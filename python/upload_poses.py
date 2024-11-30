#!/usr/bin/python3
import sys
import os
import argparse
import logging
import requests
import hashlib
import configparser
import json

from requests.auth import HTTPBasicAuth

#Command and args-------------------------------------------------------------------

description = """
Upload neural net weights to server
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-run-name', help='run name', required=True)
parser.add_argument('-poses', help='startposes or hintposes file or dir', required=True, action='append')
parser.add_argument('-upload-log-file', help='log upload data to this file', required=False)
parser.add_argument('-connection-config', help='config with serverUrl and username and password', required=False)
parser.add_argument('-set-total-weight', help='set total weight of poses to this', type=float, required=False)
parser.add_argument('-weight-factor', help='multiply weight of poses by this', type=float, required=False)
parser.add_argument('-separate-summaries', help='summarize dirs separately', action="store_true", required=False)
parser.add_argument('-separate-summaries-separate-files', help='not merging files in dir for summaries', action="store_true", required=False)
parser.add_argument('-extra-stats', help='show some extra stats', action="store_true", required=False)
parser.add_argument('-notes', help='notes or description label for poses', required=False)
args = vars(parser.parse_args())

run_name = args["run_name"]
poses_files_or_dirs = args["poses"]
upload_log_file = args["upload_log_file"]
connection_config_file = args["connection_config"]
set_total_weight = args["set_total_weight"]
weight_factor = args["weight_factor"]
separate_summaries = args["separate_summaries"]
separate_summaries_separate_files = args["separate_summaries_separate_files"]
extra_stats = args["extra_stats"]
notes = args["notes"]

loglines = []
def log(s):
    loglines.append(s)
    print(s,flush=True)

def write_log():
    if upload_log_file is not None:
        with open(upload_log_file,"a+") as f:
            for line in loglines:
                f.write(line + "\n")

if notes is None:
    notes = ""

if connection_config_file is not None:
    with open(connection_config_file,'r') as f:
        connection_config_content = "[DEFAULT]\n" + f.read()
    config_parser = configparser.ConfigParser()
    config_parser.read_string(connection_config_content)
    base_server_url = config_parser["DEFAULT"]["serverUrl"]

    if not base_server_url.endswith("/"):
        base_server_url = base_server_url + "/"

log("run_name" + ": " + run_name)
log("poses" + ": " + str(poses_files_or_dirs))
log("notes" + ": " + notes)

if connection_config_file is not None:
    username = config_parser["DEFAULT"]["username"]
    password = config_parser["DEFAULT"]["password"]
    log("username" + ": " + username)
    log("base_server_url" + ": " + base_server_url)

def compute_sum_sumsq(poses):
    sumweight = 0.0
    sumweightsq = 0.0
    for pos in poses:
        sumweight += pos["weight"]
        sumweightsq += pos["weight"] * pos["weight"]
    return (sumweight, sumweightsq)

def print_extra_stats(poses):
    sumweight = 0.0
    sumturnnum = 0
    for pos in poses:
        sumturnnum += pos["weight"] * (pos["initialTurnNumber"] + len(pos["moveLocs"]))
        sumweight += pos["weight"]
    print("Avg turn number:", sumturnnum/sumweight)

log("Loading positions")
poses_by_key = {}
poses_by_key_by_separate_files = {}

def handle_file(poses_by_key, poses_file):
    log("Loading" + ": " + poses_file)
    if separate_summaries_separate_files:
        separate_files_key = poses_file
    else:
        separate_files_key = os.path.dirname(poses_file)

    if separate_files_key not in poses_by_key_by_separate_files:
        poses_by_key_by_separate_files[separate_files_key] = {}
    poses_by_key_this_file = poses_by_key_by_separate_files[separate_files_key]

    with open(poses_file,"r") as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue
            pos = json.loads(line)
            key = (
                str(pos["initialTurnNumber"]) + "$" +
                "@".join(pos["moveLocs"]) + "$" +
                "@".join(pos["movePlas"]) + "$" +
                str(pos["xSize"]) + "$" +
                str(pos["ySize"]) + "$" +
                pos["nextPla"] + "$" +
                pos["board"] + "$" +
                pos["hintLoc"]
            )

            if len(pos["movePlas"]) > 0:
                assert pos["nextPla"] == "B" or pos["nextPla"] == "W"
                assert pos["movePlas"][0] == pos["nextPla"]
            for i in range(len(pos["movePlas"])):
                assert pos["movePlas"][i] == "B" or pos["movePlas"][i] == "W"
            for i in range(1,len(pos["movePlas"])):
                assert pos["movePlas"][i] != pos["movePlas"][i-1]

            if "weight" in pos:
                weight = pos["weight"]
            else:
                pos["weight"] = 1.0
                weight = 1.0

            if key in poses_by_key:
                poses_by_key[key]["weight"] += weight
            else:
                poses_by_key[key] = pos.copy()
            if key in poses_by_key_this_file:
                poses_by_key_this_file[key]["weight"] += weight
            else:
                poses_by_key_this_file[key] = pos.copy()

poses_files_or_dirs = sorted(poses_files_or_dirs)
for poses_file_or_dir in poses_files_or_dirs:
    if os.path.isdir(poses_file_or_dir):
        for (path,dirnames,filenames) in os.walk(poses_file_or_dir):
            dirnames.sort()
            filenames = sorted(filenames)
            for filename in filenames:
                if filename.endswith(".startposes.txt") or filename.endswith(".hintposes.txt") or filename.endswith(".bookposes.txt"):
                    handle_file(poses_by_key, os.path.join(path,filename))
    else:
        handle_file(poses_by_key, poses_file_or_dir)

if separate_summaries:
    for key, poses_by_key_this_key in poses_by_key_by_separate_files.items():
        sumweight,sumweightsq = compute_sum_sumsq(poses_by_key_this_key.values())
        if sumweight > 0 and sumweightsq > 0:
            log(key)
            log("Found %d unique positions" % len(poses_by_key_this_key.values()))
            log("Found %f total weight" % sumweight)
            log("Found %f ess" % (sumweight * sumweight / sumweightsq))
            if extra_stats:
                print_extra_stats(poses_by_key_this_key.values())
            log("%d %f %f" % (len(poses_by_key_this_key.values()), sumweight, (sumweight * sumweight / sumweightsq)))

poses = poses_by_key.values()
sumweight,sumweightsq = compute_sum_sumsq(poses)
log("Found %d unique positions" % len(poses))
log("Found %f total weight" % sumweight)
log("Found %f ess" % (sumweight * sumweight / sumweightsq))
if extra_stats:
    print_extra_stats(poses)
log("%d %f %f" % (len(poses), sumweight, (sumweight * sumweight / sumweightsq)))

if set_total_weight is not None:
    log("Setting total weight of data to " + str(set_total_weight))
    scale = set_total_weight / sumweight
    for pos in poses:
        pos["weight"] *= scale
elif weight_factor is not None:
    log("Scaling weight of data by " + str(weight_factor))
    for pos in poses:
        pos["weight"] *= weight_factor

def postStuff(to_post):
    url = base_server_url + "api/startposes/"
    result = requests.post(url,json=to_post,auth=HTTPBasicAuth(username,password))
    log("Post status code: " + str(result.status_code))
    if result.status_code == 200 or result.status_code == 201 or result.status_code == 202:
        log("Post success")
    else:
        log("Post failed")
        if result.text:
            log(result.text)
        write_log()
        sys.exit(1)


if connection_config_file is None:
    log("No connection config specified, quitting WITHOUT uploading anything")
else:
    num_total = len(poses)
    num_posted = 0

    to_post = []
    for pos in poses:
        weight = pos["weight"]
        data = {
            "run": base_server_url + "api/runs/" + run_name + "/",
            "weight": weight,
            "data": pos,
            "notes": notes,
        }
        to_post.append(data)
        if len(to_post) >= 5000:
            num_posted += len(to_post)
            log("Posting " + str(num_posted) + "/" + str(num_total))
            postStuff(to_post)
            to_post = []

    num_posted += len(to_post)
    log("Posting " + str(num_posted) + "/" + str(num_total))
    postStuff(to_post)
    to_post = []

    log("Done")
    write_log()
