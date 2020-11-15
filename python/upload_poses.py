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
parser.add_argument('-poses-file', help='startposes or hintposes file', required=True)
parser.add_argument('-upload-log-file', help='log upload data to this file', required=False)
parser.add_argument('-connection-config', help='config with serverUrl and username and password', required=False)
parser.add_argument('-set-total-weight', help='set total weight of poses to this', type=float, required=False)
parser.add_argument('-notes', help='notes or description label for poses', required=False)
args = vars(parser.parse_args())

run_name = args["run_name"]
poses_file = args["poses_file"]
upload_log_file = args["upload_log_file"]
connection_config_file = args["connection_config"]
set_total_weight = args["set_total_weight"]
notes = args["notes"]

loglines = []
def log(s):
  loglines.append(s)
  print(s,flush=True)

def write_log():
  if upload_log_file is not None:
    with open(upload_log_file,"w+") as f:
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
log("poses_file" + ": " + poses_file)
log("notes" + ": " + notes)

if connection_config_file is not None:
  username = config_parser["DEFAULT"]["username"]
  password = config_parser["DEFAULT"]["password"]
  log("username" + ": " + username)
  log("base_server_url" + ": " + base_server_url)

log("Loading positions")
poses_by_key = {}
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

    if "weight" in pos:
      weight = pos["weight"]
    else
      pos["weight"] = 1.0
      weight = 1.0

    if key in poses_by_key:
      poses_by_key[key]["weight"] += weight
    else:
      poses_by_key[key] = pos

poses = poses_by_key.values()

sumweight = 0.0
sumweightsq = 0.0

for pos in poses:
  sumweight += pos["weight"]
  sumweightsq += pos["weight"] * pos["weight"]

log("Found %d unique positions" % len(poses))
log("Found %f total weight" % sumweight)
log("Found %f ess" % (sumweight * sumweight / sumweightsq))

if set_total_weight is not None:
  log("Setting total weight of data to " + str(set_total_weight))
  scale = set_total_weight / sumweight
  for pos in poses:
    pos["weight"] *= scale

if connection_config_file is None:
  log("No connection config specified, quitting WITHOUT uploading anything")
else:
  log("Building postdata")
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
  log("Built postdata")

  log("Posting")
  url = base_server_url + "api/startposes/"
  result = requests.post(url,json=to_post,auth=HTTPBasicAuth(username,password))

  log("Post status code: " + str(result.status_code))
  log("Post result: " + str(result.text))
  if result.status_code == 200 or result.status_code == 201 or result.status_code == 202:
    log("Post success")
    write_log()
  else:
    log("Post failed")
    write_log()
    sys.exit(1)
