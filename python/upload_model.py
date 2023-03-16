#!/usr/bin/python3
import argparse
import configparser
import datetime
import hashlib
import json
import os
import sys

import requests
from requests.auth import HTTPBasicAuth
from requests_toolbelt.adapters import host_header_ssl

# Command and args-------------------------------------------------------------------

description = """
Upload neural net weights to server
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument("-run-name", help="run name", required=True)
parser.add_argument("-model-name", help="model name", required=True)
parser.add_argument("-model-file", help="model file for kg engine alone", required=True)
parser.add_argument("-model-zip", help="zipped model file with tf weights", required=True)
parser.add_argument("-upload-log-file", help="log upload data to this file", required=True)
parser.add_argument("-metadata-file", help="metadata.json file for recording some stats", required=False)
parser.add_argument("-parents-dir", help="dir with uploaded models dirs for finding parent", required=False)
parser.add_argument("-connection-config", help="config with serverUrl and username and password", required=True)
parser.add_argument(
    "-not-enabled",
    help="upload model where it is not enabled for train/rating to begin with",
    required=False,
    action="store_true",
)
parser.add_argument("-rating-only", help="upload for rating only or not", type=int, default=0, required=False)
parser.add_argument("-notes", help="extra notes to record for model", required=False)
args = vars(parser.parse_args())

run_name = args["run_name"]
model_name = args["model_name"]
model_file = args["model_file"]
model_zip = args["model_zip"]
upload_log_file = args["upload_log_file"]
metadata_file = args["metadata_file"]
parents_dir = args["parents_dir"]
connection_config_file = args["connection_config"]
not_enabled = args["not_enabled"]
rating_only = args["rating_only"]
notes = args["notes"]

loglines = []


def log(s):
    loglines.append(s)
    print(s, flush=True)


def write_log():
    with open(upload_log_file, "w+") as f:
        for line in loglines:
            f.write(line + "\n")


with open(connection_config_file, "r") as f:
    connection_config_content = "[DEFAULT]\n" + f.read()
config_parser = configparser.ConfigParser()
config_parser.read_string(connection_config_content)
base_server_url = config_parser["DEFAULT"]["serverUrl"]

if not base_server_url.endswith("/"):
    base_server_url = f"{base_server_url}/"

username = config_parser["DEFAULT"]["username"]
password = config_parser["DEFAULT"]["password"]
sslVerificationHost = None
if "sslVerificationHost" in config_parser["DEFAULT"]:
    sslVerificationHost = config_parser["DEFAULT"]["sslVerificationHost"]
sslVerifyPemPath = None
if "sslVerifyPemPath" in config_parser["DEFAULT"]:
    sslVerifyPemPath = config_parser["DEFAULT"]["sslVerifyPemPath"]

log("now" + ": " + str(datetime.datetime.now()))
log("run_name" + ": " + run_name)
log("model_name" + ": " + model_name)
log("model_file" + ": " + model_file)
log("model_zip" + ": " + model_zip)
log("parents_dir" + ": " + str(parents_dir))
log("metadata_file" + ": " + str(metadata_file))
log("username" + ": " + username)
log("base_server_url" + ": " + base_server_url)

network_size = model_name.split("-")[1]

model_file_extension = None
if model_file.endswith(".bin.gz"):
    model_file_extension = ".bin.gz"
elif model_file.endswith(".txt.gz"):
    model_file_extension = ".txt.gz"
else:
    raise ValueError("Model file must end in .bin.gz or .txt.gz")

possible_parents = []
if parents_dir is not None:
    for fname in os.listdir(parents_dir):
        if fname.startswith(network_size):
            pieces = fname.split("-")
            datasamples = int(pieces[2][1:])
            possible_parents.append((fname, datasamples))
    possible_parents.sort(key=(lambda x: x[1]))

parent_network_name_without_run = None
if possible_parents:
    parent_network_name_without_run = possible_parents[-1][0]

metadata = None
if metadata_file is not None:
    with open(metadata_file) as f:
        metadata = json.load(f)

with open(model_file, "rb") as f:
    model_file_contents = f.read()
    model_file_bytes = len(model_file_contents)
    model_file_sha256 = hashlib.sha256(model_file_contents).hexdigest()
    del model_file_contents

url = f"{base_server_url}api/networks/"

with open(model_file, "rb") as model_file_handle:
    with open(model_zip, "rb") as model_zip_handle:
        log_gamma_offset = -1.0 if network_size == "b60c320" else 0.0
        data = {
            "run": (None, f"{base_server_url}api/runs/{run_name}/"),
            "name": (None, model_name),
            "network_size": (None, network_size),
            "is_random": (None, "false"),
            "model_file": (
                model_name + model_file_extension,
                model_file_handle,
                "application/octet-stream",
            ),
            "model_file_bytes": (None, model_file_bytes),
            "model_file_sha256": (None, model_file_sha256),
            "training_games_enabled": (
                None,
                ("false" if (not_enabled or rating_only != 0) else "true"),
            ),
            "rating_games_enabled": (None, ("false" if not_enabled else "true")),
            "log_gamma_offset": (None, str(log_gamma_offset)),
            "model_zip_file": (
                f"{model_name}.zip",
                model_zip_handle,
                "application/octet-stream",
            ),
        }

        if parent_network_name_without_run is not None:
            data["parent_network"] = (
                None,
                f"{base_server_url}api/networks/{run_name}-{parent_network_name_without_run}/",
            )

        if notes is not None:
            data["notes"] = (None, notes)

        if metadata is not None:
            if "global_step_samples" in metadata:
                data["global_step_samples"] = (None, metadata["global_step_samples"])
            if "total_num_data_rows" in metadata:
                data["total_num_data_rows"] = (None, metadata["total_num_data_rows"])
            if "extra_stats" in metadata:
                data["extra_stats"] = (None, json.dumps(metadata["extra_stats"]))

        # print(requests.Request('POST', base_server_url, files=data).prepare().body)

        if sslVerificationHost is not None:
            sess = requests.Session()
            sess.mount("https://", host_header_ssl.HostHeaderSSLAdapter())
            if sslVerifyPemPath is None:
                result = sess.post(
                    url, files=data, auth=HTTPBasicAuth(username, password), headers={"Host": sslVerificationHost}
                )
            else:
                result = sess.post(
                    url,
                    files=data,
                    auth=HTTPBasicAuth(username, password),
                    headers={"Host": sslVerificationHost},
                    verify=sslVerifyPemPath,
                )
        elif sslVerifyPemPath is not None:
            result = requests.post(url, files=data, auth=HTTPBasicAuth(username, password), verify=sslVerifyPemPath)
        else:
            result = requests.post(url, files=data, auth=HTTPBasicAuth(username, password))

log(f"Post status code: {str(result.status_code)}")
log(f"Post result: {str(result.text)}")
if result.status_code == 400 and "already exist" in str(result.text):
    log("Got 400 error with substring 'already exist', network already uploaded? So assuming everything is good")
    write_log()
elif result.status_code in [200, 201, 202]:
    log("Post success")
    write_log()
else:
    log("Post failed")
    write_log()
    sys.exit(1)
