#!/usr/bin/python3
import sys
import os
import argparse
import logging
import requests
import hashlib
import configparser

from requests.auth import HTTPBasicAuth

#Command and args-------------------------------------------------------------------

description = """
Upload neural net weights to server
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-run-name', help='run name', required=True)
parser.add_argument('-model-name', help='model name', required=True)
parser.add_argument('-model-file', help='model file for kg engine alone', required=True)
parser.add_argument('-model-zip', help='zipped model file with tf weights', required=True)
parser.add_argument('-upload-log-file', help='log upload data to this file', required=True)
parser.add_argument('-parents-dir', help='dir with uploaded models dirs for finding parent', required=False)
parser.add_argument('-connection-config', help='config with serverUrl and username and password', required=True)
args = vars(parser.parse_args())

run_name = args["run_name"]
model_name = args["model_name"]
model_file = args["model_file"]
model_zip = args["model_zip"]
upload_log_file = args["upload_log_file"]
parents_dir = args["parents_dir"]
connection_config_file = args["connection_config"]

loglines = []
def log(s):
  loglines.append(s)
  print(s,flush=True)

def write_log():
  with open(upload_log_file,"w+") as f:
    for line in loglines:
      f.write(line + "\n")

with open(connection_config_file,'r') as f:
  connection_config_content = "[DEFAULT]\n" + f.read()
config_parser = configparser.ConfigParser()
config_parser.read_string(connection_config_content)
base_server_url = config_parser["DEFAULT"]["serverUrl"]

if not base_server_url.endswith("/"):
  base_server_url = base_server_url + "/"

username = config_parser["DEFAULT"]["username"]
password = config_parser["DEFAULT"]["password"]

log("run_name" + ": " + run_name)
log("model_name" + ": " + model_name)
log("model_file" + ": " + model_file)
log("model_zip" + ": " + model_zip)
log("parents_dir" + ": " + str(parents_dir))
log("username" + ": " + username)
log("base_server_url" + ": " + base_server_url)

network_size = model_name.split("-")[1]

model_file_extension = None
if model_file.endswith(".bin.gz"):
  model_file_extension = ".bin.gz"
elif model_file.endswith(".txt.gz"):
  model_file_extension = ".txt.gz"
else:
  raise Exception("Model file must end in .bin.gz or .txt.gz")

possible_parents = []
if parents_dir is not None:
  for fname in os.listdir(parents_dir):
    if fname.startswith(network_size):
      pieces = fname.split("-")
      datasamples = int(pieces[2][1:])
      possible_parents.append((fname, datasamples))
  possible_parents.sort(key=(lambda x: x[1]))

parent_network_name_without_run = None
if len(possible_parents) > 0:
  parent_network_name_without_run = possible_parents[-1][0]

with open(model_file,"rb") as f:
  model_file_contents = f.read()
  model_file_bytes = len(model_file_contents)
  model_file_sha256 = hashlib.sha256(model_file_contents).hexdigest()
  del model_file_contents

url = base_server_url + "api/networks/"

with open(model_file,"rb") as model_file_handle:
  data = {
    "run": (None, base_server_url + "api/runs/" + run_name + "/"),
    "name": (None, model_name),
    "network_size": (None, network_size),
    "is_random": (None, "false"),
    "model_file": (model_name + model_file_extension, model_file_handle, "application/octet-stream"),
    "model_file_bytes": (None, model_file_bytes),
    "model_file_sha256": (None, model_file_sha256)
    # "model_zip": (model_name + ".zip", model_zip_handle.read()),
  }

  if parent_network_name_without_run is not None:
    data["parent_network"] = (None, base_server_url + "api/networks/" + run_name + "-" + parent_network_name_without_run + "/")

  # print(requests.Request('POST', base_server_url, files=data).prepare().body)

  result = requests.post(url,files=data,auth=HTTPBasicAuth(username,password))

log("Post status code: " + str(result.status_code))
log("Post result: " + str(result.text))
if result.status_code == 409:
  log("Got 409 error, network already uploaded? So assuming everything is good")
  write_log()
elif result.status_code == 200 or result.status_code == 201 or result.status_code == 202:
  log("Post success")
  write_log()
else:
  log("Post failed")
  write_log()
  sys.exit(1)
