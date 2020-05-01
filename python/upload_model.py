#!/usr/bin/python3
import sys
import os
import argparse
import random
import time
import logging
import json
import datetime
import struct
import requests
import shutil

import modelconfigs


#Command and args-------------------------------------------------------------------

description = """
Upload neural net weights to server
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-run-name', help='run name', required=True)
parser.add_argument('-model-name', help='model name', required=True)
parser.add_argument('-model-file', help='model file for kg engine alone', required=True)
parser.add_argument('-model-zip', help='zipped model file for kg engine and tf weights', required=True)
parser.add_argument('-upload-log-file', help='log upload data to this file', required=True)
parser.add_argument('-uploaded-dir', help='dir with uploaded models for finding parent', required=True)
parser.add_argument('-base-server-url', help='base server url', required=True)
args = vars(parser.parse_args())

run_name = args["run_name"]
model_name = args["model_name"]
model_file = args["model_file"]
model_zip = args["model_zip"]
upload_log_file = args["upload_log_file"]
uploaded_dir = args["uploaded_dir"]
base_server_url = args["base_server_url"]

loglines = []
def log(s):
  loglines.append(s)
  print(s,flush=True)

log("run_name" + ": " + run_name)
log("model_name" + ": " + model_name)
log("model_file" + ": " + model_file)
log("model_zip" + ": " + model_zip)
log("uploaded_dir" + ": " + uploaded_dir)
log("base_server_url" + ": " + base_server_url)

network_size = model_name.split("-")[0]
nb_parameters = modelconfigs.num_parameters_of_name[network_size]

possible_parents = []
for fname in os.listdir(uploaded_dir):
  if os.path.isfile(os.path.join(uploaded_dir,fname)):
    if fname.startswith(network_size):
      pieces = fname.split("-")
      datasamples = long(pieces[2][1:])
      possible_parents.append((fname, datasamples))
possible_parents.sort(key=(lambda x: x[1]))

# TODO upload model_file and model_zip

url = base_server_url + "networks/"
data = {
  "name": name,
  "network_size": network_size,
  "nb_parameters": nb_parameters,
  "model_architecture_details": {},
  "model_file": "TODO", #TODO presumably we upload and then put this here?
  "parent_network": possible_parents[-1][0]
}

#TODO we need to distinguish whether it failed because network was down, or
#if it was already there. Former we retry, latter we proceed
result = requests.post(url,data=data)
log("post result: " + result.text)

with open(upload_log_file,"w+") as f:
  for line in loglines:
    f.write(line + "\n")
