#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch
import json

description = """
Utility for dumping or modifying torch checkpoint file contents
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output-json-to', help='Output json contents to here, except for model parameters and optimizer state')
parser.add_argument('-overwrite-checkpoint-from-json', help='Use this json contents to overwrite the apprpriate fields of the state dict of the checkpoint')
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_json_to = args["output_json_to"]
overwrite_checkpoint_from_json = args["overwrite_checkpoint_from_json"]

data = torch.load(checkpoint_path,map_location="cpu")

if output_json_to is not None:
  assert output_json_to.endswith(".json")
  data_to_write = dict(
    running_metrics = data["running_metrics"],
    train_state = data["train_state"],
    config = data["config"] if "config" in data else None,
  )
  with open(output_json_to,"w") as f:
    json.dump(data,f,indent=2)
  print(f"Dumped to {output_json_to}")

elif overwrite_checkpoint_from_json:
  with open(overwrite_checkpoint_from_json) as f:
    data_to_use = json.load(f)
    if "running_metrics" in data_to_use:
      print("Overwriting running_metrics")
      data["running_metrics"] = data_to_use["running_metrics"]
    if "train_state" in data_to_use:
      print("Overwriting train_state")
      data["train_state"] = data_to_use["train_state"]
    if "config" in data_to_use:
      print("Overwriting config")
      data["config"] = data_to_use["config"]

  torch.save(data, checkpoint_path)
  print(f"Updated {checkpoint_path} with new fields")

else:
  data_to_write = dict(
    running_metrics = data["running_metrics"],
    train_state = data["train_state"],
    config = data["config"] if "config" in data else None,
  )
  print(json.dumps(data_to_write,indent=2))


