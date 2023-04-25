#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch
import json

description = """
Utility for cleaning torch checkpoints for release.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output checkpoint file path', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]

data = torch.load(checkpoint_path,map_location="cpu")

if "optimizer" in data:
  del data["optimizer"]
del data["running_metrics"]
del data["metrics"]
del data["train_state"]["old_train_data_dirs"]
del data["train_state"]["data_files_used"]

if "last_val_metrics" in data:
  del data["last_val_metrics"]

torch.save(data, output_path)
print(f"Cleaned {checkpoint_path} -> {output_path} for release")
