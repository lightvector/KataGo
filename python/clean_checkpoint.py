#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch
import json

import load_model

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

if "running_metrics" in data:
    del data["running_metrics"]
if "metrics" in data:
    del data["metrics"]

if "train_state" in data:
    if "old_train_data_dirs" in data["train_state"]:
        del data["train_state"]["old_train_data_dirs"]
    if "data_files_used" in data["train_state"]:
        del data["train_state"]["data_files_used"]

if "last_val_metrics" in data:
    del data["last_val_metrics"]

torch.save(data, output_path)
print(f"Cleaned {checkpoint_path} -> {output_path} for release")
