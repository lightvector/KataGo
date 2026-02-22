#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch
import json
import re

import katago.train.load_model
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics

description = """
Hacky script to migrate a model with weight decay centered at 0 for gammas to weight decay centered at 1.
Run on a torch checkpoint.ckpt file. Replace the old one with the new one after a backup.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]

data = torch.load(checkpoint_path,map_location="cpu")

if data["config"].get("gamma_weight_decay_center_1",False):
    raise ValueError("gamma_weight_decay_center_1 is already True: " + str(data["config"]))

def adjust_gamma(name, tensor):
    mean = torch.mean(tensor).item()
    scale = math.sqrt(torch.mean(torch.square(tensor)).item())
    print(f"Converting {name} {mean=:.4f} {scale=:.4f}")
    return tensor - 1.0

for param_name in data["model"]:
    if param_name.endswith(".gamma"):
        data["model"][param_name] = adjust_gamma(param_name, data["model"][param_name])

if "optimizer" in data:
    print("Deleting optimizer state")
    del data["optimizer"]
if "swa_model" in data:
    print("Deleting swa model state")
    del data["swa_model"]

data["config"]["gamma_weight_decay_center_1"] = True

print(f"Saving to {output_path}")
torch.save(data, output_path)
print("Done")
