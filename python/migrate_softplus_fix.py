#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch

import katago.train.load_model

description = """
Hacky script to partially migrate v12 optimistic policy to v13 changing score error.
Run on a torch checkpoint.ckpt file. Replace the old one with the new one after a backup.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]

data = torch.load(checkpoint_path,map_location="cpu")
assert data["config"]["version"] == 12

def shift_bias(name, channel_idx):
    if f"module.{name}" in data["model"]:
        data["model"][f"module.{name}"][channel_idx] -= math.log(150.0 / 30.0)
    elif name in data["model"]:
        data["model"][name][channel_idx] -= math.log(150.0 / 30.0)
    else:
        assert False, f"{name} not found in saved model"
    if f"module.{name}" in data["swa_model"]:
        data["swa_model"][f"module.{name}"][channel_idx] -= math.log(150.0 / 30.0)
    elif name in data["swa_model"]:
        data["swa_model"][name][channel_idx] -= math.log(150.0 / 30.0)
    else:
        assert False, f"{name} not found in swa model"

shift_bias("value_head.linear_moremiscvaluehead.bias", channel_idx=1)
if any("intermediate_value_head" in key for key in data["model"].keys()):
    shift_bias("intermediate_value_head.linear_moremiscvaluehead.bias", channel_idx=1)

if "optimizer" in data:
    print("Deleting optimizer state")
    del data["optimizer"]
# if "swa_model" in data:
#   print("Deleting swa model state")
#   del data["swa_model"]
if "running_metrics" in data:
    print("Resetting shortterm score error running metrics")
    data["running_metrics"]["sums"]["esstloss_sum"] /= 100000.0
    data["running_metrics"]["weights"]["esstloss_sum"] /= 100000.0
    data["running_metrics"]["sums"]["Iesstloss_sum"] /= 100000.0
    data["running_metrics"]["weights"]["Iesstloss_sum"] /= 100000.0

print("Clearing export cycle counter to give time to reconverge")
assert "export_cycle_counter" in data["train_state"]
data["train_state"]["export_cycle_counter"] = 0

print("Setting version to 13")
data["config"]["version"] = 13

print(f"Saving to {output_path}")
torch.save(data, output_path)
print("Done")
