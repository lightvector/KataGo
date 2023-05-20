#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch

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
assert data["config"]["version"] == 13

# No modification of any weight is needed since for values that aren't large positive values,
# due to the fact that softplus(x) ~= exp(x) and exp(0.5*x) ** 2 ~= exp(x), the existing
# weights of the model already should be a good starting point for the new activation.

if "optimizer" in data:
  print("Deleting optimizer state")
  del data["optimizer"]
# if "swa_model" in data:
#   print("Deleting swa model state")
#   del data["swa_model"]
if "running_metrics" in data:
  print("Resetting shortterm value and score error running metrics")
  data["running_metrics"]["sums"]["evstloss_sum"] /= 100000.0
  data["running_metrics"]["weights"]["evstloss_sum"] /= 100000.0
  data["running_metrics"]["sums"]["Ievstloss_sum"] /= 100000.0
  data["running_metrics"]["weights"]["Ievstloss_sum"] /= 100000.0
  data["running_metrics"]["sums"]["esstloss_sum"] /= 100000.0
  data["running_metrics"]["weights"]["esstloss_sum"] /= 100000.0
  data["running_metrics"]["sums"]["Iesstloss_sum"] /= 100000.0
  data["running_metrics"]["weights"]["Iesstloss_sum"] /= 100000.0
print("Setting version to 14")
data["config"]["version"] = 14

print(f"Saving to {output_path}")
torch.save(data, output_path)
print("Done")
