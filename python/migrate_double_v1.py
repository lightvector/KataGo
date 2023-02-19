#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch

description = """
Hacky script to double number of v1 channels.
Run on a torch checkpoint.ckpt file. Replace the old one with the new one after a backup.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]

data = torch.load(checkpoint_path,map_location="cpu")

def noisify(tensor):
  scale = 0.02 * math.sqrt(torch.mean(torch.square(tensor)).item())
  return tensor * (0.9 + 0.15 * torch.rand(tensor.shape, device=tensor.device)) + scale * (2.0 * torch.rand(tensor.shape, device=tensor.device) - 1.0)

# Torch input channel dimension is dim 1.
def double_linear_in_dim(name, tensor, scale):
  ic = tensor.shape[1]
  print(f"{name} tensor has {ic} input channels, doubling to {ic*2}")
  tensor = torch.cat((noisify(tensor),noisify(tensor)), dim=1) * scale
  return tensor

# Torch output channel dimension is dim 0.
def double_linear_out_dim(name, tensor, scale):
  oc = tensor.shape[0]
  print(f"{name} tensor has {oc} output channels, doubling to {oc*2}")
  tensor = torch.cat((noisify(tensor),noisify(tensor)), dim=0) * scale
  return tensor

def expand_in_dim_for(name, scale):
  if f"module.{name}" in data["model"]:
    data["model"][f"module.{name}"] = double_linear_in_dim(f"module.{name}", data["model"][f"module.{name}"], scale)
  elif name in data["model"]:
    data["model"][name] = double_linear_in_dim(name, data["model"][name], scale)
  else:
    assert False, f"{name} not found in saved model"

def expand_out_dim_for(name, scale):
  if f"module.{name}" in data["model"]:
    data["model"][f"module.{name}"] = double_linear_out_dim(f"module.{name}", data["model"][f"module.{name}"], scale)
  elif name in data["model"]:
    data["model"][name] = double_linear_out_dim(name, data["model"][name], scale)
  else:
    assert False, f"{name} not found in saved model"

expand_out_dim_for("value_head.conv1.weight", scale=1.0)
expand_in_dim_for("value_head.bias1.beta", scale=1.0)
expand_in_dim_for("value_head.linear2.weight", scale=math.sqrt(0.5))
expand_in_dim_for("value_head.conv_ownership.weight", scale=math.sqrt(0.5))
expand_in_dim_for("value_head.conv_scoring.weight", scale=math.sqrt(0.5))
expand_in_dim_for("value_head.linear_s2.weight", scale=math.sqrt(0.5))
expand_in_dim_for("value_head.linear_smix.weight", scale=math.sqrt(0.5))


if any("intermediate_value_head" in key for key in data["model"].keys()):

  expand_out_dim_for("intermediate_value_head.conv1.weight", scale=1.0)
  expand_in_dim_for("intermediate_value_head.bias1.beta", scale=1.0)
  expand_in_dim_for("intermediate_value_head.linear2.weight", scale=math.sqrt(0.5))
  expand_in_dim_for("intermediate_value_head.conv_ownership.weight", scale=math.sqrt(0.5))
  expand_in_dim_for("intermediate_value_head.conv_scoring.weight", scale=math.sqrt(0.5))
  expand_in_dim_for("intermediate_value_head.linear_s2.weight", scale=math.sqrt(0.5))
  expand_in_dim_for("intermediate_value_head.linear_smix.weight", scale=math.sqrt(0.5))


old_v1_num_channels = data["config"]["v1_num_channels"]
print(f"Doubling v1_num_channels {old_v1_num_channels}")
data["config"]["v1_num_channels"] = old_v1_num_channels * 2


if "optimizer" in data:
  print("Deleting optimizer state")
  del data["optimizer"]
if "swa_model" in data:
  print("Deleting swa model state")
  del data["swa_model"]

print(f"Saving to {output_path}")
torch.save(data, output_path)
print("Done")
