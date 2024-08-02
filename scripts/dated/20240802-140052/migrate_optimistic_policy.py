#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch

description = """
Hacky script to migrate v11 model to v12 with optimistic policy.
Run on a torch checkpoint.ckpt file. Replace the old one with the new one after a backup.
And then manually edit the model.config.json to also bump the json file version to 12.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]

data = torch.load(checkpoint_path,map_location="cpu")
assert data["config"]["version"] == 11

def noisify(tensor):
    scale = 0.01 * math.sqrt(torch.mean(torch.square(tensor)).item())
    return tensor * (0.95 + 0.05 * torch.rand(tensor.shape, device=tensor.device)) + scale * (2.0 * torch.rand(tensor.shape, device=tensor.device) - 1.0)


# Torch output channel dimension is dim 0.
def expand_policy_out_dim(name, tensor):
    if tensor.shape[0] == 6:
        print(f"{name} tensor already has 6 output channels, not modifying it")
        return tensor
    elif tensor.shape[0] == 5:
        print(f"{name} tensor has 5 output channels, adding one more initialized from channel 0")
        slice0 = tensor[0:1]
        tensor = torch.cat((tensor,noisify(slice0)), dim=0)
        return tensor
    elif tensor.shape[0] == 4:
        print(f"{name} tensor has 4 output channels, adding two more initialized from channel 0")
        slice0 = tensor[0:1]
        tensor = torch.cat((tensor,noisify(slice0),noisify(slice0)), dim=0)
        return tensor
    else:
        assert False, f"Unexpected {name} tensor shape: {tensor.shape}"

def expand_policy_out_dim_for(name):
    if f"module.{name}" in data["model"]:
        data["model"][f"module.{name}"] = expand_policy_out_dim(f"module.{name}", data["model"][f"module.{name}"])
    elif name in data["model"]:
        data["model"][name] = expand_policy_out_dim(name, data["model"][name])
    else:
        assert False, f"{name} not found in saved model"

expand_policy_out_dim_for("policy_head.conv2p.weight")
expand_policy_out_dim_for("policy_head.linear_pass.weight")
if any("intermediate_policy_head" in key for key in data["model"].keys()):
    expand_policy_out_dim_for("intermediate_policy_head.conv2p.weight")
    expand_policy_out_dim_for("intermediate_policy_head.linear_pass.weight")

if "optimizer" in data:
    print("Deleting optimizer state")
    del data["optimizer"]
if "swa_model" in data:
    print("Deleting swa model state")
    del data["swa_model"]
print("Setting version to 12")
data["config"]["version"] = 12

print(f"Saving to {output_path}")
torch.save(data, output_path)
print("Done, don't forget to edit the model.config.json to version 12 if applicable")
