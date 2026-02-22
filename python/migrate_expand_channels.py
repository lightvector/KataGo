#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch
import json
import random
import re

import katago.train.load_model
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics

description = """
Hacky script to upsize number of channels in the net to match a new config with the same number of blocks and block configuration.
Run on a torch checkpoint.ckpt file. Replace the old one with the new one after a backup.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-new-config', help='New config file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
new_config_path = args["new_config"]
output_path = args["output"]

with open(new_config_path,"r") as f:
    new_config = json.load(f)

data = torch.load(checkpoint_path,map_location="cpu")

assert set(data["config"].keys()) == set(new_config.keys())

def sample_without_replacement(elts, n):
    elts = list(elts)
    result = []
    while n > 0:
        pool = elts.copy()
        num_to_take = min(n, len(pool))
        result.extend(pool[:num_to_take])
        n -= num_to_take
    return result

def expand_conv_weights(name, tensor, new_shape):
    assert len(new_shape) == 4
    assert len(tensor.shape) == 4
    # Out, in, h, w
    assert new_shape[0] >= tensor.shape[0]
    assert new_shape[1] >= tensor.shape[1]
    assert new_shape[2] == tensor.shape[2]
    assert new_shape[3] == tensor.shape[3]
    old_shape = tuple(tensor.shape)
    new_shape = tuple(new_shape)

    mean = torch.mean(tensor).item()
    scale = math.sqrt(torch.mean(torch.square(tensor)).item())
    print(f"{name} {mean=:.4f} {scale=:.4f}")

    if new_shape[0] > old_shape[0]:
        perm = tuple(sample_without_replacement(range(old_shape[0]),new_shape[0]-old_shape[0]))
        new_weights = tensor[perm,:,:,:]
        tensor = torch.cat((tensor,new_weights), dim=0)

    if new_shape[1] > old_shape[1]:
        new_weights = torch.zeros((new_shape[0],new_shape[1]-old_shape[1],new_shape[2],new_shape[3]),device=tensor.device,dtype=tensor.dtype)
        tensor = torch.cat((tensor,new_weights), dim=1)

    # Noise
    old_noise_mult = 0.9975 + 0.0050 * torch.rand((old_shape[0],new_shape[1],new_shape[2],new_shape[3]),device=tensor.device,dtype=tensor.dtype)
    new_noise_mult = 0.9500 + 0.1000 * torch.rand((new_shape[0]-old_shape[0],new_shape[1],new_shape[2],new_shape[3]),device=tensor.device,dtype=tensor.dtype)
    old_noise_add = scale * 0.0010 * torch.randn((old_shape[0],new_shape[1],new_shape[2],new_shape[3]),device=tensor.device,dtype=tensor.dtype)
    new_noise_add = scale * 0.0250 * torch.randn((new_shape[0]-old_shape[0],new_shape[1],new_shape[2],new_shape[3]),device=tensor.device,dtype=tensor.dtype)

    tensor = tensor * torch.cat((old_noise_mult,new_noise_mult), dim=0) + torch.cat((old_noise_add, new_noise_add), dim=0)
    assert tuple(tensor.shape) == new_shape
    return tensor

def expand_mat_weights(name, tensor, new_shape):
    assert len(new_shape) == 2
    assert len(tensor.shape) == 2
    # Out, in
    assert new_shape[0] >= tensor.shape[0]
    assert new_shape[1] >= tensor.shape[1]
    old_shape = tuple(tensor.shape)
    new_shape = tuple(new_shape)

    mean = torch.mean(tensor).item()
    scale = math.sqrt(torch.mean(torch.square(tensor)).item())
    print(f"{name} {mean=:.4f} {scale=:.4f}")

    if new_shape[0] > old_shape[0]:
        perm = tuple(sample_without_replacement(range(old_shape[0]),new_shape[0]-old_shape[0]))
        new_weights = tensor[perm,:]
        tensor = torch.cat((tensor,new_weights), dim=0)

    if new_shape[1] > old_shape[1]:
        new_weights = torch.zeros((new_shape[0],new_shape[1]-old_shape[1]),device=tensor.device,dtype=tensor.dtype)
        tensor = torch.cat((tensor,new_weights), dim=1)

    # Noise
    old_noise_mult = 0.9975 + 0.0050 * torch.rand((old_shape[0],new_shape[1]),device=tensor.device,dtype=tensor.dtype)
    new_noise_mult = 0.9500 + 0.1000 * torch.rand((new_shape[0]-old_shape[0],new_shape[1]),device=tensor.device,dtype=tensor.dtype)
    old_noise_add = scale * 0.0010 * torch.randn((old_shape[0],new_shape[1]),device=tensor.device,dtype=tensor.dtype)
    new_noise_add = scale * 0.0250 * torch.randn((new_shape[0]-old_shape[0],new_shape[1]),device=tensor.device,dtype=tensor.dtype)

    tensor = tensor * torch.cat((old_noise_mult,new_noise_mult), dim=0) + torch.cat((old_noise_add, new_noise_add), dim=0)
    assert tuple(tensor.shape) == new_shape
    return tensor


def expand_mat_weights_after_gpool(name, tensor, new_shape):
    assert len(new_shape) == 2
    assert len(tensor.shape) == 2
    assert new_shape[0] >= tensor.shape[0]
    assert new_shape[1] >= tensor.shape[1]

    assert tensor.shape[1] % 3 == 0
    assert new_shape[1] % 3 == 0

    oldc = tensor.shape[1] // 3
    newc = new_shape[1] // 3
    newshape_chopped = tuple(c if i != 1 else newc for (i,c) in enumerate(new_shape))
    return torch.cat(
        (
            expand_mat_weights(name+"_part0", tensor[:,:oldc], newshape_chopped),
            expand_mat_weights(name+"_part1", tensor[:,oldc:2*oldc], newshape_chopped),
            expand_mat_weights(name+"_part2", tensor[:,2*oldc:3*oldc], newshape_chopped),
        ),
        dim=1
    )


def expand_gamma(name, tensor, new_shape):
    assert len(new_shape) == 4
    assert len(tensor.shape) == 4
    # 1, out/in, 1, 1
    assert new_shape[0] == tensor.shape[0]
    assert new_shape[0] == 1
    assert new_shape[1] >= tensor.shape[1]
    assert new_shape[2] == tensor.shape[2]
    assert new_shape[2] == 1
    assert new_shape[3] == tensor.shape[3]
    assert new_shape[3] == 1
    old_shape = tuple(tensor.shape)
    new_shape = tuple(new_shape)

    mean = torch.mean(tensor).item()
    scale = math.sqrt(torch.mean(torch.square(tensor)).item())
    print(f"{name} {mean=:.4f} {scale=:.4f}")

    if new_shape[1] > old_shape[1]:
        perm = tuple(sample_without_replacement(range(old_shape[1]),new_shape[1]-old_shape[1]))
        new_weights = tensor[:,perm,:,:]
        tensor = torch.cat((tensor,new_weights), dim=1)

    # Noise
    old_noise_mult = 0.9975 + 0.0050 * torch.rand((1,old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)
    new_noise_mult = 0.9500 + 0.1000 * torch.rand((1,new_shape[1]-old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)
    old_noise_add = scale * 0.0010 * torch.randn((1,old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)
    new_noise_add = scale * 0.0100 * torch.randn((1,new_shape[1]-old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)

    tensor = tensor * torch.cat((old_noise_mult,new_noise_mult), dim=1) + torch.cat((old_noise_add, new_noise_add), dim=1)
    assert tuple(tensor.shape) == new_shape
    return tensor

def expand_beta(name, tensor, new_shape):
    assert len(new_shape) == 4
    assert len(tensor.shape) == 4
    # 1, out/in, 1, 1
    assert new_shape[0] == tensor.shape[0]
    assert new_shape[0] == 1
    assert new_shape[1] >= tensor.shape[1]
    assert new_shape[2] == tensor.shape[2]
    assert new_shape[2] == 1
    assert new_shape[3] == tensor.shape[3]
    assert new_shape[3] == 1
    old_shape = tuple(tensor.shape)
    new_shape = tuple(new_shape)

    mean = torch.mean(tensor).item()
    scale = math.sqrt(torch.mean(torch.square(tensor)).item())
    print(f"{name} {mean=:.4f} {scale=:.4f}")

    if new_shape[1] > old_shape[1]:
        perm = tuple(sample_without_replacement(range(old_shape[1]),new_shape[1]-old_shape[1]))
        new_weights = tensor[:,perm,:,:]
        tensor = torch.cat((tensor,new_weights), dim=1)

    # Noise
    old_noise_mult = 0.9975 + 0.0050 * torch.rand((1,old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)
    new_noise_mult = 0.9500 + 0.1000 * torch.rand((1,new_shape[1]-old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)
    old_noise_add = scale * 0.0010 * torch.randn((1,old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)
    new_noise_add = scale * 0.0100 * torch.randn((1,new_shape[1]-old_shape[1],1,1),device=tensor.device,dtype=tensor.dtype)

    tensor = tensor * torch.cat((old_noise_mult,new_noise_mult), dim=1) + torch.cat((old_noise_add, new_noise_add), dim=1)
    assert tuple(tensor.shape) == new_shape
    return tensor


def expand_bias(name, tensor, new_shape):
    assert len(new_shape) == 1
    assert len(tensor.shape) == 1
    # out
    assert new_shape[0] >= tensor.shape[0]
    old_shape = tuple(tensor.shape)
    new_shape = tuple(new_shape)

    mean = torch.mean(tensor).item()
    scale = math.sqrt(torch.mean(torch.square(tensor)).item())
    print(f"{name} {mean=:.4f} {scale=:.4f}")

    if new_shape[0] > old_shape[0]:
        perm = tuple(sample_without_replacement(range(old_shape[0]),new_shape[0]-old_shape[0]))
        new_weights = tensor[perm,]
        tensor = torch.cat((tensor,new_weights), dim=0)

    # Noise
    old_noise_mult = 0.9975 + 0.0050 * torch.rand((old_shape[0],),device=tensor.device,dtype=tensor.dtype)
    new_noise_mult = 0.9500 + 0.1000 * torch.rand((new_shape[0]-old_shape[0],),device=tensor.device,dtype=tensor.dtype)
    old_noise_add = scale * 0.0010 * torch.randn((old_shape[0],),device=tensor.device,dtype=tensor.dtype)
    new_noise_add = scale * 0.0100 * torch.randn((new_shape[0]-old_shape[0],),device=tensor.device,dtype=tensor.dtype)

    tensor = tensor * torch.cat((old_noise_mult,new_noise_mult), dim=0) + torch.cat((old_noise_add, new_noise_add), dim=0)
    assert tuple(tensor.shape) == new_shape
    return tensor


pos_len = 19
raw_model = Model(new_config,pos_len)
raw_model.initialize()

assert "train_state" in data

norms = Metrics.get_model_norms(raw_model)
modelnorm_normal_baseline = norms["normal"]
modelnorm_input_baseline = norms["input"]
old_modelnorm_normal_baseline = data["train_state"]["modelnorm_normal_baseline"]
old_modelnorm_input_baseline = data["train_state"].get("modelnorm_input_baseline",None)
print(f"Model norm normal baseline updating: {old_modelnorm_normal_baseline} -> {modelnorm_normal_baseline}")
print(f"Model norm input baseline updating: {old_modelnorm_input_baseline} -> {modelnorm_input_baseline}")
data["train_state"]["modelnorm_normal_baseline"] = modelnorm_normal_baseline
data["train_state"]["modelnorm_input_baseline"] = modelnorm_input_baseline

raw_model_params = {}
for name, param in raw_model.named_parameters():
    raw_model_params[name] = param
for name, param in raw_model.named_buffers():
    raw_model_params[name] = param

for param_name in data["model"]:
    if param_name.startswith("module."):
        stripped_name = param_name[len("module."):]
    else:
        stripped_name = param_name

    if stripped_name in raw_model_params:
        new_tensor = raw_model_params[stripped_name]
    elif ("module." + stripped_name) in raw_model_params:
        new_tensor = raw_model_params["module." + stripped_name]
    else:
        raise AssertionError(f"Could not find param: {stripped_name}")

    old_shape = tuple(data["model"][param_name].shape)
    new_shape = tuple(new_tensor.shape)
    print(f"Expanding {param_name} from {old_shape} to {new_shape}")
    if re.search(r"\.conv\d+\.weight$", stripped_name) or stripped_name.endswith("conv_spatial.weight") or stripped_name.endswith(".conv.weight") or stripped_name.endswith(".conv1r.weight") or stripped_name.endswith(".conv1g.weight") or stripped_name.endswith(".conv1p.weight") or stripped_name.endswith(".conv2p.weight") or stripped_name.endswith(".conv_ownership.weight") or stripped_name.endswith(".conv_scoring.weight") or stripped_name.endswith(".conv_futurepos.weight") or stripped_name.endswith(".conv_seki.weight"):
        data["model"][param_name] = expand_conv_weights(param_name, data["model"][param_name],new_shape)
    elif stripped_name.endswith("linear_g.weight") or stripped_name.endswith("linear_pass.weight") or stripped_name.endswith("value_head.linear2.weight") or stripped_name.endswith("value_head.linear_s2.weight") or stripped_name.endswith("value_head.linear_smix.weight"):
        data["model"][param_name] = expand_mat_weights_after_gpool(param_name, data["model"][param_name],new_shape)
    elif stripped_name.endswith("linear_global.weight") or stripped_name.endswith("linear_pass2.weight") or re.search(r"\.linear\d+\.weight$", stripped_name) or stripped_name.endswith("linear_valuehead.weight") or stripped_name.endswith("linear_miscvaluehead.weight") or stripped_name.endswith("linear_moremiscvaluehead.weight") or stripped_name.endswith("linear_s2off.weight") or stripped_name.endswith("linear_s2par.weight") or stripped_name.endswith("linear_s3.weight"):
        data["model"][param_name] = expand_mat_weights(param_name, data["model"][param_name],new_shape)
    elif stripped_name.endswith(".gamma"):
        data["model"][param_name] = expand_gamma(param_name, data["model"][param_name],new_shape)
    elif stripped_name.endswith(".beta"):
        data["model"][param_name] = expand_beta(param_name, data["model"][param_name],new_shape)
    elif stripped_name.endswith(".bias") or stripped_name.endswith(".running_mean") or stripped_name.endswith(".running_std"):
        data["model"][param_name] = expand_bias(param_name, data["model"][param_name],new_shape)
    else:
        raise AssertionError(f"Conversion not implemented for: {param_name} {old_shape} {new_shape}")

if "optimizer" in data:
    print("Deleting optimizer state")
    del data["optimizer"]
if "swa_model" in data:
    print("Deleting swa model state")
    del data["swa_model"]

for field in data["config"]:
    if data["config"][field] != new_config[field]:
        old_value = data["config"][field]
        new_value = new_config[field]
        print(f"Updating config field {field} from {old_value} -> {new_value}")
        data["config"][field] = new_config[field]

print(f"Saving to {output_path}")
torch.save(data, output_path)
print("Done")
