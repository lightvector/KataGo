#!/usr/bin/python3
import sys
import os
import argparse
import math
import torch

import katago.train.load_model

description = """
Hacky script to migrate v15 model to v16 with q prediction head.
Run on a torch checkpoint.ckpt file. Replace the old one with the new one after a backup.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
parser.add_argument('-new-p1-channels', help='New larger value for p1 and g1 num channels', type=int, required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]
new_p1 = args["new_p1_channels"]
new_g1 = new_p1

data = torch.load(checkpoint_path,map_location="cpu")
assert data["config"]["version"] == 15, data["config"]["version"]

def noisify(tensor, additive_scale):
    return tensor * (0.98 + 0.02 * torch.rand(tensor.shape, device=tensor.device)) + additive_scale * (2.0 * torch.rand(tensor.shape, device=tensor.device) - 1.0)

def print_channel_rms(name, tensor):
    out_channels = tensor.shape[0]
    for i in range(out_channels):
        rms = torch.sqrt(torch.mean(torch.square(tensor[i]))).item()
        print(f"{name=} out channel {i} rms {rms:.6f}")

    if len(tensor.shape) > 1:
        in_channels = tensor.shape[1]
        for j in range(in_channels):
            rms = torch.sqrt(torch.mean(torch.square(tensor[:, j]))).item()
            print(f"{name=} in channel {j} rms {rms:.6f}")


def pad_first_two_dims(tensor, dim0_pad, dim1_pad, new_dim0_scale):
    shape = tensor.shape
    assert len(shape) >= 1

    if len(shape) == 1:
        assert dim1_pad == 0

        new_shape = list(shape)
        new_shape[0] += dim0_pad

        new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
        # Copy the original tensor into the new tensor
        new_tensor[:shape[0]] = tensor

        # Initialize the new padded elements in dimension 0 with uniform random values
        if dim0_pad > 0:
            new_tensor[shape[0]:] = torch.empty(dim0_pad, dtype=tensor.dtype, device=tensor.device).uniform_(-new_dim0_scale, new_dim0_scale)

        return new_tensor
    else:
        new_shape = list(shape)
        new_shape[0] += dim0_pad
        new_shape[1] += dim1_pad

        new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
        # Copy the original tensor into the new tensor
        new_tensor[:shape[0], :shape[1]] = tensor

        # Initialize the new padded elements in dimension 0 with uniform random values
        if dim0_pad > 0:
            random_rows = torch.empty(dim0_pad, new_shape[1], *shape[2:], dtype=tensor.dtype, device=tensor.device).uniform_(-new_dim0_scale, new_dim0_scale)
            new_tensor[shape[0]:] = random_rows

        return new_tensor

# Torch output channel dimension is dim 0.
def expand_policy_dim(name, tensor, old_out, old_in, new_out, new_in, near_zero_new_out_dims, chop_in_3_parts):
    # chop_in_3_parts is for gpool since it contatenates in 3 parts so the new dim0s needs to be done in 3 parts
    if chop_in_3_parts:
        assert tensor.shape[1] == old_in
        assert old_in % 3 == 0
        assert new_in % 3 == 0
        old_in_div3 = old_in // 3
        new_in_div3 = new_in // 3
        new_tensor = torch.cat(
            (expand_policy_dim(name + " part0", tensor[:,0:old_in_div3], old_out, old_in_div3, new_out, new_in_div3, near_zero_new_out_dims, False),
             expand_policy_dim(name + " part1", tensor[:,old_in_div3:old_in_div3*2], old_out, old_in_div3, new_out, new_in_div3, near_zero_new_out_dims, False),
             expand_policy_dim(name + " part2", tensor[:,old_in_div3*2:], old_out, old_in_div3, new_out, new_in_div3, near_zero_new_out_dims, False),
            ), dim=1
        )
        assert new_tensor.shape[1] == new_in
        return new_tensor

    print(f"old {name=} {tensor.shape=}")
    assert tensor.shape[0] == old_out
    assert (len(tensor.shape) == 1 and old_in == 1 and new_in == 1) or tensor.shape[1] == old_in

    assert new_out >= old_out
    assert new_in >= old_in

    old_rms = math.sqrt(torch.mean(torch.square(tensor)).item())
    new_dim0_scale = 0.0 if near_zero_new_out_dims else old_rms

    new_tensor = pad_first_two_dims(tensor, new_out - old_out, new_in - old_in, new_dim0_scale=new_dim0_scale)
    new_tensor = noisify(new_tensor, additive_scale=old_rms*0.01)

    print(f"new {name=} {new_tensor.shape=}")
    print_channel_rms("old " + name, tensor)
    print_channel_rms("new " + name, new_tensor)
    return new_tensor


def expand_policy_dim_for(name, old_out, old_in, new_out, new_in, near_zero_new_out_dims, chop_in_3_parts=False):
    if f"module.{name}" in data["model"]:
        data["model"][f"module.{name}"] = expand_policy_dim(f"module.{name}", data["model"][f"module.{name}"], old_out, old_in, new_out, new_in, near_zero_new_out_dims, chop_in_3_parts)
    elif name in data["model"]:
        data["model"][name] = expand_policy_dim(name, data["model"][name], old_out, old_in, new_out, new_in, near_zero_new_out_dims, chop_in_3_parts)
    else:
        assert False, f"{name} not found in saved model"

trunk_c = data["config"]["trunk_num_channels"]
old_p1 = data["config"]["p1_num_channels"]
old_g1 = data["config"]["g1_num_channels"]

expand_policy_dim_for("policy_head.conv1p.weight", old_p1, trunk_c, new_p1, trunk_c, near_zero_new_out_dims=False)
expand_policy_dim_for("policy_head.conv1g.weight", old_g1, trunk_c, new_g1, trunk_c, near_zero_new_out_dims=False)
expand_policy_dim_for("policy_head.biasg.beta", 1, old_g1, 1, new_g1, near_zero_new_out_dims=True)
expand_policy_dim_for("policy_head.linear_g.weight", old_p1, 3 * old_g1, new_p1, 3 * new_g1, near_zero_new_out_dims=False, chop_in_3_parts=True)
expand_policy_dim_for("policy_head.linear_pass.weight", old_p1, 3 * old_g1, new_p1, 3 * new_g1, near_zero_new_out_dims=False, chop_in_3_parts=True)
expand_policy_dim_for("policy_head.linear_pass.bias", old_p1, 1, new_p1, 1, near_zero_new_out_dims=True)
expand_policy_dim_for("policy_head.bias2.beta", 1, old_p1, 1, new_p1, near_zero_new_out_dims=True)
expand_policy_dim_for("policy_head.conv2p.weight", 6, old_p1, 8, new_p1, near_zero_new_out_dims=True)
expand_policy_dim_for("policy_head.linear_pass2.weight", 6, old_p1, 8, new_p1, near_zero_new_out_dims=True)

if any("intermediate_policy_head" in key for key in data["model"].keys()):

    expand_policy_dim_for("intermediate_policy_head.conv1p.weight", old_p1, trunk_c, new_p1, trunk_c, near_zero_new_out_dims=False)
    expand_policy_dim_for("intermediate_policy_head.conv1g.weight", old_g1, trunk_c, new_g1, trunk_c, near_zero_new_out_dims=False)
    expand_policy_dim_for("intermediate_policy_head.biasg.beta", 1, old_g1, 1, new_g1, near_zero_new_out_dims=True)
    expand_policy_dim_for("intermediate_policy_head.linear_g.weight", old_p1, 3 * old_g1, new_p1, 3 * new_g1, near_zero_new_out_dims=False, chop_in_3_parts=True)
    expand_policy_dim_for("intermediate_policy_head.linear_pass.weight", old_p1, 3 * old_g1, new_p1, 3 * new_g1, near_zero_new_out_dims=False, chop_in_3_parts=True)
    expand_policy_dim_for("intermediate_policy_head.linear_pass.bias", old_p1, 1, new_p1, 1, near_zero_new_out_dims=True)
    expand_policy_dim_for("intermediate_policy_head.bias2.beta", 1, old_p1, 1, new_p1, near_zero_new_out_dims=True)
    expand_policy_dim_for("intermediate_policy_head.conv2p.weight", 6, old_p1, 8, new_p1, near_zero_new_out_dims=True)
    expand_policy_dim_for("intermediate_policy_head.linear_pass2.weight", 6, old_p1, 8, new_p1, near_zero_new_out_dims=True)

if "optimizer" in data:
    print("Deleting optimizer state")
    del data["optimizer"]
if "swa_model" in data:
    print("Deleting swa model state")
    del data["swa_model"]
print("Setting version to 16")
data["config"]["version"] = 16
data["config"]["p1_num_channels"] = new_p1
data["config"]["g1_num_channels"] = new_g1

print(f"Saving to {output_path}")
torch.save(data, output_path)
