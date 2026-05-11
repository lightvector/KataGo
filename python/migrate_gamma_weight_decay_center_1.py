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
Migrate a checkpoint to use gamma_weight_decay_center_1=True.

When gamma_weight_decay_center_1 is True, the model stores gamma parameters centered
at 0 and adds 1.0 at runtime (i.e. the effective scale is gamma+1.0). Weight decay then
pulls gammas toward 0, which corresponds to an effective scale of 1.0. When the flag is
False or unset, gammas are stored directly as the effective scale (initialized to 1.0),
and weight decay pulls them toward 0 effective scale.

There are two migration scenarios:

1. SHIFT MODE (default): The checkpoint was trained by KataGo (or a fork that matches
   KataGo's convention) WITHOUT gamma_weight_decay_center_1. Gammas are stored as direct
   scale values centered around 1.0. This mode subtracts 1.0 from all gammas so they
   become centered around 0 in the new representation.

   HOW TO VERIFY: Inspect gamma values before migration. They should be centered near 1.0
   (e.g. overall mean ~1.0-1.2, most values between 0.5 and 1.5).

2. CONFIG-ONLY MODE (-config-only): The checkpoint was trained by a fork that already
   used the center-1 convention internally (gammas stored centered at 0, with +1.0 added
   at runtime) but did NOT set gamma_weight_decay_center_1 in the config. This mode only
   sets the config flag without modifying any weights.

   HOW TO VERIFY: Inspect gamma values before migration. They should be centered near 0.0
   (e.g. overall mean ~0.0-0.2, most values between -0.5 and 0.5). If you also have a
   known-good KataGo checkpoint for comparison, its gammas (without the flag set) will be
   centered near 1.0, confirming the difference in convention.

In both modes, optimizer and SWA model state are deleted from the checkpoint, since they
are invalidated by the config change.

To inspect gamma distributions before deciding which mode to use:

    python -c "
    import torch
    data = torch.load('model.ckpt', map_location='cpu', weights_only=False)
    gammas = torch.cat([data['model'][k].float().flatten() for k in data['model'] if k.endswith('.gamma')])
    print(f'mean={gammas.mean():.4f} median={gammas.median():.4f} std={gammas.std():.4f}')
    print(f'near 0 (|v|<0.5): {(gammas.abs()<0.5).float().mean()*100:.1f}%')
    print(f'near 1 (|v-1|<0.5): {((gammas-1).abs()<0.5).float().mean()*100:.1f}%')
    print(f'gamma_weight_decay_center_1: {data.get(\"config\",{}).get(\"gamma_weight_decay_center_1\",\"NOT SET\")}')
    "

If gammas are near 0, use -config-only. If gammas are near 1, use the default shift mode.
"""

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
parser.add_argument('-config-only', action='store_true', help='Only set the config flag without shifting gamma values. Use this when gammas are already stored centered at 0 (e.g. from a fork that assumed center-1 without setting the flag).')
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]
config_only = args["config_only"]

data = katago.train.load_model.load_checkpoint(checkpoint_path)

if data["config"].get("gamma_weight_decay_center_1",False):
    raise ValueError("gamma_weight_decay_center_1 is already True: " + str(data["config"]))

# Print gamma stats so the user can sanity-check the mode choice
gamma_names = [k for k in data["model"] if k.endswith(".gamma")]
if gamma_names:
    all_gammas = torch.cat([data["model"][k].float().flatten() for k in gamma_names])
    mean = all_gammas.mean().item()
    median = all_gammas.median().item()
    near0 = (all_gammas.abs() < 0.5).float().mean().item() * 100
    near1 = ((all_gammas - 1.0).abs() < 0.5).float().mean().item() * 100
    print(f"Gamma stats: {mean=:.4f} {median=:.4f} near_0={near0:.1f}% near_1={near1:.1f}%")

    if config_only and near1 > near0:
        print(f"WARNING: -config-only was specified but gammas appear centered near 1, not 0.")
        print(f"  This suggests the default shift mode may be correct instead.")
        print(f"  Proceeding anyway since -config-only was explicitly requested.")
    elif not config_only and near0 > near1:
        print(f"WARNING: Shift mode but gammas appear already centered near 0, not 1.")
        print(f"  This suggests -config-only may be correct instead.")
        print(f"  Proceeding anyway since shift mode was explicitly requested.")

if config_only:
    print("Config-only mode: setting gamma_weight_decay_center_1=True without modifying gamma values.")
else:
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
if not config_only and "swa_model" in data:
    print("Deleting swa model state")
    del data["swa_model"]

data["config"]["gamma_weight_decay_center_1"] = True

print(f"Saving to {output_path}")
torch.save(data, output_path)
print("Done")
