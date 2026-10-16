#!/usr/bin/python3
import os
import argparse
import shutil
import torch
from collections import defaultdict

description = """
Migrate a checkpoint that contains defaultdict values in running_metrics or
last_val_metrics to use plain dicts instead, making the checkpoint loadable
with weights_only=True in PyTorch 2.6+.

Operates in-place, saving the original to <checkpoint>.backup first.
Safe to run on already-migrated checkpoints (idempotent).

Use this if torch.load raises an UnpicklingError about defaultdict.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
backup_path = checkpoint_path + ".backup"

if os.path.exists(backup_path):
    raise RuntimeError(
        f"Backup already exists: {backup_path}\n"
        "This suggests migration has already been run. Refusing to overwrite the backup.\n"
        "If you truly want to re-run, remove the backup file first."
    )

data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

def convert_defaultdicts(d):
    """Recursively convert any defaultdict values to plain dicts (idempotent on plain dicts)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, defaultdict):
            result[k] = dict(v)
        elif isinstance(v, dict):
            result[k] = convert_defaultdicts(v)
        else:
            result[k] = v
    return result

if "running_metrics" in data:
    data["running_metrics"] = convert_defaultdicts(data["running_metrics"])

if "last_val_metrics" in data:
    data["last_val_metrics"] = convert_defaultdicts(data["last_val_metrics"])

shutil.copy2(checkpoint_path, backup_path)
print(f"Backed up {checkpoint_path} -> {backup_path}")

torch.save(data, checkpoint_path)
print(f"Migrated {checkpoint_path} in-place")
