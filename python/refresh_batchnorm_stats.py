#!/usr/bin/python3
"""
Refresh (exactly recompute) the batchnorm running statistics of a checkpoint over
a sample of shuffled data, for both the plain model and the SWA model, and write
out a new checkpoint that is otherwise identical.

Why: batchnorm running_mean/running_std are an EWMA over recent training batches, so they lag
the current weights' true statistics. This inflates eval-mode losses in a way that varies
erratically from checkpoint to checkpoint.

This affects SWA too, and AveragedModel copies the raw model's instantaneous buffers
rather than averaging them.

This script run forwards in train mode while accumulating the exact masked population
mean/variance of each batchnorm layer's input. Then overwrites running_mean with the pooled mean
and running_std with sqrt(pooled_var + epsilon), matching the convention that
running_std is an EWMA of sqrt(batch_var + epsilon).

Also evaluates losses on the same data in various modes, paired on identical batches,
printing full losses in the same format as train.py and a small summary table:
* train mode (batch stats),
* eval mode with the original stored stats
* eval mode with the refreshed stats
"""
import sys
import os
import math
import argparse
import logging
import hashlib
import datetime
from collections import defaultdict

import torch
from torch.optim.swa_utils import AveragedModel

from katago.train import modelconfigs
from katago.train.model_pytorch import Model, NormMask, ExtraOutputs
from katago.train.metrics_pytorch import Metrics
from katago.train import data_processing_pytorch
from katago.train.load_model import load_checkpoint, load_model_state_dict, load_swa_model_state_dict
from katago.train.metrics_logging import accumulate_metrics, log_metrics
from katago.train.trainloop_helpers import detensorify_metrics

# Match train.py's defaults so the reported aggregate "loss" is comparable.
SOFT_POLICY_WEIGHT_SCALE = 8.0
VALUE_LOSS_SCALE = 0.6
TD_VALUE_LOSS_SCALES = [0.6, 0.6, 0.6]
SEKI_LOSS_SCALE = 1.0
VARIANCE_TIME_LOSS_SCALE = 1.0

SUMMARY_TABLE_KEYS = ["p0loss", "p0sopt", "vloss", "tdvloss1", "leadloss", "oloss"]


def find_batchnorm_modules(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, NormMask) and module.is_using_batchnorm]


def get_bn_buffers(model):
    return {name: buf for name, buf in model.named_buffers() if "running_" in name}


def snapshot_bn_buffers(model):
    return {name: buf.detach().clone() for name, buf in get_bn_buffers(model).items()}


def load_bn_buffers(model, values):
    # values may cover only a subset of the buffers (e.g. fresh stats cover
    # running_mean/running_std but not brenorm's auxiliary buffers).
    with torch.no_grad():
        for name, buf in get_bn_buffers(model).items():
            if name in values:
                buf.copy_(values[name])


def compute_fresh_stats(model, bn_modules, make_reader, forward_fn, num_batches, device):
    """
    Exact masked population mean/std of each batchnorm layer's input, from train-mode forwards.
    Returns {module_name: (mean, std)} as buffer-style dicts keyed like snapshot_bn_buffers.
    Restores the model's original buffers (train mode forwards update them as a side effect).
    """
    accum = {
        name: {"s1": None, "s2": None, "n": torch.zeros((), dtype=torch.float64, device=device)}
        for name, _ in bn_modules
    }
    handles = []

    def make_hook(name):
        def hook(module, args, kwargs):
            x = args[0] if len(args) > 0 else kwargs["x"]
            mask = kwargs["mask"] if "mask" in kwargs else args[1]
            mask_sum = kwargs["mask_sum"] if "mask_sum" in kwargs else args[3]
            xf = x.float()
            b1 = torch.sum(xf * mask, dim=(0, 2, 3)).double()
            b2 = torch.sum(torch.square(xf) * mask, dim=(0, 2, 3)).double()
            a = accum[name]
            if a["s1"] is None:
                a["s1"], a["s2"] = b1, b2
            else:
                a["s1"] += b1
                a["s2"] += b2
            a["n"] += float(mask_sum)
        return hook

    for name, module in bn_modules:
        handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))

    orig_buffers = snapshot_bn_buffers(model)
    model.train()
    count = 0
    with torch.no_grad():
        for batch in make_reader():
            if count >= num_batches:
                break
            forward_fn(model, batch)
            count += 1
    for handle in handles:
        handle.remove()
    load_bn_buffers(model, orig_buffers)
    model.eval()
    assert count > 0, "No data batches read for stats"

    fresh = {}
    for name, module in bn_modules:
        a = accum[name]
        mean = (a["s1"] / a["n"]).float()
        var = (a["s2"] / a["n"]).float() - torch.square(mean)
        std = torch.sqrt(var.clamp(min=0.0) + module.epsilon)
        fresh[name + ".running_mean"] = mean
        fresh[name + ".running_std"] = std
        rm = module.running_mean.float()
        rs = module.running_std.float()
        logging.info(
            f"  {name}: refreshed from {count} batches, |fresh_mean - old_mean| avg = "
            f"{(mean - rm).abs().mean().item():.5f} (mean scale {rm.abs().mean().item():.4f}), "
            f"fresh/old std ratio mean = {(std / rs).mean().item():.4f} "
            f"min = {(std / rs).min().item():.4f} max = {(std / rs).max().item():.4f}"
        )
    return fresh


def log_input_checkpoint_info(path, state_dict):
    """Log identifying info about the original checkpoint being refreshed."""
    st = os.stat(path)
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    mtime = datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")
    logging.info(f"Input checkpoint: {path}")
    logging.info(f"  size = {st.st_size} bytes ({st.st_size / 1024**3:.3f} GiB)")
    logging.info(f"  mtime = {mtime}")
    logging.info(f"  sha256 = {sha.hexdigest()}")
    train_state = state_dict.get("train_state")
    if not isinstance(train_state, dict):
        logging.info("  train_state: (not present in checkpoint)")
        return
    for key in ["global_step_samples", "total_num_data_rows", "window_start_data_row_idx"]:
        value = train_state.get(key, "(not present)")
        logging.info(f"  train_state.{key} = {value}")


def main(args):
    checkpoint_file = args["checkpoint"]
    output_ckpt = args["output_ckpt"]
    log_file = output_ckpt + ".bnrefresh.log"

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    logging.info(str(sys.argv))

    npzdir = args["npzdir"]
    pos_len = args["pos_len"]
    batch_size = args["batch_size"]
    num_stats_batches = args["num_stats_batches"]
    num_loss_batches = args["num_loss_batches"]
    use_fp16 = args["use_fp16"]
    gpu_idx = args["gpu_idx"]

    assert not os.path.exists(output_ckpt), f"Output already exists: {output_ckpt}"

    if gpu_idx is not None:
        torch.cuda.set_device(gpu_idx)
        device = torch.device("cuda", gpu_idx)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logging.warning("WARNING: No GPU, using CPU")
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    val_files = sorted(os.path.join(npzdir, f) for f in os.listdir(npzdir) if f.endswith(".npz"))
    assert len(val_files) > 0, f"No npz files in {npzdir}"

    # Load the checkpoint once; keep the raw state dict around for the final save.
    state_dict = load_checkpoint(checkpoint_file)
    log_input_checkpoint_info(checkpoint_file, state_dict)
    if "config" in state_dict:
        model_config = state_dict["config"]
    else:
        import json
        config_file = os.path.join(os.path.dirname(checkpoint_file), "model.config.json")
        logging.info(f"No config in checkpoint, so loading from: {config_file}")
        with open(config_file, "r") as f:
            model_config = json.load(f)

    model = Model(model_config, pos_len)
    model.initialize()
    model.load_state_dict(load_model_state_dict(state_dict))
    model.to(device)
    model.eval()

    swa_model = None
    if "swa_model" in state_dict:
        swa_wrapper = AveragedModel(model, device=device)
        swa_wrapper.load_state_dict(load_swa_model_state_dict(state_dict))
        swa_model = swa_wrapper.module
        swa_model.eval()
    else:
        logging.info("Checkpoint has no swa_model, refreshing plain model only")

    models = [("plain", model)] + ([("swa", swa_model)] if swa_model is not None else [])

    bn_modules_by_model = {}
    for mlabel, m in models:
        bn_modules = find_batchnorm_modules(m)
        assert len(bn_modules) > 0, "Model has no batchnorm layers, nothing to refresh"
        bn_modules_by_model[mlabel] = bn_modules
    logging.info(f"Batchnorm layers to refresh: {[name for name, _ in bn_modules_by_model['plain']]}")

    metrics_obj = Metrics(1, model)
    has_meta = model.get_has_metadata_encoder()
    has_intermediate = model.get_has_intermediate_head()

    def make_reader():
        return data_processing_pytorch.read_npz_training_data(
            val_files, batch_size, world_size=1, rank=0, pos_len=pos_len,
            device=device, randomize_symmetries=False, include_meta=has_meta,
            model_config=model_config,
        )

    def forward_fn(m, batch):
        extra_outputs = ExtraOutputs([])
        with torch.autocast("cuda", enabled=use_fp16):
            return m(
                batch["binaryInputNCHW"],
                batch["globalInputNC"],
                input_meta=(batch["metadataInputNC"] if has_meta else None),
                extra_outputs=extra_outputs,
            )

    def batch_metrics(m, batch):
        model_outputs = forward_fn(m, batch)
        postprocessed = m.postprocess_output(model_outputs)
        return metrics_obj.metrics_dict_batchwise(
            m, postprocessed, ExtraOutputs([]), batch,
            is_training=False,
            soft_policy_weight_scale=SOFT_POLICY_WEIGHT_SCALE,
            disable_optimistic_policy=False,
            meta_kata_only_soft_policy=False,
            value_loss_scale=VALUE_LOSS_SCALE,
            td_value_loss_scales=TD_VALUE_LOSS_SCALES,
            seki_loss_scale=SEKI_LOSS_SCALE,
            variance_time_loss_scale=VARIANCE_TIME_LOSS_SCALE,
            main_loss_scale=1.0,
            intermediate_loss_scale=(1.0 if has_intermediate else None),
        )

    # Compute the refreshed statistics.
    orig_buffers = {}
    fresh_buffers = {}
    for mlabel, m in models:
        logging.info(f"Computing fresh batchnorm stats for {mlabel} model over {num_stats_batches} batches of {batch_size}:")
        orig_buffers[mlabel] = snapshot_bn_buffers(m)
        fresh_buffers[mlabel] = compute_fresh_stats(
            m, bn_modules_by_model[mlabel], make_reader, forward_fn, num_stats_batches, device
        )

    # Evaluate losses under all conditions, paired on identical batches.
    conditions = [("trainmode", "train"), ("evalmode-orig", "orig"), ("evalmode-refreshed", "fresh")]
    if num_loss_batches > 0:
        sums = defaultdict(lambda: defaultdict(float))
        weights = defaultdict(lambda: defaultdict(float))
        num_batches = 0
        with torch.no_grad():
            for batch in make_reader():
                if num_batches >= num_loss_batches:
                    break
                for mlabel, m in models:
                    for clabel, ckind in conditions:
                        if ckind == "train":
                            m.train()
                        else:
                            m.eval()
                            load_bn_buffers(m, fresh_buffers[mlabel] if ckind == "fresh" else orig_buffers[mlabel])
                        metrics = detensorify_metrics(batch_metrics(m, batch))
                        key = (mlabel, clabel)
                        accumulate_metrics(sums[key], weights[key], metrics, batch_size, decay=1.0, new_weight=1.0)
                        if ckind == "train":
                            # Undo buffer EWMA updates from the train-mode forward
                            load_bn_buffers(m, orig_buffers[mlabel])
                            m.eval()
                num_batches += 1
                if num_batches % 25 == 0:
                    logging.info(f"Evaluated losses on {num_batches}/{num_loss_batches} batches")

        logging.info("")
        logging.info(f"Losses on {num_batches * batch_size} samples from {npzdir}:")
        for mlabel, _ in models:
            for clabel, _ in conditions:
                key = (mlabel, clabel)
                logging.info(f"----- {mlabel} model, {clabel} -----")
                log_metrics(sums[key], weights[key], {}, None)

        def fmt_summary_table(keys, label):
            avail = [k for k in keys if any((k + "_sum") in sums[key] for key in sums)]
            if not avail:
                return
            logging.info("")
            logging.info(f"Summary ({label}):")
            header = f"{'condition':>28} " + " ".join(f"{k:>9}" for k in avail)
            logging.info(header)
            for mlabel, _ in models:
                for clabel, _ in conditions:
                    key = (mlabel, clabel)
                    row = f"{mlabel + ' ' + clabel:>28} "
                    for k in avail:
                        sk = k + "_sum"
                        row += f"{sums[key][sk] / weights[key][sk]:9.5f} " if weights[key][sk] > 0 else f"{'n/a':>9} "
                    logging.info(row)

        fmt_summary_table(SUMMARY_TABLE_KEYS, "main heads")
        if has_intermediate:
            fmt_summary_table(["I" + k for k in SUMMARY_TABLE_KEYS], "intermediate heads - on -bnh models these are the batchnorm-normalized heads")

    # Ensure the in-memory models are left with refreshed buffers (not needed for
    # the save below, which edits the raw state dict, but tidy if this is imported).
    for mlabel, m in models:
        load_bn_buffers(m, fresh_buffers[mlabel])

    # Write the output checkpoint: identical except for the batchnorm buffers.
    def strip_prefixes(key):
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key[len("module."):] if key.startswith("module.") else key[len("_orig_mod."):]
        return key

    def overwrite_bn_keys(raw_sd, fresh, what):
        num_replaced = 0
        for raw_key in raw_sd:
            clean = strip_prefixes(raw_key)
            if clean in fresh:
                raw_sd[raw_key] = fresh[clean].detach().to(raw_sd[raw_key].dtype).cpu()
                num_replaced += 1
        assert num_replaced == len(fresh), \
            f"Expected to replace {len(fresh)} buffers in {what} but replaced {num_replaced}"
        logging.info(f"Replaced {num_replaced} batchnorm buffers in {what}")

    overwrite_bn_keys(state_dict["model"], fresh_buffers["plain"], "model")
    if swa_model is not None:
        overwrite_bn_keys(state_dict["swa_model"], fresh_buffers["swa"], "swa_model")

    torch.save(state_dict, output_ckpt)
    logging.info(f"Wrote checkpoint with refreshed batchnorm stats to: {output_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-checkpoint', help='Checkpoint file to refresh', required=True)
    parser.add_argument('-output-ckpt', help='Path to write the refreshed checkpoint', required=True)
    parser.add_argument('-npzdir', help='Directory with shuffled npz data to compute stats and losses on', required=True)
    parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, default=19, required=False)
    parser.add_argument('-batch-size', help='Batch size for stats and loss evaluation', type=int, default=256, required=False)
    parser.add_argument('-num-stats-batches', help='Number of batches for computing fresh stats', type=int, default=200, required=False)
    parser.add_argument('-num-loss-batches', help='Number of batches for the loss comparison, 0 to skip it', type=int, default=200, required=False)
    parser.add_argument('-use-fp16', help='Run forwards under fp16 autocast, matching training with -use-fp16', action="store_true", required=False)
    parser.add_argument('-gpu-idx', help='GPU idx', type=int, required=False)
    main(vars(parser.parse_args()))
