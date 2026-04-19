#!/usr/bin/python3
import sys
import os
import argparse
import time
import math
import numpy as np

import torch
import torch._dynamo
torch._dynamo.config.recompile_limit = 32
import torch.nn

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics
from katago.train import data_processing_pytorch


def main():
    description = """
    Benchmark a fresh randomly-initialized model. Reports parameter counts by tensor,
    then measures forward-pass and forward+backward+optimizer-step timing.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-model-kind', help='Model config name, e.g. b8c192nbt-fson-mish-rvglr-bnh', required=True)
    parser.add_argument('-optimizer', help='Optimizer to use', choices=['sgd', 'adam', 'muon'], default='sgd')
    parser.add_argument('-batch-size', help='Batch size', type=int, required=True)
    parser.add_argument('-data', help='Path to npz data file (e.g. ../python/testdata/benchmark_data_1024.npz)', required=True)
    parser.add_argument('-gpu', help='GPU device index', type=int, default=0)
    parser.add_argument('-pos-len', help='Board size', type=int, default=19)
    parser.add_argument('-num-iters', help='Number of benchmark iterations', type=int, default=20)
    parser.add_argument('-warmup-iters', help='Number of warmup iterations', type=int, default=5)
    parser.add_argument('-print-per-tensor-counts', help='Print parameter counts per tensor', action='store_true')
    parser.add_argument('-no-compile', help='Do not torch.compile', action='store_true')
    parser.add_argument('-use-tf32-matmul', help='Reduce float32 precision for speed on some gpus', action='store_true')
    parser.add_argument('-override-config', help='Override model config params, e.g. "gab_d1=16,tab_num_freqs=8"', type=str, default=None)
    args = vars(parser.parse_args())

    model_kind = args["model_kind"]
    optimizer_kind = args["optimizer"]
    batch_size = args["batch_size"]
    data_path = args["data"]
    gpu_idx = args["gpu"]
    pos_len = args["pos_len"]
    num_iters = args["num_iters"]
    warmup_iters = args["warmup_iters"]
    print_per_tensor = args["print_per_tensor_counts"]
    no_compile = args["no_compile"]
    use_tf32_matmul = args["use_tf32_matmul"]
    override_config_str = args["override_config"]

    device = torch.device(f"cuda:{gpu_idx}")

    if use_tf32_matmul:
        torch.set_float32_matmul_precision('high')
        print("float32 matmul precision: high (TF32)")
    else:
        print("float32 matmul precision: default")
    torch.cuda.set_device(device)

    # Load model config and create model
    assert model_kind in modelconfigs.config_of_name, f"Unknown model kind: {model_kind}, available: {list(modelconfigs.config_of_name.keys())}"
    model_config = modelconfigs.config_of_name[model_kind].copy()

    # Apply config overrides
    if override_config_str:
        for kv in override_config_str.split(","):
            kv = kv.strip()
            if not kv:
                continue
            key, val_str = kv.split("=", 1)
            key = key.strip()
            val_str = val_str.strip()
            if key in model_config:
                orig = model_config[key]
                if isinstance(orig, bool):
                    model_config[key] = val_str.lower() in ("true", "1", "yes")
                elif isinstance(orig, int):
                    model_config[key] = int(val_str)
                elif isinstance(orig, float):
                    model_config[key] = float(val_str)
                elif isinstance(orig, str):
                    model_config[key] = val_str
                else:
                    import json
                    model_config[key] = json.loads(val_str)
                print(f"Config override: {key} = {model_config[key]} (was {orig})")
            else:
                # New key: infer type from value string
                if val_str.lower() in ("true", "false"):
                    model_config[key] = val_str.lower() == "true"
                else:
                    try:
                        model_config[key] = int(val_str)
                    except ValueError:
                        try:
                            model_config[key] = float(val_str)
                        except ValueError:
                            model_config[key] = val_str
                print(f"Config override (new): {key} = {model_config[key]}")

    print(f"Model kind: {model_kind}")
    print(f"Optimizer: {optimizer_kind}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print()

    raw_model = Model(model_config, pos_len)
    raw_model.initialize()
    raw_model.to(device)

    if no_compile:
        print("torch.compile: disabled (-no-compile)")
        model = raw_model
    else:
        print("torch.compile: enabled (mode=default)")
        model = torch.compile(raw_model, mode="default")
    print()

    # Report parameter counts
    print("=" * 80)
    print("PARAMETER COUNTS")
    print("=" * 80)
    total_params = 0
    for name, param in raw_model.named_parameters():
        n = param.numel()
        total_params += n
        if print_per_tensor:
            print(f"  {n:>12,}  {str(list(param.shape)):>30s}  {name}")
    if print_per_tensor:
        print()
    print(f"  Total: {total_params:,} parameters")
    print()

    # Also report by reg group
    reg_dict = {}
    raw_model.add_reg_dict(reg_dict)
    print("Parameters by group:")
    for group_name in reg_dict:
        group_params = sum(p.numel() for p in reg_dict[group_name])
        if group_params > 0:
            print(f"  {group_name:>20s}: {group_params:>12,}")
    print()

    # Set up optimizer
    param_groups = []
    for group_name in reg_dict:
        if len(reg_dict[group_name]) > 0:
            is_muon_suitable = group_name in ("normal", "normal_attn", "normal_gab", "gab_mlp")
            param_groups.append({
                "params": reg_dict[group_name],
                "group_name": group_name,
                "lr": 1e-5,
                "weight_decay": 0.01,
                "use_muon": is_muon_suitable,
            })

    if optimizer_kind == "adam":
        optimizer = torch.optim.AdamW(param_groups, lr=1e-5)
    elif optimizer_kind == "muon":
        from muon.muon import SingleDeviceMuonWithAuxAdam
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups, adjust_lr_fn="match_rms_adamw")
    else:
        optimizer = torch.optim.SGD(param_groups, lr=1e-5, momentum=0.9)

    metrics_obj = Metrics(batch_size, 1, raw_model)

    # Load data
    print(f"Loading data from {data_path} ...")
    batch = load_batch(data_path, batch_size, pos_len, model_config, device)
    print(f"Data loaded, batch size = {batch_size}")
    print()

    # Set model to training mode
    raw_model.train()

    # Benchmark forward only
    print("=" * 80)
    print("FORWARD PASS BENCHMARK")
    print("=" * 80)
    forward_times = benchmark_forward(model, batch, num_iters, warmup_iters)
    print_timing_stats("Forward", forward_times)
    print()
    torch.cuda.empty_cache()

    # Benchmark forward + backward + optimizer step with attribution
    print("=" * 80)
    print("FORWARD + BACKWARD + OPTIMIZER STEP BENCHMARK")
    print("=" * 80)
    fwd_times, bwd_times, opt_times = benchmark_full_step(
        model, raw_model, optimizer, metrics_obj, batch, model_config, num_iters, warmup_iters,
    )
    print_timing_stats("Forward ", fwd_times)
    print_timing_stats("Backward", bwd_times)
    print_timing_stats("Opt step", opt_times)
    total_times = [f + b + o for f, b, o in zip(fwd_times, bwd_times, opt_times)]
    print_timing_stats("Total   ", total_times)
    print()
    torch.cuda.empty_cache()

    # Print proportions
    mean_fwd = sum(fwd_times) / len(fwd_times)
    mean_bwd = sum(bwd_times) / len(bwd_times)
    mean_opt = sum(opt_times) / len(opt_times)
    mean_total = mean_fwd + mean_bwd + mean_opt
    print(f"  Time attribution (with sync between phases):")
    print(f"    Forward:  {mean_fwd*1000:8.2f} ms  ({100*mean_fwd/mean_total:5.1f}%)")
    print(f"    Backward: {mean_bwd*1000:8.2f} ms  ({100*mean_bwd/mean_total:5.1f}%)")
    print(f"    Opt step: {mean_opt*1000:8.2f} ms  ({100*mean_opt/mean_total:5.1f}%)")
    print(f"    Total:    {mean_total*1000:8.2f} ms")
    print()

    # Benchmark true throughput without intermediate syncs
    print("=" * 80)
    print("FULL STEP THROUGHPUT (no intermediate sync)")
    print("=" * 80)
    throughput_times = benchmark_full_step_throughput(
        model, raw_model, optimizer, metrics_obj, batch, model_config, num_iters, warmup_iters,
    )
    print_timing_stats("Total   ", throughput_times)
    print()


def load_batch(data_path, batch_size, pos_len, model_config, device):
    """Load a single batch from an npz file."""
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    include_qvalues = model_config["version"] >= 16

    with np.load(data_path) as npz:
        binaryInputNCHWPacked = npz["binaryInputNCHWPacked"][:batch_size]
        globalInputNC = npz["globalInputNC"][:batch_size]
        policyTargetsNCMove = npz["policyTargetsNCMove"][:batch_size].astype(np.float32)
        globalTargetsNC = npz["globalTargetsNC"][:batch_size]
        scoreDistrN = npz["scoreDistrN"][:batch_size].astype(np.float32)
        valueTargetsNCHW = npz["valueTargetsNCHW"][:batch_size].astype(np.float32)
        if include_qvalues and "qValueTargetsNCMove" in npz:
            qValueTargetsNCMove = npz["qValueTargetsNCMove"][:batch_size].astype(np.float32)
        else:
            qValueTargetsNCMove = None

    binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked, axis=2)
    assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
    binaryInputNCHW = binaryInputNCHW[:, :, :pos_len * pos_len]
    binaryInputNCHW = np.reshape(binaryInputNCHW, (
        binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
    )).astype(np.float32)

    assert binaryInputNCHW.shape[1] == num_bin_features
    assert globalInputNC.shape[1] == num_global_features

    (h_base, h_builder) = data_processing_pytorch.build_history_matrices(model_config, device)

    batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW).to(device)
    batch_globalInputNC = torch.from_numpy(globalInputNC).to(device)
    batch_globalTargetsNC = torch.from_numpy(globalTargetsNC).to(device)

    (batch_binaryInputNCHW, batch_globalInputNC) = data_processing_pytorch.apply_history_matrices(
        model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
    )

    batch = dict(
        binaryInputNCHW=batch_binaryInputNCHW.contiguous(),
        globalInputNC=batch_globalInputNC,
        policyTargetsNCMove=torch.from_numpy(policyTargetsNCMove).to(device),
        globalTargetsNC=batch_globalTargetsNC,
        scoreDistrN=torch.from_numpy(scoreDistrN).to(device),
        valueTargetsNCHW=torch.from_numpy(valueTargetsNCHW).to(device),
    )
    if qValueTargetsNCMove is not None:
        batch["qValueTargetsNCMove"] = torch.from_numpy(qValueTargetsNCMove).to(device)
    return batch


def benchmark_forward(model, batch, num_iters, warmup_iters):
    """Benchmark forward pass only."""
    times = []
    for i in range(warmup_iters + num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            model_outputs = model(
                batch["binaryInputNCHW"],
                batch["globalInputNC"],
            )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        del model_outputs

        if i >= warmup_iters:
            times.append(t1 - t0)
    return times


def benchmark_full_step(model, raw_model, optimizer, metrics_obj, batch, model_config, num_iters, warmup_iters):
    """Benchmark forward + backward + optimizer step, returning separate timings."""
    fwd_times = []
    bwd_times = []
    opt_times = []

    for i in range(warmup_iters + num_iters):
        optimizer.zero_grad(set_to_none=True)

        # Forward
        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()

        model_outputs = model(
            batch["binaryInputNCHW"],
            batch["globalInputNC"],
        )
        postprocessed = raw_model.postprocess_output(model_outputs)
        metrics = metrics_obj.metrics_dict_batchwise(
            raw_model,
            postprocessed,
            extra_outputs=None,
            batch=batch,
            is_training=True,
            soft_policy_weight_scale=1.0,
            disable_optimistic_policy=False,
            meta_kata_only_soft_policy=False,
            value_loss_scale=1.0,
            td_value_loss_scales=[0.4, 1.0, 1.0],
            seki_loss_scale=0.35,
            variance_time_loss_scale=0.5,
            main_loss_scale=1.0,
            intermediate_loss_scale=0.5 if raw_model.get_has_intermediate_head() else None,
        )
        loss = metrics["loss_sum"]

        torch.cuda.synchronize()
        t_bwd_start = time.perf_counter()

        # Backward
        loss.backward()

        torch.cuda.synchronize()
        t_opt_start = time.perf_counter()

        # Optimizer step
        optimizer.step()

        torch.cuda.synchronize()
        t_opt_end = time.perf_counter()
        del model_outputs, postprocessed, metrics, loss

        if i >= warmup_iters:
            fwd_times.append(t_bwd_start - t_fwd_start)
            bwd_times.append(t_opt_start - t_bwd_start)
            opt_times.append(t_opt_end - t_opt_start)

    return fwd_times, bwd_times, opt_times


def benchmark_full_step_throughput(model, raw_model, optimizer, metrics_obj, batch, model_config, num_iters, warmup_iters):
    """Benchmark full training step without intermediate syncs, for true throughput measurement."""
    times = []

    for i in range(warmup_iters + num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        model_outputs = model(
            batch["binaryInputNCHW"],
            batch["globalInputNC"],
        )
        postprocessed = raw_model.postprocess_output(model_outputs)
        metrics = metrics_obj.metrics_dict_batchwise(
            raw_model,
            postprocessed,
            extra_outputs=None,
            batch=batch,
            is_training=True,
            soft_policy_weight_scale=1.0,
            disable_optimistic_policy=False,
            meta_kata_only_soft_policy=False,
            value_loss_scale=1.0,
            td_value_loss_scales=[0.4, 1.0, 1.0],
            seki_loss_scale=0.35,
            variance_time_loss_scale=0.5,
            main_loss_scale=1.0,
            intermediate_loss_scale=0.5 if raw_model.get_has_intermediate_head() else None,
        )
        loss = metrics["loss_sum"]
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        del model_outputs, postprocessed, metrics, loss

        if i >= warmup_iters:
            times.append(t1 - t0)

    return times


def print_timing_stats(label, times):
    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    lo = min(times)
    hi = max(times)
    print(f"  {label}: {mean*1000:8.2f} ms  (std {std*1000:6.2f} ms, min {lo*1000:8.2f} ms, max {hi*1000:8.2f} ms)")


if __name__ == "__main__":
    main()
