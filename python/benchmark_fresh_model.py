#!/usr/bin/python3
import sys
import os
import argparse
import time
import math
from collections import defaultdict
import numpy as np

import torch
import torch._dynamo
torch._dynamo.config.recompile_limit = 32
import torch.nn
import torch.distributed
import torch.multiprocessing
from torch.amp import autocast, GradScaler

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics
from katago.train import data_processing_pytorch
from katago.train import trainloop_helpers
from katago.train.metrics_logging import accumulate_metrics


def main():
    description = """
    Benchmark a fresh randomly-initialized model. Reports parameter counts by tensor,
    then measures forward-pass and forward+backward+optimizer-step timing.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-model-kind', help='Model config name, e.g. b8c192nbt-fson-mish-rvglr-bnh', required=True)
    parser.add_argument('-optimizer', help='Optimizer to use', choices=['sgd', 'adam', 'muon', 'aurora'], default='sgd')
    parser.add_argument('-ns-steps', help='Number of Newton-Schulz iterations for muon', type=int, default=5)
    parser.add_argument('-use-polar-express', help='Use Polar Express iteration instead of standard NS5 for muon', action='store_true')
    parser.add_argument('-batch-size', help='Batch size', type=int, required=True)
    parser.add_argument('-data', help='Path to npz data file (e.g. ../python/testdata/benchmark_data_1024.npz)', required=True)
    parser.add_argument('-gpu', help='GPU device index', type=int, default=0)
    parser.add_argument('-pos-len', help='Board size', type=int, default=19)
    parser.add_argument('-num-iters', help='Number of benchmark iterations', type=int, default=20)
    parser.add_argument('-warmup-iters', help='Number of warmup iterations', type=int, default=5)
    parser.add_argument('-print-per-tensor-counts', help='Print parameter counts per tensor', action='store_true')
    parser.add_argument('-no-compile', help='Do not torch.compile', action='store_true')
    parser.add_argument('-use-tf32-matmul', help='Reduce float32 precision for speed on some gpus', action='store_true')
    parser.add_argument('-use-amp', help='Use automatic mixed precision (fp16 autocast + GradScaler) for training benchmark', action='store_true')
    parser.add_argument('-use-bf16', help='Use bf16 AMP (autocast, no GradScaler) for trainloop benchmark. Mutually exclusive with -use-amp', action='store_true')
    parser.add_argument('-use-fp16', help='Cast model to fp16 for forward-only benchmark (inference only, no training)', action='store_true')
    parser.add_argument('-forward-only', help='Only benchmark the forward pass, skip training benchmarks', action='store_true')
    parser.add_argument('-override-config', help='Override model config params, e.g. "gab_d1=16,tab_num_freqs=8"', type=str, default=None)
    parser.add_argument('-mode', help='phases = existing phase-attribution benchmark, trainloop = replicate the full train.py per-batch step for realistic throughput', choices=['phases', 'trainloop'], default='phases')
    parser.add_argument('-multi-gpus', help='Comma-separated GPU ids for DDP trainloop benchmark, e.g. "0,1". Only valid with -mode trainloop', type=str, default=None)
    parser.add_argument('-master-port', help='Localhost port for DDP', type=int, default=23456)
    parser.add_argument('-print-every', help='Emulated train loss print interval in batches for trainloop mode', type=int, default=100)
    parser.add_argument('-attn-logit-penalty-cap', help='Enable the attention logit bound penalty as in train.py, for benchmarking its overhead (trainloop mode)', type=float, default=None)
    parser.add_argument('-attn-logit-penalty-coeff', help='Coeff for -attn-logit-penalty-cap', type=float, default=1e-3)
    parser.add_argument('-attn-logit-penalty-batch-frac', help='Fraction of the batch to compute the penalty on, as in train.py', type=float, default=1.0)
    args = vars(parser.parse_args())

    if args["mode"] == "trainloop":
        run_trainloop_benchmark(args)
        return

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
    use_amp = args["use_amp"]
    use_fp16 = args["use_fp16"]
    forward_only = args["forward_only"]
    override_config_str = args["override_config"]
    ns_steps = args["ns_steps"]
    use_polar_express = args["use_polar_express"]

    if use_amp and use_fp16:
        raise ValueError("-use-amp and -use-fp16 are mutually exclusive")

    if optimizer_kind not in ("muon", "aurora"):
        if ns_steps != 5:
            raise ValueError("-ns-steps can only be used with muon or aurora optimizer")
        if use_polar_express:
            raise ValueError("-use-polar-express can only be used with muon or aurora optimizer")

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
    apply_config_overrides(model_config, override_config_str)

    print(f"Model kind: {model_kind}")
    print(f"Optimizer: {optimizer_kind}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    if use_amp:
        print(f"Precision: AMP (fp16 autocast + GradScaler)")
    elif use_fp16:
        print(f"Precision: fp16 (model in half, output heads in fp32, forward-only)")
    else:
        print(f"Precision: fp32")
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
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups, adjust_lr_fn="match_rms_adamw", ns_steps=ns_steps, use_polar_express=use_polar_express)
    elif optimizer_kind == "aurora":
        from muon.muon import SingleDeviceMuonWithAuxAdam
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups, adjust_lr_fn="match_rms_adamw", use_aurora=True, ns_steps=ns_steps, use_polar_express=use_polar_express)
    else:
        optimizer = torch.optim.SGD(param_groups, lr=1e-5, momentum=0.9)

    metrics_obj = Metrics(1, raw_model)

    # Load data
    print(f"Loading data from {data_path} ...")
    batch = load_batch(data_path, batch_size, pos_len, model_config, device)
    print(f"Data loaded, batch size = {batch_size}")
    print()

    # Set model to training mode
    raw_model.train()

    if use_fp16:
        # Cast entire model to fp16, then restore output heads to fp32.
        # The model's autocast(enabled=False) + .float() guards will convert
        # fp16 trunk outputs to fp32 inputs for the fp32 head weights.
        raw_model.half()
        raw_model.policy_head.float()
        raw_model.value_head.float()
        if hasattr(raw_model, 'intermediate_policy_head'):
            raw_model.intermediate_policy_head.float()
        if hasattr(raw_model, 'intermediate_value_head'):
            raw_model.intermediate_value_head.float()
        # Cast input data to fp16
        batch["binaryInputNCHW"] = batch["binaryInputNCHW"].half()
        batch["globalInputNC"] = batch["globalInputNC"].half()

    # Benchmark forward only
    print("=" * 80)
    print("FORWARD PASS BENCHMARK")
    print("=" * 80)
    forward_times = benchmark_forward(model, batch, num_iters, warmup_iters, use_autocast=use_amp)
    print_timing_stats("Forward", forward_times)
    print()
    torch.cuda.empty_cache()

    if use_fp16 or forward_only:
        if use_fp16:
            print("Skipping training benchmarks (fp16 is forward-only)")
        else:
            print("Skipping training benchmarks (-forward-only)")
        print()
        return

    scaler = GradScaler("cuda") if use_amp else None

    # Benchmark forward + backward + optimizer step with attribution
    print("=" * 80)
    print("FORWARD + BACKWARD + OPTIMIZER STEP BENCHMARK")
    print("=" * 80)
    fwd_times, bwd_times, opt_times = benchmark_full_step(
        model, raw_model, optimizer, metrics_obj, batch, model_config, num_iters, warmup_iters,
        use_amp=use_amp, scaler=scaler,
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
        use_amp=use_amp, scaler=scaler,
    )
    print_timing_stats("Total   ", throughput_times)
    print()


def apply_config_overrides(model_config, override_config_str):
    if not override_config_str:
        return
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


def build_train_param_groups(raw_model):
    """Build optimizer param groups mirroring train.py's get_param_groups.
    Same group partition and use_muon assignment. Fixed nominal lr/wd since only throughput is measured here)."""
    reg_dict = {}
    raw_model.add_reg_dict(reg_dict)
    param_groups = []
    num_reg_dict_params = 0
    for group_name in reg_dict:
        if len(reg_dict[group_name]) > 0:
            is_muon_suitable = group_name in ("normal", "normal_attn", "normal_gab", "gab_mlp", "tab_module")
            param_groups.append({
                "params": reg_dict[group_name],
                "group_name": group_name,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "use_muon": is_muon_suitable,
            })
        num_reg_dict_params += len(reg_dict[group_name])
    num_params = len(list(raw_model.parameters()))
    assert num_params == num_reg_dict_params, "Reg dict does not have entries for all params in model"
    return param_groups


def make_optimizer(optimizer_kind, param_groups, world_size, ns_steps, use_polar_express):
    if optimizer_kind == "adam":
        return torch.optim.AdamW(param_groups, lr=1e-4)
    elif optimizer_kind in ("muon", "aurora"):
        from muon.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
        use_aurora = optimizer_kind == "aurora"
        if world_size > 1:
            return MuonWithAuxAdam(param_groups, adjust_lr_fn="match_rms_adamw", use_aurora=use_aurora, ns_steps=ns_steps, use_polar_express=use_polar_express)
        else:
            return SingleDeviceMuonWithAuxAdam(param_groups, adjust_lr_fn="match_rms_adamw", use_aurora=use_aurora, ns_steps=ns_steps, use_polar_express=use_polar_express)
    else:
        return torch.optim.SGD(param_groups, lr=1e-4, momentum=0.9)


def run_trainloop_benchmark(args):
    if args["forward_only"] or args["use_fp16"]:
        raise ValueError("-mode trainloop is a training benchmark; -forward-only / -use-fp16 are not supported")
    if args["use_amp"] and args["use_bf16"]:
        raise ValueError("-use-amp and -use-bf16 are mutually exclusive")
    if args["multi_gpus"] is not None:
        gpu_ids = [int(x) for x in args["multi_gpus"].split(",")]
    else:
        gpu_ids = [args["gpu"]]
    world_size = len(gpu_ids)
    args = dict(args)
    args["gpu_ids"] = gpu_ids

    if world_size > 1:
        torch.multiprocessing.spawn(
            trainloop_worker,
            nprocs=world_size,
            args=(world_size, args),
        )
    else:
        trainloop_worker(0, 1, args)


def trainloop_worker(rank, world_size, args):
    gpu_ids = args["gpu_ids"]
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)

    def rank0print(*a, **kw):
        if rank == 0:
            print(*a, **kw, flush=True)

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args["master_port"])
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    if args["use_tf32_matmul"]:
        torch.set_float32_matmul_precision('high')
        rank0print("float32 matmul precision: high (TF32)")
    torch.backends.cudnn.benchmark = True
    trainloop_helpers.maybe_enable_compiled_autograd()

    model_kind = args["model_kind"]
    batch_size = args["batch_size"]
    pos_len = args["pos_len"]
    use_amp = args["use_amp"]
    use_bf16 = args["use_bf16"]
    use_autocast = use_amp or use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    no_compile = args["no_compile"]
    num_iters = args["num_iters"]
    warmup_iters = args["warmup_iters"]
    print_every = args["print_every"]

    assert model_kind in modelconfigs.config_of_name, f"Unknown model kind: {model_kind}"
    model_config = modelconfigs.config_of_name[model_kind].copy()
    apply_config_overrides(model_config, args["override_config"])

    rank0print(f"Trainloop benchmark: model={model_kind} optimizer={args['optimizer']} "
               f"batch_size={batch_size}/gpu world_size={world_size} amp={use_amp} bf16={use_bf16} "
               f"compile={not no_compile} iters={num_iters} warmup={warmup_iters}")

    # Fixed seed so loss trajectories are comparable across benchmark variants.
    torch.manual_seed(20260718)
    raw_model = Model(model_config, pos_len)
    raw_model.initialize()
    raw_model.to(device)
    total_params = sum(p.numel() for p in raw_model.parameters())
    rank0print(f"Total parameters: {total_params:,}")

    attn_logit_penalty_cap = args["attn_logit_penalty_cap"]
    attn_logit_penalty_coeff = args["attn_logit_penalty_coeff"]
    if attn_logit_penalty_cap is not None:
        raw_model.attn_logit_penalty_cap = attn_logit_penalty_cap
        raw_model.attn_logit_penalty_batch_frac = args["attn_logit_penalty_batch_frac"]
        rank0print(f"Attention logit penalty enabled: cap={attn_logit_penalty_cap} coeff={attn_logit_penalty_coeff} "
                   f"batch_frac={args['attn_logit_penalty_batch_frac']}")

    ddp_model = trainloop_helpers.wrap_model_for_training(raw_model, device, world_size, no_compile)

    param_groups = build_train_param_groups(raw_model)
    optimizer = make_optimizer(args["optimizer"], param_groups, world_size, args["ns_steps"], args["use_polar_express"])
    use_muonlike = args["optimizer"] in ("muon", "aurora")

    metrics_obj = Metrics(world_size, raw_model)
    batch = load_batch(args["data"], batch_size, pos_len, model_config, device)

    model_norms_only_at_print = trainloop_helpers.get_model_norms_only_at_print()
    training_metrics_fn = trainloop_helpers.make_training_metrics_fn(metrics_obj, no_compile, model_norms_only_at_print)

    scaler = GradScaler("cuda") if use_amp else None
    step_norm_tracker = trainloop_helpers.StepNormTracker(optimizer)
    gnorm_watcher = trainloop_helpers.GnormWatcher()
    running_sums = defaultdict(float)
    running_weights = defaultdict(float)

    raw_model.train()

    # Matches train.py's cap structure for muon/adamw.
    # Value only matters in that it should essentially never clip during a throughput benchmark.
    gnorm_cap = (11000.0 if use_muonlike or args["optimizer"] == "adam" else 5500.0) * math.sqrt((batch_size * world_size) / 256.0)

    def one_batch(batch_count):
        # Matches train.py: the print-batch extras run when the post-increment batch count hits the print interval.
        is_print_batch = (batch_count + 1) % print_every == 0

        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with autocast("cuda", dtype=amp_dtype):
                model_outputs = ddp_model(
                    batch["binaryInputNCHW"],
                    batch["globalInputNC"],
                )
        else:
            model_outputs = ddp_model(
                batch["binaryInputNCHW"],
                batch["globalInputNC"],
            )
        postprocessed = raw_model.postprocess_output(model_outputs)
        metrics = training_metrics_fn(
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
            include_model_norms=not model_norms_only_at_print,
        )
        if model_norms_only_at_print and is_print_batch:
            metrics.update(metrics_obj.get_model_norm_metrics(raw_model))

        # DDP averages loss across instances, so to preserve LR as per-sample lr, we scale by world size.
        loss = metrics["loss_sum"] * world_size

        # Attention logit bound penalty, mirroring train.py.
        if attn_logit_penalty_cap is not None:
            attn_pen_sum = raw_model.attn_logit_penalty_per_sample.mean() * batch_size
            metrics["alogitpen_sum"] = attn_pen_sum.detach()
            metrics["alogitubmax_batch"] = raw_model.attn_logit_ub_batch_max
            loss = loss + attn_logit_penalty_coeff * attn_pen_sum * world_size

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        trainloop_helpers.clip_gradients_and_record(ddp_model, gnorm_cap, metrics, batch_size)

        step_norm_tracker.capture(ddp_model, is_print_batch=is_print_batch)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step_norm_tracker.record(ddp_model, metrics)

        metrics = trainloop_helpers.detensorify_metrics(metrics)
        gnorm_watcher.observe(metrics, gnorm_cap=gnorm_cap)
        accumulate_metrics(running_sums, running_weights, metrics, batch_size, decay=0.999, new_weight=1.0)
        return metrics

    rank0print("Warming up (includes compilation) ...")
    t_warm_start = time.perf_counter()
    for i in range(warmup_iters):
        one_batch(i)
    torch.cuda.synchronize()
    if world_size > 1:
        torch.distributed.barrier()
    t_warm_end = time.perf_counter()
    rank0print(f"Warmup took {t_warm_end - t_warm_start:.1f} s")

    torch.cuda.reset_peak_memory_stats()
    iter_times = []
    t_prev = time.perf_counter()
    t0 = t_prev
    metrics = None
    for i in range(num_iters):
        metrics = one_batch(warmup_iters + i)
        t_now = time.perf_counter()
        iter_times.append(t_now - t_prev)
        t_prev = t_now
    torch.cuda.synchronize()
    if world_size > 1:
        torch.distributed.barrier()
    t1 = time.perf_counter()

    peak_mem = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    if rank == 0:
        total = t1 - t0
        samples_per_sec = num_iters * batch_size * world_size / total
        print()
        print("=" * 80)
        print("TRAINLOOP THROUGHPUT")
        print("=" * 80)
        print_timing_stats("Per-iter", iter_times)
        print(f"  Total: {total:.3f} s for {num_iters} iters")
        print(f"  Throughput: {samples_per_sec:,.1f} samples/s "
              f"(batch {batch_size} x world_size {world_size})")
        print(f"  Peak GPU memory (rank 0): {peak_mem:.2f} GiB")
        print(f"  Final loss_sum: {metrics['loss_sum']:.2f}")
        print(f"  Bad-gnorm batches: {gnorm_watcher.total_bad}/{gnorm_watcher.total_observed} "
              f"({gnorm_watcher.total_nonfinite} nonfinite, {gnorm_watcher.total_extreme} extreme, "
              f"max consecutive {gnorm_watcher.max_consecutive_bad})")

    if world_size > 1:
        torch.distributed.destroy_process_group()


def load_batch(data_path, batch_size, pos_len, model_config, device):
    """Load a single batch from an npz file."""
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # Version 16 always predicts q values; version 17+ does so only when configured.
    include_qvalues = model_config["version"] == 16 or (
        model_config["version"] >= 17 and bool(model_config.get("predict_q_values"))
    )

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


def benchmark_forward(model, batch, num_iters, warmup_iters, use_autocast=False):
    """Benchmark forward pass only."""
    times = []
    for i in range(warmup_iters + num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            with autocast("cuda", enabled=use_autocast):
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


def benchmark_full_step(model, raw_model, optimizer, metrics_obj, batch, model_config, num_iters, warmup_iters, use_amp=False, scaler=None):
    """Benchmark forward + backward + optimizer step, returning separate timings."""
    fwd_times = []
    bwd_times = []
    opt_times = []

    for i in range(warmup_iters + num_iters):
        optimizer.zero_grad(set_to_none=True)

        # Forward
        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()

        with autocast("cuda", enabled=use_amp):
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
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        torch.cuda.synchronize()
        t_opt_start = time.perf_counter()

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        torch.cuda.synchronize()
        t_opt_end = time.perf_counter()
        del model_outputs, postprocessed, metrics, loss

        if i >= warmup_iters:
            fwd_times.append(t_bwd_start - t_fwd_start)
            bwd_times.append(t_opt_start - t_bwd_start)
            opt_times.append(t_opt_end - t_opt_start)

    return fwd_times, bwd_times, opt_times


def benchmark_full_step_throughput(model, raw_model, optimizer, metrics_obj, batch, model_config, num_iters, warmup_iters, use_amp=False, scaler=None):
    """Benchmark full training step without intermediate syncs, for true throughput measurement."""
    times = []

    for i in range(warmup_iters + num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
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
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
