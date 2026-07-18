"""
Helpers shared between train.py and benchmark_fresh_model.py for the per-batch
training step, so that throughput benchmarks measure the same code that real
training runs.

Optimization toggles are controlled by environment variables.
Set to "0" to disable an optimization.

KATAGO_COMPILE_MODE (default "default"):
  torch.compile mode for the model and compiled loss.
  "max-autotune-no-cudagraphs" trades minutes of extra compile/tuning time for a few percent throughput.
  It retunes per GPU model, per batch size, so it should be re-measured on each machine.
KATAGO_MODEL_NORMS_ONLY_AT_PRINT (default 1):
  Compute model norm metrics only on logging batches instead of every batch.
KATAGO_COMPILE_TRAINING_LOSS (default 1):
  torch.compile the loss/metrics computation separately from the model.
KATAGO_DDP_STATIC_GRAPH (default 1): enable DDP static-graph mode.
KATAGO_DDP_GRADIENT_AS_BUCKET_VIEW (default 1):
  Gradients are views of DDP reduction buckets, saving a copy.
KATAGO_DDP_BROADCAST_BUFFERS (default depends on norm kind):
  Skip DDP's per-forward buffer broadcast for plain batch norm, where training-mode
  forwards do not consume the running statistics. Batch renorm does consume
  them and keeps broadcasts.
KATAGO_DDP_ALIGN_CONV1X1_WEIGHT_STRIDES (default 1):
  Match 1x1 convolution parameter strides with the layout produced by CUDA convolution backward so
  DDP bucket views do not need a re-strided copy.
KATAGO_STEP_NORMS_ONLY_AT_PRINT (default 1):
  Compute the empirical step-norm metrics (which require cloning all parameters and syncing) only on logging batches.
KATAGO_DEFER_GNORM_SYNC (default 1):
  Record the clipped gradient norm as a tensor and convert to a float only at end-of-batch metric collection,
  instead of forcing a GPU sync between backward and optimizer step.
"""

import logging
import math
import os
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag, accepting only the exact values 0 and 1."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"Environment variable {name} must be exactly '0' or '1', got {value!r}")


_ALLOWED_COMPILE_MODES = (
    "default",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)


def get_compile_mode() -> str:
    mode = os.environ.get("KATAGO_COMPILE_MODE", "default")
    if mode in _ALLOWED_COMPILE_MODES:
        return mode
    allowed = "|".join(_ALLOWED_COMPILE_MODES)
    raise ValueError(f"Environment variable KATAGO_COMPILE_MODE must be one of {allowed}, got {mode!r}")


@torch.no_grad()
def align_conv1x1_weight_strides_for_ddp(raw_model) -> int:
    """Match the layout produced by convolution backward for 1x1 weights.

    CUDA convolution backward commonly returns a logically contiguous [out, in, 1, 1] gradient with strides [in, 1, in, in].
    PyTorch's default parameter allocation uses [in, 1, 1, 1] instead.
    Both layouts address exactly the same storage because the last dimensions have size 1,
    but DDP otherwise warns and copies into a differently-strided bucket view.

    Must be called before the optimizer captures parameter references.
    """
    aligned_count = 0
    for module in raw_model.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        weight = module.weight
        if (
            weight is None
            or weight.ndim != 4
            or tuple(weight.shape[2:]) != (1, 1)
        ):
            continue
        in_channels_per_group = weight.shape[1]
        desired_stride = (
            in_channels_per_group,
            1,
            in_channels_per_group,
            in_channels_per_group,
        )
        if weight.stride() == desired_stride:
            continue
        aligned_weight = torch.empty_strided(
            weight.shape,
            desired_stride,
            dtype=weight.dtype,
            device=weight.device,
        )
        aligned_weight.copy_(weight)
        module.weight = torch.nn.Parameter(
            aligned_weight,
            requires_grad=weight.requires_grad,
        )
        aligned_count += 1
    return aligned_count


def wrap_model_for_training(raw_model, device, world_size: int, no_compile: bool):
    """Apply torch.compile and DDP without changing ownership of raw_model.

    Must be called before the optimizer is constructed because 1x1 conv stride alignment can replace Parameter objects.
    """
    compile_mode = None if no_compile else get_compile_mode()
    if world_size <= 1:
        logging.info(
            f"Training model wrapper: single GPU, compile={not no_compile}, "
            f"compile_mode={compile_mode}, DDP environment flags ignored"
        )
        if no_compile:
            return raw_model
        return torch.compile(raw_model, mode=compile_mode)

    static_graph = env_flag("KATAGO_DDP_STATIC_GRAPH", default=True)
    gradient_as_bucket_view = env_flag("KATAGO_DDP_GRADIENT_AS_BUCKET_VIEW", default=True)
    align_conv1x1_weight_strides = env_flag("KATAGO_DDP_ALIGN_CONV1X1_WEIGHT_STRIDES", default=True)
    aligned_conv1x1_count = (
        align_conv1x1_weight_strides_for_ddp(raw_model)
        if align_conv1x1_weight_strides
        else 0
    )
    # Plain batch norm uses the current batch during training, so synchronizing
    # its running statistics before every forward does not affect gradients or
    # rank 0's checkpointed statistics. Batch renorm does consume its running
    # statistics in the training forward and therefore keeps DDP's behavior.
    norm_kind = raw_model.get_norm_kind()
    broadcast_buffers = env_flag(
        "KATAGO_DDP_BROADCAST_BUFFERS",
        default=norm_kind in ("brenorm", "fixbrenorm"),
    )
    # Dynamo's DDPOptimizer splits the compiled graph at DDP bucket boundaries
    # to overlap communication with backward compute. With the flex-attention
    # HOP in the graph, certain split layouts (seen on nbt3 models, torch 2.10-2.11)
    # SILENTLY CORRUPT GRADIENTS. Disable graph splitting whenever flex
    # attention is active. This also is fast or faster than splitting
    # on these models. Non-flex models keep the PyTorch default behavior.
    uses_flex_attention = bool(getattr(raw_model, "use_flex_attention", False))
    optimize_ddp = env_flag("KATAGO_DYNAMO_OPTIMIZE_DDP", default=not uses_flex_attention)
    if not no_compile:
        if uses_flex_attention and optimize_ddp:
            logging.warning(
                "KATAGO_DYNAMO_OPTIMIZE_DDP=1 with flex attention under DDP is known to "
                "silently corrupt gradients on some models. Only use this for debugging, and "
                "watch for nonfinite gnorm warnings."
            )
        torch._dynamo.config.optimize_ddp = optimize_ddp
    logging.info(
        f"Training model wrapper: DDP world_size={world_size}, compile={not no_compile}, "
        f"compile_mode={compile_mode}, static_graph={static_graph}, "
        f"gradient_as_bucket_view={gradient_as_bucket_view}, "
        f"broadcast_buffers={broadcast_buffers}, aligned_conv1x1_weights={aligned_conv1x1_count}, "
        f"dynamo_optimize_ddp={optimize_ddp if not no_compile else 'n/a'}"
    )

    ddp_kwargs = {
        "device_ids": [device],
        "broadcast_buffers": broadcast_buffers,
    }
    if static_graph:
        ddp_kwargs["static_graph"] = True
    if gradient_as_bucket_view:
        ddp_kwargs["gradient_as_bucket_view"] = True

    if no_compile:
        return DistributedDataParallel(raw_model, **ddp_kwargs)

    compiled_model = torch.compile(raw_model, mode=compile_mode)
    return DistributedDataParallel(compiled_model, **ddp_kwargs)


def maybe_enable_compiled_autograd():
    """Experimental: capture the backward pass (including DDP hooks) with
    compiled autograd for additional fusion. Off by default."""
    if env_flag("KATAGO_COMPILED_AUTOGRAD", default=False):
        torch._dynamo.config.compiled_autograd = True
        logging.info("Compiled autograd enabled")


def get_model_norms_only_at_print() -> bool:
    return env_flag("KATAGO_MODEL_NORMS_ONLY_AT_PRINT", default=True)


def make_training_metrics_fn(metrics_obj, no_compile: bool, model_norms_only_at_print: bool):
    """Return the (possibly torch.compiled) training metrics/loss function."""
    compile_requested = env_flag("KATAGO_COMPILE_TRAINING_LOSS", default=True)
    compile_training_loss = compile_requested and not no_compile
    if compile_requested and not compile_training_loss:
        logging.info("Disabling compiled training loss because no_compile is set")
    if not compile_training_loss:
        return metrics_obj.metrics_dict_batchwise
    if not model_norms_only_at_print:
        raise ValueError(
            "KATAGO_COMPILE_TRAINING_LOSS=1 requires KATAGO_MODEL_NORMS_ONLY_AT_PRINT=1 "
            "so the compiled result structure is fixed"
        )
    if not metrics_obj.seki_ema_on_device:
        raise ValueError(
            "KATAGO_COMPILE_TRAINING_LOSS=1 requires KATAGO_SEKI_EMA_ON_DEVICE=1 "
            "to avoid a per-step Python scalar guard"
        )
    logging.info(f"Compiling training loss with mode={get_compile_mode()}")
    return torch.compile(metrics_obj.metrics_dict_batchwise, mode=get_compile_mode(), dynamic=False)


def clip_gradients_and_record(ddp_model, gnorm_cap, metrics, batch_size):
    """Clip gradients by global norm and record gnorm metrics.

    With KATAGO_DEFER_GNORM_SYNC=1 the gnorm stays a tensor here so the CPU can
    enqueue the optimizer step without waiting for backward to finish.
    The conversion to float (and the finite-ness filtering) happens in detensorify_metrics at end of batch.
    """
    gnorm_tensor = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gnorm_cap).detach()
    if env_flag("KATAGO_DEFER_GNORM_SYNC", default=True):
        metrics["gnorm_batch"] = gnorm_tensor
        metrics["exgnorm_sum"] = torch.clamp(gnorm_tensor - gnorm_cap, min=0.0) * batch_size
    else:
        gnorm = gnorm_tensor.cpu().item()
        if math.isfinite(gnorm) and abs(gnorm < 1e30):
            metrics["gnorm_batch"] = gnorm
            exgnorm = max(0.0, gnorm - gnorm_cap)
            metrics["exgnorm_sum"] = exgnorm * batch_size


class StepNormTracker:
    """Tracks the empirical optimizer step vector length (overall and per group).

    capture() is called before optimizer.step() and record() after, filling
    step_norm metrics. With KATAGO_STEP_NORMS_ONLY_AT_PRINT=1 the (expensive)
    parameter cloning and comparison runs only on logging batches.
    When it does run, the per-group sums are collected with a single GPU sync rather than
    one sync per parameter tensor.
    """

    def __init__(self, optimizer):
        # Build a mapping from parameter id to group name once.
        self.param_to_group = {}
        for param_group in optimizer.param_groups:
            group_name = param_group["group_name"]
            for param in param_group["params"]:
                self.param_to_group[id(param)] = group_name
        self.only_at_print = env_flag("KATAGO_STEP_NORMS_ONLY_AT_PRINT", default=True)
        self.old_params = None

    def capture(self, ddp_model, is_print_batch=True):
        if self.only_at_print and not is_print_batch:
            self.old_params = None
            return
        old_params = {}
        for name, param in ddp_model.named_parameters():
            if param.requires_grad:
                old_params[name] = param.data.detach().clone()
        self.old_params = old_params

    def record(self, ddp_model, metrics):
        if self.old_params is None:
            return
        with torch.no_grad():
            group_sums = {}
            for name, param in ddp_model.named_parameters():
                if param.requires_grad:
                    param_diff_squared = torch.sum(torch.square(param.data - self.old_params[name]))
                    group_name = self.param_to_group.get(id(param), "unknown")
                    if group_name in group_sums:
                        group_sums[group_name] += param_diff_squared
                    else:
                        group_sums[group_name] = param_diff_squared

            group_names = list(group_sums.keys())
            # One GPU->CPU sync for all groups instead of one per parameter.
            group_values = torch.stack([group_sums[name] for name in group_names]).cpu().tolist()

            metrics["step_norm_batch"] = math.sqrt(sum(group_values))
            for group_name, norm_squared in zip(group_names, group_values):
                metrics[f"step_norm_{group_name}_batch"] = math.sqrt(norm_squared)
        self.old_params = None


class GnormWatcherError(RuntimeError):
    """Raised by GnormWatcher to halt a run whose gradients are persistently nonfinite."""
    pass


class GnormWatcher:
    """Halts the run when the gradient norm looks persistently pathological.

    A batch is "bad" when its gradient norm is nonfinite OR extremely large
    The magnitude signal matters: in the dynamo-DDPOptimizer + flex-attention
    gradient-corruption bug, gradients are finite but too large.

    Triggers:
    - consecutive: >= consecutive_warn_threshold bad batches in a row, armed
      only after the first good batch. GradScaler calibration produces an
      unbroken nonfinite run from the very first batch and must not false-positive.
    - startup: no good batch at all within the first startup_good_limit
      observations. Calibration runs end long before that.
    - rate: >= rate_warn_fraction of the last rate_window batches bad.

    Raises GnormWatcherError so the run cannot silently continue.
    Setting KATAGO_GNORM_WATCHER_HALT=0 makes it warn instead.

    All ranks compute the same post-allreduce gradient norm, so under DDP every
    rank raises at the same batch.
    """

    def __init__(self, consecutive_warn_threshold=8, rate_window=100, rate_warn_fraction=0.25,
                 extreme_cap_factor=50.0, startup_good_limit=30):
        self.consecutive_warn_threshold = consecutive_warn_threshold
        self.rate_window = rate_window
        self.rate_warn_fraction = rate_warn_fraction
        self.extreme_cap_factor = extreme_cap_factor
        self.startup_good_limit = startup_good_limit
        self.halt_on_trigger = env_flag("KATAGO_GNORM_WATCHER_HALT", default=True)
        self.window = []  # ring buffer of 0/1 bad flags
        self.window_pos = 0
        self.consecutive_bad = 0
        self.max_consecutive_bad = 0
        self.total_observed = 0
        self.total_bad = 0
        self.total_nonfinite = 0
        self.total_extreme = 0
        self.seen_good = False
        self.observations_at_last_warning = None

    def _trigger(self, message):
        message = message + " Training is likely diverging or gradients are being corrupted."
        if self.halt_on_trigger:
            message = message + " Halting the run (set KATAGO_GNORM_WATCHER_HALT=0 to only warn)."
            logging.error(message)
            raise GnormWatcherError(message)
        # Warn-only mode: at most one warning per half rate-window, to avoid
        # log spam while remaining very visible in a limping run.
        if (
            self.observations_at_last_warning is not None
            and self.total_observed - self.observations_at_last_warning < self.rate_window // 2
        ):
            return
        self.observations_at_last_warning = self.total_observed
        logging.warning(message)

    def observe(self, detensorified_metrics, gnorm_cap=None):
        gnorm = detensorified_metrics.get("gnorm_batch")
        nonfinite = gnorm is None
        extreme = (
            not nonfinite
            and gnorm_cap is not None
            and gnorm > self.extreme_cap_factor * gnorm_cap
        )
        bad = nonfinite or extreme
        self.total_observed += 1
        if bad:
            self.total_bad += 1
            if nonfinite:
                self.total_nonfinite += 1
            else:
                self.total_extreme += 1
            self.consecutive_bad += 1
            self.max_consecutive_bad = max(self.max_consecutive_bad, self.consecutive_bad)
        else:
            self.consecutive_bad = 0
            self.seen_good = True

        if len(self.window) < self.rate_window:
            self.window.append(1 if bad else 0)
        else:
            self.window[self.window_pos] = 1 if bad else 0
            self.window_pos = (self.window_pos + 1) % self.rate_window

        detail = (
            f"({self.total_nonfinite} nonfinite + {self.total_extreme} extreme "
            f"(>{self.extreme_cap_factor:g}x the clip cap) out of {self.total_observed} batches"
            + (f" latest gnorm {gnorm:.4g} vs cap {gnorm_cap:.4g}" if extreme else "")
            + ")"
        )
        if self.seen_good and self.consecutive_bad >= self.consecutive_warn_threshold:
            self._trigger(
                f"GNORM WATCHER: gradient norm has been nonfinite or extremely large for "
                f"{self.consecutive_bad} consecutive batches {detail}."
            )
        elif not self.seen_good and self.total_observed >= self.startup_good_limit:
            self._trigger(
                f"GNORM WATCHER: no healthy gradient norm in the first "
                f"{self.total_observed} batches {detail}."
            )
        elif len(self.window) >= self.rate_window:
            bad_fraction = sum(self.window) / len(self.window)
            if bad_fraction >= self.rate_warn_fraction:
                self._trigger(
                    f"GNORM WATCHER: gradient norm nonfinite or extremely large in "
                    f"{100.0 * bad_fraction:.0f}% of the last {len(self.window)} batches "
                    f"{detail}, max consecutive {self.max_consecutive_bad}."
                )


def detensorify_metrics(metrics):
    """Convert tensor metrics to floats with a single GPU sync.

    Also drops the gnorm metrics when nonfinite (they can legitimately be inf
    for a batch whose FP16 gradients overflowed and whose step was skipped),
    matching the historical behavior of excluding such batches.
    """
    ret = {}
    tensor_keys = []
    for key in metrics:
        value = metrics[key]
        if isinstance(value, torch.Tensor):
            tensor_keys.append(key)
        else:
            ret[key] = value
    if len(tensor_keys) > 0:
        stacked = torch.stack([metrics[key].detach().reshape([]).float() for key in tensor_keys]).cpu().tolist()
        for key, value in zip(tensor_keys, stacked):
            ret[key] = value
    if "gnorm_batch" in ret and not (math.isfinite(ret["gnorm_batch"]) and ret["gnorm_batch"] < 1e30):
        del ret["gnorm_batch"]
        ret.pop("exgnorm_sum", None)
    return ret


def get_local_validation_model(training_model, raw_model, world_size: int):
    """Return a local forward module that cannot initiate DDP collectives.

    Validation runs on rank 0 only. Forwarding through the DDP wrapper there
    would trigger collectives (e.g. buffer broadcasts) that other ranks never
    join. The common compile-before-DDP path preserves the compiled local
    module here, while bypassing DDP's forward-time work.
    """
    if world_size <= 1:
        return training_model

    current = training_model
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, DistributedDataParallel):
            return current.module
        original_module = getattr(current, "_orig_mod", None)
        if original_module is None or original_module is current:
            break
        current = original_module

    # Unknown wrappers are not safe to call from rank 0 alone.
    return raw_model


def set_snapshot_metrics(metric_sums, metric_weights, metrics, keys):
    """Store point-in-time metrics without depending on moving-average weight."""
    for key in keys:
        if key in metrics:
            metric_sums[key] = metrics[key]
            metric_weights[key] = 1.0
