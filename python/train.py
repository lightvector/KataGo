#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import contextlib
import json
import datetime
from datetime import timezone
import gc
import shutil
import glob
import numpy as np
import itertools
import copy
import atexit
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import GradScaler, autocast

import modelconfigs
from model_pytorch import Model, ExtraOutputs, MetadataEncoder
from metrics_pytorch import Metrics
from push_back_generator import PushBackGenerator
import load_model
import data_processing_pytorch
from metrics_logging import accumulate_metrics, log_metrics, clear_metric_nonfinite

# HANDLE COMMAND AND ARGS -------------------------------------------------------------------

if __name__ == "__main__":

    description = """
    Train neural net on Go positions from npz files of batches from selfplay.
    """

    parser = argparse.ArgumentParser(description=description,add_help=False)
    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )

    required_args.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
    required_args.add_argument('-datadir', help='Directory with a train and val subdir of npz data, output by shuffle.py', required=True)
    optional_args.add_argument('-exportdir', help='Directory to export models periodically', required=False)
    optional_args.add_argument('-exportprefix', help='Prefix to append to names of models', required=False)
    optional_args.add_argument('-initial-checkpoint', help='If no training checkpoint exists, initialize from this checkpoint', required=False)

    required_args.add_argument('-pos-len', help='Spatial edge length of expected training data, e.g. 19 for 19x19 Go', type=int, required=True)
    required_args.add_argument('-batch-size', help='Per-GPU batch size to use for training', type=int, required=True)
    optional_args.add_argument('-samples-per-epoch', help='Number of data samples to consider as one epoch', type=int, required=False)
    optional_args.add_argument('-model-kind', help='String name for what model config to use', required=False)
    optional_args.add_argument('-lr-scale', help='LR multiplier on the hardcoded schedule', type=float, required=False)
    optional_args.add_argument('-lr-scale-auto', help='LR auto scaling', required=False, action='store_true')
    optional_args.add_argument('-gnorm-clip-scale', help='Multiplier on gradient clipping threshold', type=float, required=False)
    optional_args.add_argument('-sub-epochs', help='Reload training data up to this many times per epoch', type=int, default=1, required=False)
    optional_args.add_argument('-swa-period-samples', help='How frequently to average an SWA sample, in samples', type=float, required=False)
    optional_args.add_argument('-swa-scale', help='Number of samples to average in expectation together for SWA', type=float, required=False)
    optional_args.add_argument('-lookahead-k', help='Use lookahead optimizer', type=int, default=6, required=False)
    optional_args.add_argument('-lookahead-alpha', help='Use lookahead optimizer', type=float, default=0.5, required=False)
    optional_args.add_argument('-lookahead-print', help='Only print on lookahead syncs', required=False, action='store_true')

    optional_args.add_argument('-multi-gpus', help='Use multiple gpus, comma-separated device ids', required=False)
    optional_args.add_argument('-use-fp16', help='Use fp16 training', required=False, action='store_true')

    optional_args.add_argument('-epochs-per-export', help='Export model once every this many epochs', type=int, required=False)
    optional_args.add_argument('-export-prob', help='Export model with this probablity', type=float, required=False)
    optional_args.add_argument('-max-epochs-this-instance', help='Terminate training after this many more epochs', type=int, required=False)
    optional_args.add_argument('-max-training-samples', help='Terminate training after about this many training steps in samples', type=int, required=False)
    optional_args.add_argument('-sleep-seconds-per-epoch', help='Sleep this long between epochs', type=int, required=False)
    optional_args.add_argument('-max-train-bucket-per-new-data', help='When data added, add this many train rows per data row to bucket', type=float, required=False)
    optional_args.add_argument('-max-train-bucket-size', help='Approx total number of train rows allowed if data stops', type=float, required=False)
    optional_args.add_argument('-max-train-steps-since-last-reload', help='Approx total of training allowed if shuffling stops', type=float, required=False)
    optional_args.add_argument('-stop-when-train-bucket-limited', help='Terminate due to train bucket rather than waiting for more', required=False, action='store_true')
    optional_args.add_argument('-max-val-samples', help='Approx max of validation samples per epoch', type=int, required=False)
    optional_args.add_argument('-randomize-val', help='Randomize order of validation files', required=False, action='store_true')
    optional_args.add_argument('-no-export', help='Do not export models', required=False, action='store_true')
    optional_args.add_argument('-no-repeat-files', help='Track what shuffled data was used and do not repeat, even when killed and resumed', required=False, action='store_true')
    optional_args.add_argument('-quit-if-no-data', help='If no data, quit instead of waiting for data', required=False, action='store_true')

    optional_args.add_argument('-gnorm-stats-debug', required=False, action='store_true')

    optional_args.add_argument('-brenorm-avg-momentum', type=float, help='Set brenorm running avg rate to this value', required=False)
    optional_args.add_argument('-brenorm-target-rmax', type=float, help='Gradually adjust brenorm rmax to this value', required=False)
    optional_args.add_argument('-brenorm-target-dmax', type=float, help='Gradually adjust brenorm dmax to this value', required=False)
    optional_args.add_argument('-brenorm-adjustment-scale', type=float, help='How many samples to adjust brenorm params all but 1/e of the way to target', required=False)

    optional_args.add_argument('-soft-policy-weight-scale', type=float, default=8.0, help='Soft policy loss coeff', required=False)
    optional_args.add_argument('-disable-optimistic-policy', help='Disable optimistic policy', required=False, action='store_true')
    optional_args.add_argument('-meta-kata-only-soft-policy', help='Mask soft policy on non-kata rows using sgfmeta', required=False, action='store_true')
    optional_args.add_argument('-value-loss-scale', type=float, default=0.6, help='Additional value loss coeff', required=False)
    optional_args.add_argument('-td-value-loss-scales', type=str, default="0.6,0.6,0.6", help='Additional td value loss coeffs, 3 comma separated values', required=False)
    optional_args.add_argument('-seki-loss-scale', type=float, default=1.0, help='Additional seki loss coeff', required=False)
    optional_args.add_argument('-variance-time-loss-scale', type=float, default=1.0, help='Additional variance time loss coeff', required=False)

    optional_args.add_argument('-main-loss-scale', type=float, help='Loss factor scale for main head', required=False)
    optional_args.add_argument('-intermediate-loss-scale', type=float, help='Loss factor scale for intermediate head', required=False)

    args = vars(parser.parse_args())


def get_longterm_checkpoints_dir(traindir):
    return os.path.join(traindir,"longterm_checkpoints")

def make_dirs(args):
    traindir = args["traindir"]
    exportdir = args["exportdir"]

    if not os.path.exists(traindir):
        os.makedirs(traindir)
    if exportdir is not None and not os.path.exists(exportdir):
        os.makedirs(exportdir)

    longterm_checkpoints_dir = get_longterm_checkpoints_dir(traindir)
    if not os.path.exists(longterm_checkpoints_dir):
        os.makedirs(longterm_checkpoints_dir)

def multiprocessing_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    logging.info("Running torch.distributed.init_process_group")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Returned from torch.distributed.init_process_group, my rank = {rank}, world_size={world_size}")

def multiprocessing_cleanup():
    torch.distributed.destroy_process_group()


def main(rank: int, world_size: int, args, multi_gpu_device_ids, readpipes, writepipes, barrier):
    traindir = args["traindir"]
    datadir = args["datadir"]
    exportdir = args["exportdir"]
    exportprefix = args["exportprefix"]
    initial_checkpoint = args["initial_checkpoint"]

    pos_len = args["pos_len"]
    batch_size = args["batch_size"]
    samples_per_epoch = args["samples_per_epoch"]
    model_kind = args["model_kind"]
    lr_scale = args["lr_scale"]
    lr_scale_auto = args["lr_scale_auto"]
    gnorm_clip_scale = args["gnorm_clip_scale"]
    sub_epochs = args["sub_epochs"]
    swa_period_samples = args["swa_period_samples"]
    swa_scale = args["swa_scale"]
    lookahead_k = args["lookahead_k"]
    lookahead_alpha = args["lookahead_alpha"]
    lookahead_print = args["lookahead_print"]

    use_fp16 = args["use_fp16"]

    epochs_per_export = args["epochs_per_export"]
    export_prob = args["export_prob"]
    max_epochs_this_instance = args["max_epochs_this_instance"]
    max_training_samples = args["max_training_samples"]
    sleep_seconds_per_epoch = args["sleep_seconds_per_epoch"]
    max_train_bucket_per_new_data = args["max_train_bucket_per_new_data"]
    max_train_bucket_size = args["max_train_bucket_size"]
    max_train_steps_since_last_reload = args["max_train_steps_since_last_reload"]
    stop_when_train_bucket_limited = args["stop_when_train_bucket_limited"]
    max_val_samples = args["max_val_samples"]
    randomize_val = args["randomize_val"]
    no_export = args["no_export"]
    no_repeat_files = args["no_repeat_files"]
    quit_if_no_data = args["quit_if_no_data"]

    gnorm_stats_debug = args["gnorm_stats_debug"]

    brenorm_target_rmax = args["brenorm_target_rmax"]
    brenorm_target_dmax = args["brenorm_target_dmax"]
    brenorm_avg_momentum = args["brenorm_avg_momentum"]
    brenorm_adjustment_scale = args["brenorm_adjustment_scale"]

    soft_policy_weight_scale = args["soft_policy_weight_scale"]
    disable_optimistic_policy = args["disable_optimistic_policy"]
    meta_kata_only_soft_policy = args["meta_kata_only_soft_policy"]
    value_loss_scale = args["value_loss_scale"]
    td_value_loss_scales = [float(x) for x in args["td_value_loss_scales"].split(",")]
    seki_loss_scale = args["seki_loss_scale"]
    variance_time_loss_scale = args["variance_time_loss_scale"]

    main_loss_scale = args["main_loss_scale"]
    intermediate_loss_scale = args["intermediate_loss_scale"]

    if lr_scale is None:
        lr_scale = 1.0
    if lr_scale_auto:
        assert lr_scale == 1.0, "Cannot specify both lr_scale and lr_scale_auto"

    if samples_per_epoch is None:
        samples_per_epoch = 1000000
    if max_train_bucket_size is None:
        max_train_bucket_size = 1.0e30
    if epochs_per_export is None:
        epochs_per_export = 1
    if swa_period_samples is None:
        swa_period_samples = max(1, samples_per_epoch // 2)
    if swa_scale is None:
        swa_scale = 8

    assert lookahead_alpha > 0.0 and lookahead_alpha <= 1.0
    if lookahead_alpha >= 1.0:  # 1.0 means to disable lookahead optimizer
        lookahead_alpha = None
        lookahead_k = None

    longterm_checkpoints_dir = get_longterm_checkpoints_dir(traindir)

    assert (swa_period_samples is None) == (swa_scale is None)
    assert (lookahead_k is None) == (lookahead_alpha is None)

    # SET UP LOGGING -------------------------------------------------------------

    logging.root.handlers = []
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(traindir,f"train{rank}.log"), mode="a"),
                logging.StreamHandler()
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(traindir,f"train{rank}.log"), mode="a"),
            ],
        )
    np.set_printoptions(linewidth=150)

    logging.info(str(sys.argv))

    # FIGURE OUT MULTIGPU ------------------------------------------------------------
    if world_size > 1:
        multiprocessing_setup(rank, world_size)
        atexit.register(multiprocessing_cleanup)
        assert torch.cuda.is_available()

    if True or torch.cuda.is_available():
        my_gpu_id = multi_gpu_device_ids[rank]
        torch.cuda.set_device(my_gpu_id)
        logging.info("Using GPU device: " + torch.cuda.get_device_name())
        device = torch.device("cuda", my_gpu_id)
    else:
        logging.warning("WARNING: No GPU, using CPU")
        device = torch.device("cpu")

    seed = int.from_bytes(os.urandom(7), sys.byteorder)
    logging.info(f"Seeding torch with {seed}")
    torch.manual_seed(seed)

    # LOAD MODEL ---------------------------------------------------------------------

    def lr_scale_auto_factor(train_state):
        if not lr_scale_auto:
            return 1.0

        if train_state["global_step_samples"] < 200_000_000:
            return 8.00
        if train_state["global_step_samples"] < 400_000_000:
            return 4.00
        if train_state["global_step_samples"] < 500_000_000:
            return 2.00
        if train_state["global_step_samples"] < 550_000_000:
            return 1.00
        if train_state["global_step_samples"] < 600_000_000:
            return 0.50
        if train_state["global_step_samples"] < 650_000_000:
            return 0.25
        return 0.25

    def get_checkpoint_path():
        return os.path.join(traindir,"checkpoint.ckpt")
    def get_checkpoint_prev_path(i):
        return os.path.join(traindir,f"checkpoint_prev{i}.ckpt")

    NUM_SHORTTERM_CHECKPOINTS_TO_KEEP = 4
    def save(ddp_model, swa_model, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics, path=None):
        if gnorm_stats_debug:
            logging.warning("Skipping save since debugging gnorm stats")
            return
        if rank == 0:
            state_dict = {}
            state_dict["model"] = ddp_model.state_dict()
            state_dict["optimizer"] = optimizer.state_dict()
            state_dict["metrics"] = metrics_obj.state_dict()
            state_dict["running_metrics"] = running_metrics
            state_dict["train_state"] = train_state
            state_dict["last_val_metrics"] = last_val_metrics
            state_dict["config"] = model_config

            if swa_model is not None:
                state_dict["swa_model"] = swa_model.state_dict()

            if path is not None:
                logging.info("Saving checkpoint: " + path)
                torch.save(state_dict, path + ".tmp")
                time.sleep(1)
                os.replace(path + ".tmp", path)
            else:
                logging.info("Saving checkpoint: " + get_checkpoint_path())
                for i in reversed(range(NUM_SHORTTERM_CHECKPOINTS_TO_KEEP-1)):
                    if os.path.exists(get_checkpoint_prev_path(i)):
                        os.replace(get_checkpoint_prev_path(i), get_checkpoint_prev_path(i+1))
                if os.path.exists(get_checkpoint_path()):
                    shutil.copy(get_checkpoint_path(), get_checkpoint_prev_path(0))
                torch.save(state_dict, get_checkpoint_path() + ".tmp")
                os.replace(get_checkpoint_path() + ".tmp", get_checkpoint_path())

    def get_weight_decay(raw_model, lr_scale, warmup_scale, train_state, running_metrics, group_name):
        lr_scale_with_auto = lr_scale * lr_scale_auto_factor(train_state)
        if raw_model.get_norm_kind() == "fixup" or raw_model.get_norm_kind() == "fixscale":
            if group_name == "normal" or group_name == "normal_gamma" or group_name == "output":
                return 0.000001 * world_size * batch_size / 256.0
            elif group_name == "noreg":
                return 0.00000001 * world_size * batch_size / 256.0
            elif group_name == "output_noreg":
                return 0.00000001 * world_size * batch_size / 256.0
            else:
                assert False
        elif (
            raw_model.get_norm_kind() == "bnorm" or
            raw_model.get_norm_kind() == "brenorm" or
            raw_model.get_norm_kind() == "fixbrenorm" or
            raw_model.get_norm_kind() == "fixscaleonenorm"
        ):
            if group_name == "normal" or group_name == "normal_gamma":
                adaptive_scale = 1.0
                if "sums" in running_metrics and "norm_normal_batch" in running_metrics["sums"]:
                    norm_normal_batch = running_metrics["sums"]["norm_normal_batch"] / running_metrics["weights"]["norm_normal_batch"]
                    baseline = train_state["modelnorm_normal_baseline"]
                    ratio = norm_normal_batch / (baseline + 1e-30)
                    # Adaptive weight decay keeping model norm around the baseline level so that batchnorm effective lr is held constant
                    # throughout training, covering a range of 16x from bottom to top.
                    adaptive_scale = math.pow(2.0, 2.0 * math.tanh(math.log(ratio+1e-30) * 1.5))

                # Batch norm gammas can be regularized a bit less, doing them just as much empirically seemed to be a bit more unstable
                gamma_scale = 0.125 if group_name == "normal_gamma" else 1.0

                # The theoretical scaling for keeping us confined to a surface of equal model norm should go proportionally with lr_scale.
                # because the strength of drift away from that surface goes as lr^2 and weight decay itself is scaled by lr, so we need
                # one more factor of lr to make weight decay strength equal drift strength.
                # However, at low lr it tends to be the case that gradient norm increases slightly
                # while at high lr it tends to be the case that gradient norm decreases, which means drift strength scales a bit slower
                # than expected.
                # So we scale sublinearly with lr_scale so as to slightly preadjust to this effect.
                # Adaptive scale should then help keep us there thereafter.
                return 0.00125 * world_size * batch_size / 256.0 * math.pow(lr_scale_with_auto * warmup_scale,0.75) * adaptive_scale * gamma_scale
            elif group_name == "output":
                return 0.000001 * world_size * batch_size / 256.0
            elif group_name == "noreg":
                return 0.000001 * world_size * batch_size / 256.0 * math.pow(lr_scale_with_auto * warmup_scale,0.75)
            elif group_name == "output_noreg":
                return 0.00000001 * world_size * batch_size / 256.0
            else:
                assert False
        else:
            assert False

    def get_param_groups(raw_model,train_state,running_metrics):
        reg_dict : Dict[str,List] = {}
        raw_model.add_reg_dict(reg_dict)
        param_groups = []
        param_groups.append({
            "params": reg_dict["normal"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="normal"),
            "group_name": "normal",
        })
        if len(reg_dict["normal_gamma"]) > 0:
            param_groups.append({
                "params": reg_dict["normal_gamma"],
                "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="normal_gamma"),
                "group_name": "normal_gamma",
            })
        param_groups.append({
            "params": reg_dict["output"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="output"),
            "group_name": "output",
        })
        param_groups.append({
            "params": reg_dict["noreg"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="noreg"),
            "group_name": "noreg",
        })
        param_groups.append({
            "params": reg_dict["output_noreg"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="output_noreg"),
            "group_name": "output_noreg",
        })
        num_params = len(list(raw_model.parameters()))
        num_reg_dict_params = len(reg_dict["normal"]) + len(reg_dict["normal_gamma"]) + len(reg_dict["output"]) + len(reg_dict["noreg"]) + len(reg_dict["output_noreg"])
        assert num_params == num_reg_dict_params, "Reg dict does not have entries for all params in model"
        return param_groups

    def load():
        if not os.path.exists(get_checkpoint_path()):
            logging.info("No preexisting checkpoint found at: " + get_checkpoint_path())
            for i in range(NUM_SHORTTERM_CHECKPOINTS_TO_KEEP):
                if os.path.exists(get_checkpoint_prev_path(i)):
                    raise Exception(f"No preexisting checkpoint found, but {get_checkpoint_prev_path(i)} exists, something is wrong with the training dir")

            if initial_checkpoint is not None:
                if os.path.exists(initial_checkpoint):
                    logging.info("Using initial checkpoint: {initial_checkpoint}")
                    path_to_load_from = initial_checkpoint
                else:
                    raise Exception("No preexisting checkpoint found, initial checkpoint provided is invalid: {initial_checkpoint}")
            else:
                path_to_load_from = None
        else:
            path_to_load_from = get_checkpoint_path()

        if path_to_load_from is None:
            logging.info("Initializing new model!")
            assert model_kind is not None, "Model kind is none or unspecified but the model is being created fresh"
            model_config = modelconfigs.config_of_name[model_kind]
            logging.info(str(model_config))
            raw_model = Model(model_config,pos_len)
            raw_model.initialize()

            raw_model.to(device)
            if world_size > 1:
                ddp_model = torch.nn.parallel.DistributedDataParallel(raw_model, device_ids=[device])
            else:
                ddp_model = raw_model

            swa_model = None
            if rank == 0 and swa_scale is not None:
                new_factor = 1.0 / swa_scale
                ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (cur_param - avg_param)
                swa_model = AveragedModel(raw_model, avg_fn=ema_avg)

            metrics_obj = Metrics(batch_size,world_size,raw_model)
            running_metrics = {}
            train_state = {}
            last_val_metrics = {}

            train_state["global_step_samples"] = 0

            with torch.no_grad():
                (modelnorm_normal, modelnorm_normal_gamma, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = Metrics.get_model_norms(raw_model)
                modelnorm_normal_baseline = modelnorm_normal.detach().cpu().item()
                train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline
                logging.info(f"Model norm normal baseline computed: {modelnorm_normal_baseline}")

            optimizer = torch.optim.SGD(get_param_groups(raw_model,train_state,running_metrics), lr=1.0, momentum=0.9)

            return (model_config, ddp_model, raw_model, swa_model, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics)
        else:
            state_dict = torch.load(path_to_load_from, map_location=device)
            model_config = state_dict["config"] if "config" in state_dict else modelconfigs.config_of_name[model_kind]
            logging.info(str(model_config))
            raw_model = Model(model_config,pos_len)
            raw_model.initialize()

            train_state = {}
            if "train_state" in state_dict:
                train_state = state_dict["train_state"]
            else:
                logging.info("WARNING: Train state not found in state dict, using fresh train state")

            # Do this before loading the state dict, while the model is initialized to fresh values, to get a good baseline
            if "modelnorm_normal_baseline" not in train_state:
                logging.info("Computing modelnorm_normal_baseline since not in train state")
                with torch.no_grad():
                    (modelnorm_normal, modelnorm_normal_gamma, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = Metrics.get_model_norms(raw_model)
                    modelnorm_normal_baseline = modelnorm_normal.detach().cpu().item()
                    train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline
                    logging.info(f"Model norm normal baseline computed: {modelnorm_normal_baseline}")

            # Strip off any "module." from when the model was saved with DDP or other things
            model_state_dict = load_model.load_model_state_dict(state_dict)
            raw_model.load_state_dict(model_state_dict)

            raw_model.to(device)
            if world_size > 1:
                ddp_model = torch.nn.parallel.DistributedDataParallel(raw_model, device_ids=[device])
            else:
                ddp_model = raw_model

            swa_model = None
            if rank == 0 and swa_scale is not None:
                new_factor = 1.0 / swa_scale
                ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (cur_param - avg_param)
                swa_model = AveragedModel(raw_model, avg_fn=ema_avg)
                swa_model_state_dict = load_model.load_swa_model_state_dict(state_dict)
                if swa_model_state_dict is not None:
                    swa_model.load_state_dict(swa_model_state_dict)

            metrics_obj = Metrics(batch_size,world_size,raw_model)
            if "metrics" in state_dict:
                metrics_obj.load_state_dict(state_dict["metrics"])
            else:
                logging.info("WARNING: Metrics not found in state dict, using fresh metrics")

            running_metrics = {}
            if "running_metrics" in state_dict:
                running_metrics = state_dict["running_metrics"]
            else:
                logging.info("WARNING: Running metrics not found in state dict, using fresh running metrics")

            last_val_metrics = {}
            if "last_val_metrics" in state_dict:
                last_val_metrics = state_dict["last_val_metrics"]
            else:
                logging.info("WARNING: Running metrics not found in state dict, using fresh last val metrics")

            optimizer = torch.optim.SGD(get_param_groups(raw_model,train_state,running_metrics), lr=1.0, momentum=0.9)
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
            else:
                logging.info("WARNING: Optimizer not found in state dict, using fresh optimizer")

            return (model_config, ddp_model, raw_model, swa_model, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics)

    (model_config, ddp_model, raw_model, swa_model, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics) = load()


    if "global_step_samples" not in train_state:
        train_state["global_step_samples"] = 0
    if max_train_bucket_per_new_data is not None and "train_bucket_level" not in train_state:
        train_state["train_bucket_level"] = samples_per_epoch
    if "train_steps_since_last_reload" not in train_state:
        train_state["train_steps_since_last_reload"] = 0
    if "export_cycle_counter" not in train_state:
        train_state["export_cycle_counter"] = 0
    if "window_start_data_row_idx" not in train_state:
        train_state["window_start_data_row_idx"] = 0
    if "total_num_data_rows" not in train_state:
        train_state["total_num_data_rows"] = 0
    if "old_train_data_dirs" not in train_state:
        train_state["old_train_data_dirs"] = []
    if "data_files_used" not in train_state:
        train_state["data_files_used"] = set()
    if "swa_sample_accum" not in train_state:
        train_state["swa_sample_accum"] = 0.0


    if intermediate_loss_scale is not None:
        assert raw_model.get_has_intermediate_head(), "Model must have intermediate head to use intermediate loss"

    # If the user specified an intermediate head but no loss scale, pick something reasonable by default
    if raw_model.get_has_intermediate_head():
        if intermediate_loss_scale is None and main_loss_scale is None:
            if model_config["trunk_normless"]:
                # fson-bnh default
                assert model_config["intermediate_head_blocks"] == len(model_config["block_kind"]), "If these are unequal, don't know what you intend, please specify intermediate_loss_scale"
                intermediate_loss_scale = 0.8
                main_loss_scale = 0.2
            else:
                # Intermediate head in the middle of the trunk
                intermediate_loss_scale = 0.5
                main_loss_scale = 0.5
        elif intermediate_loss_scale is None:
            assert False, "Please specify both of main_loss_scale and intermediate_loss_scale or neither when using an architecture with an intermediate head."

    logging.info(f"swa_period_samples {swa_period_samples}")
    logging.info(f"swa_scale {swa_scale}")
    logging.info(f"lookahead_alpha {lookahead_alpha}")
    logging.info(f"lookahead_k {lookahead_k}")
    logging.info(f"soft_policy_weight_scale {soft_policy_weight_scale}")
    logging.info(f"disable_optimistic_policy {disable_optimistic_policy}")
    logging.info(f"meta_kata_only_soft_policy {meta_kata_only_soft_policy}")
    logging.info(f"value_loss_scale {value_loss_scale}")
    logging.info(f"td_value_loss_scales {td_value_loss_scales}")
    logging.info(f"seki_loss_scale {seki_loss_scale}")
    logging.info(f"variance_time_loss_scale {variance_time_loss_scale}")
    logging.info(f"main_loss_scale {main_loss_scale}")
    logging.info(f"intermediate_loss_scale {intermediate_loss_scale}")

    # Print all model parameters just to get a summary
    total_num_params = 0
    total_trainable_params = 0
    logging.info("Parameters in model:")
    for name, param in raw_model.named_parameters():
        product = 1
        for dim in param.shape:
            product *= int(dim)
        if param.requires_grad:
            total_trainable_params += product
        total_num_params += product
        logging.info(f"{name}, {list(param.shape)}, {product} params")
    logging.info(f"Total num params: {total_num_params}")
    logging.info(f"Total trainable params: {total_trainable_params}")

    lookahead_cache = {}
    if lookahead_k is not None:
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                lookahead_cache[param] = torch.zeros_like(param.data)
                lookahead_cache[param] = lookahead_cache[param].copy_(param.data)
        logging.info(f"Using lookahead optimizer {lookahead_alpha} {lookahead_k}")

    # EPOCHS AND LR ---------------------------------------------------------------------

    def update_and_return_lr_and_wd():
        per_sample_lr = 0.00003 * lr_scale * lr_scale_auto_factor(train_state)

        # Warmup for initial training
        warmup_scale = 1.0
        if model_config["norm_kind"] == "fixup" or model_config["norm_kind"] == "fixscale" or model_config["norm_kind"] == "fixscaleonenorm":
            if train_state["global_step_samples"] < 1000000:
                warmup_scale = 1.0 / 5.0
            elif train_state["global_step_samples"] < 2000000:
                warmup_scale = 1.0 / 3.0
            elif train_state["global_step_samples"] < 4000000:
                warmup_scale = 1.0 / 2.0
            elif train_state["global_step_samples"] < 6000000:
                warmup_scale = 1.0 / 1.4
        elif model_config["norm_kind"] == "bnorm" or model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
            if train_state["global_step_samples"] < 250000:
                warmup_scale = 1.0 / 20.0
            elif train_state["global_step_samples"] < 500000:
                warmup_scale = 1.0 / 14.0
            elif train_state["global_step_samples"] < 750000:
                warmup_scale = 1.0 / 10.0
            elif train_state["global_step_samples"] < 1000000:
                warmup_scale = 1.0 / 7.0
            elif train_state["global_step_samples"] < 1250000:
                warmup_scale = 1.0 / 5.0
            elif train_state["global_step_samples"] < 1500000:
                warmup_scale = 1.0 / 3.0
            elif train_state["global_step_samples"] < 1750000:
                warmup_scale = 1.0 / 2.0
            elif train_state["global_step_samples"] < 2000000:
                warmup_scale = 1.0 / 1.4
            else:
                warmup_scale = 1.0 / 1.0
        else:
            assert False

        normal_weight_decay = 0.0

        for param_group in optimizer.param_groups:
            group_name = param_group["group_name"]
            if group_name == "normal":
                group_scale = 1.0
            elif group_name == "normal_gamma":
                group_scale = 1.0
            elif group_name == "output":
                group_scale = 0.5
            elif group_name == "noreg":
                group_scale = 1.0
            elif group_name == "output_noreg":
                group_scale = 0.5
            else:
                assert False

            changed = False

            # For lookahead optimizer, use weight decay appropriate for lr scale,
            # but tell optimizer to take larger steps so as to maintain the same
            # effective learning rate after lookahead averaging.
            if lookahead_alpha is not None:
                new_lr_this_group = per_sample_lr * warmup_scale * group_scale / lookahead_alpha
            else:
                new_lr_this_group = per_sample_lr * warmup_scale * group_scale

            if param_group["lr"] != new_lr_this_group:
                param_group["lr"] = new_lr_this_group
                changed = True

            new_weight_decay_this_group = get_weight_decay(
                raw_model,
                lr_scale,
                warmup_scale=warmup_scale,
                train_state=train_state,
                running_metrics=running_metrics,
                group_name=group_name,
            )
            if param_group["weight_decay"] != new_weight_decay_this_group:
                param_group["weight_decay"] = new_weight_decay_this_group
                changed = True

            if group_name == "normal":
                normal_weight_decay = param_group["weight_decay"]

            if changed:
                logging.info(f"Param group {param_group['group_name']} lr {param_group['lr']} weight_decay {param_group['weight_decay']}")

        return per_sample_lr * warmup_scale, normal_weight_decay

    last_brenorm_update_samples_this_instance = train_state["global_step_samples"]
    def maybe_update_brenorm_params():
        nonlocal last_brenorm_update_samples_this_instance

        if model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
            if "brenorm_rmax" not in train_state:
                train_state["brenorm_rmax"] = 1.0
            if "brenorm_dmax" not in train_state:
                train_state["brenorm_dmax"] = 0.0

            num_samples_elapsed = train_state["global_step_samples"] - last_brenorm_update_samples_this_instance
            factor = math.exp(-num_samples_elapsed / brenorm_adjustment_scale)
            train_state["brenorm_rmax"] = train_state["brenorm_rmax"] + (1.0 - factor) * (brenorm_target_rmax - train_state["brenorm_rmax"])
            train_state["brenorm_dmax"] = train_state["brenorm_dmax"] + (1.0 - factor) * (brenorm_target_dmax - train_state["brenorm_dmax"])

            raw_model.set_brenorm_params(brenorm_avg_momentum, train_state["brenorm_rmax"], train_state["brenorm_dmax"])
            last_brenorm_update_samples_this_instance = train_state["global_step_samples"]

    # DATA RELOADING GENERATOR ------------------------------------------------------------

    # Some globals
    last_curdatadir = None
    trainfilegenerator = None
    vdatadir = None

    def maybe_reload_training_data():
        nonlocal last_curdatadir
        nonlocal trainfilegenerator
        nonlocal vdatadir

        assert rank == 0, "Helper ddp training processes should not call maybe_reload_training_data"

        while True:
            curdatadir = os.path.realpath(datadir)

            # Different directory - new shuffle
            if curdatadir != last_curdatadir:
                if not os.path.exists(curdatadir):
                    if quit_if_no_data:
                        logging.info("Shuffled data path does not exist, there seems to be no data or not enough data yet, qutting: %s" % curdatadir)
                        sys.exit(0)
                    logging.info("Shuffled data path does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % curdatadir)
                    time.sleep(30)
                    continue

                trainjsonpath = os.path.join(curdatadir,"train.json")
                if not os.path.exists(trainjsonpath):
                    if quit_if_no_data:
                        logging.info("Shuffled data train.json file does not exist, there seems to be no data or not enough data yet, qutting: %s" % trainjsonpath)
                        sys.exit(0)
                    logging.info("Shuffled data train.json file does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % trainjsonpath)
                    time.sleep(30)
                    continue

                logging.info("Updated training data: " + curdatadir)
                last_curdatadir = curdatadir

                with open(trainjsonpath) as f:
                    datainfo = json.load(f)
                    train_state["window_start_data_row_idx"] = datainfo["range"][0]
                    train_state["total_num_data_rows"] = datainfo["range"][1]

                # Fill the buckets
                if max_train_bucket_per_new_data is not None:
                    if "train_bucket_level_at_row" not in train_state:
                        train_state["train_bucket_level_at_row"] = train_state["total_num_data_rows"]
                    if train_state["total_num_data_rows"] > train_state["train_bucket_level_at_row"]:
                        new_row_count = train_state["total_num_data_rows"] - train_state["train_bucket_level_at_row"]
                        logging.info("Advancing trainbucket row %.0f to %.0f, %.0f new rows" % (
                            train_state["train_bucket_level_at_row"], train_state["total_num_data_rows"], new_row_count
                        ))
                        train_state["train_bucket_level_at_row"] = train_state["total_num_data_rows"]
                        logging.info("Fill per data %.3f, Max bucket size %.0f" % (max_train_bucket_per_new_data, max_train_bucket_size))
                        logging.info("Old rows in bucket: %.0f" % train_state["train_bucket_level"])
                        train_state["train_bucket_level"] += new_row_count * max_train_bucket_per_new_data
                        cap = max(max_train_bucket_size, samples_per_epoch)
                        if train_state["train_bucket_level"] > cap:
                            train_state["train_bucket_level"] = cap
                        logging.info("New rows in bucket: %.0f" % train_state["train_bucket_level"])
                    if train_state["total_num_data_rows"] < train_state["train_bucket_level_at_row"]:
                        # Bucket went backward! This must be a network imported from a different run, reset the train bucket level
                        logging.warning("Train bucket last filled at %d rows but now there are only %d rows!" % (
                            train_state["train_bucket_level_at_row"], train_state["total_num_data_rows"]
                        ))
                        logging.warning("Data was deleted or this network was transplanted into a new run, resetting the train bucket fill rows")
                        train_state["train_bucket_level_at_row"] = train_state["total_num_data_rows"]

                logging.info("Train steps since last reload: %.0f -> 0" % train_state["train_steps_since_last_reload"])
                train_state["train_steps_since_last_reload"] = 0

                # Load training data files
                tdatadir = os.path.join(curdatadir,"train")
                train_files = [os.path.join(tdatadir,fname) for fname in os.listdir(tdatadir) if fname.endswith(".npz")]
                epoch0_train_files = [path for path in train_files if path not in train_state["data_files_used"]]
                if no_repeat_files:
                    logging.info(f"Dropping {len(train_files)-len(epoch0_train_files)}/{len(train_files)} files in: {tdatadir} as already used")
                else:
                    logging.info(f"Skipping {len(train_files)-len(epoch0_train_files)}/{len(train_files)} files in: {tdatadir} as already used first pass")

                if len(train_files) <= 0 or (no_repeat_files and len(epoch0_train_files) <= 0):
                    if quit_if_no_data:
                        logging.info(f"No new training files found in: {tdatadir}, quitting")
                        sys.exit(0)
                    logging.info(f"No new training files found in: {tdatadir}, waiting 30s and trying again")
                    time.sleep(30)
                    continue

                # Update history of what training data we used
                if tdatadir not in train_state["old_train_data_dirs"]:
                    train_state["old_train_data_dirs"].append(tdatadir)
                # Clear out tracking of sufficiently old files
                while len(train_state["old_train_data_dirs"]) > 20:
                    old_dir = train_state["old_train_data_dirs"][0]
                    train_state["old_train_data_dirs"] = train_state["old_train_data_dirs"][1:]
                    for filename in list(train_state["data_files_used"]):
                        if filename.startswith(old_dir):
                            train_state["data_files_used"].remove(filename)

                def train_files_gen():
                    train_files_shuffled = epoch0_train_files.copy()
                    while True:
                        random.shuffle(train_files_shuffled)
                        for filename in train_files_shuffled:
                            logging.info("Yielding training file for dataset: " + filename)
                            train_state["data_files_used"].add(filename)
                            yield filename
                        if no_repeat_files:
                            break
                        else:
                            train_files_shuffled = train_files.copy()
                            train_state["data_files_used"] = set()

                trainfilegenerator = PushBackGenerator(train_files_gen())
                vdatadir = os.path.join(curdatadir,"val")

            # Same directory as before, no new shuffle
            else:
                if max_train_steps_since_last_reload is not None:
                    if train_state["train_steps_since_last_reload"] + 0.99 * samples_per_epoch/sub_epochs > max_train_steps_since_last_reload:
                        logging.info(
                            "Too many train steps since last reload, waiting 5m and retrying (current %f)" %
                            train_state["train_steps_since_last_reload"]
                        )
                        time.sleep(300)
                        continue

            break

    # Load all the files we should train on during a subepoch
    def get_files_for_subepoch():
        nonlocal trainfilegenerator

        assert rank == 0, "Helper ddp training processes should not call get_files_for_subepoch"

        num_batches_per_epoch = int(round(samples_per_epoch / batch_size))
        num_batches_per_subepoch = num_batches_per_epoch / sub_epochs

        # Pick enough files to get the number of batches we want
        train_files_to_use = []
        batches_to_use_so_far = 0
        found_enough = False
        for filename in trainfilegenerator:
            jsonfilename = os.path.splitext(filename)[0] + ".json"
            with open(jsonfilename) as f:
                trainfileinfo = json.load(f)

            num_batches_this_file = trainfileinfo["num_rows"] // batch_size
            if num_batches_this_file <= 0:
                continue

            if batches_to_use_so_far + num_batches_this_file > num_batches_per_subepoch:
                # If we're going over the desired amount, randomly skip the file with probability equal to the
                # proportion of batches over - this makes it so that in expectation, we have the desired number of batches
                if batches_to_use_so_far > 0 and random.random() <= (batches_to_use_so_far + num_batches_this_file - num_batches_per_subepoch) / num_batches_this_file:
                    trainfilegenerator.push_back(filename)
                    found_enough = True
                    break

            train_files_to_use.append(filename)
            batches_to_use_so_far += num_batches_this_file

            #Sanity check - load a max of 100000 files.
            if batches_to_use_so_far >= num_batches_per_subepoch or len(train_files_to_use) > 100000:
                found_enough = True
                break

        if found_enough:
            return train_files_to_use
        return None

    # METRICS -----------------------------------------------------------------------------------
    def detensorify_metrics(metrics):
        ret = {}
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                ret[key] = metrics[key].detach().cpu().item()
            else:
                ret[key] = metrics[key]
        return ret

    if rank == 0:
        train_metrics_out = open(os.path.join(traindir,"metrics_train.json"),"a")
        val_metrics_out = open(os.path.join(traindir,"metrics_val.json"),"a")
    else:
        train_metrics_out = open(os.path.join(traindir,f"metrics_train_rank{rank}.json"),"a")
        val_metrics_out = open(os.path.join(traindir,f"metrics_val_rank{rank}.json"),"a")

    # TRAIN! -----------------------------------------------------------------------------------

    last_longterm_checkpoint_save_time = datetime.datetime.now()
    num_epochs_this_instance = 0
    print_train_loss_every_batches = 100 if not gnorm_stats_debug else 1000

    if "sums" not in running_metrics:
        running_metrics["sums"] = defaultdict(float)
    else:
        running_metrics["sums"] = defaultdict(float,running_metrics["sums"])
    if "weights" not in running_metrics:
        running_metrics["weights"] = defaultdict(float)
    else:
        running_metrics["weights"] = defaultdict(float,running_metrics["weights"])

    torch.backends.cudnn.benchmark = True

    if use_fp16:
        logging.info("Training in FP16! Creating scaler")
        scaler = GradScaler()
    else:
        logging.info("Training in FP32.")

    # All ddp threads should be lined up at this point before continuing
    if barrier is not None:
        barrier.wait()

    while True:
        if max_epochs_this_instance is not None and max_epochs_this_instance >= 0 and num_epochs_this_instance >= max_epochs_this_instance:
            logging.info("Hit max epochs this instance, done")
            break
        if max_training_samples is not None and train_state["global_step_samples"] >= max_training_samples:
            logging.info("Hit max training samples, done")
            break

        if rank == 0:
            maybe_reload_training_data()

            if max_train_bucket_per_new_data is not None:
                if train_state["train_bucket_level"] > 0.99 * samples_per_epoch:
                    logging.info("Consuming %.0f rows from train bucket (%.0f -> %.0f)" % (
                        samples_per_epoch, train_state["train_bucket_level"], train_state["train_bucket_level"]-samples_per_epoch
                    ))
                    train_state["train_bucket_level"] -= samples_per_epoch
                else:
                    if stop_when_train_bucket_limited:
                        logging.info(
                            "Exceeding train bucket, not enough new data rows, terminating (current level %f)" %
                            train_state["train_bucket_level"]
                        )
                        break
                    else:
                        logging.info(
                            "Exceeding train bucket, not enough new data rows, waiting 5m and retrying (current level %f)" %
                            train_state["train_bucket_level"]
                        )
                        time.sleep(300)
                        continue

        # DDP need to wait on the main process after reloading data and/or training bucket waiting
        if barrier is not None:
            barrier.wait()

        logging.info("GC collect")
        gc.collect()

        clear_metric_nonfinite(running_metrics["sums"], running_metrics["weights"])

        logging.info("=========================================================================")
        logging.info("BEGINNING NEXT EPOCH " + str(num_epochs_this_instance))
        logging.info("=========================================================================")
        logging.info("Current time: " + str(datetime.datetime.now()))
        logging.info("Global step: %d samples" % (train_state["global_step_samples"]))
        logging.info("Currently up to data row " + str(train_state["total_num_data_rows"]))
        logging.info(f"Training dir: {traindir}")
        logging.info(f"Export dir: {exportdir}")
        if use_fp16:
            logging.info(f"Current grad scale: {scaler.get_scale()}")

        lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()
        maybe_update_brenorm_params()

        # SUB EPOCH LOOP -----------
        batch_count_this_epoch = 0
        last_train_stats_time = time.perf_counter()
        for i in range(sub_epochs):

            if rank == 0:
                if i != 0:
                    maybe_reload_training_data()
                train_files_to_use = get_files_for_subepoch()
                while train_files_to_use is None or len(train_files_to_use) <= 0:
                    if quit_if_no_data:
                        logging.info("Not enough data files to fill a subepoch! Quitting.")
                        sys.exit(0)
                    logging.info("Not enough data files to fill a subepoch! Waiting 5m before retrying.")
                    time.sleep(300)
                    maybe_reload_training_data()
                    train_files_to_use = get_files_for_subepoch()

                if barrier is not None:
                    barrier.wait()
                for wpipe in writepipes:
                    wpipe.send(train_files_to_use)
                # Wait briefly just in case to reduce chance of races with filesystem or anything else
                time.sleep(5)
            else:
                if barrier is not None:
                    barrier.wait()
                train_files_to_use = readpipes[rank-1].recv()

            # DDP need to wait on the main process after reloading data and sending files to train with
            if barrier is not None:
                barrier.wait()

            logging.info("Beginning training subepoch!")
            logging.info("This subepoch, using files: " + str(train_files_to_use))
            logging.info("Currently up to data row " + str(train_state["total_num_data_rows"]))
            lookahead_counter = 0
            for batch in data_processing_pytorch.read_npz_training_data(
                train_files_to_use,
                batch_size,
                world_size,
                rank,
                pos_len=pos_len,
                device=device,
                randomize_symmetries=True,
                include_meta=raw_model.get_has_metadata_encoder(),
                model_config=model_config
            ):
                optimizer.zero_grad(set_to_none=True)
                extra_outputs = None
                # if raw_model.get_has_metadata_encoder():
                #     extra_outputs = ExtraOutputs([MetadataEncoder.OUTMEAN_KEY,MetadataEncoder.OUTLOGVAR_KEY])

                if use_fp16:
                    with autocast():
                        model_outputs = ddp_model(
                            batch["binaryInputNCHW"],
                            batch["globalInputNC"],
                            input_meta=(batch["metadataInputNC"] if raw_model.get_has_metadata_encoder() else None),
                            extra_outputs=extra_outputs,
                        )
                    model_outputs = raw_model.float32ify_output(model_outputs)
                else:
                    model_outputs = ddp_model(
                        batch["binaryInputNCHW"],
                        batch["globalInputNC"],
                        input_meta=(batch["metadataInputNC"] if raw_model.get_has_metadata_encoder() else None),
                        extra_outputs=extra_outputs,
                    )

                postprocessed = raw_model.postprocess_output(model_outputs)
                metrics = metrics_obj.metrics_dict_batchwise(
                    raw_model,
                    postprocessed,
                    extra_outputs,
                    batch,
                    is_training=True,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    disable_optimistic_policy=disable_optimistic_policy,
                    meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=seki_loss_scale,
                    variance_time_loss_scale=variance_time_loss_scale,
                    main_loss_scale=main_loss_scale,
                    intermediate_loss_scale=intermediate_loss_scale,
                )

                # DDP averages loss across instances, so to preserve LR as per-sample lr, we scale by world size.
                loss = metrics["loss_sum"] * world_size

                # Reduce gradients across DDP
                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if model_config["norm_kind"] == "fixup" or model_config["norm_kind"] == "fixscale" or model_config["norm_kind"] == "fixscaleonenorm":
                    gnorm_cap = 2500.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
                elif model_config["norm_kind"] == "bnorm" or model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
                    gnorm_cap = 5500.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
                else:
                    assert False

                if gnorm_stats_debug:
                    stats = metrics_obj.get_specific_norms_and_gradient_stats(raw_model)
                    for stat, value in stats.items():
                        metrics[stat] = value

                if "use_repvgg_learning_rate" in model_config and model_config["use_repvgg_learning_rate"]:
                    gradscale_constant = torch.tensor([[1.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,1.0]],dtype=torch.float32,device=device,requires_grad=False).view(1,1,3,3)
                    for name, param in ddp_model.named_parameters():
                        if "normactconv" in name and ".conv.weight" in name and len(param.shape) == 4 and param.shape[2] == 3 and param.shape[3] == 3:
                            param.grad *= gradscale_constant

                # Loosen gradient clipping as we shift to smaller learning rates
                gnorm_cap = gnorm_cap / math.sqrt(max(0.0000001,lr_scale * lr_scale_auto_factor(train_state)))

                gnorm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gnorm_cap).detach().cpu().item()

                if math.isfinite(gnorm) and abs(gnorm < 1e30):
                    metrics["gnorm_batch"] = gnorm
                    exgnorm = max(0.0, gnorm - gnorm_cap)
                    metrics["exgnorm_sum"] = exgnorm * batch_size

                metrics["pslr_batch"] = lr_right_now
                metrics["wdnormal_batch"] = normal_weight_decay_right_now
                metrics["gnorm_cap_batch"] = gnorm_cap
                metrics["window_start_batch"] = train_state["window_start_data_row_idx"]
                metrics["window_end_batch"] = train_state["total_num_data_rows"]

                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                batch_count_this_epoch += 1
                train_state["train_steps_since_last_reload"] += batch_size * world_size
                train_state["global_step_samples"] += batch_size * world_size

                metrics = detensorify_metrics(metrics)

                if lookahead_k is not None and lookahead_print:
                    # Only accumulate metrics when lookahead is synced if lookahead_print is True
                    if lookahead_counter == 0:
                        accumulate_metrics(running_metrics["sums"], running_metrics["weights"], metrics, batch_size, decay=math.exp(-0.001 * lookahead_k), new_weight=1.0)
                    else:
                        accumulate_metrics(running_metrics["sums"], running_metrics["weights"], metrics, batch_size, decay=1.0, new_weight=0.0)
                else:
                    accumulate_metrics(running_metrics["sums"], running_metrics["weights"], metrics, batch_size, decay=0.999, new_weight=1.0)


                if batch_count_this_epoch % print_train_loss_every_batches == 0:

                    if model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
                        metrics["brn_rmax"] = train_state["brenorm_rmax"]
                        metrics["brn_dmax"] = train_state["brenorm_dmax"]
                        metrics["brn_mmnt"] = brenorm_avg_momentum
                        upper_rclippage = []
                        lower_rclippage = []
                        dclippage = []
                        raw_model.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
                        metrics["brn_ruclip"] = sum(upper_rclippage) / max(len(upper_rclippage),1.0)
                        metrics["brn_rlclip"] = sum(lower_rclippage) / max(len(lower_rclippage),1.0)
                        metrics["brn_dclip"] = sum(dclippage) / max(len(dclippage),1.0)

                    t1 = time.perf_counter()
                    timediff = t1 - last_train_stats_time
                    last_train_stats_time = t1
                    metrics["time_since_last_print"] = timediff
                    log_metrics(running_metrics["sums"], running_metrics["weights"], metrics, train_metrics_out)

                # Update LR more frequently at the start for smoother warmup ramp and wd adjustment
                if train_state["global_step_samples"] <= 50000000 and batch_count_this_epoch % 50 == 0:
                    lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()

                # Update batch renorm parameters
                if batch_count_this_epoch % 500 == 0:
                    maybe_update_brenorm_params()

                # Perform lookahead
                in_between_lookaheads = False
                if lookahead_k is not None:
                    lookahead_counter += 1
                    if lookahead_counter >= lookahead_k:
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                slow_param_data = lookahead_cache[param]
                                slow_param_data.add_(param.data.detach() - slow_param_data, alpha=lookahead_alpha)
                                param.data.copy_(slow_param_data)
                        lookahead_counter = 0
                        in_between_lookaheads = False
                    else:
                        in_between_lookaheads = True

                # Perform SWA
                if swa_model is not None and swa_scale is not None:
                    train_state["swa_sample_accum"] += batch_size * world_size
                    # Only snap SWA when lookahead slow params are in sync.
                    if train_state["swa_sample_accum"] >= swa_period_samples and not in_between_lookaheads:
                        train_state["swa_sample_accum"] = 0
                        logging.info("Accumulating SWA")
                        swa_model.update_parameters(raw_model)

            logging.info("Finished training subepoch!")

        # END SUB EPOCH LOOP ------------

        # Discard the gradient updates from the leftover batches in the sub epoch from lookahead.
        # This wastes a very tiny bit, but makes it so that we can be in sync and deterministic on ends of subepochs/epochs.
        if lookahead_k is not None:
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    slow_param_data = lookahead_cache[param]
                    param.data.copy_(slow_param_data)

        if rank == 0:
            train_state["export_cycle_counter"] += 1

        save(ddp_model, swa_model, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics)

        num_epochs_this_instance += 1

        # Validate
        if rank == 0:
            logging.info("Beginning validation after epoch!")
            val_files = []
            if os.path.exists(vdatadir):
                val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir) if fname.endswith(".npz")]
            if randomize_val:
                random.shuffle(val_files)
            else:
                # Sort to ensure deterministic order to validation files in case we use only a subset
                val_files = sorted(val_files)
            if len(val_files) == 0:
                logging.info("No validation files, skipping validation step")
            else:
                with torch.no_grad():
                    ddp_model.eval()
                    val_metric_sums = defaultdict(float)
                    val_metric_weights = defaultdict(float)
                    val_samples = 0
                    t0 = time.perf_counter()
                    for batch in data_processing_pytorch.read_npz_training_data(
                        val_files,
                        batch_size,
                        world_size=1,  # Only the main process validates
                        rank=0,        # Only the main process validates
                        pos_len=pos_len,
                        device=device,
                        randomize_symmetries=True,
                        include_meta=raw_model.get_has_metadata_encoder(),
                        model_config=model_config
                    ):
                        model_outputs = ddp_model(
                            batch["binaryInputNCHW"],
                            batch["globalInputNC"],
                            input_meta=(batch["metadataInputNC"] if raw_model.get_has_metadata_encoder() else None),
                        )
                        postprocessed = raw_model.postprocess_output(model_outputs)
                        extra_outputs = None
                        metrics = metrics_obj.metrics_dict_batchwise(
                            raw_model,
                            postprocessed,
                            extra_outputs,
                            batch,
                            is_training=False,
                            soft_policy_weight_scale=soft_policy_weight_scale,
                            disable_optimistic_policy=disable_optimistic_policy,
                            meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                            value_loss_scale=value_loss_scale,
                            td_value_loss_scales=td_value_loss_scales,
                            seki_loss_scale=seki_loss_scale,
                            variance_time_loss_scale=variance_time_loss_scale,
                            main_loss_scale=main_loss_scale,
                            intermediate_loss_scale=intermediate_loss_scale,
                        )
                        metrics = detensorify_metrics(metrics)
                        accumulate_metrics(val_metric_sums, val_metric_weights, metrics, batch_size, decay=1.0, new_weight=1.0)
                        val_samples += batch_size
                        if max_val_samples is not None and val_samples > max_val_samples:
                            break
                        val_metric_sums["nsamp_train"] = running_metrics["sums"]["nsamp"]
                        val_metric_weights["nsamp_train"] = running_metrics["weights"]["nsamp"]
                        val_metric_sums["wsum_train"] = running_metrics["sums"]["wsum"]
                        val_metric_weights["wsum_train"] = running_metrics["weights"]["wsum"]
                    last_val_metrics["sums"] = val_metric_sums
                    last_val_metrics["weights"] = val_metric_weights
                    log_metrics(val_metric_sums, val_metric_weights, metrics, val_metrics_out)
                    t1 = time.perf_counter()
                    logging.info(f"Validation took {t1-t0} seconds")
                    ddp_model.train()

        if rank == 0:
            logging.info("Export cycle counter = " + str(train_state["export_cycle_counter"]))

            is_time_to_export = False
            if train_state["export_cycle_counter"] >= epochs_per_export:
                if no_export:
                    train_state["export_cycle_counter"] = epochs_per_export
                else:
                    train_state["export_cycle_counter"] = 0
                    is_time_to_export = True

            skip_export_this_time = False
            if export_prob is not None:
                if random.random() > export_prob:
                    skip_export_this_time = True
                    logging.info("Skipping export model this time")

            if not no_export and is_time_to_export and not skip_export_this_time and exportdir is not None and not gnorm_stats_debug:
                # Export a model for testing, unless somehow it already exists
                modelname = "%s-s%d-d%d" % (
                    exportprefix,
                    train_state["global_step_samples"],
                    train_state["total_num_data_rows"],
                )
                savepath = os.path.join(exportdir,modelname)
                savepathtmp = os.path.join(exportdir,modelname+".tmp")
                if os.path.exists(savepath):
                    logging.info("NOT saving model, already exists at: " + savepath)
                else:
                    os.mkdir(savepathtmp)
                    logging.info("SAVING MODEL FOR EXPORT TO: " + savepath)
                    save(ddp_model, swa_model, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics, path=os.path.join(savepathtmp,"model.ckpt"))
                    time.sleep(2)
                    os.rename(savepathtmp,savepath)


        if sleep_seconds_per_epoch is None:
            time.sleep(1)
        else:
            time.sleep(sleep_seconds_per_epoch)

        if rank == 0:
            now = datetime.datetime.now()
            if now - last_longterm_checkpoint_save_time >= datetime.timedelta(hours=12):
                last_longterm_checkpoint_save_time = now
                dated_name = datetime.datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                save(ddp_model, swa_model, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics, path=os.path.join(longterm_checkpoints_dir,f"{dated_name}.ckpt"))

    train_metrics_out.close()
    val_metrics_out.close()


if __name__ == "__main__":
    multi_gpus = args["multi_gpus"]
    num_gpus_used = 1
    multi_gpu_device_ids = []
    if multi_gpus is not None:
        for piece in multi_gpus.split(","):
            piece = piece.strip()
            multi_gpu_device_ids.append(int(piece))
        num_gpus_used = len(multi_gpu_device_ids)
    else:
        multi_gpu_device_ids = [0]

    make_dirs(args)

    readpipes = []
    writepipes = []

    if num_gpus_used > 1:
        torch.multiprocessing.set_start_method("spawn")

        world_size = num_gpus_used
        barrier = torch.multiprocessing.Barrier(num_gpus_used)

        for i in range(world_size - 1):
            rpipe, wpipe = torch.multiprocessing.Pipe()
            readpipes.append(rpipe)
            writepipes.append(wpipe)

        torch.multiprocessing.spawn(
            main,
            nprocs=num_gpus_used,
            args=(world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier)
        )
    else:
        rank = 0
        world_size = 1
        barrier = None
        main(rank, world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier)
