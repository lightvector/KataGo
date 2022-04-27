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

import modelconfigs
from model_pytorch import Model
from metrics_pytorch import Metrics
import data_processing_pytorch

# HANDLE COMMAND AND ARGS -------------------------------------------------------------------

if __name__ == "__main__":

  description = """
  Train neural net on Go positions from npz files of batches from selfplay.
  """

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
  parser.add_argument('-datadir', help='Directory with a train and val subdir of npz data', required=True)
  parser.add_argument('-exportdir', help='Directory to export models periodically', required=True)
  parser.add_argument('-exportprefix', help='Prefix to append to names of models', required=True)
  parser.add_argument('-initial-checkpoint', help='If no training checkpoint exists, initialize from this checkpoint', required=False)
  parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
  parser.add_argument('-batch-size', help='Batch size to use for training', type=int, required=True)
  parser.add_argument('-samples-per-epoch', help='Number of data samples to consider as one epoch', type=int, required=False)
  parser.add_argument('-multi-gpus', help='Use multiple gpus, comma-separated device ids', required=False)
  parser.add_argument('-model-kind', help='String name for what model to use', required=True)
  parser.add_argument('-lr-scale', help='LR multiplier on the hardcoded schedule', type=float, required=False)
  parser.add_argument('-gnorm-clip-scale', help='Multiplier on gradient clipping threshold', type=float, required=False)
  parser.add_argument('-sub-epochs', help='Reload training data up to this many times per epoch', type=int, required=True)
  parser.add_argument('-epochs-per-export', help='Export model once every this many epochs', type=int, required=False)
  parser.add_argument('-export-prob', help='Export model with this probablity', type=float, required=False)
  parser.add_argument('-max-epochs-this-instance', help='Terminate training after this many more epochs', type=int, required=False)
  parser.add_argument('-sleep-seconds-per-epoch', help='Sleep this long between epochs', type=int, required=False)
  parser.add_argument('-swa-sub-epoch-scale', help='Number of sub-epochs to average in expectation together for SWA', type=float, required=False)
  parser.add_argument('-max-train-bucket-per-new-data', help='When data added, add this many train rows per data row to bucket', type=float, required=False)
  parser.add_argument('-max-train-bucket-size', help='Approx total number of train rows allowed if data stops', type=float, required=False)
  parser.add_argument('-max-train-steps-since-last-reload', help='Approx total of training allowed if shuffling stops', type=float, required=False)
  parser.add_argument('-max-val-samples', help='Approx max of validation samples per epoch', type=int, required=False)
  parser.add_argument('-no-export', help='Do not export models', required=False, action='store_true')

  parser.add_argument('-brenorm-avg-momentum', type=float, help='Set brenorm running avg rate to this value', required=False)
  parser.add_argument('-brenorm-target-rmax', type=float, help='Gradually adjust brenorm rmax to this value', required=False)
  parser.add_argument('-brenorm-target-dmax', type=float, help='Gradually adjust brenorm dmax to this value', required=False)
  parser.add_argument('-brenorm-adjustment-scale', type=float, help='How many samples to adjust brenorm params all but 1/e of the way to target', required=False)

  parser.add_argument('-soft-policy-weight-scale', type=float, default=1.0, help='Soft policy loss coeff', required=False)

  parser.add_argument('-main-loss-scale', type=float, help='Loss factor scale for main head', required=False)
  parser.add_argument('-intermediate-loss-scale', type=float, help='Loss factor scale for intermediate head', required=False)
  parser.add_argument('-intermediate-distill-scale', type=float, help='Distill factor scale for intermediate head', required=False)


  args = vars(parser.parse_args())


def get_longterm_checkpoints_dir(traindir):
  return os.path.join(traindir,"longterm_checkpoints")

def make_dirs(args):
  traindir = args["traindir"]
  exportdir = args["exportdir"]

  if not os.path.exists(traindir):
    os.makedirs(traindir)
  if not os.path.exists(exportdir):
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

def dump_and_flush_json(data,filename):
  with open(filename,"w") as f:
    json.dump(data,f)
    f.flush()
    os.fsync(f.fileno())

def main(rank: int, world_size: int, args, multi_gpu_device_ids):
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
  gnorm_clip_scale = args["gnorm_clip_scale"]
  sub_epochs = args["sub_epochs"]
  epochs_per_export = args["epochs_per_export"]
  export_prob = args["export_prob"]
  max_epochs_this_instance = args["max_epochs_this_instance"]
  sleep_seconds_per_epoch = args["sleep_seconds_per_epoch"]
  swa_sub_epoch_scale = args["swa_sub_epoch_scale"]
  max_train_bucket_per_new_data = args["max_train_bucket_per_new_data"]
  max_train_bucket_size = args["max_train_bucket_size"]
  max_train_steps_since_last_reload = args["max_train_steps_since_last_reload"]
  max_val_samples = args["max_val_samples"]
  no_export = args["no_export"]

  brenorm_target_rmax = args["brenorm_target_rmax"]
  brenorm_target_dmax = args["brenorm_target_dmax"]
  brenorm_avg_momentum = args["brenorm_avg_momentum"]
  brenorm_adjustment_scale = args["brenorm_adjustment_scale"]
  soft_policy_weight_scale = args["soft_policy_weight_scale"]

  main_loss_scale = args["main_loss_scale"]
  intermediate_loss_scale = args["intermediate_loss_scale"]
  intermediate_distill_scale = args["intermediate_distill_scale"]

  if lr_scale is None:
    lr_scale = 1.0

  if samples_per_epoch is None:
    samples_per_epoch = 1000000
  if max_train_bucket_size is None:
    max_train_bucket_size = 1.0e30
  if epochs_per_export is None:
    epochs_per_export = 1

  num_batches_per_epoch = int(round(samples_per_epoch / batch_size))
  longterm_checkpoints_dir = get_longterm_checkpoints_dir(traindir)

  # SET UP LOGGING -------------------------------------------------------------

  logging.root.handlers = []
  logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
      logging.FileHandler(os.path.join(traindir,f"train{rank}.log"), mode="a"),
      logging.StreamHandler()
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

  # LOAD MODEL ---------------------------------------------------------------------

  def get_checkpoint_path():
    return os.path.join(traindir,"checkpoint.ckpt")
  def get_checkpoint_prev_path(i):
    return os.path.join(traindir,f"checkpoint_prev{i}.ckpt")

  NUM_SHORTTERM_CHECKPOINTS_TO_KEEP = 4
  def save(model, swa_model, optimizer, metrics_obj, running_metrics, train_state, path=None):
    if rank == 0:
      state_dict = {}
      state_dict["model"] = model.state_dict()
      state_dict["optimizer"] = optimizer.state_dict()
      state_dict["metrics"] = metrics_obj.state_dict()
      state_dict["running_metrics"] = running_metrics
      state_dict["train_state"] = train_state

      if swa_model is not None:
        state_dict["swa_model"] = swa_model.state_dict()

      if path is not None:
        logging.info("Saving checkpoint: " + path)
        torch.save(state_dict, path + ".tmp")
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

  def get_weight_decay(model, lr_scale, warmup_scale, train_state, running_metrics, group_name):
    if model.get_norm_kind() == "fixup" or model.get_norm_kind() == "fixscale":
      if group_name == "normal":
        return 0.000001 * world_size * batch_size / 256.0
      elif group_name == "output":
        return 0.000001 * world_size * batch_size / 256.0
      elif group_name == "noreg":
        return 0.0
      elif group_name == "output_noreg":
        return 0.0
      else:
        assert False
    elif model.get_norm_kind() == "bnorm" or model.get_norm_kind() == "brenorm" or model.get_norm_kind() == "fixbrenorm" or model.get_norm_kind() == "fixscaleonenorm":
      if group_name == "normal":
        adaptive_scale = 1.0
        if "sums" in running_metrics and "norm_normal_batch" in running_metrics["sums"]:
          norm_normal_batch = running_metrics["sums"]["norm_normal_batch"] / running_metrics["weights"]["norm_normal_batch"]
          baseline = train_state["modelnorm_normal_baseline"]
          ratio = norm_normal_batch / (baseline + 1e-30)
          # Adaptive weight decay keeping model norm around the baseline level so that batchnorm effective lr is held constant
          # throughout training, covering a range of 16x from bottom to top.
          adaptive_scale = math.pow(2.0, 2.0 * math.tanh(math.log(ratio+1e-30) * 1.5))

        # The theoretical scaling for keeping us confined to a surface of equal model norm should go proportionally with lr_scale.
        # because the strength of drift away from that surface goes as lr^2 and weight decay itself is scaled by lr, so we need
        # one more factor of lr to make weight decay strength equal drift strength.
        # However, at low lr it tends to be the case that gradient norm increases slightly
        # while at high lr it tends to be the case that gradient norm decreases, which means drift strength scales a bit slower
        # than expected.
        # So we scale sublinearly with lr_scale so as to slightly preadjust to this effect.
        # Adaptive scale should then help keep us there thereafter.
        return 0.00145 * world_size * batch_size / 256.0 * math.pow(lr_scale * warmup_scale,0.75) * adaptive_scale
      elif group_name == "output":
        return 0.000001 * world_size * batch_size / 256.0
      elif group_name == "noreg":
        return 0.0
      elif group_name == "output_noreg":
        return 0.0
      else:
        assert False
    else:
      assert False

  def get_param_groups(model,train_state,running_metrics):
    reg_dict : Dict[str,List] = {}
    model.add_reg_dict(reg_dict)
    param_groups = []
    param_groups.append({
      "params": reg_dict["normal"],
      "weight_decay": get_weight_decay(model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="normal"),
      "group_name": "normal",
    })
    param_groups.append({
      "params": reg_dict["output"],
      "weight_decay": get_weight_decay(model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="output"),
      "group_name": "output",
    })
    param_groups.append({
      "params": reg_dict["noreg"],
      "weight_decay": get_weight_decay(model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="noreg"),
      "group_name": "noreg",
    })
    param_groups.append({
      "params": reg_dict["output_noreg"],
      "weight_decay": get_weight_decay(model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="output_noreg"),
      "group_name": "output_noreg",
    })
    num_params = len(list(model.parameters()))
    num_reg_dict_params = len(reg_dict["normal"]) + len(reg_dict["output"]) + len(reg_dict["noreg"]) + len(reg_dict["output_noreg"])
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

    # Load model config
    if os.path.exists(os.path.join(traindir,"model.config.json")):
      logging.info("Loading existing model config at %s" % os.path.join(traindir,"model.config.json"))
      with open(os.path.join(traindir,"model.config.json"),"r") as f:
        model_config = json.load(f)
      if path_to_load_from is None:
        logging.warning("WARNING: No existing model but loading params from existing model.config.json!")
    else:
      model_config = modelconfigs.config_of_name[model_kind]
      logging.info("Initializing with new model config")
      assert path_to_load_from is None, "Found existing model but no existing model.config.json?"
      with open(os.path.join(traindir,"model.config.json"),"w") as f:
        json.dump(model_config,f)

    logging.info(str(model_config))

    if path_to_load_from is None:
      logging.info("Initializing new model!")
      model = Model(model_config,pos_len)
      model.initialize()

      model.to(device)
      if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

      swa_model = None
      if rank == 0 and swa_sub_epoch_scale is not None:
        new_factor = 1.0 / swa_sub_epoch_scale
        ema_avg = lambda avg_param, cur_param, num_averaged: (1.0 - new_factor) * avg_param + new_factor * cur_param
        swa_model = AveragedModel(model, avg_fn=ema_avg)

      metrics_obj = Metrics(batch_size,model)
      running_metrics = {}
      train_state = {}

      with torch.no_grad():
        (modelnorm_normal, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = metrics_obj.get_model_norms(model)
        modelnorm_normal_baseline = modelnorm_normal.detach().cpu().item()
        train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline

      optimizer = torch.optim.SGD(get_param_groups(model,train_state,running_metrics), lr=1.0, momentum=0.9)

      return (model_config, model, swa_model, optimizer, metrics_obj, running_metrics, train_state)
    else:
      state_dict = torch.load(path_to_load_from, map_location=device)
      model = Model(model_config,pos_len)
      model.initialize()

      # Strip off any "module." from when the model was saved with DDP or other things
      model_state_dict = {}
      for key in state_dict["model"]:
        old_key = key
        while key.startswith("module."):
          key = key[:7]
        model_state_dict[key] = state_dict["model"][old_key]
      model.load_state_dict(model_state_dict)

      model.to(device)
      if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

      swa_model = None
      if rank == 0 and swa_sub_epoch_scale is not None:
        new_factor = 1.0 / swa_sub_epoch_scale
        ema_avg = lambda avg_param, cur_param, num_averaged: (1.0 - new_factor) * avg_param + new_factor * cur_param
        swa_model = AveragedModel(model, avg_fn=ema_avg)
        if "swa_model" in state_dict:
          swa_model.load_state_dict(state_dict["swa_model"])

      metrics_obj = Metrics(batch_size,model)
      if "metrics" in state_dict:
        metrics_obj.load_state_dict(state_dict["metrics"])
      else:
        logging.info("WARNING: Metrics not found in state dict, using fresh metrics")

      running_metrics = {}
      if "running_metrics" in state_dict:
        running_metrics = state_dict["running_metrics"]
      else:
        logging.info("WARNING: Running metrics not found in state dict, using fresh running metrics")

      train_state = {}
      if "train_state" in state_dict:
        train_state = state_dict["train_state"]
      else:
        logging.info("WARNING: Train state not found in state dict, using fresh train state")
        with torch.no_grad():
          (modelnorm_normal, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = metrics_obj.get_model_norms(model)
          modelnorm_normal_baseline = modelnorm_normal.detach().cpu().item()
          train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline

      optimizer = torch.optim.SGD(get_param_groups(model,train_state,running_metrics), lr=1.0, momentum=0.9)
      if "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
      else:
        logging.info("WARNING: Optimizer not found in state dict, using fresh optimizer")

      return (model_config, model, swa_model, optimizer, metrics_obj, running_metrics, train_state)

  (model_config, model, swa_model, optimizer, metrics_obj, running_metrics, train_state) = load()


  if "global_step_samples" not in train_state:
    train_state["global_step_samples"] = 0
  if max_train_bucket_per_new_data is not None and "train_bucket_level" not in train_state:
    train_state["train_bucket_level"] = samples_per_epoch
  if "train_steps_since_last_reload" not in train_state:
    train_state["train_steps_since_last_reload"] = 0
  if "export_cycle_counter" not in train_state:
    train_state["export_cycle_counter"] = 0

  if intermediate_distill_scale is not None or intermediate_loss_scale is not None:
    assert model.get_has_intermediate_head(), "Model must have intermediate head to use intermediate distill or loss"


  # Print all model parameters just to get a summary
  total_num_params = 0
  total_trainable_params = 0
  logging.info("Parameters in model:")
  for name, param in model.named_parameters():
    product = 1
    for dim in param.shape:
      product *= int(dim)
    if param.requires_grad:
      total_trainable_params += product
    total_num_params += product
    logging.info(f"{name}, {list(param.shape)}, {product} params")
  logging.info(f"Total num params: {total_num_params}")
  logging.info(f"Total trainable params: {total_trainable_params}")


  # EPOCHS AND LR ---------------------------------------------------------------------

  def update_and_return_lr_and_wd():
    per_sample_lr = 0.00003 * lr_scale

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
      elif group_name == "output":
        group_scale = 0.5
      elif group_name == "noreg":
        group_scale = 1.0
      elif group_name == "output_noreg":
        group_scale = 0.5
      else:
        assert False

      param_group["lr"] = per_sample_lr * warmup_scale * group_scale
      param_group["weight_decay"] = get_weight_decay(
        model,
        lr_scale,
        warmup_scale=warmup_scale,
        train_state=train_state,
        running_metrics=running_metrics,
        group_name=group_name,
      )
      if group_name == "normal":
        normal_weight_decay = param_group["weight_decay"]

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

      model.set_brenorm_params(brenorm_avg_momentum, train_state["brenorm_rmax"], train_state["brenorm_dmax"])
      last_brenorm_update_samples_this_instance = train_state["global_step_samples"]

  # DATA RELOADING GENERATOR AND TRAINHISTORY ------------------------------------------------------------

  # Some globals
  last_curdatadir = None
  last_datainfo_row = 0
  trainfilegenerator = None
  num_train_files = 0
  vdatadir = None

  # Purely informational tracking of history of training
  trainhistory = {
    "history":[]
  }
  if os.path.isfile(os.path.join(traindir,"trainhistory.json")):
    logging.info("Loading existing training history: " + str(os.path.join(traindir,"trainhistory.json")))
    with open(os.path.join(traindir,"trainhistory.json")) as f:
      trainhistory = json.load(f)

  trainhistory["history"].append(("started",str(datetime.datetime.now(timezone.utc))))

  def save_history():
    if rank == 0:
      trainhistory["train_state"] = copy.deepcopy(train_state)
      trainhistory["extra_stats"] = copy.deepcopy(running_metrics)
      savepath = os.path.join(traindir,"trainhistory.json")
      savepathtmp = os.path.join(traindir,"trainhistory.json.tmp")
      dump_and_flush_json(trainhistory,savepathtmp)
      os.replace(savepathtmp,savepath)
      logging.info("Wrote " + savepath)

  def maybe_reload_training_data():
    nonlocal last_curdatadir
    nonlocal last_datainfo_row
    nonlocal trainfilegenerator
    nonlocal num_train_files
    nonlocal vdatadir

    if rank != 0:
      assert False # TODO need to figure out what to do here and for buckets and such
      return

    while True:
      curdatadir = os.path.realpath(datadir)

      # Different directory - new shuffle
      if curdatadir != last_curdatadir:
        if not os.path.exists(curdatadir):
          logging.info("Shuffled data path does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % curdatadir)
          time.sleep(30)
          continue

        trainjsonpath = os.path.join(curdatadir,"train.json")
        if not os.path.exists(trainjsonpath):
          logging.info("Shuffled data train.json file does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % trainjsonpath)
          time.sleep(30)
          continue

        logging.info("Updated training data: " + curdatadir)
        last_curdatadir = curdatadir

        with open(trainjsonpath) as f:
          datainfo = json.load(f)
          last_datainfo_row = datainfo["range"][1]

        if max_train_bucket_per_new_data is not None:
          if "train_bucket_level_at_row" not in trainhistory:
            train_state["train_bucket_level_at_row"] = last_datainfo_row
          if last_datainfo_row > train_state["train_bucket_level_at_row"]:
            new_row_count = last_datainfo_row - train_state["train_bucket_level_at_row"]
            logging.info("Advancing trainbucket row %.0f to %.0f, %.0f new rows" % (
              train_state["train_bucket_level_at_row"], last_datainfo_row, new_row_count
            ))
            train_state["train_bucket_level_at_row"] = last_datainfo_row
            logging.info("Fill per data %.3f, Max bucket size %.0f" % (max_train_bucket_per_new_data, max_train_bucket_size))
            logging.info("Old rows in bucket: %.0f" % train_state["train_bucket_level"])
            train_state["train_bucket_level"] += new_row_count * max_train_bucket_per_new_data
            cap = max(max_train_bucket_size, samples_per_epoch)
            if train_state["train_bucket_level"] > cap:
              train_state["train_bucket_level"] = cap
            logging.info("New rows in bucket: %.0f" % train_state["train_bucket_level"])

        logging.info("Train steps since last reload: %.0f -> 0" % train_state["train_steps_since_last_reload"])
        train_state["train_steps_since_last_reload"] = 0

        trainhistory["history"].append(("newdata",train_state["global_step_samples"],datainfo["range"]))

        # Load training data files
        tdatadir = os.path.join(curdatadir,"train")
        train_files = [os.path.join(tdatadir,fname) for fname in os.listdir(tdatadir) if fname.endswith(".npz")]
        num_train_files = len(train_files)

        # Filter down to a random subset that will comprise this epoch
        def train_files_gen():
          train_files_shuffled = train_files.copy()
          while True:
            random.shuffle(train_files_shuffled)
            for filename in train_files_shuffled:
              logging.info("Yielding training file for dataset: " + filename)
              yield filename
        trainfilegenerator = train_files_gen()

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

  # METRICS -----------------------------------------------------------------------------------
  def detensorify_metrics(metrics):
    ret = {}
    for key in metrics:
      if isinstance(metrics[key], torch.Tensor):
        ret[key] = metrics[key].detach().cpu().item()
      else:
        ret[key] = metrics[key]
    return ret

  def accumulate_metrics(metric_sums, metric_weights, metrics, batch_size, decay):
    if decay != 1.0:
      for metric in metric_sums:
        if metric.endswith("_sum"):
          metric_sums[metric] *= decay
          metric_weights[metric] *= decay

    for metric in metrics:
      if not metric.endswith("_batch"):
        metric_sums[metric] += metrics[metric]
        metric_weights[metric] += batch_size
      else:
        metric_sums[metric] += metrics[metric]
        metric_weights[metric] += 1

  def log_metrics(metric_sums, metric_weights, metrics, metrics_out):
    metrics_to_print = {}
    for metric in metric_sums:
      if metric.endswith("_sum"):
        metrics_to_print[metric[:-4]] = metric_sums[metric] / metric_weights[metric]
      elif metric.endswith("_batch"):
        metrics_to_print[metric] = metric_sums[metric] / metric_weights[metric]
        metric_sums[metric] *= 0.001
        metric_weights[metric] *= 0.001
      else:
        metrics_to_print[metric] = metric_sums[metric]
    for metric in metrics:
      if metric not in metric_sums:
        metrics_to_print[metric] = metrics[metric]

    logging.info(", ".join(["%s = %f" % (metric, metrics_to_print[metric]) for metric in metrics_to_print]))
    if metrics_out:
      metrics_out.write(json.dumps(metrics_to_print) + "\n")
      metrics_out.flush()

  train_metrics_out = open(os.path.join(traindir,"metrics_train.json"),"a")
  val_metrics_out = open(os.path.join(traindir,"metrics_val.json"),"a")

  # TRAIN! -----------------------------------------------------------------------------------

  last_longterm_checkpoint_save_time = datetime.datetime.now()
  num_epochs_this_instance = 0
  print_train_loss_every_batches = 100

  if "sums" not in running_metrics:
    running_metrics["sums"] = defaultdict(float)
  else:
    running_metrics["sums"] = defaultdict(float,running_metrics["sums"])
  if "weights" not in running_metrics:
    running_metrics["weights"] = defaultdict(float)
  else:
    running_metrics["weights"] = defaultdict(float,running_metrics["weights"])

  torch.backends.cudnn.benchmark = True

  while True:
    maybe_reload_training_data()
    logging.info("GC collect")
    gc.collect()

    lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()
    maybe_update_brenorm_params()

    logging.info("=========================================================================")
    logging.info("BEGINNING NEXT EPOCH " + str(num_epochs_this_instance))
    logging.info("=========================================================================")
    logging.info("Current time: " + str(datetime.datetime.now()))
    logging.info("Global step: %d samples" % (train_state["global_step_samples"]))
    logging.info("Currently up to data row " + str(last_datainfo_row))
    logging.info(f"Training dir: {traindir}")
    logging.info(f"Export dir: {exportdir}")

    if max_train_bucket_per_new_data is not None:
      if train_state["train_bucket_level"] > 0.99 * samples_per_epoch:
        logging.info("Consuming %.0f rows from train bucket (%.0f -> %.0f)" % (
          samples_per_epoch, train_state["train_bucket_level"], train_state["train_bucket_level"]-samples_per_epoch
        ))
        train_state["train_bucket_level"] -= samples_per_epoch
      else:
        logging.info(
          "Exceeding train bucket, not enough new data rows, waiting 5m and retrying (current level %f)" %
          train_state["train_bucket_level"]
        )
        time.sleep(300)
        continue

    # SUB EPOCH LOOP -----------
    batch_count_this_epoch = 0
    last_train_stats_time = time.perf_counter()
    num_batches_per_subepoch = num_batches_per_epoch / sub_epochs
    for i in range(sub_epochs):
      if i != 0:
        maybe_reload_training_data()

      # Pick enough files to get the number of batches we want
      train_files_to_use = []
      batches_to_use_so_far = 0
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
          if batches_to_use_so_far > 0 and random.random() >= (batches_to_use_so_far + num_batches_this_file - num_batches_per_subepoch) / num_batches_this_file:
            break

        train_files_to_use.append(filename)
        batches_to_use_so_far += num_batches_this_file

        #Sanity check - load a max of 100000 files.
        if batches_to_use_so_far >= num_batches_per_subepoch or len(train_files_to_use) > 100000:
          break

      logging.info("Beginning training subepoch!")
      logging.info("Currently up to data row " + str(last_datainfo_row))
      for batch in data_processing_pytorch.read_npz_training_data(
          train_files_to_use, batch_size, pos_len, device, randomize_symmetries=True, model_config=model_config
      ):
        optimizer.zero_grad(set_to_none=True)
        model_outputs = model(batch["binaryInputNCHW"],batch["globalInputNC"])
        postprocessed = model.postprocess_output(model_outputs)
        metrics = metrics_obj.metrics_dict_batchwise(
          model,
          postprocessed,
          batch,
          is_training=True,
          soft_policy_weight_scale=soft_policy_weight_scale,
          main_loss_scale=main_loss_scale,
          intermediate_loss_scale=intermediate_loss_scale,
          intermediate_distill_scale=intermediate_distill_scale,
        )

        # DDP averages loss across instances, so to preserve LR as per-sample lr, we scale by world size.
        loss = metrics["loss_sum"] * world_size
        # Now we have the reduced gradients
        loss.backward()

        if model_config["norm_kind"] == "fixup" or model_config["norm_kind"] == "fixscale" or model_config["norm_kind"] == "fixscaleonenorm":
          gnorm_cap = 2500.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
        elif model_config["norm_kind"] == "bnorm" or model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
          gnorm_cap = 5500.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
        else:
          assert False

        #Loosen gradient clipping as we shift to smaller learning rates
        gnorm_cap = gnorm_cap / math.sqrt(max(0.0000001,lr_scale))

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), gnorm_cap).detach().cpu().item()
        metrics["gnorm_batch"] = gnorm
        exgnorm = max(0.0, gnorm - gnorm_cap)
        metrics["exgnorm_sum"] = exgnorm * batch_size

        metrics["pslr_batch"] = lr_right_now
        metrics["wdnormal_batch"] = normal_weight_decay_right_now

        optimizer.step()
        batch_count_this_epoch += 1
        train_state["train_steps_since_last_reload"] += batch_size
        train_state["global_step_samples"] += batch_size

        metrics = detensorify_metrics(metrics)
        accumulate_metrics(running_metrics["sums"], running_metrics["weights"], metrics, batch_size, decay=0.999)
        if batch_count_this_epoch % print_train_loss_every_batches == 0:

          if model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
            metrics["brn_rmax"] = train_state["brenorm_rmax"]
            metrics["brn_dmax"] = train_state["brenorm_dmax"]
            metrics["brn_mmnt"] = brenorm_avg_momentum
            upper_rclippage = []
            lower_rclippage = []
            dclippage = []
            model.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
            metrics["brn_ruclip"] = sum(upper_rclippage) / max(len(upper_rclippage),1.0)
            metrics["brn_rlclip"] = sum(lower_rclippage) / max(len(lower_rclippage),1.0)
            metrics["brn_dclip"] = sum(dclippage) / max(len(dclippage),1.0)

          t1 = time.perf_counter()
          timediff = t1 - last_train_stats_time
          last_train_stats_time = t1
          metrics["time_since_last_print"] = timediff
          log_metrics(running_metrics["sums"], running_metrics["weights"], metrics, train_metrics_out)

        # Update LR more frequently at the start for smoother warmup ramp and wd adjustment
        if train_state["global_step_samples"] <= 50000000 and batch_count_this_epoch % 10 == 0:
          lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()

        # Update batch renorm parameters
        if batch_count_this_epoch % 500 == 0:
          maybe_update_brenorm_params()

      logging.info("Finished training subepoch!")

      if swa_model is not None and swa_sub_epoch_scale is not None:
        swa_model.update_parameters(model)

    # END SUB EPOCH LOOP ------------

    save_history()
    save(model, swa_model, optimizer, metrics_obj, running_metrics, train_state)

    num_epochs_this_instance += 1

    if rank == 0:
      train_state["export_cycle_counter"] += 1
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

      if not no_export and is_time_to_export and not skip_export_this_time:
        # Export a model for testing, unless somehow it already exists
        modelname = "%s-s%d-d%d" % (
          exportprefix,
          train_state["global_step_samples"],
          last_datainfo_row,
        )
        savepath = os.path.join(exportdir,modelname)
        savepathtmp = os.path.join(exportdir,modelname+".tmp")
        if os.path.exists(savepath):
          logging.info("NOT saving model, already exists at: " + savepath)
        else:
          os.mkdir(savepathtmp)
          logging.info("SAVING MODEL FOR EXPORT TO: " + savepath)

          save(model, swa_model, optimizer, metrics_obj, running_metrics, train_state, path=os.path.join(savepathtmp,"model.ckpt"))
          dump_and_flush_json(trainhistory,os.path.join(savepathtmp,"trainhistory.json"))
          with open(os.path.join(savepathtmp,"model.config.json"),"w") as f:
            json.dump(model_config,f)
          with open(os.path.join(savepathtmp,"saved_model","model.config.json"),"w") as f:
            json.dump(model_config,f)
          with open(os.path.join(savepathtmp,"non_swa_saved_model","model.config.json"),"w") as f:
            json.dump(model_config,f)

          time.sleep(2)
          os.rename(savepathtmp,savepath)

    # Validate
    if rank == 0:
      logging.info("Beginning validation after epoch!")
      val_files = []
      if os.path.exists(vdatadir):
        val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir) if fname.endswith(".npz")]
      val_files = sorted(val_files)
      if len(val_files) == 0:
        logging.info("No validation files, skipping validation step")
      else:
        with torch.no_grad():
          model.eval()
          val_metric_sums = defaultdict(float)
          val_metric_weights = defaultdict(float)
          val_samples = 0
          t0 = time.perf_counter()
          for batch in data_processing_pytorch.read_npz_training_data(val_files, batch_size, pos_len, device, randomize_symmetries=True, model_config=model_config):
            model_outputs = model(batch["binaryInputNCHW"],batch["globalInputNC"])
            postprocessed = model.postprocess_output(model_outputs)
            metrics = metrics_obj.metrics_dict_batchwise(
              model,
              postprocessed,
              batch,
              is_training=False,
              soft_policy_weight_scale=soft_policy_weight_scale,
              main_loss_scale=main_loss_scale,
              intermediate_loss_scale=intermediate_loss_scale,
              intermediate_distill_scale=intermediate_distill_scale,
            )
            metrics = detensorify_metrics(metrics)
            accumulate_metrics(val_metric_sums, val_metric_weights, metrics, batch_size, decay=1.0)
            val_samples += batch_size
            if max_val_samples is not None and val_samples > max_val_samples:
              break
            val_metric_sums["nsamp_train"] = running_metrics["sums"]["nsamp"]
            val_metric_weights["nsamp_train"] = running_metrics["weights"]["nsamp"]
            val_metric_sums["wsum_train"] = running_metrics["sums"]["wsum"]
            val_metric_weights["wsum_train"] = running_metrics["weights"]["wsum"]
          log_metrics(val_metric_sums, val_metric_weights, metrics, val_metrics_out)
          t1 = time.perf_counter()
          logging.info(f"Validation took {t1-t0} seconds")
          model.train()

    if max_epochs_this_instance is not None and max_epochs_this_instance >= 0 and num_epochs_this_instance >= max_epochs_this_instance:
      logging.info("Hit max epochs this instance, done")
      break

    if sleep_seconds_per_epoch is None:
      time.sleep(1)
    else:
      time.sleep(sleep_seconds_per_epoch)

    if rank == 0:
      now = datetime.datetime.now()
      if now - last_longterm_checkpoint_save_time >= datetime.timedelta(hours=12):
        last_longterm_checkpoint_save_time = now
        dated_name = datetime.datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        save(model, swa_model, optimizer, metrics_obj, running_metrics, train_state, path=os.path.join(longterm_checkpoints_dir,f"{dated_name}.ckpt"))

  close(train_metrics_out)
  close(val_metrics_out)


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
  if num_gpus_used > 1:
    torch.multiprocessing.set_start_method("spawn")
    assert False, "still need to write gradient scaling code, batch splitting, bucket logic, other multiproc handling"
    torch.multiprocessing.spawn(
      main,
      nprocs=num_gpus_used,
      args=(world_size, args, multi_gpu_device_ids)
    )
  else:
    rank = 0
    world_size = 1
    main(rank, world_size, args, multi_gpu_device_ids)
