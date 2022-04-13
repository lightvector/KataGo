#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn
from torch.optim.swa_utils import AveragedModel

import modelconfigs
from model_pytorch import Model
from metrics_pytorch import Metrics
import data_processing_pytorch

# HANDLE COMMAND AND ARGS -------------------------------------------------------------------

if __name__ == "__main__":

  description = """
  Test neural net on Go positions from npz files of batches from selfplay.
  """

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-npzdir', help='Directory with npz data', required=True)
  parser.add_argument('-model-kind', help='If specified, use this model kind instead of config', required=False)
  parser.add_argument('-config', help='Path to model.config.json', required=False)
  parser.add_argument('-checkpoint', help='Checkpoint to test', required=False)
  parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
  parser.add_argument('-batch-size', help='Batch size to use for testing', type=int, required=True)
  parser.add_argument('-use-swa', help='Use SWA model', action="store_true", required=False)
  parser.add_argument('-max-batches', help='Maximum number of batches for testing', type=int, required=False)

  args = vars(parser.parse_args())

def main(args):
  npzdir = args["npzdir"]
  model_kind = args["model_kind"]
  config_file = args["config"]
  checkpoint_file = args["checkpoint"]
  pos_len = args["pos_len"]
  batch_size = args["batch_size"]
  use_swa = args["use_swa"]
  max_batches = args["max_batches"]

  soft_policy_weight_scale = 1.0

  # SET UP LOGGING -------------------------------------------------------------

  logging.root.handlers = []
  logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
      logging.StreamHandler()
    ],
  )
  np.set_printoptions(linewidth=150)

  logging.info(str(sys.argv))

  # FIGURE OUT GPU ------------------------------------------------------------
  if True or torch.cuda.is_available():
    logging.info("Using GPU device: " + torch.cuda.get_device_name())
    device = torch.device("cuda")
  else:
    logging.warning("WARNING: No GPU, using CPU")
    device = torch.device("cpu")

  # LOAD MODEL ---------------------------------------------------------------------
  assert (model_kind is None) != (config_file is None), "Must provide exactly one of -model-kind and -config"

  if model_kind is not None:
    model_config = modelconfigs.config_of_name[model_kind]
  else:
    with open(config_file,"r") as f:
      model_config = json.load(f)
  logging.info(str(model_config))

  state_dict = None
  if checkpoint_file is None:
    logging.info("Initializing new model since no checkpoint provided")
    model = Model(model_config,pos_len)
    model.initialize()
  else:
    state_dict = torch.load(checkpoint_file)
    model = Model(model_config,pos_len)
    # Strip off any "module." from when the model was saved with DDP or other things
    model_state_dict = {}
    for key in state_dict["model"]:
      old_key = key
      while key.startswith("module."):
        key = key[:7]
      model_state_dict[key] = state_dict["model"][old_key]
    model.load_state_dict(model_state_dict)

  model.to(device)

  swa_model = None
  if use_swa:
    if state_dict is None:
      raise Exception("Cannot use swa without a trained model")
    if "swa_model" not in state_dict:
      raise Exception("Checkpoint doesn't contain swa_model")
    swa_model = AveragedModel(model, device=device)
    swa_model.load_state_dict(state_dict["swa_model"])

  metrics_obj = Metrics(batch_size,model)
  if state_dict is not None and "metrics" in state_dict:
    metrics_obj.load_state_dict(state_dict["metrics"])
  else:
    logging.info("WARNING: Metrics not found in state dict, using fresh metrics")

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

  def log_metrics(prefix, metric_sums, metric_weights, metrics, metrics_out):
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

    logging.info(prefix + ", ".join(["%s = %f" % (metric, metrics_to_print[metric]) for metric in metrics_to_print]))
    if metrics_out:
      metrics_out.write(json.dumps(metrics_to_print) + "\n")
      metrics_out.flush()

  torch.backends.cudnn.benchmark = True

  # Validate
  logging.info("Beginning test!")
  val_files = [os.path.join(npzdir,fname) for fname in os.listdir(npzdir) if fname.endswith(".npz")]
  if len(val_files) == 0:
    raise Exception("No npz files in " + npzdir)

  with torch.no_grad():
    model.eval()
    if swa_model is not None:
      swa_model.eval()
    val_metric_sums = defaultdict(float)
    val_metric_weights = defaultdict(float)
    num_batches_tested = 0
    num_samples_tested = 0
    total_inference_time = 0.0
    is_first_batch = True
    for batch in data_processing_pytorch.read_npz_training_data(val_files, batch_size, pos_len, device, randomize_symmetries=True, model_config=model_config):
      if num_batches_tested >= max_batches:
        break

      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)

      start.record()
      if swa_model is not None:
        model_outputs = swa_model(batch["binaryInputNCHW"],batch["globalInputNC"])
      else:
        model_outputs = model(batch["binaryInputNCHW"],batch["globalInputNC"])
      end.record()
      torch.cuda.synchronize()
      time_elapsed = start.elapsed_time(end) / 1000.0

      postprocessed = model.postprocess_output(model_outputs)
      metrics = metrics_obj.metrics_dict_batchwise(
        model,
        postprocessed,
        batch,
        is_training=False,
        soft_policy_weight_scale=soft_policy_weight_scale,
        main_loss_scale=1.0,
        intermediate_loss_scale=None,
        intermediate_distill_scale=None,
      )
      metrics = detensorify_metrics(metrics)

      # Ignore first batch, treat as a warmup so timings are a bit more accurate.
      if is_first_batch:
        is_first_batch = False
        continue

      accumulate_metrics(val_metric_sums, val_metric_weights, metrics, batch_size, decay=1.0)

      num_batches_tested += 1
      num_samples_tested += batch_size
      total_inference_time += time_elapsed

      if num_batches_tested % 5 == 0:
        remainder = num_batches_tested // 5
        while remainder > 1 and remainder % 2 == 0:
          remainder = remainder // 2
        if remainder in [1,3,5,7,9]:
          metrics["num_batches"] = num_batches_tested
          metrics["num_samples"] = num_samples_tested
          metrics["inferencetime"] = total_inference_time
          metrics["time/1ksamp"] = total_inference_time / num_samples_tested * 1000.0
          log_metrics("STATS SO FAR: ", val_metric_sums, val_metric_weights, metrics, None)

    metrics["num_batches"] = num_batches_tested
    metrics["num_samples"] = num_samples_tested
    metrics["inferencetime"] = total_inference_time
    metrics["time/1ksamp"] = total_inference_time / num_samples_tested * 1000.0
    log_metrics("FINAL: ", val_metric_sums, val_metric_weights, metrics, None)


if __name__ == "__main__":
  main(args)
