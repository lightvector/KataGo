#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import h5py
import contextlib
import json
import tensorflow as tf
import numpy as np

import data
from board import Board
from model import Model, Target_vars, Metrics

#Command and args-------------------------------------------------------------------

description = """
Test neural net on Go positions from an h5 file of preprocessed training positions.
Computes average loss and accuracy the same as in training.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-gamesh5', help='H5 file of preprocessed game data', required=True)
parser.add_argument('-model-file', help='model file prefix to load', required=True)
parser.add_argument('-rank-idx', help='rank to provide to model for inference', required=False)
parser.add_argument('-require-last-move', help='filter down to only instances where last move is provided', required=False, action="store_true")
parser.add_argument('-use-training-set', help='run on training set instead of test set', required=False, action="store_true")
parser.add_argument('-validation-prop', help='only use this proportion of validation set', required=False)
args = vars(parser.parse_args())

gamesh5 = args["gamesh5"]
model_file = args["model_file"]
rank_idx = (int(args["rank_idx"]) if args["rank_idx"] is not None else 0)

require_last_move = args["require_last_move"]
use_training_set = args["use_training_set"]

validation_prop = 1.0
if "validation_prop" in args and args["validation_prop"] is not None:
  validation_prop = float(args["validation_prop"])

def log(s):
  print(s,flush=True)


# Model ----------------------------------------------------------------
print("Building model", flush=True)
with open(model_file + ".config.json") as f:
  model_config = json.load(f)
model = Model(model_config)
target_vars = Target_vars(model,for_optimization=False,require_last_move=require_last_move)
metrics = Metrics(model,target_vars,include_debug_stats=False)

total_parameters = 0
for variable in tf.trainable_variables():
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  total_parameters += variable_parameters
  log("Model variable %s, %d parameters" % (variable.name,variable_parameters))

log("Built model, %d total parameters" % total_parameters)


# Open H5 file---------------------------------------------------------
print("Opening H5 file: " + gamesh5)

h5_propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
h5_settings = list(h5_propfaid.get_cache())
assert(h5_settings[2] == 1048576) #Default h5 cache size is 1 MB
h5_settings[2] *= 128 #Make it 128 MB
print("Adjusting H5 cache settings to: " + str(h5_settings))
h5_propfaid.set_cache(*h5_settings)

h5fid = h5py.h5f.open(str.encode(str(gamesh5)), fapl=h5_propfaid)
h5file = h5py.File(h5fid)
h5train = h5file["train"]
h5val = h5file["val"]
h5_chunk_size = h5train.chunks[0]
num_h5_train_rows = h5train.shape[0]
num_h5_val_rows = h5val.shape[0]

if use_training_set:
  num_h5_val_rows = num_h5_train_rows
  h5val = h5train

# Testing ------------------------------------------------------------

print("Testing", flush=True)

saver = tf.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

#Some tensorflow options
#tfconfig = tf.ConfigProto(log_device_placement=False,device_count={'GPU': 0})
tfconfig = tf.ConfigProto(log_device_placement=False)
#tfconfig.gpu_options.allow_growth = True
#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config=tfconfig) as session:
  saver.restore(session, model_file)

  sys.stdout.flush()
  sys.stderr.flush()

  log("Began session")
  log("Testing on " + str(int(num_h5_val_rows * validation_prop)) + "/" + str(num_h5_val_rows) + " rows")
  log("h5_chunk_size = " + str(h5_chunk_size))

  sys.stdout.flush()
  sys.stderr.flush()

  input_start = 0
  input_len = model.input_shape[0] * model.input_shape[1]
  policy_target_start = input_start + input_len
  policy_target_len = model.policy_target_shape[0]
  value_target_start = policy_target_start + policy_target_len
  value_target_len = 1
  target_weights_start = value_target_start + value_target_len
  target_weights_len = 1
  rank_start = target_weights_start + target_weights_len
  rank_len = model.rank_shape[0]
  side_start = rank_start + rank_len
  side_len = 1
  turn_number_start = side_start + side_len
  turn_number_len = 2
  recent_captures_start = turn_number_start + turn_number_len
  recent_captures_len = model.max_board_size * model.max_board_size
  next_moves_start = recent_captures_start + recent_captures_len
  next_moves_len = 12
  sgfhash_start = next_moves_start + next_moves_len
  sgfhash_len = 8

  def run(fetches, rows):
    assert(len(model.input_shape) == 2)
    assert(len(model.policy_target_shape) == 1)
    assert(len(model.value_target_shape) == 0)
    assert(len(model.target_weights_shape) == 0)
    assert(len(model.rank_shape) == 1)

    if not isinstance(rows, np.ndarray):
      rows = np.array(rows)

    row_inputs = rows[:,0:input_len].reshape([-1] + model.input_shape)
    row_policy_targets = rows[:,policy_target_start:policy_target_start+policy_target_len]
    row_value_target = rows[:,value_target_start]
    row_target_weights = rows[:,target_weights_start]

    ranks_input = np.zeros([rank_len])
    ranks_input[rank_idx] = 1.0
    ranks_input = [ranks_input for i in range(len(rows))]

    return session.run(fetches, feed_dict={
      model.inputs: row_inputs,
      model.ranks: ranks_input,
      target_vars.policy_targets: row_policy_targets,
      target_vars.value_target: row_value_target,
      target_vars.target_weights_from_data: row_target_weights,
      model.symmetries: [False,False,False],
      model.is_training: False
    })

  def np_array_str(arr,precision):
    return np.array_str(arr, precision=precision, suppress_small = True, max_line_width = 200)
  def merge_dicts(dicts,merge_list):
    keys = dicts[0].keys()
    return dict((key,merge_list([d[key] for d in dicts])) for key in keys)

  def run_validation_in_batches(fetches):
    #Run validation accuracy in batches to avoid out of memory error from processing one supergiant batch
    validation_batch_size = 128
    num_validation_batches = int(num_h5_val_rows * validation_prop + validation_batch_size-1)//validation_batch_size
    results = []
    for i in range(num_validation_batches):
      print(".",end="",flush=True)
      rows = h5val[i*validation_batch_size : min((i+1)*validation_batch_size, num_h5_val_rows)]
      result = run(fetches, rows)
      results.append(result)

    print("",flush=True)
    return results

  vmetrics = {
    "acc1": metrics.accuracy1,
    "acc4": metrics.accuracy4,
    "ploss": target_vars.policy_loss,
    "vloss": target_vars.value_loss,
    "vconf": metrics.valueconf,
    "wsum": target_vars.weight_sum,
  }

  def validation_stats_str(vmetrics_evaled):
    return "vacc1 %5.2f%% vacc4 %5.2f%% vploss %f vvloss %f vconf %f" % (
      vmetrics_evaled["acc1"] * 100 / vmetrics_evaled["wsum"],
      vmetrics_evaled["acc4"] * 100 / vmetrics_evaled["wsum"],
      vmetrics_evaled["ploss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["vloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["vconf"] / vmetrics_evaled["wsum"],
  )

  vmetrics_evaled = merge_dicts(run_validation_in_batches(vmetrics), np.sum)
  vstr = validation_stats_str(vmetrics_evaled)

  log(vstr)

  sys.stdout.flush()
  sys.stderr.flush()

# Finish
h5file.close()
h5fid.close()


