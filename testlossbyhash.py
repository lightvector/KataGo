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
from model import Model

#Command and args-------------------------------------------------------------------

description = """
Compute average loss by sgf hash.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-gamesh5', help='H5 file of preprocessed game data', required=True)
parser.add_argument('-model-file', help='model file prefix to load', required=True)
parser.add_argument('-rank-idx', help='rank to provide to model for inference', required=True)
parser.add_argument('-only-ranks', help='restrict only to games with these rank indices', required=False, action="append")
parser.add_argument('-require-last-move', help='filter down to only instances where last move is provided', required=False, action="store_true")
parser.add_argument('-use-training-set', help='run on training set instead of test set', required=False, action="store_true")
args = vars(parser.parse_args())

gamesh5 = args["gamesh5"]
model_file = args["model_file"]
rank_idx = int(args["rank_idx"])
only_ranks = [int(x) for x in args["only_ranks"]] if args["only_ranks"] is not None else None
require_last_move = args["require_last_move"]
use_training_set = args["use_training_set"]

def log(s):
  print(s,flush=True)


# Model ----------------------------------------------------------------
print("Building model", flush=True)
with open(model_file + ".config.json") as f:
  model_config = json.load(f)
model = Model(model_config)

policy_output = model.policy_output

#Loss function
targets = tf.placeholder(tf.float32, [None] + model.target_shape)
target_weights = tf.placeholder(tf.float32, [None] + model.target_weights_shape)

target_weights_to_use = target_weights

#Require that we have the last move
if require_last_move:
  target_weights_to_use = target_weights_to_use * tf.reduce_sum(model.inputs[:,:,18],axis=[1])

data_loss = target_weights_to_use * tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=policy_output)
weight_output = target_weights_to_use

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
  log("Testing on " + str(num_h5_val_rows) + " rows")
  log("h5_chunk_size = " + str(h5_chunk_size))

  sys.stdout.flush()
  sys.stderr.flush()

  input_start = 0
  input_len = model.input_shape[0] * model.input_shape[1]
  target_start = input_start + input_len
  target_len = model.target_shape[0]
  target_weights_start = target_start + target_len
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
  next_moves_len = 7
  sgfhash_start = next_moves_start + next_moves_len
  sgfhash_len = 8

  def run(fetches, rows):
    assert(len(model.input_shape) == 2)
    assert(len(model.target_shape) == 1)
    assert(len(model.target_weights_shape) == 0)
    input_len = model.input_shape[0] * model.input_shape[1]
    target_len = model.target_shape[0]

    row_inputs = rows[:,0:input_len].reshape([-1] + model.input_shape)
    row_targets = rows[:,target_start:target_start+target_len]
    row_target_weights = rows[:,target_weights_start]

    ranks_input = np.zeros([rank_len])
    ranks_input[rank_idx] = 1.0
    ranks_input = [ranks_input for i in range(len(rows))]

    return session.run(fetches, feed_dict={
      model.inputs: row_inputs,
      model.ranks: ranks_input,
      targets: row_targets,
      target_weights: row_target_weights,
      model.symmetries: [False,False,False],
      model.is_training: False
    })

  def np_array_str(arr,precision):
    return np.array_str(arr, precision=precision, suppress_small = True, max_line_width = 200)


  #Run validation accuracy in batches to avoid out of memory error from processing one supergiant batch
  validation_batch_size = 128
  num_validation_batches = (num_h5_val_rows+validation_batch_size-1)//validation_batch_size
  lossbyhash = {}
  weightbyhash = {}
  for i in range(num_validation_batches):
    print(".",end="",flush=True)
    rows = h5val[i*validation_batch_size : min((i+1)*validation_batch_size, num_h5_val_rows)]
    if not isinstance(rows, np.ndarray):
      rows = np.array(rows)

    (wlosses,weights) = run((data_loss,weight_output), rows)
    row_ranks = rows[:,rank_start:rank_start+rank_len]
    row_hashvalues = rows[:,sgfhash_start:sgfhash_start+sgfhash_len]
    rank_one_hot_idx = np.argmax(row_ranks,axis=1)

    for k in range(len(rows)):
      if only_ranks is None or rank_one_hot_idx[k] in only_ranks:
        sgfhash = row_hashvalues[k]
        sgfhash = (
          int(sgfhash[0]) +
          int(sgfhash[1])*0x10000 +
          int(sgfhash[2])*0x100000000 +
          int(sgfhash[3])*0x1000000000000 +
          int(sgfhash[4])*0x10000000000000000 +
          int(sgfhash[5])*0x100000000000000000000 +
          int(sgfhash[6])*0x1000000000000000000000000 +
          int(sgfhash[7])*0x10000000000000000000000000000
        )
        if sgfhash not in lossbyhash:
          lossbyhash[sgfhash] = 0.0
          weightbyhash[sgfhash] = 0

        lossbyhash[sgfhash] += wlosses[k]
        weightbyhash[sgfhash] += weights[k]

  print("",flush=True)
  sys.stderr.flush()

  print("Writing avg loss by hash",flush=True)
  with open("avglosses.csv","w") as out:
    for sgfhash in lossbyhash:
      out.write(hex(sgfhash))
      out.write(",")
      out.write(str(lossbyhash[sgfhash]/weightbyhash[sgfhash]))
      out.write(",")
      out.write(str(weightbyhash[sgfhash]))
      out.write("\n")

# Finish
h5file.close()
h5fid.close()


print("Done",flush=True)
