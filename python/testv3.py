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
import modelv3
from modelv3 import ModelV3, Target_varsV3, MetricsV3

#Command and args-------------------------------------------------------------------

description = """
Test neural net on Go positions from an h5 file of preprocessed training positions.
Computes average loss and accuracy the same as in training.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-data-file', help='tfrecords or npz file', required=True)
parser.add_argument('-checkpoint-file-prefix', help='model checkpoint file prefix to load', required=False)
parser.add_argument('-saved-model-dir', help='tf SavedModel dir to load', required=False)
parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
parser.add_argument('-batch-size', help='Expected batch size of the input data, must match tfrecords', type=int, required=True)
args = vars(parser.parse_args())

data_file = args["data_file"]
checkpoint_file_prefix = args["checkpoint_file_prefix"]
saved_model_dir = args["saved_model_dir"]
pos_len = args["pos_len"]
batch_size = args["batch_size"]

if checkpoint_file_prefix is None and saved_model_dir is None:
  raise Exception("Must specify one of -checkpoint-file-prefix or -saved-model-dir")
if checkpoint_file_prefix is not None and saved_model_dir is not None:
  raise Exception("Must specify only one of -checkpoint-file-prefix or -saved-model-dir")

def log(s):
  print(s,flush=True)

NUM_POLICY_TARGETS = 1
NUM_GLOBAL_TARGETS = 54
NUM_VALUE_SPATIAL_TARGETS = 1
EXTRA_SCORE_DISTR_RADIUS = 60
BONUS_SCORE_RADIUS = 30

log("Constructing validation input pipe")
def parse_tf_records_input(serialized_example):
  example = tf.parse_single_example(serialized_example,raw_input_features)
  binchwp = tf.decode_raw(example["binchwp"],tf.uint8)
  ginc = example["ginc"]
  ptncm = example["ptncm"]
  gtnc = example["gtnc"]
  sdn = example["sdn"]
  sbsn = example["sbsn"]
  vtnchw = example["vtnchw"]
  return {
    "binchwp": tf.reshape(binchwp,[batch_size,ModelV3.NUM_BIN_INPUT_FEATURES,(pos_len*pos_len+7)//8]),
    "ginc": tf.reshape(ginc,[batch_size,ModelV3.NUM_GLOBAL_INPUT_FEATURES]),
    "ptncm": tf.reshape(ptncm,[batch_size,NUM_POLICY_TARGETS,pos_len*pos_len+1]),
    "gtnc": tf.reshape(gtnc,[batch_size,NUM_GLOBAL_TARGETS]),
    "sdn": tf.reshape(sdn,[batch_size,pos_len*pos_len*2+EXTRA_SCORE_DISTR_RADIUS*2]),
    "sbsn": tf.reshape(sbsn,[batch_size,BONUS_SCORE_RADIUS*2+1]),
    "vtnchw": tf.reshape(vtnchw,[batch_size,NUM_VALUE_SPATIAL_TARGETS,pos_len,pos_len])
  }

if data_file.endswith(".tfrecord"):
  dataset = tf.data.Dataset.from_tensor_slices([data_file])
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.map(parse_tf_records_input)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
elif data_file.endswith(".npz"):
  features = {
    "binchwp": tf.placeholder(tf.uint8,[batch_size,ModelV3.NUM_BIN_INPUT_FEATURES,(pos_len*pos_len+7)//8]),
    "ginc": tf.placeholder(tf.float32,[batch_size,ModelV3.NUM_GLOBAL_INPUT_FEATURES]),
    "ptncm": tf.placeholder(tf.float32,[batch_size,NUM_POLICY_TARGETS,pos_len*pos_len+1]),
    "gtnc": tf.placeholder(tf.float32,[batch_size,NUM_GLOBAL_TARGETS]),
    "sdn": tf.placeholder(tf.float32,[batch_size,pos_len*pos_len*2+EXTRA_SCORE_DISTR_RADIUS*2]),
    "sbsn": tf.placeholder(tf.float32,[batch_size,BONUS_SCORE_RADIUS*2+1]),
    "vtnchw": tf.placeholder(tf.float32,[batch_size,NUM_VALUE_SPATIAL_TARGETS,pos_len,pos_len])
  }
else:
  raise Exception("Data file must be .tfrecord or .npz")


# Model ----------------------------------------------------------------
if checkpoint_file_prefix is not None:
  with open(os.path.join(os.path.dirname(checkpoint_file_prefix),"model.config.json")) as f:
    model_config = json.load(f)
elif saved_model_dir is not None:
  with open(os.path.join(saved_model_dir,"model.config.json")) as f:
    model_config = json.load(f)
else:
  assert(False)

mode = tf.estimator.ModeKeys.EVAL
print_model = False
num_batches_per_epoch = 1 #doesn't matter
(model,target_vars,metrics) = modelv3.build_model_from_tfrecords_features(features,mode,print_model,log,model_config,pos_len,num_batches_per_epoch)

total_parameters = 0
for variable in tf.trainable_variables():
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  total_parameters += variable_parameters

log("Built model, %d total parameters" % total_parameters)


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
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.2
with tf.Session(config=tfconfig) as session:
  if checkpoint_file_prefix is not None:
    saver.restore(session, checkpoint_file_prefix)
  elif saved_model_dir is not None:
    saver.restore(session, os.path.join(saved_model_dir,"saved_model","variables","variables"))
  else:
    assert(False)

  log("Began session")

  sys.stdout.flush()
  sys.stderr.flush()

  def np_array_str(arr,precision):
    return np.array_str(arr, precision=precision, suppress_small = True, max_line_width = 200)
  def merge_dicts(dicts,merge_list):
    keys = dicts[0].keys()
    return dict((key,merge_list([d[key] for d in dicts])) for key in keys)

  def run_validation_in_batches(fetches):
    results = []
    if data_file.endswith(".tfrecord"):
      try:
        while True:
          results = session.run(fetches)
          results.append(result)
      except tf.errors.OutOfRangeError:
        pass
    elif data_file.endswith(".npz"):
      with np.load(data_file) as npz:
        binchwp = npz["binaryInputNCHWPacked"]
        ginc = npz["globalInputNC"]
        ptncm = npz["policyTargetsNCMove"].astype(np.float32)
        gtnc = npz["globalTargetsNC"]
        sdn = npz["scoreDistrN"].astype(np.float32)
        sbsn = npz["selfBonusScoreN"].astype(np.float32)
        vtnchw = npz["valueTargetsNCHW"].astype(np.float32)
        nbatches = len(binchwp)//batch_size
        print("Iterating %d batches" % nbatches)
        for i in range(nbatches):
          if i % 50 == 0:
            print(".",end="")
            sys.stdout.flush()
          result = session.run(fetches,feed_dict={
            features["binchwp"]: np.array(binchwp[i*batch_size:(i+1)*batch_size]),
            features["ginc"]: np.array(ginc[i*batch_size:(i+1)*batch_size]),
            features["ptncm"]: np.array(ptncm[i*batch_size:(i+1)*batch_size]),
            features["gtnc"]: np.array(gtnc[i*batch_size:(i+1)*batch_size]),
            features["sdn"]: np.array(sdn[i*batch_size:(i+1)*batch_size]),
            features["sbsn"]: np.array(sbsn[i*batch_size:(i+1)*batch_size]),
            features["vtnchw"]: np.array(vtnchw[i*batch_size:(i+1)*batch_size])
          })
          results.append(result)
        print("")

    return results

  vmetrics = {
    # "acc1": metrics.accuracy1,
    # "acc4": metrics.accuracy4,
    "ploss": target_vars.policy_loss,
    "vloss": target_vars.value_loss,
    "oloss": target_vars.ownership_loss,
    "svloss": target_vars.scorevalue_loss,
    "sbpdfloss": target_vars.scorebelief_pdf_loss,
    "sbcdfloss": target_vars.scorebelief_cdf_loss,
    "bbpdfloss": target_vars.bonusbelief_pdf_loss,
    "bbcdfloss": target_vars.bonusbelief_cdf_loss,
    "uvloss": target_vars.utilityvar_loss,
    "rwlloss": target_vars.winloss_reg_loss,
    "rsvloss": target_vars.scorevalue_reg_loss,
    "roloss": target_vars.ownership_reg_loss,
    "rscloss": target_vars.scale_reg_loss,
    "vconf": metrics.value_conf,
    "ventr": metrics.value_entropy,
    "wsum": target_vars.weight_sum,
  }

  def validation_stats_str(vmetrics_evaled):
    # return "acc1 %5.2f%% acc4 %5.2f%% ploss %f vloss %f oloss %f svloss %f sbloss %f bbloss %f uvloss %f vconf %f ventr %f" % (
    return "ploss %f vloss %f oloss %f svloss %f sbpdfloss %f sbcdfloss %f bbpdfloss %f bbcdfloss %f uvloss %f rwlloss %f rsvloss %f roloss %f rscloss %f vconf %f ventr %f" % (
      # vmetrics_evaled["acc1"] * 100 / vmetrics_evaled["wsum"],
      # vmetrics_evaled["acc4"] * 100 / vmetrics_evaled["wsum"],
      vmetrics_evaled["ploss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["vloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["oloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["svloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["sbpdfloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["sbcdfloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["bbpdfloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["bbcdfloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["uvloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["rwlloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["rsvloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["roloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["rscloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["vconf"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["ventr"] / vmetrics_evaled["wsum"],
  )

  vmetrics_evaled = merge_dicts(run_validation_in_batches(vmetrics), np.sum)
  vstr = validation_stats_str(vmetrics_evaled)

  log(vstr)

  sys.stdout.flush()
  sys.stderr.flush()


