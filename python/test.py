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
import tensorflow as tf
import numpy as np

import data
from board import Board
from model import Model, Target_vars, Metrics, ModelUtils
import common
import tfrecordio

#Command and args-------------------------------------------------------------------

description = """
Test neural net on Go positions from a file of preprocessed training positions.
Computes average loss and accuracy the same as in training.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-data-files', help='tfrecords or npz file', required=True, nargs='+')
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
parser.add_argument('-batch-size', help='Expected batch size of the input data, must match tfrecords', type=int, required=True)
args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
data_files = args["data_files"]
name_scope = args["name_scope"]
pos_len = args["pos_len"]
batch_size = args["batch_size"]

def log(s):
  print(s,flush=True)

with open(model_config_json) as f:
  model_config = json.load(f)

log("Constructing validation input pipe")

using_tfrecords = False
using_npz = False
for data_file in data_files:
  if data_file.endswith(".tfrecord"):
    using_tfrecords = True
  elif data_file.endswith(".npz"):
    using_npz = True
  else:
    raise Exception("Data files must be .tfrecord or .npz")

if using_tfrecords and using_npz:
  raise Exception("Cannot have both .tfrecord and .npz in the same call to test")

if using_tfrecords:
  dataset = tf.data.Dataset.from_tensor_slices(data_files)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  parse_input = tfrecordio.make_tf_record_parser(model_config,pos_len,batch_size)
  dataset = dataset.map(parse_input)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
elif using_npz:
  features = tfrecordio.make_raw_input_feature_placeholders(model_config,pos_len,batch_size)


# Model ----------------------------------------------------------------

mode = tf.estimator.ModeKeys.EVAL
print_model = False
if name_scope is not None:
  with tf.name_scope(name_scope):
    (model,target_vars,metrics) = ModelUtils.build_model_from_tfrecords_features(features,mode,print_model,log,model_config,pos_len,batch_size)
else:
  (model,target_vars,metrics) = ModelUtils.build_model_from_tfrecords_features(features,mode,print_model,log,model_config,pos_len,batch_size)

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
  saver.restore(session, model_variables_prefix)

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
    if using_tfrecords:
      try:
        while True:
          result = session.run(fetches)
          results.append(result)
      except tf.errors.OutOfRangeError:
        pass
    elif using_npz:
      for data_file in data_files:
        with np.load(data_file) as npz:
          binchwp = npz["binaryInputNCHWPacked"]
          ginc = npz["globalInputNC"]
          ptncm = npz["policyTargetsNCMove"].astype(np.float32)
          gtnc = npz["globalTargetsNC"]
          sdn = npz["scoreDistrN"].astype(np.float32)
          vtnchw = npz["valueTargetsNCHW"].astype(np.float32)
          nbatches = len(binchwp)//batch_size
          print("Iterating %d batches from %s" % (nbatches,data_file))
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
              features["vtnchw"]: np.array(vtnchw[i*batch_size:(i+1)*batch_size])
            })
            results.append(result)
          print("")

    return results

  vmetrics = {
    "acc1": metrics.accuracy1,
    "acc4": metrics.accuracy4,
    "p0loss": target_vars.policy_loss,
    "p1loss": target_vars.policy1_loss,
    "vloss": target_vars.value_loss,
    "tdvloss": target_vars.td_value_loss,
    "smloss": target_vars.scoremean_loss,
    "sbpdfloss": target_vars.scorebelief_pdf_loss,
    "sbcdfloss": target_vars.scorebelief_cdf_loss,
    "oloss": target_vars.ownership_loss,
    "sloss": target_vars.scoring_loss,
    "fploss": target_vars.futurepos_loss,
    "skloss": target_vars.seki_loss,
    "rsmloss": target_vars.scoremean_reg_loss,
    "rsdloss": target_vars.scorestdev_reg_loss,
    "rscloss": target_vars.scale_reg_loss,
    "vconf": metrics.value_conf,
    "ventr": metrics.value_entropy,
    "wsum": target_vars.weight_sum,
  }

  def validation_stats_str(vmetrics_evaled):
    return "acc1 %f acc4 %f p0loss %f p1loss %f vloss %f tdvloss %f smloss %f sbpdfloss %f sbcdfloss %f oloss %f sloss %f fploss %f skloss %f rsmloss %f rsdloss %f rscloss %f vconf %f ventr %f" % (
      vmetrics_evaled["acc1"] * 100 / vmetrics_evaled["wsum"],
      vmetrics_evaled["acc4"] * 100 / vmetrics_evaled["wsum"],
      vmetrics_evaled["p0loss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["p1loss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["vloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["tdvloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["smloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["sbpdfloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["sbcdfloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["oloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["sloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["fploss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["skloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["rsmloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["rsdloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["rscloss"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["vconf"] / vmetrics_evaled["wsum"],
      vmetrics_evaled["ventr"] / vmetrics_evaled["wsum"],
  )

  vmetrics_evaled = merge_dicts(run_validation_in_batches(vmetrics), np.sum)
  vstr = validation_stats_str(vmetrics_evaled)

  log(vstr)

  sys.stdout.flush()
  sys.stderr.flush()
