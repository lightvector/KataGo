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
from model import Model
import common

#Command and args-------------------------------------------------------------------

description = """
Examine neural net weights and other stats.
This is mostly a sandbox for random ideas for things that might be cool to visualize.
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
parser.add_argument('-show-all-weight-magnitudes', help='sumsq and meansq and rmse of weights', action="store_true", required=False)
parser.add_argument('-dump', help='weights name -> dump weights', required=False)
args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
name_scope = args["name_scope"]
show_all_weight_magnitudes = args["show_all_weight_magnitudes"]
dump = args["dump"]

def log(s):
  print(s,flush=True)

# Model ----------------------------------------------------------------
print("Building model", flush=True)
with open(model_config_json) as f:
  model_config = json.load(f)

pos_len = 19 # shouldn't matter, all we're doing is exporting weights that don't depend on this
if name_scope is not None:
  with tf.name_scope(name_scope):
    model = Model(model_config,pos_len,{})
else:
  model = Model(model_config,pos_len,{})

def volume(variable):
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  return variable_parameters

total_parameters = 0
for variable in tf.global_variables():
  variable_parameters = volume(variable)
  total_parameters += variable_parameters
  log("Model variable %s, %d parameters" % (variable.name,variable_parameters))

log("Built model, %d total parameters" % total_parameters)


# Testing ------------------------------------------------------------

print("Testing", flush=True)

saver = tf.compat.v1.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

tfconfig = tf.compat.v1.ConfigProto(log_device_placement=False)
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.1
with tf.compat.v1.Session(config=tfconfig) as session:
  saver.restore(session, model_variables_prefix)

  sys.stdout.flush()
  sys.stderr.flush()

  log("Began session")

  sys.stdout.flush()
  sys.stderr.flush()

  def run(fetches):
    return session.run(fetches, feed_dict={})

  if dump is not None:
    variables = dict((variable.name,variable) for variable in tf.compat.v1.global_variables())
    for name in dump.split(","):
      variable = variables[name]
      variable = np.array(variable.eval())
      if len(variable.shape) == 0:
        print(variable)
      elif len(variable.shape) == 1:
        print(" ".join(str(variable[x0]) for x0 in range(variable.shape[0])))
      elif len(variable.shape) == 2:
        print("\n".join(" ".join(str(variable[x0,x1])
                                 for x1 in range(variable.shape[1]))
                        for x0 in range(variable.shape[0])))
      elif len(variable.shape) == 3:
        print("\n".join("\n".join(" ".join(str(variable[x0,x1,x2])
                                           for x2 in range(variable.shape[2]))
                                  for x1 in range(variable.shape[1]))
                        for x0 in range(variable.shape[0])))
      elif len(variable.shape) == 4:
        print("\n".join("\n".join("\n".join(" ".join(str(variable[x0,x1,x2,x3])
                                                     for x3 in range(variable.shape[3]))
                                           for x2 in range(variable.shape[2]))
                                  for x1 in range(variable.shape[1]))
                        for x0 in range(variable.shape[0])))


  if show_all_weight_magnitudes:
    print("name,sumsq,l2regstrength,meansq,rms")
    for variable in tf.trainable_variables():
      values = np.array(variable.eval())
      sq = np.square(values)
      reg = np.sum(sq) if any(v.name == variable.name for v in model.reg_variables) else 0
      print("%s,%f,%f,%f,%f" % (variable.name, np.sum(sq), reg, np.mean(sq), np.sqrt(np.mean(sq))))
