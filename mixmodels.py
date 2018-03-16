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
import tensorflow as tf
import numpy as np

import data
from board import Board

#Command and args-------------------------------------------------------------------

description = """
Examine neural net weights!
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-model-file', help='model file prefix to load', required=True, action='append')
parser.add_argument('-output-file', help='model file prefix to write to', required=True)
args = vars(parser.parse_args())

model_files = args["model_file"]
output_file = args["output_file"]

def log(s):
  print(s,flush=True)


# Model ----------------------------------------------------------------
print("Building model", flush=True)
import model

def volume(variable):
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  return variable_parameters

variables = {}
total_parameters = 0
for variable in tf.global_variables():
  variable_parameters = volume(variable)
  total_parameters += variable_parameters
  variables[variable.name] = variable
  log("Global variable %s, %d parameters" % (variable.name,variable_parameters))



log("Built model, %d total parameters" % total_parameters)

# Testing ------------------------------------------------------------

print("Testing", flush=True)

saver = tf.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

count = 0
accum_weights = {}

tfconfig = tf.ConfigProto(log_device_placement=False)
with tf.Session(config=tfconfig) as session:

  for model_file in model_files:
    saver.restore(session, model_file)

    def run(fetches):
      return session.run(fetches, feed_dict={})

    print("Processing: " + model_file)
    count += 1
    for name in variables:
      weights = np.array(run(variables[name]))
      if name in accum_weights:
        accum_weights[name] = accum_weights[name] + weights
      else:
        accum_weights[name] = weights

print("Normalizing...")
for name in accum_weights:
  accum_weights[name] = accum_weights[name] / count

assign_ops = dict([(name,variables[name].assign(accum_weights[name])) for name in accum_weights])

with tf.Session(config=tfconfig) as session:
  session.run(assign_ops)
  print("Saving to " + output_file)
  saver.save(session, output_file)

print("Done")
