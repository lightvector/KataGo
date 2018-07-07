#!/usr/bin/python3
import sys
import os
import argparse
import random
import time
import logging
import json
import datetime

import tensorflow as tf
import numpy as np

from model import Model

#Command and args-------------------------------------------------------------------

description = """
Export neural net weights and graph to file.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-model-file', help='model file prefix to load', required=True)
parser.add_argument('-export-dir', help='model file dir to save to', required=True)
parser.add_argument('-filename-prefix', help='filename prefix to save to within dir', required=True)
args = vars(parser.parse_args())

model_file = args["model_file"]
export_dir = args["export_dir"]
filename_prefix = args["filename_prefix"]

loglines = []
def log(s):
  loglines.append(s)
  print(s,flush=True)

log("model_file" + ": " + model_file)
log("export_dir" + ": " + export_dir)
log("filename_prefix" + ": " + filename_prefix)

# Model ----------------------------------------------------------------
print("Building model", flush=True)
with open(model_file + ".config.json") as f:
  model_config = json.load(f)
model = Model(model_config)

total_parameters = 0
for variable in tf.trainable_variables():
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  total_parameters += variable_parameters
  log("Model variable %s, %d parameters" % (variable.name,variable_parameters))

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
#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config=tfconfig) as session:
  saver.restore(session, model_file)

  sys.stdout.flush()
  sys.stderr.flush()

  log("Began session")

  sys.stdout.flush()
  sys.stderr.flush()

  tf.train.write_graph(session.graph_def,export_dir,filename_prefix + ".graph.pb")
  savepath = export_dir + "/" + filename_prefix
  saver.save(session, savepath + ".weights")
  with open(savepath + ".config.json","w") as f:
    json.dump(model_config,f)

  log("Exported at: ")
  log(str(datetime.datetime.utcnow()) + " UTC")

  with open(savepath + ".log.txt","w") as f:
    for line in loglines:
      f.write(line + "\n")

  sys.stdout.flush()
  sys.stderr.flush()


