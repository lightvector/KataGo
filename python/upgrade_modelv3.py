#!/usr/bin/python3
import sys
import os
import argparse
import random
import time
import logging
import json
import datetime
import shutil

import tensorflow as tf
import numpy as np

import modelv3
from modelv3 import ModelV3
import modelconfigs

#Command and args-------------------------------------------------------------------

description = """
Upgrade neural net to larger size.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-src-dir', help='model training/saver/checkpoint dir to load and convert', required=True)
parser.add_argument('-target-dir', help='model checkpoint dir to create and save to', required=True)
args = vars(parser.parse_args())

src_dir = args["src_dir"]
target_dir = args["target_dir"]

loglines = []
def log(s):
  loglines.append(s)
  print(s,flush=True)

log("src_dir" + ": " + str(src_dir))
log("target_dir" + ": " + str(target_dir))

if os.path.exists(target_dir):
  raise Exception("Target dir already exists: " + target_dir)

oldmodelconfig = modelconfigs.b6c96
newmodelconfig = modelconfigs.b10c128
blockmap = modelconfigs.b6_to_b10_map
noise_mag = 0.1

# Model ----------------------------------------------------------------
pos_len = 19 # shouldn't matter, all we're doing is handling weights that don't depend on this

log("SOURCE MODEL================================")
model = ModelV3(oldmodelconfig,pos_len,{})
modelv3.print_trainable_variables(log)

saver = tf.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

tfconfig = tf.ConfigProto(log_device_placement=False)
#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config=tfconfig) as session:
  saver.restore(session, tf.train.latest_checkpoint(src_dir))

  sys.stdout.flush()
  sys.stderr.flush()

  log("Began source session")

  sys.stdout.flush()
  sys.stderr.flush()

  oldweights = dict((variable.name,np.array(variable.eval())) for variable in tf.trainable_variables())

tf.reset_default_graph()

log("Upgrading model weight matrices")
newweights = {}
modelconfigs.upgrade_net(oldweights,newweights,oldmodelconfig,newmodelconfig,blockmap,noise_mag)

log("TARGET MODEL================================")
model = ModelV3(newmodelconfig,pos_len,{})
modelv3.print_trainable_variables(log)

# for variable in tf.global_variables():
#   print(variable.name)

os.mkdir(target_dir)
os.mkdir(os.path.join(target_dir,"initial_weights"))

saver = tf.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

tfconfig = tf.ConfigProto(log_device_placement=False)
#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config=tfconfig) as session:
  sys.stdout.flush()
  sys.stderr.flush()

  log("Began target session")

  sys.stdout.flush()
  sys.stderr.flush()

  log("Initializing vars")
  session.run(tf.global_variables_initializer())

  log("Assigning old model values")
  for variable in tf.trainable_variables():
    update_op = variable.assign(newweights[variable.name])
    session.run(update_op)

  log("Saving weights")
  saver.save(session, os.path.join(target_dir,"initial_weights","model"), write_state=False, write_meta_graph=False)

  with open(os.path.join(target_dir,"initial_weights","model.config.json"),"w") as f:
    json.dump(newmodelconfig,f)

log("Done")
