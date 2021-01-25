#!/usr/bin/python3
import sys
import os
import argparse
import tensorflow as tf
import numpy as np

description = """
Hacky script to sbscale3/w equal to sb3/w for migrating a model to "use_fixed_sbscaling":True.
Run on a model.ckpt-GLOBALSTEPNUMBER or similar checkpoint to produce a new one. Replace the old one with the new one after a backup.
And then manually edit the model.config.json to have use_fixed_sbscaling true.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path without the .ckpt or the .meta or .data or .index', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]

with tf.compat.v1.Session() as sess:
  for var_name, shape in tf.train.list_variables(checkpoint_path):
    if var_name == "sbscale3/w":
      continue
    var = tf.train.load_variable(checkpoint_path, var_name)
    var = tf.Variable(var,name=var_name)
    if var_name == "sb3/w":
      print("Copying sb3/w -> sbscale3/w")
      sbscale3w = tf.Variable(var,name="sbscale3/w")

  saver = tf.compat.v1.train.Saver()
  sess.run(tf.compat.v1.global_variables_initializer())
  saver.save(sess, output_path)
