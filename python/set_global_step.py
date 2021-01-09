#!/usr/bin/python3
import sys
import os
import argparse
import tensorflow as tf
import numpy as np

description = """
Hacky script to set the value of global_step in a tensorflow checkpoint
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path without the .ckpt or the .meta', required=True)
parser.add_argument('-new-value', help='New value to set to', type=int, required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
new_value = args["new_value"]
output_path = args["output"]

with tf.compat.v1.Session() as sess:
  for var_name, shape in tf.train.list_variables(checkpoint_path):
    var = tf.train.load_variable(checkpoint_path, var_name)
    if var_name == "global_step":
      var = tf.Variable(new_value, trainable=False, name=var_name, dtype=tf.int64)
    else:
      var = tf.Variable(var,name=var_name)

  saver = tf.compat.v1.train.Saver()
  sess.run(tf.compat.v1.global_variables_initializer())
  saver.save(sess, output_path)
