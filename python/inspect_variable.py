#!/usr/bin/python3
import sys
import os
import argparse
import tensorflow as tf
import numpy as np

description = """
Hacky script to inspect checkpoints
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path without the .ckpt or the .meta', required=True)
parser.add_argument('-variable', help='Variable to view the value of', required=False)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
variable_name = args["variable"]

for var_name, shape in tf.train.list_variables(checkpoint_path):
  if variable_name is None:
    print(var_name)
  else:
    if var_name == variable_name:
      print(tf.train.load_variable(checkpoint_path, var_name))
