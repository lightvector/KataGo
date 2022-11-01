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
parser.add_argument('-model-name', help='name to record in model file', required=True)
parser.add_argument('-filename-prefix', help='filename prefix to save to within dir', required=True)
parser.add_argument('-for-cuda', help='dump model file for cuda backend', action='store_true', required=False)
args = vars(parser.parse_args())

model_file = args["model_file"]
export_dir = args["export_dir"]
model_name = args["model_name"]
filename_prefix = args["filename_prefix"]
for_cuda = args["for_cuda"]

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
for variable in tf.compat.v1.trainable_variables():
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  total_parameters += variable_parameters
  log("Model variable %s, %d parameters" % (variable.name,variable_parameters))

log("Built model, %d total parameters" % total_parameters)

# Testing ------------------------------------------------------------

print("Testing", flush=True)

saver = tf.compat.v1.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

#Some tensorflow options
#tfconfig = tf.ConfigProto(log_device_placement=False,device_count={'GPU': 0})
tfconfig = tf.compat.v1.ConfigProto(log_device_placement=False)
#tfconfig.gpu_options.allow_growth = True
#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.compat.v1.Session(config=tfconfig) as session:
  saver.restore(session, model_file)

  sys.stdout.flush()
  sys.stderr.flush()

  log("Began session")

  sys.stdout.flush()
  sys.stderr.flush()

  if not for_cuda:
    tf.io.write_graph(session.graph_def,export_dir,filename_prefix + ".graph.pb")
    savepath = export_dir + "/" + filename_prefix
    saver.save(session, savepath + ".weights")
    with open(savepath + ".config.json","w") as f:
      json.dump(model_config,f)

    with open(savepath + ".graph_optimized.pb.modelVersion") as f:
      f.write(model.version)

    log("Exported at: ")
    log(str(datetime.datetime.utcnow()) + " UTC")

    with open(savepath + ".log.txt","w") as f:
      for line in loglines:
        f.write(line + "\n")

  else:
    f = open(export_dir + "/" + filename_prefix + ".txt", "w")
    def writeln(s):
      f.write(str(s)+"\n")

    writeln(model_name)
    writeln(model.version) #version
    writeln(model.max_board_size) #x
    writeln(model.max_board_size) #y
    writeln(model.num_input_features)

    variables = dict((variable.name,variable) for variable in tf.compat.v1.global_variables())
    def get_weights(name):
      return np.array(variables[name+":0"].eval())

    def write_weights(weights):
      if len(weights.shape) == 0:
        f.write(weights)
      elif len(weights.shape) == 1:
        f.write(" ".join(str(weights[x0]) for x0 in range(weights.shape[0])))
      elif len(weights.shape) == 2:
        f.write("\n".join(" ".join(str(weights[x0,x1])
                                   for x1 in range(weights.shape[1]))
                          for x0 in range(weights.shape[0])))
      elif len(weights.shape) == 3:
        f.write("\n".join("   ".join(" ".join(str(weights[x0,x1,x2])
                                              for x2 in range(weights.shape[2]))
                                     for x1 in range(weights.shape[1]))
                          for x0 in range(weights.shape[0])))
      elif len(weights.shape) == 4:
        f.write("\n".join("       ".join("   ".join(" ".join(str(weights[x0,x1,x2,x3])
                                                             for x3 in range(weights.shape[3]))
                                                    for x2 in range(weights.shape[2]))
                                         for x1 in range(weights.shape[1]))
                          for x0 in range(weights.shape[0])))
      else:
        assert(False)
      f.write("\n")

    def write_conv(name,diam,in_channels,out_channels,dilation,weights):
      writeln(name)
      writeln(diam) #y
      writeln(diam) #x
      writeln(in_channels)
      writeln(out_channels)
      writeln(dilation) #y
      writeln(dilation) #x

      assert(len(weights.shape) == 4 and list(weights.shape) == [diam,diam,in_channels,out_channels])
      write_weights(weights)

    def write_bn(name,num_channels):
      writeln(name)
      (nc,epsilon,has_bias,has_scale) = model.batch_norms[name]
      assert(nc == num_channels)

      writeln(num_channels)
      writeln(epsilon)
      writeln(1 if has_scale else 0)
      writeln(1 if has_bias else 0)

      weights = get_weights(name+"/moving_mean")
      assert(len(weights.shape) == 1 and weights.shape[0] == num_channels)
      write_weights(weights)

      weights = get_weights(name+"/moving_variance")
      assert(len(weights.shape) == 1 and weights.shape[0] == num_channels)
      write_weights(weights)

      if has_scale:
        weights = get_weights(name+"/gamma")
        assert(len(weights.shape) == 1 and weights.shape[0] == num_channels)
        write_weights(weights)

      if has_bias:
        weights = get_weights(name+"/beta")
        assert(len(weights.shape) == 1 and weights.shape[0] == num_channels)
        write_weights(weights)

    def write_activation(name):
      writeln(name)

    def write_matmul(name,in_channels,out_channels,weights):
      writeln(name)
      writeln(in_channels)
      writeln(out_channels)
      assert(len(weights.shape) == 2 and weights.shape[0] == in_channels and weights.shape[1] == out_channels)
      write_weights(weights)

    def write_matbias(name,num_channels,weights):
      writeln(name)
      writeln(num_channels)
      assert(len(weights.shape) == 1 and weights.shape[0] == num_channels)
      write_weights(weights)

    def write_initial_conv():
      (name,diam,in_channels,out_channels) = model.initial_conv
      #Fold in the special wcenter weights
      w = get_weights(name+"/w")
      wc = get_weights(name+"/wcenter")
      assert(len(w.shape) == 4)
      assert(len(wc.shape) == 4)
      assert(wc.shape[0] == 1)
      assert(wc.shape[1] == 1)
      wc = np.pad(wc,((w.shape[0]//2,w.shape[0]//2),(w.shape[1]//2,w.shape[1]//2),(0,0),(0,0)),mode="constant")
      assert(wc.shape[0] == w.shape[0])
      assert(wc.shape[1] == w.shape[1])
      write_conv(name,diam,in_channels,out_channels,1,w+wc)

    def write_block(model,block):
      trunk_num_channels = model.trunk_num_channels
      mid_num_channels = model.mid_num_channels
      regular_num_channels = model.regular_num_channels
      dilated_num_channels = model.dilated_num_channels
      gpool_num_channels = model.gpool_num_channels
      writeln(block[0])
      if block[0] == "ordinary_block":
        (kind,name,diam,trunk_num_channels,mid_num_channels) = block
        writeln(name)
        write_bn(name+"/norm1",trunk_num_channels)
        write_activation(name+"/actv1")
        write_conv(name+"/w1",diam,trunk_num_channels,mid_num_channels,1,get_weights(name+"/w1"))
        write_bn(name+"/norm2",mid_num_channels)
        write_activation(name+"/actv2")
        write_conv(name+"/w2",diam,mid_num_channels,trunk_num_channels,1,get_weights(name+"/w2"))

      elif block[0] == "dilated_block":
        (kind,name,diam,trunk_num_channels,regular_num_channels,dilated_num_channels,dilation) = block
        writeln(name)
        write_bn(name+"/norm1",trunk_num_channels)
        write_activation(name+"/actv1")
        write_conv(name+"/w1a",diam,trunk_num_channels,regular_num_channels,1,get_weights(name+"/w1a"))
        write_conv(name+"/w1b",diam,trunk_num_channels,dilated_num_channels,dilation,get_weights(name+"/w1b"))
        write_bn(name+"/norm2",regular_num_channels+dilated_num_channels)
        write_activation(name+"/actv2")
        write_conv(name+"/w2",diam,regular_num_channels+dilated_num_channels,trunk_num_channels,1,get_weights(name+"/w2"))

      elif block[0] == "gpool_block":
        (kind,name,diam,trunk_num_channels,regular_num_channels,gpool_num_channels) = block
        writeln(name)
        write_bn(name+"/norm1",trunk_num_channels)
        write_activation(name+"/actv1")
        write_conv(name+"/w1a",diam,trunk_num_channels,regular_num_channels,1,get_weights(name+"/w1a"))
        write_conv(name+"/w1b",diam,trunk_num_channels,gpool_num_channels,1,get_weights(name+"/w1b"))
        write_bn(name+"/norm1b",gpool_num_channels)
        write_activation(name+"/actv1b")
        write_matmul(name+"/w1r",gpool_num_channels*2,regular_num_channels,get_weights(name+"/w1r"))
        write_bn(name+"/norm2",regular_num_channels)
        write_activation(name+"/actv2")
        write_conv(name+"/w2",diam,regular_num_channels,trunk_num_channels,1,get_weights(name+"/w2"))

      else:
        assert(False)

    def write_trunk():
      writeln("trunk")
      writeln(len(model.blocks))
      writeln(model.trunk_num_channels)
      writeln(model.mid_num_channels)
      writeln(model.regular_num_channels)
      writeln(model.dilated_num_channels)
      writeln(model.gpool_num_channels)

      write_initial_conv()
      for block in model.blocks:
        write_block(model,block)
      write_bn("trunk/norm",model.trunk_num_channels)
      write_activation("trunk/actv")

    def write_model_conv(model_conv):
      (name,diam,in_channels,out_channels) = model_conv
      write_conv(name+"/w",diam,in_channels,out_channels,1,get_weights(name+"/w"))

    def write_policy_head():
      writeln("policyhead")
      write_model_conv(model.p1_conv)
      write_model_conv(model.g1_conv)
      write_bn("g1/norm",model.g1_num_channels)
      write_activation("g1/actv")
      write_matmul("matmulg2w",model.g2_num_channels,model.p1_num_channels,get_weights("matmulg2w"))
      write_bn("p1/norm",model.p1_num_channels)
      write_activation("p1/actv")
      write_model_conv(model.p2_conv)
      write_matmul("matmulpass",model.g2_num_channels,1,get_weights("matmulpass"))

    def write_value_head():
      writeln("valuehead")
      write_model_conv(model.v1_conv)
      write_bn("v1/norm",model.v1_num_channels)
      write_activation("v1/actv")
      write_matmul("v2/w",model.v1_num_channels,model.v2_size,get_weights("v2/w"))
      write_matbias("v2/b",model.v2_size,get_weights("v2/b"))
      write_activation("v2/actv")
      write_matmul("v3/w",model.v2_size,model.v3_size,get_weights("v3/w"))
      write_matbias("v3/b",model.v3_size,get_weights("v3/b"))

    write_trunk()
    write_policy_head()
    write_value_head()
    f.close()

    log("Exported at: ")
    log(str(datetime.datetime.utcnow()) + " UTC")

    with open(export_dir + "/log.txt","w") as f:
      for line in loglines:
        f.write(line + "\n")

  sys.stdout.flush()
  sys.stderr.flush()


