#!/usr/bin/python3
import sys
import os
import argparse
import random
import time
import logging
import json
import datetime
import struct

import tensorflow as tf
import numpy as np

from model import Model, ModelUtils
import common

tf.compat.v1.disable_eager_execution()
#Command and args-------------------------------------------------------------------

description = """
Export neural net weights and graph to file.
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
parser.add_argument('-export-dir', help='model file dir to save to', required=True)
parser.add_argument('-model-name', help='name to record in model file', required=True)
parser.add_argument('-filename-prefix', help='filename prefix to save to within dir', required=True)
parser.add_argument('-txt', help='write floats as text instead of binary', action='store_true', required=False)
parser.add_argument('-for-cuda', help='dump model file for cuda backend', action='store_true', required=False)
args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
name_scope = args["name_scope"]
export_dir = args["export_dir"]
model_name = args["model_name"]
filename_prefix = args["filename_prefix"]
binary_floats = (not args["txt"])
for_cuda = args["for_cuda"]

loglines = []
def log(s):
  loglines.append(s)
  print(s,flush=True)

log("model_variables_prefix" + ": " + str(model_variables_prefix))
log("model_config_json" + ": " + str(model_config_json))
log("name_scope" + ": " + str(name_scope))
log("export_dir" + ": " + export_dir)
log("filename_prefix" + ": " + filename_prefix)

# Model ----------------------------------------------------------------
print("Building model", flush=True)
with open(model_config_json) as f:
  model_config = json.load(f)

pos_len = 19 # shouldn't matter, all we're doing is exporting weights that don't depend on this
if name_scope is not None:
  with tf.compat.v1.variable_scope(name_scope):
    model = Model(model_config,pos_len,{})
else:
  model = Model(model_config,pos_len,{})
ModelUtils.print_trainable_variables(log)

# Testing ------------------------------------------------------------

print("Testing", flush=True)

saver = tf.compat.v1.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

#Some tensorflow options
#tfconfig = tf.compat.v1.ConfigProto(log_device_placement=False,device_count={'GPU': 0})
tfconfig = tf.compat.v1.ConfigProto(log_device_placement=False)
#tfconfig.gpu_options.allow_growth = True
#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.compat.v1.Session(config=tfconfig) as session:
  saver.restore(session, model_variables_prefix)

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
    extension = (".bin" if binary_floats else ".txt")
    mode = ("wb" if binary_floats else "w")
    f = open(export_dir + "/" + filename_prefix + extension, mode)
    def writeln(s):
      if binary_floats:
        f.write((str(s)+"\n").encode(encoding="ascii",errors="backslashreplace"))
      else:
        f.write(str(s)+"\n")
    def writestr(s):
      if binary_floats:
        f.write(s.encode(encoding="ascii",errors="backslashreplace"))
      else:
        f.write(s)

    writeln(model_name)
    writeln(model.version) #version
    writeln(model.get_num_bin_input_features(model_config))
    writeln(model.get_num_global_input_features(model_config))

    variables = dict((variable.name,variable) for variable in tf.compat.v1.global_variables())
    def get_weights(name):
      if name_scope is not None:
        return np.array(variables[name_scope+"/"+name+":0"].eval())
      else:
        return np.array(variables[name+":0"].eval())

    def write_weights(weights):
      if binary_floats:
        # Little endian
        reshaped = np.reshape(weights,[-1])
        num_weights = len(reshaped)
        writestr("@BIN@")
        f.write(struct.pack(f'<{num_weights}f',*reshaped))
        writestr("\n")
      else:
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

        writestr("\n")

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
      (nc,epsilon,has_bias,has_scale,use_fixup) = model.batch_norms[name]
      assert(nc == num_channels)

      writeln(num_channels)
      writeln(epsilon)
      writeln(1 if has_scale else 0)
      writeln(1 if has_bias else 0)

      if use_fixup:
        weights = np.zeros([num_channels])
      else:
        weights = get_weights(name+"/moving_mean")
      assert(len(weights.shape) == 1 and weights.shape[0] == num_channels)
      write_weights(weights)

      if use_fixup:
        weights = np.ones([num_channels])
      else:
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
      w = get_weights(name+"/w")
      assert(len(w.shape) == 4)
      write_conv(name,diam,in_channels,out_channels,1,w)

    def write_initial_matmul():
     (name,in_channels,out_channels) = model.initial_matmul
     w = get_weights(name)
     assert(len(w.shape) == 2)
     write_matmul(name,in_channels,out_channels,w)

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
        write_matmul(name+"/w1r",gpool_num_channels*3,regular_num_channels,get_weights(name+"/w1r"))
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
      write_initial_matmul()
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

      #Write only the this-move prediction, not the next-move prediction
      (p2name,p2diam,p2in_channels,p2out_channels) = model.p2_conv
      assert(p2out_channels == 2)
      write_conv(p2name+"/w",p2diam,p2in_channels,1,1,get_weights(p2name+"/w")[:,:,:,0:1])
      write_matmul("matmulpass",model.g2_num_channels,1,get_weights("matmulpass")[:,0:1])

    def write_value_head():
      writeln("valuehead")
      write_model_conv(model.v1_conv)
      write_bn("v1/norm",model.v1_num_channels)
      write_activation("v1/actv")
      write_matmul("v2/w",model.v1_num_channels*3,model.v2_size,get_weights("v2/w"))
      write_matbias("v2/b",model.v2_size,get_weights("v2/b"))
      write_activation("v2/actv")
      write_matmul("v3/w",model.v2_size,model.v3_size,get_weights("v3/w"))

      if model.support_japanese_rules:
        write_matbias("v3/b",model.v3_size,get_weights("v3/b"))
      else:
        w = get_weights("v3/b")
        assert(len(w.shape) == 1 and w.shape[0] == 3)
        w[2] = w[2] - 5000.0
        write_matbias("v3/b",model.v3_size,w)

      #For now, only output the scoremean and scorestdev and lead and vtime channels
      w = get_weights("mv3/w")[:,0:4]
      b = get_weights("mv3/b")[0:4]
      if model.use_scoremean_as_lead:
        w[:,2] = w[:,0]
        b[2] = b[0]

      if model.version < 9:
        write_matmul("sv3/w",model.v2_size,4,w)
        write_matbias("sv3/b",4,b)
      else:
        #Grab the shortterm channels
        w = np.concatenate([w, get_weights("mmv3/w")[:,0:2]],axis=1)
        b = np.concatenate([b, get_weights("mmv3/b")[0:2]],axis=0)
        write_matmul("sv3/w",model.v2_size,6,w)
        write_matbias("sv3/b",6,b)

      write_model_conv(model.vownership_conv)

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
