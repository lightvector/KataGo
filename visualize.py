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
from model import Model

#Command and args-------------------------------------------------------------------

description = """
Examine neural net weights!
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-model-file', help='model file prefix to load', required=True)
parser.add_argument('-conv-norm-by-xy', help='weights name -> average norm by position', required=False)
parser.add_argument('-conv-norm-by-channel', help='weights name -> matrix of average weight norms by channels', required=False)
parser.add_argument('-dump', help='weights name -> dump weights', required=False)
args = vars(parser.parse_args())

model_file = args["model_file"]
conv_norm_by_xy = args["conv_norm_by_xy"]
conv_norm_by_channel = args["conv_norm_by_channel"]
dump = args["dump"]

def log(s):
  print(s,flush=True)


# Model ----------------------------------------------------------------
print("Building model", flush=True)
model = Model(use_ranks=True)

def volume(variable):
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  return variable_parameters

total_parameters = 0
for variable in tf.trainable_variables():
  variable_parameters = volume(variable)
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

  def run(fetches):
    return session.run(fetches, feed_dict={})

  if dump is not None:
    variables = dict((variable.name,variable) for variable in tf.trainable_variables())
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

  if conv_norm_by_xy is not None:
    variables = dict((variable.name,variable) for variable in tf.trainable_variables())
    for name in conv_norm_by_xy.split(","):
      variable = variables[name]

      #Should be x,y,in_channels,out_channels
      assert(len(variable.shape) == 4)
      norms = tf.sqrt(tf.reduce_mean(variable*variable,axis=[2,3]))
      norms = np.array(run(norms))
      print(name + " " + str(volume(variable)) + " parameters")
      for y in range(norms.shape[1]):
        for x in range(norms.shape[0]):
          print(norms[x,y], end=",")
        print("")

  if conv_norm_by_channel is not None:
    variables = dict((variable.name,variable) for variable in tf.trainable_variables())

    #Each convolution weight variable has a set of channels it takes in as input and a set of channels it produces
    #as output. This is a dictionary of the mapping.
    channel_names_by_var_name = {
      "conv1/wcenter:0":("input","trunk"),
      "conv1/w:0":("input","trunk"),
      "rconv1/w1:0":("trunk","rconv1mid"),
      "rconv1/w2:0":("rconv1mid","trunk"),
      "rconv2/w1:0":("trunk","rconv2mid"),
      "rconv2/w2:0":("rconv2mid","trunk"),
      "rconv3/w1:0":("trunk","rconv3mid"),
      "rconv3/w2:0":("rconv3mid","trunk"),
      "rconv4/w1:0":("trunk","rconv4mid"),
      "rconv4/w2:0":("rconv4mid","trunk"),
      "hvconv1/w1:0":("trunk","hconv1mid"),
      "hvconv1/w2:0":("hvconv1mid","trunk"),
      "hvconv2/w1:0":("trunk","hconv2mid"),
      "hvconv2/w2:0":("hvconv2mid","trunk"),
      "p1/intermediate_conv/w:0":("trunk","policy"),
      "g1/w:0":("trunk","policyglobal"),
    }

    #Try permuting channels to group channels as best as possible

    #First, build a dictionary mapping the name of the channel to some permutation arrays that we will permute
    #as we try to optimize.
    perm_of_channel_name = {}
    for var_name in conv_norm_by_channel.split(","):
      variable = variables[var_name]
      (input_channel_name,output_channel_name) = channel_names_by_var_name[var_name]
      perm_of_channel_name[input_channel_name] = [i for i in range(variable.shape[2].value)]
      perm_of_channel_name[output_channel_name] = [i for i in range(variable.shape[3].value)]
      random.shuffle(perm_of_channel_name[input_channel_name])
      random.shuffle(perm_of_channel_name[output_channel_name])

    #Next, call out to tensorflow and collect the average norms of each input->output channel weight across all the 3x3 or other-sized
    #convolutions for that weight. These dictionaries all take the var name as a key.
    norms = {}
    #And these are the same matrix, but normalized so that the rows, or the columns, have norm 1.
    input_normalized_norms = {}  #Normalized so that each input (axis 0) maps to an output vector of norm 1
    output_normalized_norms = {} #Normalized so that each output (axis 1) maps to an input vector of norm 1
    for var_name in conv_norm_by_channel.split(","):
      variable = variables[var_name]

      #Should be x,y,in_channels,out_channels
      assert(len(variable.shape) == 4)
      norm = tf.sqrt(tf.reduce_mean(variable*variable,axis=[0,1]))
      norm = np.array(run(norm))
      norms[var_name] = norm

      input_normalized_norms[var_name] = norm / (1e-10 + np.linalg.norm(norm,axis=1,keepdims=True))
      output_normalized_norms[var_name] = norm / (1e-10 + np.linalg.norm(norm,axis=0,keepdims=True))

    #Optimization - caches by variable so that we don't need to recompute things when they don't change.
    cached_input_score_by_var_name = {}
    cached_output_score_by_var_name = {}
    #Computes the score - the sum of the norm of the differences between successive rows and columns in each permuted matrix.
    def score():
      score = 0.0
      for var_name in norms:
        input_score_for_var_name = 0.0
        output_score_for_var_name = 0.0
        inorm = input_normalized_norms[var_name]
        onorm = output_normalized_norms[var_name]
        (input_channel_name,output_channel_name) = channel_names_by_var_name[var_name]

        if var_name in cached_input_score_by_var_name:
          input_score_for_var_name = cached_input_score_by_var_name[var_name]
        else:
          perm0 = perm_of_channel_name[input_channel_name]
          for i in range(1,inorm.shape[0]):
            xa = inorm[perm0[i-1],:]
            xb = inorm[perm0[i],:]
            input_score_for_var_name += np.linalg.norm(xa-xb)

          cached_input_score_by_var_name[var_name] = input_score_for_var_name

        if var_name in cached_output_score_by_var_name:
          output_score_for_var_name = cached_output_score_by_var_name[var_name]
        else:
          perm1 = perm_of_channel_name[output_channel_name]
          for i in range(1,onorm.shape[1]):
            xa = onorm[:,perm1[i-1]]
            xb = onorm[:,perm1[i]]
            output_score_for_var_name += np.linalg.norm(xa-xb)

          cached_output_score_by_var_name[var_name] = output_score_for_var_name

        score += input_score_for_var_name
        score += output_score_for_var_name
      return score

    #Clear the cache when a given channel changes, by searching through all variables that involve that channel
    def clear_cache(channel_name):
      for var_name in norms:
        norm = norms[var_name]
        (input_channel_name,output_channel_name) = channel_names_by_var_name[var_name]
        if input_channel_name == channel_name:
          del cached_input_score_by_var_name[var_name]
        if output_channel_name == channel_name:
          del cached_output_score_by_var_name[var_name]

    def swap(channel_name,arr,i,j):
      clear_cache(channel_name)
      tmp = arr[i]
      arr[i] = arr[j]
      arr[j] = tmp

    def rotate(channel_name,arr,i,j):
      clear_cache(channel_name)

      if i > j:
        tmp = arr[j]
        for k in range(j,i):
          arr[k] = arr[k+1]
        arr[i] = tmp
      elif i < j:
        tmp = arr[j]
        for k in range(j,i,-1):
          arr[k] = arr[k-1]
        arr[i] = tmp

    #Simulated-annealing accept criterion
    def should_accept(cur_score, new_score, temperature):
      if new_score < cur_score:
        return True
      return np.random.random() < np.exp((cur_score - new_score)/temperature)

    #Now loop and actually perform the optimization
    cur_score = score()
    logspace = np.logspace(-0.5,-6.0,100000)
    for iteration in range(100000):
      temperature = logspace[iteration]
      if iteration % 50 == 0:
        print("Optimizing " + str(iteration) + " score " + str(cur_score) + " temperature " + str(temperature),flush=True)

      for channel_name in perm_of_channel_name:
        perm = perm_of_channel_name[channel_name]
        i = np.random.randint(len(perm))
        j = np.random.randint(len(perm))
        if i != j:
          swap(channel_name, perm,i,j)
          new_score = score()
          if should_accept(cur_score,new_score,temperature):
            cur_score = new_score
          else:
            swap(channel_name, perm,i,j)

      for channel_name in perm_of_channel_name:
        perm = perm_of_channel_name[channel_name]
        i = np.random.randint(len(perm))
        j = np.random.randint(len(perm))
        if i != j:
          rotate(channel_name, perm,i,j)
          new_score = score()
          if should_accept(cur_score,new_score,temperature):
            cur_score = new_score
          else:
            rotate(channel_name, perm,j,i)

    #Print the results
    for var_name in conv_norm_by_channel.split(","):
      norm = norms[var_name]
      (input_channel_name,output_channel_name) = channel_names_by_var_name[var_name]
      perm0 = perm_of_channel_name[input_channel_name]
      perm1 = perm_of_channel_name[output_channel_name]

      print(var_name + " " + str(volume(variables[var_name])) + " parameters")

      for inflow in range(norm.shape[0]):
        for outflow in range(norm.shape[1]):
          print(norm[perm0[inflow],perm1[outflow]], end=",")
        print("")


