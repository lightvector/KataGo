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
import json
import tensorflow as tf
import numpy as np

import data
from board import Board
from modelv3 import ModelV3, Target_varsV3, MetricsV3

#Command and args-------------------------------------------------------------------

description = """
Train neural net on Go positions from tf record files of batches from selfplay.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
parser.add_argument('-tdatadir', help='Directory of tf records data to train on', required=True)
parser.add_argument('-vdatadir', help='Directory of tf records data to validate on', required=True)
parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
parser.add_argument('-batch-size', help='Expected batch size of the input data, must match tfrecords', type=int, required=True)
parser.add_argument('-verbose', help='verbose', required=False, action='store_true')
parser.add_argument('-restart-file', help='restart training from file', required=False)
parser.add_argument('-restart-epoch', help='restart training epoch', required=False)
parser.add_argument('-restart-time', help='restart training time', required=False)
parser.add_argument('-fast-factor', help='divide training batches per epoch by this factor', required=False)
parser.add_argument('-validation-prop', help='only use this proportion of validation set', required=False)
args = vars(parser.parse_args())

traindir = args["traindir"]
tdatadir = args["tdatadir"]
vdatadir = args["vdatadir"]
pos_len = args["pos_len"]
batch_size = args["batch_size"]
verbose = args["verbose"]
restart_file = None
start_epoch = 0
start_elapsed = 0
fast_factor = 1
validation_prop = 1.0
logfilemode = "w"
if "restart_file" in args and args["restart_file"] is not None:
  restart_file = args["restart_file"]
  start_epoch = int(args["restart_epoch"])
  start_elapsed = float(args["restart_time"])
  logfilemode = "a"

if "fast_factor" in args and args["fast_factor"] is not None:
  fast_factor = int(args["fast_factor"])
if "validation_prop" in args and args["validation_prop"] is not None:
  validation_prop = float(args["validation_prop"])

if not os.path.exists(traindir):
  os.makedirs(traindir)

bareformatter = logging.Formatter("%(message)s")
trainlogger = logging.getLogger("tensorflow")
trainlogger.setLevel(logging.INFO)
fh = logging.FileHandler(traindir+"/train.log", mode=logfilemode)
fh.setFormatter(bareformatter)
trainlogger.addHandler(fh)

detaillogger = logging.getLogger("detaillogger")
detaillogger.setLevel(logging.INFO)
fh = logging.FileHandler(traindir+"/detail.log", mode=logfilemode)
fh.setFormatter(bareformatter)
detaillogger.addHandler(fh)

np.set_printoptions(linewidth=150)

def trainlog(s):
  print(s,flush=True)
  trainlogger.info(s)
  detaillogger.info(s)

def detaillog(s):
  detaillogger.info(s)

tf.logging.set_verbosity(tf.logging.INFO)

num_samples_per_epoch = 1000000//fast_factor
num_batches_per_epoch = int(round(num_samples_per_epoch / batch_size))

def find_var(name):
  for variable in tf.global_variables():
    if variable.name == name:
      return variable

# MODEL ----------------------------------------------------------------
def model_fn(features,labels,mode,params):

  print("Building model", flush=True)
  model_config = {}
  model_config["pos_len"] = pos_len
  with open(traindir + ".config.json","w") as f:
    json.dump(model_config,f)

  #L2 regularization coefficient
  l2_coeff_value = 0.00003

  placeholders = {}

  binchwp = features["binchwp"]
  #Unpack binary data
  bitmasks = tf.reshape(tf.constant([128,64,32,16,8,4,2,1],dtype=tf.uint8),[1,1,1,8])
  binchw = tf.reshape(tf.bitwise.bitwise_and(tf.expand_dims(binchwp,axis=3),bitmasks),[-1,ModelV3.NUM_BIN_INPUT_FEATURES,((pos_len*pos_len+7)//8)*8])
  binchw = binchw[:,:,:pos_len*pos_len]
  binhwc = tf.cast(tf.transpose(binchw, [0,2,1]),tf.float32)
  binhwc = tf.math.minimum(binhwc,tf.constant(1.0))

  placeholders["bin_inputs"] = binhwc

  placeholders["float_inputs"] = features["finc"]
  placeholders["symmetries"] = tf.greater(tf.random_uniform([3],minval=0,maxval=2,dtype=tf.int32),tf.zeros([3],dtype=tf.int32))
  placeholders["include_history"] = features["ftnc"][:,28:33]

  policy_target0 = features["ptncm"][:,0,:]
  policy_target0 = policy_target0 / tf.reduce_sum(policy_target0,axis=1,keepdims=True)
  placeholders["policy_targets"] = policy_target0

  placeholders["value_targets"] = features["ftnc"][:,0:3]
  placeholders["scorevalue_targets"] = features["ftnc"][:,3]
  placeholders["ownership_targets"] = tf.reshape(features["vtnchw"],[-1,pos_len,pos_len])
  placeholders["target_weights_from_data"] = features["ftnc"][:,0]*0 + 1
  placeholders["ownership_target_weights"] = 1.0-features["ftnc"][:,2] #1 if normal game, 0 if no result
  placeholders["l2_reg_coeff"] = tf.constant(l2_coeff_value,dtype=tf.float32)

  if mode == tf.estimator.ModeKeys.PREDICT:
    placeholders["is_training"] = tf.constant(False,dtype=tf.bool)
    model = ModelV3(model_config,placeholders)

    predictions = {}
    predictions["policy_output"] = model.policy_output
    predictions["value_output"] = model.value_output
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  if mode == tf.estimator.ModeKeys.EVAL:
    placeholders["is_training"] = tf.constant(False,dtype=tf.bool)
    model = ModelV3(model_config,placeholders)

    target_vars = Target_varsV3(model,for_optimization=True,require_last_move=False,placeholders=placeholders)
    metrics = MetricsV3(model,target_vars,include_debug_stats=False)

    wsum = tf.Variable(0.0,dtype=tf.float32)
    wsum_op = tf.assign_add(wsum,target_vars.weight_sum)
    return tf.estimator.EstimatorSpec(
      mode,
      loss=target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32),
      eval_metric_ops={
        "wsum": (wsum.read_value(),wsum_op),
        "ploss": tf.metrics.mean(target_vars.policy_loss_unreduced, weights=target_vars.target_weights_used),
        "vloss": tf.metrics.mean(target_vars.value_loss_unreduced, weights=target_vars.target_weights_used),
        "svloss": tf.metrics.mean(target_vars.scorevalue_loss_unreduced, weights=target_vars.target_weights_used),
        "oloss": tf.metrics.mean(target_vars.ownership_loss_unreduced, weights=target_vars.target_weights_used),
        "rloss": tf.metrics.mean(target_vars.reg_loss_per_weight, weights=target_vars.weight_sum),
        "pacc1": tf.metrics.mean(metrics.accuracy1_unreduced, weights=target_vars.target_weights_used),
        "ventr": tf.metrics.mean(metrics.value_entropy_unreduced, weights=target_vars.target_weights_used)
      }
    )

  if mode == tf.estimator.ModeKeys.TRAIN:
    placeholders["is_training"] = tf.constant(True,dtype=tf.bool)
    model = ModelV3(model_config,placeholders)

    target_vars = Target_varsV3(model,for_optimization=True,require_last_move=False,placeholders=placeholders)
    metrics = MetricsV3(model,target_vars,include_debug_stats=False)
    global_step = tf.train.get_global_step()
    global_step_float = tf.cast(global_step, tf.float32)
    global_epoch = global_step_float / tf.constant(num_batches_per_epoch,dtype=tf.float32)

    global_epoch_float_capped = tf.math.minimum(tf.constant(192.0),global_epoch)
    per_sample_learning_rate = (
      tf.constant(0.00020) / tf.pow(global_epoch_float_capped * tf.constant(0.1) + tf.constant(1.0), tf.constant(1.333333))
    )

    lr_adjusted_variables = model.lr_adjusted_variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #collect batch norm update operations
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.MomentumOptimizer(per_sample_learning_rate, momentum=0.9, use_nesterov=True)
      gradients = optimizer.compute_gradients(target_vars.opt_loss)
      adjusted_gradients = []
      for (grad,x) in gradients:
        adjusted_grad = grad
        if x.name in lr_adjusted_variables and grad is not None:
          adj_factor = lr_adjusted_variables[x.name]
          adjusted_grad = grad * adj_factor
          trainlog("Adjusting gradient for " + x.name + " by " + str(adj_factor))

        adjusted_gradients.append((adjusted_grad,x))
      train_step = optimizer.apply_gradients(adjusted_gradients, global_step=global_step)

    # def reduce_norm(x, axis=None, keepdims=False):
    #   return tf.sqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=keepdims))
    # relative_update_by_var = dict([
    #   (v.name,per_sample_learning_rate * reduce_norm(grad) / (1e-10 + reduce_norm(v))) for (grad,v) in adjusted_gradients if grad is not None
    # ])

    total_parameters = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      total_parameters += variable_parameters
      trainlog("Model variable %s, %d parameters" % (variable.name,variable_parameters))

    trainlog("Built model, %d total parameters" % total_parameters)

    for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
      trainlog("Additional update op on train step: %s" % update_op.name)

    def moving_mean(x,weights):
      sumwx = tf.reduce_sum(x*weights)
      sumw = tf.reduce_sum(weights)
      ema = tf.train.ExponentialMovingAverage(decay=0.999)
      op = ema.apply([sumwx,sumw])
      avg = ema.average(sumwx) / ema.average(sumw)
      return (avg,op)

    (ploss,ploss_op) = moving_mean(target_vars.policy_loss_unreduced, weights=target_vars.target_weights_used)
    (vloss,vloss_op) = moving_mean(target_vars.value_loss_unreduced, weights=target_vars.target_weights_used)
    (svloss,svloss_op) = moving_mean(target_vars.scorevalue_loss_unreduced, weights=target_vars.target_weights_used)
    (oloss,oloss_op) = moving_mean(target_vars.ownership_loss_unreduced, weights=target_vars.target_weights_used)
    (rloss,rloss_op) = moving_mean(target_vars.reg_loss_per_weight, weights=target_vars.weight_sum)
    (pacc1,pacc1_op) = moving_mean(metrics.accuracy1_unreduced, weights=target_vars.target_weights_used)
    (ventr,ventr_op) = moving_mean(metrics.value_entropy_unreduced, weights=target_vars.target_weights_used)
    (wmean,wmean_op) = tf.metrics.mean(target_vars.weight_sum)

    print_train_loss_every_batches = 10
    # print_train_loss_every_batches = num_batches_per_epoch

    logging_hook = tf.train.LoggingTensorHook({
      "nsamp": global_step * tf.constant(batch_size,dtype=tf.int64),
      "wsum": global_step_float * wmean,
      "ploss": ploss,
      "vloss": vloss,
      "svloss": svloss,
      "oloss": oloss,
      "rloss": rloss,
      "pacc1": pacc1,
      "ventr": ventr,
      "pslr": per_sample_learning_rate
    }, every_n_iter=print_train_loss_every_batches)
    return tf.estimator.EstimatorSpec(
      mode,
      loss=(target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32)),
      train_op=tf.group(train_step,ploss_op,vloss_op,svloss_op,oloss_op,rloss_op,pacc1_op,ventr_op,wmean_op),
      training_hooks = [logging_hook]
    )

# INPUTS ------------------------------------------------------------------------

NUM_POLICY_TARGETS = 1
NUM_FLOAT_TARGETS = 44
NUM_VALUE_SPATIAL_TARGETS = 1

raw_input_features = {
  "binchwp": tf.FixedLenFeature([],tf.string),
  "finc": tf.FixedLenFeature([batch_size*ModelV3.NUM_FLOAT_INPUT_FEATURES],tf.float32),
  "ptncm": tf.FixedLenFeature([batch_size*NUM_POLICY_TARGETS(pos_len*pos_len+1)],tf.float32),
  "ftnc": tf.FixedLenFeature([batch_size*NUM_FLOAT_TARGETS],tf.float32),
  "vtnchw": tf.FixedLenFeature([batch_size*NUM_VALUE_SPATIAL_TARGETS*pos_len*pos_len],tf.float32)
}
def parse_input(serialized_example):
  example = tf.parse_single_example(serialized_example,raw_input_features)
  binchwp = tf.decode_raw(example["binchwp"],tf.uint8)
  finc = example["finc"]
  ptncm = example["ptncm"]
  ftnc = example["ftnc"]
  vtnchw = example["vtnchw"]
  return {
    "binchwp": tf.reshape(binchwp,[batch_size,ModelV3.NUM_BIN_INPUT_FEATURES,(pos_len*pos_len+7)//8]),
    "finc": tf.reshape(finc,[batch_size,ModelV3.NUM_FLOAT_INPUT_FEATURES]),
    "ptncm": tf.reshape(ptncm,[batch_size,NUM_POLICY_TARGETS,pos_len*pos_len+1]),
    "ftnc": tf.reshape(ftnc,[batch_size,NUM_FLOAT_TARGETS]),
    "vtnchw": tf.reshape(vtnchw,[batch_size,NUM_VALUE_SPATIAL_TARGETS,pos_len,pos_len])
  }

def train_input_fn():
  files = [os.path.join(tdatadir,fname) for fname in os.listdir(tdatadir)]
  trainlog("Constructing train input pipe, %d files" % len(files))
  dataset = tf.data.Dataset.from_tensor_slices(files)
  dataset = dataset.shuffle(1048576)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.shuffle(1000)
  dataset = dataset.map(parse_input)
  dataset = dataset.repeat()
  return dataset

def val_input_fn():
  files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir)]
  trainlog("Constructing validation input pipe, %d files" % len(files))
  dataset = tf.data.Dataset.from_tensor_slices(files)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.map(parse_input)
  return dataset

# TRAINING PARAMETERS ------------------------------------------------------------

print("Training", flush=True)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  model_dir=traindir,
  params={},
  config=tf.estimator.RunConfig(
    save_checkpoints_steps=num_batches_per_epoch,
    keep_checkpoint_every_n_hours = 1000000,
    keep_checkpoint_max = 0
  )
)

# validate_every_batches = 100
validate_every_batches = num_batches_per_epoch

evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
  estimator,
  val_input_fn,
  every_n_iter = validate_every_batches
)

estimator.train(
  train_input_fn,
  hooks=[evaluator]
  # hooks=[]
)

