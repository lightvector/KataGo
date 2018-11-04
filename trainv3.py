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
import datetime
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
parser.add_argument('-datadir', help='Directory with a train and val subdir of tf records data', required=True)
parser.add_argument('-exportdir', help='Directory to export models periodically', required=True)
parser.add_argument('-exportsuffix', help='Suffix to append to names of models', required=True)
parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
parser.add_argument('-batch-size', help='Expected batch size of the input data, must match tfrecords', type=int, required=True)
parser.add_argument('-verbose', help='verbose', required=False, action='store_true')
args = vars(parser.parse_args())

traindir = args["traindir"]
datadir = args["datadir"]
exportdir = args["exportdir"]
exportsuffix = args["exportsuffix"]
pos_len = args["pos_len"]
batch_size = args["batch_size"]
verbose = args["verbose"]
logfilemode = "a"

if not os.path.exists(traindir):
  os.makedirs(traindir)
if not os.path.exists(exportdir):
  os.makedirs(exportdir)

bareformatter = logging.Formatter("%(message)s")
fh = logging.FileHandler(traindir+"/train.log", mode=logfilemode)
fh.setFormatter(bareformatter)

tensorflowlogger = logging.getLogger("tensorflow")
tensorflowlogger.setLevel(logging.INFO)
tensorflowlogger.addHandler(fh)

trainlogger = logging.getLogger("trainlogger")
trainlogger.setLevel(logging.INFO)
trainlogger.addHandler(fh)

np.set_printoptions(linewidth=150)

def trainlog(s):
  print(s,flush=True)
  trainlogger.info(s)

tf.logging.set_verbosity(tf.logging.INFO)

num_samples_per_epoch = 1000000
num_batches_per_epoch = int(round(num_samples_per_epoch / batch_size))

def find_var(name):
  for variable in tf.global_variables():
    if variable.name == name:
      return variable

# MODEL ----------------------------------------------------------------
def model_fn(features,labels,mode,params):

  trainlog("Building model")
  model_config = {
    "trunk_num_channels":128,
    "mid_num_channels":128,
    "regular_num_channels":96,
    "dilated_num_channels":32,
    "gpool_num_channels":32,
    "block_kind": [
      ["rconv1","regular"],
      ["rconv2","regular"],
      ["rconv3","regular"],
      ["rconv4","gpool"],
      ["rconv5","regular"],
      ["rconv6","regular"],
      ["rconv7","gpool"],
      ["rconv8","regular"]
    ],
    "p1_num_channels":48,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "v2_size":32
  }

  with open(os.path.join(traindir,"model.config.json"),"w") as f:
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

  placeholders["global_inputs"] = features["ginc"]
  placeholders["symmetries"] = tf.greater(tf.random_uniform([3],minval=0,maxval=2,dtype=tf.int32),tf.zeros([3],dtype=tf.int32))

  if mode == tf.estimator.ModeKeys.PREDICT:
    placeholders["is_training"] = tf.constant(False,dtype=tf.bool)
    model = ModelV3(model_config,pos_len,placeholders)

    predictions = {}
    predictions["policy_output"] = model.policy_output
    predictions["value_output"] = model.value_output
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  placeholders["include_history"] = features["gtnc"][:,28:33]

  policy_target0 = features["ptncm"][:,0,:]
  policy_target0 = policy_target0 / tf.reduce_sum(policy_target0,axis=1,keepdims=True)
  placeholders["policy_target"] = policy_target0
  placeholders["policy_target_weight"] = features["gtnc"][:,25]

  placeholders["value_target"] = features["gtnc"][:,0:3]
  placeholders["scorevalue_target"] = features["gtnc"][:,3]
  placeholders["utilityvar_target"] = features["gtnc"][:,21:25]
  placeholders["ownership_target"] = tf.reshape(features["vtnchw"],[-1,pos_len,pos_len])
  placeholders["target_weight_from_data"] = features["gtnc"][:,0]*0 + 1
  placeholders["ownership_target_weight"] = 1.0-features["gtnc"][:,2] #1 if normal game, 0 if no result
  placeholders["l2_reg_coeff"] = tf.constant(l2_coeff_value,dtype=tf.float32)

  if mode == tf.estimator.ModeKeys.EVAL:
    placeholders["is_training"] = tf.constant(False,dtype=tf.bool)
    model = ModelV3(model_config,pos_len,placeholders)

    target_vars = Target_varsV3(model,for_optimization=True,require_last_move=False,placeholders=placeholders)
    metrics = MetricsV3(model,target_vars,include_debug_stats=False)

    wsum = tf.Variable(
      0.0,dtype=tf.float32,name="wsum",trainable=False,
      collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
      synchronization=tf.VariableSynchronization.ON_READ,
      aggregation=tf.VariableAggregation.SUM
    )
    wsum_op = tf.assign_add(wsum,target_vars.weight_sum)
    return tf.estimator.EstimatorSpec(
      mode,
      loss=target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32),
      eval_metric_ops={
        "wsum": (wsum.read_value(),wsum_op),
        "ploss": tf.metrics.mean(target_vars.policy_loss_unreduced, weights=target_vars.target_weight_used),
        "vloss": tf.metrics.mean(target_vars.value_loss_unreduced, weights=target_vars.target_weight_used),
        "svloss": tf.metrics.mean(target_vars.scorevalue_loss_unreduced, weights=target_vars.target_weight_used),
        "uvloss": tf.metrics.mean(target_vars.utilityvar_loss_unreduced, weights=target_vars.target_weight_used),
        "oloss": tf.metrics.mean(target_vars.ownership_loss_unreduced, weights=target_vars.target_weight_used),
        "rloss": tf.metrics.mean(target_vars.reg_loss_per_weight, weights=target_vars.weight_sum),
        "pacc1": tf.metrics.mean(metrics.accuracy1_unreduced, weights=target_vars.target_weight_used),
        "ventr": tf.metrics.mean(metrics.value_entropy_unreduced, weights=target_vars.target_weight_used)
      }
    )

  if mode == tf.estimator.ModeKeys.TRAIN:
    placeholders["is_training"] = tf.constant(True,dtype=tf.bool)
    model = ModelV3(model_config,pos_len,placeholders)

    target_vars = Target_varsV3(model,for_optimization=True,require_last_move=False,placeholders=placeholders)
    metrics = MetricsV3(model,target_vars,include_debug_stats=False)
    global_step = tf.train.get_global_step()
    global_step_float = tf.cast(global_step, tf.float32)
    global_epoch = global_step_float / tf.constant(num_batches_per_epoch,dtype=tf.float32)

    global_epoch_float_capped = tf.math.minimum(tf.constant(180.0),global_epoch)
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

    (ploss,ploss_op) = moving_mean(target_vars.policy_loss_unreduced, weights=target_vars.target_weight_used)
    (vloss,vloss_op) = moving_mean(target_vars.value_loss_unreduced, weights=target_vars.target_weight_used)
    (svloss,svloss_op) = moving_mean(target_vars.scorevalue_loss_unreduced, weights=target_vars.target_weight_used)
    (uvloss,uvloss_op) = moving_mean(target_vars.utilityvar_loss_unreduced, weights=target_vars.target_weight_used)
    (oloss,oloss_op) = moving_mean(target_vars.ownership_loss_unreduced, weights=target_vars.target_weight_used)
    (rloss,rloss_op) = moving_mean(target_vars.reg_loss_per_weight, weights=target_vars.weight_sum)
    (pacc1,pacc1_op) = moving_mean(metrics.accuracy1_unreduced, weights=target_vars.target_weight_used)
    (ventr,ventr_op) = moving_mean(metrics.value_entropy_unreduced, weights=target_vars.target_weight_used)
    (wmean,wmean_op) = tf.metrics.mean(target_vars.weight_sum)

    print_train_loss_every_batches = 50
    # print_train_loss_every_batches = num_batches_per_epoch

    logging_hook = tf.train.LoggingTensorHook({
      "nsamp": global_step * tf.constant(batch_size,dtype=tf.int64),
      "wsum": global_step_float * wmean,
      "ploss": ploss,
      "vloss": vloss,
      "svloss": svloss,
      "uvloss": uvloss,
      "oloss": oloss,
      "rloss": rloss,
      "pacc1": pacc1,
      "ventr": ventr,
      "pslr": per_sample_learning_rate
    }, every_n_iter=print_train_loss_every_batches)
    return tf.estimator.EstimatorSpec(
      mode,
      loss=(target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32)),
      train_op=tf.group(train_step,ploss_op,vloss_op,svloss_op,uvloss_op,oloss_op,rloss_op,pacc1_op,ventr_op,wmean_op),
      training_hooks = [logging_hook]
    )

# INPUTS ------------------------------------------------------------------------

NUM_POLICY_TARGETS = 1
NUM_GLOBAL_TARGETS = 45
NUM_VALUE_SPATIAL_TARGETS = 1

raw_input_features = {
  "binchwp": tf.FixedLenFeature([],tf.string),
  "ginc": tf.FixedLenFeature([batch_size*ModelV3.NUM_GLOBAL_INPUT_FEATURES],tf.float32),
  "ptncm": tf.FixedLenFeature([batch_size*NUM_POLICY_TARGETS*(pos_len*pos_len+1)],tf.float32),
  "gtnc": tf.FixedLenFeature([batch_size*NUM_GLOBAL_TARGETS],tf.float32),
  "vtnchw": tf.FixedLenFeature([batch_size*NUM_VALUE_SPATIAL_TARGETS*pos_len*pos_len],tf.float32)
}
raw_input_feature_placeholders = {
  "binchwp": tf.placeholder(tf.uint8,[batch_size,ModelV3.NUM_BIN_INPUT_FEATURES,(pos_len*pos_len+7)//8]),
  "ginc": tf.placeholder(tf.float32,[batch_size,ModelV3.NUM_GLOBAL_INPUT_FEATURES]),
  "ptncm": tf.placeholder(tf.float32,[batch_size,NUM_POLICY_TARGETS,pos_len*pos_len+1]),
  "gtnc": tf.placeholder(tf.float32,[batch_size,NUM_GLOBAL_TARGETS]),
  "vtnchw": tf.placeholder(tf.float32,[batch_size,NUM_VALUE_SPATIAL_TARGETS,pos_len,pos_len])
}

def parse_input(serialized_example):
  example = tf.parse_single_example(serialized_example,raw_input_features)
  binchwp = tf.decode_raw(example["binchwp"],tf.uint8)
  ginc = example["ginc"]
  ptncm = example["ptncm"]
  gtnc = example["gtnc"]
  vtnchw = example["vtnchw"]
  return {
    "binchwp": tf.reshape(binchwp,[batch_size,ModelV3.NUM_BIN_INPUT_FEATURES,(pos_len*pos_len+7)//8]),
    "ginc": tf.reshape(ginc,[batch_size,ModelV3.NUM_GLOBAL_INPUT_FEATURES]),
    "ptncm": tf.reshape(ptncm,[batch_size,NUM_POLICY_TARGETS,pos_len*pos_len+1]),
    "gtnc": tf.reshape(gtnc,[batch_size,NUM_GLOBAL_TARGETS]),
    "vtnchw": tf.reshape(vtnchw,[batch_size,NUM_VALUE_SPATIAL_TARGETS,pos_len,pos_len])
  }

def train_input_fn(tdatadir):
  train_files = [os.path.join(tdatadir,fname) for fname in os.listdir(tdatadir)]
  trainlog("Constructing train input pipe, %d files" % len(train_files))
  dataset = tf.data.Dataset.from_tensor_slices(train_files)
  dataset = dataset.shuffle(1048576)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.shuffle(1000)
  dataset = dataset.map(parse_input)
  dataset = dataset.repeat()
  return dataset

def val_input_fn(vdatadir):
  val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir)]
  trainlog("Constructing validation input pipe, %d files" % len(val_files))
  dataset = tf.data.Dataset.from_tensor_slices(val_files)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.map(parse_input)
  return dataset

# TRAINING PARAMETERS ------------------------------------------------------------

trainlog("Beginning training")

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  model_dir=traindir,
  params={},
  config=tf.estimator.RunConfig(
    save_checkpoints_steps=num_batches_per_epoch,
    keep_checkpoint_every_n_hours = 1000000,
    keep_checkpoint_max = 10
  )
)

class CheckpointSaverListenerFunction(tf.train.CheckpointSaverListener):
  def __init__(self,f):
    self.func_to_call = f

  def begin(self):
    pass
  def before_save(self, session, global_step_value):
    pass
  def after_save(self, session, global_step_value):
    self.func_to_call(global_step_value)
  def end(self, session, global_step_value):
    pass

def dump_and_flush_json(data,filename):
  with open(filename,"w") as f:
    json.dump(data,f)
    f.flush()
    os.fsync(f.fileno())

trainhistory = []
if os.path.isfile(os.path.join(traindir,"trainhistory.json")):
  trainlog("Loading existing training history: " + str(os.path.join(traindir,"trainhistory.json")))
  with open(os.path.join(traindir,"trainhistory.json")) as f:
    trainhistory = json.load(f)

def save_history(global_step_value):
  trainhistory.append(("nsamp",int(global_step_value * batch_size)))
  savepath = os.path.join(traindir,"trainhistory.json")
  savepathtmp = os.path.join(traindir,"trainhistory.json.tmp")
  dump_and_flush_json(trainhistory,savepathtmp)
  os.rename(savepathtmp,savepath)
  trainlog("Wrote " + savepath)

last_curdatadir = ""
last_datainfo_row = 0
while True:
  curdatadir = os.path.realpath(datadir)
  if curdatadir != last_curdatadir:
    trainlog("Updated training data: " + curdatadir)
    with open(os.path.join(curdatadir,"train.json")) as f:
      datainfo = json.load(f)
      last_datainfo_row = max(end_row_idx for (fname,(start_row_idx,end_row_idx)) in datainfo)
    trainhistory.append(("newdata",datainfo))

  trainlog("=========================================================================")
  trainlog("BEGINNING NEXT EPOCH")
  trainlog("=========================================================================")
  trainlog("Current time: " + str(datetime.datetime.now()))
  globalstep = int(estimator.get_variable_value("global_step:0"))
  trainlog("Global step: %d (%d samples)" % (globalstep, globalstep*batch_size))

  #Train
  trainlog("Beginning training epoch!")
  tdatadir = os.path.join(curdatadir,"train")
  estimator.train(
    (lambda: train_input_fn(tdatadir)),
    steps=num_batches_per_epoch,
    saving_listeners=[
      CheckpointSaverListenerFunction(save_history)
    ]
  )

  # #Validate
  trainlog("Beginning validation after epoch!")
  vdatadir = os.path.join(curdatadir,"val")
  val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir)]
  if len(val_files) == 0:
    trainlog("No validation files, skipping validation step")
  else:
    estimator.evaluate(
      (lambda: val_input_fn(vdatadir))
    )

  #Export a model for testing, unless somehow it already exists
  globalstep = int(estimator.get_variable_value("global_step:0"))
  modelname = "s%d-d%d-%s" % (
    globalstep*batch_size,
    last_datainfo_row,
    exportsuffix
  )
  savepath = os.path.join(exportdir,modelname)
  savepathtmp = os.path.join(exportdir,modelname+".tmp")
  if os.path.exists(savepath):
    trainlog("NOT saving model, already exists at: " + savepath)
  else:
    trainlog("SAVING MODEL TO: " + savepath)
    estimator.export_saved_model(
      os.path.join(savepathtmp,"savedmodel"),
      tf.estimator.export.build_raw_serving_input_receiver_fn(raw_input_feature_placeholders)
    )
    dump_and_flush_json(trainhistory,os.path.join(savepathtmp,"trainhistory.json"))
    os.rename(savepathtmp,savepath)


  time.sleep(1)

