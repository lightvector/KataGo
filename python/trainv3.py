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
import gc
import tensorflow as tf
import numpy as np

import data
from board import Board
import modelv3
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


# MODEL ----------------------------------------------------------------
printed_model_yet = False

def model_fn(features,labels,mode,params):
  global printed_model_yet
  print_model = not printed_model_yet

  built = modelv3.build_model_from_tfrecords_features(features,mode,print_model,trainlog,model_config,pos_len,num_batches_per_epoch)

  if mode == tf.estimator.ModeKeys.PREDICT:
    model = built
    predictions = {}
    predictions["policy_output"] = model.policy_output
    predictions["value_output"] = model.value_output
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  if mode == tf.estimator.ModeKeys.EVAL:
    (model,target_vars,metrics) = built
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
    (model,target_vars,metrics,global_step,global_step_float,per_sample_learning_rate,train_step) = built
    printed_model_yet = True

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

    print_train_loss_every_batches = 100

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

    printed_model_yet = True

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

def train_input_fn(train_files_to_use,total_num_train_files):
  trainlog("Constructing train input pipe, %d/%d files used" % (len(train_files_to_use),total_num_train_files))
  # def genfiles():
  #   trainlog("Shuffling/reshuffling training files for dataset")
  #   train_files_shuffled = train_files.copy()
  #   random.shuffle(train_files_shuffled)
  #   for filename in train_files_shuffled:
  #     trainlog("Yielding training file for dataset: " + filename)
  #     yield filename
  # dataset = tf.data.Dataset.from_generator(genfiles,tf.string,output_shapes=tf.TensorShape([]))

  dataset = tf.data.Dataset.from_tensor_slices(train_files_to_use)
  dataset = dataset.shuffle(65536)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.shuffle(1000)
  dataset = dataset.map(parse_input)
  # dataset = dataset.repeat()
  return dataset

def val_input_fn(vdatadir):
  val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir) if fname.endswith(".tfrecord")]
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

last_curdatadir = None
last_datainfo_row = 0
globalstep = None
while True:
  curdatadir = os.path.realpath(datadir)
  if curdatadir != last_curdatadir:
    if not os.path.exists(curdatadir):
      trainlog("Training data path does not exist, waiting and trying again later: %s" % curdatadir)
      time.sleep(300)
      continue
    trainjsonpath = os.path.join(curdatadir,"train.json")
    if not os.path.exists(trainjsonpath):
      trainlog("Training data json file does not exist, waiting and trying again later: %s" % trainjsonpath)
      time.sleep(300)
      continue

    trainlog("Updated training data: " + curdatadir)
    last_curdatadir = curdatadir

    with open(trainjsonpath) as f:
      datainfo = json.load(f)
      last_datainfo_row = max(end_row_idx for (fname,(start_row_idx,end_row_idx)) in datainfo)
    trainhistory.append(("newdata",datainfo))

  trainlog("GC collect")
  gc.collect()

  trainlog("=========================================================================")
  trainlog("BEGINNING NEXT EPOCH")
  trainlog("=========================================================================")
  trainlog("Current time: " + str(datetime.datetime.now()))
  if globalstep is not None:
    trainlog("Global step: %d (%d samples)" % (globalstep, globalstep*batch_size))

  #Load training data files
  tdatadir = os.path.join(curdatadir,"train")
  train_files = [os.path.join(tdatadir,fname) for fname in os.listdir(tdatadir) if fname.endswith(".tfrecord")]

  #Filter down to a random subset that will comprise this epoch
  def train_files_gen():
    train_files_shuffled = train_files.copy()
    while True:
      random.shuffle(train_files_shuffled)
      for filename in train_files_shuffled:
        trainlog("Yielding training file for dataset: " + filename)
        yield filename

  #Sanity check - load a max of 100000 files.
  train_files_to_use = []
  batches_to_use_so_far = 0
  for filename in train_files_gen():
    train_files_to_use.append(filename)
    jsonfilename = os.path.splitext(filename)[0] + ".json"
    with open(jsonfilename) as f:
      trainfileinfo = json.load(f)
    batches_to_use_so_far += trainfileinfo["num_batches"]
    if batches_to_use_so_far >= num_batches_per_epoch or len(train_files_to_use) > 100000:
      break

  #Train
  trainlog("Beginning training epoch!")
  estimator.train(
    (lambda: train_input_fn(train_files_to_use,len(train_files))),
    saving_listeners=[
      CheckpointSaverListenerFunction(save_history)
    ]
  )

  #Validate
  trainlog("Beginning validation after epoch!")
  vdatadir = os.path.join(curdatadir,"val")
  val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir) if fname.endswith(".tfrecord")]
  if len(val_files) == 0:
    trainlog("No validation files, skipping validation step")
  else:
    estimator.evaluate(
      (lambda: val_input_fn(vdatadir))
    )

  globalstep = int(estimator.get_variable_value("global_step:0"))

  #Export a model for testing, unless somehow it already exists
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
    saved_to = estimator.export_saved_model(
      savepathtmp,
      tf.estimator.export.build_raw_serving_input_receiver_fn(raw_input_feature_placeholders)
    )
    os.rename(saved_to, os.path.join(savepathtmp,"saved_model"))
    dump_and_flush_json(trainhistory,os.path.join(savepathtmp,"trainhistory.json"))
    with open(os.path.join(savepathtmp,"model.config.json"),"w") as f:
      json.dump(model_config,f)

    time.sleep(1)
    os.rename(savepathtmp,savepath)


  time.sleep(1)

