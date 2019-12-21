#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import contextlib
import json
import datetime
import gc
import shutil
import glob
import tensorflow as tf
import numpy as np
import itertools

import data
from board import Board
from model import Model, Target_vars, Metrics, ModelUtils
import modelconfigs
import tfrecordio

#Command and args-------------------------------------------------------------------

description = """
Train neural net on Go positions from tf record files of batches from selfplay.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
parser.add_argument('-datadir', help='Directory with a train and val subdir of tf records data', required=True)
parser.add_argument('-exportdir', help='Directory to export models periodically', required=True)
parser.add_argument('-exportprefix', help='Prefix to append to names of models', required=True)
parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
parser.add_argument('-batch-size', help='Expected batch size of the input data, must match tfrecords', type=int, required=True)
parser.add_argument('-samples-per-epoch', help='Number of data samples to consider as one epoch', type=int, required=True)
parser.add_argument('-multi-gpus', help='Use multiple gpus, comma-separated device ids', required=False)
parser.add_argument('-gpu-memory-frac', help='Fraction of gpu memory to use', type=float, required=True)
parser.add_argument('-model-kind', help='String name for what model to use', required=True)
parser.add_argument('-lr-scale', help='LR multiplier on the hardcoded schedule', type=float, required=False)
parser.add_argument('-sub-epochs', help='Reload training data up to this many times per epoch', type=int, required=True)
parser.add_argument('-swa-sub-epoch-scale', help='Number of sub-epochs to average in expectation together for SWA', type=float, required=False)
parser.add_argument('-verbose', help='verbose', required=False, action='store_true')
parser.add_argument('-no-export', help='Do not export models', required=False, action='store_true')
args = vars(parser.parse_args())

traindir = args["traindir"]
datadir = args["datadir"]
exportdir = args["exportdir"]
exportprefix = args["exportprefix"]
pos_len = args["pos_len"]
batch_size = args["batch_size"]
samples_per_epoch = args["samples_per_epoch"]
multi_gpus = args["multi_gpus"]
gpu_memory_frac = args["gpu_memory_frac"]
model_kind = args["model_kind"]
lr_scale = args["lr_scale"]
sub_epochs = args["sub_epochs"]
swa_sub_epoch_scale = args["swa_sub_epoch_scale"]
verbose = args["verbose"]
no_export = args["no_export"]
logfilemode = "a"

if not os.path.exists(traindir):
  os.makedirs(traindir)
if not os.path.exists(exportdir):
  os.makedirs(exportdir)

longterm_checkpoints_dir = os.path.join(traindir,"longterm_checkpoints")
if not os.path.exists(longterm_checkpoints_dir):
  os.makedirs(longterm_checkpoints_dir)

bareformatter = logging.Formatter("%(message)s")
fh = logging.FileHandler(os.path.join(traindir,"train.log"), mode=logfilemode)
fh.setFormatter(bareformatter)

trainlogger = logging.getLogger("trainlogger")
trainlogger.setLevel(logging.INFO)
trainlogger.addHandler(fh)
trainlogger.propagate=False

np.set_printoptions(linewidth=150)

def trainlog(s):
  print(s,flush=True)
  trainlogger.info(s)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

num_batches_per_epoch = int(round(samples_per_epoch / batch_size))

model_config = modelconfigs.config_of_name[model_kind]

with open(os.path.join(traindir,"model.config.json"),"w") as f:
  json.dump(model_config,f)

trainlog(str(sys.argv))


# FIGURE OUT MULTIGPU ------------------------------------------------------------
num_gpus_used = 1
multi_gpu_device_ids = []
if multi_gpus is not None:
  for piece in multi_gpus.split(","):
    piece = piece.strip()
    multi_gpu_device_ids.append("/GPU:" + str(int(piece)))
  num_gpus_used = len(multi_gpu_device_ids)


# MODEL ----------------------------------------------------------------
printed_model_yet = False
initial_weights_already_loaded = False

if swa_sub_epoch_scale is not None:
  with tf.device("/cpu:0"):
    with tf.compat.v1.variable_scope("swa_model"):
      swa_model = Model(model_config,pos_len,placeholders={})
      swa_saver = tf.compat.v1.train.Saver(
        max_to_keep = 10000000,
        save_relative_paths = True,
      )
    swa_assign_placeholders = {}
    swa_wvalues = {}
    swa_weight = 0.0
    assign_ops = []
    for variable in itertools.chain(tf.compat.v1.model_variables(), tf.compat.v1.trainable_variables()):
      if variable.name.startswith("swa_model/"):
        placeholder = tf.compat.v1.placeholder(variable.dtype,variable.shape)
        assign_ops.append(tf.compat.v1.assign(variable,placeholder))
        swa_assign_placeholders[variable.name] = placeholder
        swa_wvalues[variable.name] = np.zeros([elt.value for elt in variable.shape])
    swa_assign_op = tf.group(*assign_ops)
  trainlog("Build SWA graph for SWA update and saving, %d variables" % len(swa_assign_placeholders))

def accumulate_swa(estimator):
  global swa_weight
  assert(swa_sub_epoch_scale is not None)

  old_factor = 1.0 - 1.0 / swa_sub_epoch_scale
  new_factor = 1.0 / swa_sub_epoch_scale

  new_swa_weight = swa_weight * old_factor + new_factor

  for swa_variable_name in swa_assign_placeholders:
    assert(swa_variable_name.startswith("swa_model/"))
    variable_name = swa_variable_name[len("swa_model/"):]
    swa_wvalues[swa_variable_name] *= old_factor
    swa_wvalues[swa_variable_name] += new_factor * estimator.get_variable_value(variable_name)

  swa_weight = new_swa_weight

def save_swa(savedir):
  global swa_weight
  assert(swa_sub_epoch_scale is not None)
  assignments = {}

  for swa_variable_name in swa_assign_placeholders:
    assert(swa_variable_name.startswith("swa_model/"))
    assignments[swa_assign_placeholders[swa_variable_name]] = swa_wvalues[swa_variable_name] / swa_weight

  with tf.compat.v1.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(swa_assign_op, assignments)
    if not os.path.exists(savedir):
      os.mkdir(savedir)
    os.mkdir(os.path.join(savedir,"saved_model"))
    os.mkdir(os.path.join(savedir,"saved_model","variables"))
    swa_saver.save(sess,os.path.join(savedir,"saved_model","variables","variables"), write_meta_graph=False, write_state=False)

def model_fn(features,labels,mode,params):
  global printed_model_yet
  global initial_weights_already_loaded

  print_model = not printed_model_yet

  num_globalsteps_per_epoch = num_batches_per_epoch
  built = ModelUtils.build_model_from_tfrecords_features(features,mode,print_model,trainlog,model_config,pos_len,num_globalsteps_per_epoch,lr_scale)

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
      collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES, tf.compat.v1.GraphKeys.METRIC_VARIABLES],
      synchronization=tf.VariableSynchronization.ON_READ,
      aggregation=tf.VariableAggregation.SUM
    )
    wsum_op = tf.assign_add(wsum,target_vars.weight_sum)
    return tf.estimator.EstimatorSpec(
      mode,
      loss=target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32),
      eval_metric_ops={
        #"wsum": (wsum.read_value(),wsum_op),
        "p0loss": tf.compat.v1.metrics.mean(target_vars.policy_loss_unreduced, weights=target_vars.target_weight_used),
        "p1loss": tf.compat.v1.metrics.mean(target_vars.policy1_loss_unreduced, weights=target_vars.target_weight_used),
        "vloss": tf.compat.v1.metrics.mean(target_vars.value_loss_unreduced, weights=target_vars.target_weight_used),
        "tdvloss": tf.compat.v1.metrics.mean(target_vars.td_value_loss_unreduced, weights=target_vars.target_weight_used),
        "smloss": tf.compat.v1.metrics.mean(target_vars.scoremean_loss_unreduced, weights=target_vars.target_weight_used),
        "leadloss": tf.compat.v1.metrics.mean(target_vars.lead_loss_unreduced, weights=target_vars.target_weight_used),
        "vtimeloss": tf.compat.v1.metrics.mean(target_vars.variance_time_loss_unreduced, weights=target_vars.target_weight_used),
        "sbpdfloss": tf.compat.v1.metrics.mean(target_vars.scorebelief_pdf_loss_unreduced, weights=target_vars.target_weight_used),
        "sbcdfloss": tf.compat.v1.metrics.mean(target_vars.scorebelief_cdf_loss_unreduced, weights=target_vars.target_weight_used),
        "oloss": tf.compat.v1.metrics.mean(target_vars.ownership_loss_unreduced, weights=target_vars.target_weight_used),
        "sloss": tf.compat.v1.metrics.mean(target_vars.scoring_loss_unreduced, weights=target_vars.target_weight_used),
        "fploss": tf.compat.v1.metrics.mean(target_vars.futurepos_loss_unreduced, weights=target_vars.target_weight_used),
        "rsdloss": tf.compat.v1.metrics.mean(target_vars.scorestdev_reg_loss_unreduced, weights=target_vars.target_weight_used),
        "rloss": tf.compat.v1.metrics.mean(target_vars.reg_loss_per_weight, weights=target_vars.weight_sum),
        "rscloss": tf.compat.v1.metrics.mean(target_vars.scale_reg_loss_unreduced, weights=target_vars.target_weight_used),
        "pacc1": tf.compat.v1.metrics.mean(metrics.accuracy1_unreduced, weights=target_vars.target_weight_used),
        "ventr": tf.compat.v1.metrics.mean(metrics.value_entropy_unreduced, weights=target_vars.target_weight_used),
        "ptentr": tf.compat.v1.metrics.mean(metrics.policy_target_entropy_unreduced, weights=target_vars.target_weight_used)
      }
    )

  if mode == tf.estimator.ModeKeys.TRAIN:
    (model,target_vars,metrics,global_step,global_step_float,per_sample_learning_rate,train_step) = built
    printed_model_yet = True

    def moving_mean(name,x,weights):
      sumwx = tf.reduce_sum(x*weights,name="printstats/wx/"+name)
      sumw = tf.reduce_sum(weights,name="printstats/w/"+name)
      moving_wx = tf.compat.v1.get_variable(initializer=tf.zeros([]),name=(name+"/moving_wx"),trainable=False)
      moving_w = tf.compat.v1.get_variable(initializer=tf.zeros([]),name=(name+"/moving_w"),trainable=False)

      decay = 0.999
      with tf.name_scope(name):
        wx_op = tf.keras.backend.moving_average_update(moving_wx,sumwx,decay)
        w_op = tf.keras.backend.moving_average_update(moving_w,sumw,decay)
        op = tf.group(wx_op,w_op)

      avg = (moving_wx + sumwx * (1.0-decay)) / (moving_w + sumw * (1.0-decay))
      return (avg,op)

    (p0loss,p0loss_op) = moving_mean("p0loss",target_vars.policy_loss_unreduced, weights=target_vars.target_weight_used)
    (p1loss,p1loss_op) = moving_mean("p1loss",target_vars.policy1_loss_unreduced, weights=target_vars.target_weight_used)
    (vloss,vloss_op) = moving_mean("vloss",target_vars.value_loss_unreduced, weights=target_vars.target_weight_used)
    (tdvloss,tdvloss_op) = moving_mean("tdvloss",target_vars.td_value_loss_unreduced, weights=target_vars.target_weight_used)
    (smloss,smloss_op) = moving_mean("smloss",target_vars.scoremean_loss_unreduced, weights=target_vars.target_weight_used)
    (leadloss,leadloss_op) = moving_mean("leadloss",target_vars.lead_loss_unreduced, weights=target_vars.target_weight_used)
    (vtimeloss,vtimeloss_op) = moving_mean("vtimeloss",target_vars.variance_time_loss_unreduced, weights=target_vars.target_weight_used)
    (sbpdfloss,sbpdfloss_op) = moving_mean("sbpdfloss",target_vars.scorebelief_pdf_loss_unreduced, weights=target_vars.target_weight_used)
    (sbcdfloss,sbcdfloss_op) = moving_mean("sbcdfloss",target_vars.scorebelief_cdf_loss_unreduced, weights=target_vars.target_weight_used)
    (oloss,oloss_op) = moving_mean("oloss",target_vars.ownership_loss_unreduced, weights=target_vars.target_weight_used)
    (sloss,sloss_op) = moving_mean("sloss",target_vars.scoring_loss_unreduced, weights=target_vars.target_weight_used)
    (fploss,fploss_op) = moving_mean("fploss",target_vars.futurepos_loss_unreduced, weights=target_vars.target_weight_used)
    (skloss,skloss_op) = moving_mean("skloss",target_vars.seki_loss_unreduced, weights=target_vars.target_weight_used)
    (rsdloss,rsdloss_op) = moving_mean("rsdloss",target_vars.scorestdev_reg_loss_unreduced, weights=target_vars.target_weight_used)
    (rloss,rloss_op) = moving_mean("rloss",target_vars.reg_loss_per_weight, weights=target_vars.weight_sum)
    (rscloss,rscloss_op) = moving_mean("rscloss",target_vars.scale_reg_loss_unreduced, weights=target_vars.target_weight_used)
    (pacc1,pacc1_op) = moving_mean("pacc1",metrics.accuracy1_unreduced, weights=target_vars.target_weight_used)
    (ptentr,ptentr_op) = moving_mean("ptentr",metrics.policy_target_entropy_unreduced, weights=target_vars.target_weight_used)
    (gnorm,gnorm_op) = moving_mean("gnorm",metrics.gnorm, weights=1.0)
    (exgnorm,exgnorm_op) = moving_mean("excessgnorm",metrics.excess_gnorm, weights=1.0)
    (wmean,wmean_op) = tf.compat.v1.metrics.mean(target_vars.weight_sum)

    print_train_loss_every_batches = 100

    logging_hook = tf.estimator.LoggingTensorHook({
      "nsamp": global_step * tf.constant(batch_size,dtype=tf.int64),
      "wsum": global_step_float * wmean * tf.constant(float(num_gpus_used)),
      "p0loss": p0loss,
      "p1loss": p1loss,
      "vloss": vloss,
      "tdvloss": tdvloss,
      "smloss": smloss,
      "leadloss": leadloss,
      "vtimeloss": vtimeloss,
      "sbpdfloss": sbpdfloss,
      "sbcdfloss": sbcdfloss,
      "oloss": oloss,
      "sloss": sloss,
      "fploss": fploss,
      "skloss": skloss,
      "skw": target_vars.seki_weight_scale,
      "rsdloss": rsdloss,
      "rloss": rloss,
      "rscloss": rscloss,
      "pacc1": pacc1,
      "ptentr": ptentr,
      "pslr": per_sample_learning_rate,
      "gnorm": gnorm,
      "exgnorm": exgnorm,
    }, every_n_iter=print_train_loss_every_batches)

    printed_model_yet = True

    sys.stdout.flush()
    sys.stderr.flush()

    initial_weights_dir = os.path.join(traindir,"initial_weights")
    if os.path.exists(initial_weights_dir) and not initial_weights_already_loaded:
      print("Initial weights found at: " + initial_weights_dir)
      checkpoint_path = os.path.join(initial_weights_dir,"model")
      vars_in_checkpoint = tf.contrib.framework.list_variables(checkpoint_path)
      print("Checkpoint contains:")
      for var in vars_in_checkpoint:
        print(var)

      print("Modifying graph to load weights from checkpoint upon init...")
      sys.stdout.flush()
      sys.stderr.flush()

      variables_to_restore = tf.trainable_variables()
      assignment_mapping = { v.name.split(":")[0] : v for v in variables_to_restore }
      tf.train.init_from_checkpoint(checkpoint_path, assignment_mapping)
      initial_weights_already_loaded = True

    return tf.estimator.EstimatorSpec(
      mode,
      loss=(target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32)),
      train_op=tf.group(
        train_step,
        p0loss_op,p1loss_op,vloss_op,tdvloss_op,smloss_op,leadloss_op,vtimeloss_op,sbpdfloss_op,sbcdfloss_op,
        oloss_op,sloss_op,fploss_op,skloss_op,rsdloss_op,rloss_op,rscloss_op,pacc1_op,ptentr_op,wmean_op,
        gnorm_op,exgnorm_op
      ),
      training_hooks = [logging_hook]
    )

# INPUTS ------------------------------------------------------------------------

raw_input_feature_placeholders = tfrecordio.make_raw_input_feature_placeholders(model_config,pos_len,batch_size)
if num_gpus_used > 1:
  parse_input = tfrecordio.make_tf_record_parser(model_config,pos_len,batch_size,multi_num_gpus = num_gpus_used)
else:
  parse_input = tfrecordio.make_tf_record_parser(model_config,pos_len,batch_size,multi_num_gpus = None)

def train_input_fn(train_files_to_use,total_num_train_files,batches_to_use,mode,input_context):
  assert(mode == tf.estimator.ModeKeys.TRAIN)
  if input_context:
    assert(input_context.num_input_pipelines == 1)
  trainlog("Constructing train input pipe, %d/%d files used (%d batches)" % (len(train_files_to_use),total_num_train_files,batches_to_use))
  dataset = tf.data.Dataset.from_tensor_slices(train_files_to_use)
  dataset = dataset.shuffle(1024)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.shuffle(100)
  dataset = dataset.map(parse_input)
  dataset = dataset.prefetch(2)
  if num_gpus_used > 1:
    dataset = dataset.unbatch()
  return dataset

def val_input_fn(vdatadir):
  val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir) if fname.endswith(".tfrecord")]
  trainlog("Constructing validation input pipe, %d files" % len(val_files))
  dataset = tf.data.Dataset.from_tensor_slices(val_files)
  dataset = dataset.flat_map(lambda fname: tf.data.TFRecordDataset(fname,compression_type="ZLIB"))
  dataset = dataset.map(parse_input)
  dataset = dataset.prefetch(2)
  if num_gpus_used > 1:
    dataset = dataset.unbatch()
  return dataset

# TRAINING PARAMETERS ------------------------------------------------------------

trainlog("Beginning training")

if multi_gpus is None:
  session_config = tf.ConfigProto()
  session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=traindir,
    params={},
    config=tf.estimator.RunConfig(
      save_checkpoints_steps=1000000000, #We get checkpoints every time we complete an epoch anyways
      keep_checkpoint_every_n_hours = 1000000,
      keep_checkpoint_max = 10,
      session_config = session_config
    )
  )
else:
  session_config = tf.ConfigProto(allow_soft_placement=True)
  session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac
  multigpu_strategy = tf.distribute.MirroredStrategy(devices=multi_gpu_device_ids)
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=traindir,
    params={},
    config=tf.estimator.RunConfig(
      save_checkpoints_steps=1000000000, #We get checkpoints every time we complete an epoch anyways
      keep_checkpoint_every_n_hours = 1000000,
      keep_checkpoint_max = 10,
      session_config = session_config,
      train_distribute = multigpu_strategy,
      eval_distribute = multigpu_strategy,
    )
  )


class CheckpointSaverListenerFunction(tf.estimator.CheckpointSaverListener):
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

trainhistory = {
  "files":[],
  "history":[]
}
if os.path.isfile(os.path.join(traindir,"trainhistory.json")):
  trainlog("Loading existing training history: " + str(os.path.join(traindir,"trainhistory.json")))
  with open(os.path.join(traindir,"trainhistory.json")) as f:
    trainhistory = json.load(f)
elif os.path.isfile(os.path.join(traindir,"initial_weights","trainhistory.json")):
  trainlog("Loading previous model's training history: " + str(os.path.join(traindir,"initial_weights","trainhistory.json")))
  with open(os.path.join(traindir,"initial_weights","trainhistory.json")) as f:
    trainhistory = json.load(f)
else:
  trainhistory["history"].append(("initialized",model_config))

def save_history(global_step_value):
  trainhistory["history"].append(("nsamp",int(global_step_value * batch_size)))
  savepath = os.path.join(traindir,"trainhistory.json")
  savepathtmp = os.path.join(traindir,"trainhistory.json.tmp")
  dump_and_flush_json(trainhistory,savepathtmp)
  os.replace(savepathtmp,savepath)
  trainlog("Wrote " + savepath)


# DATA RELOADING GENERATOR ------------------------------------------------------------

last_curdatadir = None
last_datainfo_row = 0
trainfilegenerator = None
num_train_files = 0
vdatadir = None
def maybe_reload_training_data():
  global last_curdatadir
  global last_datainfo_row
  global trainfilegenerator
  global trainhistory
  global num_train_files
  global vdatadir

  while True:
    curdatadir = os.path.realpath(datadir)
    if curdatadir != last_curdatadir:
      if not os.path.exists(curdatadir):
        trainlog("Training data path does not exist, waiting and trying again later: %s" % curdatadir)
        time.sleep(30)
        continue

      trainjsonpath = os.path.join(curdatadir,"train.json")
      if not os.path.exists(trainjsonpath):
        trainlog("Training data json file does not exist, waiting and trying again later: %s" % trainjsonpath)
        time.sleep(30)
        continue

      trainlog("Updated training data: " + curdatadir)
      last_curdatadir = curdatadir

      with open(trainjsonpath) as f:
        datainfo = json.load(f)
        last_datainfo_row = datainfo["range"][1]
      trainhistory["files"] = datainfo["files"]
      trainhistory["history"].append(("newdata",datainfo["range"]))

      #Load training data files
      tdatadir = os.path.join(curdatadir,"train")
      train_files = [os.path.join(tdatadir,fname) for fname in os.listdir(tdatadir) if fname.endswith(".tfrecord")]
      num_train_files = len(train_files)

      #Filter down to a random subset that will comprise this epoch
      def train_files_gen():
        train_files_shuffled = train_files.copy()
        while True:
          random.shuffle(train_files_shuffled)
          for filename in train_files_shuffled:
            trainlog("Yielding training file for dataset: " + filename)
            yield filename
      trainfilegenerator = train_files_gen()

      vdatadir = os.path.join(curdatadir,"val")
    break

# TRAIN! -----------------------------------------------------------------------------------

#Tensorflow doesn't offer a good way to save checkpoints more sparsely, so we have to manually do it.
last_longterm_checkpoint_save_time = datetime.datetime.now()

globalstep = None
while True:
  maybe_reload_training_data()

  trainlog("GC collect")
  gc.collect()

  trainlog("=========================================================================")
  trainlog("BEGINNING NEXT EPOCH")
  trainlog("=========================================================================")
  trainlog("Current time: " + str(datetime.datetime.now()))
  if globalstep is not None:
    trainlog("Global step: %d (%d samples)" % (globalstep, globalstep*batch_size))

  #SUB EPOCH LOOP -----------
  num_batches_per_subepoch = num_batches_per_epoch / sub_epochs
  for i in range(sub_epochs):
    if i != 0:
      maybe_reload_training_data()

    #Pick enough files to get the number of batches we want
    train_files_to_use = []
    batches_to_use_so_far = 0
    for filename in trainfilegenerator:
      jsonfilename = os.path.splitext(filename)[0] + ".json"
      with open(jsonfilename) as f:
        trainfileinfo = json.load(f)

      num_batches_this_file = trainfileinfo["num_batches"]
      if num_batches_this_file <= 0:
        continue

      if batches_to_use_so_far + num_batches_this_file > num_batches_per_subepoch:
        #If we're going over the desired amount, randomly skip the file with probability equal to the
        #proportion of batches over - this makes it so that in expectation, we have the desired number of batches
        if batches_to_use_so_far > 0 and random.random() >= (batches_to_use_so_far + num_batches_this_file - num_batches_per_subepoch) / num_batches_this_file:
          break

      train_files_to_use.append(filename)
      batches_to_use_so_far += num_batches_this_file

      #Sanity check - load a max of 100000 files.
      if batches_to_use_so_far >= num_batches_per_subepoch or len(train_files_to_use) > 100000:
        break

    #Train
    trainlog("Beginning training subepoch!")
    trainlog("Currently up to data row " + str(last_datainfo_row))
    estimator.train(
      (lambda mode, input_context=None: train_input_fn(train_files_to_use,num_train_files,batches_to_use_so_far,mode,input_context)),
      saving_listeners=[
        CheckpointSaverListenerFunction(save_history)
      ]
    )
    trainlog("Finished training subepoch!")

    if swa_sub_epoch_scale is not None:
      accumulate_swa(estimator)

  #END SUB EPOCH LOOP ------------

  globalstep = int(estimator.get_variable_value("global_step:0"))

  if not no_export:
    #Export a model for testing, unless somehow it already exists
    modelname = "%s-s%d-d%d" % (
      exportprefix,
      globalstep*batch_size,
      last_datainfo_row,
    )
    savepath = os.path.join(exportdir,modelname)
    savepathtmp = os.path.join(exportdir,modelname+".tmp")
    if os.path.exists(savepath):
      trainlog("NOT saving model, already exists at: " + savepath)
    else:
      os.mkdir(savepathtmp)
      trainlog("SAVING MODEL TO: " + savepath)
      if swa_sub_epoch_scale is not None:
        #Also save non-swa model
        saved_to = estimator.export_saved_model(
          savepathtmp,
          tf.estimator.export.build_raw_serving_input_receiver_fn(raw_input_feature_placeholders)
        )
        if saved_to != os.path.join(savepathtmp,"non_swa_saved_model"):
          os.rename(saved_to, os.path.join(savepathtmp,"non_swa_saved_model"))
        save_swa(savepathtmp)
      else:
        saved_to = estimator.export_saved_model(
          savepathtmp,
          tf.estimator.export.build_raw_serving_input_receiver_fn(raw_input_feature_placeholders)
        )
        if saved_to != os.path.join(savepathtmp,"saved_model"):
          os.rename(saved_to, os.path.join(savepathtmp,"saved_model"))

      dump_and_flush_json(trainhistory,os.path.join(savepathtmp,"trainhistory.json"))
      with open(os.path.join(savepathtmp,"model.config.json"),"w") as f:
        json.dump(model_config,f)
      with open(os.path.join(savepathtmp,"saved_model","model.config.json"),"w") as f:
        json.dump(model_config,f)
      with open(os.path.join(savepathtmp,"non_swa_saved_model","model.config.json"),"w") as f:
        json.dump(model_config,f)

      time.sleep(1)
      os.rename(savepathtmp,savepath)

  #Validate
  trainlog("Beginning validation after epoch!")
  val_files = []
  if os.path.exists(vdatadir):
    val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir) if fname.endswith(".tfrecord")]
  if len(val_files) == 0:
    trainlog("No validation files, skipping validation step")
  else:
    estimator.evaluate(
      (lambda: val_input_fn(vdatadir))
    )

  time.sleep(1)

  now = datetime.datetime.now()
  if now - last_longterm_checkpoint_save_time >= datetime.timedelta(hours=3):
    last_longterm_checkpoint_save_time = now
    ckpt_path = estimator.latest_checkpoint()
    #Tensorflow checkpoints have multiple pieces
    for ckpt_part in glob.glob(ckpt_path + "*"):
      print("Copying checkpoint longterm: " + ckpt_part)
      shutil.copy(ckpt_part, longterm_checkpoints_dir)
