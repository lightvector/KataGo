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
import copy

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
parser.add_argument('-samples-per-epoch', help='Number of data samples to consider as one epoch', type=int, required=False)
parser.add_argument('-multi-gpus', help='Use multiple gpus, comma-separated device ids', required=False)
parser.add_argument('-gpu-memory-frac', help='Fraction of gpu memory to use', type=float, required=True)
parser.add_argument('-model-kind', help='String name for what model to use', required=True)
parser.add_argument('-lr-scale', help='LR multiplier on the hardcoded schedule', type=float, required=False)
parser.add_argument('-lr-scale-before-export', help='LR multiplier on the hardcoded schedule just before export', type=float, required=False)
parser.add_argument('-lr-scale-before-export-epochs', help='Number of epochs for -lr-scale-before-export', type=int, required=False)
parser.add_argument('-gnorm-clip-scale', help='Multiplier on gradient clipping threshold', type=float, required=False)
parser.add_argument('-sub-epochs', help='Reload training data up to this many times per epoch', type=int, required=True)
parser.add_argument('-epochs-per-export', help='Export model once every this many epochs', type=int, required=False)
parser.add_argument('-export-prob', help='Export model with this probablity', type=float, required=False)
parser.add_argument('-max-epochs-this-instance', help='Terminate training after this many more epochs', type=int, required=False)
parser.add_argument('-sleep-seconds-per-epoch', help='Sleep this long between epochs', type=int, required=False)
parser.add_argument('-swa-sub-epoch-scale', help='Number of sub-epochs to average in expectation together for SWA', type=float, required=False)
parser.add_argument('-max-train-bucket-per-new-data', help='When data added, add this many train rows per data row to bucket', type=float, required=False)
parser.add_argument('-max-train-bucket-size', help='Approx total number of train rows allowed if data stops', type=float, required=False)
parser.add_argument('-max-train-steps-since-last-reload', help='Approx total of training allowed if shuffling stops', type=float, required=False)
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
lr_scale_before_export = args["lr_scale_before_export"]
lr_scale_before_export_epochs = args["lr_scale_before_export_epochs"]
gnorm_clip_scale = args["gnorm_clip_scale"]
sub_epochs = args["sub_epochs"]
epochs_per_export = args["epochs_per_export"]
export_prob = args["export_prob"]
max_epochs_this_instance = args["max_epochs_this_instance"]
sleep_seconds_per_epoch = args["sleep_seconds_per_epoch"]
swa_sub_epoch_scale = args["swa_sub_epoch_scale"]
max_train_bucket_per_new_data = args["max_train_bucket_per_new_data"]
max_train_bucket_size = args["max_train_bucket_size"]
max_train_steps_since_last_reload = args["max_train_steps_since_last_reload"]
verbose = args["verbose"]
no_export = args["no_export"]
logfilemode = "a"

if samples_per_epoch is None:
  samples_per_epoch = 1000000

if max_train_bucket_size is None:
  max_train_bucket_size = 1.0e30

if lr_scale_before_export is None:
  lr_scale_before_export = lr_scale

if lr_scale_before_export_epochs is None:
  lr_scale_before_export_epochs = 1

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

if epochs_per_export is None:
  epochs_per_export = 1

trainlog(str(sys.argv))

if os.path.exists(os.path.join(traindir,"model.config.json")):
  trainlog("Loading existing model config at %s" % os.path.join(traindir,"model.config.json"))
  with open(os.path.join(traindir,"model.config.json"),"r") as f:
    model_config = json.load(f)
else:
  model_config = modelconfigs.config_of_name[model_kind]
  trainlog("Initializing with new model config")
  with open(os.path.join(traindir,"model.config.json"),"w") as f:
    json.dump(model_config,f)

trainlog(str(model_config))

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
# Avoid loading initial weights, just ignore them, if we've already started training and we have weights.
# We detect this by detecting the TF estimator "checkpoint" index file.
initial_weights_already_loaded = os.path.exists(os.path.join(traindir,"checkpoint"))

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

  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU':0})) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(swa_assign_op, assignments)
    if not os.path.exists(savedir):
      os.mkdir(savedir)
    os.mkdir(os.path.join(savedir,"saved_model"))
    os.mkdir(os.path.join(savedir,"saved_model","variables"))
    swa_saver.save(sess,os.path.join(savedir,"saved_model","variables","variables"), write_meta_graph=True, write_state=False)


class CustomLoggingHook(tf.estimator.LoggingTensorHook):

  def __init__(self, *args, **kwargs):
    self.handle_logging_values = kwargs.pop('handle_logging_values')
    super().__init__(*args, **kwargs)

  def after_run(self, run_context, run_values):
    if run_values.results is not None:
      self.handle_logging_values(run_values.results)
    super().after_run(run_context, run_values)

num_epochs_this_instance = 0
global_latest_extra_stats = {}
def update_global_latest_extra_stats(results):
  global global_latest_extra_stats
  for key in results:
    global_latest_extra_stats[key] = results[key].item()

def model_fn(features,labels,mode,params):
  global num_epochs_this_instance
  global printed_model_yet
  global initial_weights_already_loaded

  print_model = not printed_model_yet

  lr_scale_to_use = lr_scale
  if (num_epochs_this_instance + lr_scale_before_export_epochs) % epochs_per_export <= num_epochs_this_instance % epochs_per_export:
    lr_scale_to_use = lr_scale_before_export

  built = ModelUtils.build_model_from_tfrecords_features(features,mode,print_model,trainlog,model_config,pos_len,batch_size,lr_scale_to_use,gnorm_clip_scale,num_gpus_used)

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
    if model.version >= 9:
      eval_metric_ops["evstloss"] = tf.compat.v1.metrics.mean(target_vars.shortterm_value_error_loss_unreduced, weights=target_vars.target_weight_used)
      eval_metric_ops["esstloss"] = tf.compat.v1.metrics.mean(target_vars.shortterm_score_error_loss_unreduced, weights=target_vars.target_weight_used)
    if model.version >= 10:
      eval_metric_ops["tdsloss"] = tf.compat.v1.metrics.mean(target_vars.td_score_loss_unreduced, weights=target_vars.target_weight_used)

    return tf.estimator.EstimatorSpec(
      mode,
      loss=target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32),
      eval_metric_ops=eval_metric_ops
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
      with tf.compat.v1.variable_scope(name):
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
    if model.version >= 9:
      (evstloss,evstloss_op) = moving_mean("evstloss",target_vars.shortterm_value_error_loss_unreduced, weights=target_vars.target_weight_used)
      (esstloss,esstloss_op) = moving_mean("esstloss",target_vars.shortterm_score_error_loss_unreduced, weights=target_vars.target_weight_used)
      # (evstm,evstm_op) = moving_mean("evstm",metrics.shortterm_value_error_mean_unreduced, weights=target_vars.target_weight_used)
      # (evstv,evstv_op) = moving_mean("evstv",metrics.shortterm_value_error_var_unreduced, weights=target_vars.target_weight_used)
      # (esstm,esstm_op) = moving_mean("esstm",metrics.shortterm_score_error_mean_unreduced, weights=target_vars.target_weight_used)
      # (esstv,esstv_op) = moving_mean("esstv",metrics.shortterm_score_error_var_unreduced, weights=target_vars.target_weight_used)
    if model.version >= 10:
      (tdsloss,tdsloss_op) = moving_mean("tdsloss",target_vars.td_score_loss_unreduced, weights=target_vars.target_weight_used)
    (pacc1,pacc1_op) = moving_mean("pacc1",metrics.accuracy1_unreduced, weights=target_vars.target_weight_used)
    (ptentr,ptentr_op) = moving_mean("ptentr",metrics.policy_target_entropy_unreduced, weights=target_vars.target_weight_used)
    #NOTE: These two are going to be smaller if using more GPUs since it's the gradient norm as measured on the instance batch
    #rather than the global batch.
    #Also, somewhat awkwardly, we say the weight is 1.0 rather than 1.0/num_gpus_used because tensorflow seems to have "meany"
    #behavior where it updates sumw via the mean of the two separate updates of the gpus rather than the sum.
    (gnorm,gnorm_op) = moving_mean("gnorm",metrics.gnorm, weights=1.0)
    (exgnorm,exgnorm_op) = moving_mean("excessgnorm",metrics.excess_gnorm, weights=1.0)
    (wmean,wmean_op) = tf.compat.v1.metrics.mean(target_vars.weight_sum)

    # print_op = tf.print(
    #   metrics.gnorm,
    #   target_vars.weight_sum,
    #   target_vars.opt_loss,
    #   metrics.tmp,
    #   foo[0],
    #   output_stream=sys.stdout
    # )

    print_train_loss_every_batches = 100

    logvars = {
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
      "exgnorm": exgnorm
    }
    if model.version >= 9:
      logvars["evstloss"] = evstloss
      logvars["esstloss"] = esstloss
      # logvars["evstm"] = evstm
      # logvars["evstv"] = evstv
      # logvars["esstm"] = esstm
      # logvars["esstv"] = esstv
    if model.version >= 10:
      logvars["tdsloss"] = tdsloss

    logging_hook = CustomLoggingHook(logvars, every_n_iter=print_train_loss_every_batches, handle_logging_values=update_global_latest_extra_stats)

    printed_model_yet = True

    sys.stdout.flush()
    sys.stderr.flush()

    initial_weights_dir = os.path.join(traindir,"initial_weights")
    if os.path.exists(initial_weights_dir) and not initial_weights_already_loaded:
      print("Initial weights dir found at: " + initial_weights_dir)
      checkpoint_path = None
      for initial_weights_file in os.listdir(initial_weights_dir):
        if initial_weights_file.startswith("model") and initial_weights_file.endswith(".index"):
          checkpoint_path = os.path.join(initial_weights_dir, initial_weights_file[0:len(initial_weights_file)-len(".index")])
          break
      if checkpoint_path is not None:
        print("Initial weights checkpoint to use found at: " + checkpoint_path)
        vars_in_checkpoint = tf.contrib.framework.list_variables(checkpoint_path)
        varname_in_checkpoint = {}
        print("Checkpoint contains:")
        for varandshape in vars_in_checkpoint:
          print(varandshape)
          varname_in_checkpoint[varandshape[0]] = True

        print("Modifying graph to load weights from checkpoint upon init...")
        sys.stdout.flush()
        sys.stderr.flush()

        variables_to_restore = tf.compat.v1.global_variables()
        assignment_mapping = {}
        for v in variables_to_restore:
          name = v.name.split(":")[0] # drop the ":0" at the end of each var
          if name in varname_in_checkpoint:
            assignment_mapping[name] = v
          elif ("swa_model/"+name) in varname_in_checkpoint:
            assignment_mapping[("swa_model/"+name)] = v

        tf.compat.v1.train.init_from_checkpoint(checkpoint_path, assignment_mapping)
        initial_weights_already_loaded = True

    ops = [
      train_step,
      p0loss_op,p1loss_op,vloss_op,tdvloss_op,smloss_op,leadloss_op,vtimeloss_op,sbpdfloss_op,sbcdfloss_op,
      oloss_op,sloss_op,fploss_op,skloss_op,rsdloss_op,rloss_op,rscloss_op,pacc1_op,ptentr_op,wmean_op,
      gnorm_op,exgnorm_op
    ]
    if model.version >= 9:
      ops.append(evstloss_op)
      ops.append(esstloss_op)
      # ops.append(evstm_op)
      # ops.append(evstv_op)
      # ops.append(esstm_op)
      # ops.append(esstv_op)
    if model.version >= 10:
      ops.append(tdsloss_op)

    return tf.estimator.EstimatorSpec(
      mode,
      loss=(target_vars.opt_loss / tf.constant(batch_size,dtype=tf.float32)),
      train_op=tf.group(*ops),
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
  session_config = tf.compat.v1.ConfigProto()
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
  session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac
  multigpu_strategy = tf.distribute.MirroredStrategy(
    devices=multi_gpu_device_ids,
    cross_device_ops=tf.distribute.ReductionToOneDevice(
      reduce_to_device="/device:CPU:0"
    )
  )
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


# DATA RELOADING GENERATOR AND TRAINHISTORY ------------------------------------------------------------

# Some globals
last_curdatadir = None
last_datainfo_row = 0
trainfilegenerator = None
num_train_files = 0
vdatadir = None

trainhistory = {
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

if max_train_bucket_per_new_data is not None and "train_bucket_level" not in trainhistory:
  trainhistory["train_bucket_level"] = samples_per_epoch
if "train_steps_since_last_reload" not in trainhistory:
  trainhistory["train_steps_since_last_reload"] = 0
if "export_cycle_counter" not in trainhistory:
  trainhistory["export_cycle_counter"] = 0

def save_history(global_step_value):
  global trainhistory
  if global_step_value is not None:
    trainhistory["history"].append(("nsamp",int(global_step_value * batch_size)))
    trainhistory["train_step"] = int(global_step_value * batch_size)
  trainhistory["total_num_data_rows"] = last_datainfo_row
  trainhistory["extra_stats"] = copy.deepcopy(global_latest_extra_stats)
  savepath = os.path.join(traindir,"trainhistory.json")
  savepathtmp = os.path.join(traindir,"trainhistory.json.tmp")
  dump_and_flush_json(trainhistory,savepathtmp)
  os.replace(savepathtmp,savepath)
  trainlog("Wrote " + savepath)

def maybe_reload_training_data():
  global last_curdatadir
  global last_datainfo_row
  global trainfilegenerator
  global trainhistory
  global num_train_files
  global vdatadir

  while True:
    curdatadir = os.path.realpath(datadir)

    # Different directory - new shuffle
    if curdatadir != last_curdatadir:
      if not os.path.exists(curdatadir):
        trainlog("Shuffled data path does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % curdatadir)
        time.sleep(30)
        continue

      trainjsonpath = os.path.join(curdatadir,"train.json")
      if not os.path.exists(trainjsonpath):
        trainlog("Shuffled data train.json file does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % trainjsonpath)
        time.sleep(30)
        continue

      trainlog("Updated training data: " + curdatadir)
      last_curdatadir = curdatadir

      with open(trainjsonpath) as f:
        datainfo = json.load(f)
        last_datainfo_row = datainfo["range"][1]

      if max_train_bucket_per_new_data is not None:
        if "train_bucket_level_at_row" not in trainhistory:
          trainhistory["train_bucket_level_at_row"] = last_datainfo_row
        if last_datainfo_row > trainhistory["train_bucket_level_at_row"]:
          new_row_count = last_datainfo_row - trainhistory["train_bucket_level_at_row"]
          trainlog("Advancing trainbucket row %.0f to %.0f, %.0f new rows" % (
            trainhistory["train_bucket_level_at_row"], last_datainfo_row, new_row_count
          ))
          trainhistory["train_bucket_level_at_row"] = last_datainfo_row
          trainlog("Fill per data %.3f, Max bucket size %.0f" % (max_train_bucket_per_new_data, max_train_bucket_size))
          trainlog("Old rows in bucket: %.0f" % trainhistory["train_bucket_level"])
          trainhistory["train_bucket_level"] += new_row_count * max_train_bucket_per_new_data
          cap = max(max_train_bucket_size, samples_per_epoch)
          if trainhistory["train_bucket_level"] > cap:
            trainhistory["train_bucket_level"] = cap
          trainlog("New rows in bucket: %.0f" % trainhistory["train_bucket_level"])

      trainlog("Train steps since last reload: %.0f -> 0" % trainhistory["train_steps_since_last_reload"])
      trainhistory["train_steps_since_last_reload"] = 0

      # Remove legacy value from this dictionary
      if "files" in trainhistory:
        del trainhistory["files"]
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

    # Same directory as before, no new shuffle
    else:
      if max_train_steps_since_last_reload is not None:
        if trainhistory["train_steps_since_last_reload"] + 0.99 * samples_per_epoch/sub_epochs > max_train_steps_since_last_reload:
          trainlog(
            "Too many train steps since last reload, waiting 5m and retrying (current %f)" %
            trainhistory["train_steps_since_last_reload"]
          )
          time.sleep(300)
          continue

    break

# TRAIN! -----------------------------------------------------------------------------------

#Tensorflow doesn't offer a good way to save checkpoints more sparsely, so we have to manually do it.
last_longterm_checkpoint_save_time = datetime.datetime.now()

globalstep = None
try:
  globalstep = int(estimator.get_variable_value("global_step:0"))
except ValueError:
  pass # Will happen on the start of a new model, first iteration

while True:
  maybe_reload_training_data()
  save_history(globalstep)
  trainlog("GC collect")
  gc.collect()

  trainlog("=========================================================================")
  trainlog("BEGINNING NEXT EPOCH " + str(num_epochs_this_instance))
  trainlog("=========================================================================")
  trainlog("Current time: " + str(datetime.datetime.now()))
  if globalstep is not None:
    trainlog("Global step: %d (%d samples)" % (globalstep, globalstep*batch_size))
    trainlog("Currently up to data row " + str(last_datainfo_row))

    if max_train_bucket_per_new_data is not None:
      if trainhistory["train_bucket_level"] > 0.99 * samples_per_epoch:
        trainlog("Consuming %.0f rows from train bucket (%.0f -> %.0f)" % (
          samples_per_epoch, trainhistory["train_bucket_level"], trainhistory["train_bucket_level"]-samples_per_epoch
        ))
        trainhistory["train_bucket_level"] -= samples_per_epoch
      else:
        trainlog(
          "Exceeding train bucket, not enough new data rows, waiting 5m and retrying (current level %f)" %
          trainhistory["train_bucket_level"]
        )
        time.sleep(300)
        continue

  #SUB EPOCH LOOP -----------
  num_batches_per_subepoch = num_batches_per_epoch / sub_epochs
  for i in range(sub_epochs):
    if i != 0:
      maybe_reload_training_data()
      save_history(globalstep)

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
    trainhistory["train_steps_since_last_reload"] += num_batches_per_subepoch * batch_size

    if swa_sub_epoch_scale is not None:
      accumulate_swa(estimator)

  #END SUB EPOCH LOOP ------------
  num_epochs_this_instance += 1
  trainhistory["export_cycle_counter"] += 1
  trainlog("Export cycle counter = " + str(trainhistory["export_cycle_counter"]))

  is_time_to_export = False
  if trainhistory["export_cycle_counter"] >= epochs_per_export:
    if no_export:
      trainhistory["export_cycle_counter"] = epochs_per_export
    else:
      trainhistory["export_cycle_counter"] = 0
      is_time_to_export = True

  globalstep = int(estimator.get_variable_value("global_step:0"))

  skip_export_this_time = False
  if export_prob is not None:
    if random.random() > export_prob:
      skip_export_this_time = True
      trainlog("Skipping export model this time")

  if not no_export and is_time_to_export and not skip_export_this_time:
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

  if max_epochs_this_instance is not None and max_epochs_this_instance >= 0 and num_epochs_this_instance >= max_epochs_this_instance:
    print("Done")
    break

  if sleep_seconds_per_epoch is None:
    time.sleep(1)
  else:
    time.sleep(sleep_seconds_per_epoch)

  now = datetime.datetime.now()
  if now - last_longterm_checkpoint_save_time >= datetime.timedelta(hours=12):
    last_longterm_checkpoint_save_time = now
    ckpt_path = estimator.latest_checkpoint()
    #Tensorflow checkpoints have multiple pieces
    for ckpt_part in glob.glob(ckpt_path + "*"):
      print("Copying checkpoint longterm: " + ckpt_part)
      shutil.copy(ckpt_part, longterm_checkpoints_dir)
