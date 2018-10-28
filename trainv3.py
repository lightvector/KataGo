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
  binchw = tf.reshape(tf.bitwise.bitwise_and(tf.expand_dims(binchwp,axis=3),bitmasks),[-1,20,((pos_len*pos_len+7)//8)*8])
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

raw_input_features = {
  "binchwp": tf.FixedLenFeature([],tf.string),
  "finc": tf.FixedLenFeature([batch_size*15],tf.float32),
  "ptncm": tf.FixedLenFeature([batch_size*3*(pos_len*pos_len+1)],tf.float32),
  "ftnc": tf.FixedLenFeature([batch_size*44],tf.float32),
  "vtnchw": tf.FixedLenFeature([batch_size*1*pos_len*pos_len],tf.float32)
}
def parse_input(serialized_example):
  example = tf.parse_single_example(serialized_example,raw_input_features)
  binchwp = tf.decode_raw(example["binchwp"],tf.uint8)
  finc = example["finc"]
  ptncm = example["ptncm"]
  ftnc = example["ftnc"]
  vtnchw = example["vtnchw"]
  return {
    "binchwp": tf.reshape(binchwp,[batch_size,20,(pos_len*pos_len+7)//8]),
    "finc": tf.reshape(finc,[batch_size,15]),
    "ptncm": tf.reshape(ptncm,[batch_size,3,pos_len*pos_len+1]),
    "ftnc": tf.reshape(ftnc,[batch_size,44]),
    "vtnchw": tf.reshape(vtnchw,[batch_size,1,pos_len,pos_len])
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
  # hooks=[evaluator]
  hooks=[]
)

# # Training ------------------------------------------------------------

# saver = tf.train.Saver(
#   max_to_keep = 10000,
#   save_relative_paths = True,
# )

# #Some tensorflow options
# tfconfig = tf.ConfigProto(log_device_placement=False)
# #tfconfig.gpu_options.allow_growth = True
# #tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
# with tf.Session(config=tfconfig) as session:
#   if restart_file is not None:
#     saver.restore(session, restart_file)
#   else:
#     session.run(tf.global_variables_initializer())

#   sys.stdout.flush()
#   sys.stderr.flush()

#   trainlog("Began session")
#   trainlog("Training on " + str(num_h5_train_rows) + " rows, validating on " + str(int(num_h5_val_rows * validation_prop)) + "/" + str(num_h5_val_rows) + " rows")
#   trainlog("Epoch size = " + str(num_samples_per_epoch))
#   trainlog("h5_chunk_size = " + str(h5_chunk_size))
#   trainlog("Batch size = " + str(batch_size))
#   trainlog("L2 coeff value = " + str(l2_coeff_value))
#   trainlog("use_ranks = " + str(use_ranks))
#   trainlog("predict_pass = " + str(predict_pass))

#   sys.stdout.flush()
#   sys.stderr.flush()

#   input_start = 0
#   input_len = model.input_shape[0] * model.input_shape[1]
#   policy_target_start = input_start + input_len
#   policy_target_len = model.policy_target_shape[0]
#   value_target_start = policy_target_start + policy_target_len
#   value_target_len = 1
#   target_weights_start = value_target_start + value_target_len
#   target_weights_len = 1
#   rank_start = target_weights_start + target_weights_len
#   rank_len = model.rank_shape[0]
#   side_start = rank_start + rank_len
#   side_len = 1
#   turn_number_start = side_start + side_len
#   turn_number_len = 2
#   recent_captures_start = turn_number_start + turn_number_len
#   recent_captures_len = model.max_board_size * model.max_board_size
#   next_moves_start = recent_captures_start + recent_captures_len
#   next_moves_len = 12
#   sgf_hash_start = next_moves_start + next_moves_len
#   sgf_hash_len = 8
#   include_history_start = sgf_hash_start + sgf_hash_len
#   include_history_len = 5
#   total_row_len = include_history_start + include_history_len

#   def run(fetches, rows, training, symmetries, pslr=0.0):
#     assert(len(model.input_shape) == 2)
#     assert(len(model.policy_target_shape) == 1)
#     assert(len(model.value_target_shape) == 0)
#     assert(len(model.target_weights_shape) == 0)
#     assert(len(model.rank_shape) == 1)

#     if not isinstance(rows, np.ndarray):
#       rows = np.array(rows)

#     assert(rows.shape[1] == total_row_len)

#     row_inputs = rows[:,0:input_len].reshape([-1] + model.input_shape)
#     row_policy_targets = rows[:,policy_target_start:policy_target_start+policy_target_len]
#     row_value_target = rows[:,value_target_start]
#     row_target_weights = rows[:,target_weights_start]
#     if use_ranks:
#       row_ranks = rows[:,rank_start:rank_start+rank_len]
#     row_include_history = rows[:,include_history_start:include_history_start+include_history_len]

#     if use_ranks:
#       return session.run(fetches, feed_dict={
#         model.inputs: row_inputs,
#         target_vars.policy_targets: row_policy_targets,
#         target_vars.value_target: row_value_target,
#         target_vars.target_weights_from_data: row_target_weights,
#         model.ranks: row_ranks,
#         model.symmetries: symmetries,
#         model.include_history: row_include_history,
#         per_sample_learning_rate: pslr,
#         target_vars.l2_reg_coeff: l2_coeff_value,
#         model.is_training: training
#       })
#     else:
#       return session.run(fetches, feed_dict={
#         model.inputs: row_inputs,
#         target_vars.policy_targets: row_policy_targets,
#         target_vars.value_target: row_value_target,
#         target_vars.target_weights_from_data: row_target_weights,
#         model.symmetries: symmetries,
#         model.include_history: row_include_history,
#         per_sample_learning_rate: pslr,
#         target_vars.l2_reg_coeff: l2_coeff_value,
#         model.is_training: training
#       })

#   def np_array_str(arr,precision):
#     return np.array_str(arr, precision=precision, suppress_small = True, max_line_width = 200)
#   def merge_dicts(dicts,merge_list):
#     keys = dicts[0].keys()
#     return dict((key,merge_list([d[key] for d in dicts])) for key in keys)

#   def run_validation_in_batches(fetches):
#     #Run validation accuracy in batches to avoid out of memory error from processing one supergiant batch
#     validation_batch_size = 256
#     num_validation_batches = int(num_h5_val_rows * validation_prop + validation_batch_size-1)//validation_batch_size
#     results = []
#     for i in range(num_validation_batches):
#       rows = h5val[i*validation_batch_size : min((i+1)*validation_batch_size, num_h5_val_rows)]
#       result = run(fetches, rows, symmetries=[False,False,False], training=False)
#       results.append(result)
#     return results

#   tmetrics = {
#     "acc1": metrics.accuracy1,
#     "acc4": metrics.accuracy4,
#     "ploss": target_vars.policy_loss,
#     "vloss": target_vars.value_loss,
#     "rloss": target_vars.reg_loss,
#     "wsum": target_vars.weight_sum,
#   }

#   vmetrics = {
#     "acc1": metrics.accuracy1,
#     "acc4": metrics.accuracy4,
#     "ploss": target_vars.policy_loss,
#     "vloss": target_vars.value_loss,
#     "wsum": target_vars.weight_sum,
#   }

#   def train_stats_str(tmetrics_evaled):
#     return "tacc1 %5.2f%% tacc4 %5.2f%% tploss %f tvloss %f trloss %f" % (
#       tmetrics_evaled["acc1"] * 100 / tmetrics_evaled["wsum"],
#       tmetrics_evaled["acc4"] * 100 / tmetrics_evaled["wsum"],
#       tmetrics_evaled["ploss"] / tmetrics_evaled["wsum"],
#       tmetrics_evaled["vloss"] / tmetrics_evaled["wsum"],
#       tmetrics_evaled["rloss"] / tmetrics_evaled["wsum"],
#     )

#   def validation_stats_str(vmetrics_evaled):
#     return "vacc1 %5.2f%% vacc4 %5.2f%% vploss %f vvloss %f" % (
#       vmetrics_evaled["acc1"] * 100 / vmetrics_evaled["wsum"],
#       vmetrics_evaled["acc4"] * 100 / vmetrics_evaled["wsum"],
#       vmetrics_evaled["ploss"] / vmetrics_evaled["wsum"],
#       vmetrics_evaled["vloss"] / vmetrics_evaled["wsum"],
#   )

#   def time_str(elapsed):
#     return "time %.3f" % elapsed

#   def log_detail_stats(relupdates):
#     results = run_validation_in_batches([metrics.activated_prop_by_layer, metrics.mean_output_by_layer, metrics.stdev_output_by_layer])
#     [apbls,mobls,sobls] = list(map(list, zip(*results)))

#     apbl = merge_dicts(apbls, (lambda x: np.mean(x,axis=0)))
#     mobl = merge_dicts(mobls, (lambda x: np.mean(x,axis=0)))
#     sobl = merge_dicts(sobls, (lambda x: np.sqrt(np.mean(np.square(x),axis=0))))

#     for key in apbl:
#       detaillog("%s: activated_prop %s" % (key, np_array_str(apbl[key], precision=3)))
#       detaillog("%s: mean_output %s" % (key, np_array_str(mobl[key], precision=4)))
#       detaillog("%s: stdev_output %s" % (key, np_array_str(sobl[key], precision=4)))

#     (mw,nw) = session.run([metrics.mean_weights_by_var, metrics.norm_weights_by_var])

#     for key in mw:
#       detaillog("%s: mean weight %f" % (key, mw[key]))
#     for key in nw:
#       detaillog("%s: norm weight %f" % (key, nw[key]))

#     if relupdates is not None:
#       for key in relupdates:
#         detaillog("%s: relative update %f" % (key,relupdates[key]))

#   def make_batch_generator():
#     while(True):
#       chunk_perm = np.random.permutation(num_h5_train_rows // h5_chunk_size)
#       batch_perm = np.random.permutation(h5_chunk_size // batch_size)
#       for chunk_perm_idx in range(len(chunk_perm)):
#         chunk_start = chunk_perm[chunk_perm_idx] * h5_chunk_size
#         chunk_end = chunk_start + h5_chunk_size
#         chunk = np.array(h5train[chunk_start:chunk_end])
#         for batch_perm_idx in range(len(batch_perm)):
#           batch_start = batch_perm[batch_perm_idx] * batch_size
#           batch_end = batch_start + batch_size
#           yield chunk[batch_start:batch_end]
#         np.random.shuffle(batch_perm)

#   batch_generator = make_batch_generator()
#   def run_batches(num_batches):
#     tmetrics_results = []
#     relupdates = dict([(key,0.0) for key in relative_update_by_var])

#     for i in range(num_batches):
#       rows = next(batch_generator)

#       # input_len = model.input_shape[0] * model.input_shape[1]
#       # target_len = model.target_shape[0]
#       # row_inputs = rows[:,0:input_len].reshape([-1] + model.input_shape)
#       # row_targets = rows[:,input_len:input_len+target_len]
#       # row_target_weights = rows[:,input_len+target_len]
#       # for j in range(len(row_inputs)):
#       #   print("BOARD")
#       #   print((row_inputs[i,:,0] + row_inputs[i,:,1] + row_inputs[i,:,2]*2).reshape([19,19]))
#       #   print("MYLIB")
#       #   print((row_inputs[i,:,3] + row_inputs[i,:,4]*2 + row_inputs[i,:,5]*3).reshape([19,19]))
#       #   print("OPPLIB")
#       #   print((row_inputs[i,:,6] + row_inputs[i,:,7]*2 + row_inputs[i,:,8]*3).reshape([19,19]))
#       #   print("LAST")
#       #   print((row_inputs[i,:,9] + row_inputs[i,:,10]*2 + row_inputs[i,:,11]*3).reshape([19,19]))
#       #   print("KO")
#       #   print((row_inputs[i,:,12]).reshape([19,19]))
#       #   print("TARGET")
#       #   print(row_targets[i].reshape([19,19]))
#       #   print("WEIGHT")
#       #   print(row_target_weights[i])

#       # assert(False)

#       (tmetrics_result, brelupdates, _) = run(
#         fetches=[tmetrics, relative_update_by_var, train_step],
#         rows=rows,
#         training=True,
#         symmetries=[np.random.random() < 0.5, np.random.random() < 0.5, np.random.random() < 0.5],
#         pslr=lr.lr()
#       )

#       tmetrics_results.append(tmetrics_result)
#       for key in brelupdates:
#         relupdates[key] += brelupdates[key]

#       if i % (max(1,num_batches // 30)) == 0:
#         print(".", end='', flush=True)

#     tmetrics_evaled = merge_dicts(tmetrics_results,np.sum)
#     for key in relupdates:
#       relupdates[key] = relupdates[key] / num_batches
#     return (tmetrics_evaled,relupdates)

#   vmetrics_evaled = merge_dicts(run_validation_in_batches(vmetrics), np.sum)
#   vstr = validation_stats_str(vmetrics_evaled)

#   trainlog("Initial: %s" % (vstr))
#   log_detail_stats(relupdates=None)

#   start_time = time.perf_counter()

#   if start_epoch > 0:
#     lr.report_epoch_done(start_epoch-1)

#   for e in range(num_epochs):
#     epoch = start_epoch + e
#     print("Epoch %d" % (epoch), end='', flush=True)
#     (tmetrics_evaled,relupdates) = run_batches(num_batches_per_epoch)
#     vmetrics_evaled = merge_dicts(run_validation_in_batches(vmetrics), np.sum)
#     lr.report_epoch_done(epoch)
#     print("")

#     elapsed = time.perf_counter() - start_time + start_elapsed

#     tstr = train_stats_str(tmetrics_evaled)
#     vstr = validation_stats_str(vmetrics_evaled)
#     timestr = time_str(elapsed)

#     trainlogger.info("Epoch %d--------------------------------------------------" % (epoch))
#     detaillogger.info("Epoch %d--------------------------------------------------" % (epoch))

#     trainlog("%s %s lr %f %s" % (tstr,vstr,lr.lr(),timestr))
#     log_detail_stats(relupdates)

#     #Save model every 4 epochs
#     if epoch % 4 == 0 or epoch == num_epochs-1:
#       savepath = traindir + "/model" + str(epoch)
#       with open(savepath + ".config.json","w") as f:
#         json.dump(model_config,f)
#       saver.save(session, savepath)

#   vmetrics_evaled = merge_dicts(run_validation_in_batches(vmetrics), np.sum)
#   vstr = validation_stats_str(vmetrics_evaled)
#   trainlog("Final: %s" % (vstr))


# # Finish
# h5file.close()
# h5fid.close()


