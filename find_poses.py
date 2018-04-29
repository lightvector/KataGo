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
import pickle
import tensorflow as tf
import numpy as np

import data
from board import Board
from model import Model

#Command and args-------------------------------------------------------------------

description = """
Extract positions from Go games!
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-gamesh5', help='H5 files of preprocessed game data', required=True, action="append")
parser.add_argument('-output', help='', required=True)
parser.add_argument('-model-file', help='model file prefix to load', required=True)
parser.add_argument('-use-training-set', help='run on training set instead of test set', required=False, action="store_true")
parser.add_argument('-user-rank', help='', required=True)
parser.add_argument('-better-rank', help='', required=True)
parser.add_argument('-user-min-score', help='', required=True)
parser.add_argument('-user-max-score', help='', required=True)
parser.add_argument('-min-better-at-min', help='', required=True)
parser.add_argument('-min-better-at-max', help='', required=True)
args = vars(parser.parse_args())

gamesh5_files = args["gamesh5"]
output_file = args["output"]
model_file = args["model_file"]
use_training_set = args["use_training_set"]

user_rank = int(args["user_rank"])
better_rank = int(args["better_rank"])
user_min_score = float(args["user_min_score"])
user_max_score = float(args["user_max_score"])
min_better_at_min = float(args["min_better_at_min"])
min_better_at_max = float(args["min_better_at_max"])

def log(s):
  print(s,flush=True)

# Model ----------------------------------------------------------------
print("Building model", flush=True)
model = Model(use_ranks=True)

policy_probs_output = tf.nn.softmax(model.policy_output)

total_parameters = 0
for variable in tf.trainable_variables():
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  total_parameters += variable_parameters
  log("Model variable %s, %d parameters" % (variable.name,variable_parameters))

log("Built model, %d total parameters" % total_parameters)

#H5 file format
assert(len(model.input_shape) == 2)
assert(len(model.target_shape) == 1)
assert(len(model.target_weights_shape) == 0)
assert(len(model.rank_shape) == 1)
input_start = 0
input_len = model.input_shape[0] * model.input_shape[1]
target_start = input_start + input_len
target_len = model.target_shape[0]
target_weights_start = target_start + target_len
target_weights_len = 1
rank_start = target_weights_start + target_weights_len
rank_len = model.rank_shape[0]
side_start = rank_start + rank_len
side_len = 1
turn_number_start = side_start + side_len
turn_number_len = 2
recent_captures_start = turn_number_start + turn_number_len
recent_captures_len = model.max_board_size * model.max_board_size


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

  log("Began session, loaded model")

  num_processed = [0]
  num_used = [0]

  buf_pla = []
  buf_positions = []
  buf_recent_capture_locs = []
  buf_last_moves = []
  buf_correct_net_moves = []
  buf_correct_lz_moves = []
  buf_big_credit_moves = []
  buf_medium_credit_moves = []
  buf_small_credit_moves = []

  def run(inputs,ranks):
    fetches = policy_probs_output
    policy_probs = session.run(fetches, feed_dict={
      model.inputs: inputs,
      model.ranks: ranks,
      model.symmetries: [False,False,False],
      model.is_training: False
    })

    return np.array(policy_probs)

  def run_in_batches(h5val,f):
    num_h5_val_rows = h5val.shape[0]
    #Run validation accuracy in batches to avoid out of memory error from processing one supergiant batch
    validation_batch_size = 64
    num_validation_batches = num_h5_val_rows//validation_batch_size

    pro_ranks_input = np.zeros([rank_len])
    pro_ranks_input[0] = 1.0
    pro_ranks_input = [pro_ranks_input for i in range(validation_batch_size)]

    for i in range(num_validation_batches):
      print("Batch: " + str(i) + "/" + str(num_validation_batches) + " Used " + str(num_used[0]) + "/" + str(num_processed[0]), flush=True)
      rows = h5val[i*validation_batch_size : min((i+1)*validation_batch_size, num_h5_val_rows)]

      if not isinstance(rows, np.ndarray):
        rows = np.array(rows)

      row_inputs = rows[:,0:input_len].reshape([-1] + model.input_shape)
      row_targets = rows[:,target_start:target_start+target_len]
      row_target_weights = rows[:,target_weights_start]
      row_ranks = rows[:,rank_start:rank_start+rank_len]
      pro_probs = run(row_inputs, pro_ranks_input)

      for i in range(len(rows)):
        f(row_inputs[i],row_targets[i],row_ranks[i],pro_probs[i],rows[i])


  def print_board(inputs,pla,recent_captures,a,b):
    pla,opp = (("X","O") if pla == 1 else ("0","X"))
    print("----------------------------------------------------------------------")
    print("TO MOVE: " + pla)
    for y in range(19):
      for x in range(19):
        loc = y*19+x
        if inputs[loc,18] == 1.0:
          print("3",end="")
        elif inputs[loc,19] == 1.0:
          print("2",end="")
        elif inputs[loc,20] == 1.0:
          print("1",end="")
        elif inputs[loc,1] == 1.0:
          print(pla,end="")
        elif inputs[loc,2] == 1.0:
          print(opp,end="")
        elif loc == a:
          print("A",end="")
        elif loc == b:
          print("B",end="")
        elif recent_captures[loc] > 0 and recent_captures[loc] <= 3:
          print("*",end="")
        elif x in [3,9,15] and y in [3,9,15]:
          print(",",end="")
        else:
          print(".",end="")

        if inputs[loc,18] == 1.0:
          print(" ",end="")
        elif inputs[loc,19] == 1.0:
          print(" ",end="")
        elif inputs[loc,20] == 1.0:
          print(" ",end="")
        else:
          print(" ",end="")

      print("")
    print("",flush=True)


  def basic_filter(inputs,turn_number,turns_total):
    if (turn_number > 15 and #Exclude super-early opening
        turn_number < turns_total - 20 and #Exclude dame-filling and ultra-micro-endgame
        np.max(inputs[:,18]) > 0 and #History features present
        np.max(inputs[:,19]) > 0 and
        np.max(inputs[:,20]) > 0):
      return True
    return False

  strong_ranks = [0,8,9,35,62,63] #GoGoD, KGS 8d-9d, OGS Fox 9d, OGS 8d-9d
  def strong_game_filter(rank_one_hot_idx,pro_probs,real_move):
    if (real_move == np.argmax(pro_probs) and
        rank_one_hot_idx in strong_ranks and
        pro_probs[real_move] > 0.65):
      return True
    return False

  def nth_largest(arr,n):
    if len(arr) <= n:
      return None
    return -(np.partition((-arr).flatten(), n)[n])


  def get_expected_score(pro_probs,player_probs):
    probratios = pro_probs / np.max(pro_probs)
    scores = np.zeros(probratios.shape)
    scores[probratios >= 0] = 0.00
    scores[probratios >= 0.05] = 0.15
    scores[probratios >= 0.15] = 0.35
    scores[probratios >= 0.30] = 0.60
    scores[probratios >= 0.999999] = 1.00
    return np.sum(scores * player_probs)

  # def excess_expected_surprisal(observer_probs,player_probs):
  #   return np.sum(player_probs * (np.log(observer_probs+1e-30) - np.log(player_probs+1e-30)))

  def get_probs(inputs,rank):
    ranks = np.zeros([rank_len])
    ranks[rank] = 1.0
    probs = run([inputs],[ranks])
    probs = probs[0]
    return probs


  def write_position(inputs,target,ranks,pro_probs,row,user_expected_score,better_expected_score,real_move):
    pla = int(row[side_start]+1)
    recent_captures = row[recent_captures_start:recent_captures_start+recent_captures_len]
    max_pro_prob = np.max(pro_probs)

    position = np.zeros([19*19],dtype=np.int8)
    recent_capture_locs = []
    last_moves = np.zeros([3],dtype=np.int16)
    correct_net_moves = []
    correct_lz_moves = []
    big_credit_moves = []
    medium_credit_moves = []
    small_credit_moves = []

    correct_net_moves.append(real_move)
    for y in range(19):
      for x in range(19):
        loc = y*19+x
        if inputs[loc,1] == 1.0:
          position[loc] = 1
        elif inputs[loc,2] == 1.0:
          position[loc] = 2

        if recent_captures[loc] > 0 and recent_captures[loc] <= 3:
          recent_capture_locs.append(loc)

        if inputs[loc,18] == 1.0:
          last_moves[2] = loc
        elif inputs[loc,19] == 1.0:
          last_moves[1] = loc
        elif inputs[loc,20] == 1.0:
          last_moves[0] = loc

        if pro_probs[loc] >= max_pro_prob * 0.30:
          big_credit_moves.append(loc)
        elif pro_probs[loc] >= max_pro_prob * 0.15:
          medium_credit_moves.append(loc)
        elif pro_probs[loc] >= max_pro_prob * 0.05:
          small_credit_moves.append(loc)

    buf_pla.append(pla)
    buf_positions.append(position)
    buf_recent_capture_locs.append(recent_capture_locs)
    buf_last_moves.append(last_moves)
    buf_correct_net_moves.append(correct_net_moves)
    buf_correct_lz_moves.append(correct_lz_moves)
    buf_big_credit_moves.append(big_credit_moves)
    buf_medium_credit_moves.append(medium_credit_moves)
    buf_small_credit_moves.append(small_credit_moves)

    # print_board(inputs,pla,recent_captures,a=real_move,b=None)
    # print("Used " + str(num_used[0]) + "/" + str(num_processed[0]) +
    #       " pro prob " + str(pro_probs[real_move]) +
    #       " user expected score " + str(user_expected_score)
    #       " better expected score " + str(better_expected_score)
    # )

    # for beginner_rank in (list(range(10,36)) + [0]):
    #   beginner_probs = get_probs(inputs,beginner_rank)
    #   beginner_move = np.argmax(beginner_probs)
    #   beginner_second_move = np.argsort(-beginner_probs)[1]
    #   print("rank %2d: (%2d,%2d), %4.1f%%, (%2d,%2d) %4.1f%%, expected %6.3f" % (
    #     beginner_rank,
    #     beginner_move%19, beginner_move//19,
    #     beginner_probs[beginner_move]*100,
    #     beginner_second_move%19, beginner_second_move//19,
    #     beginner_probs[beginner_second_move]*100,
    #           get_expected_score(pro_probs,beginner_probs),
    #   ),flush=True)


  def process_position(inputs,target,ranks,pro_probs,row):
    num_processed[0] += 1
    turn_number = row[turn_number_start]
    turns_total = row[turn_number_start+1]
    if basic_filter(inputs,turn_number,turns_total):
      rank_one_hot_idx = np.argmax(ranks)
      real_move = np.argmax(target)
      if strong_game_filter(rank_one_hot_idx,pro_probs,real_move):
        user_expected_score = get_expected_score(pro_probs,get_probs(inputs,user_rank))
        better_expected_score = get_expected_score(pro_probs,get_probs(inputs,better_rank))
        pro_expected_score = get_expected_score(pro_probs,pro_probs)

        prop = (user_expected_score - user_min_score) / (user_max_score - user_min_score)
        min_better_score = min_better_at_min + prop * (min_better_at_max - min_better_at_min)
        # if True:
        if (user_expected_score >= user_min_score and
            user_expected_score <= user_max_score and
            better_expected_score >= min_better_score and
            pro_expected_score >= min_better_score):
          num_used[0] += 1
          write_position(inputs,target,ranks,pro_probs,row,user_expected_score,better_expected_score,real_move)


  def process_file(gamesh5_file):
    # Open H5 file---------------------------------------------------------
    print("Opening H5 file: " + gamesh5_file)
    sys.stdout.flush()
    sys.stderr.flush()

    h5_propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    h5_settings = list(h5_propfaid.get_cache())
    assert(h5_settings[2] == 1048576) #Default h5 cache size is 1 MB
    h5_settings[2] *= 128 #Make it 128 MB
    print("Adjusting H5 cache settings to: " + str(h5_settings))
    h5_propfaid.set_cache(*h5_settings)

    h5fid = h5py.h5f.open(str.encode(str(gamesh5_file)), fapl=h5_propfaid)
    h5file = h5py.File(h5fid)
    h5train = h5file["train"]
    h5val = h5file["val"]
    h5_chunk_size = h5train.chunks[0]
    num_h5_train_rows = h5train.shape[0]
    num_h5_val_rows = h5val.shape[0]

    if use_training_set:
      num_h5_val_rows = num_h5_train_rows
      h5val = h5train

    log("Loaded " + str(num_h5_val_rows) + " rows")
    log("h5_chunk_size = " + str(h5_chunk_size))

    sys.stdout.flush()
    sys.stderr.flush()

    run_in_batches(h5val,process_position)

    sys.stdout.flush()
    sys.stderr.flush()

    h5file.close()
    h5fid.close()

  for gamesh5_file in gamesh5_files:
    process_file(gamesh5_file)


  data = (
    buf_pla,
    buf_positions,
    buf_recent_capture_locs,
    buf_last_moves,
    buf_correct_net_moves,
    buf_correct_lz_moves,
    buf_big_credit_moves,
    buf_medium_credit_moves,
    buf_small_credit_moves,
  )

  with open(output_file,"wb") as out:
    pickle.dump(data,out)

  print("Done, wrote to " + output_file)
