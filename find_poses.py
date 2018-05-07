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
import json
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
parser.add_argument('-output-dir', help='', required=True)
parser.add_argument('-model-file', help='model file prefix to load', required=True)
parser.add_argument('-config-file', help='json file of config for rank conditions', required=True)
parser.add_argument('-data-prop', help='proportion of data to use', required=True)
parser.add_argument('-use-training-set', help='run on training set instead of test set', required=False, action="store_true")
args = vars(parser.parse_args())

gamesh5_files = args["gamesh5"]
output_dir = args["output_dir"]
model_file = args["model_file"]
config_file = args["config_file"]
data_prop = float(args["data_prop"])
use_training_set = args["use_training_set"]

with open(config_file) as infile:
  config = json.load(infile)

def log(s):
  print(s,flush=True)
np.set_printoptions(linewidth=150)

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
next_moves_start = recent_captures_start + recent_captures_len
next_moves_len = 7
sgfhash_start = next_moves_start + next_moves_len
sgfhash_len = 4

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
  num_used_counts = {}
  num_used_by_output_matrix = {}

  ko_filter_key = "kofiltered.pickle"
  entries_by_output_key = {}
  for output_key in ([key for key in config] + [ko_filter_key]):
    entries_by_output_key[output_key] = []
    num_used_by_output_matrix[output_key] = {}
    for output_key2 in ([key for key in config] + [ko_filter_key]):
      num_used_by_output_matrix[output_key][output_key2] = 0

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
    num_validation_batches = (num_h5_val_rows+validation_batch_size-1)//validation_batch_size
    num_validation_batches = math.floor(data_prop * num_validation_batches)

    pro_ranks_input = np.zeros([rank_len])
    pro_ranks_input[0] = 1.0
    pro_ranks_input = [pro_ranks_input for i in range(validation_batch_size)]

    for i in range(num_validation_batches):
      print("Batch: " + str(i) + "/" + str(num_validation_batches) +
            " Used " + str(num_used[0]) + "/" + str(num_processed[0]) +
            " " + str(num_used_counts) +
            " " + str(num_used_by_output_matrix),
            flush=True)
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
        elif recent_captures[loc] > 0 and recent_captures[loc] <= 5:
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
        np.max(inputs[:,20]) > 0 and
        np.max(inputs[:,21]) > 0 and
        np.max(inputs[:,22]) > 0):
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
    batch_probs = run([inputs],[ranks])
    probs = batch_probs[0]
    return probs

  def get_multi_rank_probs(inputs,ranks):
    batch_inputs = []
    batch_ranks = []
    for rank in ranks:
      arr = np.zeros([rank_len])
      arr[rank] = 1.0
      batch_inputs.append(inputs)
      batch_ranks.append(arr)
    batch_probs = run(batch_inputs,batch_ranks)
    return batch_probs

  def is_on_board(x,y):
    return (x >= 0 and x < 19 and y >= 0 and y < 19)

  def is_maybe_ko_recapture(pla,position,last_moves,real_move):
    real_move_x = int(real_move) % 19
    real_move_y = int(real_move) // 19
    last_move3_x = int(last_moves[2]) % 19
    last_move3_y = int(last_moves[2]) // 19
    real_move_adjs = []
    if is_on_board(real_move_x,real_move_y-1):
      real_move_adjs.append((real_move_x,real_move_y-1))
    if is_on_board(real_move_x-1,real_move_y):
      real_move_adjs.append((real_move_x-1,real_move_y))
    if is_on_board(real_move_x+1,real_move_y):
      real_move_adjs.append((real_move_x+1,real_move_y))
    if is_on_board(real_move_x,real_move_y+1):
      real_move_adjs.append((real_move_x,real_move_y+1))

    if (last_move3_x,last_move3_y) not in real_move_adjs:
      return False
    for (x,y) in real_move_adjs:
      if position[x+y*19] != 3-pla:
        return False
    return True

  def write_position(inputs,target,ranks,pro_probs,row,real_move,outputs_to_include_in):
    pla = int(row[side_start]+1)
    opp = 3-pla
    recent_captures = row[recent_captures_start:recent_captures_start+recent_captures_len]
    max_pro_prob = np.max(pro_probs)

    position = np.zeros([19*19],dtype=np.int8)
    recent_capture_locs = []
    last_moves = np.zeros([5],dtype=np.int16)
    correct_net_moves = []
    correct_lz_moves = []
    big_credit_moves = []
    medium_credit_moves = []
    small_credit_moves = []
    next_moves = np.zeros([5],dtype=np.int16)

    #By default, set the last moves all to an offboard number if there was no such move (or it was a pass)
    for i in range(len(last_moves)):
      last_moves[i] = 19*19

    correct_net_moves.append(real_move)
    for y in range(19):
      for x in range(19):
        loc = y*19+x
        if inputs[loc,1] == 1.0:
          position[loc] = pla
        elif inputs[loc,2] == 1.0:
          position[loc] = opp

        if recent_captures[loc] > 0 and recent_captures[loc] <= 5:
          recent_capture_locs.append(loc)

        if inputs[loc,18] == 1.0:
          last_moves[4] = loc
        elif inputs[loc,19] == 1.0:
          last_moves[3] = loc
        elif inputs[loc,20] == 1.0:
          last_moves[2] = loc
        elif inputs[loc,21] == 1.0:
          last_moves[1] = loc
        elif inputs[loc,22] == 1.0:
          last_moves[0] = loc

        if loc in correct_net_moves or loc in correct_lz_moves:
          pass
        elif pro_probs[loc] >= max_pro_prob * 0.28:
          big_credit_moves.append(loc)
        elif pro_probs[loc] >= max_pro_prob * 0.14:
          medium_credit_moves.append(loc)
        elif pro_probs[loc] >= max_pro_prob * 0.06:
          small_credit_moves.append(loc)

    for i in range(next_moves_len):
      next_moves[i] = row[next_moves_start+i]

    sgfhash = row[sgfhash_start:sgfhash_start+sgfhash_len]
    sgfhash = hex(int(sgfhash[0]) + int(sgfhash[1])*0x10000 + int(sgfhash[2])*0x100000000 + int(sgfhash[3])*0x1000000000000)

    if is_maybe_ko_recapture(pla,position,last_moves,real_move):
      if random.random() < 0.25:
        outputs_to_include_in = [ko_filter_key]
      else:
        outputs_to_include_in.append(ko_filter_key)

    entry = (pla,position,recent_capture_locs,last_moves,correct_net_moves,correct_lz_moves,big_credit_moves,medium_credit_moves,small_credit_moves,next_moves,sgfhash)

    for output_key in outputs_to_include_in:
      arr = entries_by_output_key[output_key]
      arr.append(entry)
      for output_key2 in outputs_to_include_in:
        num_used_by_output_matrix[output_key][output_key2] += 1

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
        pro_expected_score = get_expected_score(pro_probs,pro_probs)
        outputs_to_include_in = []

        #Batch up all the ranks we need to run a neural net eval for
        desired_ranks = []
        for output_key in config:
          output_specs = config[output_key]
          for output_spec in output_specs:
            if output_spec["user_rank"] not in desired_ranks and output_spec["user_rank"] != 0:
              desired_ranks.append(output_spec["user_rank"])
            if output_spec["better_rank"] not in desired_ranks and output_spec["better_rank"] != 0:
              desired_ranks.append(output_spec["better_rank"])

        probs = get_multi_rank_probs(inputs,desired_ranks)
        probs_by_rank = {}
        for i in range(len(desired_ranks)):
          probs_by_rank[desired_ranks[i]] = probs[i]
        probs_by_rank[0] = pro_probs

        for output_key in config:
          output_specs = config[output_key]
          for output_spec in output_specs:
            user_probs = probs_by_rank[output_spec["user_rank"]]
            better_probs = probs_by_rank[output_spec["better_rank"]]
            user_expected_score = get_expected_score(pro_probs,user_probs)
            better_expected_score = get_expected_score(pro_probs,better_probs)
            prop = (user_expected_score - output_spec["user_min_score"]) / (output_spec["user_max_score"] - output_spec["user_min_score"])
            min_better_score = output_spec["min_better_at_min"] + prop * (output_spec["min_better_at_max"] - output_spec["min_better_at_min"])
            if(user_expected_score >= output_spec["user_min_score"] and
               user_expected_score <= output_spec["user_max_score"] and
               better_expected_score >= min_better_score and
               pro_expected_score >= min_better_score):
              #If any of the specs in the config want to include this row, then include it
              outputs_to_include_in.append(output_key)
              break

        num_outputs_to_include_in = len(outputs_to_include_in)
        if num_outputs_to_include_in > 0:
          num_used[0] += 1
          if num_outputs_to_include_in not in num_used_counts:
            num_used_counts[num_outputs_to_include_in] = 0
          num_used_counts[num_outputs_to_include_in] += 1
          write_position(inputs,target,ranks,pro_probs,row,real_move,outputs_to_include_in)


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

  for output_key in entries_by_output_key:
    entries = entries_by_output_key[output_key]
    random.shuffle(entries)

    path = output_dir + "/" + output_key
    with open(path,"wb") as out:
      pickle.dump(entries,out)
      print("Done, wrote to " + path)

  print("Done")
