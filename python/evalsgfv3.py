#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import re
import logging
import colorsys
import json
import tensorflow as tf
import numpy as np

import data
from board import Board
import modelv3
from modelv3 import ModelV3, Target_varsV3, MetricsV3

description = """
Evaluate raw neural net output directly on a position in an sgf
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-model', help='Path to model to use', required=True)
parser.add_argument('-model-config', help='Path to model config json to use', required=False)
parser.add_argument('-sgf', help="SGF file to evaluate", required=True)
parser.add_argument('-move', help="Move number to evaluate, 0-indexed", required=True)
parser.add_argument('-debug', help="Debug sandbox", action="store_true", required=False)
args = vars(parser.parse_args())

modelpath = args["model"]
modelconfigpath = args["model_config"]
if modelconfigpath is None:
  modelconfigpath = os.path.join(os.path.dirname(modelpath),"model.config.json")
sgf_file = args["sgf"]
movenum = int(args["move"])
debug = args["debug"]

np.set_printoptions(linewidth=150)

# Model ----------------------------------------------------------------

pos_len = 19
with open(modelconfigpath) as f:
  model_config = json.load(f)
model = ModelV3(model_config,pos_len,{})
policy_output = tf.nn.softmax(model.policy_output)
value_output = tf.nn.softmax(model.value_output)
scorevalue_output = tf.tanh(model.miscvalues_output[:,0])
scorebelief_output = tf.nn.softmax(scorebelief_output)
ownership_output = tf.tanh(model.ownership_output)

# Moves ----------------------------------------------------------------

def fetch_output(session, board, boards, moves, use_history_prop, rules, fetches):
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(moves)
  model.fill_row_features(board,pla,opp,boards,moves,move_idx,rules,bin_input_data,global_input_data,use_history_prop=use_history_prop,idx=0)
  outputs = session.run(fetches, feed_dict={
    model.bin_inputs: bin_input_data,
    model.global_inputs: global_input_data,
    model.symmetries: [False,False,False],
    model.is_training: False,
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  return [output[0] for output in outputs]

def get_policy_output(session, board, boards, moves, use_history_prop, rank_one_hot):
  return fetch_output(session,board,boards,moves,use_history_prop,rank_one_hot,[policy_output])

def get_moves_and_probs_of_policy(policy):
  moves_and_probs = []
  for i in range(len(policy)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy)-1:
      moves_and_probs.append((Board.PASS_LOC,policy[i]))
    elif board.would_be_legal(pla,move) and not board.is_simple_eye(pla,move):
      moves_and_probs.append((move,policy[i]))
  return moves_and_probs

# Basic parsing --------------------------------------------------------
colstr = 'ABCDEFGHJKLMNOPQRST'
def str_coord(loc,board):
  if loc == Board.PASS_LOC:
    return 'pass'
  x = board.loc_x(loc)
  y = board.loc_y(loc)
  return '%c%d' % (colstr[x], board.size - y)

(metadata,setups,moves,rules) = data.load_sgf_moves_exn(sgf_file)

board_size = metadata.size
board = Board(size=board_size)
moves = []
boards = [board.copy()]

def setstone(pla,loc):
  board.play(pla,loc)
  moves.clear()
  boards.clear()
  boards.append(board.copy())
def play(pla,loc):
  board.play(pla,loc)
  moves.append((pla,loc))
  boards.append(board.copy())

for (pla,loc) in setups:
  setstone(pla,loc)

for i in range(movenum):
  (pla,loc) = moves[i]
  play(pla,loc)

print(board.to_string())

saver = tf.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

with tf.Session() as session:

  if not debug:
    saver.restore(session, modelpath)
    (policy,value,scorevalue,scorebelief,ownership) = fetch_output(session, board, boards, moves, 1.0, rules, (
      policy_output,
      value_output,
      scorevalue_output,
      scorebelief_output,
      ownership_output
    ))

    moves_and_probs = get_moves_and_probs_of_policy(policy)
    moves_and_probs = sorted(moves_and_probs, key=lambda moveandprob: moveandprob[1], reverse=True)

    print("Value: " + str(value))
    print("ScoreValue: " + str(scorevalue))
    print("Policy: ")
    for i in range(max(len(moves_and_probs),40)):
      (move,prob) = moves_and_probs[i]
      print("%s %4.1f%%" % (str_coord(move,board),prob*100))
    print("Ownership: ")
    for y in range(board_size):
      for x in range(board_size):
        print("%+5.0f " % (ownership[y*pos_len+x] * 1000),end="")
      print()

    print("ScoreBelief: ")
    for i in range(17,-1,-1):
      print("%+6.1" % (-(i*20+0.5)),end="")
      for j in range(20):
        idx = 360-(i*20+j)
        print(" %5.0f" % (scorebelief[idx] * 10000),end="")
      print()
    for i in range(18):
      print("%+6.1" % ((i*20+0.5)),end="")
      for j in range(20):
        idx = 361+(i*20+j)
        print(" %5.0f" % (scorebelief[idx] * 10000),end="")
      print()

  else:
    [transformed_input] = fetch_output(session, board, boards, moves[:movenum], use_history_prop=1.0, rank_one_hot = play_rank_one_hot, fetches = [model.transformed_input])
    transformed_input = np.array(transformed_input)
    assert(len(transformed_input.shape) == 3)
    for i in range(transformed_input.shape[2]):
      print(i)
      print(transformed_input[:,:,i])





