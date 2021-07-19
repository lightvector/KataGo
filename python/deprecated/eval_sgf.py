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
from model import Model

description = """
Evaluate raw neural net output directly on a position in an sgf
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-model', help='Path to model to use', required=True)
parser.add_argument('-rank-one-hot', help='Model plays like this rankonehot', required=False)
parser.add_argument('-sgf', help="SGF file to evaluate", required=True)
parser.add_argument('-move', help="Move number to evaluate, 0-indexed", required=True)
parser.add_argument('-debug', help="Debug sandbox", action="store_true", required=False)
args = vars(parser.parse_args())

modelpath = args["model"]
sgf_file = args["sgf"]
movenum = int(args["move"])
debug = args["debug"]

play_rank_one_hot = 0
if "rank_one_hot" in args and args["rank_one_hot"] != "" and args["rank_one_hot"] is not None:
  play_rank_one_hot = int(args["rank_one_hot"])

np.set_printoptions(linewidth=150)

# Model ----------------------------------------------------------------

with open(modelpath + ".config.json") as f:
  model_config = json.load(f)
model = Model(model_config)
policy_output = tf.nn.softmax(model.policy_output)

# Moves ----------------------------------------------------------------

def fetch_output(session, board, boards, moves, use_history_prop, rank_one_hot, fetches):
  input_data = np.zeros(shape=[1]+model.input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(moves)
  self_komi = (7.5 if pla == Board.WHITE else -7.5)
  model.fill_row_features(board,pla,opp,boards,moves,move_idx,input_data,self_komi,use_history_prop=use_history_prop,idx=0)
  row_ranks = np.zeros(shape=[1]+model.rank_shape)
  row_ranks[0,rank_one_hot] = 1.0
  outputs = session.run(fetches, feed_dict={
    model.inputs: input_data,
    model.ranks: row_ranks,
    model.symmetries: [False,False,False],
    model.is_training: False,
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  return [output[0] for output in outputs]

def get_policy_output(session, board, boards, moves, use_history_prop, rank_one_hot):
  return fetch_output(session,board,boards,moves,use_history_prop,rank_one_hot,[policy_output])

def get_moves_and_probs(session, board, boards, moves, use_history_prop, rank_one_hot):
  pla = board.pla
  [policy] = get_policy_output(session, board, boards, moves, use_history_prop, rank_one_hot)
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

board_size = 19
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

(metadata,setups,moves) = data.load_sgf_moves_exn(sgf_file)
assert(metadata.size == 19) #Neural net only works with 19x19 right now

for (pla,loc) in setups:
  setstone(pla,loc)

for i in range(movenum):
  (pla,loc) = moves[i]
  play(pla,loc)

print(board.to_string())

saver = tf.compat.v1.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

with tf.compat.v1.Session() as session:

  if not debug:
    saver.restore(session, modelpath)
    moves_and_probs = get_moves_and_probs(session, board, boards, moves, use_history_prop=1.0, rank_one_hot = play_rank_one_hot)
    moves_and_probs = sorted(moves_and_probs, key=lambda moveandprob: moveandprob[1], reverse=True)

    for i in range(len(moves_and_probs)):
      (move,prob) = moves_and_probs[i]
      print("%s %4.1f%%" % (str_coord(move,board),prob*100))

  else:
    [transformed_input] = fetch_output(session, board, boards, moves[:movenum], use_history_prop=1.0, rank_one_hot = play_rank_one_hot, fetches = [model.transformed_input])
    transformed_input = np.array(transformed_input)
    assert(len(transformed_input.shape) == 3)
    for i in range(transformed_input.shape[2]):
      print(i)
      print(transformed_input[:,:,i])





