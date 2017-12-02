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
import tensorflow as tf
import numpy as np

from board import Board

description = """
Train neural net on Go games!
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-model', help='Path to model to use', required=True)
parser.add_argument('-white', help='Model plays white', action="store_true", required=False)
args = vars(parser.parse_args())

modelpath = args["model"]
is_white = args["white"]

# Model ----------------------------------------------------------------
print("Building model", flush=True)
import model
output_layer = tf.nn.softmax(model.output_layer)

# Moves ----------------------------------------------------------------

def genmove(session, board, moves):
  input_data = np.zeros(shape=[1]+model.input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(moves)
  model.fill_row_features(board,pla,opp,moves,move_idx,input_data,target_data=None,target_data_weights=None,for_training=False,idx=0)

  output = session.run(fetches=[output_layer], feed_dict={
    model.inputs: input_data,
    model.symmetries: [False,False,False],
    model.is_training: False
  })
  output = output[0][0]

  moves_and_probs = []
  for i in range(len(output)):
    move = model.tensor_pos_to_loc(i,board)
    if board.would_be_legal(pla,move) and not board.is_simple_eye(pla,move):
      moves_and_probs.append((move,output[i]))

  if len(moves_and_probs) <= 0:
    return None

  moves_and_probs = sorted(moves_and_probs, key=lambda moveandprob: moveandprob[1], reverse=True)

  #Generate a random number biased small and then find the appropriate move to make
  #Interpolate from moving uniformly to choosing from the triangular distribution
  alpha = 1
  beta = 1 + (1 - 0.5 ** (len(moves) / 30))
  r = np.random.beta(alpha,beta)
  probsum = 0.0
  i = 0
  while True:
    (move,prob) = moves_and_probs[i]
    probsum += prob
    if i >= len(moves_and_probs)-1 or probsum > r:
      return move
    i += 1

#Adapted from https://github.com/pasky/michi/blob/master/michi.py, which is distributed under MIT license
#https://opensource.org/licenses/MIT
def run_gtp(session):
  known_commands = [
    'boardsize',
    'clear_board',
    'komi',
    'play',
    'genmove',
    'final_score',
    'quit',
    'name',
    'version',
    'known_command',
    'list_commands',
    'protocol_version',
  ]

  board_size = 19
  board = Board(size=board_size)
  moves = []

  colstr = 'ABCDEFGHJKLMNOPQRST'
  def parse_coord(s):
    if s == 'pass':
      return None
    return board.loc(colstr.index(s[0].upper()), board_size - int(s[1:]))

  def str_coord(loc):
    if loc is None:
      return 'pass'
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    return '%c%d' % (colstr[x], board_size - y)

  while True:
    try:
      line = input().strip()
    except EOFError:
      break
    if line == '':
      continue
    command = [s.lower() for s in line.split()]
    if re.match('\d+', command[0]):
      cmdid = command[0]
      command = command[1:]
    else:
      cmdid = ''

    ret = ''
    if command[0] == "boardsize":
      if int(command[1]) > model.max_board_size:
        print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
        ret = None
      board_size = int(command[1])
      board = Board(size=board_size)
      moves = []
    elif command[0] == "clear_board":
      board = Board(size=board_size)
      moves = []
    elif command[0] == "komi":
      pass
    elif command[0] == "play":
      loc = parse_coord(command[2])
      if loc is not None:
        moves.append((board.pla,loc))
        board.play(board.pla,loc)
      else:
        moves.append((board.pla,loc))
        board.do_pass()
    elif command[0] == "genmove":
      loc = genmove(session, board, moves)
      if loc is not None:
        moves.append((board.pla,loc))
        board.play(board.pla,loc)
        ret = str_coord(loc)
      else:
        moves.append((board.pla,loc))
        board.do_pass()
        ret = 'pass'
    elif command[0] == "final_score":
      ret = '0'
    elif command[0] == "name":
      ret = 'simplenn'
    elif command[0] == "version":
      ret = '0.1'
    elif command[0] == "list_commands":
      ret = '\n'.join(known_commands)
    elif command[0] == "known_command":
      ret = 'true' if command[1] in known_commands else 'false'
    elif command[0] == "protocol_version":
      ret = '2'
    elif command[0] == "quit":
      print('=%s \n\n' % (cmdid,), end='')
      break
    else:
      print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
      ret = None

    if ret is not None:
      print('=%s %s\n\n' % (cmdid, ret,), end='')
    else:
      print('?%s ???\n\n' % (cmdid,), end='')
    sys.stdout.flush()

saver = tf.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

with tf.Session() as session:
  saver.restore(session, modelpath)
  run_gtp(session)



