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

#Feature extraction functions-------------------------------------------------------------------

max_board_size = 19
input_shape = [19,19,13]
target_shape = [19*19]

prob_to_include_prev1 = 1.0
prob_to_include_prev2 = 1.0
prob_to_include_prev3 = 1.0

def fill_row_features(board, pla, opp, moves, move_idx, input_data, idx):
  for y in range(19):
    for x in range(19):
      input_data[idx,y,x,0] = 1.0
      loc = board.loc(x,y)
      stone = board.board[loc]
      if stone == pla:
        input_data[idx,y,x,1] = 1.0
        libs = board.num_liberties(loc)
        if libs == 1:
          input_data[idx,y,x,3] = 1.0
        elif libs == 2:
          input_data[idx,y,x,4] = 1.0
        elif libs == 3:
          input_data[idx,y,x,5] = 1.0

      elif stone == opp:
        input_data[idx,y,x,2] = 1.0
        libs = board.num_liberties(loc)
        if libs == 1:
          input_data[idx,y,x,6] = 1.0
        elif libs == 2:
          input_data[idx,y,x,7] = 1.0
        elif libs == 3:
          input_data[idx,y,x,8] = 1.0

  if move_idx >= 1 and random.random() < prob_to_include_prev1:
    prev1_loc = moves[move_idx-1][1]
    if prev1_loc is not None:
      input_data[idx,board.loc_y(prev1_loc),board.loc_x(prev1_loc),9] = 1.0

    if move_idx >= 2 and random.random() < prob_to_include_prev2:
      prev2_loc = moves[move_idx-2][1]
      if prev2_loc is not None:
        input_data[idx,board.loc_y(prev2_loc),board.loc_x(prev2_loc),10] = 1.0

      if move_idx >= 3 and random.random() < prob_to_include_prev3:
        prev3_loc = moves[move_idx-3][1]
        if prev3_loc is not None:
          input_data[idx,board.loc_y(prev3_loc),board.loc_x(prev3_loc),11] = 1.0

  if board.simple_ko_point is not None:
    input_data[idx,board.loc_y(board.simple_ko_point),board.loc_x(board.simple_ko_point),12] = 1.0

# Build model -------------------------------------------------------------

is_training = tf.placeholder(tf.bool)
reg_variables = []

def batchnorm(name,tensor):
  return tf.layers.batch_normalization(
    tensor,
    axis=-1, #Because channels are our last axis, -1 refers to that via wacky python indexing
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    training=is_training,
    name=name,
  )

def init_stdev(num_inputs,num_outputs):
  #xavier
  #return math.sqrt(2.0 / (num_inputs + num_outputs))
  #herangzhen
  return math.sqrt(2.0 / (num_inputs))

def weight_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs) / 1.0
  initial = tf.truncated_normal(shape=shape, stddev=stdev)
  variable = tf.Variable(initial,name=name)
  reg_variables.append(variable)
  return variable

def bias_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs) / 2.0
  initial = tf.truncated_normal(shape=shape, mean=stdev, stddev=stdev)
  variable = tf.Variable(initial,name=name)
  reg_variables.append(variable)
  return variable

def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

#Indexing:
#batch, bsize, bsize, channel

#Input layer
inputs = tf.placeholder(tf.float32, [None] + input_shape)

outputs_by_layer = []
cur_layer = inputs
cur_num_channels = input_shape[2]

#Convolutional RELU layer 1
conv1diam = 3
conv1num_channels = 32
conv1w = weight_variable("conv1w",[conv1diam,conv1diam,cur_num_channels,conv1num_channels],cur_num_channels*conv1diam**2,conv1num_channels)
# conv1b = bias_variable("conv1b",[conv1num_channels],cur_num_channels,conv1num_channels)

cur_layer = tf.nn.relu(batchnorm("conv1norm",conv2d(cur_layer, conv1w)))
cur_num_channels = conv1num_channels
outputs_by_layer.append(("conv1",cur_layer))

#Convolutional RELU layer 2
conv2diam = 3
conv2num_channels = 16
conv2w = weight_variable("conv2w",[conv2diam,conv2diam,cur_num_channels,conv2num_channels],cur_num_channels*conv2diam**2,conv2num_channels)
# conv2b = bias_variable("conv2b",[conv2num_channels],cur_num_channels,conv2num_channels)

cur_layer = tf.nn.relu(batchnorm("conv2norm",conv2d(cur_layer, conv2w)))
cur_num_channels = conv2num_channels
outputs_by_layer.append(("conv2",cur_layer))

#Convolutional RELU layer 3
conv3diam = 3
conv3num_channels = 8
conv3w = weight_variable("conv3w",[conv3diam,conv3diam,cur_num_channels,conv3num_channels],cur_num_channels*conv3diam**2,conv3num_channels)
# conv3b = bias_variable("conv3b",[conv3num_channels],cur_num_channels,conv3num_channels)

cur_layer = tf.nn.relu(batchnorm("conv3norm",conv2d(cur_layer, conv3w)))
cur_num_channels = conv3num_channels
outputs_by_layer.append(("conv3",cur_layer))

#Convolutional linear output layer 4
conv4diam = 5
conv4num_channels = 1
conv4w = weight_variable("conv4w",[conv4diam,conv4diam,cur_num_channels,conv4num_channels],cur_num_channels*conv4diam**2,conv4num_channels)

cur_layer = conv2d(cur_layer, conv4w)
cur_num_channels = conv4num_channels
outputs_by_layer.append(("conv4",cur_layer))

#Output
assert(cur_num_channels == 1)
output_layer = tf.reshape(cur_layer, [-1] + target_shape)

#-----------------------------------------------------------------------------

output_layer = tf.nn.softmax(output_layer)

def genmove(session, board, moves):
  input_data = np.zeros(shape=[1]+input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(moves)
  fill_row_features(board,pla,opp,moves,move_idx,input_data,idx=0)

  output = session.run(fetches=[output_layer], feed_dict={
    inputs: input_data,
    is_training: False
  })
  output = output[0][0]

  moves_and_probs = []
  for i in range(len(output)):
    move = None #pass
    if i < 361:
      move = board.loc(i%19,i//19)
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
      if int(command[1]) != board_size:
        print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
        ret = None
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



