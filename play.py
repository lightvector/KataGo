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

from board import Board
from model import Model

description = """
Play go with a trained neural net!
Implements a basic GTP engine that uses the neural net directly to play moves..
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-model', help='Path to model to use', required=True)
parser.add_argument('-rank-one-hot', help='Model plays like this rankonehot', required=False)
args = vars(parser.parse_args())

modelpath = args["model"]

play_rank_one_hot = [0]
if "rank_one_hot" in args and args["rank_one_hot"] != "":
  play_rank_one_hot[0] = int(args["rank_one_hot"])

# Model ----------------------------------------------------------------

with open(modelpath + ".config.json") as f:
  model_config = json.load(f)
model = Model(model_config)
policy_output = tf.nn.softmax(model.policy_output)

# Moves ----------------------------------------------------------------

def fetch_output(session, board, moves, use_history_prop, rank_one_hot, fetch):
  input_data = np.zeros(shape=[1]+model.input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(moves)
  self_komi = (7.5 if pla == Board.WHITE else -7.5) #TODO use real komi of the game?
  model.fill_row_features(board,pla,opp,moves,move_idx,input_data,self_komi,use_history_prop=use_history_prop,idx=0)
  row_ranks = np.zeros(shape=[1]+model.rank_shape)
  row_ranks[0,rank_one_hot] = 1.0
  output = session.run(fetches=[fetch], feed_dict={
    model.inputs: input_data,
    model.ranks: row_ranks,
    model.symmetries: [False,False,False],
    model.is_training: False
  })
  return output[0][0]

def get_policy_output(session, board, moves, use_history_prop, rank_one_hot):
  return fetch_output(session,board,moves,use_history_prop,rank_one_hot,policy_output)

def get_moves_and_probs(session, board, moves, use_history_prop, rank_one_hot):
  pla = board.pla
  policy = get_policy_output(session, board, moves, use_history_prop, rank_one_hot)
  moves_and_probs = []
  for i in range(len(policy)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy)-1:
      moves_and_probs.append((Board.PASS_LOC,policy[i]))
    elif board.would_be_legal(pla,move) and not board.is_simple_eye(pla,move):
      moves_and_probs.append((move,policy[i]))
  return moves_and_probs

def genmove(session, board, moves, use_history_prop):
  moves_and_probs = get_moves_and_probs(session, board, moves, use_history_prop, play_rank_one_hot[0])
  moves_and_probs = sorted(moves_and_probs, key=lambda moveandprob: moveandprob[1], reverse=True)

  if len(moves_and_probs) <= 0:
    return Board.PASS_LOC

  #Generate a random number biased small and then find the appropriate move to make
  #Interpolate from moving uniformly to choosing from the triangular distribution
  alpha = 1
  beta = 1 + math.sqrt(max(0,len(moves)-20))
  r = np.random.beta(alpha,beta)
  probsum = 0.0
  i = 0
  while True:
    (move,prob) = moves_and_probs[i]
    probsum += prob
    if i >= len(moves_and_probs)-1 or probsum > r:
      return move
    i += 1

def get_layer_values(session, board, moves, layer, channel):
  layer = fetch_output(session,board,moves,use_history_prop=1.0,rank_one_hot=play_rank_one_hot[0],fetch=layer)
  layer = layer.reshape([board.size * board.size,-1])
  locs_and_values = []
  for i in range(board.size * board.size):
    loc = model.tensor_pos_to_loc(i,board)
    locs_and_values.append((loc,layer[i,channel]))
  return locs_and_values

def get_input_feature(board, moves, feature_idx):
  input_data = np.zeros(shape=[1]+model.input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(moves)
  self_komi = (7.5 if pla == Board.WHITE else -7.5)
  model.fill_row_features(board,pla,opp,moves,move_idx,input_data,self_komi,use_history_prop=1.0,idx=0)

  locs_and_values = []
  for i in range(board.size * board.size):
    loc = model.tensor_pos_to_loc(i,board)
    locs_and_values.append((loc,input_data[0,i,feature_idx]))
  return locs_and_values


def fill_gfx_commands_for_heatmap(gfx_commands, locs_and_values, board, normalization_div, is_percent):
  divisor = 1.0
  if normalization_div == "max":
    max_abs_value = max(abs(value) for (loc,value) in locs_and_values)
    divisor = max(0.0000000001,max_abs_value) #avoid divide by zero
  elif normalization_div is not None:
    divisor = normalization_div

  #Caps value at 1.0, using an asymptotic curve
  def loose_cap(x):
    def transformed_softplus(x):
      return -math.log(math.exp(-(x-1.0)*8.0)+1.0)/8.0+1.0
    base = transformed_softplus(0.0)
    return (transformed_softplus(x) - base) / (1.0 - base)

  #Softly curves a value so that it ramps up faster than linear in that range
  def soft_curve(x,x0,x1):
    p = (x-x0)/(x1-x0)
    def curve(p):
      return math.sqrt(p+0.16)-0.4
    p = curve(p) / curve(1.0)
    return x0 + p * (x1-x0)

  for (loc,value) in locs_and_values:
    if loc != Board.PASS_LOC:
      value = value / divisor
      if value < 0:
        value = -value
        huestart = 0.50
        huestop = 0.86
      else:
        huestart = -0.02
        huestop = 0.45

      value = loose_cap(value)

      def lerp(p,x0,x1,y0,y1):
        return y0 + (y1-y0) * (p-x0)/(x1-x0)

      if value <= 0.04:
        hue = huestart
        lightness = 0.5
        saturation = value / 0.04
        (r,g,b) = colorsys.hls_to_rgb((hue+1)%1, lightness, saturation)
      elif value <= 0.70:
        # value = soft_curve(value,0.04,0.70)
        hue = lerp(value,0.04,0.70,huestart,huestop)
        val = 1.0
        saturation = 1.0
        (r,g,b) = colorsys.hsv_to_rgb((hue+1)%1, val, saturation)
      else:
        hue = huestop
        lightness = lerp(value,0.70,1.00,0.5,0.95)
        saturation = 1.0
        (r,g,b) = colorsys.hls_to_rgb((hue+1)%1, lightness, saturation)

      r = ("%02x" % int(r*255))
      g = ("%02x" % int(g*255))
      b = ("%02x" % int(b*255))
      gfx_commands.append("COLOR #%s%s%s %s" % (r,g,b,str_coord(loc,board)))

  locs_and_values = sorted(locs_and_values, key=lambda loc_and_value: loc_and_value[1])
  locs_and_values_rev = sorted(locs_and_values, key=lambda loc_and_value: loc_and_value[1], reverse=True)
  texts = []
  texts_rev = []
  maxlen_per_side = 10
  if len(locs_and_values) > 0 and locs_and_values[0][1] < 0:
    maxlen_per_side = 5

    for i in range(min(len(locs_and_values),maxlen_per_side)):
      (loc,value) = locs_and_values[i]
      if is_percent:
        texts.append("%s %4.1f%%" % (str_coord(loc,board),value*100))
      else:
        texts.append("%s %.3f" % (str_coord(loc,board),value))
    texts.reverse()

  for i in range(min(len(locs_and_values_rev),maxlen_per_side)):
    (loc,value) = locs_and_values_rev[i]
    if is_percent:
      texts_rev.append("%s %4.1f%%" % (str_coord(loc,board),value*100))
    else:
      texts_rev.append("%s %.3f" % (str_coord(loc,board),value))

  gfx_commands.append("TEXT " + ", ".join(texts_rev + texts))


# Basic parsing --------------------------------------------------------
colstr = 'ABCDEFGHJKLMNOPQRST'
def parse_coord(s,board):
  if s == 'pass':
    return Board.PASS_LOC
  return board.loc(colstr.index(s[0].upper()), board.size - int(s[1:]))

def str_coord(loc,board):
  if loc == Board.PASS_LOC:
    return 'pass'
  x = board.loc_x(loc)
  y = board.loc_y(loc)
  return '%c%d' % (colstr[x], board.size - y)


# GTP Implementation -----------------------------------------------------

#Adapted from https://github.com/pasky/michi/blob/master/michi.py, which is distributed under MIT license
#https://opensource.org/licenses/MIT
def run_gtp(session):
  known_commands = [
    'boardsize',
    'clear_board',
    'komi',
    'play',
    'genmove',
    'quit',
    'name',
    'version',
    'known_command',
    'list_commands',
    'protocol_version',
    'gogui-analyze_commands',
    'policy',
    'policy-half-history',
    'policy-no-history',
  ]
  known_analyze_commands = [
    'gfx/Policy/policy',
    'gfx/PolicyHalfHistory/policy-half-history',
    'gfx/PolicyNoHistory/policy-no-history',
  ]

  board_size = 19
  board = Board(size=board_size)
  moves = []

  layerdict = dict(model.outputs_by_layer)

  rank_policy_command_lookup = dict()
  layer_command_lookup = dict()

  def add_rank_policy_command_lookup(name, rank_one_hot):
    command_name = (name + "(" + str(rank_one_hot) + ")").replace("/",":")
    known_commands.append(command_name)
    known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
    rank_policy_command_lookup[command_name.lower()] = rank_one_hot

  add_rank_policy_command_lookup("policy_GoGoD",0)
  add_rank_policy_command_lookup("policy_KGS1d",1)
  add_rank_policy_command_lookup("policy_KGS2d",2)
  add_rank_policy_command_lookup("policy_KGS3d",3)
  add_rank_policy_command_lookup("policy_KGS4d",4)
  add_rank_policy_command_lookup("policy_KGS5d",5)
  add_rank_policy_command_lookup("policy_KGS6d",6)
  add_rank_policy_command_lookup("policy_KGS7d",7)
  add_rank_policy_command_lookup("policy_KGS8d",8)
  add_rank_policy_command_lookup("policy_KGS9d",9)
  add_rank_policy_command_lookup("policy_Fox17k",10)
  add_rank_policy_command_lookup("policy_Fox16k",11)
  add_rank_policy_command_lookup("policy_Fox15k",12)
  add_rank_policy_command_lookup("policy_Fox14k",13)
  add_rank_policy_command_lookup("policy_Fox13k",14)
  add_rank_policy_command_lookup("policy_Fox12k",15)
  add_rank_policy_command_lookup("policy_Fox11k",16)
  add_rank_policy_command_lookup("policy_Fox10k",17)
  add_rank_policy_command_lookup("policy_Fox9k",18)
  add_rank_policy_command_lookup("policy_Fox8k",19)
  add_rank_policy_command_lookup("policy_Fox7k",20)
  add_rank_policy_command_lookup("policy_Fox6k",21)
  add_rank_policy_command_lookup("policy_Fox5k",22)
  add_rank_policy_command_lookup("policy_Fox4k",23)
  add_rank_policy_command_lookup("policy_Fox3k",24)
  add_rank_policy_command_lookup("policy_Fox2k",25)
  add_rank_policy_command_lookup("policy_Fox1k",26)
  add_rank_policy_command_lookup("policy_Fox1d",27)
  add_rank_policy_command_lookup("policy_Fox2d",28)
  add_rank_policy_command_lookup("policy_Fox3d",29)
  add_rank_policy_command_lookup("policy_Fox4d",30)
  add_rank_policy_command_lookup("policy_Fox5d",31)
  add_rank_policy_command_lookup("policy_Fox6d",32)
  add_rank_policy_command_lookup("policy_Fox7d",33)
  add_rank_policy_command_lookup("policy_Fox8d",34)
  add_rank_policy_command_lookup("policy_Fox9d",35)
  add_rank_policy_command_lookup("policy_OGS19k",36)
  add_rank_policy_command_lookup("policy_OGS18k",37)
  add_rank_policy_command_lookup("policy_OGS17k",38)
  add_rank_policy_command_lookup("policy_OGS16k",39)
  add_rank_policy_command_lookup("policy_OGS15k",40)
  add_rank_policy_command_lookup("policy_OGS14k",41)
  add_rank_policy_command_lookup("policy_OGS13k",42)
  add_rank_policy_command_lookup("policy_OGS12k",43)
  add_rank_policy_command_lookup("policy_OGS11k",44)
  add_rank_policy_command_lookup("policy_OGS10k",45)
  add_rank_policy_command_lookup("policy_OGS9k",46)
  add_rank_policy_command_lookup("policy_OGS8k",47)
  add_rank_policy_command_lookup("policy_OGS7k",48)
  add_rank_policy_command_lookup("policy_OGS6k",49)
  add_rank_policy_command_lookup("policy_OGS5k",50)
  add_rank_policy_command_lookup("policy_OGS4k",51)
  add_rank_policy_command_lookup("policy_OGS3k",52)
  add_rank_policy_command_lookup("policy_OGS2k",53)
  add_rank_policy_command_lookup("policy_OGS1k",54)
  add_rank_policy_command_lookup("policy_OGS1d",55)
  add_rank_policy_command_lookup("policy_OGS2d",56)
  add_rank_policy_command_lookup("policy_OGS3d",57)
  add_rank_policy_command_lookup("policy_OGS4d",58)
  add_rank_policy_command_lookup("policy_OGS5d",59)
  add_rank_policy_command_lookup("policy_OGS6d",60)
  add_rank_policy_command_lookup("policy_OGS7d",61)
  add_rank_policy_command_lookup("policy_OGS8d",62)
  add_rank_policy_command_lookup("policy_OGS9d",63)

  def add_extra_board_size_visualizations(layer_name, layer, normalization_div):
    assert(layer.shape[1].value == board_size)
    assert(layer.shape[2].value == board_size)
    num_channels = layer.shape[3].value
    for i in range(num_channels):
      command_name = layer_name + "-" + str(i)
      command_name = command_name.replace("/",":")
      known_commands.append(command_name)
      known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
      layer_command_lookup[command_name.lower()] = (layer,i,normalization_div)

  def add_board_size_visualizations(layer_name, normalization_div):
    layer = layerdict[layer_name]
    add_extra_board_size_visualizations(layer_name, layer, normalization_div)

  add_board_size_visualizations("conv1",normalization_div=6)
  add_board_size_visualizations("rconv1",normalization_div=14)
  add_board_size_visualizations("rconv2",normalization_div=20)
  add_board_size_visualizations("rconv3",normalization_div=26)
  add_board_size_visualizations("rconv4",normalization_div=36)
  add_board_size_visualizations("rconv5",normalization_div=40)
  add_board_size_visualizations("rconv6",normalization_div=44)
  add_board_size_visualizations("rconv6/conv1a",normalization_div=12)
  add_board_size_visualizations("rconv6/conv1b",normalization_div=12)
  add_board_size_visualizations("rconv7",normalization_div=48)
  add_board_size_visualizations("rconv8",normalization_div=52)
  add_board_size_visualizations("rconv9",normalization_div=55)
  add_board_size_visualizations("rconv10",normalization_div=58)
  add_board_size_visualizations("rconv10/conv1a",normalization_div=12)
  add_board_size_visualizations("rconv10/conv1b",normalization_div=12)
  add_board_size_visualizations("rconv11",normalization_div=61)
  add_board_size_visualizations("rconv12",normalization_div=64)
  add_board_size_visualizations("g1",normalization_div=6)
  add_board_size_visualizations("p1",normalization_div=2)

  input_feature_command_lookup = dict()
  def add_input_feature_visualizations(layer_name, feature_idx, normalization_div):
    command_name = layer_name
    command_name = command_name.replace("/",":")
    known_commands.append(command_name)
    known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
    input_feature_command_lookup[command_name] = (feature_idx,normalization_div)

  for i in range(model.input_shape[1]):
    add_input_feature_visualizations("input-" + str(i),i, normalization_div=1)


  linear = tf.cumsum(tf.ones([19],dtype=tf.float32),axis=0,exclusive=True) / 18.0
  color_calibration = tf.stack(axis=0,values=[
    linear,
    linear*0.5,
    linear*0.2,
    linear*0.1,
    linear*0.05,
    linear*0.02,
    linear*0.01,
    -linear,
    -linear*0.5,
    -linear*0.2,
    -linear*0.1,
    -linear*0.05,
    -linear*0.02,
    -linear*0.01,
    linear*2-1,
    tf.zeros([19],dtype=tf.float32),
    tf.zeros([19],dtype=tf.float32),
    tf.zeros([19],dtype=tf.float32),
    tf.zeros([19],dtype=tf.float32),
  ])
  add_extra_board_size_visualizations("colorcalibration", tf.reshape(color_calibration,[1,19,19,1]),normalization_div=None)

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
      pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
      loc = parse_coord(command[2],board)
      moves.append((pla,loc))
      board.play(pla,loc)
    elif command[0] == "genmove":
      loc = genmove(session, board, moves, use_history_prop=1.0)
      moves.append((board.pla,loc))
      board.play(board.pla,loc)
      ret = str_coord(loc,board)
    # elif command[0] == "final_score":
    #   ret = '0'
    elif command[0] == "name":
      ret = 'simplenn'
    elif command[0] == "version":
      ret = '0.1'
    elif command[0] == "list_commands":
      ret = '\n'.join(known_commands)
    elif command[0] == "known_command":
      ret = 'true' if command[1] in known_commands else 'false'
    elif command[0] == "gogui-analyze_commands":
      ret = '\n'.join(known_analyze_commands)
    elif command[0] == "rank":
      try:
        parsed = int(command[1])
      except ValueError:
        parsed = None
      if parsed is not None:
        play_rank_one_hot[0] = parsed
    elif command[0] == "policy":
      moves_and_probs = get_moves_and_probs(session, board, moves, use_history_prop=1.0, rank_one_hot = play_rank_one_hot[0])
      gfx_commands = []
      fill_gfx_commands_for_heatmap(gfx_commands, moves_and_probs, board, normalization_div=None, is_percent=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "policy-half-history":
      moves_and_probs = get_moves_and_probs(session, board, moves, use_history_prop=0.5, rank_one_hot = play_rank_one_hot[0])
      gfx_commands = []
      fill_gfx_commands_for_heatmap(gfx_commands, moves_and_probs, board, normalization_div=None, is_percent=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "policy-no-history":
      moves_and_probs = get_moves_and_probs(session, board, moves, use_history_prop=0.0, rank_one_hot = play_rank_one_hot[0])
      gfx_commands = []
      fill_gfx_commands_for_heatmap(gfx_commands, moves_and_probs, board, normalization_div=None, is_percent=True)
      ret = "\n".join(gfx_commands)
    elif command[0] in rank_policy_command_lookup:
      rank_one_hot = rank_policy_command_lookup[command[0]]
      moves_and_probs = get_moves_and_probs(session, board, moves, use_history_prop=1.0, rank_one_hot = rank_one_hot)
      gfx_commands = []
      fill_gfx_commands_for_heatmap(gfx_commands, moves_and_probs, board, normalization_div=None, is_percent=True)
      ret = "\n".join(gfx_commands)
    elif command[0] in layer_command_lookup:
      (layer,channel,normalization_div) = layer_command_lookup[command[0]]
      locs_and_values = get_layer_values(session, board, moves, layer, channel)
      gfx_commands = []
      fill_gfx_commands_for_heatmap(gfx_commands, locs_and_values, board, normalization_div, is_percent=False)
      ret = "\n".join(gfx_commands)

    elif command[0] in input_feature_command_lookup:
      (feature_idx,normalization_div) = input_feature_command_lookup[command[0]]
      locs_and_values = get_input_feature(board, moves, feature_idx)
      gfx_commands = []
      fill_gfx_commands_for_heatmap(gfx_commands, locs_and_values, board, normalization_div, is_percent=False)
      ret = "\n".join(gfx_commands)

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



