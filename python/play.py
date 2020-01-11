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
import common

description = """
Play go with a trained neural net!
Implements a basic GTP engine that uses the neural net directly to play moves.
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
name_scope = args["name_scope"]

#Hardcoded max board size
pos_len = 19

# Model ----------------------------------------------------------------

with open(model_config_json) as f:
  model_config = json.load(f)

if name_scope is not None:
  with tf.compat.v1.variable_scope(name_scope):
    model = Model(model_config,pos_len,{})
else:
  model = Model(model_config,pos_len,{})
policy0_output = tf.nn.softmax(model.policy_output[:,:,0])
policy1_output = tf.nn.softmax(model.policy_output[:,:,1])
value_output = tf.nn.softmax(model.value_output)
scoremean_output = 20.0 * model.miscvalues_output[:,0]
scorestdev_output = 20.0 * tf.math.softplus(model.miscvalues_output[:,1])
lead_output = 20.0 * model.miscvalues_output[:,2]
vtime_output = 150.0 * tf.math.softplus(model.miscvalues_output[:,3])
ownership_output = tf.tanh(model.ownership_output)
scoring_output = model.scoring_output
futurepos_output = tf.tanh(model.futurepos_output)
seki_output = tf.nn.softmax(model.seki_output[:,:,:,0:3])
seki_output = seki_output[:,:,:,1] - seki_output[:,:,:,2]
seki_output2 = tf.sigmoid(model.seki_output[:,:,:,3])
scorebelief_output = tf.nn.softmax(model.scorebelief_output)
sbscale_output = model.sbscale3_layer

class GameState:
  def __init__(self,board_size):
    self.board_size = board_size
    self.board = Board(size=board_size)
    self.moves = []
    self.boards = [self.board.copy()]


# Moves ----------------------------------------------------------------

def fetch_output(session, gs, rules, fetches):
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = gs.board.pla
  opp = Board.get_opp(pla)
  move_idx = len(gs.moves)
  model.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)
  outputs = session.run(fetches, feed_dict={
    model.bin_inputs: bin_input_data,
    model.global_inputs: global_input_data,
    model.symmetries: [False,False,False],
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  return [output[0] for output in outputs]

def get_outputs(session, gs, rules):
  [policy0,
   policy1,
   value,
   scoremean,
   scorestdev,
   lead,
   vtime,
   ownership,
   scoring,
   futurepos,
   seki,
   seki2,
   scorebelief,
   sbscale
  ] = fetch_output(session,gs,rules,[
    policy0_output,
    policy1_output,
    value_output,
    scoremean_output,
    scorestdev_output,
    lead_output,
    vtime_output,
    ownership_output,
    scoring_output,
    futurepos_output,
    seki_output,
    seki_output2,
    scorebelief_output,
    sbscale_output
  ])
  board = gs.board

  moves_and_probs0 = []
  for i in range(len(policy0)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy0)-1:
      moves_and_probs0.append((Board.PASS_LOC,policy0[i]))
    elif board.would_be_legal(board.pla,move):
      moves_and_probs0.append((move,policy0[i]))

  moves_and_probs1 = []
  for i in range(len(policy1)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy1)-1:
      moves_and_probs1.append((Board.PASS_LOC,policy1[i]))
    elif board.would_be_legal(board.pla,move):
      moves_and_probs1.append((move,policy1[i]))

  ownership_flat = ownership.reshape([model.pos_len * model.pos_len])
  ownership_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        ownership_by_loc.append((loc,ownership_flat[pos]))
      else:
        ownership_by_loc.append((loc,-ownership_flat[pos]))

  scoring_flat = scoring.reshape([model.pos_len * model.pos_len])
  scoring_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        scoring_by_loc.append((loc,scoring_flat[pos]))
      else:
        scoring_by_loc.append((loc,-scoring_flat[pos]))

  futurepos0_flat = futurepos[:,:,0].reshape([model.pos_len * model.pos_len])
  futurepos0_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        futurepos0_by_loc.append((loc,futurepos0_flat[pos]))
      else:
        futurepos0_by_loc.append((loc,-futurepos0_flat[pos]))

  futurepos1_flat = futurepos[:,:,1].reshape([model.pos_len * model.pos_len])
  futurepos1_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        futurepos1_by_loc.append((loc,futurepos1_flat[pos]))
      else:
        futurepos1_by_loc.append((loc,-futurepos1_flat[pos]))

  seki_flat = seki.reshape([model.pos_len * model.pos_len])
  seki_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        seki_by_loc.append((loc,seki_flat[pos]))
      else:
        seki_by_loc.append((loc,-seki_flat[pos]))

  seki_flat2 = seki2.reshape([model.pos_len * model.pos_len])
  seki_by_loc2 = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      seki_by_loc2.append((loc,seki_flat2[pos]))

  moves_and_probs = sorted(moves_and_probs0, key=lambda moveandprob: moveandprob[1], reverse=True)
  #Generate a random number biased small and then find the appropriate move to make
  #Interpolate from moving uniformly to choosing from the triangular distribution
  alpha = 1
  beta = 1 + math.sqrt(max(0,len(gs.moves)-20))
  r = np.random.beta(alpha,beta)
  probsum = 0.0
  i = 0
  genmove_result = Board.PASS_LOC
  while True:
    (move,prob) = moves_and_probs[i]
    probsum += prob
    if i >= len(moves_and_probs)-1 or probsum > r:
      genmove_result = move
      break
    i += 1

  return {
    "policy0": policy0,
    "policy1": policy1,
    "moves_and_probs0": moves_and_probs0,
    "moves_and_probs1": moves_and_probs1,
    "value": value,
    "scoremean": scoremean,
    "scorestdev": scorestdev,
    "lead": lead,
    "vtime": vtime,
    "ownership": ownership,
    "ownership_by_loc": ownership_by_loc,
    "scoring": scoring,
    "scoring_by_loc": scoring_by_loc,
    "futurepos": futurepos,
    "futurepos0_by_loc": futurepos0_by_loc,
    "futurepos1_by_loc": futurepos1_by_loc,
    "seki": seki,
    "seki_by_loc": seki_by_loc,
    "seki2": seki2,
    "seki_by_loc2": seki_by_loc2,
    "scorebelief": scorebelief,
    "sbscale": sbscale,
    "genmove_result": genmove_result
  }

def get_layer_values(session, gs, rules, layer, channel):
  board = gs.board
  [layer] = fetch_output(session,gs,rules=rules,fetches=[layer])
  layer = layer.reshape([model.pos_len * model.pos_len,-1])
  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      locs_and_values.append((loc,layer[pos,channel]))
  return locs_and_values

def get_input_feature(gs, rules, feature_idx):
  board = gs.board
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(gs.moves)
  model.fill_row_features(board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)

  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      locs_and_values.append((loc,bin_input_data[0,pos,feature_idx]))
  return locs_and_values

def get_pass_alive(board, rules):
  pla = board.pla
  opp = Board.get_opp(pla)
  area = [-1 for i in range(board.arrsize)]
  nonPassAliveStones = False
  safeBigTerritories = True
  unsafeBigTerritories = False
  board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])

  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      locs_and_values.append((loc,area[loc]))
  return locs_and_values


def get_gfx_commands_for_heatmap(locs_and_values, board, normalization_div, is_percent, value_and_score_from=None, hotcold=False):
  gfx_commands = []
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

  if hotcold:
    for (loc,value) in locs_and_values:
      if loc != Board.PASS_LOC:
        value = value / divisor

        if value < 0:
          value = -loose_cap(-value)
        else:
          value = loose_cap(value)

        interpoints = [
          (-1.00,(0,0,0)),
          (-0.85,(15,0,50)),
          (-0.60,(60,0,160)),
          (-0.35,(0,0,255)),
          (-0.15,(0,100,255)),
          ( 0.00,(115,115,115)),
          ( 0.15,(250,45,40)),
          ( 0.25,(255,55,0)),
          ( 0.60,(255,255,20)),
          ( 0.85,(255,255,128)),
          ( 1.00,(255,255,255)),
        ]

        def lerp(p,y0,y1):
          return y0 + p*(y1-y0)

        i = 0
        while i < len(interpoints):
          if value <= interpoints[i][0]:
            break
          i += 1
        i -= 1

        if i < 0:
          (r,g,b) = interpoints[0][1]
        if i >= len(interpoints)-1:
          (r,g,b) = interpoints[len(interpoints)-1][1]

        p = (value - interpoints[i][0]) / (interpoints[i+1][0] - interpoints[i][0])

        (r0,g0,b0) = interpoints[i][1]
        (r1,g1,b1) = interpoints[i+1][1]
        r = lerp(p,r0,r1)
        g = lerp(p,g0,g1)
        b = lerp(p,b0,b1)

        r = ("%02x" % int(r))
        g = ("%02x" % int(g))
        b = ("%02x" % int(b))
        gfx_commands.append("COLOR #%s%s%s %s" % (r,g,b,str_coord(loc,board)))

  else:
    for (loc,value) in locs_and_values:
      if loc != Board.PASS_LOC:
        value = value / divisor
        if value < 0:
          value = -value
          huestart = 0.50
          huestop = 0.86
        else:
          huestart = -0.02
          huestop = 0.38

        value = loose_cap(value)

        def lerp(p,x0,x1,y0,y1):
          return y0 + (y1-y0) * (p-x0)/(x1-x0)

        if value <= 0.03:
          hue = huestart
          lightness = 0.00 + 0.50 * (value / 0.03)
          saturation = value / 0.03
          (r,g,b) = colorsys.hls_to_rgb((hue+1)%1, lightness, saturation)
        elif value <= 0.60:
          hue = lerp(value,0.03,0.60,huestart,huestop)
          val = 1.0
          saturation = 1.0
          (r,g,b) = colorsys.hsv_to_rgb((hue+1)%1, val, saturation)
        else:
          hue = huestop
          lightness = lerp(value,0.60,1.00,0.5,0.95)
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
  texts_value = []
  maxlen_per_side = 1000
  if len(locs_and_values) > 0 and locs_and_values[0][1] < 0:
    maxlen_per_side = 500

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

  if value_and_score_from is not None:
    value = value_and_score_from["value"]
    score = value_and_score_from["scoremean"]
    lead = value_and_score_from["lead"]
    vtime = value_and_score_from["vtime"]
    texts_value.append("wv %.2fc nr %.2f%% ws %.1f wl %.1f vt %.1f" % (
      100*(value[0]-value[1] if board.pla == Board.WHITE else value[1] - value[0]),
      100*value[2],
      (score if board.pla == Board.WHITE else -score),
      (lead if board.pla == Board.WHITE else -lead),
      vtime
    ))

  gfx_commands.append("TEXT " + ", ".join(texts_value + texts_rev + texts))
  return gfx_commands

def print_scorebelief(gs,outputs):
  board = gs.board
  scorebelief = outputs["scorebelief"]
  scoremean = outputs["scoremean"]
  scorestdev = outputs["scorestdev"]
  sbscale = outputs["sbscale"]

  scorebelief = list(scorebelief)
  if board.pla != Board.WHITE:
    scorebelief.reverse()
    scoremean = -scoremean

  scoredistrmid = pos_len * pos_len + Model.EXTRA_SCORE_DISTR_RADIUS
  ret = ""
  ret += "TEXT "
  ret += "SBScale: " + str(sbscale) + "\n"
  ret += "ScoreBelief: \n"
  for i in range(17,-1,-1):
    ret += "TEXT "
    ret += "%+6.1f" %(-(i*20+0.5))
    for j in range(20):
      idx = scoredistrmid-(i*20+j)-1
      ret += " %4.0f" % (scorebelief[idx] * 10000)
    ret += "\n"
  for i in range(18):
    ret += "TEXT "
    ret += "%+6.1f" %((i*20+0.5))
    for j in range(20):
      idx = scoredistrmid+(i*20+j)
      ret += " %4.0f" % (scorebelief[idx] * 10000)
    ret += "\n"

  beliefscore = 0
  beliefscoresq = 0
  beliefwin = 0
  belieftotal = 0
  for idx in range(scoredistrmid*2):
    score = idx-scoredistrmid+0.5
    if score > 0:
      beliefwin += scorebelief[idx]
    else:
      beliefwin -= scorebelief[idx]
    belieftotal += scorebelief[idx]
    beliefscore += score*scorebelief[idx]
    beliefscoresq += score*score*scorebelief[idx]

  beliefscoremean = beliefscore/belieftotal
  beliefscoremeansq = beliefscoresq/belieftotal
  beliefscorevar = max(0,beliefscoremeansq-beliefscoremean*beliefscoremean)
  beliefscorestdev = math.sqrt(beliefscorevar)

  ret += "TEXT BeliefWin: %.2fc\n" % (100*beliefwin/belieftotal)
  ret += "TEXT BeliefScoreMean: %.1f\n" % (beliefscoremean)
  ret += "TEXT BeliefScoreStdev: %.1f\n" % (beliefscorestdev)
  ret += "TEXT ScoreMean: %.1f\n" % (scoremean)
  ret += "TEXT ScoreStdev: %.1f\n" % (scorestdev)
  return ret


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
    'showboard',
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
    'setrule',
    'policy',
    'policy1',
    'logpolicy',
    'ownership',
    'scoring',
    'futurepos0',
    'futurepos1',
    'seki',
    'seki2',
    'scorebelief',
    'passalive',
  ]
  known_analyze_commands = [
    'gfx/Policy/policy',
    'gfx/Policy1/policy1',
    'gfx/LogPolicy/logpolicy',
    'gfx/Ownership/ownership',
    'gfx/Scoring/scoring',
    'gfx/FuturePos0/futurepos0',
    'gfx/FuturePos1/futurepos1',
    'gfx/Seki/seki',
    'gfx/Seki2/seki2',
    'gfx/ScoreBelief/scorebelief',
    'gfx/PassAlive/passalive',
  ]

  board_size = 19
  gs = GameState(board_size)

  rules = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5
  }

  layerdict = dict(model.outputs_by_layer)
  weightdict = dict()
  for v in tf.trainable_variables():
    weightdict[v.name] = v

  layer_command_lookup = dict()


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

  def add_layer_visualizations(layer_name, normalization_div):
    if layer_name in layerdict:
      layer = layerdict[layer_name]
      add_extra_board_size_visualizations(layer_name, layer, normalization_div)

  add_layer_visualizations("conv1",normalization_div=6)
  add_layer_visualizations("rconv1",normalization_div=14)
  add_layer_visualizations("rconv2",normalization_div=20)
  add_layer_visualizations("rconv3",normalization_div=26)
  add_layer_visualizations("rconv4",normalization_div=36)
  add_layer_visualizations("rconv5",normalization_div=40)
  add_layer_visualizations("rconv6",normalization_div=40)
  add_layer_visualizations("rconv7",normalization_div=44)
  add_layer_visualizations("rconv7/conv1a",normalization_div=12)
  add_layer_visualizations("rconv7/conv1b",normalization_div=12)
  add_layer_visualizations("rconv8",normalization_div=48)
  add_layer_visualizations("rconv9",normalization_div=52)
  add_layer_visualizations("rconv10",normalization_div=55)
  add_layer_visualizations("rconv11",normalization_div=58)
  add_layer_visualizations("rconv11/conv1a",normalization_div=12)
  add_layer_visualizations("rconv11/conv1b",normalization_div=12)
  add_layer_visualizations("rconv12",normalization_div=58)
  add_layer_visualizations("rconv13",normalization_div=64)
  add_layer_visualizations("rconv14",normalization_div=66)
  add_layer_visualizations("g1",normalization_div=6)
  add_layer_visualizations("p1",normalization_div=2)
  add_layer_visualizations("v1",normalization_div=4)

  input_feature_command_lookup = dict()
  def add_input_feature_visualizations(layer_name, feature_idx, normalization_div):
    command_name = layer_name
    command_name = command_name.replace("/",":")
    known_commands.append(command_name)
    known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
    input_feature_command_lookup[command_name] = (feature_idx,normalization_div)

  for i in range(model.bin_input_shape[1]):
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
    linear,
    -linear,
    tf.zeros([19],dtype=tf.float32)
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
      if int(command[1]) > model.pos_len:
        print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
        ret = None
      board_size = int(command[1])
      gs = GameState(board_size)
    elif command[0] == "clear_board":
      gs = GameState(board_size)
    elif command[0] == "showboard":
      ret = "\n" + gs.board.to_string().strip()
    elif command[0] == "komi":
      rules["whiteKomi"] = float(command[1])
    elif command[0] == "play":
      pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
      loc = parse_coord(command[2],gs.board)
      gs.board.play(pla,loc)
      gs.moves.append((pla,loc))
      gs.boards.append(gs.board.copy())
    elif command[0] == "genmove":
      outputs = get_outputs(session, gs, rules)
      loc = outputs["genmove_result"]
      pla = gs.board.pla

      if len(command) > 1:
        pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
      gs.board.play(pla,loc)
      gs.moves.append((pla,loc))
      gs.boards.append(gs.board.copy())
      ret = str_coord(loc,gs.board)

    elif command[0] == "name":
      ret = 'KataGo Raw Neural Net Debug/Test Script'
    elif command[0] == "version":
      ret = '1.0'
    elif command[0] == "list_commands":
      ret = '\n'.join(known_commands)
    elif command[0] == "known_command":
      ret = 'true' if command[1] in known_commands else 'false'
    elif command[0] == "gogui-analyze_commands":
      ret = '\n'.join(known_analyze_commands)
    elif command[0] == "setrule":
      ret = ""
      if command[1] == "korule":
        rules["koRule"] = command[2].upper()
      elif command[1] == "scoringrule":
        rules["scoringRule"] = command[2].upper()
      elif command[1] == "taxrule":
        rules["taxRule"] = command[2].upper()
      elif command[1] == "multistonesuicidelegal":
        rules["multiStoneSuicideLegal"] = (command[2].lower() == "true")
      elif command[1] == "hasbutton":
        rules["hasButton"] = (command[2].lower() == "true")
      elif command[1] == "encorephase":
        rules["encorePhase"] = int(command[2])
      elif command[1] == "passwouldendphase":
        rules["passWouldEndPhase"] = (command[2].lower() == "true")
      elif command[1] == "whitekomi" or command[1] == "komi":
        rules["whiteKomi"] = float(command[2])
      elif command[1] == "asym":
        rules["asymPowersOfTwo"] = float(command[2])
      else:
        ret = "Unknown rules setting"
    elif command[0] == "policy":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["moves_and_probs0"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=outputs)
      ret = "\n".join(gfx_commands)
    elif command[0] == "policy1":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["moves_and_probs1"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=outputs)
      ret = "\n".join(gfx_commands)
    elif command[0] == "logpolicy":
      outputs = get_outputs(session, gs, rules)
      moves_and_logprobs = [(move,max(0.0,4.9+math.log10(prob))) for (move,prob) in outputs["moves_and_probs0"]]
      gfx_commands = get_gfx_commands_for_heatmap(moves_and_logprobs, gs.board, normalization_div=6, is_percent=False, value_and_score_from=outputs)
      ret = "\n".join(gfx_commands)
    elif command[0] == "ownership":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["ownership_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "scoring":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["scoring_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "futurepos0":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["futurepos0_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "futurepos1":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["futurepos1_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "seki":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["seki_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None)
      ret = "\n".join(gfx_commands)
    elif command[0] == "seki2":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["seki_by_loc2"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None)
      ret = "\n".join(gfx_commands)
    elif command[0] in layer_command_lookup:
      (layer,channel,normalization_div) = layer_command_lookup[command[0]]
      locs_and_values = get_layer_values(session, gs, rules, layer, channel)
      gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div, is_percent=False)
      ret = "\n".join(gfx_commands)

    elif command[0] in input_feature_command_lookup:
      (feature_idx,normalization_div) = input_feature_command_lookup[command[0]]
      locs_and_values = get_input_feature(gs, rules, feature_idx)
      gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div, is_percent=False)
      ret = "\n".join(gfx_commands)

    elif command[0] == "passalive":
      locs_and_values = get_pass_alive(gs.board, rules)
      gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div=None, is_percent=False)
      ret = "\n".join(gfx_commands)

    elif command[0] == "scorebelief":
      outputs = get_outputs(session, gs, rules)
      ret = print_scorebelief(gs,outputs)

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
  saver.restore(session, model_variables_prefix)
  run_gtp(session)
