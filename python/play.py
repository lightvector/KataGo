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
import numpy as np

from katago.game.board import Board
from katago.game.features import Features
from katago.game.gamestate import GameState

from typing import Dict, Any, List

import torch
import torch.nn

from katago.train import modelconfigs
from katago.train.model_pytorch import Model, EXTRA_SCORE_DISTR_RADIUS, ExtraOutputs
from katago.train.data_processing_pytorch import apply_symmetry
from katago.train.load_model import load_model

description = """
Play go with a trained neural net!
Implements a basic GTP engine that uses the neural net directly to play moves.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint to test', required=False)
parser.add_argument('-use-swa', help='Use SWA model', action="store_true", required=False)
parser.add_argument('-device', help='Pytorch device, like cpu or cuda:0', required=False)

args = vars(parser.parse_args())

checkpoint_file = args["checkpoint"]
use_swa = args["use_swa"]
device = args["device"]

# Hardcoded max board size
pos_len = 19

# Model ----------------------------------------------------------------

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stderr),
    ],
)
np.set_printoptions(linewidth=150)
torch.set_printoptions(precision=7,sci_mode=False,linewidth=100000,edgeitems=1000,threshold=1000000)

model, swa_model, _ = load_model(checkpoint_file, use_swa, device=device, pos_len=pos_len, verbose=True)
if swa_model is not None:
    model = swa_model
model_config = model.config
model.eval()

features = Features(model_config, pos_len)


# Moves ----------------------------------------------------------------

def get_input_feature(gs, feature_idx):
    bin_input_data, global_input_data = gs.get_input_features(features)
    locs_and_values = []
    for y in range(gs.board.y_size):
        for x in range(gs.board.x_size):
            loc = board.loc(x,y)
            locs_and_values.append((loc,bin_input_data[0,feature_idx,y,x]))
    return locs_and_values

def get_pass_alive(gs):
    board = gs.board
    rules = gs.rules
    pla = board.pla
    opp = Board.get_opp(pla)
    area = [-1 for i in range(board.arrsize)]
    nonPassAliveStones = False
    safeBigTerritories = True
    unsafeBigTerritories = False
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])

    locs_and_values = []
    for y in range(board.y_size):
        for x in range(board.x_size):
            loc = board.loc(x,y)
            locs_and_values.append((loc,area[loc]))
    return locs_and_values


def get_gfx_commands_for_heatmap(locs_and_values, board, normalization_div, is_percent, value_and_score_from=None, hotcold=False):
    gfx_commands = []
    divisor = 1.0
    if normalization_div == "max":
        max_abs_value = max(abs(value) for (loc,value) in locs_and_values)
        divisor = max(0.0000000001,max_abs_value) # avoid divide by zero
    elif normalization_div is not None:
        divisor = normalization_div

    # Caps value at 1.0, using an asymptotic curve
    def loose_cap(x):
        def transformed_softplus(x):
            return -math.log(math.exp(-(x-1.0)*8.0)+1.0)/8.0+1.0
        base = transformed_softplus(0.0)
        return (transformed_softplus(x) - base) / (1.0 - base)

    # Softly curves a value so that it ramps up faster than linear in that range
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
    lead = outputs["lead"]
    scoremean = outputs["scoremean"]
    scorestdev = outputs["scorestdev"]

    scorebelief = list(scorebelief)
    # Flip so that it's in perspective of the player playing
    if board.pla != Board.WHITE:
        scorebelief.reverse()
        scoremean = -scoremean
        lead = -lead

    scoredistrmid = pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS
    ret = ""
    ret += "TEXT "
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
    ret += "TEXT BeliefScoreMean: %.2f\n" % (beliefscoremean)
    ret += "TEXT BeliefScoreStdev: %.2f\n" % (beliefscorestdev)
    ret += "TEXT Lead: %.3f\n" % (lead)
    ret += "TEXT ScoreMean: %.3f\n" % (scoremean)
    ret += "TEXT ScoreStdev: %.3f\n" % (scorestdev)
    ret += "TEXT Value: %s\n" % str(["%.3f" % x for x in outputs["value"]])
    ret += "TEXT TDValue: %s\n" % str(["%.3f" % x for x in outputs["td_value"]])
    ret += "TEXT TDValue2: %s\n" % str(["%.3f" % x for x in outputs["td_value2"]])
    ret += "TEXT TDValue3: %s\n" % str(["%.3f" % x for x in outputs["td_value3"]])
    ret += "TEXT TDScore: %s\n" % str(["%.3f" % x for x in outputs["td_score"]])
    ret += "TEXT Estv: %s\n" % str(outputs["estv"])
    ret += "TEXT Ests: %s\n" % str(outputs["ests"])
    ret += "TEXT Vtime: %s\n" % str(outputs["vtime"])
    return ret


# Basic parsing --------------------------------------------------------
colstr = 'ABCDEFGHJKLMNOPQRST'
def parse_coord(s,board):
    if s == 'pass':
        return Board.PASS_LOC
    return board.loc(colstr.index(s[0].upper()), board.y_size - int(s[1:]))

def str_coord(loc,board):
    if loc == Board.PASS_LOC:
        return 'pass'
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    return '%c%d' % (colstr[x], board.y_size - y)


# GTP Implementation -----------------------------------------------------

# Adapted from https://github.com/pasky/michi/blob/master/michi.py, which is distributed under MIT license
# https://opensource.org/licenses/MIT

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
gs = GameState(board_size, GameState.RULES_TT)

input_feature_command_lookup = dict()
def add_input_feature_visualizations(layer_name, feature_idx, normalization_div):
    command_name = layer_name
    command_name = command_name.replace("/",":")
    known_commands.append(command_name)
    known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
    input_feature_command_lookup[command_name] = (feature_idx,normalization_div)
for i in range(model.bin_input_shape[1]):
    add_input_feature_visualizations("input-" + str(i),i, normalization_div=1)

attention_feature_command_lookup = dict()
def add_attention_visualizations(extra_output_name, extra_output):
    for c in range(extra_output.shape[0]):
        command_name = extra_output_name
        command_name = command_name.replace("/",":")
        command_name += ":" + str(c)
        known_commands.append(command_name)
        known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
        attention_feature_command_lookup[command_name] = (extra_output_name, c)
with torch.no_grad():
    dummy_outputs = gs.get_model_outputs(model)
    extra_attention_output_names = [name for name in dummy_outputs["available_extra_outputs"] if name.endswith(".attention") or name.endswith(".reverse_attention")]
    for name in extra_attention_output_names:
        add_attention_visualizations(name,dummy_outputs[name])
    del dummy_outputs


def get_board_matrix_str(matrix, scale, formatstr):
    ret = ""
    matrix = matrix.reshape([features.pos_len,features.pos_len])
    for y in range(features.pos_len):
        for x in range(features.pos_len):
            ret += formatstr % (scale * matrix[y,x])
            ret += " "
        ret += "\n"
    return ret

def get_policy_matrix_str(matrix, gs, scale, formatstr):
    ret = ""
    for y in range(gs.board.y_size):
        for x in range(gs.board.x_size):
            loc = gs.board.loc(x,y)
            pos = features.loc_to_tensor_pos(loc,gs.board)
            if gs.board.would_be_legal(gs.board.pla,loc):
                ret += formatstr % (scale * matrix[pos])
            else:
                ret += "  -" + (" " * (len(formatstr % (0.0)) - 3))
            ret += " "
        ret += "\n"
    loc = Board.PASS_LOC
    pos = features.loc_to_tensor_pos(loc,gs.board)
    ret += "Pass: " + (formatstr % (scale * matrix[pos]))
    return ret



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
        if int(command[1]) > features.pos_len:
            print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
            ret = None
        board_size = int(command[1])
        if len(command) >= 3:
            x_size = board_size
            y_size = int(command[2])
            board_size = (x_size,y_size)
        gs = GameState(board_size, gs.rules)
    elif command[0] == "clear_board":
        gs = GameState(board_size, gs.rules)
    elif command[0] == "showboard":
        ret = "\n" + gs.board.to_string().strip()
    elif command[0] == "komi":
        gs.rules["whiteKomi"] = float(command[1])
    elif command[0] == "play":
        pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
        loc = parse_coord(command[2],gs.board)
        gs.play(pla,loc)
    elif command[0] == "genmove":
        outputs = gs.get_model_outputs(model)
        loc = outputs["genmove_result"]
        pla = gs.board.pla

        if len(command) > 1:
            pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
        gs.play(pla,loc)
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
    elif command[0] == "setrules":
        if command[1].lower() == "chinese":
            gs.rules = GameState.RULES_CHINESE.copy()
        elif command[1].lower() == "japanese":
            gs.rules = GameState.RULES_JAPANESE.copy()
        elif command[1].lower() == "tromp_taylor":
            gs.rules = GameState.RULES_TT.copy()
        else:
            ret = "Unknown rules"
    elif command[0] == "setrule":
        ret = ""
        if command[1] == "korule":
            gs.rules["koRule"] = command[2].upper()
        elif command[1] == "scoringrule":
            gs.rules["scoringRule"] = command[2].upper()
        elif command[1] == "taxrule":
            gs.rules["taxRule"] = command[2].upper()
        elif command[1] == "multistonesuicidelegal":
            gs.rules["multiStoneSuicideLegal"] = (command[2].lower() == "true")
        elif command[1] == "hasbutton":
            gs.rules["hasButton"] = (command[2].lower() == "true")
        elif command[1] == "encorephase":
            gs.rules["encorePhase"] = int(command[2])
        elif command[1] == "passwouldendphase":
            gs.rules["passWouldEndPhase"] = (command[2].lower() == "true")
        elif command[1] == "whitekomi" or command[1] == "komi":
            gs.rules["whiteKomi"] = float(command[2])
        elif command[1] == "asym":
            gs.rules["asymPowersOfTwo"] = float(command[2])
        else:
            ret = "Unknown rules setting"
    elif command[0] == "policy":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["moves_and_probs0"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=outputs)
        ret = "\n".join(gfx_commands)
    elif command[0] == "policy1":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["moves_and_probs1"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=outputs)
        ret = "\n".join(gfx_commands)
    elif command[0] == "logpolicy":
        outputs = gs.get_model_outputs(model)
        moves_and_logprobs = [(move,max(0.0,4.9+math.log10(prob))) for (move,prob) in outputs["moves_and_probs0"]]
        gfx_commands = get_gfx_commands_for_heatmap(moves_and_logprobs, gs.board, normalization_div=6, is_percent=False, value_and_score_from=outputs)
        ret = "\n".join(gfx_commands)
    elif command[0] == "ownership":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["ownership_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
        ret = "\n".join(gfx_commands)
    elif command[0] == "scoring":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["scoring_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
        ret = "\n".join(gfx_commands)
    elif command[0] == "futurepos0":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["futurepos0_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
        ret = "\n".join(gfx_commands)
    elif command[0] == "futurepos1":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["futurepos1_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
        ret = "\n".join(gfx_commands)
    elif command[0] == "seki":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["seki_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None)
        ret = "\n".join(gfx_commands)
    elif command[0] == "seki2":
        outputs = gs.get_model_outputs(model)
        gfx_commands = get_gfx_commands_for_heatmap(outputs["seki_by_loc2"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None)
        ret = "\n".join(gfx_commands)

    elif command[0] == "policy_raw":
        outputs = gs.get_model_outputs(model)
        ret = "\n"

        policysum = 0.0
        for y in range(gs.board.y_size):
            for x in range(gs.board.x_size):
                loc = gs.board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,gs.board)
                gs.board.would_be_legal(gs.board.pla,loc)
                policysum += outputs["policy0"][pos]
        loc = Board.PASS_LOC
        pos = features.loc_to_tensor_pos(loc,gs.board)
        policysum += outputs["policy0"][pos]

        ret += get_policy_matrix_str(outputs["policy0"], gs, 100.0 / policysum, "%6.3f")

    elif command[0] == "policy1_raw":
        outputs = gs.get_model_outputs(model)
        ret = "\n"
        ret += get_policy_matrix_str(outputs["policy1"], gs, 100.0, "%6.3f")

    elif command[0] == "ownership_raw":
        outputs = gs.get_model_outputs(model)
        ret = get_board_matrix_str(outputs["ownership"], 100.0, "%+7.3f")
    elif command[0] == "scoring_raw":
        outputs = gs.get_model_outputs(model)
        ret = get_board_matrix_str(outputs["scoring"], 100.0, "%+7.3f")
    elif command[0] == "futurepos0_raw":
        outputs = gs.get_model_outputs(model)
        ret = get_board_matrix_str(outputs["futurepos"][0], 100.0, "%+7.3f")
    elif command[0] == "futurepos1_raw":
        outputs = gs.get_model_outputs(model)
        ret = get_board_matrix_str(outputs["futurepos"][1], 100.0, "%+7.3f")
    elif command[0] == "seki_raw":
        outputs = gs.get_model_outputs(model)
        ret = get_board_matrix_str(outputs["seki"], 100.0, "%+7.3f")
    elif command[0] == "seki2_raw":
        outputs = gs.get_model_outputs(model)
        ret = get_board_matrix_str(outputs["seki2"], 100.0, "%+7.3f")
    elif command[0] == "qwinloss_raw":
        outputs = gs.get_model_outputs(model)
        ret = "\n"
        ret += get_policy_matrix_str(outputs["qwinloss"], gs, 100.0, "%+7.3fc")
    elif command[0] == "qscore_raw":
        outputs = gs.get_model_outputs(model)
        ret = "\n"
        ret += get_policy_matrix_str(outputs["qscore"], gs, 1.0, "%+6.2f")

    elif command[0] in input_feature_command_lookup:
        (feature_idx,normalization_div) = input_feature_command_lookup[command[0]]
        locs_and_values = get_input_feature(gs, feature_idx)
        gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div, is_percent=False)
        ret = "\n".join(gfx_commands)

    elif command[0] in attention_feature_command_lookup:
        (extra_output_name,channel_idx) = attention_feature_command_lookup[command[0]]
        outputs = gs.get_model_outputs(model, extra_output_names=[extra_output_name])
        output = outputs[extra_output_name] # shape c, hw
        output = output[channel_idx]
        locs_and_values = []
        board = gs.board
        for y in range(board.y_size):
            for x in range(board.x_size):
                loc = board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,board)
                locs_and_values.append((loc,output[pos]))

        normalization_div = "max"
        gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div, is_percent=True)
        ret = "\n".join(gfx_commands)

    elif command[0] == "passalive":
        locs_and_values = get_pass_alive(gs)
        gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div=None, is_percent=False)
        ret = "\n".join(gfx_commands)

    elif command[0] == "scorebelief":
        outputs = gs.get_model_outputs(model)
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
