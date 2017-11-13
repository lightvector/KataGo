#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import tensorflow as tf

from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves

description = """
Train neural net on Go games!
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-weightsdir', help='Dir to write for training weights', required=True)
parser.add_argument('-gamesdir', help='Dir of games to read', required=True)
parser.add_argument('-verbose', help='verbose', required=False, action='store_true')
args = vars(parser.parse_args())

weightsdir = args["weightsdir"]
gamesdir = args["gamesdir"]
verbose = args["verbose"]

def load_sgf_moves(path):
  sgf_file = open(path,"rb")
  contents = sgf_file.read()
  sgf_file.close()
  try:
    game = sgf.Sgf_game.from_bytes(contents)
  except:
    if verbose:
      traceback.print_exc()
    raise Exception("Error parsing sgf file: " + path)

  board, moves = sgf_moves.get_setup_and_moves(game)
  for (color,loc) in moves:
    if loc is not None: # pass
      (row,col) = loc
      board.play(row,col,color)
    print(ascii_boards.render_board(board))

def collect_game_files(gamesdir):
  files = []
  for root, directories, filenames in os.walk(gamesdir):
    for filename in filenames:
      files.append(os.path.join(root,filename))
  return files

print("Collecting games in " + gamesdir)
game_files = collect_game_files(gamesdir)
game_files = [path for path in game_files if path.endswith(".sgf")]
print("Collected %d games" % (len(game_files)))

for path in game_files:
  load_sgf_moves(path)
