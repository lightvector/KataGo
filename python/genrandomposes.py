#!/usr/bin/python

import argparse
import os
import re
import json
import random
import math
import sys
from pathlib import Path
from typing import List

import sgfmill
import sgfmill.sgf
import sgfmill.sgf_moves
from query_analysis_engine_example import KataGo, sgfmill_to_str
from katago.game.board import Board

def is_fair_enough_pos(board: Board, katago: KataGo, initial_player: int, score_rejection_pickiness: float) -> bool:
    query = {}
    query["id"] = ""
    query["moves"] = []
    query["initialStones"] = []
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x,y)
            pla = board.board[loc]
            if pla != 0:
                query["initialStones"].append((("B" if pla == 1 else "W"), sgfmill_to_str((y,x))))

    query["rules"] = "Japanese"
    query["initialPlayer"] = ("B" if initial_player == 1 else "W")
    query["komi"] = 7.0
    query["boardXSize"] = board.size
    query["boardYSize"] = board.size
    query["maxVisits"] = 400

    result = katago.query_raw(query)
    winrate = result['rootInfo']['winrate']
    score = result['rootInfo']['scoreLead']

    if abs(score) > 150:
        return False
    keep_prob = 1.0 / (1.0 + score_rejection_pickiness * abs(score))
    if random.random() < keep_prob:
        # print(board.to_string())
        # print(query["initialPlayer"])
        print(winrate, score)
        sys.stdout.flush()
        return True
    return False

def randint_exponential(scale):
    r = 0
    while r <= 0:
        r = random.random()
    return int(math.floor(-math.log(r) * scale))

def maybe_generate_one_pos(katago, score_rejection_pickiness, out):
    board = Board(19)
    both_plays = 1 + randint_exponential(5) + randint_exponential(5) + randint_exponential(12)
    extra_b_plays = randint_exponential(1.5)
    extra_w_plays = randint_exponential(1)

    plays = []
    for _ in range(both_plays+extra_b_plays):
        plays.append(1)
    for _ in range(both_plays+extra_w_plays):
        plays.append(2)
    random.shuffle(plays)

    if len(plays) > 140:
        return False

    for pla in plays:
        choices = []
        weights = []
        for y in range(board.size):
            for x in range(board.size):
                line = min(y+1,x+1,19-y,19-x)
                if line <= 1:
                    relprob = 1
                elif line <= 2:
                    relprob = 4
                else:
                    relprob = 20
                choices.append((x,y))
                weights.append(relprob)
        (x,y) = random.choices(choices,weights=weights,k=1)[0]
        loc = board.loc(x,y)
        if board.would_be_legal(pla,loc):
            board.play(pla,loc)

    initial_pla = random.choice([1,2])
    if is_fair_enough_pos(board, katago, initial_pla, score_rejection_pickiness):
        to_write = {}
        to_write["board"] = ""
        num_stones = 0
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x,y)
                if board.board[loc] == 1:
                    to_write["board"] += "X"
                    num_stones += 1
                elif board.board[loc] == 2:
                    to_write["board"] += "O"
                    num_stones += 1
                else:
                    to_write["board"] += "."
            to_write["board"] += "/"
        to_write["hintLoc"] = "null"
        to_write["nextPla"] = ("B" if initial_pla == 1 else "W")
        to_write["initialTurnNumber"] = num_stones
        to_write["moveLocs"] = []
        to_write["movePlas"] = []
        to_write["weight"] = 1.0
        to_write["xSize"] = board.size
        to_write["ySize"] = board.size
        out.write(json.dumps(to_write) + "\n")
        out.flush()
        return True
    return False

def main(katago_path, config_path, model_path, score_rejection_pickiness, num_to_generate, out_file):
    with open(out_file,"w") as out:

        katago = KataGo(
            katago_path,
            config_path,
            model_path,
            additional_args=["-override-config","numSearchThreadsPerAnalysisThread=8,reportAnalysisWinratesAs=BLACK"]
        )

        num_kept = 0
        while True:
            kept = maybe_generate_one_pos(katago, score_rejection_pickiness, out)
            if kept:
                num_kept += 1
                if num_kept >= num_to_generate:
                    break
                if num_kept % 100 == 0:
                    print(f"Kept {num_kept} so far")
                    sys.stdout.flush()
    katago.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-katago-path", required=True)
    parser.add_argument("-config-path", required=True)
    parser.add_argument("-model-path", required=True)
    parser.add_argument("-score-rejection-pickiness", required=True, type=float)
    parser.add_argument("-num-to-generate", required=True, type=int)
    parser.add_argument("-out-file", required=True)
    args = vars(parser.parse_args())

    main(
        args["katago_path"],
        args["config_path"],
        args["model_path"],
        args["score_rejection_pickiness"],
        args["num_to_generate"],
        args["out_file"]
    )
