#!/usr/bin/python

import argparse
import os
import re
import json
import random
import math
import sys
import threading
from pathlib import Path
from typing import List

import sgfmill
import sgfmill.sgf
import sgfmill.sgf_moves
from query_analysis_engine_example import KataGo, sgfmill_to_str
from katago.game.board import Board

def make_query(query_id: str, board: Board, initial_player: int, visits: int) -> dict:
    query = {}
    query["id"] = query_id
    query["moves"] = []
    query["initialStones"] = []
    for y in range(board.y_size):
        for x in range(board.x_size):
            loc = board.loc(x,y)
            pla = board.board[loc]
            if pla != 0:
                # Board rows count from the top, sgfmill rows count from the bottom.
                query["initialStones"].append((("B" if pla == 1 else "W"), sgfmill_to_str((board.y_size-1-y,x))))

    query["rules"] = "Japanese"
    query["initialPlayer"] = ("B" if initial_player == 1 else "W")
    query["komi"] = 7.0
    query["boardXSize"] = board.x_size
    query["boardYSize"] = board.y_size
    query["maxVisits"] = visits
    return query

def query_batch(katago: KataGo, queries: List[dict]) -> dict:
    """Sends all queries at once and returns the results keyed by query id.
    The analysis engine answers in completion order, not submission order. A query that fails maps to None."""
    # Write from a separate thread while this thread reads. Writing everything before reading can deadlock once
    # katago's stdout pipe fills up, since katago then stops taking input until its output is read.
    def write_all():
        for query in queries:
            katago.katago.stdin.write((json.dumps(query) + "\n").encode())
        katago.katago.stdin.flush()
    writer = threading.Thread(target=write_all)
    writer.start()

    results = {}
    while len(results) < len(queries):
        line = katago.katago.stdout.readline()
        if not line:
            raise Exception("Unexpected katago exit")
        line = line.decode().strip()
        if line == "":
            continue
        response = json.loads(line)
        if "warning" in response:
            print("Warning from katago: " + line)
            continue
        if "error" in response:
            print("Error from katago: " + line)
            results[response.get("id")] = None
            continue
        results[response["id"]] = response
    writer.join()
    return results

def is_fair_enough_result(result: dict, score_rejection_pickiness: float) -> bool:
    if result is None:
        return False
    winrate = result['rootInfo']['winrate']
    score = result['rootInfo']['scoreLead']

    if abs(score) > 150:
        return False
    keep_prob = 1.0 / (1.0 + score_rejection_pickiness * abs(score))
    if random.random() < keep_prob:
        print(winrate, score)
        sys.stdout.flush()
        return True
    return False

def randint_exponential(scale):
    r = 0
    while r <= 0:
        r = random.random()
    return int(math.floor(-math.log(r) * scale))

def generate_candidate():
    """Returns a random (board, initial_pla), or None if this attempt is discarded for having too many plays."""
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
        return None

    for pla in plays:
        choices = []
        weights = []
        for y in range(board.y_size):
            for x in range(board.x_size):
                line = min(y+1,x+1,board.y_size-y,board.x_size-x)
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
    return (board, initial_pla)

def write_pos(board: Board, initial_pla: int, out):
    to_write = {}
    to_write["board"] = ""
    num_stones = 0
    for y in range(board.y_size):
        for x in range(board.x_size):
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
    to_write["xSize"] = board.x_size
    to_write["ySize"] = board.y_size
    out.write(json.dumps(to_write) + "\n")

def main(katago_path, config_path, model_path, score_rejection_pickiness, num_to_generate, out_file, visits, batch_size):
    with open(out_file,"w") as out:

        # One search thread per query, with batch_size queries in parallel, keeps the search for each position the
        # same as a single-threaded search while still giving the GPU large batches.
        overrides = ",".join([
            "numAnalysisThreads=" + str(batch_size),
            "numSearchThreadsPerAnalysisThread=1",
            "nnMaxBatchSize=" + str(batch_size),
            "reportAnalysisWinratesAs=BLACK",
        ])
        katago = KataGo(
            katago_path,
            config_path,
            model_path,
            additional_args=["-override-config",overrides]
        )

        num_kept = 0
        num_queried = 0
        while num_kept < num_to_generate:
            candidates = {}
            while len(candidates) < batch_size:
                candidate = generate_candidate()
                if candidate is not None:
                    candidates[str(num_queried + len(candidates))] = candidate
            queries = [make_query(query_id, board, initial_pla, visits) for query_id, (board, initial_pla) in candidates.items()]
            results = query_batch(katago, queries)
            num_queried += len(queries)

            # Process in submission order so the kept positions do not depend on katago's completion order.
            for query_id, (board, initial_pla) in candidates.items():
                if is_fair_enough_result(results.get(query_id), score_rejection_pickiness):
                    write_pos(board, initial_pla, out)
                    num_kept += 1
                    if num_kept % 100 == 0:
                        print(f"Kept {num_kept} so far, queried {num_queried}")
                        sys.stdout.flush()
                    if num_kept >= num_to_generate:
                        break
            out.flush()
    katago.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-katago-path", required=True)
    parser.add_argument("-config-path", required=True)
    parser.add_argument("-model-path", required=True)
    parser.add_argument("-score-rejection-pickiness", required=True, type=float)
    parser.add_argument("-num-to-generate", required=True, type=int)
    parser.add_argument("-out-file", required=True)
    parser.add_argument("-visits", type=int, default=400, help="Visits per position for the fairness check")
    parser.add_argument("-batch-size", type=int, default=256, help="Number of positions queried in parallel")
    args = vars(parser.parse_args())

    main(
        args["katago_path"],
        args["config_path"],
        args["model_path"],
        args["score_rejection_pickiness"],
        args["num_to_generate"],
        args["out_file"],
        args["visits"],
        args["batch_size"],
    )
