"""
This is a simple python program that demonstrates how to run KataGo's
analysis engine as a subprocess and send it a query. It queries the
result of playing the 4-4 point on an empty board and prints out
the json response.
"""

import argparse
import json
import subprocess
import time
from threading import Thread
import sgfmill
import sgfmill.boards
import sgfmill.ascii_boards
from typing import Tuple, List, Optional, Union, Literal

Color = Union[Literal["b"],Literal["w"]]
Move = Union[Literal["pass"],Tuple[int,int]]

def sgfmill_to_str(move: Move) -> str:
  if move == "pass":
    return "pass"
  (y,x) = move
  return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x] + str(y+1)

class KataGo:

  def __init__(self, katago_path: str, config_path: str, model_path: str):
    self.query_counter = 0
    katago = subprocess.Popen(
      [katago_path, "analysis", "-config", config_path, "-model", model_path],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    self.katago = katago
    def printforever():
      while katago.poll() is None:
        data = katago.stderr.readline()
        time.sleep(0)
        if data:
          print("KataGo: ", data.decode(), end="")
      data = katago.stderr.read()
      if data:
        print("KataGo: ", data.decode(), end="")
    self.stderrthread = Thread(target=printforever)
    self.stderrthread.start()

  def close(self):
    self.katago.stdin.close()


  def query(self, initial_board: sgfmill.boards.Board, moves: List[Tuple[Color,Move]], komi: float, max_visits=None):
    query = {}

    query["id"] = str(self.query_counter)
    self.query_counter += 1

    query["moves"] = [(color,sgfmill_to_str(move)) for color, move in moves]
    query["initialStones"] = []
    for y in range(initial_board.side):
      for x in range(initial_board.side):
        color = initial_board.get(y,x)
        if color:
          query["initialStones"].append((color,sgfmill_to_str((y,x))))
    query["rules"] = "Chinese"
    query["komi"] = komi
    query["boardXSize"] = initial_board.side
    query["boardYSize"] = initial_board.side
    query["includePolicy"] = True
    if max_visits is not None:
      query["maxVisits"] = max_visits

    self.katago.stdin.write((json.dumps(query) + "\n").encode())
    self.katago.stdin.flush()

    # print(json.dumps(query))

    line = ""
    while line == "":
      if self.katago.poll():
        time.sleep(1)
        raise Exception("Unexpected katago exit")
      line = self.katago.stdout.readline()
      line = line.decode().strip()
      # print("Got: " + line)
    response = json.loads(line)

    # print(response)
    return response

if __name__ == "__main__":
  description = """
  Example script showing how to run KataGo analysis engine and query it from python.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
    "-katago-path",
    help="Path to katago executable",
    required=True,
  )
  parser.add_argument(
    "-config-path",
    help="Path to KataGo analysis config (e.g. cpp/configs/analysis_example.cfg in KataGo repo)",
    required=True,
  )
  parser.add_argument(
    "-model-path",
    help="Path to neural network .bin.gz file",
    required=True,
  )
  args = vars(parser.parse_args())
  print(args)

  katago = KataGo(args["katago_path"], args["config_path"], args["model_path"])

  board = sgfmill.boards.Board(19)
  komi = 6.5
  moves = [("b",(3,3))]

  displayboard = board.copy()
  for color, move in moves:
    if move != "pass":
      row,col = move
      displayboard.play(row,col,color)
  print(sgfmill.ascii_boards.render_board(displayboard))

  print("Query result: ")
  print(katago.query(board, moves, komi))

  katago.close()


