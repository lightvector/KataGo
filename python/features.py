import math
import numpy as np
from typing import Callable, List

from board import Board, Loc, Pos, Player
from modelconfigs import ModelConfig

class Features:
  def __init__(self, config: ModelConfig, pos_len: int):
    self.config = config
    self.pos_len = pos_len
    self.version = modelconfigs.get_version(config)
    self.pass_pos = self.pos_len * self.pos_len

  def xy_to_tensor_pos(self,x,y):
    return y * self.pos_len + x
  def loc_to_tensor_pos(self,loc,board):
    assert(loc != Board.PASS_LOC)
    return board.loc_y(loc) * self.pos_len + board.loc_x(loc)

  def tensor_pos_to_loc(self,pos,board):
    if pos == self.pass_pos:
      return None
    pos_len = self.pos_len
    bsize = board.size
    assert(self.pos_len >= bsize)
    x = pos % pos_len
    y = pos // pos_len
    if x < 0 or x >= bsize or y < 0 or y >= bsize:
      return board.loc(-10,-10) #Return an illegal move since this is offboard
    return board.loc(x,y)

  def sym_tensor_pos(self,pos,symmetry):
    if pos == self.pass_pos:
      return pos
    pos_len = self.pos_len
    x = pos % pos_len
    y = pos // pos_len
    if symmetry >= 4:
      symmetry -= 4
      tmp = x
      x = y
      y = tmp
    if symmetry >= 2:
      symmetry -= 2
      x = pos_len-x-1
    if symmetry >= 1:
      symmetry -= 1
      y = pos_len-y-1
    return y * pos_len + x

  #Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
  def iterLadders(self, board, f):
    chainHeadsSolved = {}
    copy = board.copy()

    bsize = board.size
    assert(self.pos_len >= bsize)

    for y in range(bsize):
      for x in range(bsize):
        pos = self.xy_to_tensor_pos(x,y)
        loc = board.loc(x,y)
        stone = board.board[loc]

        if stone == Board.BLACK or stone == Board.WHITE:
          libs = board.num_liberties(loc)
          if libs == 1 or libs == 2:
            head = board.group_head[loc]
            if head in chainHeadsSolved:
              laddered = chainHeadsSolved[head]
              if laddered:
                f(loc,pos,[])
            else:
              #Perform search on copy so as not to mess up tracking of solved heads
              if libs == 1:
                workingMoves = []
                laddered = copy.searchIsLadderCaptured(loc,True)
              else:
                workingMoves = copy.searchIsLadderCapturedAttackerFirst2Libs(loc)
                laddered = len(workingMoves) > 0

              chainHeadsSolved[head] = laddered
              if laddered:
                f(loc,pos,workingMoves)


  #Returns the new idx, which could be the same as idx if this isn't a good training row
  def fill_row_features(self, board, pla, opp, boards, moves, move_idx, rules, bin_input_data, global_input_data, idx):
    assert(self.version == 10)

    bsize = board.size
    assert(self.pos_len >= bsize)
    assert(len(boards) > 0)
    assert(board.zobrist == boards[move_idx].zobrist)

    for y in range(bsize):
      for x in range(bsize):
        pos = self.xy_to_tensor_pos(x,y)
        bin_input_data[idx,pos,0] = 1.0
        loc = board.loc(x,y)
        stone = board.board[loc]
        if stone == pla:
          bin_input_data[idx,pos,1] = 1.0
        elif stone == opp:
          bin_input_data[idx,pos,2] = 1.0

        if stone == pla or stone == opp:
          libs = board.num_liberties(loc)
          if libs == 1:
            bin_input_data[idx,pos,3] = 1.0
          elif libs == 2:
            bin_input_data[idx,pos,4] = 1.0
          elif libs == 3:
            bin_input_data[idx,pos,5] = 1.0

    #Python code does NOT handle superko
    if board.simple_ko_point is not None:
      pos = self.loc_to_tensor_pos(board.simple_ko_point,board)
      bin_input_data[idx,pos,6] = 1.0
    #Python code does NOT handle ko-prohibited encore spots or anything relating to the encore
    #so features 7 and 8 leave that blank

    if move_idx >= 1 and moves[move_idx-1][0] == opp:
      prev1_loc = moves[move_idx-1][1]
      if prev1_loc is not None and prev1_loc != Board.PASS_LOC:
        pos = self.loc_to_tensor_pos(prev1_loc,board)
        bin_input_data[idx,pos,9] = 1.0
      elif prev1_loc == Board.PASS_LOC:
        global_input_data[idx,0] = 1.0

      if move_idx >= 2 and moves[move_idx-2][0] == pla:
        prev2_loc = moves[move_idx-2][1]
        if prev2_loc is not None and prev2_loc != Board.PASS_LOC:
          pos = self.loc_to_tensor_pos(prev2_loc,board)
          bin_input_data[idx,pos,10] = 1.0
        elif prev2_loc == Board.PASS_LOC:
          global_input_data[idx,1] = 1.0

        if move_idx >= 3 and moves[move_idx-3][0] == opp:
          prev3_loc = moves[move_idx-3][1]
          if prev3_loc is not None and prev3_loc != Board.PASS_LOC:
            pos = self.loc_to_tensor_pos(prev3_loc,board)
            bin_input_data[idx,pos,11] = 1.0
          elif prev3_loc == Board.PASS_LOC:
            global_input_data[idx,2] = 1.0

          if move_idx >= 4 and moves[move_idx-4][0] == pla:
            prev4_loc = moves[move_idx-4][1]
            if prev4_loc is not None and prev4_loc != Board.PASS_LOC:
              pos = self.loc_to_tensor_pos(prev4_loc,board)
              bin_input_data[idx,pos,12] = 1.0
            elif prev4_loc == Board.PASS_LOC:
              global_input_data[idx,3] = 1.0

            if move_idx >= 5 and moves[move_idx-5][0] == opp:
              prev5_loc = moves[move_idx-5][1]
              if prev5_loc is not None and prev5_loc != Board.PASS_LOC:
                pos = self.loc_to_tensor_pos(prev5_loc,board)
                bin_input_data[idx,pos,13] = 1.0
              elif prev5_loc == Board.PASS_LOC:
                global_input_data[idx,4] = 1.0

    def addLadderFeature(loc,pos,workingMoves):
      assert(board.board[loc] == Board.BLACK or board.board[loc] == Board.WHITE)
      bin_input_data[idx,pos,14] = 1.0
      if board.board[loc] == opp and board.num_liberties(loc) > 1:
        for workingMove in workingMoves:
          workingPos = self.loc_to_tensor_pos(workingMove,board)
          bin_input_data[idx,workingPos,17] = 1.0

    self.iterLadders(board, addLadderFeature)

    if move_idx > 0:
      prevBoard = boards[move_idx-1]
    else:
      prevBoard = board
    def addPrevLadderFeature(loc,pos,workingMoves):
      assert(prevBoard.board[loc] == Board.BLACK or prevBoard.board[loc] == Board.WHITE)
      bin_input_data[idx,pos,15] = 1.0
    self.iterLadders(prevBoard, addPrevLadderFeature)

    if move_idx > 1:
      prevPrevBoard = boards[move_idx-2]
    else:
      prevPrevBoard = prevBoard
    def addPrevPrevLadderFeature(loc,pos,workingMoves):
      assert(prevPrevBoard.board[loc] == Board.BLACK or prevPrevBoard.board[loc] == Board.WHITE)
      bin_input_data[idx,pos,16] = 1.0
    self.iterLadders(prevPrevBoard, addPrevPrevLadderFeature)

    #Features 18,19 - area
    area = [0 for i in range(board.arrsize)]

    if rules["scoringRule"] == "SCORING_AREA" and rules["taxRule"] == "TAX_NONE":
      safeBigTerritories = True
      nonPassAliveStones = True
      unsafeBigTerritories = True
      board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])
    else:
      hasAreaFeature = False
      keepTerritories = False
      keepStones = False
      if rules["scoringRule"] == "SCORING_AREA" and (rules["taxRule"] == "TAX_SEKI" or rules["taxRule"] == "TAX_ALL"):
        hasAreaFeature = True
        keepTerritories = False
        keepStones = True
      elif rules["scoringRule"] == "SCORING_TERRITORY" and rules["taxRule"] == "TAX_NONE":
        if rules["encorePhase"] >= 2:
          hasAreaFeature = True
          keepTerritories = True
          keepStones = False
      elif rules["scoringRule"] == "SCORING_TERRITORY" and (rules["taxRule"] == "TAX_SEKI" or rules["taxRule"] == "TAX_ALL"):
        if rules["encorePhase"] >= 2:
          hasAreaFeature = True
          keepTerritories = False
          keepStones = False
      else:
        assert(False)

      if hasAreaFeature:
        board.calculateNonDameTouchingArea(
          area,
          keepTerritories,
          keepStones,
          rules["multiStoneSuicideLegal"]
        )

    for y in range(bsize):
      for x in range(bsize):
        loc = board.loc(x,y)
        pos = self.xy_to_tensor_pos(x,y)

        if area[loc] == pla:
          bin_input_data[idx,pos,18] = 1.0
        elif area[loc] == opp:
          bin_input_data[idx,pos,19] = 1.0

    #Features 20,21 - second encore phase starting stones, we just set them to the current stones in pythong
    #since we don't really have a jp rules impl
    if rules["encorePhase"] >= 2:
      for y in range(bsize):
        for x in range(bsize):
          pos = self.xy_to_tensor_pos(x,y)
          loc = board.loc(x,y)
          stone = board.board[loc]
          if stone == pla:
            bin_input_data[idx,pos,20] = 1.0
          elif stone == opp:
            bin_input_data[idx,pos,21] = 1.0


    #Not quite right, japanese rules aren't really implemented in the python
    bArea = board.size * board.size
    whiteKomi = rules["whiteKomi"]
    if rules["scoringRule"] == "SCORING_TERRITORY":
      selfKomi = (whiteKomi+1 if pla == Board.WHITE else -whiteKomi)
    else:
      selfKomi = (whiteKomi if pla == Board.WHITE else -whiteKomi)

    if selfKomi > bArea+1:
      selfKomi = bArea+1
    if selfKomi < -bArea-1:
      selfKomi = -bArea-1
    global_input_data[idx,5] = selfKomi/20.0

    if rules["koRule"] == "KO_SIMPLE":
      pass
    elif rules["koRule"] == "KO_POSITIONAL" or rules["koRule"] == "KO_SPIGHT":
      global_input_data[idx,6] = 1.0
      global_input_data[idx,7] = 0.5
    elif rules["koRule"] == "KO_SITUATIONAL":
      global_input_data[idx,6] = 1.0
      global_input_data[idx,7] = -0.5
    else:
      assert(False)

    if rules["multiStoneSuicideLegal"]:
      global_input_data[idx,8] = 1.0

    if rules["scoringRule"] == "SCORING_AREA":
      pass
    elif rules["scoringRule"] == "SCORING_TERRITORY":
      global_input_data[idx,9] = 1.0
    else:
      assert(False)

    if rules["taxRule"] == "TAX_NONE":
      pass
    elif rules["taxRule"] == "TAX_SEKI":
      global_input_data[idx,10] = 1.0
    elif rules["taxRule"] == "TAX_ALL":
      global_input_data[idx,10] = 1.0
      global_input_data[idx,11] = 1.0
    else:
      assert(False)

    if rules["encorePhase"] > 0:
      global_input_data[idx,12] = 1.0
    if rules["encorePhase"] > 1:
      global_input_data[idx,13] = 1.0
    passWouldEndPhase = rules["passWouldEndPhase"]
    global_input_data[idx,14] = (1.0 if passWouldEndPhase else 0.0)

    global_input_data[idx,15] = 1.0
    global_input_data[idx,16] = rules["asymPowersOfTwo"]

    if "hasButton" in rules and rules["hasButton"] and Board.PASS_LOC not in [move[1] for move in moves]:
      global_input_data[idx,17] = 1.0

    if rules["scoringRule"] == "SCORING_AREA" or rules["encorePhase"] > 1:
      boardAreaIsEven = (board.size % 2 == 0)

      drawableKomisAreEven = boardAreaIsEven

      if drawableKomisAreEven:
        komiFloor = math.floor(selfKomi / 2.0) * 2.0
      else:
        komiFloor = math.floor((selfKomi-1.0) / 2.0) * 2.0 + 1.0

      delta = selfKomi - komiFloor
      assert(delta >= -0.0001)
      assert(delta <= 2.0001)
      if delta < 0.0:
        delta = 0.0
      if delta > 2.0:
        delta = 2.0

      if delta < 0.5:
        wave = delta
      elif delta < 1.5:
        wave = 1.0-delta
      else:
        wave = delta-2.0

      global_input_data[idx,18] = wave

    return idx+1
