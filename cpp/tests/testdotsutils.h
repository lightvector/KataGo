#pragma once

#include "../program/playutils.h"

using namespace std;

inline Rand DOTS_RANDOM("DOTS_RANDOM");

struct XYMove {
  int x;
  int y;
  Player player;

  XYMove(const int x, const int y, const Player player) : x(x), y(y), player(player) {}

  [[nodiscard]] std::string toString() const {
    return "(" + to_string(x) + "," + to_string(y) + "," + PlayerIO::colorToChar(player) + ")";
  }
};

struct BoardWithMoveRecords {
  Board& board;
  vector<Board::MoveRecord>& moveRecords;

  BoardWithMoveRecords(Board& initBoard, vector<Board::MoveRecord>& initMoveRecords) : board(initBoard), moveRecords(initMoveRecords) {}

  void playMove(const int x, const int y, const Player player) const {
    moveRecords.push_back(board.playMoveRecorded(Location::getLoc(x, y, board.x_size), player));
  }

  void playGroundingMove(const Player player) const {
    moveRecords.push_back(board.playMoveRecorded(Board::PASS_LOC, player));
  }

  [[nodiscard]] State getState(const int x, const int y) const {
    return board.getState(Location::getLoc(x, y, board.x_size));
  }

  [[nodiscard]] bool isLegal(const int x, const int y, const Player player) const {
    return board.isLegal(Location::getLoc(x, y, board.x_size), player, true, false);
  }

  [[nodiscard]] bool isSuicide(const int x, const int y, const Player player) const {
    return board.isSuicide(Location::getLoc(x, y, board.x_size), player);
  }

  [[nodiscard]] bool wouldBeCapture(const int x, const int y, const Player player) const {
    return board.wouldBeCapture(Location::getLoc(x, y, board.x_size), player);
  }

  int getWhiteScore() const {
    return board.numBlackCaptures - board.numWhiteCaptures;
  }

  int getBlackScore() const {
    return -getWhiteScore();
  }

  void undo() const {
    board.undo(moveRecords.back());
    moveRecords.pop_back();
  }
};

Board parseDotsFieldDefault(const string& input, const vector<XYMove>& extraMoves = {});

Board parseDotsField(const string& input, bool startPosIsRandom, bool suicide, bool captureEmptyBases, bool freeCapturedDots, const vector<XYMove>& extraMoves);

