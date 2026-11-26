"""
Unit tests for the incremental zobrist hash in katago.game.board.Board.

Board.zobrist is updated incrementally as stones are placed and captured. These tests check that it always equals
the hash recomputed from scratch for the stones currently on the board, in particular after captures.
"""

import random

from katago.game.board import Board


def recomputed_zobrist(board):
    h = 0
    for loc in range(board.arrsize):
        if board.board[loc] == Board.BLACK or board.board[loc] == Board.WHITE:
            h ^= Board.ZOBRIST_STONE[board.board[loc]][loc]
    return h


def test_zobrist_after_single_capture():
    board = Board(9)
    board.play(Board.WHITE, board.loc(1, 0))
    board.play(Board.BLACK, board.loc(0, 0))
    board.play(Board.BLACK, board.loc(2, 0))
    board.play(Board.BLACK, board.loc(1, 1))  # captures the white stone at (1,0)
    assert board.board[board.loc(1, 0)] == Board.EMPTY
    assert board.zobrist == recomputed_zobrist(board)


def test_zobrist_matches_recomputation_in_random_games():
    rand = random.Random(12345)
    for _ in range(100):
        board = Board(5)
        pla = Board.BLACK
        for _ in range(80):
            moves = [loc for loc in range(board.arrsize) if board.board[loc] == Board.EMPTY and board.would_be_legal(pla, loc)]
            if not moves:
                break
            board.play(pla, rand.choice(moves))
            assert board.zobrist == recomputed_zobrist(board)
            pla = Board.get_opp(pla)


def test_zobrist_restored_by_undo():
    rand = random.Random(54321)
    for _ in range(100):
        board = Board(5)
        pla = Board.BLACK
        for _ in range(80):
            moves = [loc for loc in range(board.arrsize) if board.board[loc] == Board.EMPTY and board.would_be_legal(pla, loc)]
            if not moves:
                break
            zobrist_before = board.zobrist
            record = board.playRecordedUnsafe(pla, rand.choice(moves))
            assert board.zobrist == recomputed_zobrist(board)
            board.undo(record)
            assert board.zobrist == zobrist_before
            board.play(pla, rand.choice(moves))
            pla = Board.get_opp(pla)
