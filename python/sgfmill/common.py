"""Domain-dependent utility functions for sgfmill.

This module is designed to be used with 'from common import *'.

This is for Go-specific utilities.

"""

__all__ = ["opponent_of", "colour_name", "format_vertex", "format_vertex_list", "move_from_vertex"]

_opponents = {"b": "w", "w": "b"}


def opponent_of(colour):
    """Return the opponent colour.

    colour -- 'b' or 'w'

    Returns 'b' or 'w'.

    """
    try:
        return _opponents[colour]
    except KeyError as e:
        raise ValueError from e


def colour_name(colour):
    """Return the (lower-case) full name of a colour.

    colour -- 'b' or 'w'

    """
    try:
        return {"b": "black", "w": "white"}[colour]
    except KeyError as e:
        raise ValueError from e


column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def format_vertex(move):
    """Return coordinates as a string like 'A1', or 'pass'.

    move -- pair (row, col), or None for a pass

    The result is suitable for use directly in GTP responses.

    """
    if move is None:
        return "pass"
    row, col = move
    if not 0 <= row < 25 or not 0 <= col < 25:
        raise ValueError
    return column_letters[col] + str(row + 1)


def format_vertex_list(moves):
    """Return a list of coordinates as a string like 'A1,B2'."""
    return ",".join(map(format_vertex, moves))


def move_from_vertex(vertex, board_size):
    """Interpret a string representing a vertex, as specified by GTP.

    Returns a pair of coordinates (row, col) in range(0, board_size)

    Raises ValueError with an appropriate message if 'vertex' isn't a valid GTP
    vertex specification for a board of size 'board_size'.

    """
    if not 0 < board_size <= 25:
        raise ValueError("board_size out of range")
    try:
        s = vertex.lower()
    except Exception as e:
        raise ValueError("invalid vertex") from e
    if s == "pass":
        return None
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        if col_c > "i":
            col = ord(col_c) - ord("b")
        else:
            col = ord(col_c) - ord("a")
        row = int(s[1:]) - 1
        if row < 0:
            raise ValueError
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid vertex: '{s}'") from exc
    if col >= board_size or row >= board_size:
        raise ValueError(f"vertex is off board: '{s}'")
    return row, col
