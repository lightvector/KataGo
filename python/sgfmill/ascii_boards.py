"""ASCII board representation."""

from . import boards
from .common import column_letters


def render_grid(point_formatter, size):
    """Render a board-shaped grid as a list of strings.

    point_formatter -- function (row, col) -> string of length 2.

    Returns a list of strings.

    """
    column_header_string = " ".join(column_letters[i] for i in range(size))
    if size > 9:
        rowstart = "%2d "
        padding = " "
    else:
        rowstart = "%d "
        padding = ""
    result = [
        rowstart % (row + 1) + " ".join(point_formatter(row, col) for col in range(size))
        for row in range(size - 1, -1, -1)
    ]
    result.append(f"{padding}  {column_header_string}")
    return result


_point_strings = {
    None: ".",
    "b": "X",
    "w": "O",
}


def render_board(board):
    """Render an sgfmill Board in ascii.

    Returns a string without final newline.

    """

    def format_pt(row, col):
        return _point_strings.get(board.get(row, col), "?")

    return "\n".join(render_grid(format_pt, board.side))


def interpret_diagram(diagram, size, board=None):
    """Set up the position from a diagram.

    diagram -- board representation as from render_board()
    size    -- int

    Returns a Board.

    If the optional 'board' parameter is provided, it must be an empty board of
    the right size; the same object will be returned.

    Ignores leading and trailing whitespace.

    An ill-formed diagram may give ValueError or a 'best guess'.

    """
    if board is None:
        board = boards.Board(size)
    else:
        if board.side != size:
            raise ValueError("wrong board size, must be %d" % size)
        if not board.is_empty():
            raise ValueError("board not empty")
    lines = diagram.strip().split("\n")
    colours = {"#": "b", "o": "w", ".": None}
    if size > 9:
        extra_offset = 1
    else:
        extra_offset = 0
    try:
        for row, col in board.board_points:
            colour = colours[lines[size - row - 1][3 * (col + 1) + extra_offset]]
            if colour is not None:
                board.play(row, col, colour)
    except Exception as e:
        raise ValueError from e
    return board
