"""Higher-level processing of moves and positions from SGF games."""

from . import sgf_properties

def get_setup_and_moves(sgf_game, board=None):
    """Return the initial setup and the following moves from an Sgf_game.

    Returns a pair (board, plays)

      board -- boards.Board
      plays -- list of pairs (colour, move)
               moves are (row, col), or None for a pass.

    The board represents the position described by AB and/or AW properties
    in the root node.

    The moves are from the game's 'leftmost' variation.

    Raises ValueError if this position isn't legal.

    Raises ValueError if there are any AB/AW/AE properties after the root
    node.

    Doesn't check whether the moves are legal.

    If the optional 'board' parameter is provided, it must be an empty board of
    the right size; the same object will be returned.
    See sgf_board_interface.Interface_for_get_setup_and_moves
    for the interface it should support.

    """
    size = sgf_game.get_size()
    if board is None:
        from . import boards
        board = boards.Board(size)
    else:
        if board.side != size:
            raise ValueError("wrong board size, must be %d" % size)
        if not board.is_empty():
            raise ValueError("board not empty")
    root = sgf_game.get_root()
    nodes = sgf_game.main_sequence_iter()
    ab, aw, ae = root.get_setup_stones()
    if ab or aw:
        is_legal = board.apply_setup(ab, aw, ae)
        if not is_legal:
            raise ValueError("setup position not legal")
        colour, raw = root.get_raw_move()
        if colour is not None:
            raise ValueError("mixed setup and moves in root node")
        next(nodes)
    moves = []
    for node in nodes:
        if node.has_setup_stones():
            raise ValueError("setup properties after the root node")
        colour, raw = node.get_raw_move()
        if colour is not None:
            moves.append((colour, sgf_properties.interpret_go_point(raw, size)))
    return board, moves

def set_initial_position(sgf_game, board):
    """Add setup stones to an Sgf_game reflecting a board position.

    sgf_game -- Sgf_game
    board    -- Board object

    Replaces any existing setup stones in the Sgf_game's root node.

    'board' may be a boards.Board instance, or anything providing the interface
    described by sgf_board_interface.Interface_for_set_initial_position().

    """
    stones = {'b' : set(), 'w' : set()}
    for (colour, point) in board.list_occupied_points():
        stones[colour].add(point)
    sgf_game.get_root().set_setup_stones(stones['b'], stones['w'])

def indicate_first_player(sgf_game):
    """Add a PL property to the root node if appropriate.

    Looks at the first child of the root to see who the first player is, and
    sets PL it isn't the expected player (ie, black normally, but white if
    there is a handicap), or if there are non-handicap setup stones.

    """
    root = sgf_game.get_root()
    first_player, move = root[0].get_move()
    if first_player is None:
        return
    has_handicap = root.has_property("HA")
    if root.has_property("AW"):
        specify_pl = True
    elif root.has_property("AB") and not has_handicap:
        specify_pl = True
    elif not has_handicap and first_player == 'w':
        specify_pl = True
    elif has_handicap and first_player == 'b':
        specify_pl = True
    else:
        specify_pl = False
    if specify_pl:
        root.set('PL', first_player)

