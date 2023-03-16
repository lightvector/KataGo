"""Go board representation."""

from itertools import chain

from .common import opponent_of


class _Group:
    """Represent a solidly-connected group.

    Public attributes:
      colour
      points
      is_surrounded

    Points are coordinate pairs (row, col).

    """


class _Region:
    """Represent an empty region.

    Public attributes:
      points
      neighbouring_colours

    Points are coordinate pairs (row, col).

    """

    def __init__(self):
        self.points = set()
        self.neighbouring_colours = set()


class Board:
    """A legal Go position.

    Supports playing stones with captures, and area scoring.

    Public attributes:
      side         -- board size (int >= 2)
      board_points -- list of coordinates of all points on the board

    """

    def __init__(self, side):
        self.side = side
        if side < 2:
            raise ValueError
        self.board_points = [
            (_row, _col) for _row in range(side) for _col in range(side)
        ]
        self.board = []
        for row in range(side):
            self.board.append([None] * side)
        self._is_empty = True

    def copy(self):
        """Return an independent copy of this Board."""
        b = Board(self.side)
        b.board = [self.board[i][:] for i in range(self.side)]
        b._is_empty = self._is_empty
        return b

    def _make_group(self, row, col, colour):
        points = set()
        is_surrounded = True
        to_handle = set()
        to_handle.add((row, col))
        while to_handle:
            point = to_handle.pop()
            points.add(point)
            r, c = point
            for neighbour in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                (r1, c1) = neighbour
                if not ((0 <= r1 < self.side) and (0 <= c1 < self.side)):
                    continue
                neigh_colour = self.board[r1][c1]
                if neigh_colour is None:
                    is_surrounded = False
                elif neigh_colour == colour:
                    if neighbour not in points:
                        to_handle.add(neighbour)
        group = _Group()
        group.colour = colour
        group.points = points
        group.is_surrounded = is_surrounded
        return group

    def _make_empty_region(self, row, col):
        points = set()
        neighbouring_colours = set()
        to_handle = set()
        to_handle.add((row, col))
        while to_handle:
            point = to_handle.pop()
            points.add(point)
            r, c = point
            for neighbour in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                (r1, c1) = neighbour
                if not ((0 <= r1 < self.side) and (0 <= c1 < self.side)):
                    continue
                neigh_colour = self.board[r1][c1]
                if neigh_colour is None:
                    if neighbour not in points:
                        to_handle.add(neighbour)
                else:
                    neighbouring_colours.add(neigh_colour)
        region = _Region()
        region.points = points
        region.neighbouring_colours = neighbouring_colours
        return region

    def _find_surrounded_groups(self):
        """Find solidly-connected groups with 0 liberties.

        Returns a list of _Groups.

        """
        surrounded = []
        handled = set()
        for row, col in self.board_points:
            colour = self.board[row][col]
            if colour is None:
                continue
            point = (row, col)
            if point in handled:
                continue
            group = self._make_group(row, col, colour)
            if group.is_surrounded:
                surrounded.append(group)
            handled.update(group.points)
        return surrounded

    def is_empty(self):
        """Return a boolean indicating whether the board is empty."""
        return self._is_empty

    def get(self, row, col):
        """Return the state of the specified point.

        Returns a colour, or None for an empty point.

        Raises IndexError if the coordinates are out of range.

        """
        if row < 0 or col < 0:
            raise IndexError
        return self.board[row][col]

    def play(self, row, col, colour):
        """Play a move on the board.

        Raises IndexError if the coordinates are out of range.

        Raises ValueError if the specified point isn't empty.

        Performs any necessary captures. Allows self-captures. Doesn't enforce
        any ko rule.

        Returns the point forbidden by simple ko, or None

        """
        if row < 0 or col < 0:
            raise IndexError
        opponent = opponent_of(colour)
        if self.board[row][col] is not None:
            raise ValueError
        self.board[row][col] = colour
        self._is_empty = False
        surrounded = self._find_surrounded_groups()
        simple_ko_point = None
        if surrounded:
            if len(surrounded) == 1:
                to_capture = surrounded
                if len(to_capture[0].points) == self.side * self.side:
                    self._is_empty = True
            else:
                to_capture = [group for group in surrounded if group.colour == opponent]
                if len(to_capture) == 1 and len(to_capture[0].points) == 1:
                    self_capture = [
                        group for group in surrounded if group.colour == colour
                    ]
                    if len(self_capture[0].points) == 1:
                        (simple_ko_point,) = to_capture[0].points
            for group in to_capture:
                for r, c in group.points:
                    self.board[r][c] = None
        return simple_ko_point

    def apply_setup(self, black_points, white_points, empty_points):
        """Add setup stones or removals to the position.

        This is intended to support SGF AB/AW/AE commands.

        Each parameter is an iterable of coordinate pairs (row, col).

        Applies all the setup specifications, then removes any groups with no
        liberties (so the resulting position is always legal).

        If the same point is specified in more than one list, the order in which
        they're applied is undefined.

        Returns a boolean saying whether the position was legal as specified.

        Raises IndexError if any coordinates are out of range.

        """
        for row, col in chain(black_points, white_points, empty_points):
            if row < 0 or col < 0 or row >= self.side or col >= self.side:
                raise IndexError
        for row, col in black_points:
            self.board[row][col] = "b"
        for row, col in white_points:
            self.board[row][col] = "w"
        for row, col in empty_points:
            self.board[row][col] = None
        captured = self._find_surrounded_groups()
        for group in captured:
            for row, col in group.points:
                self.board[row][col] = None
        self._is_empty = True
        for row, col in self.board_points:
            if self.board[row][col] is not None:
                self._is_empty = False
                break
        return not (captured)

    def list_occupied_points(self):
        """List all nonempty points.

        Returns a list of pairs (colour, (row, col))

        """
        result = []
        for row, col in self.board_points:
            colour = self.board[row][col]
            if colour is not None:
                result.append((colour, (row, col)))
        return result

    def area_score(self):
        """Calculate the area score of a position.

        Assumes all stones are alive.

        Returns black score minus white score.

        Doesn't take komi into account.

        """
        scores = {"b": 0, "w": 0}
        handled = set()
        for row, col in self.board_points:
            colour = self.board[row][col]
            if colour is not None:
                scores[colour] += 1
                continue
            point = (row, col)
            if point in handled:
                continue
            region = self._make_empty_region(row, col)
            region_size = len(region.points)
            for colour in ("b", "w"):
                if colour in region.neighbouring_colours:
                    scores[colour] += region_size
            handled.update(region.points)
        return scores["b"] - scores["w"]
