"""Description of the board interfaces required by sgf_moves."""

class Interface_for_get_setup_and_moves:
    """Interface required by sgf_moves.get_setup_and_moves().

    Required public attributes:
      side -- board size (int >= 2)

    """

    def is_empty(self):
        """Return a boolean indicating whether the board is empty."""
        raise NotImplementedError

    def apply_setup(self, black_points, white_points, empty_points):
        """Add setup stones or removals to the position.

        See boards.Board.apply_setup() for details.

        """
        raise NotImplementedError

class Interface_for_set_initial_position:
    """Interface required by sgf_moves.set_initial_position()."""

    def list_occupied_points(self):
        """List all nonempty points.

        Returns a list of pairs (colour, (row, col))

        """
        raise NotImplementedError
