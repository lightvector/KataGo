import argparse
import elo
from elo import GameRecord
import itertools
import math
import os
import re

from dataclasses import dataclass
from sgfmill import sgf
from typing import List, Dict, Tuple, Set, Sequence

class GoGameResultSummary(elo.GameResultSummary):

    def __init__(
        self,
        elo_prior_games: float,
        estimate_first_player_advantage: bool,
    ):
        super().__init__(elo_prior_games, estimate_first_player_advantage)
        self._should_warn_handicap_komi = False

    # @override
    def print_elos(self):
        super().print_elos()
        if self._should_warn_handicap_komi:
            print("WARNING: There are handicap games or games with komi < 5.5 or komi > 7.5, these games may not be fair?")

    # @override
    def is_game_file(self, input_file: str) -> bool:
        lower = input_file.lower()
        return input_file.endswith(".sgf") or input_file.endswith(".sgfs")

    # @override
    def get_game_records(self, input_file: str) -> List[GameRecord]:
        if input_file.lower().endswith(".sgfs"):
            with open(input_file, "rb") as f:
                sgfs_strings = f.readlines()

            records = []
            for sgf in sgfs_strings:
                record = self.sgf_string_to_game_record(sgf, input_file)
                if record is not None:
                    records.append(record)
            return records
        else:
            with open(input_file, "rb") as f:
                sgf = f.read()

            records = []
            record = self.sgf_string_to_game_record(sgf, input_file)
            if record is not None:
                records.append(record)
            return records

    def sgf_string_to_game_record(self, sgf_string, debug_source = None) -> GameRecord:
        try:
            # sgfmill for some reason can't handle rectangular boards, even though it's part of the SGF spec.
            # So lie and say that they're square, so that we can load them.
            sgf_string = re.sub(r'SZ\[(\d+):\d+\]', r'SZ[\1]', sgf_string.decode("utf-8"))
            sgf_string = sgf_string.encode("utf-8")

            game = sgf.Sgf_game.from_bytes(sgf_string)
            winner = game.get_winner()
        except ValueError:
            print ('\033[91m'+f"A sgf string is damaged in {debug_source}, and its record has been skipped!"+ '\x1b[0m')
            return
        pla_black = game.get_player_name('b')
        pla_white = game.get_player_name('w')
        if (game.get_handicap() is not None) or game.get_komi() < 5.5 or game.get_komi() > 7.5:
            self._should_warn_handicap_komi = True

        game_record = GameRecord(player1=pla_black,player2=pla_white)
        if (winner == 'b'):
            game_record.win += 1
        elif (winner == 'w'):
            game_record.loss += 1
        else:
            game_record.draw += 1
        return game_record



if __name__ == "__main__":
    description = """
    Summarize SGF/SGFs files and estimate Bayes Elo score for each of the player.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "input-files-or-dirs",
        help="sgf/sgfs files or directories of them",
        nargs="+",
    )
    parser.add_argument(
        "-recursive",
        help="Recursively search subdirectories of input directories",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-elo-prior-games",
        help="Prior for Bayes Elo calculation, using input as the prior number of games to stabilize the results",
        required=False,
        type=float,
        default=2,
    )
    parser.add_argument(
        "-estimate-first-player-advantage",
        help="Attempt to estimate first player advantage instead of assuming fair game",
        required=False,
        action="store_true",
    )
    args = vars(parser.parse_args())
    print(args)

    input_files_or_dirs = args["input-files-or-dirs"]
    recursive = args["recursive"]
    elo_prior_games = args["elo_prior_games"]
    estimate_first_player_advantage = args["estimate_first_player_advantage"]

    game_result_summary = GoGameResultSummary(
        elo_prior_games=elo_prior_games,
        estimate_first_player_advantage=estimate_first_player_advantage,
    )
    for input_file_or_dir in input_files_or_dirs:
        game_result_summary.add_games_from_file_or_dir(input_file_or_dir, recursive=recursive)

    game_result_summary.print_elos()
