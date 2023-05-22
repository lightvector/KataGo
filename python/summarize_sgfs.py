import argparse
import elo
import itertools
import math
import os

from dataclasses import dataclass
from sgfmill import sgf
from typing import List, Dict, Tuple, Set, Sequence

@dataclass
class Record:
    win: int = 0
    lost: int = 0
    draw: int = 0

class GameResultSummary:
    """
    Summrize Go games results in sgf file format under a list of directories (optionally recursively in subdirs).
    Also supports katago "sgfs" file, which is simply a bunch of sgf files (with no newlines) concatenated one per line.

    Example:
      Call it from terminal:
        :$python summarize_sgfs.py [input_directory1 input directory2]

      call it by other function:
      import summarize_sgfs
      elo_prior_games = 4
      estimate_first_player_advantage = False
      game_result_summary = summarize_sgfs.GameResultSummary(elo_prior_games, estimate_first_player_advantage)
      game_result_summary.add_games(input_file_or_dir)
      game_result_summary.print_game_results()
      game_result_summary.print_elos()
    """

    def __init__(
        self,
        elo_prior_games: float,
        estimate_first_player_advantage: bool,
    ):
        self.results = {}  # dict of { (black_player_name, white_player_name) : Record }

        self._all_sgfs_files = set()
        self._all_sgf_files = set()
        self._should_warn_handicap_komi = False
        self._elo_prior_games = elo_prior_games # number of games for bayesian prior around Elo 0
        self._estimate_first_player_advantage = estimate_first_player_advantage
        self._elo_info = None
        self._game_count = 0

    def add_games(self, input_file_or_dir: str, recursive=False):
        """Add sgfs found in input_file_or_dir into the results. Repeated paths to the same file will be ignored."""
        new_files = self._add_files(input_file_or_dir, recursive)

    def clear(self):
        """Clear all data added."""
        self.results = {}
        self._all_sgfs_files = set()
        self._all_sgf_files = set()
        self._should_warn_handicap_komi = False
        self._elo_info = None

    def print_game_results(self):
        """Print tables of wins and win percentage."""
        pla_names = set(itertools.chain(*(name_pair for name_pair in self.results.keys())))
        self._print_result_matrix(pla_names)

    def print_elos(self):
        """Print game results and maximum likelihood posterior Elos."""
        elo_info = self._compute_elos_if_needed()
        real_players = [player for player in elo_info.players if player != elo.P1_ADVANTAGE_NAME]
        self._print_result_matrix(real_players)
        print("Elos (+/- one approx standard error):")
        print(elo_info)

        print("Pairwise approx % likelihood of superiority of row over column:")
        los_matrix = []
        for player in real_players:
            los_row = []
            for player2 in real_players:
                los = elo_info.get_approx_likelihood_of_superiority(player,player2)
                los_row.append(f"{los*100:.2f}")
            los_matrix.append(los_row)
        self._print_matrix(real_players,los_matrix)

        print(f"Used a prior of {self._elo_prior_games} games worth that each player is near Elo 0.")
        if self._should_warn_handicap_komi:
            print("WARNING: There are handicap games or games with komi < 5.5 or komi > 7.5, these games may not be fair?")

    def get_elos(self) -> elo.EloInfo:
        return self._compute_elos_if_needed()

    def get_game_results(self) -> Dict:
        """Return a dictionary of game results as { (black_player_name, white_player_name) : Record }

          You can retrieve results by player's name like:
          results[(black_player_name, white_player_name)].win
          results[(black_player_name, white_player_name)].lost
          results[(black_player_name, white_player_name)].draw
        """
        return self.results

    # Private functions ------------------------------------------------------------------------------------

    def _compute_elos_if_needed(self):
        if self._elo_info is None:
            self._elo_info = self._estimate_elo()
        return self._elo_info

    def _add_files(self, input_file_or_dir, recursive):
        print(f"Searching and adding files in {input_file_or_dir}, please wait...")

        if not os.path.exists(input_file_or_dir):
            raise Exception(f"There is no file or directory with name: {input_file_or_dir}")

        files = []
        if input_file_or_dir.lower().endswith((".sgf", ".sgfs")):
            files.append(input_file_or_dir)
        elif recursive:
            for (dirpath, dirnames, filenames) in os.walk(input_file_or_dir):
                files += [os.path.join(dirpath, file) for file in filenames]
        else:
            files = [os.path.join(input_file_or_dir, file) for file in os.listdir(input_file_or_dir)]

        new_sgfs_files = set([file for file in files if file.split(".")[-1].lower() == "sgfs"])
        new_sgf_files = set([file for file in files if file.split(".")[-1].lower() == "sgf"])

        # Remove duplicates
        new_sgfs_files = new_sgfs_files.difference(self._all_sgfs_files)
        new_sgf_files = new_sgf_files.difference(self._all_sgf_files)

        self._all_sgfs_files = self._all_sgfs_files.union(new_sgfs_files)
        self._all_sgf_files = self._all_sgf_files.union(new_sgf_files)

        self._add_new_games_to_result_dict(new_sgf_files, new_sgfs_files, input_file_or_dir)

        print(f"Added {len(new_sgfs_files)} new sgfs files and {len(new_sgf_files)} new sgf files from {input_file_or_dir}")

    def _add_new_games_to_result_dict(self, new_sgf_files, new_sgfs_files, source):
        idx = 0
        for sgfs in new_sgfs_files:
            self._add_one_sgfs_file_to_result(sgfs)
            idx += 1
            if (idx % 10 == 0):
                print(f"Added {idx}/{len(new_sgfs_files)} sgfs files for {source}")

        idx = 0
        for sgf in new_sgf_files:
            self._add_one_sgf_file_to_result(sgf)
            idx += 1
            if (idx % 10 == 0):
                print(f"Added {idx}/{len(new_sgf_files)} sgf files for {source}")

    def _add_one_sgfs_file_to_result(self, sgfs_file_name):
        """Add a single sgfs file. Each line of an sgfs file should be the contents of a valid sgf file with no newlines."""
        if not os.path.exists(sgfs_file_name):
            raise Exception(f"There is no SGFs file named: {sgfs_file_name}")

        with open(sgfs_file_name, "rb") as f:
            sgfs_strings = f.readlines()

        for sgf in sgfs_strings:
            self._add_a_single_sgf_string(sgf, sgfs_file_name)

    def _add_one_sgf_file_to_result(self, sgf_file_name):
        """Add a single sgf file."""
        if not os.path.exists(sgf_file_name):
            raise Exception(f"There is no SGF file named: {sgf_file_name}")

        with open(sgf_file_name, "rb") as f:
            sgf = f.read()

        self._add_a_single_sgf_string(sgf, sgf_file_name)

    def _add_a_single_sgf_string(self, sgf_string, debug_source = None):
        """add a single game in a sgf string save the results in self.results."""
        try:
            game = sgf.Sgf_game.from_bytes(sgf_string)
            winner = game.get_winner()
        except ValueError:
            print ('\033[91m'+f"A sgf string is damaged in {debug_source}, and its record has been skipped!"+ '\x1b[0m')
            return
        pla_black = game.get_player_name('b')
        pla_white = game.get_player_name('w')
        if (game.get_handicap() is not None) or game.get_komi() < 5.5 or game.get_komi() > 7.5:
            self._should_warn_handicap_komi = True

        if (pla_black, pla_white) not in self.results:
            self.results[(pla_black, pla_white)] = Record()

        if (winner == 'b'):
            self.results[(pla_black, pla_white)].win += 1
        elif (winner == 'w'):
            self.results[(pla_black, pla_white)].lost += 1
        else:
            self.results[(pla_black, pla_white)].draw += 1
        self._game_count += 1

    def _estimate_elo(self) -> elo.EloInfo:
        """Estimate and print elo values. This function must be called after add all the sgfs/sgf files"""
        pla_names = set(itertools.chain(*(name_pair for name_pair in self.results.keys())))
        data = []
        for pla_black in pla_names:
            for pla_white in pla_names:
                if (pla_black == pla_white):
                    continue
                else:
                    if (pla_black, pla_white) not in self.results:
                        continue
                    record = self.results[(pla_black, pla_white)]
                    total = record.win + record.lost + record.draw
                    assert total >= 0
                    if total == 0:
                        continue

                    win = record.win + 0.5 * record.draw
                    winrate = win / total
                    data.extend(elo.likelihood_of_games(
                        pla_black,
                        pla_white,
                        total,
                        winrate,
                        include_first_player_advantage=self._estimate_first_player_advantage
                    ))

        for pla in pla_names:
            data.extend(elo.make_single_player_prior(pla, self._elo_prior_games,0))
        data.extend(elo.make_center_elos_prior(list(pla_names),0)) # Add this in case user put elo_prior_games = 0
        if self._estimate_first_player_advantage:
            data.extend(elo.make_single_player_prior(elo.P1_ADVANTAGE_NAME, (1.0 + self._elo_prior_games) * 2.0, 0))

        info = elo.compute_elos(data, verbose=True)
        return info

    def _print_matrix(self,pla_names,results_matrix):
        per_elt_space = 2
        for sublist in results_matrix:
            for elt in sublist:
                per_elt_space = max(per_elt_space, len(str(elt)))
        per_elt_space += 2

        per_name_space = 1 if len(pla_names) == 0 else max(len(name) for name in pla_names)
        per_name_space += 1
        if per_name_space > per_elt_space:
            per_elt_space += 1

        row_format = f"{{:>{per_name_space}}}" +   f"{{:>{per_elt_space}}}" * len(results_matrix)
        print(row_format.format("", *[name[:per_elt_space-2] for name in pla_names]))
        for name, row in zip(pla_names, results_matrix):
            print(row_format.format(name, *row))

    def _print_result_matrix(self, pla_names):
        print(f"Total games: {self._game_count}")
        print("Games by player:")
        for pla1 in pla_names:
            total = 0
            for pla2 in pla_names:
                if (pla1 == pla2):
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else Record()
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else Record()
                    total += pla1_pla2.win + pla2_pla1.win + pla1_pla2.lost + pla2_pla1.lost + pla1_pla2.draw + pla2_pla1.draw
            print(f"{pla1}: {total:.1f}")

        print("Wins by row player against column player:")
        result_matrix = []
        for pla1 in pla_names:
            row = []
            for pla2 in pla_names:
                if (pla1 == pla2):
                    row.append("-")
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else Record()
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else Record()
                    win = pla1_pla2.win + pla2_pla1.lost + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.lost + pla2_pla1.lost + pla1_pla2.draw + pla2_pla1.draw
                    row.append(f"{win:.1f}/{total:.1f}")
            result_matrix.append(row)
        self._print_matrix(pla_names,result_matrix)

        print("Win% by row player against column player:")
        result_matrix = []
        for pla1 in pla_names:
            row = []
            for pla2 in pla_names:
                if (pla1 == pla2):
                    row.append("-")
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else Record()
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else Record()
                    win = pla1_pla2.win + pla2_pla1.lost + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.lost + pla2_pla1.lost + pla1_pla2.draw + pla2_pla1.draw
                    if total <= 0:
                        row.append("-")
                    else:
                        row.append(f"{win/total*100.0:.1f}%")
            result_matrix.append(row)

        self._print_matrix(pla_names,result_matrix)

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

    game_result_summary = GameResultSummary(
        elo_prior_games=elo_prior_games,
        estimate_first_player_advantage=estimate_first_player_advantage,
    )
    for input_file_or_dir in input_files_or_dirs:
        game_result_summary.add_games(input_file_or_dir, recursive=recursive)

    game_result_summary.print_elos()
