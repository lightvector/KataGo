import glob
import itertools

from sgfmill import sgf
import elo
import os
import math
import argparse
from typing import List, Dict, Tuple, Set, Sequence
from dataclasses import dataclass
import warnings


@dataclass
class Record:
  win: int = 0
  lost: int = 0
  draw: int = 0


class GameResultSummary:
  """
  Summrize Go games results in sgf or sgfs file format under a directory (and its subdirectory).
  The sgfs file is a special file format provided by Katago, and each row is a single game in a sgf string.
  sgf file is not required to be in a single line.

  Public functions include:
  self.add_games(input_file_dir, search_subdir = False): add files into the results and duplicate files will not be
  added.
  self.add_a_game_file(input_file_name): add a new game file into the results and duplicate file will not be added.
  self.clear(): clear all the results
  self.print_elos(): print the estimated Bayes Elo scores
  self.print_game_results(): print the game results
  self.get_game_results(self) -> Dict

  Example:
    call it from terminal:
      :$python summarize_sgfs.py -input-dirs [input_directory1 input directory2] -elo-prior 3400

    call it by other function:
    import summarize_sgfs
    elo_prior = 50
    game_result_summary = summarize_sgfs.GameResultSummary(elo_prior)
    game_result_summary.add_games(input_file_dir)
    game_result_summary.print_game_results()
    game_result_summary.print_elos()
  """

  def __init__(
      self,
      elo_prior: float
  ):
    self.results = {}  # a dictionary as {(black_player_name, white_player_name):Record}

    self._all_sgfs_files = set()
    self._all_sgf_files = set()
    self._should_warn_handicap_komi = False
    self._elo_prior = elo_prior #number of prior games to stablize the results.

  """ Public functions """

  def add_games(self, input_file_dir, search_subdir=False):
    """add files under input_file_dir into the results and duplicate files will not be added."""
    new_files = self._add_game_file_names(input_file_dir, search_subdir)
    self._add_new_games_to_result_dict(new_files)

  def add_a_game_file(self, input_file_name):
    """add a new game file (sgfs or sgf) into the results and duplicate file will not be added."""
    new_file = self._add_a_game_file_name(input_file_name)
    self._add_new_games_to_result_dict(new_file)

  def clear(self):
    """Clear all the records and reset to empty sets and empty dictionaries"""
    self.results = {}  # a dictionary as {(black_player_name, white_player_name):Record}
    self._all_sgfs_files = set()
    self._all_sgf_files = set()
    self._should_warn_handicap_komi = False

  def print_elos(self) -> elo.EloInfo:
    """Print estimated Bayes Elo scores centered as Elo prior"""
    elo_info = self._estimate_elo()
    print(elo_info)
    print(f"The prior is set as {self._elo_prior}, and the estimated Bayes Elo:")
    if self._should_warn_handicap_komi:
      warnings.warn("There are handicap games or games with komi < 5.5 or komi > 7.5")
    return elo_info

  def print_game_results(self):
    """Print game results in a matrix with item in row i and column j as the numbers of games player i beaten player
    j. We split draw game as 0.5 to each player."""
    print("Player information:")
    self._print_player_info()
    print("Game Results by Player ID:")
    self._print_result_matrix()
    return

  def get_game_results(self) -> Dict:
    """Return a dictionary of game results as {(black_player_name, white_player_name):Record}
      We can retrieve results by player's name as
      results[(black_player_name, white_player_name)].win
      results[(black_player_name, white_player_name)].lost
      results[(black_player_name, white_player_name)].draw
    """
    return self.results

  """ Private functions starts here, and no more public functions below """

  def _add_game_file_names(self, input_file_dir, search_subdir):
    # handle user specifying multiple sgf files to the script via glob on the terminal, for example foo/bar/*.sgf"
    print(f"Searching and adding files in {input_file_dir}, please wait ...")
    if input_file_dir.lower().endswith((".sgf", ".sgfs")):
      files = glob.glob(input_file_dir)
      for file in files:
        self.add_a_game_file(file)
      print(f"We found {len(files)} new sgfs/sgf files in after parse {input_file_dir}.")
      return {"new_sgfs_files": [], "new_sgf_files": []}

    if not os.path.exists(input_file_dir):
      print(f"There is no directory under name {input_file_dir}")
      return

    files = []
    if (search_subdir):
      for (dirpath, dirnames, filenames) in os.walk(input_file_dir):
        files += [os.path.join(dirpath, file) for file in filenames]
    else:
      files = [os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir)]

    new_sgfs_files = set([file for file in files if file.split(".")[-1].lower() ==
                           "sgfs"])
    new_sgf_files = set([file for file in files if file.split(".")[-1].lower() == "sgf"])

    #update the new_sgfs_files and sgf_files to remove duplication
    new_sgfs_files = new_sgfs_files.difference(self._all_sgfs_files)
    new_sgf_files = new_sgf_files.difference(self._all_sgf_files)

    self._all_sgfs_files = self._all_sgfs_files.union(new_sgfs_files)
    self._all_sgf_files = self._all_sgf_files.union(new_sgf_files)
    print(f"We found {len(new_sgfs_files)} new sgfs files and {len(new_sgf_files)} new sgf files in the "
          f"search "
          f"directory {input_file_dir}.")
    return {"new_sgfs_files": new_sgfs_files, "new_sgf_files": new_sgf_files}

  def _add_a_game_file_name(self, input_file_name):
    new_sgf_files = []
    new_sgfs_files =[]
    if not os.path.isfile(input_file_name):
      print(f"There is no file with name {input_file_name}.")
      return {"new_sgfs_files": new_sgfs_files, "new_sgf_files": new_sgf_files}

    if input_file_name in self._all_sgfs_files or input_file_name in self._all_sgf_files:
      print(f"File name {input_file_name} has been added before")
      return {"new_sgfs_files": new_sgfs_files, "new_sgf_files": [new_sgf_files]}

    if input_file_name.split(".")[-1].lower() == "sgf":
      new_sgf_files = [input_file_name]
      self._all_sgf_files = self._all_sgf_files.union({input_file_name})
    elif input_file_name.split(".")[-1].lower() == "sgfs":
      new_sgfs_files = [input_file_name]
      self._all_sgfs_files = self._all_sgfs_files.union({input_file_name})
    else:
      print(f"{input_file_name} is not sgf or sgfs file, no game was added")
      return
    return {"new_sgfs_files": new_sgfs_files, "new_sgf_files": new_sgf_files}

  def _add_new_games_to_result_dict(self, new_files):
    """add all sgfs files first"""
    new_sgf_files = new_files["new_sgf_files"]
    new_sgfs_files = new_files["new_sgfs_files"]

    idx = 1
    for sgfs in new_sgfs_files:
      self._add_one_sgfs_file_to_result(sgfs)
      if (idx % 10 == 0):
        print(f"Addedd {idx} files out of {len(new_sgfs_files)} sgfs files.")
      idx += 1
    #print(f"We have added additional {len(new_sgfs_files)} sgfs files into the results.")

    idx = 1
    for sgf in new_sgf_files:
      self._add_one_sgf_file_to_result(sgf)
      if (idx % 10 == 0):
        print(f"Added {idx} files out of {len(new_sgf_files)} sgf files.")
      idx += 1
    #print(f"We have added additional {len(new_sgf_files)} sgf files into the results.")

  def _add_one_sgfs_file_to_result(self, sgfs_file_name):
    """Add a single sgfs file. Each row of such sgf file contain a single game as a sgf string"""
    if not os.path.exists(sgfs_file_name):
      print(f"There is no SGFs file as {sgfs_file_name}")
      return

    with open(sgfs_file_name, "rb") as f:
      sgfs_strings = f.readlines()

    for sgf in sgfs_strings:
      self._add_a_single_sgf_string(sgf)

  def _add_one_sgf_file_to_result(self, sgf_file_name):
    """Add a single sgf file."""
    if not os.path.exists(sgf_file_name):
      print(f"There is no SGF file as {sgf_file_name}")
      return

    with open(sgf_file_name, "rb") as f:
      sgf = f.read()

    self._add_a_single_sgf_string(sgf)

  def _add_a_single_sgf_string(self, sgf_string):
    """add a single game in a sgf string save the results in self.results."""
    game = sgf.Sgf_game.from_bytes(sgf_string)
    winner = game.get_winner()
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

  def _estimate_elo(self) -> elo.EloInfo:
    """Estimate and print elo values. This function must be called after add all the sgfs/sgf files"""
    pla_names = set(itertools.chain(*(name_pair for name_pair in self.results.keys())))
    data = []
    for pla_black in pla_names:
      for pla_white in pla_names:
        if (pla_black == pla_white):
          continue
        else:
          record = self.results[(pla_black, pla_white)]
          total = record.win + record.lost + record.draw
          win = record.win + 0.5 * record.draw
          winrate = win / total
          data.extend(elo.likelihood_of_games(pla_black, pla_white, total, winrate, False))
    for pla in pla_names:
      data.extend(elo.make_single_player_prior(pla, self._elo_prior,0))
    data.extend(elo.make_center_elos_prior(list(pla_names),0)) #add this in case user put elo_prior = 0
    info = elo.compute_elos(data, verbose=True)
    return info

  def _print_player_info(self):
    pla_names = set(itertools.chain(*(name_pair for name_pair in self.results.keys())))
    max_len = len(max(pla_names, key=len))
    title = 'Player Name'
    title_space = max(len(title), max_len) + 1
    print(f"{title:<{title_space}}: Player ID")
    idx = 0
    for pla in pla_names:
      print(f"{pla:<{title_space}}: {idx}")
      idx += 1

  def _print_result_matrix(self):
    results_matrix = self._build_result_matrix()
    max_game_played_space = int(math.log10(max([max(sublist) for sublist in results_matrix]))) + 5
    row_format = f"{{:>{max_game_played_space}}}" * (len(results_matrix) + 1)
    print(row_format.format("", *range(len(results_matrix))))
    for playerID, row in zip(range(len(results_matrix)), results_matrix):
      print(row_format.format(playerID, *row))

  def _build_result_matrix(self) -> list:
    pla_names = list(set(itertools.chain(*(name_pair for name_pair in self.results.keys()))))
    results_matrix = []
    for pla1 in pla_names:
      row = []
      for pla2 in pla_names:
        if (pla1 == pla2):
          row.append(0)
          continue
        else:
          pla1_pla2 = self.results[(pla1, pla2)]
          pla2_pla1 = self.results[(pla2, pla1)]
          win = pla1_pla2.win + pla2_pla1.lost + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
          row.append(win)
      results_matrix.append(row)
    return results_matrix


if __name__ == "__main__":
  description = """
  Summarize SGF/SGFs files and estimate Bayes Elo score for each of the player.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-input-dirs', help='sgf/sgfs files input directories. If multiple directories, seperate them '
                                          'with a space', required=True,
                      nargs='+')
  parser.add_argument('-search-subdir', help='If we also need to search all sub-directories', required=False,
                      default=False)
  parser.add_argument('-elo-prior', help='Prior for Bayes Elo calculation, using input as the prior number of games '
                                         'to stabilize the results',
                      required=False, type=float, default=0)
  args = vars(parser.parse_args())

  input_dirs = args["input_dirs"]
  search_subdir = args["search_subdir"]
  elo_prior = args["elo_prior"]

  game_result_summary = GameResultSummary(elo_prior)
  for dir in input_dirs:
    game_result_summary.add_games(dir, search_subdir)

  game_result_summary.print_game_results()
  game_result_summary.print_elos()
