from sgfmill import sgf
import elo
import os
import math
import argparse
from typing import List, Dict, Tuple, Set, Sequence


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

  Example:
    call it from terminal:
      :$python summarize_sgfs.py -input-dir [files input directory] -elo-prior 3400

    call it by other function:
    import summarize_sgfs
    elo_prior = 3400
    gameResultSummary = summarize_sgfs.GameResultSummary(elo_prior)
    gameResultSummary.add_games(input_file_dir)
    gameResultSummary.print_game_results()
    gameResultSummary.print_elos()
  """
  def __init__(
      self,
      elo_prior:float
  ):
    self.results = []
    self.players = {}

    self._all_sgfs_files = set()
    self._all_sgf_files = set()
    self._new_sgfs_files = set()
    self._new_sgf_files = set()
    self._elo_prior = elo_prior

  """ Public functions """
  def add_games(self, input_file_dir, search_subdir = False):
    self._add_game_file_names(input_file_dir, search_subdir)
    self._add_new_games_to_resultMatrix()
  def add_a_game_file(self, input_file_name):
    self._add_a_game_file_name(input_file_name)
    self._add_new_games_to_resultMatrix()

  def clear(self):
    """Clear all the records and reset to empty sets and empty dictionaries"""
    self.results = []
    self.players = {}
    self._all_sgfs_files = set()
    self._all_sgf_files = set()
    self._new_sgfs_files = set()
    self._new_sgf_files = set()

  def print_elos(self) -> elo.EloInfo:
    elo_info = self._estimate_elo()
    print(elo_info)
    return elo_info

  def print_game_results(self):
    print("Player information:")
    self._print_player_info()
    print("Game Results by Player ID:")
    self._print_resultMatrix()
    return

  def get_game_results(self) ->Dict:
    return {"PlayerInfo": self.players, "ResultMatrix": self.results}

  """ Private functions """
  def _add_game_file_names(self, input_file_dir, search_subdir):
    if not os.path.exists(input_file_dir):
      print(f"There is no directory under name {input_file_dir}")
      return None
    _files=[]
    if(search_subdir):
      for (dirpath, dirnames, filenames) in os.walk(input_file_dir):
        _files += [os.path.join(dirpath, file) for file in filenames]
    else:
       _files = [os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir)]

    _new_sgfs_files = set([file for file in _files if file.split(".")[-1].lower() ==
                                                        "sgfs"])
    _new_sgf_files = set([file for file in _files if file.split(".")[-1].lower() == "sgf"])

    self._new_sgfs_files = _new_sgfs_files.difference(self._all_sgfs_files)
    self._new_sgf_files = _new_sgf_files.difference(self._all_sgf_files)

    self._all_sgfs_files = self._all_sgfs_files.union(_new_sgfs_files)
    self._all_sgf_files = self._all_sgf_files.union(_new_sgf_files)
    print(f"We found {len(self._new_sgfs_files)} new sgfs files and {len(self._new_sgf_files)} new sgf files in the "
          f"search "
          f"directory.")

  def _add_a_game_file_name(self, input_file_name):
    if not os.path.isfile(input_file_name):
      print(f"There is no file with name {input_file_name}.")
      return None
    if input_file_name.split(".")[-1].lower() == "sgf":
      self._new_sgf_files = {input_file_name}.difference(self._all_sgf_files)
      if len(self._new_sgf_files) == 0:
        print(f"File name {input_file_name} has been added before")
        return None
      else:
        self._all_sgf_files = self._all_sgf_files.union({input_file_name})
    elif input_file_name.split(".")[-1].lower() == "sgfs":
      self._new_sgfs_files = {input_file_name}.difference(self._all_sgfs_files)
      if len(self._new_sgfs_files) == 0:
        print(f"File name {input_file_name} has been added before")
        return None
      else:
        self._all_sgfs_files = self._all_sgfs_files.union({input_file_name})
    else:
      print(f"{input_file_name} is not sgf or sgfs file, no game was added")
      return None

  def _add_new_games_to_resultMatrix(self):
    """add all sgfs files first"""
    idx = 1
    for sgfs in self._new_sgfs_files:
        self._add_one_sgfs_file_to_resultMatrix(sgfs)
        if (idx%10 == 0):
          print(f"Addedd {idx} files out of {len(self._new_sgfs_files)} sgfs files.")
        idx+=1
    print(f"We have added additional {len(self._new_sgfs_files)} sgfs files into the results.")

    idx = 1
    for sgf in self._new_sgf_files:
      self._add_one_sgf_file_to_resultMatrix(sgf)
      if (idx % 10 == 0):
        print(f"Added {idx} files out of {len(self._new_sgf_files)} sgf files.")
      idx += 1
    print(f"We have added additional {len(self._new_sgf_files)} sgf files into the results.")

  def _add_one_sgfs_file_to_resultMatrix(self,sgfs_file_name):
    """Add a single sgfs file. Each row of such sgf file contain a single game as a sgf string"""
    if not os.path.exists(sgfs_file_name):
      print(f"There is no SGFs file as {sgfs_file_name}")
      return None

    with open(sgfs_file_name, "rb") as f:
      sgfs_strings = f.readlines()

    for sgf in sgfs_strings:
      self._add_a_single_sgf_string(sgf)

  def _add_one_sgf_file_to_resultMatrix(self,sgf_file_name):
    """Add a single sgf file."""
    if not os.path.exists(sgf_file_name):
      print(f"There is no SGF file as {sgf_file_name}")
      return None

    with open(sgf_file_name, "rb") as f:
      sgf = f.read()

    self._add_a_single_sgf_string(sgf)

  def _add_a_single_sgf_string(self,sgf_string):
    """add a single game in a sgf string save the results in self.results matrix. The self.results[i][j] save the
       number of games that player i beats player j"""
    game = sgf.Sgf_game.from_bytes(sgf_string)
    winner = game.get_winner()
    pla_black = game.get_player_name('b')
    pla_white = game.get_player_name('w')
    self._player_info_helper(pla_black)
    self._player_info_helper(pla_white)

    if (winner == 'b'):
      self.results[self.players[pla_black]][self.players[pla_white]] += 1
    elif (winner == 'w'):
      self.results[self.players[pla_white]][self.players[pla_black]] += 1
    else:
      self.results[self.players[pla_black]][self.players[pla_white]] += 0.5
      self.results[self.players[pla_white]][self.players[pla_black]] += 0.5

  def _player_info_helper(self, pla_name):
    """Check if the player has already added to the players dictionary. If not, we will add the player into players and expand the results table"""
    if pla_name in self.players.keys():
      return
    else:
      players_len = len(self.players)
      self.players[pla_name] = players_len
      self._expand_result_table()

  def _expand_result_table(self):
    """This is helper function to example result table when we add a new player"""
    players_len = len(self.players)
    last_player = [0] * players_len
    for i in range(players_len - 1):
      self.results[i].append(0)
    self.results.append(last_player)

  def _estimate_elo(self) -> elo.EloInfo:
    """Estimate and print elo values. This function must be called after parse all the sgfs files"""
    data = []
    pla_names = list(self.players.keys())
    n_pla = len(pla_names)
    for i in range(n_pla - 1):
      for j in range(i + 1, n_pla):
        win = self.results[self.players[pla_names[i]]][self.players[pla_names[j]]]
        lost = self.results[self.players[pla_names[j]]][self.players[pla_names[i]]]
        total_games = win + lost
        data.extend(elo.likelihood_of_games(pla_names[i], pla_names[j], total_games, win / total_games, False))

    data.extend(elo.make_center_elos_prior(pla_names, self._elo_prior))
    info = elo.compute_elos(data, verbose=True)
    return info

  def _print_player_info(self):
    max_len = len(max(self.players.keys(),key=len))
    title = 'Player Name'
    title_space = max(len(title), max_len)+1
    print(f"{title:<{title_space}}: Player ID")
    for pla in self.players:
      print(f"{pla:<{title_space}}: {self.players[pla]}")

  def _print_resultMatrix(self):
    max_game_played_space = int(math.log10(max([max(sublist) for sublist in self.results])))+5
    row_format =f"{{:>{max_game_played_space}}}" * (len(self.players) + 1)
    print(row_format.format("", *self.players.values()))
    for playerID, row in zip(self.players.values(), self.results):
      print(row_format.format(playerID, *row))


if __name__ == "__main__":
  description = """
  Summarize SGF/SGFs files and estimate Bayes Elo score for each of the player.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-input-dir', help='sgf/sgfs files input directory', required=True)
  parser.add_argument('-search-subdir', help='If we also need to search all sub-directories', required=False,
                      default=False)
  parser.add_argument('-elo-prior', help='A Bayesian prior that the mean of all player Elos is the specified Elo ',
                      required=False, type=float, default=0)
  args = vars(parser.parse_args())

  input_dir = args["input_dir"]
  search_subdir = args["search_subdir"]
  elo_prior = args["elo_prior"]

  gameResultSummary = GameResultSummary(elo_prior)
  gameResultSummary.add_games(input_dir, search_subdir)
  gameResultSummary.print_game_results()
  gameResultSummary.print_elos()


