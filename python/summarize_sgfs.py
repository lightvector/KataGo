from sgfmill import sgf
import elo
import os
import argparse

class SummarizeSGFs:
  """Summarize Go games in sgfs file format. Each row of sgfs game is a single game in a sgf string."""

  def __init__(
      self,
      file_path
  ):
    self.file_path = file_path
    self.sgf_files = []
    self.players = {}
    self.results = []
    self.get_sgfs_files()
    self.parse_all_sgfs_files()

  def get_sgfs_files(self):
    """Load all the sgfs file names in file_path directory into a list"""
    if not os.path.exists(self.file_path):
      print(f"There is no directory under name {self.file_path}")
      return None
    self.sgf_files = os.listdir(self.file_path)

    for file in self.sgf_files:
      if file.split(".")[-1].lower() != "sgfs":
        self.sgf_files.remove(file)
    print(f"Total {len(self.sgf_files)} sgfs files have been found under directory {self.file_path}.")


  def parse_all_sgfs_files(self):
    """Parse all sgfs files"""
    idx=1
    for file in self.sgf_files:
        self.parse_one_sgfs_file(file)
        print(f"Parsed {idx} files out of {len(self.sgf_files)} files.")
        idx+=1
    self.estimate_elo()


  def estimate_elo(self):
    """Estimate and print elo values. This function must be called after parse all the sgfs files"""
    data = []
    pla_names = list(self.players.keys())
    n_pla = len(pla_names)
    for i in range(n_pla-1):
      for j in range(i+1,n_pla):
        won = self.results[self.players[pla_names[i]]][self.players[pla_names[j]]]
        lost = self.results[self.players[pla_names[j]]][self.players[pla_names[i]]]
        total_games = won + lost
        data.extend(elo.likelihood_of_games(pla_names[i], pla_names[j], total_games, won/total_games, False))

    data.extend(elo.make_center_elos_prior(pla_names, 0))
    info = elo.compute_elos(data, verbose=True)
    print(info)

  def parse_one_sgfs_file(self, sgfs_file_name):
    """Parse a single sgfs file. Each row of such sgf file contain a single game as a sgf string"""
    file_location = os.path.join(self.file_path, sgfs_file_name)
    if not os.path.exists(file_location):
      print(f"There is no SGF file as {file_location}")
      return None

    with open(file_location, "rb") as f:
      sgfs_strings = f.readlines()

    for sgf in sgfs_strings:
      self.parse_a_single_sgf_string(sgf)

  def parse_a_single_sgf_string(self, sgf_string):
    """parse a single game in a sgf string save the results in self.results matrix. The self.results[i][j] save the
    number of games that player i beats player j"""
    game = sgf.Sgf_game.from_bytes(sgf_string)
    winner = game.get_winner()
    pla_black = game.get_player_name('b')
    pla_white = game.get_player_name('w')
    self.player_info_helper(pla_black)
    self.player_info_helper(pla_white)

    if(winner == 'b'):
      self.results[self.players[pla_black]][self.players[pla_white]]+=1
    elif(winner == 'w'):
      self.results[self.players[pla_white]][self.players[pla_black]] += 1
    else:
      self.results[self.players[pla_black]][self.players[pla_white]] += 0.5
      self.results[self.players[pla_white]][self.players[pla_black]] += 0.5

  def expand_result_table(self):
    """This is helper function to example result table when we add a new player"""
    players_len = len(self.players)
    last_player = [0]*players_len
    for i in range(players_len-1):
      self.results[i].append(0)
    self.results.append(last_player)


  def player_info_helper(self, pla_name):
    """Check if the player has already added to the players dictionary. If not, we will add the player into players and expand the results table"""
    if pla_name in self.players.keys():
      return
    else:
      players_len = len(self.players)
      self.players[pla_name]= players_len
      self.expand_result_table()


if __name__ == "__main__":
  description = """
  Summarize SGFs files and estimate Bayes Elo score for each of the player.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-sgfs-input-dir', help='sgfs files input directory', required=True)
  args = vars(parser.parse_args())

  file_path = args["sgfs_input_dir"]
  SummarizeSGFs(file_path)


