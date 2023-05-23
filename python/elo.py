import copy
import math
import numpy as np
import scipy.stats
from typing import List, Dict, Tuple, Set, Sequence

Player = str
PlayerIdx = int

ELO_PER_STRENGTH = 400.0 * math.log10(math.exp(1.0))
P1_ADVANTAGE_NAME = "P1Advantage"

class EloInfo:
    """Summarizes maximum likelihood Elos and uncertainties for a group of players."""

    def __init__(
        self,
        players: List[Player],
        elo: Dict[Player,float],
        elo_stderr: Dict[Player,float],
        elo_covariance: Dict[Tuple[Player,Player],float],
        effective_game_count: Dict[Player,float],
    ):
        self.players = players
        self.elo = elo
        self.elo_stderr = elo_stderr
        self.elo_covariance = elo_covariance
        self.effective_game_count = effective_game_count

        self.players = sorted(self.players, key=(lambda player: -elo[player] if player != P1_ADVANTAGE_NAME else 1e50))

    def get_players(self) -> List[Player]:
        return self.players

    def get_elo(self, p1: Player) -> float:
        """Returns the maximum likelihood Elo of p1"""
        return self.elo[p1]

    def get_approx_elo_stderr(self, p1: Player) -> float:
        """Returns an approximation of the standard error on the Elo of p1, ASSUMING all other players Elos are equal to their maximum likelihood value.
        This approximation may underestimate if the amount of data is very small."""
        return self.elo_stderr[p1]

    def get_elo_difference(self, p1: Player, p2: Player) -> float:
        """Returns the maximum likelhood difference in Elo between p1 and p2"""
        return self.elo[p1] - self.elo[p2]

    def get_approx_elo_difference_stderr(self, p1: Player, p2: Player) -> float:
        """Returns an approximation of the standard error on difference in Elo between p1 and p2.
        This approximation may underestimate if the amount of data is very small."""
        return math.sqrt(
          self.elo_covariance[(p1,p1)] - self.elo_covariance[(p1,p2)] - self.elo_covariance[(p2,p1)] + self.elo_covariance[(p2,p2)]
        )

    def get_approx_likelihood_of_superiority(self, p1: Player, p2: Player) -> float:
        """Returns an approximation of the likelihood that elo(p1) - elo(p2) > 0, given the data.
        This approximation may be overconfident or inaccurate if the amount of data is very small."""
        if p1 == p2:
            return 0.5
        mean = self.get_elo_difference(p1,p2)
        stderr = self.get_approx_elo_difference_stderr(p1,p2)
        return scipy.stats.t.cdf(mean/stderr,df=self.effective_game_count[p1]-1)

    def __str__(self) -> str:
        lines = []
        for player in self.players:
            lines.append(f"{player:20s}: {self.elo[player]:8.2f} +/- {self.elo_stderr[player]:5.2f}")
        return "\n".join(lines)


class Likelihood:
    """Summarizes the information in an observed datapoint about player Elos, or a prior about them.

    Represents that sum_{p in playercombo} Strength(p)*playercombo[p] + offset has likelihood function f raised to the power of weight.
    If kind is SIGMOID_KIND, f is the sigmoid function.
    If kind is GAUSSIAN_KIND, f is the pdf of a unit gaussian

    where strength is such that 1 unit of strength is e:1 odds of winning in a head-to-head game.
    """

    SIGMOID_KIND = 1
    GAUSSIAN_KIND = 2

    def __init__(
        self,
        playercombo: Dict[Player,float],
        offset: float,
        weight: float,
        gamecount: float,
        kind: int,
    ):
        self.playercombo = playercombo
        self.offset = offset
        self.weight = weight
        self.gamecount = gamecount
        self.kind = kind
        assert kind == Likelihood.SIGMOID_KIND or kind == Likelihood.GAUSSIAN_KIND, "invalid kind"

    def add_idxs(self, player_to_idx: Dict[Player,PlayerIdx]):
        self.pidxcombo : List[Tuple[PlayerIdx,float]] = [(player_to_idx[player],coeff) for (player,coeff) in self.playercombo.items()]

    LOG_ONE_OVER_SQRT_TWO_PI = math.log(1.0 / math.sqrt(2.0 * math.pi))

    def get_loglikelihood(self, strengths: np.array) -> float:
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            if strength_total < -40:
                return strength_total
            return -self.weight * math.log(1.0 + math.exp(-strength_total))
        else:
            return self.weight * (Likelihood.LOG_ONE_OVER_SQRT_TWO_PI - 0.5 * strength_total * strength_total)

    def accum_dloglikelihood_dstrength(self, strengths: np.array, accum: np.array):
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            dloglikelihood_dstrength_total = self.weight / (1.0 + math.exp(strength_total))
        else:
            dloglikelihood_dstrength_total = -self.weight * strength_total
        for (pidx,coeff) in self.pidxcombo:
            accum[pidx] += coeff * dloglikelihood_dstrength_total

    def accum_d2loglikelihood_dstrength2(self, strengths: np.array, accum: np.array):
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            denom = math.exp(-0.5 * strength_total) + math.exp(0.5 * strength_total)
            d2loglikelihood_dstrength_total2 = -self.weight / (denom * denom)
        else:
            d2loglikelihood_dstrength_total2 = -self.weight

        for (pidx1,coeff1) in self.pidxcombo:
            for (pidx2,coeff2) in self.pidxcombo:
                accum[pidx1,pidx2] += coeff1 * coeff2 * d2loglikelihood_dstrength_total2

    def accum_d2loglikelihood_dstrength2_scalepow(self, strengths: np.array, accum: np.array, scale: float, power: float):
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            denom = math.exp(-0.5 * strength_total) + math.exp(0.5 * strength_total)
            d2loglikelihood_dstrength_total2 = -self.weight / (denom * denom)
        else:
            d2loglikelihood_dstrength_total2 = -self.weight

        for (pidx1,coeff1) in self.pidxcombo:
            for (pidx2,coeff2) in self.pidxcombo:
                x = coeff1 * coeff2 * d2loglikelihood_dstrength_total2
                accum[pidx1,pidx2] += scale * (x ** power)


def likelihood_of_games(
    p1: Player,
    p2: Player,
    num_games: float,
    p1_won_proportion: float,
    include_first_player_advantage: bool,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing the result of set of games between p1 and p2

    These Likelihoods can accumulated with any other Likelihoods, and then all passed to compute_elos
    to compute maximum likelihood Elos for all the players.

    NOTE: For performance reasons, you should try to minimize the number of these you create. If p1 and p2 played
    a large number of games, don't call this function once per game. Instead, call it once for all the games
    together (or twice, if you are using include_first_player_advantage=True, separately reporting the stats
    that occured when each side was the first player).

    NOTE: If specifying include_first_player_advantage=True, make sure to add a make_single_player_prior on
    P1_ADVANTAGE_NAME.

    Arguments:
    p1: Name of the first player
    p2: Name of the second player
    num_games: The number of games played
    p1_won_proportion: The proportion of games that p1 won among those games played, counting draws as 0.5.
    include_first_player_advantage: If true, will also make the computation take into account that the first player
      might have an advantage (or a disadvantage!) and it will try to estimate the amount of that advantage.

    Returns:
    List of likelihood objects summarizing the information.
    """

    ret = []
    assert p1_won_proportion >= 0.0 and p1_won_proportion <= 1.0
    assert num_games >= 0.0
    assert p1 != p2

    if num_games > 0.0:
        if not include_first_player_advantage:
            if p1_won_proportion > 0.0:
                ret.append(Likelihood(
                    playercombo={p1: 1.0, p2: -1.0},
                    offset=0.0,
                    weight=p1_won_proportion*num_games,
                    gamecount=p1_won_proportion*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))
            if p1_won_proportion < 1.0:
                ret.append(Likelihood(
                    playercombo={p2: 1.0, p1: -1.0},
                    offset=0.0,
                    weight=(1.0-p1_won_proportion)*num_games,
                    gamecount=(1.0-p1_won_proportion)*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))
        else:
            if p1_won_proportion > 0.0:
                ret.append(Likelihood(
                    playercombo={p1: 1.0, p2: -1.0, P1_ADVANTAGE_NAME: 1.0},
                    offset=0.0,
                    weight=p1_won_proportion*num_games,
                    gamecount=p1_won_proportion*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))
            if p1_won_proportion < 1.0:
                ret.append(Likelihood(
                    playercombo={p2: 1.0, p1: -1.0, P1_ADVANTAGE_NAME: -1.0},
                    offset=0.0,
                    weight=(1.0-p1_won_proportion)*num_games,
                    gamecount=(1.0-p1_won_proportion)*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))

    return ret

def make_single_player_prior(
    p1: Player,
    num_games: float,
    elo: float,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing a Bayesian prior that p1 is the specified Elo.

    The strength of the prior that p1 is the specified Elo is as if p1 were observed to have played
    num_games many games against a known player of that Elo and won half and lost half.

    Returns:
    List of likelihood objects summarizing the information.
    """
    ret = []
    assert num_games >= 0.0
    assert np.isfinite(elo)
    if num_games > 0.0:
        ret.append(Likelihood(
            playercombo={p1: 1.0},
            offset=(-elo / ELO_PER_STRENGTH),
            weight=0.5*num_games,
            gamecount=0.5*num_games,
            kind=Likelihood.SIGMOID_KIND
        ))
        ret.append(Likelihood(
            playercombo={p1: -1.0},
            offset=(elo / ELO_PER_STRENGTH),
            weight=0.5*num_games,
            gamecount=0.5*num_games,
            kind=Likelihood.SIGMOID_KIND
        ))
    return ret


def make_sequential_prior(
    players: List[Player],
    num_games: float,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing a Bayesian prior that each player in the sequence is similar in strength to the previous.

    This can be used, for example, if there were a sequence of changes between different versions, such that each version on average
    is expected to be more similar to its neighbors.

    The strength of the prior between each sequential pair of players is as if they were observed to have played
    num_games many games against each other and won half and lost half.

    Returns:
    List of likelihood objects summarizing the information.
    """
    ret = []
    assert num_games >= 0.0
    assert len(set(players)) == len(players), "players must not contain any duplicates"

    if len(players) < 1:
        return ret

    for i in range(len(players)-1):
        ret.extend(likelihood_of_games(
            p1=players[i],
            p2=players[i+1],
            num_games=num_games,
            p1_won_proportion=0.5,
            include_first_player_advantage=False,
        ))
    return ret


def make_center_elos_prior(
    players: Sequence[Player],
    elo: float,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing a Bayesian prior that the mean of all player Elos is the specified Elo.

    This prior will have no effect on the relative Elos of the players, unless it fights with another that sets players to
    specific Elos, such as make_single_player_prior. It can simply be used to center all the Elos of the players.

    Returns:
    List of likelihood objects summarizing the information.
    """
    ret = []
    assert np.isfinite(elo)
    assert len(set(players)) == len(players), "players must not contain any duplicates"
    playercombo = { player: 1.0 for player in players }
    ret.append(Likelihood(
        playercombo=playercombo,
        offset=-len(players) * elo / ELO_PER_STRENGTH,
        weight=0.001,
        gamecount=0.0,
        kind=Likelihood.GAUSSIAN_KIND
    ))
    return ret


def compute_elos(
    data: List[Likelihood],
    tolerance: float = 0.001,
    max_iters: int = 1000,
    verbose: bool = False,
) -> EloInfo:
    """Compute maximum-likelihood Elo ratings given the data and any priors.

    NOTE: It is recommend that you specify some sort of prior, even if relatively weak, for numerical stability.
    When you call this function, it is up to you to make sure that the data and priors you have provided result in all
    all Elos of all players being "anchored". For example, this function may crash or fail or return bad values if:

    * There is a player with no data at all, and no prior for that player.
      (since that would mean an Elo for that player cannot be defined).

    * There is a player who has only won and never lost, and there is no prior that restricts that player's rating.
      (since then nothing could stop player's Elo from going to infinity).

    * All players *have* won and lost against other players, but there is nothing that anchors the Elo of the
      population as a whole, such no player having a make_single_player_prior AND there being no make_center_elos_prior.
      (since even if all players Elos are known relative to one another, the Elos to report are undefined - there is nothing
       to say where the Elos should be centered or what value they should start from).

    Examples of things that are normally sufficient to anchor everyone:
    If all players have won and lost against other players, adding a make_center_elos_prior is sufficient.
    If all players have won and lost against other players, AND every player has transitively beat every other player and
    transitively lost to every other player, then adding a make_single_player_prior to one player is sufficient.
    Regardless of whether the players have won or lost, adding a make_single_player_prior to every player is sufficient.
    Regardless of whether the players have won or lost, adding both a make_sequential_prior and a make_center_elos_prior is sufficient.

    NOTE: Even aside from ensuring numeric stability and anchoring as above, it is probably good practice to add some mild prior
    beyond that anyways. If your players are sequential (e.g. a series of different nets), a weak make_sequential_prior could be good.
    If you want to just put all players on equal footing unbiasedly, adding a weak make_single_player_prior to each player that its Elo
    is 0 (or 1000, or whatever) is also good.

    And if you have actual prior beliefs about the players Elos, feel free to add those.

    Arguments:
    data: A single list of all the likelihoods from all your games and priors.
    tolerance: Stop soon after the Elos stop changing by more than this.
    max_iters: Bail out if the optimization takes more than this many iterations.
    verbose: Print out the iteration as it proceeds.

    Returns:
    Elos. Yay!
    """

    players = []
    for d in data:
        players.extend(d.playercombo.keys())
    players = list(set(players))
    players.sort()
    player_to_idx = { player: i for (i,player) in enumerate(players) }

    data = [copy.copy(d) for d in data]
    for d in data:
        d.add_idxs(player_to_idx)

    num_players = len(players)

    def compute_loglikelihood(strengths: np.array) -> float:
        total = 0.0
        for d in data:
            total += d.get_loglikelihood(strengths)
        return total

    # Gauss newton
    def find_ascent_vector(strengths: np.array) -> np.array:
        dloglikelihood_dstrength = np.zeros(num_players,dtype=np.float64)
        d2loglikelihood_dstrength2 = np.zeros((num_players,num_players),dtype=np.float64)

        for d in data:
            d.accum_dloglikelihood_dstrength(strengths, dloglikelihood_dstrength)
            d.accum_d2loglikelihood_dstrength2(strengths, d2loglikelihood_dstrength2)

        ascent = -np.linalg.solve(d2loglikelihood_dstrength2,dloglikelihood_dstrength)
        return ascent

    def line_search_ascend(strengths: np.array, cur_loglikelihood: float) -> Tuple[np.array,float]:
        ascent = find_ascent_vector(strengths)
        # Try up to this many times to find an improvement
        for i in range(30):
            new_strengths = strengths + ascent
            new_loglikelihood = compute_loglikelihood(new_strengths)
            if new_loglikelihood > cur_loglikelihood:
                return (new_strengths, new_loglikelihood)
            # Shrink ascent step and try again
            ascent *= 0.6
        return (strengths,cur_loglikelihood)

    strengths = np.zeros(num_players,dtype=np.float64)
    loglikelihood = compute_loglikelihood(strengths)
    iters_since_big_change = 0
    last_elo_change = None
    for i in range(max_iters):
        if verbose:
            print(f"Beginning iteration {i}, cur log likelihood {loglikelihood}, last elo change {last_elo_change}")

        (new_strengths, new_loglikelihood) = line_search_ascend(strengths, loglikelihood)
        elodiff = (new_strengths - strengths) * ELO_PER_STRENGTH
        last_elo_change = 0 if len(elodiff) <= 0 else np.max(np.abs(elodiff))

        strengths = new_strengths
        loglikelihood = new_loglikelihood

        iters_since_big_change += 1
        if np.any(elodiff > tolerance):
            iters_since_big_change = 0

        if iters_since_big_change > 3:
            break


    d2loglikelihood_dstrength2 = np.zeros((num_players,num_players),dtype=np.float64)
    for d in data:
        d.accum_d2loglikelihood_dstrength2(strengths, d2loglikelihood_dstrength2)
    strength_precision = -d2loglikelihood_dstrength2
    elo_precision = strength_precision / (ELO_PER_STRENGTH * ELO_PER_STRENGTH)
    elo_covariance = np.linalg.inv(elo_precision)

    sqrt_ess_numerator = np.zeros((num_players,num_players),dtype=np.float64)
    ess_denominator = np.zeros((num_players,num_players),dtype=np.float64)
    for d in data:
        if d.gamecount > 0.0:
            d.accum_d2loglikelihood_dstrength2_scalepow(strengths, sqrt_ess_numerator, scale = 1.0, power=1.0)
            d.accum_d2loglikelihood_dstrength2_scalepow(strengths, ess_denominator, scale = 1.0 / d.gamecount, power=2.0)

    info = EloInfo(
      players = players,
      elo = { player: ELO_PER_STRENGTH * strengths[player_to_idx[player]] for player in players },
      elo_stderr = { player: math.sqrt(1.0 / elo_precision[player_to_idx[player],player_to_idx[player]]) for player in players },
      elo_covariance = { (p1,p2): elo_covariance[player_to_idx[p1],player_to_idx[p2]] for p1 in players for p2 in players },
      effective_game_count = {
          player: (np.square(sqrt_ess_numerator[player_to_idx[player],player_to_idx[player]]) /
                   ess_denominator[player_to_idx[player],player_to_idx[player]])
          for player in players
      },
    )
    return info


# Testing code
if __name__ == "__main__":
    data = []
    data.extend(likelihood_of_games("Alice","Bob", 18, 12/18, False))
    data.extend(likelihood_of_games("Bob","Carol", 18, 12/18, False))
    data.extend(likelihood_of_games("Carol","Dan", 36, 12/18, False))
    data.extend(likelihood_of_games("Dan","Eve", 48, 40/48, False))
    data.extend(make_center_elos_prior(["Alice","Bob","Carol","Dan","Eve"],0))
    info = compute_elos(data,verbose=True)

    for player in info.players:
        for player2 in info.players:
            print(info.get_approx_likelihood_of_superiority(player,player2),end=" ")
        print()
