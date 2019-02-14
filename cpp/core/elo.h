#ifndef ELO_H
#define ELO_H

#include "../core/global.h"

namespace ComputeElos {
  STRUCT_NAMED_PAIR(double,firstWins,double,secondWins,WLRecord);
    
  //winMatrix[a*numPlayers+b] should be a matrix of the record a has versus b when a is playing first.
  //priorWL is the number of wins and number of losses against a virtual 0-elo opponent
  vector<double> computeElos(
    const WLRecord* winMatrix,
    int numPlayers,
    double priorWL,
    int maxIters,
    double tolerance,
    ostream* out
  );

  //Approximately compute the standard deviation of all players' Elos, assuming each time that all other
  //player Elos are completely confident.
  //Uses a local normal approximation at the final optimal point.
  vector<double> computeApproxEloStdevs(
    const vector<double>& elos,
    const WLRecord* winMatrix,
    int numPlayers,
    double priorWL
  );

  //What's the probability of winning correspnding to this elo difference?
  double probWin(double eloDiff);

  void runTests();
}


#endif
