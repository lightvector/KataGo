#include "../core/elo.h"
#include "../core/test.h"

#include <cmath>

static const double ELO_PER_LOG_GAMMA = 173.717792761;

static double logOnePlusExpX(double x) {
  if(x >= 50)
    return 50;
  return log(1+exp(x));
}

static double logOnePlusExpXSecondDerivative(double x) {
  double halfX = 0.5 * x;
  double denom = exp(halfX) + exp(-halfX);
  return 1 / (denom * denom);
}

double ComputeElos::probWin(double eloDiff) {
  double logGammaDiff = eloDiff / ELO_PER_LOG_GAMMA;
  return 1.0 / (1.0 + exp(-logGammaDiff));
}

static double logLikelihoodOfWL(
  double eloFirstMinusSecond,
  ComputeElos::WLRecord winRecord
) {
  double logGammaFirstMinusSecond = eloFirstMinusSecond / ELO_PER_LOG_GAMMA;
  double logProbFirstWin = -logOnePlusExpX(-logGammaFirstMinusSecond);
  double logProbSecondWin = -logOnePlusExpX(logGammaFirstMinusSecond);
  return winRecord.firstWins * logProbFirstWin + winRecord.secondWins * logProbSecondWin;
}

static double logLikelihoodOfWLSecondDerivative(
  double eloFirstMinusSecond,
  ComputeElos::WLRecord winRecord
) {
  double logGammaFirstMinusSecond = eloFirstMinusSecond / ELO_PER_LOG_GAMMA;
  double logProbFirstWinSecondDerivative = -logOnePlusExpXSecondDerivative(-logGammaFirstMinusSecond);
  double logProbSecondWinSecondDerivative = -logOnePlusExpXSecondDerivative(logGammaFirstMinusSecond);
  return (winRecord.firstWins * logProbFirstWinSecondDerivative + winRecord.secondWins * logProbSecondWinSecondDerivative)
    / (ELO_PER_LOG_GAMMA * ELO_PER_LOG_GAMMA);
}

//Compute only the part of the log likelihood depending on given player
static double computeLocalLogLikelihood(
  int player,
  const vector<double>& elos,
  const ComputeElos::WLRecord* winMatrix,
  int numPlayers,
  double priorWL
) {
  double logLikelihood = 0.0;
  for(int y = 0; y<numPlayers; y++) {
    if(y == player)
      continue;
    logLikelihood += logLikelihoodOfWL(elos[player] - elos[y], winMatrix[player*numPlayers+y]);
    logLikelihood += logLikelihoodOfWL(elos[y] - elos[player], winMatrix[y*numPlayers+player]);
  }
  logLikelihood += logLikelihoodOfWL(elos[player] - 0.0, ComputeElos::WLRecord(priorWL,priorWL));
     
  return logLikelihood;
}

//Compute the second derivative of the log likelihood with respect to the current player
static double computeLocalLogLikelihoodSecondDerivative(
  int player,
  const vector<double>& elos,
  const ComputeElos::WLRecord* winMatrix,
  int numPlayers,
  double priorWL
) {
  double logLikelihoodSecondDerivative = 0.0;
  for(int y = 0; y<numPlayers; y++) {
    if(y == player)
      continue;
    logLikelihoodSecondDerivative += logLikelihoodOfWLSecondDerivative(elos[player] - elos[y], winMatrix[player*numPlayers+y]);
    logLikelihoodSecondDerivative += logLikelihoodOfWLSecondDerivative(elos[y] - elos[player], winMatrix[y*numPlayers+player]);
  }
  logLikelihoodSecondDerivative += logLikelihoodOfWLSecondDerivative(elos[player] - 0.0, ComputeElos::WLRecord(priorWL,priorWL));
     
  return logLikelihoodSecondDerivative;
}


//Approximately compute the standard deviation of all players' Elos, assuming each time that all other
//player Elos are completely confident.
vector<double> ComputeElos::computeApproxEloStdevs(
  const vector<double>& elos,
  const ComputeElos::WLRecord* winMatrix,
  int numPlayers,
  double priorWL
) {
  //Very crude - just discretely model the distribution and look at what its stdev is
  vector<double> eloStdevs(numPlayers,0.0);

  const int radius = 1500;
  vector<double> relProbs(radius*2+1,0.0);
  const double step = 1.0; //one-elo increments
  
  for(int player = 0; player < numPlayers; player++) {
    double logLikelihood = computeLocalLogLikelihood(player,elos,winMatrix,numPlayers,priorWL);
    double sumRelProbs = 0.0;
    vector<double> tempElos = elos;
    for(int i = 0; i < radius*2+1; i++) {
      double elo = elos[player] + (i - radius) * step;
      tempElos[player] = elo;
      double newLogLikelihood = computeLocalLogLikelihood(player,tempElos,winMatrix,numPlayers,priorWL);
      relProbs[i] = exp(newLogLikelihood-logLikelihood);
      sumRelProbs += relProbs[i];
    }

    double secondMomentAroundElo = 0.0;
    for(int i = 0; i < radius*2+1; i++) {
      double elo = elos[player] + (i - radius) * step;
      secondMomentAroundElo += relProbs[i] / sumRelProbs * (elo - elos[player]) * (elo - elos[player]);
    }
    eloStdevs[player] = sqrt(secondMomentAroundElo);
  }
  return eloStdevs;
}

//MM algorithm
/*
vector<double> ComputeElos::computeElos(
  const ComputeElos::WLRecord* winMatrix,
  int numPlayers,
  double priorWL,
  int maxIters,
  double tolerance,
  ostream* out
) {
  vector<double> logGammas(numPlayers,0.0);

  vector<double> numWins(numPlayers,0.0);
  for(int x = 0; x<numPlayers; x++) {
    for(int y = 0; y<numPlayers; y++) {
      if(x == y)
        continue;
      numWins[x] += winMatrix[x*numPlayers+y].firstWins;
      numWins[y] += winMatrix[x*numPlayers+y].secondWins;
    }
  }

  vector<double> matchLogGammaSums(numPlayers*numPlayers);
  vector<double> priorMatchLogGammaSums(numPlayers);

  auto recomputeLogGammaSums = [&]() {
    for(int x = 0; x<numPlayers; x++) {
      for(int y = 0; y<numPlayers; y++) {
        if(x == y)
          continue;
        double maxLogGamma = std::max(logGammas[x],logGammas[y]);
        matchLogGammaSums[x*numPlayers+y] = maxLogGamma + log(exp(logGammas[x] - maxLogGamma) + exp(logGammas[y] - maxLogGamma));
      }
      double maxLogGamma = std::max(logGammas[x],0.0);
      priorMatchLogGammaSums[x] = maxLogGamma + log(exp(logGammas[x] - maxLogGamma) + exp(0.0 - maxLogGamma));
    }
  };
  
  auto iterate = [&]() {
    recomputeLogGammaSums();
    
    double maxEloDiff = 0;
    for(int x = 0; x<numPlayers; x++) {
      double oldLogGamma = logGammas[x];

      double sumInvDifficulty = 0.0;
      for(int y = 0; y<numPlayers; y++) {
        if(x == y)
          continue;
        double numGamesXY = winMatrix[x*numPlayers+y].firstWins + winMatrix[x*numPlayers+y].secondWins;
        double numGamesYX = winMatrix[y*numPlayers+x].firstWins + winMatrix[y*numPlayers+x].secondWins;
        sumInvDifficulty += numGamesXY / exp(matchLogGammaSums[x*numPlayers+y] - oldLogGamma);
        sumInvDifficulty += numGamesYX / exp(matchLogGammaSums[y*numPlayers+x] - oldLogGamma);
      }
      sumInvDifficulty += priorWL / exp(priorMatchLogGammaSums[x] - oldLogGamma);
      sumInvDifficulty += priorWL / exp(priorMatchLogGammaSums[x] - oldLogGamma);
      
      double logGammaDiff = log((numWins[x] + priorWL) / sumInvDifficulty);
      double newLogGamma = oldLogGamma + logGammaDiff;
      logGammas[x] = newLogGamma;

      double eloDiff = ELO_PER_LOG_GAMMA * abs(logGammaDiff);
      maxEloDiff = std::max(eloDiff,maxEloDiff);
    }
    return maxEloDiff;
  };

  for(int i = 0; i<maxIters; i++) {
    double maxEloDiff = iterate();
    if(out != NULL && i % 50 == 0) {
      (*out) << "Iteration " << i << " maxEloDiff = " << maxEloDiff << endl;
    }
    if(maxEloDiff < tolerance)
      break;
  }

  vector<double> elos(numPlayers,0.0);
  for(int x = 0; x<numPlayers; x++) {
    elos[x] = ELO_PER_LOG_GAMMA * logGammas[x];
  }
  return elos;
}
*/


vector<double> ComputeElos::computeElos(
  const ComputeElos::WLRecord* winMatrix,
  int numPlayers,
  double priorWL,
  int maxIters,
  double tolerance,
  ostream* out
) {
  vector<double> elos(numPlayers,0.0);


  //General gradient-free algorithm
  vector<double> nextDelta(numPlayers,100.0);  
  auto iterate = [&]() {
    double maxEloDiff = 0;
    for(int x = 0; x<numPlayers; x++) {
      double oldElo = elos[x];
      double hiElo = oldElo + nextDelta[x];
      double loElo = oldElo - nextDelta[x];

      double likelihood = computeLocalLogLikelihood(x,elos,winMatrix,numPlayers,priorWL);
      elos[x] = hiElo;
      double likelihoodHi = computeLocalLogLikelihood(x,elos,winMatrix,numPlayers,priorWL);
      elos[x] = loElo;
      double likelihoodLo = computeLocalLogLikelihood(x,elos,winMatrix,numPlayers,priorWL);

      if(likelihoodHi > likelihood) {
        elos[x] = hiElo;
        nextDelta[x] *= 1.1;
      }
      else if(likelihoodLo > likelihood) {
        elos[x] = loElo;
        nextDelta[x] *= 1.1;
      }
      else {
        elos[x] = oldElo;
        nextDelta[x] *= 0.4;
      }

      double eloDiff = nextDelta[x];
      maxEloDiff = std::max(eloDiff,maxEloDiff);
    }
    return maxEloDiff;
  };

  
  for(int i = 0; i<maxIters; i++) {
    double maxEloDiff = iterate();
    if(out != NULL && i % 50 == 0) {
      (*out) << "Iteration " << i << " maxEloDiff = " << maxEloDiff << endl;
    }
    if(maxEloDiff < tolerance)
      break;
  }
  
  return elos;
}


void ComputeElos::runTests() {
  ostringstream out;

  auto printEloStuff = [&](vector<double>& elos, ComputeElos::WLRecord* winMatrix, int numPlayers, double priorWL) {
    vector<double> eloStdevs = ComputeElos::computeApproxEloStdevs(elos,winMatrix,numPlayers,priorWL);
    for(int i = 0; i<numPlayers; i++) {
      double local2d = computeLocalLogLikelihoodSecondDerivative(i,elos,winMatrix,numPlayers,priorWL);

      double elo = elos[i];
      double eloHi = elos[i] + 1.0;
      double eloLo = elos[i] - 1.0;
      elos[i] = eloHi; double llHi = computeLocalLogLikelihood(i,elos,winMatrix,numPlayers,priorWL);
      elos[i] = eloLo; double llLo = computeLocalLogLikelihood(i,elos,winMatrix,numPlayers,priorWL);
      elos[i] = elo; double llMid = computeLocalLogLikelihood(i,elos,winMatrix,numPlayers,priorWL);

      double approx2d = llHi + llLo - 2.0 * llMid;
      
      out << "Elo " << i << " = " << elos[i] << " stdev " << eloStdevs[i] << " 2nd der " << local2d << " approx " << approx2d << endl;
    }
  };
  
  { 
    const char* name = "Elo test 0";

    int numPlayers = 1;
    ComputeElos::WLRecord* winMatrix = new ComputeElos::WLRecord[numPlayers*numPlayers];
    double priorWL = 0.1;
    int maxIters = 1000;
    double tolerance = 0.0001;

    vector<double> elos = ComputeElos::computeElos(winMatrix,numPlayers,priorWL,maxIters,tolerance,&out);
    
    string expected = R"%%(
Iteration 0 maxEloDiff = 40
Elo 0 = 0 stdev 780.104 2nd der -1.65684e-06 approx -1.65684e-06
)%%";

    printEloStuff(elos,winMatrix,numPlayers,priorWL);
    
    TestCommon::expect(name,out,expected);
    delete[] winMatrix;
  }

  { 
    const char* name = "Elo test 1";

    int numPlayers = 3;
    ComputeElos::WLRecord* winMatrix = new ComputeElos::WLRecord[numPlayers*numPlayers];
    double priorWL = 1.0;
    int maxIters = 1000;
    double tolerance = 0.0001;

    winMatrix[0*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+2] = ComputeElos::WLRecord(200.0,0.0);
    winMatrix[2*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[2*numPlayers+1] = ComputeElos::WLRecord(100.0,0.0);
    winMatrix[2*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);

    vector<double> elos = ComputeElos::computeElos(winMatrix,numPlayers,priorWL,maxIters,tolerance,&out);
    
    string expected = R"%%(
Iteration 0 maxEloDiff = 110
Iteration 50 maxEloDiff = 0.189831
Iteration 100 maxEloDiff = 0.14169
Iteration 150 maxEloDiff = 0.105757
Iteration 200 maxEloDiff = 0.0287043
Iteration 250 maxEloDiff = 0.00779085
Iteration 300 maxEloDiff = 0.00211458
Iteration 350 maxEloDiff = 0.00157832
Iteration 400 maxEloDiff = 0.000428383
Elo 0 = 4.39805e-07 stdev 313.547 2nd der -1.65684e-05 approx -1.65684e-05
Elo 1 = 59.9906 stdev 21.2381 2nd der -0.00222709 approx -0.00222709
Elo 2 = -59.9762 stdev 21.2381 2nd der -0.00222709 approx -0.00222709
)%%";

    printEloStuff(elos,winMatrix,numPlayers,priorWL);
    
    TestCommon::expect(name,out,expected);
    delete[] winMatrix;
  }

  { 
    const char* name = "Elo test 2";

    int numPlayers = 3;
    ComputeElos::WLRecord* winMatrix = new ComputeElos::WLRecord[numPlayers*numPlayers];
    double priorWL = 1.0;
    int maxIters = 1000;
    double tolerance = 0.0001;

    winMatrix[0*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+2] = ComputeElos::WLRecord(0.0,1.0);
    winMatrix[1*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+2] = ComputeElos::WLRecord(5.0,0.0);
    winMatrix[2*numPlayers+0] = ComputeElos::WLRecord(0.0,5.0);
    winMatrix[2*numPlayers+1] = ComputeElos::WLRecord(1.0,0.0);
    winMatrix[2*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);

    vector<double> elos = ComputeElos::computeElos(winMatrix,numPlayers,priorWL,maxIters,tolerance,&out);
    
    string expected = R"%%(
Iteration 0 maxEloDiff = 110
Elo 0 = 76.5227 stdev 162.965 2nd der -4.7933e-05 approx -4.7933e-05
Elo 1 = 76.5227 stdev 162.965 2nd der -4.7933e-05 approx -4.7933e-05
Elo 2 = -161.285 stdev 123.894 2nd der -7.77407e-05 approx -7.77407e-05
)%%";

    printEloStuff(elos,winMatrix,numPlayers,priorWL);
    
    TestCommon::expect(name,out,expected);
    delete[] winMatrix;
  }

  { 
    const char* name = "Elo test 3";

    int numPlayers = 3;
    ComputeElos::WLRecord* winMatrix = new ComputeElos::WLRecord[numPlayers*numPlayers];
    double priorWL = 1.0;
    int maxIters = 1000;
    double tolerance = 0.0001;

    winMatrix[0*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+2] = ComputeElos::WLRecord(0.0,1.0);
    winMatrix[2*numPlayers+0] = ComputeElos::WLRecord(5.0,1.0);
    winMatrix[2*numPlayers+1] = ComputeElos::WLRecord(0.0,5.0);
    winMatrix[2*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);

    vector<double> elos = ComputeElos::computeElos(winMatrix,numPlayers,priorWL,maxIters,tolerance,&out);
    
    string expected = R"%%(
Iteration 0 maxEloDiff = 110
Elo 0 = -190.848 stdev 161.134 2nd der -4.97053e-05 approx -4.97053e-05
Elo 1 = 190.848 stdev 161.134 2nd der -4.97053e-05 approx -4.97053e-05
Elo 2 = 0 stdev 106.849 2nd der -9.11264e-05 approx -9.11263e-05
)%%";

    printEloStuff(elos,winMatrix,numPlayers,priorWL);
    
    TestCommon::expect(name,out,expected);
    delete[] winMatrix;
  }
  
  { 
    const char* name = "Elo test 3";

    int numPlayers = 3;
    ComputeElos::WLRecord* winMatrix = new ComputeElos::WLRecord[numPlayers*numPlayers];
    double priorWL = 0.1;
    int maxIters = 10000;
    double tolerance = 0.0001;

    winMatrix[0*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+2] = ComputeElos::WLRecord(0.0,1.0);
    winMatrix[2*numPlayers+0] = ComputeElos::WLRecord(5.0,1.0);
    winMatrix[2*numPlayers+1] = ComputeElos::WLRecord(0.0,5.0);
    winMatrix[2*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);

    vector<double> elos = ComputeElos::computeElos(winMatrix,numPlayers,priorWL,maxIters,tolerance,&out);
    
    string expected = R"%%(
Iteration 0 maxEloDiff = 110
Elo 0 = -266.471 stdev 234.178 2nd der -2.99835e-05 approx -2.99835e-05
Elo 1 = 266.471 stdev 234.178 2nd der -2.99835e-05 approx -2.99835e-05
Elo 2 = -7.72341e-07 stdev 128.942 2nd der -5.96895e-05 approx -5.96895e-05
)%%";

    printEloStuff(elos,winMatrix,numPlayers,priorWL);
    
    TestCommon::expect(name,out,expected);
    delete[] winMatrix;
  }

  { 
    const char* name = "Elo test 4";

    int numPlayers = 3;
    ComputeElos::WLRecord* winMatrix = new ComputeElos::WLRecord[numPlayers*numPlayers];
    double priorWL = 0.01;
    int maxIters = 10000;
    double tolerance = 0.0001;

    winMatrix[0*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[0*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+0] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+1] = ComputeElos::WLRecord(0.0,0.0);
    winMatrix[1*numPlayers+2] = ComputeElos::WLRecord(0.0,1.0);
    winMatrix[2*numPlayers+0] = ComputeElos::WLRecord(7.0,1.0);
    winMatrix[2*numPlayers+1] = ComputeElos::WLRecord(0.0,5.0);
    winMatrix[2*numPlayers+2] = ComputeElos::WLRecord(0.0,0.0);

    vector<double> elos = ComputeElos::computeElos(winMatrix,numPlayers,priorWL,maxIters,tolerance,&out);
    
    string expected = R"%%(
Iteration 0 maxEloDiff = 110
Iteration 50 maxEloDiff = 0.0690295
Iteration 100 maxEloDiff = 0.0515235
Iteration 150 maxEloDiff = 0.0384571
Iteration 200 maxEloDiff = 0.0287043
Iteration 250 maxEloDiff = 0.0214248
Iteration 300 maxEloDiff = 0.0159915
Iteration 350 maxEloDiff = 0.032824
Iteration 400 maxEloDiff = 0.0244998
Iteration 450 maxEloDiff = 0.00664968
Iteration 500 maxEloDiff = 0.0136491
Iteration 550 maxEloDiff = 0.0101877
Iteration 600 maxEloDiff = 0.00760407
Iteration 650 maxEloDiff = 0.00567567
Iteration 700 maxEloDiff = 0.00423631
Iteration 750 maxEloDiff = 0.00316197
Iteration 800 maxEloDiff = 0.00649026
Iteration 850 maxEloDiff = 0.00176157
Iteration 900 maxEloDiff = 0.00361579
Iteration 950 maxEloDiff = 0.00269882
Iteration 1000 maxEloDiff = 0.0020144
Iteration 1050 maxEloDiff = 0.00150354
Iteration 1100 maxEloDiff = 0.00112224
Iteration 1150 maxEloDiff = 0.00083764
Iteration 1200 maxEloDiff = 0.00171934
Iteration 1250 maxEloDiff = 0.00128331
Iteration 1300 maxEloDiff = 0.000957861
Iteration 1350 maxEloDiff = 0.000714946
Iteration 1400 maxEloDiff = 0.000533635
Iteration 1450 maxEloDiff = 0.000398304
Iteration 1500 maxEloDiff = 0.000817557
Iteration 1550 maxEloDiff = 0.0002219
Iteration 1600 maxEloDiff = 0.00045547
Iteration 1650 maxEloDiff = 0.000339962
Iteration 1700 maxEloDiff = 0.000253747
Iteration 1750 maxEloDiff = 0.000189397
Iteration 1800 maxEloDiff = 0.000141365
Elo 0 = -322.047 stdev 246.125 2nd der -2.92533e-05 approx -2.92534e-05
Elo 1 = 292.708 stdev 248.472 2nd der -2.78531e-05 approx -2.78531e-05
Elo 2 = 14.549 stdev 129.788 2nd der -5.71067e-05 approx -5.71068e-05
)%%";

    printEloStuff(elos,winMatrix,numPlayers,priorWL);
    
    TestCommon::expect(name,out,expected);
    delete[] winMatrix;
  }

}
