#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

// Updates mirroringPla, mirrorAdvantage, mirrorCenterSymmetryError
void Search::updateMirroring() {
  mirroringPla = C_EMPTY;
  mirrorAdvantage = 0.0;
  mirrorCenterSymmetryError = 1e10;

  if(searchParams.antiMirror) {
    const Board& board = rootBoard;
    const BoardHistory& hist = rootHistory;
    int mirrorCount = 0;
    int totalCount = 0;
    double mirrorEwms = 0;
    double totalEwms = 0;
    bool lastWasMirror = false;
    for(int i = 1; i<hist.moveHistory.size(); i++) {
      if(hist.moveHistory[i].pla != rootPla) {
        lastWasMirror = false;
        if(hist.moveHistory[i].loc == Location::getMirrorLoc(hist.moveHistory[i-1].loc,board.x_size,board.y_size)) {
          mirrorCount += 1;
          mirrorEwms += 1;
          lastWasMirror = true;
        }
        totalCount += 1;
        totalEwms += 1;
        mirrorEwms *= 0.75;
        totalEwms *= 0.75;
      }
    }
    //If at most of the moves in the game are mirror moves, and many of the recent moves were mirrors, and the last move
    //was a mirror, then the opponent is mirroring.
    if(mirrorCount >= 7.0 + 0.5 * totalCount && mirrorEwms >= 0.45 * totalEwms && lastWasMirror) {
      mirroringPla = getOpp(rootPla);

      double blackExtraPoints = 0.0;
      int numHandicapStones = hist.computeNumHandicapStones();
      if(hist.rules.scoringRule == Rules::SCORING_AREA) {
        if(numHandicapStones > 0)
          blackExtraPoints += numHandicapStones-1;
        bool blackGetsLastMove = (board.x_size % 2 == 1 && board.y_size % 2 == 1) == (numHandicapStones == 0 || numHandicapStones % 2 == 1);
        if(blackGetsLastMove)
          blackExtraPoints += 1;
      }
      if(numHandicapStones > 0 && hist.rules.whiteHandicapBonusRule == Rules::WHB_N)
        blackExtraPoints -= numHandicapStones;
      if(numHandicapStones > 0 && hist.rules.whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE)
        blackExtraPoints -= numHandicapStones-1;
      mirrorAdvantage = mirroringPla == P_BLACK ? blackExtraPoints - hist.rules.komi : hist.rules.komi - blackExtraPoints;
    }

    if(board.x_size >= 7 && board.y_size >= 7) {
      mirrorCenterSymmetryError = 0.0;
      int halfX = board.x_size / 2;
      int halfY = board.y_size / 2;
      int unmatchedMirrorPlaStones = 0;
      for(int dy = -3; dy <= 3; dy++) {
        for(int dx = -3; dx <= 3; dx++) {
          Loc loc = Location::getLoc(halfX+dx,halfY+dy,board.x_size);
          Loc mirrorLoc = Location::getMirrorLoc(loc,board.x_size,board.y_size);
          if(loc == mirrorLoc)
            continue;
          Color c0 = board.colors[loc];
          Color c1 = board.colors[mirrorLoc];
          if(c0 == getOpp(mirroringPla) && c1 != mirroringPla)
            mirrorCenterSymmetryError += 1.0;
          if(c0 == mirroringPla && c1 == C_EMPTY)
            unmatchedMirrorPlaStones += 1;
        }
      }
      if(mirrorCenterSymmetryError > 0.0)
        mirrorCenterSymmetryError += 0.2 * unmatchedMirrorPlaStones;
      if(mirrorCenterSymmetryError >= 1.0)
        mirrorCenterSymmetryError = 0.5 * mirrorCenterSymmetryError * (1.0 + mirrorCenterSymmetryError);
    }
  }
}

bool Search::isMirroringSinceSearchStart(const BoardHistory& threadHistory, int skipRecent) const {
  int xSize = threadHistory.initialBoard.x_size;
  int ySize = threadHistory.initialBoard.y_size;
  for(size_t i = rootHistory.moveHistory.size()+1; i+skipRecent < threadHistory.moveHistory.size(); i += 2) {
    if(threadHistory.moveHistory[i].loc != Location::getMirrorLoc(threadHistory.moveHistory[i-1].loc,xSize,ySize))
      return false;
  }
  return true;
}

void Search::maybeApplyAntiMirrorPolicy(
  float& nnPolicyProb,
  const Loc moveLoc,
  const float* policyProbs,
  const Player movePla,
  const SearchThread* thread
) const {
  int xSize = thread->board.x_size;
  int ySize = thread->board.y_size;

  double weight = 0.0;

  //Put significant prior probability on the opponent continuing to mirror, at least for the next few turns.
  if(movePla == getOpp(rootPla) && thread->history.moveHistory.size() > 0) {
    Loc prevLoc = thread->history.moveHistory[thread->history.moveHistory.size()-1].loc;
    if(prevLoc == Board::PASS_LOC)
      return;
    Loc mirrorLoc = Location::getMirrorLoc(prevLoc,xSize,ySize);
    if(policyProbs[getPos(mirrorLoc)] < 0)
      mirrorLoc = Board::PASS_LOC;
    if(moveLoc == mirrorLoc) {
      weight = 1.0;
      Loc centerLoc = Location::getCenterLoc(xSize,ySize);
      bool isDifficult = centerLoc != Board::NULL_LOC && thread->board.colors[centerLoc] == mirroringPla && mirrorAdvantage >= -0.5;
      if(isDifficult)
        weight *= 3.0;
    }
  }
  //Put a small prior on playing the center or attaching to center, bonusing moves that are relatively more likely.
  else if(movePla == rootPla && moveLoc != Board::PASS_LOC) {
    if(Location::isCentral(moveLoc,xSize,ySize))
      weight = 0.3;
    else {
      if(Location::isNearCentral(moveLoc,xSize,ySize))
        weight = 0.05;

      Loc centerLoc = Location::getCenterLoc(xSize,ySize);
      if(centerLoc != Board::NULL_LOC) {
        if(rootBoard.colors[centerLoc] == getOpp(movePla)) {
          if(thread->board.isAdjacentToChain(moveLoc,centerLoc))
            weight = 0.05;
          else {
            int distanceSq = Location::euclideanDistanceSquared(moveLoc,centerLoc,xSize);
            if(distanceSq <= 2)
              weight = 0.05;
            else if(distanceSq <= 4)
              weight = 0.03;
          }
        }
      }
    }
  }

  if(weight > 0) {
    weight = weight / (1.0 + sqrt(thread->history.moveHistory.size() - rootHistory.moveHistory.size()));
    nnPolicyProb = nnPolicyProb + (1.0f - nnPolicyProb) * (float)weight;
  }
}

//Force the search to dump playouts down a mirror move, so as to encourage moves that cause mirror moves
//to have bad values, and also tolerate us playing certain countering moves even if their values are a bit worse.
void Search::maybeApplyAntiMirrorForcedExplore(
  double& childUtility,
  const double parentUtility,
  const Loc moveLoc,
  const float* policyProbs,
  const double thisChildWeight,
  const double totalChildWeight,
  const Player movePla,
  const SearchThread* thread,
  const SearchNode& parent
) const {
  assert(mirroringPla == getOpp(rootPla));

  int xSize = thread->board.x_size;
  int ySize = thread->board.y_size;
  Loc centerLoc = Location::getCenterLoc(xSize,ySize);
  //The difficult case is when the opponent has occupied tengen, and ALSO the komi favors them.
  //In such a case, we're going to have a hard time.
  //Technically there are other configurations (like if the opponent makes a diamond around tengen)
  //but we're not going to worry about breaking that.
  bool isDifficult = centerLoc != Board::NULL_LOC && thread->board.colors[centerLoc] == mirroringPla && mirrorAdvantage >= -0.5;
  // bool isSemiDifficult = !isDifficult && mirrorAdvantage >= 6.5;
  bool isRoot = &parent == rootNode;

  //Force mirroring pla to dump playouts down mirror moves
  if(movePla == mirroringPla && thread->history.moveHistory.size() > 0) {
    Loc prevLoc = thread->history.moveHistory[thread->history.moveHistory.size()-1].loc;
    if(prevLoc == Board::PASS_LOC)
      return;
    Loc mirrorLoc = Location::getMirrorLoc(prevLoc,xSize,ySize);
    if(policyProbs[getPos(mirrorLoc)] < 0)
      mirrorLoc = Board::PASS_LOC;
    if(moveLoc == mirrorLoc) {
      double proportionToDump = 0.0;
      double proportionToBias = 0.0;
      if(isDifficult) {
        proportionToDump = 0.20;
        if(mirrorLoc != Board::PASS_LOC) {
          proportionToDump = std::max(
            proportionToDump,
            1.0 / (0.75 + 0.5 * sqrt(Location::euclideanDistanceSquared(centerLoc,mirrorLoc,xSize)))
            / std::max(1.0,mirrorCenterSymmetryError)
          );
        }
        proportionToBias = 0.75;
      }
      else if(mirrorAdvantage >= 5.0) {
        proportionToDump = 0.15;
        proportionToBias = 0.50;
      }
      else if(mirrorAdvantage >= -5.0) {
        proportionToDump = 0.10 + mirrorAdvantage;
        proportionToBias = 0.30 + mirrorAdvantage * 4;
      }
      else {
        proportionToDump = 0.05;
        proportionToBias = 0.10;
      }

      if(mirrorLoc == Board::PASS_LOC)
        proportionToDump *= (moveLoc == centerLoc ? 0.35 : 0.35 / std::max(1.0,sqrt(mirrorCenterSymmetryError)));
      if(mirrorCenterSymmetryError >= 1.0) {
        proportionToDump /= mirrorCenterSymmetryError;
        proportionToBias /= mirrorCenterSymmetryError;
      }

      if(thisChildWeight < proportionToDump * totalChildWeight) {
        childUtility += (parent.nextPla == P_WHITE ? 100.0 : -100.0);
      }
      if(thisChildWeight < proportionToBias * totalChildWeight) {
        childUtility += (parent.nextPla == P_WHITE ? 0.18 : -0.18) * std::max(0.3, 1.0 - 0.7 * parentUtility * parentUtility);
      }
      if(thisChildWeight < 0.5 * proportionToBias * totalChildWeight) {
        childUtility += (parent.nextPla == P_WHITE ? 0.36 : -0.36) * std::max(0.3, 1.0 - 0.7 * parentUtility * parentUtility);
      }
    }
  }
  //Encourage us to find refuting moves, even if they look a little bad, in the difficult case
  //Force us to dump playouts down tengen if possible, to encourage us to make tengen into a good move.
  else if(movePla == rootPla && moveLoc != Board::PASS_LOC) {
    double proportionToDump = 0.0;
    if(isDifficult) {
      if(thread->board.isAdjacentToChain(moveLoc,centerLoc)) {
        childUtility += (parent.nextPla == P_WHITE ? 0.75 : -0.75) / (1.0 + thread->board.getNumLiberties(centerLoc))
          / std::max(1.0,mirrorCenterSymmetryError) * std::max(0.3, 1.0 - 0.7 * parentUtility * parentUtility);
        proportionToDump = 0.10 / thread->board.getNumLiberties(centerLoc);
      }
      int distanceSq = Location::euclideanDistanceSquared(moveLoc,centerLoc,xSize);
      if(distanceSq <= 2)
        proportionToDump = std::max(proportionToDump, 0.010);
      else if(distanceSq <= 4)
        proportionToDump = std::max(proportionToDump, 0.005);

      //proportionToDump *= (1.0 / (1.0 + sqrt(thread->history.moveHistory.size() - rootHistory.moveHistory.size())));
    }
    if(moveLoc == centerLoc) {
      if(isRoot)
        proportionToDump = 0.06;
      else
        proportionToDump = 0.12;
    }

    double utilityLoss = (parent.nextPla == P_WHITE) ? parentUtility - childUtility : childUtility - parentUtility;
    if(utilityLoss > 0 && utilityLoss * proportionToDump > 0.03)
      proportionToDump += 0.5 * (0.03 / utilityLoss - proportionToDump);

    if(thread->history.moveHistory.size() > 0) {
      Loc prevLoc = thread->history.moveHistory[thread->history.moveHistory.size()-1].loc;
      if(prevLoc != Board::NULL_LOC && prevLoc != Board::PASS_LOC) {
        int centerDistanceSquared = Location::euclideanDistanceSquared(centerLoc,prevLoc,xSize);
        if(centerDistanceSquared <= 16)
          proportionToDump *= 0.900;
        if(centerDistanceSquared <= 5)
          proportionToDump *= 0.825;
        if(centerDistanceSquared <= 2)
          proportionToDump *= 0.750;
      }
    }

    if(thisChildWeight < proportionToDump * totalChildWeight) {
      childUtility += (parent.nextPla == P_WHITE ? 100.0 : -100.0);
    }
  }
}

void Search::hackNNOutputForMirror(std::shared_ptr<NNOutput>& result) const {
  // Root player gets a bonus/penalty based on the strength of the center.
  int centerPos = getPos(Location::getCenterLoc(rootBoard));
  double totalWLProb = result->whiteWinProb + result->whiteLossProb;
  double ownScale = mirrorCenterSymmetryError <= 0.0 ? 0.7 : 0.3;
  double wl = (result->whiteWinProb - result->whiteLossProb) / (totalWLProb+1e-10);
  wl = std::min(std::max(wl,-1.0+1e-15),1.0-1e-15);
  wl = tanh(atanh(wl) + ownScale * result->whiteOwnerMap[centerPos]);
  double whiteNewWinProb = 0.5 + 0.5 * wl;
  whiteNewWinProb = totalWLProb * whiteNewWinProb;

  result->whiteWinProb = (float)whiteNewWinProb;
  result->whiteLossProb = (float)(totalWLProb - whiteNewWinProb);
}

