#include "board.h"
#include <vector>

#include "../program/play.h"

using namespace std;

static constexpr int PLACED_PLAYER_SHIFT = PLAYER_BITS_COUNT;
static constexpr int EMPTY_TERRITORY_SHIFT = PLACED_PLAYER_SHIFT + PLAYER_BITS_COUNT;
static constexpr int TERRITORY_FLAG_SHIFT = EMPTY_TERRITORY_SHIFT + PLAYER_BITS_COUNT;
static constexpr int GROUNDED_FLAG_SHIFT = TERRITORY_FLAG_SHIFT + 1;

static constexpr State TERRITORY_FLAG = 1 << TERRITORY_FLAG_SHIFT;
static constexpr State GROUNDED_FLAG = static_cast<State>(1 << GROUNDED_FLAG_SHIFT);
static constexpr State INVALIDATE_TERRITORY_MASK = ~(ACTIVE_MASK | ACTIVE_MASK << EMPTY_TERRITORY_SHIFT);

Loc Location::xm1y(Loc loc) {
  return loc - 1;
}

Loc Location::xm1ym1(Loc loc, int x_size) {
  return loc - 1 - (x_size+1);
}

Loc Location::xym1(Loc loc, int x_size) {
  return loc - (x_size+1);
}

Loc Location::xp1ym1(Loc loc, int x_size) {
  return loc + 1 - (x_size+1);
}

Loc Location::xp1y(Loc loc) {
  return loc + 1;
}

Loc Location::xp1yp1(Loc loc, int x_size) {
  return loc + 1 + (x_size+1);
}

Loc Location::xyp1(Loc loc, int x_size) {
  return loc + (x_size+1);
}

Loc Location::xm1yp1(Loc loc, int x_size) {
  return loc - 1 + (x_size+1);
}

int Location::getGetBigJumpInitialIndex(const Loc loc0, const Loc loc1, const int x_size) {
  const int diff = loc1 - loc0;
  const int stride = x_size + 1;

  if (diff == -1 || diff == -1 - stride) {
    return RIGHT_TOP_INDEX;
  }

  if(diff == -stride || diff == +1 - stride) {
    return RIGHT_BOTTOM_INDEX;
  }

  if(diff == 1 || diff == 1 + stride) {
    return LEFT_BOTTOM_INDEX;
  }

  if(diff == stride || diff == -1 + stride) {
    return LEFT_TOP_INDEX;
  }

  return -1;
}

Loc Location::getNextLocCW(const Loc loc0, const Loc loc1, const int x_size) {
  const int diff = loc1 - loc0;
  const int stride = x_size + 1;

  if (diff == -1) return xm1ym1(loc0, x_size);
  if (diff == -1 - stride) return xym1(loc0, x_size);
  if (diff == -stride) return xp1ym1(loc0, x_size);
  if (diff == +1 - stride) return xp1y(loc0);
  if (diff == +1) return xp1yp1(loc0, x_size);
  if (diff == +1 + stride) return xyp1(loc0, x_size);
  if (diff == +stride) return xm1yp1(loc0, x_size);
  if (diff == -1 + stride) return xm1y(loc0);

  assert(false && "Incorrect locations");
  return 0;
}

Color getActiveColor(const State state) {
  return static_cast<Color>(state & ACTIVE_MASK);
}

bool isTerritory(const State s) {
  return (s & TERRITORY_FLAG) == TERRITORY_FLAG;
}

Color getPlacedDotColor(const State s) {
  return static_cast<Player>(s >> PLACED_PLAYER_SHIFT & ACTIVE_MASK);
}

bool isPlaced(const State s, const Player pla) {
  return (s >> PLACED_PLAYER_SHIFT & ACTIVE_MASK) == pla;
}

bool isActive(const State s, const Player pla) {
  return (s & ACTIVE_MASK) == pla;
}

State setTerritoryAndActivePlayer(const State s, const Player pla) {
  return static_cast<State>(TERRITORY_FLAG | (s & INVALIDATE_TERRITORY_MASK | pla));
}

Color getEmptyTerritoryColor(const State s) {
  return static_cast<Player>(s >> EMPTY_TERRITORY_SHIFT & ACTIVE_MASK);
}

bool isWithinEmptyTerritory(const State s, const Player pla) {
  return (s >> EMPTY_TERRITORY_SHIFT & ACTIVE_MASK) == pla;
}

State Board::getState(const Loc loc) const {
  return colors[loc];
}

void Board::setState(const Loc loc, const State state) {
  colors[loc] = state;
}

bool Board::isDots() const {
  return rules.isDots;
}

bool isGrounded(const State state) {
  return (state & GROUNDED_FLAG) == GROUNDED_FLAG;
}

bool isGroundedOrWall(const State state, const Player pla) {
  // Use bit tricks for grounding detecting.
  // If the active player is C_WALL, then the result is also true.
  return (state & GROUNDED_FLAG) == GROUNDED_FLAG && (state & pla) == pla;
}

void Board::setGrounded(const Loc loc) {
  colors[loc] = static_cast<Color>(colors[loc] | GROUNDED_FLAG);
}

void Board::clearGrounded(const Loc loc) {
  colors[loc] = static_cast<Color>(colors[loc] & ~GROUNDED_FLAG);
}

bool Board::isVisited(const Loc loc) const {
  return visited_data[loc];
}

void Board::setVisited(const Loc loc) const {
  visited_data[loc] = true;
}

void Board::clearVisited(const Loc loc) const {
  visited_data[loc] = false;
}

void Board::clearVisited(const vector<Loc>& locations) const {
  for (const Loc& loc : locations) {
    clearVisited(loc);
  }
}

int Board::calculateOwnershipAndWhiteScore(Color* result, const Color groundingPlayer) const {
  int whiteCaptures = 0;
  int blackCaptures = 0;

  for (int y = 0; y < y_size; y++) {
    for (int x = 0; x < x_size; x++) {
      const Loc loc = Location::getLoc(x, y, x_size);
      const State state = getState(loc);
      const Color activeColor = getActiveColor(state);
      const Color placedDotColor = getPlacedDotColor(state);
      Color ownershipColor = C_EMPTY;
      if (activeColor != C_EMPTY) {
        if (isGrounded(state) || groundingPlayer == C_EMPTY) {
          if (placedDotColor != C_EMPTY && activeColor != placedDotColor) {
            ownershipColor = activeColor;
            if (placedDotColor == P_BLACK) {
              blackCaptures++;
            } else {
              whiteCaptures++;
            }
          }
        } else {
          // If the game is finished by grounding by a player,
          // Remove its ungrounded dots to get a more refined ownership and score.
          if (groundingPlayer == C_WHITE && placedDotColor == P_WHITE) {
            ownershipColor = P_BLACK;
            whiteCaptures++;
          } else if (groundingPlayer == C_BLACK && placedDotColor == P_BLACK) {
            ownershipColor = P_WHITE;
            blackCaptures++;
          }
        }
      }
      result[loc] = ownershipColor;
    }
  }

  if (groundingPlayer == C_WHITE) {
    // White wins by grounding
    assert(blackScoreIfWhiteGrounds == whiteCaptures - blackCaptures);
    return -blackScoreIfWhiteGrounds;
  }

  if (groundingPlayer == C_BLACK) {
    // Black wins by grounding
    assert(whiteScoreIfBlackGrounds == blackCaptures - whiteCaptures);
    return whiteScoreIfBlackGrounds;
  }

  assert(numBlackCaptures == blackCaptures && numWhiteCaptures == whiteCaptures);
  return numBlackCaptures - numWhiteCaptures;
}

Board::MoveRecord::MoveRecord(
  const Loc newLoc,
  const Player newPla,
  const State newPreviousState,
  const vector<Base>& newBases,
  const vector<Loc>& newEmptyBaseInvalidateLocations,
  const vector<Loc>& newGroundingLocations
) {
  ko_loc = NULL_LOC;
  capDirs = 0;

  loc = newLoc;
  pla = newPla;
  previousState = newPreviousState;
  bases = newBases;
  emptyBaseInvalidateLocations = newEmptyBaseInvalidateLocations;
  groundingLocations = newGroundingLocations;
}

bool Board::isSuicideDots(const Loc loc, const Player pla) const {
  const State state = getState(loc);
  if (Player opp = getOpp(pla); getActiveColor(state) == C_EMPTY && getEmptyTerritoryColor(state) == opp) {
    return !wouldBeCaptureDots(loc, pla);
  }

  return false;
}

bool Board::wouldBeCaptureDots(const Loc loc, const Player pla) const {
  // TODO: optimize and get rid of `const_cast`
  auto moveRecord = const_cast<Board*>(this)->tryPlayMoveRecordedDots(loc, pla, false);

  bool result = false;

  if (moveRecord.pla != C_EMPTY) {
    for (const Base& base : moveRecord.bases) {
      if (base.is_real && base.pla == pla) {
        result = true;
        break;
      }
    }

    const_cast<Board*>(this)->undoDots(moveRecord);
  }

  return result;
}

Board::MoveRecord Board::playMoveRecordedDots(const Loc loc, const Player pla) {
  const MoveRecord& result = tryPlayMoveRecordedDots(loc, pla, true);
  assert(result.pla == pla);
  return result;
}

void Board::playMoveAssumeLegalDots(const Loc loc, const Player pla) {
  const State originalState = getState(loc);

  if (loc == RESIGN_LOC) {
  } else if (loc == PASS_LOC) {
    auto initEmptyBaseInvalidateLocations = vector<Loc>();
    auto bases = vector<Base>();
    ground(pla, initEmptyBaseInvalidateLocations, bases);
  } else {
    colors[loc] = static_cast<Color>(pla | pla << PLACED_PLAYER_SHIFT);
    const Hash128 hashValue = ZOBRIST_BOARD_HASH[loc][pla];
    pos_hash ^= hashValue;
    if (rules.multiStoneSuicideLegal) {
      numLegalMovesIfSuiAllowed--;
    }

    bool atLeastOneRealBaseIsGrounded = false;
    int unconnectedLocationsSize = 0;
    const std::array<Loc, 4> unconnectedLocations = getUnconnectedLocations(loc, pla, unconnectedLocationsSize);
    bool capturing = false;
    if (unconnectedLocationsSize >= 2) {
      auto bases = vector<Base>();
      tryCapture(loc, pla, unconnectedLocations, unconnectedLocationsSize, atLeastOneRealBaseIsGrounded, bases);
      capturing = !bases.empty();
    }

    const Color opp = getOpp(pla);
    if (!capturing) {
      if (getEmptyTerritoryColor(originalState) == opp) {
        vector<Base> oppBases;
        captureWhenEmptyTerritoryBecomesRealBase(loc, opp, oppBases, atLeastOneRealBaseIsGrounded);
      }
    } else if (isWithinEmptyTerritory(originalState, opp)) {
      invalidateAdjacentEmptyTerritoryIfNeeded(loc);
    }

    if (pla == P_BLACK) {
      whiteScoreIfBlackGrounds++;
    } else if (pla == P_WHITE) {
      blackScoreIfWhiteGrounds++;
    }

    if (atLeastOneRealBaseIsGrounded) {
      fillGrounding(loc);
    } else if(
        const Player locActivePlayer = getColor(loc); // Can't use pla because of a possible suicidal move
        isGroundedOrWall(getState(Location::xm1y(loc)), locActivePlayer) ||
        isGroundedOrWall(getState(Location::xym1(loc, x_size)), locActivePlayer) ||
        isGroundedOrWall(getState(Location::xp1y(loc)), locActivePlayer) ||
        isGroundedOrWall(getState(Location::xyp1(loc, x_size)), locActivePlayer)
    ) {
      fillGrounding(loc);
    }
  }
}

Board::MoveRecord Board::tryPlayMoveRecordedDots(Loc loc, Player pla, const bool isSuicideLegal) {
  State originalState = getState(loc);

  vector<Base> bases;
  vector<Loc> initEmptyBaseInvalidateLocations;
  vector<Loc> newGroundingLocations;

  if (loc == RESIGN_LOC) {
  } else if (loc == PASS_LOC) {
    ground(pla, initEmptyBaseInvalidateLocations, bases);
  } else {
    colors[loc] = static_cast<Color>(pla | pla << PLACED_PLAYER_SHIFT);
    const Hash128 hashValue = ZOBRIST_BOARD_HASH[loc][pla];
    pos_hash ^= hashValue;
    if (rules.multiStoneSuicideLegal) {
      numLegalMovesIfSuiAllowed--;
    }

    bool atLeastOneRealBaseIsGrounded = false;
    int unconnectedLocationsSize = 0;
    const std::array<Loc, 4> unconnectedLocations = getUnconnectedLocations(loc, pla, unconnectedLocationsSize);
    if (unconnectedLocationsSize >= 2) {
       tryCapture(loc, pla, unconnectedLocations, unconnectedLocationsSize, atLeastOneRealBaseIsGrounded, bases);
    }

    const Color opp = getOpp(pla);
    if (bases.empty()) {
      if (getEmptyTerritoryColor(originalState) == opp) {
        if (isSuicideLegal) {
          captureWhenEmptyTerritoryBecomesRealBase(loc, opp, bases, atLeastOneRealBaseIsGrounded);
        } else {
          colors[loc] = originalState;
          pos_hash ^= hashValue;
          if (rules.multiStoneSuicideLegal) {
            numLegalMovesIfSuiAllowed++;
          }
          return {};
        }
      }
    } else if (isWithinEmptyTerritory(originalState, opp)) {
      invalidateAdjacentEmptyTerritoryIfNeeded(loc);
      initEmptyBaseInvalidateLocations = vector(closureOrInvalidateLocsBuffer);
    }

    if (pla == P_BLACK) {
      whiteScoreIfBlackGrounds++;
    } else if (pla == P_WHITE) {
      blackScoreIfWhiteGrounds++;
    }

    if (atLeastOneRealBaseIsGrounded) {
      newGroundingLocations = fillGrounding(loc);
    } else if(
        const Player locActivePlayer = getColor(loc); // Can't use pla because of a possible suicidal move
        isGroundedOrWall(getState(Location::xm1y(loc)), locActivePlayer) ||
        isGroundedOrWall(getState(Location::xym1(loc, x_size)), locActivePlayer) ||
        isGroundedOrWall(getState(Location::xp1y(loc)), locActivePlayer) ||
        isGroundedOrWall(getState(Location::xyp1(loc, x_size)), locActivePlayer)
    ) {
      newGroundingLocations = fillGrounding(loc);
    }
  }

  return {loc, pla, originalState, bases, initEmptyBaseInvalidateLocations, newGroundingLocations};
}

void Board::undoDots(MoveRecord& moveRecord) {
  if (moveRecord.loc == RESIGN_LOC) {
    return; // Resin doesn't really change the state
  }

  const bool isGroundingMove = moveRecord.loc == PASS_LOC;

  for (const Loc& loc : moveRecord.groundingLocations) {
    const State state = getState(loc);
    const Player mainPla = getActiveColor(state);
    if (getPlacedDotColor(state) != C_EMPTY) {
      if (mainPla == P_BLACK) {
        whiteScoreIfBlackGrounds++;
      } else {
        blackScoreIfWhiteGrounds++;
      }
    }
    clearGrounded(loc);
  }

  if (!isGroundingMove) {
    if (moveRecord.pla == P_BLACK) {
      whiteScoreIfBlackGrounds--;
    } else if (moveRecord.pla == P_WHITE) {
      blackScoreIfWhiteGrounds--;
    }
  }

  const Player emptyTerritoryPlayer = isGroundingMove ? moveRecord.pla : getOpp(moveRecord.pla);
  for (const Loc& loc : moveRecord.emptyBaseInvalidateLocations) {
    assert(0 == getState(loc));
    setState(loc, static_cast<State>(emptyTerritoryPlayer << EMPTY_TERRITORY_SHIFT));
  }

  for (auto it = moveRecord.bases.rbegin(); it != moveRecord.bases.rend(); ++it) {
    for (size_t index = 0; index < it->rollback_locations.size(); index++) {
      const State rollbackState = it->rollback_states[index];
      const Loc rollbackLocation = it->rollback_locations[index];
      setState(rollbackLocation, rollbackState);
      if (it->is_real) {
        updateScoreAndHashForTerritory(rollbackLocation, rollbackState, it->pla, true);
      }
    }
  }

  if (!isGroundingMove) {
    setState(moveRecord.loc, moveRecord.previousState);
    pos_hash ^= ZOBRIST_BOARD_HASH[moveRecord.loc][moveRecord.pla];
    if (rules.multiStoneSuicideLegal) {
      numLegalMovesIfSuiAllowed++;
    }
  }
}

vector<Loc> Board::fillGrounding(const Loc loc) {
  vector<Loc> groundedLocs;

  walkStack.clear();
  walkStack.push_back(loc);
  const Player pla = getColor(loc);
  assert(pla != C_EMPTY && pla != C_WALL);
  setGrounded(loc);
  if (pla == P_BLACK) {
    whiteScoreIfBlackGrounds--;
  } else {
    blackScoreIfWhiteGrounds--;
  }
  groundedLocs.push_back(loc);

  while (!walkStack.empty()) {
    const Loc currentLoc = walkStack.back();
    walkStack.pop_back();

    forEachAdjacent(currentLoc, [&](const Loc adj) {
      if (const State state = getState(adj); !isGrounded(state) && isActive(state, pla)) {
        setGrounded(adj);
        if (const Player placedColor = getPlacedDotColor(state); placedColor != C_EMPTY) {
          if (pla == P_BLACK) {
            whiteScoreIfBlackGrounds--;
          } else {
            blackScoreIfWhiteGrounds--;
          }
        }
        groundedLocs.push_back(adj);
        walkStack.push_back(adj);
      }
    });
  }

  return groundedLocs;
}

void Board::captureWhenEmptyTerritoryBecomesRealBase(
  const Loc initLoc,
  const Player opp,
  vector<Base>& bases,
  bool& isGrounded) {
  Loc loc = initLoc;

  // Searching for an opponent dot that makes a closure that contains the `initialPosition`.
  // The closure always exists, otherwise there is an error in previous calculations.
  while (loc > 0) {
    loc = Location::xm1y(loc);

    // Try to peek an active opposite player dot
    if (getColor(loc) != opp) continue;

    int unconnectedLocationsSize = 0;
    const std::array<Loc, 4> unconnectedLocations = getUnconnectedLocations(loc, opp, unconnectedLocationsSize);
    if (unconnectedLocationsSize >= 1) {
      bases.clear();
      tryCapture(loc, opp, unconnectedLocations, unconnectedLocationsSize, isGrounded, bases);
      // The found base always should be real and include the `iniLoc`
      for (const Base& oppBase : bases) {
        if (oppBase.is_real) {
          return;
        }
      }
    }
  }

  assert(false && "Opp empty territory should be enclosed by an outer closure");
}

void Board::tryCapture(
  const Loc loc,
  const Player pla,
  const std::array<Loc, 4>& unconnectedLocations,
  const int unconnectedLocationsSize,
  bool& atLeastOneRealBaseIsGrounded,
  std::vector<Base>& bases) {
  auto currentClosures = vector<vector<Loc>>();

  for (int index = 0; index < unconnectedLocationsSize; index++) {
    const Loc unconnectedLoc = unconnectedLocations[index];

    // Optimization: it doesn't make sense to check the latest unconnected dot
    // when all previous connections form minimal bases
    // because the latest always forms a base with maximal square that should be dropped
    if (const size_t closuresSize = currentClosures.size();
       closuresSize > 0 && closuresSize == unconnectedLocations.size() - 1) {
      break;
    }

    tryGetCounterClockwiseClosure(loc, unconnectedLoc, pla);

    // Sort the given closures in ascending order
    if (!closureOrInvalidateLocsBuffer.empty()) {
      bool added = false;

      auto newClosure = vector(closureOrInvalidateLocsBuffer);

      for (auto it = currentClosures.begin(); it != currentClosures.end(); ++it) {
        if (closureOrInvalidateLocsBuffer.size() < it->size()) {
          currentClosures.insert(it, newClosure);
          added = true;
          break;
        }
      }

      if (!added) {
        currentClosures.emplace_back(newClosure);
      }
    }
  }

  atLeastOneRealBaseIsGrounded = false;
  for (const vector<Loc>& currentClosure: currentClosures) {
    Base base = buildBase(currentClosure, pla);
    bases.push_back(base);

    if (!atLeastOneRealBaseIsGrounded && base.is_real) {
      for (const Loc& closureLoc : currentClosure) {
        forEachAdjacent(closureLoc, [&](const Loc adj) {
          atLeastOneRealBaseIsGrounded = atLeastOneRealBaseIsGrounded || isGroundedOrWall(getState(adj), base.pla);
        });
        if (atLeastOneRealBaseIsGrounded) {
          break;
        }
      }
    }
  }
}

void Board::ground(const Player pla, vector<Loc>& emptyBaseInvalidatePositions, vector<Base>& bases) {
  const Color opp = getOpp(pla);

  for (int y = 0; y < y_size; y++) {
    for (int x = 0; x < x_size; x++) {
      const Loc loc = Location::getLoc(x, y, x_size);
      if (const State state = getState(loc); !isGrounded(state) && isActive(state, pla)) {
        bool createRealBase = false;
        getTerritoryLocations(pla, loc, true, createRealBase);
        assert(createRealBase);

        for (const Loc& territoryLoc : territoryLocationsBuffer) {
          invalidateAdjacentEmptyTerritoryIfNeeded(territoryLoc);
          for (const Loc& invalidateLoc : closureOrInvalidateLocsBuffer) {
            emptyBaseInvalidatePositions.push_back(invalidateLoc);
          }
        }

        bases.push_back(createBaseAndUpdateStates(opp, true));
      }
    }
  }
}

std::array<Loc, 4> Board::getUnconnectedLocations(const Loc loc, const Player pla, int& size) const {
  const Loc xm1y = Location::xm1y(loc);
  const Loc xym1 = Location::xym1(loc, x_size);
  const Loc xp1y = Location::xp1y(loc);
  const Loc xyp1 = Location::xyp1(loc, x_size);

  std::array<Loc, 4> unconnectedLocationsBuffer;
  size = 0;
  checkAndAddUnconnectedLocation(unconnectedLocationsBuffer, size, getColor(xp1y), pla, Location::xp1yp1(loc, x_size), xyp1);
  checkAndAddUnconnectedLocation(unconnectedLocationsBuffer, size, getColor(xyp1), pla, Location::xm1yp1(loc, x_size), xm1y);
  checkAndAddUnconnectedLocation(unconnectedLocationsBuffer, size, getColor(xm1y), pla, Location::xm1ym1(loc, x_size), xym1);
  checkAndAddUnconnectedLocation(unconnectedLocationsBuffer, size, getColor(xym1), pla, Location::xp1ym1(loc, x_size), xp1y);

  return unconnectedLocationsBuffer;
}

void Board::checkAndAddUnconnectedLocation(std::array<Loc, 4>& unconnectedLocationsBuffer, int& size, const Player checkPla, const Player currentPla, const Loc addLoc1, const Loc addLoc2) const {
  if (checkPla != currentPla) {
    if (getColor(addLoc1) == currentPla) {
      unconnectedLocationsBuffer[size++] = addLoc1;
    } else if (getColor(addLoc2) == currentPla) {
      unconnectedLocationsBuffer[size++] = addLoc2;
    }
  }
}

void Board::tryGetCounterClockwiseClosure(const Loc initialLoc, const Loc startLoc, const Player pla) const {
  closureOrInvalidateLocsBuffer.clear();
  closureOrInvalidateLocsBuffer.push_back(initialLoc);
  setVisited(initialLoc);
  closureOrInvalidateLocsBuffer.push_back(startLoc);
  setVisited(startLoc);

  Loc currentLoc = startLoc;
  Loc nextLoc = initialLoc;
  Loc loc;

  do {
    const int initialIndex = Location::getGetBigJumpInitialIndex(currentLoc, nextLoc, x_size);
    int currentIndex = initialIndex;

    bool breakSearchingLoop = false;

    do {
      loc = currentLoc + adj_offsets[currentIndex++];
      if (currentIndex == 8) currentIndex = 0;

      const State state = getState(loc);
      const Color activeColor = getActiveColor(state);
      if (activeColor == C_WALL) {
        // Optimization: there is no need to walk anymore because the border can't enclosure anything
        breakSearchingLoop = true;
        break;
      }

      if(activeColor == pla) {
        if(loc == initialLoc) {
          breakSearchingLoop = true;
          break;
        }

        if(isVisited(loc)) {
          // Remove trailing dots
          Loc lastLoc;
          do {
            lastLoc = closureOrInvalidateLocsBuffer.back();
            closureOrInvalidateLocsBuffer.pop_back();
            clearVisited(lastLoc);
          } while(lastLoc != loc);
        }

        closureOrInvalidateLocsBuffer.push_back(loc);
        setVisited(loc);
        nextLoc = currentLoc;
        currentLoc = loc;
        break;
      }
    } while (currentIndex != initialIndex);

    if (breakSearchingLoop) {
      break;
    }
  }
  while (true);

  if (loc != initialLoc || closureOrInvalidateLocsBuffer.size() < 4) {
    clearVisited(closureOrInvalidateLocsBuffer);
    closureOrInvalidateLocsBuffer.clear();
    return;
  }

  int square = 0;
  const int stride = x_size + 1;

  const Loc prevLoc = closureOrInvalidateLocsBuffer.back();
  // Store the previously calculated coordinates because division is an expensive operation
  int prevX = prevLoc % stride;
  int prevY = prevLoc / stride;

  for (const Loc& l : closureOrInvalidateLocsBuffer) {
    const int x = l % stride;
    const int y = l / stride;

    square += prevY * x - y * prevX;

    prevX = x;
    prevY = y;
    clearVisited(l);
  }

  if (square <= 0) {
    closureOrInvalidateLocsBuffer.clear();
  }
}

Board::Base Board::buildBase(const vector<Loc>& closure, const Player pla) {
  for (const Loc& closureLoc : closure) {
    setVisited(closureLoc);
  }

  const Loc territoryFirstLoc = Location::getNextLocCW(closure.at(1), closure.at(0), x_size);
  bool createRealBase;
  getTerritoryLocations(pla, territoryFirstLoc, false, createRealBase);
  clearVisited(closure);

  return createBaseAndUpdateStates(pla, createRealBase);
}

void Board::getTerritoryLocations(const Player pla, const Loc firstLoc, const bool grounding, bool& createRealBase) const {
  walkStack.clear();
  territoryLocationsBuffer.clear();

  createRealBase = grounding ? false : rules.dotsCaptureEmptyBases;
  const Player opp = getOpp(pla);

  State state = getState(firstLoc);
  Color activeColor = getActiveColor(state);
  assert(activeColor != C_WALL);

  bool legalLoc = false;
  if (grounding) {
    createRealBase = true;
    // In a rare case it's possible to encounter an empty ungrounded loc that should be de-facto grounded.
    // However, currently it's to set up its grounding due to limitations of the grounding algorithm that doesn't traverse diagonals.
    // That's why we have to check adj locs on grounding to prevent adding incorrect locs and causing out of bounds exception.
    legalLoc = activeColor == pla && !isGroundedOrWall(state, pla);
  } else if (activeColor != pla || !isTerritory(state)) {  // Ignore already captured territory
    createRealBase = createRealBase || isPlaced(state, opp);
    legalLoc = true; // If no grounding, empty locations can be handled as well
  }

  if (legalLoc) {
    territoryLocationsBuffer.push_back(firstLoc);
    setVisited(firstLoc);
    walkStack.push_back(firstLoc);
  }

  while (!walkStack.empty()) {
    const Loc loc = walkStack.back();
    walkStack.pop_back();

    forEachAdjacent(loc, [&](const Loc adj) {
      if (isVisited(adj)) return;

      state = getState(adj);
      activeColor = getActiveColor(state);

      bool isAdjLegal = false;
      if (grounding) {
        createRealBase = true;
        isAdjLegal = activeColor == pla && !isGroundedOrWall(state, pla);
      } else {
        assert(activeColor != C_WALL);
        if (activeColor != pla || !isTerritory(state)) {  // Ignore already captured territory
          createRealBase = createRealBase || isPlaced(state, opp);
          isAdjLegal = true; // If no grounding, empty locations can be handled as well
        }
      }

      if (isAdjLegal) {
        territoryLocationsBuffer.push_back(adj);
        setVisited(adj);
        walkStack.push_back(adj);
      }
    });
  }

  clearVisited(territoryLocationsBuffer);
}

Board::Base Board::createBaseAndUpdateStates(Player basePla, bool isReal) {
  auto rollbackLocations = vector<Loc>();
  rollbackLocations.reserve(territoryLocationsBuffer.size());
  auto rollbackStates = vector<State>();
  rollbackStates.reserve(territoryLocationsBuffer.size());

  for (const Loc& territoryLoc : territoryLocationsBuffer) {
    State state = getState(territoryLoc);

    if (Player activePlayer = getActiveColor(state); activePlayer != basePla) {
      State newState;
      if (isReal) {
        updateScoreAndHashForTerritory(territoryLoc, state, basePla, false);
        newState = setTerritoryAndActivePlayer(state, basePla);
      } else {
        newState = static_cast<State>(basePla << EMPTY_TERRITORY_SHIFT);
      }

      rollbackLocations.push_back(territoryLoc);
      rollbackStates.push_back(state);
      setState(territoryLoc, newState);
    }
  }

  return {basePla, rollbackLocations, rollbackStates, isReal};
}

void Board::updateScoreAndHashForTerritory(const Loc loc, const State state, const Player basePla, const bool rollback) {
  const Color currentColor = getActiveColor(state);
  const Player baseOppPla = getOpp(basePla);

  if (isPlaced(state, baseOppPla)) {
    // The `getTerritoryPositions` never returns positions inside already owned territory,
    // so there is no need to check for the territory flag.
    if (basePla == P_BLACK) {
      if (!rollback) {
        numWhiteCaptures++;
      } else {
        numWhiteCaptures--;
      }
    } else {
      if (!rollback) {
        numBlackCaptures++;
      } else {
        numBlackCaptures--;
      }
    }
  } else if (isPlaced(state, basePla) && isActive(state, baseOppPla)) {
    // No diff for the territory of the current player
    if (basePla == P_BLACK) {
      if (!rollback) {
        numBlackCaptures--;
      } else {
        numBlackCaptures++;
      }
    } else {
      if (!rollback) {
        numWhiteCaptures--;
      } else {
        numWhiteCaptures++;
      }
    }
  }

  if (currentColor == C_EMPTY) {
    if (rules.multiStoneSuicideLegal) {
      if (!rollback) {
        numLegalMovesIfSuiAllowed--;
      } else {
        numLegalMovesIfSuiAllowed++;
      }
    }
    pos_hash ^= ZOBRIST_BOARD_HASH[loc][basePla];
  } else if (currentColor == baseOppPla) {
    // Simulate unmaking the opponent move and making the player's move
    const auto positionsHash = ZOBRIST_BOARD_HASH[loc];
    pos_hash ^= positionsHash[baseOppPla];
    pos_hash ^= positionsHash[basePla];
  }
}

void Board::invalidateAdjacentEmptyTerritoryIfNeeded(const Loc loc) {
  walkStack.clear();
  walkStack.push_back(loc);
  closureOrInvalidateLocsBuffer.clear();

  while(!walkStack.empty()) {
    const Loc lastLoc = walkStack.back();
    walkStack.pop_back();

    FOREACHADJ(
      Loc adj = lastLoc + ADJOFFSET;

      if (!isVisited(adj) && getEmptyTerritoryColor(getState(adj)) != C_EMPTY) {
        closureOrInvalidateLocsBuffer.push_back(adj);
        setState(adj, C_EMPTY);
        setVisited(adj);

        walkStack.push_back(adj);
      }
    )
  }

  clearVisited(closureOrInvalidateLocsBuffer);
}

void Board::makeMoveAndCalculateCapturesAndBases(
  const Player pla,
  const Loc loc,
  vector<Color>& captures,
  vector<Color>& bases) const {
  if(isLegal(loc, pla, rules.multiStoneSuicideLegal, false)) {
    MoveRecord moveRecord = const_cast<Board*>(this)->playMoveRecordedDots(loc, pla);

    for(Base& base: moveRecord.bases) {
      if (base.is_real) {
        const bool suicide = base.pla != pla;
        if (!suicide) {
          captures[loc] = static_cast<Color>(captures[loc] | base.pla);
        }

        for(const Loc& rollbackLoc: base.rollback_locations) {
          // Consider empty bases as well
          bases[rollbackLoc] = static_cast<Color>(bases[rollbackLoc] | base.pla);
        }
      }
    }

    const_cast<Board*>(this)->undo(moveRecord);
  }
}

void Board::calculateOneMoveCaptureAndBasePositionsForDots(vector<Color>& captures, vector<Color>& bases) const {
  const int fieldSize = (x_size + 1) * (y_size + 1);
  captures.resize(fieldSize);
  bases.resize(fieldSize);

  for (int y = 0; y < y_size; y++) {
    for (int x = 0; x < x_size; x++) {
      const Loc loc = Location::getLoc(x, y, x_size);

      const State state = getState(loc);
      const Color emptyTerritoryColor = getEmptyTerritoryColor(state);

      // It doesn't make sense to calculate capturing when dot placed into own empty territory
      if (emptyTerritoryColor != P_BLACK) {
        makeMoveAndCalculateCapturesAndBases(P_BLACK, loc, captures, bases);
      }

      if (emptyTerritoryColor != P_WHITE) {
        makeMoveAndCalculateCapturesAndBases(P_WHITE, loc, captures, bases);
      }
    }
  }
}
