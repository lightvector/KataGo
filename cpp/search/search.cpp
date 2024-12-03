
//-------------------------------------------------------------------------------------
//This file contains the main core logic of the search.
//-------------------------------------------------------------------------------------

#include "../search/search.h"

#include <algorithm>
#include <numeric>

#include "../core/fancymath.h"
#include "../core/timer.h"
#include "../game/graphhash.h"
#include "../search/distributiontable.h"
#include "../search/patternbonustable.h"
#include "../search/searchnode.h"
#include "../search/searchnodetable.h"
#include "../search/subtreevaluebiastable.h"

using namespace std;

//-----------------------------------------------------------------------------------------

static string makeSeed(const Search& search, int threadIdx) {
  stringstream ss;
  ss << search.randSeed;
  ss << "$searchThread$";
  ss << threadIdx;
  ss << "$";
  ss << search.rootBoard.pos_hash;
  ss << "$";
  ss << search.rootHistory.moveHistory.size();
  ss << "$";
  ss << search.numSearchesBegun;
  return ss.str();
}

SearchThread::SearchThread(int tIdx, const Search& search)
  :threadIdx(tIdx),
   pla(search.rootPla),board(search.rootBoard),
   history(search.rootHistory),
   graphHash(search.rootGraphHash),
   graphPath(),
   shouldCountPlayout(false),
   rand(makeSeed(search,tIdx)),
   nnResultBuf(),
   statsBuf(),
   upperBoundVisitsLeft(1e30),
   oldNNOutputsToCleanUp(),
   illegalMoveHashes()
{
  statsBuf.resize(NNPos::MAX_NN_POLICY_SIZE);
  graphPath.reserve(256);

  //Reserving even this many is almost certainly overkill but should guarantee that we never have hit allocation here.
  oldNNOutputsToCleanUp.reserve(8);
}
SearchThread::~SearchThread() {
  for(size_t i = 0; i<oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.resize(0);
}

//-----------------------------------------------------------------------------------------

static const double VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0;

Search::Search(SearchParams params, NNEvaluator* nnEval, Logger* lg, const string& rSeed)
  :Search(params,nnEval,NULL,lg,rSeed)
{}
Search::Search(SearchParams params, NNEvaluator* nnEval, NNEvaluator* humanEval, Logger* lg, const string& rSeed)
  :rootPla(P_BLACK),
   rootBoard(),
   rootHistory(),
   rootGraphHash(),
   rootHintLoc(Board::NULL_LOC),
   avoidMoveUntilByLocBlack(),avoidMoveUntilByLocWhite(),avoidMoveUntilRescaleRoot(false),
   rootSymmetries(),
   rootPruneOnlySymmetries(),
   rootSafeArea(NULL),
   recentScoreCenter(0.0),
   mirroringPla(C_EMPTY),
   mirrorAdvantage(0.0),
   mirrorCenterSymmetryError(1e10),
   alwaysIncludeOwnerMap(false),
   searchParams(params),numSearchesBegun(0),searchNodeAge(0),
   plaThatSearchIsFor(C_EMPTY),plaThatSearchIsForLastSearch(C_EMPTY),
   lastSearchNumPlayouts(0),
   effectiveSearchTimeCarriedOver(0.0),
   randSeed(rSeed),
   rootKoHashTable(NULL),
   valueWeightDistribution(NULL),
   patternBonusTable(NULL),
   externalPatternBonusTable(nullptr),
   nonSearchRand(rSeed + string("$nonSearchRand")),
   logger(lg),
   nnEvaluator(nnEval),
   humanEvaluator(humanEval),
   nnXLen(),
   nnYLen(),
   policySize(),
   rootNode(NULL),
   nodeTable(NULL),
   mutexPool(NULL),
   subtreeValueBiasTable(NULL),
   numThreadsSpawned(0),
   threads(NULL),
   threadTasks(NULL),
   threadTasksRemaining(NULL),
   oldNNOutputsToCleanUpMutex(),
   oldNNOutputsToCleanUp()
{
  assert(logger != NULL);
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  assert(nnXLen > 0 && nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen > 0 && nnYLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(nnXLen,nnYLen);

  if(humanEvaluator != NULL) {
    if(humanEvaluator->getNNXLen() != nnXLen || humanEvaluator->getNNYLen() != nnYLen)
      throw StringError("Search::init - humanEval has different nnXLen or nnYLen");
  }

  rootKoHashTable = new KoHashTable();

  rootSafeArea = new Color[Board::MAX_ARR_SIZE];

  valueWeightDistribution = new DistributionTable(
    [](double z) { return FancyMath::tdistpdf(z,VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    [](double z) { return FancyMath::tdistcdf(z,VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    -50.0,
    50.0,
    2000
  );

  rootNode = NULL;
  nodeTable = new SearchNodeTable(params.nodeTableShardsPowerOfTwo);
  mutexPool = new MutexPool(nodeTable->mutexPool->getNumMutexes());

  rootHistory.clear(rootBoard,rootPla,Rules(),0);
  rootKoHashTable->recompute(rootHistory);
}

Search::~Search() {
  clearSearch();

  delete[] rootSafeArea;
  delete rootKoHashTable;
  delete valueWeightDistribution;

  delete nodeTable;
  delete mutexPool;
  delete subtreeValueBiasTable;
  delete patternBonusTable;
  killThreads();
}

const Board& Search::getRootBoard() const {
  return rootBoard;
}
const BoardHistory& Search::getRootHist() const {
  return rootHistory;
}
Player Search::getRootPla() const {
  return rootPla;
}

Player Search::getPlayoutDoublingAdvantagePla() const {
  return searchParams.playoutDoublingAdvantagePla == C_EMPTY ? plaThatSearchIsFor : searchParams.playoutDoublingAdvantagePla;
}

int Search::getPos(Loc moveLoc) const {
  return NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
}

void Search::setPosition(Player pla, const Board& board, const BoardHistory& history) {
  clearSearch();
  rootPla = pla;
  plaThatSearchIsFor = C_EMPTY;
  rootBoard = board;
  rootHistory = history;
  rootKoHashTable->recompute(rootHistory);
  avoidMoveUntilByLocBlack.clear();
  avoidMoveUntilByLocWhite.clear();
}

void Search::setPlayerAndClearHistory(Player pla) {
  clearSearch();
  rootPla = pla;
  plaThatSearchIsFor = C_EMPTY;
  rootBoard.clearSimpleKoLoc();
  Rules rules = rootHistory.rules;
  //Preserve this value even when we get multiple moves in a row by some player
  bool assumeMultipleStartingBlackMovesAreHandicap = rootHistory.assumeMultipleStartingBlackMovesAreHandicap;
  rootHistory.clear(rootBoard,rootPla,rules,rootHistory.encorePhase);
  rootHistory.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

  rootKoHashTable->recompute(rootHistory);

  //If changing the player alone, don't clear these, leave the user's setting - the user may have tried
  //to adjust the player or will be calling runWholeSearchAndGetMove with a different player and will
  //still want avoid moves to apply.
  //avoidMoveUntilByLocBlack.clear();
  //avoidMoveUntilByLocWhite.clear();
}

void Search::setPlayerIfNew(Player pla) {
  if(pla != rootPla)
    setPlayerAndClearHistory(pla);
}

void Search::setKomiIfNew(float newKomi) {
  if(rootHistory.rules.komi != newKomi) {
    clearSearch();
    rootHistory.setKomi(newKomi);
  }
}

void Search::setAvoidMoveUntilByLoc(const std::vector<int>& bVec, const std::vector<int>& wVec) {
  if(avoidMoveUntilByLocBlack == bVec && avoidMoveUntilByLocWhite == wVec)
    return;
  clearSearch();
  avoidMoveUntilByLocBlack = bVec;
  avoidMoveUntilByLocWhite = wVec;
}

void Search::setAvoidMoveUntilRescaleRoot(bool b) {
  avoidMoveUntilRescaleRoot = b;
}

void Search::setRootHintLoc(Loc loc) {
  //When we positively change the hint loc, we clear the search to make absolutely sure
  //that the hintloc takes effect, and that all nnevals (including the root noise that adds the hintloc) has a chance to happen
  if(loc != Board::NULL_LOC && rootHintLoc != loc)
    clearSearch();
  rootHintLoc = loc;
}

void Search::setAlwaysIncludeOwnerMap(bool b) {
  if(!alwaysIncludeOwnerMap && b)
    clearSearch();
  alwaysIncludeOwnerMap = b;
}

void Search::setRootSymmetryPruningOnly(const std::vector<int>& v) {
  if(rootPruneOnlySymmetries == v)
    return;
  clearSearch();
  rootPruneOnlySymmetries = v;
}


void Search::setParams(SearchParams params) {
  clearSearch();
  searchParams = params;
}

void Search::setParamsNoClearing(SearchParams params) {
  searchParams = params;
}

void Search::setExternalPatternBonusTable(std::unique_ptr<PatternBonusTable>&& table) {
  if(table == externalPatternBonusTable)
    return;
  //Probably not actually needed so long as we do a fresh search to refresh and use the new table
  //but this makes behavior consistent with all the other setters.
  clearSearch();
  externalPatternBonusTable = std::move(table);
}

void Search::setCopyOfExternalPatternBonusTable(const std::unique_ptr<PatternBonusTable>& table) {
  setExternalPatternBonusTable(table == nullptr ? nullptr : std::make_unique<PatternBonusTable>(*table));
}

void Search::setNNEval(NNEvaluator* nnEval) {
  clearSearch();
  nnEvaluator = nnEval;
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  assert(nnXLen > 0 && nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen > 0 && nnYLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(nnXLen,nnYLen);

  if(humanEvaluator != NULL) {
    if(humanEvaluator->getNNXLen() != nnXLen || humanEvaluator->getNNYLen() != nnYLen)
      throw StringError("Search::setNNEval - humanEval has different nnXLen or nnYLen");
  }
}

void Search::clearSearch() {
  effectiveSearchTimeCarriedOver = 0.0;
  if(rootNode != NULL) {
    deleteAllTableNodesMulithreaded();
    //Root is not stored in node table
    if(rootNode != NULL) {
      delete rootNode;
      rootNode = NULL;
    }
  }
  clearOldNNOutputs();
  searchNodeAge = 0;
}

bool Search::isLegalTolerant(Loc moveLoc, Player movePla) const {
  //Tolerate sgf files or GTP reporting suicide moves, even if somehow the rules are set to disallow them.
  bool multiStoneSuicideLegal = true;

  //If we somehow have the same player making multiple moves in a row (possible in GTP or an sgf file),
  //clear the ko loc - the simple ko loc of a player should not prohibit the opponent playing there!
  if(movePla != rootPla) {
    Board copy = rootBoard;
    copy.clearSimpleKoLoc();
    return copy.isLegal(moveLoc,movePla,multiStoneSuicideLegal);
  }
  else {
    return rootHistory.isLegalTolerant(rootBoard,moveLoc,movePla);
  }
}

bool Search::isLegalStrict(Loc moveLoc, Player movePla) const {
  return movePla == rootPla && rootHistory.isLegal(rootBoard,moveLoc,movePla);
}

bool Search::makeMove(Loc moveLoc, Player movePla) {
  return makeMove(moveLoc,movePla,false);
}

bool Search::makeMove(Loc moveLoc, Player movePla, bool preventEncore) {
  if(!isLegalTolerant(moveLoc,movePla))
    return false;

  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);

  //If the white handicap bonus changes due to the move, we will also need to recompute everything since this is
  //basically like a change to the komi.
  float oldWhiteHandicapBonusScore = rootHistory.whiteHandicapBonusScore;

  //Compute these first so we can know if we need to set forceNonTerminal below.
  rootHistory.makeBoardMoveAssumeLegal(rootBoard,moveLoc,rootPla,rootKoHashTable,preventEncore);
  rootPla = getOpp(rootPla);
  rootKoHashTable->recompute(rootHistory);

  if(rootNode != NULL) {
    SearchNode* child = NULL;
    {
      SearchNodeChildrenReference children = rootNode->getChildren();
      int childrenCapacity = children.getCapacity();
      for(int i = 0; i<childrenCapacity; i++) {
        SearchNode* childCandidate = children[i].getIfAllocated();
        if(childCandidate == NULL)
          break;
        if(children[i].getMoveLocRelaxed() == moveLoc) {
          child = childCandidate;
          break;
        }
      }
    }

    //Just in case, make sure the child has an nnOutput, otherwise no point keeping it.
    //This is a safeguard against any oddity involving node preservation into states that
    //were considered terminal.
    if(child != NULL) {
      NNOutput* nnOutput = child->getNNOutput();
      if(nnOutput == NULL)
        child = NULL;
    }

    if(child != NULL) {
      //Account for time carried over
      {
        int64_t rootVisits = rootNode->stats.visits.load(std::memory_order_acquire);
        int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
        double visitProportion = (double)childVisits / (double)rootVisits;
        if(visitProportion > 1)
          visitProportion = 1;
        effectiveSearchTimeCarriedOver = effectiveSearchTimeCarriedOver * visitProportion * searchParams.treeReuseCarryOverTimeFactor;
      }

      SearchNode* oldRootNode = rootNode;

      //Okay, this is now our new root! Create a copy so as to keep the root out of the node table.
      const bool copySubtreeValueBias = false;
      const bool forceNonTerminal = rootHistory.isGameFinished; // Make sure the root isn't considered terminal if game would be finished.
      rootNode = new SearchNode(*child, forceNonTerminal, copySubtreeValueBias);
      //Sweep over the new root marking it as good (calling NULL function), and then delete anything unmarked.
      //This will include the old copy of the child that we promoted to root.
      applyRecursivelyAnyOrderMulithreaded({rootNode}, NULL);
      bool old = true;
      deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(old);
      //Old root is not stored in node table, delete it too.
      delete oldRootNode;
    }
    else {
      clearSearch();
    }
  }

  //Explicitly clear avoid move arrays when we play a move - user needs to respecify them if they want them.
  avoidMoveUntilByLocBlack.clear();
  avoidMoveUntilByLocWhite.clear();

  //If we're newly inferring some moves as handicap that we weren't before, clear since score will be wrong.
  if(rootHistory.whiteHandicapBonusScore != oldWhiteHandicapBonusScore)
    clearSearch();

  //In the case that we are conservativePass and a pass would end the game, need to clear the search.
  //This is because deeper in the tree, such a node would have been explored as ending the game, but now that
  //it's a root pass, it needs to be treated as if it no longer ends the game.
  if(searchParams.conservativePass && rootHistory.passWouldEndGame(rootBoard,rootPla))
    clearSearch();

  //In the case that we're preventing encore, and the phase would have ended, we also need to clear the search
  //since the search was conducted on the assumption that we're going into encore now.
  if(preventEncore && rootHistory.passWouldEndPhase(rootBoard,rootPla))
    clearSearch();

  return true;
}


Loc Search::runWholeSearchAndGetMove(Player movePla) {
  return runWholeSearchAndGetMove(movePla,false);
}

Loc Search::runWholeSearchAndGetMove(Player movePla, bool pondering) {
  runWholeSearch(movePla,pondering);
  return getChosenMoveLoc();
}

void Search::runWholeSearch(Player movePla) {
  runWholeSearch(movePla,false);
}

void Search::runWholeSearch(Player movePla, bool pondering) {
  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);
  std::atomic<bool> shouldStopNow(false);
  runWholeSearch(shouldStopNow,pondering);
}

void Search::runWholeSearch(std::atomic<bool>& shouldStopNow) {
  runWholeSearch(shouldStopNow, false);
}

void Search::runWholeSearch(std::atomic<bool>& shouldStopNow, bool pondering) {
  std::function<void()>* searchBegun = NULL;
  runWholeSearch(shouldStopNow,searchBegun,pondering,TimeControls(),1.0);
}

void Search::runWholeSearch(
  std::atomic<bool>& shouldStopNow,
  std::function<void()>* searchBegun,
  bool pondering,
  const TimeControls& tc,
  double searchFactor
) {

  ClockTimer timer;
  atomic<int64_t> numPlayoutsShared(0);

  if(!std::atomic_is_lock_free(&numPlayoutsShared))
    logger->write("Warning: int64_t atomic numPlayoutsShared is not lock free");
  if(!std::atomic_is_lock_free(&shouldStopNow))
    logger->write("Warning: bool atomic shouldStopNow is not lock free");

  //Do this first, just in case this causes us to clear things and have 0 effective time carried over
  beginSearch(pondering);
  if(searchBegun != NULL)
    (*searchBegun)();
  const int64_t numNonPlayoutVisits = getRootVisits();

  //Compute caps on search
  int64_t maxVisits = pondering ? searchParams.maxVisitsPondering : searchParams.maxVisits;
  int64_t maxPlayouts = pondering ? searchParams.maxPlayoutsPondering : searchParams.maxPlayouts;
  double maxTime = pondering ? searchParams.maxTimePondering : searchParams.maxTime;

  {
    //Possibly reduce computation time, for human friendliness
    if(rootHistory.moveHistory.size() >= 1 && rootHistory.moveHistory[rootHistory.moveHistory.size()-1].loc == Board::PASS_LOC) {
      if(rootHistory.moveHistory.size() >= 3 && rootHistory.moveHistory[rootHistory.moveHistory.size()-3].loc == Board::PASS_LOC)
        searchFactor *= searchParams.searchFactorAfterTwoPass;
      else
        searchFactor *= searchParams.searchFactorAfterOnePass;
    }

    if(searchFactor != 1.0) {
      double cap = (double)((int64_t)1L << 62);
      maxVisits = (int64_t)ceil(std::min(cap, maxVisits * searchFactor));
      maxPlayouts = (int64_t)ceil(std::min(cap, maxPlayouts * searchFactor));
      maxTime = maxTime * searchFactor;
    }
  }

  int capThreads = 0x3fffFFFF;
  if(searchParams.minPlayoutsPerThread > 0.0) {
    int64_t numNewPlayouts = std::min(maxVisits - numNonPlayoutVisits, maxPlayouts);
    double cap = numNewPlayouts / searchParams.minPlayoutsPerThread;
    if(!std::isnan(cap) && cap < (double)0x3fffFFFF) {
      capThreads = std::max(1, (int)floor(cap));
    }
  }

  //Apply time controls. These two don't particularly need to be synchronized with each other so its fine to have two separate atomics.
  std::atomic<double> tcMaxTime(1e30);
  std::atomic<double> upperBoundVisitsLeftDueToTime(1e30);
  const bool hasMaxTime = maxTime < 1.0e12;
  const bool hasTc = !pondering && !tc.isEffectivelyUnlimitedTime();
  if(!pondering && (hasTc || hasMaxTime)) {
    int64_t rootVisits = numPlayoutsShared.load(std::memory_order_relaxed) + numNonPlayoutVisits;
    double timeUsed = timer.getSeconds();
    double tcLimit = 1e30;
    if(hasTc) {
      tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
      tcMaxTime.store(tcLimit, std::memory_order_release);
    }
    double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit,maxTime));
    upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
  }

  std::function<void(int)> searchLoop = [
    this,&timer,&numPlayoutsShared,numNonPlayoutVisits,&tcMaxTime,&upperBoundVisitsLeftDueToTime,&tc,
    &hasMaxTime,&hasTc,
    &shouldStopNow,maxVisits,maxPlayouts,maxTime,pondering,searchFactor
  ](int threadIdx) {
    SearchThread* stbuf = new SearchThread(threadIdx,*this);

    int64_t numPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
    try {
      double lastTimeUsedRecomputingTcLimit = 0.0;
      while(true) {
        double timeUsed = 0.0;
        if(hasTc || hasMaxTime)
          timeUsed = timer.getSeconds();

        double tcMaxTimeLimit = 0.0;
        if(hasTc)
          tcMaxTimeLimit = tcMaxTime.load(std::memory_order_acquire);

        bool shouldStop =
          (numPlayouts >= maxPlayouts) ||
          (numPlayouts + numNonPlayoutVisits >= maxVisits);

        //Time limits cannot stop us from doing at least a little search so we have a non-null tree
        if(hasMaxTime && numPlayouts >= 2 && timeUsed >= maxTime)
          shouldStop = true;
        if(hasTc && numPlayouts >= 2 && timeUsed >= tcMaxTimeLimit)
          shouldStop = true;

        //But an explicit stop signal can stop us from doing any search
        if(shouldStop || shouldStopNow.load(std::memory_order_relaxed)) {
          shouldStopNow.store(true,std::memory_order_relaxed);
          break;
        }

        //Thread 0 alone is responsible for recomputing time limits every once in a while
        //Cap of 10 times per second.
        if(!pondering && (hasTc || hasMaxTime) && threadIdx == 0 && timeUsed >= lastTimeUsedRecomputingTcLimit + 0.1) {
          int64_t rootVisits = numPlayouts + numNonPlayoutVisits;
          double tcLimit = 1e30;
          if(hasTc) {
            tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
            tcMaxTime.store(tcLimit, std::memory_order_release);
          }
          double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit,maxTime));
          upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
        }

        double upperBoundVisitsLeft = 1e30;
        if(hasTc)
          upperBoundVisitsLeft = upperBoundVisitsLeftDueToTime.load(std::memory_order_acquire);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxPlayouts - numPlayouts);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxVisits - numPlayouts - numNonPlayoutVisits);

        bool finishedPlayout = runSinglePlayout(*stbuf, upperBoundVisitsLeft);
        if(finishedPlayout) {
          numPlayouts = numPlayoutsShared.fetch_add((int64_t)1, std::memory_order_relaxed);
          numPlayouts += 1;
        }
        else {
          //In the case that we didn't finish a playout, give other threads a chance to run before we try again
          //so that it's more likely we become unstuck.
          std::this_thread::yield();
        }
      }
    }
    catch(...) {
      transferOldNNOutputs(*stbuf);
      delete stbuf;
      throw;
    }

    transferOldNNOutputs(*stbuf);
    delete stbuf;
  };

  double actualSearchStartTime = timer.getSeconds();
  performTaskWithThreads(&searchLoop, capThreads);

  //If the search did not actually do anything, we need to still make sure to update the root node if it needs
  //such an update (since root params may differ from tree params).
  if(rootNode != NULL && rootNode->nodeAge.load(std::memory_order_acquire) != searchNodeAge) {
    //Also check if the root node even got an nn eval or not. It might not be, if we quit the search instantly upon
    //start due to an explicit stop signal
    if(rootNode->getNNOutput() != nullptr) {
      const int threadIdx = 0;
      const bool isRoot = true;
      SearchThread thread(threadIdx,*this);
      maybeRecomputeExistingNNOutput(thread,*rootNode,isRoot);
    }
  }

  //Relaxed load is fine since numPlayoutsShared should be synchronized already due to the joins
  lastSearchNumPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
  effectiveSearchTimeCarriedOver += timer.getSeconds() - actualSearchStartTime;
}

//If we're being asked to search from a position where the game is over, this is fine. Just keep going, the boardhistory
//should reasonably tolerate just continuing. We do NOT want to clear history because we could inadvertently make a move
//that an external ruleset COULD think violated superko.
void Search::beginSearch(bool pondering) {
  if(rootBoard.x_size > nnXLen || rootBoard.y_size > nnYLen)
    throw StringError("Search got from NNEval nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) + " but was asked to search board with larger x or y size");

  rootBoard.checkConsistency();

  numSearchesBegun++;

  //Avoid any issues in principle from rolling over
  if(searchNodeAge > 0x3FFFFFFF)
    clearSearch();

  if(!pondering)
    plaThatSearchIsFor = rootPla;
  //If we begin the game with a ponder, then assume that "we" are the opposing side until we see otherwise.
  if(plaThatSearchIsFor == C_EMPTY)
    plaThatSearchIsFor = getOpp(rootPla);

  if(plaThatSearchIsForLastSearch != plaThatSearchIsFor) {
    //In the case we are doing playoutDoublingAdvantage without a specific player (so, doing the root player)
    //and the player that the search is for changes, we need to clear the tree since we need new evals for the new way around
    if(searchParams.playoutDoublingAdvantage != 0 && searchParams.playoutDoublingAdvantagePla == C_EMPTY)
      clearSearch();
    //If we are doing pattern bonus and the player the search is for changes, clear the search. Recomputing the search tree
    //recursively *would* fix all our utilities, but the problem is the playout distribution will still be matching the
    //old probabilities without a lot of new search, so clearing ensures a better distribution.
    if(searchParams.avoidRepeatedPatternUtility != 0 || externalPatternBonusTable != nullptr)
      clearSearch();
    //If we have a human SL net and the parameters are different for the different sides, clear the search.
    if(humanEvaluator != NULL) {
      if((searchParams.humanSLPlaExploreProbWeightless != searchParams.humanSLOppExploreProbWeightless) ||
         (searchParams.humanSLPlaExploreProbWeightful != searchParams.humanSLOppExploreProbWeightful) ||
         (searchParams.humanSLPlaExploreProbWeightless != searchParams.humanSLRootExploreProbWeightless) ||
         (searchParams.humanSLPlaExploreProbWeightful != searchParams.humanSLRootExploreProbWeightful))
        clearSearch();
    }
  }
  plaThatSearchIsForLastSearch = plaThatSearchIsFor;
  //cout << "BEGINSEARCH " << PlayerIO::playerToString(rootPla) << " " << PlayerIO::playerToString(plaThatSearchIsFor) << endl;

  clearOldNNOutputs();
  computeRootValues();

  //Prepare value bias table if we need it
  if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable == NULL && !(searchParams.antiMirror && mirroringPla != C_EMPTY))
    subtreeValueBiasTable = new SubtreeValueBiasTable(searchParams.subtreeValueBiasTableNumShards);

  //Refresh pattern bonuses if needed
  if(patternBonusTable != NULL) {
    delete patternBonusTable;
    patternBonusTable = NULL;
  }
  if(searchParams.avoidRepeatedPatternUtility != 0 || externalPatternBonusTable != nullptr) {
    if(externalPatternBonusTable != nullptr)
      patternBonusTable = new PatternBonusTable(*externalPatternBonusTable);
    else
      patternBonusTable = new PatternBonusTable();
    if(searchParams.avoidRepeatedPatternUtility != 0) {
      double bonus = plaThatSearchIsFor == P_WHITE ? -searchParams.avoidRepeatedPatternUtility : searchParams.avoidRepeatedPatternUtility;
      patternBonusTable->addBonusForGameMoves(rootHistory,bonus,plaThatSearchIsFor);
    }
    //Clear any pattern bonus on the root node itself
    if(rootNode != NULL)
      rootNode->patternBonusHash = Hash128();
  }

  if(searchParams.rootSymmetryPruning) {
    const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
    if(rootPruneOnlySymmetries.size() > 0)
      SymmetryHelpers::markDuplicateMoveLocs(rootBoard,rootHistory,&rootPruneOnlySymmetries,avoidMoveUntilByLoc,rootSymDupLoc,rootSymmetries);
    else
      SymmetryHelpers::markDuplicateMoveLocs(rootBoard,rootHistory,NULL,avoidMoveUntilByLoc,rootSymDupLoc,rootSymmetries);
  }
  else {
    //Just in case, don't leave the values undefined.
    std::fill(rootSymDupLoc,rootSymDupLoc+Board::MAX_ARR_SIZE, false);
    rootSymmetries.clear();
    rootSymmetries.push_back(0);
  }

  SearchThread dummyThread(-1, *this);

  if(rootNode == NULL) {
    //Avoid storing the root node in the nodeTable, guarantee that it never is part of a cycle, allocate it directly.
    //Also force that it is non-terminal.
    const bool forceNonTerminal = rootHistory.isGameFinished; // Make sure the root isn't considered terminal if game would be finished.
    rootNode = new SearchNode(rootPla, forceNonTerminal, createMutexIdxForNode(dummyThread));
  }
  else {
    //If the root node has any existing children, then prune things down if there are moves that should not be allowed at the root.
    SearchNode& node = *rootNode;
    SearchNodeChildrenReference children = node.getChildren();
    int childrenCapacity = children.getCapacity();
    bool anyFiltered = false;
    if(childrenCapacity > 0) {

      //This filtering, by deleting children, doesn't conform to the normal invariants that hold during search.
      //However nothing else should be running at this time and the search hasn't actually started yet, so this is okay.
      //Also we can't be affecting the tree since the root node isn't in the table and can't be transposed to.
      int numGoodChildren = 0;
      vector<SearchNode*> filteredNodes;
      {
        int i = 0;
        for(; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          int64_t edgeVisits = children[i].getEdgeVisits();
          Loc moveLoc = children[i].getMoveLoc();
          if(child == NULL)
            break;
          //Remove the child from its current spot
          children[i].store(NULL);
          children[i].setEdgeVisits(0);
          children[i].setMoveLoc(Board::NULL_LOC);
          //Maybe add it back. Specifically check for legality just in case weird graph interaction in the
          //tree gives wrong legality - ensure that once we are the root, we are strict on legality.
          if(rootHistory.isLegal(rootBoard,moveLoc,rootPla) && isAllowedRootMove(moveLoc)) {
            children[numGoodChildren].store(child);
            children[numGoodChildren].setEdgeVisits(edgeVisits);
            children[numGoodChildren].setMoveLoc(moveLoc);
            numGoodChildren++;
          }
          else {
            anyFiltered = true;
            filteredNodes.push_back(child);
          }
        }
        for(; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          (void)child;
          assert(child == NULL);
        }
      }

      if(anyFiltered) {
        //Fix up the node state and child arrays.
        node.collapseChildrenCapacity(numGoodChildren);
        children = node.getChildren();
        childrenCapacity = children.getCapacity();

        //Fix up the number of visits of the root node after doing this filtering
        int64_t newNumVisits = 0;
        for(int i = 0; i<childrenCapacity; i++) {
          const SearchNode* child = children[i].getIfAllocated();
          if(child == NULL)
            break;
          int64_t edgeVisits = children[i].getEdgeVisits();
          newNumVisits += edgeVisits;
        }

        //For the node's own visit itself
        newNumVisits += 1;

        //Set the visits in place
        while(node.statsLock.test_and_set(std::memory_order_acquire));
        node.stats.visits.store(newNumVisits,std::memory_order_release);
        node.statsLock.clear(std::memory_order_release);

        //Update all other stats
        recomputeNodeStats(node, dummyThread, 0, true);
      }
    }

    //Recursively update all stats in the tree if we have dynamic score values
    //And also to clear out lastResponseBiasDeltaSum and lastResponseBiasWeight
    if(searchParams.dynamicScoreUtilityFactor != 0 || searchParams.subtreeValueBiasFactor != 0 || patternBonusTable != NULL) {
      recursivelyRecomputeStats(node);
      if(anyFiltered) {
        //Recursive stats recomputation resulted in us marking all nodes we have. Anything filtered is old now, delete it.
        bool old = true;
        deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(old);
      }
    }
    else {
      if(anyFiltered) {
        //Sweep over the entire child marking it as good (calling NULL function), and then delete anything unmarked.
        applyRecursivelyAnyOrderMulithreaded({rootNode}, NULL);
        bool old = true;
        deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(old);
      }
    }
  }

  //Clear unused stuff in value bias table since we may have pruned rootNode stuff
  if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable != NULL)
    subtreeValueBiasTable->clearUnusedSynchronous();

  //Mark all nodes old for the purposes of updating old nnoutputs
  searchNodeAge++;
}

uint32_t Search::createMutexIdxForNode(SearchThread& thread) const {
  return thread.rand.nextUInt() & (mutexPool->getNumMutexes()-1);
}

//Based on sha256 of "search.cpp FORCE_NON_TERMINAL_HASH"
static const Hash128 FORCE_NON_TERMINAL_HASH = Hash128(0xd4c31800cb8809e2ULL,0xf75f9d2083f2ffcaULL);

//Must be called AFTER making the bestChildMoveLoc in the thread board and hist.
SearchNode* Search::allocateOrFindNode(SearchThread& thread, Player nextPla, Loc bestChildMoveLoc, bool forceNonTerminal, Hash128 graphHash) {
  //Hash to use as a unique id for this node in the table, for transposition detection.
  //If this collides, we will be sad, but it should be astronomically rare since our hash is 128 bits.
  Hash128 childHash;
  if(searchParams.useGraphSearch) {
    childHash = graphHash;
    if(forceNonTerminal)
      childHash ^= FORCE_NON_TERMINAL_HASH;
  }
  else {
    childHash = thread.board.pos_hash ^ Hash128(thread.rand.nextUInt64(),thread.rand.nextUInt64());
  }

  uint32_t nodeTableIdx = nodeTable->getIndex(childHash.hash0);
  std::mutex& mutex = nodeTable->mutexPool->getMutex(nodeTableIdx);
  std::lock_guard<std::mutex> lock(mutex);

  SearchNode* child = NULL;
  std::map<Hash128,SearchNode*>& nodeMap = nodeTable->entries[nodeTableIdx];

  while(true) {
    auto insertLoc = nodeMap.lower_bound(childHash);

    if(insertLoc != nodeMap.end() && insertLoc->first == childHash) {
      //Attempt to transpose to invalid node - rerandomize hash and just store this node somewhere arbitrary.
      if(insertLoc->second->nextPla != nextPla) {
        childHash = thread.board.pos_hash ^ Hash128(thread.rand.nextUInt64(),thread.rand.nextUInt64());
        continue;
      }
      child = insertLoc->second;
    }
    else {
      child = new SearchNode(nextPla, forceNonTerminal, createMutexIdxForNode(thread));

      //Also perform subtree value bias and pattern bonus handling under the mutex. These parameters are no atomic, so
      //if the node is accessed concurrently by other nodes through the table, we need to make sure these parameters are fully
      //fully-formed before we make the node accessible to anyone.

      if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable != NULL) {
        //TODO can we make subtree value bias not depend on prev move loc?
        if(thread.history.moveHistory.size() >= 2) {
          Loc prevMoveLoc = thread.history.moveHistory[thread.history.moveHistory.size()-2].loc;
          if(prevMoveLoc != Board::NULL_LOC) {
            child->subtreeValueBiasTableEntry = subtreeValueBiasTable->get(getOpp(thread.pla), prevMoveLoc, bestChildMoveLoc, thread.history.getRecentBoard(1));
          }
        }
      }

      if(patternBonusTable != NULL)
        child->patternBonusHash = patternBonusTable->getHash(getOpp(thread.pla), bestChildMoveLoc, thread.history.getRecentBoard(1));

      //Insert into map! Use insertLoc as hint.
      nodeMap.insert(insertLoc, std::make_pair(childHash,child));
    }
    break;
  }
  return child;
}

void Search::clearOldNNOutputs() {
  for(size_t i = 0; i<oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.resize(0);
}
void Search::transferOldNNOutputs(SearchThread& thread) {
  std::lock_guard<std::mutex> lock(oldNNOutputsToCleanUpMutex);
  for(size_t i = 0; i<thread.oldNNOutputsToCleanUp.size(); i++)
    oldNNOutputsToCleanUp.push_back(thread.oldNNOutputsToCleanUp[i]);
  thread.oldNNOutputsToCleanUp.resize(0);
}

void Search::removeSubtreeValueBias(SearchNode* node) {
  if(node->subtreeValueBiasTableEntry != nullptr) {
    double deltaUtilitySumToSubtract = node->lastSubtreeValueBiasDeltaSum * searchParams.subtreeValueBiasFreeProp;
    double weightSumToSubtract = node->lastSubtreeValueBiasWeight * searchParams.subtreeValueBiasFreeProp;

    SubtreeValueBiasEntry& entry = *(node->subtreeValueBiasTableEntry);
    while(entry.entryLock.test_and_set(std::memory_order_acquire));
    entry.deltaUtilitySum -= deltaUtilitySumToSubtract;
    entry.weightSum -= weightSumToSubtract;
    entry.entryLock.clear(std::memory_order_release);
    node->subtreeValueBiasTableEntry = nullptr;
  }
}

//Delete ALL nodes where nodeAge < searchNodeAge if old is true, else all nodes where nodeAge >= searchNodeAge
//Also clears subtreevaluebias for deleted nodes.
void Search::deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(bool old) {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  assert(numAdditionalThreads >= 0);
  std::function<void(int)> g = [&](int threadIdx) {
    size_t idx0 = (size_t)((uint64_t)(threadIdx) * nodeTable->entries.size() / (numAdditionalThreads+1));
    size_t idx1 = (size_t)((uint64_t)(threadIdx+1) * nodeTable->entries.size() / (numAdditionalThreads+1));
    for(size_t i = idx0; i<idx1; i++) {
      std::map<Hash128,SearchNode*>& nodeMap = nodeTable->entries[i];
      for(auto it = nodeMap.cbegin(); it != nodeMap.cend();) {
        SearchNode* node = it->second;
        if(old == (node->nodeAge.load(std::memory_order_acquire) < searchNodeAge)) {
          removeSubtreeValueBias(node);
          delete node;
          it = nodeMap.erase(it);
        }
        else
          ++it;
      }
    }
  };
  performTaskWithThreads(&g, 0x3FFFffff);
}

//Delete ALL nodes. More efficient than deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded if deleting everything.
//Doesn't clear subtree value bias.
void Search::deleteAllTableNodesMulithreaded() {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  assert(numAdditionalThreads >= 0);
  std::function<void(int)> g = [&](int threadIdx) {
    size_t idx0 = (size_t)((uint64_t)(threadIdx) * nodeTable->entries.size() / (numAdditionalThreads+1));
    size_t idx1 = (size_t)((uint64_t)(threadIdx+1) * nodeTable->entries.size() / (numAdditionalThreads+1));
    for(size_t i = idx0; i<idx1; i++) {
      std::map<Hash128,SearchNode*>& nodeMap = nodeTable->entries[i];
      for(auto it = nodeMap.cbegin(); it != nodeMap.cend(); ++it) {
        delete it->second;
      }
      nodeMap.clear();
    }
  };
  performTaskWithThreads(&g, 0x3FFFffff);
}

//This function should NOT ever be called concurrently with any other threads modifying the search tree.
//However, it does thread-safely modify things itself, so can safely in theory run concurrently with things
//like ownership computation or analysis that simply read the tree.
void Search::recursivelyRecomputeStats(SearchNode& n) {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  std::vector<SearchThread*> dummyThreads(numAdditionalThreads+1, NULL);
  for(int threadIdx = 0; threadIdx<numAdditionalThreads+1; threadIdx++)
    dummyThreads[threadIdx] = new SearchThread(threadIdx, *this);

  std::function<void(SearchNode*,int)> f = [&](SearchNode* node, int threadIdx) {
    assert(threadIdx >= 0 && threadIdx < dummyThreads.size());
    SearchThread& thread = *(dummyThreads[threadIdx]);

    bool foundAnyChildren = false;
    SearchNodeChildrenReference children = node->getChildren();
    int childrenCapacity = children.getCapacity();
    int i = 0;
    for(; i<childrenCapacity; i++) {
      SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      foundAnyChildren = true;
    }
    for(; i<childrenCapacity; i++) {
      SearchNode* child = children[i].getIfAllocated();
      (void)child;
      assert(child == NULL);
    }

    //If this node has children, it MUST also have an nnOutput.
    if(foundAnyChildren) {
      NNOutput* nnOutput = node->getNNOutput();
      (void)nnOutput; //avoid warning when we have no asserts
      assert(nnOutput != NULL);
    }

    //Also, something is wrong if we have virtual losses at this point
    int32_t numVirtualLosses = node->virtualLosses.load(std::memory_order_acquire);
    (void)numVirtualLosses;
    assert(numVirtualLosses == 0);

    bool isRoot = (node == rootNode);

    //If the node has no children, then just update its utility directly
    //Again, this would be a little wrong if this function were running concurrently with anything else in the
    //case that new children were added in the meantime. Although maybe it would be okay.
    if(!foundAnyChildren) {
      int64_t numVisits = node->stats.visits.load(std::memory_order_acquire);
      double weightSum = node->stats.weightSum.load(std::memory_order_acquire);
      double winLossValueAvg = node->stats.winLossValueAvg.load(std::memory_order_acquire);
      double noResultValueAvg = node->stats.noResultValueAvg.load(std::memory_order_acquire);
      double scoreMeanAvg = node->stats.scoreMeanAvg.load(std::memory_order_acquire);
      double scoreMeanSqAvg = node->stats.scoreMeanSqAvg.load(std::memory_order_acquire);

      //It's possible that this node has 0 weight in the case where it's the root node
      //and has 0 visits because we began a search and then stopped it before any playouts happened.
      //In that case, there's not much to recompute.
      if(weightSum <= 0.0) {
        assert(numVisits == 0);
        assert(isRoot);
      }
      else {
        double resultUtility = getResultUtility(winLossValueAvg, noResultValueAvg);
        double scoreUtility = getScoreUtility(scoreMeanAvg, scoreMeanSqAvg);
        double newUtilityAvg = resultUtility + scoreUtility;
        newUtilityAvg += getPatternBonus(node->patternBonusHash,getOpp(node->nextPla));
        double newUtilitySqAvg = newUtilityAvg * newUtilityAvg;

        while(node->statsLock.test_and_set(std::memory_order_acquire));
        node->stats.utilityAvg.store(newUtilityAvg,std::memory_order_release);
        node->stats.utilitySqAvg.store(newUtilitySqAvg,std::memory_order_release);
        node->statsLock.clear(std::memory_order_release);
      }
    }
    else {
      //Otherwise recompute it using the usual method
      recomputeNodeStats(*node, thread, 0, isRoot);
    }
  };

  vector<SearchNode*> nodes;
  nodes.push_back(&n);
  applyRecursivelyPostOrderMulithreaded(nodes,&f);

  for(int threadIdx = 0; threadIdx<numAdditionalThreads+1; threadIdx++)
    delete dummyThreads[threadIdx];
}


void Search::computeRootValues() {
  //rootSafeArea is strictly pass-alive groups and strictly safe territory.
  bool nonPassAliveStones = false;
  bool safeBigTerritories = false;
  bool unsafeBigTerritories = false;
  bool isMultiStoneSuicideLegal = rootHistory.rules.multiStoneSuicideLegal;
  rootBoard.calculateArea(
    rootSafeArea,
    nonPassAliveStones,
    safeBigTerritories,
    unsafeBigTerritories,
    isMultiStoneSuicideLegal
  );

  //Figure out how to set recentScoreCenter
  {
    bool foundExpectedScoreFromTree = false;
    double expectedScore = 0.0;
    if(rootNode != NULL) {
      const SearchNode& node = *rootNode;
      int64_t numVisits = node.stats.visits.load(std::memory_order_acquire);
      double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
      double scoreMeanAvg = node.stats.scoreMeanAvg.load(std::memory_order_acquire);
      if(numVisits > 0 && weightSum > 0) {
        foundExpectedScoreFromTree = true;
        expectedScore = scoreMeanAvg;
      }
    }

    //Grab a neural net evaluation for the current position and use that as the center
    if(!foundExpectedScoreFromTree) {
      NNResultBuf nnResultBuf;
      NNResultBuf humanResultBuf;
      bool includeOwnerMap = true;
      bool includeHumanResult = humanEvaluator != NULL && searchParams.humanSLValueProportion > 0;
      computeRootNNEvaluation(nnResultBuf,humanResultBuf,includeOwnerMap,includeHumanResult);
      expectedScore = nnResultBuf.result->whiteScoreMean;
      if(includeHumanResult) {
        expectedScore += searchParams.humanSLValueProportion * ((double)(humanResultBuf.result->whiteScoreMean) - expectedScore);
      }
    }

    recentScoreCenter = expectedScore * (1.0 - searchParams.dynamicScoreCenterZeroWeight);
    double cap =  sqrt(rootBoard.x_size * rootBoard.y_size) * searchParams.dynamicScoreCenterScale;
    if(recentScoreCenter > expectedScore + cap)
      recentScoreCenter = expectedScore + cap;
    if(recentScoreCenter < expectedScore - cap)
      recentScoreCenter = expectedScore - cap;
  }

  //If we're using graph search, we recompute the graph hash from scratch at the start of search.
  if(searchParams.useGraphSearch)
    rootGraphHash = GraphHash::getGraphHashFromScratch(rootHistory, rootPla, searchParams.graphSearchRepBound, searchParams.drawEquivalentWinsForWhite);
  else
    rootGraphHash = Hash128();

  Player opponentWasMirroringPla = mirroringPla;
  //Update mirroringPla, mirrorAdvantage, mirrorCenterSymmetryError
  updateMirroring();

  //Clear search if opponent mirror status changed, so that our tree adjusts appropriately
  if(opponentWasMirroringPla != mirroringPla) {
    clearSearch();
    delete subtreeValueBiasTable;
    subtreeValueBiasTable = NULL;
  }
}


bool Search::runSinglePlayout(SearchThread& thread, double upperBoundVisitsLeft) {
  //Store this value, used for futile-visit pruning this thread's root children selections.
  thread.upperBoundVisitsLeft = upperBoundVisitsLeft;

  //Prep this value, playoutDescend will set it to true if we do have a playout
  thread.shouldCountPlayout = false;

  bool finishedPlayout = playoutDescend(thread,*rootNode,true);
  (void)finishedPlayout;

  //Restore thread state back to the root state
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;
  thread.graphHash = rootGraphHash;
  thread.graphPath.clear();

  return thread.shouldCountPlayout;
}

bool Search::playoutDescend(
  SearchThread& thread, SearchNode& node,
  bool isRoot
) {
  //Hit terminal node, finish
  //forceNonTerminal marks special nodes where we cannot end the game, and is set IF they would normally be finished.
  //This includes the root if the root would be game-ended, since if we are searching a position
  //we presumably want to actually explore deeper and get a result. Also it includes the node following a pass from the root in
  //the case where we are conservativePass and the game would be ended. For friendlyPassOk rules, it may include deeper nodes.
  //Note that we also carefully clear the search when a pass from the root would be terminal, so nodes should never need to switch
  //status after tree reuse in the latter case.
  if(thread.history.isGameFinished && !node.forceNonTerminal) {
    //Avoid running "too fast", by making sure that a leaf evaluation takes roughly the same time as a genuine nn eval
    //This stops a thread from building a silly number of visits to distort MCTS statistics while other threads are stuck on the GPU.
    nnEvaluator->waitForNextNNEvalIfAny();
    if(thread.history.isNoResult) {
      double winLossValue = 0.0;
      double noResultValue = 1.0;
      double scoreMean = 0.0;
      double scoreMeanSq = 0.0;
      double lead = 0.0;
      double weight = (searchParams.useUncertainty && nnEvaluator->supportsShorttermError()) ? searchParams.uncertaintyMaxWeight : 1.0;
      addLeafValue(node, winLossValue, noResultValue, scoreMean, scoreMeanSq, lead, weight, true, false);
      thread.shouldCountPlayout = true;
      return true;
    }
    else {
      double winLossValue = 2.0 * ScoreValue::whiteWinsOfWinner(thread.history.winner, searchParams.drawEquivalentWinsForWhite) - 1;
      double noResultValue = 0.0;
      double scoreMean = ScoreValue::whiteScoreDrawAdjust(thread.history.finalWhiteMinusBlackScore,searchParams.drawEquivalentWinsForWhite,thread.history);
      double scoreMeanSq = ScoreValue::whiteScoreMeanSqOfScoreGridded(thread.history.finalWhiteMinusBlackScore,searchParams.drawEquivalentWinsForWhite);
      double lead = scoreMean;
      double weight = (searchParams.useUncertainty && nnEvaluator->supportsShorttermError()) ? searchParams.uncertaintyMaxWeight : 1.0;
      addLeafValue(node, winLossValue, noResultValue, scoreMean, scoreMeanSq, lead, weight, true, false);
      thread.shouldCountPlayout = true;
      return true;
    }
  }

  SearchNodeState nodeState = node.state.load(std::memory_order_acquire);
  if(nodeState == SearchNode::STATE_UNEVALUATED) {
    //Always attempt to set a new nnOutput. That way, if some GPU is slow and malfunctioning, we don't get blocked by it.
    {
      bool suc = initNodeNNOutput(thread,node,isRoot,false,false);
      //Leave the node as unevaluated - only the thread that first actually set the nnOutput into the node
      //gets to update the state, to avoid races where we update the state while the node stats aren't updated yet.
      if(!suc)
        return false;
    }

    bool suc = node.state.compare_exchange_strong(nodeState, SearchNode::STATE_EVALUATING, std::memory_order_seq_cst);
    if(!suc) {
      //Presumably someone else got there first.
      //Just give up on this playout and try again from the start.
      return false;
    }
    else {
      //Perform the nn evaluation and finish!
      node.initializeChildren();
      node.state.store(SearchNode::STATE_EXPANDED0, std::memory_order_seq_cst);
      thread.shouldCountPlayout = true;
      return true;
    }
  }
  else if(nodeState == SearchNode::STATE_EVALUATING) {
    //Just give up on this playout and try again from the start.
    return false;
  }

  assert(nodeState >= SearchNode::STATE_EXPANDED0);
  maybeRecomputeExistingNNOutput(thread,node,isRoot);

  //Find the best child to descend down
  int numChildrenFound;
  int bestChildIdx;
  Loc bestChildMoveLoc;
  bool countEdgeVisit;

  SearchNode* child = NULL;
  while(true) {
    selectBestChildToDescend(thread,node,nodeState,numChildrenFound,bestChildIdx,bestChildMoveLoc,countEdgeVisit,isRoot);

    //The absurdly rare case that the move chosen is not legal
    //(this should only happen either on a bug or where the nnHash doesn't have full legality information or when there's an actual hash collision).
    //Regenerate the neural net call and continue
    //Could also be true if we have an illegal move due to graph search and we had a cycle and superko interaction, or a true collision
    //on an older path that results in bad transposition between positions that don't transpose.
    if(bestChildIdx >= 0 && !thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla)) {
      bool isReInit = true;
      initNodeNNOutput(thread,node,isRoot,true,isReInit);

      {
        NNOutput* nnOutput = node.getNNOutput();
        assert(nnOutput != NULL);
        Hash128 nnHash = nnOutput->nnHash;
        //In case of a cycle or bad transposition, this will fire a lot, so limit it to once per thread per search.
        if(thread.illegalMoveHashes.find(nnHash) == thread.illegalMoveHashes.end()) {
          thread.illegalMoveHashes.insert(nnHash);
          logger->write("WARNING: Chosen move not legal so regenerated nn output, nnhash=" + nnHash.toString());
          ostringstream out;
          thread.history.printBasicInfo(out,thread.board);
          thread.history.printDebugInfo(out,thread.board);
          out << Location::toString(bestChildMoveLoc,thread.board) << endl;
          logger->write(out.str());
        }
      }

      //Give up on this playout now that we've forced the nn output to be consistent legality of this path.
      //Return TRUE though, so that the parent path we traversed increments its edge visits.
      //We want the search to continue as best it can, so we increment visits so search will still make progress
      //even if this keeps happening in some really bad transposition or something.
      thread.shouldCountPlayout = true;
      return true;
    }

    if(bestChildIdx <= -1) {
      //This might happen if all moves have been forbidden. The node will just get stuck counting visits without expanding
      //and we won't do any search.
      addCurrentNNOutputAsLeafValue(node,false);
      thread.shouldCountPlayout = true;
      return true;
    }

    //Do we think we are searching a new child for the first time?
    if(bestChildIdx >= numChildrenFound) {
      assert(bestChildIdx == numChildrenFound);
      assert(bestChildIdx < NNPos::MAX_NN_POLICY_SIZE);
      bool suc = node.maybeExpandChildrenCapacityForNewChild(nodeState, numChildrenFound+1);
      //Someone else is expanding. Loop again trying to select the best child to explore.
      if(!suc) {
        std::this_thread::yield();
        nodeState = node.state.load(std::memory_order_acquire);
        continue;
      }

      SearchNodeChildrenReference children = node.getChildren(nodeState);
      int childrenCapacity = children.getCapacity();
      assert(childrenCapacity > bestChildIdx);
      (void)childrenCapacity;

      //We can only test this before we make the move, so do it now.
      const bool canForceNonTerminalDueToFriendlyPass =
        bestChildMoveLoc == Board::PASS_LOC &&
        thread.history.shouldSuppressEndGameFromFriendlyPass(thread.board, thread.pla);

      //Make the move! We need to make the move before we create the node so we can see the new state and get the right graphHash.
      thread.history.makeBoardMoveAssumeLegal(thread.board,bestChildMoveLoc,thread.pla,rootKoHashTable);
      thread.pla = getOpp(thread.pla);
      if(searchParams.useGraphSearch)
        thread.graphHash = GraphHash::getGraphHash(
          thread.graphHash, thread.history, thread.pla, searchParams.graphSearchRepBound, searchParams.drawEquivalentWinsForWhite
        );

      //If conservative pass, passing from the root is always non-terminal
      //If friendly passing rules, we might also be non-terminal
      const bool forceNonTerminal = bestChildMoveLoc == Board::PASS_LOC && thread.history.isGameFinished && (
        (searchParams.conservativePass && (&node == rootNode)) ||
        canForceNonTerminalDueToFriendlyPass
      );
      child = allocateOrFindNode(thread, thread.pla, bestChildMoveLoc, forceNonTerminal, thread.graphHash);
      child->virtualLosses.fetch_add(1,std::memory_order_release);

      {
        //Lock mutex to store child and move loc in a synchronized way
        std::lock_guard<std::mutex> lock(mutexPool->getMutex(node.mutexIdx));
        SearchNode* existingChild = children[bestChildIdx].getIfAllocated();
        if(existingChild == NULL) {
          //Set relaxed *first*, then release this value via storing the child. Anyone who load-acquires the child
          //is guaranteed by release semantics to see the move as well.
          SearchChildPointer& childPointer = children[bestChildIdx];
          childPointer.setMoveLocRelaxed(bestChildMoveLoc);
          childPointer.store(child);
        }
        else {
          //Someone got there ahead of us. We already made a move so we can't just loop again. Instead just fail this playout and try again.
          //Even if the node was newly allocated, no need to delete the node, it will get cleaned up next time we mark and sweep the node table later.
          //Clean up virtual losses in case the node is a transposition and is being used.
          child->virtualLosses.fetch_add(-1,std::memory_order_release);
          return false;
        }
      }

      //If edge visits is too much smaller than the child's visits, we can avoid descending.
      //Instead just add edge visits and treat that as a visit.
      //If we're not counting edge visits, then we're deliberately trying to add child visits beyond edge visits, skip
      if(countEdgeVisit && maybeCatchUpEdgeVisits(thread, node, child, nodeState, bestChildIdx)) {
        updateStatsAfterPlayout(node,thread,isRoot);
        child->virtualLosses.fetch_add(-1,std::memory_order_release);
        thread.shouldCountPlayout = true;
        return true;
      }
    }
    //Searching an existing child
    else {
      SearchNodeChildrenReference children = node.getChildren(nodeState);
      child = children[bestChildIdx].getIfAllocated();
      assert(child != NULL);

      child->virtualLosses.fetch_add(1,std::memory_order_release);

      //If edge visits is too much smaller than the child's visits, we can avoid descending.
      //Instead just add edge visits and treat that as a visit.
      //If we're not counting edge visits, then we're deliberately trying to add child visits beyond edge visits, skip
      if(countEdgeVisit && maybeCatchUpEdgeVisits(thread, node, child, nodeState, bestChildIdx)) {
        updateStatsAfterPlayout(node,thread,isRoot);
        child->virtualLosses.fetch_add(-1,std::memory_order_release);
        thread.shouldCountPlayout = true;
        return true;
      }

      //Make the move!
      thread.history.makeBoardMoveAssumeLegal(thread.board,bestChildMoveLoc,thread.pla,rootKoHashTable);
      thread.pla = getOpp(thread.pla);
      if(searchParams.useGraphSearch)
        thread.graphHash = GraphHash::getGraphHash(
          thread.graphHash, thread.history, thread.pla, searchParams.graphSearchRepBound, searchParams.drawEquivalentWinsForWhite
        );
    }

    break;
  }

  //If somehow we find ourselves in a cycle, increment edge visits and terminate the playout.
  //Basically if the search likes a cycle... just reinforce playing around the cycle and hope we return something
  //reasonable in the end of the search.
  //Note that this means that child visits >= edge visits is NOT an invariant.
  {
    std::pair<std::unordered_set<SearchNode*>::iterator,bool> result = thread.graphPath.insert(child);
    //No insertion, child was already there
    if(!result.second) {
      if(countEdgeVisit) {
        SearchNodeChildrenReference children = node.getChildren(nodeState);
        children[bestChildIdx].addEdgeVisits(1);
        updateStatsAfterPlayout(node,thread,isRoot);
        thread.shouldCountPlayout = true;
      }
      child->virtualLosses.fetch_add(-1,std::memory_order_release);
      // If we didn't count an edge visit, none of the parents need to update either.
      return countEdgeVisit;
    }
  }

  //Recurse!
  bool shouldUpdateChildAncestors = playoutDescend(thread,*child,false);

  //Update this node stats
  shouldUpdateChildAncestors = shouldUpdateChildAncestors && countEdgeVisit;
  if(shouldUpdateChildAncestors) {
    nodeState = node.state.load(std::memory_order_acquire);
    SearchNodeChildrenReference children = node.getChildren(nodeState);
    children[bestChildIdx].addEdgeVisits(1);
    updateStatsAfterPlayout(node,thread,isRoot);
  }
  child->virtualLosses.fetch_add(-1,std::memory_order_release);

  return shouldUpdateChildAncestors;
}


//If edge visits is too much smaller than the child's visits, we can avoid descending.
//Instead just add edge visits and return immediately.
bool Search::maybeCatchUpEdgeVisits(
  SearchThread& thread,
  SearchNode& node,
  SearchNode* child,
  const SearchNodeState& nodeState,
  const int bestChildIdx
) {
  //Don't need to do this since we already are pretty recent as of finding the best child.
  //nodeState = node.state.load(std::memory_order_acquire);
  SearchNodeChildrenReference children = node.getChildren(nodeState);
  SearchChildPointer& childPointer = children[bestChildIdx];

  // int64_t maxNumToAdd = 1;
  // if(searchParams.graphSearchCatchUpProp > 0.0) {
  //   int64_t parentVisits = node.stats.visits.load(std::memory_order_acquire);
  //   //Truncate down
  //   maxNumToAdd = 1 + (int64_t)(searchParams.graphSearchCatchUpProp * parentVisits);
  // }
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  int64_t edgeVisits = childPointer.getEdgeVisits();

  //If we want to leak through some of the time, then we keep searching the transposition node even if we'd be happy to stop here with
  //how many visits it has
  if(searchParams.graphSearchCatchUpLeakProb > 0.0 && edgeVisits < childVisits && thread.rand.nextBool(searchParams.graphSearchCatchUpLeakProb))
    return false;

  //If the edge visits exceeds the child then we need to search the child more, but as long as that's not the case,
  //we can add more edge visits.
  constexpr int64_t numToAdd = 1;
  // int64_t numToAdd;
  do {
    if(edgeVisits >= childVisits)
      return false;
    // numToAdd = std::min((childVisits - edgeVisits + 3) / 4, maxNumToAdd);
  } while(!childPointer.compexweakEdgeVisits(edgeVisits, edgeVisits + numToAdd));

  return true;
}
