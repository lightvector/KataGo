#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

void Search::computeRootNNEvaluation(NNResultBuf& nnResultBuf, bool includeOwnerMap) {
  Board board = rootBoard;
  const BoardHistory& hist = rootHistory;
  Player pla = rootPla;
  bool skipCache = false;
  bool isRoot = true;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.conservativePassAndIsRoot = searchParams.conservativePass && isRoot;
  nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == pla;
  nnInputParams.policyOptimism = searchParams.rootPolicyOptimism;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }
  if(searchParams.ignorePreRootHistory || searchParams.ignoreAllHistory)
    nnInputParams.maxHistory = 0;
  nnEvaluator->evaluate(
    board, hist, pla, &searchParams.humanSLProfile,
    nnInputParams,
    nnResultBuf, skipCache, includeOwnerMap
  );
}

bool Search::needsHumanOutputAtRoot() const {
  return humanEvaluator != NULL && (searchParams.humanSLProfile.initialized || !humanEvaluator->requiresSGFMetadata());
}
bool Search::needsHumanOutputInTree() const {
  return needsHumanOutputAtRoot() && (
    searchParams.humanSLPlaExploreProbWeightless > 0 ||
    searchParams.humanSLPlaExploreProbWeightful > 0 ||
    searchParams.humanSLOppExploreProbWeightless > 0 ||
    searchParams.humanSLOppExploreProbWeightful > 0
  );
}

//If isReInit is false, among any threads trying to store, the first one wins
//If isReInit is true, we always replace, even for threads that come later.
//Returns true if a nnOutput was set where there was none before.
bool Search::initNodeNNOutput(
  SearchThread& thread, SearchNode& node,
  bool isRoot, bool skipCache, bool isReInit
) {
  bool includeOwnerMap = isRoot || alwaysIncludeOwnerMap;
  bool antiMirrorDifficult = false;
  if(searchParams.antiMirror && mirroringPla != C_EMPTY && mirrorAdvantage >= -0.5 &&
     Location::getCenterLoc(thread.board) != Board::NULL_LOC && thread.board.colors[Location::getCenterLoc(thread.board)] == getOpp(rootPla) &&
     isMirroringSinceSearchStart(thread.history,4) // skip recent 4 ply to be a bit tolerant
  ) {
    includeOwnerMap = true;
    antiMirrorDifficult = true;
  }
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.conservativePassAndIsRoot = searchParams.conservativePass && isRoot;
  nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == thread.pla;
  nnInputParams.policyOptimism = isRoot ? searchParams.rootPolicyOptimism : searchParams.policyOptimism;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(thread.pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }
  if(searchParams.ignoreAllHistory)
    nnInputParams.maxHistory = 0;
  else if(searchParams.ignorePreRootHistory) {
    nnInputParams.maxHistory = isRoot ? 0 : std::max(0, (int)thread.history.moveHistory.size() - (int)rootHistory.moveHistory.size());
  }

  std::shared_ptr<NNOutput>* result = NULL;
  std::shared_ptr<NNOutput>* humanResult = NULL;
  if(isRoot && searchParams.rootNumSymmetriesToSample > 1) {
    result = nnEvaluator->averageMultipleSymmetries(
      thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
      nnInputParams,
      thread.nnResultBuf, includeOwnerMap,
      thread.rand, searchParams.rootNumSymmetriesToSample
    );
    if(needsHumanOutputInTree() || (isRoot && needsHumanOutputAtRoot())) {
      humanResult = humanEvaluator->averageMultipleSymmetries(
        thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
        nnInputParams,
        thread.nnResultBuf, includeOwnerMap,
        thread.rand, searchParams.rootNumSymmetriesToSample
      );
    }
  }
  else {
    nnEvaluator->evaluate(
      thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
      nnInputParams,
      thread.nnResultBuf, skipCache, includeOwnerMap
    );
    result = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
    if(needsHumanOutputInTree() || (isRoot && needsHumanOutputAtRoot())) {
      humanEvaluator->evaluate(
        thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
        nnInputParams,
        thread.nnResultBuf, skipCache, includeOwnerMap
      );
      humanResult = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
    }
  }

  if(antiMirrorDifficult) {
    // Copy
    std::shared_ptr<NNOutput>* newNNOutputSharedPtr = new std::shared_ptr<NNOutput>(new NNOutput(**result));
    std::shared_ptr<NNOutput>* tmp = result;
    result = newNNOutputSharedPtr;
    delete tmp;
    hackNNOutputForMirror(*result);
  }

  assert((*result)->noisedPolicyProbs == NULL);
  std::shared_ptr<NNOutput>* noisedResult = maybeAddPolicyNoiseAndTemp(thread,isRoot,result->get());
  if(noisedResult != NULL) {
    std::shared_ptr<NNOutput>* tmp = result;
    result = noisedResult;
    delete tmp;
  }

  node.nodeAge.store(searchNodeAge,std::memory_order_release);
  //If this is a re-initialization of the nnOutput, we don't want to add any visits or anything.
  //Also don't bother updating any of the stats. Technically we should do so because winLossValueSum
  //and such will have changed potentially due to a new orientation of the neural net eval
  //slightly affecting the evals, but this is annoying to recompute from scratch, and on the next
  //visit updateStatsAfterPlayout should fix it all up anyways.
  if(isReInit) {
    if(humanResult != NULL)
      node.storeHumanOutput(humanResult,thread); // ignore the wasNullBefore from this one
    bool wasNullBefore = node.storeNNOutput(result,thread);
    return wasNullBefore;
  }
  else {
    // Store human result first, so that the presence of the main result guarantees
    // that the human result exists in the case we have a human evaluator.
    if(humanResult != NULL) {
      bool humanSuc = node.storeHumanOutputIfNull(humanResult);
      if(!humanSuc)
        delete humanResult;
    }
    bool suc = node.storeNNOutputIfNull(result);
    if(!suc)
      delete result;
    else {
      addCurrentNNOutputAsLeafValue(node,true);
    }
    return suc;
  }
}


//Assumes node already has an nnOutput
bool Search::maybeRecomputeExistingNNOutput(
  SearchThread& thread, SearchNode& node, bool isRoot
) {
  bool recomputeHappened = false;
  //Right now only the root node currently ever needs to recompute, and only if it's old
  if(isRoot && node.nodeAge.load(std::memory_order_acquire) != searchNodeAge) {
    //See if we're the lucky thread that gets to do the update!
    //Threads that pass by here later will NOT wait for us to be done before proceeding with search.
    //We accept this and tolerate that for a few iterations potentially we will be using the OLD policy - without noise,
    //or without root temperature, etc.
    //Or if we have none of those things, then we'll not end up updating anything except the age, which is okay too.
    uint32_t oldAge = node.nodeAge.exchange(searchNodeAge,std::memory_order_acq_rel);
    if(oldAge < searchNodeAge) {
      NNOutput* nnOutput = node.getNNOutput();
      NNOutput* humanOutput = node.getHumanOutput();
      assert(nnOutput != NULL);

      //Recompute if we have no ownership map, since we need it for getEndingWhiteScoreBonus
      //If conservative passing, then we may also need to recompute the root policy ignoring the history if a pass ends the game
      //If averaging a bunch of symmetries, then we need to recompute it too
      //If root needs different optimism, we need to recompute it.
      //If humanSL is missing, but we want it, we need to recompute.
      //Also do so when ignoring history pre root
      if(nnOutput->whiteOwnerMap == NULL ||
         (searchParams.conservativePass && thread.history.passWouldEndGame(thread.board,thread.pla)) ||
         searchParams.rootNumSymmetriesToSample > 1 ||
         searchParams.rootPolicyOptimism != searchParams.policyOptimism ||
         (searchParams.ignorePreRootHistory && !searchParams.ignoreAllHistory) ||
         (humanOutput == NULL && needsHumanOutputAtRoot())
      ) {
        //We *can* use cached evaluations even though parameters are changing, because:
        //conservativePass is part of the nn hash
        //Symmetry averaging skips the cache on its own when it does symmetry sampling without replacement
        //The optimism is part of the nn hash
        //When pre-root history is ignored at the root, maxHistory is 0 and the nn cache distinguishes 0 from nonzero.
        const bool skipCache = false;
        initNodeNNOutput(thread,node,isRoot,skipCache,true);
        recomputeHappened = true;
      }
      //We also need to recompute the root nn if we have root noise or temperature and that's missing.
      else {
        //We don't need to go all the way to the nnEvaluator, we just need to maybe add those transforms
        //to the existing policy.
        std::shared_ptr<NNOutput>* result = maybeAddPolicyNoiseAndTemp(thread,isRoot,nnOutput);
        if(result != NULL) {
          node.storeNNOutput(result,thread);
          recomputeHappened = true;
        }
      }
    }
  }
  return recomputeHappened;
}
