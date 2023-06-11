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
  nnEvaluator->evaluate(
    board, hist, pla,
    nnInputParams,
    nnResultBuf, skipCache, includeOwnerMap
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

  std::shared_ptr<NNOutput>* result;
  if(isRoot && searchParams.rootNumSymmetriesToSample > 1) {
    vector<std::shared_ptr<NNOutput>> ptrs;
    std::array<int, SymmetryHelpers::NUM_SYMMETRIES> symmetryIndexes;
    std::iota(symmetryIndexes.begin(), symmetryIndexes.end(), 0);
    for(int i = 0; i<searchParams.rootNumSymmetriesToSample; i++) {
      std::swap(symmetryIndexes[i], symmetryIndexes[thread.rand.nextInt(i,SymmetryHelpers::NUM_SYMMETRIES-1)]);
      nnInputParams.symmetry = symmetryIndexes[i];
      bool skipCacheThisIteration = true; //Skip cache since there's no guarantee which symmetry is in the cache
      nnEvaluator->evaluate(
        thread.board, thread.history, thread.pla,
        nnInputParams,
        thread.nnResultBuf, skipCacheThisIteration, includeOwnerMap
      );
      ptrs.push_back(std::move(thread.nnResultBuf.result));
    }
    result = new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
  }
  else {
    nnEvaluator->evaluate(
      thread.board, thread.history, thread.pla,
      nnInputParams,
      thread.nnResultBuf, skipCache, includeOwnerMap
    );
    result = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
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
    bool wasNullBefore = node.storeNNOutput(result,thread);
    return wasNullBefore;
  }
  else {
    bool suc = node.storeNNOutputIfNull(result);
    if(!suc) {
      delete result;
      return false;
    }
    addCurrentNNOutputAsLeafValue(node,true);
    return true;
  }
}


//Assumes node already has an nnOutput
void Search::maybeRecomputeExistingNNOutput(
  SearchThread& thread, SearchNode& node, bool isRoot
) {
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
      assert(nnOutput != NULL);

      //Recompute if we have no ownership map, since we need it for getEndingWhiteScoreBonus
      //If conservative passing, then we may also need to recompute the root policy ignoring the history if a pass ends the game
      //If averaging a bunch of symmetries, then we need to recompute it too
      //If root needs different optimism, we need to recompute it.
      if(nnOutput->whiteOwnerMap == NULL ||
         (searchParams.conservativePass && thread.history.passWouldEndGame(thread.board,thread.pla)) ||
         searchParams.rootNumSymmetriesToSample > 1 ||
         searchParams.rootPolicyOptimism != searchParams.policyOptimism
      ) {
        //We *can* use cached evaluations even though parameters are changing, because:
        //conservativePass is part of the nn hash
        //Symmetry averaging skips the cache on its own when it does symmetry sampling without replacement
        //The optimism is part of the nn hash
        const bool skipCache = false;
        initNodeNNOutput(thread,node,isRoot,skipCache,true);
      }
      //We also need to recompute the root nn if we have root noise or temperature and that's missing.
      else {
        //We don't need to go all the way to the nnEvaluator, we just need to maybe add those transforms
        //to the existing policy.
        std::shared_ptr<NNOutput>* result = maybeAddPolicyNoiseAndTemp(thread,isRoot,nnOutput);
        if(result != NULL)
          node.storeNNOutput(result,thread);
      }
    }
  }
}
