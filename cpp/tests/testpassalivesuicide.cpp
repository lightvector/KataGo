#include "../tests/tests.h"
#include "../tests/testsearchcommon.h"

#include <sstream>

#include "../book/book.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../search/search.h"

using namespace std;
using namespace TestCommon;

//Tests for BoardHistory::alwaysComputePassAliveUnderSuicideRules and its plumbing.
//Prints only a single completion line on success - all checks are testAsserts.
void Tests::runPassAliveSuicideModeTests() {
  //A position where pass-alive computation differs between suicide-legal and suicide-illegal modes.
  //Under suicide-illegal computation, the three white stones are pass-dead inside black
  //pass-alive territory (an ecosystem of mutually-supporting black chains). Under suicide-legal
  //computation, none of the black chains are pass-alive at all.
  Board board = Board::parseBoard(4,5,R"%%(
...x
xx..
oxxx
oox.
.x.x
)%%");

  Rules rules = Rules::parseRules("chinese");
  testAssert(!rules.multiStoneSuicideLegal);

  //Sanity check that this position really does discriminate the two computations.
  {
    Color areaNoSuicide[Board::MAX_ARR_SIZE];
    Color areaSuicide[Board::MAX_ARR_SIZE];
    board.calculateArea(areaNoSuicide,false,false,false,false);
    board.calculateArea(areaSuicide,false,false,false,true);
    Loc whiteLoc = Location::getLoc(0,2,board.x_size);
    testAssert(board.colors[whiteLoc] == C_WHITE);
    testAssert(areaNoSuicide[whiteLoc] == C_BLACK);
    testAssert(areaSuicide[whiteLoc] == C_EMPTY);
  }

  //Basic flag behavior and effect on scoring
  {
    BoardHistory hist(board,P_BLACK,rules,0,false);
    testAssert(!hist.alwaysComputePassAliveUnderSuicideRules);
    testAssert(!hist.suicideLegalForPassAlive());

    BoardHistory histFlagged(board,P_BLACK,rules,0,false);
    histFlagged.setAlwaysComputePassAliveUnderSuicideRules(true);
    testAssert(histFlagged.alwaysComputePassAliveUnderSuicideRules);
    testAssert(histFlagged.suicideLegalForPassAlive());

    //The flag changes the situation-and-rules hash exactly when the rules don't already have suicide legal
    Hash128 hashOff = BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5);
    Hash128 hashOn = BoardHistory::getSituationRulesAndKoHash(board,histFlagged,P_BLACK,0.5);
    testAssert(hashOff != hashOn);

    //And changes area scoring on this position
    BoardHistory histScore(board,P_BLACK,rules,0,false);
    Board boardCopy = board;
    histScore.endAndScoreGameNow(boardCopy);
    BoardHistory histScoreFlagged(board,P_BLACK,rules,0,false);
    histScoreFlagged.setAlwaysComputePassAliveUnderSuicideRules(true);
    Board boardCopy2 = board;
    histScoreFlagged.endAndScoreGameNow(boardCopy2);
    testAssert(histScore.isScored && histScoreFlagged.isScored);
    testAssert(histScore.finalWhiteMinusBlackScore != histScoreFlagged.finalWhiteMinusBlackScore);

    //And changes territory scoring too (the countTerritoryAreaScoreWhiteMinusBlack path)
    Rules japRules = Rules::parseRules("japanese");
    testAssert(!japRules.multiStoneSuicideLegal);
    BoardHistory histTerrScore(board,P_BLACK,japRules,0,false);
    Board boardCopy3 = board;
    histTerrScore.endAndScoreGameNow(boardCopy3);
    BoardHistory histTerrScoreFlagged(board,P_BLACK,japRules,0,false);
    histTerrScoreFlagged.setAlwaysComputePassAliveUnderSuicideRules(true);
    Board boardCopy4 = board;
    histTerrScoreFlagged.endAndScoreGameNow(boardCopy4);
    testAssert(histTerrScore.isScored && histTerrScoreFlagged.isScored);
    testAssert(histTerrScore.finalWhiteMinusBlackScore != histTerrScoreFlagged.finalWhiteMinusBlackScore);

    //Copying and clear() preserve the flag
    BoardHistory copied(histFlagged);
    testAssert(copied.alwaysComputePassAliveUnderSuicideRules);
    BoardHistory assigned;
    assigned = histFlagged;
    testAssert(assigned.alwaysComputePassAliveUnderSuicideRules);
    BoardHistory cleared(histFlagged);
    cleared.clear(board,P_BLACK,rules,0);
    testAssert(cleared.alwaysComputePassAliveUnderSuicideRules);
    testAssert(histFlagged.copyToInitial().alwaysComputePassAliveUnderSuicideRules);
  }

  //When the rules already have suicide legal, the flag is a no-op for hashing and scoring
  {
    Rules tromp = Rules::parseRules("tromp-taylor");
    testAssert(tromp.multiStoneSuicideLegal);
    BoardHistory hist(board,P_BLACK,tromp,0,false);
    BoardHistory histFlagged(board,P_BLACK,tromp,0,false);
    histFlagged.setAlwaysComputePassAliveUnderSuicideRules(true);
    testAssert(hist.suicideLegalForPassAlive() && histFlagged.suicideLegalForPassAlive());
    Hash128 hashOff = BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5);
    Hash128 hashOn = BoardHistory::getSituationRulesAndKoHash(board,histFlagged,P_BLACK,0.5);
    testAssert(hashOff == hashOn);
  }

  //MiscNNInputParams override behavior for the nn cache hash
  {
    BoardHistory hist(board,P_BLACK,rules,0,false);
    BoardHistory histFlagged(board,P_BLACK,rules,0,false);
    histFlagged.setAlwaysComputePassAliveUnderSuicideRules(true);

    MiscNNInputParams noOverride;
    MiscNNInputParams forceOn;
    forceOn.passAliveSuicideRulesOverride = 1;
    MiscNNInputParams forceOff;
    forceOff.passAliveSuicideRulesOverride = 0;

    testAssert(!noOverride.getSuicideLegalForPassAlive(hist));
    testAssert(noOverride.getSuicideLegalForPassAlive(histFlagged));
    testAssert(forceOn.getSuicideLegalForPassAlive(hist));
    testAssert(!forceOff.getSuicideLegalForPassAlive(histFlagged));

    //Hash reflects the effective featurization: flag-off + force-on == flag-on + no-override, etc.
    Hash128 offPlain = NNInputs::getHash(board,hist,P_BLACK,noOverride);
    Hash128 onPlain = NNInputs::getHash(board,histFlagged,P_BLACK,noOverride);
    Hash128 offForcedOn = NNInputs::getHash(board,hist,P_BLACK,forceOn);
    Hash128 onForcedOff = NNInputs::getHash(board,histFlagged,P_BLACK,forceOff);
    testAssert(offPlain != onPlain);
    testAssert(offForcedOn == onPlain);
    testAssert(onForcedOff == offPlain);
  }

  //Search maintains the invariant that its root history's flag always matches what its params
  //and nnEval resolve to, regardless of the history set into it.
  {
    Logger logger(nullptr,false,false);
    NNEvaluator* nnEval = TestSearchCommon::startNNEval(
      "",logger,"passAliveSuicideModeTest",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,
      0,true,false,false,true,false
    );
    //With no loaded model, auto resolves to false
    testAssert(!nnEval->modelPreferPassAliveUnderSuicideRules());

    SearchParams params = SearchParams::forTestsV2();
    testAssert(params.alwaysComputePassAliveUnderSuicideRules == enabled_t::Auto);

    {
      Search search(params,nnEval,&logger,"passAliveStampTestSeed");
      testAssert(!search.getRootHist().alwaysComputePassAliveUnderSuicideRules);
      testAssert(!Search::resolveAlwaysComputePassAliveUnderSuicideRules(params,nnEval));

      //Setting a history with the flag on gets re-stamped to the search's own resolution
      BoardHistory histFlagged(board,P_BLACK,rules,0,false);
      histFlagged.setAlwaysComputePassAliveUnderSuicideRules(true);
      search.setPosition(P_BLACK,board,histFlagged);
      testAssert(!search.getRootHist().alwaysComputePassAliveUnderSuicideRules);

      //Forcing the param stamps the flag on, even though the last set position had it off
      SearchParams paramsOn = params;
      paramsOn.alwaysComputePassAliveUnderSuicideRules = enabled_t::True;
      testAssert(Search::resolveAlwaysComputePassAliveUnderSuicideRules(paramsOn,nnEval));
      search.setParams(paramsOn);
      testAssert(search.getRootHist().alwaysComputePassAliveUnderSuicideRules);

      //And even the no-clearing param setter keeps the invariant
      search.setParamsNoClearing(params);
      testAssert(!search.getRootHist().alwaysComputePassAliveUnderSuicideRules);
      search.setParamsNoClearing(paramsOn);
      testAssert(search.getRootHist().alwaysComputePassAliveUnderSuicideRules);

      //setPlayerAndClearHistory and setKomiIfNew preserve it too
      search.setPlayerAndClearHistory(P_WHITE);
      testAssert(search.getRootHist().alwaysComputePassAliveUnderSuicideRules);
      search.setKomiIfNew(5.5f);
      testAssert(search.getRootHist().alwaysComputePassAliveUnderSuicideRules);
    }

    nnEval->killServerThreads();
    delete nnEval;
  }

  //Books record the pass-alive computation mode and stamp it on their histories.
  //Mode-true books require book version >= 3 (so old binaries reject them cleanly)
  {
    Rules bookRules = Rules::parseRules("chinese");
    Board initialBoard(9,9);
    BookParams bparams;
    for(bool mode : {false,true}) {
      Book book(Book::LATEST_BOOK_VERSION, initialBoard, bookRules, P_BLACK, 3, mode, bparams);
      testAssert(book.alwaysComputePassAliveUnderSuicideRules == mode);
      testAssert(book.getInitialHist().alwaysComputePassAliveUnderSuicideRules == mode);

      std::ostringstream saved;
      book.saveToStream(saved);
      {
        std::istringstream headerIn(saved.str());
        testAssert(Book::readAlwaysComputePassAliveUnderSuicideRulesOfHeader(headerIn) == mode);
      }
      std::istringstream loadIn(saved.str());
      Book* loaded = Book::loadFromStream(loadIn);
      testAssert(loaded->alwaysComputePassAliveUnderSuicideRules == mode);
      testAssert(loaded->bookVersion == Book::LATEST_BOOK_VERSION);
      testAssert(loaded->getInitialHist().alwaysComputePassAliveUnderSuicideRules == mode);
      delete loaded;
    }

    //Constructing a mode-true book at a pre-migration version is an error.
    {
      bool threw = false;
      try {
        Book book(2, initialBoard, bookRules, P_BLACK, 3, true, bparams);
      }
      catch(const StringError&) {
        threw = true;
      }
      testAssert(threw);
    }

    //A pre-migration book file (header key absent entirely, version 2) loads as mode false.
    {
      Book book(2, initialBoard, bookRules, P_BLACK, 3, false, bparams);
      std::ostringstream saved;
      book.saveToStream(saved);

      string contents = saved.str();
      const string key = "\"alwaysComputePassAliveUnderSuicideRules\":false,";
      size_t keyPos = contents.find(key);
      testAssert(keyPos != string::npos);
      contents.erase(keyPos, key.size());

      {
        std::istringstream headerIn(contents);
        testAssert(!Book::readAlwaysComputePassAliveUnderSuicideRulesOfHeader(headerIn));
      }
      std::istringstream loadIn(contents);
      Book* loaded = Book::loadFromStream(loadIn);
      testAssert(!loaded->alwaysComputePassAliveUnderSuicideRules);
      testAssert(loaded->getInitialHist().alwaysComputePassAliveUnderSuicideRules == false);
      delete loaded;
    }
  }

  //The nn cache cannot mix results across featurization modes: with suicide-illegal rules the two
  //modes hash differently (distinct cache entries), and with suicide-legal rules the mode is a
  //genuine no-op and deliberately shares cache entries.
  {
    Logger logger(nullptr,false,false);
    NNEvaluator* nnEval = TestSearchCommon::startNNEval(
      "",logger,"passAliveSuicideCacheTest",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,
      0,true,false,false,true,false
    );

    MiscNNInputParams noOverride;
    MiscNNInputParams forceOn;
    forceOn.passAliveSuicideRulesOverride = 1;

    auto sameOutputs = [](const NNOutput& a, const NNOutput& b) {
      if(a.whiteWinProb != b.whiteWinProb || a.whiteLossProb != b.whiteLossProb || a.whiteScoreMean != b.whiteScoreMean)
        return false;
      for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
        if(a.policyProbs[i] != b.policyProbs[i])
          return false;
      return true;
    };

    //Suicide-illegal rules: the modes must hit distinct cache entries.
    {
      BoardHistory hist(board,P_BLACK,rules,0,false);
      NNResultBuf buf;
      nnEval->evaluate(board,hist,P_BLACK,noOverride,buf,false,false);
      std::shared_ptr<NNOutput> outPlain = std::move(buf.result);
      nnEval->evaluate(board,hist,P_BLACK,forceOn,buf,false,false);
      std::shared_ptr<NNOutput> outForcedOn = std::move(buf.result);
      nnEval->evaluate(board,hist,P_BLACK,noOverride,buf,false,false);
      std::shared_ptr<NNOutput> outPlainAgain = std::move(buf.result);

      //The debug-skip evaluator returns fresh random outputs on every cache miss, so distinct
      //entries differ and a repeated query only matches if it hit the cache.
      testAssert(!sameOutputs(*outPlain,*outForcedOn));
      testAssert(sameOutputs(*outPlain,*outPlainAgain));
    }

    //Suicide-legal rules: the override is a no-op and must share the cache entry.
    {
      Rules tromp = Rules::parseRules("tromp-taylor");
      BoardHistory hist(board,P_BLACK,tromp,0,false);
      NNResultBuf buf;
      nnEval->evaluate(board,hist,P_BLACK,noOverride,buf,false,false);
      std::shared_ptr<NNOutput> outPlain = std::move(buf.result);
      nnEval->evaluate(board,hist,P_BLACK,forceOn,buf,false,false);
      std::shared_ptr<NNOutput> outForcedOn = std::move(buf.result);
      testAssert(sameOutputs(*outPlain,*outForcedOn));
    }

    nnEval->killServerThreads();
    delete nnEval;
  }

  cout << "Ran pass-alive suicide mode tests" << endl;
}
