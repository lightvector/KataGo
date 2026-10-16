#include "../tests/tests.h"
#include "../tests/testsearchcommon.h"

#include <sstream>

#include "../book/book.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../search/search.h"

using namespace std;
using namespace TestCommon;

//Tests for BoardHistoryModes::excludeTerritoryAdjacentToAtari and its plumbing.
//Prints only a single completion line on success - all checks are testAsserts.
void Tests::runExcludeTerritoryAtariModeTests() {
  //A whole-board seki with an unfilled ko mouth. The top-right black group's only liberty is its
  //eye at (5,1), so the group is in atari and under rules v3 (excludeTerritoryAdjacentToAtari)
  //the eye is not a point for black under territory scoring with TaxRule NONE, while under
  //rules v2 it is.
  Board board = Board::parseBoard(7,7,R"%%(
...oxxx
oooox.x
xxxxoxx
o.xoooo
ooxox.o
oxxo.xo
o.xooxx
)%%");

  Rules terrNoTaxRules;
  terrNoTaxRules.koRule = Rules::KO_SIMPLE;
  terrNoTaxRules.scoringRule = Rules::SCORING_TERRITORY;
  terrNoTaxRules.taxRule = Rules::TAX_NONE;
  terrNoTaxRules.multiStoneSuicideLegal = false;
  terrNoTaxRules.komi = -0.5f;

  const Loc eyeLoc = Location::getLoc(5,1,board.x_size);
  testAssert(board.colors[eyeLoc] == C_EMPTY);
  testAssert(board.getNumLiberties(Location::getLoc(5,0,board.x_size)) == 1);

  //Sanity check that this position really does discriminate the two computations.
  {
    Color areaV2[Board::MAX_ARR_SIZE];
    Color areaV3[Board::MAX_ARR_SIZE];
    int count;
    board.calculateIndependentLifeArea(areaV2,count,true,false,false,false);
    board.calculateIndependentLifeArea(areaV3,count,true,false,true,false);
    testAssert(areaV2[eyeLoc] == C_BLACK);
    testAssert(areaV3[eyeLoc] == C_EMPTY);
  }

  //Basic flag behavior and effect on scoring
  {
    BoardHistory hist(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
    testAssert(!hist.modes.excludeTerritoryAdjacentToAtari);

    BoardHistory histFlagged(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
    histFlagged.setModes(BoardHistoryModes(false,true));
    testAssert(histFlagged.modes.excludeTerritoryAdjacentToAtari);

    //The flag changes the situation-and-rules hash under territory scoring with TaxRule NONE
    Hash128 hashOff = BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5);
    Hash128 hashOn = BoardHistory::getSituationRulesAndKoHash(board,histFlagged,P_BLACK,0.5);
    testAssert(hashOff != hashOn);

    //And changes territory scoring by exactly the ko mouth point (black loses 1 point)
    BoardHistory histScore(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
    Board boardCopy = board;
    histScore.endAndScoreGameNow(boardCopy);
    BoardHistory histScoreFlagged(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,true));
    Board boardCopy2 = board;
    histScoreFlagged.endAndScoreGameNow(boardCopy2);
    testAssert(histScore.isScored && histScoreFlagged.isScored);
    testAssert(histScoreFlagged.finalWhiteMinusBlackScore == histScore.finalWhiteMinusBlackScore + 1.0f);

    //Copying and clear() preserve the flag
    BoardHistory copied(histFlagged);
    testAssert(copied.modes.excludeTerritoryAdjacentToAtari);
    BoardHistory assigned;
    assigned = histFlagged;
    testAssert(assigned.modes.excludeTerritoryAdjacentToAtari);
    BoardHistory cleared(histFlagged);
    cleared.clear(board,P_BLACK,terrNoTaxRules,0);
    testAssert(cleared.modes.excludeTerritoryAdjacentToAtari);
    testAssert(histFlagged.copyToInitial().modes.excludeTerritoryAdjacentToAtari);
  }

  //Under any other scoring/tax rules, the flag is a no-op for hashing and scoring
  {
    //Area scoring with TaxRule NONE
    Rules areaRules = terrNoTaxRules;
    areaRules.scoringRule = Rules::SCORING_AREA;
    BoardHistory hist(board,P_BLACK,areaRules,0,BoardHistoryModes(false,false));
    BoardHistory histFlagged(board,P_BLACK,areaRules,0,BoardHistoryModes(false,true));
    testAssert(
      BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5) ==
      BoardHistory::getSituationRulesAndKoHash(board,histFlagged,P_BLACK,0.5)
    );
    Board boardCopy = board;
    hist.endAndScoreGameNow(boardCopy);
    Board boardCopy2 = board;
    histFlagged.endAndScoreGameNow(boardCopy2);
    testAssert(hist.finalWhiteMinusBlackScore == histFlagged.finalWhiteMinusBlackScore);

    //Territory scoring with TaxRule SEKI
    Rules terrSekiRules = terrNoTaxRules;
    terrSekiRules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist2(board,P_BLACK,terrSekiRules,0,BoardHistoryModes(false,false));
    BoardHistory hist2Flagged(board,P_BLACK,terrSekiRules,0,BoardHistoryModes(false,true));
    testAssert(
      BoardHistory::getSituationRulesAndKoHash(board,hist2,P_BLACK,0.5) ==
      BoardHistory::getSituationRulesAndKoHash(board,hist2Flagged,P_BLACK,0.5)
    );
    Board boardCopy3 = board;
    hist2.endAndScoreGameNow(boardCopy3);
    Board boardCopy4 = board;
    hist2Flagged.endAndScoreGameNow(boardCopy4);
    testAssert(hist2.finalWhiteMinusBlackScore == hist2Flagged.finalWhiteMinusBlackScore);
  }

  //The two flags hash independently
  {
    BoardHistory hist(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
    Hash128 h00 = BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5,BoardHistoryModes(false,false));
    Hash128 h01 = BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5,BoardHistoryModes(false,true));
    Hash128 h10 = BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5,BoardHistoryModes(true,false));
    Hash128 h11 = BoardHistory::getSituationRulesAndKoHash(board,hist,P_BLACK,0.5,BoardHistoryModes(true,true));
    testAssert(h00 != h01 && h00 != h10 && h00 != h11);
    testAssert(h01 != h10 && h01 != h11 && h10 != h11);
  }

  //MiscNNInputParams override behavior for the nn cache hash
  {
    BoardHistory hist(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
    BoardHistory histFlagged(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,true));

    MiscNNInputParams noOverride;
    MiscNNInputParams forceOn;
    forceOn.excludeTerritoryAdjAtariOverride = 1;
    MiscNNInputParams forceOff;
    forceOff.excludeTerritoryAdjAtariOverride = 0;

    testAssert(!noOverride.getExcludeTerritoryAdjacentToAtari(hist));
    testAssert(noOverride.getExcludeTerritoryAdjacentToAtari(histFlagged));
    testAssert(forceOn.getExcludeTerritoryAdjacentToAtari(hist));
    testAssert(!forceOff.getExcludeTerritoryAdjacentToAtari(histFlagged));

    //Hash reflects the effective featurization: flag-off + force-on == flag-on + no-override, etc.
    Hash128 offPlain = NNInputs::getHash(board,hist,P_BLACK,noOverride);
    Hash128 onPlain = NNInputs::getHash(board,histFlagged,P_BLACK,noOverride);
    Hash128 offForcedOn = NNInputs::getHash(board,hist,P_BLACK,forceOn);
    Hash128 onForcedOff = NNInputs::getHash(board,histFlagged,P_BLACK,forceOff);
    testAssert(offPlain != onPlain);
    testAssert(offForcedOn == onPlain);
    testAssert(onForcedOff == offPlain);
  }

  //Search maintains the invariant that its root history's modes always match what its params
  //and nnEval resolve to, regardless of the history set into it.
  {
    Logger logger(nullptr,false,false);
    NNEvaluator* nnEval = TestSearchCommon::startNNEval(
      "",logger,"excludeTerritoryAtariModeTest",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,
      0,true,false,false,true,false
    );
    //With no loaded model, auto resolves to false
    testAssert(!nnEval->modelPreferExcludeTerritoryAdjacentToAtari());

    SearchParams params = SearchParams::forTestsV2();
    testAssert(params.excludeTerritoryAdjacentToAtari == enabled_t::Auto);

    {
      Search search(params,nnEval,&logger,"excludeTerritoryAtariStampTestSeed");
      testAssert(!search.getRootHist().modes.excludeTerritoryAdjacentToAtari);
      testAssert(!Search::resolveExcludeTerritoryAdjacentToAtari(params,nnEval));

      //Setting a history with the flag on gets re-stamped to the search's own resolution
      BoardHistory histFlagged(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,true));
      search.setPosition(P_BLACK,board,histFlagged);
      testAssert(!search.getRootHist().modes.excludeTerritoryAdjacentToAtari);

      //Forcing the param stamps the flag on, even though the last set position had it off
      SearchParams paramsOn = params;
      paramsOn.excludeTerritoryAdjacentToAtari = enabled_t::True;
      testAssert(Search::resolveExcludeTerritoryAdjacentToAtari(paramsOn,nnEval));
      search.setParams(paramsOn);
      testAssert(search.getRootHist().modes.excludeTerritoryAdjacentToAtari);

      //And even the no-clearing param setter keeps the invariant
      search.setParamsNoClearing(params);
      testAssert(!search.getRootHist().modes.excludeTerritoryAdjacentToAtari);
      search.setParamsNoClearing(paramsOn);
      testAssert(search.getRootHist().modes.excludeTerritoryAdjacentToAtari);

      //setPlayerAndClearHistory and setKomiIfNew preserve it too
      search.setPlayerAndClearHistory(P_WHITE);
      testAssert(search.getRootHist().modes.excludeTerritoryAdjacentToAtari);
      search.setKomiIfNew(5.5f);
      testAssert(search.getRootHist().modes.excludeTerritoryAdjacentToAtari);
    }

    nnEval->killServerThreads();
    delete nnEval;
  }

  //Books record the BoardHistoryModes and stamp them on their histories.
  //Books flagged with this mode require book version >= 4, the version that introduced it, so that
  //binaries predating the mode reject them cleanly rather than ignoring the unrecognized header key
  //and mis-hashing the whole book.
  {
    Rules bookRules = Rules::parseRules("chinese");
    Board initialBoard(9,9);
    BookParams bparams;
    for(bool mode : {false,true}) {
      Book book(Book::LATEST_BOOK_VERSION, initialBoard, bookRules, P_BLACK, 3, BoardHistoryModes(false,mode), bparams);
      testAssert(book.historyModes.excludeTerritoryAdjacentToAtari == mode);
      testAssert(book.getInitialHist().modes.excludeTerritoryAdjacentToAtari == mode);

      std::ostringstream saved;
      book.saveToStream(saved);
      {
        std::istringstream headerIn(saved.str());
        testAssert(Book::readHistoryModesOfHeader(headerIn) == BoardHistoryModes(false,mode));
      }
      std::istringstream loadIn(saved.str());
      Book* loaded = Book::loadFromStream(loadIn);
      testAssert(loaded->historyModes.excludeTerritoryAdjacentToAtari == mode);
      testAssert(loaded->bookVersion == Book::LATEST_BOOK_VERSION);
      testAssert(loaded->getInitialHist().modes.excludeTerritoryAdjacentToAtari == mode);
      delete loaded;
    }

    //Flagging a book with a mode at a version predating that mode is an error. Version 3 introduced
    //alwaysComputePassAliveUnderSuicideRules and version 4 introduced this mode, so a version-3 book
    //may carry the former but not the latter.
    auto constructThrows = [&](int version, const BoardHistoryModes& modes) {
      bool threw = false;
      try {
        Book book(version, initialBoard, bookRules, P_BLACK, 3, modes, bparams);
      }
      catch(const StringError&) {
        threw = true;
      }
      return threw;
    };
    testAssert(constructThrows(2,BoardHistoryModes(false,true)));
    testAssert(constructThrows(3,BoardHistoryModes(false,true)));
    testAssert(constructThrows(2,BoardHistoryModes(true,false)));
    testAssert(!constructThrows(3,BoardHistoryModes(true,false)));
    testAssert(!constructThrows(4,BoardHistoryModes(false,true)));
    testAssert(Book::LATEST_BOOK_VERSION >= 4);

    //A book file with the header key absent entirely loads as mode false.
    {
      Book book(Book::LATEST_BOOK_VERSION, initialBoard, bookRules, P_BLACK, 3, BoardHistoryModes(), bparams);
      std::ostringstream saved;
      book.saveToStream(saved);

      string contents = saved.str();
      const string key = "\"excludeTerritoryAdjacentToAtari\":false,";
      size_t keyPos = contents.find(key);
      testAssert(keyPos != string::npos);
      contents.erase(keyPos, key.size());

      {
        std::istringstream headerIn(contents);
        testAssert(Book::readHistoryModesOfHeader(headerIn) == BoardHistoryModes());
      }
      std::istringstream loadIn(contents);
      Book* loaded = Book::loadFromStream(loadIn);
      testAssert(!loaded->historyModes.excludeTerritoryAdjacentToAtari);
      delete loaded;
    }
  }

  //The nn cache cannot mix results across featurization modes: under territory scoring with
  //TaxRule NONE the two modes hash differently (distinct cache entries), and under other rules
  //the mode is a genuine no-op and deliberately shares cache entries.
  {
    Logger logger(nullptr,false,false);
    NNEvaluator* nnEval = TestSearchCommon::startNNEval(
      "",logger,"excludeTerritoryAtariCacheTest",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,
      0,true,false,false,true,false
    );

    MiscNNInputParams noOverride;
    MiscNNInputParams forceOn;
    forceOn.excludeTerritoryAdjAtariOverride = 1;

    auto sameOutputs = [](const NNOutput& a, const NNOutput& b) {
      if(a.whiteWinProb != b.whiteWinProb || a.whiteLossProb != b.whiteLossProb || a.whiteScoreMean != b.whiteScoreMean)
        return false;
      for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
        if(a.policyProbs[i] != b.policyProbs[i])
          return false;
      return true;
    };

    //Territory scoring with TaxRule NONE: the modes must hit distinct cache entries.
    {
      BoardHistory hist(board,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
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

    //Area scoring: the override is a no-op and must share the cache entry.
    {
      Rules areaRules = terrNoTaxRules;
      areaRules.scoringRule = Rules::SCORING_AREA;
      BoardHistory hist(board,P_BLACK,areaRules,0,BoardHistoryModes(false,false));
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

  //A pass-alive white group with both phenomena at once: a dead black throw-in stone in atari
  //(A5, with liberty A4) inside its pass-alive territory, and a ko mouth (D7, the sole liberty
  //of the isolated white ko stone E7). The exclusion applies only to chains of the territory
  //owner's own color, so the throw-in does NOT block A4/A5 from counting for white, while the
  //own-color ko stone in atari DOES block the mouth D7 - the only point the modes disagree on.
  {
    Board tboard = Board::parseBoard(7,7,R"%%(
.oo.ox.
oo.oxx.
xoooox.
.oooox.
oooo.x.
o.ooox.
ooooox.
)%%");
    const Loc throwInLoc = Location::getLoc(0,2,tboard.x_size);
    const Loc libertyLoc = Location::getLoc(0,3,tboard.x_size);
    const Loc mouthLoc = Location::getLoc(3,0,tboard.x_size);
    const Loc koStoneLoc = Location::getLoc(4,0,tboard.x_size);
    testAssert(tboard.colors[throwInLoc] == C_BLACK);
    testAssert(tboard.getNumLiberties(throwInLoc) == 1);
    testAssert(tboard.colors[koStoneLoc] == C_WHITE);
    testAssert(tboard.getNumLiberties(koStoneLoc) == 1);

    Color areaV2[Board::MAX_ARR_SIZE];
    Color areaV3[Board::MAX_ARR_SIZE];
    int count;
    tboard.calculateIndependentLifeArea(areaV2,count,true,false,false,false);
    tboard.calculateIndependentLifeArea(areaV3,count,true,false,true,false);
    //Opposing-color atari does not block the territory...
    testAssert(areaV2[libertyLoc] == C_WHITE);
    testAssert(areaV3[libertyLoc] == C_WHITE);
    testAssert(areaV2[throwInLoc] == C_WHITE);
    testAssert(areaV3[throwInLoc] == C_WHITE);
    //...but the own-color ko stone in atari blocks the ko mouth, and nothing else differs.
    testAssert(areaV2[mouthLoc] == C_WHITE);
    testAssert(areaV3[mouthLoc] == C_EMPTY);
    for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
      testAssert(areaV2[i] == areaV3[i] || i == mouthLoc);

    BoardHistory histV2(tboard,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
    Board tb1 = tboard;
    histV2.endAndScoreGameNow(tb1);
    BoardHistory histV3(tboard,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,true));
    Board tb2 = tboard;
    histV3.endAndScoreGameNow(tb2);
    testAssert(histV2.finalWhiteMinusBlackScore == histV3.finalWhiteMinusBlackScore + 1.0f);
  }

  //NN input features 18/19 (current territory) reflect the flag under territory scoring with
  //TaxRule NONE in encore phase 2: the atari-adjacent eye counts for black only under v2.
  {
    const int nnXLen = 7;
    const int nnYLen = 7;
    const int numSpatial = NNInputs::NUM_FEATURES_SPATIAL_V7;
    std::vector<float> rowBinV2(numSpatial*nnXLen*nnYLen);
    std::vector<float> rowBinV3(numSpatial*nnXLen*nnYLen);
    std::vector<float> rowGlobalV2(NNInputs::NUM_FEATURES_GLOBAL_V7);
    std::vector<float> rowGlobalV3(NNInputs::NUM_FEATURES_GLOBAL_V7);
    MiscNNInputParams nnInputParams;
    BoardHistory histV2(board,P_BLACK,terrNoTaxRules,2,BoardHistoryModes(false,false));
    BoardHistory histV3(board,P_BLACK,terrNoTaxRules,2,BoardHistoryModes(false,true));
    NNInputs::fillRowV7(board,histV2,P_BLACK,nnInputParams,nnXLen,nnYLen,false,rowBinV2.data(),rowGlobalV2.data());
    NNInputs::fillRowV7(board,histV3,P_BLACK,nnInputParams,nnXLen,nnYLen,false,rowBinV3.data(),rowGlobalV3.data());
    int eyePos = NNPos::locToPos(eyeLoc,board.x_size,nnXLen,nnYLen);
    //Feature 18 is the current player's territory; black is to move, so the eye is feature 18 under v2.
    testAssert(rowBinV2[18*nnXLen*nnYLen + eyePos] == 1.0f);
    testAssert(rowBinV3[18*nnXLen*nnYLen + eyePos] == 0.0f);
    //The eye is the sole atari-adjacent counted point on this board, so it is the only difference.
    for(int i = 0; i<numSpatial*nnXLen*nnYLen; i++) {
      if(i != 18*nnXLen*nnYLen + eyePos)
        testAssert(rowBinV2[i] == rowBinV3[i]);
    }
  }

  //Torazu sanmoku ("three points without capturing")
  {
    Board tboard = Board::parseBoard(9,9,R"%%(
.xxo.....
oxxo.....
xooo.....
xxxoooooo
xxxxxxxxx
.........
.........
.........
.........
)%%");
    const Loc sharedLibLoc = Location::getLoc(0,0,tboard.x_size);
    const Loc dangoLoc = Location::getLoc(1,0,tboard.x_size);
    const Loc loneStoneLoc = Location::getLoc(0,1,tboard.x_size);
    testAssert(tboard.colors[sharedLibLoc] == C_EMPTY);
    //Mutual atari: black's four-stone dango and white's lone stone share the 1-1 point.
    testAssert(tboard.colors[dangoLoc] == C_BLACK);
    testAssert(tboard.getNumLiberties(dangoLoc) == 1);
    testAssert(tboard.colors[loneStoneLoc] == C_WHITE);
    testAssert(tboard.getNumLiberties(loneStoneLoc) == 1);

    Color areaV2[Board::MAX_ARR_SIZE];
    Color areaV3[Board::MAX_ARR_SIZE];
    int count;
    tboard.calculateIndependentLifeArea(areaV2,count,true,false,false,false);
    tboard.calculateIndependentLifeArea(areaV3,count,true,false,true,false);
    //The shared liberty is nobody's territory either way, and nothing else differs.
    testAssert(areaV2[sharedLibLoc] == C_EMPTY);
    testAssert(areaV3[sharedLibLoc] == C_EMPTY);
    for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
      testAssert(areaV2[i] == areaV3[i]);

    BoardHistory histV2(tboard,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,false));
    Board tb1 = tboard;
    histV2.endAndScoreGameNow(tb1);
    BoardHistory histV3(tboard,P_BLACK,terrNoTaxRules,0,BoardHistoryModes(false,true));
    Board tb2 = tboard;
    histV3.endAndScoreGameNow(tb2);
    testAssert(histV2.finalWhiteMinusBlackScore == histV3.finalWhiteMinusBlackScore);
  }

  cout << "Ran exclude-territory-adjacent-to-atari mode tests" << endl;
}
