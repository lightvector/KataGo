#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/fancymath.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../core/timer.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../dataio/poswriter.h"
#include "../dataio/files.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <csignal>

using namespace std;

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

static void handleStartAnnotations(Sgf* rootSgf) {
  std::function<bool(Sgf*)> hasStartNode = [&hasStartNode](Sgf* sgf) {
    for(SgfNode* node : sgf->nodes) {
      if(node->hasProperty("C")) {
        std::string comment = node->getSingleProperty("C");
        if(comment.find("%START%") != std::string::npos) {
          return true;
        }
      }
    }
    for(Sgf* child : sgf->children) {
      if(hasStartNode(child)) {
        return true;
      }
    }
    return false;
  };

  std::function<void(Sgf*)> markNodes = [&markNodes](Sgf* sgf) {
    bool isInStartSubtree = false;
    for(SgfNode* node : sgf->nodes) {
      if(node->hasProperty("C")) {
        std::string comment = node->getSingleProperty("C");
        if(comment.find("%START%") != std::string::npos) {
          isInStartSubtree = true;
          break;
        }
      }
      node->appendComment("%NOSAMPLE%");
      node->appendComment("%NOHINT%");
    }
    if(!isInStartSubtree) {
      for(Sgf* child : sgf->children)
        markNodes(child);
    }
  };

  if(hasStartNode(rootSgf)) {
    markNodes(rootSgf);
  }
}

int MainCmds::samplesgfs(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  vector<string> sgfFilesFromCmdline;
  vector<string> sgfDirs;
  vector<string> sgfsDirs;
  string outDir;
  vector<string> excludeHashesFiles;
  double sampleProb;
  double sampleWeight;
  double forceSampleWeight;
  double turnWeightLambda;
  double minWeight;
  int64_t maxDepth;
  int64_t maxNodeCount;
  int64_t maxBranchCount;
  double minTurnNumberBoardAreaProp;
  double maxTurnNumberBoardAreaProp;
  bool flipIfPassOrWFirst;
  double afterPassFactor;
  bool allowGameOver;
  bool hashComments;
  double trainingWeight;
  int verbosity;
  bool tolerateIllegalMoves;

  string valueFluctuationModelFile;
  double valueFluctuationTurnScale;
  double valueFluctuationForwardTurnScale;
  double valueFluctuationMaxWeight;
  bool valueFluctuationMakeKomiFair;
  double valueFluctuationWeightBySurprise;
  double valueFluctuationWeightByCount;
  double valueFluctuationWeightByUncertainty;
  bool debugValueFluctuation;

  int minMinRank;
  int minMinRating;
  string requiredPlayerName;
  int maxHandicap;
  double maxKomi;

  int numThreads;

  bool forTesting;

  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");

    TCLAP::MultiArg<string> sgfArg("","sgf","Sgf file",false,"SGF");
    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::MultiArg<string> sgfsDirArg("","sgfsdir","Directory of sgfs files",false,"DIR");
    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::ValueArg<double> sampleProbArg("","sample-prob","Probability to sample each position",true,0.0,"PROB");
    TCLAP::ValueArg<double> sampleWeightArg("","sample-weight","",false,1.0,"Weight");
    TCLAP::ValueArg<double> forceSampleWeightArg("","force-sample-weight","",false,5.0,"Weight");
    TCLAP::ValueArg<double> turnWeightLambdaArg("","turn-weight-lambda","Adjust weight for writing down each position",true,0.0,"LAMBDA");
    TCLAP::ValueArg<double> minWeightArg("","min-weight","",false,0.0,"Weight");
    TCLAP::ValueArg<string> maxDepthArg("","max-depth","Max depth allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<string> maxNodeCountArg("","max-node-count","Max node count allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<string> maxBranchCountArg("","max-branch-count","Max branch count allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<double> minTurnNumberBoardAreaPropArg("","min-turn-number-board-area-prop","Only use turn number >= this board area",false,-1.0,"PROP");
    TCLAP::ValueArg<double> maxTurnNumberBoardAreaPropArg("","max-turn-number-board-area-prop","Only use turn number <= this board area",false,10000.0,"PROP");
    TCLAP::SwitchArg flipIfPassOrWFirstArg("","flip-if-pass","Try to heuristically find cases where an sgf passes to simulate white<->black");
    TCLAP::ValueArg<double> afterPassFactorArg("","after-pass-factor","Scale down weight of positions following a pass",false, 1.0, "FACTOR");
    TCLAP::SwitchArg allowGameOverArg("","allow-game-over","Allow sampling game over positions in sgf");
    TCLAP::SwitchArg hashCommentsArg("","hash-comments","Hash comments in sgf");
    TCLAP::ValueArg<double> trainingWeightArg("","training-weight","Scale the loss function weight from data from games that originate from this position",false,1.0,"WEIGHT");
    TCLAP::ValueArg<int> verbosityArg("","verbosity","Print more stuff",false,0,"INT");
    TCLAP::SwitchArg tolerateIllegalMovesArg("","tolerate-illegal-moves","Tolerate illegal moves");

    TCLAP::ValueArg<string> valueFluctuationModelFileArg("","value-fluctuation-model","Upweight positions prior to value fluctuations",false,string(),"MODELFILE");
    TCLAP::ValueArg<double> valueFluctuationTurnScaleArg("","value-fluctuation-turn-scale","How much prior on average",false,1.0,"AVGTURNS");
    TCLAP::ValueArg<double> valueFluctuationForwardTurnScaleArg("","value-fluctuation-forward-turn-scale","How much prior on average",false,1.0,"AVGTURNS");
    TCLAP::ValueArg<double> valueFluctuationMaxWeightArg("","value-fluctuation-max-weight","",false,10.0,"MAXWEIGHT");
    TCLAP::SwitchArg valueFluctuationMakeKomiFairArg("","value-fluctuation-make-komi-fair","");
    TCLAP::ValueArg<double> valueFluctuationWeightBySurpriseArg("","value-fluctuation-weight-by-surprise","",false,0.0,"SCALE");
    TCLAP::ValueArg<double> valueFluctuationWeightByCountArg("","value-fluctuation-weight-by-count","",false,1.0,"SCALE");
    TCLAP::ValueArg<double> valueFluctuationWeightByUncertaintyArg("","value-fluctuation-weight-by-uncertainty","",false,0.0,"SCALE");
    TCLAP::SwitchArg debugValueFluctuationArg("","debug-value-fluctuation","");
    TCLAP::ValueArg<int> minMinRankArg("","min-min-rank","Require both players in a game to have rank at least this",false,Sgf::RANK_UNKNOWN,"INT");
    TCLAP::ValueArg<int> minMinRatingArg("","min-min-rating","Require both players in a game to have rating at least this",false,-1000000000,"INT");
    TCLAP::ValueArg<string> requiredPlayerNameArg("","required-player-name","Require player making the move to have this name",false,string(),"NAME");
    TCLAP::ValueArg<int> maxHandicapArg("","max-handicap","Require no more than this big handicap in stones",false,100,"INT");
    TCLAP::ValueArg<double> maxKomiArg("","max-komi","Require absolute value of game komi to be at most this",false,1000,"KOMI");

    TCLAP::ValueArg<int> numThreadsArg("","num-threads","Number of threads to process",false,1,"INT");

    TCLAP::SwitchArg forTestingArg("","for-testing","For testing");

    cmd.add(sgfArg);
    cmd.add(sgfDirArg);
    cmd.add(sgfsDirArg);
    cmd.add(outDirArg);
    cmd.add(excludeHashesArg);
    cmd.add(sampleProbArg);
    cmd.add(sampleWeightArg);
    cmd.add(forceSampleWeightArg);
    cmd.add(turnWeightLambdaArg);
    cmd.add(minWeightArg);
    cmd.add(maxDepthArg);
    cmd.add(maxNodeCountArg);
    cmd.add(maxBranchCountArg);
    cmd.add(minTurnNumberBoardAreaPropArg);
    cmd.add(maxTurnNumberBoardAreaPropArg);
    cmd.add(flipIfPassOrWFirstArg);
    cmd.add(afterPassFactorArg);
    cmd.add(allowGameOverArg);
    cmd.add(hashCommentsArg);
    cmd.add(trainingWeightArg);
    cmd.add(verbosityArg);
    cmd.add(tolerateIllegalMovesArg);
    cmd.add(valueFluctuationModelFileArg);
    cmd.add(valueFluctuationTurnScaleArg);
    cmd.add(valueFluctuationForwardTurnScaleArg);
    cmd.add(valueFluctuationMaxWeightArg);
    cmd.add(valueFluctuationMakeKomiFairArg);
    cmd.add(valueFluctuationWeightBySurpriseArg);
    cmd.add(valueFluctuationWeightByCountArg);
    cmd.add(valueFluctuationWeightByUncertaintyArg);
    cmd.add(debugValueFluctuationArg);
    cmd.add(minMinRankArg);
    cmd.add(minMinRatingArg);
    cmd.add(requiredPlayerNameArg);
    cmd.add(maxHandicapArg);
    cmd.add(maxKomiArg);
    cmd.add(numThreadsArg);
    cmd.add(forTestingArg);
    cmd.parseArgs(args);
    sgfFilesFromCmdline = sgfArg.getValue();
    sgfDirs = sgfDirArg.getValue();
    sgfsDirs = sgfsDirArg.getValue();
    outDir = outDirArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    sampleProb = sampleProbArg.getValue();
    sampleWeight = sampleWeightArg.getValue();
    minWeight = minWeightArg.getValue();
    forceSampleWeight = forceSampleWeightArg.getValue();
    turnWeightLambda = turnWeightLambdaArg.getValue();
    maxDepth = Global::stringToInt64(maxDepthArg.getValue());
    maxNodeCount = Global::stringToInt64(maxNodeCountArg.getValue());
    maxBranchCount = Global::stringToInt64(maxBranchCountArg.getValue());
    minTurnNumberBoardAreaProp = minTurnNumberBoardAreaPropArg.getValue();
    maxTurnNumberBoardAreaProp = maxTurnNumberBoardAreaPropArg.getValue();
    flipIfPassOrWFirst = flipIfPassOrWFirstArg.getValue();
    afterPassFactor = afterPassFactorArg.getValue();
    allowGameOver = allowGameOverArg.getValue();
    hashComments = hashCommentsArg.getValue();
    trainingWeight = trainingWeightArg.getValue();
    verbosity = verbosityArg.getValue();
    tolerateIllegalMoves = tolerateIllegalMovesArg.getValue();
    valueFluctuationModelFile = valueFluctuationModelFileArg.getValue();
    valueFluctuationTurnScale = valueFluctuationTurnScaleArg.getValue();
    valueFluctuationForwardTurnScale = valueFluctuationForwardTurnScaleArg.getValue();
    valueFluctuationMaxWeight = valueFluctuationMaxWeightArg.getValue();
    valueFluctuationMakeKomiFair = valueFluctuationMakeKomiFairArg.getValue();
    valueFluctuationWeightBySurprise = valueFluctuationWeightBySurpriseArg.getValue();
    valueFluctuationWeightByCount = valueFluctuationWeightByCountArg.getValue();
    valueFluctuationWeightByUncertainty = valueFluctuationWeightByUncertaintyArg.getValue();
    debugValueFluctuation = debugValueFluctuationArg.getValue();
    minMinRank = minMinRankArg.getValue();
    minMinRating = minMinRatingArg.getValue();
    requiredPlayerName = requiredPlayerNameArg.getValue();
    maxHandicap = maxHandicapArg.getValue();
    maxKomi = maxKomiArg.getValue();
    numThreads = numThreadsArg.getValue();
    forTesting = forTestingArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outDir);

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTimeStamp = !forTesting;
  Logger logger(nullptr, logToStdout, logToStderr, logTimeStamp);
  logger.addFile(outDir + "/" + "log.log");
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);
  for(const string& s: sgfFilesFromCmdline)
    sgfFiles.push_back(s);

  logger.write("Found " + Global::int64ToString((int64_t)sgfFiles.size()) + " sgf files!");

  vector<string> sgfsFiles;
  FileHelpers::collectMultiSgfsFromDirsOrFiles(sgfsDirs,sgfsFiles);
  logger.write("Found " + Global::int64ToString((int64_t)sgfsFiles.size()) + " sgfs files!");

  if(forTesting) {
    std::sort(sgfFiles.begin(),sgfFiles.end());
    std::sort(sgfsFiles.begin(),sgfsFiles.end());
  }

  set<Hash128> excludeHashes = Sgf::readExcludes(excludeHashesFiles);
  logger.write("Loaded " + Global::uint64ToString(excludeHashes.size()) + " excludes");

  NNEvaluator* valueFluctuationNNEval = NULL;
  if(valueFluctuationModelFile != "") {
    if(valueFluctuationTurnScale < 1.0 || valueFluctuationTurnScale > 100000000.0)
      throw StringError("Invalid valueFluctuationTurnScale");
    if(valueFluctuationForwardTurnScale < 1.0 || valueFluctuationForwardTurnScale > 100000000.0)
      throw StringError("Invalid valueFluctuationForwardTurnScale");
    if(valueFluctuationMaxWeight <= 0.0 || valueFluctuationMaxWeight > 100000000.0)
      throw StringError("Invalid valueFluctuationMaxWeight");
    ConfigParser cfg;
    if(forTesting)
      cfg.overrideKey("nnRandSeed","forTesting");

    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = numThreads;
    const int defaultMaxBatchSize = std::max(8,((numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    valueFluctuationNNEval = Setup::initializeNNEvaluator(
      valueFluctuationModelFile,valueFluctuationModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
    logger.write("Loaded neural net");
  }

  // ---------------------------------------------------------------------------------------------------

  auto isPlayerOkay = [&](const Sgf* sgf, Player pla) {
    if(requiredPlayerName != "") {
      if(sgf->getPlayerName(pla) != requiredPlayerName)
        return false;
    }
    return true;
  };

  auto isSgfOkay = [&](const Sgf* sgf, string& reasonBuf) {
    if(maxHandicap < 100 && sgf->getHandicapValue() > maxHandicap)
    {reasonBuf = "handicap"; return false;}
    if(sgf->depth() > maxDepth)
    {reasonBuf = "depth" + Global::intToString(sgf->depth()); return false;}
    if(std::fabs(sgf->getKomiOrDefault(7.5f)) > maxKomi)
    {reasonBuf = "komi"; return false;}
    if(minMinRank != Sgf::RANK_UNKNOWN) {
      if(sgf->getRank(P_BLACK) < minMinRank || sgf->getRank(P_WHITE) < minMinRank)
      {reasonBuf = "rank"; return false;}
    }
    if(minMinRating > -10000000) {
      if(sgf->getRating(P_BLACK) < minMinRating || sgf->getRating(P_WHITE) < minMinRating)
      {reasonBuf = "rating"; return false;}
    }
    if(!isPlayerOkay(sgf,P_BLACK) && !isPlayerOkay(sgf,P_WHITE))
    {reasonBuf = "player " + sgf->getPlayerName(P_BLACK) + " " + sgf->getPlayerName(P_WHITE); return false;}
    return true;
  };

  // ---------------------------------------------------------------------------------------------------
  std::mutex mutex;

  PosWriter posWriter("startposes.txt", outDir, 1, 0, 100000);
  posWriter.start();

  // ---------------------------------------------------------------------------------------------------

  int64_t numKept = 0;
  double weightKept = 0;
  std::set<Hash128> uniqueHashes;
  std::function<void(Sgf::PositionSample&, const BoardHistory&, const string&)> posHandler =
    [sampleProb,sampleWeight,forceSampleWeight,&posWriter,turnWeightLambda,&numKept,&weightKept,&seedRand,minTurnNumberBoardAreaProp,maxTurnNumberBoardAreaProp,afterPassFactor,trainingWeight,minWeight](
      Sgf::PositionSample& posSample, const BoardHistory& hist, const string& comments
    ) {
      assert(posSample.getCurrentTurnNumber() == hist.getCurrentTurnNumber());
      double minTurnNumber = minTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      double maxTurnNumber = maxTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      if(posSample.getCurrentTurnNumber() < minTurnNumber || posSample.getCurrentTurnNumber() > maxTurnNumber)
        return;
      if(comments.size() > 0 && comments.find("%NOSAMPLE%") != string::npos)
        return;

      if(seedRand.nextBool(sampleProb)) {
        Sgf::PositionSample posSampleToWrite = posSample;
        int64_t startTurn = posSampleToWrite.getCurrentTurnNumber();
        posSampleToWrite.weight = sampleWeight * exp(-startTurn * turnWeightLambda) * posSampleToWrite.weight;
        if(posSampleToWrite.moves.size() > 0 && posSampleToWrite.moves[posSampleToWrite.moves.size()-1].loc == Board::PASS_LOC)
          posSampleToWrite.weight *= afterPassFactor;
        if(comments.size() > 0 && comments.find("%SAMPLE%") != string::npos)
          posSampleToWrite.weight = std::max(posSampleToWrite.weight,forceSampleWeight);
        if(comments.size() > 0 && comments.find("%SAMPLELIGHT%") != string::npos)
          posSampleToWrite.weight = std::max(posSampleToWrite.weight,0.5*forceSampleWeight);
        if(posSampleToWrite.weight < minWeight)
          return;
        posSampleToWrite.trainingWeight = trainingWeight;
        posWriter.writePos(posSampleToWrite);
        numKept += 1;
        weightKept += posSampleToWrite.weight;
      }
    };

  std::map<string,int64_t> sgfCountUsedByPlayerName;
  std::map<string,int64_t> sgfCountUsedByResult;

  double totalWeightFromCount = 0.0;
  double totalWeightFromSurprise = 0.0;
  double totalWeightFromUncertainty = 0.0;
  int64_t numExcluded = 0;
  int64_t numSgfsFilteredTopLevel = 0;
  auto trySgf = [&](Sgf* sgf) {
    std::unique_lock<std::mutex> lock(mutex);

    if(contains(excludeHashes,sgf->hash)) {
      numExcluded += 1;
      return;
    }

    int64_t depth = sgf->depth();
    int64_t nodeCount = sgf->nodeCount();
    int64_t branchCount = sgf->branchCount();
    if(depth > maxDepth || nodeCount > maxNodeCount || branchCount > maxBranchCount) {
      logger.write(
        "Skipping due to violating limits depth " + Global::int64ToString(depth) +
        " nodes " + Global::int64ToString(nodeCount) +
        " branches " + Global::int64ToString(branchCount) +
        " " + sgf->fileName
      );
      numSgfsFilteredTopLevel += 1;
      return;
    }

    try {
      string reasonBuf;
      if(!isSgfOkay(sgf,reasonBuf)) {
        if(verbosity >= 2)
          logger.write("Filtering due to not okay (" + reasonBuf + "): " + sgf->fileName);
        numSgfsFilteredTopLevel += 1;
        return;
      }
    }
    catch(const StringError& e) {
      logger.write("Filtering due to error checking okay: " + sgf->fileName + ": " + e.what());
      numSgfsFilteredTopLevel += 1;
      return;
    }

    handleStartAnnotations(sgf);

    if(valueFluctuationNNEval == NULL) {
      bool hashParent = false;
      Rand iterRand;
      sgf->iterAllUniquePositions(uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, forTesting ? NULL : &iterRand, posHandler);
      if(verbosity >= 2)
        logger.write("Handled " + sgf->fileName + " kept weight " + Global::doubleToString(weightKept));
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_BLACK)] += 1;
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_WHITE)] += 1;
      sgfCountUsedByResult[sgf->getRootPropertyWithDefault("RE","")] += 1;
    }
    else {
      string fileName = sgf->fileName;
      CompactSgf compactSgf(sgf);
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules rules = compactSgf.getRulesOrFailAllowUnspecified(Rules::getSimpleTerritory());
      compactSgf.setupInitialBoardAndHist(rules, board, nextPla, hist);

      if(valueFluctuationMakeKomiFair) {
        Rand rand;
        string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
        SearchParams params = SearchParams::basicDecentParams();
        Search* search = new Search(params,valueFluctuationNNEval,&logger,searchRandSeed);
        OtherGameProperties otherGameProps;
        int64_t numVisits = 30;
        lock.unlock();
        PlayUtils::adjustKomiToEven(search, search, board, hist, nextPla, numVisits, otherGameProps, rand);
        lock.lock();
      }

      const bool preventEncore = false;
      const vector<Move>& sgfMoves = compactSgf.moves;

      vector<Board> boards;
      vector<BoardHistory> hists;
      vector<Player> nextPlas;
      vector<shared_ptr<NNOutput>> nnOutputs;
      vector<double> winLossValues;
      vector<Move> moves;

      for(size_t m = 0; m<sgfMoves.size()+1; m++) {
        MiscNNInputParams nnInputParams;
        nnInputParams.conservativePassAndIsRoot = true;
        if(forTesting)
          nnInputParams.symmetry = 0;
        NNResultBuf buf;
        bool skipCache = true;
        bool includeOwnerMap = false;
        lock.unlock();
        valueFluctuationNNEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);
        lock.lock();

        boards.push_back(board);
        hists.push_back(hist);
        nextPlas.push_back(nextPla);
        nnOutputs.push_back(std::move(buf.result));
        shared_ptr<NNOutput>& nnOutput = nnOutputs[nnOutputs.size()-1];
        winLossValues.push_back(nnOutput->whiteWinProb - nnOutput->whiteLossProb);

        if(m >= sgfMoves.size())
          break;

        moves.push_back(sgfMoves[m]);

        //Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
        if(hist.isGameFinished || hist.encorePhase > 0)
          break;
        //Quit out if consecutive moves by the same player, to keep the history clean and "normal"
        if(sgfMoves[m].pla != nextPla && m > 0) {
          logger.write("Ending SGF " + fileName + " early due to non-alternating players on turn " + Global::intToString(m));
          break;
        }

        bool suc = hist.isLegal(board,sgfMoves[m].loc,sgfMoves[m].pla);
        if(!suc) {
          //Only log on errors that aren't simply due to ko rules, but quit out regardless
          suc = hist.makeBoardMoveTolerant(board,sgfMoves[m].loc,sgfMoves[m].pla,preventEncore);
          if(!suc)
            logger.write("Illegal move in " + fileName + " turn " + Global::intToString(m) + " move " + Location::toString(sgfMoves[m].loc, board.x_size, board.y_size));
          break;
        }
        hist.makeBoardMoveAssumeLegal(board,sgfMoves[m].loc,sgfMoves[m].pla,NULL,preventEncore);
        nextPla = getOpp(sgfMoves[m].pla);
      }
      boards.push_back(board);
      hists.push_back(hist);
      nextPlas.push_back(nextPla);

      if(winLossValues.size() <= 1)
        return;

      double minTurnNumber = minTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      double maxTurnNumber = maxTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      //At this point we transition from indexing by move index alone to indexing by turn number in case the board
      //started in the middle of a nonempty position.
      double totalSurprise = 0.0;
      double totalUncertainty = 0.0;
      vector<double> winrateVariance;
      for(size_t i = 0; i<winLossValues.size()-1; i++) {
        int64_t turnNumber = hists[i].getCurrentTurnNumber();
        if(winrateVariance.size() <= turnNumber)
          winrateVariance.resize(turnNumber+1);
        if(turnNumber >= minTurnNumber && turnNumber <= maxTurnNumber) {
          winrateVariance[turnNumber] = (winLossValues[i+1]-winLossValues[i]) * (winLossValues[i+1]-winLossValues[i]);
          totalSurprise += FancyMath::binaryCrossEntropy(0.5 + 0.5 * winLossValues[i], 0.5 + 0.5 * winLossValues[winLossValues.size()-1], 0.001);
          totalUncertainty += std::max(0.0, 1.0 - winLossValues[i] * winLossValues[i]);
        }
        else
          winrateVariance[turnNumber] = 0.0;
      }

      //Apply exponential blur
      vector<double> winrateVarianceBlurred(winrateVariance.size());
      {
        double blurSum = 0.0;
        for(size_t i = winrateVariance.size(); i--;) {
          blurSum *= 1.0 - 1.0 / valueFluctuationTurnScale;
          blurSum += winrateVariance[i];
          winrateVarianceBlurred[i] = blurSum / valueFluctuationTurnScale; // Normalize so blur only sums to 1
        }
      }

      //Apply exponential forward blur
      if(valueFluctuationForwardTurnScale > 1.0) {
        vector<double> winrateVarianceMoreBlurred(winrateVariance.size());
        double blurSum = 0.0;
        for(size_t i = 0; i < winrateVariance.size(); i++) {
          blurSum *= 1.0 - 1.0 / valueFluctuationForwardTurnScale;
          blurSum += winrateVarianceBlurred[i];
          winrateVarianceMoreBlurred[i] = blurSum / valueFluctuationForwardTurnScale; // Normalize so blur only sums to 1
        }
        winrateVarianceBlurred = winrateVarianceMoreBlurred;
      }

      double totalWeight = 0.0;
      int totalCount = 0;
      for(size_t i = 0; i < winrateVariance.size(); i++) {
        if(i >= minTurnNumber && i <= maxTurnNumber) {
          // Rescale so the blur has total weight valueFluctuationTurnScale + valueFluctuationForwardTurnScale
          winrateVarianceBlurred[i] *= valueFluctuationTurnScale + valueFluctuationForwardTurnScale;
          totalWeight += winrateVarianceBlurred[i];
          totalCount += 1;
        }
        else {
          winrateVarianceBlurred[i] = 0;
        }
      }

      if(totalCount <= 0 || totalWeight <= 0)
        return;

      //Normalize
      double desiredTotalWeight = (
        valueFluctuationWeightByCount * totalCount +
        valueFluctuationWeightBySurprise * totalSurprise +
        valueFluctuationWeightByUncertainty * totalUncertainty
      );
      totalWeightFromCount += valueFluctuationWeightByCount * totalCount;
      totalWeightFromSurprise += valueFluctuationWeightBySurprise * totalSurprise;
      totalWeightFromUncertainty += valueFluctuationWeightByUncertainty * totalUncertainty;
      vector<double> desiredWeight(winrateVariance.size());
      for(size_t i = 0; i<desiredWeight.size(); i++) {
        desiredWeight[i] = std::min(valueFluctuationMaxWeight, winrateVarianceBlurred[i] / totalWeight * desiredTotalWeight);
      }

      if(debugValueFluctuation) {
        ostringstream out;
        std::vector<string> extraComments;
        for(size_t i = 0; i<winLossValues.size(); i++) {
          int64_t turnIdx = std::min((int64_t)(desiredWeight.size()-1), hists[i].getCurrentTurnNumber());
          turnIdx = std::max(turnIdx,(int64_t)0);
          extraComments.push_back(Global::strprintf("%.3f %.3f",winLossValues[i],desiredWeight[turnIdx]));
        }
        WriteSgf::writeSgf(
          out, "B", "W",
          hists[hists.size()-1],
          extraComments
        );
        logger.write(out.str());
      }

      for(int64_t m = 0; m<(int64_t)moves.size(); m++) {
        Sgf::PositionSample sample;
        const int numMovesToRecord = 8;
        int64_t startIdx = std::max((int64_t)0,m-numMovesToRecord);
        sample.board = boards[startIdx];
        sample.nextPla = nextPlas[startIdx];
        for(int64_t j = startIdx; j<m; j++)
          sample.moves.push_back(moves[j]);
        sample.initialTurnNumber = hist.initialTurnNumber + startIdx;
        sample.hintLoc = Board::NULL_LOC;

        assert(desiredWeight.size() > 0);
        int64_t turnIdx = std::min((int64_t)(desiredWeight.size()-1), hists[m].getCurrentTurnNumber());
        turnIdx = std::max(turnIdx,(int64_t)0);
        sample.weight = desiredWeight[turnIdx];

        sample.trainingWeight = trainingWeight;

        if(sample.weight < 0.1)
          continue;

        posHandler(sample, hists[m], "");
        // cout << fileName << " " << m << " " << desiredWeight[m] << endl;
      }

      // Block all the main line hashes and then iterate through the whole SGF to catch side variations, weight them
      // the same way as the same turn in the main line.
      std::set<Hash128> blockedSituationHashes;
      for(size_t m = 0; m<hists.size(); m++) {
        blockedSituationHashes.insert(
          BoardHistory::getSituationAndSimpleKoAndPrevPosHash(hists[m].getRecentBoard(0),hists[m],hists[m].presumedNextMovePla)
        );
      }

      std::function<void(Sgf::PositionSample&, const BoardHistory&, const string&)> posHandler2 =
        [&blockedSituationHashes, &desiredWeight, &posHandler, trainingWeight](
          Sgf::PositionSample& posSample, const BoardHistory& posHist, const string& comments
        ) {
          assert(posSample.getCurrentTurnNumber() == posHist.getCurrentTurnNumber());
          // cout << "AAAA " << (posHist.initialTurnNumber + (int)posHist.moveHistory.size()) << endl;
          if(contains(
               blockedSituationHashes,
               BoardHistory::getSituationAndSimpleKoAndPrevPosHash(posHist.getRecentBoard(0),posHist,posHist.presumedNextMovePla)
             ))
            return;
          // cout << "BBBB" << endl;
          Sgf::PositionSample posSampleWeighted = posSample;
          if(desiredWeight.size() > 0) {
            int64_t turnIdx = std::min((int64_t)(desiredWeight.size()-1), posHist.getCurrentTurnNumber());
            turnIdx = std::max(turnIdx,(int64_t)0);
            posSampleWeighted.weight = desiredWeight[turnIdx];
          }
          posSampleWeighted.trainingWeight = trainingWeight;
          posHandler(posSampleWeighted, posHist, comments);
        };

      bool hashParent = false;
      Rand iterRand;
      sgf->iterAllUniquePositions(uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, forTesting ? NULL : &iterRand, posHandler2);

      if(verbosity >= 2)
        cout << "Handled " << fileName << " kept weight " << weightKept << endl;
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_BLACK)] += 1;
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_WHITE)] += 1;
      sgfCountUsedByResult[sgf->getRootPropertyWithDefault("RE","")] += 1;
    }
  };

  {
    auto processSgfFile = [&](int threadIdx, size_t index) {
      (void)threadIdx;
      Sgf* sgf = NULL;
      try {
        sgf = Sgf::loadFile(sgfFiles[index]);
        trySgf(sgf);
      }
      catch(const StringError& e) {
        if(tolerateIllegalMoves)
          logger.write("Invalid SGF " + sgfFiles[index] + ": " + e.what());
        else
          throw;
      }
      if(sgf != NULL) {
        delete sgf;
      }
    };
    Parallel::iterRange(
      numThreads,
      sgfFiles.size(),
      logger,
      std::function<void(int,size_t)>(processSgfFile)
    );
  };

  for(size_t i = 0; i<sgfsFiles.size(); i++) {
    std::vector<Sgf*> sgfs;
    try {
      sgfs = Sgf::loadSgfsFile(sgfsFiles[i]);
    }
    catch(const StringError& e) {
      if(tolerateIllegalMoves)
        logger.write("Invalid SGFS " + sgfsFiles[i] + ": " + e.what());
      else
        throw;
      continue;
    }

    auto processSgf = [&](int threadIdx, size_t index) {
      (void)threadIdx;
      try {
        trySgf(sgfs[index]);
      }
      catch(const StringError& e) {
        if(tolerateIllegalMoves)
          logger.write("Bad sgf in SGFS" + sgfsFiles[index] + ": " + e.what());
        else
          throw;
      }
    };
    Parallel::iterRange(
      numThreads,
      sgfs.size(),
      logger,
      std::function<void(int,size_t)>(processSgf)
    );
    for(size_t j = 0; j<sgfs.size(); j++) {
      delete sgfs[j];
    }
  }

  logger.write("Kept " + Global::int64ToString(numKept) + " start positions");
  logger.write("Excluded " + Global::int64ToString(numExcluded) + "/" + Global::uint64ToString(sgfFiles.size()) + " sgf files");
  logger.write("Filtered " + Global::int64ToString(numSgfsFilteredTopLevel) + "/" + Global::uint64ToString(sgfFiles.size()) + " sgf files");
  if(valueFluctuationNNEval != NULL) {
    logger.write("totalWeightFromCount " + Global::doubleToString(totalWeightFromCount));
    logger.write("totalWeightFromSurprise " + Global::doubleToString(totalWeightFromSurprise));
    logger.write("totalWeightFromUncertainty " + Global::doubleToString(totalWeightFromUncertainty));
  }

  if(verbosity >= 1) {
    logger.write("SGF count used by player name:");
    for(const auto &elt: sgfCountUsedByPlayerName) {
      logger.write(elt.first + " " + Global::int64ToString(elt.second));
    }
    logger.write("SGF count used by result:");
    for(const auto &elt: sgfCountUsedByResult) {
      logger.write(elt.first + " " + Global::int64ToString(elt.second));
    }
  }

  // ---------------------------------------------------------------------------------------------------

  posWriter.flushAndStop();

  if(valueFluctuationNNEval != NULL)
    delete valueFluctuationNNEval;

  logger.write("All done");

  ScoreValue::freeTables();
  return 0;
}

static bool maybeGetValuesAfterMove(
  Search* search, Loc moveLoc,
  Player nextPla, const Board& board, const BoardHistory& hist,
  double quickSearchFactor,
  ReportedSearchValues& values
) {
  Board newBoard = board;
  BoardHistory newHist = hist;
  Player newNextPla = nextPla;

  if(moveLoc != Board::NULL_LOC) {
    if(!hist.isLegal(newBoard,moveLoc,newNextPla))
      return false;
    newHist.makeBoardMoveAssumeLegal(newBoard,moveLoc,newNextPla,NULL);
    newNextPla = getOpp(newNextPla);
  }

  search->setPosition(newNextPla,newBoard,newHist);

  if(quickSearchFactor != 1.0) {
    SearchParams oldSearchParams = search->searchParams;
    SearchParams newSearchParams = oldSearchParams;
    newSearchParams.maxVisits = 1 + (int64_t)(oldSearchParams.maxVisits * quickSearchFactor);
    newSearchParams.maxPlayouts = 1 + (int64_t)(oldSearchParams.maxPlayouts * quickSearchFactor);
    search->setParamsNoClearing(newSearchParams);
    search->runWholeSearch(newNextPla,shouldStop);
    search->setParamsNoClearing(oldSearchParams);
  }
  else {
    search->runWholeSearch(newNextPla,shouldStop);
  }

  if(shouldStop.load(std::memory_order_acquire))
    return false;
  values = search->getRootValuesRequireSuccess();
  return true;
}



//We want surprising moves that turned out not poorly
//The more surprising, the more we will weight it
static double surpriseWeight(double policyProb, Rand& rand, bool alwaysAddWeight) {
  if(policyProb < 0)
    return 0;
  double weight = 0.12 / (policyProb + 0.02) - 0.5;
  if(alwaysAddWeight && weight < 1.0)
    weight = 1.0;

  if(weight <= 0)
    return 0;
  if(weight < 0.2) {
    if(rand.nextDouble() * 0.2 >= weight)
      return 0;
    return 0.2;
  }
  return weight;
}

struct PosQueueEntry {
  BoardHistory* hist;
  bool markedAsHintPos;
  bool markedAsHintPosLight;
};

int MainCmds::dataminesgfs(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfFilesFromCmdline;
  vector<string> sgfDirs;
  vector<string> sgfsDirs;
  string outDir;
  int numProcessThreads;
  vector<string> excludeHashesFiles;
  bool gameMode;
  bool treeMode;
  bool surpriseMode;
  bool autoKomi;
  bool tolerateIllegalMoves;
  int sgfSplitCount;
  int sgfSplitIdx;
  int64_t maxDepth;
  double turnWeightLambda;
  int maxPosesPerOutFile;
  double utilityThreshold;
  double gameModeFastThreshold;
  bool flipIfPassOrWFirst;
  bool allowGameOver;
  bool manualHintOnly;
  double trainingWeight;

  int minTurn;
  int minRank;
  int minMinRank;
  string requiredPlayerName;
  int maxHandicap;
  double maxKomi;
  double maxAutoKomi;
  double maxPolicy;
  double minHintWeight;
  double hintScale;
  double moreUtilityWeight;
  int startPosesBeforeHintsLen;

  bool forTesting;

  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> sgfArg("","sgf","Sgf file",false,"SGF");
    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::MultiArg<string> sgfsDirArg("","sgfsdir","Directory of sgfs files",false,"DIR");
    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::ValueArg<int> numProcessThreadsArg("","threads","Number of threads",true,1,"THREADS");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::SwitchArg gameModeArg("","game-mode","Game mode");
    TCLAP::SwitchArg treeModeArg("","tree-mode","Tree mode");
    TCLAP::SwitchArg surpriseModeArg("","surprise-mode","Surprise mode");
    TCLAP::SwitchArg autoKomiArg("","auto-komi","Auto komi");
    TCLAP::SwitchArg tolerateIllegalMovesArg("","tolerate-illegal-moves","Tolerate illegal moves");
    TCLAP::ValueArg<int> sgfSplitCountArg("","sgf-split-count","Number of splits",false,1,"N");
    TCLAP::ValueArg<int> sgfSplitIdxArg("","sgf-split-idx","Which split",false,0,"IDX");
    TCLAP::ValueArg<int> maxDepthArg("","max-depth","Max depth allowed for sgf",false,1000000,"INT");
    TCLAP::ValueArg<double> turnWeightLambdaArg("","turn-weight-lambda","Adjust weight for writing down each position",false,0.0,"LAMBDA");
    TCLAP::ValueArg<int> maxPosesPerOutFileArg("","max-poses-per-out-file","Number of hintposes per output file",false,100000,"INT");
    TCLAP::ValueArg<double> utilityThresholdArg("","utility-threshold","Utility threshold for expensive pass",false,0.01,"UTILS");
    TCLAP::ValueArg<double> gameModeFastThresholdArg("","game-mode-fast-threshold","Utility threshold for game mode fast pass",false,0.005,"UTILS");
    TCLAP::SwitchArg flipIfPassOrWFirstArg("","flip-if-pass","Try to heuristically find cases where an sgf passes to simulate white<->black");
    TCLAP::SwitchArg allowGameOverArg("","allow-game-over","Allow sampling game over positions in sgf");
    TCLAP::SwitchArg manualHintOnlyArg("","manual-hint-only","Allow only positions marked for hint in the sgf");
    TCLAP::ValueArg<double> trainingWeightArg("","training-weight","Scale the loss function weight from data from games that originate from this position",false,1.0,"WEIGHT");
    TCLAP::ValueArg<int> minTurnArg("","min-turn","Only get hints for the given turn or later",false,0,"TURNIDX");
    TCLAP::ValueArg<int> minRankArg("","min-rank","Require player making the move to have rank at least this",false,Sgf::RANK_UNKNOWN,"INT");
    TCLAP::ValueArg<int> minMinRankArg("","min-min-rank","Require both players in a game to have rank at least this",false,Sgf::RANK_UNKNOWN,"INT");
    TCLAP::ValueArg<string> requiredPlayerNameArg("","required-player-name","Require player making the move to have this name",false,string(),"NAME");
    TCLAP::ValueArg<int> maxHandicapArg("","max-handicap","Require no more than this big handicap in stones",false,100,"INT");
    TCLAP::ValueArg<double> maxKomiArg("","max-komi","Require absolute value of game komi to be at most this",false,1000,"KOMI");
    TCLAP::ValueArg<double> maxAutoKomiArg("","max-auto-komi","If absolute value of auto komi would exceed this, skip position",false,1000,"KOMI");
    TCLAP::ValueArg<double> maxPolicyArg("","max-policy","Chop off moves with raw policy more than this",false,1000,"POLICY");
    TCLAP::ValueArg<double> minHintWeightArg("","min-hint-weight","Hinted moves get at least this weight",false,0.0,"WEIGHT");
    TCLAP::ValueArg<double> hintScaleArg("","hint-scale","Manually hinted moves get weight scaled by this",false,1.0,"FACTOR");
    TCLAP::ValueArg<double> moreUtilityWeightArg("","more-utility-weight","Increase weight when hint move is uniquely better than other moves",false,0.0,"WEIGHTSCALE");
    TCLAP::ValueArg<int> startPosesBeforeHintsLenArg("","start-poses-before-hints","Add weight for startposes before hints in game mode",false,0,"NMOVES");

    TCLAP::SwitchArg forTestingArg("","for-testing","For testing");

    cmd.add(sgfArg);
    cmd.add(sgfDirArg);
    cmd.add(sgfsDirArg);
    cmd.add(outDirArg);
    cmd.add(numProcessThreadsArg);
    cmd.add(excludeHashesArg);
    cmd.add(gameModeArg);
    cmd.add(treeModeArg);
    cmd.add(surpriseModeArg);
    cmd.add(autoKomiArg);
    cmd.add(tolerateIllegalMovesArg);
    cmd.add(sgfSplitCountArg);
    cmd.add(sgfSplitIdxArg);
    cmd.add(maxDepthArg);
    cmd.add(turnWeightLambdaArg);
    cmd.add(maxPosesPerOutFileArg);
    cmd.add(utilityThresholdArg);
    cmd.add(gameModeFastThresholdArg);
    cmd.add(flipIfPassOrWFirstArg);
    cmd.add(allowGameOverArg);
    cmd.add(manualHintOnlyArg);
    cmd.add(trainingWeightArg);
    cmd.add(minTurnArg);
    cmd.add(minRankArg);
    cmd.add(minMinRankArg);
    cmd.add(requiredPlayerNameArg);
    cmd.add(maxHandicapArg);
    cmd.add(maxKomiArg);
    cmd.add(maxAutoKomiArg);
    cmd.add(maxPolicyArg);
    cmd.add(minHintWeightArg);
    cmd.add(hintScaleArg);
    cmd.add(moreUtilityWeightArg);
    cmd.add(startPosesBeforeHintsLenArg);
    cmd.add(forTestingArg);
    cmd.parseArgs(args);

    nnModelFile = cmd.getModelFile();
    sgfFilesFromCmdline = sgfArg.getValue();
    sgfDirs = sgfDirArg.getValue();
    sgfsDirs = sgfsDirArg.getValue();
    outDir = outDirArg.getValue();
    numProcessThreads = numProcessThreadsArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    gameMode = gameModeArg.getValue();
    treeMode = treeModeArg.getValue();
    surpriseMode = surpriseModeArg.getValue();
    autoKomi = autoKomiArg.getValue();
    tolerateIllegalMoves = tolerateIllegalMovesArg.getValue();
    sgfSplitCount = sgfSplitCountArg.getValue();
    sgfSplitIdx = sgfSplitIdxArg.getValue();
    maxDepth = maxDepthArg.getValue();
    turnWeightLambda = turnWeightLambdaArg.getValue();
    maxPosesPerOutFile = maxPosesPerOutFileArg.getValue();
    utilityThreshold = utilityThresholdArg.getValue();
    gameModeFastThreshold = gameModeFastThresholdArg.getValue();
    flipIfPassOrWFirst = flipIfPassOrWFirstArg.getValue();
    allowGameOver = allowGameOverArg.getValue();
    manualHintOnly = manualHintOnlyArg.getValue();
    trainingWeight = trainingWeightArg.getValue();
    minTurn = minTurnArg.getValue();
    minRank = minRankArg.getValue();
    minMinRank = minMinRankArg.getValue();
    requiredPlayerName = requiredPlayerNameArg.getValue();
    maxHandicap = maxHandicapArg.getValue();
    maxKomi = maxKomiArg.getValue();
    maxAutoKomi = maxAutoKomiArg.getValue();
    maxPolicy = maxPolicyArg.getValue();
    minHintWeight = minHintWeightArg.getValue();
    hintScale = hintScaleArg.getValue();
    moreUtilityWeight = moreUtilityWeightArg.getValue();
    startPosesBeforeHintsLen = startPosesBeforeHintsLenArg.getValue();
    forTesting = forTestingArg.getValue();

    if((int)gameMode + (int)treeMode + (int)surpriseMode != 1)
      throw StringError("Must specify either -game-mode or -tree-mode or -surprise-mode");
    if(startPosesBeforeHintsLen != 0 && gameMode != 1)
      throw StringError("startPosesBeforeHintsLen only works with -game-mode");

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outDir);

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.addFile(outDir + "/" + "log.log");
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);
  if(!forTesting)
    logger.write("Git revision " + Version::getGitRevision());

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_ANALYSIS);
  //Ignore temperature, noise
  params.chosenMoveTemperature = 0;
  params.chosenMoveTemperatureEarly = 0;
  params.rootNoiseEnabled = false;
  params.rootDesiredPerChildVisitsCoeff = 0;
  params.rootPolicyTemperature = 1.0;
  params.rootPolicyTemperatureEarly = 1.0;
  params.rootFpuReductionMax = params.fpuReductionMax * 0.5;

  //Disable dynamic utility so that utilities are always comparable
  params.staticScoreUtilityFactor += params.dynamicScoreUtilityFactor;
  params.dynamicScoreUtilityFactor = 0;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  GameInitializer* gameInit = new GameInitializer(cfg,logger);
  cfg.warnUnusedKeys(cerr,&logger);
  Setup::maybeWarnHumanSLParams(params,nnEval,NULL,cerr,&logger);

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);
  FileHelpers::collectMultiSgfsFromDirsOrFiles(sgfsDirs,sgfFiles);

  for(const string& s: sgfFilesFromCmdline)
    sgfFiles.push_back(s);

  logger.write("Found " + Global::int64ToString((int64_t)sgfFiles.size()) + " sgf(s) files!");

  if(forTesting)
    std::sort(sgfFiles.begin(),sgfFiles.end());
  else {
    seedRand.shuffle(sgfFiles);
  }

  set<Hash128> excludeHashes = Sgf::readExcludes(excludeHashesFiles);
  logger.write("Loaded " + Global::uint64ToString(excludeHashes.size()) + " excludes");


  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // ---------------------------------------------------------------------------------------------------
  PosWriter posWriter("hintposes.txt", outDir, sgfSplitCount, sgfSplitIdx, maxPosesPerOutFile);

  //COMMON ---------------------------------------------------------------------------------------------------
  std::atomic<int64_t> numSgfsDone(0);
  std::atomic<int64_t> numFilteredIndivdualPoses(0);
  std::atomic<int64_t> numFilteredSgfs(0);

  auto isPlayerOkay = [&](const Sgf* sgf, Player pla) {
    if(minRank != Sgf::RANK_UNKNOWN) {
      if(sgf->getRank(pla) < minRank)
        return false;
    }
    if(requiredPlayerName != "") {
      if(sgf->getPlayerName(pla) != requiredPlayerName)
        return false;
    }
    return true;
  };


  auto isSgfOkay = [&](const Sgf* sgf, string& reasonBuf) {
    if(maxHandicap < 100 && sgf->getHandicapValue() > maxHandicap)
    {reasonBuf = "handicap"; return false;}
    if(sgf->depth() > maxDepth)
    {reasonBuf = "depth" + Global::intToString(sgf->depth()); return false;}
    if(std::fabs(sgf->getKomiOrDefault(7.5f)) > maxKomi)
    {reasonBuf = "komi"; return false;}
    if(minMinRank != Sgf::RANK_UNKNOWN) {
      if(sgf->getRank(P_BLACK) < minMinRank || sgf->getRank(P_WHITE) < minMinRank)
      {reasonBuf = "rank"; return false;}
    }
    if(!isPlayerOkay(sgf,P_BLACK) && !isPlayerOkay(sgf,P_WHITE))
    {reasonBuf = "player " + sgf->getPlayerName(P_BLACK) + " " + sgf->getPlayerName(P_WHITE); return false;}
    return true;
  };

  auto expensiveEvaluateMove = [&posWriter,&turnWeightLambda,&maxAutoKomi,&maxHandicap,&numFilteredIndivdualPoses,&surpriseMode,&minHintWeight,&hintScale,&logger,trainingWeight,moreUtilityWeight,utilityThreshold,minTurn](
    Search* search, Loc missedLoc,
    Player nextPla, const Board& board, const BoardHistory& hist,
    const Sgf::PositionSample& sample, bool markedAsHintPos, bool markedAsHintPosLight
  ) {
    if(shouldStop.load(std::memory_order_acquire))
      return 0.0;

    if(std::fabs(hist.rules.komi) > maxAutoKomi) {
      numFilteredIndivdualPoses.fetch_add(1);
      return 0.0;
    }
    if(hist.computeNumHandicapStones() > maxHandicap) {
      numFilteredIndivdualPoses.fetch_add(1);
      return 0.0;
    }
    if(minTurn > 0 && (size_t)(sample.initialTurnNumber + sample.moves.size()) < minTurn)
      return 0.0;

    {
      int numStonesOnBoard = 0;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(board.colors[loc] != C_EMPTY)
            numStonesOnBoard += 1;
        }
      }
      if(numStonesOnBoard < 6)
        return 0.0;
    }

    if(surpriseMode) {
      // TODO Very simple logic - If a full search gives a different move than a quick search and
      // judges the move to be way better than the quick search's move, then record as a hintpos.
      // If a full search gives a very worse value than a quick search, then record as a sample position.

      ReportedSearchValues veryQuickValues;
      {
        bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/50.0,veryQuickValues);
        if(!suc)
          return 0.0;
      }
      Loc veryQuickMoveLoc = search->getChosenMoveLoc();
      ReportedSearchValues baseValues;
      {
        bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0,baseValues);
        if(!suc)
          return 0.0;
      }
      Loc moveLoc = search->getChosenMoveLoc();

      if(moveLoc != veryQuickMoveLoc) {
        ReportedSearchValues veryQuickAfterMoveValues;
        {
          bool suc = maybeGetValuesAfterMove(search,veryQuickMoveLoc,nextPla,board,hist,1.0/2.0,veryQuickAfterMoveValues);
          if(!suc)
            return 0.0;
        }
        ReportedSearchValues baseAfterMoveValues;
        {
          bool suc = maybeGetValuesAfterMove(search,moveLoc,nextPla,board,hist,1.0/2.0,baseAfterMoveValues);
          if(!suc)
            return 0.0;
        }
        if(
          (nextPla == P_WHITE && baseAfterMoveValues.utility - veryQuickAfterMoveValues.utility > 0.2) ||
          (nextPla == P_BLACK && baseAfterMoveValues.utility - veryQuickAfterMoveValues.utility < -0.2)
        ) {
          Sgf::PositionSample sampleToWrite = sample;
          sampleToWrite.weight += std::fabs(baseValues.utility - veryQuickValues.utility);
          sampleToWrite.hintLoc = moveLoc;
          sampleToWrite.trainingWeight = trainingWeight;
          posWriter.writePos(sampleToWrite);
          if(sampleToWrite.hasPreviousPositions(1))
            posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.5));
          if(sampleToWrite.hasPreviousPositions(2))
            posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.25).previousPosition(sampleToWrite.weight * 0.25));
          logger.write("Surprising good " + Global::doubleToString(sampleToWrite.weight));
          return sampleToWrite.weight;
        }
      }

      if(
        (nextPla == P_WHITE && baseValues.utility - veryQuickValues.utility < -0.2) ||
        (nextPla == P_BLACK && baseValues.utility - veryQuickValues.utility > 0.2)
      ) {
        Sgf::PositionSample sampleToWrite = sample;
        sampleToWrite.weight = 1.0 + std::fabs(baseValues.utility - veryQuickValues.utility);
        sampleToWrite.hintLoc = Board::NULL_LOC;
        sampleToWrite.trainingWeight = trainingWeight;
        posWriter.writePos(sampleToWrite);
        if(sampleToWrite.hasPreviousPositions(1))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.5));
        if(sampleToWrite.hasPreviousPositions(2))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.25).previousPosition(sampleToWrite.weight * 0.25));
        logger.write("Inevitable bad " + Global::doubleToString(sampleToWrite.weight));
        return sampleToWrite.weight;
      }
      if(
        (nextPla == P_WHITE && baseValues.utility - veryQuickValues.utility > 0.2) ||
        (nextPla == P_BLACK && baseValues.utility - veryQuickValues.utility < -0.2)
      ) {
        Sgf::PositionSample sampleToWrite = sample;
        sampleToWrite.weight = 1.0 + std::fabs(baseValues.utility - veryQuickValues.utility);
        sampleToWrite.hintLoc = Board::NULL_LOC;
        sampleToWrite.trainingWeight = trainingWeight;
        posWriter.writePos(sampleToWrite);
        if(sampleToWrite.hasPreviousPositions(1))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.5));
        if(sampleToWrite.hasPreviousPositions(2))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.25).previousPosition(sampleToWrite.weight * 0.25));
        logger.write("Inevitable good " + Global::doubleToString(sampleToWrite.weight));
        return sampleToWrite.weight;
      }
      return 0.0;
    }

    ReportedSearchValues veryQuickValues;
    {
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/25.0,veryQuickValues);
      if(!suc)
        return 0.0;
    }
    Loc veryQuickMoveLoc = search->getChosenMoveLoc();

    ReportedSearchValues quickValues;
    {
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/5.0,quickValues);
      if(!suc)
        return 0.0;
    }
    Loc quickMoveLoc = search->getChosenMoveLoc();

    ReportedSearchValues baseValues;
    {
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0,baseValues);
      if(!suc)
        return 0.0;
    }
    Loc moveLoc = search->getChosenMoveLoc();

    // const Player perspective = P_WHITE;
    // {
    //   ostringstream preOut;
    //   Board::printBoard(preOut, search->getRootBoard(), moveLoc, &(search->getRootHist().moveHistory));
    //   search->printTree(preOut, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10),perspective);
    //   cout << preOut.str() << endl;
    //   cout << Location::toString(missedLoc,board) << endl;
    // }

    Sgf::PositionSample sampleToWrite = sample;
    sampleToWrite.trainingWeight = trainingWeight;
    sampleToWrite.weight += std::fabs(baseValues.utility - quickValues.utility);
    sampleToWrite.weight += std::fabs(baseValues.utility - veryQuickValues.utility);

    constexpr double NULL_UTILITY = -1e10; // Placeholder if no utility is found
    double selfUtilityOfHintMove = NULL_UTILITY;
    double selfUtilityOfBestOtherMove = NULL_UTILITY;
    if(moreUtilityWeight > 0.0) {
      vector<AnalysisData> analysisData;
      const int minMovesToTryToGet = 0;
      const bool includeWeightFactors = false;
      const int maxPVDepth = 0;
      const bool duplicateForSymmetries = true;
      search->getAnalysisData(analysisData, minMovesToTryToGet, includeWeightFactors, maxPVDepth, duplicateForSymmetries);
      for(const AnalysisData& data: analysisData) {
        if(data.move == missedLoc)
          selfUtilityOfHintMove = (nextPla == P_WHITE ? data.utility : -data.utility);
        else
          selfUtilityOfBestOtherMove = std::max(selfUtilityOfBestOtherMove, (nextPla == P_WHITE ? data.utility : -data.utility));
      }
    }

    //Bot DOES see the move?
    if(moveLoc == missedLoc) {
      if(moreUtilityWeight > 0.0 && selfUtilityOfHintMove > NULL_UTILITY && selfUtilityOfBestOtherMove > NULL_UTILITY) {
        sampleToWrite.weight += moreUtilityWeight * (-0.3 + sqrt(0.09 + std::max(0.0, selfUtilityOfHintMove - selfUtilityOfBestOtherMove)));
      }

      if(quickMoveLoc == moveLoc)
        sampleToWrite.weight = sampleToWrite.weight * 0.75 - 0.1;
      if(veryQuickMoveLoc == moveLoc)
        sampleToWrite.weight = sampleToWrite.weight * 0.75 - 0.1;

      sampleToWrite.weight *= exp(-sampleToWrite.initialTurnNumber * turnWeightLambda);

      if(markedAsHintPos)
        sampleToWrite.weight *= hintScale;

      if(sampleToWrite.weight < minHintWeight && markedAsHintPos)
        sampleToWrite.weight = minHintWeight;
      if(sampleToWrite.weight < minHintWeight / 2 && markedAsHintPosLight)
        sampleToWrite.weight = minHintWeight / 2;

      if(sampleToWrite.weight > 0.1) {
        //Still good to learn from given that policy was really low
        posWriter.writePos(sampleToWrite);
        return sampleToWrite.weight;
      }
    }

    //Bot doesn't see the move?
    else if(moveLoc != missedLoc) {

      ReportedSearchValues moveValues;
      ReportedSearchValues missedValues;
      // If marked as hint pos and no utility bonus, then we don't need these values
      if(markedAsHintPos && moreUtilityWeight <= 0.0) {}
      else {
        if(!maybeGetValuesAfterMove(search,moveLoc,nextPla,board,hist,1.0,moveValues))
          return 0.0;
        if(!maybeGetValuesAfterMove(search,missedLoc,nextPla,board,hist,1.0,missedValues))
          return 0.0;
      }
      if(moreUtilityWeight > 0.0) {
        selfUtilityOfBestOtherMove = std::max(selfUtilityOfBestOtherMove, (nextPla == P_WHITE ? moveValues.utility : -moveValues.utility));
        selfUtilityOfHintMove = std::max(selfUtilityOfHintMove, (nextPla == P_WHITE ? missedValues.utility : -missedValues.utility));
      }

      bool shouldWriteMove = false;
      if(markedAsHintPos) {
        //If marked as a hint pos, always trust that it should be better and add it.
        shouldWriteMove = true;
      }
      else {
        // ostringstream out0;
        // out0 << "SGF MOVE " << Location::toString(missedLoc,board) << endl;
        // search->printTree(out0, search->rootNode, PrintTreeOptions().maxDepth(0),perspective);
        // cout << out0.str() << endl;

        //If the move is this minimum amount better, then record this position as a hint
        //Otherwise the bot actually thinks the move isn't better, so we reject it as an invalid hint.
        ReportedSearchValues postValues = search->getRootValuesRequireSuccess();
        if((nextPla == P_WHITE && missedValues.utility > moveValues.utility + utilityThreshold) ||
           (nextPla == P_BLACK && missedValues.utility < moveValues.utility - utilityThreshold)) {
          shouldWriteMove = true;
        }
      }

      double lightFactor = 1.0;
      if(!shouldWriteMove && markedAsHintPosLight) {
        shouldWriteMove = true;
        lightFactor = 1.0/3.0;
      }

      if(shouldWriteMove) {
        //Moves that the bot didn't see get written out more
        sampleToWrite.weight = (sampleToWrite.weight * 1.5 + 1.0) * lightFactor;

        if(moreUtilityWeight > 0.0 && selfUtilityOfHintMove > NULL_UTILITY && selfUtilityOfBestOtherMove > NULL_UTILITY) {
          sampleToWrite.weight += moreUtilityWeight * (-0.3 + sqrt(0.09 + std::max(0.0, selfUtilityOfHintMove - selfUtilityOfBestOtherMove)));
        }

        sampleToWrite.weight *= exp(-sampleToWrite.initialTurnNumber * turnWeightLambda);
        if(markedAsHintPos)
          sampleToWrite.weight *= hintScale;
        if(sampleToWrite.weight < minHintWeight && markedAsHintPos)
          sampleToWrite.weight = minHintWeight;
        if(sampleToWrite.weight < minHintWeight / 2 && markedAsHintPosLight)
          sampleToWrite.weight = minHintWeight / 2;
        if(sampleToWrite.weight > 0.1) {
          posWriter.writePos(sampleToWrite);
          return sampleToWrite.weight;
        }
      }
    }

    return 0.0;
  };

  // ---------------------------------------------------------------------------------------------------
  //GAME MODE

  auto processSgfGame = [&posWriter,&logger,&gameInit,&nnEval,&expensiveEvaluateMove,autoKomi,&gameModeFastThreshold,&maxDepth,&numFilteredSgfs,&maxHandicap,&maxPolicy,allowGameOver,manualHintOnly,trainingWeight,startPosesBeforeHintsLen,minTurn](
    Search* search, Rand& rand, const string& fileName, CompactSgf* sgf, bool blackOkay, bool whiteOkay
  ) {
    //Don't use the SGF rules - randomize them for a bit more entropy
    Rules rules = gameInit->createRules();

    Board board;
    Player nextPla;
    BoardHistory hist;
    sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);
    if(!gameInit->isAllowedBSize(board.x_size,board.y_size)) {
      numFilteredSgfs.fetch_add(1);
      return;
    }
    if(board.x_size != 19 || board.y_size != 19) {
      numFilteredSgfs.fetch_add(1);
      return;
    }

    const bool preventEncore = true;
    const vector<Move>& sgfMoves = sgf->moves;

    if((int64_t)sgfMoves.size() > maxDepth) {
      numFilteredSgfs.fetch_add(1);
      return;
    }
    if(hist.computeNumHandicapStones() > maxHandicap) {
      numFilteredSgfs.fetch_add(1);
      return;
    }

    vector<Board> boards;
    vector<BoardHistory> hists;
    vector<Player> nextPlas;
    vector<shared_ptr<NNOutput>> nnOutputs;
    vector<double> winLossValues;
    vector<double> scoreLeads;

    vector<Move> moves;
    vector<double> policyPriors;

    for(int m = 0; m<sgfMoves.size()+1; m++) {
      MiscNNInputParams nnInputParams;
      NNResultBuf buf;
      bool skipCache = true; //Always ignore cache so that we get more entropy on repeated board positions due to symmetries
      bool includeOwnerMap = false;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

      ReportedSearchValues superQuickValues;
      {
        bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/60.0,superQuickValues);
        if(!suc)
          break;
      }

      boards.push_back(board);
      hists.push_back(hist);
      nextPlas.push_back(nextPla);
      nnOutputs.push_back(std::move(buf.result));

      shared_ptr<NNOutput>& nnOutput = nnOutputs[nnOutputs.size()-1];

      winLossValues.push_back(superQuickValues.winLossValue);
      scoreLeads.push_back(superQuickValues.lead);

      if(m < sgfMoves.size()) {
        moves.push_back(sgfMoves[m]);
        int pos = NNPos::locToPos(sgfMoves[m].loc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
        policyPriors.push_back(nnOutput->policyProbs[pos]);
      }

      if(m >= sgfMoves.size())
        break;

      //Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
      if(!allowGameOver && (hist.isGameFinished || hist.encorePhase > 0))
        break;

      //Quit out if consecutive moves by the same player, to keep the history clean and "normal"
      if(sgfMoves[m].pla != nextPla && m > 0) {
        logger.write("Ending SGF " + fileName + " early due to non-alternating players on turn " + Global::intToString(m));
        break;
      }

      bool suc = hist.isLegal(board,sgfMoves[m].loc,sgfMoves[m].pla);
      if(!suc) {
        //Only log on errors that aren't simply due to ko rules, but quit out regardless
        suc = hist.makeBoardMoveTolerant(board,sgfMoves[m].loc,sgfMoves[m].pla,preventEncore);
        if(!suc)
          logger.write("Illegal move in " + fileName + " turn " + Global::intToString(m) + " move " + Location::toString(sgfMoves[m].loc, board.x_size, board.y_size));
        break;
      }
      hist.makeBoardMoveAssumeLegal(board,sgfMoves[m].loc,sgfMoves[m].pla,NULL,preventEncore);
      nextPla = getOpp(sgfMoves[m].pla);
    }
    boards.push_back(board);
    hists.push_back(hist);
    nextPlas.push_back(nextPla);

    if(winLossValues.size() <= 0)
      return;
    if(shouldStop.load(std::memory_order_acquire))
      return;

    vector<double> futureValue(winLossValues.size()+1);
    vector<double> futureLead(winLossValues.size()+1);
    vector<double> pastValue(winLossValues.size());
    vector<double> pastLead(winLossValues.size());
    futureValue[winLossValues.size()] = winLossValues[winLossValues.size()-1];
    futureLead[winLossValues.size()] = scoreLeads[winLossValues.size()-1];
    for(int i = (int)winLossValues.size()-1; i >= 0; i--) {
      futureValue[i] = 0.10 * winLossValues[i] + 0.90 * futureValue[i+1];
      futureLead[i] = 0.10 * scoreLeads[i] + 0.90 * futureLead[i+1];
    }
    pastValue[0] = winLossValues[0];
    pastLead[0] = scoreLeads[0];
    for(int i = 1; i<(int)winLossValues.size(); i++) {
      pastValue[i] = 0.5 * winLossValues[i] + 0.5 * pastValue[i-1];
      pastLead[i] = 0.5 * scoreLeads[i] + 0.5 * pastLead[i-1];
    }

    const double scoreLeadWeight = 0.01;
    const double sumThreshold = gameModeFastThreshold;

    //cout << fileName << endl;
    std::map<int,double> startPosesBeforeHintsWeights;
    for(int m = 0; m<moves.size(); m++) {

      if(shouldStop.load(std::memory_order_acquire))
        break;

      if((nextPlas[m] == P_BLACK && !blackOkay) || (nextPlas[m] == P_WHITE && !whiteOkay))
        continue;

      if(m < minTurn)
        continue;

      //cout << m << endl;
      //Look for surprising moves that turned out not poorly
      //The more surprising, the more times we will write it out.
      if(policyPriors[m] > maxPolicy)
        continue;
      double weight = surpriseWeight(policyPriors[m],rand,false);
      if(weight <= 0)
        continue;

      double pastSum = pastValue[m] + pastLead[m]*scoreLeadWeight;
      double futureSum = futureValue[m] + futureLead[m]*scoreLeadWeight;
      if((nextPlas[m] == P_WHITE && futureSum > pastSum + sumThreshold) ||
         (nextPlas[m] == P_BLACK && futureSum < pastSum - sumThreshold)) {
        //Good
      }
      else
        continue;

      Sgf::PositionSample sample;
      const int numMovesToRecord = 8;
      int startIdx = std::max(0,m-numMovesToRecord);
      sample.board = boards[startIdx];
      sample.nextPla = nextPlas[startIdx];
      for(int j = startIdx; j<m; j++)
        sample.moves.push_back(moves[j]);
      sample.initialTurnNumber = hists[m].initialTurnNumber + startIdx;
      sample.hintLoc = moves[m].loc;
      sample.weight = weight;
      sample.trainingWeight = trainingWeight;

      if(autoKomi) {
        const int64_t numVisits = 10;
        OtherGameProperties props;
        PlayUtils::adjustKomiToEven(search,NULL,boards[m],hists[m],nextPlas[m],numVisits,props,rand);
      }

      double wroteHintPosWeight = expensiveEvaluateMove(
        search, moves[m].loc, nextPlas[m], boards[m], hists[m],
        sample, false, false
      );

      if(wroteHintPosWeight > 0 && startPosesBeforeHintsLen > 0) {
        for(int i = std::max(0,m-startPosesBeforeHintsLen); i <= m+1; i++) {
          double newWeight = std::max(startPosesBeforeHintsWeights[i], wroteHintPosWeight / sqrt(startPosesBeforeHintsLen));
          startPosesBeforeHintsWeights[i] = newWeight;
        }
      }
    }

    for(int m = 0; m<moves.size(); m++) {
      if(startPosesBeforeHintsWeights[m] > 0) {
        Sgf::PositionSample sample;
        const int numMovesToRecord = 8;
        int startIdx = std::max(0,m-numMovesToRecord);
        sample.board = boards[startIdx];
        sample.nextPla = nextPlas[startIdx];
        for(int j = startIdx; j<m; j++)
          sample.moves.push_back(moves[j]);
        sample.initialTurnNumber = hists[m].initialTurnNumber + startIdx;
        sample.hintLoc = Board::NULL_LOC;
        sample.weight = startPosesBeforeHintsWeights[m];
        sample.trainingWeight = trainingWeight;
        posWriter.writePos(sample);
      }
    }

    logger.write("Sgf processed: " + fileName);
  };

  const int maxSgfQueueSize = 128;
  ThreadSafeQueue<Sgf*> sgfQueue(maxSgfQueueSize);
  auto processSgfLoop = [&logger,&processSgfGame,&sgfQueue,&params,&nnEval,&numSgfsDone,&isPlayerOkay,&tolerateIllegalMoves]() {
    Rand rand;
    string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
    Search* search = new Search(params,nnEval,&logger,searchRandSeed);

    while(true) {
      if(shouldStop.load(std::memory_order_acquire))
        break;

      Sgf* sgfRaw;
      bool success = sgfQueue.waitPop(sgfRaw);
      if(!success)
        break;

      bool blackOkay = isPlayerOkay(sgfRaw,P_BLACK);
      bool whiteOkay = isPlayerOkay(sgfRaw,P_WHITE);

      CompactSgf* sgf = NULL;
      try {
        sgf = new CompactSgf(sgfRaw);
      }
      catch(const StringError& e) {
        if(!tolerateIllegalMoves)
          throw;
        else {
          logger.write(e.what());
        }
      }
      if(sgf != NULL)
        processSgfGame(search,rand,sgf->fileName,sgf,blackOkay,whiteOkay);

      numSgfsDone.fetch_add(1);
      delete sgf;
      delete sgfRaw;
    }
    delete search;
  };



  // ---------------------------------------------------------------------------------------------------
  //TREE MODE

  auto treePosHandler = [&gameInit,&nnEval,&expensiveEvaluateMove,&autoKomi,&maxPolicy,&flipIfPassOrWFirst,&surpriseMode,trainingWeight](
    Search* search, Rand& rand, const BoardHistory& treeHist, bool markedAsHintPos, bool markedAsHintPosLight
  ) {
    if(shouldStop.load(std::memory_order_acquire))
      return;
    if(treeHist.moveHistory.size() > 0x3FFFFFFF)
      throw StringError("Too many moves in history");
    int moveHistorySize = (int)treeHist.moveHistory.size();
    if(moveHistorySize <= 0)
      return;

    //Snap the position 8 turns ago so as to include 8 moves of history.
    int turnsAgoToSnap = 0;
    while(turnsAgoToSnap < 8) {
      if(turnsAgoToSnap >= moveHistorySize)
        break;
      //If a player played twice in a row, then instead snap so as not to have a move history
      //with a double move by the same player.
      if(turnsAgoToSnap > 0 && treeHist.moveHistory[moveHistorySize - turnsAgoToSnap - 1].pla == treeHist.moveHistory[moveHistorySize - turnsAgoToSnap].pla)
        break;
      turnsAgoToSnap++;
    }
    int startTurn = moveHistorySize - turnsAgoToSnap;
    //If the start turn is past the end of the last move, we don't actually have a move we're judging if it's good, so we quit.
    if(startTurn >= moveHistorySize)
      return;

    //Play moves out until we get back to where we need to be.
    //This is hacky and makes everything quadratic, but whatever
    Board board = treeHist.initialBoard;
    for(int i = 0; i<startTurn; i++) {
      bool multiStoneSuicideLegal = true;
      //Just in case
      if(!board.isLegal(treeHist.moveHistory[i].loc,treeHist.moveHistory[i].pla,multiStoneSuicideLegal))
        return;
      board.playMoveAssumeLegal(treeHist.moveHistory[i].loc,treeHist.moveHistory[i].pla);
    }

    Sgf::PositionSample sample;
    sample.board = board;
    sample.nextPla = treeHist.moveHistory[startTurn].pla;
    for(int j = startTurn; j<moveHistorySize-1; j++)
      sample.moves.push_back(treeHist.moveHistory[j]);
    sample.initialTurnNumber = treeHist.initialTurnNumber + startTurn;
    sample.hintLoc = treeHist.moveHistory[moveHistorySize-1].loc;
    sample.weight = 0.0; //dummy, filled in below
    sample.trainingWeight = trainingWeight;

    //Don't use the SGF rules - randomize them for a bit more entropy
    Rules rules = gameInit->createRules();

    //Now play the rest of the moves out, except the last, which we keep as the potential hintloc
    int encorePhase = 0;
    Player pla = sample.nextPla;
    BoardHistory hist(board,pla,rules,encorePhase);
    int numSampleMoves = (int)sample.moves.size();
    for(int i = 0; i<numSampleMoves; i++) {
      if(!hist.isLegal(board,sample.moves[i].loc,sample.moves[i].pla))
        return;
      assert(sample.moves[i].pla == pla);
      hist.makeBoardMoveAssumeLegal(board,sample.moves[i].loc,sample.moves[i].pla,NULL);
      pla = getOpp(pla);
    }

    //Make sure the hinted move is legal too under our randomized rules.
    int hintIdx = (int)treeHist.moveHistory.size()-1;
    assert(treeHist.moveHistory[hintIdx].pla == pla);
    assert(treeHist.moveHistory[hintIdx].loc == sample.hintLoc);
    if(!hist.isLegal(board,sample.hintLoc,pla))
      return;

    if(autoKomi) {
      const int64_t numVisits = 10;
      OtherGameProperties props;
      PlayUtils::adjustKomiToEven(search,NULL,board,hist,pla,numVisits,props,rand);
    }

    MiscNNInputParams nnInputParams;
    bool skipCache = true; //Always ignore cache so that we get more entropy on repeated board positions due to symmetries
    bool includeOwnerMap = false;

    double policyProb = 0.0;
    {
      // Take 1.1 * the geometric mean of a few samples, so as to greatly upweight the importance of anomalous low values.
      double acc = 0.0;
      int count = 0;
      for(int samples = 0; samples < 4; samples++) {
        NNResultBuf buf;
        nnEval->evaluate(board,hist,pla,nnInputParams,buf,skipCache,includeOwnerMap);
        shared_ptr<NNOutput>& nnOutput = buf.result;
        int pos = NNPos::locToPos(sample.hintLoc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
        double prob = nnOutput->policyProbs[pos];
        assert(prob >= 0.0);
        acc += log(prob + 1e-30);
        count += 1;
      }
      assert(count > 0);
      policyProb = 1.1 * exp(acc / count);
    }

    if(policyProb > maxPolicy)
      return;
    bool alwaysAddWeight = markedAsHintPos || markedAsHintPosLight || surpriseMode;
    double weight = surpriseWeight(policyProb,rand,alwaysAddWeight);
    if(weight <= 0)
      return;
    sample.weight = weight;

    if(flipIfPassOrWFirst) {
      if(treeHist.hasBlackPassOrWhiteFirst())
        sample = sample.getColorFlipped();
    }

    expensiveEvaluateMove(
      search, sample.hintLoc, pla, board, hist,
      sample, markedAsHintPos, markedAsHintPosLight
    );
  };


  const int64_t maxPosQueueSize = 1024;
  ThreadSafeQueue<PosQueueEntry> posQueue(maxPosQueueSize);
  std::atomic<int64_t> numPosesBegun(0);
  std::atomic<int64_t> numPosesDone(0);
  std::atomic<int64_t> numPosesEnqueued(0);

  auto processPosLoop = [&logger,&posQueue,&params,&numPosesBegun,&numPosesDone,&numPosesEnqueued,&nnEval,&treePosHandler,&seedRand,forTesting]() {
    Rand rand(forTesting ? "testseed" : Global::uint64ToString(seedRand.nextUInt64()));
    string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
    Search* search = new Search(params,nnEval,&logger,searchRandSeed);

    while(true) {
      if(shouldStop.load(std::memory_order_acquire))
        break;

      PosQueueEntry p;
      bool success = posQueue.waitPop(p);
      if(!success)
        break;
      BoardHistory* hist = p.hist;
      bool markedAsHintPos = p.markedAsHintPos;
      bool markedAsHintPosLight = p.markedAsHintPosLight;

      int64_t numEnqueued = numPosesEnqueued.load();
      int64_t numBegun = 1+numPosesBegun.fetch_add(1);
      if(numBegun % 20 == 0)
        logger.write("Begun " + Global::int64ToString(numBegun) + "/" + Global::int64ToString(numEnqueued) + " poses");

      treePosHandler(search, rand, *hist, markedAsHintPos, markedAsHintPosLight);

      int64_t numDone = 1+numPosesDone.fetch_add(1);
      if(numDone % 20 == 0)
        logger.write("Done " + Global::int64ToString(numDone) + "/" + Global::int64ToString(numEnqueued) + " poses");

      delete hist;
    }
    delete search;
    posQueue.setReadOnly();
  };


  // ---------------------------------------------------------------------------------------------------

  //Begin writing
  posWriter.start();

  vector<std::thread> threads;
  for(int i = 0; i<numProcessThreads; i++) {
    if(gameMode)
      threads.push_back(std::thread(processSgfLoop));
    else if(treeMode)
      threads.push_back(std::thread(processPosLoop));
    else if(surpriseMode)
      threads.push_back(std::thread(processPosLoop));
  }

  // ---------------------------------------------------------------------------------------------------

  int64_t numSgfsBegun = 0;
  int64_t numSgfsSkipped = 0;
  int64_t numSgfsFilteredTopLevel = 0;

  std::set<Hash128> uniqueHashes;

  auto logSgfProgress = [&]() {
    logger.write(
      "Begun " + Global::int64ToString(numSgfsBegun) + " / " + Global::int64ToString(sgfFiles.size()) + " sgfs, " +
      string("done ") + Global::int64ToString(numSgfsDone.load()) + " sgfs, " +
      string("skipped ") + Global::int64ToString(numSgfsSkipped) + " sgfs, " +
      string("filtered ") + Global::int64ToString(numSgfsFilteredTopLevel + numFilteredSgfs.load()) + " sgfs, " +
      string("filtered ") + Global::int64ToString(numFilteredIndivdualPoses.load()) + " individual poses"
    );
  };

  for(size_t i = 0; i<sgfFiles.size(); i++) {
    numSgfsBegun += 1;
    if(numSgfsBegun % std::min((size_t)20, 1 + sgfFiles.size() / 60) == 0)
      logSgfProgress();

    const string& fileName = sgfFiles[i];

    std::vector<Sgf*> sgfs = Sgf::loadSgfOrSgfsLogAndIgnoreErrors(fileName,logger);
    if(!forTesting)
      seedRand.shuffle(sgfs);

    for(size_t j = 0; j<sgfs.size(); j++) {
      Sgf* sgf = sgfs[j];

      if(contains(excludeHashes,sgf->hash)) {
        logger.write("Filtering due to exclude: " + fileName);
        numSgfsFilteredTopLevel += 1;
        delete sgf;
        continue;
      }
      try {
        string reasonBuf;
        if(!isSgfOkay(sgf,reasonBuf)) {
          logger.write("Filtering due to not okay (" + reasonBuf + "): " + fileName);
          numSgfsFilteredTopLevel += 1;
          delete sgf;
          continue;
        }
      }
      catch(const StringError& e) {
        logger.write("Filtering due to error checking okay: " + fileName + ": " + e.what());
        numSgfsFilteredTopLevel += 1;
        delete sgf;
        continue;
      }
      if(sgfSplitCount > 1 && ((int)(sgf->hash.hash0 & 0x7FFFFFFF) % sgfSplitCount) != sgfSplitIdx) {
        numSgfsSkipped += 1;
        delete sgf;
        continue;
      }

      logger.write("Starting " + fileName);
      handleStartAnnotations(sgf);

      if(gameMode) {
        sgfQueue.waitPush(sgf);
      }
      else {
        bool hashComments = true; //Hash comments so that if we see a position without %HINT% and one with, we make sure to re-load it.
        bool blackOkay = isPlayerOkay(sgf,P_BLACK);
        bool whiteOkay = isPlayerOkay(sgf,P_WHITE);
        try {
          bool hashParent = true; //Hash parent so that we distinguish hint moves that reach the same position but were different moves from different starting states.
          sgf->iterAllUniquePositions(
            uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, forTesting ? NULL : &seedRand,
            [&](Sgf::PositionSample& unusedSample, const BoardHistory& hist, const string& comments) {
              if(comments.size() > 0 && comments.find("%NOHINT%") != string::npos)
                return;
              if(hist.moveHistory.size() <= 0)
                return;
              int hintIdx = (int)hist.moveHistory.size()-1;
              if((hist.moveHistory[hintIdx].pla == P_BLACK && !blackOkay) || (hist.moveHistory[hintIdx].pla == P_WHITE && !whiteOkay))
                return;

              bool markedAsHintPos = (comments.size() > 0 && comments.find("%HINT%") != string::npos);
              bool markedAsHintPosLight = (comments.size() > 0 && comments.find("%HINTLIGHT%") != string::npos);
              if(manualHintOnly && !markedAsHintPos && !markedAsHintPosLight)
                return;

              //unusedSample doesn't have enough history, doesn't have hintloc the way we want it
              int64_t numEnqueued = 1+numPosesEnqueued.fetch_add(1);
              if(numEnqueued % 500 == 0)
                logger.write("Enqueued " + Global::int64ToString(numEnqueued) + " poses");
              PosQueueEntry entry;
              entry.hist = new BoardHistory(hist);
              assert(hist.getCurrentTurnNumber() == unusedSample.getCurrentTurnNumber());
              entry.markedAsHintPos = markedAsHintPos;
              entry.markedAsHintPosLight = markedAsHintPosLight;
              posQueue.waitPush(entry);
            }
          );
        }
        catch(const StringError& e) {
          if(!tolerateIllegalMoves)
            throw;
          else
            logger.write(e.what());
        }
        numSgfsDone.fetch_add(1);
        delete sgf;
      }
    }
  }
  logSgfProgress();
  logger.write("All sgfs loaded, waiting for finishing analysis");
  logger.write(Global::uint64ToString(sgfQueue.size()) + " sgfs still enqueued");

  sgfQueue.setReadOnly();
  posQueue.setReadOnly();
  for(size_t i = 0; i<threads.size(); i++)
    threads[i].join();

  logSgfProgress();
  logger.write("Waiting for final writing and cleanup");

  posWriter.flushAndStop();

  logger.write(nnEval->getModelFileName());
  logger.write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
  logger.write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
  logger.write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));

  logger.write("All done");

  delete gameInit;
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}






int MainCmds::trystartposes(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> startPosesFiles;
  double minWeight;
  bool autoKomi;
  bool randomSample;
  try {
    KataGoCommandLine cmd("Try running searches starting from startposes");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> startPosesFileArg("","startposes","Startposes file",true,"DIR");
    TCLAP::ValueArg<double> minWeightArg("","min-weight","Minimum weight of startpos to try",false,0.0,"WEIGHT");
    TCLAP::SwitchArg autoKomiArg("","auto-komi","Auto komi");
    TCLAP::SwitchArg randomSampleArg("","random-sample","Weighted random sample");

    cmd.add(startPosesFileArg);
    cmd.add(minWeightArg);
    cmd.add(autoKomiArg);
    cmd.add(randomSampleArg);
    cmd.parseArgs(args);
    nnModelFile = cmd.getModelFile();
    startPosesFiles = startPosesFileArg.getValue();
    minWeight = minWeightArg.getValue();
    autoKomi = autoKomiArg.getValue();
    randomSample = randomSampleArg.getValue();
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_ANALYSIS);
  //Ignore temperature, noise
  params.chosenMoveTemperature = 0;
  params.chosenMoveTemperatureEarly = 0;
  params.rootNoiseEnabled = false;
  params.rootDesiredPerChildVisitsCoeff = 0;
  params.rootPolicyTemperature = 1.0;
  params.rootPolicyTemperatureEarly = 1.0;
  params.rootFpuReductionMax = params.fpuReductionMax * 0.5;

  //Disable dynamic utility so that utilities are always comparable
  params.staticScoreUtilityFactor += params.dynamicScoreUtilityFactor;
  params.dynamicScoreUtilityFactor = 0;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  vector<Sgf::PositionSample> startPoses;
  for(size_t i = 0; i<startPosesFiles.size(); i++) {
    const string& startPosesFile = startPosesFiles[i];
    vector<string> lines = FileUtils::readFileLines(startPosesFile,'\n');
    for(size_t j = 0; j<lines.size(); j++) {
      string line = Global::trim(lines[j]);
      if(line.size() > 0) {
        try {
          Sgf::PositionSample posSample = Sgf::PositionSample::ofJsonLine(line);
          startPoses.push_back(posSample);
        }
        catch(const StringError& err) {
          logger.write(string("ERROR parsing startpos:") + err.what());
        }
      }
    }
  }
  string searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());
  Search* search = new Search(params,nnEval,&logger,searchRandSeed);

  std::vector<double> startPosCumProbs;
  double cumProb = 0;
  for(size_t i = 0; i<startPoses.size(); i++) {
    cumProb += startPoses[i].weight;
    startPosCumProbs.push_back(cumProb);
  }

  // ---------------------------------------------------------------------------------------------------

  for(size_t s = 0; s<startPoses.size(); s++) {
    size_t r;
    if(randomSample)
      r = seedRand.nextIndexCumulative(startPosCumProbs.data(),startPosCumProbs.size());
    else
      r = s;

    const Sgf::PositionSample& startPos = startPoses[r];
    if(startPos.weight < minWeight)
      continue;

    Rules rules = PlayUtils::genRandomRules(seedRand);
    Board board = startPos.board;
    Player pla = startPos.nextPla;
    BoardHistory hist;
    hist.clear(board,pla,rules,0);
    hist.setInitialTurnNumber(startPos.initialTurnNumber);
    bool allLegal = true;
    for(size_t i = 0; i<startPos.moves.size(); i++) {
      bool isLegal = hist.makeBoardMoveTolerant(board,startPos.moves[i].loc,startPos.moves[i].pla,false);
      if(!isLegal) {
        allLegal = false;
        break;
      }
      pla = getOpp(startPos.moves[i].pla);
    }
    if(!allLegal) {
      throw StringError("Illegal move in startpos: " + Sgf::PositionSample::toJsonLine(startPos));
    }

    if(autoKomi) {
      const int64_t numVisits = 10;
      OtherGameProperties props;
      PlayUtils::adjustKomiToEven(search,NULL,board,hist,pla,numVisits,props,seedRand);
    }

    Loc hintLoc = startPos.hintLoc;

    {
      ReportedSearchValues values;
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,pla,board,hist,1.0,values);
      (void)suc;
      assert(suc);
      cout << "Searching startpos: " << "\n";
      cout << "Weight: " << startPos.weight << "\n";
      cout << "Training Weight: " << startPos.trainingWeight << "\n";
      cout << search->getRootHist().rules.toString() << "\n";
      Board::printBoard(cout, search->getRootBoard(), search->getChosenMoveLoc(), &(search->getRootHist().moveHistory));
      search->printTree(cout, search->rootNode, PrintTreeOptions().maxDepth(1),P_WHITE);
      cout << endl;
    }

    if(hintLoc != Board::NULL_LOC) {
      if(search->getChosenMoveLoc() == hintLoc) {
        cout << "There was a hintpos " << Location::toString(hintLoc,board) << ", but it was the chosen move" << "\n";
        cout << endl;
      }
      else {
        ReportedSearchValues values;
        cout << "There was a hintpos " << Location::toString(hintLoc,board) << ", re-searching after playing it: " << "\n";
        bool suc = maybeGetValuesAfterMove(search,hintLoc,pla,board,hist,1.0,values);
        (void)suc;
        assert(suc);
        Board::printBoard(cout, search->getRootBoard(), search->getChosenMoveLoc(), &(search->getRootHist().moveHistory));
        search->printTree(cout, search->rootNode, PrintTreeOptions().maxDepth(1),P_WHITE);
        cout << endl;
      }
    }
  }

  delete search;
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}


int MainCmds::viewstartposes(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  vector<string> startPosesFiles;
  double minWeight;
  int idxToView;
  bool checkLegality;
  bool autoKomi;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","",false);
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> startPosesFileArg("","start-poses-file","Startposes file",true,"DIR");
    TCLAP::ValueArg<double> minWeightArg("","min-weight","Min weight of startpos to view",false,0.0,"WEIGHT");
    TCLAP::ValueArg<int> idxArg("","idx","Index of startpos to view in file",false,-1,"IDX");
    TCLAP::SwitchArg checkLegalityArg("","check-legality","Print startposes that are illegal or that have illegal hints");
    TCLAP::SwitchArg autoKomiArg("","auto-komi","Auto komi");
    cmd.add(startPosesFileArg);
    cmd.add(minWeightArg);
    cmd.add(idxArg);
    cmd.add(checkLegalityArg);
    cmd.add(autoKomiArg);
    cmd.parseArgs(args);
    startPosesFiles = startPosesFileArg.getValue();
    minWeight = minWeightArg.getValue();
    idxToView = idxArg.getValue();
    checkLegality = checkLegalityArg.getValue();
    autoKomi = autoKomiArg.getValue();

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  const bool logToStderrDefault = false;
  const bool logTimeDefault = true;
  const bool logConfigContents = cfg.getFileName() != "";
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault, logTimeDefault, logConfigContents);

  Rules rules;
  AsyncBot* bot = NULL;
  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    const bool loadKomiFromCfg = false;
    rules = Setup::loadSingleRules(cfg,loadKomiFromCfg);
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int expectedConcurrentEvals = params.numThreads;
      const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");

    string searchRandSeed;
    if(cfg.contains("searchRandSeed"))
      searchRandSeed = cfg.getString("searchRandSeed");
    else
      searchRandSeed = Global::uint64ToString(rand.nextUInt64());

    bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
  }

  vector<Sgf::PositionSample> startPoses;
  for(size_t i = 0; i<startPosesFiles.size(); i++) {
    const string& startPosesFile = startPosesFiles[i];
    vector<string> lines = FileUtils::readFileLines(startPosesFile,'\n');
    for(size_t j = 0; j<lines.size(); j++) {
      string line = Global::trim(lines[j]);
      if(line.size() > 0) {
        try {
          Sgf::PositionSample posSample = Sgf::PositionSample::ofJsonLine(line);
          startPoses.push_back(posSample);
        }
        catch(const StringError& err) {
          cout << (string("ERROR parsing startpos:") + err.what()) << endl;
        }
      }
    }
  }

  for(size_t s = 0; s<startPoses.size(); s++) {
    const Sgf::PositionSample& startPos = startPoses[s];
    if(startPos.weight < minWeight)
      continue;
    if(idxToView >= 0 && s != idxToView)
      continue;

    Board board = startPos.board;
    Player pla = startPos.nextPla;
    BoardHistory hist;
    hist.clear(board,pla,rules,0);
    hist.setInitialTurnNumber(startPos.initialTurnNumber);

    bool allLegal = true;
    for(size_t i = 0; i<startPos.moves.size(); i++) {
      bool isLegal = hist.makeBoardMoveTolerant(board,startPos.moves[i].loc,startPos.moves[i].pla,false);
      if(!isLegal) {
        allLegal = false;
        break;
      }
      pla = getOpp(startPos.moves[i].pla);
    }
    if(!allLegal) {
      if(checkLegality) {
        cout << "Illegal move in startpos in " + Global::concat(startPosesFiles,",") + ": " + Sgf::PositionSample::toJsonLine(startPos) << endl;
        continue;
      }
      else
        throw StringError("Illegal move in startpos: " + Sgf::PositionSample::toJsonLine(startPos));
    }

    if(checkLegality) {
      if(startPos.moves.size() > 0 && startPos.moves[0].pla != startPos.nextPla) {
        cout << "Mismatching nextPla in startpos in " + Global::concat(startPosesFiles,",") + ": " + Sgf::PositionSample::toJsonLine(startPos) << endl;
      }
    }

    if(autoKomi && bot != NULL) {
      const int64_t numVisits = 10;
      OtherGameProperties props;
      PlayUtils::adjustKomiToEven(bot->getSearchStopAndWait(),NULL,board,hist,pla,numVisits,props,rand);
    }

    Loc hintLoc = startPos.hintLoc;
    if(checkLegality) {
      if(hintLoc != Board::NULL_LOC) {
        bool isLegal = hist.isLegal(board,hintLoc,pla);
        if(!isLegal) {
          cout << "Illegal hint in startpos in " + Global::concat(startPosesFiles,",") + ": " + Sgf::PositionSample::toJsonLine(startPos) << endl;
          Board::printBoard(cout, board, hintLoc, &(hist.moveHistory));
          continue;
        }
      }
    }

    if(bot != NULL || !checkLegality) {
      cout << "StartPos: " << s << "/" << startPoses.size() << "\n";
      cout << "Next pla: " << PlayerIO::playerToString(pla) << "\n";
      cout << "Weight: " << startPos.weight << "\n";
      cout << "TrainingWeight: " << startPos.trainingWeight << "\n";
      cout << "StartPosInitialNextPla: " << PlayerIO::playerToString(startPos.nextPla) << "\n";
      cout << "StartPosMoves: ";
      for(int i = 0; i<(int)startPos.moves.size(); i++)
        cout << (startPos.moves[i].pla == P_WHITE ? "w" : "b") << Location::toString(startPos.moves[i].loc,board) << " ";
      cout << "\n";
      cout << "Auto komi: " << hist.rules.komi << "\n";
      Board::printBoard(cout, board, hintLoc, &(hist.moveHistory));
      cout << endl;

      if(bot != NULL) {
        bot->setPosition(pla,board,hist);
        if(hintLoc != Board::NULL_LOC)
          bot->setRootHintLoc(hintLoc);
        else
          bot->setRootHintLoc(Board::NULL_LOC);
        bot->genMoveSynchronous(bot->getSearch()->rootPla,TimeControls());
        const Search* search = bot->getSearchStopAndWait();
        PrintTreeOptions options;
        Player perspective = P_WHITE;
        search->printTree(cout, search->rootNode, options, perspective);
      }
    }
  }

  if(bot != NULL)
    delete bot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}


int MainCmds::checksgfhintpolicy(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfDirs;
  try {
    KataGoCommandLine cmd("Check policy for hint positions in sgfs");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",true,"DIR");
    cmd.add(sgfDirArg);
    cmd.parseArgs(args);

    nnModelFile = cmd.getModelFile();
    sgfDirs = sgfDirArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = 1;
    const int defaultMaxBatchSize = 8;
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirs(sgfDirs, sgfFiles);
  logger.write("Found " + Global::int64ToString((int64_t)sgfFiles.size()) + " sgf files!");

  int64_t numHintPositions = 0;
  double logPolicySum = 0.0;
  double logPolicyWeight = 0.0;

  for(size_t i = 0; i<sgfFiles.size(); i++) {
    Sgf* sgf = NULL;
    try {
      sgf = Sgf::loadFile(sgfFiles[i]);
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + sgfFiles[i] + ": " + e.what());
      continue;
    }

    std::set<Hash128> uniqueHashes;
    bool hashComments = true;
    bool hashParent = true;
    bool flipIfPassOrWFirst = false;
    bool allowGameOver = false;
    Rand rand;

    const std::vector<Rules> rulesToUse = {
      Rules::parseRules("chinese"),
      Rules::parseRules("japanese")
    };

    logger.write("Processing sgf: " + sgfFiles[i] + " hint positions " + Global::int64ToString(numHintPositions));
    sgf->iterAllUniquePositions(
      uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, &rand,
      [&](Sgf::PositionSample& posSample, const BoardHistory& hist, const string& comments) {
        if(comments.find("%HINT%") == string::npos)
          return;
        (void)hist; // Ignore, we want the position before the hint move

        if(!posSample.hasPreviousPositions(1))
          return;
        numHintPositions++;
        Sgf::PositionSample priorPosSample = posSample.previousPosition(1.0);

        for(const Rules& rules: rulesToUse) {
          Player nextPla;
          BoardHistory histBefore;
          bool suc = priorPosSample.tryGetCurrentBoardHistory(rules,nextPla,histBefore);
          testAssert(suc);
          Board board = histBefore.getRecentBoard(0);

          for(int symmetry = 0; symmetry < 8; symmetry++) {
            MiscNNInputParams nnInputParams;
            NNResultBuf buf;
            bool skipCache = true;
            bool includeOwnerMap = false;
            nnEval->evaluate(board,histBefore,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

            shared_ptr<NNOutput> nnOutput = std::move(buf.result);
            int pos = NNPos::locToPos(posSample.moves[posSample.moves.size()-1].loc, board.x_size, nnOutput->nnXLen, nnOutput->nnYLen);
            double policy = nnOutput->policyProbs[pos];
            logPolicySum += log(policy + 1e-30);
            logPolicyWeight += 1.0;
          }
        }
      }
    );

    delete sgf;
  }

  double averageLogPolicy = logPolicySum / logPolicyWeight;

  cout << "Total number of hint positions: " << numHintPositions << endl;
  cout << "Average log policy across all hints: " << averageLogPolicy << endl;

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  return 0;
}


int MainCmds::genposesfromselfplayinit(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  string outDir;
  int numPosesToWrite;
  int numProcessThreads;
  int maxPosesPerOutFile;

  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::ValueArg<int> numPosesToWriteArg("","poses","Number of poses to write",true,100,"N");
    TCLAP::ValueArg<int> numProcessThreadsArg("","threads","Number of threads",true,1,"THREADS");
    TCLAP::ValueArg<int> maxPosesPerOutFileArg("","max-poses-per-out-file","Number of hintposes per output file",false,100000,"INT");

    cmd.add(outDirArg);
    cmd.add(numPosesToWriteArg);
    cmd.add(numProcessThreadsArg);
    cmd.add(maxPosesPerOutFileArg);
    cmd.parseArgs(args);

    nnModelFile = cmd.getModelFile();
    outDir = outDirArg.getValue();
    numPosesToWrite = numPosesToWriteArg.getValue();
    numProcessThreads = numProcessThreadsArg.getValue();
    maxPosesPerOutFile = maxPosesPerOutFileArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outDir);

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.addFile(outDir + "/" + "log.log");
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);
  logger.write("Git revision " + Version::getGitRevision());

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_ANALYSIS);

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  {
    GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
    cfg.warnUnusedKeys(cerr,&logger);
    Setup::maybeWarnHumanSLParams(params,nnEval,NULL,cerr,&logger);
    delete gameRunner;
  }

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // ---------------------------------------------------------------------------------------------------
  const int sgfSplitIdx = 0;
    const int sgfSplitCount = 1;
  PosWriter posWriter("outposes.txt", outDir, sgfSplitCount, sgfSplitIdx, maxPosesPerOutFile);

  std::atomic<int64_t> nextPosIdx(0);
  auto genPosLoop = [&]() {
    Rand rand;
    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = nnModelFile;
    botSpec.nnEval = nnEval;
    botSpec.baseParams = SearchParams();
    botSpec.baseParams.maxVisits = 5;

    GameRunner* gameRunner = new GameRunner(cfg, Global::uint64ToString(rand.nextUInt64()), playSettings, logger);
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    while(true) {
      if(shouldStop.load(std::memory_order_acquire))
        break;
      int64_t posIdx = nextPosIdx.fetch_add(1, std::memory_order_acq_rel);
      if(posIdx >= numPosesToWrite)
        break;

      string seed = Global::uint64ToString(rand.nextUInt64());
      ForkData* forkData = new ForkData();
      FinishedGameData* data = gameRunner->runGame(seed, botSpec, botSpec, forkData, NULL, logger, shouldStopFunc, shouldPause, nullptr, nullptr, nullptr);

      Sgf::PositionSample sampleToWrite;
      Sgf::PositionSample::writePosOfHist(sampleToWrite, data->startHist, data->startPla);

      // Random symmetry
      int symmetry = (int)rand.nextInt(0,7);
      Board symBoard = SymmetryHelpers::getSymBoard(sampleToWrite.board, symmetry);
      sampleToWrite.board = symBoard;
      for(size_t i = 0; i<sampleToWrite.moves.size(); i++) {
        sampleToWrite.moves[i].loc = SymmetryHelpers::getSymLoc(sampleToWrite.moves[i].loc, sampleToWrite.board, symmetry);
      }
      sampleToWrite.hintLoc = SymmetryHelpers::getSymLoc(sampleToWrite.hintLoc, sampleToWrite.board, symmetry);

      posWriter.writePos(sampleToWrite);

      delete data;
      delete forkData;
    }

    delete gameRunner;
  };

  // ---------------------------------------------------------------------------------------------------

  //Begin writing
  posWriter.start();

  vector<std::thread> threads;
  for(int i = 0; i<numProcessThreads; i++) {
    threads.push_back(std::thread(genPosLoop));
  }

  for(size_t i = 0; i<threads.size(); i++)
    threads[i].join();

  logger.write("Waiting for final writing and cleanup");
  posWriter.flushAndStop();

  logger.write(nnEval->getModelFileName());
  logger.write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
  logger.write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
  logger.write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));

  logger.write("All done");

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
