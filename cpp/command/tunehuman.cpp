#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/sgfmetadata.h"
#include "../dataio/trainingwrite.h"
#include "../search/search.h"
#include "../search/searchparams.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../program/playsettings.h"
#include "../program/humansltuner.h"
#include "../command/commandline.h"
#include "../main.h"

#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

using namespace std;

int MainCmds::tunehuman(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string baselineConfigPath;
  string profile;
  double targetElo = 0.0;
  string outputConfigPath;
  string modelFile;
  string humanModelFile;
  double eloTol = 25.0;
  int gamesPerRound = 32;
  int maxRounds = 24;
  int numGameThreadsArgVal = -1;
  string seedStr = "tunehuman";
  string resumeFile;
  int searchVisits = -1;
  int maxVisitsCap = -1;
  double piklFloor = 0.02;
  double piklMax = 1.0e4;
  double dtauMax = 0.6;
  double candHumanRootExplore = -1.0;
  double xLo = 0.0;
  double xHi = 3.0;
  double komi = 7.5;
  string candColor = "auto";

  try {
    KataGoCommandLine cmd("Tune human-SL play parameters to hit a target ELO offset vs a baseline config.");
    cmd.addModelFileArg();
    cmd.addHumanModelFileArg();
    TCLAP::ValueArg<string> baselineConfigArg("","baseline-config","Baseline human-SL config (defines ELO 0).",true,"","FILE");
    TCLAP::ValueArg<string> profileArg("","profile","Candidate humanSLProfile, e.g. preaz_8d.",true,"","PROFILE");
    TCLAP::ValueArg<double> targetEloArg("","target-elo","Desired (candidate - baseline) ELO. Negative = weaker.",true,0.0,"ELO");
    TCLAP::ValueArg<string> outputConfigArg("","output-config","Where to write the tuned config.",true,"","FILE");
    TCLAP::ValueArg<double> eloTolArg("","elo-tol","Stop when 1-sigma CI half-width (ELO) <= this.",false,25.0,"ELO");
    TCLAP::ValueArg<int> gamesPerRoundArg("","games-per-round","Games per dial value per round.",false,32,"N");
    TCLAP::ValueArg<int> maxRoundsArg("","max-rounds","Hard cap on rounds.",false,24,"N");
    TCLAP::ValueArg<int> numGameThreadsArg("","num-game-threads","Parallel games within a round.",false,-1,"N");
    TCLAP::ValueArg<string> seedArg("","seed","Master seed for reproducibility.",false,"tunehuman","SEED");
    TCLAP::ValueArg<string> resumeFileArg("","resume-file","Per-round checkpoint file for resumable calibration. Empty = auto (<output-config>.samples).",false,"","FILE");
    TCLAP::ValueArg<int> searchVisitsArg("","search-visits","Visits in the piklLambda segment (>=2). -1 = auto (anchor to baseline maxVisits).",false,-1,"N");
    TCLAP::ValueArg<int> maxVisitsCapArg("","max-visits-cap","Visits at the strong end. -1 = auto (anchor to baseline maxVisits).",false,-1,"N");
    TCLAP::ValueArg<double> piklFloorArg("","pikl-floor","Smallest piklLambda (strongest).",false,0.02,"F");
    TCLAP::ValueArg<double> piklMaxArg("","pikl-max","Largest active piklLambda.",false,1.0e4,"F");
    TCLAP::ValueArg<double> dtauMaxArg("","dtau-max","Max temperature offset at the weak end.",false,0.6,"F");
    TCLAP::ValueArg<double> candHumanRootExploreArg("","cand-human-root-explore","Override the CANDIDATE's humanSLRootExploreProbWeightless (lower = less human-policy exploration = stronger). -1 = use baseline config's value.",false,-1.0,"F");
    TCLAP::ValueArg<double> xLoArg("","x-lo","Low end of the strength coordinate search range.",false,0.0,"F");
    TCLAP::ValueArg<double> xHiArg("","x-hi","High end of the strength coordinate search range.",false,3.0,"F");
    TCLAP::ValueArg<double> komiArg("","komi","Komi for the games. Use 0.5 for a KGS 1-rank handicap (stronger=White gets no compensation).",false,7.5,"F");
    TCLAP::ValueArg<string> candColorArg("","cand-color","Candidate's color: auto (alternate, removes color bias), black, or white. Use 'black' with -komi 0.5 for a 1-rank handicap match (weaker candidate as Black).",false,"auto","COLOR");
    cmd.add(baselineConfigArg);
    cmd.add(profileArg);
    cmd.add(targetEloArg);
    cmd.add(outputConfigArg);
    cmd.add(eloTolArg);
    cmd.add(gamesPerRoundArg);
    cmd.add(maxRoundsArg);
    cmd.add(numGameThreadsArg);
    cmd.add(seedArg);
    cmd.add(resumeFileArg);
    cmd.add(searchVisitsArg);
    cmd.add(maxVisitsCapArg);
    cmd.add(piklFloorArg);
    cmd.add(piklMaxArg);
    cmd.add(dtauMaxArg);
    cmd.add(candHumanRootExploreArg);
    cmd.add(xLoArg);
    cmd.add(xHiArg);
    cmd.add(komiArg);
    cmd.add(candColorArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    humanModelFile = cmd.getHumanModelFile();
    baselineConfigPath = baselineConfigArg.getValue();
    profile = profileArg.getValue();
    targetElo = targetEloArg.getValue();
    outputConfigPath = outputConfigArg.getValue();
    eloTol = eloTolArg.getValue();
    gamesPerRound = gamesPerRoundArg.getValue();
    maxRounds = maxRoundsArg.getValue();
    numGameThreadsArgVal = numGameThreadsArg.getValue();
    seedStr = seedArg.getValue();
    resumeFile = resumeFileArg.getValue();
    searchVisits = searchVisitsArg.getValue();
    maxVisitsCap = maxVisitsCapArg.getValue();
    piklFloor = piklFloorArg.getValue();
    piklMax = piklMaxArg.getValue();
    dtauMax = dtauMaxArg.getValue();
    candHumanRootExplore = candHumanRootExploreArg.getValue();
    xLo = xLoArg.getValue();
    xHi = xHiArg.getValue();
    komi = komiArg.getValue();
    candColor = candColorArg.getValue();
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  catch(const StringError& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  // ---- validation ----
  if(humanModelFile.empty()) { cerr << "Error: -human-model is required." << endl; return 1; }
  if(!FileUtils::exists(baselineConfigPath)) { cerr << "Error: baseline-config not found: " << baselineConfigPath << endl; return 1; }
  if(!FileUtils::exists(modelFile)) { cerr << "Error: model not found: " << modelFile << endl; return 1; }
  if(!FileUtils::exists(humanModelFile)) { cerr << "Error: human-model not found: " << humanModelFile << endl; return 1; }
  if(gamesPerRound < 1) { cerr << "Error: -games-per-round must be >= 1." << endl; return 1; }
  if(xLo >= xHi) { cerr << "Error: -x-lo must be < -x-hi." << endl; return 1; }
  if(candColor != "auto" && candColor != "black" && candColor != "white") {
    cerr << "Error: -cand-color must be auto, black, or white." << endl; return 1;
  }
  if(eloTol <= 0.0) { cerr << "Error: -elo-tol must be > 0." << endl; return 1; }
  if(searchVisits != -1 && searchVisits < 2) { cerr << "Error: -search-visits must be >= 2 (piklLambda needs >1 visit), or -1 for auto." << endl; return 1; }
  if(maxVisitsCap != -1 && maxVisitsCap < 1) { cerr << "Error: -max-visits-cap must be >= 1, or -1 for auto." << endl; return 1; }
  if(maxRounds < 1) { cerr << "Error: -max-rounds must be >= 1." << endl; return 1; }
  if(maxRounds < 4)
    cout << "WARNING: -max-rounds " << maxRounds << " < 4: calibration needs at least 4 rounds to"
         << " reach 'converged' (it requires 4 distinct dial samples). It will still run and write"
         << " a best-achievable config, but converged will be false." << endl;

  int numGameThreads = numGameThreadsArgVal > 0
    ? numGameThreadsArgVal
    : std::max(1, std::min(gamesPerRound, (int)std::thread::hardware_concurrency()));

  cout << "tunehuman parsed configuration:" << endl;
  cout << "  baseline-config = " << baselineConfigPath << endl;
  cout << "  profile         = " << profile << endl;
  cout << "  target-elo      = " << targetElo << endl;
  cout << "  output-config   = " << outputConfigPath << endl;
  cout << "  model           = " << modelFile << endl;
  cout << "  human-model     = " << humanModelFile << endl;
  cout << "  elo-tol         = " << eloTol << endl;
  cout << "  games-per-round = " << gamesPerRound << endl;
  cout << "  max-rounds      = " << maxRounds << endl;
  cout << "  num-game-threads= " << numGameThreads << endl;
  cout << "  seed            = " << seedStr << endl;
  cout << "  resume-file     = " << (resumeFile.empty() ? string("auto (<output-config>.samples)") : resumeFile) << endl;
  cout << "  search-visits   = " << (searchVisits < 0 ? string("auto") : Global::intToString(searchVisits)) << endl;
  cout << "  max-visits-cap  = " << (maxVisitsCap < 0 ? string("auto") : Global::intToString(maxVisitsCap)) << endl;
  cout << "  pikl-floor      = " << piklFloor << endl;
  cout << "  pikl-max        = " << piklMax << endl;
  cout << "  dtau-max        = " << dtauMax << endl;
  cout << "  cand-human-root-explore = " << (candHumanRootExplore < 0.0 ? string("(baseline)") : Global::doubleToString(candHumanRootExplore)) << endl;
  cout << "  x-lo / x-hi     = " << xLo << " / " << xHi << endl;
  cout << "  komi            = " << komi << endl;
  cout << "  cand-color      = " << candColor << (candColor == "auto" ? " (alternate)" : (komi != 7.5 ? " (handicap match)" : "")) << endl;

  // ---- load baseline config, logger, params, nets ----
  ConfigParser baselineCfg(baselineConfigPath);
  Logger logger(&baselineCfg, true, false, true, false); // log to stdout, with time, don't dump config

  const bool hasHumanModel = true;
  SearchParams baselineParams = Setup::loadSingleParams(baselineCfg, Setup::SETUP_FOR_GTP, hasHumanModel);
  string baselineText = baselineCfg.getContents();

  // ---- resolve the candidate visit budget, anchored to the baseline's own maxVisits ----
  // Honors "don't spend more compute than the baseline unless explicitly asked": with both
  // -search-visits and -max-visits-cap on auto (-1), the budget collapses onto baselineParams.maxVisits,
  // so the dial's visits never exceed the baseline. Visits rise above baseline only on explicit opt-in.
  VisitBudget vb = resolveVisitBudget(baselineParams.maxVisits, searchVisits, maxVisitsCap);
  string baselineDesc = vb.baselineHasCap ? Global::int64ToString(baselineParams.maxVisits) : string("uncapped");
  logger.write(
    "Resolved visit budget: mid(segment-B)=" + Global::intToString(vb.midVisits) +
    " cap(segment-C)=" + Global::intToString(vb.maxVisitsCap) +
    " (baseline maxVisits=" + baselineDesc + ")");
  if(!vb.baselineHasCap)
    logger.write("INFO: baseline config has no maxVisits cap (search bounded by time/playouts); "
                 "anchoring tuner visit budget to mid=" + Global::intToString(vb.midVisits) +
                 " cap=" + Global::intToString(vb.maxVisitsCap) + "." +
                 ((searchVisits == -1 || maxVisitsCap == -1)
                    ? string(" Pass -search-visits/-max-visits-cap to override.") : string("")));
  if(maxVisitsCap != -1 && maxVisitsCap < vb.midVisits)
    logger.write("WARNING: -max-visits-cap " + Global::intToString(maxVisitsCap) +
                 " < resolved search-visits " + Global::intToString(vb.midVisits) +
                 "; raised cap to " + Global::intToString(vb.midVisits) +
                 " to keep strength monotone in segment C.");
  if(vb.flooredFromBelow2)
    logger.write("NOTE: resolved segment-B visits were below 2 (piklLambda needs >=2 visits to act); "
                 "running segment B at the 2-visit minimum (baseline maxVisits=" + baselineDesc +
                 ", -search-visits=" + (searchVisits < 0 ? string("auto") : Global::intToString(searchVisits)) + ").");
  // Loud over-baseline warning: fire whenever the OPERATOR explicitly set a lever above the anchor the
  // budget is judged against (effectiveBaseline). This is independent of the mandatory sub-2 mid floor
  // (so an explicit big -max-visits-cap still warns even when -search-visits was floored), and covers
  // both finite-cap baselines and uncapped baselines (where the anchor is the legacy 100).
  bool userMidRaise = (searchVisits != -1) && (vb.midVisits > vb.effectiveBaseline);
  bool userCapRaise = (maxVisitsCap != -1) && (vb.maxVisitsCap > vb.effectiveBaseline);
  if(userMidRaise || userCapRaise)
    logger.write("WARNING: you explicitly set a visit budget (mid=" + Global::intToString(vb.midVisits) +
                 ", cap=" + Global::intToString(vb.maxVisitsCap) + ") above the " +
                 (vb.baselineHasCap
                    ? ("baseline maxVisits=" + baselineDesc)
                    : ("legacy anchor of " + Global::intToString(vb.effectiveBaseline) + " (baseline is uncapped)")) +
                 ". A weaker target may then cost MORE compute than the baseline, and a large visit count "
                 "significantly increases time per move. Omit -search-visits/-max-visits-cap to anchor to the baseline.");

  SearchParams candidateBaseParams = baselineParams;
  try {
    candidateBaseParams.humanSLProfile = SGFMetadata::getProfile(profile);
  }
  catch(const StringError& e) {
    cerr << "Error: invalid -profile '" << profile << "': " << e.what() << endl;
    return 1;
  }

  Rand seedRand(seedStr);
  int maxBotThreads = std::max(1, baselineParams.numThreads);
  int expectedConcurrentEvals = maxBotThreads * numGameThreads;
  const int defaultMaxBatchSize = std::max(8, ((expectedConcurrentEvals + 3) / 4) * 4);
  const bool defaultRequireExactNNLen = true; // fixed 19x19
  const bool disableFP16 = false;
  const string expectedSha256 = "";
  const int boardLen = 19;

  NNEvaluator* mainNNEval = Setup::initializeNNEvaluator(
    modelFile, modelFile, expectedSha256, baselineCfg, logger, seedRand, expectedConcurrentEvals,
    boardLen, boardLen, defaultMaxBatchSize, defaultRequireExactNNLen, disableFP16, Setup::SETUP_FOR_GTP);
  logger.write("Loaded main net");

  NNEvaluator* humanNNEval = Setup::initializeNNEvaluator(
    humanModelFile, humanModelFile, expectedSha256, baselineCfg, logger, seedRand, expectedConcurrentEvals,
    boardLen, boardLen, defaultMaxBatchSize, defaultRequireExactNNLen, disableFP16, Setup::SETUP_FOR_GTP);
  logger.write("Loaded human SL net");
  if(!humanNNEval->requiresSGFMetadata())
    logger.write("WARNING: -human-model was not trained from SGF metadata; profile may have no effect.");

  // ---- minimal game-setup config (rules/board/komi only; bot strength comes from BotSpec) ----
  // Inherit the board ruleset from the baseline config (a deployed gtp_human<rank>.cfg, e.g.
  // "rules = japanese") so tuning games are scored EXACTLY like real play. Calibrating under a
  // different ruleset than the configs are deployed with would be an avoidable confound (area vs
  // territory scoring changes endgame play, and the human-SL net's KGS-rank conditioning is most
  // faithful under the ruleset its KGS training games used). Falls back to Japanese if unspecified.
  Rules gameRules = Rules::parseRules(baselineCfg.contains("rules") ? baselineCfg.getString("rules") : "japanese");
  logger.write("Tuning-game ruleset (inherited from baseline config): " + gameRules.toStringNoKomi());
  std::map<string,string> gameCfgMap = {
    {"koRules", Rules::writeKoRule(gameRules.koRule)},
    {"scoringRules", Rules::writeScoringRule(gameRules.scoringRule)},
    {"taxRules", Rules::writeTaxRule(gameRules.taxRule)},
    {"multiStoneSuicideLegals", gameRules.multiStoneSuicideLegal ? "true" : "false"},
    {"hasButtons", gameRules.hasButton ? "true" : "false"},
    {"bSizes", "19"},
    {"bSizeRelProbs", "1"},
    {"komiMean", Global::doubleToString(komi)},
    {"komiStdev", "0.0"},
    {"komiAllowIntegerProb", "0.0"},
    {"logSearchInfo", "false"},
    {"logMoves", "false"},
    {"maxMovesPerGame", "1200"},
  };
  ConfigParser gameCfg(gameCfgMap);
  PlaySettings playSettings; // default: forSelfPlay=false, allowResignation=false, no fork/cheap/reduce
  GameRunner* gameRunner = new GameRunner(gameCfg, playSettings, logger);

  // ---- dial config + target ----
  StrengthDialConfig dialConfig;
  dialConfig.piklFloor = piklFloor;
  dialConfig.piklMax = piklMax;
  dialConfig.searchVisits = vb.midVisits;
  dialConfig.maxVisitsCap = vb.maxVisitsCap;
  dialConfig.dtauMax = dtauMax;

  // When segment C is flat (cap == mid, the auto outcome), the strong third of the dial [2,3] collapses
  // to a single indistinguishable point. effectiveXHi restricts calibration to [xLo, 2.0] so we neither
  // waste rounds on that plateau nor let an unreachable-strong target settle mid-plateau and dodge the
  // boundary warning.
  double effXHi = effectiveXHi(vb, xLo, xHi);
  if(effXHi < xHi)
    logger.write("INFO: strong-end visit budget equals mid (segment C is flat at " +
                 Global::intToString(vb.midVisits) + " visits); restricting calibration to x in [" +
                 Global::doubleToString(xLo) + ", 2.0]. Raise -max-visits-cap above the baseline to "
                 "calibrate stronger play.");
  if(vb.maxVisitsCap == vb.midVisits && xLo >= 2.0)
    logger.write("WARNING: segment C is flat (cap == mid) and -x-lo " + Global::doubleToString(xLo) +
                 " >= 2.0, so the entire calibration range lies on the flat strong plateau; every dial maps "
                 "to identical play and calibration cannot discriminate strength. Raise -max-visits-cap above "
                 "the baseline, or lower -x-lo below 2.0.");

  const double TEMP_CAP = 1.0;
  auto clipTemp = [TEMP_CAP](double v) { return v < 0.0 ? 0.0 : (v > TEMP_CAP ? TEMP_CAP : v); };
  double targetWinrate = 1.0 / (1.0 + std::pow(10.0, -targetElo / 400.0));

  // ---- playAt(x): set candidate dials, play gamesPerRound games candidate-vs-baseline ----
  int roundCounter = 0;
  auto playAt = [&](double x) -> std::pair<double,int> {
    int round = roundCounter++;
    StrengthDialParams dials = strengthDialToParams(x, dialConfig);

    SearchParams cand = candidateBaseParams;
    cand.humanSLChosenMovePiklLambda = dials.piklLambda;
    cand.maxVisits = dials.maxVisits;
    cand.chosenMoveTemperature = clipTemp(baselineParams.chosenMoveTemperature + dials.deltaTau);
    cand.chosenMoveTemperatureEarly = clipTemp(baselineParams.chosenMoveTemperatureEarly + dials.deltaTau);
    // Optional: strengthen the candidate by reducing its human-policy SEARCH exploration (the piklLambda
    // lever only affects move SELECTION; the ~100-ELO preaz_9d-vs-rank_9d gap lives in which moves get
    // explored). Lower = less human exploration = closer to pure main-net search = stronger.
    if(candHumanRootExplore >= 0.0)
      cand.humanSLRootExploreProbWeightless = candHumanRootExplore;

    std::atomic<int> nextGameIdx(0);
    double candidateWins = 0.0;
    int countedGames = 0;
    std::mutex tallyMutex;

    auto worker = [&]() {
      while(true) {
        int gameIdx = nextGameIdx.fetch_add(1);
        if(gameIdx >= gamesPerRound)
          break;
        bool candIsBlack = (candColor == "black") ? true
                         : (candColor == "white") ? false
                         : (gameIdx % 2 == 0);   // auto: alternate to remove color bias
        string seed = seedStr + ":r" + Global::intToString(round) + ":g" + Global::intToString(gameIdx);

        MatchPairer::BotSpec specCand;
        specCand.botIdx = 0; specCand.botName = "cand";
        specCand.nnEval = mainNNEval; specCand.humanEval = humanNNEval;
        specCand.baseParams = cand;
        MatchPairer::BotSpec specBase;
        specBase.botIdx = 1; specBase.botName = "base";
        specBase.nnEval = mainNNEval; specBase.humanEval = humanNNEval;
        specBase.baseParams = baselineParams;

        const MatchPairer::BotSpec& specB = candIsBlack ? specCand : specBase;
        const MatchPairer::BotSpec& specW = candIsBlack ? specBase : specCand;

        std::function<bool()> shouldStop = []() { return false; };
        std::function<void(const MatchPairer::BotSpec&, Search*)> noopAfterInit =
          [](const MatchPairer::BotSpec&, Search*) {};

        FinishedGameData* g = gameRunner->runGame(
          seed, specB, specW, NULL, NULL, logger,
          shouldStop, nullptr, nullptr, noopAfterInit, nullptr);
        if(g == NULL)
          continue;

        bool counted = true;
        double winInc = 0.0;
        if(g->endHist.isNoResult) {
          counted = false;
        } else {
          Player winner = g->endHist.winner;
          Player candPlayerColor = candIsBlack ? P_BLACK : P_WHITE;
          if(winner == C_EMPTY) winInc = 0.5;          // draw
          else winInc = (winner == candPlayerColor) ? 1.0 : 0.0;
        }
        delete g;

        if(counted) {
          std::lock_guard<std::mutex> lock(tallyMutex);
          candidateWins += winInc;
          countedGames += 1;
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(numGameThreads);
    for(int t = 0; t < numGameThreads; t++)
      threads.emplace_back(worker);
    for(size_t t = 0; t < threads.size(); t++)
      threads[t].join();

    return std::make_pair(candidateWins, countedGames);
  };

  // ---- progress logging per round ----
  auto onRound = [&](int round, double xStar, double eloSe, int distinctXs, int totalGames) {
    StrengthDialParams d = strengthDialToParams(xStar, dialConfig);
    logger.write(
      "Round " + Global::intToString(round) +
      ": x*=" + Global::doubleToString(xStar) +
      " eloSe=" + Global::doubleToString(eloSe) +
      " distinctX=" + Global::intToString(distinctXs) +
      " games=" + Global::intToString(totalGames) +
      " dial[piklLambda=" + Global::doubleToString(d.piklLambda) +
      " maxVisits=" + Global::intToString(d.maxVisits) +
      " deltaTau=" + Global::doubleToString(d.deltaTau) + "]");
  };

  // ---- resume support: a per-round checkpoint so an interrupted run continues instead of restarting ----
  // The checkpoint stores each round's (x, wins, games). A signature header guards against pooling samples
  // from a different matchup/dial: only the fields that define the candidate-vs-baseline winrate at a given
  // dial x are included (NOT target/tol/range/seed, which affect only sampling and stopping).
  string resumeFilePath = resumeFile.empty() ? (outputConfigPath + ".samples") : resumeFile;
  string resumeHeader =
    string("# tunehuman-samples v1") +
    " profile=" + profile +
    " model=" + modelFile +
    " human=" + humanModelFile +
    " baseline=" + baselineConfigPath +
    " mid=" + Global::intToString(vb.midVisits) +
    " cap=" + Global::intToString(vb.maxVisitsCap) +
    " piklFloor=" + Global::doubleToString(piklFloor) +
    " piklMax=" + Global::doubleToString(piklMax) +
    " dtau=" + Global::doubleToString(dtauMax);

  std::vector<CalibrationSample> initialSamples;
  if(FileUtils::exists(resumeFilePath)) {
    ifstream in(resumeFilePath);
    if(!in.good()) { cerr << "Error: cannot open resume-file for reading: " << resumeFilePath << endl; return 1; }
    string line;
    bool headerSeen = false;
    int lineNum = 0;
    while(std::getline(in, line)) {
      lineNum++;
      string t = Global::trim(line);
      if(t.empty())
        continue;
      if(t[0] == '#') {
        if(!headerSeen) {
          if(t != resumeHeader) {
            cerr << "Error: resume-file " << resumeFilePath << " was written for a different configuration.\n"
                 << "  found:    " << t << "\n"
                 << "  expected: " << resumeHeader << "\n"
                 << "Remove it to start fresh, or pass a different -resume-file." << endl;
            return 1;
          }
          headerSeen = true;
        }
        continue;
      }
      // Tolerate a malformed line (warn + skip) rather than fail: a hard kill mid-append can leave a
      // truncated final line, and a fatal parse would then permanently block resume. Bad numeric tokens
      // are skipped the same way (tryStringToDouble never throws).
      std::vector<string> parts = Global::split(t, ' ');
      CalibrationSample s;
      if(parts.size() != 3 ||
         !Global::tryStringToDouble(parts[0], s.x) ||
         !Global::tryStringToDouble(parts[1], s.wins) ||
         !Global::tryStringToDouble(parts[2], s.games)) {
        logger.write("WARNING: skipping malformed resume-file line " + Global::intToString(lineNum) + " in " +
                     resumeFilePath + " (likely a partial write from an interrupted run): '" + line + "'");
        continue;
      }
      // Semantic gate: a kill mid-write can truncate the LAST token (games) into a still-parseable but
      // wrong value (e.g. wins > games), which would silently poison the fit. Every clean round has
      // games >= 1 and 0 <= wins <= games, so this never rejects a valid sample -- only a corrupt one.
      if(!std::isfinite(s.x) || !std::isfinite(s.wins) || !std::isfinite(s.games) ||
         s.games < 1.0 || s.wins < -1e-9 || s.wins > s.games + 1e-9) {
        logger.write("WARNING: skipping out-of-range resume-file line " + Global::intToString(lineNum) + " in " +
                     resumeFilePath + " (likely a partial write from an interrupted run): '" + line + "'");
        continue;
      }
      initialSamples.push_back(s);
    }
    in.close();
    if(!headerSeen) {
      if(!initialSamples.empty()) {
        cerr << "Error: resume-file " << resumeFilePath << " has samples but no recognizable signature header." << endl;
        return 1;
      }
      // File exists but is empty/headerless (0-byte file, or a header write killed mid-flush by the
      // runtime cap). Recreate the header so the per-round appends below land in a well-formed file --
      // otherwise the NEXT restart would see samples-without-header and fatally refuse to resume.
      ofstream hdrOut(resumeFilePath, std::ios::trunc);
      if(!hdrOut.good()) { cerr << "Error: cannot (re)create resume-file: " << resumeFilePath << endl; return 1; }
      hdrOut << resumeHeader << "\n";
      hdrOut.close();
      logger.write("Resume-file " + resumeFilePath + " had no header (empty/partial write); recreated it.");
    }
    logger.write("Resuming calibration from " + Global::intToString((int)initialSamples.size()) +
                 " checkpointed round(s) in " + resumeFilePath + ".");
  }
  else {
    ofstream hdrOut(resumeFilePath);
    if(!hdrOut.good()) { cerr << "Error: cannot create resume-file: " << resumeFilePath << endl; return 1; }
    hdrOut << resumeHeader << "\n";
    hdrOut.close();
    logger.write("Checkpointing each round to " + resumeFilePath + " (resumable across restarts).");
  }

  // Durable per-round append: open/flush/close each call so a hard kill mid-run can't lose a completed round.
  auto onSampleCollected = [&](double x, double wins, double games) {
    ofstream app(resumeFilePath, std::ios::app);
    app << Global::doubleToStringHighPrecision(x) << " "
        << Global::doubleToStringHighPrecision(wins) << " "
        << Global::doubleToStringHighPrecision(games) << "\n";
    app.close();
  };

  // ---- run calibration ----
  uint64_t rngSeed = (uint64_t)std::hash<std::string>()(seedStr);
  CalibrationResult result = calibrateToTarget(
    playAt, xLo, effXHi, targetWinrate, gamesPerRound, maxRounds, eloTol, rngSeed, 0.5, onRound,
    initialSamples, onSampleCollected);

  // ---- compute final dials + fitted ELO ----
  StrengthDialParams finalDials = strengthDialToParams(result.xStar, dialConfig);
  double tempBase = clipTemp(baselineParams.chosenMoveTemperature + finalDials.deltaTau);
  double tempEarly = clipTemp(baselineParams.chosenMoveTemperatureEarly + finalDials.deltaTau);
  double fittedWinrate = result.model.predict(result.xStar);
  double fittedElo = 400.0 * std::log10(fittedWinrate / (1.0 - fittedWinrate));
  bool reachedBoundary = (result.xStar <= xLo + 1e-6) || (result.xStar >= effXHi - 1e-6);

  // ---- build header + overridden config text ----
  std::ostringstream hdr;
  hdr << "# Tuned by `katago tunehuman`.\n";
  hdr << "# baseline-config : " << baselineConfigPath << "\n";
  hdr << "# profile         : " << profile << "\n";
  hdr << "# models          : " << modelFile << " / " << humanModelFile << "\n";
  hdr << "# target-elo      : " << targetElo << "   (targetWinrate " << targetWinrate << " vs baseline)\n";
  hdr << "# achieved        : fitted " << fittedElo << " ELO  +/- " << result.eloSe
      << " (1-sigma),  over " << result.totalGames << " games, " << result.rounds
      << " rounds, converged=" << (result.converged ? "yes" : "no") << "\n";
  hdr << "# dial            : x*=" << result.xStar << "  piklLambda=" << finalDials.piklLambda
      << "  maxVisits=" << finalDials.maxVisits << "  deltaTau=" << finalDials.deltaTau << "\n";
  hdr << "# seed            : " << seedStr << "\n";
  if(reachedBoundary) {
    hdr << "# WARNING: target ELO not reachable within the dial range; best-achievable shown.\n";
    hdr << "#          Widen -max-visits-cap / -dtau-max / -x-lo / -x-hi to extend the range.\n";
    logger.write("WARNING: target ELO not reachable within dial range; wrote best-achievable config (x* at boundary).");
  }
  hdr << "\n";

  std::vector<std::pair<string,string>> overrides;
  overrides.push_back(std::make_pair("humanSLProfile", profile));
  overrides.push_back(std::make_pair("humanSLChosenMovePiklLambda", Global::doubleToString(finalDials.piklLambda)));
  overrides.push_back(std::make_pair("maxVisits", Global::intToString(finalDials.maxVisits)));
  overrides.push_back(std::make_pair("chosenMoveTemperature", Global::doubleToString(tempBase)));
  overrides.push_back(std::make_pair("chosenMoveTemperatureEarly", Global::doubleToString(tempEarly)));
  if(candHumanRootExplore >= 0.0)
    overrides.push_back(std::make_pair("humanSLRootExploreProbWeightless", Global::doubleToString(candHumanRootExplore)));

  string finalText = hdr.str() + overrideConfigText(baselineText, overrides);

  ofstream out;
  FileUtils::open(out, outputConfigPath);
  out << finalText;
  out.close();

  logger.write(
    "Wrote tuned config to " + outputConfigPath +
    " (fitted " + Global::doubleToString(fittedElo) + " ELO +/- " + Global::doubleToString(result.eloSe) +
    ", " + Global::intToString(result.totalGames) + " games, " + Global::intToString(result.rounds) +
    " rounds, converged=" + (result.converged ? "yes" : "no") + ")");

  delete gameRunner;
  delete mainNNEval;
  delete humanNNEval;
  return 0;
}
