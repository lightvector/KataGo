#include "../tests/tests.h"

#include "../core/fileutils.h"
#include "../neuralnet/nneval.h"
#include "../dataio/sgf.h"

#include "../external/nlohmann_json/json.hpp"

//------------------------
#include "../core/using.h"
//------------------------
using json = nlohmann::json;

// Absolute-output check of a backend against compiled-in reference data blended across a
// sampling of neural nets from the training run (see backendreferencedata.cpp for the data
// schema). Unlike runBackendErrorTest, whose reference is by default the same backend's own
// unbatched fp32 outputs, this can catch a backend that is systematically wrong in fp16 and
// fp32 alike, so the two are complementary.
//
// The reference blend is what a net from the run looks like, so only nets from the run are
// expected to pass. Human-imitation nets and nets trained for other special purposes deviate
// from the run's own trajectory by more than a backend bug would, and can fail on their own
// merits. Nets too small for any threshold to be meaningful are exempted outright rather than
// checked, see MIN_EFF_PARAMS_TO_CHECK.
//
// The detection floor is the spread of legitimate nets across the run on each metric, so
// thresholds are necessarily far looser than runBackendErrorTest's.
// Metrics and thresholds must stay valid for every net the run still uses, as nets improve:
//  - Policy KL(candidate || smoothed mixture): improving nets sharpen within the mixture's
//    support, so this stays bounded, and it fires when the backend puts mass on moves no
//    reference net ever considered.
//  - Binary KL of the candidate's winrate vs the reference average winrate.
//  - Mean absolute error vs the reference averages for lead, score mean, score stdev, and
//    (loosely) the shortterm winloss/score error predictions.
//  - Consensus top move: on positions where the mixture is confident, any reasonable net
//    assigns the consensus move substantial mass.
//  - Ownership MAE vs the mixture mean, bounded by a multiple of the max MAE that any
//    reference net itself achieved.
//
// Positions may individually override the policy optimism and pda used for their eval (the
// function arguments are just the defaults), and may cap the history planes via maxHistory.
// Backends implement the blend of the two policy output channels themselves, so varying
// optimism across positions exercises that blend.

struct BackendRefPosData {
  Sgf::PositionSample sample;
  Rules rules;
  double policyOptimism = 0.0;
  double pda = 0.0;
  int maxHistory = 1000;
  //Full averaged policy across reference nets, length xSize*ySize+1, row-major y*xSize+x with
  //pass as the last entry.
  std::vector<double> policyMixture;
  bool hasPolicyMixture = false;
  double winrateAvg = 0.0;
  bool hasWinrateAvg = false;
  double leadAvg = 0.0;
  bool hasLeadAvg = false;
  double scoreMeanAvg = 0.0;
  bool hasScoreMeanAvg = false;
  double scoreStdevAvg = 0.0;
  bool hasScoreStdevAvg = false;
  double stWinlossErrorAvg = 0.0;
  bool hasStWinlossErrorAvg = false;
  double stScoreErrorAvg = 0.0;
  bool hasStScoreErrorAvg = false;
  std::vector<double> ownership;
  bool hasOwnership = false;
  double ownershipAllowedMAE = 0.0;
};

// Base limits, calibrated August 2026 against 31 nets spanning 2020-2026: kata1-b6c96 through
// b28c512nbt, b40c768nbt and experimental transformers, including the worst and earliest net
// of each size class. Each was evaluated on the compiled-in positions against the compiled-in
// reference blend. Limits are set so that with the size lenience below, every legitimate net
// measured at most ~2/3 of each limit.
// Before checking, continuous limits are scaled up by the overall lenience (the size lenience
// below times the caller's lenienceFactor) and consensusMinProb is scaled down by it.
struct BackendRefLimits {
  double mixtureSmoothing = 1e-5;    // epsilon of uniform-over-legal mixed into the mixture
  double avgPolicyKL = 0.24;         // mean over positions of KL(candidate || smoothed mixture), nats
  double p95PolicyKL = 0.60;         // 95th percentile over positions
  double consensusTopProb = 0.80;    // mixture top prob >= this marks a consensus position
  double consensusMinProb = 0.10;    // candidate must assign at least this to the consensus move
  int maxConsensusViolations = 3;
  double avgWinrateKL = 0.040;       // binary KL of candidate winrate vs reference average, nats
  double p95WinrateKL = 0.19;
  //Lead, score stdev, and the shortterm error predictions get extra slack beyond the
  //calibration: they are canaries that don't drive play the way policy/winrate do, and the
  //second-order ones could shift with future training hyperparameter changes.
  double avgLeadAE = 1.8;            // abs error vs reference average, points
  double p95LeadAE = 5.3;
  double avgScoreMeanAE = 1.55;
  double p95ScoreMeanAE = 4.7;
  double avgScoreStdevAE = 5.1;
  double p95ScoreStdevAE = 9.9;
  double avgStWinlossErrorAE = 0.096; // abs error vs reference average shorttermWinlossError
  double p95StWinlossErrorAE = 0.33;
  double avgStScoreErrorAE = 1.3;    // abs error vs reference average shorttermScoreError, points
  double p95StScoreErrorAE = 3.9;
  //Ownership is checked via the ratio of the candidate's per-position MAE to the stored
  //per-position allowed MAE, which normalizes for how much positions legitimately vary
  //(settled positions have tiny cross-net spread, complex middlegames large), then
  //thresholded avg/p95 like the other metrics. The denominator is padded and floored:
  //fully-decided positions have allowed MAE as small as ~0.0004, which would hugely amplify
  //benign numerics like a tiny uniform offset on saturated outputs.
  double ownershipMAEPad = 0.005;
  double ownershipMAEFloor = 0.015;
  double avgOwnershipRatio = 1.35;
  double p95OwnershipRatio = 1.6;
};

// Smaller/weaker nets legitimately deviate more from the blended reference. Measured mean
// policy KL vs the blend rises smoothly as parameter count shrinks: ~1.4x at 47M params, ~5x
// at 10M, ~11x at 1M, relative to the largest nets. Nested bottleneck nets behave like larger
// plain nets per parameter, and transformers larger still, so their parameters are counted at
// a multiple. The exponent is deliberately a bit steeper than the measured scaling: small nets
// only play rating games rather than generating training data, so extra buffer against
// stochastic variation is cheap there, while nets at or above the current main-run size stay
// near lenience 1.
static double computeBackendRefEffParams(const NNEvaluator* nnEval) {
  double effParams = (double)nnEval->getNumModelParameters();
  if(nnEval->modelHasAnyNestedBottleneckBlocks())
    effParams *= 2.0;
  if(nnEval->modelHasAnyTransformerBlocks())
    effParams *= 2.0;
  return effParams;
}

static double computeBackendRefSizeLenience(const NNEvaluator* nnEval) {
  return std::max(1.0, pow(1.4e8 / std::max(1.0, computeBackendRefEffParams(nnEval)), 0.75));
}

//Size classes at or below b6c96 include the run's very first nets, barely better than random.
//No thresholds can both pass those and stay meaningful, so below this the checks are advisory.
static constexpr double MIN_EFF_PARAMS_TO_CHECK = 2e6;

static bool parseBackendRefScalar(const json& obj, const char* key, double& out) {
  if(obj.find(key) == obj.end() || obj[key].is_null())
    return false;
  out = obj[key].get<double>();
  return true;
}

static BackendRefPosData parseBackendRefPosData(const string& line, double defaultPolicyOptimism, double defaultPda) {
  BackendRefPosData data;
  json obj = json::parse(line);
  data.sample = Sgf::PositionSample::ofJsonLine(obj["sample"].dump());
  data.rules = Rules::parseRules(obj["rules"].get<string>());
  data.rules.komi = (float)obj["komi"].get<double>();
  data.policyOptimism = defaultPolicyOptimism;
  if(parseBackendRefScalar(obj,"policyOptimism",data.policyOptimism)) {
    if(!(data.policyOptimism >= 0.0 && data.policyOptimism <= 1.0))
      throw StringError("Backend reference data: policyOptimism out of range");
  }
  data.pda = defaultPda;
  if(parseBackendRefScalar(obj,"pda",data.pda)) {
    if(!(data.pda >= -4.0 && data.pda <= 4.0))
      throw StringError("Backend reference data: pda out of range");
  }
  if(obj.find("maxHistory") != obj.end() && !obj["maxHistory"].is_null()) {
    data.maxHistory = obj["maxHistory"].get<int>();
    if(data.maxHistory < 0 || data.maxHistory > 1000)
      throw StringError("Backend reference data: maxHistory out of range");
  }
  if(obj.find("policyMixture") != obj.end() && !obj["policyMixture"].is_null()) {
    data.policyMixture = obj["policyMixture"].get<std::vector<double>>();
    if(data.policyMixture.size() != (size_t)(data.sample.board.x_size * data.sample.board.y_size + 1))
      throw StringError("Backend reference data: policyMixture size does not match board size");
    for(const double& p: data.policyMixture) {
      if(!(p >= 0.0 && p <= 1.0 + 1e-6))
        throw StringError("Backend reference data: bad policyMixture prob");
    }
    data.hasPolicyMixture = true;
  }
  data.hasWinrateAvg = parseBackendRefScalar(obj,"winrateAvg",data.winrateAvg);
  data.hasLeadAvg = parseBackendRefScalar(obj,"leadAvg",data.leadAvg);
  data.hasScoreMeanAvg = parseBackendRefScalar(obj,"scoreMeanAvg",data.scoreMeanAvg);
  data.hasScoreStdevAvg = parseBackendRefScalar(obj,"scoreStdevAvg",data.scoreStdevAvg);
  data.hasStWinlossErrorAvg = parseBackendRefScalar(obj,"shorttermWinlossErrorAvg",data.stWinlossErrorAvg);
  data.hasStScoreErrorAvg = parseBackendRefScalar(obj,"shorttermScoreErrorAvg",data.stScoreErrorAvg);
  if(obj.find("ownership") != obj.end() && !obj["ownership"].is_null()) {
    if(obj.find("ownershipAllowedMAE") == obj.end() || obj["ownershipAllowedMAE"].is_null())
      throw StringError("Backend reference data: ownership provided without ownershipAllowedMAE");
    //Stored as white ownership times 100, integers in [-100,100].
    data.ownership = obj["ownership"].get<std::vector<double>>();
    data.ownershipAllowedMAE = obj["ownershipAllowedMAE"].get<double>();
    if(data.ownership.size() != (size_t)(data.sample.board.x_size * data.sample.board.y_size))
      throw StringError("Backend reference data: ownership size does not match board size");
    for(double& o: data.ownership) {
      if(!(o >= -100.0 && o <= 100.0))
        throw StringError("Backend reference data: ownership value out of range");
      o *= 0.01;
    }
    if(!(data.ownershipAllowedMAE > 0.0))
      throw StringError("Backend reference data: ownershipAllowedMAE must be positive");
    data.hasOwnership = true;
  }
  return data;
}

static double backendRefAvg(const std::vector<double>& vec) {
  if(vec.size() <= 0)
    return 0.0;
  double sum = 0;
  for(const double& x: vec)
    sum += x;
  return sum / (double)vec.size();
}
static double backendRefPercentile(std::vector<double> vec, double frac) {
  if(vec.size() <= 0)
    return 0.0;
  std::sort(vec.begin(),vec.end());
  return vec[(size_t)((double)(vec.size()-1) * frac)];
}
static double backendRefMax(const std::vector<double>& vec) {
  if(vec.size() <= 0)
    return 0.0;
  return *std::max_element(vec.begin(),vec.end());
}
static void backendRefReportStats(const string& name, const std::vector<double>& vec, Logger& logger) {
  auto rpad = [](const string& s, size_t n) {
    if(s.size() < n)
      return s + std::string(n - s.size(),' ');
    return s;
  };
  logger.write(
    rpad("backend reference " + name + ":", 46) +
    Global::strprintf(
      " %8.5f  %8.5f  %8.5f  %8.5f",
      backendRefAvg(vec), backendRefPercentile(vec,0.95), backendRefPercentile(vec,0.99), backendRefMax(vec)
    )
  );
}

bool Tests::runBackendReferenceTest(
  NNEvaluator* nnEval,
  Logger& logger,
  bool verbose,
  double policyOptimismForTest,
  double pdaForTest,
  double nnPolicyTemperatureForTest,
  double lenienceFactor,
  const string& referenceDataFileOverride,
  const string& dumpCandidateFileName
) {
  BackendRefLimits limits;
  const double lenience = computeBackendRefSizeLenience(nnEval) * lenienceFactor;
  const double consensusMinProbUsed = std::max(0.005, limits.consensusMinProb / lenience);

  std::vector<string> jsonLines;
  if(referenceDataFileOverride != "") {
    for(const string& line: FileUtils::readFileLines(referenceDataFileOverride,'\n')) {
      if(Global::trim(line) != "")
        jsonLines.push_back(line);
    }
    if(verbose)
      logger.write("Loaded " + Global::uint64ToString((uint64_t)jsonLines.size()) + " backend reference positions from: " + referenceDataFileOverride);
  }
  else
    jsonLines = TestCommon::getBackendReferenceJsonData();

  std::vector<BackendRefPosData> refData;
  for(const string& line: jsonLines)
    refData.push_back(parseBackendRefPosData(line, policyOptimismForTest, pdaForTest));

  if(refData.size() <= 0) {
    logger.write("Backend reference test: no reference positions, skipping");
    return true;
  }

  std::ofstream dumpOut;
  if(dumpCandidateFileName != "") {
    FileUtils::open(dumpOut,dumpCandidateFileName);
    if(!dumpOut)
      throw StringError("Unable to open dump file: " + dumpCandidateFileName);
  }

  // Per-position aggregates
  std::vector<double> policyKL;
  std::vector<double> winrateKL;
  std::vector<double> leadAE;
  std::vector<double> scoreMeanAE;
  std::vector<double> scoreStdevAE;
  std::vector<double> stWinlossErrorAE;
  std::vector<double> stScoreErrorAE;
  std::vector<double> consensusProbs;      // candidate prob on the consensus move, over consensus positions
  std::vector<double> ownershipMAERatio;   // candidate MAE / stored allowed MAE, over ownership positions
  int numConsensusViolations = 0;

  for(size_t posIdx = 0; posIdx < refData.size(); posIdx++) {
    const BackendRefPosData& data = refData[posIdx];

    //Featurize per the model's own declared BoardHistoryModes preferences. Nets with different modes
    //thus see slightly different inputs on the same position - part of the model-family spread the
    //calibrated thresholds must absorb.
    const BoardHistoryModes modelModes(
      nnEval->modelPreferPassAliveUnderSuicideRules(), nnEval->modelPreferExcludeTerritoryAdjacentToAtari());
    BoardHistory hist;
    Player nextPla = C_EMPTY;
    bool histOkay = data.sample.tryGetCurrentBoardHistory(data.rules, nextPla, hist, modelModes);
    if(!histOkay)
      throw StringError("Backend reference test: reference position " + Global::uint64ToString((uint64_t)posIdx) + " has illegal moves");
    const Board& board = hist.getRecentBoard(0);

    MiscNNInputParams nnInputParams;
    nnInputParams.symmetry = (int)(BoardHistory::getSituationRulesAndKoHash(board,hist,hist.presumedNextMovePla,0.5).hash0 & 7);
    nnInputParams.policyOptimism = data.policyOptimism;
    nnInputParams.playoutDoublingAdvantage = data.pda;
    nnInputParams.maxHistory = data.maxHistory;
    nnInputParams.nnPolicyTemperature = (float)nnPolicyTemperatureForTest;
    nnInputParams.passAliveSuicideRulesOverride = modelModes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
    nnInputParams.excludeTerritoryAdjAtariOverride = modelModes.excludeTerritoryAdjacentToAtari ? 1 : 0;

    NNResultBuf buf;
    const bool skipCache = true;
    const bool includeOwnerMap = true;
    SGFMetadata sgfMeta = SGFMetadata::getProfile("preaz_5k");
    nnEval->evaluate(board,hist,hist.presumedNextMovePla,&sgfMeta,nnInputParams,buf,skipCache,includeOwnerMap);
    const std::shared_ptr<NNOutput>& out = buf.result;

    int numLegal = 0;
    for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++) {
      if(out->policyProbs[i] >= 0)
        numLegal += 1;
    }
    testAssert(numLegal > 0);

    const int mixtureLen = board.x_size * board.y_size + 1;
    auto mixtureIdxOfLoc = [&](Loc loc) {
      if(loc == Board::PASS_LOC)
        return mixtureLen-1;
      return Location::getY(loc,board.x_size) * board.x_size + Location::getX(loc,board.x_size);
    };

    if(data.hasPolicyMixture) {
      const double eps = limits.mixtureSmoothing;
      double klSum = 0.0;
      for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++) {
        double c = out->policyProbs[i];
        if(c <= 1e-30)
          continue;
        Loc loc = NNPos::posToLoc(i, board.x_size, board.y_size, out->nnXLen, out->nnYLen);
        double m = data.policyMixture[mixtureIdxOfLoc(loc)];
        m = (1.0-eps)*m + eps/numLegal;
        klSum += c * (log(c) - log(m));
      }
      policyKL.push_back(klSum);

      int topIdx = 0;
      for(int i = 1; i<mixtureLen; i++) {
        if(data.policyMixture[i] > data.policyMixture[topIdx])
          topIdx = i;
      }
      if(data.policyMixture[topIdx] >= limits.consensusTopProb) {
        Loc topLoc = (topIdx == mixtureLen-1) ? Board::PASS_LOC : Location::getLoc(topIdx % board.x_size, topIdx / board.x_size, board.x_size);
        int candPos = out->getPos(topLoc,board);
        double candidateProb = out->policyProbs[candPos];
        if(candidateProb < 0)
          throw StringError("Backend reference test: consensus move is illegal on reference position " + Global::uint64ToString((uint64_t)posIdx));
        consensusProbs.push_back(candidateProb);
        if(candidateProb < consensusMinProbUsed)
          numConsensusViolations += 1;
      }
    }

    double winrate = 0.5*(1.0 + out->whiteWinProb - out->whiteLossProb);
    double scoreStdev = sqrt(std::max(0.0, (double)out->whiteScoreMeanSq - (double)out->whiteScoreMean*(double)out->whiteScoreMean));
    if(data.hasWinrateAvg) {
      auto clamp01 = [](double x) { return std::min(1.0-1e-6, std::max(1e-6, x)); };
      double r = clamp01(data.winrateAvg);
      double c = clamp01(winrate);
      winrateKL.push_back(r*log(r/c) + (1.0-r)*log((1.0-r)/(1.0-c)));
    }
    if(data.hasLeadAvg)
      leadAE.push_back(std::abs((double)out->whiteLead - data.leadAvg));
    if(data.hasScoreMeanAvg)
      scoreMeanAE.push_back(std::abs((double)out->whiteScoreMean - data.scoreMeanAvg));
    if(data.hasScoreStdevAvg)
      scoreStdevAE.push_back(std::abs(scoreStdev - data.scoreStdevAvg));
    //Model versions below 9 lack these heads entirely (the outputs would be softplus-of-zero
    //garbage rather than real predictions), so skip the checks for such models.
    if(data.hasStWinlossErrorAvg && nnEval->supportsShorttermError() && out->shorttermWinlossError >= 0)
      stWinlossErrorAE.push_back(std::abs((double)out->shorttermWinlossError - data.stWinlossErrorAvg));
    if(data.hasStScoreErrorAvg && nnEval->supportsShorttermError() && out->shorttermScoreError >= 0)
      stScoreErrorAE.push_back(std::abs((double)out->shorttermScoreError - data.stScoreErrorAvg));

    testAssert(out->whiteOwnerMap != NULL);
    if(data.hasOwnership) {
      double maeSum = 0.0;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          int pos = NNPos::xyToPos(x,y,out->nnXLen);
          maeSum += std::abs(data.ownership[y*board.x_size+x] - (double)out->whiteOwnerMap[pos]);
        }
      }
      double mae = maeSum / (board.x_size * board.y_size);
      ownershipMAERatio.push_back(mae / std::max(data.ownershipAllowedMAE + limits.ownershipMAEPad, limits.ownershipMAEFloor));
    }

    if(dumpCandidateFileName != "") {
      json dump;
      dump["posIdx"] = (int64_t)posIdx;
      dump["sample"] = json::parse(Sgf::PositionSample::toJsonLine(data.sample));
      dump["rules"] = data.rules.toStringNoKomi();
      dump["komi"] = data.rules.komi;
      //Effective values used, including ones that came from the defaults.
      dump["policyOptimism"] = data.policyOptimism;
      dump["pda"] = data.pda;
      dump["maxHistory"] = data.maxHistory;
      dump["symmetryUsed"] = nnInputParams.symmetry;
      dump["modelName"] = nnEval->getInternalModelName();
      dump["modelVersion"] = nnEval->getModelVersion();
      //Full policy in the same layout as policyMixture: row-major y*xSize+x, pass last.
      std::vector<double> policy(mixtureLen, 0.0);
      for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++) {
        double c = out->policyProbs[i];
        if(c < 0)
          continue;
        Loc loc = NNPos::posToLoc(i, board.x_size, board.y_size, out->nnXLen, out->nnYLen);
        policy[mixtureIdxOfLoc(loc)] = c;
      }
      dump["policy"] = policy;
      dump["whiteWinProb"] = out->whiteWinProb;
      dump["whiteLossProb"] = out->whiteLossProb;
      dump["whiteNoResultProb"] = out->whiteNoResultProb;
      dump["winrate"] = winrate;
      dump["whiteLead"] = out->whiteLead;
      dump["whiteScoreMean"] = out->whiteScoreMean;
      dump["scoreStdev"] = scoreStdev;
      //Null for models that lack these heads, so blending excludes them.
      if(nnEval->supportsShorttermError()) {
        dump["shorttermWinlossError"] = out->shorttermWinlossError;
        dump["shorttermScoreError"] = out->shorttermScoreError;
      }
      else {
        dump["shorttermWinlossError"] = nullptr;
        dump["shorttermScoreError"] = nullptr;
      }
      std::vector<double> ownership;
      for(int y = 0; y<board.y_size; y++)
        for(int x = 0; x<board.x_size; x++)
          ownership.push_back((double)out->whiteOwnerMap[NNPos::xyToPos(x,y,out->nnXLen)]);
      dump["ownership"] = ownership;
      dumpOut << dump.dump() << "\n";
    }
  }

  if(dumpCandidateFileName != "") {
    dumpOut.close();
    logger.write("Dumped candidate outputs for " + Global::uint64ToString((uint64_t)refData.size()) + " positions to: " + dumpCandidateFileName);
  }

  if(verbose) {
    logger.write(
      "Backend reference test: " + Global::uint64ToString((uint64_t)refData.size()) + " positions ("
      + Global::uint64ToString((uint64_t)policyKL.size()) + " policy mixture, "
      + Global::uint64ToString((uint64_t)consensusProbs.size()) + " consensus, "
      + Global::uint64ToString((uint64_t)winrateKL.size()) + " winrate, "
      + Global::uint64ToString((uint64_t)stWinlossErrorAE.size()) + " shortterm, "
      + Global::uint64ToString((uint64_t)ownershipMAERatio.size()) + " ownership) on model "
      + nnEval->getInternalModelName());
    logger.write(Global::strprintf(
      "Lenience: %.3f (size lenience %.3f x factor %.3f)",
      lenience, computeBackendRefSizeLenience(nnEval), lenienceFactor));
    logger.write("Reporting avg, 95%, 99%, max over positions:");
    backendRefReportStats("policyKL (nats)", policyKL, logger);
    backendRefReportStats("winrateKL (nats)", winrateKL, logger);
    backendRefReportStats("leadAE (points)", leadAE, logger);
    backendRefReportStats("scoreMeanAE (points)", scoreMeanAE, logger);
    backendRefReportStats("scoreStdevAE (points)", scoreStdevAE, logger);
    backendRefReportStats("stWinlossErrorAE", stWinlossErrorAE, logger);
    backendRefReportStats("stScoreErrorAE (points)", stScoreErrorAE, logger);
    backendRefReportStats("ownershipMAERatio", ownershipMAERatio, logger);
    if(consensusProbs.size() > 0)
      logger.write(Global::strprintf(
        "backend reference min consensus move prob: %8.5f  (%d of %d below %.5f, %d allowed)",
        *std::min_element(consensusProbs.begin(),consensusProbs.end()),
        numConsensusViolations, (int)consensusProbs.size(), consensusMinProbUsed, limits.maxConsensusViolations));
  }

  bool success = true;
  auto failCheck = [&](const string& msg) {
    //Always log failures even when not verbose, so that contribute logs record which check tripped.
    logger.write("Backend reference test failed check for " + nnEval->getInternalModelName() + ": " + msg);
    success = false;
  };
  auto checkLimit = [&](double value, double limit, const char* name) {
    if(!(value <= limit))
      failCheck(Global::strprintf("%s %.5f > limit %.5f", name, value, limit));
  };
  auto checkViolations = [&](int count, int maxAllowed, const char* name) {
    if(count > maxAllowed)
      failCheck(Global::strprintf("%s %d > allowed %d", name, count, maxAllowed));
  };

  auto p95 = [&](const std::vector<double>& v) { return backendRefPercentile(v, 0.95); };
  checkLimit(backendRefAvg(policyKL), limits.avgPolicyKL * lenience, "avg policyKL");
  checkLimit(p95(policyKL), limits.p95PolicyKL * lenience, "p95 policyKL");
  checkViolations(numConsensusViolations, limits.maxConsensusViolations, "consensus move violations");
  checkLimit(backendRefAvg(winrateKL), limits.avgWinrateKL * lenience, "avg winrateKL");
  checkLimit(p95(winrateKL), limits.p95WinrateKL * lenience, "p95 winrateKL");
  checkLimit(backendRefAvg(leadAE), limits.avgLeadAE * lenience, "lead MAE");
  checkLimit(p95(leadAE), limits.p95LeadAE * lenience, "p95 leadAE");
  checkLimit(backendRefAvg(scoreMeanAE), limits.avgScoreMeanAE * lenience, "scoreMean MAE");
  checkLimit(p95(scoreMeanAE), limits.p95ScoreMeanAE * lenience, "p95 scoreMeanAE");
  checkLimit(backendRefAvg(scoreStdevAE), limits.avgScoreStdevAE * lenience, "scoreStdev MAE");
  checkLimit(p95(scoreStdevAE), limits.p95ScoreStdevAE * lenience, "p95 scoreStdevAE");
  checkLimit(backendRefAvg(stWinlossErrorAE), limits.avgStWinlossErrorAE * lenience, "stWinlossError MAE");
  checkLimit(p95(stWinlossErrorAE), limits.p95StWinlossErrorAE * lenience, "p95 stWinlossErrorAE");
  checkLimit(backendRefAvg(stScoreErrorAE), limits.avgStScoreErrorAE * lenience, "stScoreError MAE");
  checkLimit(p95(stScoreErrorAE), limits.p95StScoreErrorAE * lenience, "p95 stScoreErrorAE");
  checkLimit(backendRefAvg(ownershipMAERatio), limits.avgOwnershipRatio * lenience, "avg ownershipMAERatio");
  checkLimit(p95(ownershipMAERatio), limits.p95OwnershipRatio * lenience, "p95 ownershipMAERatio");

  bool failuresExempted = false;
  if(!success && computeBackendRefEffParams(nnEval) < MIN_EFF_PARAMS_TO_CHECK) {
    logger.write("Backend reference test: failed checks ignored for " + nnEval->getInternalModelName() + ", model is below the minimum size for checking");
    success = true;
    failuresExempted = true;
  }

  if(verbose) {
    if(failuresExempted)
      logger.write("Backend reference test PASSED (checks not applicable at this model size)");
    else
      logger.write(string("Backend reference test ") + (success ? "PASSED" : "FAILED"));
  }
  return success;
}
