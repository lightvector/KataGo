#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../dataio/files.h"
#include "../neuralnet/modelversion.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <csignal>

using namespace std;

static ValueTargets makeForcedWinnerValueTarget(Player winner) {
  ValueTargets targets;
  if(winner == C_EMPTY) {
    targets.win = 0.5f;
    targets.loss = 0.5f;
    targets.noResult = 0.0f;
    targets.score = 0.0f;
    targets.hasLead = false;
    targets.lead = 0.0f;
    return targets;
  }
  assert(winner == P_BLACK || winner == P_WHITE);
  targets.win = winner == P_WHITE ? 1.0f : 0.0f;
  targets.loss = winner == P_BLACK ? 1.0f : 0.0f;
  targets.noResult = 0.0f;
  targets.score = winner == P_WHITE ? 0.5f : -0.5f;
  targets.hasLead = false;
  targets.lead = 0.0f;
  return targets;
}

static void getNNEval(
  const Board& board,
  const BoardHistory& hist,
  Player nextPla,
  double drawEquivalentWinsForWhite,
  Player playoutDoublingAdvantagePla,
  double playoutDoublingAdvantage,
  int maxHistory,
  NNEvaluator* nnEval,
  NNResultBuf& buf
) {
  bool skipCache = true;
  bool includeOwnerMap = true;
  MiscNNInputParams nnInputParams;
  // Use conservative pass so that if players interleave passing in the middle of their moves
  // when it would not be valid under a strict passing ruleset, it doesn't create a huge swing in value.
  nnInputParams.conservativePassAndIsRoot = true;
  nnInputParams.enablePassingHacks = true;
  nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
  nnInputParams.playoutDoublingAdvantage = (playoutDoublingAdvantagePla == getOpp(nextPla) ? -playoutDoublingAdvantage : playoutDoublingAdvantage);
  nnInputParams.maxHistory = maxHistory;
  Board copy(board);
  nnEval->evaluate(copy,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);
}

// static double getPassProb(
//   const Board& board,
//   const BoardHistory& hist,
//   Player nextPla,
//   double drawEquivalentWinsForWhite,
//   Player playoutDoublingAdvantagePla,
//   double playoutDoublingAdvantage,
//   NNEvaluator* nnEval
// ) {
//   NNResultBuf buf;
//   getNNEval(board,hist,nextPla,drawEquivalentWinsForWhite,playoutDoublingAdvantagePla,playoutDoublingAdvantage,1000,nnEval,buf);
//   int passPos = NNPos::getPassPos(nnEval->getNNXLen(),nnEval->getNNYLen());
//   return buf.result->policyProbs[passPos];
// }

static ValueTargets makeWhiteValueTarget(
  const Board& board,
  const BoardHistory& hist,
  Player nextPla,
  double drawEquivalentWinsForWhite,
  Player playoutDoublingAdvantagePla,
  double playoutDoublingAdvantage,
  NNEvaluator* nnEval
) {
  ValueTargets targets;
  if(hist.isGameFinished) {
    if(hist.isNoResult) {
      targets.win = 0.0f;
      targets.loss = 0.0f;
      targets.noResult = 1.0f;
      targets.score = 0.0f;
      targets.hasLead = false;
      targets.lead = 0.0f;
    }
    else {
      BoardHistory copyHist(hist);
      copyHist.endAndScoreGameNow(board);
      targets.win = (float)ScoreValue::whiteWinsOfWinner(copyHist.winner, drawEquivalentWinsForWhite);
      targets.loss = 1.0f - targets.win;
      targets.noResult = 0.0f;
      targets.score = (float)ScoreValue::whiteScoreDrawAdjust(copyHist.finalWhiteMinusBlackScore,drawEquivalentWinsForWhite,hist);
      targets.hasLead = false;
      targets.lead = targets.score;
    }
    return targets;
  }

  NNResultBuf buf;
  int maxHistory = 1000;
  getNNEval(board,hist,nextPla,drawEquivalentWinsForWhite,playoutDoublingAdvantagePla,playoutDoublingAdvantage,maxHistory,nnEval,buf);

  targets.win = buf.result->whiteWinProb;
  targets.loss = buf.result->whiteLossProb;
  targets.noResult = buf.result->whiteNoResultProb;
  targets.score = buf.result->whiteScoreMean;
  targets.hasLead = false;
  targets.lead = buf.result->whiteLead;
  return targets;
}

static std::set<string> loadStrippedTxtFileLines(const string& filePath) {
  std::vector<string> lines = FileUtils::readFileLines(filePath,'\n');
  std::set<string> ret;
  for(const string& line: lines) {
    string s = Global::trim(line);
    if(s.size() > 0)
      ret.insert(s);
  }
  return ret;
}

static std::set<string> loadStrippedTxtFileLinesWithComments(const string& filePath) {
  std::vector<string> lines = FileUtils::readFileLines(filePath,'\n');
  std::set<string> ret;
  for(const string& line: lines) {
    string s = Global::trim(Global::stripComments(line));
    if(s.size() > 0)
      ret.insert(s);
  }
  return ret;
}

static bool isLikelyBot(const string& username, const string& whatDataSource) {
  string s = Global::trim(Global::toLower(username));
  if(s.find("gnugo") != std::string::npos)
    return true;
  if(s.find("gnu go") != std::string::npos)
    return true;
  if(s.find("fuego") != std::string::npos)
    return true;
  if(s.find("katago") != std::string::npos)
    return true;
  if(s.find("kata-bot") != std::string::npos)
    return true;
  if(s.find("crazystone") != std::string::npos)
    return true;
  if(s.find("leela") != std::string::npos)
    return true;
  if(s.find("pachi") != std::string::npos)
    return true;
  if(s.find("manyfaces") != std::string::npos)
    return true;
  if(s.find("amybot") != std::string::npos)
    return true;
  if(s.find("randombot") != std::string::npos)
    return true;
  if(s.find("random_bot") != std::string::npos)
    return true;
  if(s.find("random bot") != std::string::npos)
    return true;
  if(s.find("weakbot") != std::string::npos)
    return true;
  if(s.find("weak_bot") != std::string::npos)
    return true;
  if(s.find("weak bot") != std::string::npos)
    return true;
  if(whatDataSource == "ogs") {
    if(s.find("_bot_") != std::string::npos)
      return true;
    if(Global::isSuffix(s,"_bot"))
      return true;
    if(Global::isSuffix(s,"-bot"))
      return true;
  }
  if(whatDataSource == "kgs") {
    if(Global::isPrefix(s,"ayabot"))
      return true;
    if(Global::isPrefix(s,"ayamc"))
      return true;
    if(Global::isPrefix(s,"dcnn"))
      return true;
    if(Global::isPrefix(s,"jbxkata"))
      return true;
    if(Global::isPrefix(s,"hirabot"))
      return true;
    if(Global::isPrefix(s,"swissbot"))
      return true;
    if(Global::isPrefix(s,"golois"))
      return true;
    if(Global::isPrefix(s,"hiramc"))
      return true;
    if(Global::isPrefix(s,"petgo"))
      return true;
    if(Global::isPrefix(s,"easybot"))
      return true;
    if(Global::isPrefix(s,"zen19"))
      return true;
    if(Global::isPrefix(s,"darkfmcts"))
      return true;
    if(Global::isPrefix(s,"mfgo") && s.find("kyu") != std::string::npos)
      return true;
    if(Global::isPrefix(s,"mfgo") && s.find("dnnbot") != std::string::npos)
      return true;
    if(Global::isPrefix(s,"nexus") && s.find("5k") != std::string::npos)
      return true;
    if(Global::isPrefix(s,"nexus") && s.find("10k") != std::string::npos)
      return true;
    if(Global::isPrefix(s,"valkyria"))
      return true;
    if(Global::isPrefix(s,"mogobot"))
      return true;
    if(Global::isPrefix(s,"orego"))
      return true;
    if(Global::isPrefix(s,"neuralz"))
      return true;
    if(Global::isPrefix(s,"rankbot"))
      return true;
    if(Global::isPrefix(s,"sai0bot"))
      return true;
    if(Global::isPrefix(s,"postneo"))
      return true;
    if(s == "ericabot")
      return true;
    if(s == "bonobot")
      return true;
    if(s == "czechbot")
      return true;
  }

  return false;
}

static void parseSGFRank(
  const string& rankStr,
  int& inverseRankRet,
  bool& isUnrankedRet,
  bool& isUnknownRet,
  const string& whatDataSource
) {
  int rankNumber = 0;
  bool isKyu = false;
  bool isKyuAma = false;
  bool isDan = false;
  bool isDanAma = false;
  bool isPro = false;
  bool isUnranked = false;
  bool isUnknown = false;

  string rankStrLower = Global::toLower(rankStr);
  try {
    if(rankStr == "") {
      isUnknown = true;
    }
    else if(rankStr == "-") {
      isUnranked = true;
    }
    //-------------------------------------------------------------
    //Don't know how strong "insei" or "ama" are
    else if(
      rankStrLower == "insei"
      || rankStrLower == "ama"
      || rankStrLower == "ama."
      || rankStrLower == "amateur"
      || rankStrLower == "nr"
    ) {
      isUnknown = true;
    }
    else if(
      rankStrLower == "meijin"
      || rankStrLower == "kisei"
      || rankStrLower == "judan"
      || rankStrLower == "holder"
      || rankStrLower == "challenger"
      || rankStrLower == "oza"
      || rankStrLower == "tianyuan"
      || rankStrLower == "mingren"
      || rankStrLower == "tengen"
      || rankStrLower == "guoshou"
      || rankStrLower == "gosei"

    ) {
      rankNumber = 9;
      isPro = true;
    }
    //-------------------------------------------------------------
    else if(Global::isSuffix(rankStr," kyu")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr," kyu"));
      isKyu = true;
    }
    else if(Global::isSuffix(rankStr," k")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr," k"));
      isKyu = true;
    }
    else if(Global::isSuffix(rankStr,"kyu")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"kyu"));
      isKyu = true;
    }
    else if(Global::isSuffix(rankStr,"k")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"k"));
      isKyu = true;
    }
    //UTF-8 for 级(kyu/grade)
    else if(Global::isSuffix(rankStr,"\xE7\xBA\xA7")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"\xE7\xBA\xA7"));
      isKyu = true;
    }
    //-------------------------------------------------------------
    else if(Global::isSuffix(rankStr,"k ama")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"k ama"));
      isKyuAma = true;
    }
    //-------------------------------------------------------------
    else if(Global::isSuffix(rankStr," dan")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr," dan"));
      isDan = true;
    }
    else if(Global::isSuffix(rankStr," d")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr," d"));
      isDan = true;
    }
    else if(Global::isSuffix(rankStr,"dan")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"dan"));
      isDan = true;
    }
    else if(Global::isSuffix(rankStr,"d")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"d"));
      isDan = true;
    }
    //UTF-8 for 段(dan)
    else if(Global::isSuffix(rankStr,"\xE6\xAE\xB5")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"\xE6\xAE\xB5"));
      isDan = true;
    }
    //-------------------------------------------------------------
    else if(Global::isSuffix(rankStr,"d ama")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"d ama"));
      isDanAma = true;
    }
    else if(Global::isSuffix(rankStr,"a")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"a"));
      isDanAma = true;
    }
    //-------------------------------------------------------------
    else if(Global::isSuffix(rankStr," pro")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr," pro"));
      isPro = true;
    }
    else if(Global::isSuffix(rankStr," p")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr," p"));
      isPro = true;
    }
    else if(Global::isSuffix(rankStr,"pro")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"pro"));
      isPro = true;
    }
    else if(Global::isSuffix(rankStr,"p")) {
      rankNumber = Global::stringToInt(Global::chopSuffix(rankStr,"p"));
      isPro = true;
    }
    //-------------------------------------------------------------
    else {
      throw StringError("Unable to parse rank: " + rankStr);
    }
    //-------------------------------------------------------------
  }
  catch(const StringError&) {
    throw StringError("Unable to parse rank: " + rankStr);
  }

  assert((int)isKyu + (int)isKyuAma + (int)isDan + (int)isDanAma + (int)isPro + (int)isUnranked + (int)isUnknown == 1);

  //Special handling for gogod ranks. Gogod labels pro dan as d, and amateur dan sometimes as "a" or "d ama".
  //Also Gogod kyu labels are not reliable to modern kyu since they could be historical very strong kyu.
  if(whatDataSource == "gogod") {
    if(isDan) {
      isPro = true;
      isDan = false;
    }
    else if(isKyu) {
      isUnknown = true;
    }
  }

  //--------------------------------------------------------------

  if(isUnknown) {
    inverseRankRet = 0;
    isUnrankedRet = false;
    isUnknownRet = true;
  }
  else if(isUnranked) {
    inverseRankRet = 0;
    isUnrankedRet = true;
    isUnknownRet = false;
  }
  else if(isPro) {
    inverseRankRet = 1;
    isUnrankedRet = false;
    isUnknownRet = false;
  }
  else if(isDan || isDanAma) {
    if(rankNumber <= 0)
      throw StringError("Unable to parse rank: " + rankStr);
    if(rankNumber > 9)
      rankNumber = 9;
    inverseRankRet = 10 - rankNumber;
    isUnrankedRet = false;
    isUnknownRet = false;
  }
  else {
    if(rankNumber <= 0)
      throw StringError("Unable to parse rank: " + rankStr);
    inverseRankRet = 9 + rankNumber;
    isUnrankedRet = false;
    isUnknownRet = false;
  }
}

struct KGSCsvLine {
  SimpleDate date;
  string sgfWUsername;
  string sgfWRank;
  string sgfBUsername;
  string sgfBRank;
  int indexThisDayZeroIndexed;
  bool isRated;
  int boardSize;
  int handicap;
  float komi;
  string unknownField; // Not sure what this is
  float result;

  string getKey() {
    return date.toString() + "|$#" + sgfWUsername + "|$#" + sgfBUsername + "|" + Global::intToString(indexThisDayZeroIndexed);
  }
  static string makeKey(SimpleDate date, string sgfWUsername, string sgfBUsername, int indexThisDayZeroIndexed) {
    return date.toString() + "|$#" + sgfWUsername + "|$#" + sgfBUsername + "|" + Global::intToString(indexThisDayZeroIndexed);
  }
};

static void readKgsCsv(
  const string& kgsCsv,
  SimpleDate kgsCsvMinDate,
  SimpleDate kgsCsvMaxDate,
  Logger& logger,
  std::map<string,KGSCsvLine>& kgsCsvMap
) {
  std::vector<string> lines = FileUtils::readFileLines(kgsCsv, '\n');
  logger.write("Loaded KGS csv with " + Global::uint64ToString(lines.size()) + " lines");
  for(const string& origLine: lines) {
    std::vector<string> pieces = Global::split(Global::trim(origLine),',');
    if(pieces.size() != 12)
      continue;
    for(size_t i = 0; i<pieces.size(); i++)
      pieces[i] = Global::trim(pieces[i],"\"");

    KGSCsvLine data;
    data.date = SimpleDate(pieces[0]);
    if(data.date < kgsCsvMinDate || data.date > kgsCsvMaxDate)
      continue;

    data.sgfWUsername = pieces[1];
    data.sgfWRank = pieces[2];
    data.sgfBUsername = pieces[3];
    data.sgfBRank = pieces[4];
    data.indexThisDayZeroIndexed = Global::stringToInt(pieces[5]);
    if(pieces[6] == "ranked" || pieces[6] == "tournament")
      data.isRated = true;
    else if(pieces[6] == "free" || pieces[6] == "teaching" || pieces[6] == "simul")
      data.isRated = false;
    else
      throw StringError("Unknown option for rating field in kgs csv: " + pieces[6]);
    data.boardSize = Global::stringToInt(pieces[7]);
    data.handicap = Global::stringToInt(pieces[8]);
    data.komi = Global::stringToFloat(pieces[9]);
    data.unknownField = pieces[10];
    data.result = -Global::stringToFloat(pieces[11]);
    kgsCsvMap[data.getKey()] = data;
  }
  logger.write("Parsed KGS csv and kept " + Global::uint64ToString(kgsCsvMap.size()) + " lines");
}

static string parseFileNameToKgsCsvKey(
  const string& fileName,
  const string& sgfWUsername,
  const string& sgfBUsername
) {
  std::vector<string> fileNamePieces = Global::split(fileName,'/');
  if(fileNamePieces.size() >= 4) {
    int year, month, day;
    if(
      Global::tryStringToInt(fileNamePieces[fileNamePieces.size()-4],year) &&
      Global::tryStringToInt(fileNamePieces[fileNamePieces.size()-3],month) &&
      Global::tryStringToInt(fileNamePieces[fileNamePieces.size()-2],day)
    ) {
      SimpleDate fileDirDate;
      bool parsedDate = false;
      try {
        fileDirDate = SimpleDate(year,month,day);
        parsedDate = true;
      }
      catch(const StringError& e) {
        (void)e;
      }
      if(parsedDate) {
        string fileNameBase = fileNamePieces[fileNamePieces.size()-1];
        if(Global::isSuffix(fileNameBase,".sgf"))
          fileNameBase = Global::chopSuffix(fileNameBase,".sgf");
        string expectedFileNamePrefix = sgfWUsername + "-" + sgfBUsername;
        if(Global::isPrefix(fileNameBase,expectedFileNamePrefix)) {
          string remainder = Global::chopPrefix(fileNameBase,expectedFileNamePrefix);
          int indexThisDayOneIndexed;
          if(remainder == "")
            indexThisDayOneIndexed = 1;
          else if(Global::isPrefix(remainder,"-") && Global::tryStringToInt(remainder,indexThisDayOneIndexed)) {
            indexThisDayOneIndexed = -indexThisDayOneIndexed;
          }
          else {
            indexThisDayOneIndexed = -1;
          }
          if(indexThisDayOneIndexed > 0) {
            int indexThisDayZeroIndexed = indexThisDayOneIndexed - 1;
            return KGSCsvLine::makeKey(fileDirDate, sgfWUsername, sgfBUsername, indexThisDayZeroIndexed);
          }
        }
      }
    }
  }
  return string();
}


int MainCmds::writetrainingdata(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfDirs;
  string noTrainUsersFile;
  string noGameUsersFile;
  string isBotUsersFile;
  string excludeGamesFile;
  bool noTrainOnBots;
  bool useFancyBotUsers;
  string whatDataSource;
  string kgsCsv;
  SimpleDate kgsCsvMinDate("0000-01-01");
  SimpleDate kgsCsvMaxDate("9999-12-31");
  size_t maxFilesToLoad;
  bool shuffleFiles;
  double keepProb;
  double gameKeepFracMin;
  double gameKeepFracMax;
  int maxVisits;
  string outputDir;
  int verbosity;

  try {
    KataGoCommandLine cmd("Generate training data from sgfs.");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::ValueArg<string> noTrainUsersArg("","no-train-users-file","Avoid training on these player's moves",true,string(),"TXTFILE");
    TCLAP::ValueArg<string> noGameUsersArg("","no-game-users-file","Avoid training on games with these players",true,string(),"TXTFILE");
    TCLAP::ValueArg<string> isBotUsersArg("","is-bot-users-file","Mark these usernames as bots",true,string(),"TXTFILE");
    TCLAP::ValueArg<string> excludeGamesArg("","exclude-games-file","Exclude games whose paths contain these strings",false,string(),"TXTFILE");
    TCLAP::SwitchArg noTrainOnBotsArg("","no-train-on-bots","Use bot users files as additional no train users file");
    TCLAP::SwitchArg useFancyBotUsersArg("","use-fancy-bot-users","Use some hardcoded rules to mark as bots some common names");
    TCLAP::ValueArg<string> whatDataSourceArg("","what-data-source","What data source",true,string(),"NAME");
    TCLAP::ValueArg<string> kgsCsvArg("","kgs-csv","KGS keeps whether games are rated as separate csv",false,string(),"CSVFILE");
    TCLAP::ValueArg<string> kgsCsvMinDateArg("","kgs-csv-min-date","Min date to use from csv",false,string(),"DATE");
    TCLAP::ValueArg<string> kgsCsvMaxDateArg("","kgs-csv-max-date","Max date to use from csv",false,string(),"DATE");
    TCLAP::ValueArg<size_t> maxFilesToLoadArg("","max-files-to-load","Max sgf files to try to load",false,(size_t)10000000000000ULL,"NUM");
    TCLAP::SwitchArg shuffleFilesArg("","shuffle-files","Shuffle order of files handled");
    TCLAP::ValueArg<double> keepProbArg("","keep-prob","Keep poses with this prob",false,1.0,"PROB");
    TCLAP::ValueArg<double> gameKeepFracMinArg("","game-keep-frac-min","Keep games frac min",false,0.0,"MIN");
    TCLAP::ValueArg<double> gameKeepFracMaxArg("","game-keep-frac-max","Keep games frac max",false,1.0,"MAX");
    TCLAP::ValueArg<int> maxVisitsArg("","max-visits","Max visits",false,30,"NUM");
    TCLAP::ValueArg<string> outputDirArg("","output-dir","Dir to output files",true,string(),"DIR");
    TCLAP::ValueArg<int> verbosityArg("","verbosity","1-3",false,1,"NUM");

    cmd.add(sgfDirArg);
    cmd.add(noTrainUsersArg);
    cmd.add(noGameUsersArg);
    cmd.add(isBotUsersArg);
    cmd.add(excludeGamesArg);
    cmd.add(noTrainOnBotsArg);
    cmd.add(useFancyBotUsersArg);
    cmd.add(whatDataSourceArg);
    cmd.add(kgsCsvArg);
    cmd.add(kgsCsvMinDateArg);
    cmd.add(kgsCsvMaxDateArg);
    cmd.add(maxFilesToLoadArg);
    cmd.add(shuffleFilesArg);
    cmd.add(keepProbArg);
    cmd.add(gameKeepFracMinArg);
    cmd.add(gameKeepFracMaxArg);
    cmd.add(maxVisitsArg);
    cmd.add(outputDirArg);
    cmd.add(verbosityArg);

    cmd.parseArgs(args);

    nnModelFile = cmd.getModelFile();
    sgfDirs = sgfDirArg.getValue();
    noTrainUsersFile = noTrainUsersArg.getValue();
    noGameUsersFile = noGameUsersArg.getValue();
    isBotUsersFile = isBotUsersArg.getValue();
    excludeGamesFile = excludeGamesArg.getValue();
    noTrainOnBots = noTrainOnBotsArg.getValue();
    useFancyBotUsers = useFancyBotUsersArg.getValue();
    whatDataSource = whatDataSourceArg.getValue();
    kgsCsv = kgsCsvArg.getValue();
    if(kgsCsvMinDateArg.getValue() != "")
      kgsCsvMinDate = SimpleDate(kgsCsvMinDateArg.getValue());
    if(kgsCsvMaxDateArg.getValue() != "")
      kgsCsvMaxDate = SimpleDate(kgsCsvMaxDateArg.getValue());
    maxFilesToLoad = maxFilesToLoadArg.getValue();
    shuffleFiles = shuffleFilesArg.getValue();
    keepProb = keepProbArg.getValue();
    gameKeepFracMin = gameKeepFracMinArg.getValue();
    gameKeepFracMax = gameKeepFracMaxArg.getValue();
    maxVisits = maxVisitsArg.getValue();
    outputDir = outputDirArg.getValue();
    verbosity = verbosityArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTimeStamp = true;
  Logger logger(nullptr, logToStdout, logToStderr, logTimeStamp);
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);

  {
    MakeDir::make(outputDir);
    std::vector<string> outputDirFiles = FileUtils::listFiles(outputDir);
    for(const string& file: outputDirFiles) {
      if(Global::isSuffix(file,".npz"))
        throw StringError("outputDir already contains npz files: " + outputDir);
    }
  }
  logger.addFile(outputDir + "/" + "log.log");

  const int numWorkerThreads = 64;
  const int numSearchThreads = 1;
  const int numTotalThreads = numWorkerThreads * numSearchThreads;

  const int dataBoardLen = cfg.getInt("dataBoardLen",3,37);
  const int maxApproxRowsPerTrainFile = cfg.getInt("maxApproxRowsPerTrainFile",1,100000000);

  const std::vector<std::pair<int,int>> allowedBoardSizes =
    cfg.getNonNegativeIntDashedPairs("allowedBoardSizes", 2, Board::MAX_LEN);

  if(dataBoardLen > Board::MAX_LEN)
    throw StringError("dataBoardLen > maximum board len, must recompile to increase");

  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  const int inputsVersion = 7;
  const int numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V7;
  const int numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V7;

  const std::set<string> noTrainUsers = loadStrippedTxtFileLines(noTrainUsersFile);
  const std::set<string> noGameUsers = loadStrippedTxtFileLines(noGameUsersFile);
  const std::set<string> isBotUsers = loadStrippedTxtFileLines(isBotUsersFile);
  const std::set<string> excludeFiles =
    (excludeGamesFile != "") ? loadStrippedTxtFileLinesWithComments(excludeGamesFile) : std::set<string>();

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int maxConcurrentEvals = numTotalThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = numTotalThreads;
    const int defaultMaxBatchSize = std::max(8,((numTotalThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  string searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());
  SearchParams params = SearchParams::basicDecentParams();
  params.maxVisits = maxVisits;
  params.chosenMoveTemperatureEarly = 0.1;
  params.chosenMoveTemperature = 0.1;
  params.useUncertainty = false; // To prevent weird selection effects at low playouts
  params.numThreads = numSearchThreads;
  params.rootEndingBonusPoints = 0.8; // More aggressive game ending
  params.conservativePass = false; // false since we want game completions to be consistent with strict rules
  params.enablePassingHacks = true;

  std::map<string,KGSCsvLine> kgsCsvMap;
  if(kgsCsv != "") {
    if(whatDataSource != "kgs")
      throw StringError("Provided kgs csv but data source is not kgs");
    readKgsCsv(kgsCsv, kgsCsvMinDate, kgsCsvMaxDate, logger, kgsCsvMap);
  }

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);
  logger.write("Collected " + Global::int64ToString(sgfFiles.size()) + " files");

  if(shuffleFiles) {
    seedRand.shuffle(sgfFiles);
  }

  Setup::initializeSession(cfg);

  cfg.warnUnusedKeys(cerr,&logger);

  // Done loading!
  // ------------------------------------------------------------------------------------
  std::atomic<int64_t> numSgfsDone(0);
  std::atomic<int64_t> numSgfErrors(0);

  std::mutex statsLock;
  std::map<string,int64_t> gameCountByUsername;
  std::map<string,int64_t> gameCountByRank;
  std::map<string,int64_t> gameCountByBSize;
  std::map<string,int64_t> gameCountByRules;
  std::map<string,int64_t> gameCountByKomi;
  std::map<string,int64_t> gameCountByHandicap;
  std::map<string,int64_t> gameCountByResult;
  std::map<string,int64_t> gameCountByTimeControl;
  std::map<string,int64_t> gameCountByIsRated;
  std::map<string,int64_t> gameCountByEvent;
  std::map<string,int64_t> gameCountByPlace;

  std::map<string,int64_t> acceptedGameCountByUsername;
  std::map<string,int64_t> acceptedGameCountByRank;
  std::map<string,int64_t> acceptedGameCountByBSize;
  std::map<string,int64_t> acceptedGameCountByRules;
  std::map<string,int64_t> acceptedGameCountByKomi;
  std::map<string,int64_t> acceptedGameCountByHandicap;
  std::map<string,int64_t> acceptedGameCountByResult;
  std::map<string,int64_t> acceptedGameCountByTimeControl;
  std::map<string,int64_t> acceptedGameCountByIsRated;
  std::map<string,int64_t> acceptedGameCountByEvent;
  std::map<string,int64_t> acceptedGameCountByPlace;
  std::map<string,int64_t> acceptedGameCountByYear;

  std::map<string,int64_t> acceptedGameCountByUsage;
  std::map<string,int64_t> doneGameCountByReason;
  std::map<string,int64_t> warningsByReason;

  auto reportSgfDone = [&](bool wasSuccess, const string& reasonLabel) {
    if(!wasSuccess)
      numSgfErrors.fetch_add(1);
    int64_t numErrors = numSgfErrors.load();
    int64_t numDone = numSgfsDone.fetch_add(1) + 1;
    {
      std::lock_guard<std::mutex> lock(statsLock);
      doneGameCountByReason[reasonLabel] += 1;
    }

    if(numDone == sgfFiles.size() || numDone % 100 == 0) {
      logger.write(
        "Done " + Global::int64ToString(numDone) + " / " + Global::int64ToString(sgfFiles.size()) + " sgfs, " +
        string("errors ") + Global::int64ToString(numErrors)
      );
    }
  };

  std::vector<TrainingWriteBuffers*> threadDataBuffers;
  std::vector<Rand*> threadRands;
  std::vector<Search*> threadSearchers;
  for(int threadIdx = 0; threadIdx<numWorkerThreads; threadIdx++) {
    threadRands.push_back(new Rand());
    const bool hasMetadataInput = true;
    threadDataBuffers.push_back(
      new TrainingWriteBuffers(
        inputsVersion,
        (maxApproxRowsPerTrainFile * 4/3 + Board::MAX_PLAY_SIZE * 2 + 100),
        numBinaryChannels,
        numGlobalChannels,
        dataBoardLen,
        dataBoardLen,
        hasMetadataInput
      )
    );
    Search* search = new Search(params,nnEval,&logger,searchRandSeed);
    search->setAlwaysIncludeOwnerMap(true);
    threadSearchers.push_back(search);
  }

  auto saveDataBuffer = [&](TrainingWriteBuffers* dataBuffer, Rand& rand) {
    int64_t numRows = dataBuffer->curRows;
    string resultingFilename = outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz";
    string tmpFilename = resultingFilename + ".tmp";
    dataBuffer->writeToZipFile(tmpFilename);
    dataBuffer->clear();
    FileUtils::rename(tmpFilename,resultingFilename);
    logger.write("Saved output file with " + Global::int64ToString(numRows) + " rows: " + resultingFilename);
  };


  auto processSgf = [&](int threadIdx, size_t index) {
    TrainingWriteBuffers* dataBuffer = threadDataBuffers[threadIdx];
    Rand& rand = *threadRands[threadIdx];

    const string& fileName = sgfFiles[index];
    {
      double d = Hash::seededHashFloat(fileName,"writeTrainingDataKeepFrac");
      if(d < gameKeepFracMin || d >= gameKeepFracMax) {
        reportSgfDone(false,"SGFNotInKeepFrac");
        return;
      }
    }

    for(const string& excluded: excludeFiles) {
      if(fileName.find(excluded) != string::npos) {
        reportSgfDone(false,"SGFExcluded");
        return;
      }
    }

    std::unique_ptr<Sgf> sgfRaw = NULL;
    XYSize xySize;
    try {
      sgfRaw = std::unique_ptr<Sgf>(Sgf::loadFile(fileName));
      xySize = sgfRaw->getXYSize();
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      reportSgfDone(false,"SGFInvalid");
      return;
    }

    if(xySize.x > dataBoardLen || xySize.y > dataBoardLen) {
      logger.write(
        "SGF board size > dataBoardLen in " + fileName + ":"
        + " " + Global::intToString(xySize.x)
        + " " + Global::intToString(xySize.y)
        + " " + Global::intToString(dataBoardLen)
      );
      reportSgfDone(false,"SGFGreaterThanDataBoardLen");
      return;
    }
    {
      bool foundBoardSize = false;
      for(const std::pair<int,int> p: allowedBoardSizes) {
        if(p.first == xySize.x && p.second == xySize.y)
          foundBoardSize = true;
        if(p.first == xySize.y && p.second == xySize.x)
          foundBoardSize = true;
      }
      if(!foundBoardSize) {
        logger.write(
          "SGF board size not in allowedBoardSizes in " + fileName + ":"
          + " " + Global::intToString(xySize.x)
          + " " + Global::intToString(xySize.y)
        );
        reportSgfDone(false,"SGFDisallowedBoardSize");
        return;
      }
    }

    const int boardArea = xySize.x * xySize.y;
    const double sqrtBoardArea = sqrt(boardArea);
    const string sizeStr = " (size " + Global::intToString(xySize.x) + "x" + Global::intToString(xySize.y) + ")";
    const string bSizeStr = Global::intToString(xySize.x) + "x" + Global::intToString(xySize.y);

    std::unique_ptr<CompactSgf> sgf = NULL;
    try {
      sgf = std::make_unique<CompactSgf>(sgfRaw.get());
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      reportSgfDone(false,"SGFInvalidCompact");
      return;
    }

    string sgfBUsername;
    string sgfWUsername;
    string sgfBRank;
    string sgfWRank;
    string sgfRules;
    string sgfKomi;
    string sgfHandicap;
    string sgfResult;
    string sgfEvent;
    string sgfDate;
    string sgfPlace;
    try {
      sgfBUsername = Global::trim(sgfRaw->getRootPropertyWithDefault("PB", ""));
      sgfWUsername = Global::trim(sgfRaw->getRootPropertyWithDefault("PW", ""));
      sgfBRank = sgfRaw->getRootPropertyWithDefault("BR", "");
      sgfWRank = sgfRaw->getRootPropertyWithDefault("WR", "");
      sgfRules = sgfRaw->getRootPropertyWithDefault("RU", "");
      sgfKomi = sgfRaw->getRootPropertyWithDefault("KM", "");
      sgfHandicap = sgfRaw->getRootPropertyWithDefault("HA", "");
      sgfResult = sgfRaw->getRootPropertyWithDefault("RE", "");
      sgfEvent = sgfRaw->getRootPropertyWithDefault("EV", "");
      sgfDate = sgfRaw->getRootPropertyWithDefault("DT", "");
      sgfPlace = sgfRaw->getRootPropertyWithDefault("PC", "");
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      reportSgfDone(false,"SGFInvalid");
      return;
    }

    bool sgfGameIsRated = false;
    bool sgfGameRatednessIsUnknown = false;
    bool foundKgsEntry = false;
    KGSCsvLine kgsEntry;
    if(whatDataSource == "ogs") {
      string sgfGC = sgfRaw->getRootPropertyWithDefault("GC", "");
      std::vector<string> pieces = Global::split(sgfGC,',');
      bool isRated = false;
      bool isUnrated = false;
      for(const string& s: pieces) {
        if(s == "ranked")
          isRated = true;
        if(s == "unranked")
          isUnrated = true;
      }
      if(!isRated && !isUnrated) {
        logger.write("Unknown rating status in SGF " + fileName);
        reportSgfDone(false,"GameUnknownRatingStatus");
        return;
      }
      sgfGameIsRated = isRated;
    }
    else if(whatDataSource == "kgs") {
      sgfGameRatednessIsUnknown = true;
      if(sgfBRank == "")
        sgfBRank = "-";
      if(sgfWRank == "")
        sgfWRank = "-";
      if(kgsCsv != "") {
        string key = parseFileNameToKgsCsvKey(fileName, sgfWUsername, sgfBUsername);
        if(key == "") {
          logger.write("WARNING: Unable to parse KGS csv key from file path for " + fileName);
          std::lock_guard<std::mutex> lock(statsLock);
          warningsByReason["UnableParseKGSCsvKey"] += 1;
        }
        else {
          const auto& iter = kgsCsvMap.find(key);
          if(iter != kgsCsvMap.end()) {
            kgsEntry = iter->second;
            foundKgsEntry = true;
            sgfGameIsRated = kgsEntry.isRated;
            sgfGameRatednessIsUnknown = false;
          }
          else {
            logger.write("WARNING: Unable to find KGS csv key for " + fileName);
            std::lock_guard<std::mutex> lock(statsLock);
            warningsByReason["UnableFindKGSCsvKey"] += 1;
          }
        }
      }
    }
    else if(whatDataSource == "gogod") {
      sgfGameIsRated = true;
    }
    else if(whatDataSource == "go4go") {
      sgfGameIsRated = true;
    }
    else {
      throw StringError("Unknown what-data-source " + whatDataSource);
    }

    string sgfTM = sgfRaw->getRootPropertyWithDefault("TM", "");
    string sgfOT = sgfRaw->getRootPropertyWithDefault("OT", "");
    string sgfLC = sgfRaw->getRootPropertyWithDefault("LC", "");
    string sgfLT = sgfRaw->getRootPropertyWithDefault("LT", "");
    string sgfTC = sgfRaw->getRootPropertyWithDefault("TC", "");
    string sgfTT = sgfRaw->getRootPropertyWithDefault("TT", "");
    string sgfTimeControl;
    if(whatDataSource == "ogs" || whatDataSource == "kgs") {
      if(sgfTM != "" && sgfOT != "")
        sgfTimeControl = sgfTM + "+" + sgfOT;
      else if(sgfTM != "" && (sgfLC != "" || sgfLT != ""))
        sgfTimeControl = sgfTM + "+" + sgfLC + "x" + sgfLT;
      else if(sgfTM != "")
        sgfTimeControl = sgfTM;
      else if(sgfOT != "")
        sgfTimeControl = sgfOT;
      else if(sgfLC != "" || sgfLT != "")
        sgfTimeControl = sgfLC + "x" + sgfLT;
    }
    else if(whatDataSource == "fox") {
      if(sgfTM != "" && sgfOT != "")
        sgfTimeControl = sgfTM + "+" + sgfOT;
      else if(sgfTM != "" && (sgfTC != "" || sgfTT != ""))
        sgfTimeControl = sgfTM + "+" + sgfTC + "x" + sgfTT;
      else if(sgfTM != "")
        sgfTimeControl = sgfTM;
      else if(sgfOT != "")
        sgfTimeControl = sgfOT;
      else if(sgfTC != "" || sgfTT != "")
        sgfTimeControl = sgfTC + "x" + sgfTT;
    }
    else if(whatDataSource == "gogod") {
      sgfTimeControl = "";
    }
    else if(whatDataSource == "go4go") {
      sgfTimeControl = "";
    }
    else {
      throw StringError("Unknown data source: " + whatDataSource);
    }

    {
      std::lock_guard<std::mutex> lock(statsLock);
      gameCountByUsername[sgfBUsername] += 1;
      gameCountByUsername[sgfWUsername] += 1;
      gameCountByRank[sgfBRank] += 1;
      gameCountByRank[sgfWRank] += 1;
      gameCountByBSize[bSizeStr] += 1;
      gameCountByRules[sgfRules] += 1;
      gameCountByKomi[sgfKomi] += 1;
      gameCountByHandicap[sgfHandicap] += 1;
      gameCountByResult[sgfResult] += 1;
      gameCountByTimeControl[sgfTimeControl] += 1;
      gameCountByIsRated[sgfGameRatednessIsUnknown ? "unknown" : sgfGameIsRated ? "true" : "false"] += 1;
      gameCountByEvent[sgfEvent] += 1;
      gameCountByPlace[sgfPlace] += 1;
    }

    if(noGameUsers.find(sgfBUsername) != noGameUsers.end() || noGameUsers.find(sgfWUsername) != noGameUsers.end()) {
      logger.write("Filtering due to undesired user in sgf " + fileName);
      reportSgfDone(false,"UserInNoGameUsers");
      return;
    }

    bool isBotB = isBotUsers.find(sgfBUsername) != isBotUsers.end();
    bool isBotW = isBotUsers.find(sgfWUsername) != isBotUsers.end();
    if(useFancyBotUsers) {
      isBotB = isBotB || isLikelyBot(sgfBUsername,whatDataSource);
      isBotW = isBotW || isLikelyBot(sgfWUsername,whatDataSource);
    }

    const bool shouldTrainB = noTrainUsers.find(sgfBUsername) == noTrainUsers.end() && !(noTrainOnBots && isBotB);
    const bool shouldTrainW = noTrainUsers.find(sgfWUsername) == noTrainUsers.end() && !(noTrainOnBots && isBotW);

    if(!shouldTrainB && !shouldTrainW) {
      if(verbosity >= 2)
        logger.write("Filtering due to both users no training in sgf " + fileName);
      reportSgfDone(false,"UserBothNoTrain");
      return;
    }

    int sgfHandicapParsed = -1;
    if(sgfHandicap != "") {
      const int maxAllowedHandicap = (int)(sqrtBoardArea / 2.0);
      if(Global::tryStringToInt(sgfHandicap,sgfHandicapParsed) && sgfHandicapParsed >= 0 && sgfHandicapParsed <= maxAllowedHandicap) {
        // Yay
      }
      else {
        logger.write("Unable to parse handicap or handicap too extreme in sgf " + fileName + ": " + sgfHandicap + sizeStr);
        reportSgfDone(false,"GameBadHandicap");
        return;
      }
    }
    //Some Fox games have implict handicap placement, rather than trying to match/replicate we just ignore it.
    if(whatDataSource == "fox" && sgfHandicapParsed != 0) {
      logger.write("Skipping handicap game, not implemented, for sgf: " + fileName + ": " + sgfHandicap + sizeStr);
      reportSgfDone(false,"GameSkipHandicap");
      return;
    }

    Rules rules;
    bool rulesDisableValueTraining = false;
    if(whatDataSource == "ogs" || whatDataSource == "kgs") {
      try {
        rules = sgf->getRulesOrFail();
      }
      catch(const StringError& e) {
        logger.write("Failed to get rules for sgf: " + fileName + " " + e.what());
        reportSgfDone(false,"GameBadRules");
        return;
      }
    }
    else if(whatDataSource == "fox") {
      try {
        rules = sgf->getRulesOrFail();
      }
      catch(const StringError& e) {
        logger.write("Failed to get rules for sgf: " + fileName + " " + e.what());
        reportSgfDone(false,"GameBadRules");
        return;
      }
      //Ugly hack for fox that seems to have komi 0 when the real komi is 6.5
      if(sgfRules == "Japanese" && sgfKomi == "0") {
        rules.komi = 6.5;
      }
      else {
        throw StringError("Unhandled case " + sgfKomi + " " + sgfRules + " " + fileName);
      }
    }
    else if(whatDataSource == "gogod") {
      string sgfPlaceAndEvent = sgfPlace + " | " + sgfEvent;

      // Filter out unusual events and games that are hard to know the rules and player strengths
      if(
        sgfPlaceAndEvent.find("Environmental Go") != string::npos
        || sgfRules.find("gives komi") != string::npos
        || sgfPlaceAndEvent.find("Children") != string::npos
        || sgfPlaceAndEvent.find("Student") != string::npos
        || sgfPlaceAndEvent.find("Youth") != string::npos
        || sgfPlaceAndEvent.find("Young ") != string::npos
        || sgfPlaceAndEvent.find("1-dans v. 9-dans") != string::npos
        || sgfPlaceAndEvent.find("Teaching") != string::npos
        || sgfPlaceAndEvent.find("teaching") != string::npos
        || sgfPlaceAndEvent.find("pair go") != string::npos
        || sgfPlaceAndEvent.find("Pair go") != string::npos
        || sgfPlaceAndEvent.find("Amateur") != string::npos
        || sgfPlaceAndEvent.find("amateur") != string::npos
        || sgfPlaceAndEvent.find("Internet") != string::npos
        || sgfPlaceAndEvent.find("Online") != string::npos
        || sgfPlaceAndEvent.find("online") != string::npos
        || sgfPlaceAndEvent.find("on-line") != string::npos
        || sgfPlaceAndEvent.find("High School") != string::npos
        || sgfPlaceAndEvent.find("Middle School") != string::npos
        || sgfPlaceAndEvent.find("Primary School") != string::npos
        || sgfPlaceAndEvent.find("Elementary School") != string::npos
        || sgfPlaceAndEvent.find("Cross-Straits") != string::npos
        || sgfPlaceAndEvent.find("Computer") != string::npos
        || sgfPlaceAndEvent.find("computer") != string::npos
        || sgfPlaceAndEvent.find("Machine") != string::npos
        || sgfPlaceAndEvent.find("Sunjang") != string::npos
        || sgfRules.find("Sunjang") != string::npos
        || sgfPlaceAndEvent.find("World AI ") != string::npos
        || Global::isPrefix(sgfPlaceAndEvent,"AI ")
        || sgfBRank.find("&") != string::npos
        || sgfWRank.find("&") != string::npos
      ) {
        if(verbosity >= 2)
          logger.write("Unable to handle rules in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
        reportSgfDone(false,"UnsupportedRules");
        return;
      }

      bool whiteWinsJigo = false;
      if(sgfRules == "Chinese" || sgfRules == "Korean" || sgfRules == "Japanese" || sgfRules == "NZ" || sgfRules == "AGA") {
        rules = Rules::parseRules(sgfRules);
      }
      else if(sgfRules == "Ing" || sgfRules == "Ing Goe") {
        rules = Rules::parseRules("Chinese");
        if(rules.komi == 7.5) {
          //Good!
        }
        // Ing rules with komi 8 are basically 7.5 with black wins draws
        else if(rules.komi == 8) {
          rules.komi = 7.5;
        }
        else {
          logger.write("Unable to handle rules, need more cases in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
          reportSgfDone(false,"GameRulesParsing");
          return;
        }
      }
      else if(Global::isPrefix(sgfRules,"Tang")) {
        rules = Rules::parseRules("Japanese");
        rules.taxRule = Rules::TAX_ALL;
        rulesDisableValueTraining = true; // Since we're not entirely sure on the evaluation, also many games are incomplete
      }
      else if(sgfRules == "Old Chinese") {
        rules = Rules::parseRules("Chinese");
        rulesDisableValueTraining = true; // Since we're not entirely sure on the evaluation, also many games are incomplete
      }
      else {
        if(sgfRules == "W wins jigo") {
          whiteWinsJigo = true;
        }
        bool mentionsKorea = sgfPlaceAndEvent.find("Korea") != string::npos;
        bool mentionsJapan = sgfPlaceAndEvent.find("Japan") != string::npos;
        bool mentionsChina = sgfPlaceAndEvent.find("China") != string::npos || sgfPlaceAndEvent.find("Chinese") != string::npos;
        if((int)mentionsKorea + (int)mentionsJapan + (int)mentionsChina > 1) {
          logger.write("Unable to handle rules, need more cases in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
          reportSgfDone(false,"GameRulesParsing");
          return;
        }

        if(
          mentionsKorea
          || sgfPlaceAndEvent.find("Hanguk Kiweon") != string::npos
          || sgfPlaceAndEvent.find("LG Cup") != string::npos
          || sgfPlaceAndEvent.find("Samsung Cup") != string::npos
          || sgfPlaceAndEvent.find("Seoul") != string::npos
          || sgfPlaceAndEvent.find("BC Card") != string::npos
          || sgfPlaceAndEvent.find("Myeongin") != string::npos
          || sgfPlaceAndEvent.find("Hayago") != string::npos
          || sgfPlaceAndEvent.find("KBS Cup") != string::npos
          || sgfPlaceAndEvent.find("Paedal") != string::npos
          || sgfPlaceAndEvent.find("Kuksu") != string::npos
          || sgfPlaceAndEvent.find("Nongshim Cup") != string::npos
          || sgfPlaceAndEvent.find("Maxim Cup") != string::npos
          || sgfPlaceAndEvent.find("Tong Yang Cup") != string::npos
          || sgfPlaceAndEvent.find("Kiseong") != string::npos
          || sgfPlaceAndEvent.find("Kiwang") != string::npos
          || sgfPlaceAndEvent.find("Ki wang") != string::npos
          || sgfPlaceAndEvent.find("Chaegowi") != string::npos
          || sgfPlaceAndEvent.find("Wangwi") != string::npos
          || sgfPlaceAndEvent.find("GG Auction") != string::npos
          || sgfPlaceAndEvent.find("GS Caltex") != string::npos
          || sgfPlaceAndEvent.find("Ha Ch'an-seok Cup") != string::npos
        ) {
          rules = Rules::parseRules("Korean");
        }
        else if(
          mentionsJapan
          || sgfPlaceAndEvent.find("Nihon Ki-in") != string::npos
          || sgfPlaceAndEvent.find("Kansai Ki-in") != string::npos
          || sgfPlaceAndEvent.find("Tokyo") != string::npos
          || sgfPlaceAndEvent.find("Honinbo") != string::npos
          || sgfPlaceAndEvent.find("Oza") != string::npos
          || sgfPlaceAndEvent.find("Tengen") != string::npos
          || sgfPlaceAndEvent.find("Gosei") != string::npos
          || sgfPlaceAndEvent.find("Judan") != string::npos
          || sgfPlaceAndEvent.find("Ryusei") != string::npos
          || sgfPlaceAndEvent.find("Kisei") != string::npos
          || sgfPlaceAndEvent.find("Meijin") != string::npos
          || sgfPlaceAndEvent.find("NHK Cup") != string::npos
          || sgfPlaceAndEvent.find("Osaka") != string::npos
          || sgfPlaceAndEvent.find("Saikyo") != string::npos
          || sgfPlaceAndEvent.find("Daiwa Cup") != string::npos
          || sgfPlaceAndEvent.find("Castle Game") != string::npos
          || sgfPlaceAndEvent.find("Globis Cup") != string::npos
          || sgfPlaceAndEvent.find("Shinei") != string::npos
          || sgfPlaceAndEvent.find("Shinjin-O") != string::npos
          || sgfPlaceAndEvent.find("Kakusei") != string::npos
          || sgfPlaceAndEvent.find("Fujitsu Cup") != string::npos
          || sgfPlaceAndEvent.find("Toyota") != string::npos
          || sgfPlaceAndEvent.find("Oteai") != string::npos
       ) {
          rules = Rules::parseRules("Japanese");
        }
        else if(
          mentionsChina
          || sgfPlaceAndEvent.find("Beijing") != string::npos
          || sgfPlaceAndEvent.find("Chunlan Cup") != string::npos
          || sgfPlaceAndEvent.find("MLily Cup") != string::npos
          || sgfPlaceAndEvent.find("Lanke Cup") != string::npos
          || sgfPlaceAndEvent.find("CCTV Cup") != string::npos
        ) {
          rules = Rules::parseRules("Chinese");
        }
        else {
          logger.write("Unable to handle rules, need more cases in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
          reportSgfDone(false,"GameRulesParsing");
          return;
        }

        // If komi is unrecorded, just go ahead and use modern komi, maybe some results will differ but otherwise it's not too bad.
        if(sgfKomi != "") {
          if(!Global::tryStringToFloat(sgfKomi,rules.komi)) {
            logger.write("Unable to handle rules, need more cases in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
            reportSgfDone(false,"GameRulesParsing");
            return;
          }
        }

        // gogod uses zi for chinese komi, we should check if there's any weirdness
        if(rules.scoringRule == Rules::SCORING_AREA && rules.komi >= 4) {
          logger.write("Unable to handle rules, need more cases in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
          reportSgfDone(false,"GameRulesParsing");
          return;
        }
        if(rules.scoringRule == Rules::SCORING_AREA) {
          rules.komi *= 2;
        }

        if(whiteWinsJigo) {
          if(rules.komi != (int)rules.komi) {
            logger.write("Unable to handle rules, need more cases in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
            reportSgfDone(false,"GameRulesParsing");
            return;
          }
          rules.komi += 0.5f;
        }

        if(!Rules::komiIsIntOrHalfInt(rules.komi)) {
          logger.write("Unable to handle rules, need more cases in sgf " + fileName + ": " + sgfRules + "|" + sgfKomi + "|" + sgfPlace + "|" + sgfEvent);
          reportSgfDone(false,"GameRulesParsing");
          return;
        }
      }
    }
    else if(whatDataSource == "go4go") {
      try {
        rules = sgf->getRulesOrFail();
      }
      catch(const StringError& e) {
        logger.write("Failed to get rules for sgf: " + fileName + " " + e.what());
        reportSgfDone(false,"GameBadRules");
        return;
      }
      // Basically everything using komi 8 in go4go is "komi 8 but black wins ties
      if(rules.komi == 8)
        rules.komi = 7.5;
    }
    else {
      throw StringError("Unknown data source: " + whatDataSource);
    }

    // No friendly pass since we want to complete consistent with strict rules
    rules.friendlyPassOk = false;

    Board board;
    Player nextPla;
    BoardHistory hist;
    try {
      sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);
    }
    catch(const StringError& e) {
      logger.write("Bad initial setup in sgf " + fileName + " " + e.what());
      reportSgfDone(false,"BadInitialSetup");
      return;
    }
    const vector<Move>& sgfMoves = sgf->moves;

    if(sgfMoves.size() > boardArea * 1.5 + 40.0) {
      logger.write("Too many moves in sgf " + fileName + ": " + Global::uint64ToString(sgfMoves.size()) + sizeStr);
      reportSgfDone(false,"MovesTooManyMoves");
      return;
    }

    bool sgfGameEnded = false;
    bool sgfGameEndedByTime = false;
    bool sgfGameEndedByResign = false;
    bool sgfGameEndedByScore = false;
    bool sgfGameEndedByForfeit = false;
    bool sgfGameEndedButRecordIncomplete = false;
    bool sgfGameEndedByOther = false;
    Player sgfGameWinner = C_EMPTY;
    double sgfGameWhiteFinalScore = 0.0;

    if(sgfResult != "") {
      string s = Global::trim(Global::toLower(sgfResult));
      // Remove annotations for score in zi, we don't use the actual score so it doesn't matter if
      // it's inaccurate.
      {
        size_t found = s.find(" (zi)");
        if (found != std::string::npos)
          s.erase(found, 5);
      }
      {
        size_t found = s.find(" {zi}");
        if (found != std::string::npos)
          s.erase(found, 5);
      }
      // Remove annotations for ko connecting
      {
        size_t foundConnect = s.find("connect");
        if(foundConnect == string::npos)
          foundConnect = s.find("win");
        size_t foundKo = s.find("ko");
        if(foundConnect != string::npos && foundKo != string::npos) {
          size_t foundL = s.find_last_of('(',foundConnect);
          if(foundL == string::npos)
            foundL = s.find_last_of('{',foundConnect);
          size_t foundR = s.find_first_of(')',foundKo);
          if(foundR == string::npos)
            foundR = s.find_first_of('}',foundKo);
          if(foundL != string::npos && foundKo != string::npos && foundL < foundR) {
            s.erase(foundL, foundR-foundL+1);
            s = Global::trim(s);
          }
        }
      }

      // Don't use games with time penalty that might have changed result
      if(s.find("time penalty") != string::npos)
        rulesDisableValueTraining = true;

      if(s == "b+r" || s == "black+r" || s == "b+resign") {
        sgfGameEnded = true;
        sgfGameEndedByResign = true;
        sgfGameWinner = P_BLACK;
      }
      else if(s == "w+r" || s == "white+r" || s == "w+resign") {
        sgfGameEnded = true;
        sgfGameEndedByResign = true;
        sgfGameWinner = P_WHITE;
      }
      else if(s == "b+t" || s == "black+t" || s == "b+time") {
        sgfGameEnded = true;
        sgfGameEndedByTime = true;
        sgfGameWinner = P_BLACK;
      }
      else if(s == "w+t" || s == "white+t" || s == "w+time") {
        sgfGameEnded = true;
        sgfGameEndedByTime = true;
        sgfGameWinner = P_WHITE;
      }
      else if(s == "b+f" || s == "black+f" || s == "b+forfeit") {
        sgfGameEnded = true;
        sgfGameEndedByForfeit = true;
        sgfGameWinner = P_BLACK;
      }
      else if(s == "w+f" || s == "white+f" || s == "w+forfeit") {
        sgfGameEnded = true;
        sgfGameEndedByForfeit = true;
        sgfGameWinner = P_WHITE;
      }
      else if(s == "void") {
        sgfGameEnded = true;
        sgfGameEndedByOther = true;
      }
      else if(s == "?") {
        sgfGameEnded = true;
        sgfGameEndedByOther = true;
      }
      else if(s == "draw" || s == "0" || s == "jigo") {
        sgfGameEnded = true;
        sgfGameEndedByScore = true;
        sgfGameWinner = C_EMPTY;
        sgfGameWhiteFinalScore = 0.0;
      }
      else if(
        (Global::isPrefix(s,"b+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"b+"), sgfGameWhiteFinalScore))
        ||
        (Global::isPrefix(s,"black+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"black+"), sgfGameWhiteFinalScore))
      ) {
        if(
          std::isfinite(sgfGameWhiteFinalScore)
          && std::abs(sgfGameWhiteFinalScore) < boardArea
        ) {
          sgfGameEnded = true;
          sgfGameEndedByScore = true;
          sgfGameWinner = P_BLACK;
          sgfGameWhiteFinalScore *= -1;
          if(sgfGameWhiteFinalScore == 0.0)
            sgfGameWinner = C_EMPTY;
        }
        else {
          logger.write("Game ended with invalid score " + fileName + " result " + sgfResult + sizeStr);
          reportSgfDone(false,"ResultInvalidScore");
          return;
        }
      }
      else if(
        (Global::isPrefix(s,"w+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"w+"), sgfGameWhiteFinalScore))
        ||
        (Global::isPrefix(s,"white+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"white+"), sgfGameWhiteFinalScore))
      ) {
        if(
          std::isfinite(sgfGameWhiteFinalScore)
          && std::abs(sgfGameWhiteFinalScore) < boardArea
        ) {
          sgfGameEnded = true;
          sgfGameEndedByScore = true;
          sgfGameWinner = P_WHITE;
          sgfGameWhiteFinalScore *= 1;
          if(sgfGameWhiteFinalScore == 0.0)
            sgfGameWinner = C_EMPTY;
        }
        else {
          logger.write("Game ended with invalid score " + fileName + " result " + sgfResult + sizeStr);
          reportSgfDone(false,"ResultInvalidScore");
          return;
        }
      }
      else if(
        (
          s.find("oves beyond") != string::npos ||
          s.find("oves after") != string::npos ||
          s.find("urther moves") != string::npos
        )
        && (
          s.find("not known") != string::npos ||
          s.find("not recorded") != string::npos ||
          s.find("not given") != string::npos ||
          s.find("no further") != string::npos ||
          s.find("omitted") != string::npos
        )
      ) {
        if(Global::isPrefix(s,"b+") || Global::isPrefix(s,"black+")) {
          sgfGameEnded = true;
          sgfGameEndedButRecordIncomplete = true;
          sgfGameWinner = P_BLACK;
        }
        else if(Global::isPrefix(s,"w+") || Global::isPrefix(s,"white+")) {
          sgfGameEnded = true;
          sgfGameEndedButRecordIncomplete = true;
          sgfGameWinner = P_WHITE;
        }
        else if(Global::isPrefix(s,"draw") || Global::isPrefix(s,"jigo")) {
          sgfGameEnded = true;
          sgfGameEndedButRecordIncomplete = true;
          sgfGameWinner = C_EMPTY;
        }
        else {
          logger.write("Game ended with invalid score " + fileName + " result " + sgfResult + sizeStr);
          reportSgfDone(false,"ResultInvalidScore");
          return;
        }
      }
      else if(s == "b+") {
        sgfGameEnded = true;
        sgfGameEndedButRecordIncomplete = true;
        sgfGameWinner = P_BLACK;
      }
      else if(s == "w+") {
        sgfGameEnded = true;
        sgfGameEndedButRecordIncomplete = true;
        sgfGameWinner = P_BLACK;
      }
      else {
        logger.write("Game ended with unknown result " + fileName + " result " + sgfResult);
        reportSgfDone(false,"ResultUnknown");
        return;
      }
    }

    bool assumeMultipleStartingBlackMovesAreHandicap;
    int overrideNumHandicapStones = -1;
    bool tcIsUnknown = false;
    bool tcIsNone = false;
    bool tcIsAbsolute = false;
    bool tcIsSimple = false;
    bool tcIsByoYomi = false;
    bool tcIsCanadian = false;
    bool tcIsFischer = false;
    double mainTimeSeconds = 0.0;
    double periodTimeSeconds = 0.0;
    int byoYomiPeriods = 0;
    int canadianMoves = 0;
    SimpleDate gameDate;
    int sgfSource;
    if(whatDataSource == "ogs" || whatDataSource == "kgs") {
      assumeMultipleStartingBlackMovesAreHandicap = false;
      if(sgfHandicapParsed >= 0)
        overrideNumHandicapStones = sgfHandicapParsed;

      if(sgfTM != "" && sgfOT != "") {
        if(sgfOT.find(" byo-yomi") != std::string::npos) {
          tcIsByoYomi = true;
          mainTimeSeconds = Global::stringToDouble(sgfTM);
          vector<string> pieces = Global::split(Global::split(sgfOT,' ')[0],'x');
          if(pieces.size() != 2)
            throw StringError("Could not parse OT: " + sgfOT);
          byoYomiPeriods = Global::stringToInt(pieces[0]);
          periodTimeSeconds = Global::stringToDouble(pieces[1]);
          if(byoYomiPeriods <= 0 || byoYomiPeriods > 1000 || !std::isfinite(periodTimeSeconds) || periodTimeSeconds <= 0 || periodTimeSeconds > 100000000) {
            logger.write("Could not parse OT: " + sgfOT);
            reportSgfDone(false,"BadTimeControl");
            return;
          }
        }
        else if(sgfOT.find(" fischer") != std::string::npos) {
          tcIsFischer = true;
          mainTimeSeconds = Global::stringToDouble(sgfTM);
          vector<string> pieces = Global::split(sgfOT,' ');
          periodTimeSeconds = Global::stringToDouble(pieces[0]);
          if(!std::isfinite(periodTimeSeconds) || periodTimeSeconds <= 0 || periodTimeSeconds > 100000000) {
            logger.write("Could not parse OT: " + sgfOT);
            reportSgfDone(false,"BadTimeControl");
            return;
          }
        }
        else if(sgfOT.find(" simple") != std::string::npos) {
          tcIsSimple = true;
          mainTimeSeconds = Global::stringToDouble(sgfTM);
          vector<string> pieces = Global::split(sgfOT,' ');
          periodTimeSeconds = Global::stringToDouble(pieces[0]);
          if(!std::isfinite(periodTimeSeconds) || periodTimeSeconds <= 0 || periodTimeSeconds > 100000000) {
            logger.write("Could not parse OT: " + sgfOT);
            reportSgfDone(false,"BadTimeControl");
            return;
          }
        }
        else if(sgfOT.find(" canadian") != std::string::npos || sgfOT.find(" Canadian") != std::string::npos) {
          tcIsCanadian = true;
          mainTimeSeconds = Global::stringToDouble(sgfTM);
          vector<string> pieces = Global::split(Global::split(sgfOT,' ')[0],'/');
          if(pieces.size() != 2)
            throw StringError("Could not parse OT: " + sgfOT);
          canadianMoves = Global::stringToInt(pieces[0]);
          periodTimeSeconds = Global::stringToDouble(pieces[1]);
          if(canadianMoves <= 0 || canadianMoves > 1000 || !std::isfinite(periodTimeSeconds) || periodTimeSeconds <= 0 || periodTimeSeconds > 100000000) {
            logger.write("Could not parse OT: " + sgfOT);
            reportSgfDone(false,"BadTimeControl");
            return;
          }
          // Switch it to permove time to make it more comparable
          periodTimeSeconds = periodTimeSeconds / canadianMoves;
        }
        if(!std::isfinite(mainTimeSeconds) || mainTimeSeconds < 0 || mainTimeSeconds > 1000000000)
          throw StringError("Could not parse TM: " + sgfTM);
      }
      else if(sgfTM != "") {
        tcIsAbsolute = true;
        mainTimeSeconds = Global::stringToDouble(sgfTM);
        if(!std::isfinite(mainTimeSeconds) || mainTimeSeconds <= 0 || mainTimeSeconds > 1000000000)
          throw StringError("Could not parse TM: " + sgfTM);
      }
      else if(sgfOT != "") {
        throw StringError("Unexpected OGS/KGS time control with OT only: " + sgfOT + " in " + fileName);
      }
      else {
        tcIsNone = true;
      }

      if(whatDataSource == "ogs") {
        gameDate = SimpleDate(sgfDate);
        sgfSource = SGFMetadata::SOURCE_OGS;
      }
      else if(whatDataSource == "kgs") {
        gameDate = SimpleDate(sgfDate);
        sgfSource = SGFMetadata::SOURCE_KGS;
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
    else if(whatDataSource == "fox") {
      assumeMultipleStartingBlackMovesAreHandicap = false;
      if(sgfHandicapParsed >= 0)
        overrideNumHandicapStones = sgfHandicapParsed;

      if(sgfTM != "" && sgfTC != "" && sgfTT != "") {
        tcIsByoYomi = true;
        mainTimeSeconds = Global::stringToDouble(sgfTM);
        byoYomiPeriods = Global::stringToInt(sgfTC);
        periodTimeSeconds = Global::stringToDouble(sgfTT);
        if(byoYomiPeriods <= 0 || byoYomiPeriods > 1000 || !std::isfinite(periodTimeSeconds) || periodTimeSeconds <= 0 || periodTimeSeconds > 100000000) {
          logger.write("Could not parse time control: " + sgfTM + " " + sgfTC + " " + sgfTT);
          reportSgfDone(false,"BadTimeControl");
          return;
        }
      }
      else if(sgfTM != "") {
        tcIsAbsolute = true;
        mainTimeSeconds = Global::stringToDouble(sgfTM);
        if(!std::isfinite(mainTimeSeconds) || mainTimeSeconds <= 0 || mainTimeSeconds > 1000000000)
          throw StringError("Could not parse TM: " + sgfTM);
      }
      else {
        throw StringError("Unexpected Fox time control: " + sgfTimeControl + " in " + fileName);
      }

      gameDate = SimpleDate(sgfDate);
      sgfSource = SGFMetadata::SOURCE_FOX;
    }
    else if(whatDataSource == "gogod") {
      assumeMultipleStartingBlackMovesAreHandicap = false;
      if(sgfHandicapParsed >= 0)
        overrideNumHandicapStones = sgfHandicapParsed;

      //gogod doesn't seem to have TC
      tcIsUnknown = true;

      // Handle unusual date annotations
      string sgfDateToParse;
      size_t startIdx = sgfDate.find_first_of("0123456789");
      if(startIdx != string::npos
         && sgfDate.size() >= startIdx+4
         && Global::isDigit(sgfDate[startIdx+1])
         && Global::isDigit(sgfDate[startIdx+2])
         && Global::isDigit(sgfDate[startIdx+3])
      ) {
        if(sgfDate.size() >= startIdx+7
           && sgfDate[startIdx+4] == '-'
           && Global::isDigit(sgfDate[startIdx+5])
           && Global::isDigit(sgfDate[startIdx+6])
        ) {
          if(sgfDate.size() >= startIdx+10
             && sgfDate[startIdx+7] == '-'
             && Global::isDigit(sgfDate[startIdx+8])
             && Global::isDigit(sgfDate[startIdx+9])
          ) {
            sgfDateToParse = sgfDate.substr(startIdx,10);
          }
          else {
            sgfDateToParse = sgfDate.substr(startIdx,7) + "-01";
          }
        }
        else {
          sgfDateToParse = sgfDate.substr(startIdx,4) + "-01-01";
        }
      }
      else {
        logger.write("Unable to handle date" + fileName + ": " + sgfDate);
        reportSgfDone(false,"GameDateParsing");
        return;
      }

      if(Global::isSuffix(sgfDateToParse,"-00-00"))
        sgfDateToParse = Global::chopSuffix(sgfDateToParse,"-00-00") + "-01-01";
      else if(Global::isSuffix(sgfDateToParse,"-00"))
        sgfDateToParse = Global::chopSuffix(sgfDateToParse,"-00") + "-01";

      try {
        gameDate = SimpleDate(sgfDateToParse);
      }
      catch(const StringError& e) {
        logger.write("Failed to get date for sgf: " + fileName + " " + e.what());
        reportSgfDone(false,"GameBadDate");
        return;
      }
      sgfSource = SGFMetadata::SOURCE_GOGOD;
    }
    else if(whatDataSource == "go4go") {
      assumeMultipleStartingBlackMovesAreHandicap = false;
      if(sgfHandicapParsed >= 0)
        overrideNumHandicapStones = sgfHandicapParsed;
      tcIsUnknown = true;
      gameDate = SimpleDate(sgfDate);
      sgfSource = SGFMetadata::SOURCE_GO4GO;
    }
    else {
      throw StringError("Unknown data source: " + whatDataSource);
    }

    // Error check
    if(foundKgsEntry) {
      if(overrideNumHandicapStones > 0 && overrideNumHandicapStones != kgsEntry.handicap)
        throw StringError("Handicap doesn't match csv for " + fileName);
      float k;
      if(Global::tryStringToFloat(sgfKomi,k) && k != kgsEntry.komi)
        throw StringError("Komi doesn't match csv for " + fileName);
      if(sgfGameEndedByScore && sgfGameWhiteFinalScore != kgsEntry.result)
        throw StringError("Score doesn't match csv for " + fileName);
    }

    SGFMetadata sgfMeta;
    try {
      parseSGFRank(sgfBRank, sgfMeta.inverseBRank, sgfMeta.bIsUnranked, sgfMeta.bRankIsUnknown, whatDataSource);
      parseSGFRank(sgfWRank, sgfMeta.inverseWRank, sgfMeta.wIsUnranked, sgfMeta.wRankIsUnknown, whatDataSource);
    }
    catch(const std::exception& e) {
      logger.write("Failed to get rules for sgf: " + fileName + " " + e.what());
      reportSgfDone(false,"GameBadRankParse");
      return;
    }
    sgfMeta.bIsHuman = !isBotB;
    sgfMeta.wIsHuman = !isBotW;
    sgfMeta.gameIsUnrated = !sgfGameIsRated;
    sgfMeta.gameRatednessIsUnknown = sgfGameRatednessIsUnknown;

    sgfMeta.tcIsUnknown = tcIsUnknown;
    sgfMeta.tcIsNone = tcIsNone;
    sgfMeta.tcIsAbsolute = tcIsAbsolute;
    sgfMeta.tcIsSimple = tcIsSimple;
    sgfMeta.tcIsByoYomi = tcIsByoYomi;
    sgfMeta.tcIsCanadian = tcIsCanadian;
    sgfMeta.tcIsFischer = tcIsFischer;

    sgfMeta.mainTimeSeconds = mainTimeSeconds;
    sgfMeta.periodTimeSeconds = periodTimeSeconds;
    sgfMeta.byoYomiPeriods = byoYomiPeriods;
    sgfMeta.canadianMoves = canadianMoves;

    sgfMeta.boardArea = boardArea;
    sgfMeta.gameDate = gameDate;

    sgfMeta.source = sgfSource;

    const int consecBlackMovesTurns = 6 + boardArea / 40;
    const int skipEarlyPassesTurns = 12 + boardArea / 20;
    const int disallowEarlyPassesTurns = 14 + boardArea / 10;
    const int minGameLenToWrite = 15 + boardArea / 8;
    const int maxKataFinishMoves = 30 + boardArea / 5;

    // If there are any passes in the early moves, start only on the move after the latest such pass.
    int startGameAt = 0;
    for(size_t m = 0; m<sgfMoves.size() && m < skipEarlyPassesTurns; m++) {
      if(sgfMoves[m].loc == Board::PASS_LOC)
        startGameAt = m+2;
    }
    // If there are any passes in moves semi-early moves, reject, we're invalid.
    for(size_t m = skipEarlyPassesTurns; m<sgfMoves.size() && m < disallowEarlyPassesTurns; m++) {
      if(sgfMoves[m].loc == Board::PASS_LOC) {
        logger.write(
          "Pass during moves " +
          Global::intToString(skipEarlyPassesTurns) + "-" + Global::intToString(disallowEarlyPassesTurns)
          + " in " + fileName + sizeStr
        );
        reportSgfDone(false,"MovesSemiEarlyPass");
        return;
      }
    }

    // If there are multiple black moves in a row to start, start at the last one.
    for(size_t m = 1; m<sgfMoves.size() && m < consecBlackMovesTurns; m++) {
      if(sgfMoves[m].pla == P_BLACK && sgfMoves[m-1].pla == P_BLACK) {
        startGameAt = std::max(startGameAt,(int)m);
      }
    }

    if(startGameAt >= sgfMoves.size()) {
      if(verbosity >= 2)
        logger.write("Game too short overlaps with start in " + fileName + " turns " + Global::uint64ToString(sgfMoves.size()) + sizeStr);
      reportSgfDone(false,"MovesTooShortOverlaps");
      return;
    }

    // Fastforward through initial bit before we start
    for(size_t m = 0; m<startGameAt; m++) {
      Move move = sgfMoves[m];

      // Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
      if(hist.isGameFinished || hist.encorePhase > 0) {
        logger.write("Game unexpectedly ended near start in " + fileName + sizeStr);
        reportSgfDone(false,"MovesUnexpectedGameEndNearStart");
        return;
      }
      bool suc = hist.isLegal(board,move.loc,move.pla);
      if(!suc) {
        logger.write("Illegal move near start in " + fileName + " move " + Location::toString(move.loc, board.x_size, board.y_size) + sizeStr);
        reportSgfDone(false,"MovesIllegalMoveNearStart");
        return;
      }
      bool preventEncore = false;
      hist.makeBoardMoveAssumeLegal(board,move.loc,move.pla,NULL,preventEncore);
    }

    // Use the actual first player of the game.
    nextPla = sgfMoves[startGameAt].pla;

    // Set up handicap behavior
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setOverrideNumHandicapStones(overrideNumHandicapStones);
    int numExtraBlack = hist.computeNumHandicapStones();

    const double drawEquivalentWinsForWhite = 0.5;
    const Player playoutDoublingAdvantagePla = C_EMPTY;
    const double playoutDoublingAdvantage = 0.0;

    vector<Board> boards;
    vector<BoardHistory> hists;
    vector<Player> nextPlas;
    vector<ValueTargets> whiteValueTargets;
    vector<Loc> moves;
    vector<vector<PolicyTargetMove>> policyTargets;
    vector<double> trainingWeights;

    for(size_t m = startGameAt; m<sgfMoves.size()+1; m++) {
      boards.push_back(board);
      hists.push_back(hist);
      nextPlas.push_back(nextPla);

      ValueTargets targets = makeWhiteValueTarget(
        board, hist, nextPla, drawEquivalentWinsForWhite, playoutDoublingAdvantagePla, playoutDoublingAdvantage, nnEval
      );
      whiteValueTargets.push_back(targets);

      if(m >= sgfMoves.size())
        break;
      // Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
      if(hist.isGameFinished || hist.encorePhase > 0)
        break;

      Move move = sgfMoves[m];
      moves.push_back(move.loc);
      policyTargets.push_back(vector<PolicyTargetMove>());
      policyTargets[policyTargets.size()-1].push_back(PolicyTargetMove(move.loc,1));

      // We want policies learned from human moves to be compatible with KataGo using them in a search which may use
      // stricter computer rules. So if a player passes, we do NOT train on it. And do NOT train on the opponent's response
      // to a pass, to avoid nonpunishment of bad passes under strict game end rules.
      double trainingWeight = 1.0;
      if(move.loc == Board::PASS_LOC || (moves.size() > 1 && moves[moves.size()-2] == Board::PASS_LOC))
        trainingWeight = 0.0;
      trainingWeights.push_back(trainingWeight);

      // Bad data checks
      if(move.pla != nextPla && m > 0) {
        logger.write("Bad SGF " + fileName + " due to non-alternating players on turn " + Global::intToString(m));
        reportSgfDone(false,"MovesNonAlternating");
        return;
      }
      bool suc = hist.isLegal(board,move.loc,move.pla);
      if(!suc) {
        logger.write("Illegal move in " + fileName + " turn " + Global::intToString(m) + " move " + Location::toString(move.loc, board.x_size, board.y_size));
        reportSgfDone(false,"MovesIllegal");
        return;
      }
      bool preventEncore = false;
      hist.makeBoardMoveAssumeLegal(board,move.loc,move.pla,NULL,preventEncore);
      nextPla = getOpp(move.pla);
    }

    // Now that we finished game play from the SGF, pop off as many passes as possible
    while(moves.size() > 0 && moves[moves.size()-1] == Board::PASS_LOC) {
      boards.pop_back();
      hists.pop_back();
      nextPlas.pop_back();
      whiteValueTargets.pop_back();
      moves.pop_back();
      policyTargets.pop_back();
      trainingWeights.pop_back();

      board = boards[boards.size()-1];
      hist = hists[hists.size()-1];
      nextPla = nextPlas[nextPlas.size()-1];
    }

    if(policyTargets.size() < minGameLenToWrite) {
      if(verbosity >= 2)
        logger.write("Game too short in " + fileName + " turns " + Global::uint64ToString(policyTargets.size()) + sizeStr);
      reportSgfDone(false,"MovesTooShort");
      return;
    }

    int endIdxFromSgf = (int)policyTargets.size();
    (void)endIdxFromSgf;

    string usageStr;
    float valueTargetWeight = 0.0f;
    bool hasOwnershipTargets = false;
    bool hasForcedWinner = false; // Did we manually modify value targets to declare the winner?
    Color finalOwnership[Board::MAX_ARR_SIZE];
    Color finalFullArea[Board::MAX_ARR_SIZE];
    float finalWhiteScoring[Board::MAX_ARR_SIZE];
    std::fill(finalFullArea,finalFullArea+Board::MAX_ARR_SIZE,C_EMPTY);
    std::fill(finalOwnership,finalOwnership+Board::MAX_ARR_SIZE,C_EMPTY);
    std::fill(finalWhiteScoring,finalWhiteScoring+Board::MAX_ARR_SIZE,0.0f);

    // If this was a scored position, fill out the remaining turns with KataGo making moves, just to clean
    // up the position and carry out encore for JP-style rules.
    if(sgfGameEndedByScore && !rulesDisableValueTraining) {
      Search* search = threadSearchers[threadIdx];

      // Before we have KataGo play anything, require KataGo to agree that the game ended in a position
      // that was basically finished.
      bool gameIsNearEnd = false;
      {
        ReportedSearchValues valuesIfBlackFirst;
        vector<double> ownershipIfBlackFirst;
        {
          search->setPosition(P_BLACK,board,hist);
          search->runWholeSearchAndGetMove(P_BLACK);
          valuesIfBlackFirst = search->getRootValuesRequireSuccess();
          ownershipIfBlackFirst = search->getAverageTreeOwnership();
        }
        ReportedSearchValues valuesIfWhiteFirst;
        vector<double> ownershipIfWhiteFirst;
        {
          search->setPosition(P_WHITE,board,hist);
          search->runWholeSearchAndGetMove(P_WHITE);
          valuesIfWhiteFirst = search->getRootValuesRequireSuccess();
          ownershipIfWhiteFirst = search->getAverageTreeOwnership();
        }

        double absAvgLead = abs(0.5 * (valuesIfBlackFirst.lead + valuesIfWhiteFirst.lead));
        double leadDiff = abs(valuesIfBlackFirst.lead - valuesIfWhiteFirst.lead);
        double winlossDiff = abs(valuesIfBlackFirst.winLossValue - valuesIfWhiteFirst.winLossValue);
        // Remaining difference between side to move is less than 4 pointsish, and the winrate is close
        if(leadDiff < 3.5 + 0.05 * absAvgLead && winlossDiff < 0.2) {
          const double maxFractionOfBoardUnsettled = 0.10 + 6.0 / boardArea;
          int blackCountUnsettled = 0;
          int whiteCountUnsettled = 0;
          int blackWhiteVeryDifferent = 0;
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              int pos = NNPos::xyToPos(x,y,nnEval->getNNXLen());
              if(abs(ownershipIfBlackFirst[pos]) < 0.9)
                blackCountUnsettled += 1;
              if(abs(ownershipIfWhiteFirst[pos]) < 0.9)
                whiteCountUnsettled += 1;
              if(abs(ownershipIfBlackFirst[pos] - ownershipIfWhiteFirst[pos]) > 1.0)
                blackWhiteVeryDifferent += 1;
            }
          }
          if(
            blackWhiteVeryDifferent / boardArea < maxFractionOfBoardUnsettled
            && blackCountUnsettled / boardArea < maxFractionOfBoardUnsettled
            && whiteCountUnsettled / boardArea < maxFractionOfBoardUnsettled
          ) {
            gameIsNearEnd = true;
          }
        }
      }

      if(!gameIsNearEnd) {
        logger.write("SGF " + fileName + " ended with score but is not near ending position");
        // if(verbosity >= 3) {
        //   ostringstream out;
        //   out << board << endl;
        //   logger.write(out.str());
        // }
        valueTargetWeight = 0.0f;
        hasOwnershipTargets = false;
        usageStr = "NoValueScoredNotNearEnding";
      }
      else {
        bool gameFinishedProperly = false;
        vector<Loc> locsBuf;
        vector<double> playSelectionValuesBuf;
        for(int extraM = 0; extraM<maxKataFinishMoves; extraM++) {
          search->setPosition(nextPla,board,hist);
          Loc moveLoc = search->runWholeSearchAndGetMove(nextPla);

          moves.push_back(moveLoc);
          policyTargets.push_back(vector<PolicyTargetMove>());
          Play::extractPolicyTarget(policyTargets[policyTargets.size()-1],search,search->rootNode,locsBuf,playSelectionValuesBuf);

          // KataGo cleanup moves get weighted a tiny bit, so we can preserve the instinct to cleanup
          // in the right way for strict rules as according to KataGo's cleanup instincts.
          trainingWeights.push_back(0.05);

          bool suc = hist.isLegal(board,moveLoc,nextPla);
          (void)suc;
          assert(suc);

          bool preventEncore = false;
          hist.makeBoardMoveAssumeLegal(board,moveLoc,nextPla,NULL,preventEncore);
          nextPla = getOpp(nextPla);

          boards.push_back(board);
          hists.push_back(hist);
          nextPlas.push_back(nextPla);

          ValueTargets targets = makeWhiteValueTarget(
            board, hist, nextPla, drawEquivalentWinsForWhite, playoutDoublingAdvantagePla, playoutDoublingAdvantage, nnEval
          );
          whiteValueTargets.push_back(targets);

          if(hist.isGameFinished) {
            gameFinishedProperly = true;
            break;
          }
        }

        if(gameFinishedProperly) {
          // Ownership stuff!
          hasOwnershipTargets = true;
          hists[hists.size()-1].endAndScoreGameNow(board,finalOwnership);
          board.calculateArea(finalFullArea, true, true, true, hist.rules.multiStoneSuicideLegal);
          NNInputs::fillScoring(board,finalOwnership,hist.rules.taxRule == Rules::TAX_ALL,finalWhiteScoring);

          // Make sure KataGo didn't leave huge unscored regions due to passing weirdness, and make sure the scoring agrees with
          // a historyless nn eval of the position.
          bool weirdScoringState = false;
          {
            NNResultBuf buf;
            const int maxHistory = 0;
            getNNEval(board,hist,nextPla,drawEquivalentWinsForWhite,playoutDoublingAdvantagePla,playoutDoublingAdvantage,maxHistory,nnEval,buf);
            int ownershipDisagreeCount = 0;
            int unscoredCount = 0;
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                int pos = NNPos::xyToPos(x,y,nnEval->getNNXLen());
                Loc loc = Location::getLoc(x,y,board.x_size);
                if(buf.result->whiteOwnerMap[pos] > 0.90 && finalOwnership[loc] != P_WHITE)
                  ownershipDisagreeCount += 1;
                else if(buf.result->whiteOwnerMap[pos] < -0.90 && finalOwnership[loc] != P_BLACK)
                  ownershipDisagreeCount += 1;

                if(finalOwnership[loc] == C_EMPTY)
                  unscoredCount += 1;
              }
            }

            const double maxFractionOfBoardUnscored = 0.01 + 5.0 / boardArea;
            const double maxFractionOfBoardDisagree = 0.01 + 5.0 / boardArea;
            if(
              unscoredCount > boardArea * maxFractionOfBoardUnscored ||
              ownershipDisagreeCount > boardArea * maxFractionOfBoardDisagree
            ) {
              weirdScoringState = true;
            }
          }

          // Does our result match the sgf recorded result for winner?
          // If no, reduce weight a lot
          // Also check for weird scoring state
          if(weirdScoringState) {
            logger.write("SGF " + fileName + " ended with score and was finishable but scoring state was weird");
            valueTargetWeight = 0.0;
            hasOwnershipTargets = false;
            usageStr = "NoValueWeirdScoringState";
            if(verbosity >= 3) {
              ostringstream out;
              out << endIdxFromSgf << endl;
              out << boards[endIdxFromSgf] << endl;
              out << boards.size() << endl;
              out << board << endl;
              WriteSgf::writeSgf(out, "", "", hist, NULL, true, true);
              logger.write(out.str());
            }
          }
          else if(hist.winner == sgfGameWinner) {
            if(verbosity >= 2)
              logger.write("SGF " + fileName + " ended with score and was good data " + Global::uint64ToString(moves.size()-endIdxFromSgf));
            valueTargetWeight = 1.0f;
            usageStr = "GoodValueAndOwnerScored";
          }
          else {
            logger.write("SGF " + fileName + " ended with score and was finishable but got different result than sgf " + Global::uint64ToString(moves.size()-endIdxFromSgf));
            valueTargetWeight = 0.1f;
            usageStr = "LowWeightValueAndOwnerDisagreeingResult";
            if(verbosity >= 3) {
              ostringstream out;
              out << endIdxFromSgf << endl;
              out << boards[endIdxFromSgf] << endl;
              out << boards.size() << endl;
              out << board << endl;
              WriteSgf::writeSgf(out, "", "", hist, NULL, true, true);
              logger.write(out.str());
            }
          }
        }
        else {
          logger.write("SGF " + fileName + " ended with score but it doesn't finish in a reasonable time");
          if(verbosity >= 3) {
            ostringstream out;
            out << endIdxFromSgf << endl;
            out << boards[endIdxFromSgf] << endl;
            out << boards.size() << endl;
            out << board << endl;
            logger.write(out.str());
          }
          valueTargetWeight = 0.0f;
          hasOwnershipTargets = false;
          usageStr = "NoValueDoesntFinish";
        }
      }
    }
    else if((sgfGameEndedByResign || sgfGameEndedByTime) && !rulesDisableValueTraining) {
      const double whiteFinalWinProb = std::max(whiteValueTargets[whiteValueTargets.size()-1].win, 0.0f);
      const double whiteFinalLossProb = std::max(whiteValueTargets[whiteValueTargets.size()-1].loss, 0.0f);
      if(sgfGameWinner == P_WHITE) {
        valueTargetWeight = 2.0f * (float)std::max(0.0, whiteFinalWinProb - 0.5);

        // Assume white catches up in score at the same rate from turn 0 to 200 - does that leave white ahead?
        if(numExtraBlack > 0 && board.x_size == 19 && board.y_size == 19 && whiteValueTargets.size() < 200 && whiteValueTargets.size() > 30) {
          double change = whiteValueTargets[whiteValueTargets.size()-1].lead - whiteValueTargets[0].lead;
          double extrapolatedChange = change * (200.0 / whiteValueTargets.size());
          if(whiteValueTargets[0].lead + extrapolatedChange > 0.5) {
            // Black resigned while ahead, but white was catching up in a handicp game, so count full weight.
            valueTargetWeight = 1.0f;
            usageStr = "FullValueBlackHandicapResign";
          }
        }
        hasOwnershipTargets = false;
        hasForcedWinner = true;
        whiteValueTargets[whiteValueTargets.size()-1] = makeForcedWinnerValueTarget(P_WHITE);
        usageStr = (
          valueTargetWeight <= 0.0 ? "NoValue" :
          valueTargetWeight <= 0.9 ? "ReducedValue" :
          "FullValue"
        );
      }
      else if(sgfGameWinner == P_BLACK) {
        // Player resigned when they were still ahead. Downweight and/or don't use.
        valueTargetWeight = 2.0f * (float)std::max(0.0, whiteFinalLossProb - 0.5);
        hasOwnershipTargets = false;
        hasForcedWinner = true;
        whiteValueTargets[whiteValueTargets.size()-1] = makeForcedWinnerValueTarget(P_BLACK);
        usageStr = (
          valueTargetWeight <= 0.0 ? "NoValue" :
          valueTargetWeight <= 0.9 ? "ReducedValue" :
          "FullValue"
        );
      }
      else {
        ASSERT_UNREACHABLE;
      }
      // For games ending by time, treat them similarly to resigns except use a lower and sharper weighting for outcome.
      // Square so that we require the winrate to have been more clear to accept such a loss, and further downeight by constant factor.
      if(sgfGameEndedByTime) {
        valueTargetWeight = 0.5f * valueTargetWeight * valueTargetWeight;
        usageStr += "EndedByTime";
      }
      else {
        usageStr += "EndedByResign";
      }
    }
    else if(sgfGameEndedButRecordIncomplete && !rulesDisableValueTraining) {
      const double whiteFinalWinProb = std::max(whiteValueTargets[whiteValueTargets.size()-1].win, 0.0f);
      const double whiteFinalLossProb = std::max(whiteValueTargets[whiteValueTargets.size()-1].loss, 0.0f);
      if(sgfGameWinner == P_WHITE) {
        // Downweight results that are really very contrary to the lead as misrecords but otherwise use it.
        valueTargetWeight = (float)pow(whiteFinalWinProb, 0.25);
        hasOwnershipTargets = false;
        hasForcedWinner = true;
        whiteValueTargets[whiteValueTargets.size()-1] = makeForcedWinnerValueTarget(P_WHITE);
        usageStr = (
          valueTargetWeight <= 0.0 ? "NoValue" :
          valueTargetWeight <= 0.9 ? "ReducedValue" :
          "FullValue"
        );
      }
      else if(sgfGameWinner == P_BLACK) {
        // Downweight results that are really very contrary to the lead as misrecords but otherwise use it.
        valueTargetWeight = (float)pow(whiteFinalLossProb, 0.25);
        hasOwnershipTargets = false;
        hasForcedWinner = true;
        whiteValueTargets[whiteValueTargets.size()-1] = makeForcedWinnerValueTarget(P_BLACK);
      }
      else if(sgfGameWinner == C_EMPTY) {
        // Downweight results that are really very contrary to the lead as misrecords but otherwise use it.
        valueTargetWeight = (float)pow(whiteFinalWinProb * whiteFinalLossProb, 0.25);
        hasOwnershipTargets = false;
        hasForcedWinner = true;
        whiteValueTargets[whiteValueTargets.size()-1] = makeForcedWinnerValueTarget(C_EMPTY);
      }
      else {
        ASSERT_UNREACHABLE;
      }

      usageStr = (
        valueTargetWeight <= 0.0 ? "NoValue" :
        valueTargetWeight <= 0.9 ? "ReducedValue" :
        "FullValue"
      );
      usageStr += "EndedButRecordIncomplete";
    }
    else {
      valueTargetWeight = 0.0f;
      hasOwnershipTargets = false;
      usageStr += "NoValueOtherResult";
      (void)sgfGameEnded;
      (void)sgfGameEndedByForfeit;
      (void)sgfGameEndedByOther;
    }

    // Everything in gogod is sketchy for value learning, due to the uneven mix of data
    // and very different komis.
    if(whatDataSource == "gogod") {
      valueTargetWeight *= 0.2f;
    }

    // The lead target needs to be the final game outcome because otherwise it will just learn to
    // judge what KataGo's notion of the lead is which is not reflective of the player level
    // being modeled.
    // Only fill in the lead when we have ownership targets, with the game scored correctly.
    // Lead is going to be pretty high variance,
    if(hasOwnershipTargets) {
      assert(!hasForcedWinner);
      for(size_t m = 0; m<whiteValueTargets.size()-1; m++) {
        const ValueTargets& finalTargets = whiteValueTargets[whiteValueTargets.size()-1];
        whiteValueTargets[m].hasLead = true;
        whiteValueTargets[m].lead = finalTargets.score;
      }
    }

    for(size_t m = 0; m<policyTargets.size(); m++) {
      int turnIdx = (int)m;
      int64_t unreducedNumVisits = 0;
      const double policySurprise = 0;
      const double policyEntropy = 0;
      const double searchEntropy = 0;

      NNRawStats nnRawStats;
      nnRawStats.whiteWinLoss = 0;
      nnRawStats.whiteScoreMean = 0;
      nnRawStats.policyEntropy = 0;

      // If we manually hacked the td value, prevent an abrupt game end with a forced value from biasing it too much
      // by disabling the td targets if we get too close.
      float tdValueTargetWeight = valueTargetWeight;
      if(hasForcedWinner && whiteValueTargets.size() - m < (5 + boardArea / 8)) {
        tdValueTargetWeight = 0.0f;
      }

      // Lead is going to be a lot higher variance than KataGo normally handles, so reduce it a bunch.
      float leadTargetWeightFactor = 0.25f;

      const bool isSidePosition = false;
      const int numNeuralNetsBehindLatest = 0;
      const Hash128 gameHash;
      const std::vector<ChangedNeuralNet*> changedNeuralNets;
      const bool hitTurnLimit = false;
      const int mode = 0;

      if(
        trainingWeights[m] > 1e-8
        && ((nextPlas[m] == P_BLACK && shouldTrainB) || (nextPlas[m] == P_WHITE && shouldTrainW))
        && rand.nextDouble() < keepProb
      ) {
        dataBuffer->addRow(
          boards[m],
          hists[m],
          nextPlas[m],
          hists[0],
          hists[hists.size()-1],
          turnIdx,
          (float)trainingWeights[m],
          unreducedNumVisits,
          &policyTargets[m],
          (m+1 < policyTargets.size() && rand.nextDouble() < trainingWeights[m+1] ? &policyTargets[m+1] : NULL),
          policySurprise,
          policyEntropy,
          searchEntropy,
          whiteValueTargets,
          turnIdx,
          valueTargetWeight,
          tdValueTargetWeight,
          leadTargetWeightFactor,
          nnRawStats,
          &boards[boards.size()-1],
          (hasOwnershipTargets ? finalFullArea : NULL),
          (hasOwnershipTargets ? finalOwnership : NULL),
          (hasOwnershipTargets ? finalWhiteScoring : NULL),
          &boards,
          isSidePosition,
          numNeuralNetsBehindLatest,
          drawEquivalentWinsForWhite,
          playoutDoublingAdvantagePla,
          playoutDoublingAdvantage,
          gameHash,
          changedNeuralNets,
          hitTurnLimit,
          numExtraBlack,
          mode,
          &sgfMeta,
          rand
        );
      }
    }

    if(dataBuffer->curRows >= maxApproxRowsPerTrainFile)
      saveDataBuffer(dataBuffer,rand);

    {
      std::lock_guard<std::mutex> lock(statsLock);
      if(shouldTrainB)
        acceptedGameCountByUsername[sgfBUsername] += 1;
      if(shouldTrainW)
        acceptedGameCountByUsername[sgfWUsername] += 1;
      if(shouldTrainB)
        acceptedGameCountByRank[sgfBRank] += 1;
      if(shouldTrainW)
        acceptedGameCountByRank[sgfWRank] += 1;
      acceptedGameCountByBSize[bSizeStr] += 1;
      acceptedGameCountByRules[sgfRules] += 1;
      acceptedGameCountByKomi[sgfKomi] += 1;
      acceptedGameCountByHandicap[sgfHandicap] += 1;
      acceptedGameCountByResult[sgfResult] += 1;
      acceptedGameCountByTimeControl[sgfTimeControl] += 1;
      acceptedGameCountByIsRated[sgfGameRatednessIsUnknown ? "unknown" : sgfGameIsRated ? "true" : "false"] += 1;
      acceptedGameCountByEvent[sgfEvent] += 1;
      acceptedGameCountByPlace[sgfPlace] += 1;
      acceptedGameCountByUsage[usageStr] += 1;
      acceptedGameCountByYear[Global::intToString(gameDate.year)] += 1;
    }

    reportSgfDone(true,"Used");
  };

  Parallel::iterRange(
    numWorkerThreads,
    std::min(maxFilesToLoad,sgfFiles.size()),
    std::function<void(int,size_t)>(processSgf)
  );

  auto saveDataBufferJob = [&](int threadIdx, size_t bufferToSaveIdx) {
    (void)threadIdx;
    Rand& rand = *threadRands[bufferToSaveIdx];
    TrainingWriteBuffers* dataBuffer = threadDataBuffers[bufferToSaveIdx];
    if(dataBuffer->curRows > 0)
      saveDataBuffer(dataBuffer,rand);
  };


  Parallel::iterRange(
    std::min(8,numWorkerThreads),
    numWorkerThreads,
    std::function<void(int,size_t)>(saveDataBufferJob)
  );

  auto printGameCountsMap = [&](
    const std::map<string,int64_t> counts,
    const string& label,
    bool sortByCount,
    std::map<string,int64_t>* accCounts
  ) {
    logger.write("===================================================");
    logger.write("Counts by " + label);
    std::vector<std::pair<string, int64_t>> pairVec(counts.begin(), counts.end());
    if(sortByCount) {
      std::sort(
        pairVec.begin(),
        pairVec.end(),
        [](const std::pair<string, int64_t>& a, const std::pair<string, int64_t>& b) {
          return a.second > b.second;
        }
      );
    }
    for(const auto& keyAndCount: pairVec) {
      const string& key = keyAndCount.first;
      if(accCounts != NULL && contains(*accCounts,key)) {
        logger.write(key + ": " + Global::int64ToString(keyAndCount.second) + " (acc " + Global::int64ToString((*accCounts)[key]) + ")");
      }
      else {
        logger.write(key + ": " + Global::int64ToString(keyAndCount.second));
      }
    }
    logger.write("===================================================");
  };
  printGameCountsMap(gameCountByUsername,"username",true,&acceptedGameCountByUsername);
  printGameCountsMap(acceptedGameCountByUsername,"username (accepted)",true,NULL);
  printGameCountsMap(gameCountByRank,"rank",false,&acceptedGameCountByRank);
  printGameCountsMap(acceptedGameCountByRank,"rank (accepted)",false,NULL);
  printGameCountsMap(gameCountByBSize,"bSize",true,&acceptedGameCountByBSize);
  printGameCountsMap(acceptedGameCountByBSize,"bSize (accepted)",true,NULL);
  printGameCountsMap(gameCountByTimeControl,"time control",false,&acceptedGameCountByTimeControl);
  printGameCountsMap(acceptedGameCountByTimeControl,"time control (accepted)",false,NULL);
  printGameCountsMap(gameCountByRules,"rules",false,&acceptedGameCountByRules);
  printGameCountsMap(acceptedGameCountByRules,"rules (accepted)",false,NULL);
  printGameCountsMap(gameCountByKomi,"komi",false,&acceptedGameCountByKomi);
  printGameCountsMap(acceptedGameCountByKomi,"komi (accepted)",false,NULL);
  printGameCountsMap(gameCountByHandicap,"handicap",false,&acceptedGameCountByHandicap);
  printGameCountsMap(acceptedGameCountByHandicap,"handicap (accepted)",false,NULL);
  printGameCountsMap(gameCountByResult,"result",false,&acceptedGameCountByResult);
  printGameCountsMap(acceptedGameCountByResult,"result (accepted)",false,NULL);
  printGameCountsMap(gameCountByEvent,"event",false,&acceptedGameCountByEvent);
  printGameCountsMap(acceptedGameCountByEvent,"event (accepted)",false,NULL);
  if(whatDataSource != "ogs") {
    printGameCountsMap(gameCountByPlace,"place",false,&acceptedGameCountByPlace);
    printGameCountsMap(acceptedGameCountByPlace,"place (accepted)",false,NULL);
  }
  printGameCountsMap(gameCountByIsRated,"isRated",false,&acceptedGameCountByIsRated);
  printGameCountsMap(acceptedGameCountByIsRated,"isRated (accepted)",false,NULL);
  printGameCountsMap(acceptedGameCountByUsage,"usage (accepted)",false,NULL);
  printGameCountsMap(acceptedGameCountByYear,"year (accepted)",false,NULL);
  printGameCountsMap(doneGameCountByReason,"done",false,NULL);
  printGameCountsMap(warningsByReason,"warnings",false,NULL);

  logger.write(nnEval->getModelFileName());
  logger.write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
  logger.write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
  logger.write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));

  logger.write("All done");

  for(int i = 0; i<numWorkerThreads; i++) {
    delete threadDataBuffers[i];
    delete threadRands[i];
    delete threadSearchers[i];
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
