#include "core/global.h"

namespace MainCmds {
  int analysis(const std::vector<std::string>& args);
  int benchmark(const std::vector<std::string>& args);
  int contribute(const std::vector<std::string>& args);
  int evalsgf(const std::vector<std::string>& args);
  int gatekeeper(const std::vector<std::string>& args);
  int genconfig(const std::vector<std::string>& args, const std::string& firstCommand);
  int gtp(const std::vector<std::string>& args);
  int tuner(const std::vector<std::string>& args);
  int match(const std::vector<std::string>& args);
  int matchauto(const std::vector<std::string>& args);
  int selfplay(const std::vector<std::string>& args);

  int runtests(const std::vector<std::string>& args);
  int runnnlayertests(const std::vector<std::string>& args);
  int runnnontinyboardtest(const std::vector<std::string>& args);
  int runnnsymmetriestest(const std::vector<std::string>& args);
  int runoutputtests(const std::vector<std::string>& args);
  int runsearchtests(const std::vector<std::string>& args);
  int runsearchtestsv3(const std::vector<std::string>& args);
  int runsearchtestsv8(const std::vector<std::string>& args);
  int runselfplayinittests(const std::vector<std::string>& args);
  int runselfplayinitstattests(const std::vector<std::string>& args);
  int runsekitrainwritetests(const std::vector<std::string>& args);
  int runnnonmanyposestest(const std::vector<std::string>& args);
  int runnnbatchingtest(const std::vector<std::string>& args);
  int runownershiptests(const std::vector<std::string>& args);
  int runtinynntests(const std::vector<std::string>& args);
  int runnnevalcanarytests(const std::vector<std::string>& args);
  int runbeginsearchspeedtest(const std::vector<std::string>& args);
  int runownershipspeedtest(const std::vector<std::string>& args);
  int runsleeptest(const std::vector<std::string>& args);

  int samplesgfs(const std::vector<std::string>& args);
  int dataminesgfs(const std::vector<std::string>& args);
  int genbook(const std::vector<std::string>& args);
  int checkbook(const std::vector<std::string>& args);

  int trystartposes(const std::vector<std::string>& args);
  int viewstartposes(const std::vector<std::string>& args);

  int demoplay(const std::vector<std::string>& args);
  int printclockinfo(const std::vector<std::string>& args);
  int sampleinitializations(const std::vector<std::string>& args);

  int sandbox();
}

namespace Version {
  std::string getKataGoVersion();
  std::string getKataGoVersionForHelp();
  std::string getKataGoVersionFullInfo();
  std::string getGitRevision();
  std::string getGitRevisionWithBackend();
}
