#include "../core/logger.h"

#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/config_parser.h"
#include "../core/rand.h"
#include "../core/makedir.h"

using namespace std;

Logger::Logger(
  ConfigParser* cfg,
  bool logToStdoutDefault,
  bool logToStderrDefault,
  bool logTimeDefault,
  bool logConfigContents_
):
  logToStdout(logToStdoutDefault),
  logToStderr(logToStderrDefault),
  logTime(logTimeDefault),
  logConfigContents(logConfigContents_),
  header(),
  ostreams(),
  files(),
  logBufs(),
  mutex(),
  isDisabled(false)
{
  if(cfg) {
    header = "Running with following config:\n" + cfg->getAllKeyVals();
    if(cfg->contains("logToStdout"))
      logToStdout = cfg->getBool("logToStdout");

    if(cfg->contains("logToStderr"))
      logToStderr = cfg->getBool("logToStderr");

    if(cfg->contains("logTimeStamp"))
      logTime = cfg->getBool("logTimeStamp");

    if((int)cfg->contains("logFile") + (int)cfg->contains("logDir") + (int)cfg->contains("logDirDated") > 1)
      throw StringError("Cannot specify more than one of logFile and logDir and logDirDated in config");
    else if(cfg->contains("logFile"))
      addFile(cfg->getString("logFile"), false);
    else if(cfg->contains("logDir")) {
      MakeDir::make(cfg->getString("logDir"));
      Rand rand;
      addFile(cfg->getString("logDir") + "/" + DateTime::getCompactDateTimeString() + "-" + Global::uint32ToHexString(rand.nextUInt()) + ".log", false);
    }
    else if(cfg->contains("logDirDated")) {
      MakeDir::make(cfg->getString("logDirDated"));
      Rand rand;
      MakeDir::make(cfg->getString("logDirDated") + "/" + DateTime::getCompactDateTimeString());
      addFile(cfg->getString("logDirDated") + "/" + DateTime::getCompactDateTimeString() + "/" + Global::uint32ToHexString(rand.nextUInt()) + ".log", false);
    }
  }

  if(!header.empty()) {
    write(header);
  }
}

Logger::~Logger()
{
  for(size_t i = 0; i<logBufs.size(); i++)
    delete logBufs[i];

  for(size_t i = 0; i<files.size(); i++) {
    files[i]->close();
    delete files[i];
  }
}

bool Logger::isLoggingToStdout() const {
  return logToStdout;
}

bool Logger::isLoggingToStderr() const {
  return logToStderr;
}

void Logger::addOStream(ostream& out, bool afterCreation) {
  lock_guard<std::mutex> lock(mutex);
  ostreams.push_back(&out);

  if(afterCreation && !header.empty()) {
    time_t time = DateTime::getNow();
    writeLocked(header, true, out, time);
  }
}
void Logger::addFile(const string& file, bool afterCreation) {
  if(file == "")
    return;
  ofstream* out = new ofstream();
  try {
    FileUtils::open(*out, file, ofstream::app);
  }
  catch(const StringError& e) {
    write(string("WARNING: could not open file for logging: ") + e.what());
    cerr << "WARNING: could not open file for logging: " << e.what() << endl;
    out->close();
    delete out;
    return;
  }

  lock_guard<std::mutex> lock(mutex);
  files.push_back(out);

  if(afterCreation && !header.empty()) {
    time_t time = DateTime::getNow();
    writeLocked(header, true, *out, time);
  }
}

void Logger::setDisabled(bool b) {
  lock_guard<std::mutex> lock(mutex);
  isDisabled = b;
}

void Logger::write(const string& str, bool endLine) {
  lock_guard<std::mutex> lock(mutex);
  time_t time = DateTime::getNow();

  if(logToStdout) {
    writeLocked(str, endLine, cout, time);
  }
  if(logToStderr) {
    writeLocked(str, endLine, cerr, time);
  }
  for(size_t i = 0; i<ostreams.size(); i++) {
    ostream& out = *(ostreams[i]);
    writeLocked(str, endLine, out, time);
  }
  for(size_t i = 0; i<files.size(); i++) {
    ofstream& out = *(files[i]);
    writeLocked(str, endLine, out, time);
  }
}

void Logger::writeLocked(const std::string &str, bool endLine, std::ostream &out, const time_t& time)
{
  if(isDisabled)
    return;
  const char* timeFormat = "%F %T%z: ";

  if(logTime) {
    DateTime::writeTimeToStream(out, timeFormat, time);
    out << str;
  }
  else
    out << ": " << str;
  if(endLine)
    out << std::endl;
  else
    out << std::flush;
}

void Logger::write(const string& str) {
  write(str,true);
}

void Logger::writeNoEndline(const string& str) {
  write(str,false);
}

ostream* Logger::createOStream() {
  unique_lock<std::mutex> lock(mutex);
  LogBuf* logBuf = new LogBuf(this);
  logBufs.push_back(logBuf);
  lock.unlock();
  return new ostream(logBuf);
}

LogBuf::LogBuf(Logger* l)
  :stringbuf(),logger(l)
{}

LogBuf::~LogBuf()
{}

int LogBuf::sync() {
  const string& str = this->str();
  logger->writeNoEndline(str);
  this->str("");
  return 0;
}

void Logger::logThreadUncaught(const string& name, Logger* logger, std::function<void()> f) {
  try {
    f();
  }
  catch(const exception& e) {
    if(logger != NULL)
      logger->write(string("ERROR: " + name + " loop thread failed: ") + e.what());
    else
      cerr << (string("ERROR: " + name + " loop thread failed: " )+ e.what()) << endl;
    std::this_thread::sleep_for(std::chrono::duration<double>(5.0));
    throw;
  }
  catch(...) {
    if(logger != NULL)
      logger->write("ERROR: " + name + " loop thread failed with unexpected throw");
    else
      cerr << "ERROR: " + name + " loop thread failed with unexpected throw" << endl;
    std::this_thread::sleep_for(std::chrono::duration<double>(5.0));
    throw;
  }
}
