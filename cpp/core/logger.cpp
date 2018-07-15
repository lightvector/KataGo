#include "../core/logger.h"
#include <iomanip>
#include <chrono>

Logger::Logger()
  :logToStdout(false),logToStderr(false),files()
{}

Logger::~Logger()
{
  for(size_t i = 0; i<logBufs.size(); i++)
    delete logBufs[i];

  for(size_t i = 0; i<files.size(); i++) {
    files[i]->close();
    delete files[i];
  }
}

void Logger::setLogToStdout(bool b) {
  logToStdout = b;
}
void Logger::setLogToStderr(bool b) {
  logToStderr = b;
}
void Logger::addFile(const string& file) {
  files.push_back(new ofstream(file, ofstream::app));
}

void Logger::write(const string& str) {
  lock_guard<std::mutex> lock(mutex);
  time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  if(logToStdout)
    cout << std::put_time(std::localtime(&time), "%F %T%z: ") << str << std::flush;
  if(logToStderr)
    cerr << std::put_time(std::localtime(&time), "%F %T%z: ") << str << std::flush;
  for(size_t i = 0; i<files.size(); i++)
    (*files[i]) << std::put_time(std::localtime(&time), "%F %T%z: ") << str << std::flush;
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
  logger->write(str);
  return 0;
}
