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

void Logger::write(const string& str, bool endLine) {
  lock_guard<std::mutex> lock(mutex);
  time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  if(logToStdout) {
    cout << std::put_time(std::localtime(&time), "%F %T%z: ") << str;
    if(endLine) cout << std::endl; else cout << std::flush;
  }
  if(logToStderr) {
    cerr << std::put_time(std::localtime(&time), "%F %T%z: ") << str;
    if(endLine) cerr << std::endl; else cerr << std::flush;
  }
  for(size_t i = 0; i<files.size(); i++) {
    (*files[i]) << std::put_time(std::localtime(&time), "%F %T%z: ") << str;
    if(endLine) (*files[i]) << std::endl; else (*files[i]) << std::flush;
  }
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
