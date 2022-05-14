#ifndef CORE_LOGGER_H
#define CORE_LOGGER_H

#include <fstream>
#include <sstream>

#include "../core/global.h"
#include "../core/multithread.h"

class LogBuf;
class ConfigParser;

class Logger {
 public:
  // TODO: Logger(ConfigFile) for proper setup and values logging
  Logger(ConfigParser *cfg = nullptr, bool logToStdout = false, bool logToStderr = false, bool logTime = true);
  ~Logger();

  Logger(const Logger& other) = delete;
  Logger& operator=(const Logger& other) = delete;

  bool isLoggingToStdout() const;
  bool isLoggingToStderr() const;

  void setLogToStdout(bool b, bool afterCreation = true);
  void setLogToStderr(bool b, bool afterCreation = true);
  void setLogTime(bool b);
  void addOStream(std::ostream& out, bool afterCreation = true); //User responsible for cleaning up the ostream, logger does not take ownership
  void addFile(const std::string& file, bool afterCreation = true);

  //write and ostreams returned are synchronized with other calls to write and other ostream calls
  //The lifetime of the Logger must exceed the lifetimes of any of the ostreams created from it.
  //The caller is responsible for freeing the ostreams
  void write(const std::string& str);
  void writeNoEndline(const std::string& str);
  std::ostream* createOStream();

  static void logThreadUncaught(const std::string& name, Logger* logger, std::function<void()> f);

 private:
  bool logToStdout;
  bool logToStderr;
  bool logTime;
  std::vector<std::ostream*> ostreams;
  std::vector<std::ofstream*> files;
  std::vector<LogBuf*> logBufs;
  std::string logHeader;
  std::mutex mutex;

  void write(const std::string& str, bool endLine);
  void writeLocked(const std::string& str, bool endLine, std::ostream& out, const time_t& time);
};

class LogBuf final : public std::stringbuf {
 public:
  LogBuf(Logger* logger);
  ~LogBuf() final;

  LogBuf(const LogBuf& other) = delete;
  LogBuf& operator=(const LogBuf& other) = delete;

  virtual int sync();
 private:
  Logger* logger;
};

#endif  // CORE_LOGGER_H_
