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
  Logger(
    ConfigParser* cfg = nullptr,
    // The following are defaults that may be overriden by the user in the config.
    bool logToStdoutDefault = false,
    bool logToStderrDefault = false,
    bool logTimeDefault = true,
    //Log config contents on startup?
    bool logConfigContents = true
  );
  ~Logger();

  Logger(const Logger& other) = delete;
  Logger& operator=(const Logger& other) = delete;

  bool isLoggingToStdout() const;
  bool isLoggingToStderr() const;

  void addOStream(std::ostream& out, bool afterCreation = true); // User responsible for cleaning up the ostream, logger does not take ownership
  void addFile(const std::string& file, bool afterCreation = true);

  void setDisabled(bool b);

  // write and ostreams returned are synchronized with other calls to write and other ostream calls
  // The lifetime of the Logger must exceed the lifetimes of any of the ostreams created from it.
  // The caller is responsible for freeing the ostreams
  void write(const std::string& str);
  void writeNoEndline(const std::string& str);
  std::ostream* createOStream();

  static void logThreadUncaught(const std::string& name, Logger* logger, std::function<void()> f);

 private:
  //------------
  // Constant after initialization
  bool logToStdout;
  bool logToStderr;
  bool logTime;
  bool logConfigContents;
  std::string header;

  //------------

  std::vector<std::ostream*> ostreams;
  std::vector<std::ofstream*> files;
  std::vector<LogBuf*> logBufs;
  std::mutex mutex;

  bool isDisabled;

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
