#ifndef DATAIO_POSWRITER_H_
#define DATAIO_POSWRITER_H_

#include "../core/global.h"
#include "../core/threadsafequeue.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"

class PosWriter {
 public:
  PosWriter(
    const std::string& suffix,
    const std::string& outDir,
    int sgfSplitCount,
    int sgfSplitIdx,
    int maxPosesPerOutFile
  );
  ~PosWriter();

  void start();
  void flushAndStop();
  void writeLine(const std::string& line);
  void writePos(const Sgf::PositionSample& pos);

 private:
  std::string suffix;
  std::string outDir;
  int sgfSplitCount;
  int sgfSplitIdx;
  int maxPosesPerOutFile;
  ThreadSafeQueue<std::string*> toWriteQueue;
  std::thread* writeLoopThread;
};

#endif
