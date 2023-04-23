#include "../dataio/poswriter.h"

#include "../core/fileutils.h"

//------------------------
#include "../core/using.h"
//------------------------

PosWriter::PosWriter(
  const std::string& s,
  const std::string& out,
  int sgfCount,
  int sgfIdx,
  int maxPerFile
) :
  suffix(s),
  outDir(out),
  sgfSplitCount(sgfCount),
  sgfSplitIdx(sgfIdx),
  maxPosesPerOutFile(maxPerFile),
  toWriteQueue(),
  writeLoopThread(NULL)
{}

PosWriter::~PosWriter() {
  flushAndStop();
}

void PosWriter::flushAndStop() {
  toWriteQueue.setReadOnly();
  if(writeLoopThread != NULL) {
    writeLoopThread->join();
  }
  delete writeLoopThread;
  writeLoopThread = NULL;
}

static void writeLoop(
  const std::string& suffix,
  const std::string& outDir,
  int sgfSplitCount,
  int sgfSplitIdx,
  int maxPosesPerOutFile,
  ThreadSafeQueue<string*>* toWriteQueue
) {
  int fileCounter = 0;
  int numWrittenThisFile = 0;
  ofstream* out = NULL;
  while(true) {
    string* message;
    bool suc = toWriteQueue->waitPop(message);
    if(!suc)
      break;

    if(out == NULL || numWrittenThisFile > maxPosesPerOutFile) {
      if(out != NULL) {
        out->close();
        delete out;
      }

      string fileNameToWrite;
      if(sgfSplitCount > 1)
        fileNameToWrite = outDir + "/" + Global::intToString(fileCounter) + "." + Global::intToString(sgfSplitIdx) + "." + suffix;
      else
        fileNameToWrite = outDir + "/" + Global::intToString(fileCounter) + "." + suffix;

      out = new ofstream();
      FileUtils::open(*out,fileNameToWrite);
      fileCounter += 1;
      numWrittenThisFile = 0;
    }
    (*out) << *message << endl;
    numWrittenThisFile += 1;
    delete message;
  }

  if(out != NULL) {
    out->close();
    delete out;
  }
}

void PosWriter::start() {
  assert(!toWriteQueue.isReadOnly());
  assert(writeLoopThread == NULL);
  writeLoopThread = new std::thread(writeLoop, suffix, outDir, sgfSplitCount, sgfSplitIdx, maxPosesPerOutFile, &toWriteQueue);
}

void PosWriter::writeLine(const std::string& line) {
  toWriteQueue.waitPush(new string(line));
}

void PosWriter::writePos(const Sgf::PositionSample& pos) {
  toWriteQueue.waitPush(new string(Sgf::PositionSample::toJsonLine(pos)));
}
