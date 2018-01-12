#include "core/global.h"
#include "core/rand.h"
#include "fastboard.h"
#include "sgf.h"
#include "datapool.h"

#include <H5Cpp.h>
using namespace H5;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

//Data and feature row parameters
static const int maxBoardSize = 19;
static const int numFeatures = 23;
static const int inputLen = maxBoardSize * maxBoardSize * numFeatures;
static const int chainLen = maxBoardSize * maxBoardSize + 1;
static const int targetLen = maxBoardSize * maxBoardSize;
static const int targetWeightsLen = 1;
static const int totalRowLen = inputLen + chainLen + targetLen + targetWeightsLen;

//HDF5 parameters
static const int chunkHeight = 2000;
static const int deflateLevel = 6;
static const int h5Dimension = 2;

static int xyToTensorPos(int x, int y, int offset) {
  return (y+offset) * maxBoardSize + (x+offset);
}
static int locToTensorPos(Loc loc, int bSize, int offset) {
  return (Location::getY(loc,bSize) + offset) * maxBoardSize + (Location::getX(loc,bSize) + offset);
}

static void setRow(float* row, int pos, int feature, float value) {
  row[pos*numFeatures + feature] = value;
}

static const int TARGET_NEXT_MOVE = 0;
static const int TARGET_LADDER_CAPTURE = 1;

static void fillRow(const FastBoard& board, const vector<Move>& moves, int nextMoveIdx, int target, float* row, Rand& rand) {
  assert(board.x_size == board.y_size);
  assert(nextMoveIdx < moves.size());

  Player pla = moves[nextMoveIdx].pla;
  Player opp = getEnemy(pla);
  int bSize = board.x_size;
  int offset = (maxBoardSize - bSize) / 2;

  int nextChainLabel = 1;
  int chainLabelsByHeadLoc[FastBoard::MAX_ARR_SIZE];
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    chainLabelsByHeadLoc[i] = 0;

  for(int y = 0; y<bSize; y++) {
    for(int x = 0; x<bSize; x++) {
      int pos = xyToTensorPos(x,y,offset);
      Loc loc = Location::getLoc(x,y,bSize);

      //Feature 0 - on board
      setRow(row,pos,0, 1.0);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5,6 and 7,8,9,10 - pla 1,2,3,4 libs and opp 1,2,3,4 libs.
      if(stone == pla) {
        setRow(row,pos,1, 1.0);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRow(row,pos,3, 1.0);
        else if(libs == 2) setRow(row,pos,4, 1.0);
        else if(libs == 3) setRow(row,pos,5, 1.0);
        else if(libs == 4) setRow(row,pos,6, 1.0);
      }
      else if(stone == opp) {
        setRow(row,pos,2, 1.0);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRow(row,pos,7, 1.0);
        else if(libs == 2) setRow(row,pos,8, 1.0);
        else if(libs == 3) setRow(row,pos,9, 1.0);
        else if(libs == 4) setRow(row,pos,10, 1.0);
      }

      if(stone == pla || stone == opp) {
        //Fill chain feature
        Loc headLoc = board.chain_head[loc];
        if(chainLabelsByHeadLoc[headLoc] == 0)
          chainLabelsByHeadLoc[headLoc] = (nextChainLabel++);

        row[inputLen + pos] = chainLabelsByHeadLoc[headLoc];
      }
      else {
        //Feature 11,12,13 - 1, 2, 3 liberties after own play.
        //Feature 14,15,16 - 1, 2, 3 liberties after opponent play
        int plaLibAfterPlay = board.getNumLibertiesAfterPlay(loc,pla,4);
        int oppLibAfterPlay = board.getNumLibertiesAfterPlay(loc,opp,4);
        if(plaLibAfterPlay == 1)      setRow(row,pos,11, 1.0);
        else if(plaLibAfterPlay == 2) setRow(row,pos,12, 1.0);
        else if(plaLibAfterPlay == 3) setRow(row,pos,13, 1.0);

        if(oppLibAfterPlay == 1)      setRow(row,pos,14, 1.0);
        else if(oppLibAfterPlay == 2) setRow(row,pos,15, 1.0);
        else if(oppLibAfterPlay == 3) setRow(row,pos,16, 1.0);
      }
    }
  }

  //Last chain entry is the number of chain segments
  row[inputLen + maxBoardSize * maxBoardSize] = nextChainLabel;

  //Feature 17 - simple ko location
  if(board.ko_loc != FastBoard::NULL_LOC) {
    int pos = locToTensorPos(board.ko_loc,bSize,offset);
    setRow(row,pos,17, 1.0);
  }

  //Probabilistically include prev move features
  //Features 18,19,20,21,22
  bool includePrev1 = rand.nextDouble() < 0.9;
  bool includePrev2 = includePrev1 && rand.nextDouble() < 0.95;
  bool includePrev3 = includePrev2 && rand.nextDouble() < 0.95;
  bool includePrev4 = includePrev3 && rand.nextDouble() < 0.98;
  bool includePrev5 = includePrev4 && rand.nextDouble() < 0.98;

  if(nextMoveIdx >= 1 && moves[nextMoveIdx-1].pla == opp && includePrev1) {
    Loc prev1Loc = moves[nextMoveIdx-1].loc;
    if(prev1Loc != FastBoard::PASS_LOC) {
      int pos = locToTensorPos(prev1Loc,bSize,offset);
      setRow(row,pos,18, 1.0);
    }
    if(nextMoveIdx >= 2 && moves[nextMoveIdx-2].pla == pla && includePrev2) {
      Loc prev2Loc = moves[nextMoveIdx-2].loc;
      if(prev2Loc != FastBoard::PASS_LOC) {
        int pos = locToTensorPos(prev2Loc,bSize,offset);
        setRow(row,pos,19, 1.0);
      }
      if(nextMoveIdx >= 3 && moves[nextMoveIdx-3].pla == opp && includePrev3) {
        Loc prev3Loc = moves[nextMoveIdx-3].loc;
        if(prev3Loc != FastBoard::PASS_LOC) {
          int pos = locToTensorPos(prev3Loc,bSize,offset);
          setRow(row,pos,20, 1.0);
        }
        if(nextMoveIdx >= 4 && moves[nextMoveIdx-4].pla == pla && includePrev4) {
          Loc prev4Loc = moves[nextMoveIdx-4].loc;
          if(prev4Loc != FastBoard::PASS_LOC) {
            int pos = locToTensorPos(prev4Loc,bSize,offset);
            setRow(row,pos,21, 1.0);
          }
          if(nextMoveIdx >= 5 && moves[nextMoveIdx-5].pla == opp && includePrev5) {
            Loc prev5Loc = moves[nextMoveIdx-5].loc;
            if(prev5Loc != FastBoard::PASS_LOC) {
              int pos = locToTensorPos(prev5Loc,bSize,offset);
              setRow(row,pos,22, 1.0);
            }
          }
        }
      }
    }
  }

  //Target
  if(target == TARGET_NEXT_MOVE) {
    Loc nextMoveLoc = moves[nextMoveIdx].loc;
    assert(nextMoveLoc != FastBoard::PASS_LOC);
    int nextMovePos = locToTensorPos(nextMoveLoc,bSize,offset);
    row[inputLen + chainLen + nextMovePos] = 1.0;
  }
  else if(target == TARGET_LADDER_CAPTURE) {
    Loc chainHeadsSolved[bSize*bSize];
    float chainHeadsSolvedValue[bSize*bSize];
    int numChainHeadsSolved = 0;
    FastBoard copy(board);
    vector<Loc> buf;

    for(int y = 0; y<bSize; y++) {
      for(int x = 0; x<bSize; x++) {
        int pos = xyToTensorPos(x,y,offset);
        Loc loc = Location::getLoc(x,y,bSize);
        Color stone = board.colors[loc];
        if(stone == pla || stone == opp) {
          int libs = board.getNumLiberties(loc);
          if(libs == 1 || libs == 2) {
            bool alreadySolved = false;
            Loc head = board.chain_head[loc];
            for(int i = 0; i<numChainHeadsSolved; i++) {
              if(chainHeadsSolved[i] == head) {
                alreadySolved = true;
                row[inputLen + chainLen + pos] = chainHeadsSolvedValue[i];
                break;
              }
            }
            if(!alreadySolved) {
              float value = 0.0;
              if(libs == 1) {
                //Perform search on copy so as not to mess up tracking of solved heads
                bool laddered = copy.searchIsLadderCaptured(loc,true,buf);
                if(laddered)
                  value = 1.0;
              }
              else if(libs == 2) {
                //Perform search on copy so as not to mess up tracking of solved heads
                bool laddered = copy.searchIsLadderCaptured(loc,false,buf);
                if(laddered)
                  value = 1.0;
              }

              chainHeadsSolved[numChainHeadsSolved] = head;
              chainHeadsSolvedValue[numChainHeadsSolved] = value;
              numChainHeadsSolved++;
              row[inputLen + chainLen + pos] = value;
            }
          }
        }
      }
    }
  }

  //Weight of the row, currently always 1.0
  row[inputLen + chainLen + targetLen] = 1.0;
}

//Returns number of rows processed
static size_t processSgf(Sgf* sgf, vector<Move>& placementsBuf, vector<Move>& movesBuf, DataPool& dataPool, Rand& rand, int target, set<Hash>& posHashes) {
  int bSize;
  try {
    bSize = sgf->getBSize();

    //Apply some filters
    if(bSize != 19)
      return 0;

    sgf->getPlacements(placementsBuf,bSize);
    sgf->getMoves(movesBuf,bSize);
  }
  catch(const IOError &e) {
    cout << "Skipping sgf file: " << sgf->fileName << ": " << e.message << endl;
  }

  FastBoard board(bSize);
  for(int j = 0; j<placementsBuf.size(); j++) {
    Move m = placementsBuf[j];
    bool suc = board.setStone(m.loc,m.pla);
    if(!suc) {
      cout << sgf->fileName << endl;
      cout << ("Illegal stone placement " + Global::intToString(j)) << endl;
      cout << board << endl;
      return 0;
    }
  }

  //If there are multiple black moves in a row, then make them all right now.
  //Sometimes sgfs break the standard and do handicap setup in this way.
  int j = 0;
  if(movesBuf.size() > 1 && movesBuf[0].pla == P_BLACK && movesBuf[1].pla == P_BLACK) {
    for(; j<movesBuf.size(); j++) {
      Move m = movesBuf[j];
      if(m.pla != P_BLACK)
        break;
      bool suc = board.playMove(m.loc,m.pla);
      if(!suc) {
        cout << sgf->fileName << endl;
        cout << ("Illegal move! " + Global::intToString(j)) << endl;
        cout << board << endl;
      }
    }
  }

  size_t numRowsProcessed = 0;
  Player prevPla = C_EMPTY;
  for(; j<movesBuf.size(); j++) {
    Move m = movesBuf[j];

    //Forbid consecutive moves by the same player
    if(m.pla == prevPla) {
      cout << sgf->fileName << endl;
      cout << ("Multiple moves in a row by same player at " + Global::intToString(j)) << endl;
      cout << board << endl;
      break;
    }

    //For now, only generate training rows for non-passes
    if(m.loc != FastBoard::PASS_LOC) {
      float* newRow = dataPool.addNewRow(rand);
      fillRow(board,movesBuf,j,target,newRow,rand);
      posHashes.insert(board.pos_hash);
      numRowsProcessed++;
    }

    bool suc = board.playMove(m.loc,m.pla);
    if(!suc) {
      cout << sgf->fileName << endl;
      cout << ("Illegal move! " + Global::intToString(j)) << endl;
      cout << board << endl;
      break;
    }

    prevPla = m.pla;
  }

  return numRowsProcessed;
}



int main(int argc, const char* argv[]) {
  assert(sizeof(size_t) == 8);
  FastBoard::initHash();

  cout << "Command: ";
  for(int i = 0; i<argc; i++)
    cout << argv[i] << " ";
  cout << endl;

  vector<string> gamesDirs;
  string outputFile;
  int trainPoolSize;
  int testSize;
  int target;

  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0");
    TCLAP::MultiArg<string> gamesdirArg("","gamesdir","Directory of sgf files",true,"DIR");
    TCLAP::ValueArg<string> outputArg("","output","H5 file to write",true,string(),"FILE");
    TCLAP::ValueArg<size_t> trainPoolSizeArg("","train-pool-size","Pool size for shuffling training rows",true,(size_t)0,"SIZE");
    TCLAP::ValueArg<size_t> testSizeArg("","test-size","Number of testing rows",true,(size_t)0,"SIZE");
    TCLAP::ValueArg<string> targetArg("","target","nextmove or ladder",true,string(),"TARGET");
    cmd.add(gamesdirArg);
    cmd.add(outputArg);
    cmd.add(trainPoolSizeArg);
    cmd.add(testSizeArg);
    cmd.add(targetArg);
    cmd.parse(argc,argv);
    gamesDirs = gamesdirArg.getValue();
    outputFile = outputArg.getValue();
    trainPoolSize = trainPoolSizeArg.getValue();
    testSize = testSizeArg.getValue();

    if(targetArg.getValue() == "nextmove")
      target = TARGET_NEXT_MOVE;
    else if(targetArg.getValue() == "ladder")
      target = TARGET_LADDER_CAPTURE;
    else
      throw IOError("Must specify target nextmove or ladder");
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    return 1;
  }

  //Print some stats-----------------------------------------------------------------
  cout << "maxBoardSize " << maxBoardSize << endl;
  cout << "numFeatures " << numFeatures << endl;
  cout << "inputLen " << inputLen << endl;
  cout << "chainLen " << chainLen << endl;
  cout << "targetLen " << targetLen << endl;
  cout << "targetWeightsLen " << targetWeightsLen << endl;
  cout << "totalRowLen " << totalRowLen << endl;
  cout << "chunkHeight " << chunkHeight << endl;
  cout << "deflateLevel " << deflateLevel << endl;
  cout << "target " << target << endl;

  //Collect SGF files-----------------------------------------------------------------
  const string suffix = ".sgf";
  auto filter = [&suffix](const string& name) {
    return Global::isSuffix(name,suffix);
  };

  vector<string> files;
  for(int i = 0; i<gamesDirs.size(); i++)
    Global::collectFiles(gamesDirs[i], filter, files);
  cout << "Found " << files.size() << " sgf files!" << endl;

  cout << "Opening h5 file..." << endl;
  H5File* h5File = new H5File(H5std_string(outputFile), H5F_ACC_TRUNC);
  hsize_t maxDims[h5Dimension] = {H5S_UNLIMITED, totalRowLen};
  hsize_t chunkDims[h5Dimension] = {chunkHeight, totalRowLen};
  hsize_t initFileDims[h5Dimension] = {0, totalRowLen};

  DSetCreatPropList dataSetProps;
  dataSetProps.setChunk(h5Dimension,chunkDims);
  dataSetProps.setDeflate(deflateLevel);

  H5std_string trainSetName("train");
  H5std_string testSetName("test");
  DataSet* trainDataSet = new DataSet(h5File->createDataSet(trainSetName, PredType::IEEE_F32LE, DataSpace(h5Dimension,initFileDims,maxDims), dataSetProps));

  size_t curTrainDataSetRow = 0;
  std::function<void(const float*,size_t)> writeTrainRow = [&curTrainDataSetRow,&trainDataSet](const float* rows, size_t numRows) {
    hsize_t newDims[h5Dimension] = {curTrainDataSetRow+numRows,totalRowLen};
    trainDataSet->extend(newDims);
    DataSpace fileSpace = trainDataSet->getSpace();
    hsize_t memDims[h5Dimension] = {numRows,totalRowLen};
    DataSpace memSpace(h5Dimension,memDims);
    hsize_t start[h5Dimension] = {curTrainDataSetRow,0};
    hsize_t count[h5Dimension] = {numRows,totalRowLen};
    fileSpace.selectHyperslab(H5S_SELECT_SET, count, start);
    trainDataSet->write(rows, PredType::NATIVE_FLOAT, memSpace, fileSpace);
    curTrainDataSetRow += numRows;
  };

  DataPool dataPool(totalRowLen,trainPoolSize,testSize,chunkHeight,writeTrainRow);

  //Process SGFS to make rows----------------------------------------------------------
  Rand rand;
  cout << "Loading SGFS..." << endl;
  vector<Sgf*> sgfs = Sgf::loadFiles(files);

  //Shuffle sgfs
  cout << "Shuffling SGFS..." << endl;
  for(int i = 1; i<sgfs.size(); i++) {
    int r = rand.nextUInt(i+1);
    Sgf* tmp = sgfs[i];
    sgfs[i] = sgfs[r];
    sgfs[r] = tmp;
  }

  cout << "Processing SGFS..." << endl;
  size_t numRowsProcessed = 0;
  vector<Move> placementsBuf;
  vector<Move> movesBuf;
  set<Hash> posHashes;
  for(int i = 0; i<sgfs.size(); i++) {
    if(i > 0 && i % 100 == 0)
      cout << "Processed " << i << " sgfs, " << numRowsProcessed << " rows, " << curTrainDataSetRow << " rows written..." << endl;

    Sgf* sgf = sgfs[i];
    numRowsProcessed += processSgf(sgf, placementsBuf, movesBuf, dataPool, rand, target, posHashes);
  }

  //Empty out pools--------------------------------------------------------------------
  cout << "Emptying training pool" << endl;
  dataPool.finishAndWriteTrainPool(rand);
  //Close the training dataset
  delete trainDataSet;

  //Open the testing dataset
  DataSet* testDataSet = new DataSet(h5File->createDataSet(testSetName, PredType::IEEE_F32LE, DataSpace(h5Dimension,initFileDims,maxDims), dataSetProps));

  size_t curTestDataSetRow = 0;
  std::function<void(const float*,size_t)> writeTestRow = [&curTestDataSetRow,&testDataSet](const float* rows, size_t numRows) {
    hsize_t newDims[h5Dimension] = {curTestDataSetRow+numRows,totalRowLen};
    testDataSet->extend(newDims);
    DataSpace fileSpace = testDataSet->getSpace();
    hsize_t memDims[h5Dimension] = {numRows,totalRowLen};
    DataSpace memSpace(h5Dimension,memDims);
    hsize_t start[h5Dimension] = {curTestDataSetRow,0};
    hsize_t count[h5Dimension] = {numRows,totalRowLen};
    fileSpace.selectHyperslab(H5S_SELECT_SET, count, start);
    testDataSet->write(rows, PredType::NATIVE_FLOAT, memSpace, fileSpace);
    curTestDataSetRow += numRows;
  };

  cout << "Writing testing set" << endl;
  dataPool.writeTestPool(writeTestRow, rand);

  //Close the testing dataset
  delete testDataSet;

  //Close the h5 file
  delete h5File;

  cout << "Done" << endl;
  cout << numRowsProcessed << " rows" << endl;
  cout << posHashes.size() << " unique pos hashes" << endl;

  //Cleanup----------------------------------------------------------------------------
  for(int i = 0; i<sgfs.size(); i++) {
    delete sgfs[i];
  }

  return 0;
}
