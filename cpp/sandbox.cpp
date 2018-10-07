
#include "core/global.h"
#include "core/timer.h"
#include "core/logger.h"
#include "game/board.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nneval.h"
#include "search/searchparams.h"
#include "search/search.h"
#include "search/asyncbot.h"
#include "main.h"

int MainCmds::sandbox() {
  Board::initHash();

  Logger logger;
  logger.setLogToStdout(true);
  logger.addFile("tmp.txt");

  string tensorflowGpuVisibleDeviceList = ""; //use default
  double tensorflowPerProcessGpuMemoryFraction = -1; //use default
  NeuralNet::globalInitialize(tensorflowGpuVisibleDeviceList,tensorflowPerProcessGpuMemoryFraction);

  int modelFileIdx = 0;
  int maxBatchSize = 1;
  int posLen = 19;
  int nnCacheSizePowerOfTwo = 16;
  bool debugSkipNeuralNet = false;
  NNEvaluator* nnEval = new NNEvaluator(
    // "/efs/data/GoNN/exportedmodels/tensorflow/value24-140/model.graph_optimized.pb",
    "/efs/data/GoNN/exportedmodels/cuda/value33-140/model.txt",
    // "/efs/data/GoNN/exportedmodels/cuda/value24-140/model.txt",
    modelFileIdx,
    maxBatchSize,
    posLen,
    nnCacheSizePowerOfTwo,
    debugSkipNeuralNet
  );

  int numNNServerThreads = 1;
  bool doRandomize = true;
  string randSeed = "abc";
  int defaultSymmetry = 0;
  vector<int> cudaGpuIdxByServerThread = {0};
  bool cudaUseFP16 = false;
  bool cudaUseNHWC = false;
  nnEval->spawnServerThreads(
    numNNServerThreads,doRandomize,randSeed,defaultSymmetry,logger,cudaGpuIdxByServerThread,cudaUseFP16,cudaUseNHWC
  );

  Rules rules;
  rules.koRule = Rules::KO_POSITIONAL;
  rules.scoringRule = Rules::SCORING_AREA;
  rules.multiStoneSuicideLegal = true;
  rules.komi = 7.5f;

  Player pla = P_BLACK;
//   Board board = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . x . .
// 18 . . x o . . . . . . x o . . o . o x .
// 17 . . x o . . o x . . . . o . . o x . .
// 16 . . x o . . o x x o . x . . . o x . .
// 15 . x o o x . x . x x x . x . . o x . .
// 14 . x o . . . x x o o o o x . x o o x .
// 13 . x o . . . . . o x x x x . . . o x .
// 12 . . o . . x x x . o . o o o o . o . .
// 11 . . . . o x o o o o . o . x . o . . .
// 10 . o o o o o x . . o x x x . o x x . .
//  9 . x . x o o x x x x o o x . x o x . .
//  8 . . . x x x x . . x . o o x . o x . .
//  7 . . . o o . x x . x . . . . . x . x .
//  6 . . o x x x . x x o o . o . . x . x .
//  5 . . o o o o x x . . . o . o . o x . .
//  4 . o o x x o o . x o o x . o . o x . .
//  3 . o x x . o o x x x . x . o x x o x .
//  2 o x . x x o . o . . . . . o . . o x .
//  1 . o x x o . . o . . . . . . . . . . .
// )");

  Board board = Board::parseBoard(19,19,R"(
   A B C D E F G H J K L M N O P Q R S T
19 . o . . o . . . . . . . . . . . . . .
18 o x x x x o o . . . . . . . . . . . .
17 . o o o x x o . . x . . . . . x . . .
16 . . . o o . o . . . . . . . . . . . .
15 . . x . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . .
13 . . x . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . x . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . x o .
 8 . . . . . . . . . . . . . . . x . o .
 7 . . . . . . . . . . . . o x x o o o .
 6 . . . . . . . . . . . x o o o x x x .
 5 . . . . . . . . . . . . x x x o o x .
 4 . . . o . . . . . . . . o . . o o x .
 3 . . . . . . . . . . . . . x x x o x .
 2 . . . . . . . . . . . x . . . . o o .
 1 . . . . . . . . . . . . . . . . . . .
)");


  BoardHistory hist(board,pla,rules);

  ostream* logStream = logger.createOStream();
  NNResultBuf buf;
  nnEval->evaluate(board, hist, pla, buf, logStream, false);

  for(int y = 0; y<NNPos::MAX_BOARD_LEN; y++) {
    for(int x = 0; x<NNPos::MAX_BOARD_LEN; x++) {
      if(buf.result->policyProbs[x+y*NNPos::MAX_BOARD_LEN] >= 0)
        printf("%6.2f%%", buf.result->policyProbs[x+y*NNPos::MAX_BOARD_LEN] * 100.0);
      else
        printf("   .   ");
    }
    cout << endl;
  }
  printf("%4.1f%%", buf.result->policyProbs[NNPos::NN_POLICY_SIZE-1] * 100.0);
  cout << endl;
  cout << buf.result->whiteValue << endl;

  delete logStream;

  SearchParams params;
  params.maxPlayouts = 180;
  params.numThreads = 1;
  params.fpuUseParentAverage = false;
  // params.moveProbModelExponent = 0.0;
  // params.moveProbModelPolicyExponent = 0.0;

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "def");
  bot->setPosition(pla,board,hist);

  cout << bot->getRootBoard() << endl;
  bot->genMoveSynchronous(pla);
  bot->getSearch()->printTree(cout, bot->getSearch()->rootNode, PrintTreeOptions().maxDepth(1));
  cout << "NN rows: " << nnEval->numRowsProcessed() << endl;
  cout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
  cout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;
  bot->clearSearch();
  nnEval->clearCache();

  bot->makeMove(Location::ofString("F15",board),pla);
  cout << bot->getRootBoard() << endl;
  bot->genMoveSynchronous(getOpp(pla));
  bot->getSearch()->printTree(cout, bot->getSearch()->rootNode, PrintTreeOptions().maxDepth(1));

  cout << "NN rows: " << nnEval->numRowsProcessed() << endl;
  cout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
  cout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;

  // cout << "sizeof(uint8_t) " << sizeof(uint8_t) << endl;
  // cout << "sizeof(uint16_t) " << sizeof(uint16_t) << endl;
  // cout << "sizeof(uint32_t) " << sizeof(uint32_t) << endl;
  // cout << "sizeof(uint64_t) " << sizeof(uint64_t) << endl;
  // cout << "sizeof(std::atomic_flag) " << sizeof(std::atomic_flag) << endl;;
  // cout << "sizeof(std::mutex) " << sizeof(std::mutex) << endl;;
  // cout << "sizeof(std::shared_ptr<NNOutput>) " << sizeof(std::shared_ptr<NNOutput>) << endl;;

  // {
  //   atomic<bool>* b = new atomic<bool>(false);
  //   cout << "atomic<bool> lock free " << std::atomic_is_lock_free(b) << endl;
  //   delete b;
  // }
  // {
  //   atomic<uint64_t>* b = new atomic<uint64_t>(0);
  //   cout << "atomic<uint64_t> lock free " << std::atomic_is_lock_free(b) << endl;
  //   delete b;
  // }

  nnEval->killServerThreads();
  delete bot;
  delete nnEval;

  cout << "Done" << endl;
  return 0;
}






// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);
//   logger.addFile("tmp.txt");

//   string tensorflowGpuVisibleDeviceList = ""; //use default
//   double tensorflowPerProcessGpuMemoryFraction = -1; //use default
//   NeuralNet::globalInitialize(tensorflowGpuVisibleDeviceList,tensorflowPerProcessGpuMemoryFraction);

//   LoadedModel* loadedModel = NeuralNet::loadModelFile("/efs/data/GoNN/exportedmodels/cuda/value33-140/model.txt", 0);
//   // LoadedModel* loadedModel = NeuralNet::loadModelFile("/efs/data/GoNN/exportedmodels/cuda/value24-140/model.txt", 0);
//   // LoadedModel* loadedModel = NeuralNet::loadModelFile("/efs/data/GoNN/exportedmodels/tensorflow/value24-140/model.graph_optimized.pb", 0);
//   bool cudaUseFP16 = true;
//   bool cudaUseNHWC = true;
//   int maxBatchSize = 256;
//   LocalGpuHandle* gpuHandle = NeuralNet::createLocalGpuHandle(loadedModel,&logger,maxBatchSize,0,cudaUseFP16,cudaUseNHWC);
//   InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel,maxBatchSize);

//   bool* syms = NeuralNet::getSymmetriesInplace(inputBuffers);
//   syms[0] = false;
//   syms[1] = false;
//   syms[2] = false;

//   Rules rules;
//   rules.koRule = Rules::KO_POSITIONAL;
//   rules.scoringRule = Rules::SCORING_AREA;
//   rules.multiStoneSuicideLegal = true;
//   rules.komi = 7.5f;

//   Player pla = P_WHITE;
//   Board board = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . x . .
// 18 . . x o . . . . . . x o . . o . o x .
// 17 . . x o . . o x . . . . o . . o x . .
// 16 . . x o . . o x x o . x . . . o x . .
// 15 . x o o x . x . x x x . x . . o x . .
// 14 . x o . . . x x o o o o x . x o o x .
// 13 . x o . . . . . o x x x x . . . o x .
// 12 . . o . . x x x . o . o o o o . o . .
// 11 . . . . o x o o o o . o . x . o . . .
// 10 . o o o o o x . . o x x x . o x x . .
//  9 . x . x o o x x x x o o x . x o x . .
//  8 . . . x x x x . . x . o o x . o x . .
//  7 . . . o o . x x . x . . . . . x . x .
//  6 . . o x x x . x x o o . o . . x . x .
//  5 . . o o o o x x . . . o . o . o x . .
//  4 . o o x x o o . x o o x . o . o x . .
//  3 . o x x . o o x x x . x . o x x o x .
//  2 o x . x x o . o . . . . . o . . o x .
//  1 . o x x o . . o . . . . . . . . . . .
// )");

//   Board board2 = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . . . .
// 18 . . . . . . . . . . . . . . . . . . .
// 17 . . . x . . . . . . . . . . . . . . .
// 16 . . . . . . . . . . . . . . . . . . .
// 15 . . . . . . . . . . . . . . . . . . .
// 14 . . . . . . . . . . . . . . . . . . .
// 13 . . . . . . . . . . . . . . . . . . .
// 12 . . . . . . . . . . . . . . . . . . .
// 11 . . . . . . . . . . . . . . . . . . .
// 10 . . . . . . . . . . . . . . . . . . .
//  9 . . . . . . . . . . . . . . . . . . .
//  8 . . . . . . . . . . . . . . . . . . .
//  7 . . . . . . . . . . . . . . . . . . .
//  6 . . . . . . . . . . . . . . . . . . .
//  5 . . . . . . . . . . . . . . . . . . .
//  4 . . . . . . . . . . . . . . . . . . .
//  3 . . . . . . . . . . . . . . . . . . .
//  2 . . . . . . . . . . . . . . . . . . .
//  1 . . . . . . . . . . . . . . . . . . .
// )");


//   Board board3 = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . x . . x . . x . . . . . . . . . . .
// 18 . x . . . . . x . . . . . x . . x . .
// 17 . . . x . . . . x . . x . . x . . x .
// 16 . x . . x . . . . . x . . x . . x . .
// 15 . x . . x . . . . . x . . x . . . . .
// 14 . x . . x . . . . . . . . x . . x . .
// 13 . . . . x . . . . x . . x o . . . x o
// 12 x . . x o x . . x o x . . x o . . . .
// 11 x . . x o x . . x o x . . x o . . . .
// 10 x . . x o x . . x o x . . x o . . . .
//  9 x . . x o x . . x o x . . x o . . . .
//  8 x . . x o x . . x o x . . x o . . . .
//  7 x . . x o . . . x o . . . x o x x . .
//  6 x x . . x x . . x x . . x x . . . . .
//  5 x x . . x x . . x x . . x x . x . . .
//  4 x x . . x x . . x x . . x x . . . . .
//  3 x x . . x x . . x x . . x x . . . . .
//  2 x x . . x x . . x x . . x x . . . . .
//  1 . . . . . . . . . . . . . . . . . . .
// )");


//   BoardHistory hist(board,pla,rules);
//   BoardHistory hist2(board2,pla,rules);
//   BoardHistory hist3(board3,pla,rules);

//   //int batchSize = 5;
//   // int batchSize = maxBatchSize;
//   int batchSize = 32;
//   for(int i = 0; i<batchSize; i++) {
//     float* row = NeuralNet::getRowInplace(inputBuffers,i);
//     if(i % 3 == 0)
//       NNInputs::fillRowV1(board, hist, pla, row);
//     else if(i % 3 == 1)
//       NNInputs::fillRowV1(board2, hist2, pla, row);
//     else
//       NNInputs::fillRowV1(board3, hist3, pla, row);
//   }

//   vector<NNOutput*> outputs;
//   NeuralNet::getOutput(gpuHandle,inputBuffers,batchSize,outputs);

//   for(int i = 0; i<outputs.size(); i++) {
//     NNOutput* result = outputs[i];
//     for(int y = 0; y<NNPos::MAX_BOARD_LEN; y++) {
//       for(int x = 0; x<NNPos::MAX_BOARD_LEN; x++) {
//         printf("%7.4f ", result->policyProbs[x+y*NNPos::MAX_BOARD_LEN]);
//       }
//       cout << endl;
//     }
//     printf("%6.4f ", result->policyProbs[NNPos::NN_POLICY_SIZE-1]);
//     cout << endl;
//     cout << result->whiteValue << endl;
//   }

//   for(int i = 0; i<outputs.size(); i++)
//     delete outputs[i];

//   NeuralNet::freeInputBuffers(inputBuffers);
//   NeuralNet::freeLocalGpuHandle(gpuHandle);
//   NeuralNet::freeLoadedModel(loadedModel);

//   cout << "Done" << endl;
//   return 0;
// }




// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cudnn.h>
// #include "neuralnet/cudahelpers.h"
// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   cudaSetDevice(0);

//   int batchSize = 3;
//   int ySize = 13;
//   int xSize = 19;
//   int cSize = 5;

//   float inputArr[batchSize][cSize][ySize][xSize];
//   float outputArr[batchSize][cSize][ySize][xSize];

//   size_t inputBytes = sizeof(inputArr);
//   float* inputBuf = NULL;
//   cudaMalloc(&inputBuf, inputBytes);
//   size_t outputBytes = sizeof(outputArr);
//   float* outputBuf = NULL;
//   cudaMalloc(&outputBuf, outputBytes);

//   for(int b = 0; b<batchSize; b++) {
//     int ctr = 0;
//     for(int c = 0; c<cSize; c++) {
//       for(int y = 0; y<ySize; y++) {
//         for(int x = 0; x<xSize; x++) {
//           inputArr[b][c][y][x] = ctr++;
//         }
//       }
//     }
//   }

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   cudaMemcpy(inputBuf, inputArr, inputBytes, cudaMemcpyHostToDevice);

//   int reps = 100;

//   //Warmup
//   customCudaMirrorNCHW(inputBuf,outputBuf,batchSize,cSize,ySize,xSize,false,true);

//   cudaThreadSynchronize();
//   cudaEventRecord(start);
//   for(int i = 0; i<reps; i++)
//     customCudaMirrorNCHW(inputBuf,outputBuf,batchSize,cSize,ySize,xSize,false,true);
//   cudaThreadSynchronize();
//   cudaEventRecord(stop);

//   cudaMemcpy(outputArr, outputBuf, outputBytes, cudaMemcpyDeviceToHost);

//   cudaEventSynchronize(stop);
//   float timems;
//   cudaEventElapsedTime(&timems, start, stop);

//   for(int b = 0; b<batchSize; b++) {
//     for(int c = 0; c<cSize; c++) {
//       for(int y = 0; y<ySize; y++) {
//         for(int x = 0; x<xSize; x++) {
//           cout << outputArr[b][c][y][x] << " ";
//         }
//         cout << endl;
//       }
//       cout << endl;
//     }
//     cout << endl;
//   }
//   cout << "cuda time ms " << timems / reps << endl;

//   cudaFree(inputBuf);
//   cudaFree(outputBuf);

//   cout << "Done" << endl;
//   return 0;
// }




// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cudnn.h>
// #include "neuralnet/cudahelpers.h"
// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   cudaSetDevice(0);

//   int batchSize = 3;
//   int ySize = 13;
//   int xSize = 19;
//   int cSize = 5;

//   float inputArr[batchSize][ySize][xSize][cSize];
//   float outputArr[batchSize][ySize][xSize][cSize];

//   size_t inputBytes = sizeof(inputArr);
//   float* inputBuf = NULL;
//   cudaMalloc(&inputBuf, inputBytes);
//   size_t outputBytes = sizeof(outputArr);
//   float* outputBuf = NULL;
//   cudaMalloc(&outputBuf, outputBytes);

//   for(int b = 0; b<batchSize; b++) {
//     int ctr = 0;
//     for(int y = 0; y<ySize; y++) {
//       for(int x = 0; x<xSize; x++) {
//         for(int c = 0; c<cSize; c++) {
//           inputArr[b][y][x][c] = ctr++;
//         }
//       }
//     }
//   }

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   cudaMemcpy(inputBuf, inputArr, inputBytes, cudaMemcpyHostToDevice);

//   int reps = 100;

//   //Warmup
//   customCudaMirrorNHWC(inputBuf,outputBuf,batchSize,ySize,xSize,cSize,false,false);

//   cudaThreadSynchronize();
//   cudaEventRecord(start);
//   for(int i = 0; i<reps; i++)
//     customCudaMirrorNHWC(inputBuf,outputBuf,batchSize,ySize,xSize,cSize,false,false);
//   cudaThreadSynchronize();
//   cudaEventRecord(stop);

//   cudaMemcpy(outputArr, outputBuf, outputBytes, cudaMemcpyDeviceToHost);

//   cudaEventSynchronize(stop);
//   float timems;
//   cudaEventElapsedTime(&timems, start, stop);

//   for(int b = 0; b<batchSize; b++) {
//     for(int y = 0; y<ySize; y++) {
//       for(int x = 0; x<xSize; x++) {
//         cout << "(";
//         for(int c = 0; c<cSize; c++) {
//           cout << outputArr[b][y][x][c] << " ";
//         }
//         cout << ") ";
//       }
//       cout << endl;
//     }
//     cout << endl;
//   }
//   cout << "cuda time ms " << timems / reps << endl;

//   cudaFree(inputBuf);
//   cudaFree(outputBuf);

//   cout << "Done" << endl;
//   return 0;
// }






// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cudnn.h>
// #include "neuralnet/cudahelpers.h"
// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   cudaSetDevice(0);

//   int batchSize = 3;
//   int ySize = 13;
//   int xSize = 19;
//   int cSize = 5;

//   float inputArr[batchSize][ySize][xSize][cSize];
//   float outputArr[batchSize][xSize][ySize][cSize];

//   size_t inputBytes = sizeof(inputArr);
//   float* inputBuf = NULL;
//   cudaMalloc(&inputBuf, inputBytes);
//   size_t outputBytes = sizeof(outputArr);
//   float* outputBuf = NULL;
//   cudaMalloc(&outputBuf, outputBytes);

//   for(int b = 0; b<batchSize; b++) {
//     int ctr = 0;
//     for(int y = 0; y<ySize; y++) {
//       for(int x = 0; x<xSize; x++) {
//         for(int c = 0; c<cSize; c++) {
//           inputArr[b][y][x][c] = ctr++;
//         }
//       }
//     }
//   }

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   cudaMemcpy(inputBuf, inputArr, inputBytes, cudaMemcpyHostToDevice);

//   int reps = 100;

//   //Warmup
//   customCudaNHWCTranspose(inputBuf,outputBuf,xSize,ySize,cSize,batchSize);

//   cudaThreadSynchronize();
//   cudaEventRecord(start);
//   for(int i = 0; i<reps; i++)
//     customCudaNHWCTranspose(inputBuf,outputBuf,xSize,ySize,cSize,batchSize);
//   cudaThreadSynchronize();
//   cudaEventRecord(stop);

//   cudaMemcpy(outputArr, outputBuf, outputBytes, cudaMemcpyDeviceToHost);

//   cudaEventSynchronize(stop);
//   float timems;
//   cudaEventElapsedTime(&timems, start, stop);

//   for(int b = 0; b<batchSize; b++) {
//     for(int x = 0; x<xSize; x++) {
//       for(int y = 0; y<ySize; y++) {
//         cout << "(";
//         for(int c = 0; c<cSize; c++) {
//           cout << outputArr[b][x][y][c] << " ";
//         }
//         cout << ") ";
//       }
//       cout << endl;
//     }
//     cout << endl;
//   }
//   cout << "cuda time ms " << timems / reps << endl;

//   cudaFree(inputBuf);
//   cudaFree(outputBuf);

//   cout << "Done" << endl;
//   return 0;
// }





// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cudnn.h>
// #include "neuralnet/cudahelpers.h"
// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   cudaSetDevice(0);

//   int batchSize = 7;
//   int ySize = 45;
//   int xSize = 45;

//   float inputArr[batchSize][ySize][xSize];
//   float outputArr[batchSize][xSize][ySize];

//   size_t inputBytes = sizeof(inputArr);
//   float* inputBuf = NULL;
//   cudaMalloc(&inputBuf, inputBytes);
//   size_t outputBytes = sizeof(outputArr);
//   float* outputBuf = NULL;
//   cudaMalloc(&outputBuf, outputBytes);

//   for(int b = 0; b<batchSize; b++) {
//     int ctr = 0;
//     for(int y = 0; y<ySize; y++) {
//       for(int x = 0; x<xSize; x++) {
//         inputArr[b][y][x] = ctr++;
//       }
//     }
//   }

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   cudaMemcpy(inputBuf, inputArr, inputBytes, cudaMemcpyHostToDevice);

//   int reps = 100;

//   //Warmup
//   customCudaNCHWTranspose(inputBuf,outputBuf,xSize,ySize,batchSize);

//   cudaThreadSynchronize();
//   cudaEventRecord(start);
//   for(int i = 0; i<reps; i++)
//     customCudaNCHWTranspose(inputBuf,outputBuf,xSize,ySize,batchSize);
//   cudaThreadSynchronize();
//   cudaEventRecord(stop);

//   cudaMemcpy(outputArr, outputBuf, outputBytes, cudaMemcpyDeviceToHost);

//   cudaEventSynchronize(stop);
//   float timems;
//   cudaEventElapsedTime(&timems, start, stop);

//   for(int b = 0; b<batchSize; b++) {
//     for(int x = 0; x<xSize; x++) {
//       for(int y = 0; y<ySize; y++) {
//         cout << outputArr[b][x][y] << " ";
//       }
//       cout << endl;
//     }
//     cout << endl;
//   }
//   cout << "cuda time ms " << timems / reps << endl;

//   cudaFree(inputBuf);
//   cudaFree(outputBuf);

//   cout << "Done" << endl;
//   return 0;
// }











// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cudnn.h>
// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   cudaSetDevice(0);

//   cudnnStatus_t status;
//   cudnnHandle_t cudnn;
//   status = cudnnCreate(&cudnn);
//   assert(status == CUDNN_STATUS_SUCCESS);

//   int batchSize = 2;
//   int numChannels = 3;
//   int ySize = 3;
//   int xSize = 3;

//   cudnnTensorDescriptor_t inputDescriptor;
//   status = cudnnCreateTensorDescriptor(&inputDescriptor);
//   assert(status == CUDNN_STATUS_SUCCESS);
//   status = cudnnSetTensor4dDescriptor(
//     inputDescriptor,
//     CUDNN_TENSOR_NCHW,
//     CUDNN_DATA_FLOAT,
//     1,
//     batchSize*numChannels,
//     ySize,
//     xSize
//   );
//   assert(status == CUDNN_STATUS_SUCCESS);

//   cudnnTensorDescriptor_t biasDescriptor;
//   status = cudnnCreateTensorDescriptor(&biasDescriptor);
//   assert(status == CUDNN_STATUS_SUCCESS);
//   status = cudnnSetTensor4dDescriptor(
//     biasDescriptor,
//     CUDNN_TENSOR_NCHW,
//     CUDNN_DATA_FLOAT,
//     1,
//     batchSize*numChannels,
//     1,
//     1
//   );
//   assert(status == CUDNN_STATUS_SUCCESS);

//   float inputArr[batchSize][numChannels][ySize][xSize] = {
//     {
//       {
//         {0,1,2},
//         {3,4,5},
//         {6,7,8},
//       },
//       {
//         {0,1,2},
//         {3,4,5},
//         {6,7,8},
//       },
//       {
//         {0,1,2},
//         {3,4,5},
//         {6,7,8},
//       },
//     },
//     {
//       {
//         {0,0,0},
//         {1,1,1},
//         {2,2,2},
//       },
//       {
//         {3,4,5},
//         {3,4,5},
//         {3,4,5},
//       },
//       {
//         {6,7,8},
//         {8,6,7},
//         {7,8,6},
//       },
//     },
//   };

//   float biasArr[batchSize][numChannels] = {
//     {10,20,30},{40,50,60}
//   };

//   size_t inputBytes = sizeof(inputArr);
//   float* inputBuf = NULL;
//   cudaMalloc(&inputBuf, inputBytes);
//   cudaMemcpy(inputBuf, inputArr, inputBytes, cudaMemcpyHostToDevice);

//   size_t biasBytes = sizeof(biasArr);
//   float* biasBuf = NULL;
//   cudaMalloc(&biasBuf, biasBytes);
//   cudaMemcpy(biasBuf, biasArr, biasBytes, cudaMemcpyHostToDevice);

//   const float alpha = 1.0f;
//   const float beta = 1.0f;
//   status = cudnnAddTensor(cudnn,&alpha,biasDescriptor,biasBuf,&beta,inputDescriptor,inputBuf);
//   assert(status == CUDNN_STATUS_SUCCESS);

//   cudaMemcpy(inputArr, inputBuf, inputBytes, cudaMemcpyDeviceToHost);

//   for(int b = 0; b<batchSize; b++) {
//     for(int c = 0; c<numChannels; c++) {
//       for(int y = 0; y<ySize; y++) {
//         for(int x = 0; x<xSize; x++) {
//           cout << inputArr[b][c][y][x] << " ";
//         }
//         cout << endl;
//       }
//       cout << endl;
//     }
//     cout << endl;
//   }

//   cudaFree(inputBuf);
//   cudaFree(biasBuf);

//   cudnnDestroyTensorDescriptor(inputDescriptor);
//   cudnnDestroyTensorDescriptor(biasDescriptor);

//   cudnnDestroy(cudnn);

//   cout << "Done" << endl;
//   return 0;
// }








// #include <tensorflow/c/c_api.h>
// #include <tensorflow/cc/client/client_session.h>
// #include <tensorflow/cc/ops/standard_ops.h>
// #include <tensorflow/core/framework/tensor.h>
// #include <tensorflow/core/framework/tensor_shape.h>
// #include <tensorflow/core/platform/env.h>
// #include <tensorflow/core/public/session.h>
// #include <iostream>
// using namespace std;
// using namespace tensorflow;

// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cudnn.h>

// #include "neuralnet/cudaerrorcheck.h"
// #include "neuralnet/cudahelpers.h"

// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   CUDA_ERR(cudaSetDevice(0));

//   cublasHandle_t cublasHandle;
//   CUBLAS_ERR(cublasCreate(&cublasHandle));


//   int n = 2;
//   int ic = 4;
//   int oc = 3;

//   float* in;
//   float* mat;
//   float* out;
//   CUDA_ERR(cudaMalloc(&in, n*ic*sizeof(float)));
//   CUDA_ERR(cudaMalloc(&mat, ic*oc*sizeof(float)));
//   CUDA_ERR(cudaMallocManaged(&out, n*oc*sizeof(float)));

//   float invals[n][ic] = {
//     {0,1,2,3},
//     {5,6,2,1},
//   };

//   float matvals[ic][oc] = {
//     {1,-1,2},
//     {1,0,3},
//     {1,0,4},
//     {1,0,5},
//   };

//   CUDA_ERR(cudaMemcpy(in,invals,n*ic*sizeof(float),cudaMemcpyHostToDevice));
//   CUDA_ERR(cudaMemcpy(mat,matvals,ic*oc*sizeof(float),cudaMemcpyHostToDevice));

//   float alpha = 1.0;
//   float beta = 0.0;
//   ClockTimer timer;
//   CUBLAS_ERR(cublasSgemm(
//     cublasHandle,
//     CUBLAS_OP_T,
//     CUBLAS_OP_T,
//     n,oc,ic,
//     &alpha,
//     in,ic,
//     mat,oc,
//     &beta,
//     out,n
//   ));

//   cudaDeviceSynchronize();

//   double timeTaken = timer.getSeconds();
//   cout << "timeTaken " << timeTaken << endl;

//   // float result[n];
//   // cudaMemcpy(result, cc, n*sizeof(float), cudaMemcpyDeviceToHost);

//   for(int i = 0; i<n*oc; i++) {
//     cout << out[i] << " ";
//   }
//   // for(int i = 0; i<n; i++) {
//   //   cout << result[i] << " ";
//   // }
//   cout << "Yay" << endl;
//   cudaFree(in);
//   cudaFree(mat);
//   cudaFree(out);

//   cublasDestroy(cublasHandle);

//   return 0;
// }


// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   checkCudaErrors(cudaSetDevice(0));

//   cublasHandle_t cublasHandle;
//   checkCudaErrors(cublasCreate(&cublasHandle));


//   int n = 23;
//   int c = 13;

//   float* aa;
//   float* cc;
//   cudaMallocManaged(&aa, n*c*sizeof(float));
//   cudaMallocManaged(&cc, n*sizeof(float));

//   for(int i = 0; i < n*c; i++) {
//     aa[i] = i;
//   }

//   ClockTimer timer;
//   cudaPoolRowsSum(aa,cc,n,c);
//   float scale = 1.0f / c;
//   checkCudaErrors(cublasSscal(cublasHandle, n, &scale, cc, 1));
//   cudaDeviceSynchronize();

//   double timeTaken = timer.getSeconds();
//   cout << "timeTaken " << timeTaken << endl;

//   // float result[n];
//   // cudaMemcpy(result, cc, n*sizeof(float), cudaMemcpyDeviceToHost);

//   for(int i = 0; i<n; i++) {
//     cout << cc[i] << " ";
//   }
//   // for(int i = 0; i<n; i++) {
//   //   cout << result[i] << " ";
//   // }
//   cout << "Yay" << endl;
//   cudaFree(aa);
//   cudaFree(cc);

//   cublasDestroy(cublasHandle);

//   return 0;
// }



// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   checkCudaErrors(cudaSetDevice(0));

//   int n = 23;
//   int ca = 13;
//   int cb = 3;
//   int hw = 101;

//   float* aa;
//   float* bb;
//   float* cc;
//   cudaMallocManaged(&aa, n*ca*hw*sizeof(float));
//   cudaMallocManaged(&bb, n*cb*hw*sizeof(float));
//   cudaMallocManaged(&cc, n*(ca+cb)*hw*sizeof(float));

//   for(int i = 0; i < n*ca*hw; i++) {
//     aa[i] = 100000 + i*1.0f;
//   }
//   for(int i = 0; i < n*cb*hw; i++) {
//     bb[i] = i*1.0f;
//   }

//   ClockTimer timer;
//   cudaChannelConcat(aa,bb,cc,ca*hw,cb*hw,n);
//   double result = timer.getSeconds();
//   cout << "result " << result << endl;

//   for(int i = 0; i<n; i++) {
//     for(int j = 0; j<(ca+cb)*hw; j++) {
//       cout << cc[i*(ca+cb)*hw+j] << " ";
//     }
//     cout << endl;
//   }
//   cout << "Yay" << endl;
//   cudaFree(aa);
//   cudaFree(bb);
//   cudaFree(cc);

//   return 0;
// }



// static void checkCudnnStatus(const cudnnStatus_t& status, const char* subLabel) {
//   if(status != CUDNN_STATUS_SUCCESS)
//     throw StringError("CUDNN Error: " + string(subLabel) + ": " + cudnnGetErrorString(status));
// }

// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);

//   checkCudaErrors(cudaSetDevice(0));

//   cudnnStatus_t status;
//   cudnnHandle_t cudnn;
//   status = cudnnCreate(&cudnn);
//   checkCudnnStatus(status,"cudnnCreate");

//   int ySize = 5;
//   int xSize = 5;
//   int inChannels = 2;
//   int outChannels = 2;
//   int convYSize = 3;
//   int convXSize = 3;
//   int batchSize = 1;

//   cudnnTensorDescriptor_t inputDescriptor;
//   status = cudnnCreateTensorDescriptor(&inputDescriptor);
//   checkCudnnStatus(status,"cudnnCreateTensorDescriptor");
//   status = cudnnSetTensor4dDescriptor(
//     inputDescriptor,
//     CUDNN_TENSOR_NHWC,
//     CUDNN_DATA_FLOAT,
//     batchSize,
//     inChannels,
//     ySize,
//     xSize
//   );
//   checkCudnnStatus(status,"cudnnSetTensor4dDescriptor");

//   // cudnnTensorDescriptor_t inputDescriptor;
//   // status = cudnnCreateTensorDescriptor(&inputDescriptor);
//   // checkCudnnStatus(status,"cudnnCreateTensorDescriptor");
//   // status = cudnnSetTensor4dDescriptor(
//   //   inputDescriptor,
//   //   CUDNN_TENSOR_NCHW,
//   //   CUDNN_DATA_FLOAT,
//   //   batchSize,
//   //   inChannels,
//   //   ySize,
//   //   xSize
//   // );
//   // checkCudnnStatus(status,"cudnnSetTensor4dDescriptor");

//   cudnnTensorDescriptor_t outputDescriptor;
//   status = cudnnCreateTensorDescriptor(&outputDescriptor);
//   checkCudnnStatus(status,"cudnnCreateTensorDescriptor");
//   status = cudnnSetTensor4dDescriptor(
//     outputDescriptor,
//     CUDNN_TENSOR_NHWC,
//     CUDNN_DATA_FLOAT,
//     batchSize,
//     outChannels,
//     ySize,
//     xSize
//   );
//   checkCudnnStatus(status,"cudnnSetTensor4dDescriptor");

//   cudnnFilterDescriptor_t kernelDescriptor;
//   status = cudnnCreateFilterDescriptor(&kernelDescriptor);
//   checkCudnnStatus(status,"cudnnCreateFilterDescriptor");
//   status = cudnnSetFilter4dDescriptor(
//     kernelDescriptor,
//     CUDNN_DATA_FLOAT,
//     CUDNN_TENSOR_NCHW,
//     outChannels,
//     inChannels,
//     convYSize,
//     convXSize
//   );
//   checkCudnnStatus(status,"cudnnSetFilter4dDescriptor");

//   int paddingY = 1;
//   int paddingX = 1;
//   int yStride = 1;
//   int xStride = 1;
//   int dilationY = 1;
//   int dilationX = 1;

//   cudnnConvolutionDescriptor_t convolutionDescriptor;
//   status = cudnnCreateConvolutionDescriptor(&convolutionDescriptor);
//   checkCudnnStatus(status,"cudnnCreateConvolutionDescriptor");
//   status = cudnnSetConvolution2dDescriptor(
//     convolutionDescriptor,
//     paddingY,
//     paddingX,
//     yStride,
//     xStride,
//     dilationY,
//     dilationX,
//     CUDNN_CROSS_CORRELATION,
//     CUDNN_DATA_FLOAT
//   );
//   checkCudnnStatus(status,"cudnnSetConvolution2dDescriptor");

//   size_t bytesMemoryLimit = 0;
//   cudnnConvolutionFwdAlgo_t convolutionAlgorithm;
//   status = cudnnGetConvolutionForwardAlgorithm(
//     cudnn,
//     inputDescriptor,
//     kernelDescriptor,
//     convolutionDescriptor,
//     outputDescriptor,
//     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//     bytesMemoryLimit,
//     &convolutionAlgorithm
//   );
//   checkCudnnStatus(status,"cudnnGetConvolutionForwardAlgorithm");

//   size_t workspaceBytes = 0;
//   status = cudnnGetConvolutionForwardWorkspaceSize(
//     cudnn,
//     inputDescriptor,
//     kernelDescriptor,
//     convolutionDescriptor,
//     outputDescriptor,
//     convolutionAlgorithm,
//     &workspaceBytes
//   );
//   checkCudnnStatus(status,"cudnnGetConvolutionForwardWorkspaceSize");
//   cout << "Workspace size: " << workspaceBytes << endl;

//   float inputArr[batchSize][ySize][xSize][inChannels] = {
//     {
//       {{1,1},{1,0},{1,1},{1,0},{1,1}},
//       {{1,0},{2,2},{3,0},{4,2},{5},0},
//       {{1,3},{4,0},{9,3},{16,0},{25,3}},
//       {{1,0},{8,4},{27,0},{81,4},{125,0}},
//       {{1,5},{16,0},{81,5},{243,0},{625,5}},
//     },
//   };

//   // float inputArr[batchSize][inChannels][ySize][xSize] = {{
//   //   {
//   //     {1,1,1,1,1},
//   //     {1,2,3,4,5},
//   //     {1,4,9,16,25},
//   //     {1,8,27,81,125},
//   //     {1,16,81,243,625},
//   //   },
//   //   {
//   //     {1,0,1,0,1},
//   //     {0,2,0,2,0},
//   //     {3,0,3,0,3},
//   //     {0,4,0,4,0},
//   //     {5,0,5,0,5},
//   //   }
//   // }};

//   float kernelArr[outChannels][inChannels][convYSize][convXSize] = {
//     {
//       {
//         {0,0,0},
//         {0,1,0},
//         {0,0,0},
//       },
//       {
//         {0,0,0},
//         {0,1,0},
//         {0,0,0},
//       },
//     },
//     {
//       {
//         {0,0,0},
//         {0,0,1},
//         {0,0,0},
//       },
//       {
//         {0,0,0},
//         {0,0,-1},
//         {0,0,0},
//       },
//     },
//   };

//   void* gpuWorkspace = NULL;
//   cudaMalloc(&gpuWorkspace,workspaceBytes);

//   size_t inputBytes = sizeof(inputArr);
//   float* inputBuf = NULL;
//   cudaMalloc(&inputBuf, inputBytes);
//   cudaMemcpy(inputBuf, inputArr, inputBytes, cudaMemcpyHostToDevice);

//   size_t outputBytes = batchSize * outChannels * ySize * xSize * sizeof(float);
//   float* outputBuf = NULL;
//   cudaMalloc(&outputBuf, outputBytes);
//   cudaMemset(outputBuf, 0.0f, outputBytes);

//   size_t kernelBytes = sizeof(kernelArr);
//   float* kernelBuf = NULL;
//   cudaMalloc(&kernelBuf, kernelBytes);
//   cudaMemcpy(kernelBuf, kernelArr, kernelBytes, cudaMemcpyHostToDevice);

//   const float alpha = 1;
//   const float beta = 0;
//   status = cudnnConvolutionForward(
//     cudnn,
//     &alpha,
//     inputDescriptor,
//     inputBuf,
//     kernelDescriptor,
//     kernelBuf,
//     convolutionDescriptor,
//     convolutionAlgorithm,
//     gpuWorkspace,
//     workspaceBytes,
//     &beta,
//     outputDescriptor,
//     outputBuf
//   );
//   checkCudnnStatus(status,"cudnnConvolutionForward");

//   float outputArr[batchSize][ySize][xSize][outChannels];
//   cudaMemcpy(outputArr, outputBuf, outputBytes, cudaMemcpyDeviceToHost);

//   for(int b = 0; b<batchSize; b++) {
//     for(int c = 0; c<outChannels; c++) {
//       for(int y = 0; y<ySize; y++) {
//         for(int x = 0; x<xSize; x++) {
//           cout << outputArr[b][y][x][c] << " ";
//         }
//         cout << endl;
//       }
//       cout << endl;
//     }
//   }

//   // for(int b = 0; b<batchSize; b++) {
//   //   for(int c = 0; c<inChannels; c++) {
//   //     for(int y = 0; y<ySize; y++) {
//   //       for(int x = 0; x<xSize; x++) {
//   //         inputArr[b][c][y][x] *= -1;
//   //       }
//   //     }
//   //   }
//   // }
//   for(int b = 0; b<batchSize; b++) {
//     for(int y = 0; y<ySize; y++) {
//       for(int x = 0; x<xSize; x++) {
//         for(int c = 0; c<inChannels; c++) {
//           inputArr[b][y][x][c] *= -1;
//         }
//       }
//     }
//   }


//   cudaMemcpy(inputBuf, inputArr, inputBytes, cudaMemcpyHostToDevice);

//   status = cudnnConvolutionForward(
//     cudnn,
//     &alpha,
//     inputDescriptor,
//     inputBuf,
//     kernelDescriptor,
//     kernelBuf,
//     convolutionDescriptor,
//     convolutionAlgorithm,
//     gpuWorkspace,
//     workspaceBytes,
//     &beta,
//     outputDescriptor,
//     outputBuf
//   );
//   checkCudnnStatus(status,"cudnnConvolutionForward");

//   cudaMemcpy(outputArr, outputBuf, outputBytes, cudaMemcpyDeviceToHost);

//   cout << "-----------" << endl;
//   for(int b = 0; b<batchSize; b++) {
//     for(int c = 0; c<outChannels; c++) {
//       for(int y = 0; y<ySize; y++) {
//         for(int x = 0; x<xSize; x++) {
//           cout << outputArr[b][y][x][c] << " ";
//         }
//         cout << endl;
//       }
//       cout << endl;
//     }
//   }

//   cudaFree(inputBuf);
//   cudaFree(outputBuf);
//   cudaFree(kernelBuf);
//   cudaFree(gpuWorkspace);

//   cudnnDestroyTensorDescriptor(inputDescriptor);
//   cudnnDestroyTensorDescriptor(outputDescriptor);
//   cudnnDestroyFilterDescriptor(kernelDescriptor);
//   cudnnDestroyConvolutionDescriptor(convolutionDescriptor);

//   cudnnDestroy(cudnn);

//   cout << "Done" << endl;
//   return 0;
// }







// static void checkStatus(const Status& status, const char* subLabel) {
//   if(!status.ok())
//     throw StringError("NN Eval Error: " + string(subLabel) + status.ToString());
// }


// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);
//   logger.addFile("tmp.txt");

//   Session* session;
//   Status status;

//   string gpuVisibleDeviceList = ""; //use default
//   double perProcessGPUMemoryFraction = -1; //use default
//   SessionOptions sessionOptions = SessionOptions();
//   if(gpuVisibleDeviceList.length() > 0)
//     sessionOptions.config.mutable_gpu_options()->set_visible_device_list(gpuVisibleDeviceList);
//   if(perProcessGPUMemoryFraction >= 0.0)
//     sessionOptions.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(perProcessGPUMemoryFraction);

//   status = NewSession(sessionOptions, &session);

//   checkStatus(status, "creating session");

//   int modelFileIdx = 0;
//   int maxBatchSize = 8;
//   int nnCacheSizePowerOfTwo = 16;
//   bool debugSkipNeuralNet = false;
//   NNEvaluator* nnEval = new NNEvaluator(
//     session,
//     "/efs/data/GoNN/exportedmodels/value10-84/model.graph_optimized.pb",
//     modelFileIdx,
//     maxBatchSize,
//     nnCacheSizePowerOfTwo,
//     debugSkipNeuralNet
//   );

//   int numNNServerThreads = 2;
//   bool doRandomize = true;
//   string randSeed = "abc";
//   int defaultSymmetry = 0;
//   nnEval->spawnServerThreads(
//     numNNServerThreads,doRandomize,randSeed,defaultSymmetry,logger
//   );

//   Rules rules;
//   rules.koRule = Rules::KO_POSITIONAL;
//   rules.scoringRule = Rules::SCORING_AREA;
//   rules.multiStoneSuicideLegal = true;
//   rules.komi = 7.5f;

//   Player pla = P_WHITE;
//   Board board = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . x . .
// 18 . . x o . . . . . . x o . . o . o x .
// 17 . . x o . . o x . . . . o . . o x . .
// 16 . . x o . . o x x o . x . . . o x . .
// 15 . x o o x . x . x x x . x . . o x . .
// 14 . x o . . . x x o o o o x . x o o x .
// 13 . x o . . . . . o x x x x . . . o x .
// 12 . . o . . x x x . o . o o o o . o . .
// 11 . . . . o x o o o o . o . x . o . . .
// 10 . o o o o o x . . o x x x . o x x . .
//  9 . x . x o o x x x x o o x . x o x . .
//  8 . . . x x x x . . x . o o.x . o x . .
//  7 . . . o o . x x . x . . . . . x . x .
//  6 . . o x x x . x x o o . o . . x . x .
//  5 . . o o o o x x . . . o . o . o x . .
//  4 . o o x x o o . x o o x . o . o x . .
//  3 . o x x . o o x x x . x . o x x o x .
//  2 o x . x x o . o . . . . . o . . o x .
//  1 . o x x o . . o . . . . . . . . . . .
// )");

//   BoardHistory hist(board,pla,rules);
//   SearchParams params;
//   params.maxPlayouts = 1000;
//   params.numThreads = 6;

//   AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "def");
//   bot->setPosition(pla,board,hist);

//   Loc moveLoc = bot->genMoveSynchronous(pla);
//   bot->clearSearch();
//   nnEval->clearCache();
//   ClockTimer timer;
//   moveLoc = bot->genMoveSynchronous(pla);

//   double seconds = timer.getSeconds();
//   cout << board << endl;
//   cout << "MoveLoc: " << Location::toString(moveLoc,board) << endl;
//   cout << "Seconds: " << seconds << endl;
//   bot->getSearch()->printTree(cout, bot->getSearch()->rootNode, PrintTreeOptions().maxDepth(1));

//   cout << "NN rows: " << nnEval->numRowsProcessed() << endl;
//   cout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
//   cout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;

//   cout << "sizeof(uint8_t) " << sizeof(uint8_t) << endl;
//   cout << "sizeof(uint16_t) " << sizeof(uint16_t) << endl;
//   cout << "sizeof(uint32_t) " << sizeof(uint32_t) << endl;
//   cout << "sizeof(uint64_t) " << sizeof(uint64_t) << endl;
//   cout << "sizeof(std::atomic_flag) " << sizeof(std::atomic_flag) << endl;;
//   cout << "sizeof(std::mutex) " << sizeof(std::mutex) << endl;;
//   cout << "sizeof(std::shared_ptr<NNOutput>) " << sizeof(std::shared_ptr<NNOutput>) << endl;;

//   {
//     atomic<bool>* b = new atomic<bool>(false);
//     cout << "atomic<bool> lock free " << std::atomic_is_lock_free(b) << endl;
//     delete b;
//   }
//   {
//     atomic<uint64_t>* b = new atomic<uint64_t>(0);
//     cout << "atomic<uint64_t> lock free " << std::atomic_is_lock_free(b) << endl;
//     delete b;
//   }

//   nnEval->killServerThreads();
//   delete bot;
//   delete nnEval;

//   cout << "Done" << endl;
//   return 0;
// }




// int MainCmds::sandbox() {
//   Board::initHash();

//   Logger logger;
//   logger.setLogToStdout(true);
//   logger.addFile("tmp.txt");

//   int maxBatchSize = 8;
//   int nnCacheSizePowerOfTwo = 16;
//   NNEvaluator* nnEval = new NNEvaluator("/efs/data/GoNN/exportedmodels/value10-84/model.graph_optimized.pb", maxBatchSize, nnCacheSizePowerOfTwo);

//   int numNNServerThreads = 1;
//   bool doRandomize = true;
//   string randSeed = "abc";
//   int defaultSymmetry = 0;
//   vector<std::thread*> nnServerThreads = nnEval->spawnServerThreads(numNNServerThreads,doRandomize,randSeed,defaultSymmetry,logger);

//   Rules rules;
//   rules.koRule = Rules::KO_POSITIONAL;
//   rules.scoringRule = Rules::SCORING_AREA;
//   rules.multiStoneSuicideLegal = true;
//   rules.komi = 7.5f;

//   Player pla = P_WHITE;
//   Board board = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . x . .
// 18 . . x o . . . . . . x o . . o . o x .
// 17 . . x o . . o x . . . . o . . o x . .
// 16 . . x o . . o x x o . x . . . o x . .
// 15 . x o o x . x . x x x . x . . o x . .
// 14 . x o . . . x x o o o o x . x o o x .
// 13 . x o . . . . . o x x x x . . . o x .
// 12 . . o . . x x x . o . o o o o . o . .
// 11 . . . . o x o o o o . o . x . o . . .
// 10 . o o o o o x . . o x x x . o x x . .
//  9 . x . x o o x x x x o o x . x o x . .
//  8 . . . x x x x . . x . o o.x . o x . .
//  7 . . . o o . x x . x . . . . . x . x .
//  6 . . o x x x . x x o o . o . . x . x .
//  5 . . o o o o x x . . . o . o . o x . .
//  4 . o o x x o o . x o o x . o . o x . .
//  3 . o x x . o o x x x . x . o x x o x .
//  2 o x . x x o . o . . . . . o . . o x .
//  1 . o x x o . . o . . . . . . . . . . .
// )");

//   BoardHistory hist(board,pla,rules);
//   SearchParams params;

//   Search* search = new Search(params, nnEval);
//   search->setPosition(pla,board,hist);

//   search->beginSearch();
//   SearchThread* searchThread = new SearchThread(0,*search,&logger);

//   ClockTimer timer;
//   for(int i = 0; i<300; i++)
//     search->runSinglePlayout(*searchThread);

//   double seconds = timer.getSeconds();
//   cout << board << endl;
//   cout << "Seconds: " << seconds << endl;
//   search->printTree(cout, search->rootNode, PrintTreeOptions().maxDepth(1));

//   cout << "sizeof(uint8_t) " << sizeof(uint8_t) << endl;
//   cout << "sizeof(uint16_t) " << sizeof(uint16_t) << endl;
//   cout << "sizeof(uint32_t) " << sizeof(uint32_t) << endl;
//   cout << "sizeof(uint64_t) " << sizeof(uint64_t) << endl;
//   cout << "sizeof(std::atomic_flag) " << sizeof(std::atomic_flag) << endl;;
//   cout << "sizeof(std::mutex) " << sizeof(std::mutex) << endl;;
//   cout << "sizeof(std::shared_ptr<NNOutput>) " << sizeof(std::shared_ptr<NNOutput>) << endl;;

//   {
//     atomic<bool>* b = new atomic<bool>(false);
//     cout << "atomic<bool> lock free " << std::atomic_is_lock_free(b) << endl;
//     delete b;
//   }
//   {
//     atomic<uint64_t>* b = new atomic<uint64_t>(0);
//     cout << "atomic<uint64_t> lock free " << std::atomic_is_lock_free(b) << endl;
//     delete b;
//   }

//   nnEval->killServers();
//   for(size_t i = 0; i<nnServerThreads.size(); i++)
//     nnServerThreads[i]->join();
//   for(size_t i = 0; i<nnServerThreads.size(); i++)
//     delete nnServerThreads[i];

//   delete searchThread;
//   delete search;
//   delete nnEval;

//   cout << "Done" << endl;
//   return 0;
// }





// int MainCmds::sandbox() {
//   Board::initHash();

//   int maxBatchSize = 8;
//   NNEvaluator* nnEval = new NNEvaluator("/efs/data/GoNN/exportedmodels/value10-84/model.graph_optimized.pb", maxBatchSize);

//   auto serveEvals = [&nnEval](int threadIdx) {
//     NNServerBuf* buf = new NNServerBuf(*nnEval);
//     Rand rand("NNServerThread " + Global::intToString(threadIdx));
//     try {
//       nnEval->serve(*buf,&rand,0);
//     }
//     catch(const exception& e) {
//       cout << "NN Server Thread: " << e.what() << endl;
//     }
//     catch(const string& e) {
//       cout << "NN Server Thread: " << e << endl;
//     }
//     catch(...) {
//       cout << "Unexpected throw in NN server thread" << endl;
//     }
//   };

//   thread nnServerThread(serveEvals,0);

//   Rules rules;
//   rules.koRule = Rules::KO_POSITIONAL;
//   rules.scoringRule = Rules::SCORING_AREA;
//   rules.multiStoneSuicideLegal = true;
//   rules.komi = 7.5f;

//   Board board;
//   BoardHistory boardHistory(board,P_BLACK,rules);
//   Player nextPlayer = P_WHITE;

//   Loc loc = Location::getLoc(2,3,board.x_size);
//   boardHistory.makeBoardMoveAssumeLegal(board,loc,P_BLACK,NULL);

//   NNResultBuf resultBuf;

//   for(int i = 0; i<10; i++) {
//     nnEval->evaluate(board,boardHistory,nextPlayer,resultBuf);

//     shared_ptr<NNOutput> output = std::move(resultBuf.result);

//     for(int y = 0; y<NNPos::MAX_BOARD_LEN; y++) {
//       for(int x = 0; x<NNPos::MAX_BOARD_LEN; x++) {
//         float prob = output->policyProbs[x+y*NNPos::MAX_BOARD_LEN];
//         if(prob < 0)
//           printf("    %%");
//         else
//           printf("%4.1f%%", prob * 100.0);
//       }
//       cout << endl;
//     }
//     printf("%4.1f%%", output->policyProbs[NNPos::NN_POLICY_SIZE-1] * 100.0);
//     cout << endl;
//     cout << output->whiteValue << endl;
//     cout << endl;
//     sleep(1);
//   }

//   nnEval->killServers();
//   nnServerThread.join();
//   cout << "Done" << endl;

// }



// int MainCmds::sandbox() {
//   Board::initHash();

//   auto checkStatus = [](Status status, const char* subLabel) {
//     if(!status.ok())
//       throw StringError("NNEvaluator initialization failed: " + string(subLabel) + ": " + status.ToString());
//   };

//   Session* session;
//   Status status;
//   GraphDef graphDef1;
//   GraphDef graphDef2;

//   //Create session
//   status = NewSession(SessionOptions(), &session);
//   checkStatus(status,"creating session");

//   //Read graph from file
//   status = ReadBinaryProto(Env::Default(), string("/efs/data/GoNN/exportedmodels/value10-84/model.graph_optimized.pb"), &graphDef1);
//   checkStatus(status,"reading graph1");
//   status = ReadBinaryProto(Env::Default(), string("/efs/data/GoNN/exportedmodels/value18-140/model.graph_optimized.pb"), &graphDef2);
//   checkStatus(status,"reading graph2");

//   auto addPrefixToGraph = [](GraphDef& graphDef, const string& prefix) {
//     //string device = "/gpu:0";
//     for(int i = 0; i < graphDef.node_size(); ++i)
//     {
//       auto node = graphDef.mutable_node(i);
//       //node->set_device(device);
//       string* name = node->mutable_name();
//       *name = prefix + *name;
//       int inputSize = node->input_size();
//       for(int j = 0; j<inputSize; j++) {
//         string* inputName = node->mutable_input(j);
//         if(inputName->size() > 0 && (*inputName)[0] == '^')
//           *inputName = "^" + prefix + inputName->substr(1);
//         else
//           *inputName = prefix + *inputName;
//       }
//     }
//   };
//   addPrefixToGraph(graphDef1,"g1/");
//   addPrefixToGraph(graphDef2,"g2/");

//   //Add graph to session
//   status = session->Create(graphDef1);
//   checkStatus(status,"adding graph1 to session");
//   status = session->Extend(graphDef2);
//   checkStatus(status,"adding graph2 to session");

//   int outputBatchSize = 4;

//   //Set up inputs
//   TensorShape inputsShape;
//   TensorShape symmetriesShape;
//   TensorShape isTrainingShape;
//   int inputsShapeArr[3] = {outputBatchSize,NNPos::MAX_BOARD_AREA,NNInputs::NUM_FEATURES_V1};
//   status = TensorShapeUtils::MakeShape(inputsShapeArr,3,&inputsShape);
//   checkStatus(status,"making inputs shape");
//   int symmetriesShapeArr[1] = {NNInputs::NUM_SYMMETRY_BOOLS};
//   status = TensorShapeUtils::MakeShape(symmetriesShapeArr,1,&symmetriesShape);
//   checkStatus(status,"making symmetries shape");
//   int isTrainingShapeArr[0] = {};
//   status = TensorShapeUtils::MakeShape(isTrainingShapeArr,0,&isTrainingShape);
//   checkStatus(status,"making isTraining shape");

//   Rules rules;
//   rules.koRule = Rules::KO_POSITIONAL;
//   rules.scoringRule = Rules::SCORING_AREA;
//   rules.multiStoneSuicideLegal = true;
//   rules.komi = 7.5f;

//   Board board1 = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . . . .
// 18 . . x o . . . . . . x o . . o . o x .
// 17 . . x o . . o x . . . . o . . o x . .
// 16 . . x o . . o x x o . x . . . o x . .
// 15 . x o o x . x . x x x . x . . o x . .
// 14 . x o . . . x x o o o o x . x o o x .
// 13 . x o . . . . . o x x x x . . . o x .
// 12 . . o . . x x x . o . o o o o . o . .
// 11 . . . . o x o o o o . o . x . o . . .
// 10 . o o o o o x . . o x x x . o x x . .
//  9 . x . x o o x x x x o o x . x o x . .
//  8 . . . x x x x . . x . o o x . o x . .
//  7 . . . o o . x x . x . . . . . x . x .
//  6 . . o x x x . x x o o . o . . x . x .
//  5 . . o o o o x x . . . o . o . o x . .
//  4 . o o x x o o . x o o x . o . o x . .
//  3 . o x x . o o x x x . x . o x x o x .
//  2 o x . x x o . o . . . . . o . . o x .
//  1 . o x x o . . o . . . . . . . . . . .
// )");
//   Board board2 = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . . . .
// 18 . . . . . . . . . . . . . . . . . . .
// 17 . . . . . . . . . . . . . . o x . . .
// 16 . . . o . . . . . . . . . . . . . . .
// 15 . . . . . . . . . . . . . . . . x . .
// 14 . . . . . . . . . . . . . . . . . . .
// 13 . . . . . . . . . . . . . . . . . . .
// 12 . . . . . . . . . . . . . . . . . . .
// 11 . . . . . . . . . . . . . . . . . . .
// 10 . . . . . . . . . . . . . . . . . . .
//  9 . . . . . . . . . . . . . . . . . . .
//  8 . . . . . . . . . . . . . . . . . . .
//  7 . . . . . . . . . . . . . . . . . . .
//  6 . . . . . . . . . . . . . . . . . . .
//  5 . . . . . . . . . . . . . . . . . . .
//  4 . . . . . . . . . . . . . . . x . . .
//  3 . . . o . . . . . . . . . . . . . . .
//  2 . . . . . . . . . . . . . . . . . . .
//  1 . . . . . . . . . . . . . . . . . . .
// )");

//   Player nextPlayer = P_BLACK;
//   BoardHistory hist1(board1,nextPlayer,rules);
//   BoardHistory hist2(board2,nextPlayer,rules);

//   auto runInputsInLoop = [&](vector<float>* res, int which) {
//     vector<float>& results = *res;

//     Tensor inputs(DT_FLOAT,inputsShape);
//     Tensor symmetries(DT_BOOL,symmetriesShape);
//     Tensor isTraining(DT_BOOL,isTrainingShape);

//     auto symmetriesMap = symmetries.tensor<bool, 1>();
//     symmetriesMap(0) = false;
//     symmetriesMap(1) = false;
//     symmetriesMap(2) = false;

//     auto isTrainingMap = isTraining.tensor<bool, 0>();
//     isTrainingMap(0) = false;

//     // Tensor sliced = inputs.Slice(0,1);

//     vector<pair<string,Tensor>> inputsList1 = {
//       {"g1/inputs",inputs},
//       {"g1/symmetries",symmetries},
//       {"g1/is_training",isTraining},
//     };
//     vector<pair<string,Tensor>> inputsList2 = {
//       {"g2/inputs",inputs},
//       {"g2/symmetries",symmetries},
//       {"g2/is_training",isTraining},
//     };

//     for(int i = 0; i<500; i++) {
//       float* row = inputs.flat<float>().data();
//       for(int j = 0; j<outputBatchSize * NNInputs::ROW_SIZE_V1; j++)
//         row[j] = 0.0f;

//       NNInputs::fillRowV1(board1, hist1, nextPlayer, row);
//       NNInputs::fillRowV1(board2, hist2, nextPlayer, row + NNInputs::ROW_SIZE_V1);
//       NNInputs::fillRowV1(board1, hist1, nextPlayer, row + NNInputs::ROW_SIZE_V1*2);
//       NNInputs::fillRowV1(board2, hist2, nextPlayer, row + NNInputs::ROW_SIZE_V1*3);

//       vector<Tensor> outputs;
//       // cout << "Running" << endl;
//       if(which == 1)
//         status = session->Run(inputsList1, {"g1/policy_output","g1/value_output"}, {}, &outputs);
//       else
//         status = session->Run(inputsList2, {"g2/policy_output","g2/value_output"}, {}, &outputs);

//       checkStatus(status,"running inference");
//       assert(outputs.size() == 2);

//       assert(outputs[0].dims() == 2);
//       assert(outputs[1].dims() == 1);
//       assert(outputs[0].dim_size(0) == outputBatchSize); //batch
//       assert(outputs[0].dim_size(1) == NNPos::NN_POLICY_SIZE);
//       assert(outputs[1].dim_size(0) == outputBatchSize); //batch

//       // auto policyMap = outputs[0].matrix<float>();
//       auto valueMap = outputs[1].vec<float>();

//       for(int batch = 0; batch < outputBatchSize; batch++) {
//         // float policy[NNPos::NN_POLICY_SIZE];
//         // float maxPolicy = -1e10f;
//         // for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
//         //   policy[i] = policyMap(batch,i);
//         //   if(policy[i] > maxPolicy)
//         //     maxPolicy = policy[i];
//         // }
//         // float policySum = 0.0f;
//         // for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
//         //   policy[i] = exp(policy[i] - maxPolicy);
//         //   policySum += policy[i];
//         // }
//         // for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
//         //   policy[i] /= policySum;
//         // }

//         float value = valueMap(batch);

//         // for(int y = 0; y<NNPos::MAX_BOARD_LEN; y++) {
//         //   for(int x = 0; x<NNPos::MAX_BOARD_LEN; x++) {
//         //     printf("%4.1f%%", policy[x+y*NNPos::MAX_BOARD_LEN] * 100.0);
//         //   }
//         //   cout << endl;
//         // }
//         // printf("%4.1f%%", policy[NNPos::NN_POLICY_SIZE-1] * 100.0);
//         // cout << endl;
//         // cout << value << endl;
//         results.push_back(value);
//       }
//     }
//   };

//   int numThreads = 4;

//   vector<std::thread> threads;
//   vector<float> results[numThreads];
//   for(int i = 0; i<numThreads; i++)
//     results[i] = vector<float>();

//   for(int i = 0; i<numThreads; i++) {
//     if(i % 2 == 0)
//       threads.push_back(std::thread(runInputsInLoop,&(results[i]),1));
//     else
//       threads.push_back(std::thread(runInputsInLoop,&(results[i]),2));
//   }
//   for(int i = 0; i<numThreads; i++) {
//     threads[i].join();
//   }
//   for(int i = 0; i<numThreads; i++) {
//     for(int j = 0; j<results[i].size(); j++) {
//       printf("%.9f", results[i][j]);
//       cout << endl;
//     }
//   }

//   cout << "Done" << endl;

//   return 0;
// }


// int MainCmds::sandbox() {
//   Board::initHash();

//   auto checkStatus = [](Status status, const char* subLabel) {
//     if(!status.ok())
//       throw StringError("NNEvaluator initialization failed: " + string(subLabel) + ": " + status.ToString());
//   };

//   Session* session;
//   Status status;
//   GraphDef graphDef;

//   //Create session
//   status = NewSession(SessionOptions(), &session);
//   checkStatus(status,"creating session");

//   //Read graph from file
//   // status = ReadTextProto(Env::Default(), string("/efs/data/GoNN/exportedmodels/value10-84/model.graph.pb"), &graphDef);
//   // status = ReadBinaryProto(Env::Default(), string("/efs/data/GoNN/exportedmodels/value10-84/model.graph_frozen.pb"), &graphDef);
//   status = ReadBinaryProto(Env::Default(), string("/efs/data/GoNN/exportedmodels/value10-84/model.graph_optimized.pb"), &graphDef);
//   checkStatus(status,"reading graph");

//   //Add graph to session
//   status = session->Create(graphDef);
//   checkStatus(status,"adding graph to session");

//   //Set up inputs
//   TensorShape inputsShape;
//   TensorShape symmetriesShape;
//   TensorShape isTrainingShape;
//   int inputsShapeArr[3] = {2,NNPos::MAX_BOARD_AREA,NNInputs::NUM_FEATURES_V1};
//   status = TensorShapeUtils::MakeShape(inputsShapeArr,3,&inputsShape);
//   checkStatus(status,"making inputs shape");
//   int symmetriesShapeArr[1] = {NNInputs::NUM_SYMMETRY_BOOLS};
//   status = TensorShapeUtils::MakeShape(symmetriesShapeArr,1,&symmetriesShape);
//   checkStatus(status,"making symmetries shape");
//   int isTrainingShapeArr[0] = {};
//   status = TensorShapeUtils::MakeShape(isTrainingShapeArr,0,&isTrainingShape);
//   checkStatus(status,"making isTraining shape");

//   Tensor inputs(DT_FLOAT,inputsShape);
//   Tensor symmetries(DT_BOOL,symmetriesShape);
//   Tensor isTraining(DT_BOOL,isTrainingShape);

//   float* row = inputs.flat<float>().data();

//   // float row[NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES_V1];
//   for(int i = 0; i<2*NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES_V1; i++)
//     row[i] = 0.0f;

//   Rules rules;
//   rules.koRule = Rules::KO_POSITIONAL;
//   rules.scoringRule = Rules::SCORING_AREA;
//   rules.multiStoneSuicideLegal = true;
//   rules.komi = 7.5f;

//   Board board1 = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . . . .
// 18 . . x o . . . . . . x o . . o . o x .
// 17 . . x o . . o x . . . . o . . o x . .
// 16 . . x o . . o x x o . x . . . o x . .
// 15 . x o o x . x . x x x . x . . o x . .
// 14 . x o . . . x x o o o o x . x o o x .
// 13 . x o . . . . . o x x x x . . . o x .
// 12 . . o . . x x x . o . o o o o . o . .
// 11 . . . . o x o o o o . o . x . o . . .
// 10 . o o o o o x . . o x x x . o x x . .
//  9 . x . x o o x x x x o o x . x o x . .
//  8 . . . x x x x . . x . o o x . o x . .
//  7 . . . o o . x x . x . . . . . x . x .
//  6 . . o x x x . x x o o . o . . x . x .
//  5 . . o o o o x x . . . o . o . o x . .
//  4 . o o x x o o . x o o x . o . o x . .
//  3 . o x x . o o x x x . x . o x x o x .
//  2 o x . x x o . o . . . . . o . . o x .
//  1 . o x x o . . o . . . . . . . . . . .
// )");
//   Board board2 = Board::parseBoard(19,19,R"(
//    A B C D E F G H J K L M N O P Q R S T
// 19 . . . . . . . . . . . . . . . . . . .
// 18 . . . . . . . . . . . . . . . . . . .
// 17 . . . . . . . . . . . . . . o x . . .
// 16 . . . o . . . . . . . . . . . . . . .
// 15 . . . . . . . . . . . . . . . . x . .
// 14 . . . . . . . . . . . . . . . . . . .
// 13 . . . . . . . . . . . . . . . . . . .
// 12 . . . . . . . . . . . . . . . . . . .
// 11 . . . . . . . . . . . . . . . . . . .
// 10 . . . . . . . . . . . . . . . . . . .
//  9 . . . . . . . . . . . . . . . . . . .
//  8 . . . . . . . . . . . . . . . . . . .
//  7 . . . . . . . . . . . . . . . . . . .
//  6 . . . . . . . . . . . . . . . . . . .
//  5 . . . . . . . . . . . . . . . . . . .
//  4 . . . . . . . . . . . . . . . x . . .
//  3 . . . o . . . . . . . . . . . . . . .
//  2 . . . . . . . . . . . . . . . . . . .
//  1 . . . . . . . . . . . . . . . . . . .
// )");

//   Player nextPlayer = P_BLACK;
//   BoardHistory hist1(board1,nextPlayer,rules);
//   BoardHistory hist2(board2,nextPlayer,rules);

//   NNInputs::fillRowV1(board1, hist1, nextPlayer, row);
//   NNInputs::fillRowV1(board2, hist2, nextPlayer, row + NNInputs::ROW_SIZE_V1);

//   auto symmetriesMap = symmetries.tensor<bool, 1>();
//   symmetriesMap(0) = false;
//   symmetriesMap(1) = false;
//   symmetriesMap(2) = false;

//   auto isTrainingMap = isTraining.tensor<bool, 0>();
//   isTrainingMap(0) = false;

//   cout << "ISALIGNED " << inputs.IsAligned() << endl;
//   Tensor sliced = inputs.Slice(0,1);
//   cout << "ISALIGNED " << sliced.IsAligned() << endl;
//   int outputBatchSize = 1;

//   vector<pair<string,Tensor>> inputsList = {
//     {"inputs",sliced},
//     {"symmetries",symmetries},
//     {"is_training",isTraining},
//   };

//   vector<Tensor> outputs;

//   status = session->Run(inputsList, {"policy_output","value_output"}, {}, &outputs);
//   checkStatus(status,"running inference");
//   assert(outputs.size() == 2);

//   assert(outputs[0].dims() == 2);
//   assert(outputs[1].dims() == 1);
//   assert(outputs[0].dim_size(0) == outputBatchSize); //batch
//   assert(outputs[0].dim_size(1) == NNPos::NN_POLICY_SIZE);
//   assert(outputs[1].dim_size(0) == outputBatchSize); //batch

//   auto policyMap = outputs[0].matrix<float>();
//   auto valueMap = outputs[1].vec<float>();

//   for(int batch = 0; batch < outputBatchSize; batch++) {
//     float policy[NNPos::NN_POLICY_SIZE];
//     float maxPolicy = -1e10f;
//     for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
//       policy[i] = policyMap(batch,i);
//       if(policy[i] > maxPolicy)
//         maxPolicy = policy[i];
//     }
//     float policySum = 0.0f;
//     for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
//       policy[i] = exp(policy[i] - maxPolicy);
//       policySum += policy[i];
//     }
//     for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
//       policy[i] /= policySum;
//     }

//     float value = valueMap(batch);

//     for(int y = 0; y<NNPos::MAX_BOARD_LEN; y++) {
//       for(int x = 0; x<NNPos::MAX_BOARD_LEN; x++) {
//         printf("%4.1f%%", policy[x+y*NNPos::MAX_BOARD_LEN] * 100.0);
//       }
//       cout << endl;
//     }
//     printf("%4.1f%%", policy[NNPos::NN_POLICY_SIZE-1] * 100.0);
//     cout << endl;
//     cout << value << endl;
//   }

//   return 0;
// }
