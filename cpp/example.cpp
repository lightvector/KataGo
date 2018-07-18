// example.cpp

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

// #include <tensorflow/c/c_api.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <iostream>
using namespace std;
using namespace tensorflow;


int main() {
  Board::initHash();

  Logger logger;
  logger.setLogToStdout(true);
  logger.addFile("tmp.txt");

  int maxBatchSize = 8;
  int nnCacheSizePowerOfTwo = 16;
  bool debugSkipNeuralNet = false;
  NNEvaluator* nnEval = new NNEvaluator(
    "/efs/data/GoNN/exportedmodels/value10-84/model.graph_optimized.pb",
    maxBatchSize,
    nnCacheSizePowerOfTwo,
    debugSkipNeuralNet
  );

  int numNNServerThreads = 1;
  bool doRandomize = true;
  string randSeed = "abc";
  int defaultSymmetry = 0;
  vector<string> gpuVisibleDeviceListByThread = {}; //use default
  double perProcessGPUMemoryFraction = -1; //use default
  nnEval->spawnServerThreads(
    numNNServerThreads,doRandomize,randSeed,defaultSymmetry,logger,
    gpuVisibleDeviceListByThread,
    perProcessGPUMemoryFraction
  );

  Rules rules;
  rules.koRule = Rules::KO_POSITIONAL;
  rules.scoringRule = Rules::SCORING_AREA;
  rules.multiStoneSuicideLegal = true;
  rules.komi = 7.5f;

  Player pla = P_WHITE;
  Board board = Board::parseBoard(19,19,R"(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . x . .
18 . . x o . . . . . . x o . . o . o x .
17 . . x o . . o x . . . . o . . o x . .
16 . . x o . . o x x o . x . . . o x . .
15 . x o o x . x . x x x . x . . o x . .
14 . x o . . . x x o o o o x . x o o x .
13 . x o . . . . . o x x x x . . . o x .
12 . . o . . x x x . o . o o o o . o . .
11 . . . . o x o o o o . o . x . o . . .
10 . o o o o o x . . o x x x . o x x . .
 9 . x . x o o x x x x o o x . x o x . .
 8 . . . x x x x . . x . o o.x . o x . .
 7 . . . o o . x x . x . . . . . x . x .
 6 . . o x x x . x x o o . o . . x . x .
 5 . . o o o o x x . . . o . o . o x . .
 4 . o o x x o o . x o o x . o . o x . .
 3 . o x x . o o x x x . x . o x x o x .
 2 o x . x x o . o . . . . . o . . o x .
 1 . o x x o . . o . . . . . . . . . . .
)");

  BoardHistory hist(board,pla,rules);
  SearchParams params;
  params.maxPlayouts = 1000;
  params.numThreads = 1;

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger);
  bot->setPosition(pla,board,hist);

  Loc moveLoc = bot->genMoveSynchronous(pla);
  bot->clearSearch();
  nnEval->clearCache();
  ClockTimer timer;
  moveLoc = bot->genMoveSynchronous(pla);

  double seconds = timer.getSeconds();
  cout << board << endl;
  cout << "MoveLoc: " << Location::toString(moveLoc,board) << endl;
  cout << "Seconds: " << seconds << endl;
  bot->getSearch()->printTree(cout, bot->getSearch()->rootNode, PrintTreeOptions().maxDepth(1));

  cout << "NN rows: " << nnEval->numRowsProcessed() << endl;
  cout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
  cout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;

  cout << "sizeof(uint8_t) " << sizeof(uint8_t) << endl;
  cout << "sizeof(uint16_t) " << sizeof(uint16_t) << endl;
  cout << "sizeof(uint32_t) " << sizeof(uint32_t) << endl;
  cout << "sizeof(uint64_t) " << sizeof(uint64_t) << endl;
  cout << "sizeof(std::atomic_flag) " << sizeof(std::atomic_flag) << endl;;
  cout << "sizeof(std::mutex) " << sizeof(std::mutex) << endl;;
  cout << "sizeof(std::shared_ptr<NNOutput>) " << sizeof(std::shared_ptr<NNOutput>) << endl;;

  {
    atomic<bool>* b = new atomic<bool>(false);
    cout << "atomic<bool> lock free " << std::atomic_is_lock_free(b) << endl;
    delete b;
  }
  {
    atomic<uint64_t>* b = new atomic<uint64_t>(0);
    cout << "atomic<uint64_t> lock free " << std::atomic_is_lock_free(b) << endl;
    delete b;
  }

  nnEval->killServerThreads();
  delete bot;
  delete nnEval;

  cout << "Done" << endl;
  return 0;
}




// int main() {
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





// int main() {
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





// int main() {
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
