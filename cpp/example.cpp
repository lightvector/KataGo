// example.cpp

#include "core/global.h"
#include "game/board.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nneval.h"
#include "search/searchparams.h"
#include "search/search.h"

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

  NNEvaluator* nnEval = new NNEvaluator("/efs/data/GoNN/exportedmodels/value10-84/model.graph_optimized.pb");

  Rules rules;
  rules.koRule = Rules::KO_POSITIONAL;
  rules.scoringRule = Rules::SCORING_AREA;
  rules.multiStoneSuicideLegal = true;
  rules.komi = 7.5f;

  Player pla = P_WHITE;
  Board board = Board::parseBoard(19,19,R"(
...................
...................
...................
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
..............x.o..
...o..........ox...
...................
...................
)");

  BoardHistory hist(board,pla,rules);
  SearchParams params;
  int mutexPoolSize = 4096;

  Search* search = new Search(rules, params, mutexPoolSize);
  search->setPosition(pla,board,hist);

  search->beginSearch("randseed",nnEval);
  SearchThread* thread = new SearchThread(0,*search);

  search->runSinglePlayout(*thread);
  thread->board.checkConsistency();
  search->runSinglePlayout(*thread);
  thread->board.checkConsistency();
  search->runSinglePlayout(*thread);

  cout << board << endl;
  search->printTree(cout, search->rootNode, PrintTreeOptions().maxDepth(10));


  delete thread;
  delete search;
  delete nnEval;

  return 0;

  // Board board;
  // BoardHistory boardHistory(board,P_BLACK,rules);
  // Player nextPlayer = P_WHITE;

  // Loc loc = Location::getLoc(2,3,board.x_size);
  // boardHistory.makeBoardMoveAssumeLegal(board,loc,P_BLACK,NULL);

  // for(int symmetry = 0; symmetry < 8; symmetry++) {
  //   shared_ptr<NNOutput> output = nnEval->evaluate(board,boardHistory,nextPlayer,symmetry);

  //   cout << "SYMMETRY " << symmetry << endl;
  //   for(int y = 0; y<NNPos::MAX_BOARD_LEN; y++) {
  //     for(int x = 0; x<NNPos::MAX_BOARD_LEN; x++) {
  //       float prob = output->policyProbs[x+y*NNPos::MAX_BOARD_LEN];
  //       if(prob < 0)
  //         printf("    %%");
  //       else
  //         printf("%4.1f%%", prob * 100.0);
  //     }
  //     cout << endl;
  //   }
  //   printf("%4.1f%%", output->policyProbs[NNPos::NN_POLICY_SIZE-1] * 100.0);
  //   cout << endl;
  //   cout << output->value << endl;
  //   cout << endl;
  // }

}

// int main() {
//   Board::initHash();

//   auto checkStatus = [](Status status, const char* subLabel) {
//     if(!status.ok())
//       throw StringError("NNEvaluator initialization failed: ", string(subLabel) + ": " + status.ToString());
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
//   int inputsShapeArr[3] = {2,NNPos::MAX_BOARD_AREA,NNInputs::NUM_FEATURES};
//   status = TensorShapeUtils::MakeShape(inputsShapeArr,3,&inputsShape);
//   checkStatus(status,"making inputs shape");
//   int symmetriesShapeArr[1] = {NNInputs::NUM_SYMMETRIES};
//   status = TensorShapeUtils::MakeShape(symmetriesShapeArr,1,&symmetriesShape);
//   checkStatus(status,"making symmetries shape");
//   int isTrainingShapeArr[0] = {};
//   status = TensorShapeUtils::MakeShape(isTrainingShapeArr,0,&isTrainingShape);
//   checkStatus(status,"making isTraining shape");

//   Tensor inputs(DT_FLOAT,inputsShape);
//   Tensor symmetries(DT_BOOL,symmetriesShape);
//   Tensor isTraining(DT_BOOL,isTrainingShape);

//   float* row = inputs.flat<float>().data();

//   // float row[NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES];
//   for(int i = 0; i<2*NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES; i++)
//     row[i] = 0.0f;

//   cout << "AAAA" << endl;
//   Board board;
//   vector<Move> moveHistory;
//   int moveHistoryLen = 0;
//   Player nextPlayer = P_BLACK;
//   float selfKomi = -7.5f;

//   NNInputs::fillRow(board, moveHistory, moveHistoryLen, nextPlayer, selfKomi, row);

//   nextPlayer = P_WHITE;
//   selfKomi = 7.5f;
//   board.playMove(Location::getLoc(2,3,board.x_size),P_BLACK);
//   NNInputs::fillRow(board, moveHistory, moveHistoryLen, nextPlayer, selfKomi, row + NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES);

//   // auto inputsMap = inputs.tensor<float, 3>();
//   // for(int i = 0; i<NNPos::MAX_BOARD_AREA; i++) {
//   //   for(int j = 0; j<NNInputs::NUM_FEATURES; j++) {
//   //     inputsMap(0,i,j) = row[i*NNInputs::NUM_FEATURES + j];
//   //   }
//   // }

//   cout << "BBBB" << endl;


//   auto symmetriesMap = symmetries.tensor<bool, 1>();
//   symmetriesMap(0) = false;
//   symmetriesMap(1) = false;
//   symmetriesMap(2) = false;

//   auto isTrainingMap = isTraining.tensor<bool, 0>();
//   isTrainingMap(0) = false;

//   vector<pair<string,Tensor>> inputsList = {
//     {"inputs",inputs},
//     {"symmetries",symmetries},
//     {"is_training",isTraining},
//   };

//   vector<Tensor> outputs;

//   cout << "CCCC" << endl;

//   status = session->Run(inputsList, {"policy_output","value_output"}, {}, &outputs);
//   checkStatus(status,"running inference");
//   assert(outputs.size() == 2);

//   cout << "DDDD" << endl;

//   assert(outputs[0].dims() == 2);
//   assert(outputs[1].dims() == 1);
//   assert(outputs[0].dim_size(0) == 2); //batch
//   assert(outputs[0].dim_size(1) == NNPos::NN_POLICY_SIZE);
//   assert(outputs[1].dim_size(0) == 2); //batch

//   auto policyMap = outputs[0].matrix<float>();
//   auto valueMap = outputs[1].vec<float>();

//   for(int batch = 0; batch < 2; batch++) {
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

//   // float row[NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES];
//   for(int i = 0; i<2*NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES; i++)
//     row[i] = 0.0f;

//   cout << "EEEE" << endl;
//   NNInputs::fillRow(board, moveHistory, moveHistoryLen, nextPlayer, selfKomi, row);

//   nextPlayer = P_BLACK;
//   selfKomi = -7.5f;
//   board.playMove(Location::getLoc(3,3,board.x_size),P_WHITE);
//   NNInputs::fillRow(board, moveHistory, moveHistoryLen, nextPlayer, selfKomi, row + NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES);

//   status = session->Run(inputsList, {"policy_output","value_output"}, {}, &outputs);
//   checkStatus(status,"running inference");
//   assert(outputs.size() == 2);

//   cout << "DDDD" << endl;

//   assert(outputs[0].dims() == 2);
//   assert(outputs[1].dims() == 1);
//   assert(outputs[0].dim_size(0) == 2); //batch
//   assert(outputs[0].dim_size(1) == NNPos::NN_POLICY_SIZE);
//   assert(outputs[1].dim_size(0) == 2); //batch

//   policyMap = outputs[0].matrix<float>();
//   valueMap = outputs[1].vec<float>();

//   for(int batch = 0; batch < 2; batch++) {
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
