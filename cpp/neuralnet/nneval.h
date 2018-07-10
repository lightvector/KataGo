#ifndef NNEVAL_H
#define NNEVAL_H

#include <memory>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

using tensorflow::Tensor;
using tensorflow::Session;

#include "../core/global.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"

struct NNOutput {
  //From the perspective of the player to move at the time of the eval
  float value;

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  float policyProbs[NNPos::NN_POLICY_SIZE];
};

class NNEvaluator {
 public:
  static const int NUM_SYMMETRIES = 8;

  NNEvaluator(const string& pbModelFile);
  ~NNEvaluator();

  //TODO not thread-safe right now!
  shared_ptr<NNOutput> evaluate(
    Board& board, const BoardHistory& history, Player nextPlayer, int symmetry
  );

 private:

  Session* session;
  vector<pair<string,Tensor>> inputsList;
  vector<string> outputNames;
  vector<string> targetNames;
  vector<Tensor> outputsBuf;

  float* inputsBuffer;
  bool* symmetriesBuffer;
};

#endif
