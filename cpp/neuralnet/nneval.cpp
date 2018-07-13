
#include "../neuralnet/nneval.h"

using namespace tensorflow;

NNOutput::NNOutput() {}
NNOutput::NNOutput(const NNOutput& other) {
  whiteValue = other.whiteValue;
  std::copy(other.policyProbs, other.policyProbs+NNPos::NN_POLICY_SIZE, policyProbs);
}

double NNOutput::whiteValueOfWinner(Player winner) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return -1.0;
  return 0.0;
}

double NNOutput::whiteValueOfScore(double finalWhiteMinusBlackScore, int bSize) {
  return tanh(finalWhiteMinusBlackScore / (bSize*2));
}


static const int BATCH_SIZE = 1;

static void checkStatus(Status status, const char* subLabel) {
  if(!status.ok())
    throw StringError("NN Eval Error: " + string(subLabel) + status.ToString());
}

NNEvaluator::NNEvaluator(const string& pbModelFile)
{
  Status status;
  GraphDef graphDef;

  //Create session
  status = NewSession(SessionOptions(), &session);
  checkStatus(status,"creating session");

  //Read graph from file
  status = ReadBinaryProto(Env::Default(), pbModelFile, &graphDef);
  checkStatus(status,"reading graph");

  //Add graph to session
  status = session->Create(graphDef);
  checkStatus(status,"adding graph to session");

  //Set up inputs
  TensorShape inputsShape;
  TensorShape symmetriesShape;
  TensorShape isTrainingShape;
  int inputsShapeArr[3] = {BATCH_SIZE,NNPos::MAX_BOARD_AREA,NNInputs::NUM_FEATURES_V1};
  status = TensorShapeUtils::MakeShape(inputsShapeArr,3,&inputsShape);
  checkStatus(status,"making inputs shape");
  int symmetriesShapeArr[1] = {NNInputs::NUM_SYMMETRY_BOOLS};
  status = TensorShapeUtils::MakeShape(symmetriesShapeArr,1,&symmetriesShape);
  checkStatus(status,"making symmetries shape");
  int isTrainingShapeArr[0] = {};
  status = TensorShapeUtils::MakeShape(isTrainingShapeArr,0,&isTrainingShape);
  checkStatus(status,"making isTraining shape");

  Tensor inputs(DT_FLOAT,inputsShape);
  Tensor symmetries(DT_BOOL,symmetriesShape);
  Tensor isTraining(DT_BOOL,isTrainingShape);

  inputsList = {
    {"inputs",inputs},
    {"symmetries",symmetries},
    {"is_training",isTraining},
  };

  outputNames = {
    string("policy_output"),
    string("value_output")
  };
  targetNames = {};

  inputsBuffer = inputs.flat<float>().data();
  symmetriesBuffer = symmetries.flat<bool>().data();

  auto isTrainingMap = isTraining.tensor<bool, 0>();
  isTrainingMap(0) = false;
}

NNEvaluator::~NNEvaluator()
{
  //Clear these out - these are direct pointers into the inputs and symmetries tensor
  //and are invalid once inputList is cleared and those are freed
  inputsBuffer = NULL;
  symmetriesBuffer = NULL;

  //Explictly clean up tensors - their destructors should get called.
  inputsList.clear();
  outputsBuf.clear();

  session->Close();
  session = NULL;
}

shared_ptr<NNOutput> NNEvaluator::evaluate(
  Board& board, const BoardHistory& history, Player nextPlayer, int symmetry
) {
  shared_ptr<NNOutput> nnOutput = std::make_shared<NNOutput>();
  outputsBuf.clear();

  int rowSize = NNPos::MAX_BOARD_AREA * NNInputs::NUM_FEATURES_V1;
  int bufferSize = rowSize * BATCH_SIZE;

  std::fill(inputsBuffer,inputsBuffer+bufferSize,0.0f);

  //TODO send this for batching to another thread? Probably would do so by synchronizedly
  //acquiring a buffer from that thread to be filled that doesn't conflict with threads
  //filling other entries for the same batch

  int batch = 0;

  NNInputs::fillRowV1(board, history, nextPlayer, inputsBuffer+batch*rowSize);

  assert(symmetry >= 0 && symmetry <= NUM_SYMMETRIES);
  symmetriesBuffer[0] = (symmetry & 0x1) != 0;
  symmetriesBuffer[1] = (symmetry & 0x2) != 0;
  symmetriesBuffer[2] = (symmetry & 0x4) != 0;

  Status status;
  status = session->Run(inputsList, outputNames, targetNames, &outputsBuf);
  checkStatus(status,"running inference");

  assert(outputsBuf.size() == 2);
  assert(outputsBuf[0].dims() == 2);
  assert(outputsBuf[1].dims() == 1);
  assert(outputsBuf[0].dim_size(0) == BATCH_SIZE);
  assert(outputsBuf[0].dim_size(1) == NNPos::NN_POLICY_SIZE);
  assert(outputsBuf[1].dim_size(0) == BATCH_SIZE);

  auto policyMap = outputsBuf[0].matrix<float>();
  auto valueMap = outputsBuf[1].vec<float>();

  assert(board.x_size == board.y_size);
  int bSize = board.x_size;
  int offset = NNPos::getOffset(bSize);

  float* policy = nnOutput->policyProbs;
  float maxPolicy = -1e25f;

  bool isLegal[NNPos::NN_POLICY_SIZE];
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
    Loc loc = NNPos::posToLoc(i,bSize,offset);
    isLegal[i] = history.isLegal(board,loc,nextPlayer);

    float policyValue = isLegal[i] ? policyMap(batch,i) : -1e30f;

    policy[i] = policyValue;
    if(policyValue > maxPolicy)
      maxPolicy = policyValue;
  }

  float policySum = 0.0f;
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
    policy[i] = exp(policy[i] - maxPolicy);
    policySum += policy[i];
  }

  //Somehow all legal moves rounded to 0 probability
  if(policySum <= 0.0)
    throw StringError("NN Eval Error: Policy all rounded to 0.0");

  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++)
    policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;

  if(nextPlayer == P_WHITE)
    nnOutput->whiteValue = tanh(valueMap(batch));
  else
    nnOutput->whiteValue = -tanh(valueMap(batch));

  return nnOutput;
}
