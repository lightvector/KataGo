
//TODO remove this define
#define USE_TENSORFLOW_BACKEND
#ifdef USE_TENSORFLOW_BACKEND

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include "../neuralnet/nninterface.h"

static tensorflow::Session* globalSession = NULL;

static void checkStatus(const tensorflow::Status& status, const char* subLabel) {
  if(!status.ok())
    throw StringError("NN Eval Error: " + string(subLabel) + status.ToString());
}

void NeuralNet::globalInitialize(
  const string& tensorflowGpuVisibleDeviceList,
  double tensorflowPerProcessGpuMemoryFraction
) {
  tensorflow::Status status;
  tensorflow::SessionOptions sessionOptions = tensorflow::SessionOptions();
  if(tensorflowGpuVisibleDeviceList.length() > 0)
    sessionOptions.config.mutable_gpu_options()->set_visible_device_list(tensorflowGpuVisibleDeviceList);
  if(tensorflowPerProcessGpuMemoryFraction >= 0.0)
    sessionOptions.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(tensorflowPerProcessGpuMemoryFraction);
  status = NewSession(sessionOptions, &globalSession);
  checkStatus(status,"creating session");
  assert(globalSession != NULL);
}

void NeuralNet::globalCleanup() {
  if(globalSession != NULL)
    globalSession->Close();
  globalSession = NULL;
}


struct LocalGpuHandle {
  //Empty for tensorflow
};
struct LoadedModel {
  string graphPrefix;
  tensorflow::GraphDef* graphDef;
};
struct InputBuffers {
  vector<string> outputNames;
  vector<string> targetNames;

  float* inputsBuffer;
  bool* symmetriesBuffer;
  vector<pair<string,tensorflow::Tensor>> inputsList;
  vector<tensorflow::Tensor> outputsBuf;

  vector<pair<string,tensorflow::Tensor>> slicedInputsList;
};


static void addPrefixToGraph(tensorflow::GraphDef* graphDef, const string& prefix) {
  //string device = "/gpu:0";
  for(int i = 0; i < graphDef->node_size(); ++i)
  {
    auto node = graphDef->mutable_node(i);
    //node->set_device(device);
    string* name = node->mutable_name();
    *name = prefix + *name;
    int inputSize = node->input_size();
    for(int j = 0; j<inputSize; j++) {
      string* inputName = node->mutable_input(j);
      if(inputName->size() > 0 && (*inputName)[0] == '^')
        *inputName = "^" + prefix + inputName->substr(1);
      else
        *inputName = prefix + *inputName;
    }
  }
};


LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  assert(globalSession != NULL);
  tensorflow::Status status;

  LoadedModel* loadedModel = new LoadedModel();
  loadedModel->graphPrefix = "m" + Global::intToString(modelFileIdx);
  loadedModel->graphDef = new tensorflow::GraphDef();

  //Read graph from file
  status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), file, loadedModel->graphDef);
  checkStatus(status,"reading binary proto graph from file");
  addPrefixToGraph(loadedModel->graphDef,loadedModel->graphPrefix);

  //Add graph to session
  status = globalSession->Extend(*(loadedModel->graphDef));
  checkStatus(status,"adding graph to session");

  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel->graphDef;
  delete loadedModel;
}

LocalGpuHandle* NeuralNet::createLocalGpuHandle(LoadedModel* loadedModel, int maxBatchSize, int cudaGpuIdxForThisThread) {
  assert(globalSession != NULL);
  (void)loadedModel;
  (void)maxBatchSize;
  (void)cudaGpuIdxForThisThread;
  return NULL;
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
  (void)gpuHandle;
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize) {
  assert(globalSession != NULL);
  tensorflow::Status status;

  InputBuffers* buffers = new InputBuffers();

  buffers->outputNames = {
    loadedModel->graphPrefix + "policy_output",
    loadedModel->graphPrefix + "value_output"
  };
  buffers->targetNames = {};

  //Set up inputs
  tensorflow::TensorShape inputsShape;
  tensorflow::TensorShape symmetriesShape;
  tensorflow::TensorShape isTrainingShape;
  int inputsShapeArr[3] = {maxBatchSize,NNPos::MAX_BOARD_AREA,NNInputs::NUM_FEATURES_V1};
  status = tensorflow::TensorShapeUtils::MakeShape(inputsShapeArr,3,&inputsShape);
  checkStatus(status,"making inputs shape");
  int symmetriesShapeArr[1] = {NNInputs::NUM_SYMMETRY_BOOLS};
  status = tensorflow::TensorShapeUtils::MakeShape(symmetriesShapeArr,1,&symmetriesShape);
  checkStatus(status,"making symmetries shape");
  int isTrainingShapeArr[0] = {};
  status = tensorflow::TensorShapeUtils::MakeShape(isTrainingShapeArr,0,&isTrainingShape);
  checkStatus(status,"making isTraining shape");

  tensorflow::Tensor inputs(tensorflow::DT_FLOAT,inputsShape);
  tensorflow::Tensor symmetries(tensorflow::DT_BOOL,symmetriesShape);
  tensorflow::Tensor isTraining(tensorflow::DT_BOOL,isTrainingShape);

  assert(inputs.IsAligned());
  assert(symmetries.IsAligned());

  buffers->inputsBuffer = inputs.flat<float>().data();
  buffers->symmetriesBuffer = symmetries.flat<bool>().data();
  auto isTrainingMap = isTraining.tensor<bool, 0>();
  isTrainingMap(0) = false;

  buffers->inputsList = {
    {loadedModel->graphPrefix+"inputs",inputs},
    {loadedModel->graphPrefix+"symmetries",symmetries},
    {loadedModel->graphPrefix+"is_training",isTraining},
  };
  buffers->outputsBuf = vector<tensorflow::Tensor>();

  return buffers;
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  //Simply null these - these are direct pointers into the inputs and symmetries tensor
  //and are invalid once inputList is cleared and those are freed
  buffers->inputsBuffer = NULL;
  buffers->symmetriesBuffer = NULL;

  buffers->outputNames.clear();
  buffers->targetNames.clear();
  buffers->inputsList.clear();
  buffers->outputsBuf.clear();
  delete buffers;
}

float* NeuralNet::getRowInplace(InputBuffers* buffers, int rowIdx) {
  return buffers->inputsBuffer + rowIdx * NNInputs::ROW_SIZE_V1;
}
bool* NeuralNet::getSymmetriesInplace(InputBuffers* buffers) {
  return buffers->symmetriesBuffer;
}

void NeuralNet::getOutput(LocalGpuHandle* gpuHandle, InputBuffers* buffers, int numFilledRows, vector<NNOutput*>& outputs) {
  (void)gpuHandle;
  tensorflow::Status status;

  buffers->slicedInputsList = buffers->inputsList;
  buffers->slicedInputsList[0].second = buffers->inputsList[0].second.Slice(0,numFilledRows);

  status = globalSession->Run(buffers->slicedInputsList, buffers->outputNames, buffers->targetNames, &(buffers->outputsBuf));
  checkStatus(status,"running inference");

  assert(buffers->outputsBuf.size() == 2);
  assert(buffers->outputsBuf[0].dims() == 2);
  assert(buffers->outputsBuf[1].dims() == 1);
  assert(buffers->outputsBuf[0].dim_size(0) == numFilledRows);
  assert(buffers->outputsBuf[0].dim_size(1) == NNPos::NN_POLICY_SIZE);
  assert(buffers->outputsBuf[1].dim_size(0) == numFilledRows);

  assert(buffers->outputsBuf[0].IsAligned());
  assert(buffers->outputsBuf[1].IsAligned());

  float* policyData = buffers->outputsBuf[0].flat<float>().data();
  float* valueData = buffers->outputsBuf[1].flat<float>().data();

  outputs.clear();

  for(int row = 0; row < numFilledRows; row++) {
    NNOutput* output = new NNOutput();
    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //probabilities and white value
    //Also we don't fill in the nnHash here either
    std::copy(
      policyData + row * NNPos::NN_POLICY_SIZE,
      policyData + (row+1) * NNPos::NN_POLICY_SIZE,
      policyProbs
    );
    output->whiteValue = valueData[row];
    outputs.push_back(output);
  }

  buffers->outputsBuf.clear();
}

#endif

