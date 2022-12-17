#ifndef coremlbackend_h
#define coremlbackend_h

struct CoreMLLoadedModel {
  int modelXLen;
  int modelYLen;
  ModelDesc modelDesc;

  CoreMLLoadedModel();
  CoreMLLoadedModel(const CoreMLLoadedModel&) = delete;
  CoreMLLoadedModel& operator=(const CoreMLLoadedModel&) = delete;
};

struct CoreMLComputeHandle {
  int nnXLen;
  int nnYLen;
  int modelXLen;
  int modelYLen;
  bool inputsUseNHWC;
  int version;
  int gpuIndex;
  bool isCoreML;

  CoreMLComputeHandle(const CoreMLLoadedModel* loadedModel,
                      int nnXLen,
                      int nnYLen,
                      int gpuIdx,
                      bool inputsNHWC,
                      int serverThreadIdx,
                      bool useFP16);
  
  CoreMLComputeHandle() = delete;
  CoreMLComputeHandle(const CoreMLComputeHandle&) = delete;
  CoreMLComputeHandle& operator=(const CoreMLComputeHandle&) = delete;
};

struct CoreMLInputBuffers {
  int maxBatchSize;
  int modelXLen;
  int modelYLen;

  size_t policyResultChannels;

  size_t singleSpatialElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singlePolicyResultElts;
  size_t singlePolicyProbsElts;
  size_t singleValueResultElts;
  size_t singleOwnershipResultElts;
  size_t singleOwnerMapElts;
  size_t singleMiscValuesResultElts;
  size_t singleMoreMiscValuesResultElts;

  size_t rowSpatialBufferElts;
  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t policyResultBufferElts;
  size_t policyProbsBufferElts;
  size_t valueResultBufferElts;
  size_t ownershipResultBufferElts;
  size_t ownerMapBufferElts;
  size_t miscValuesResultBufferElts;
  size_t moreMiscValuesResultsBufferElts;

  float* rowSpatialBuffer;
  float* userInputBuffer;        // Host pointer
  float* userInputGlobalBuffer;  // Host pointer

  float* policyResults;
  float* policyProbsBuffer;
  float* valueResults;
  float* ownershipResults;
  float* ownerMapBuffer;
  float* miscValuesResults;
  float* moreMiscValuesResults;

  CoreMLInputBuffers(const CoreMLLoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen);

  ~CoreMLInputBuffers() {
    delete[] rowSpatialBuffer;
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] policyResults;
    delete[] policyProbsBuffer;
    delete[] valueResults;
    delete[] ownershipResults;
    delete[] ownerMapBuffer;
    delete[] miscValuesResults;
    delete[] moreMiscValuesResults;
  }

  CoreMLInputBuffers() = delete;
  CoreMLInputBuffers(const CoreMLInputBuffers&) = delete;
  CoreMLInputBuffers& operator=(const CoreMLInputBuffers&) = delete;
};

void initCoreMLBackends();

int createCoreMLBackend(int modelIndex,
                        int modelXLen,
                        int modelYLen,
                        int serverThreadIdx,
                        bool useFP16);

void freeCoreMLBackend(int modelIndex);
int getCoreMLBackendNumSpatialFeatures(int modelIndex);
int getCoreMLBackendNumGlobalFeatures(int modelIndex);

void getCoreMLBackendOutput(float* userInputBuffer,
                            float* userInputGlobalBuffer,
                            float* policyOutput,
                            float* valueOutput,
                            float* ownershipOutput,
                            float* miscValuesOutput,
                            float* moreMiscValuesOutput,
                            int modelIndex);

void getCoreMLHandleOutput(CoreMLComputeHandle* gpuHandle,
                           CoreMLInputBuffers* inputBuffers,
                           int numBatchEltsFilled,
                           NNResultBuf** inputBufs,
                           std::vector<NNOutput*>& outputs);

#endif /* coremlbackend_h */
