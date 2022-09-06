#ifndef coremlbackend_h
#define coremlbackend_h

void* createCoreMLModel(int modelXLen, int modelYLen);
void freeCoreMLModel(void* context);
void createCoreMLBackend(void* coreMLContext, int modelIndex, int modelXLen, int modelYLen);
void freeCoreMLBackend(int modelIndex);

void getCoreMLBackendOutput(float* userInputBuffer,
                            float* userInputGlobalBuffer,
                            float* policyOutput,
                            float* valueOutput,
                            float* ownershipOutput,
                            float* miscValuesOutput,
                            float* moreMiscValuesOutput,
                            int modelIndex);

#endif /* coremlbackend_h */
