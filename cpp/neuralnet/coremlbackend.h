#ifndef coremlbackend_h
#define coremlbackend_h

void initCoreMLBackends();
int createCoreMLBackend(int modelIndex, int modelXLen, int modelYLen);
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
