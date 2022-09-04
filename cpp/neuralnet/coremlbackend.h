#ifndef coremlbackend_h
#define coremlbackend_h

void initCoreMLBackend(int modelIndex);
void resetCoreMLBackend(int modelIndex);

void getCoreMLBackendOutput(float* userInputBuffer,
                            float* userInputGlobalBuffer,
                            float* policyOutput,
                            float* valueOutput,
                            float* ownershipOutput,
                            float* miscValuesOutput,
                            float* moreMiscValuesOutput,
                            int modelIndex);

#endif /* coremlbackend_h */
