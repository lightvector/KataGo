#import <Foundation/Foundation.h>
#import <CoreML/MLMultiArray.h>
#import "katago-Swift.h"

void getCoreMLBackendOutput(float* userInputBuffer, float* userInputGlobalBuffer, float* policyOutput, float* valueOutput, float* ownershipOutput, float* miscValuesOutput, float* moreMiscValuesOutput) {
    NSError *error = nil;

    [[CoreMLBackend shared] getOutputWithBinInputs: userInputBuffer globalInputs: userInputGlobalBuffer policyOutput: policyOutput valueOutput: valueOutput ownershipOutput: ownershipOutput miscValuesOutput: miscValuesOutput moreMiscValuesOutput: moreMiscValuesOutput error: &error];
}
