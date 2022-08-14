#import <Foundation/Foundation.h>
#import <CoreML/MLMultiArray.h>
#import "katago-Swift.h"

void getCoreMLBackendOutput(float* userInputBuffer, float* userInputGlobalBuffer, float* policyResults) {
    NSError *error = nil;

    [[CoreMLBackend shared] getOutputWithBin_inputs: userInputBuffer global_inputs: userInputGlobalBuffer policy_output: policyResults error: &error];
}
