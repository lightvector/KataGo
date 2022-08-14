#import <Foundation/Foundation.h>
#import <CoreML/MLMultiArray.h>
#import "katago-Swift.h"

void getCoreMLBackendOutput(float* userInputBuffer, float* userInputGlobalBuffer, float* policyResults) {
    NSError *error = nil;

    [[CoreMLBackend shared] getOutputWithBinInputs: userInputBuffer globalInputs: userInputGlobalBuffer policyOutput: policyResults error: &error];
}
