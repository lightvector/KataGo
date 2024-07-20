//
//  testc.m
//  testc
//
//  Created by Chin-Chang Yang on 2023/11/5.
//

#import <XCTest/XCTest.h>
#import "../neuralnet/nninterface.h"
#import "../main.h"

@interface TestNN : XCTestCase

@end

@implementation TestNN

- (void)testNNLayer {
    std::vector<std::string> args;
    MainCmds::runnnlayertests(args);
}

- (void)testOwnership {
    std::vector<std::string> args;
    args.push_back("katago");
    args.push_back("gtp.cfg");
    args.push_back("model.bin.gz");
    // Create new CoreML files
    MainCmds::runownershiptests(args);
    // Reuse the CoreML files
    MainCmds::runownershiptests(args);
}

- (void)testOwnershipV8 {
    std::vector<std::string> args;
    args.push_back("katago");
    args.push_back("metal_gtp.cfg");
    args.push_back("modelv8.bin.gz");
    MainCmds::runownershiptests(args);
}

- (void)testPrintDevices {
    NeuralNet::printDevices();
}

@end
