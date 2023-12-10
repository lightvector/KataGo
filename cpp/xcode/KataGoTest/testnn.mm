//
//  testc.m
//  testc
//
//  Created by Chin-Chang Yang on 2023/11/5.
//

#import <XCTest/XCTest.h>
#import "../main.h"

@interface TestNN : XCTestCase

@end

@implementation TestNN

- (void)testNNLayer {
    std::vector<std::string> args;
    MainCmds::runnnlayertests(args);
}

- (void)testOutput {
    std::vector<std::string> args;
    MainCmds::runoutputtests(args);
}

- (void)testNNOnTinyBoard {
    std::vector<std::string> args;
    args.push_back("katago");
    args.push_back("model.bin.gz");
    args.push_back("false");
    args.push_back("false");
    args.push_back("0");
    args.push_back("false");
    MainCmds::runnnontinyboardtest(args);
}

- (void)testNNOnTinyBoardFp16 {
    std::vector<std::string> args;
    args.push_back("katago");
    args.push_back("model.bin.gz");
    args.push_back("false");
    args.push_back("false");
    args.push_back("0");
    args.push_back("true");
    MainCmds::runnnontinyboardtest(args);
}

- (void)testNNSymmetries {
    std::vector<std::string> args;
    args.push_back("katago");
    args.push_back("model.bin.gz");
    args.push_back("false");
    args.push_back("false");
    args.push_back("false");
    MainCmds::runnnsymmetriestest(args);
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

- (void)testGpuError {
    std::vector<std::string> args;
    args.push_back("katago");
    args.push_back("-config");
    args.push_back("gtp.cfg");
    args.push_back("-model");
    args.push_back("model.bin.gz");
    args.push_back("-boardsize");
    args.push_back("9");
    args.push_back("-quick");
    MainCmds::testgpuerror(args);
}

@end
