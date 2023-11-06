//
//  testc.m
//  testc
//
//  Created by Chin-Chang Yang on 2023/11/5.
//

#import <XCTest/XCTest.h>
#import "../tests/tests.h"

@interface TestNN : XCTestCase

@end

@implementation TestNN

- (void)testNNLayer {
    Tests::runNNLayerTests();
}

- (void)testNNSymmetry {
    Tests::runNNSymmetryTests();
}

@end
