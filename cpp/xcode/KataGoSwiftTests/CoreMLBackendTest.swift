//
//  CoreMLBackendTest.swift
//  KataGoSwiftTests
//
//  Created by Chin-Chang Yang on 2024/7/20.
//

import XCTest

final class CoreMLBackendTest: XCTestCase {
    
    func testNilCoreMLBackend() {
        let backend = maybeCreateCoreMLBackend(xLen: 1,
                                               yLen: 1)

        XCTAssertNil(backend)
    }

    func testCoreMLBackendMeta() {
        let backend = maybeCreateCoreMLBackend(metaEncoderVersion: 1)!
        var binInputs = [Float32](repeating: 1, count: backend.spatialSize)
        var globalInputs = [Float32](repeating: 1, count: backend.numGlobalFeatures)
        var metaInputs = [Float32](repeating: 1, count: backend.numMetaFeatures)
        // See the contents in Predictions tab of a mlpackage file
        let policyOutputsSize = 1 * 2 * 362
        let valueOutputsSize = 1 * 3
        let ownershipOutputsSize = 1 * 1 * 19 * 19
        let miscValuesOutputsSize = 1 * 10
        let moreMiscValuesOutputsSize = 1 * 8
        var policyOutputs = [Float32](repeating: 1, count: policyOutputsSize)
        var valueOutputs = [Float32](repeating: 1, count: valueOutputsSize)
        var ownershipOutputs = [Float32](repeating: 1, count: ownershipOutputsSize)
        var miscValuesOutputs = [Float32](repeating: 1, count: miscValuesOutputsSize)
        var moreMiscValuesOutputs = [Float32](repeating: 1, count: moreMiscValuesOutputsSize)
        let batchSize = 1

        backend.getBatchOutput(binInputs: &binInputs,
                               globalInputs: &globalInputs,
                               metaInputs: &metaInputs,
                               policyOutputs: &policyOutputs,
                               valueOutputs: &valueOutputs,
                               ownershipOutputs: &ownershipOutputs,
                               miscValuesOutputs: &miscValuesOutputs,
                               moreMiscValuesOutputs: &moreMiscValuesOutputs,
                               batchSize: batchSize)

        XCTAssertEqual(policyOutputs[0], -14.865191, accuracy: 1e-8)
        XCTAssertEqual(policyOutputs[policyOutputsSize - 1], -4.618183, accuracy: 1e-8)
        XCTAssertEqual(valueOutputs[0], -2.6804342, accuracy: 1e-8)
        XCTAssertEqual(valueOutputs[valueOutputsSize - 1], -10.766362, accuracy: 1e-8)
        XCTAssertEqual(ownershipOutputs[0], -0.057577543, accuracy: 1e-8)
        XCTAssertEqual(ownershipOutputs[ownershipOutputsSize - 1], -0.08216003, accuracy: 1e-8)
        XCTAssertEqual(miscValuesOutputs[0], -15.050249, accuracy: 1e-8)
        XCTAssertEqual(miscValuesOutputs[miscValuesOutputsSize - 1], -8.116829, accuracy: 1e-8)
        XCTAssertEqual(moreMiscValuesOutputs[0], -4.3661594, accuracy: 1e-8)
        XCTAssertEqual(moreMiscValuesOutputs[moreMiscValuesOutputsSize - 1], -20.357855, accuracy: 1e-8)
    }
}

