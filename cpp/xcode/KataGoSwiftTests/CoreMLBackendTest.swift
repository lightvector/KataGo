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
        let backend = maybeCreateCoreMLBackend(metaEncoderVersion: 1,
                                               useCpuAndNeuralEngine: false)!

        checkBackendOutput(backend: backend)
    }

    func testCoreMLBackendMetaNE() {
        let backend = maybeCreateCoreMLBackend(metaEncoderVersion: 1,
                                               useCpuAndNeuralEngine: true)!

        checkBackendOutput(backend: backend)
    }

    func checkBackendOutput(backend: CoreMLBackend) {
        var binInputs = [Float32](repeating: 1, count: backend.spatialSize)
        var globalInputs = [Float32](repeating: 1, count: backend.numGlobalFeatures)
        var metaInputs = [Float32](repeating: 1, count: backend.numMetaFeatures)
        // See the contents in Predictions tab of a mlpackage file
        let policyOutputsSize = 1 * 6 * 362
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

        XCTAssertEqual(policyOutputs[0], -14.86533, accuracy: 1e-3)
        XCTAssertEqual(policyOutputs[policyOutputsSize - 1], -4.618265, accuracy: 1e-3)
        XCTAssertEqual(valueOutputs[0], -2.6803048, accuracy: 1e-3)
        XCTAssertEqual(valueOutputs[valueOutputsSize - 1], -10.766384, accuracy: 1e-3)
        XCTAssertEqual(ownershipOutputs[0], -0.05757516, accuracy: 1e-3)
        XCTAssertEqual(ownershipOutputs[ownershipOutputsSize - 1], -0.08216501, accuracy: 1e-3)
        XCTAssertEqual(miscValuesOutputs[0], -15.050129, accuracy: 1e-3)
        XCTAssertEqual(miscValuesOutputs[miscValuesOutputsSize - 1], -8.116809, accuracy: 1e-3)
        XCTAssertEqual(moreMiscValuesOutputs[0], -4.365787, accuracy: 1e-3)
        XCTAssertEqual(moreMiscValuesOutputs[moreMiscValuesOutputsSize - 1], -20.357615, accuracy: 1e-3)

    }
}
