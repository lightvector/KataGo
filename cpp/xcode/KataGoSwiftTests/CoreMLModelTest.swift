//
//  CoreMLModelTest.swift
//  KataGoSwiftTests
//
//  Created by Chin-Chang Yang on 2024/7/20.
//

import XCTest

final class CoreMLModelTest: XCTestCase {
    func testFreshCompileBundleMLModel() {
        let modelName = CoreMLBackend.getModelName()

        let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName,
                                                       computeUnits: .cpuAndNeuralEngine)

        XCTAssertNotNil(mlmodel)
    }
}
