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
        let permanentURL = try! KataGoModel.getMLModelCPermanentURL(modelName: modelName)
        let savedDigestURL = try! KataGoModel.getSavedDigestURL(modelName: modelName)
        try! FileManager.default.removeItem(at: permanentURL)
        try! FileManager.default.removeItem(at: savedDigestURL)

        let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName,
                                                       computeUnits: .cpuAndNeuralEngine)

        XCTAssertNotNil(mlmodel)
    }

    func testCompileBundleMLModelWhenOldMLModelNotExists() {
        let modelName = CoreMLBackend.getModelName()

        _ = KataGoModel.compileBundleMLModel(modelName: modelName,
                                             computeUnits: .cpuAndNeuralEngine)

        let permanentURL = try! KataGoModel.getMLModelCPermanentURL(modelName: modelName)
        try! FileManager.default.removeItem(at: permanentURL)

        let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName,
                                                       computeUnits: .cpuAndNeuralEngine)

        XCTAssertNotNil(mlmodel)
    }

    func testCompileBundleMLModelWhenDigestChanges() {
        let modelName = CoreMLBackend.getModelName()

        _ = KataGoModel.compileBundleMLModel(modelName: modelName,
                                             computeUnits: .cpuAndNeuralEngine)

        let savedDigestURL = try! KataGoModel.getSavedDigestURL(modelName: modelName)
        try! "".write(to: savedDigestURL, atomically: true, encoding: .utf8)

        let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName,
                                                       computeUnits: .cpuAndNeuralEngine)

        XCTAssertNotNil(mlmodel)
    }
}
