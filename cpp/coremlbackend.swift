//
//  coremlbackend.swift
//  KataGo
//
//  Created by Chin-Chang Yang on 2023/11/8.
//

import Foundation
import CoreML
import OSLog

class CoreMLBackend {
    private static var backends: [Int: CoreMLBackend] = [:]
    private static var modelIndex: Int = -1

    class func reserveBackends() {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        if backends.isEmpty {
            backends.reserveCapacity(2)
        }
    }

    class func clearBackends() {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        backends.removeAll()
    }

    class func getNextModelIndex() -> Int {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        // The next CoreMLBackend index is the current index + 1.
        modelIndex = modelIndex + 1

        // The CoreMLBackend index is returned.
        return modelIndex;
    }

    class func getBackend(at index: Int) -> CoreMLBackend? {
        return backends[index]
    }

    class func getModelName(useFP16: Bool) -> String {
        let COMPILE_MAX_BOARD_LEN = 19
        let precision = useFP16 ? 16 : 32
        return "KataGoModel\(COMPILE_MAX_BOARD_LEN)x\(COMPILE_MAX_BOARD_LEN)fp\(precision)"
    }

    class func createInstance(xLen: Int, yLen: Int, useFP16: Bool) -> Int {
        // The next ML model index is retrieved.
        let modelIndex = getNextModelIndex()

        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        // Get the model name.
        let modelName = getModelName(useFP16: useFP16)

        // Compile the model in Bundle.
        if let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName) {
            // The CoreMLBackend object is created.
            backends[modelIndex] = CoreMLBackend(model: mlmodel, xLen: xLen, yLen: yLen)
        } else {
            fatalError("Unable to compile bundle MLModel from model: \(modelName)")
        }

        // The ML model index is returned.
        return modelIndex;
    }

    class func destroyInstance(index: Int) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        backends[index] = nil
    }

    let model: KataGoModel
    let xLen: Int
    let yLen: Int
    let version: Int
    let numSpatialFeatures: Int
    let numGlobalFeatures: Int

    init(model: MLModel, xLen: Int, yLen: Int) {
        self.model = KataGoModel(model: model)
        self.xLen = xLen
        self.yLen = yLen

        // The model version must be at least 8.
        if let versionString = model.modelDescription.metadata[MLModelMetadataKey.versionString] as? String {
            if let versionInt = Int(versionString) {
                self.version = versionInt
            } else {
                self.version = -1
            }
        } else {
            self.version = -1
        }

        assert(self.version >= 8, "version must not be smaller than 8: \(self.version)")

        // The number of spatial features must be 22.
        self.numSpatialFeatures = 22

        // The number of global features must be 19.
        self.numGlobalFeatures = 19
    }

    func getBatchOutput(binInputs: UnsafeMutablePointer<Float32>,
                        globalInputs: UnsafeMutablePointer<Float32>,
                        policyOutputs: UnsafeMutablePointer<Float32>,
                        valueOutputs: UnsafeMutablePointer<Float32>,
                        ownershipOutputs: UnsafeMutablePointer<Float32>,
                        miscValuesOutputs: UnsafeMutablePointer<Float32>,
                        moreMiscValuesOutputs: UnsafeMutablePointer<Float32>,
                        batchSize: Int) {

        autoreleasepool {
            do {
                let spatialStrides = [numSpatialFeatures * yLen * xLen,
                                      yLen * xLen,
                                      xLen,
                                      1] as [NSNumber]

                let globalStrides = [numGlobalFeatures, 1] as [NSNumber]
                let spatialSize = numSpatialFeatures * yLen * xLen

                let inputArray = try (0..<batchSize).map { index -> KataGoModelInput in
                    let binInputsArray = try MLMultiArray(
                        dataPointer: binInputs.advanced(by: index * spatialSize),
                        shape: [1, numSpatialFeatures, yLen, xLen] as [NSNumber],
                        dataType: .float,
                        strides: spatialStrides)

                    let globalInputsArray = try MLMultiArray(
                        dataPointer: globalInputs.advanced(by: index * numGlobalFeatures),
                        shape: [1, numGlobalFeatures] as [NSNumber],
                        dataType: .float,
                        strides: globalStrides)

                    return KataGoModelInput(input_spatial: binInputsArray, input_global: globalInputsArray)
                }

                let inputBatch = KataGoModelInputBatch(inputArray: inputArray)
                let options = MLPredictionOptions()
                let outputBatch = try model.prediction(from: inputBatch, options: options)

                outputBatch.outputArray.enumerated().forEach { index, output in
                    let policyOutputBase = policyOutputs.advanced(by: index * output.output_policy.count)
                    let valueOutputBase = valueOutputs.advanced(by: index * output.out_value.count)
                    let ownershipOutputBase = ownershipOutputs.advanced(by: index * output.out_ownership.count)
                    let miscValuesOutputBase = miscValuesOutputs.advanced(by: index * output.out_miscvalue.count)
                    let moreMiscValuesOutputBase = moreMiscValuesOutputs.advanced(by: index * output.out_moremiscvalue.count)

                    (0..<output.output_policy.count).forEach { i in
                        policyOutputBase[i] = output.output_policy[i].floatValue
                    }

                    (0..<output.out_value.count).forEach { i in
                        valueOutputBase[i] = output.out_value[i].floatValue
                    }

                    (0..<output.out_ownership.count).forEach { i in
                        ownershipOutputBase[i] = output.out_ownership[i].floatValue
                    }

                    (0..<output.out_miscvalue.count).forEach { i in
                        miscValuesOutputBase[i] = output.out_miscvalue[i].floatValue
                    }

                    (0..<output.out_moremiscvalue.count).forEach { i in
                        moreMiscValuesOutputBase[i] = output.out_moremiscvalue[i].floatValue
                    }
                }
            } catch {
                Logger().error("An error occurred: \(error)")
            }
        }
    }
}

public func createCoreMLContext() {
    CoreMLBackend.reserveBackends()
}

public func destroyCoreMLContext() {
    CoreMLBackend.clearBackends()
}

public func createCoreMLBackend(modelXLen: Int,
                                modelYLen: Int,
                                serverThreadIdx: Int,
                                useFP16: Bool) -> Int {

    // Load the model.
    let modelIndex = CoreMLBackend.createInstance(xLen: modelXLen,
                                                  yLen: modelYLen,
                                                  useFP16: useFP16)

    Logger().info("CoreML backend thread \(serverThreadIdx): Model-\(modelIndex) \(modelXLen)x\(modelYLen) useFP16 \(useFP16)");

    // Return the model index.
    return modelIndex;
}

public func freeCoreMLBackend(modelIndex: Int) {
    CoreMLBackend.destroyInstance(index: modelIndex)
}

public func getCoreMLBackendVersion(modelIndex: Int) -> Int {
    let backend = CoreMLBackend.getBackend(at: modelIndex)
    let version = backend?.version ?? -1
    return version
}

public func getCoreMLHandleBatchOutput(userInputBuffer: UnsafeMutablePointer<Float32>,
                                       userInputGlobalBuffer: UnsafeMutablePointer<Float32>,
                                       policyOutputs: UnsafeMutablePointer<Float32>,
                                       valueOutputs: UnsafeMutablePointer<Float32>,
                                       ownershipOutputs: UnsafeMutablePointer<Float32>,
                                       miscValuesOutputs: UnsafeMutablePointer<Float32>,
                                       moreMiscValuesOutputs: UnsafeMutablePointer<Float32>,
                                       modelIndex: Int,
                                       batchSize: Int) {

    if let model = CoreMLBackend.getBackend(at: modelIndex) {
        model.getBatchOutput(binInputs: userInputBuffer,
                             globalInputs: userInputGlobalBuffer,
                             policyOutputs: policyOutputs,
                             valueOutputs: valueOutputs,
                             ownershipOutputs: ownershipOutputs,
                             miscValuesOutputs: miscValuesOutputs,
                             moreMiscValuesOutputs: moreMiscValuesOutputs,
                             batchSize: batchSize)
    } else {
        fatalError("Unable to get CoreML backend at model index: \(modelIndex)")
    }
}
