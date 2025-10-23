//
//  coremlbackend.swift
//  KataGo
//
//  Created by Chin-Chang Yang on 2023/11/8.
//

import Foundation
import CoreML

extension MLModel {
    var version: Int32 {
        let versionString = modelDescription.metadata[MLModelMetadataKey.versionString] as! String
        let versionInt = Int32(versionString)!
        return versionInt
    }

    var metaDescription: String {
        let description = modelDescription.metadata[MLModelMetadataKey.description] as! String
        return description
    }

    var nnXLen: Int? {
        if let match = metaDescription.firstMatch(of: #/KataGo\s+(\d+)x(\d+)/#) {
            let nnXLen = Int(match.1)
            return nnXLen
        } else {
            return nil
        }
    }

    var nnYLen: Int? {
        if let match = metaDescription.firstMatch(of: #/KataGo\s+(\d+)x(\d+)/#) {
            let nnYLen = Int(match.2)
            return nnYLen
        } else {
            return nil
        }
    }
}

public class CoreMLBackend {

    class func getModelName(xLen: Int = 19,
                            yLen: Int = 19,
                            useFP16: Bool = true,
                            metaEncoderVersion: Int = 0) -> String {
        let precision = useFP16 ? 16 : 32
        let encoder = (metaEncoderVersion > 0) ? "m\(metaEncoderVersion)" : ""
        return "KataGoModel\(xLen)x\(yLen)fp\(precision)\(encoder)"
    }

    var model: KataGoModel
    let xLen: Int
    let yLen: Int
    public let version: Int32
    let numSpatialFeatures: Int
    let numGlobalFeatures: Int
    let numMetaFeatures: Int
    let metaEncoderVersion: Int
    let modelName: String
    let modelDirectory: String

    var spatialSize: Int {
        numSpatialFeatures * yLen * xLen
    }

    public var modelXLen: Int32 { Int32(xLen) }
    public var modelYLen: Int32 { Int32(yLen) }

    init(model: MLModel, metaEncoderVersion: Int, modelName: String, modelDirectory: String) {
        self.model = KataGoModel(model: model)
        self.metaEncoderVersion = metaEncoderVersion
        self.modelName = modelName
        self.modelDirectory = modelDirectory

        self.xLen = model.nnXLen ?? 19
        assert(self.xLen >= 2)

        self.yLen = model.nnYLen ?? 19
        assert(self.yLen >= 2)

        // The model version must be at least 8.
        self.version = model.version
        assert(self.version >= 8)

        // The number of spatial features must be 22.
        self.numSpatialFeatures = 22

        // The number of global features must be 19.
        self.numGlobalFeatures = 19

        // The number of meta features must be 192.
        self.numMetaFeatures = 192
    }

    public func getBatchOutput(binInputs: UnsafeMutablePointer<Float32>,
                               globalInputs: UnsafeMutablePointer<Float32>,
                               metaInputs: UnsafeMutablePointer<Float32>,
                               policyOutputs: UnsafeMutablePointer<Float32>,
                               valueOutputs: UnsafeMutablePointer<Float32>,
                               ownershipOutputs: UnsafeMutablePointer<Float32>,
                               miscValuesOutputs: UnsafeMutablePointer<Float32>,
                               moreMiscValuesOutputs: UnsafeMutablePointer<Float32>,
                               batchSize: Int) {

        autoreleasepool {
            let spatialStrides = [numSpatialFeatures * yLen * xLen,
                                  yLen * xLen,
                                  xLen,
                                  1] as [NSNumber]

            let globalStrides = [numGlobalFeatures, 1] as [NSNumber]

            let inputArray = (0..<batchSize).map { index -> KataGoModelInput in
                let binInputsArray = try! MLMultiArray(
                    dataPointer: binInputs.advanced(by: index * spatialSize),
                    shape: [1, numSpatialFeatures, yLen, xLen] as [NSNumber],
                    dataType: .float,
                    strides: spatialStrides)

                let globalInputsArray = try! MLMultiArray(
                    dataPointer: globalInputs.advanced(by: index * numGlobalFeatures),
                    shape: [1, numGlobalFeatures] as [NSNumber],
                    dataType: .float,
                    strides: globalStrides)

                if metaEncoderVersion == 0 {
                    return KataGoModelInput(input_spatial: binInputsArray, input_global: globalInputsArray)
                } else {
                    let metaStrides = [numMetaFeatures, 1] as [NSNumber]

                    let metaInputsArray = try! MLMultiArray(
                        dataPointer: metaInputs.advanced(by: index * numMetaFeatures),
                        shape: [1, numMetaFeatures] as [NSNumber],
                        dataType: .float,
                        strides: metaStrides)

                    return KataGoModelInput(input_spatial: binInputsArray,
                                            input_global: globalInputsArray,
                                            input_meta: metaInputsArray)
                }
            }

            let inputBatch = KataGoModelInputBatch(inputArray: inputArray)
            let options = MLPredictionOptions()

            let outputBatch = safelyPredict(from: inputBatch, options: options)!
            assert(outputBatch.count == batchSize)

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
        }
    }

    func safelyPredict(from inputBatch: KataGoModelInputBatch,
                       options: MLPredictionOptions) -> KataGoModelOutputBatch? {
        if let prediction = try? model.prediction(from: inputBatch, options: options) {
            return prediction
        }

        let computeUnits = model.model.configuration.computeUnits

        for mustCompile in [false, true] {
            if let prediction = compileAndPredict(with: computeUnits, from: inputBatch, options: options, mustCompile: mustCompile) {
                return prediction
            }
        }

        return nil
    }

    private func compileAndPredict(with computeUnits: MLComputeUnits,
                                   from inputBatch: KataGoModelInputBatch,
                                   options: MLPredictionOptions,
                                   mustCompile: Bool) -> KataGoModelOutputBatch? {
        if let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName, computeUnits: computeUnits, mustCompile: mustCompile, modelDirectory: modelDirectory) {
            model = KataGoModel(model: mlmodel)
            if let outputBatch = try? model.prediction(from: inputBatch, options: options) {
                return outputBatch
            }
        }

        return nil
    }
}

public func maybeCreateCoreMLBackend(condition: Bool = true,
                                     serverThreadIdx: Int = 0,
                                     xLen: Int = 19,
                                     yLen: Int = 19,
                                     useFP16: Bool = false,
                                     metaEncoderVersion: Int = 0,
                                     useCpuAndNeuralEngine: Bool = true,
                                     modelDirectory: String = "") -> CoreMLBackend? {
    guard condition else { return nil }

    // Get the model name.
    let modelName = CoreMLBackend.getModelName(xLen: xLen, yLen: yLen, useFP16: useFP16, metaEncoderVersion: metaEncoderVersion)

    // Specify compute units.
    let computeUnits: MLComputeUnits = useCpuAndNeuralEngine ? .cpuAndNeuralEngine : .all

    // Compile the model in Bundle.
    let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName,
                                                   computeUnits: computeUnits,
                                                   mustCompile: false,
                                                   modelDirectory: modelDirectory)

    if let mlmodel {
        printError("CoreML backend \(serverThreadIdx): useFP16 \(useFP16) metaEncoderVersion \(metaEncoderVersion) useCpuAndNeuralEngine \(useCpuAndNeuralEngine)");
        printError("CoreML backend \(serverThreadIdx): \(mlmodel.metaDescription)");

        // The CoreMLBackend object is created.
        return CoreMLBackend(model: mlmodel,
                             metaEncoderVersion: metaEncoderVersion,
                             modelName: modelName,
                             modelDirectory: modelDirectory)
    } else {
        printError("Unable to compile bundle MLModel from model: \(modelName)")
        return nil
    }
}
