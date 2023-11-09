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

    class func getBackend(at index: Int) -> CoreMLBackend {
        return backends[index]!
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
        let mlmodel = KataGoModel.compileBundleMLModel(modelName: modelName)
        
        // The CoreMLBackend object is created.
        backends[modelIndex] = CoreMLBackend(model: mlmodel!, xLen: xLen, yLen: yLen)

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
        self.version = Int(model.modelDescription.metadata[MLModelMetadataKey.versionString] as! String)!
        assert(self.version >= 8, "version must not be smaller than 8: \(self.version)")

        // The number of spatial features must be 22.
        self.numSpatialFeatures = 22

        // The number of global features must be 19.
        self.numGlobalFeatures = 19
    }

    func getOutput(binInputs: UnsafeMutablePointer<Float32>,
                   globalInputs: UnsafeMutablePointer<Float32>,
                   policyOutputs: UnsafeMutablePointer<Float32>,
                   valueOutputs: UnsafeMutablePointer<Float32>,
                   ownershipOutputs: UnsafeMutablePointer<Float32>,
                   miscValuesOutputs: UnsafeMutablePointer<Float32>,
                   moreMiscValuesOutputs: UnsafeMutablePointer<Float32>) {

        autoreleasepool {
            // Strides are used to access the data in the MLMultiArray.
            let strides = [numSpatialFeatures * yLen * xLen,
                           yLen * xLen,
                           xLen,
                           1] as [NSNumber]
            
            // Create the MLMultiArray for the spatial features.
            let bin_inputs_array = try! MLMultiArray(dataPointer: binInputs,
                                                     shape: [1, numSpatialFeatures, yLen, xLen] as [NSNumber],
                                                     dataType: .float,
                                                     strides: strides)
            
            // Create the MLMultiArray for the global features.
            let global_inputs_array = try! MLMultiArray(dataPointer: globalInputs,
                                                        shape: [1, numGlobalFeatures] as [NSNumber],
                                                        dataType: .float,
                                                        strides: [numGlobalFeatures, 1] as [NSNumber])
            
            let input = KataGoModelInput(input_spatial: bin_inputs_array,
                                         input_global: global_inputs_array)
            
            let options = MLPredictionOptions()
            
            let output = model.prediction(from: input, options: options)
            
            // Copy the output to the output buffers.
            for i in 0..<output.output_policy.count {
                policyOutputs[i] = output.output_policy[i].floatValue
            }
            
            for i in 0..<output.out_value.count {
                valueOutputs[i] = output.out_value[i].floatValue
            }
            
            for i in 0..<output.out_ownership.count {
                ownershipOutputs[i] = output.out_ownership[i].floatValue
            }
            
            for i in 0..<output.out_miscvalue.count {
                miscValuesOutputs[i] = output.out_miscvalue[i].floatValue
            }
            
            for i in 0..<output.out_moremiscvalue.count {
                moreMiscValuesOutputs[i] = output.out_moremiscvalue[i].floatValue
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
    return CoreMLBackend.getBackend(at: modelIndex).version
}

public func getCoreMLHandleOutput(userInputBuffer: UnsafeMutablePointer<Float32>,
                                  userInputGlobalBuffer: UnsafeMutablePointer<Float32>,
                                  policyOutputs: UnsafeMutablePointer<Float32>,
                                  valueOutputs: UnsafeMutablePointer<Float32>,
                                  ownershipOutputs: UnsafeMutablePointer<Float32>,
                                  miscValuesOutputs: UnsafeMutablePointer<Float32>,
                                  moreMiscValuesOutputs: UnsafeMutablePointer<Float32>,
                                  modelIndex: Int) {
    
    let model = CoreMLBackend.getBackend(at: modelIndex)

    model.getOutput(binInputs: userInputBuffer,
                    globalInputs: userInputGlobalBuffer,
                    policyOutputs: policyOutputs,
                    valueOutputs: valueOutputs,
                    ownershipOutputs: ownershipOutputs,
                    miscValuesOutputs: miscValuesOutputs,
                    moreMiscValuesOutputs: moreMiscValuesOutputs)
}
