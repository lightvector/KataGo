import Foundation
import CoreML

extension UnsafeMutableRawPointer {
    func printAsFloat() {
        print("data[0]=\(load(fromByteOffset: 0, as: Float32.self))")
        print("data[1]=\(load(fromByteOffset: 4, as: Float32.self))")
        print("data[2]=\(load(fromByteOffset: 8, as: Float32.self))")
        print("data[3]=\(load(fromByteOffset: 12, as: Float32.self))")
        print("data[4]=\(load(fromByteOffset: 16, as: Float32.self))")
    }
}

extension MLMultiArray {
    func copyFloat(to output: UnsafeMutableRawPointer) {
        output.copyMemory(from: dataPointer, byteCount: count * MemoryLayout<Float>.size)
    }
}

extension KataGoModelInput {
    func printData(of featureName: String) {
        let array = featureValue(for: featureName)!.multiArrayValue!
        let maxPrintCount = 5
        let printCount = min(array.count, maxPrintCount)

        print("\(featureName) shape: \(array.shape)")

        for i in 0..<printCount {
            print("\(featureName)[\(i)] = \(array[i].floatValue)")
        }
    }

    func printData() {
        for featureName in featureNames {
            printData(of: featureName)
        }
    }
}

extension KataGoModelOutput {
    func printData(of featureName: String) {
        let array = featureValue(for: featureName)!.multiArrayValue!
        let maxPrintCount = 5
        let printCount = min(array.count, maxPrintCount)

        print("\(featureName) shape: \(array.shape)")

        for i in 0..<printCount {
            print("\(featureName)[\(i)] = \(array[i].floatValue)")
        }
    }

    func printData() {
        for featureName in featureNames {
            printData(of: featureName)
        }
    }
}

@objc
class CoreMLBackend: NSObject {
    static var models: [Int: CoreMLBackend] = [:]
    let model: KataGoModel
    let includeHistory: MLMultiArray
    let symmetries: MLMultiArray

    @objc class func getModel(at index: Int) -> CoreMLBackend {
        if let model = models[index] {
            return model
        } else {
            let model = CoreMLBackend()
            models[index] = model
            return model
        }
    }

    private override init() {
        model = try! KataGoModel()
        includeHistory = MLMultiArray(MLShapedArray<Float>(scalars: [1, 1, 1, 1, 1], shape: [1, 5]))
        symmetries = try! MLMultiArray([0, 0, 0])
    }

    @objc func getOutput(binInputs: UnsafeMutableRawPointer, globalInputs: UnsafeMutableRawPointer, policyOutput: UnsafeMutableRawPointer, valueOutput: UnsafeMutableRawPointer, ownershipOutput: UnsafeMutableRawPointer, miscValuesOutput: UnsafeMutableRawPointer, moreMiscValuesOutput: UnsafeMutableRawPointer) throws {
        let bin_inputs_array = try MLMultiArray(dataPointer: binInputs, shape: [1, 361, 22], dataType: MLMultiArrayDataType.float32, strides: [1, 1, 361])

        let global_inputs_array = try MLMultiArray(dataPointer: globalInputs, shape: [1, 19], dataType: MLMultiArrayDataType.float32, strides: [1, 1])

        let input = KataGoModelInput(
            swa_model_bin_inputs: bin_inputs_array,
            swa_model_global_inputs: global_inputs_array,
            swa_model_include_history: includeHistory,
            swa_model_symmetries: symmetries)

        let output = try model.prediction(input: input)
        output.swa_model_policy_output.copyFloat(to: policyOutput)
        output.swa_model_value_output.copyFloat(to: valueOutput)
        output.swa_model_ownership_output.copyFloat(to: ownershipOutput)
        output.swa_model_miscvalues_output.copyFloat(to: miscValuesOutput)
        output.swa_model_moremiscvalues_output.copyFloat(to: moreMiscValuesOutput)
    }
}
