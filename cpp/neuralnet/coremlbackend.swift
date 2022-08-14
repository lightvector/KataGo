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

extension KataGob40c256Input {
    func printBinInputs() {
        let max_length = 3
        let lengths = swa_model_bin_inputs.shape.map({length in min(length.intValue, max_length)})

        for i in 0..<lengths[0] {
            let ii = NSNumber(value: i)
            for j in 0..<lengths[1] {
                let jj = NSNumber(value: j)
                for k in 0..<lengths[2] {
                    let kk = NSNumber(value: k)
                    print("bin_inputs[\(k)][\(j)][\(i)]=\(swa_model_bin_inputs[[kk, jj, ii]].floatValue)")
                }
            }
        }

        print(swa_model_bin_inputs.strides)
    }

    func printGlobalInputs() {
        let lengths = swa_model_global_inputs.shape.map({length in length.intValue})

        for i in 0..<lengths[0] {
            let ii = NSNumber(value: i)
            for j in 0..<lengths[1] {
                let jj = NSNumber(value: j)
                print("global_inputs[\(j)][\(i)]=\(swa_model_global_inputs[[jj, ii]].floatValue)")
            }
        }

        print(swa_model_global_inputs.strides)
    }

    func printData() {
        printBinInputs()
        printGlobalInputs()
        print(swa_model_include_history)
        print(swa_model_symmetries)
    }
}

extension KataGob40c256Output {
    func printData() {
        for i in 0..<swa_model_policy_output.shape.count {
            print("policy_output shape[\(i)]=\(swa_model_policy_output.shape[i])")
        }

        let lengths = swa_model_policy_output.shape.map({length in min(length.intValue, 3)})

        for i in 0..<lengths[0] {
            let ii = NSNumber(value: i)
            for j in 0..<lengths[1] {
                let jj = NSNumber(value: j)
                for k in 0..<lengths[2] {
                    let kk = NSNumber(value: k)
                    print("policy_output[\(k)][\(j)][\(i)]=\(swa_model_policy_output[[kk, jj, ii]].floatValue)")
                }
            }
        }
    }

    func copy(to output: UnsafeMutableRawPointer) {
        let byteCount = swa_model_policy_output.count * MemoryLayout<Float32>.size
        output.copyMemory(from: swa_model_policy_output.dataPointer, byteCount: byteCount)
    }
}

@objc
class CoreMLBackend: NSObject {
    @objc static let shared = CoreMLBackend()
    let model: KataGob40c256
    let includeHistory: MLMultiArray
    let symmetries: MLMultiArray

    private override init() {
        model = try! KataGob40c256()
        includeHistory = MLMultiArray(MLShapedArray<Float>(scalars: [1, 1, 1, 1, 1], shape: [1, 5]))
        symmetries = try! MLMultiArray([0, 0, 0])
    }

    @objc func getOutput(binInputs: UnsafeMutableRawPointer, globalInputs: UnsafeMutableRawPointer, policyOutput: UnsafeMutableRawPointer) throws {

        binInputs.printAsFloat()
        globalInputs.printAsFloat()

        let bin_inputs_array = try MLMultiArray(dataPointer: binInputs, shape: [1, 361, 22], dataType: MLMultiArrayDataType.float32, strides: [1, 1, 361])

        let global_inputs_array = try MLMultiArray(dataPointer: globalInputs, shape: [1, 19], dataType: MLMultiArrayDataType.float32, strides: [1, 1])

        let input = KataGob40c256Input(
            swa_model_bin_inputs: bin_inputs_array,
            swa_model_global_inputs: global_inputs_array,
            swa_model_include_history: includeHistory,
            swa_model_symmetries: symmetries)

        input.printData()

        /* swa_model_policy_output as 1 x 362 x 2 3-dimensional array of floats */
        let output = try model.prediction(input: input)
        output.printData()
        output.copy(to: policyOutput)
    }
}
