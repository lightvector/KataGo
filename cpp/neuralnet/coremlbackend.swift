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
    let bin_inputs_shape: [NSNumber]
    let bin_inputs_strides: [NSNumber]
    let global_inputs_shape: [NSNumber]
    let global_inputs_strides: [NSNumber]
    let include_history: MLMultiArray
    let symmetries: MLMultiArray

    private override init() {
        let all = MLModelConfiguration()
        all.computeUnits = .all
        model = try! KataGob40c256(configuration: all)
        bin_inputs_shape = [1, 361, 22]
        bin_inputs_strides = [1, 1, 361]
        global_inputs_shape = [1, 19]
        global_inputs_strides = [1, 1]
        include_history = MLMultiArray(MLShapedArray<Float>(scalars: [1, 1, 1, 1, 1], shape: [1, 5]))
        symmetries = try! MLMultiArray([0, 0, 0])
    }

    func dump_raw_bin_inputs(_ bin_inputs: UnsafeMutableRawPointer) {
        print("raw_bin_inputs[0]=\(bin_inputs.load(fromByteOffset: 0, as: Float32.self))")
        print("raw_bin_inputs[1]=\(bin_inputs.load(fromByteOffset: 4, as: Float32.self))")
        print("raw_bin_inputs[2]=\(bin_inputs.load(fromByteOffset: 8, as: Float32.self))")
        print("raw_bin_inputs[3]=\(bin_inputs.load(fromByteOffset: 12, as: Float32.self))")
        print("raw_bin_inputs[4]=\(bin_inputs.load(fromByteOffset: 16, as: Float32.self))")
    }

    func dump_raw_global_inputs(_ global_inputs: UnsafeMutableRawPointer) {
        print("raw_global_inputs[0]=\(global_inputs.load(fromByteOffset: 0, as: Float32.self))")
        print("raw_global_inputs[1]=\(global_inputs.load(fromByteOffset: 4, as: Float32.self))")
        print("raw_global_inputs[2]=\(global_inputs.load(fromByteOffset: 8, as: Float32.self))")
        print("raw_global_inputs[3]=\(global_inputs.load(fromByteOffset: 12, as: Float32.self))")
        print("raw_global_inputs[4]=\(global_inputs.load(fromByteOffset: 16, as: Float32.self))")
    }

    @objc func getOutput(bin_inputs: UnsafeMutableRawPointer, global_inputs: UnsafeMutableRawPointer, policy_output: UnsafeMutableRawPointer) throws {

        bin_inputs.printAsFloat()
        global_inputs.printAsFloat()

        let bin_inputs_array = try MLMultiArray(dataPointer: bin_inputs, shape: bin_inputs_shape, dataType: MLMultiArrayDataType.float32, strides: bin_inputs_strides)

        let global_inputs_array = try MLMultiArray(dataPointer: global_inputs, shape: global_inputs_shape, dataType: MLMultiArrayDataType.float32, strides: global_inputs_strides)

        let input = KataGob40c256Input(
            swa_model_bin_inputs: bin_inputs_array,
            swa_model_global_inputs: global_inputs_array,
            swa_model_include_history: include_history,
            swa_model_symmetries: symmetries)

        input.printData()

        /* swa_model_policy_output as 1 x 362 x 2 3-dimensional array of floats */
        let output = try model.prediction(input: input)
        output.printData()
        output.copy(to: policy_output)
    }
}
