import Foundation
import CoreML

/// A class that handles output to standard error.
class StandardError: TextOutputStream {
    func write(_ string: String) {
        try? FileHandle.standardError.write(contentsOf: Data(string.utf8))
    }
}

/// Print to standard error
func printError(_ item: Any) {
    var instance = StandardError()
    print(item, to: &instance)
}

// NOTE: Model caching and conversion are now handled in C++ using the native katagocoreml library.
// The Python-based CoreMLConverter and ModelCacheManager have been removed to eliminate Python dependency.

/// Context storing board dimensions and settings
public class CoreMLComputeContext {
    public let nnXLen: Int32
    public let nnYLen: Int32
    public let useFP16: Bool

    init(nnXLen: Int32, nnYLen: Int32, useFP16: Bool) {
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.useFP16 = useFP16
    }
}

/// Create a Core ML compute context
public func createCoreMLComputeContext(
    nnXLen: Int32,
    nnYLen: Int32,
    useFP16: Bool
) -> CoreMLComputeContext {
    return CoreMLComputeContext(nnXLen: nnXLen, nnYLen: nnYLen, useFP16: useFP16)
}

/// Handle that wraps the loaded MLModel for inference
public class CoreMLComputeHandle {
    let model: MLModel
    let nnXLen: Int32
    let nnYLen: Int32
    let optimizeIdentityMask: Bool
    let numInputChannels: Int
    let numInputGlobalChannels: Int
    let numInputMetaChannels: Int
    let numPolicyChannels: Int
    let numValueChannels: Int
    let numScoreValueChannels: Int
    let numOwnershipChannels: Int

    /// Model input/output names matching KataGoCoremltools output
    struct IONames {
        static let spatialInput = "spatial_input"
        static let globalInput = "global_input"
        static let inputMask = "input_mask"
        static let metaInput = "meta_input"

        static let policyOutput = "policy_p2_conv"
        static let policyPassOutput = "policy_pass"
        static let valueOutput = "value_v3_bias"
        static let ownershipOutput = "value_ownership_conv"
        static let scoreValueOutput = "value_sv3_bias"
    }

    init(model: MLModel, nnXLen: Int32, nnYLen: Int32,
         optimizeIdentityMask: Bool,
         numInputChannels: Int,
         numInputGlobalChannels: Int,
         numInputMetaChannels: Int,
         numPolicyChannels: Int,
         numValueChannels: Int,
         numScoreValueChannels: Int,
         numOwnershipChannels: Int) {
        self.model = model
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.optimizeIdentityMask = optimizeIdentityMask
        self.numInputChannels = numInputChannels
        self.numInputGlobalChannels = numInputGlobalChannels
        self.numInputMetaChannels = numInputMetaChannels
        self.numPolicyChannels = numPolicyChannels
        self.numValueChannels = numValueChannels
        self.numScoreValueChannels = numScoreValueChannels
        self.numOwnershipChannels = numOwnershipChannels
    }

    /// Run inference on a batch of inputs
    public func apply(
        spatialInput: UnsafeMutablePointer<Float32>,
        globalInput: UnsafeMutablePointer<Float32>,
        metaInput: UnsafeMutablePointer<Float32>,
        maskInput: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        autoreleasepool {
            // Process batch elements sequentially (Core ML optimized for batch=1)
            for b in 0..<batchSize {
                do {
                    try runSingleInference(
                        batchIndex: b,
                        spatialInput: spatialInput,
                        globalInput: globalInput,
                        metaInput: metaInput,
                        maskInput: maskInput,
                        policy: policy,
                        policyPass: policyPass,
                        value: value,
                        scoreValue: scoreValue,
                        ownership: ownership
                    )
                } catch {
                    printError("Core ML inference error: \(error)")
                }
            }
        }
    }

    private func runSingleInference(
        batchIndex: Int,
        spatialInput: UnsafeMutablePointer<Float32>,
        globalInput: UnsafeMutablePointer<Float32>,
        metaInput: UnsafeMutablePointer<Float32>,
        maskInput: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>
    ) throws {
        let spatialSize = Int(nnXLen) * Int(nnYLen) * numInputChannels
        let spatialOffset = batchIndex * spatialSize

        // Create MLMultiArray for spatial input (1, C, H, W)
        let spatialArray = try MLMultiArray(
            shape: [1, NSNumber(value: numInputChannels),
                   NSNumber(value: nnYLen), NSNumber(value: nnXLen)],
            dataType: .float32)

        // Copy spatial data
        let spatialPtr = spatialArray.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<spatialSize {
            spatialPtr[i] = spatialInput[spatialOffset + i]
        }

        // Create global input array (1, C) - rank 2 as expected by converter
        let globalArray = try MLMultiArray(
            shape: [1, NSNumber(value: numInputGlobalChannels)],
            dataType: .float32)
        let globalPtr = globalArray.dataPointer.assumingMemoryBound(to: Float32.self)
        let globalOffset = batchIndex * numInputGlobalChannels
        for i in 0..<numInputGlobalChannels {
            globalPtr[i] = globalInput[globalOffset + i]
        }

        // Build feature provider dictionary
        var inputDict: [String: MLFeatureValue] = [
            IONames.spatialInput: MLFeatureValue(multiArray: spatialArray),
            IONames.globalInput: MLFeatureValue(multiArray: globalArray)
        ]

        // Add mask input (always required, even with optimize_identity_mask=True)
        // When optimize_identity_mask=True, the mask is still required as input but
        // internal mask operations are optimized away for ~6.5% speedup
        let maskArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: nnYLen), NSNumber(value: nnXLen)],
            dataType: .float32)
        let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Float32.self)
        let maskSize = Int(nnXLen) * Int(nnYLen)
        let maskOffset = batchIndex * maskSize
        for i in 0..<maskSize {
            maskPtr[i] = maskInput[maskOffset + i]
        }
        inputDict[IONames.inputMask] = MLFeatureValue(multiArray: maskArray)

        // Add meta input if model has it
        if numInputMetaChannels > 0 {
            let metaArray = try MLMultiArray(
                shape: [1, NSNumber(value: numInputMetaChannels)],
                dataType: .float32)
            let metaPtr = metaArray.dataPointer.assumingMemoryBound(to: Float32.self)
            let metaOffset = batchIndex * numInputMetaChannels
            for i in 0..<numInputMetaChannels {
                metaPtr[i] = metaInput[metaOffset + i]
            }
            inputDict[IONames.metaInput] = MLFeatureValue(multiArray: metaArray)
        }

        // Run prediction
        let featureProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let prediction = try model.prediction(from: featureProvider)

        // Extract outputs and copy to output buffers
        extractOutputs(
            prediction: prediction,
            batchIndex: batchIndex,
            policy: policy,
            policyPass: policyPass,
            value: value,
            scoreValue: scoreValue,
            ownership: ownership
        )
    }

    /// Copy MLMultiArray data to destination buffer, respecting strides.
    /// Core ML may return non-contiguous arrays, especially for spatial outputs after GPU computation.
    private func copyMultiArray(
        _ array: MLMultiArray,
        to dest: UnsafeMutablePointer<Float32>,
        destOffset: Int
    ) {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
        let totalElements = shape.reduce(1, *)

        // Check if contiguous (strides match expected for row-major C-order)
        var isContiguous = true
        var expectedStride = 1
        for i in (0..<shape.count).reversed() {
            if strides[i] != expectedStride {
                isContiguous = false
                break
            }
            expectedStride *= shape[i]
        }

        if isContiguous {
            // Fast path: direct memcpy-style copy
            for i in 0..<totalElements {
                dest[destOffset + i] = ptr[i]
            }
        } else {
            // Slow path: copy with strides (handles non-contiguous layouts)
            copyWithStrides(
                from: ptr,
                to: dest,
                destOffset: destOffset,
                shape: shape,
                strides: strides,
                dim: 0,
                srcOffset: 0,
                destIdx: 0
            )
        }
    }

    /// Recursively copy array elements respecting strides (NCHW order)
    @discardableResult
    private func copyWithStrides(
        from src: UnsafePointer<Float32>,
        to dest: UnsafeMutablePointer<Float32>,
        destOffset: Int,
        shape: [Int],
        strides: [Int],
        dim: Int,
        srcOffset: Int,
        destIdx: Int
    ) -> Int {
        var currentDestIdx = destIdx

        if dim == shape.count - 1 {
            // Innermost dimension: copy elements
            for i in 0..<shape[dim] {
                dest[destOffset + currentDestIdx] = src[srcOffset + i * strides[dim]]
                currentDestIdx += 1
            }
        } else {
            // Recurse into next dimension
            for i in 0..<shape[dim] {
                currentDestIdx = copyWithStrides(
                    from: src,
                    to: dest,
                    destOffset: destOffset,
                    shape: shape,
                    strides: strides,
                    dim: dim + 1,
                    srcOffset: srcOffset + i * strides[dim],
                    destIdx: currentDestIdx
                )
            }
        }

        return currentDestIdx
    }

    private func extractOutputs(
        prediction: MLFeatureProvider,
        batchIndex: Int,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>
    ) {
        // Extract policy output (1, policyChannels, H, W)
        // Must use stride-aware copy as Core ML may return non-contiguous arrays
        if let policyArray = prediction.featureValue(for: IONames.policyOutput)?.multiArrayValue {
            let policyOffset = batchIndex * Int(nnXLen) * Int(nnYLen) * numPolicyChannels
            copyMultiArray(policyArray, to: policy, destOffset: policyOffset)
        }

        // Extract policy pass output (1, numPolicyChannels)
        if let passArray = prediction.featureValue(for: IONames.policyPassOutput)?.multiArrayValue {
            let passOffset = batchIndex * numPolicyChannels
            copyMultiArray(passArray, to: policyPass, destOffset: passOffset)
        }

        // Extract value output (1, 3)
        if let valueArray = prediction.featureValue(for: IONames.valueOutput)?.multiArrayValue {
            let valueOffset = batchIndex * numValueChannels
            copyMultiArray(valueArray, to: value, destOffset: valueOffset)
        }

        // Extract score value output (1, numScoreValueChannels)
        if let svArray = prediction.featureValue(for: IONames.scoreValueOutput)?.multiArrayValue {
            let svOffset = batchIndex * numScoreValueChannels
            copyMultiArray(svArray, to: scoreValue, destOffset: svOffset)
        }

        // Extract ownership output (1, 1, H, W)
        // Must use stride-aware copy as Core ML may return non-contiguous arrays
        if let ownArray = prediction.featureValue(for: IONames.ownershipOutput)?.multiArrayValue {
            let ownOffset = batchIndex * Int(nnXLen) * Int(nnYLen) * numOwnershipChannels
            copyMultiArray(ownArray, to: ownership, destOffset: ownOffset)
        }
    }
}

/// Create compute handle - loads pre-converted Core ML model
/// Model conversion is now handled in C++ using the native katagocoreml library
public func createCoreMLComputeHandle(
    coremlModelPath: String,
    serverThreadIdx: Int,
    requireExactNNLen: Bool,
    numInputChannels: Int32,
    numInputGlobalChannels: Int32,
    numInputMetaChannels: Int32,
    numPolicyChannels: Int32,
    numValueChannels: Int32,
    numScoreValueChannels: Int32,
    numOwnershipChannels: Int32,
    context: CoreMLComputeContext
) -> CoreMLComputeHandle? {

    let optimizeMask = requireExactNNLen  // When true: skips internal mask operations (~6.5% speedup)
    let mlpackagePath = URL(fileURLWithPath: coremlModelPath)

    // Load Core ML model (already converted by C++ katagocoreml library)
    do {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine + GPU + CPU

        printError("Core ML backend \(serverThreadIdx): Compiling model...")
        let compiledURL = try MLModel.compileModel(at: mlpackagePath)

        printError("Core ML backend \(serverThreadIdx): Loading compiled model...")
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        printError("Core ML backend \(serverThreadIdx): Model loaded successfully, \(context.nnXLen)x\(context.nnYLen)")

        return CoreMLComputeHandle(
            model: model,
            nnXLen: context.nnXLen,
            nnYLen: context.nnYLen,
            optimizeIdentityMask: optimizeMask,
            numInputChannels: Int(numInputChannels),
            numInputGlobalChannels: Int(numInputGlobalChannels),
            numInputMetaChannels: Int(numInputMetaChannels),
            numPolicyChannels: Int(numPolicyChannels),
            numValueChannels: Int(numValueChannels),
            numScoreValueChannels: Int(numScoreValueChannels),
            numOwnershipChannels: Int(numOwnershipChannels)
        )
    } catch {
        printError("Core ML backend: Failed to load model: \(error)")
        return nil
    }
}

/// Print available Core ML compute units
public func printCoreMLDevices() {
    printError("Core ML backend: Using Apple Neural Engine + GPU + CPU acceleration")
}
