import Foundation
import CoreML
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

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
        // Process batch elements in parallel using Grand Central Dispatch
        // Each inference is independent, reading/writing to different buffer offsets
        DispatchQueue.concurrentPerform(iterations: batchSize) { b in
            autoreleasepool {
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

        // Copy spatial data using fast memcpy
        let spatialPtr = spatialArray.dataPointer.assumingMemoryBound(to: Float32.self)
        memcpy(spatialPtr, spatialInput.advanced(by: spatialOffset), spatialSize * MemoryLayout<Float32>.size)

        // Create global input array (1, C) - rank 2 as expected by converter
        let globalArray = try MLMultiArray(
            shape: [1, NSNumber(value: numInputGlobalChannels)],
            dataType: .float32)
        let globalPtr = globalArray.dataPointer.assumingMemoryBound(to: Float32.self)
        let globalOffset = batchIndex * numInputGlobalChannels
        memcpy(globalPtr, globalInput.advanced(by: globalOffset), numInputGlobalChannels * MemoryLayout<Float32>.size)

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
        memcpy(maskPtr, maskInput.advanced(by: maskOffset), maskSize * MemoryLayout<Float32>.size)
        inputDict[IONames.inputMask] = MLFeatureValue(multiArray: maskArray)

        // Add meta input if model has it
        if numInputMetaChannels > 0 {
            let metaArray = try MLMultiArray(
                shape: [1, NSNumber(value: numInputMetaChannels)],
                dataType: .float32)
            let metaPtr = metaArray.dataPointer.assumingMemoryBound(to: Float32.self)
            let metaOffset = batchIndex * numInputMetaChannels
            memcpy(metaPtr, metaInput.advanced(by: metaOffset), numInputMetaChannels * MemoryLayout<Float32>.size)
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
            // Fast path: direct memcpy
            memcpy(dest.advanced(by: destOffset), ptr, totalElements * MemoryLayout<Float32>.size)
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
        config.computeUnits = .cpuAndNeuralEngine  // Exclude GPU for hybrid mode

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
    printError("Core ML backend: Hybrid mode - CoreML (CPU+ANE) + MPSGraph (GPU)")
}

// MARK: - Throughput Tracker for Adaptive Batch Sizing

/// Tracks throughput for CoreML and MPSGraph paths to adaptively adjust batch split ratio
public class ThroughputTracker {
    private var coreMLSamplesPerSec: Double = 1.0
    private var mpsGraphSamplesPerSec: Double = 1.0
    private let alpha: Double = 0.3  // EMA smoothing factor (higher = faster adaptation)
    private let lock = NSLock()

    /// Update CoreML throughput measurement
    public func updateCoreML(samples: Int, duration: TimeInterval) {
        guard duration > 0, samples > 0 else { return }
        let newRate = Double(samples) / duration
        lock.lock()
        coreMLSamplesPerSec = alpha * newRate + (1 - alpha) * coreMLSamplesPerSec
        lock.unlock()
    }

    /// Update MPSGraph throughput measurement
    public func updateMPSGraph(samples: Int, duration: TimeInterval) {
        guard duration > 0, samples > 0 else { return }
        let newRate = Double(samples) / duration
        lock.lock()
        mpsGraphSamplesPerSec = alpha * newRate + (1 - alpha) * mpsGraphSamplesPerSec
        lock.unlock()
    }

    /// Get optimal CoreML ratio (0.0 to 1.0) based on measured throughput
    public func getOptimalCoreMLRatio() -> Float {
        lock.lock()
        let total = coreMLSamplesPerSec + mpsGraphSamplesPerSec
        let ratio = total > 0 ? Float(coreMLSamplesPerSec / total) : 0.5
        lock.unlock()
        return ratio
    }

    /// Get current throughput stats for logging
    public func getStats() -> (coreML: Double, mpsGraph: Double, ratio: Float) {
        lock.lock()
        let stats = (coreMLSamplesPerSec, mpsGraphSamplesPerSec, getOptimalCoreMLRatio())
        lock.unlock()
        return stats
    }
}

// MARK: - MPSGraph-based Model for GPU Inference

/// GPU-based model using MPSGraph for inference
public class MPSGraphModelHandle {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let graph: MPSGraph
    let nnXLen: Int32
    let nnYLen: Int32
    let numInputChannels: Int
    let numInputGlobalChannels: Int
    let numInputMetaChannels: Int
    let numPolicyChannels: Int
    let numValueChannels: Int
    let numScoreValueChannels: Int
    let numOwnershipChannels: Int

    // Layers
    let input: InputLayer
    let inputGlobal: InputGlobalLayer
    let inputMeta: InputMetaLayer
    let mask: MaskLayer
    let trunk: Trunk
    let policyHead: PolicyHead
    let valueHead: ValueHead
    let targetTensors: [MPSGraphTensor]

    public init?(
        modelDesc: SWModelDesc,
        nnXLen: Int32,
        nnYLen: Int32,
        optimizeIdentityMask: Bool = false
    ) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            printError("MPSGraph backend: Failed to create Metal device")
            return nil
        }

        self.device = device
        guard let queue = device.makeCommandQueue() else {
            printError("MPSGraph backend: Failed to create command queue")
            return nil
        }
        self.commandQueue = queue
        self.graph = MPSGraph()
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.numInputChannels = modelDesc.numInputChannels.intValue
        self.numInputGlobalChannels = modelDesc.numInputGlobalChannels.intValue
        self.numInputMetaChannels = modelDesc.numInputMetaChannels.intValue
        self.numPolicyChannels = 2  // Policy has 2 channels
        self.numValueChannels = modelDesc.numValueChannels.intValue
        self.numScoreValueChannels = modelDesc.numScoreValueChannels.intValue
        self.numOwnershipChannels = modelDesc.numOwnershipChannels.intValue

        let nnXLenNS = nnXLen as NSNumber
        let nnYLenNS = nnYLen as NSNumber

        input = InputLayer(
            graph: graph,
            nnXLen: nnXLenNS,
            nnYLen: nnYLenNS,
            numChannels: modelDesc.numInputChannels)

        inputGlobal = InputGlobalLayer(
            graph: graph,
            numGlobalFeatures: modelDesc.numInputGlobalChannels)

        inputMeta = InputMetaLayer(
            graph: graph,
            numMetaFeatures: modelDesc.numInputMetaChannels)

        mask = MaskLayer(
            graph: graph,
            nnXLen: nnXLenNS,
            nnYLen: nnYLenNS)

        // Use constant tensors when mask is all 1s (requireExactNNLen=true)
        let maskSum: MaskSumLayer
        let maskSumSqrtS14M01: MaskSumSqrtS14M01Layer
        let maskSumSqrtS14M01SquareS01: MaskSumSqrtS14M01SquareS01Layer

        if optimizeIdentityMask {
            maskSum = MaskSumLayer(
                graph: graph,
                nnXLen: nnXLenNS,
                nnYLen: nnYLenNS)
            maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(
                graph: graph,
                nnXLen: nnXLenNS,
                nnYLen: nnYLenNS)
            maskSumSqrtS14M01SquareS01 = MaskSumSqrtS14M01SquareS01Layer(
                graph: graph,
                nnXLen: nnXLenNS,
                nnYLen: nnYLenNS)
        } else {
            maskSum = MaskSumLayer(
                graph: graph,
                maskTensor: mask.tensor)
            maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(
                graph: graph,
                maskSum: maskSum)
            maskSumSqrtS14M01SquareS01 = MaskSumSqrtS14M01SquareS01Layer(
                graph: graph,
                maskSumSqrtS14M01: maskSumSqrtS14M01)
        }

        trunk = Trunk(
            graph: graph,
            descriptor: modelDesc.trunk,
            inputTensor: input.tensor,
            inputGlobalTensor: inputGlobal.tensor,
            inputMetaTensor: inputMeta.tensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            nnXLen: nnXLenNS,
            nnYLen: nnYLenNS,
            optimizeIdentityMask: optimizeIdentityMask)

        policyHead = PolicyHead(
            graph: graph,
            descriptor: modelDesc.policyHead,
            sourceTensor: trunk.resultTensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            nnXLen: nnXLenNS,
            nnYLen: nnYLenNS,
            optimizeIdentityMask: optimizeIdentityMask)

        valueHead = ValueHead(
            graph: graph,
            descriptor: modelDesc.valueHead,
            sourceTensor: trunk.resultTensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
            nnXLen: nnXLenNS,
            nnYLen: nnYLenNS,
            optimizeIdentityMask: optimizeIdentityMask)

        targetTensors = [
            policyHead.policyTensor,
            policyHead.policyPassTensor,
            valueHead.valueTensor,
            valueHead.scoreValueTensor,
            valueHead.ownershipTensor,
        ]

        printError("MPSGraph backend: Initialized on \(device.name)\(optimizeIdentityMask ? " (mask optimized)" : "")")
    }

    /// Run inference on a batch using MPSGraph (GPU)
    public func apply(
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        inputMeta inputMetaPointer: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        let channelAxis = InputShape.getChannelAxis()
        let numInputChannels = input.shape[channelAxis]
        let nnXLenNS = nnXLen as NSNumber
        let nnYLenNS = nnYLen as NSNumber

        let inputShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: numInputChannels,
            nnYLen: nnYLenNS,
            nnXLen: nnXLenNS)

        let inputDescriptor = MPSNDArrayDescriptor(
            dataType: input.tensor.dataType,
            shape: inputShape)

        let inputArray = MPSNDArray(
            device: device,
            descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)

        let numInputGlobalChannels = inputGlobal.shape[channelAxis]

        let inputGlobalShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: numInputGlobalChannels,
            nnYLen: 1,
            nnXLen: 1)

        let inputGlobalDescriptor = MPSNDArrayDescriptor(
            dataType: inputGlobal.tensor.dataType,
            shape: inputGlobalShape)

        let inputGlobalArray = MPSNDArray(
            device: device,
            descriptor: inputGlobalDescriptor)

        inputGlobalArray.writeBytes(inputGlobalPointer)

        let numInputMetaChannels = inputMeta.shape[channelAxis]

        let inputMetaShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: numInputMetaChannels,
            nnYLen: 1,
            nnXLen: 1)

        let inputMetaDescriptor = MPSNDArrayDescriptor(
            dataType: inputMeta.tensor.dataType,
            shape: inputMetaShape)

        let inputMetaArray = MPSNDArray(
            device: device,
            descriptor: inputMetaDescriptor)

        inputMetaArray.writeBytes(inputMetaPointer)

        let maskShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: 1,
            nnYLen: nnYLenNS,
            nnXLen: nnXLenNS)

        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: mask.tensor.dataType,
            shape: maskShape)

        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        // Extract mask from first channel of spatial input
        var maskStrideArray = [
            MemoryLayout<Float32>.size,
            Int(nnXLen) * MemoryLayout<Float32>.size,
            Int(nnYLen) * Int(nnXLen) * MemoryLayout<Float32>.size,
            numInputChannels.intValue * Int(nnYLen) * Int(nnXLen) * MemoryLayout<Float32>.size,
        ]

        maskArray.writeBytes(inputPointer, strideBytes: &maskStrideArray)

        let feeds = [
            input.tensor: MPSGraphTensorData(inputArray),
            inputGlobal.tensor: MPSGraphTensorData(inputGlobalArray),
            inputMeta.tensor: MPSGraphTensorData(inputMetaArray),
            mask.tensor: MPSGraphTensorData(maskArray),
        ]

        let fetch = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: targetTensors,
            targetOperations: nil)

        fetch[policyHead.policyTensor]?.mpsndarray().readBytes(policy)
        fetch[policyHead.policyPassTensor]?.mpsndarray().readBytes(policyPass)
        fetch[valueHead.valueTensor]?.mpsndarray().readBytes(value)
        fetch[valueHead.scoreValueTensor]?.mpsndarray().readBytes(scoreValue)
        fetch[valueHead.ownershipTensor]?.mpsndarray().readBytes(ownership)
    }
}

// MARK: - Hybrid Compute Handle

/// Hybrid compute handle that dispatches to both CoreML (CPU+ANE) and MPSGraph (GPU)
public class HybridComputeHandle {
    let coremlHandle: CoreMLComputeHandle
    let mpsGraphHandle: MPSGraphModelHandle
    let throughputTracker: ThroughputTracker
    let coremlQueue: DispatchQueue
    let mpsGraphQueue: DispatchQueue
    let nnXLen: Int32
    let nnYLen: Int32

    public init(
        coremlHandle: CoreMLComputeHandle,
        mpsGraphHandle: MPSGraphModelHandle
    ) {
        self.coremlHandle = coremlHandle
        self.mpsGraphHandle = mpsGraphHandle
        self.throughputTracker = ThroughputTracker()
        self.coremlQueue = DispatchQueue(label: "com.katago.coreml", qos: .userInitiated)
        self.mpsGraphQueue = DispatchQueue(label: "com.katago.mpsgraph", qos: .userInitiated)
        self.nnXLen = coremlHandle.nnXLen
        self.nnYLen = coremlHandle.nnYLen
    }

    /// Run hybrid inference - splits batch between CoreML and MPSGraph
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
        // Get optimal split ratio based on throughput
        let ratio = throughputTracker.getOptimalCoreMLRatio()
        let coreMLBatchSize = max(1, min(batchSize - 1, Int(Float(batchSize) * ratio)))
        let mpsGraphBatchSize = batchSize - coreMLBatchSize

        // Calculate buffer offsets
        let spatialSize = Int(nnXLen) * Int(nnYLen) * coremlHandle.numInputChannels
        let globalSize = coremlHandle.numInputGlobalChannels
        let metaSize = coremlHandle.numInputMetaChannels
        let maskSize = Int(nnXLen) * Int(nnYLen)
        let policySize = Int(nnXLen) * Int(nnYLen) * coremlHandle.numPolicyChannels
        let valueSize = coremlHandle.numValueChannels
        let scoreValueSize = coremlHandle.numScoreValueChannels
        let ownershipSize = Int(nnXLen) * Int(nnYLen) * coremlHandle.numOwnershipChannels

        let group = DispatchGroup()

        // CoreML path (CPU + ANE)
        if coreMLBatchSize > 0 {
            group.enter()
            coremlQueue.async { [self] in
                let start = CFAbsoluteTimeGetCurrent()

                coremlHandle.apply(
                    spatialInput: spatialInput,
                    globalInput: globalInput,
                    metaInput: metaInput,
                    maskInput: maskInput,
                    policy: policy,
                    policyPass: policyPass,
                    value: value,
                    scoreValue: scoreValue,
                    ownership: ownership,
                    batchSize: coreMLBatchSize
                )

                let duration = CFAbsoluteTimeGetCurrent() - start
                throughputTracker.updateCoreML(samples: coreMLBatchSize, duration: duration)
                group.leave()
            }
        }

        // MPSGraph path (GPU)
        if mpsGraphBatchSize > 0 {
            group.enter()
            mpsGraphQueue.async { [self] in
                let start = CFAbsoluteTimeGetCurrent()

                // Offset pointers for MPSGraph batch portion
                let spatialOffset = coreMLBatchSize * spatialSize
                let globalOffset = coreMLBatchSize * globalSize
                let metaOffset = coreMLBatchSize * metaSize
                let policyOffset = coreMLBatchSize * policySize
                let valueOffset = coreMLBatchSize * valueSize
                let scoreValueOffset = coreMLBatchSize * scoreValueSize
                let ownershipOffset = coreMLBatchSize * ownershipSize

                autoreleasepool {
                    mpsGraphHandle.apply(
                        input: spatialInput.advanced(by: spatialOffset),
                        inputGlobal: globalInput.advanced(by: globalOffset),
                        inputMeta: metaInput.advanced(by: metaOffset),
                        policy: policy.advanced(by: policyOffset),
                        policyPass: policyPass.advanced(by: coreMLBatchSize * coremlHandle.numPolicyChannels),
                        value: value.advanced(by: valueOffset),
                        scoreValue: scoreValue.advanced(by: scoreValueOffset),
                        ownership: ownership.advanced(by: ownershipOffset),
                        batchSize: mpsGraphBatchSize
                    )
                }

                let duration = CFAbsoluteTimeGetCurrent() - start
                throughputTracker.updateMPSGraph(samples: mpsGraphBatchSize, duration: duration)
                group.leave()
            }
        }

        // Wait for both paths to complete
        group.wait()
    }
}

/// Create a hybrid compute handle
public func createHybridComputeHandle(
    coremlModelPath: String,
    modelDesc: SWModelDesc,
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
) -> HybridComputeHandle? {

    // Create CoreML handle (CPU + ANE)
    guard let coremlHandle = createCoreMLComputeHandle(
        coremlModelPath: coremlModelPath,
        serverThreadIdx: serverThreadIdx,
        requireExactNNLen: requireExactNNLen,
        numInputChannels: numInputChannels,
        numInputGlobalChannels: numInputGlobalChannels,
        numInputMetaChannels: numInputMetaChannels,
        numPolicyChannels: numPolicyChannels,
        numValueChannels: numValueChannels,
        numScoreValueChannels: numScoreValueChannels,
        numOwnershipChannels: numOwnershipChannels,
        context: context
    ) else {
        printError("Hybrid backend \(serverThreadIdx): Failed to create CoreML handle")
        return nil
    }

    // Create MPSGraph handle (GPU)
    guard let mpsGraphHandle = MPSGraphModelHandle(
        modelDesc: modelDesc,
        nnXLen: context.nnXLen,
        nnYLen: context.nnYLen,
        optimizeIdentityMask: requireExactNNLen
    ) else {
        printError("Hybrid backend \(serverThreadIdx): Failed to create MPSGraph handle")
        return nil
    }

    printError("Hybrid backend \(serverThreadIdx): Initialized CoreML (CPU+ANE) + MPSGraph (GPU)")

    return HybridComputeHandle(
        coremlHandle: coremlHandle,
        mpsGraphHandle: mpsGraphHandle
    )
}
