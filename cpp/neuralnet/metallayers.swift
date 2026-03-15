// MPSGraph layer implementations shared between Metal and CoreML backends
// Extracted from metalbackend.swift to enable hybrid CoreML + MPSGraph execution

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// MARK: - Helper Extensions

/// An extension to the Data struct for handling float data with optional FP16 conversion.
extension Data {
    /// Initializes a new Data instance using an UnsafeMutablePointer<Float32>, with optional conversion to FP16 format.
    init(
        floatsNoCopy: UnsafeMutablePointer<Float32>,
        shape: [NSNumber]
    ) {
        self.init(
            bytesNoCopy: floatsNoCopy,
            count: shape.countBytesOfFloat32(),
            deallocator: .none)
    }
}

/// Extension to MPSNDArray to convert from MPSGraphTensor, and to read/write bytes from/to UnsafeMutableRawPointer
extension MPSNDArray {
    /// Read bytes from the buffer
    func readBytes(_ buffer: UnsafeMutableRawPointer) {
        self.readBytes(buffer, strideBytes: nil)
    }

    /// Write bytes to the buffer
    func writeBytes(_ buffer: UnsafeMutableRawPointer) {
        self.writeBytes(buffer, strideBytes: nil)
    }
}

/// Extension to Array to count number of elements and bytes
extension Array where Element == NSNumber {
    /// Count number of elements
    func countElements() -> Int {
        return reduce(1, { $0 * $1.intValue })
    }

    /// Count number of bytes
    func countBytesOfFloat32() -> Int {
        return countElements() * MemoryLayout<Float32>.size
    }
}

/// Extension to MPSGraph to the mish activation function
extension MPSGraph {
    /// Mish activation: x * tanh(softplus(x))
    func mish(tensor: MPSGraphTensor) -> MPSGraphTensor {
        assert(tensor.dataType == .float32)

        let one = 1.0
        let threshold = 20.0
        let thresholdTensor = constant(threshold, dataType: tensor.dataType)
        let minimumTensor = minimum(tensor, thresholdTensor, name: nil)
        let expTensor = exponent(with: minimumTensor, name: nil)
        let oneTensor = constant(one, dataType: tensor.dataType)
        let addTensor = addition(expTensor, oneTensor, name: nil)
        let logTensor = logarithm(with: addTensor, name: nil)
        let lessTensor = lessThan(tensor, thresholdTensor, name: nil)
        let selectTensor = select(
            predicate: lessTensor, trueTensor: logTensor, falseTensor: tensor, name: nil)
        let tanhTensor = tanh(with: selectTensor, name: nil)
        let mulTensor = multiplication(tensor, tanhTensor, name: nil)

        return mulTensor
    }
}

// MARK: - Input Shape Utilities

/// A structure that represents the input shape (internal - not exposed to C++)
struct InputShape {
    /// Create a shape for the input tensor
    static func create(
        batchSize: NSNumber,
        numChannels: NSNumber,
        nnYLen: NSNumber,
        nnXLen: NSNumber
    ) -> [NSNumber] {
        return [batchSize, numChannels, nnYLen, nnXLen]
    }

    /// Get the channel axis
    static func getChannelAxis() -> Int {
        return 1
    }

    /// Get the HW axes
    static func getHWAxes() -> [NSNumber] {
        return [2, 3] as [NSNumber]
    }
}

// MARK: - Input Layers

/// A structure that represents the input layer
struct InputLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    init(
        graph: MPSGraph,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        numChannels: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        shape = InputShape.create(
            batchSize: -1,
            numChannels: numChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents an input global layer for a neural network model.
struct InputGlobalLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    init(
        graph: MPSGraph,
        numGlobalFeatures: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        shape = InputShape.create(
            batchSize: -1,
            numChannels: numGlobalFeatures,
            nnYLen: 1,
            nnXLen: 1)

        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure representing the input meta layer for a neural network graph.
struct InputMetaLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    init(
        graph: MPSGraph,
        numMetaFeatures: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        shape = InputShape.create(
            batchSize: -1,
            numChannels: numMetaFeatures,
            nnYLen: 1,
            nnXLen: 1)

        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)
    }
}

/// A structure that represents a mask layer for a neural network model.
struct MaskLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    init(
        graph: MPSGraph,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        shape = InputShape.create(
            batchSize: -1,
            numChannels: 1,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

// MARK: - Mask Processing Layers

/// A structure that represents a layer which performs the summation operation on a mask layer.
struct MaskSumLayer {
    let tensor: MPSGraphTensor

    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    init(
        graph: MPSGraph,
        maskTensor: MPSGraphTensor
    ) {
        let hwAxes = InputShape.getHWAxes()

        self.tensor = graph.reductionSum(
            with: maskTensor,
            axes: hwAxes,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }

    /// Optimized init for when mask is all 1s (requireExactNNLen=true)
    /// Returns constant tensor with boardSize value
    init(
        graph: MPSGraph,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        let boardSize = Double(nnXLen.intValue * nnYLen.intValue)
        self.tensor = graph.constant(
            boardSize,
            shape: [1, 1, 1, 1],
            dataType: dataType)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents sqrt(maskSum) * 0.1 - 1.4
struct MaskSumSqrtS14M01Layer {
    let tensor: MPSGraphTensor

    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    init(
        graph: MPSGraph,
        maskSum: MaskSumLayer
    ) {
        let sqrtMaskSum = graph.squareRoot(with: maskSum.tensor, name: nil)

        let fourTeen = graph.constant(
            14.0,
            shape: [1],
            dataType: maskSum.tensor.dataType)

        let subtracted = graph.subtraction(sqrtMaskSum, fourTeen, name: nil)

        let zeroPointone = graph.constant(
            0.1,
            shape: [1],
            dataType: maskSum.tensor.dataType)

        self.tensor = graph.multiplication(
            subtracted,
            zeroPointone,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }

    /// Optimized init for when mask is all 1s (requireExactNNLen=true)
    /// Returns constant tensor: (sqrt(boardSize) - 14) * 0.1
    init(
        graph: MPSGraph,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        let boardSize = Double(nnXLen.intValue * nnYLen.intValue)
        let value = (sqrt(boardSize) - 14.0) * 0.1
        self.tensor = graph.constant(
            value,
            shape: [1, 1, 1, 1],
            dataType: dataType)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure for (sqrt(maskSum) * 0.1 - 1.4)^2 - 0.1
struct MaskSumSqrtS14M01SquareS01Layer {
    let tensor: MPSGraphTensor

    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    init(
        graph: MPSGraph,
        maskSumSqrtS14M01: MaskSumSqrtS14M01Layer
    ) {
        let squared = graph.square(with: maskSumSqrtS14M01.tensor, name: nil)

        let zeroPointone = graph.constant(
            0.1,
            shape: [1],
            dataType: maskSumSqrtS14M01.tensor.dataType)

        self.tensor = graph.subtraction(
            squared,
            zeroPointone,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }

    /// Optimized init for when mask is all 1s (requireExactNNLen=true)
    /// Returns constant tensor: ((sqrt(boardSize) - 14) * 0.1)^2 - 0.1
    init(
        graph: MPSGraph,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        let boardSize = Double(nnXLen.intValue * nnYLen.intValue)
        let sqrtS14M01 = (sqrt(boardSize) - 14.0) * 0.1
        let value = sqrtS14M01 * sqrtS14M01 - 0.1
        self.tensor = graph.constant(
            value,
            shape: [1, 1, 1, 1],
            dataType: dataType)

        assert(self.tensor.shape?.count == 4)
    }
}

// MARK: - Layer Descriptors

/// An enumeration of the different kinds of activation function.
public enum ActivationKind {
    case identity
    case relu
    case mish
}

/// A struct that represents a description of convolutional layer.
public struct SWConvLayerDesc {
    let convYSize: NSNumber
    let convXSize: NSNumber
    let inChannels: NSNumber
    let outChannels: NSNumber
    let dilationY: Int
    let dilationX: Int
    let weights: UnsafeMutablePointer<Float32>

    init(
        convYSize: NSNumber,
        convXSize: NSNumber,
        inChannels: NSNumber,
        outChannels: NSNumber,
        dilationY: Int,
        dilationX: Int,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.convYSize = convYSize
        self.convXSize = convXSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dilationY = dilationY
        self.dilationX = dilationX
        self.weights = weights
    }
}

public func createSWConvLayerDesc(
    convYSize: Int32,
    convXSize: Int32,
    inChannels: Int32,
    outChannels: Int32,
    dilationY: Int32,
    dilationX: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWConvLayerDesc {
    return SWConvLayerDesc(
        convYSize: convYSize as NSNumber,
        convXSize: convXSize as NSNumber,
        inChannels: inChannels as NSNumber,
        outChannels: outChannels as NSNumber,
        dilationY: Int(dilationY),
        dilationX: Int(dilationX),
        weights: weights)
}

/// A struct that represents a description of a batch normalization layer.
public struct SWBatchNormLayerDesc {
    let numChannels: NSNumber
    let mergedScale: UnsafeMutablePointer<Float32>
    let mergedBias: UnsafeMutablePointer<Float32>

    init(
        numChannels: NSNumber,
        mergedScale: UnsafeMutablePointer<Float32>,
        mergedBias: UnsafeMutablePointer<Float32>
    ) {
        self.numChannels = numChannels
        self.mergedScale = mergedScale
        self.mergedBias = mergedBias
    }
}

public func createSWBatchNormLayerDesc(
    numChannels: Int32,
    mergedScale: UnsafeMutablePointer<Float32>,
    mergedBias: UnsafeMutablePointer<Float32>
) -> SWBatchNormLayerDesc {
    return SWBatchNormLayerDesc(
        numChannels: numChannels as NSNumber,
        mergedScale: mergedScale,
        mergedBias: mergedBias)
}

/// A struct that represents a matrix multiplication layer descriptor
public struct SWMatMulLayerDesc {
    let inChannels: NSNumber
    let outChannels: NSNumber
    let weights: UnsafeMutablePointer<Float32>

    init(
        inChannels: NSNumber,
        outChannels: NSNumber,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
}

public func createSWMatMulLayerDesc(
    inChannels: Int32,
    outChannels: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWMatMulLayerDesc {
    return SWMatMulLayerDesc(
        inChannels: inChannels as NSNumber,
        outChannels: outChannels as NSNumber,
        weights: weights)
}

/// A struct that represents the bias layer description.
public struct SWMatBiasLayerDesc {
    let numChannels: NSNumber
    let weights: UnsafeMutablePointer<Float32>

    init(
        numChannels: NSNumber,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.numChannels = numChannels
        self.weights = weights
    }
}

public func createSWMatBiasLayerDesc(
    numChannels: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWMatBiasLayerDesc {
    return SWMatBiasLayerDesc(
        numChannels: numChannels as NSNumber,
        weights: weights)
}

// MARK: - Core Layers

/// A class that represents a convolutional layer using MPSGraph
class ConvLayer {
    let resultTensor: MPSGraphTensor
    let convDescriptor = MPSGraphConvolution2DOpDescriptor(
        strideInX: 1,
        strideInY: 1,
        dilationRateInX: 1,
        dilationRateInY: 1,
        groups: 1,
        paddingStyle: .TF_SAME,
        dataLayout: .NCHW,
        weightsLayout: .OIHW)!

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        descriptor: SWConvLayerDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {
        let weightsShape = [
            descriptor.outChannels,
            descriptor.inChannels,
            descriptor.convYSize,
            descriptor.convXSize,
        ]

        let weightsData = Data(
            floatsNoCopy: descriptor.weights,
            shape: weightsShape)

        let weightsTensor = graph.constant(
            weightsData,
            shape: weightsShape,
            dataType: sourceTensor.dataType)

        resultTensor = graph.convolution2D(
            sourceTensor,
            weights: weightsTensor,
            descriptor: convDescriptor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that represents a batch normalization layer.
class BatchNormLayer {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        descriptor: SWBatchNormLayerDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        let scaleBiasShape = InputShape.create(
            batchSize: 1,
            numChannels: descriptor.numChannels,
            nnYLen: 1,
            nnXLen: 1)

        let mergedScaleData = Data(
            floatsNoCopy: descriptor.mergedScale,
            shape: scaleBiasShape)

        let mergedBiasData = Data(
            floatsNoCopy: descriptor.mergedBias,
            shape: scaleBiasShape)

        let scaleTensor = graph.constant(
            mergedScaleData,
            shape: scaleBiasShape,
            dataType: sourceTensor.dataType)

        let biasTensor = graph.constant(
            mergedBiasData,
            shape: scaleBiasShape,
            dataType: sourceTensor.dataType)

        let scaled = graph.multiplication(
            sourceTensor,
            scaleTensor,
            name: nil)

        let normalized = graph.addition(
            scaled,
            biasTensor,
            name: nil)

        // Skip mask multiplication when all mask values are 1
        if optimizeIdentityMask {
            resultTensor = normalized
        } else {
            resultTensor = graph.multiplication(
                normalized,
                maskTensor,
                name: nil)
        }

        assert(resultTensor.shape?.count == 4)
    }
}

/// A structure that represents an activation layer
struct ActivationLayer {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        activationKind: ActivationKind
    ) {
        switch activationKind {
        case .relu:
            resultTensor = graph.reLU(with: sourceTensor, name: nil)
        case .mish:
            resultTensor = graph.mish(tensor: sourceTensor)
        default:
            resultTensor = sourceTensor
        }

        assert(resultTensor.shape == sourceTensor.shape)
    }
}

/// A structure representing a matrix multiplication layer.
struct MatMulLayer {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        descriptor: SWMatMulLayerDesc,
        sourceTensor: MPSGraphTensor
    ) {
        assert(
            (sourceTensor.shape?.count == 4) || (sourceTensor.shape?[1] == descriptor.inChannels))
        assert(
            (sourceTensor.shape?.count == 2) || (sourceTensor.shape?[1] == descriptor.inChannels))

        let weightsShape = [
            descriptor.inChannels,
            descriptor.outChannels,
        ]

        let weightsData = Data(
            floatsNoCopy: descriptor.weights,
            shape: weightsShape)

        let weightsTensor = graph.constant(
            weightsData,
            shape: weightsShape,
            dataType: sourceTensor.dataType)

        let shape = [-1, descriptor.inChannels]

        let reshapedSource = graph.reshape(
            sourceTensor,
            shape: shape,
            name: nil)

        resultTensor = graph.matrixMultiplication(
            primary: reshapedSource,
            secondary: weightsTensor,
            name: nil)

        assert(resultTensor.shape?.count == 2)
    }
}

/// A structure that performs matrix bias operations
struct MatBiasLayer {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        descriptor: SWMatBiasLayerDesc,
        sourceTensor: MPSGraphTensor
    ) {
        assert(
            (sourceTensor.shape?.count == 2) && (sourceTensor.shape?[1] == descriptor.numChannels))

        let weightsShape = [1, descriptor.numChannels]

        let weightsData = Data(
            floatsNoCopy: descriptor.weights,
            shape: weightsShape)

        let weightsTensor = graph.constant(
            weightsData,
            shape: weightsShape,
            dataType: sourceTensor.dataType)

        resultTensor = graph.addition(
            sourceTensor,
            weightsTensor,
            name: nil)
    }
}

/// A structure that performs bias operations in NC coordinates.
struct AddNCBiasLayer {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        biasTensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        numChannels: NSNumber
    ) {
        let shape = InputShape.create(
            batchSize: -1,
            numChannels: numChannels,
            nnYLen: 1,
            nnXLen: 1)

        assert(biasTensor.shape?[1] == shape[1])

        let reshaped = graph.reshape(biasTensor, shape: shape, name: nil)
        resultTensor = graph.addition(sourceTensor, reshaped, name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == nnYLen)
        assert(resultTensor.shape?[3] == nnXLen)
    }
}

// MARK: - Pooling Layers

/// A structure that represents a global pooling layer
struct GlobalPoolingLayer {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        optimizeIdentityMask: Bool = false
    ) {
        let hwAxes = InputShape.getHWAxes()
        let channelAxis = InputShape.getChannelAxis()

        let sumTensor = graph.reductionSum(
            with: sourceTensor,
            axes: hwAxes,
            name: nil)

        let meanTensor = graph.division(sumTensor, maskSumTensor, name: nil)

        let meanMaskTensor = graph.multiplication(
            meanTensor,
            maskSumSqrtS14M01Tensor,
            name: nil)

        let maxTensor: MPSGraphTensor
        if optimizeIdentityMask {
            // When all mask values are 1, directly compute max without mask adjustment
            maxTensor = graph.reductionMaximum(
                with: sourceTensor,
                axes: hwAxes,
                name: nil)
        } else {
            // Mask out invalid positions by subtracting 1 (making them very negative)
            let oneTensor = graph.constant(1.0, dataType: sourceTensor.dataType)
            let maskM1Tensor = graph.subtraction(maskTensor, oneTensor, name: nil)
            let addition = graph.addition(sourceTensor, maskM1Tensor, name: nil)

            maxTensor = graph.reductionMaximum(
                with: addition,
                axes: hwAxes,
                name: nil)
        }

        resultTensor = graph.concatTensors(
            [
                meanTensor,
                meanMaskTensor,
                maxTensor,
            ],
            dimension: channelAxis,
            name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == 1)
        assert(resultTensor.shape?[3] == 1)
    }
}

/// A structure that represents a layer that performs global pooling on the input tensor
struct GlobalPoolingValueLayer {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor
    ) {
        let hwAxes = InputShape.getHWAxes()
        let channelAxis = InputShape.getChannelAxis()

        let sumTensor = graph.reductionSum(
            with: sourceTensor,
            axes: hwAxes,
            name: nil)

        let meanTensor = graph.division(sumTensor, maskSumTensor, name: nil)

        let meanMaskTensor = graph.multiplication(
            meanTensor,
            maskSumSqrtS14M01Tensor,
            name: nil)

        let meanMaskSquareTensor = graph.multiplication(
            meanTensor,
            maskSumSqrtS14M01SquareS01Tensor,
            name: nil)

        resultTensor = graph.concatTensors(
            [
                meanTensor,
                meanMaskTensor,
                meanMaskSquareTensor,
            ],
            dimension: channelAxis,
            name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == 1)
        assert(resultTensor.shape?[3] == 1)
    }
}

// MARK: - Block Descriptors

/// Base class for block descriptors
public class BlockDescriptor {
}

/// A class that represents a residual block.
public class SWResidualBlockDesc: BlockDescriptor {
    let preBN: SWBatchNormLayerDesc
    let preActivation: ActivationKind
    let regularConv: SWConvLayerDesc
    let midBN: SWBatchNormLayerDesc
    let midActivation: ActivationKind
    let finalConv: SWConvLayerDesc

    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        regularConv: SWConvLayerDesc,
        midBN: SWBatchNormLayerDesc,
        midActivation: ActivationKind,
        finalConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

public func createSWResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    regularConv: SWConvLayerDesc,
    midBN: SWBatchNormLayerDesc,
    midActivation: ActivationKind,
    finalConv: SWConvLayerDesc
) -> SWResidualBlockDesc {
    return SWResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        regularConv: regularConv,
        midBN: midBN,
        midActivation: midActivation,
        finalConv: finalConv)
}

/// A class that represents a residual block with global pooling.
public class SWGlobalPoolingResidualBlockDesc: BlockDescriptor {
    let preBN: SWBatchNormLayerDesc
    let preActivation: ActivationKind
    let regularConv: SWConvLayerDesc
    let gpoolConv: SWConvLayerDesc
    let gpoolBN: SWBatchNormLayerDesc
    let gpoolActivation: ActivationKind
    let gpoolToBiasMul: SWMatMulLayerDesc
    let midBN: SWBatchNormLayerDesc
    let midActivation: ActivationKind
    let finalConv: SWConvLayerDesc

    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        regularConv: SWConvLayerDesc,
        gpoolConv: SWConvLayerDesc,
        gpoolBN: SWBatchNormLayerDesc,
        gpoolActivation: ActivationKind,
        gpoolToBiasMul: SWMatMulLayerDesc,
        midBN: SWBatchNormLayerDesc,
        midActivation: ActivationKind,
        finalConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.gpoolConv = gpoolConv
        self.gpoolBN = gpoolBN
        self.gpoolActivation = gpoolActivation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

public func createSWGlobalPoolingResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    regularConv: SWConvLayerDesc,
    gpoolConv: SWConvLayerDesc,
    gpoolBN: SWBatchNormLayerDesc,
    gpoolActivation: ActivationKind,
    gpoolToBiasMul: SWMatMulLayerDesc,
    midBN: SWBatchNormLayerDesc,
    midActivation: ActivationKind,
    finalConv: SWConvLayerDesc
) -> SWGlobalPoolingResidualBlockDesc {
    return SWGlobalPoolingResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        regularConv: regularConv,
        gpoolConv: gpoolConv,
        gpoolBN: gpoolBN,
        gpoolActivation: gpoolActivation,
        gpoolToBiasMul: gpoolToBiasMul,
        midBN: midBN,
        midActivation: midActivation,
        finalConv: finalConv)
}

/// A class that represents a nested bottleneck residual block
public class SWNestedBottleneckResidualBlockDesc: BlockDescriptor {
    let preBN: SWBatchNormLayerDesc
    let preActivation: ActivationKind
    let preConv: SWConvLayerDesc
    let blockDescriptors: [BlockDescriptor]
    let postBN: SWBatchNormLayerDesc
    let postActivation: ActivationKind
    let postConv: SWConvLayerDesc

    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        preConv: SWConvLayerDesc,
        blockDescriptors: [BlockDescriptor],
        postBN: SWBatchNormLayerDesc,
        postActivation: ActivationKind,
        postConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.preConv = preConv
        self.blockDescriptors = blockDescriptors
        self.postBN = postBN
        self.postActivation = postActivation
        self.postConv = postConv
    }
}

public func createSWNestedBottleneckResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    preConv: SWConvLayerDesc,
    blockDescriptors: [BlockDescriptor],
    postBN: SWBatchNormLayerDesc,
    postActivation: ActivationKind,
    postConv: SWConvLayerDesc
) -> SWNestedBottleneckResidualBlockDesc {
    return SWNestedBottleneckResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        preConv: preConv,
        blockDescriptors: blockDescriptors,
        postBN: postBN,
        postActivation: postActivation,
        postConv: postConv)
}

public class BlockDescriptorBuilder {
    public var blockDescriptors: [BlockDescriptor] = []

    init() {}

    public func enque(with descriptor: BlockDescriptor) {
        blockDescriptors.append(descriptor)
    }
}

public func createBlockDescriptorBuilder() -> BlockDescriptorBuilder {
    return BlockDescriptorBuilder()
}

// MARK: - Block Implementations

/// A class that represents a Residual Block layer
class ResidualBlock {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        descriptor: SWResidualBlockDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        let preBN = BatchNormLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.preBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let preActivation = ActivationLayer(
            graph: graph,
            sourceTensor: preBN.resultTensor,
            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.regularConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let midBN = BatchNormLayer(
            graph: graph,
            sourceTensor: regularConv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.midBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let midActivation = ActivationLayer(
            graph: graph,
            sourceTensor: midBN.resultTensor,
            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(
            graph: graph,
            sourceTensor: midActivation.resultTensor,
            descriptor: descriptor.finalConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        resultTensor = graph.addition(
            sourceTensor,
            finalConv.resultTensor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class representing a residual block with global pooling
class GlobalPoolingResidualBlock {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        descriptor: SWGlobalPoolingResidualBlockDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        let maskSum = MaskSumLayer(tensor: maskSumTensor)
        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(tensor: maskSumSqrtS14M01Tensor)

        let preBN = BatchNormLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.preBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let preActivation = ActivationLayer(
            graph: graph,
            sourceTensor: preBN.resultTensor,
            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.regularConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let gpoolConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.gpoolConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let gpoolBN = BatchNormLayer(
            graph: graph,
            sourceTensor: gpoolConv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.gpoolBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let gpoolActivation = ActivationLayer(
            graph: graph,
            sourceTensor: gpoolBN.resultTensor,
            activationKind: descriptor.gpoolActivation)

        let gpoolConcat = GlobalPoolingLayer(
            graph: graph,
            sourceTensor: gpoolActivation.resultTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            optimizeIdentityMask: optimizeIdentityMask)

        assert(gpoolConcat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels)

        let gpoolToBiasMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.gpoolToBiasMul,
            sourceTensor: gpoolConcat.resultTensor)

        let added = AddNCBiasLayer(
            graph: graph,
            sourceTensor: regularConv.resultTensor,
            biasTensor: gpoolToBiasMul.resultTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.gpoolToBiasMul.outChannels)

        let midBN = BatchNormLayer(
            graph: graph,
            sourceTensor: added.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.midBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let midActivation = ActivationLayer(
            graph: graph,
            sourceTensor: midBN.resultTensor,
            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(
            graph: graph,
            sourceTensor: midActivation.resultTensor,
            descriptor: descriptor.finalConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        resultTensor = graph.addition(
            sourceTensor,
            finalConv.resultTensor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// A structure that represents a block stack
struct BlockStack {
    let resultTensor: MPSGraphTensor

    static func processBlockDescriptors(
        _ graph: MPSGraph,
        _ sourceTensor: MPSGraphTensor,
        _ maskTensor: MPSGraphTensor,
        _ maskSumTensor: MPSGraphTensor,
        _ maskSumSqrtS14M01Tensor: MPSGraphTensor,
        _ blockDescriptors: [BlockDescriptor],
        _ index: Int,
        _ nnXLen: NSNumber,
        _ nnYLen: NSNumber,
        _ optimizeIdentityMask: Bool
    ) -> MPSGraphTensor {
        guard index < blockDescriptors.count else {
            return sourceTensor
        }

        let blockDescriptor = blockDescriptors[index]
        let blockInput: MPSGraphTensor

        switch blockDescriptor {
        case let globalPoolingDescriptor as SWGlobalPoolingResidualBlockDesc:
            let globalPooling = GlobalPoolingResidualBlock(
                graph: graph,
                sourceTensor: sourceTensor,
                maskTensor: maskTensor,
                maskSumTensor: maskSumTensor,
                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                descriptor: globalPoolingDescriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen,
                optimizeIdentityMask: optimizeIdentityMask)

            blockInput = globalPooling.resultTensor
        case let nestedBottleneckDescriptor as SWNestedBottleneckResidualBlockDesc:
            let nestedBottleneck = NestedBottleneckResidualBlock(
                graph: graph,
                sourceTensor: sourceTensor,
                maskTensor: maskTensor,
                maskSumTensor: maskSumTensor,
                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                descriptor: nestedBottleneckDescriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen,
                optimizeIdentityMask: optimizeIdentityMask)

            blockInput = nestedBottleneck.resultTensor
        case let residualBlockDescriptor as SWResidualBlockDesc:
            let ordinary = ResidualBlock(
                graph: graph,
                sourceTensor: sourceTensor,
                maskTensor: maskTensor,
                descriptor: residualBlockDescriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen,
                optimizeIdentityMask: optimizeIdentityMask)

            blockInput = ordinary.resultTensor
        default:
            blockInput = sourceTensor
        }

        return processBlockDescriptors(
            graph,
            blockInput,
            maskTensor,
            maskSumTensor,
            maskSumSqrtS14M01Tensor,
            blockDescriptors,
            index + 1,
            nnXLen,
            nnYLen,
            optimizeIdentityMask)
    }

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        blockDescriptors: [BlockDescriptor],
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        resultTensor = BlockStack.processBlockDescriptors(
            graph,
            sourceTensor,
            maskTensor,
            maskSumTensor,
            maskSumSqrtS14M01Tensor,
            blockDescriptors,
            0,
            nnXLen,
            nnYLen,
            optimizeIdentityMask)
    }
}

/// A structure that represents a nested bottleneck residual block
struct NestedBottleneckResidualBlock {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        descriptor: SWNestedBottleneckResidualBlockDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        let preBN = BatchNormLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.preBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let preActivation = ActivationLayer(
            graph: graph,
            sourceTensor: preBN.resultTensor,
            activationKind: descriptor.preActivation)

        let preConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.preConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let blocks = BlockStack(
            graph: graph,
            sourceTensor: preConv.resultTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSumTensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
            blockDescriptors: descriptor.blockDescriptors,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let postBN = BatchNormLayer(
            graph: graph,
            sourceTensor: blocks.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.postBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let postActivation = ActivationLayer(
            graph: graph,
            sourceTensor: postBN.resultTensor,
            activationKind: descriptor.postActivation)

        let postConv = ConvLayer(
            graph: graph,
            sourceTensor: postActivation.resultTensor,
            descriptor: descriptor.postConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        resultTensor = graph.addition(
            sourceTensor,
            postConv.resultTensor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

// MARK: - SGF Metadata Encoder

/// Class representing the description of the SGF Metadata Encoder.
public class SWSGFMetadataEncoderDesc {
    let version: Int
    let numInputMetaChannels: Int
    let mul1: SWMatMulLayerDesc
    let bias1: SWMatBiasLayerDesc
    let act1: ActivationKind
    let mul2: SWMatMulLayerDesc
    let bias2: SWMatBiasLayerDesc
    let act2: ActivationKind
    let mul3: SWMatMulLayerDesc

    init(
        version: Int,
        numInputMetaChannels: Int,
        mul1: SWMatMulLayerDesc,
        bias1: SWMatBiasLayerDesc,
        act1: ActivationKind,
        mul2: SWMatMulLayerDesc,
        bias2: SWMatBiasLayerDesc,
        act2: ActivationKind,
        mul3: SWMatMulLayerDesc
    ) {
        self.version = version
        self.numInputMetaChannels = numInputMetaChannels
        self.mul1 = mul1
        self.bias1 = bias1
        self.act1 = act1
        self.mul2 = mul2
        self.bias2 = bias2
        self.act2 = act2
        self.mul3 = mul3
    }
}

public func createSWSGFMetadataEncoderDesc(
    version: Int32,
    numInputMetaChannels: Int32,
    mul1: SWMatMulLayerDesc,
    bias1: SWMatBiasLayerDesc,
    act1: ActivationKind,
    mul2: SWMatMulLayerDesc,
    bias2: SWMatBiasLayerDesc,
    act2: ActivationKind,
    mul3: SWMatMulLayerDesc
) -> SWSGFMetadataEncoderDesc? {
    return SWSGFMetadataEncoderDesc(
        version: Int(version),
        numInputMetaChannels: Int(numInputMetaChannels),
        mul1: mul1,
        bias1: bias1,
        act1: act1,
        mul2: mul2,
        bias2: bias2,
        act2: act2,
        mul3: mul3)
}

/// A class that encodes SGF metadata.
class SGFMetadataEncoder {
    let resultTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        descriptor: SWSGFMetadataEncoderDesc,
        sourceTensor: MPSGraphTensor
    ) {
        let mul1 = MatMulLayer(
            graph: graph,
            descriptor: descriptor.mul1,
            sourceTensor: sourceTensor)

        let bias1 = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.bias1,
            sourceTensor: mul1.resultTensor)

        let act1 = ActivationLayer(
            graph: graph,
            sourceTensor: bias1.resultTensor,
            activationKind: descriptor.act1)

        let mul2 = MatMulLayer(
            graph: graph,
            descriptor: descriptor.mul2,
            sourceTensor: act1.resultTensor)

        let bias2 = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.bias2,
            sourceTensor: mul2.resultTensor)

        let act2 = ActivationLayer(
            graph: graph,
            sourceTensor: bias2.resultTensor,
            activationKind: descriptor.act2)

        let mul3 = MatMulLayer(
            graph: graph,
            descriptor: descriptor.mul3,
            sourceTensor: act2.resultTensor)

        resultTensor = mul3.resultTensor

        assert(resultTensor.shape?.count == 2)
    }
}

// MARK: - Trunk

/// A class that describes a trunk for a neural network
public class SWTrunkDesc {
    let version: Int
    let trunkNumChannels: NSNumber
    let midNumChannels: NSNumber
    let regularNumChannels: NSNumber
    let gpoolNumChannels: NSNumber
    let initialConv: SWConvLayerDesc
    let initialMatMul: SWMatMulLayerDesc
    let sgfMetadataEncoder: SWSGFMetadataEncoderDesc?
    let blockDescriptors: [BlockDescriptor]
    let trunkTipBN: SWBatchNormLayerDesc
    let trunkTipActivation: ActivationKind

    init(
        version: Int,
        trunkNumChannels: NSNumber,
        midNumChannels: NSNumber,
        regularNumChannels: NSNumber,
        gpoolNumChannels: NSNumber,
        initialConv: SWConvLayerDesc,
        initialMatMul: SWMatMulLayerDesc,
        sgfMetadataEncoder: SWSGFMetadataEncoderDesc?,
        blockDescriptors: [BlockDescriptor],
        trunkTipBN: SWBatchNormLayerDesc,
        trunkTipActivation: ActivationKind
    ) {
        self.version = version
        self.trunkNumChannels = trunkNumChannels
        self.midNumChannels = midNumChannels
        self.regularNumChannels = regularNumChannels
        self.gpoolNumChannels = gpoolNumChannels
        self.initialConv = initialConv
        self.initialMatMul = initialMatMul
        self.sgfMetadataEncoder = sgfMetadataEncoder
        self.blockDescriptors = blockDescriptors
        self.trunkTipBN = trunkTipBN
        self.trunkTipActivation = trunkTipActivation
    }
}

public func createSWTrunkDesc(
    version: Int32,
    trunkNumChannels: Int32,
    midNumChannels: Int32,
    regularNumChannels: Int32,
    gpoolNumChannels: Int32,
    initialConv: SWConvLayerDesc,
    initialMatMul: SWMatMulLayerDesc,
    sgfMetadataEncoder: SWSGFMetadataEncoderDesc?,
    blockDescriptors: [BlockDescriptor],
    trunkTipBN: SWBatchNormLayerDesc,
    trunkTipActivation: ActivationKind
) -> SWTrunkDesc {
    return SWTrunkDesc(
        version: Int(version),
        trunkNumChannels: trunkNumChannels as NSNumber,
        midNumChannels: midNumChannels as NSNumber,
        regularNumChannels: regularNumChannels as NSNumber,
        gpoolNumChannels: gpoolNumChannels as NSNumber,
        initialConv: initialConv,
        initialMatMul: initialMatMul,
        sgfMetadataEncoder: sgfMetadataEncoder,
        blockDescriptors: blockDescriptors,
        trunkTipBN: trunkTipBN,
        trunkTipActivation: trunkTipActivation)
}

/// A structure representing a ResNet trunk for a neural network
struct Trunk {
    let resultTensor: MPSGraphTensor

    static func getBlockSourceTensor(
        graph: MPSGraph,
        descriptor: SWSGFMetadataEncoderDesc?,
        initialAdd: AddNCBiasLayer,
        inputMetaTensor: MPSGraphTensor?,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        numChannels: NSNumber
    ) -> MPSGraphTensor {
        var blockSourceTensor: MPSGraphTensor

        if let inputMetaTensor,
            let descriptor, descriptor.numInputMetaChannels > 0
        {
            let encoded = SGFMetadataEncoder(
                graph: graph,
                descriptor: descriptor,
                sourceTensor: inputMetaTensor)

            let encodedAdd = AddNCBiasLayer(
                graph: graph,
                sourceTensor: initialAdd.resultTensor,
                biasTensor: encoded.resultTensor,
                nnXLen: nnXLen,
                nnYLen: nnYLen,
                numChannels: numChannels)

            blockSourceTensor = encodedAdd.resultTensor
        } else {
            blockSourceTensor = initialAdd.resultTensor
        }

        return blockSourceTensor
    }

    init(
        graph: MPSGraph,
        descriptor: SWTrunkDesc,
        inputTensor: MPSGraphTensor,
        inputGlobalTensor: MPSGraphTensor,
        inputMetaTensor: MPSGraphTensor?,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        let initialConv = ConvLayer(
            graph: graph,
            sourceTensor: inputTensor,
            descriptor: descriptor.initialConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let initialMatMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.initialMatMul,
            sourceTensor: inputGlobalTensor)

        let initialAdd = AddNCBiasLayer(
            graph: graph,
            sourceTensor: initialConv.resultTensor,
            biasTensor: initialMatMul.resultTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.initialMatMul.outChannels)

        let blockSourceTensor = Trunk.getBlockSourceTensor(
            graph: graph,
            descriptor: descriptor.sgfMetadataEncoder,
            initialAdd: initialAdd,
            inputMetaTensor: inputMetaTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.initialMatMul.outChannels)

        let blocks = BlockStack(
            graph: graph,
            sourceTensor: blockSourceTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSumTensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
            blockDescriptors: descriptor.blockDescriptors,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let trunkTipBN = BatchNormLayer(
            graph: graph,
            sourceTensor: blocks.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.trunkTipBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let trunkTipActivation = ActivationLayer(
            graph: graph,
            sourceTensor: trunkTipBN.resultTensor,
            activationKind: descriptor.trunkTipActivation)

        resultTensor = trunkTipActivation.resultTensor

        assert(resultTensor.shape?.count == 4)
    }
}

// MARK: - Policy Head

/// A class that describes a policy head for a neural network
public struct SWPolicyHeadDesc {
    let version: Int
    let p1Conv: SWConvLayerDesc
    let g1Conv: SWConvLayerDesc
    let g1BN: SWBatchNormLayerDesc
    let g1Activation: ActivationKind
    let gpoolToBiasMul: SWMatMulLayerDesc
    let p1BN: SWBatchNormLayerDesc
    let p1Activation: ActivationKind
    let p2Conv: SWConvLayerDesc
    let gpoolToPassMul: SWMatMulLayerDesc
    let gpoolToPassBias: SWMatBiasLayerDesc?
    let passActivation: ActivationKind?
    let gpoolToPassMul2: SWMatMulLayerDesc?

    init(
        version: Int,
        p1Conv: SWConvLayerDesc,
        g1Conv: SWConvLayerDesc,
        g1BN: SWBatchNormLayerDesc,
        g1Activation: ActivationKind,
        gpoolToBiasMul: SWMatMulLayerDesc,
        p1BN: SWBatchNormLayerDesc,
        p1Activation: ActivationKind,
        p2Conv: SWConvLayerDesc,
        gpoolToPassMul: SWMatMulLayerDesc,
        gpoolToPassBias: SWMatBiasLayerDesc?,
        passActivation: ActivationKind?,
        gpoolToPassMul2: SWMatMulLayerDesc?
    ) {
        self.version = version
        self.p1Conv = p1Conv
        self.g1Conv = g1Conv
        self.g1BN = g1BN
        self.g1Activation = g1Activation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.p1BN = p1BN
        self.p1Activation = p1Activation
        self.p2Conv = p2Conv
        self.gpoolToPassMul = gpoolToPassMul
        self.gpoolToPassBias = gpoolToPassBias
        self.passActivation = passActivation
        self.gpoolToPassMul2 = gpoolToPassMul2

        assert(
            (version >= 15)
                || ((gpoolToPassBias == nil) && (passActivation == nil) && (gpoolToPassMul2 == nil))
        )
        assert(
            (version < 15)
                || ((gpoolToPassBias != nil) && (passActivation != nil) && (gpoolToPassMul2 != nil))
        )
    }
}

public func createSWPolicyHeadDesc(
    version: Int32,
    p1Conv: SWConvLayerDesc,
    g1Conv: SWConvLayerDesc,
    g1BN: SWBatchNormLayerDesc,
    g1Activation: ActivationKind,
    gpoolToBiasMul: SWMatMulLayerDesc,
    p1BN: SWBatchNormLayerDesc,
    p1Activation: ActivationKind,
    p2Conv: SWConvLayerDesc,
    gpoolToPassMul: SWMatMulLayerDesc,
    gpoolToPassBias: SWMatBiasLayerDesc,
    passActivation: ActivationKind,
    gpoolToPassMul2: SWMatMulLayerDesc
) -> SWPolicyHeadDesc {
    if version >= 15 {
        return SWPolicyHeadDesc(
            version: Int(version),
            p1Conv: p1Conv,
            g1Conv: g1Conv,
            g1BN: g1BN,
            g1Activation: g1Activation,
            gpoolToBiasMul: gpoolToBiasMul,
            p1BN: p1BN,
            p1Activation: p1Activation,
            p2Conv: p2Conv,
            gpoolToPassMul: gpoolToPassMul,
            gpoolToPassBias: gpoolToPassBias,
            passActivation: passActivation,
            gpoolToPassMul2: gpoolToPassMul2)
    } else {
        return SWPolicyHeadDesc(
            version: Int(version),
            p1Conv: p1Conv,
            g1Conv: g1Conv,
            g1BN: g1BN,
            g1Activation: g1Activation,
            gpoolToBiasMul: gpoolToBiasMul,
            p1BN: p1BN,
            p1Activation: p1Activation,
            p2Conv: p2Conv,
            gpoolToPassMul: gpoolToPassMul,
            gpoolToPassBias: nil,
            passActivation: nil,
            gpoolToPassMul2: nil)
    }
}

/// A structure that represents a policy head of a neural network.
struct PolicyHead {
    let policyTensor: MPSGraphTensor
    let policyPassTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        descriptor: SWPolicyHeadDesc,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        let p1Conv = ConvLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            descriptor: descriptor.p1Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let g1Conv = ConvLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            descriptor: descriptor.g1Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let g1BN = BatchNormLayer(
            graph: graph,
            sourceTensor: g1Conv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.g1BN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let g1Activation = ActivationLayer(
            graph: graph,
            sourceTensor: g1BN.resultTensor,
            activationKind: descriptor.g1Activation)

        let g1Concat = GlobalPoolingLayer(
            graph: graph,
            sourceTensor: g1Activation.resultTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSumTensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
            optimizeIdentityMask: optimizeIdentityMask)

        assert(g1Concat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels)

        let gpoolToBiasMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.gpoolToBiasMul,
            sourceTensor: g1Concat.resultTensor)

        let added = AddNCBiasLayer(
            graph: graph,
            sourceTensor: p1Conv.resultTensor,
            biasTensor: gpoolToBiasMul.resultTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.gpoolToBiasMul.outChannels)

        let p1BN = BatchNormLayer(
            graph: graph,
            sourceTensor: added.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.p1BN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let p1Activation = ActivationLayer(
            graph: graph,
            sourceTensor: p1BN.resultTensor,
            activationKind: descriptor.p1Activation)

        let p2Conv = ConvLayer(
            graph: graph,
            sourceTensor: p1Activation.resultTensor,
            descriptor: descriptor.p2Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        policyTensor = p2Conv.resultTensor

        assert(g1Concat.resultTensor.shape?[1] == descriptor.gpoolToPassMul.inChannels)

        let gpoolToPassMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.gpoolToPassMul,
            sourceTensor: g1Concat.resultTensor)

        if let gpoolToPassBias = descriptor.gpoolToPassBias,
            let passActivation = descriptor.passActivation,
            let gpoolToPassMul2 = descriptor.gpoolToPassMul2
        {
            assert(descriptor.version >= 15)

            let gpoolToPassBiasLayer = MatBiasLayer(
                graph: graph,
                descriptor: gpoolToPassBias,
                sourceTensor: gpoolToPassMul.resultTensor)

            let passActivationLayer = ActivationLayer(
                graph: graph,
                sourceTensor: gpoolToPassBiasLayer.resultTensor,
                activationKind: passActivation)

            let gpoolToPassMul2Layer = MatMulLayer(
                graph: graph,
                descriptor: gpoolToPassMul2,
                sourceTensor: passActivationLayer.resultTensor)

            policyPassTensor = gpoolToPassMul2Layer.resultTensor
        } else {
            assert(descriptor.version < 15)
            policyPassTensor = gpoolToPassMul.resultTensor
        }

        assert(policyTensor.shape?.count == 4)
        assert(policyPassTensor.shape?.count == 2)
    }
}

// MARK: - Value Head

/// A struct that describes the value head of a neural network
public struct SWValueHeadDesc {
    let version: Int
    let v1Conv: SWConvLayerDesc
    let v1BN: SWBatchNormLayerDesc
    let v1Activation: ActivationKind
    let v2Mul: SWMatMulLayerDesc
    let v2Bias: SWMatBiasLayerDesc
    let v2Activation: ActivationKind
    let v3Mul: SWMatMulLayerDesc
    let v3Bias: SWMatBiasLayerDesc
    let sv3Mul: SWMatMulLayerDesc
    let sv3Bias: SWMatBiasLayerDesc
    let vOwnershipConv: SWConvLayerDesc

    init(
        version: Int,
        v1Conv: SWConvLayerDesc,
        v1BN: SWBatchNormLayerDesc,
        v1Activation: ActivationKind,
        v2Mul: SWMatMulLayerDesc,
        v2Bias: SWMatBiasLayerDesc,
        v2Activation: ActivationKind,
        v3Mul: SWMatMulLayerDesc,
        v3Bias: SWMatBiasLayerDesc,
        sv3Mul: SWMatMulLayerDesc,
        sv3Bias: SWMatBiasLayerDesc,
        vOwnershipConv: SWConvLayerDesc
    ) {
        self.version = version
        self.v1Conv = v1Conv
        self.v1BN = v1BN
        self.v1Activation = v1Activation
        self.v2Mul = v2Mul
        self.v2Bias = v2Bias
        self.v2Activation = v2Activation
        self.v3Mul = v3Mul
        self.v3Bias = v3Bias
        self.sv3Mul = sv3Mul
        self.sv3Bias = sv3Bias
        self.vOwnershipConv = vOwnershipConv
    }
}

public func createSWValueHeadDesc(
    version: Int32,
    v1Conv: SWConvLayerDesc,
    v1BN: SWBatchNormLayerDesc,
    v1Activation: ActivationKind,
    v2Mul: SWMatMulLayerDesc,
    v2Bias: SWMatBiasLayerDesc,
    v2Activation: ActivationKind,
    v3Mul: SWMatMulLayerDesc,
    v3Bias: SWMatBiasLayerDesc,
    sv3Mul: SWMatMulLayerDesc,
    sv3Bias: SWMatBiasLayerDesc,
    vOwnershipConv: SWConvLayerDesc
) -> SWValueHeadDesc {
    return SWValueHeadDesc(
        version: Int(version),
        v1Conv: v1Conv,
        v1BN: v1BN,
        v1Activation: v1Activation,
        v2Mul: v2Mul,
        v2Bias: v2Bias,
        v2Activation: v2Activation,
        v3Mul: v3Mul,
        v3Bias: v3Bias,
        sv3Mul: sv3Mul,
        sv3Bias: sv3Bias,
        vOwnershipConv: vOwnershipConv)
}

/// A structure that creates a value head for the neural network
struct ValueHead {
    let valueTensor: MPSGraphTensor
    let scoreValueTensor: MPSGraphTensor
    let ownershipTensor: MPSGraphTensor

    init(
        graph: MPSGraph,
        descriptor: SWValueHeadDesc,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        optimizeIdentityMask: Bool = false
    ) {
        let v1Conv = ConvLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            descriptor: descriptor.v1Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let v1BN = BatchNormLayer(
            graph: graph,
            sourceTensor: v1Conv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.v1BN,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            optimizeIdentityMask: optimizeIdentityMask)

        let v1Activation = ActivationLayer(
            graph: graph,
            sourceTensor: v1BN.resultTensor,
            activationKind: descriptor.v1Activation)

        let v1Mean =
            GlobalPoolingValueLayer(
                graph: graph,
                sourceTensor: v1Activation.resultTensor,
                maskSumTensor: maskSumTensor,
                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01Tensor)

        assert(v1Mean.resultTensor.shape?[1] == descriptor.v2Mul.inChannels)

        let v2Mul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.v2Mul,
            sourceTensor: v1Mean.resultTensor)

        let v2Bias = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.v2Bias,
            sourceTensor: v2Mul.resultTensor)

        let v2Activation = ActivationLayer(
            graph: graph,
            sourceTensor: v2Bias.resultTensor,
            activationKind: descriptor.v2Activation)

        let v3Mul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.v3Mul,
            sourceTensor: v2Activation.resultTensor)

        let v3Bias = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.v3Bias,
            sourceTensor: v3Mul.resultTensor)

        let sv3Mul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.sv3Mul,
            sourceTensor: v2Activation.resultTensor)

        let sv3Bias = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.sv3Bias,
            sourceTensor: sv3Mul.resultTensor)

        let vOwnershipConv = ConvLayer(
            graph: graph,
            sourceTensor: v1Activation.resultTensor,
            descriptor: descriptor.vOwnershipConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        valueTensor = v3Bias.resultTensor
        scoreValueTensor = sv3Bias.resultTensor
        ownershipTensor = vOwnershipConv.resultTensor

        assert(valueTensor.shape?.count == 2)
        assert(scoreValueTensor.shape?.count == 2)
        assert(ownershipTensor.shape?.count == 4)
    }
}

// MARK: - Model Descriptor

/// A struct that describes a neural network model used for playing the game of Go.
public struct SWModelDesc {
    let version: Int
    let name: String
    let numInputChannels: NSNumber
    let numInputGlobalChannels: NSNumber
    let numInputMetaChannels: NSNumber
    let numValueChannels: NSNumber
    let numScoreValueChannels: NSNumber
    let numOwnershipChannels: NSNumber
    let numPolicyChannels: NSNumber
    let trunk: SWTrunkDesc
    let policyHead: SWPolicyHeadDesc
    let valueHead: SWValueHeadDesc

    init(
        version: Int,
        name: String,
        numInputChannels: NSNumber,
        numInputGlobalChannels: NSNumber,
        numInputMetaChannels: NSNumber,
        numValueChannels: NSNumber,
        numScoreValueChannels: NSNumber,
        numOwnershipChannels: NSNumber,
        numPolicyChannels: NSNumber,
        trunk: SWTrunkDesc,
        policyHead: SWPolicyHeadDesc,
        valueHead: SWValueHeadDesc
    ) {
        self.version = version
        self.name = name
        self.numInputChannels = numInputChannels
        self.numInputGlobalChannels = numInputGlobalChannels
        self.numInputMetaChannels = numInputMetaChannels
        self.numValueChannels = numValueChannels
        self.numScoreValueChannels = numScoreValueChannels
        self.numOwnershipChannels = numOwnershipChannels
        self.numPolicyChannels = numPolicyChannels
        self.trunk = trunk
        self.policyHead = policyHead
        self.valueHead = valueHead
    }
}

public func createSWModelDesc(
    version: Int32,
    name: String,
    numInputChannels: Int32,
    numInputGlobalChannels: Int32,
    numInputMetaChannels: Int32,
    numValueChannels: Int32,
    numScoreValueChannels: Int32,
    numOwnershipChannels: Int32,
    numPolicyChannels: Int32,
    trunk: SWTrunkDesc,
    policyHead: SWPolicyHeadDesc,
    valueHead: SWValueHeadDesc
) -> SWModelDesc {
    return SWModelDesc(
        version: Int(version),
        name: name,
        numInputChannels: numInputChannels as NSNumber,
        numInputGlobalChannels: numInputGlobalChannels as NSNumber,
        numInputMetaChannels: numInputMetaChannels as NSNumber,
        numValueChannels: numValueChannels as NSNumber,
        numScoreValueChannels: numScoreValueChannels as NSNumber,
        numOwnershipChannels: numOwnershipChannels as NSNumber,
        numPolicyChannels: numPolicyChannels as NSNumber,
        trunk: trunk,
        policyHead: policyHead,
        valueHead: valueHead)
}

// MARK: - MPSGraph Model (for GPU inference)

/// A structure representing a neural network model for processing Go game states using MPSGraph.
struct MPSGraphModel {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let graph: MPSGraph
    let nnXLen: NSNumber
    let nnYLen: NSNumber
    let version: Int
    let numValueChannels: NSNumber
    let numScoreValueChannels: NSNumber
    let numOwnershipChannels: NSNumber
    let input: InputLayer
    let inputGlobal: InputGlobalLayer
    let inputMeta: InputMetaLayer
    let mask: MaskLayer
    let trunk: Trunk
    let policyHead: PolicyHead
    let valueHead: ValueHead
    let targetTensors: [MPSGraphTensor]

    init(
        device: MTLDevice,
        graph: MPSGraph,
        descriptor: SWModelDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.graph = graph
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.version = descriptor.version
        self.numValueChannels = descriptor.numValueChannels
        self.numScoreValueChannels = descriptor.numScoreValueChannels
        self.numOwnershipChannels = descriptor.numOwnershipChannels

        input = InputLayer(
            graph: graph,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.numInputChannels)

        inputGlobal = InputGlobalLayer(
            graph: graph,
            numGlobalFeatures: descriptor.numInputGlobalChannels)

        inputMeta = InputMetaLayer(
            graph: graph,
            numMetaFeatures: descriptor.numInputMetaChannels)

        mask = MaskLayer(
            graph: graph,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let maskSum = MaskSumLayer(
            graph: graph,
            maskTensor: mask.tensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(
            graph: graph,
            maskSum: maskSum)

        let maskSumSqrtS14M01SquareS01 = MaskSumSqrtS14M01SquareS01Layer(
            graph: graph,
            maskSumSqrtS14M01: maskSumSqrtS14M01)

        trunk = Trunk(
            graph: graph,
            descriptor: descriptor.trunk,
            inputTensor: input.tensor,
            inputGlobalTensor: inputGlobal.tensor,
            inputMetaTensor: inputMeta.tensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        policyHead = PolicyHead(
            graph: graph,
            descriptor: descriptor.policyHead,
            sourceTensor: trunk.resultTensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        valueHead = ValueHead(
            graph: graph,
            descriptor: descriptor.valueHead,
            sourceTensor: trunk.resultTensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        targetTensors = [
            policyHead.policyTensor,
            policyHead.policyPassTensor,
            valueHead.valueTensor,
            valueHead.scoreValueTensor,
            valueHead.ownershipTensor,
        ]
    }

    /// Applies the model to the given input data
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

        let inputShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: numInputChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

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
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: mask.tensor.dataType,
            shape: maskShape)

        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        var maskStrideArray = [
            MemoryLayout<Float32>.size,
            nnXLen.intValue * MemoryLayout<Float32>.size,
            nnYLen.intValue * nnXLen.intValue * MemoryLayout<Float32>.size,
            numInputChannels.intValue * nnYLen.intValue * nnXLen.intValue
                * MemoryLayout<Float32>.size,
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

        assert(fetch[policyHead.policyTensor] != nil)
        assert(fetch[policyHead.policyPassTensor] != nil)
        assert(fetch[valueHead.valueTensor] != nil)
        assert(fetch[valueHead.scoreValueTensor] != nil)
        assert(fetch[valueHead.ownershipTensor] != nil)

        fetch[policyHead.policyTensor]?.mpsndarray().readBytes(policy)
        fetch[policyHead.policyPassTensor]?.mpsndarray().readBytes(policyPass)
        fetch[valueHead.valueTensor]?.mpsndarray().readBytes(value)
        fetch[valueHead.scoreValueTensor]?.mpsndarray().readBytes(scoreValue)
        fetch[valueHead.ownershipTensor]?.mpsndarray().readBytes(ownership)
    }
}

// MARK: - Test Infrastructure

/// Helper struct for testing individual network layers using MPSGraph
struct NetworkTester {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let graph: MPSGraph
    let inputTensor: MPSGraphTensor
    let maskTensor: MPSGraphTensor
    let outputTensor: MPSGraphTensor
    let inputShape: [NSNumber]
    let maskShape: [NSNumber]
    let outputShape: [NSNumber]

    /// Initialize a network tester for testing a single layer
    init(
        device: MTLDevice,
        graph: MPSGraph,
        inputTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        outputTensor: MPSGraphTensor,
        batchSize: NSNumber,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        inChannels: NSNumber,
        outChannels: NSNumber
    ) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.graph = graph
        self.inputTensor = inputTensor
        self.maskTensor = maskTensor
        self.outputTensor = outputTensor
        self.inputShape = InputShape.create(
            batchSize: batchSize,
            numChannels: inChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)
        self.maskShape = InputShape.create(
            batchSize: batchSize,
            numChannels: 1,
            nnYLen: nnYLen,
            nnXLen: nnXLen)
        self.outputShape = InputShape.create(
            batchSize: batchSize,
            numChannels: outChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)
    }

    /// Run the test with given input and mask data, writing results to output
    func run(
        inputPointer: UnsafePointer<Float32>,
        maskPointer: UnsafePointer<Float32>,
        outputPointer: UnsafeMutablePointer<Float32>
    ) {
        let inputDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: inputShape)

        let inputArray = MPSNDArray(
            device: device,
            descriptor: inputDescriptor)

        inputArray.writeBytes(UnsafeMutableRawPointer(mutating: inputPointer))

        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: maskShape)

        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        maskArray.writeBytes(UnsafeMutableRawPointer(mutating: maskPointer))

        let feeds = [
            inputTensor: MPSGraphTensorData(inputArray),
            maskTensor: MPSGraphTensorData(maskArray),
        ]

        let fetch = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [outputTensor],
            targetOperations: nil)

        fetch[outputTensor]?.mpsndarray().readBytes(outputPointer)
    }
}

// MARK: - ConvLayer Test Extension

extension ConvLayer {
    /// Test the convolution layer with given parameters
    static func test(
        descriptor: SWConvLayerDesc,
        batchSize: Int32,
        nnXLen: Int32,
        nnYLen: Int32,
        inputPointer: UnsafePointer<Float32>,
        outputPointer: UnsafeMutablePointer<Float32>
    ) -> Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return false
        }

        let graph = MPSGraph()

        let inputShape = InputShape.create(
            batchSize: -1 as NSNumber,
            numChannels: descriptor.inChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputTensor = graph.placeholder(
            shape: inputShape,
            dataType: .float32,
            name: nil)

        let convLayer = ConvLayer(
            graph: graph,
            sourceTensor: inputTensor,
            descriptor: descriptor,
            nnXLen: nnXLen as NSNumber,
            nnYLen: nnYLen as NSNumber)

        // Run the graph
        let commandQueue = device.makeCommandQueue()!

        let actualInputShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: descriptor.inChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: actualInputShape)

        let inputArray = MPSNDArray(
            device: device,
            descriptor: inputDescriptor)

        inputArray.writeBytes(UnsafeMutableRawPointer(mutating: inputPointer))

        let feeds = [inputTensor: MPSGraphTensorData(inputArray)]

        let fetch = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [convLayer.resultTensor],
            targetOperations: nil)

        fetch[convLayer.resultTensor]?.mpsndarray().readBytes(outputPointer)

        return true
    }
}

// MARK: - BatchNormLayer Test Extension

extension BatchNormLayer {
    /// Test the batch normalization layer with given parameters
    static func test(
        descriptor: SWBatchNormLayerDesc,
        batchSize: Int32,
        nnXLen: Int32,
        nnYLen: Int32,
        inputPointer: UnsafePointer<Float32>,
        maskPointer: UnsafePointer<Float32>,
        outputPointer: UnsafeMutablePointer<Float32>
    ) -> Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return false
        }

        let graph = MPSGraph()

        let inputShape = InputShape.create(
            batchSize: -1 as NSNumber,
            numChannels: descriptor.numChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputTensor = graph.placeholder(
            shape: inputShape,
            dataType: .float32,
            name: nil)

        let maskShape = InputShape.create(
            batchSize: -1 as NSNumber,
            numChannels: 1,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let maskTensor = graph.placeholder(
            shape: maskShape,
            dataType: .float32,
            name: nil)

        let bnLayer = BatchNormLayer(
            graph: graph,
            sourceTensor: inputTensor,
            maskTensor: maskTensor,
            descriptor: descriptor,
            nnXLen: nnXLen as NSNumber,
            nnYLen: nnYLen as NSNumber)

        // Run the graph
        let commandQueue = device.makeCommandQueue()!

        let actualInputShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: descriptor.numChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: actualInputShape)

        let inputArray = MPSNDArray(
            device: device,
            descriptor: inputDescriptor)

        inputArray.writeBytes(UnsafeMutableRawPointer(mutating: inputPointer))

        let actualMaskShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: 1,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: actualMaskShape)

        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        maskArray.writeBytes(UnsafeMutableRawPointer(mutating: maskPointer))

        let feeds = [
            inputTensor: MPSGraphTensorData(inputArray),
            maskTensor: MPSGraphTensorData(maskArray),
        ]

        let fetch = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [bnLayer.resultTensor],
            targetOperations: nil)

        fetch[bnLayer.resultTensor]?.mpsndarray().readBytes(outputPointer)

        return true
    }
}

// MARK: - ResidualBlock Test Extension

extension ResidualBlock {
    /// Test the residual block with given parameters
    static func test(
        descriptor: SWResidualBlockDesc,
        batchSize: Int32,
        nnXLen: Int32,
        nnYLen: Int32,
        inputPointer: UnsafePointer<Float32>,
        maskPointer: UnsafePointer<Float32>,
        outputPointer: UnsafeMutablePointer<Float32>
    ) -> Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return false
        }

        let graph = MPSGraph()

        let inputShape = InputShape.create(
            batchSize: -1 as NSNumber,
            numChannels: descriptor.preBN.numChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputTensor = graph.placeholder(
            shape: inputShape,
            dataType: .float32,
            name: nil)

        let maskShape = InputShape.create(
            batchSize: -1 as NSNumber,
            numChannels: 1,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let maskTensor = graph.placeholder(
            shape: maskShape,
            dataType: .float32,
            name: nil)

        let resBlock = ResidualBlock(
            graph: graph,
            sourceTensor: inputTensor,
            maskTensor: maskTensor,
            descriptor: descriptor,
            nnXLen: nnXLen as NSNumber,
            nnYLen: nnYLen as NSNumber)

        // Run the graph
        let commandQueue = device.makeCommandQueue()!

        let actualInputShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: descriptor.preBN.numChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: actualInputShape)

        let inputArray = MPSNDArray(
            device: device,
            descriptor: inputDescriptor)

        inputArray.writeBytes(UnsafeMutableRawPointer(mutating: inputPointer))

        let actualMaskShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: 1,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: actualMaskShape)

        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        maskArray.writeBytes(UnsafeMutableRawPointer(mutating: maskPointer))

        let feeds = [
            inputTensor: MPSGraphTensorData(inputArray),
            maskTensor: MPSGraphTensorData(maskArray),
        ]

        let fetch = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [resBlock.resultTensor],
            targetOperations: nil)

        fetch[resBlock.resultTensor]?.mpsndarray().readBytes(outputPointer)

        return true
    }
}

// MARK: - GlobalPoolingResidualBlock Test Extension

extension GlobalPoolingResidualBlock {
    /// Test the global pooling residual block with given parameters
    static func test(
        descriptor: SWGlobalPoolingResidualBlockDesc,
        batchSize: Int32,
        nnXLen: Int32,
        nnYLen: Int32,
        inputPointer: UnsafePointer<Float32>,
        maskPointer: UnsafePointer<Float32>,
        outputPointer: UnsafeMutablePointer<Float32>
    ) -> Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return false
        }

        let graph = MPSGraph()

        let inputShape = InputShape.create(
            batchSize: -1 as NSNumber,
            numChannels: descriptor.preBN.numChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputTensor = graph.placeholder(
            shape: inputShape,
            dataType: .float32,
            name: nil)

        let maskShape = InputShape.create(
            batchSize: -1 as NSNumber,
            numChannels: 1,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let maskTensor = graph.placeholder(
            shape: maskShape,
            dataType: .float32,
            name: nil)

        // Compute mask sum and related tensors from mask
        let maskSum = MaskSumLayer(graph: graph, maskTensor: maskTensor)
        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph, maskSum: maskSum)

        let gpoolBlock = GlobalPoolingResidualBlock(
            graph: graph,
            sourceTensor: inputTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            descriptor: descriptor,
            nnXLen: nnXLen as NSNumber,
            nnYLen: nnYLen as NSNumber)

        // Run the graph
        let commandQueue = device.makeCommandQueue()!

        let actualInputShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: descriptor.preBN.numChannels,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let inputDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: actualInputShape)

        let inputArray = MPSNDArray(
            device: device,
            descriptor: inputDescriptor)

        inputArray.writeBytes(UnsafeMutableRawPointer(mutating: inputPointer))

        let actualMaskShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: 1,
            nnYLen: nnYLen as NSNumber,
            nnXLen: nnXLen as NSNumber)

        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: actualMaskShape)

        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        maskArray.writeBytes(UnsafeMutableRawPointer(mutating: maskPointer))

        let feeds = [
            inputTensor: MPSGraphTensorData(inputArray),
            maskTensor: MPSGraphTensorData(maskArray),
        ]

        let fetch = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [gpoolBlock.resultTensor],
            targetOperations: nil)

        fetch[gpoolBlock.resultTensor]?.mpsndarray().readBytes(outputPointer)

        return true
    }
}

// MARK: - Public Test Functions (callable from C++)

/// Test the convolution layer
public func testConvLayer(
    descriptor: SWConvLayerDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    inputPointer: UnsafePointer<Float32>,
    outputPointer: UnsafeMutablePointer<Float32>
) -> Bool {
    return ConvLayer.test(
        descriptor: descriptor,
        batchSize: batchSize,
        nnXLen: nnXLen,
        nnYLen: nnYLen,
        inputPointer: inputPointer,
        outputPointer: outputPointer)
}

/// Test the batch normalization layer
public func testBatchNormLayer(
    descriptor: SWBatchNormLayerDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    inputPointer: UnsafePointer<Float32>,
    maskPointer: UnsafePointer<Float32>,
    outputPointer: UnsafeMutablePointer<Float32>
) -> Bool {
    return BatchNormLayer.test(
        descriptor: descriptor,
        batchSize: batchSize,
        nnXLen: nnXLen,
        nnYLen: nnYLen,
        inputPointer: inputPointer,
        maskPointer: maskPointer,
        outputPointer: outputPointer)
}

/// Test the residual block
public func testResidualBlock(
    descriptor: SWResidualBlockDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    inputPointer: UnsafePointer<Float32>,
    maskPointer: UnsafePointer<Float32>,
    outputPointer: UnsafeMutablePointer<Float32>
) -> Bool {
    return ResidualBlock.test(
        descriptor: descriptor,
        batchSize: batchSize,
        nnXLen: nnXLen,
        nnYLen: nnYLen,
        inputPointer: inputPointer,
        maskPointer: maskPointer,
        outputPointer: outputPointer)
}

/// Test the global pooling residual block
public func testGlobalPoolingResidualBlock(
    descriptor: SWGlobalPoolingResidualBlockDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    inputPointer: UnsafePointer<Float32>,
    maskPointer: UnsafePointer<Float32>,
    outputPointer: UnsafeMutablePointer<Float32>
) -> Bool {
    return GlobalPoolingResidualBlock.test(
        descriptor: descriptor,
        batchSize: batchSize,
        nnXLen: nnXLen,
        nnYLen: nnYLen,
        inputPointer: inputPointer,
        maskPointer: maskPointer,
        outputPointer: outputPointer)
}
