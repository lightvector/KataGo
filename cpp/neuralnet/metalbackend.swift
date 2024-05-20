import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import OSLog

class DefaultDevice {
    static var device = MTLCreateSystemDefaultDevice()!
}

class StandardError: TextOutputStream {
    /// A shared instance of the StandardError class.
    static var instance = StandardError()

    /// Writes the given string to standard error output.
    func write(_ string: String) {
        /// Attempts to write the contents of a Data object containing the UTF8-encoded string to
        /// the standard error file handle.
        try? FileHandle.standardError.write(contentsOf: Data(string.utf8))
    }
}

/// An extension to the Data struct for handling float data with optional FP16 conversion.
extension Data {
    /// Initializes a new Data instance using an UnsafeMutablePointer<Float32>, with optional conversion to FP16 format.
    /// - Parameters:
    ///   - floatsNoCopy: An UnsafeMutablePointer<Float32> containing the float data.
    ///   - shape: An array of NSNumber objects representing the shape of the data.
    init(floatsNoCopy: UnsafeMutablePointer<Float32>,
         shape: [NSNumber]) {
        self.init(bytesNoCopy: floatsNoCopy,
                  count: shape.countBytesOfFloat32(),
                  deallocator: .none)
    }
}

/// Extension to MPSNDArray to convert from MPSGraphTensor, and to read/write bytes from/to UnsafeMutableRawPointer
extension MPSNDArray {
    /// Read bytes from the buffer
    /// - Parameter buffer: The buffer to read
    func readBytes(_ buffer: UnsafeMutableRawPointer) {
        self.readBytes(buffer, strideBytes: nil)
    }

    /// Write bytes to the buffer
    /// - Parameter buffer: The buffer to write
    func writeBytes(_ buffer: UnsafeMutableRawPointer) {
        self.writeBytes(buffer, strideBytes: nil)
    }
}

/// Extension to Array to count number of elements and bytes
extension Array where Element == NSNumber {
    /// Count number of elements
    /// - Returns: Number of elements
    func countElements() -> Int {
        return reduce(1, { $0 * $1.intValue })
    }

    /// Count number of bytes
    /// - Parameter dataType: The data type
    /// - Returns: Number of bytes
    func countBytesOfFloat32() -> Int {
        return countElements() * MemoryLayout<Float32>.size
    }
}

/// Extension to MPSGraph to the mish activation function
extension MPSGraph {
    /// This function applies the Mish activation function on the input tensor `x`. The Mish function is defined as
    /// x * tanh(Softplus(x)), where Softplus(x) is defined as log(1 + exp(min(x, 10.39))) if x < 10.39 and x otherwise.
    /// The threshold of softplus is modified to 10.39, which is different from the original 20. This is because
    /// exp(10.39) = 32532.666936 < 32767.0 < 65504.0, so the result of exp(10.39) can be represented by float16. If the threshold
    /// of softplus is 20, the result of exp(20) is 485165195.40979004, which is out of range of float16.
    /// - Parameter tensor: The input tensor of mish activation function
    /// - Returns: The output tensor of mish activation function
    func mish(tensor: MPSGraphTensor) -> MPSGraphTensor {
        let threshold = 10.39
        let thresholdTensor = constant(threshold, dataType: tensor.dataType)
        let minimumTensor = minimum(tensor, thresholdTensor, name: nil)
        let expTensor = exponent(with: minimumTensor, name: nil)
        let one = 1.0
        let oneTensor = constant(one, dataType: tensor.dataType)
        let addTensor = addition(expTensor, oneTensor, name: nil)
        let logTensor = logarithm(with: addTensor, name: nil)
        let lessTensor = lessThan(tensor, thresholdTensor, name: nil)
        let selectTensor = select(predicate: lessTensor, trueTensor: logTensor, falseTensor: tensor, name: nil)
        let tanhTensor = tanh(with: selectTensor, name: nil)
        let mulTensor = multiplication(tensor, tanhTensor, name: nil)
        return mulTensor
    }
}

/// A structure that represents the input shape
struct InputShape {
    /// Create a shape for the input tensor
    /// - Parameters:
    ///   - batchSize: Batch size
    ///   - numChannels: Number of channels
    ///   - nnYLen: Y length
    ///   - nnXLen: X length
    /// - Returns: The shape
    static func create(batchSize: NSNumber,
                       numChannels: NSNumber,
                       nnYLen: NSNumber,
                       nnXLen: NSNumber) -> [NSNumber] {
        let shape = [batchSize,
                     numChannels,
                     nnYLen,
                     nnXLen]
        return shape
    }

    /// Get the channel axis
    /// - Returns: The channel axis
    static func getChannelAxis() -> Int {
        return 1
    }

    /// Get the HW axes
    /// - Returns: The HW axes
    static func getHWAxes() -> [NSNumber] {
        let hwAxes = [2, 3] as [NSNumber]
        return hwAxes
    }
}

/// A structure that represents the input layer
struct InputLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    /// Initialize a InputLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    ///   - numChannels: Number of channels
    init(graph: MPSGraph,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         numChannels: NSNumber) {
        shape = InputShape.create(batchSize: -1,
                                  numChannels: numChannels,
                                  nnYLen: nnYLen,
                                  nnXLen: nnXLen)

        self.tensor = graph.placeholder(shape: shape,
                                        dataType: MPSDataType.float32,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents an input global layer for a neural network model.
struct InputGlobalLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    /// Initializes an InputGlobalLayer object with a graph, batch size, number of global features, data type, and input shape.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - numGlobalFeatures: The number of global features.
    init(graph: MPSGraph,
         numGlobalFeatures: NSNumber) {
        shape = InputShape.create(batchSize: -1,
                                  numChannels: numGlobalFeatures,
                                  nnYLen: 1,
                                  nnXLen: 1)

        self.tensor = graph.placeholder(shape: shape,
                                        dataType: MPSDataType.float32,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure representing the input meta layer for a neural network graph.
struct InputMetaLayer {
    /// A `MPSGraphTensor` representing the placeholder tensor in the graph.
    let tensor: MPSGraphTensor
    /// An array of `NSNumber` representing the shape of the tensor placeholder.
    let shape: [NSNumber]

    /// Initializes a new `InputMetaLayer` instance with the given graph and number of meta features.
    ///
    /// - Parameters:
    ///   - graph: The `MPSGraph` instance where the placeholder tensor will be created.
    ///   - numMetaFeatures: The number of meta features (channels) for the input tensor.
    ///
    /// This initializer sets the shape of the input tensor using a helper function `InputShape.create` with
    /// a dynamic batch size (-1), the specified number of channels, and a spatial size of 1x1 (nnYLen and nnXLen).
    /// It also creates a placeholder tensor in the MPS graph with the specified shape and data type `float32`.
    init(graph: MPSGraph, numMetaFeatures: NSNumber) {
        // Define the shape of the input tensor with dynamic batch size, specified number of channels, and spatial dimensions 1x1.
        shape = InputShape.create(batchSize: -1,
                                  numChannels: numMetaFeatures,
                                  nnYLen: 1,
                                  nnXLen: 1)

        // Create a placeholder tensor in the graph with the above-defined shape and data type float32.
        self.tensor = graph.placeholder(shape: shape,
                                        dataType: MPSDataType.float32,
                                        name: nil)
    }
}

/// A structure that represents a mask layer for a neural network model.
struct MaskLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    /// Initializes a MaskLayer object with a graph, batch size, x and y lengths, data type, and input shape.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - nnXLen: The length of the x-axis.
    ///   - nnYLen: The length of the y-axis.
    init(graph: MPSGraph,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        shape = InputShape.create(batchSize: -1,
                                  numChannels: 1,
                                  nnYLen: nnYLen,
                                  nnXLen: nnXLen)

        self.tensor = graph.placeholder(shape: shape,
                                        dataType: MPSDataType.float32,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents a layer which performs the summation operation on a mask layer.
struct MaskSumLayer {
    let tensor: MPSGraphTensor

    /// Initializes a MaskSumLayer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes a MaskSumLayer object with a graph, a mask layer, and a boolean flag indicating whether to use NHWC or NCHW format.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - maskTensor: The mask tensor.
    init(graph: MPSGraph,
         maskTensor: MPSGraphTensor) {
        let hwAxes = InputShape.getHWAxes()

        self.tensor = graph.reductionSum(with: maskTensor,
                                         axes: hwAxes,
                                         name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents a layer which performs square root, subtraction, and multiplication operations on a MaskSumLayer object.
struct MaskSumSqrtS14M01Layer {
    let tensor: MPSGraphTensor

    /// Initializes a MaskSumSqrtS14M01Layer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes a MaskSumSqrtS14M01Layer object with a graph, a MaskSumLayer object, and a boolean flag indicating whether to use 16-bit floating-point data type.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - maskSum: The MaskSumLayer object.
    init(graph: MPSGraph,
         maskSum: MaskSumLayer) {
        let sqrtMaskSum = graph.squareRoot(with: maskSum.tensor, name: nil)

        let fourTeen = graph.constant(14.0,
                                      shape: [1],
                                      dataType: MPSDataType.float32)

        let subtracted = graph.subtraction(sqrtMaskSum, fourTeen, name: nil)

        let zeroPointone = graph.constant(0.1,
                                          shape: [1],
                                          dataType: MPSDataType.float32)

        self.tensor = graph.multiplication(subtracted,
                                           zeroPointone,
                                           name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents a layer which performs squaring and subtraction operations on a MaskSumSqrtS14M01Layer object.
struct MaskSumSqrtS14M01SquareS01Layer {
    let tensor: MPSGraphTensor

    /// Initializes a MaskSumSqrtS14M01SquareS01Layer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes a MaskSumSqrtS14M01SquareS01Layer object with a graph, a MaskSumSqrtS14M01Layer object, and a boolean flag indicating whether to use 16-bit floating-point data type.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - maskSumSqrtS14M01: The MaskSumSqrtS14M01Layer object.
    init(graph: MPSGraph,
         maskSumSqrtS14M01: MaskSumSqrtS14M01Layer) {
        let squared = graph.square(with: maskSumSqrtS14M01.tensor, name: nil)

        let zeroPointone = graph.constant(0.1,
                                          shape: [1],
                                          dataType: MPSDataType.float32)

        self.tensor = graph.subtraction(squared,
                                        zeroPointone,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A Swift structure that represents a network tester, which tests various neural network configurations.
struct NetworkTester {

    /// A static function that tests a custom neural network configuration with the given parameters.
    /// - Parameters:
    ///   - batchSize: The number of input batches.
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    ///   - numChannels: The number of channels in the input tensor.
    ///   - input: A pointer to the input data.
    ///   - mask: A pointer to the mask data.
    ///   - output: A pointer to the output data.
    ///   - networkBuilder: A closure that takes an MPSGraph, InputLayer, and MaskLayer, and returns an MPSGraphTensor representing the custom network configuration.
    static func test(batchSize: NSNumber,
                     nnXLen: NSNumber,
                     nnYLen: NSNumber,
                     numChannels: NSNumber,
                     input: UnsafeMutablePointer<Float32>,
                     mask: UnsafeMutablePointer<Float32>,
                     output: UnsafeMutablePointer<Float32>,
                     networkBuilder: (MPSGraph, InputLayer, MaskLayer) -> MPSGraphTensor) {

        // Create a Metal device.
        let device = DefaultDevice.device

        // Create a MPSGraph.
        let graph = MPSGraph()

        // Create the input and mask layers.
        let inputLayer = InputLayer(graph: graph,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    numChannels: numChannels)

        let maskLayer = MaskLayer(graph: graph,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        // Build the custom network configuration using the provided networkBuilder closure.
        let resultTensor = networkBuilder(graph, inputLayer, maskLayer)

        // Create input shape
        let inputShape = InputShape.create(batchSize: batchSize,
                                           numChannels: numChannels,
                                           nnYLen: nnYLen,
                                           nnXLen: nnXLen)

        // Create MPSNDArrayDescriptors from the input shape.
        let sourceDescriptor = MPSNDArrayDescriptor(dataType: inputLayer.tensor.dataType,
                                                    shape: inputShape)

        // Create MPSNDArray from the source descriptor.
        let sourceArray = MPSNDArray(device: device,
                                     descriptor: sourceDescriptor)

        // Create a mask shape
        let maskShape = InputShape.create(batchSize: batchSize,
                                          numChannels: 1,
                                          nnYLen: nnYLen,
                                          nnXLen: nnXLen)

        // Create MPSNDArrayDescriptors from the mask shape.
        let maskDescriptor = MPSNDArrayDescriptor(dataType: maskLayer.tensor.dataType,
                                                  shape: maskShape)

        // Create MPSNDArray from the mask descriptor.
        let maskArray = MPSNDArray(device: device,
                                   descriptor: maskDescriptor)

        // Write input and mask data to their respective MPSNDArrays, converting to FP16 if necessary.
        sourceArray.writeBytes(input)
        maskArray.writeBytes(mask)

        // Create MPSGraphTensorData objects from the source and mask arrays.
        let sourceTensorData = MPSGraphTensorData(sourceArray)
        let maskTensorData = MPSGraphTensorData(maskArray)

        // Execute the graph and fetch the result.
        let fetch = graph.run(feeds: [inputLayer.tensor: sourceTensorData,
                                      maskLayer.tensor: maskTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        // Read the output data from the result tensor, converting from FP16 to FP32 if necessary.
        fetch[resultTensor]?.mpsndarray().readBytes(output)
    }
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

    /// Initializes a SWConvLayerDesc object.
    /// - Parameters:
    ///   - convYSize: The Y size of the convolution.
    ///   - convXSize: The X size of the convolution.
    ///   - inChannels: The number of input channels.
    ///   - outChannels: The number of output channels.
    ///   - dilationY: The dilation in the Y direction.
    ///   - dilationX: The dilation in the X direction.
    ///   - weights: A pointer to the weights.
    init(convYSize: NSNumber,
         convXSize: NSNumber,
         inChannels: NSNumber,
         outChannels: NSNumber,
         dilationY: Int,
         dilationX: Int,
         weights: UnsafeMutablePointer<Float32>) {
        self.convYSize = convYSize
        self.convXSize = convXSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dilationY = dilationY
        self.dilationX = dilationX
        self.weights = weights
    }
}

public func createSWConvLayerDesc(convYSize: Int32,
                                  convXSize: Int32,
                                  inChannels: Int32,
                                  outChannels: Int32,
                                  dilationY: Int32,
                                  dilationX: Int32,
                                  weights: UnsafeMutablePointer<Float32>) -> SWConvLayerDesc {
    return SWConvLayerDesc(convYSize: convYSize as NSNumber,
                           convXSize: convXSize as NSNumber,
                           inChannels: inChannels as NSNumber,
                           outChannels: outChannels as NSNumber,
                           dilationY: Int(dilationY),
                           dilationX: Int(dilationX),
                           weights: weights)
}

/// A class that represents a convolutional layer using MPSGraph
class ConvLayer {
    /// The result tensor of the convolutional operation
    let resultTensor: MPSGraphTensor
    /// The convolution 2D operation descriptor
    let convDescriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                                           strideInY: 1,
                                                           dilationRateInX: 1,
                                                           dilationRateInY: 1,
                                                           groups: 1,
                                                           paddingStyle: .TF_SAME,
                                                           dataLayout: .NCHW,
                                                           weightsLayout: .OIHW)!

    /// Class method that tests the convolutional layer by running a forward pass
    /// - Parameters:
    ///   - descriptor: A descriptor for the convolutional layer
    ///   - nnXLen: The width of the input tensor
    ///   - nnYLen: The height of the input tensor
    ///   - batchSize: The batch size of the input tensor
    ///   - input: A pointer to the input tensor data
    ///   - output: A pointer to the output tensor data
    class func test(descriptor: SWConvLayerDesc,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    batchSize: NSNumber,
                    input: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {
        let device = DefaultDevice.device
        let graph = MPSGraph()

        let source = InputLayer(graph: graph,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                numChannels: descriptor.inChannels)

        let conv = ConvLayer(graph: graph,
                             sourceTensor: source.tensor,
                             descriptor: descriptor,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen)

        let inputShape = InputShape.create(batchSize: batchSize,
                                           numChannels: descriptor.inChannels,
                                           nnYLen: nnYLen,
                                           nnXLen: nnXLen)

        let sourceDescriptor = MPSNDArrayDescriptor(dataType: source.tensor.dataType,
                                                    shape: inputShape)

        let sourceArray = MPSNDArray(device: device,
                                     descriptor: sourceDescriptor)

        sourceArray.writeBytes(input)
        let sourceTensorData = MPSGraphTensorData(sourceArray)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData],
                              targetTensors: [conv.resultTensor],
                              targetOperations: nil)

        fetch[conv.resultTensor]?.mpsndarray().readBytes(output)
    }

    /// Initializes a ConvLayer object
    /// - Parameters:
    ///   - graph: An MPSGraph object
    ///   - sourceTensor: The input tensor for the convolutional layer
    ///   - descriptor: A descriptor for the convolutional layer
    ///   - nnXLen: The width of the input tensor
    ///   - nnYLen: The height of the input tensor
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         descriptor: SWConvLayerDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        let weightsShape = [descriptor.outChannels,
                            descriptor.inChannels,
                            descriptor.convYSize,
                            descriptor.convXSize]

        let weightsData = Data(floatsNoCopy: descriptor.weights,
                               shape: weightsShape)

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: MPSDataType.float32)

        resultTensor = graph.convolution2D(sourceTensor,
                                           weights: weightsTensor,
                                           descriptor: convDescriptor,
                                           name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testConvLayer(descriptor: SWConvLayerDesc,
                          nnXLen: Int32,
                          nnYLen: Int32,
                          batchSize: Int32,
                          input: UnsafeMutablePointer<Float32>,
                          output: UnsafeMutablePointer<Float32>) {
    ConvLayer.test(descriptor: descriptor,
                   nnXLen: nnXLen as NSNumber,
                   nnYLen: nnYLen as NSNumber,
                   batchSize: batchSize as NSNumber,
                   input: input,
                   output: output)
}

/// A struct that represents a description of a batch normalization layer.
public struct SWBatchNormLayerDesc {
    let numChannels: NSNumber
    let epsilon: Float32
    let hasScale: NSNumber
    let hasBias: NSNumber
    let mean: UnsafeMutablePointer<Float32>
    let variance: UnsafeMutablePointer<Float32>
    let scale: UnsafeMutablePointer<Float32>
    let bias: UnsafeMutablePointer<Float32>

    /// Initializes a SWBatchNormLayerDesc object.
    /// - Parameters:
    ///   - numChannels: The number of channels in the input tensor.
    ///   - epsilon: A small value added to the variance to avoid division by zero.
    ///   - hasScale: A flag indicating whether scaling is applied.
    ///   - hasBias: A flag indicating whether bias is applied.
    ///   - mean: A pointer to the mean.
    ///   - variance: A pointer to the variance.
    ///   - scale: A pointer to the scale.
    ///   - bias: A pointer to the bias.
    init(numChannels: NSNumber,
         epsilon: Float32,
         hasScale: NSNumber,
         hasBias: NSNumber,
         mean: UnsafeMutablePointer<Float32>,
         variance: UnsafeMutablePointer<Float32>,
         scale: UnsafeMutablePointer<Float32>,
         bias: UnsafeMutablePointer<Float32>) {
        self.numChannels = numChannels
        self.epsilon = epsilon
        self.hasScale = hasScale
        self.hasBias = hasBias
        self.mean = mean
        self.variance = variance
        self.scale = scale
        self.bias = bias
    }
}

public func createSWBatchNormLayerDesc(numChannels: Int32,
                                       epsilon: Float32,
                                       hasScale: Bool,
                                       hasBias: Bool,
                                       mean: UnsafeMutablePointer<Float32>,
                                       variance: UnsafeMutablePointer<Float32>,
                                       scale: UnsafeMutablePointer<Float32>,
                                       bias: UnsafeMutablePointer<Float32>) -> SWBatchNormLayerDesc {
    return SWBatchNormLayerDesc(numChannels: numChannels as NSNumber,
                                epsilon: epsilon,
                                hasScale: hasScale as NSNumber,
                                hasBias: hasBias as NSNumber,
                                mean: mean,
                                variance: variance,
                                scale: scale,
                                bias: bias)
}

/// A class that represents a batch normalization layer.
class BatchNormLayer {
    let resultTensor: MPSGraphTensor

    /// Executes a test for the batch normalization layer.
    /// - Parameters:
    ///   - descriptor: The description of the batch normalization layer.
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    ///   - batchSize: The number of input batches.
    ///   - input: A pointer to the input data.
    ///   - mask: A pointer to the mask data.
    ///   - output: A pointer to the output data.
    class func test(descriptor: SWBatchNormLayerDesc,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    batchSize: NSNumber,
                    input: UnsafeMutablePointer<Float32>,
                    mask: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        NetworkTester.test(batchSize: batchSize,
                           nnXLen: nnXLen,
                           nnYLen: nnYLen,
                           numChannels: descriptor.numChannels,
                           input: input,
                           mask: mask,
                           output: output) { graph, inputLayer, maskLayer in

            let batchNorm = BatchNormLayer(graph: graph,
                                           sourceTensor: inputLayer.tensor,
                                           maskTensor: maskLayer.tensor,
                                           descriptor: descriptor,
                                           nnXLen: nnXLen,
                                           nnYLen: nnYLen)

            return batchNorm.resultTensor
        }
    }

    /// Initializes a BatchNormLayer object with the specified parameters, and computes the normalized and masked result tensor.
    /// - Parameters:
    ///   - graph: The MPSGraph object used to build the BatchNormLayer.
    ///   - sourceTensor: The input tensor to the BatchNormLayer.
    ///   - maskTensor: The mask tensor to apply to the normalized tensor.
    ///   - descriptor: The BatchNormLayer descriptor containing parameters such as the number of channels, mean, variance, scale, and bias.
    ///   - nnXLen: The length of the input tensor in the X direction.
    ///   - nnYLen: The length of the input tensor in the Y direction.
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         descriptor: SWBatchNormLayerDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        let meanShape = InputShape.create(batchSize: 1,
                                          numChannels: descriptor.numChannels,
                                          nnYLen: 1,
                                          nnXLen: 1)

        let meanData = Data(floatsNoCopy: descriptor.mean,
                            shape: meanShape)

        let varianceData = Data(floatsNoCopy: descriptor.variance,
                                shape: meanShape)

        let scaleData = Data(floatsNoCopy: descriptor.scale,
                             shape: meanShape)

        let biasData = Data(floatsNoCopy: descriptor.bias,
                            shape: meanShape)

        let meanTensor = graph.constant(meanData,
                                        shape: meanShape,
                                        dataType: MPSDataType.float32)

        let varianceTensor = graph.constant(varianceData,
                                            shape: meanShape,
                                            dataType: MPSDataType.float32)

        let scaleTensor = graph.constant(scaleData,
                                         shape: meanShape,
                                         dataType: MPSDataType.float32)

        let biasTensor = graph.constant(biasData,
                                        shape: meanShape,
                                        dataType: MPSDataType.float32)

        let normalized = graph.normalize(sourceTensor,
                                         mean: meanTensor,
                                         variance: varianceTensor,
                                         gamma: scaleTensor,
                                         beta: biasTensor,
                                         epsilon: descriptor.epsilon,
                                         name: nil)

        resultTensor = graph.multiplication(normalized,
                                            maskTensor,
                                            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testBatchNormLayer(descriptor: SWBatchNormLayerDesc,
                               nnXLen: Int32,
                               nnYLen: Int32,
                               batchSize: Int32,
                               input: UnsafeMutablePointer<Float32>,
                               mask: UnsafeMutablePointer<Float32>,
                               output: UnsafeMutablePointer<Float32>) {
    BatchNormLayer.test(descriptor: descriptor,
                        nnXLen: nnXLen as NSNumber,
                        nnYLen: nnYLen as NSNumber,
                        batchSize: batchSize as NSNumber,
                        input: input,
                        mask: mask,
                        output: output)
}

/// An enumeration of the different kinds of activation function.
public enum ActivationKind {
    case identity
    case relu
    case mish
}

/// A structure that represents an activation layer
struct ActivationLayer {
    let resultTensor: MPSGraphTensor

    /// Initialize an ActivationLayer object
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - activationKind: The activation kind
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         activationKind: ActivationKind) {

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

/// A class that represents a residual block in a convolutional neural network.
public class SWResidualBlockDesc: BlockDescriptor {
    /// A description of the batch normalization layer that is applied before the first convolutional layer.
    let preBN: SWBatchNormLayerDesc

    /// The type of activation function that is applied before the first convolutional layer.
    let preActivation: ActivationKind

    /// A description of the convolutional layer that is applied in the middle of the residual block.
    let regularConv: SWConvLayerDesc

    /// A description of the batch normalization layer that is applied after the middle convolutional layer.
    let midBN: SWBatchNormLayerDesc

    /// The type of activation function that is applied after the middle convolutional layer.
    let midActivation: ActivationKind

    /// A description of the convolutional layer that is applied at the end of the residual block.
    let finalConv: SWConvLayerDesc

    /// Initializes a `SWResidualBlockDesc` object.
    /// - Parameters:
    ///   - preBN: A description of the batch normalization layer that is applied before the first convolutional layer.
    ///   - preActivation: The type of activation function that is applied before the first convolutional layer.
    ///   - regularConv: A description of the convolutional layer that is applied in the middle of the residual block.
    ///   - midBN: A description of the batch normalization layer that is applied after the middle convolutional layer.
    ///   - midActivation: The type of activation function that is applied after the middle convolutional layer.
    ///   - finalConv: A description of the convolutional layer that is applied at the end of the residual block.
    init(preBN: SWBatchNormLayerDesc,
         preActivation: ActivationKind,
         regularConv: SWConvLayerDesc,
         midBN: SWBatchNormLayerDesc,
         midActivation: ActivationKind,
         finalConv: SWConvLayerDesc) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

public func createSWResidualBlockDesc(preBN: SWBatchNormLayerDesc,
                                      preActivation: ActivationKind,
                                      regularConv: SWConvLayerDesc,
                                      midBN: SWBatchNormLayerDesc,
                                      midActivation: ActivationKind,
                                      finalConv: SWConvLayerDesc) -> SWResidualBlockDesc {
    return SWResidualBlockDesc(preBN: preBN,
                               preActivation: preActivation,
                               regularConv: regularConv,
                               midBN: midBN,
                               midActivation: midActivation,
                               finalConv: finalConv)
}

/// A class that represents a Residual Block layer
class ResidualBlock {
    let resultTensor: MPSGraphTensor

    /// A function that runs tests on the Residual Block layer
    ///
    /// - Parameters:
    ///   - descriptor: The Residual Block descriptor
    ///   - batchSize: Batch size
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    ///   - input: The input float32 pointer
    ///   - mask: The mask float32 pointer
    ///   - output: The output float32 pointer
    class func test(descriptor: SWResidualBlockDesc,
                    batchSize: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    input: UnsafeMutablePointer<Float32>,
                    mask: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        NetworkTester.test(batchSize: batchSize,
                           nnXLen: nnXLen,
                           nnYLen: nnYLen,
                           numChannels: descriptor.preBN.numChannels,
                           input: input,
                           mask: mask,
                           output: output) { graph, inputLayer, maskLayer in

            let block = ResidualBlock(graph: graph,
                                      sourceTensor: inputLayer.tensor,
                                      maskTensor: maskLayer.tensor,
                                      descriptor: descriptor,
                                      nnXLen: nnXLen,
                                      nnYLen: nnYLen)

            return block.resultTensor
        }
    }

    /// Initialize a ResidualBlock object
    ///
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - descriptor: The Residual Block descriptor
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         descriptor: SWResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen)

        let preActivation = ActivationLayer(graph: graph,
                                            sourceTensor: preBN.resultTensor,
                                            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(graph: graph,
                                    sourceTensor: preActivation.resultTensor,
                                    descriptor: descriptor.regularConv,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen)

        let midBN = BatchNormLayer(graph: graph,
                                   sourceTensor: regularConv.resultTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen)

        let midActivation = ActivationLayer(graph: graph,
                                            sourceTensor: midBN.resultTensor,
                                            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(graph: graph,
                                  sourceTensor: midActivation.resultTensor,
                                  descriptor: descriptor.finalConv,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        resultTensor = graph.addition(sourceTensor,
                                      finalConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testResidualBlock(descriptor: SWResidualBlockDesc,
                              batchSize: Int32,
                              nnXLen: Int32,
                              nnYLen: Int32,
                              input: UnsafeMutablePointer<Float32>,
                              mask: UnsafeMutablePointer<Float32>,
                              output: UnsafeMutablePointer<Float32>) {
    ResidualBlock.test(descriptor: descriptor,
                       batchSize: batchSize as NSNumber,
                       nnXLen: nnXLen as NSNumber,
                       nnYLen: nnYLen as NSNumber,
                       input: input,
                       mask: mask,
                       output: output)
}

/// A structure that represents a global pooling layer
struct GlobalPoolingLayer {
    /// The resulting tensor after applying the global pooling operation
    let resultTensor: MPSGraphTensor

    /// Initialize a GlobalPoolingLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The source tensor to be pooled
    ///   - maskSumTensor: The sum of the mask
    ///   - maskSumSqrtS14M01Tensor: The multiplication of subtraction of square root of the sum of the mask
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor) {
        let hwAxes = InputShape.getHWAxes()
        let channelAxis = InputShape.getChannelAxis()

        let sumTensor = graph.reductionSum(with: sourceTensor,
                                           axes: hwAxes,
                                           name: nil)

        let meanTensor = graph.division(sumTensor, maskSumTensor, name: nil)

        let meanMaskTensor = graph.multiplication(meanTensor,
                                                  maskSumSqrtS14M01Tensor,
                                                  name: nil)

        let maxTensor = graph.reductionMaximum(with: sourceTensor,
                                               axes: hwAxes,
                                               name: nil)

        resultTensor = graph.concatTensors([meanTensor,
                                            meanMaskTensor,
                                            maxTensor],
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

    /// Initialize a GlobalPoolingValueLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The input tensor
    ///   - maskSumTensor: The sum of the mask
    ///   - maskSumSqrtS14M01Tensor: The multiplication of subtraction of square root of the sum of the mask
    ///   - maskSumSqrtS14M01SquareS01Tensor: The subtraction of square of multiplication of subtraction of square root of the sum of the mask
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor) {
        let hwAxes = InputShape.getHWAxes()
        let channelAxis = InputShape.getChannelAxis()

        let sumTensor = graph.reductionSum(with: sourceTensor,
                                           axes: hwAxes,
                                           name: nil)

        let meanTensor = graph.division(sumTensor, maskSumTensor, name: nil)

        let meanMaskTensor = graph.multiplication(meanTensor,
                                                  maskSumSqrtS14M01Tensor,
                                                  name: nil)

        let meanMaskSquareTensor = graph.multiplication(meanTensor,
                                                        maskSumSqrtS14M01SquareS01Tensor,
                                                        name: nil)

        resultTensor = graph.concatTensors([meanTensor,
                                            meanMaskTensor,
                                            meanMaskSquareTensor],
                                           dimension: channelAxis,
                                           name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == 1)
        assert(resultTensor.shape?[3] == 1)
    }
}

/// A struct that represents a matrix multiplication layer descriptor
public struct SWMatMulLayerDesc {
    /// The number of input channels
    let inChannels: NSNumber
    /// The number of output channels
    let outChannels: NSNumber
    /// The weights used for the matrix multiplication
    let weights: UnsafeMutablePointer<Float32>

    /// Initialize a SWMatMulLayerDesc object
    /// - Parameters:
    ///   - inChannels: The number of input channels
    ///   - outChannels: The number of output channels
    ///   - weights: The weights used for the matrix multiplication
    init(inChannels: NSNumber,
         outChannels: NSNumber,
         weights: UnsafeMutablePointer<Float32>) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
}

public func createSWMatMulLayerDesc(inChannels: Int32,
                                    outChannels: Int32,
                                    weights: UnsafeMutablePointer<Float32>) -> SWMatMulLayerDesc {
    return SWMatMulLayerDesc(inChannels: inChannels as NSNumber,
                             outChannels: outChannels as NSNumber,
                             weights: weights)
}

/// A structure representing a matrix multiplication layer.
struct MatMulLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes a MatMulLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - descriptor: The matrix multiplication layer descriptor.
    ///   - sourceTensor: The input tensor to the layer.
    init(graph: MPSGraph,
         descriptor: SWMatMulLayerDesc,
         sourceTensor: MPSGraphTensor) {

        assert((sourceTensor.shape?.count == 4) || (sourceTensor.shape?[1] == descriptor.inChannels))
        assert((sourceTensor.shape?.count == 2) || (sourceTensor.shape?[1] == descriptor.inChannels))

        let weightsShape = [descriptor.inChannels,
                            descriptor.outChannels]

        let weightsData = Data(floatsNoCopy: descriptor.weights,
                               shape: weightsShape)

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: MPSDataType.float32)

        let shape = [-1, descriptor.inChannels]

        let reshapedSource = graph.reshape(sourceTensor,
                                           shape: shape,
                                           name: nil)

        resultTensor = graph.matrixMultiplication(primary: reshapedSource,
                                                  secondary: weightsTensor,
                                                  name: nil)

        assert(resultTensor.shape?.count == 2)
    }
}

/// An Objective-C class that represents the bias layer description used in Swift.
public struct SWMatBiasLayerDesc {
    /// The number of channels.
    let numChannels: NSNumber
    /// The pointer to the weights.
    let weights: UnsafeMutablePointer<Float32>

    /// Initialize an instance of SWMatBiasLayerDesc.
    /// - Parameters:
    ///   - numChannels: The number of channels.
    ///   - weights: The pointer to the weights.
    init(numChannels: NSNumber,
         weights: UnsafeMutablePointer<Float32>) {
        self.numChannels = numChannels
        self.weights = weights
    }
}

public func createSWMatBiasLayerDesc(numChannels: Int32,
                                     weights: UnsafeMutablePointer<Float32>) -> SWMatBiasLayerDesc {
    return SWMatBiasLayerDesc(numChannels: numChannels as NSNumber,
                              weights: weights)
}

/// A structure that performs matrix bias operations
struct MatBiasLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes a MatBiasLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - descriptor: The descriptor that contains information about the layer
    ///   - sourceTensor: The input tensor to the layer.
    init(graph: MPSGraph,
         descriptor: SWMatBiasLayerDesc,
         sourceTensor: MPSGraphTensor) {

        assert((sourceTensor.shape?.count == 2) && (sourceTensor.shape?[1] == descriptor.numChannels))

        let weightsShape = [1, descriptor.numChannels]

        let weightsData = Data(floatsNoCopy: descriptor.weights,
                               shape: weightsShape)

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: MPSDataType.float32)

        resultTensor = graph.addition(sourceTensor,
                                      weightsTensor,
                                      name: nil)
    }
}

/// A structure that performs bias operations in NC coordinates.
struct AddNCBiasLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes an AddNCBiasLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - sourceTensor: The input tensor to the layer.
    ///   - biasTensor: The bias tensor.
    ///   - nnXLen: The x length.
    ///   - nnYLen: The y length.
    ///   - numChannels: The number of channels.
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         biasTensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         numChannels: NSNumber) {
        let shape = InputShape.create(batchSize: -1,
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

/// A class that represents a residual block with global pooling.
public class SWGlobalPoolingResidualBlockDesc: BlockDescriptor {
    /// The batch normalization layer before the residual block.
    let preBN: SWBatchNormLayerDesc

    /// The pre-activation function of the residual block.
    let preActivation: ActivationKind

    /// The regular convolutional layer in the residual block.
    let regularConv: SWConvLayerDesc

    /// The convolutional layer for global pooling.
    let gpoolConv: SWConvLayerDesc

    /// The batch normalization layer after the global pooling convolutional layer.
    let gpoolBN: SWBatchNormLayerDesc

    /// The activation function after the global pooling batch normalization layer.
    let gpoolActivation: ActivationKind

    /// The matrix multiplication layer that multiplies the global pooled output with a bias.
    let gpoolToBiasMul: SWMatMulLayerDesc

    /// The batch normalization layer after the matrix multiplication layer.
    let midBN: SWBatchNormLayerDesc

    /// The activation function after the mid batch normalization layer.
    let midActivation: ActivationKind

    /// The final convolutional layer in the residual block.
    let finalConv: SWConvLayerDesc

    /// Initialize a SWGlobalPoolingResidualBlockDesc object.
    /// - Parameters:
    ///   - preBN: The batch normalization layer before the residual block.
    ///   - preActivation: The pre-activation function of the residual block.
    ///   - regularConv: The regular convolutional layer in the residual block.
    ///   - gpoolConv: The convolutional layer for global pooling.
    ///   - gpoolBN: The batch normalization layer after the global pooling convolutional layer.
    ///   - gpoolActivation: The activation function after the global pooling batch normalization layer.
    ///   - gpoolToBiasMul: The matrix multiplication layer that multiplies the global pooled output with a bias.
    ///   - midBN: The batch normalization layer after the matrix multiplication layer.
    ///   - midActivation: The activation function after the mid batch normalization layer.
    ///   - finalConv: The final convolutional layer in the residual block.
    init(preBN: SWBatchNormLayerDesc,
         preActivation: ActivationKind,
         regularConv: SWConvLayerDesc,
         gpoolConv: SWConvLayerDesc,
         gpoolBN: SWBatchNormLayerDesc,
         gpoolActivation: ActivationKind,
         gpoolToBiasMul: SWMatMulLayerDesc,
         midBN: SWBatchNormLayerDesc,
         midActivation: ActivationKind,
         finalConv: SWConvLayerDesc) {
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

public func createSWGlobalPoolingResidualBlockDesc(preBN: SWBatchNormLayerDesc,
                                                   preActivation: ActivationKind,
                                                   regularConv: SWConvLayerDesc,
                                                   gpoolConv: SWConvLayerDesc,
                                                   gpoolBN: SWBatchNormLayerDesc,
                                                   gpoolActivation: ActivationKind,
                                                   gpoolToBiasMul: SWMatMulLayerDesc,
                                                   midBN: SWBatchNormLayerDesc,
                                                   midActivation: ActivationKind,
                                                   finalConv: SWConvLayerDesc) -> SWGlobalPoolingResidualBlockDesc {

    return SWGlobalPoolingResidualBlockDesc(preBN: preBN,
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

/// A class representing a residual block with global pooling
class GlobalPoolingResidualBlock {
    let resultTensor: MPSGraphTensor

    /// A method to test the global pooling residual block
    ///
    /// - Parameters:
    ///   - descriptor: The descriptor of the global pooling residual block
    ///   - batchSize: The batch size
    ///   - nnXLen: The X length
    ///   - nnYLen: The Y length
    ///   - input: The input pointer
    ///   - mask: The mask pointer
    ///   - output: The output pointer
    class func test(descriptor: SWGlobalPoolingResidualBlockDesc,
                    batchSize: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    input: UnsafeMutablePointer<Float32>,
                    mask: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        NetworkTester.test(batchSize: batchSize,
                           nnXLen: nnXLen,
                           nnYLen: nnYLen,
                           numChannels: descriptor.preBN.numChannels,
                           input: input,
                           mask: mask,
                           output: output) { graph, inputLayer, maskLayer in

            let maskSum = MaskSumLayer(graph: graph,
                                       maskTensor: maskLayer.tensor)

            let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                           maskSum: maskSum)

            let block =
            GlobalPoolingResidualBlock(graph: graph,
                                       sourceTensor: inputLayer.tensor,
                                       maskTensor: maskLayer.tensor,
                                       maskSumTensor: maskSum.tensor,
                                       maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                       descriptor: descriptor,
                                       nnXLen: nnXLen,
                                       nnYLen: nnYLen)

            return block.resultTensor
        }
    }

    /// Initialize a GlobalPoolingResidualBlock object
    ///
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The source tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The mask sum tensor
    ///   - maskSumSqrtS14M01Tensor: The mask sum square tensor
    ///   - descriptor: The descriptor of the global pooling residual block
    ///   - nnXLen: The X length
    ///   - nnYLen: The Y length
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         descriptor: SWGlobalPoolingResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        let maskSum = MaskSumLayer(tensor: maskSumTensor)
        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(tensor: maskSumSqrtS14M01Tensor)

        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen)

        let preActivation = ActivationLayer(graph: graph,
                                            sourceTensor: preBN.resultTensor,
                                            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(graph: graph,
                                    sourceTensor: preActivation.resultTensor,
                                    descriptor: descriptor.regularConv,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen)

        let gpoolConv = ConvLayer(graph: graph,
                                  sourceTensor: preActivation.resultTensor,
                                  descriptor: descriptor.gpoolConv,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        let gpoolBN = BatchNormLayer(graph: graph,
                                     sourceTensor: gpoolConv.resultTensor,
                                     maskTensor: maskTensor,
                                     descriptor: descriptor.gpoolBN,
                                     nnXLen: nnXLen,
                                     nnYLen: nnYLen)

        let gpoolActivation = ActivationLayer(graph: graph,
                                              sourceTensor: gpoolBN.resultTensor,
                                              activationKind: descriptor.gpoolActivation)

        let gpoolConcat = GlobalPoolingLayer(graph: graph,
                                             sourceTensor: gpoolActivation.resultTensor,
                                             maskSumTensor: maskSum.tensor,
                                             maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor)

        assert(gpoolConcat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels)

        let gpoolToBiasMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToBiasMul,
                                         sourceTensor: gpoolConcat.resultTensor)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: regularConv.resultTensor,
                                   biasTensor: gpoolToBiasMul.resultTensor,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.gpoolToBiasMul.outChannels)

        let midBN = BatchNormLayer(graph: graph,
                                   sourceTensor: added.resultTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen)

        let midActivation = ActivationLayer(graph: graph,
                                            sourceTensor: midBN.resultTensor,
                                            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(graph: graph,
                                  sourceTensor: midActivation.resultTensor,
                                  descriptor: descriptor.finalConv,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        resultTensor = graph.addition(sourceTensor,
                                      finalConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testGlobalPoolingResidualBlock(descriptor: SWGlobalPoolingResidualBlockDesc,
                                           batchSize: Int32,
                                           nnXLen: Int32,
                                           nnYLen: Int32,
                                           input: UnsafeMutablePointer<Float32>,
                                           mask: UnsafeMutablePointer<Float32>,
                                           output: UnsafeMutablePointer<Float32>) {
    GlobalPoolingResidualBlock.test(descriptor: descriptor,
                                    batchSize: batchSize as NSNumber,
                                    nnXLen: nnXLen as NSNumber,
                                    nnYLen: nnYLen as NSNumber,
                                    input: input,
                                    mask: mask,
                                    output: output)
}

/// A class that represents a nested bottleneck residual block
public class SWNestedBottleneckResidualBlockDesc: BlockDescriptor {
    /// The batch normalization layer before the residual block.
    let preBN: SWBatchNormLayerDesc

    /// The pre-activation function of the residual block.
    let preActivation: ActivationKind

    /// The convolutional layer before the residual block.
    let preConv: SWConvLayerDesc

    /// The list of blocks that make up the trunk
    let blockDescriptors: [BlockDescriptor]

    /// The batch normalization layer after the residual block.
    let postBN: SWBatchNormLayerDesc

    /// The activation function after the post batch normalization layer.
    let postActivation: ActivationKind

    /// The convolutional layer after the post activation layer.
    let postConv: SWConvLayerDesc

    /// Initialize a SWNestedBottleneckResidualBlockDesc object.
    /// - Parameters:
    ///   - preBN: The batch normalization layer before the residual block.
    ///   - preActivation: The pre-activation function of the residual block.
    ///   - preConv: The convolutional layer before the residual block.
    ///   - postBN: The batch normalization layer after the residual block.
    ///   - postActivation: The activation function after the post batch normalization layer.
    ///   - postConv: The convolutional layer after the post activation layer.
    init(preBN: SWBatchNormLayerDesc,
         preActivation: ActivationKind,
         preConv: SWConvLayerDesc,
         blockDescriptors: [BlockDescriptor],
         postBN: SWBatchNormLayerDesc,
         postActivation: ActivationKind,
         postConv: SWConvLayerDesc) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.preConv = preConv
        self.blockDescriptors = blockDescriptors
        self.postBN = postBN
        self.postActivation = postActivation
        self.postConv = postConv
    }
}

public func createSWNestedBottleneckResidualBlockDesc(preBN: SWBatchNormLayerDesc,
                                                      preActivation: ActivationKind,
                                                      preConv: SWConvLayerDesc,
                                                      blockDescriptors: [BlockDescriptor],
                                                      postBN: SWBatchNormLayerDesc,
                                                      postActivation: ActivationKind,
                                                      postConv: SWConvLayerDesc) -> SWNestedBottleneckResidualBlockDesc {
    return SWNestedBottleneckResidualBlockDesc(preBN: preBN,
                                               preActivation: preActivation,
                                               preConv: preConv,
                                               blockDescriptors: blockDescriptors,
                                               postBN: postBN,
                                               postActivation: postActivation,
                                               postConv: postConv)
}

public class BlockDescriptor {
}

public class BlockDescriptorBuilder {
    public var blockDescriptors: [BlockDescriptor] = []

    public func enque(with descriptor: BlockDescriptor) {
        blockDescriptors.append(descriptor)
    }
}

public func createBlockDescriptorBuilder() -> BlockDescriptorBuilder {
    return BlockDescriptorBuilder()
}

/// A structure that represents a block stack
struct BlockStack {
    /// The resulting tensor after processing the block stack
    let resultTensor: MPSGraphTensor

    /// Process block descriptors
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - blockDescriptors: The block descriptors
    ///   - index: The index of the block descriptor
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    /// - Returns: The result tensor
    static func processBlockDescriptors(_ graph: MPSGraph,
                                        _ sourceTensor: MPSGraphTensor,
                                        _ maskTensor: MPSGraphTensor,
                                        _ maskSumTensor: MPSGraphTensor,
                                        _ maskSumSqrtS14M01Tensor: MPSGraphTensor,
                                        _ blockDescriptors: [BlockDescriptor],
                                        _ index: Int,
                                        _ nnXLen: NSNumber,
                                        _ nnYLen: NSNumber) -> MPSGraphTensor {
        guard index < blockDescriptors.count else {
            return sourceTensor
        }

        let blockDescriptor = blockDescriptors[index]
        let blockInput: MPSGraphTensor

        switch blockDescriptor {
        case let globalPoolingDescriptor as SWGlobalPoolingResidualBlockDesc:
            let globalPooling = GlobalPoolingResidualBlock(graph: graph,
                                                           sourceTensor: sourceTensor,
                                                           maskTensor: maskTensor,
                                                           maskSumTensor: maskSumTensor,
                                                           maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                                           descriptor: globalPoolingDescriptor,
                                                           nnXLen: nnXLen,
                                                           nnYLen: nnYLen)

            blockInput = globalPooling.resultTensor
        case let nestedBottleneckDescriptor as SWNestedBottleneckResidualBlockDesc:
            let nestedBottleneck = NestedBottleneckResidualBlock(graph: graph,
                                                                 sourceTensor: sourceTensor,
                                                                 maskTensor: maskTensor,
                                                                 maskSumTensor: maskSumTensor,
                                                                 maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                                                 descriptor: nestedBottleneckDescriptor,
                                                                 nnXLen: nnXLen,
                                                                 nnYLen: nnYLen)

            blockInput = nestedBottleneck.resultTensor
        case let residualBlockDescriptor as SWResidualBlockDesc:
            let ordinary = ResidualBlock(graph: graph,
                                         sourceTensor: sourceTensor,
                                         maskTensor: maskTensor,
                                         descriptor: residualBlockDescriptor,
                                         nnXLen: nnXLen,
                                         nnYLen: nnYLen)

            blockInput = ordinary.resultTensor
        default:
            blockInput = sourceTensor
        }

        return processBlockDescriptors(graph,
                                       blockInput,
                                       maskTensor,
                                       maskSumTensor,
                                       maskSumSqrtS14M01Tensor,
                                       blockDescriptors,
                                       index + 1,
                                       nnXLen,
                                       nnYLen)
    }

    /// Initialize a BlockStack object
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - blockDescriptors: The block descriptors
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         blockDescriptors: [BlockDescriptor],
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        resultTensor = BlockStack.processBlockDescriptors(graph,
                                                          sourceTensor,
                                                          maskTensor,
                                                          maskSumTensor,
                                                          maskSumSqrtS14M01Tensor,
                                                          blockDescriptors,
                                                          0,
                                                          nnXLen,
                                                          nnYLen)
    }
}

/// A structure that represents a nested bottleneck residual block
struct NestedBottleneckResidualBlock {
    /// The resulting tensor after processing the nested bottleneck residual block
    let resultTensor: MPSGraphTensor

    /// Initialize a ResidualBlock object
    ///
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - descriptor: The nested bottleneck residual block descriptor
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         descriptor: SWNestedBottleneckResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {

        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen)

        let preActivation = ActivationLayer(graph: graph,
                                            sourceTensor: preBN.resultTensor,
                                            activationKind: descriptor.preActivation)

        let preConv = ConvLayer(graph: graph,
                                sourceTensor: preActivation.resultTensor,
                                descriptor: descriptor.preConv,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen)

        let blocks = BlockStack(graph: graph,
                                sourceTensor: preConv.resultTensor,
                                maskTensor: maskTensor,
                                maskSumTensor: maskSumTensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                blockDescriptors: descriptor.blockDescriptors,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen)

        let postBN = BatchNormLayer(graph: graph,
                                    sourceTensor: blocks.resultTensor,
                                    maskTensor: maskTensor,
                                    descriptor: descriptor.postBN,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen)

        let postActivation = ActivationLayer(graph: graph,
                                             sourceTensor: postBN.resultTensor,
                                             activationKind: descriptor.postActivation)

        let postConv = ConvLayer(graph: graph,
                                 sourceTensor: postActivation.resultTensor,
                                 descriptor: descriptor.postConv,
                                 nnXLen: nnXLen,
                                 nnYLen: nnYLen)

        resultTensor = graph.addition(sourceTensor,
                                      postConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// Class representing the description of the SGF Metadata Encoder.
///
/// This encoder consists of three matrix multiplication layers, each followed by a bias and an activation function.
public class SWSGFMetadataEncoderDesc {
    /// Version of the SGF Metadata Encoder.
    let version: Int

    /// Number of input metadata channels.
    let numInputMetaChannels: Int

    /// Description of the first multiplication layer.
    let mul1: SWMatMulLayerDesc

    /// Description of the bias for the first layer.
    let bias1: SWMatBiasLayerDesc

    /// Activation kind for the first layer.
    let act1: ActivationKind

    /// Description of the second multiplication layer.
    let mul2: SWMatMulLayerDesc

    /// Description of the bias for the second layer.
    let bias2: SWMatBiasLayerDesc

    /// Activation kind for the second layer.
    let act2: ActivationKind

    /// Description of the third multiplication layer.
    let mul3: SWMatMulLayerDesc

    /// Initializes a new instance of the `SWSGFMetadataEncoderDesc` class.
    ///
    /// - Parameters:
    ///   - version: The version of the SGF Metadata Encoder.
    ///   - numInputMetaChannels: The number of input metadata channels.
    ///   - mul1: Description of the first multiplication layer.
    ///   - bias1: Description of the bias for the first layer.
    ///   - act1: Activation kind for the first layer.
    ///   - mul2: Description of the second multiplication layer.
    ///   - bias2: Description of the bias for the second layer.
    ///   - act2: Activation kind for the second layer.
    ///   - mul3: Description of the third multiplication layer.
    init(version: Int,
         numInputMetaChannels: Int,
         mul1: SWMatMulLayerDesc,
         bias1: SWMatBiasLayerDesc,
         act1: ActivationKind,
         mul2: SWMatMulLayerDesc,
         bias2: SWMatBiasLayerDesc,
         act2: ActivationKind,
         mul3: SWMatMulLayerDesc) {
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

/// Creates an instance of `SWSGFMetadataEncoderDesc` using the specified parameters.
///
/// - Parameters:
///   - version: An `Int32` representing the version of the encoder descriptor.
///   - numInputMetaChannels: An `Int32` specifying the number of input metadata channels.
///   - mul1: A `SWMatMulLayerDesc` representing the description of the first matrix multiplication layer.
///   - bias1: A `SWMatBiasLayerDesc` representing the description of the bias for the first layer.
///   - act1: An `ActivationKind` specifying the activation function applied after the first layer.
///   - mul2: A `SWMatMulLayerDesc` representing the description of the second matrix multiplication layer.
///   - bias2: A `SWMatBiasLayerDesc` representing the description of the bias for the second layer.
///   - act2: An `ActivationKind` specifying the activation function applied after the second layer.
///   - mul3: A `SWMatMulLayerDesc` representing the description of the third matrix multiplication layer.
///
/// - Returns:
///   An instance of `SWSGFMetadataEncoderDesc` initialized with the provided parameters.
public func createSWSGFMetadataEncoderDesc(version: Int32,
                                           numInputMetaChannels: Int32,
                                           mul1: SWMatMulLayerDesc,
                                           bias1: SWMatBiasLayerDesc,
                                           act1: ActivationKind,
                                           mul2: SWMatMulLayerDesc,
                                           bias2: SWMatBiasLayerDesc,
                                           act2: ActivationKind,
                                           mul3: SWMatMulLayerDesc) -> SWSGFMetadataEncoderDesc? {
    return SWSGFMetadataEncoderDesc(version: Int(version),
                                    numInputMetaChannels: Int(numInputMetaChannels),
                                    mul1: mul1,
                                    bias1: bias1,
                                    act1: act1,
                                    mul2: mul2,
                                    bias2: bias2,
                                    act2: act2,
                                    mul3: mul3)
}

/// A class that describes SGF metadata encoder.
/// SGFMetadataEncoder takes a graph, a descriptor object defining various parameters for the encoding process,
/// and an input tensor, and performs a sequence of matrix multiplications, bias additions, and activation functions
/// to produce a final encoded tensor.
class SGFMetadataEncoder {
    /// The resulting tensor after encoding the metadata.
    let resultTensor: MPSGraphTensor

    /// Initializes an `SGFMetadataEncoder` instance and performs the encoding process.
    ///
    /// - Parameters:
    ///   - graph: The computational graph object used to define and manage tensor operations.
    ///   - descriptor: An object holding all the required parameters, including matrix multiplication, biases,
    ///                 and activation functions for each layer.
    ///   - sourceTensor: The initial input tensor containing the metadata to be encoded.
    init(graph: MPSGraph,
         descriptor: SWSGFMetadataEncoderDesc,
         sourceTensor: MPSGraphTensor) {

        // First matrix multiplication layer.
        let mul1 = MatMulLayer(graph: graph,
                               descriptor: descriptor.mul1,
                               sourceTensor: sourceTensor)

        // Adding bias to the result of the first matrix multiplication.
        let bias1 = MatBiasLayer(graph: graph,
                                 descriptor: descriptor.bias1,
                                 sourceTensor: mul1.resultTensor)

        // Applying the first activation function to the biased tensor.
        let act1 = ActivationLayer(graph: graph,
                                   sourceTensor: bias1.resultTensor,
                                   activationKind: descriptor.act1)

        // Second matrix multiplication layer taking the output of the first activation layer.
        let mul2 = MatMulLayer(graph: graph,
                               descriptor: descriptor.mul2,
                               sourceTensor: act1.resultTensor)

        // Adding bias to the result of the second matrix multiplication.
        let bias2 = MatBiasLayer(graph: graph,
                                 descriptor: descriptor.bias2,
                                 sourceTensor: mul2.resultTensor)

        // Applying the second activation function to the biased tensor.
        let act2 = ActivationLayer(graph: graph,
                                   sourceTensor: bias2.resultTensor,
                                   activationKind: descriptor.act2)

        // Third and final matrix multiplication layer taking the output of the second activation layer.
        let mul3 = MatMulLayer(graph: graph,
                               descriptor: descriptor.mul3,
                               sourceTensor: act2.resultTensor)

        // Setting the final result tensor to the output of the last matrix multiplication layer.
        resultTensor = mul3.resultTensor

        assert(resultTensor.shape?.count == 2)
    }
}

/// A class that describes a trunk for a neural network
public class SWTrunkDesc {
    /// The version of the ResNet trunk
    let version: Int
    /// Number of channels for the trunk
    let trunkNumChannels: NSNumber
    /// Number of channels for the mid section
    let midNumChannels: NSNumber
    /// Number of channels for the regular section
    let regularNumChannels: NSNumber
    /// Number of channels for the global pooling section
    let gpoolNumChannels: NSNumber
    /// The description of the initial convolutional layer
    let initialConv: SWConvLayerDesc
    /// The description of the initial matrix multiplication layer
    let initialMatMul: SWMatMulLayerDesc
    /// The description of the SGF metadata encoder
    let sgfMetadataEncoder: SWSGFMetadataEncoderDesc?
    /// The list of blocks that make up the trunk
    let blockDescriptors: [BlockDescriptor]
    /// The description of the batch normalization layer that is applied at the end of the trunk
    let trunkTipBN: SWBatchNormLayerDesc
    /// The activation function that is applied at the end of the trunk
    let trunkTipActivation: ActivationKind

    /// Initializes a SWTrunkDesc object
    /// - Parameters:
    ///   - version: The version of the ResNet trunk
    ///   - trunkNumChannels: Number of channels for the trunk
    ///   - midNumChannels: Number of channels for the mid section
    ///   - regularNumChannels: Number of channels for the regular section
    ///   - gpoolNumChannels: Number of channels for the global pooling section
    ///   - initialConv: The description of the initial convolutional layer
    ///   - initialMatMul: The description of the initial matrix multiplication layer
    ///   - sgfMetadataEncoder: The description of the SGF metadata encoder
    ///   - blockDescriptors: The list of blocks that make up the trunk
    ///   - trunkTipBN: The description of the batch normalization layer that is applied at the end of the trunk
    ///   - trunkTipActivation: The activation function that is applied at the end of the trunk
    init(version: Int,
         trunkNumChannels: NSNumber,
         midNumChannels: NSNumber,
         regularNumChannels: NSNumber,
         gpoolNumChannels: NSNumber,
         initialConv: SWConvLayerDesc,
         initialMatMul: SWMatMulLayerDesc,
         sgfMetadataEncoder: SWSGFMetadataEncoderDesc?,
         blockDescriptors: [BlockDescriptor],
         trunkTipBN: SWBatchNormLayerDesc,
         trunkTipActivation: ActivationKind) {
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

public func createSWTrunkDesc(version: Int32,
                              trunkNumChannels: Int32,
                              midNumChannels: Int32,
                              regularNumChannels: Int32,
                              gpoolNumChannels: Int32,
                              initialConv: SWConvLayerDesc,
                              initialMatMul: SWMatMulLayerDesc,
                              sgfMetadataEncoder: SWSGFMetadataEncoderDesc?,
                              blockDescriptors: [BlockDescriptor],
                              trunkTipBN: SWBatchNormLayerDesc,
                              trunkTipActivation: ActivationKind) -> SWTrunkDesc {
    return SWTrunkDesc(version: Int(version),
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
    /// The resulting tensor after processing the trunk
    let resultTensor: MPSGraphTensor

    /// Returns the block source tensor by processing the input meta tensor, if available, and adding a bias term.
    ///
    /// - Parameters:
    ///     - graph: The Metal Performance Shaders (MPS) graph.
    ///     - descriptor: The SGF metadata encoder descriptor.
    ///     - initialAdd: The initial add operation result tensor.
    ///     - inputMetaTensor: The input meta tensor.
    ///     - nnXLen: The X length of the neural network (NN).
    ///     - nnYLen: The Y length of the neural network (NN).
    ///     - numChannels: The number of channels of the initial add operation result tensor.
    ///
    /// - Returns:
    ///     - blockSourceTensor: The processed block source tensor.
    ///
    /// This function is used to get the block source tensor by processing the input meta tensor, if available.
    /// If the input meta tensor is not available, it returns the result tensor from the initial add operation.
    /// The function uses SGF metadata encoder and AddNCBiasLayer to process the input meta tensor.
    static func getBlockSourceTensor(graph: MPSGraph,
                                     descriptor: SWSGFMetadataEncoderDesc?,
                                     initialAdd: AddNCBiasLayer,
                                     inputMetaTensor: MPSGraphTensor?,
                                     nnXLen: NSNumber,
                                     nnYLen: NSNumber,
                                     numChannels: NSNumber) -> MPSGraphTensor {
        var blockSourceTensor: MPSGraphTensor

        if let inputMetaTensor,
           let descriptor, descriptor.numInputMetaChannels > 0 {
            let encoded = SGFMetadataEncoder(graph: graph,
                                            descriptor: descriptor,
                                            sourceTensor: inputMetaTensor)

            let encodedAdd = AddNCBiasLayer(graph: graph,
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

    /// Initializes a Trunk object
    /// - Parameters:
    ///   - graph: The graph used to build the trunk
    ///   - descriptor: A SWTrunkDesc object that describes the trunk
    ///   - inputTensor: The input tensor
    ///   - inputGlobalTensor: The input global tensor
    ///   - inputMetaTensor: The input meta tensor
    ///   - maskTensor: The tensor used to mask input activations
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - nnXLen: The length of the X dimension of the input tensor
    ///   - nnYLen: The length of the Y dimension of the input tensor
    init(graph: MPSGraph,
         descriptor: SWTrunkDesc,
         inputTensor: MPSGraphTensor,
         inputGlobalTensor: MPSGraphTensor,
         inputMetaTensor: MPSGraphTensor?,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {

        let initialConv = ConvLayer(graph: graph,
                                    sourceTensor: inputTensor,
                                    descriptor: descriptor.initialConv,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen)

        let initialMatMul = MatMulLayer(graph: graph,
                                        descriptor: descriptor.initialMatMul,
                                        sourceTensor: inputGlobalTensor)

        let initialAdd = AddNCBiasLayer(graph: graph,
                                        sourceTensor: initialConv.resultTensor,
                                        biasTensor: initialMatMul.resultTensor,
                                        nnXLen: nnXLen,
                                        nnYLen: nnYLen,
                                        numChannels: descriptor.initialMatMul.outChannels)

        let blockSourceTensor = Trunk.getBlockSourceTensor(graph: graph,
                                                           descriptor: descriptor.sgfMetadataEncoder,
                                                           initialAdd: initialAdd,
                                                           inputMetaTensor: inputMetaTensor,
                                                           nnXLen: nnXLen,
                                                           nnYLen: nnYLen,
                                                           numChannels: descriptor.initialMatMul.outChannels)

        let blocks = BlockStack(graph: graph,
                                sourceTensor: blockSourceTensor,
                                maskTensor: maskTensor,
                                maskSumTensor: maskSumTensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                blockDescriptors: descriptor.blockDescriptors,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen)

        let trunkTipBN = BatchNormLayer(graph: graph,
                                        sourceTensor: blocks.resultTensor,
                                        maskTensor: maskTensor,
                                        descriptor: descriptor.trunkTipBN,
                                        nnXLen: nnXLen,
                                        nnYLen: nnYLen)

        let trunkTipActivation = ActivationLayer(graph: graph,
                                                 sourceTensor: trunkTipBN.resultTensor,
                                                 activationKind: descriptor.trunkTipActivation)

        resultTensor = trunkTipActivation.resultTensor

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that describes a policy head for a neural network, responsible for predicting
/// the best moves for the current player and the opposing player on the subsequent turn.
public struct SWPolicyHeadDesc {
    /// The version of the policy head
    let version: Int
    /// The 1x1 convolution layer for P
    let p1Conv: SWConvLayerDesc
    /// The 1x1 convolution layer for G
    let g1Conv: SWConvLayerDesc
    /// The batch normalization layer for G
    let g1BN: SWBatchNormLayerDesc
    /// The activation function for G
    let g1Activation: ActivationKind
    /// The global pooling bias structure that pools the output of G to bias the output of P
    let gpoolToBiasMul: SWMatMulLayerDesc
    /// The batch normalization layer for P
    let p1BN: SWBatchNormLayerDesc
    /// The activation function for P
    let p1Activation: ActivationKind
    /// The 1x1 convolution layer with 2 channels for outputting two policy distributions
    let p2Conv: SWConvLayerDesc
    /// The fully connected linear layer for outputting logits for the pass move
    let gpoolToPassMul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer for model version >= 15
    let gpoolToPassBias: SWMatBiasLayerDesc?
    /// The activation function for the bias layer in model version >= 15
    let passActivation: ActivationKind?
    /// The fully connected linear layer for outputting logits for the pass move in model version >= 15
    let gpoolToPassMul2: SWMatMulLayerDesc?

    /// Initializes a SWPolicyHeadDesc object with the given parameters
    /// - Parameters:
    ///   - version: The version of the policy head
    ///   - p1Conv: The 1x1 convolution layer for P
    ///   - g1Conv: The 1x1 convolution layer for G
    ///   - g1BN: The batch normalization layer for G
    ///   - g1Activation: The activation function for G
    ///   - gpoolToBiasMul: The global pooling bias structure that pools the output of G to bias the output of P
    ///   - p1BN: The batch normalization layer for P
    ///   - p1Activation: The activation function for P
    ///   - p2Conv: The 1x1 convolution layer with 2 channels for outputting two policy distributions
    ///   - gpoolToPassMul: The fully connected linear layer for outputting logits for the pass move
    init(version: Int,
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
         gpoolToPassMul2: SWMatMulLayerDesc?) {
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

        assert((version >= 15) || ((gpoolToPassBias == nil) && (passActivation == nil) && (gpoolToPassMul2 == nil)))
        assert((version < 15) || ((gpoolToPassBias != nil) && (passActivation != nil) && (gpoolToPassMul2 != nil)))
    }
}

public func createSWPolicyHeadDesc(version: Int32,
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
                                   gpoolToPassMul2: SWMatMulLayerDesc) -> SWPolicyHeadDesc {
    if version >= 15 {
        return SWPolicyHeadDesc(version: Int(version),
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
        return SWPolicyHeadDesc(version: Int(version),
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
    /// The tensor that holds the policy prediction of the neural network
    let policyTensor: MPSGraphTensor
    /// The tensor that holds the policy pass of the neural network
    let policyPassTensor: MPSGraphTensor

    /// Initializes a PolicyHead object
    /// - Parameters:
    ///   - graph: The MPSGraph object to which the policy head is added
    ///   - descriptor: The description of the policy head
    ///   - sourceTensor: The input tensor to the policy head
    ///   - maskTensor: The mask tensor for the input tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor and a small epsilon
    ///   - nnXLen: The number of X pixels in the input tensor
    ///   - nnYLen: The number of Y pixels in the input tensor
    init(graph: MPSGraph,
         descriptor: SWPolicyHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {

        let p1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.p1Conv,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        let g1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.g1Conv,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        let g1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: g1Conv.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.g1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        let g1Activation = ActivationLayer(graph: graph,
                                           sourceTensor: g1BN.resultTensor,
                                           activationKind: descriptor.g1Activation)

        let g1Concat = GlobalPoolingLayer(graph: graph,
                                          sourceTensor: g1Activation.resultTensor,
                                          maskSumTensor: maskSumTensor,
                                          maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor)

        assert(g1Concat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels)

        let gpoolToBiasMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToBiasMul,
                                         sourceTensor: g1Concat.resultTensor)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: p1Conv.resultTensor,
                                   biasTensor: gpoolToBiasMul.resultTensor,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.gpoolToBiasMul.outChannels)

        let p1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: added.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.p1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        let p1Activation = ActivationLayer(graph: graph,
                                           sourceTensor: p1BN.resultTensor,
                                           activationKind: descriptor.p1Activation)

        let p2Conv = ConvLayer(graph: graph,
                               sourceTensor: p1Activation.resultTensor,
                               descriptor: descriptor.p2Conv,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        policyTensor = p2Conv.resultTensor

        assert(g1Concat.resultTensor.shape?[1] == descriptor.gpoolToPassMul.inChannels)

        let gpoolToPassMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToPassMul,
                                         sourceTensor: g1Concat.resultTensor)

        if let gpoolToPassBias = descriptor.gpoolToPassBias,
           let passActivation = descriptor.passActivation,
           let gpoolToPassMul2 = descriptor.gpoolToPassMul2 {
            assert(descriptor.version >= 15)

            let gpoolToPassBiasLayer = MatBiasLayer(graph: graph,
                                                    descriptor: gpoolToPassBias,
                                                    sourceTensor: gpoolToPassMul.resultTensor)

            let passActivationLayer = ActivationLayer(graph: graph,
                                                      sourceTensor: gpoolToPassBiasLayer.resultTensor,
                                                      activationKind: passActivation)

            let gpoolToPassMul2Layer = MatMulLayer(graph: graph,
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

/// A struct that describes the value head of a neural network
public struct SWValueHeadDesc {
    /// The version of the value head
    let version: Int
    /// The description of the first convolutional layer in the value head
    let v1Conv: SWConvLayerDesc
    /// The description of the batch normalization layer after the first convolutional layer in the value head
    let v1BN: SWBatchNormLayerDesc
    /// The activation function that is applied after the first batch normalization layer in the value head
    let v1Activation: ActivationKind
    /// The description of the matrix multiplication layer that is applied to the output of the first convolutional layer in the value head
    let v2Mul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    let v2Bias: SWMatBiasLayerDesc
    /// The activation function that is applied after the bias layer in the value head
    let v2Activation: ActivationKind
    /// The description of the matrix multiplication layer that is applied to the output of the bias layer in the value head
    let v3Mul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    let v3Bias: SWMatBiasLayerDesc
    /// The description of the matrix multiplication layer that is applied to the output of the third bias layer in the value head
    let sv3Mul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    let sv3Bias: SWMatBiasLayerDesc
    /// The description of the convolutional layer that is applied to the board ownership map in the value head
    let vOwnershipConv: SWConvLayerDesc

    /// Initializes a SWValueHeadDesc object
    /// - Parameters:
    ///   - version: The version of the value head
    ///   - v1Conv: The description of the first convolutional layer in the value head
    ///   - v1BN: The description of the batch normalization layer after the first convolutional layer in the value head
    ///   - v1Activation: The activation function that is applied after the first batch normalization layer in the value head
    ///   - v2Mul: The description of the matrix multiplication layer that is applied to the output of the first convolutional layer in the value head
    ///   - v2Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - v2Activation: The activation function that is applied after the bias layer in the value head
    ///   - v3Mul: The description of the matrix multiplication layer that is applied to the output of the bias layer in the value head
    ///   - v3Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - sv3Mul: The description of the matrix multiplication layer that is applied to the output of the third bias layer in the value head
    ///   - sv3Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - vOwnershipConv: The description of the convolutional layer that is applied to the board ownership map in the value head
    init(version: Int,
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
         vOwnershipConv: SWConvLayerDesc) {
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

public func createSWValueHeadDesc(version: Int32,
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
                                  vOwnershipConv: SWConvLayerDesc) -> SWValueHeadDesc {
    return SWValueHeadDesc(version: Int(version),
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

/// A structure that creates a value head for the neural network, which produces the value, score value, and ownership tensors.
struct ValueHead {
    /// The tensor that represents the value of the board
    let valueTensor: MPSGraphTensor
    /// The tensor that represents the score value of the board
    let scoreValueTensor: MPSGraphTensor
    /// The tensor that represents the ownership of the board
    let ownershipTensor: MPSGraphTensor

    /// Initializes the value head using a graph, a descriptor, a source tensor, and other relevant tensors.
    /// - Parameters:
    ///   - graph: The graph used to perform calculations on tensors
    ///   - descriptor: The SWValueHeadDesc object that describes the value head
    ///   - sourceTensor: The tensor used to source data to the neural network
    ///   - maskTensor: The tensor used to mask out invalid moves
    ///   - maskSumTensor: The tensor used to sum up the mask tensor values
    ///   - maskSumSqrtS14M01Tensor: The tensor used to calculate a square root value
    ///   - maskSumSqrtS14M01SquareS01Tensor: The tensor used to calculate a square value
    ///   - nnXLen: The x-axis length of the neural network
    ///   - nnYLen: The y-axis length of the neural network
    init(graph: MPSGraph,
         descriptor: SWValueHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {

        let v1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.v1Conv,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        let v1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: v1Conv.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.v1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        let v1Activation = ActivationLayer(graph: graph,
                                           sourceTensor: v1BN.resultTensor,
                                           activationKind: descriptor.v1Activation)

        let v1Mean =
        GlobalPoolingValueLayer(graph: graph,
                                sourceTensor: v1Activation.resultTensor,
                                maskSumTensor: maskSumTensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01Tensor)

        assert(v1Mean.resultTensor.shape?[1] == descriptor.v2Mul.inChannels)

        let v2Mul = MatMulLayer(graph: graph,
                                descriptor: descriptor.v2Mul,
                                sourceTensor: v1Mean.resultTensor)

        let v2Bias = MatBiasLayer(graph: graph,
                                  descriptor: descriptor.v2Bias,
                                  sourceTensor: v2Mul.resultTensor)

        let v2Activation = ActivationLayer(graph: graph,
                                           sourceTensor: v2Bias.resultTensor,
                                           activationKind: descriptor.v2Activation)

        let v3Mul = MatMulLayer(graph: graph,
                                descriptor: descriptor.v3Mul,
                                sourceTensor: v2Activation.resultTensor)

        let v3Bias = MatBiasLayer(graph: graph,
                                  descriptor: descriptor.v3Bias,
                                  sourceTensor: v3Mul.resultTensor)

        let sv3Mul = MatMulLayer(graph: graph,
                                 descriptor: descriptor.sv3Mul,
                                 sourceTensor: v2Activation.resultTensor)

        let sv3Bias = MatBiasLayer(graph: graph,
                                   descriptor: descriptor.sv3Bias,
                                   sourceTensor: sv3Mul.resultTensor)

        let vOwnershipConv = ConvLayer(graph: graph,
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


/// A struct that describes a neural network model used for playing the game of Go.
public struct SWModelDesc {

    static let defaultDesc = createDefaultDesc()

    static func createDefaultDesc() -> SWModelDesc {

        var unityConvWeights = [Float](repeating: 1, count: 1)
        var unityMatMulWeights = [Float](repeating: 1, count: 1)
        var meanWeights = [Float](repeating: 0, count: 1)
        var varianceWeights = [Float](repeating: 0.9, count: 1)
        var scaleWeights = [Float](repeating: 1, count: 1)
        var biasWeights = [Float](repeating: 0, count: 1)
        var gpoolMatMulWeights = [Float](repeating: 3, count: 3)
        var zeroMatBiasWeights = [Float](repeating: 0, count: 1)

        let unityConv = SWConvLayerDesc(convYSize: 1,
                                        convXSize: 1,
                                        inChannels: 1,
                                        outChannels: 1,
                                        dilationY: 1,
                                        dilationX: 1,
                                        weights: &unityConvWeights)

        let unityMatMul = SWMatMulLayerDesc(inChannels: 1,
                                            outChannels: 1,
                                            weights: &unityMatMulWeights)


        let unityBatchNorm = SWBatchNormLayerDesc(numChannels: 1,
                                                  epsilon: 0.1,
                                                  hasScale: false,
                                                  hasBias: false,
                                                  mean: &meanWeights,
                                                  variance: &varianceWeights,
                                                  scale: &scaleWeights,
                                                  bias: &biasWeights)

        let unityResidual = SWResidualBlockDesc(preBN: unityBatchNorm,
                                                preActivation: ActivationKind.relu,
                                                regularConv: unityConv,
                                                midBN: unityBatchNorm,
                                                midActivation: ActivationKind.relu,
                                                finalConv: unityConv)

        let gpoolMatMul = SWMatMulLayerDesc(inChannels: 3,
                                            outChannels: 1,
                                            weights: &gpoolMatMulWeights)

        let globalPooling =
        SWGlobalPoolingResidualBlockDesc(preBN: unityBatchNorm,
                                         preActivation: ActivationKind.relu,
                                         regularConv: unityConv,
                                         gpoolConv: unityConv,
                                         gpoolBN: unityBatchNorm,
                                         gpoolActivation: ActivationKind.relu,
                                         gpoolToBiasMul: gpoolMatMul,
                                         midBN: unityBatchNorm,
                                         midActivation: ActivationKind.relu,
                                         finalConv: unityConv)

        let blocks: [BlockDescriptor] = [unityResidual,
                                         BlockDescriptor(),
                                         globalPooling,
                                         unityResidual]

        let trunkDesc = SWTrunkDesc(version: 0,
                                    trunkNumChannels: 1,
                                    midNumChannels: 1,
                                    regularNumChannels: 1,
                                    gpoolNumChannels: 1,
                                    initialConv: unityConv,
                                    initialMatMul: unityMatMul,
                                    sgfMetadataEncoder: nil,
                                    blockDescriptors: blocks,
                                    trunkTipBN: unityBatchNorm,
                                    trunkTipActivation: ActivationKind.relu)

        let policyHead = SWPolicyHeadDesc(version: 0,
                                          p1Conv: unityConv,
                                          g1Conv: unityConv,
                                          g1BN: unityBatchNorm,
                                          g1Activation: ActivationKind.relu,
                                          gpoolToBiasMul: gpoolMatMul,
                                          p1BN: unityBatchNorm,
                                          p1Activation: ActivationKind.relu,
                                          p2Conv: unityConv,
                                          gpoolToPassMul: gpoolMatMul,
                                          gpoolToPassBias: nil,
                                          passActivation: nil,
                                          gpoolToPassMul2: nil)

        let zeroMatBias = SWMatBiasLayerDesc(numChannels: 1,
                                             weights: &zeroMatBiasWeights)

        let valueHead = SWValueHeadDesc(version: 0,
                                        v1Conv: unityConv,
                                        v1BN: unityBatchNorm,
                                        v1Activation: ActivationKind.relu,
                                        v2Mul: gpoolMatMul,
                                        v2Bias: zeroMatBias,
                                        v2Activation: ActivationKind.relu,
                                        v3Mul: unityMatMul,
                                        v3Bias: zeroMatBias,
                                        sv3Mul: unityMatMul,
                                        sv3Bias: zeroMatBias,
                                        vOwnershipConv: unityConv)

        let modelDesc = createSWModelDesc(version: 8,
                                          name: "default",
                                          numInputChannels: 1,
                                          numInputGlobalChannels: 1,
                                          numInputMetaChannels: 0,
                                          numValueChannels: 1,
                                          numScoreValueChannels: 1,
                                          numOwnershipChannels: 1,
                                          trunk: trunkDesc,
                                          policyHead: policyHead,
                                          valueHead: valueHead)

        return modelDesc
    }

    /// The version of the model.
    let version: Int
    /// The name of the model.
    let name: String
    /// Number of channels for input features.
    let numInputChannels: NSNumber
    /// Number of channels for global input features.
    let numInputGlobalChannels: NSNumber
    /// Number of channels for meta input features.
    let numInputMetaChannels: NSNumber
    /// Number of channels for the value head output.
    let numValueChannels: NSNumber
    /// Number of channels for the score value head output.
    let numScoreValueChannels: NSNumber
    /// Number of channels for the ownership head output.
    let numOwnershipChannels: NSNumber
    /// The description of the trunk that makes up the backbone of the model.
    let trunk: SWTrunkDesc
    /// The description of the policy head that predicts the probability of playing at a particular position.
    let policyHead: SWPolicyHeadDesc
    /// The description of the value head that predicts the expected outcome of a game state.
    let valueHead: SWValueHeadDesc

    /// Initializes an SWModelDesc object.
    /// - Parameters:
    ///   - version: The version of the model.
    ///   - name: The name of the model.
    ///   - numInputChannels: Number of channels for input features.
    ///   - numInputGlobalChannels: Number of channels for global input features.
    ///   - numInputMetaChannels: Number of channels for meta input features.
    ///   - numValueChannels: Number of channels for the value head output.
    ///   - numScoreValueChannels: Number of channels for the score value head output.
    ///   - numOwnershipChannels: Number of channels for the ownership head output.
    ///   - trunk: The description of the trunk that makes up the backbone of the model.
    ///   - policyHead: The description of the policy head that predicts the probability of playing at a particular position.
    ///   - valueHead: The description of the value head that predicts the expected outcome of a game state.
    init(version: Int,
         name: String,
         numInputChannels: NSNumber,
         numInputGlobalChannels: NSNumber,
         numInputMetaChannels: NSNumber,
         numValueChannels: NSNumber,
         numScoreValueChannels: NSNumber,
         numOwnershipChannels: NSNumber,
         trunk: SWTrunkDesc,
         policyHead: SWPolicyHeadDesc,
         valueHead: SWValueHeadDesc) {
        self.version = version
        self.name = name
        self.numInputChannels = numInputChannels
        self.numInputGlobalChannels = numInputGlobalChannels
        self.numInputMetaChannels = numInputMetaChannels
        self.numValueChannels = numValueChannels
        self.numScoreValueChannels = numScoreValueChannels
        self.numOwnershipChannels = numOwnershipChannels
        self.trunk = trunk
        self.policyHead = policyHead
        self.valueHead = valueHead
    }
}

public func createSWModelDesc(version: Int32,
                              name: String,
                              numInputChannels: Int32,
                              numInputGlobalChannels: Int32,
                              numInputMetaChannels: Int32,
                              numValueChannels: Int32,
                              numScoreValueChannels: Int32,
                              numOwnershipChannels: Int32,
                              trunk: SWTrunkDesc,
                              policyHead: SWPolicyHeadDesc,
                              valueHead: SWValueHeadDesc) -> SWModelDesc {
    return SWModelDesc(version: Int(version),
                       name: name,
                       numInputChannels: numInputChannels as NSNumber,
                       numInputGlobalChannels: numInputGlobalChannels as NSNumber,
                       numInputMetaChannels: numInputMetaChannels as NSNumber,
                       numValueChannels: numValueChannels as NSNumber,
                       numScoreValueChannels: numScoreValueChannels as NSNumber,
                       numOwnershipChannels: numOwnershipChannels as NSNumber,
                       trunk: trunk,
                       policyHead: policyHead,
                       valueHead: valueHead)
}

/// A structure representing a neural network model for processing Go game states.
struct Model {

    static let defaultNnXLen: NSNumber = 19
    static let defaultNnYLen: NSNumber = 19

    static let defaultModel = Model(device: DefaultDevice.device,
                                    graph: MPSGraph(),
                                    descriptor: SWModelDesc.defaultDesc,
                                    nnXLen: defaultNnXLen,
                                    nnYLen: defaultNnYLen)

    /// The Metal device
    let device: MTLDevice
    /// The command queue used to execute the graph on the GPU
    let commandQueue: MTLCommandQueue
    /// The Metal Performance Shaders graph object used for building and executing the graph
    let graph: MPSGraph
    /// The length of the neural network input in the x dimension
    let nnXLen: NSNumber
    /// The length of the neural network input in the y dimension
    let nnYLen: NSNumber
    /// The version of the model
    let version: Int
    /// The number of channels in the value output layer
    let numValueChannels: NSNumber
    /// The number of channels in the score value output layer
    let numScoreValueChannels: NSNumber
    /// The number of channels in the ownership output layer
    let numOwnershipChannels: NSNumber
    /// The input layer of the neural network
    let input: InputLayer
    /// The global input layer of the neural network
    let inputGlobal: InputGlobalLayer
    /// The meta input layer of the neural network
    let inputMeta: InputMetaLayer
    /// The mask layer of the neural network
    let mask: MaskLayer
    /// The trunk of the neural network
    let trunk: Trunk
    /// The policy head of the neural network
    let policyHead: PolicyHead
    /// The value head of the neural network
    let valueHead: ValueHead
    /// The dictionary that maps the output tensors to the tensor data
    let targetTensors: [MPSGraphTensor]

    /// Initializes a Model object.
    /// - Parameters:
    ///   - device: The Metal device to use for computations.
    ///   - graph: The Metal Performance Shaders graph object used for building and executing the graph.
    ///   - descriptor: The description of the model.
    ///   - nnXLen: The length of the neural network input in the x dimension.
    ///   - nnYLen: The length of the neural network input in the y dimension.
    init(device: MTLDevice,
         graph: MPSGraph,
         descriptor: SWModelDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.graph = graph
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.version = descriptor.version
        self.numValueChannels = descriptor.numValueChannels
        self.numScoreValueChannels = descriptor.numScoreValueChannels
        self.numOwnershipChannels = descriptor.numOwnershipChannels

        input = InputLayer(graph: graph,
                           nnXLen: nnXLen,
                           nnYLen: nnYLen,
                           numChannels: descriptor.numInputChannels)

        inputGlobal = InputGlobalLayer(graph: graph,
                                       numGlobalFeatures: descriptor.numInputGlobalChannels)

        inputMeta = InputMetaLayer(graph: graph,
                                   numMetaFeatures: descriptor.numInputMetaChannels)

        mask = MaskLayer(graph: graph,
                         nnXLen: nnXLen,
                         nnYLen: nnYLen)

        let maskSum = MaskSumLayer(graph: graph,
                                   maskTensor: mask.tensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum)

        let maskSumSqrtS14M01SquareS01 = MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                                                         maskSumSqrtS14M01: maskSumSqrtS14M01)

        trunk = Trunk(graph: graph,
                      descriptor: descriptor.trunk,
                      inputTensor: input.tensor,
                      inputGlobalTensor: inputGlobal.tensor,
                      inputMetaTensor: inputMeta.tensor,
                      maskTensor: mask.tensor,
                      maskSumTensor: maskSum.tensor,
                      maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                      nnXLen: nnXLen,
                      nnYLen: nnYLen)

        policyHead = PolicyHead(graph: graph,
                                descriptor: descriptor.policyHead,
                                sourceTensor: trunk.resultTensor,
                                maskTensor: mask.tensor,
                                maskSumTensor: maskSum.tensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen)

        valueHead = ValueHead(graph: graph,
                              descriptor: descriptor.valueHead,
                              sourceTensor: trunk.resultTensor,
                              maskTensor: mask.tensor,
                              maskSumTensor: maskSum.tensor,
                              maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                              maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
                              nnXLen: nnXLen,
                              nnYLen: nnYLen)

        targetTensors = [policyHead.policyTensor,
                         policyHead.policyPassTensor,
                         valueHead.valueTensor,
                         valueHead.scoreValueTensor,
                         valueHead.ownershipTensor]
    }

    /// Applies the model to the given input data, and generates predictions for policy, value and ownership
    /// - Parameters:
    ///   - inputPointer: UnsafeMutablePointer to a flattened 2D array of floats representing the input state
    ///   - inputGlobalPointer: UnsafeMutablePointer to a flattened array of floats representing global state features
    ///   - inputMetaPointer: UnsafeMutablePointer to a flattened array of floats representing the metadata
    ///   - policy: UnsafeMutablePointer to a flattened 2D array of floats representing predicted policy
    ///   - policyPass: UnsafeMutablePointer to a flattened array of floats representing predicted probability of passing
    ///   - value: UnsafeMutablePointer to a flattened array of floats representing predicted value
    ///   - scoreValue: UnsafeMutablePointer to a flattened array of floats representing predicted score value
    ///   - ownership: UnsafeMutablePointer to a flattened 2D array of floats representing predicted ownership
    ///   - batchSize: The batch size
    func apply(input inputPointer: UnsafeMutablePointer<Float32>,
               inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
               inputMeta inputMetaPointer: UnsafeMutablePointer<Float32>,
               policy: UnsafeMutablePointer<Float32>,
               policyPass: UnsafeMutablePointer<Float32>,
               value: UnsafeMutablePointer<Float32>,
               scoreValue: UnsafeMutablePointer<Float32>,
               ownership: UnsafeMutablePointer<Float32>,
               batchSize: Int) {

        let channelAxis = InputShape.getChannelAxis()
        let numInputChannels = input.shape[channelAxis]

        let inputShape = InputShape.create(batchSize: batchSize as NSNumber,
                                           numChannels: numInputChannels,
                                           nnYLen: nnYLen,
                                           nnXLen: nnXLen)

        let inputDescriptor = MPSNDArrayDescriptor(dataType: input.tensor.dataType,
                                                   shape: inputShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)

        let numInputGlobalChannels = inputGlobal.shape[channelAxis]

        let inputGlobalShape = InputShape.create(batchSize: batchSize as NSNumber,
                                                 numChannels: numInputGlobalChannels,
                                                 nnYLen: 1,
                                                 nnXLen: 1)

        let inputGlobalDescriptor = MPSNDArrayDescriptor(dataType: inputGlobal.tensor.dataType,
                                                         shape: inputGlobalShape)

        let inputGlobalArray = MPSNDArray(device: device,
                                          descriptor: inputGlobalDescriptor)

        inputGlobalArray.writeBytes(inputGlobalPointer)

        let numInputMetaChannels = inputMeta.shape[channelAxis]

        let inputMetaShape = InputShape.create(batchSize: batchSize as NSNumber,
                                               numChannels: numInputMetaChannels,
                                               nnYLen: 1,
                                               nnXLen: 1)

        let inputMetaDescriptor = MPSNDArrayDescriptor(dataType: inputMeta.tensor.dataType,
                                                       shape: inputMetaShape)

        let inputMetaArray = MPSNDArray(device: device,
                                        descriptor: inputMetaDescriptor)

        inputMetaArray.writeBytes(inputMetaPointer)

        let maskShape = InputShape.create(batchSize: batchSize as NSNumber,
                                          numChannels: 1,
                                          nnYLen: nnYLen,
                                          nnXLen: nnXLen)

        let maskDescriptor = MPSNDArrayDescriptor(dataType: mask.tensor.dataType,
                                                  shape: maskShape)

        let maskArray = MPSNDArray(device: device,
                                   descriptor: maskDescriptor)

        var maskStrideArray = [MemoryLayout<Float32>.size,
                               nnXLen.intValue * MemoryLayout<Float32>.size,
                               nnYLen.intValue * nnXLen.intValue * MemoryLayout<Float32>.size,
                               numInputChannels.intValue * nnYLen.intValue * nnXLen.intValue * MemoryLayout<Float32>.size]

        maskArray.writeBytes(inputPointer, strideBytes: &maskStrideArray)

        let feeds = [input.tensor: MPSGraphTensorData(inputArray),
                     inputGlobal.tensor: MPSGraphTensorData(inputGlobalArray),
                     inputMeta.tensor: MPSGraphTensorData(inputMetaArray),
                     mask.tensor: MPSGraphTensorData(maskArray)]

        let fetch = graph.run(with: commandQueue,
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

// A enum to represent enabled/disabled/auto option of a feature.
public enum SWEnable {
    case False
    case True
    case Auto
}

/// A class that represents context of GPU devices.
public class MetalComputeContext {

    static let defaultNnXLen: NSNumber = 19
    static let defaultNnYLen: NSNumber = 19
    static let defaultId: Int32 = -1

    static let defaultContext = MetalComputeContext(nnXLen: defaultNnXLen,
                                                    nnYLen: defaultNnYLen,
                                                    id: defaultId)

    static var contexts: [Int32: MetalComputeContext] = [:]

    static let initialId: Int32 = 0
    static private var nextId: Int32 = initialId

    private class func getNextId() -> Int32 {
        let id = nextId
        nextId = nextId + 1
        return id
    }

    /// Create a context.
    /// - Parameters:
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    /// - Returns: The ID of the compute context
    class func createInstance(nnXLen: NSNumber,
                              nnYLen: NSNumber) -> Int32 {

        let id = getNextId()

        let context = MetalComputeContext(nnXLen: nnXLen,
                                          nnYLen: nnYLen,
                                          id: id)

        contexts[id] = context

        print("Metal compute context \(id): \(nnXLen)x\(nnYLen)",
              to: &StandardError.instance)

        return id
    }

    /// Destroy the context.
    class func destroyInstance(id: Int32) {
        contexts[id] = nil
    }

    /// Get the context.
    /// - Returns: The context.
    class func getInstance(id: Int32) -> MetalComputeContext {
        return contexts[id] ?? defaultContext
    }

    let nnXLen: NSNumber
    let nnYLen: NSNumber
    let id: Int32

    /// Initialize a context.
    /// - Parameters:
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    ///   - id: The ID of the compute context
    private init(nnXLen: NSNumber,
                 nnYLen: NSNumber,
                 id: Int32) {
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.id = id
    }
}

public func createMetalComputeContext(nnXLen: Int32,
                                      nnYLen: Int32) -> Int32 {

    return MetalComputeContext.createInstance(nnXLen: nnXLen as NSNumber,
                                              nnYLen: nnYLen as NSNumber)
}

public func destroyMetalComputeContext(id: Int32) {
    MetalComputeContext.destroyInstance(id: id)
}

/// A class that represents a handle of GPU device.
public class MetalComputeHandle {
    static let defaultId: Int32 = -1
    static let defaultHandle = MetalComputeHandle(model: Model.defaultModel, id: defaultId)
    static var handles: [Int32: MetalComputeHandle] = [:]
    static let initialId: Int32 = 0
    static var nextId: Int32 = initialId

    private class func getNextId() -> Int32 {
        let id = nextId
        nextId = nextId + 1
        return id
    }

    /// Creates a new handle of GPU device.
    /// - Parameters:
    ///   - descriptor: The descriptor of the model.
    ///   - contextId: The id of the ComputeContext object.
    class func createInstance(descriptor: SWModelDesc,
                              contextId: Int32) -> Int32 {

        let device = DefaultDevice.device
        let context = MetalComputeContext.getInstance(id: contextId)

        let model = Model(device: device,
                          graph: MPSGraph(),
                          descriptor: descriptor,
                          nnXLen: context.nnXLen,
                          nnYLen: context.nnYLen)

        let id = getNextId()
        let handle = MetalComputeHandle(model: model, id: id)

        handles[id] = handle

        print("Metal backend \(id): \(device.name), Model version \(descriptor.version) \(descriptor.name)",
              to: &StandardError.instance)

        return id
    }

    /// Destroy the handle.
    class func destroyInstance(id: Int32) {
        handles[id] = nil
    }

    /// Get the handle.
    /// - Returns: The handle.
    class func getInstance(id: Int32) -> MetalComputeHandle {
        return handles[id] ?? defaultHandle
    }

    let model: Model
    let id: Int32

    private init(model: Model, id: Int32) {
        self.model = model
        self.id = id
    }
}

public func createMetalComputeHandle(descriptor: SWModelDesc,
                                     contextId: Int32) -> Int32 {

    return MetalComputeHandle.createInstance(descriptor: descriptor,
                                             contextId: contextId)
}

public func destroyMetalComputeHandle(handleId id: Int32) {
    MetalComputeHandle.destroyInstance(id: id)
}

public func printMetalDevices() {
    let device = DefaultDevice.device

    print("Found Metal Device: \(device.name)",
          to: &StandardError.instance)
}

///
/// Retrieves and processes output data using the Metal backend.
///
/// This function interfaces with the Metal framework to process and obtain
/// output data based on the provided input buffers. It is designed to manage
/// various pieces of data relevant to a specific batch operation and populate
/// multiple output buffers. The function utilizes a backend method for the
/// actual processing.
///
/// - Parameters:
///   - handleId: A compute handle ID
///   - userInputBuffer: An UnsafeMutablePointer to a Float32 array representing
///     the user input buffer. This buffer contains the main input data required
///     for processing.
///   - userInputGlobalBuffer: An UnsafeMutablePointer to a Float32 array that
///     holds global input data shared across the batch operation.
///   - userInputMetaBuffer: An UnsafeMutablePointer to a Float32 array containing
///     metadata associated with the user input.
///   - policyOutput: An UnsafeMutablePointer to a Float32 array where the policy
///     output will be stored. This output is generally used in scenarios
///     involving machine learning models to represent predictive policies.
///   - policyPassOutput: An UnsafeMutablePointer to a Float32 array to store the
///     policy pass output.
///   - valueOutput: An UnsafeMutablePointer to a Float32 array for storing
///     computed value outputs.
///   - ownershipOutput: An UnsafeMutablePointer to a Float32 array to hold the
///     output representing ownership values.
///   - scoreValueOutput: An UnsafeMutablePointer to a Float32 array for storing
///     score values.
///   - batchSize: An Int specifying the size of the batch to be processed. This
///     indicates how many sets of input and corresponding outputs are being handled.
///
public func getMetalHandleOutput(handleId: Int32,
                                 userInputBuffer: UnsafeMutablePointer<Float32>,
                                 userInputGlobalBuffer: UnsafeMutablePointer<Float32>,
                                 userInputMetaBuffer: UnsafeMutablePointer<Float32>,
                                 policyOutput: UnsafeMutablePointer<Float32>,
                                 policyPassOutput: UnsafeMutablePointer<Float32>,
                                 valueOutput: UnsafeMutablePointer<Float32>,
                                 ownershipOutput: UnsafeMutablePointer<Float32>,
                                 scoreValueOutput: UnsafeMutablePointer<Float32>,
                                 batchSize: Int) {

    autoreleasepool {
        let handle = MetalComputeHandle.getInstance(id: handleId)

        handle.model.apply(input: userInputBuffer,
                           inputGlobal: userInputGlobalBuffer,
                           inputMeta: userInputMetaBuffer,
                           policy: policyOutput,
                           policyPass: policyPassOutput,
                           value: valueOutput,
                           scoreValue: scoreValueOutput,
                           ownership: ownershipOutput,
                           batchSize: batchSize)
    }
}

public func getMetalContextXLen(id: Int32) -> Int32 {
    return Int32(MetalComputeContext.getInstance(id: id).nnXLen.intValue)
}

public func getMetalContextYLen(id: Int32) -> Int32 {
    return Int32(MetalComputeContext.getInstance(id: id).nnYLen.intValue)
}
