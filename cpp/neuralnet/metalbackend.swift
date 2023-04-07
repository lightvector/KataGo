import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

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

/// A struct to handle writing data to an MPSNDArray.
struct MPSNDArrayDataWriter {
    /// The target MPSNDArray instance.
    private let mpsNDArray: MPSNDArray
    /// A closure that writes data to the MPSNDArray instance.
    private let dataWriter: (UnsafeMutablePointer<Float32>) -> Void

    /// Initializes an MPSNDArrayDataWriter with the given MPSNDArray.
    /// - Parameters:
    ///   - mpsNDArray: The target MPSNDArray instance.
    init(mpsNDArray: MPSNDArray) {
        self.mpsNDArray = mpsNDArray

        dataWriter = { pointerFP32 in
            mpsNDArray.writeBytes(pointerFP32)
        }
    }

    /// Writes data to the associated MPSNDArray instance using the dataWriter closure.
    /// - Parameter pointerFP32: A pointer to the memory buffer containing the data in FP32 format.
    func writeData(pointerFP32: UnsafeMutablePointer<Float32>) {
        dataWriter(pointerFP32)
    }
}

/// A struct to handle reading data from an MPSNDArray.
struct MPSNDArrayDataReader {
    /// A closure that reads data from the MPSNDArray instance.
    private let dataReader: (UnsafeMutablePointer<Float32>, MPSNDArray?) -> Void

    /// Initializes an MPSNDArrayDataReader
    init() {
        dataReader = { pointerFP32, mpsNDArray in
            // Reads bytes from a MPSNDArray to the Float32 buffer
            mpsNDArray?.readBytes(pointerFP32, strideBytes: nil)
        }
    }

    /// Reads data from the given MPSNDArray instance using the dataReader closure.
    /// - Parameter pointerFP32: A pointer to the memory buffer containing the data in FP32 format.
    /// - Parameter mpsNDArray: The given MPSNDArray instance
    func readData(pointerFP32: UnsafeMutablePointer<Float32>, mpsNDArray: MPSNDArray?) {
        dataReader(pointerFP32, mpsNDArray)
    }
}

/// Extension to MPSGraphTensor to count number of elements
extension MPSGraphTensor {
    /// Count number of elements
    /// - Returns: Number of elements
    func countElements() -> Int? {
        return shape?.reduce(1, { $0 * $1.intValue })
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
    ///   - batchSize: Batch size
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    ///   - numChannels: Number of channels
    init(graph: MPSGraph,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         numChannels: NSNumber) {
        shape = InputShape.create(batchSize: batchSize,
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
    ///   - batchSize: The batch size.
    ///   - numGlobalFeatures: The number of global features.
    init(graph: MPSGraph,
         batchSize: NSNumber,
         numGlobalFeatures: NSNumber) {
        shape = InputShape.create(batchSize: batchSize,
                                  numChannels: numGlobalFeatures,
                                  nnYLen: 1,
                                  nnXLen: 1)

        self.tensor = graph.placeholder(shape: shape,
                                        dataType: MPSDataType.float32,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents a mask layer for a neural network model.
struct MaskLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    /// Initializes a MaskLayer object with a graph, batch size, x and y lengths, data type, and input shape.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - batchSize: The batch size.
    ///   - nnXLen: The length of the x-axis.
    ///   - nnYLen: The length of the y-axis.
    init(graph: MPSGraph,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        shape = InputShape.create(batchSize: batchSize,
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

        // Create a Metal device and an MPS graph.
        let device = MetalBackend.defaultDevice
        let graph = MPSGraph()

        // Create the input and mask layers.
        let inputLayer = InputLayer(graph: graph,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    numChannels: numChannels)

        let maskLayer = MaskLayer(graph: graph,
                                  batchSize: batchSize,
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
        let sourceArrayWriter = MPSNDArrayDataWriter(mpsNDArray: sourceArray)
        sourceArrayWriter.writeData(pointerFP32: input)
        let maskArrayWriter = MPSNDArrayDataWriter(mpsNDArray: maskArray)
        maskArrayWriter.writeData(pointerFP32: mask)

        // Create MPSGraphTensorData objects from the source and mask arrays.
        let sourceTensorData = MPSGraphTensorData(sourceArray)
        let maskTensorData = MPSGraphTensorData(maskArray)

        // Execute the graph and fetch the result.
        let fetch = graph.run(feeds: [inputLayer.tensor: sourceTensorData,
                                      maskLayer.tensor: maskTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        // Read the output data from the result tensor, converting from FP16 to FP32 if necessary.
        let outputArrayReader = MPSNDArrayDataReader()

        outputArrayReader.readData(pointerFP32: output,
                                   mpsNDArray: fetch[resultTensor]?.mpsndarray())
    }
}

/// A class that represents a description of convolutional layer.
@objc class SWConvLayerDesc: NSObject {
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
    @objc init(convYSize: NSNumber,
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

/// A class that represents a convolutional layer using MPSGraph
@objc class ConvLayer: NSObject {
    /// The result tensor of the convolutional operation
    let resultTensor: MPSGraphTensor

    /// Class method that tests the convolutional layer by running a forward pass
    /// - Parameters:
    ///   - descriptor: A descriptor for the convolutional layer
    ///   - nnXLen: The width of the input tensor
    ///   - nnYLen: The height of the input tensor
    ///   - batchSize: The batch size of the input tensor
    ///   - input: A pointer to the input tensor data
    ///   - output: A pointer to the output tensor data
    @objc class func test(descriptor: SWConvLayerDesc,
                          nnXLen: NSNumber,
                          nnYLen: NSNumber,
                          batchSize: NSNumber,
                          input: UnsafeMutablePointer<Float32>,
                          output: UnsafeMutablePointer<Float32>) {
        let device = MetalBackend.defaultDevice
        let graph = MPSGraph()

        let source = InputLayer(graph: graph,
                                batchSize: batchSize,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                numChannels: descriptor.inChannels)

        let conv = ConvLayer(graph: graph,
                             sourceTensor: source.tensor,
                             descriptor: descriptor,
                             batchSize: batchSize,
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

        let sourceArrayDataWriter = MPSNDArrayDataWriter(mpsNDArray: sourceArray)
        sourceArrayDataWriter.writeData(pointerFP32: input)

        let sourceTensorData = MPSGraphTensorData(sourceArray)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData],
                              targetTensors: [conv.resultTensor],
                              targetOperations: nil)

        let outputArrayReader = MPSNDArrayDataReader()

        outputArrayReader.readData(pointerFP32: output,
                                   mpsNDArray: fetch[conv.resultTensor]?.mpsndarray())
    }

    /// Initializes a ConvLayer object
    /// - Parameters:
    ///   - graph: An MPSGraph object
    ///   - sourceTensor: The input tensor for the convolutional layer
    ///   - descriptor: A descriptor for the convolutional layer
    ///   - batchSize: The batch size of the input tensor
    ///   - nnXLen: The width of the input tensor
    ///   - nnYLen: The height of the input tensor
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         descriptor: SWConvLayerDesc,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber) {
        let dataLayout: MPSGraphTensorNamedDataLayout = .NCHW

        let weightsShape = [descriptor.outChannels,
                            descriptor.inChannels,
                            descriptor.convYSize,
                            descriptor.convXSize]

        let convDescriptor =
        MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                          strideInY: 1,
                                          dilationRateInX: 1,
                                          dilationRateInY: 1,
                                          groups: 1,
                                          paddingStyle: .TF_SAME,
                                          dataLayout: dataLayout,
                                          weightsLayout: .OIHW)!

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

/// A class that represents a description of a batch normalization layer.
@objc class SWBatchNormLayerDesc: NSObject {
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
    @objc init(numChannels: NSNumber,
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

/// A class that represents a batch normalization layer.
@objc class BatchNormLayer: NSObject {
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
    @objc class func test(descriptor: SWBatchNormLayerDesc,
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
                                           nnYLen: nnYLen,
                                           batchSize: batchSize)

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
    ///   - batchSize: The number of inputs in the batch.
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         descriptor: SWBatchNormLayerDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {
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

/// An enumeration of the different kinds of activation function.
@objc enum ActivationKind: Int {
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
@objc class SWResidualBlockDesc: NSObject {
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
    @objc init(preBN: SWBatchNormLayerDesc,
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

/// A class that represents a Residual Block layer
@objc class ResidualBlock: NSObject {
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
    @objc class func test(descriptor: SWResidualBlockDesc,
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
                                      nnYLen: nnYLen,
                                      batchSize: batchSize)

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
    ///   - batchSize: Batch size
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         descriptor: SWResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {
        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize)

        let preActivation = ActivationLayer(graph: graph,
                                            sourceTensor: preBN.resultTensor,
                                            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(graph: graph,
                                    sourceTensor: preActivation.resultTensor,
                                    descriptor: descriptor.regularConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen)

        let midBN = BatchNormLayer(graph: graph,
                                   sourceTensor: regularConv.resultTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize)

        let midActivation = ActivationLayer(graph: graph,
                                            sourceTensor: midBN.resultTensor,
                                            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(graph: graph,
                                  sourceTensor: midActivation.resultTensor,
                                  descriptor: descriptor.finalConv,
                                  batchSize: batchSize,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        resultTensor = graph.addition(sourceTensor,
                                      finalConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
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

/// A class that represents a matrix multiplication layer descriptor
@objc class SWMatMulLayerDesc: NSObject {
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
    @objc init(inChannels: NSNumber,
               outChannels: NSNumber,
               weights: UnsafeMutablePointer<Float32>) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
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
@objc class SWMatBiasLayerDesc: NSObject {
    /// The number of channels.
    let numChannels: NSNumber
    /// The pointer to the weights.
    let weights: UnsafeMutablePointer<Float32>

    /// Initialize an instance of SWMatBiasLayerDesc.
    /// - Parameters:
    ///   - numChannels: The number of channels.
    ///   - weights: The pointer to the weights.
    @objc init(numChannels: NSNumber,
               weights: UnsafeMutablePointer<Float32>) {
        self.numChannels = numChannels
        self.weights = weights
    }
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
    ///   - batchSize: The batch size.
    ///   - nnXLen: The x length.
    ///   - nnYLen: The y length.
    ///   - numChannels: The number of channels.
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         biasTensor: MPSGraphTensor,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         numChannels: NSNumber) {
        let shape = InputShape.create(batchSize: batchSize,
                                      numChannels: numChannels,
                                      nnYLen: 1,
                                      nnXLen: 1)

        assert(biasTensor.countElements() == shape.countElements())
        let reshaped = graph.reshape(biasTensor, shape: shape, name: nil)
        resultTensor = graph.addition(sourceTensor, reshaped, name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == nnYLen)
        assert(resultTensor.shape?[3] == nnXLen)
    }
}

/// A class that represents a residual block with global pooling.
@objc class SWGlobalPoolingResidualBlockDesc: NSObject {
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
    @objc init(preBN: SWBatchNormLayerDesc,
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

/// A class representing a residual block with global pooling
@objc class GlobalPoolingResidualBlock: NSObject {
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
    @objc class func test(descriptor: SWGlobalPoolingResidualBlockDesc,
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
                                       nnYLen: nnYLen,
                                       batchSize: batchSize)

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
    ///   - batchSize: The batch size
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         descriptor: SWGlobalPoolingResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {
        let maskSum = MaskSumLayer(tensor: maskSumTensor)
        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(tensor: maskSumSqrtS14M01Tensor)

        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize)

        let preActivation = ActivationLayer(graph: graph,
                                            sourceTensor: preBN.resultTensor,
                                            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(graph: graph,
                                    sourceTensor: preActivation.resultTensor,
                                    descriptor: descriptor.regularConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen)

        let gpoolConv = ConvLayer(graph: graph,
                                  sourceTensor: preActivation.resultTensor,
                                  descriptor: descriptor.gpoolConv,
                                  batchSize: batchSize,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        let gpoolBN = BatchNormLayer(graph: graph,
                                     sourceTensor: gpoolConv.resultTensor,
                                     maskTensor: maskTensor,
                                     descriptor: descriptor.gpoolBN,
                                     nnXLen: nnXLen,
                                     nnYLen: nnYLen,
                                     batchSize: batchSize)

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
                                   batchSize: batchSize,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.gpoolToBiasMul.outChannels)

        let midBN = BatchNormLayer(graph: graph,
                                   sourceTensor: added.resultTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize)

        let midActivation = ActivationLayer(graph: graph,
                                            sourceTensor: midBN.resultTensor,
                                            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(graph: graph,
                                  sourceTensor: midActivation.resultTensor,
                                  descriptor: descriptor.finalConv,
                                  batchSize: batchSize,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen)

        resultTensor = graph.addition(sourceTensor,
                                      finalConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that represents a nested bottleneck residual block
@objc class SWNestedBottleneckResidualBlockDesc: NSObject {
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
    @objc init(preBN: SWBatchNormLayerDesc,
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

/// An enumeration of the different kinds of blocks that can be used in a residual network.
@objc enum BlockKind: Int {
    case ordinary
    case globalPooling
    case nestedBottleneck
}

/// A class that represents a block descriptor that is used to define the characteristics of a residual block.
@objc class BlockDescriptor: NSObject {
    /// The kind of the block, it can be ordinary, globalPooling, or nestedBottleneck.
    let kind: BlockKind

    /// The descriptor for the ordinary residual block, if the kind is ordinary.
    let ordinary: SWResidualBlockDesc?

    /// The descriptor for the global pooling residual block, if the kind is globalPooling.
    let globalPooling: SWGlobalPoolingResidualBlockDesc?

    /// The descriptor for the nested bottleneck residual block, if the kind is nestedBottleneck.
    let nestedBottleneck: SWNestedBottleneckResidualBlockDesc?

    /// Initializes a block descriptor object with the given parameters.
    /// - Parameters:
    ///   - ordinary: The descriptor for the ordinary residual block, if the kind is ordinary.
    @objc init(ordinary: SWResidualBlockDesc) {
        self.kind = BlockKind.ordinary
        self.ordinary = ordinary
        self.globalPooling = nil
        self.nestedBottleneck = nil
    }

    /// Initializes a block descriptor object with the given parameters.
    /// - Parameters:
    ///   - globalPooling: The descriptor for the global pooling residual block, if the kind is globalPooling.
    @objc init(globalPooling: SWGlobalPoolingResidualBlockDesc) {
        self.kind = BlockKind.globalPooling
        self.ordinary = nil
        self.globalPooling = globalPooling
        self.nestedBottleneck = nil
    }

    /// Initializes a block descriptor object with the given parameters.
    /// - Parameters:
    ///   - nestedBottleneck: The descriptor for the nested bottleneck residual block, if the kind is nestedBottleneck.
    @objc init(nestedBottleneck: SWNestedBottleneckResidualBlockDesc) {
        self.kind = BlockKind.nestedBottleneck
        self.ordinary = nil
        self.globalPooling = nil
        self.nestedBottleneck = nestedBottleneck
    }
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
    ///   - batchSize: Batch size
    /// - Returns: The result tensor
    static func processBlockDescriptors(_ graph: MPSGraph,
                                        _ sourceTensor: MPSGraphTensor,
                                        _ maskTensor: MPSGraphTensor,
                                        _ maskSumTensor: MPSGraphTensor,
                                        _ maskSumSqrtS14M01Tensor: MPSGraphTensor,
                                        _ blockDescriptors: [BlockDescriptor],
                                        _ index: Int,
                                        _ nnXLen: NSNumber,
                                        _ nnYLen: NSNumber,
                                        _ batchSize: NSNumber) -> MPSGraphTensor {
        guard index < blockDescriptors.count else {
            return sourceTensor
        }

        let blockDescriptor = blockDescriptors[index]
        let blockInput: MPSGraphTensor

        switch blockDescriptor.kind {
        case .globalPooling:
            let globalPooling = GlobalPoolingResidualBlock(graph: graph,
                                                           sourceTensor: sourceTensor,
                                                           maskTensor: maskTensor,
                                                           maskSumTensor: maskSumTensor,
                                                           maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                                           descriptor: blockDescriptor.globalPooling!,
                                                           nnXLen: nnXLen,
                                                           nnYLen: nnYLen,
                                                           batchSize: batchSize)

            blockInput = globalPooling.resultTensor
        case .nestedBottleneck:
            let nestedBottleneck = NestedBottleneckResidualBlock(graph: graph,
                                                                 sourceTensor: sourceTensor,
                                                                 maskTensor: maskTensor,
                                                                 maskSumTensor: maskSumTensor,
                                                                 maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                                                 descriptor: blockDescriptor.nestedBottleneck!,
                                                                 nnXLen: nnXLen,
                                                                 nnYLen: nnYLen,
                                                                 batchSize: batchSize)

            blockInput = nestedBottleneck.resultTensor
        case .ordinary:
            let ordinary = ResidualBlock(graph: graph,
                                         sourceTensor: sourceTensor,
                                         maskTensor: maskTensor,
                                         descriptor: blockDescriptor.ordinary!,
                                         nnXLen: nnXLen,
                                         nnYLen: nnYLen,
                                         batchSize: batchSize)

            blockInput = ordinary.resultTensor
        }

        return processBlockDescriptors(graph,
                                       blockInput,
                                       maskTensor,
                                       maskSumTensor,
                                       maskSumSqrtS14M01Tensor,
                                       blockDescriptors,
                                       index + 1,
                                       nnXLen,
                                       nnYLen,
                                       batchSize)
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
    ///   - batchSize: Batch size
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         blockDescriptors: [BlockDescriptor],
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {
        resultTensor = BlockStack.processBlockDescriptors(graph,
                                                          sourceTensor,
                                                          maskTensor,
                                                          maskSumTensor,
                                                          maskSumSqrtS14M01Tensor,
                                                          blockDescriptors,
                                                          0,
                                                          nnXLen,
                                                          nnYLen,
                                                          batchSize)
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
    ///   - batchSize: Batch size
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         descriptor: SWNestedBottleneckResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {

        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize)

        let preActivation = ActivationLayer(graph: graph,
                                            sourceTensor: preBN.resultTensor,
                                            activationKind: descriptor.preActivation)

        let preConv = ConvLayer(graph: graph,
                                sourceTensor: preActivation.resultTensor,
                                descriptor: descriptor.preConv,
                                batchSize: batchSize,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen)

        let blocks = BlockStack(graph: graph,
                                sourceTensor: preConv.resultTensor,
                                maskTensor: maskTensor,
                                maskSumTensor: maskSumTensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                blockDescriptors: descriptor.blockDescriptors,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                batchSize: batchSize)

        let postBN = BatchNormLayer(graph: graph,
                                    sourceTensor: blocks.resultTensor,
                                    maskTensor: maskTensor,
                                    descriptor: descriptor.postBN,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    batchSize: batchSize)

        let postActivation = ActivationLayer(graph: graph,
                                             sourceTensor: postBN.resultTensor,
                                             activationKind: descriptor.postActivation)

        let postConv = ConvLayer(graph: graph,
                                 sourceTensor: postActivation.resultTensor,
                                 descriptor: descriptor.postConv,
                                 batchSize: batchSize,
                                 nnXLen: nnXLen,
                                 nnYLen: nnYLen)

        resultTensor = graph.addition(sourceTensor,
                                      postConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that describes a trunk for a neural network
@objc class SWTrunkDesc: NSObject {
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
    ///   - blockDescriptors: The list of blocks that make up the trunk
    ///   - trunkTipBN: The description of the batch normalization layer that is applied at the end of the trunk
    ///   - trunkTipActivation: The activation function that is applied at the end of the trunk
    @objc init(version: Int,
               trunkNumChannels: NSNumber,
               midNumChannels: NSNumber,
               regularNumChannels: NSNumber,
               gpoolNumChannels: NSNumber,
               initialConv: SWConvLayerDesc,
               initialMatMul: SWMatMulLayerDesc,
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
        self.blockDescriptors = blockDescriptors
        self.trunkTipBN = trunkTipBN
        self.trunkTipActivation = trunkTipActivation
    }
}

/// A structure representing a ResNet trunk for a neural network
struct Trunk {
    /// The resulting tensor after processing the trunk
    let resultTensor: MPSGraphTensor

    /// Initializes a Trunk object
    /// - Parameters:
    ///   - graph: The graph used to build the trunk
    ///   - descriptor: A SWTrunkDesc object that describes the trunk
    ///   - inputTensor: The input tensor
    ///   - inputGlobalTensor: The input global tensor
    ///   - maskTensor: The tensor used to mask input activations
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - nnXLen: The length of the X dimension of the input tensor
    ///   - nnYLen: The length of the Y dimension of the input tensor
    ///   - batchSize: The batch size of the input tensor
    ///   - numSpatialFeatures: The number of spatial features in the input tensor
    ///   - numGlobalFeatures: The number of global features in the input tensor
    init(graph: MPSGraph,
         descriptor: SWTrunkDesc,
         inputTensor: MPSGraphTensor,
         inputGlobalTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         numSpatialFeatures: NSNumber,
         numGlobalFeatures: NSNumber) {

        let initialConv = ConvLayer(graph: graph,
                                    sourceTensor: inputTensor,
                                    descriptor: descriptor.initialConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen)

        let initialMatMul = MatMulLayer(graph: graph,
                                        descriptor: descriptor.initialMatMul,
                                        sourceTensor: inputGlobalTensor)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: initialConv.resultTensor,
                                   biasTensor: initialMatMul.resultTensor,
                                   batchSize: batchSize,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.initialMatMul.outChannels)

        let blocks = BlockStack(graph: graph,
                                sourceTensor: added.resultTensor,
                                maskTensor: maskTensor,
                                maskSumTensor: maskSumTensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                blockDescriptors: descriptor.blockDescriptors,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                batchSize: batchSize)

        let trunkTipBN = BatchNormLayer(graph: graph,
                                        sourceTensor: blocks.resultTensor,
                                        maskTensor: maskTensor,
                                        descriptor: descriptor.trunkTipBN,
                                        nnXLen: nnXLen,
                                        nnYLen: nnYLen,
                                        batchSize: batchSize)

        let trunkTipActivation = ActivationLayer(graph: graph,
                                                 sourceTensor: trunkTipBN.resultTensor,
                                                 activationKind: descriptor.trunkTipActivation)

        resultTensor = trunkTipActivation.resultTensor

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that describes a policy head for a neural network, responsible for predicting
/// the best moves for the current player and the opposing player on the subsequent turn.
@objc class SWPolicyHeadDesc: NSObject {
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
    @objc init(version: Int,
               p1Conv: SWConvLayerDesc,
               g1Conv: SWConvLayerDesc,
               g1BN: SWBatchNormLayerDesc,
               g1Activation: ActivationKind,
               gpoolToBiasMul: SWMatMulLayerDesc,
               p1BN: SWBatchNormLayerDesc,
               p1Activation: ActivationKind,
               p2Conv: SWConvLayerDesc,
               gpoolToPassMul: SWMatMulLayerDesc) {
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
    ///   - batchSize: The batch size of the input tensor
    init(graph: MPSGraph,
         descriptor: SWPolicyHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {

        let p1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.p1Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        let g1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.g1Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        let g1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: g1Conv.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.g1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize)

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
                                   batchSize: batchSize,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.gpoolToBiasMul.outChannels)

        let p1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: added.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.p1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize)

        let p1Activation = ActivationLayer(graph: graph,
                                           sourceTensor: p1BN.resultTensor,
                                           activationKind: descriptor.p1Activation)

        let p2Conv = ConvLayer(graph: graph,
                               sourceTensor: p1Activation.resultTensor,
                               descriptor: descriptor.p2Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        assert(g1Concat.resultTensor.shape?[1] == descriptor.gpoolToPassMul.inChannels)

        let gpoolToPassMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToPassMul,
                                         sourceTensor: g1Concat.resultTensor)

        policyTensor = p2Conv.resultTensor
        policyPassTensor = gpoolToPassMul.resultTensor

        assert(policyTensor.shape?.count == 4)
        assert(policyPassTensor.shape?.count == 2)
    }
}

/// A class that describes the value head of a neural network
@objc class SWValueHeadDesc: NSObject {
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
    @objc init(version: Int,
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
    ///   - batchSize: The size of the batch
    init(graph: MPSGraph,
         descriptor: SWValueHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {

        let v1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.v1Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen)

        let v1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: v1Conv.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.v1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize)

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
                                       batchSize: batchSize,
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


/// A class that describes a neural network model used for playing the game of Go.
@objc class SWModelDesc : NSObject {
    /// The version of the model.
    let version: Int
    /// The name of the model.
    let name: String
    /// Number of channels for input features.
    let numInputChannels: NSNumber
    /// Number of channels for global input features.
    let numInputGlobalChannels: NSNumber
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
    ///   - numValueChannels: Number of channels for the value head output.
    ///   - numScoreValueChannels: Number of channels for the score value head output.
    ///   - numOwnershipChannels: Number of channels for the ownership head output.
    ///   - trunk: The description of the trunk that makes up the backbone of the model.
    ///   - policyHead: The description of the policy head that predicts the probability of playing at a particular position.
    ///   - valueHead: The description of the value head that predicts the expected outcome of a game state.
    @objc init(version: Int,
               name: String,
               numInputChannels: NSNumber,
               numInputGlobalChannels: NSNumber,
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
        self.numValueChannels = numValueChannels
        self.numScoreValueChannels = numScoreValueChannels
        self.numOwnershipChannels = numOwnershipChannels
        self.trunk = trunk
        self.policyHead = policyHead
        self.valueHead = valueHead
    }
}

/// A structure representing a neural network model for processing Go game states.
struct Model {
    /// The Metal Performance Shaders graph object used for building and executing the graph
    let graph: MPSGraph
    /// The length of the neural network input in the x dimension
    let nnXLen: NSNumber
    /// The length of the neural network input in the y dimension
    let nnYLen: NSNumber
    /// The batch size of the neural network input
    let batchSize: NSNumber
    /// The version of the model
    let version: Int
    /// The number of channels in the input layer
    let numInputChannels: NSNumber
    /// The number of channels in the global input layer
    let numInputGlobalChannels: NSNumber
    /// The number of channels in the value output layer
    let numValueChannels: NSNumber
    /// The number of channels in the score value output layer
    let numScoreValueChannels: NSNumber
    /// The number of channels in the ownership output layer
    let numOwnershipChannels: NSNumber
    /// The command queue used to execute the graph on the GPU
    let commandQueue: MTLCommandQueue
    /// The input layer of the neural network
    let input: InputLayer
    /// The global input layer of the neural network
    let inputGlobal: InputGlobalLayer
    /// The trunk of the neural network
    let trunk: Trunk
    /// The policy head of the neural network
    let policyHead: PolicyHead
    /// The value head of the neural network
    let valueHead: ValueHead
    /// The input layer as a Metal Performance Shaders n-dimensional array
    let inputArray: MPSNDArray
    /// The data writer for the input array
    let inputArrayWriter: MPSNDArrayDataWriter
    /// The global input layer as a Metal Performance Shaders n-dimensional array
    let inputGlobalArray: MPSNDArray
    /// The data writer for the global input array
    let inputGlobalArrayWriter: MPSNDArrayDataWriter
    /// The data reader for the policy array
    let policyArrayReader: MPSNDArrayDataReader
    /// The data reader for the policy pass array
    let policyPassArrayReader: MPSNDArrayDataReader
    /// The data reader for the value array
    let valueArrayReader: MPSNDArrayDataReader
    /// The data reader for the score value array
    let scoreValueArrayReader: MPSNDArrayDataReader
    /// The data reader for the ownership array
    let ownershipArrayReader: MPSNDArrayDataReader
    /// The dictionary that maps the input tensors to the tensor data
    let feeds: [MPSGraphTensor: MPSGraphTensorData]
    /// The dictionary that maps the output tensors to the tensor data
    let targetTensors: [MPSGraphTensor]

    /// Initializes a Model object.
    /// - Parameters:
    ///   - device: The Metal device to use for computations.
    ///   - graph: The Metal Performance Shaders graph object used for building and executing the graph.
    ///   - descriptor: The description of the model.
    ///   - nnXLen: The length of the neural network input in the x dimension.
    ///   - nnYLen: The length of the neural network input in the y dimension.
    ///   - batchSize: The batch size of the neural network input.
    init(device: MPSGraphDevice,
         graph: MPSGraph,
         descriptor: SWModelDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber) {
        self.graph = graph
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.batchSize = batchSize
        self.version = descriptor.version
        self.numInputChannels = descriptor.numInputChannels
        self.numInputGlobalChannels = descriptor.numInputGlobalChannels
        self.numValueChannels = descriptor.numValueChannels
        self.numScoreValueChannels = descriptor.numScoreValueChannels
        self.numOwnershipChannels = descriptor.numOwnershipChannels
        commandQueue = (device.metalDevice?.makeCommandQueue())!

        input = InputLayer(graph: graph,
                           batchSize: batchSize,
                           nnXLen: nnXLen,
                           nnYLen: nnYLen,
                           numChannels: descriptor.numInputChannels)

        inputGlobal = InputGlobalLayer(graph: graph,
                                       batchSize: batchSize,
                                       numGlobalFeatures: descriptor.numInputGlobalChannels)

        let startOfMask: [NSNumber] = [0, 0, 0, 0]

        let endOfMask = InputShape.create(batchSize: batchSize,
                                          numChannels: 1,
                                          nnYLen: nnYLen,
                                          nnXLen: nnXLen)

        let maskTensor = graph.sliceTensor(input.tensor,
                                           starts: startOfMask,
                                           ends: endOfMask,
                                           strides: [1, 1, 1, 1],
                                           name: nil)

        let maskSum = MaskSumLayer(graph: graph,
                                   maskTensor: maskTensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum)

        let maskSumSqrtS14M01SquareS01 = MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                                                         maskSumSqrtS14M01: maskSumSqrtS14M01)

        trunk = Trunk(graph: graph,
                      descriptor: descriptor.trunk,
                      inputTensor: input.tensor,
                      inputGlobalTensor: inputGlobal.tensor,
                      maskTensor: maskTensor,
                      maskSumTensor: maskSum.tensor,
                      maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                      nnXLen: nnXLen,
                      nnYLen: nnYLen,
                      batchSize: batchSize,
                      numSpatialFeatures: descriptor.numInputChannels,
                      numGlobalFeatures: descriptor.numInputGlobalChannels)

        policyHead = PolicyHead(graph: graph,
                                descriptor: descriptor.policyHead,
                                sourceTensor: trunk.resultTensor,
                                maskTensor: maskTensor,
                                maskSumTensor: maskSum.tensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                batchSize: batchSize)

        valueHead = ValueHead(graph: graph,
                              descriptor: descriptor.valueHead,
                              sourceTensor: trunk.resultTensor,
                              maskTensor: maskTensor,
                              maskSumTensor: maskSum.tensor,
                              maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                              maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
                              nnXLen: nnXLen,
                              nnYLen: nnYLen,
                              batchSize: batchSize)

        let metalDevice = device.metalDevice!

        let inputShape = InputShape.create(batchSize: batchSize,
                                           numChannels: descriptor.numInputChannels,
                                           nnYLen: nnYLen,
                                           nnXLen: nnXLen)

        let inputDescriptor = MPSNDArrayDescriptor(dataType: input.tensor.dataType,
                                                   shape: inputShape)

        inputArray = MPSNDArray(device: metalDevice,
                                descriptor: inputDescriptor)

        inputArrayWriter = MPSNDArrayDataWriter(mpsNDArray: inputArray)

        let inputGlobalShape = InputShape.create(batchSize: batchSize,
                                                 numChannels: descriptor.numInputGlobalChannels,
                                                 nnYLen: 1,
                                                 nnXLen: 1)

        let inputGlobalDescriptor = MPSNDArrayDescriptor(dataType: inputGlobal.tensor.dataType,
                                                         shape: inputGlobalShape)

        inputGlobalArray = MPSNDArray(device: metalDevice,
                                      descriptor: inputGlobalDescriptor)

        inputGlobalArrayWriter = MPSNDArrayDataWriter(mpsNDArray: inputGlobalArray)

        policyArrayReader = MPSNDArrayDataReader()
        policyPassArrayReader = MPSNDArrayDataReader()
        valueArrayReader = MPSNDArrayDataReader()
        scoreValueArrayReader = MPSNDArrayDataReader()
        ownershipArrayReader = MPSNDArrayDataReader()

        feeds = [input.tensor: MPSGraphTensorData(inputArray),
                 inputGlobal.tensor: MPSGraphTensorData(inputGlobalArray)]

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
    ///   - policy: UnsafeMutablePointer to a flattened 2D array of floats representing predicted policy
    ///   - policyPass: UnsafeMutablePointer to a flattened array of floats representing predicted probability of passing
    ///   - value: UnsafeMutablePointer to a flattened array of floats representing predicted value
    ///   - scoreValue: UnsafeMutablePointer to a flattened array of floats representing predicted score value
    ///   - ownership: UnsafeMutablePointer to a flattened 2D array of floats representing predicted ownership
    func apply(input inputPointer: UnsafeMutablePointer<Float32>,
               inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
               policy: UnsafeMutablePointer<Float32>,
               policyPass: UnsafeMutablePointer<Float32>,
               value: UnsafeMutablePointer<Float32>,
               scoreValue: UnsafeMutablePointer<Float32>,
               ownership: UnsafeMutablePointer<Float32>,
               batchSize: Int) {

        inputArrayWriter.writeData(pointerFP32: inputPointer)
        inputGlobalArrayWriter.writeData(pointerFP32: inputGlobalPointer)

        let commandBuffer = MPSCommandBuffer(commandBuffer: commandQueue.makeCommandBuffer()!)

        let fetch = graph.encode(to: commandBuffer,
                                 feeds: feeds,
                                 targetTensors: targetTensors,
                                 targetOperations: nil,
                                 executionDescriptor: nil)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        policyArrayReader.readData(pointerFP32: policy,
                                   mpsNDArray: fetch[policyHead.policyTensor]?.mpsndarray())

        policyPassArrayReader.readData(pointerFP32: policyPass,
                                       mpsNDArray: fetch[policyHead.policyPassTensor]?.mpsndarray())

        valueArrayReader.readData(pointerFP32: value,
                                  mpsNDArray: fetch[valueHead.valueTensor]?.mpsndarray())

        scoreValueArrayReader.readData(pointerFP32: scoreValue,
                                       mpsNDArray: fetch[valueHead.scoreValueTensor]?.mpsndarray())

        ownershipArrayReader.readData(pointerFP32: ownership,
                                      mpsNDArray: fetch[valueHead.ownershipTensor]?.mpsndarray())
    }
}

// A enum to represent enabled/disabled/auto option of a feature.
@objc enum SWEnable: Int {
    case False
    case True
    case Auto
}

/// A class that represents context of GPU devices.
@objc class MetalComputeContext: NSObject {
    static let defaultNnXLen: NSNumber = 19
    static let defaultNnYLen: NSNumber = 19

    static let defaultInstance = MetalComputeContext(nnXLen: defaultNnXLen,
                                                     nnYLen: defaultNnYLen)

    static var instance = defaultInstance

    let nnXLen: NSNumber
    let nnYLen: NSNumber

    /// Create a context.
    /// - Parameters:
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    ///   - useFP16Mode: use FP16 mode or not.
    ///   - useNHWCMode: use NHWC mode or not.
    @objc class func createInstance(nnXLen: NSNumber,
                                    nnYLen: NSNumber,
                                    useFP16Mode: SWEnable,
                                    useNHWCMode: SWEnable) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        instance = MetalComputeContext(nnXLen: nnXLen,
                                       nnYLen: nnYLen)
    }

    /// Destroy the context.
    @objc class func destroyInstance() {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        instance = defaultInstance
    }

    /// Get the context.
    /// - Returns: The context.
    @objc class func getInstance() -> MetalComputeContext {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        return instance
    }

    /// Initialize a context.
    /// - Parameters:
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    private init(nnXLen: NSNumber,
                 nnYLen: NSNumber) {
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
    }
}

/// A class that represents a handle of GPU device.
@objc class MetalComputeHandle: NSObject {
    static var handles: [Int: MetalComputeHandle] = [:]
    let model: Model

    /// Creates a new handle of GPU device.
    /// - Parameters:
    ///   - gpuIdxForThisThread: The index of GPU device.
    ///   - descriptor: The descriptor of the model.
    ///   - batchSize: The batch size.
    ///   - serverThreadIdx: The index of the server thread.
    @objc class func createInstance(at gpuIdxForThisThread: Int,
                                    descriptor: SWModelDesc,
                                    batchSize: NSNumber,
                                    serverThreadIdx: Int) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        handles[gpuIdxForThisThread] = MetalComputeHandle(descriptor: descriptor,
                                                          batchSize: batchSize,
                                                          gpuIdxForThisThread: gpuIdxForThisThread,
                                                          serverThreadIdx: serverThreadIdx)
    }

    /// Gets the handle of GPU device.
    /// - Parameter gpuIdxForThisThread: The index of GPU device.
    /// - Returns: The handle of GPU device.
    @objc class func getInstance(at gpuIdxForThisThread: Int) -> MetalComputeHandle {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        return handles[gpuIdxForThisThread]!
    }

    /// Initializes a new instance of the `MetalComputeHandle` class.
    /// - Parameters:
    ///   - descriptor: The descriptor of the model.
    ///   - batchSize: The batch size.
    ///   - gpuIdx: The index of GPU device.
    ///   - threadIdx: The index of the server thread.
    private init(descriptor: SWModelDesc,
                 batchSize: NSNumber,
                 gpuIdxForThisThread gpuIdx: Int,
                 serverThreadIdx threadIdx: Int) {

        let context = MetalComputeContext.getInstance()
        let devices = MTLCopyAllDevices()
        let mtlDevice: MTLDevice

        // Select a GPU device.
        if ((gpuIdx >= 0) && (gpuIdx < devices.count)) {
            mtlDevice = devices[gpuIdx]
        } else {
            mtlDevice = MetalBackend.defaultDevice
        }

        let device = MPSGraphDevice(mtlDevice: mtlDevice)

        NSLog("Metal backend thread \(threadIdx): \(mtlDevice.name) Model version \(descriptor.version)")
        NSLog("Metal backend thread \(threadIdx): \(mtlDevice.name) Model name \(descriptor.name)")

        // Create a model.
        model = Model(device: device,
                      graph: MPSGraph(),
                      descriptor: descriptor,
                      nnXLen: context.nnXLen,
                      nnYLen: context.nnYLen,
                      batchSize: batchSize)

        NSLog("Metal backend thread \(threadIdx): \(mtlDevice.name) batchSize=\(batchSize)")
    }
}

/// A class that represents Metal backend.
@objc class MetalBackend : NSObject {
    static let defaultDevice = MTLCreateSystemDefaultDevice()!

    /// Print all available devices.
    @objc class func printDevices() {
        let devices = MTLCopyAllDevices()

        (0..<devices.count).forEach {
            print("Found Metal Device \($0): \(devices[$0].name) (isLowPower:\(devices[$0].isLowPower), isRemovable:\(devices[$0].isRemovable))")
        }
    }

    /// Get width of the input tensor.
    /// - Returns: The width of the input tensor.
    @objc class func getContextXLen() -> Int {
        return MetalComputeContext.getInstance().nnXLen.intValue
    }

    /// Get height of the input tensor.
    /// - Returns: The height of the input tensor.
    @objc class func getContextYLen() -> Int {
        return MetalComputeContext.getInstance().nnYLen.intValue
    }

    /// Get output data from the model.
    /// - Parameters:
    ///   - userInputBuffer: The input data.
    ///   - userInputGlobalBuffer: The global input data.
    ///   - policyOutput: The policy output data.
    ///   - policyPassOutput: The policy pass output data.
    ///   - valueOutput: The value output data.
    ///   - ownershipOutput: The ownership output data.
    ///   - scoreValueOutput: The score value output data.
    ///   - gpuIdx: The index of the GPU to use.
    @objc class func getOutput(userInputBuffer: UnsafeMutablePointer<Float32>,
                               userInputGlobalBuffer: UnsafeMutablePointer<Float32>,
                               policyOutput: UnsafeMutablePointer<Float32>,
                               policyPassOutput: UnsafeMutablePointer<Float32>,
                               valueOutput: UnsafeMutablePointer<Float32>,
                               ownershipOutput: UnsafeMutablePointer<Float32>,
                               scoreValueOutput: UnsafeMutablePointer<Float32>,
                               gpuIdx: Int) {
        autoreleasepool {
            let handle = MetalComputeHandle.getInstance(at: gpuIdx)

            handle.model.apply(input: userInputBuffer,
                               inputGlobal: userInputGlobalBuffer,
                               policy: policyOutput,
                               policyPass: policyPassOutput,
                               value: valueOutput,
                               scoreValue: scoreValueOutput,
                               ownership: ownershipOutput,
                               batchSize: 1)
        }
    }
}
