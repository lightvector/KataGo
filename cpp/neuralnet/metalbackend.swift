import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Extension to convert float32 to float16
extension UnsafeMutablePointer<Float32> {
    /// Convert to Float16
    /// - Parameter length: The length of the array
    /// - Returns: An array of Float16
    func toFP16(length: Int) -> UnsafeMutablePointer<Float16> {
        let fp16Pointer = UnsafeMutablePointer<Float16>.allocate(capacity: length)

        for i in 0..<length {
            fp16Pointer[i] = Float16(self[i])
        }

        return fp16Pointer
    }

    /// Convert to Float16
    /// - Parameters:
    ///   - fp16Pointer: Pointer to the destination buffer
    ///   - length: Number of elements to convert
    func toFP16(_ fp16Pointer: UnsafeMutablePointer<Float16>, length: Int) {
        for i in 0..<length {
            fp16Pointer[i] = Float16(self[i])
        }
    }
}

/// Extension to UnsafeMutablePointer to convert Float16 to Float32
extension UnsafeMutablePointer<Float16> {
    /// Convert to Float32
    /// - Parameters:
    ///   - fp32Pointer: Pointer to Float32
    ///   - length: Length of the array
    func toFP32(_ fp32Pointer: UnsafeMutablePointer<Float32>, length: Int) {
        for i in 0..<length {
            fp32Pointer[i] = Float32(self[i])
        }
    }
}

/// Extension to MPSNDArray to convert from MPSGraphTensor, and to read/write bytes from/to UnsafeMutableRawPointer
extension MPSNDArray {
    /// Initialize a MPSNDArray object with the data type and the shape of the tensor
    /// - Parameters:
    ///   - device: the metal deivce that the tensor is intended for
    ///   - tensor: the tensor to use shape and data type from
    convenience init(device: MTLDevice, tensor: MPSGraphTensor) {
        // Metal backend uses a fixed batch size,
        // so every shape is determined at compile time.
        let descriptor = MPSNDArrayDescriptor(dataType: tensor.dataType,
                                              shape: tensor.shape!)

        self.init(device: device, descriptor: descriptor)
    }

    /// Write bytes to the buffer
    /// - Parameter buffer: The buffer to write
    func writeBytes(_ buffer: UnsafeMutableRawPointer) {
        self.writeBytes(buffer, strideBytes: nil)
    }

    /// Read bytes from the buffer
    /// - Parameter buffer: The buffer to read
    func readBytes(_ buffer: UnsafeMutableRawPointer) {
        self.readBytes(buffer, strideBytes: nil)
    }
}

/// Extension to MPSGraphTensor to count number of elements
extension MPSGraphTensor {
    /// Count number of elements
    /// - Returns: Number of elements
    func countElements() -> Int {
        var result = shape![0].intValue
        for i in 1..<shape!.count {
            result *= shape![i].intValue
        }
        return result
    }
}

/// Extension to MPSDataType to initialize by using a boolean value of using FP16 or not, and to convert to MemoryLayout size
extension MPSDataType {
    /// Initialize a MPSDataType object
    /// - Parameter useFP16: If true, use MPSDataType.float16, otherwise use MPSDataType.float32
    init(useFP16: Bool) {
        if useFP16 {
            self.init(rawValue: MPSDataType.float16.rawValue)!
        } else {
            self.init(rawValue: MPSDataType.float32.rawValue)!
        }
    }

    /// Convert to MemoryLayout size
    /// - Returns: MemoryLayout size
    func toMemoryLayoutSize() -> Int {
        let memoryLayoutSize: Int
        switch self {
        case .float16:
            memoryLayoutSize = MemoryLayout<Float16>.size
        default:
            precondition(self == .float32)
            memoryLayoutSize = MemoryLayout<Float32>.size
        }
        return memoryLayoutSize
    }
}

/// Extension to Array to count number of elements and bytes
extension Array where Element == NSNumber {
    /// Count number of elements
    /// - Returns: Number of elements
    func countElements() -> Int {
        var result = 1.0
        for x in self {
            result *= x.doubleValue
        }
        return Int(result)
    }

    /// Count number of bytes
    /// - Parameter dataType: The data type
    /// - Returns: Number of bytes
    func countBytes(of dataType: MPSDataType) -> Int {
        return countElements() * dataType.toMemoryLayoutSize()
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

/// A class that represents the input shape
class InputShape {
    /// Create a shape for the input tensor
    /// - Parameters:
    ///   - batchSize: Batch size
    ///   - numChannels: Number of channels
    ///   - nnYLen: Y length
    ///   - nnXLen: X length
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW
    /// - Returns: The shape
    class func create(batchSize: NSNumber,
                      numChannels: NSNumber,
                      nnYLen: NSNumber,
                      nnXLen: NSNumber,
                      useNHWC: Bool) -> [NSNumber] {
        let shape: [NSNumber]
        if useNHWC {
            shape = [batchSize,
                     nnYLen,
                     nnXLen,
                     numChannels]
        } else {
            shape = [batchSize,
                     numChannels,
                     nnYLen,
                     nnXLen]
        }
        return shape
    }

    /// Get the channel axis
    /// - Parameter useNHWC: If true, use NHWC, otherwise use NCHW
    /// - Returns: The channel axis
    class func getChannelAxis(useNHWC: Bool) -> Int {
        return useNHWC ? 3 : 1
    }

    /// Get the HW axes
    /// - Parameter useNHWC: If true, use NHWC, otherwise use NCHW
    /// - Returns: The HW axes
    class func getHWAxes(useNHWC: Bool) -> [NSNumber] {
        let hwAxes: [NSNumber]
        if useNHWC {
            hwAxes = [1, 2]
        } else {
            hwAxes = [2, 3]
        }
        return hwAxes
    }
}

/// A class that represents the input layer
class InputLayer {
    let tensor: MPSGraphTensor

    /// Initialize a InputLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - batchSize: Batch size
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    ///   - numChannels: Number of channels
    ///   - useFP16: If true, use FP16, otherwise use FP32
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW
    init(graph: MPSGraph,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         numChannels: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let shape = InputShape.create(batchSize: batchSize,
                                      numChannels: numChannels,
                                      nnYLen: nnYLen,
                                      nnXLen: nnXLen,
                                      useNHWC: useNHWC)

        let dataType = MPSDataType.init(useFP16: useFP16)

        self.tensor = graph.placeholder(shape: shape,
                                        dataType: dataType,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A class that represents an input global layer for a neural network model.
class InputGlobalLayer {
    let tensor: MPSGraphTensor

    /// Initializes an InputGlobalLayer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes an InputGlobalLayer object with a graph, batch size, number of global features, data type, and input shape.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - batchSize: The batch size.
    ///   - numGlobalFeatures: The number of global features.
    ///   - useFP16: If true, use 16-bit floating-point data type. Otherwise, use 32-bit.
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW.
    init(graph: MPSGraph,
         batchSize: NSNumber,
         numGlobalFeatures: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let shape = InputShape.create(batchSize: batchSize,
                                      numChannels: numGlobalFeatures,
                                      nnYLen: 1,
                                      nnXLen: 1,
                                      useNHWC: useNHWC)

        let dataType = MPSDataType.init(useFP16: useFP16)

        self.tensor = graph.placeholder(shape: shape,
                                        dataType: dataType,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A class that represents a mask layer for a neural network model.
class MaskLayer {
    let tensor: MPSGraphTensor

    /// Initializes a MaskLayer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes a MaskLayer object with a graph, batch size, x and y lengths, data type, and input shape.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - batchSize: The batch size.
    ///   - nnXLen: The length of the x-axis.
    ///   - nnYLen: The length of the y-axis.
    ///   - useFP16: If true, use 16-bit floating-point data type. Otherwise, use 32-bit.
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW.
    init(graph: MPSGraph,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let shape = InputShape.create(batchSize: batchSize,
                                      numChannels: 1,
                                      nnYLen: nnYLen,
                                      nnXLen: nnXLen,
                                      useNHWC: useNHWC)

        let dataType = MPSDataType.init(useFP16: useFP16)

        self.tensor = graph.placeholder(shape: shape,
                                        dataType: dataType,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
        assert(self.tensor.shape == shape)
    }
}

/// A class that represents a layer which performs the summation operation on a mask layer.
class MaskSumLayer {
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
    ///   - mask: The mask layer.
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW.
    init(graph: MPSGraph,
         mask: MaskLayer,
         useNHWC: Bool) {
        let hwAxes = InputShape.getHWAxes(useNHWC: useNHWC)

        self.tensor = graph.reductionSum(with: mask.tensor,
                                         axes: hwAxes,
                                         name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A class that represents a layer which performs square root, subtraction, and multiplication operations on a MaskSumLayer object.
class MaskSumSqrtS14M01Layer {
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
    ///   - useFP16: If true, use 16-bit floating-point data type. Otherwise, use 32-bit.
    init(graph: MPSGraph,
         maskSum: MaskSumLayer,
         useFP16: Bool) {
        let dataType = MPSDataType.init(useFP16: useFP16)
        let sqrtMaskSum = graph.squareRoot(with: maskSum.tensor, name: nil)

        let fourTeen = graph.constant(14.0,
                                      shape: sqrtMaskSum.shape!,
                                      dataType: dataType)

        let subtracted = graph.subtraction(sqrtMaskSum, fourTeen, name: nil)

        let zeroPointone = graph.constant(0.1,
                                          shape: sqrtMaskSum.shape!,
                                          dataType: dataType)

        self.tensor = graph.multiplication(subtracted,
                                           zeroPointone,
                                           name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A class that represents a layer which performs squaring and subtraction operations on a MaskSumSqrtS14M01Layer object.
class MaskSumSqrtS14M01SquareS01Layer {
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
    ///   - useFP16: If true, use 16-bit floating-point data type. Otherwise, use 32-bit.
    init(graph: MPSGraph,
         maskSumSqrtS14M01: MaskSumSqrtS14M01Layer,
         useFP16: Bool) {
        let dataType = MPSDataType.init(useFP16: useFP16)
        let squared = graph.square(with: maskSumSqrtS14M01.tensor, name: nil)

        let zeroPointone = graph.constant(0.1,
                                          shape: squared.shape!,
                                          dataType: dataType)

        self.tensor = graph.subtraction(squared,
                                        zeroPointone,
                                        name: nil)

        assert(self.tensor.shape?.count == 4)
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
    @objc
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
    ///   - useFP16: If true, use FP16 mode. If false, use FP32 mode
    ///   - useNHWC: If true, use NHWC mode. If false, use NCHW mode
    ///   - input: A pointer to the input tensor data
    ///   - output: A pointer to the output tensor data
    @objc
    class func test(descriptor: SWConvLayerDesc,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    batchSize: NSNumber,
                    useFP16: Bool,
                    useNHWC: Bool,
                    input: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {
        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)
        let graph = MPSGraph()

        let source = InputLayer(graph: graph,
                                batchSize: batchSize,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                numChannels: descriptor.inChannels,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        let conv = ConvLayer(graph: graph,
                             sourceTensor: source.tensor,
                             descriptor: descriptor,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let sourceArray = MPSNDArray(device: device.metalDevice!,
                                     tensor: source.tensor)

        if useFP16 {
            let inLength = source.tensor.countElements()

            sourceArray.writeBytes(input.toFP16(length: inLength))
        } else {
            sourceArray.writeBytes(input)
        }

        let sourceTensorData = MPSGraphTensorData(sourceArray)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData],
                              targetTensors: [conv.resultTensor],
                              targetOperations: nil)

        if useFP16 {
            let outLength = conv.resultTensor.countElements()
            let outputFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: outLength)

            fetch[conv.resultTensor]?.mpsndarray().readBytes(outputFP16)

            for i in 0..<outLength {
                output[i] = Float32(outputFP16[i])
            }
        } else {
            fetch[conv.resultTensor]?.mpsndarray().readBytes(output)
        }
    }

    /// Initializes a ConvLayer object
    /// - Parameters:
    ///   - graph: An MPSGraph object
    ///   - sourceTensor: The input tensor for the convolutional layer
    ///   - descriptor: A descriptor for the convolutional layer
    ///   - batchSize: The batch size of the input tensor
    ///   - nnXLen: The width of the input tensor
    ///   - nnYLen: The height of the input tensor
    ///   - useFP16: If true, use FP16 mode. If false, use FP32 mode
    ///   - useNHWC: If true, use NHWC mode. If false, use NCHW mode
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         descriptor: SWConvLayerDesc,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let dataType = MPSDataType.init(useFP16: useFP16)

        let dataLayout: MPSGraphTensorNamedDataLayout = useNHWC ? .NHWC : .NCHW

        let weightsShape = [descriptor.outChannels,
                            descriptor.inChannels,
                            descriptor.convYSize,
                            descriptor.convXSize]

        let convDescriptor =
        MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                          strideInY: 1,
                                          dilationRateInX: descriptor.dilationX,
                                          dilationRateInY: descriptor.dilationY,
                                          groups: 1,
                                          paddingStyle: .TF_SAME,
                                          dataLayout: dataLayout,
                                          weightsLayout: .OIHW)!

        let byteCount = weightsShape.countBytes(of: dataType)
        let weightsData: Data

        if useFP16 {
            let length = weightsShape.countElements()

            weightsData = Data(bytesNoCopy: descriptor.weights.toFP16(length: length),
                               count: byteCount,
                               deallocator: .free)
        } else {
            weightsData = Data(bytesNoCopy: descriptor.weights,
                               count: byteCount,
                               deallocator: .none)
        }

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: dataType)

        resultTensor = graph.convolution2D(sourceTensor,
                                           weights: weightsTensor,
                                           descriptor: convDescriptor,
                                           name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that represents a description of a batch normalization layer.
@objc
class SWBatchNormLayerDesc: NSObject {
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
    @objc
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

/// A class that represents a batch normalization layer.
@objc
class BatchNormLayer: NSObject {
    let resultTensor: MPSGraphTensor

    /// Executes a test for the batch normalization layer.
    /// - Parameters:
    ///   - descriptor: The description of the batch normalization layer.
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    ///   - batchSize: The number of input batches.
    ///   - useFP16: Indicates whether the layer should use 16-bit floating point numbers.
    ///   - useNHWC: Indicates whether the layer should use NHWC data layout.
    ///   - input: A pointer to the input data.
    ///   - maskPointer: A pointer to the mask data.
    ///   - output: A pointer to the output data.
    @objc
    class func test(descriptor: SWBatchNormLayerDesc,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    batchSize: NSNumber,
                    useFP16: Bool,
                    useNHWC: Bool,
                    input: UnsafeMutablePointer<Float32>,
                    mask maskPointer: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)
        let graph = MPSGraph()

        let source = InputLayer(graph: graph,
                                batchSize: batchSize,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                numChannels: descriptor.numChannels,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        let mask = MaskLayer(graph: graph,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let batchNorm = BatchNormLayer(graph: graph,
                                       sourceTensor: source.tensor,
                                       maskTensor: mask.tensor,
                                       descriptor: descriptor,
                                       nnXLen: nnXLen,
                                       nnYLen: nnYLen,
                                       batchSize: batchSize,
                                       useFP16: useFP16,
                                       useNHWC: useNHWC)

        let sourceArray = MPSNDArray(device: device.metalDevice!,
                                     tensor: source.tensor)

        let maskArray = MPSNDArray(device: device.metalDevice!,
                                   tensor: mask.tensor)

        if useFP16 {
            let inLength = source.tensor.countElements()
            let maskLength = mask.tensor.countElements()

            sourceArray.writeBytes(input.toFP16(length: inLength))

            maskArray.writeBytes(maskPointer.toFP16(length: maskLength))
        } else {
            sourceArray.writeBytes(input)
            maskArray.writeBytes(maskPointer)
        }

        let sourceTensorData = MPSGraphTensorData(sourceArray)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [batchNorm.resultTensor],
                              targetOperations: nil)

        if useFP16 {
            let outLength = batchNorm.resultTensor.countElements()
            let outputFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: outLength)

            fetch[batchNorm.resultTensor]?.mpsndarray().readBytes(outputFP16)

            for i in 0..<outLength {
                output[i] = Float32(outputFP16[i])
            }
        } else {
            fetch[batchNorm.resultTensor]?.mpsndarray().readBytes(output)
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
    ///   - useFP16: A boolean value indicating whether or not to use 16-bit floating point numbers.
    ///   - useNHWC: A boolean value indicating whether or not to use NHWC data format.
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         descriptor: SWBatchNormLayerDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let meanShape = InputShape.create(batchSize: 1,
                                          numChannels: descriptor.numChannels,
                                          nnYLen: 1,
                                          nnXLen: 1,
                                          useNHWC: useNHWC)

        let dataType = MPSDataType.init(useFP16: useFP16)
        let byteCount = meanShape.countBytes(of: dataType)
        let meanData: Data
        let varianceData: Data
        let scaleData: Data
        let biasData: Data

        if useFP16 {
            let length = meanShape.countElements()

            meanData = Data(bytesNoCopy: descriptor.mean.toFP16(length: length),
                            count: byteCount,
                            deallocator: .free)

            varianceData = Data(bytesNoCopy: descriptor.variance.toFP16(length: length),
                                count: byteCount,
                                deallocator: .free)

            scaleData = Data(bytesNoCopy: descriptor.scale.toFP16(length: length),
                             count: byteCount,
                             deallocator: .free)

            biasData = Data(bytesNoCopy: descriptor.bias.toFP16(length: length),
                            count: byteCount,
                            deallocator: .free)
        } else {
            meanData = Data(bytesNoCopy: descriptor.mean,
                            count: byteCount,
                            deallocator: .none)

            varianceData = Data(bytesNoCopy: descriptor.variance,
                                count: byteCount,
                                deallocator: .none)

            scaleData = Data(bytesNoCopy: descriptor.scale,
                             count: byteCount,
                             deallocator: .none)

            biasData = Data(bytesNoCopy: descriptor.bias,
                            count: byteCount,
                            deallocator: .none)
        }

        let meanTensor = graph.constant(meanData,
                                        shape: meanShape,
                                        dataType: dataType)

        let varianceTensor = graph.constant(varianceData,
                                            shape: meanShape,
                                            dataType: dataType)

        let scaleTensor = graph.constant(scaleData,
                                         shape: meanShape,
                                         dataType: dataType)

        let biasTensor = graph.constant(biasData,
                                        shape: meanShape,
                                        dataType: dataType)

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

/// A class that represents an activation layer
class ActivationLayer {
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
    @objc
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
    ///   - useFP16: If true, use FP16, otherwise use FP32
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW
    ///   - input: The input float32 pointer
    ///   - maskPointer: The mask float32 pointer
    ///   - output: The output float32 pointer
    @objc
    class func test(descriptor: SWResidualBlockDesc,
                    batchSize: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    useFP16: Bool,
                    useNHWC: Bool,
                    input: UnsafeMutablePointer<Float32>,
                    mask maskPointer: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)
        let graph = MPSGraph()

        let source = InputLayer(graph: graph,
                                batchSize: batchSize,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                numChannels: descriptor.preBN.numChannels,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        let mask = MaskLayer(graph: graph,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let block = ResidualBlock(graph: graph,
                                  sourceTensor: source.tensor,
                                  maskTensor: mask.tensor,
                                  descriptor: descriptor,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let sourceArray = MPSNDArray(device: device.metalDevice!,
                                     tensor: source.tensor)

        let maskArray = MPSNDArray(device: device.metalDevice!,
                                   tensor: mask.tensor)

        if useFP16 {
            let inLength = source.tensor.countElements()
            let maskLength = mask.tensor.countElements()

            sourceArray.writeBytes(input.toFP16(length: inLength))

            maskArray.writeBytes(maskPointer.toFP16(length: maskLength))
        } else {
            sourceArray.writeBytes(input)
            maskArray.writeBytes(maskPointer)
        }

        let sourceTensorData = MPSGraphTensorData(sourceArray)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [block.resultTensor],
                              targetOperations: nil)

        if useFP16 {
            let outLength = block.resultTensor.countElements()
            let outputFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: outLength)

            fetch[block.resultTensor]?.mpsndarray().readBytes(outputFP16)

            for i in 0..<outLength {
                output[i] = Float32(outputFP16[i])
            }
        } else {
            fetch[block.resultTensor]?.mpsndarray().readBytes(output)
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
    ///   - useFP16: If true, use FP16, otherwise use FP32
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         descriptor: SWResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let preActivation = ActivationLayer(graph: graph,
                                            sourceTensor: preBN.resultTensor,
                                            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(graph: graph,
                                    sourceTensor: preActivation.resultTensor,
                                    descriptor: descriptor.regularConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    useFP16: useFP16,
                                    useNHWC: useNHWC)

        let midBN = BatchNormLayer(graph: graph,
                                   sourceTensor: regularConv.resultTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let midActivation = ActivationLayer(graph: graph,
                                            sourceTensor: midBN.resultTensor,
                                            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(graph: graph,
                                  sourceTensor: midActivation.resultTensor,
                                  descriptor: descriptor.finalConv,
                                  batchSize: batchSize,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        resultTensor = graph.addition(sourceTensor,
                                      finalConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that represents a global pooling layer
class GlobalPoolingLayer {
    /// The resulting tensor after applying the global pooling operation
    let resultTensor: MPSGraphTensor

    /// Initialize a GlobalPoolingLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The source tensor to be pooled
    ///   - maskSumTensor: The sum of the mask
    ///   - maskSumSqrtS14M01Tensor: The multiplication of subtraction of square root of the sum of the mask
    ///   - useFP16: If true, use FP16, otherwise use FP32
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        let hwAxes = InputShape.getHWAxes(useNHWC: useNHWC)
        let channelAxis = InputShape.getChannelAxis(useNHWC: useNHWC)

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
        assert(useNHWC || (resultTensor.shape?[2] == 1))
        assert(useNHWC || (resultTensor.shape?[3] == 1))
        assert(!useNHWC || (resultTensor.shape?[1] == 1))
        assert(!useNHWC || (resultTensor.shape?[2] == 1))
    }
}

/// A class that represents a layer that performs global pooling on the input tensor
class GlobalPoolingValueLayer {
    let resultTensor: MPSGraphTensor

    /// Initialize a GlobalPoolingValueLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The input tensor
    ///   - maskSumTensor: The sum of the mask
    ///   - maskSumSqrtS14M01Tensor: The multiplication of subtraction of square root of the sum of the mask
    ///   - maskSumSqrtS14M01SquareS01Tensor: The subtraction of square of multiplication of subtraction of square root of the sum of the mask
    ///   - useFP16: If true, use FP16, otherwise use FP32
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        let hwAxes = InputShape.getHWAxes(useNHWC: useNHWC)
        let channelAxis = InputShape.getChannelAxis(useNHWC: useNHWC)

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
        assert(useNHWC || (resultTensor.shape?[2] == 1))
        assert(useNHWC || (resultTensor.shape?[3] == 1))
        assert(!useNHWC || (resultTensor.shape?[1] == 1))
        assert(!useNHWC || (resultTensor.shape?[2] == 1))
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
    @objc
    init(inChannels: NSNumber,
         outChannels: NSNumber,
         weights: UnsafeMutablePointer<Float32>) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
}

/// A class representing a matrix multiplication layer.
class MatMulLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes a MatMulLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - descriptor: The matrix multiplication layer descriptor.
    ///   - sourceTensor: The input tensor to the layer.
    ///   - useFP16: If true, use FP16, otherwise use FP32.
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW.
    init(graph: MPSGraph,
         descriptor: SWMatMulLayerDesc,
         sourceTensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {

        assert(useNHWC ||
               (descriptor.outChannels == 1) ||
               (sourceTensor.shape?.count == 2) ||
               ((sourceTensor.shape?.count == 4) &&
                (sourceTensor.shape?[2] == 1) && (sourceTensor.shape?[3] == 1)))

        assert((sourceTensor.shape?.count == 4) || (sourceTensor.shape?[1] == descriptor.inChannels))

        assert((sourceTensor.shape?.count == 2) || useNHWC || (sourceTensor.shape?[1] == descriptor.inChannels))

        assert((sourceTensor.shape?.count == 2) || (!useNHWC) || (sourceTensor.shape?[3] == descriptor.inChannels))

        let dataType = MPSDataType.init(useFP16: useFP16)

        let weightsShape = [descriptor.inChannels,
                            descriptor.outChannels]

        let byteCount = weightsShape.countBytes(of: dataType)
        let weightsData: Data

        if useFP16 {
            let length = weightsShape.countElements()

            weightsData = Data(bytesNoCopy: descriptor.weights.toFP16(length: length),
                               count: byteCount,
                               deallocator: .free)
        } else {
            weightsData = Data(bytesNoCopy: descriptor.weights,
                               count: byteCount,
                               deallocator: .none)
        }

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: dataType)

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
    @objc
    init(numChannels: NSNumber,
         weights: UnsafeMutablePointer<Float32>) {
        self.numChannels = numChannels
        self.weights = weights
    }
}

/// A class that performs matrix bias operations
class MatBiasLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes a MatBiasLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - descriptor: The descriptor that contains information about the layer
    ///   - sourceTensor: The input tensor to the layer.
    ///   - useFP16: If true, use FP16, otherwise use FP32.
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW.
    init(graph: MPSGraph,
         descriptor: SWMatBiasLayerDesc,
         sourceTensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {

        assert((sourceTensor.shape?.count == 2) && (sourceTensor.shape?[1] == descriptor.numChannels))

        let dataType = MPSDataType.init(useFP16: useFP16)
        let weightsShape = [1, descriptor.numChannels]
        let byteCount = weightsShape.countBytes(of: dataType)
        let weightsData: Data

        if useFP16 {
            let length = weightsShape.countElements()

            weightsData = Data(bytesNoCopy: descriptor.weights.toFP16(length: length),
                               count: byteCount,
                               deallocator: .free)
        } else {
            weightsData = Data(bytesNoCopy: descriptor.weights,
                               count: byteCount,
                               deallocator: .none)
        }

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: dataType)

        resultTensor = graph.addition(sourceTensor,
                                      weightsTensor,
                                      name: nil)
    }
}

/// A class that performs bias operations in NC coordinates.
class AddNCBiasLayer {
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
    ///   - useFP16: If true, use FP16, otherwise use FP32.
    ///   - useNHWC: If true, use NHWC, otherwise use NCHW.
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         biasTensor: MPSGraphTensor,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         numChannels: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let shape = InputShape.create(batchSize: batchSize,
                                      numChannels: numChannels,
                                      nnYLen: 1,
                                      nnXLen: 1,
                                      useNHWC: useNHWC)

        assert(biasTensor.countElements() == shape.countElements())
        let reshaped = graph.reshape(biasTensor, shape: shape, name: nil)
        resultTensor = graph.addition(sourceTensor, reshaped, name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(useNHWC || resultTensor.shape?[2] == nnYLen)
        assert(useNHWC || resultTensor.shape?[3] == nnXLen)
        assert(!useNHWC || resultTensor.shape?[1] == nnYLen)
        assert(!useNHWC || resultTensor.shape?[2] == nnXLen)
    }
}

/// A class that represents a residual block with global pooling.
@objc
class SWGlobalPoolingResidualBlockDesc: NSObject {
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
    @objc
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

/// A class representing a residual block with global pooling
@objc
class GlobalPoolingResidualBlock: NSObject {
    let resultTensor: MPSGraphTensor

    /// A method to test the global pooling residual block
    ///
    /// - Parameters:
    ///   - descriptor: The descriptor of the global pooling residual block
    ///   - batchSize: The batch size
    ///   - nnXLen: The X length
    ///   - nnYLen: The Y length
    ///   - useFP16: If true, use 16-bit floating point format, otherwise use 32-bit
    ///   - useNHWC: If true, use NHWC format, otherwise use NCHW format
    ///   - input: The input pointer
    ///   - maskPointer: The mask pointer
    ///   - output: The output pointer
    @objc
    class func test(descriptor: SWGlobalPoolingResidualBlockDesc,
                    batchSize: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    useFP16: Bool,
                    useNHWC: Bool,
                    input: UnsafeMutablePointer<Float32>,
                    mask maskPointer: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)
        let graph = MPSGraph()

        let source = InputLayer(graph: graph,
                                batchSize: batchSize,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                numChannels: descriptor.preBN.numChannels,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        let mask = MaskLayer(graph: graph,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let maskSum = MaskSumLayer(graph: graph, mask: mask, useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum,
                                                       useFP16: useFP16)

        let block =
        GlobalPoolingResidualBlock(graph: graph,
                                   sourceTensor: source.tensor,
                                   maskTensor: mask.tensor,
                                   maskSumTensor: maskSum.tensor,
                                   maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                   descriptor: descriptor,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let sourceArray = MPSNDArray(device: device.metalDevice!,
                                     tensor: source.tensor)

        let maskArray = MPSNDArray(device: device.metalDevice!,
                                   tensor: mask.tensor)

        if useFP16 {
            let inLength = source.tensor.countElements()
            let maskLength = mask.tensor.countElements()

            sourceArray.writeBytes(input.toFP16(length: inLength))

            maskArray.writeBytes(maskPointer.toFP16(length: maskLength))
        } else {
            sourceArray.writeBytes(input)
            maskArray.writeBytes(maskPointer)
        }

        let sourceTensorData = MPSGraphTensorData(sourceArray)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [block.resultTensor],
                              targetOperations: nil)

        if useFP16 {
            let outLength = block.resultTensor.countElements()
            let outputFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: outLength)

            fetch[block.resultTensor]?.mpsndarray().readBytes(outputFP16)

            for i in 0..<outLength {
                output[i] = Float32(outputFP16[i])
            }
        } else {
            fetch[block.resultTensor]?.mpsndarray().readBytes(output)
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
    ///   - useFP16: If true, use 16-bit floating point format, otherwise use 32-bit
    ///   - useNHWC: If true, use NHWC format, otherwise use NCHW format
    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         descriptor: SWGlobalPoolingResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let mask = MaskLayer(tensor: maskTensor)
        let maskSum = MaskSumLayer(tensor: maskSumTensor)
        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(tensor: maskSumSqrtS14M01Tensor)

        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: mask.tensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let preReLU = graph.reLU(with: preBN.resultTensor, name: nil)

        let regularConv = ConvLayer(graph: graph,
                                    sourceTensor: preReLU,
                                    descriptor: descriptor.regularConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    useFP16: useFP16,
                                    useNHWC: useNHWC)

        let gpoolConv = ConvLayer(graph: graph,
                                  sourceTensor: preReLU,
                                  descriptor: descriptor.gpoolConv,
                                  batchSize: batchSize,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let gpoolBN = BatchNormLayer(graph: graph,
                                     sourceTensor: gpoolConv.resultTensor,
                                     maskTensor: mask.tensor,
                                     descriptor: descriptor.gpoolBN,
                                     nnXLen: nnXLen,
                                     nnYLen: nnYLen,
                                     batchSize: batchSize,
                                     useFP16: useFP16,
                                     useNHWC: useNHWC)

        let gpoolReLU = graph.reLU(with: gpoolBN.resultTensor, name: nil)

        let gpoolConcat = GlobalPoolingLayer(graph: graph,
                                             sourceTensor: gpoolReLU,
                                             maskSumTensor: maskSum.tensor,
                                             maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                             useFP16: useFP16,
                                             useNHWC: useNHWC)

        assert(useNHWC || (gpoolConcat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels))
        assert(!useNHWC || (gpoolConcat.resultTensor.shape?[3] == descriptor.gpoolToBiasMul.inChannels))

        let gpoolToBiasMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToBiasMul,
                                         sourceTensor: gpoolConcat.resultTensor,
                                         useFP16: useFP16,
                                         useNHWC: useNHWC)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: regularConv.resultTensor,
                                   biasTensor: gpoolToBiasMul.resultTensor,
                                   batchSize: batchSize,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.gpoolToBiasMul.outChannels,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let midBN = BatchNormLayer(graph: graph,
                                   sourceTensor: added.resultTensor,
                                   maskTensor: mask.tensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let midReLU = graph.reLU(with: midBN.resultTensor, name: nil)

        let finalConv = ConvLayer(graph: graph,
                                  sourceTensor: midReLU,
                                  descriptor: descriptor.finalConv,
                                  batchSize: batchSize,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        resultTensor = graph.addition(sourceTensor,
                                      finalConv.resultTensor,
                                      name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// An enumeration of the different kinds of blocks that can be used in a residual network.
@objc enum BlockKind: Int {
    case ordinary
    case dilated
    case globalPooling
}

/// A class that represents a block descriptor that is used to define the characteristics of a residual block.
@objc
class BlockDescriptor: NSObject {
    /// The kind of the block, it can be ordinary, dilated or globalPooling.
    let kind: BlockKind

    /// The descriptor for the ordinary residual block, if the kind is ordinary.
    let ordinary: SWResidualBlockDesc?

    /// The descriptor for the global pooling residual block, if the kind is globalPooling.
    let globalPooling: SWGlobalPoolingResidualBlockDesc?

    /// Initializes a block descriptor object with the given parameters.
    ///
    /// - Parameters:
    ///   - kind: The kind of the block.
    ///   - ordinary: The descriptor for the ordinary residual block, if the kind is ordinary.
    ///   - globalPooling: The descriptor for the global pooling residual block, if the kind is globalPooling.
    @objc
    init(kind: BlockKind,
         ordinary: SWResidualBlockDesc?,
         globalPooling: SWGlobalPoolingResidualBlockDesc?) {
        self.kind = kind
        self.ordinary = ordinary
        self.globalPooling = globalPooling
    }
}

/// A class that describes a trunk for a neural network
@objc
class SWTrunkDesc: NSObject {
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
    let blocks: [BlockDescriptor]
    /// The description of the batch normalization layer that is applied at the end of the trunk
    let trunkTipBN: SWBatchNormLayerDesc

    /// Initializes a SWTrunkDesc object
    /// - Parameters:
    ///   - version: The version of the ResNet trunk
    ///   - trunkNumChannels: Number of channels for the trunk
    ///   - midNumChannels: Number of channels for the mid section
    ///   - regularNumChannels: Number of channels for the regular section
    ///   - gpoolNumChannels: Number of channels for the global pooling section
    ///   - initialConv: The description of the initial convolutional layer
    ///   - initialMatMul: The description of the initial matrix multiplication layer
    ///   - blocks: The list of blocks that make up the trunk
    ///   - trunkTipBN: The description of the batch normalization layer that is applied at the end of the trunk
    @objc
    init(version: Int,
         trunkNumChannels: NSNumber,
         midNumChannels: NSNumber,
         regularNumChannels: NSNumber,
         gpoolNumChannels: NSNumber,
         initialConv: SWConvLayerDesc,
         initialMatMul: SWMatMulLayerDesc,
         blocks: [BlockDescriptor],
         trunkTipBN: SWBatchNormLayerDesc) {
        self.version = version
        self.trunkNumChannels = trunkNumChannels
        self.midNumChannels = midNumChannels
        self.regularNumChannels = regularNumChannels
        self.gpoolNumChannels = gpoolNumChannels
        self.initialConv = initialConv
        self.initialMatMul = initialMatMul
        self.blocks = blocks
        self.trunkTipBN = trunkTipBN
    }
}

/// A class representing a ResNet trunk for a neural network
class Trunk {
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
    ///   - useFP16: Whether to use FP16 precision
    ///   - useNHWC: Whether to use NHWC format
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
         numGlobalFeatures: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {

        let initialConv = ConvLayer(graph: graph,
                                    sourceTensor: inputTensor,
                                    descriptor: descriptor.initialConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    useFP16: useFP16,
                                    useNHWC: useNHWC)

        let initialMatMul = MatMulLayer(graph: graph,
                                        descriptor: descriptor.initialMatMul,
                                        sourceTensor: inputGlobalTensor,
                                        useFP16: useFP16,
                                        useNHWC: useNHWC)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: initialConv.resultTensor,
                                   biasTensor: initialMatMul.resultTensor,
                                   batchSize: batchSize,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.initialMatMul.outChannels,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        var blockInput = added.resultTensor

        for block in descriptor.blocks {
            assert((block.kind == .ordinary) || (block.kind == .globalPooling))

            switch block.kind {
            case .ordinary:
                let ordinary = ResidualBlock(graph: graph,
                                             sourceTensor: blockInput,
                                             maskTensor: maskTensor,
                                             descriptor: block.ordinary!,
                                             nnXLen: nnXLen,
                                             nnYLen: nnYLen,
                                             batchSize: batchSize,
                                             useFP16: useFP16,
                                             useNHWC: useNHWC)

                blockInput = ordinary.resultTensor
            default:
                let globalPooling =
                GlobalPoolingResidualBlock(graph: graph,
                                           sourceTensor: blockInput,
                                           maskTensor: maskTensor,
                                           maskSumTensor: maskSumTensor,
                                           maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                           descriptor: block.globalPooling!,
                                           nnXLen: nnXLen,
                                           nnYLen: nnYLen,
                                           batchSize: batchSize,
                                           useFP16: useFP16,
                                           useNHWC: useNHWC)

                blockInput = globalPooling.resultTensor
            }
        }

        let trunkTipBN = BatchNormLayer(graph: graph,
                                        sourceTensor: blockInput,
                                        maskTensor: maskTensor,
                                        descriptor: descriptor.trunkTipBN,
                                        nnXLen: nnXLen,
                                        nnYLen: nnYLen,
                                        batchSize: batchSize,
                                        useFP16: useFP16,
                                        useNHWC: useNHWC)

        let trunkTipReLU = graph.reLU(with: trunkTipBN.resultTensor, name: nil)

        resultTensor = trunkTipReLU

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that describes a policy head for a neural network
@objc
class SWPolicyHeadDesc: NSObject {
    /// The version of the policy head
    let version: Int
    /// The description of the first convolutional layer of the policy head
    let p1Conv: SWConvLayerDesc
    /// The description of the first global pooling convolutional layer of the policy head
    let g1Conv: SWConvLayerDesc
    /// The description of the batch normalization layer that is applied after the first global pooling convolutional layer
    let g1BN: SWBatchNormLayerDesc
    /// The description of the matrix multiplication layer that converts the global pooling convolutional output to bias
    let gpoolToBiasMul: SWMatMulLayerDesc
    /// The description of the batch normalization layer that is applied after the first convolutional layer
    let p1BN: SWBatchNormLayerDesc
    /// The description of the second convolutional layer of the policy head
    let p2Conv: SWConvLayerDesc
    /// The description of the matrix multiplication layer that converts the global pooling convolutional output to pass
    let gpoolToPassMul: SWMatMulLayerDesc

    /// Initializes a SWPolicyHeadDesc object
    /// - Parameters:
    ///   - version: The version of the policy head
    ///   - p1Conv: The description of the first convolutional layer of the policy head
    ///   - g1Conv: The description of the first global pooling convolutional layer of the policy head
    ///   - g1BN: The description of the batch normalization layer that is applied after the first global pooling convolutional layer
    ///   - gpoolToBiasMul: The description of the matrix multiplication layer that converts the global pooling convolutional output to bias
    ///   - p1BN: The description of the batch normalization layer that is applied after the first convolutional layer
    ///   - p2Conv: The description of the second convolutional layer of the policy head
    ///   - gpoolToPassMul: The description of the matrix multiplication layer that converts the global pooling convolutional output to pass
    @objc
    init(version: Int,
         p1Conv: SWConvLayerDesc,
         g1Conv: SWConvLayerDesc,
         g1BN: SWBatchNormLayerDesc,
         gpoolToBiasMul: SWMatMulLayerDesc,
         p1BN: SWBatchNormLayerDesc,
         p2Conv: SWConvLayerDesc,
         gpoolToPassMul: SWMatMulLayerDesc) {
        self.version = version
        self.p1Conv = p1Conv
        self.g1Conv = g1Conv
        self.g1BN = g1BN
        self.gpoolToBiasMul = gpoolToBiasMul
        self.p1BN = p1BN
        self.p2Conv = p2Conv
        self.gpoolToPassMul = gpoolToPassMul
    }
}

/// A class that represents a policy head of a neural network.
class PolicyHead {
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
    ///   - useFP16: A boolean flag that determines whether the policy head uses FP16
    ///   - useNHWC: A boolean flag that determines whether the policy head uses NHWC
    init(graph: MPSGraph,
         descriptor: SWPolicyHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {

        let p1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.p1Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let g1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.g1Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let g1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: g1Conv.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.g1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let g1ReLU = graph.reLU(with: g1BN.resultTensor, name: nil)

        let g1Concat = GlobalPoolingLayer(graph: graph,
                                          sourceTensor: g1ReLU,
                                          maskSumTensor: maskSumTensor,
                                          maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                          useFP16: useFP16,
                                          useNHWC: useNHWC)

        assert(useNHWC || (g1Concat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels))
        assert(!useNHWC || (g1Concat.resultTensor.shape?[3] == descriptor.gpoolToBiasMul.inChannels))

        let gpoolToBiasMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToBiasMul,
                                         sourceTensor: g1Concat.resultTensor,
                                         useFP16: useFP16,
                                         useNHWC: useNHWC)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: p1Conv.resultTensor,
                                   biasTensor: gpoolToBiasMul.resultTensor,
                                   batchSize: batchSize,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   numChannels: descriptor.gpoolToBiasMul.outChannels,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let p1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: added.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.p1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let p1ReLU = graph.reLU(with: p1BN.resultTensor, name: nil)

        let p2Conv = ConvLayer(graph: graph,
                               sourceTensor: p1ReLU,
                               descriptor: descriptor.p2Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        assert(useNHWC || (g1Concat.resultTensor.shape?[1] == descriptor.gpoolToPassMul.inChannels))
        assert(!useNHWC || (g1Concat.resultTensor.shape?[3] == descriptor.gpoolToPassMul.inChannels))

        let gpoolToPassMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToPassMul,
                                         sourceTensor: g1Concat.resultTensor,
                                         useFP16: useFP16,
                                         useNHWC: useNHWC)

        policyTensor = p2Conv.resultTensor
        policyPassTensor = gpoolToPassMul.resultTensor

        assert(policyTensor.shape?.count == 4)
        assert(policyPassTensor.shape?.count == 2)
    }
}

/// A class that describes the value head of a neural network
@objc
class SWValueHeadDesc: NSObject {
    /// The version of the value head
    let version: Int
    /// The description of the first convolutional layer in the value head
    let v1Conv: SWConvLayerDesc
    /// The description of the batch normalization layer after the first convolutional layer in the value head
    let v1BN: SWBatchNormLayerDesc
    /// The description of the matrix multiplication layer that is applied to the output of the first convolutional layer in the value head
    let v2Mul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    let v2Bias: SWMatBiasLayerDesc
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
    ///   - v2Mul: The description of the matrix multiplication layer that is applied to the output of the first convolutional layer in the value head
    ///   - v2Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - v3Mul: The description of the matrix multiplication layer that is applied to the output of the bias layer in the value head
    ///   - v3Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - sv3Mul: The description of the matrix multiplication layer that is applied to the output of the third bias layer in the value head
    ///   - sv3Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - vOwnershipConv: The description of the convolutional layer that is applied to the board ownership map in the value head
    @objc
    init(version: Int, v1Conv: SWConvLayerDesc, v1BN: SWBatchNormLayerDesc, v2Mul: SWMatMulLayerDesc, v2Bias: SWMatBiasLayerDesc, v3Mul: SWMatMulLayerDesc, v3Bias: SWMatBiasLayerDesc, sv3Mul: SWMatMulLayerDesc, sv3Bias: SWMatBiasLayerDesc, vOwnershipConv: SWConvLayerDesc) {
        self.version = version
        self.v1Conv = v1Conv
        self.v1BN = v1BN
        self.v2Mul = v2Mul
        self.v2Bias = v2Bias
        self.v3Mul = v3Mul
        self.v3Bias = v3Bias
        self.sv3Mul = sv3Mul
        self.sv3Bias = sv3Bias
        self.vOwnershipConv = vOwnershipConv
    }
}

/// A class that creates a value head for the neural network, which produces the value, score value, and ownership tensors.
class ValueHead {
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
    ///   - useFP16: A boolean value indicating whether to use half-precision floating-point numbers
    ///   - useNHWC: A boolean value indicating whether to use NHWC (channel last) format for the tensor shape
    init(graph: MPSGraph,
         descriptor: SWValueHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {

        let v1Conv = ConvLayer(graph: graph,
                               sourceTensor: sourceTensor,
                               descriptor: descriptor.v1Conv,
                               batchSize: batchSize,
                               nnXLen: nnXLen,
                               nnYLen: nnYLen,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let v1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: v1Conv.resultTensor,
                                  maskTensor: maskTensor,
                                  descriptor: descriptor.v1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let v1ReLU = graph.reLU(with: v1BN.resultTensor, name: nil)

        let v1Mean =
        GlobalPoolingValueLayer(graph: graph,
                                sourceTensor: v1ReLU,
                                maskSumTensor: maskSumTensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                                maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01Tensor,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        assert(useNHWC || (v1Mean.resultTensor.shape?[1] == descriptor.v2Mul.inChannels))
        assert(!useNHWC || (v1Mean.resultTensor.shape?[3] == descriptor.v2Mul.inChannels))

        let v2Mul = MatMulLayer(graph: graph,
                                descriptor: descriptor.v2Mul,
                                sourceTensor: v1Mean.resultTensor,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        let v2Bias = MatBiasLayer(graph: graph,
                                  descriptor: descriptor.v2Bias,
                                  sourceTensor: v2Mul.resultTensor,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let v2ReLU = graph.reLU(with: v2Bias.resultTensor, name: nil)

        let v3Mul = MatMulLayer(graph: graph,
                                descriptor: descriptor.v3Mul,
                                sourceTensor: v2ReLU,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        let v3Bias = MatBiasLayer(graph: graph,
                                  descriptor: descriptor.v3Bias,
                                  sourceTensor: v3Mul.resultTensor,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let sv3Mul = MatMulLayer(graph: graph,
                                 descriptor: descriptor.sv3Mul,
                                 sourceTensor: v2ReLU,
                                 useFP16: useFP16,
                                 useNHWC: useNHWC)

        let sv3Bias = MatBiasLayer(graph: graph,
                                   descriptor: descriptor.sv3Bias,
                                   sourceTensor: sv3Mul.resultTensor,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let vOwnershipConv = ConvLayer(graph: graph,
                                       sourceTensor: v1ReLU,
                                       descriptor: descriptor.vOwnershipConv,
                                       batchSize: batchSize,
                                       nnXLen: nnXLen,
                                       nnYLen: nnYLen,
                                       useFP16: useFP16,
                                       useNHWC: useNHWC)

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
    @objc
    init(version: Int,
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

/// A class representing a neural network model for processing Go game states.
class Model {
    /// The Metal Performance Shaders graph object used for building and executing the graph
    let graph: MPSGraph
    /// The length of the neural network input in the x dimension
    let nnXLen: NSNumber
    /// The length of the neural network input in the y dimension
    let nnYLen: NSNumber
    /// The batch size of the neural network input
    let batchSize: NSNumber
    /// A flag that indicates whether or not to use the half-precision floating point format for computations
    let useFP16: Bool
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
    /// The number of elements in the input layer
    let inputCount: Int
    /// A pointer to the half-precision floating point input data
    let inputFP16: UnsafeMutablePointer<Float16>?
    /// The number of elements in the global input layer
    let inputGlobalCount: Int
    /// A pointer to the half-precision floating point global input data
    let inputGlobalFP16: UnsafeMutablePointer<Float16>?
    /// The number of elements in the policy output layer
    let policyCount: Int
    /// A pointer to the half-precision floating point policy output data
    let policyFP16: UnsafeMutablePointer<Float16>?
    /// The number of elements in the policy pass output layer
    let policyPassCount: Int
    /// A pointer to the half-precision floating point policy pass output data
    let policyPassFP16: UnsafeMutablePointer<Float16>?
    /// The number of elements in the value output layer
    let valueCount: Int
    /// A pointer to the half-precision floating point value output data
    let valueFP16: UnsafeMutablePointer<Float16>?
    /// The number of elements in the score value output layer
    let scoreValueCount: Int
    /// A pointer to the half-precision floating point score value output data
    let scoreValueFP16: UnsafeMutablePointer<Float16>?
    /// The number of elements in the ownership output layer
    let ownershipCount: Int
    /// A pointer to the half-precision floating point ownership output data
    let ownershipFP16: UnsafeMutablePointer<Float16>?
    /// The input layer as a Metal Performance Shaders n-dimensional array
    let inputArray: MPSNDArray
    /// The global input layer as a Metal Performance Shaders n-dimensional array
    let inputGlobalArray: MPSNDArray
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
    ///   - useFP16: A flag that indicates whether or not to use the half-precision floating point format for computations.
    ///   - useNHWC: A flag that indicates whether or not to use the NHWC format for computations.
    init(device: MPSGraphDevice,
         graph: MPSGraph,
         descriptor: SWModelDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        self.graph = graph
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.batchSize = batchSize
        self.useFP16 = useFP16
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
                           numChannels: descriptor.numInputChannels,
                           useFP16: useFP16,
                           useNHWC: useNHWC)

        inputGlobal = InputGlobalLayer(graph: graph,
                                       batchSize: batchSize,
                                       numGlobalFeatures: descriptor.numInputGlobalChannels,
                                       useFP16: useFP16,
                                       useNHWC: useNHWC)

        let startOfMask: [NSNumber] = [0, 0, 0, 0]

        let endOfMask = InputShape.create(batchSize: batchSize,
                                          numChannels: 1,
                                          nnYLen: nnYLen,
                                          nnXLen: nnXLen,
                                          useNHWC: useNHWC)

        let maskTensor = graph.sliceTensor(input.tensor,
                                           starts: startOfMask,
                                           ends: endOfMask,
                                           strides: [1, 1, 1, 1],
                                           name: nil)

        let mask = MaskLayer(tensor: maskTensor)

        let maskSum = MaskSumLayer(graph: graph,
                                   mask: mask,
                                   useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum,
                                                       useFP16: useFP16)

        let maskSumSqrtS14M01SquareS01 = MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                                                         maskSumSqrtS14M01: maskSumSqrtS14M01,
                                                                         useFP16: useFP16)

        trunk = Trunk(graph: graph,
                      descriptor: descriptor.trunk,
                      inputTensor: input.tensor,
                      inputGlobalTensor: inputGlobal.tensor,
                      maskTensor: mask.tensor,
                      maskSumTensor: maskSum.tensor,
                      maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                      nnXLen: nnXLen,
                      nnYLen: nnYLen,
                      batchSize: batchSize,
                      numSpatialFeatures: descriptor.numInputChannels,
                      numGlobalFeatures: descriptor.numInputGlobalChannels,
                      useFP16: useFP16,
                      useNHWC: useNHWC)

        policyHead = PolicyHead(graph: graph,
                                descriptor: descriptor.policyHead,
                                sourceTensor: trunk.resultTensor,
                                maskTensor: mask.tensor,
                                maskSumTensor: maskSum.tensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                nnXLen: nnXLen,
                                nnYLen: nnYLen,
                                batchSize: batchSize,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

        valueHead = ValueHead(graph: graph,
                              descriptor: descriptor.valueHead,
                              sourceTensor: trunk.resultTensor,
                              maskTensor: mask.tensor,
                              maskSumTensor: maskSum.tensor,
                              maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                              maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
                              nnXLen: nnXLen,
                              nnYLen: nnYLen,
                              batchSize: batchSize,
                              useFP16: useFP16,
                              useNHWC: useNHWC)

        inputCount = input.tensor.countElements()
        inputGlobalCount = inputGlobal.tensor.countElements()
        policyCount = policyHead.policyTensor.countElements()
        policyPassCount = policyHead.policyPassTensor.countElements()
        valueCount = valueHead.valueTensor.countElements()
        scoreValueCount = valueHead.scoreValueTensor.countElements()
        ownershipCount = valueHead.ownershipTensor.countElements()

        if useFP16 {
            inputFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: inputCount)
            inputGlobalFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: inputGlobalCount)
            policyFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: policyCount)
            policyPassFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: policyPassCount)
            valueFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: valueCount)
            scoreValueFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: scoreValueCount)
            ownershipFP16 = UnsafeMutablePointer<Float16>.allocate(capacity: ownershipCount)
        } else {
            inputFP16 = nil
            inputGlobalFP16 = nil
            policyFP16 = nil
            policyPassFP16 = nil
            valueFP16 = nil
            scoreValueFP16 = nil
            ownershipFP16 = nil
        }

        inputArray = MPSNDArray(device: device.metalDevice!,
                                tensor: input.tensor)

        inputGlobalArray = MPSNDArray(device: device.metalDevice!,
                                      tensor: inputGlobal.tensor)

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
               ownership: UnsafeMutablePointer<Float32>) {
        if let inputFP16 {
            assert(useFP16)
            inputPointer.toFP16(inputFP16, length: inputCount)
            inputArray.writeBytes(inputFP16)
        } else {
            assert(!useFP16)
            inputArray.writeBytes(inputPointer)
        }

        if let inputGlobalFP16 {
            inputGlobalPointer.toFP16(inputGlobalFP16, length: inputGlobalCount)
            inputGlobalArray.writeBytes(inputGlobalFP16)
        } else {
            inputGlobalArray.writeBytes(inputGlobalPointer)
        }

        let commandBuffer = MPSCommandBuffer(commandBuffer: commandQueue.makeCommandBuffer()!)

        let fetch = graph.encode(to: commandBuffer,
                                 feeds: feeds,
                                 targetTensors: targetTensors,
                                 targetOperations: nil,
                                 executionDescriptor: nil)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let policyFP16 {
            fetch[policyHead.policyTensor]?.mpsndarray().readBytes(policyFP16)

            policyFP16.toFP32(policy, length: policyCount)
        } else {
            fetch[policyHead.policyTensor]?.mpsndarray().readBytes(policy)

        }

        if let policyPassFP16 {
            fetch[policyHead.policyPassTensor]?.mpsndarray().readBytes(policyPassFP16)

            policyPassFP16.toFP32(policyPass, length: policyPassCount)
        } else {
            fetch[policyHead.policyPassTensor]?.mpsndarray().readBytes(policyPass)
        }

        if let valueFP16 {
            fetch[valueHead.valueTensor]?.mpsndarray().readBytes(valueFP16)

            valueFP16.toFP32(value, length: valueCount)
        } else {
            fetch[valueHead.valueTensor]?.mpsndarray().readBytes(value)
        }

        if let scoreValueFP16 {
            fetch[valueHead.scoreValueTensor]?.mpsndarray().readBytes(scoreValueFP16)

            scoreValueFP16.toFP32(scoreValue, length: scoreValueCount)
        } else {
            fetch[valueHead.scoreValueTensor]?.mpsndarray().readBytes(scoreValue)
        }

        if let ownershipFP16 {
            fetch[valueHead.ownershipTensor]?.mpsndarray().readBytes(ownershipFP16)

            ownershipFP16.toFP32(ownership, length: ownershipCount)
        } else {
            fetch[valueHead.ownershipTensor]?.mpsndarray().readBytes(ownership)
        }
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
    static let defaultUseFP16Mode: SWEnable = .Auto
    static let defaultUseNHWCMode: SWEnable = .Auto

    static var instance = MetalComputeContext(nnXLen: defaultNnXLen,
                                              nnYLen: defaultNnYLen,
                                              useFP16Mode: defaultUseFP16Mode,
                                              useNHWCMode: defaultUseNHWCMode)
    let nnXLen: NSNumber
    let nnYLen: NSNumber
    let useFP16Mode: SWEnable
    let useNHWCMode: SWEnable

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
                                       nnYLen: nnYLen,
                                       useFP16Mode: useFP16Mode,
                                       useNHWCMode: useNHWCMode)
    }

    /// Destroy the context.
    @objc class func destroyInstance() {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        instance = MetalComputeContext(nnXLen: defaultNnXLen,
                                       nnYLen: defaultNnYLen,
                                       useFP16Mode: defaultUseFP16Mode,
                                       useNHWCMode: defaultUseNHWCMode)
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
    ///   - useFP16Mode: use FP16 mode or not.
    ///   - useNHWCMode: use NHWC mode or not.
    private init(nnXLen: NSNumber,
                 nnYLen: NSNumber,
                 useFP16Mode: SWEnable,
                 useNHWCMode: SWEnable) {
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.useFP16Mode = useFP16Mode
        self.useNHWCMode = useNHWCMode
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
        let useFP16: Bool
        let useNHWC: Bool
        let devices = MTLCopyAllDevices()
        let mtlDevice: MTLDevice

        // Select a GPU device.
        if ((gpuIdx >= 0) && (gpuIdx < devices.count)) {
            mtlDevice = devices[gpuIdx]
        } else {
            mtlDevice = MTLCreateSystemDefaultDevice()!
        }

        let device = MPSGraphDevice(mtlDevice: mtlDevice)

        NSLog("Metal backend thread \(threadIdx): \(mtlDevice.name) Model version \(descriptor.version)")

        NSLog("Metal backend thread \(threadIdx): \(mtlDevice.name) Model name \(descriptor.name)")

        // Select useFP16 mode.
        switch context.useFP16Mode {
        case .True: useFP16 = true
        default: useFP16 = false
        }

        // Select useNHWC mode.
        switch context.useNHWCMode {
        case .True: useNHWC = true
        default: useNHWC = false
        }

        // Create a model.
        model = Model(device: device,
                      graph: MPSGraph(),
                      descriptor: descriptor,
                      nnXLen: context.nnXLen,
                      nnYLen: context.nnYLen,
                      batchSize: batchSize,
                      useFP16: useFP16,
                      useNHWC: useNHWC)

        NSLog("Metal backend thread \(threadIdx): \(mtlDevice.name) useFP16=\(useFP16) useNHWC=\(useNHWC) batchSize=\(batchSize)")
    }
}

/// A class that represents Metal backend.
@objc class MetalBackend : NSObject {

    /// Print all available devices.
    @objc class func printDevices() {
        let devices = MTLCopyAllDevices()

        for i in 0..<devices.count {
            print("Found Metal Device \(i): \(devices[i].name) (isLowPower:\(devices[i].isLowPower), isRemovable:\(devices[i].isRemovable))")
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
                               ownership: ownershipOutput)
        }
    }
}
