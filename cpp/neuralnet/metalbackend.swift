import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

extension UnsafeMutablePointer<Float32> {
    func printAsFloat(_ length: Int) {
        for i in 0..<length {
            print("data[\(i)]=\(self[i])")
        }
    }
}

extension MPSNDArray {
    func dumpFloats(name: String?, length: Int) {
        print(name ?? "")
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)
        readBytes(buffer, strideBytes: nil)
        buffer.printAsFloat(length)
    }
}

extension MPSGraphTensorData {
    convenience init?(device: MPSGraphDevice, tensor: MPSGraphTensor) {
        if let metalDevice = device.metalDevice {
            if let shape = tensor.shape {
                self.init(MPSNDArray(device: metalDevice,
                                     descriptor: MPSNDArrayDescriptor(dataType: tensor.dataType,
                                                                      shape: shape)))
            } else {
                return nil
            }
        } else {
            return nil
        }
    }
}

extension Array where Element == NSNumber {
    func product() -> NSNumber {
        var result = 1.0
        for x in self {
            result *= x.doubleValue
        }

        return result as NSNumber
    }

    func asShapeCount(of dataType: MPSDataType) -> Int {
        assert(dataType == .float32)
        return product().intValue * MemoryLayout<Float32>.size
    }
}

/// Source layer in NxHxWxC or NxCxHxW
class SourceLayer {
    let tensor: MPSGraphTensor
    let layout: MPSGraphTensorNamedDataLayout

    init(graph: MPSGraph,
         tensor: MPSGraphTensor?,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         numChannels: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let shape: [NSNumber]
        let dataType = MPSDataType.float32

        if useNHWC {
            shape = [batchSize,
                     nnYLen,
                     nnXLen,
                     numChannels]

            layout = MPSGraphTensorNamedDataLayout.NHWC
        } else {
            shape = [batchSize,
                     numChannels,
                     nnYLen,
                     nnXLen]

            layout = MPSGraphTensorNamedDataLayout.NCHW
        }

        self.tensor = tensor ?? graph.placeholder(shape: shape,
                                                  dataType: dataType,
                                                  name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

class InputGlobalLayer {
    let tensor: MPSGraphTensor

    init(graph: MPSGraph,
         tensor: MPSGraphTensor?,
         batchSize: NSNumber,
         numGlobalFeatures: NSNumber,
         useFP16: Bool) {
        let shape = [batchSize, numGlobalFeatures]
        let dataType = MPSDataType.float32

        self.tensor = tensor ?? graph.placeholder(shape: shape,
                                                  dataType: dataType,
                                                  name: nil)

        assert(self.tensor.shape?.count == 2)
    }
}

class MaskLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    init(graph: MPSGraph,
         tensor: MPSGraphTensor?,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let dataType = MPSDataType.float32

        if useNHWC {
            shape = [batchSize,
                     nnYLen,
                     nnXLen,
                     1]
        } else {
            shape = [batchSize,
                     1,
                     nnYLen,
                     nnXLen]
        }

        self.tensor = tensor ?? graph.placeholder(shape: shape,
                                                  dataType: dataType,
                                                  name: nil)

        assert(self.tensor.shape?.count == 4)
        assert(self.tensor.shape == shape)
    }
}

class MaskSumLayer {
    let tensor: MPSGraphTensor

    init(graph: MPSGraph,
         tensor: MPSGraphTensor?,
         mask: MaskLayer,
         useNHWC: Bool) {
        let hwAxes: [NSNumber]

        if useNHWC {
            hwAxes = [1, 2]
        } else {
            hwAxes = [2, 3]
        }

        self.tensor = tensor ?? graph.reductionSum(with: mask.tensor,
                                                   axes: hwAxes,
                                                   name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

class MaskSumSqrtS14M01Layer {
    let tensor: MPSGraphTensor

    init(graph: MPSGraph,
         tensor: MPSGraphTensor?,
         maskSum: MaskSumLayer,
         useFP16: Bool) {
        if let knownTensor = tensor {
            self.tensor = knownTensor
        } else {
            let dataType = MPSDataType.float32
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
        }

        assert(self.tensor.shape?.count == 4)
    }
}

class MaskSumSqrtS14M01SquareS01Layer {
    let tensor: MPSGraphTensor

    init(graph: MPSGraph,
         tensor: MPSGraphTensor?,
         maskSumSqrtS14M01: MaskSumSqrtS14M01Layer,
         useFP16: Bool) {
        if let knownTensor = tensor {
            self.tensor = knownTensor
        } else {
            let dataType = MPSDataType.float32

            let squared = graph.square(with: maskSumSqrtS14M01.tensor, name: nil)

            let zeroPointone = graph.constant(0.1,
                                              shape: squared.shape!,
                                              dataType: dataType)

            self.tensor = graph.subtraction(squared,
                                            zeroPointone,
                                            name: nil)
        }

        assert(self.tensor.shape?.count == 4)
    }
}

@objc
class SWConvLayerDesc: NSObject {
    let convYSize: NSNumber
    let convXSize: NSNumber
    let inChannels: NSNumber
    let outChannels: NSNumber
    let dilationY: Int
    let dilationX: Int
    let weights: UnsafeMutablePointer<Float32>

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

@objc
class ConvLayer: NSObject {
    let graph: MPSGraph
    let source: SourceLayer
    let resultTensor: MPSGraphTensor

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

        let layer = ConvLayer(graph: MPSGraph(),
                              sourceTensor: nil,
                              descriptor: descriptor,
                              batchSize: batchSize,
                              nnXLen: nnXLen,
                              nnYLen: nnYLen,
                              useFP16: useFP16,
                              useNHWC: useNHWC)

        layer.apply(device: device, input: input, output: output)
    }

    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor?,
         descriptor: SWConvLayerDesc,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        let dataType = MPSDataType.float32

        let weightsShape = [descriptor.outChannels,
                            descriptor.inChannels,
                            descriptor.convYSize,
                            descriptor.convXSize]

        source = SourceLayer(graph: graph,
                             tensor: sourceTensor,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             numChannels: descriptor.inChannels,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let convDescriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                                               strideInY: 1,
                                                               dilationRateInX: descriptor.dilationX,
                                                               dilationRateInY: descriptor.dilationY,
                                                               groups: 1,
                                                               paddingStyle: .TF_SAME,
                                                               dataLayout: source.layout,
                                                               weightsLayout: .OIHW)!

        self.graph = graph

        let weightsData = Data(bytes: descriptor.weights,
                               count: weightsShape.asShapeCount(of: dataType))

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: dataType)

        resultTensor = graph.convolution2D(source.tensor,
                                           weights: weightsTensor,
                                           descriptor: convDescriptor,
                                           name: nil)
    }

    func apply(device: MPSGraphDevice,
               input: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        let sourceTensorData = MPSGraphTensorData(device: device,
                                                  tensor: source.tensor)!

        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        fetch[resultTensor]?.mpsndarray().readBytes(output, strideBytes: nil)
    }
}

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

@objc
class BatchNormLayer: NSObject {
    let graph: MPSGraph
    let source: SourceLayer
    let mask: MaskLayer
    let resultTensor: MPSGraphTensor

    @objc
    class func test(descriptor: SWBatchNormLayerDesc,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    batchSize: NSNumber,
                    useFP16: Bool,
                    useNHWC: Bool,
                    input: UnsafeMutablePointer<Float32>,
                    mask: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let layer = BatchNormLayer(graph: MPSGraph(),
                                   sourceTensor: nil,
                                   maskTensor: nil,
                                   descriptor: descriptor,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        layer.apply(device: device,
                    input: input,
                    maskPointer: mask,
                    output: output)
    }

    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor?,
         maskTensor: MPSGraphTensor?,
         descriptor: SWBatchNormLayerDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        let meanShape: [NSNumber]
        let dataType = MPSDataType.float32

        if useNHWC {
            meanShape = [1,
                         1,
                         1,
                         descriptor.numChannels]
        } else {
            meanShape = [1,
                         descriptor.numChannels,
                         1,
                         1]
        }

        self.graph = graph

        source = SourceLayer(graph: graph,
                             tensor: sourceTensor,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             numChannels: descriptor.numChannels,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        mask = MaskLayer(graph: graph,
                         tensor: maskTensor,
                         batchSize: batchSize,
                         nnXLen: nnXLen,
                         nnYLen: nnYLen,
                         useFP16: useFP16,
                         useNHWC: useNHWC)

        let meanCount = meanShape.asShapeCount(of: dataType)

        let meanData = Data(bytes: descriptor.mean,
                            count: meanCount)

        let meanTensor = graph.constant(meanData,
                                        shape: meanShape,
                                        dataType: dataType)

        let varianceData = Data(bytes: descriptor.variance,
                                count: meanCount)

        let varianceTensor = graph.constant(varianceData,
                                            shape: meanShape,
                                            dataType: dataType)

        let scaleData = Data(bytes: descriptor.scale,
                             count: meanCount)

        let scaleTensor = graph.constant(scaleData,
                                         shape: meanShape,
                                         dataType: dataType)

        let biasData = Data(bytes: descriptor.bias,
                            count: meanCount)

        let biasTensor = graph.constant(biasData,
                                        shape: meanShape,
                                        dataType: dataType)

        let normalized = graph.normalize(source.tensor,
                                         mean: meanTensor,
                                         variance: varianceTensor,
                                         gamma: scaleTensor,
                                         beta: biasTensor,
                                         epsilon: descriptor.epsilon,
                                         name: nil)

        resultTensor = graph.multiplication(normalized,
                                            mask.tensor,
                                            name: nil)
    }

    func apply(device: MPSGraphDevice,
               input: UnsafeMutablePointer<Float32>,
               maskPointer: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        let sourceTensorData = MPSGraphTensorData(device: device,
                                                  tensor: source.tensor)!

        let maskTensorData = MPSGraphTensorData(device: device,
                                                tensor: mask.tensor)!

        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)
        maskTensorData.mpsndarray().writeBytes(maskPointer, strideBytes: nil)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        fetch[resultTensor]?.mpsndarray().readBytes(output, strideBytes: nil)
    }
}

@objc
class SWResidualBlockDesc: NSObject {
    let preBN: SWBatchNormLayerDesc
    let preActivation: NSString?
    let regularConv: SWConvLayerDesc
    let midBN: SWBatchNormLayerDesc
    let midActivation: NSString?
    let finalConv: SWConvLayerDesc

    @objc
    init(preBN: SWBatchNormLayerDesc,
         preActivation: NSString?,
         regularConv: SWConvLayerDesc,
         midBN: SWBatchNormLayerDesc,
         midActivation: NSString?,
         finalConv: SWConvLayerDesc) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

@objc
class ResidualBlock: NSObject {
    let graph: MPSGraph
    let source: SourceLayer
    let mask: MaskLayer
    let resultTensor: MPSGraphTensor

    @objc
    class func test(descriptor: SWResidualBlockDesc,
                    batchSize: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    useFP16: Bool,
                    useNHWC: Bool,
                    input: UnsafeMutablePointer<Float32>,
                    mask: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let layer = ResidualBlock(graph: MPSGraph(),
                                  sourceTensor: nil,
                                  maskTensor: nil,
                                  descriptor: descriptor,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        layer.apply(device: device,
                    input: input,
                    maskPointer: mask,
                    output: output)
    }

    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor?,
         maskTensor: MPSGraphTensor?,
         descriptor: SWResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        self.graph = graph

        source = SourceLayer(graph: graph,
                             tensor: sourceTensor,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             numChannels: descriptor.preBN.numChannels,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        mask = MaskLayer(graph: graph,
                         tensor: maskTensor,
                         batchSize: batchSize,
                         nnXLen: nnXLen,
                         nnYLen: nnYLen,
                         useFP16: useFP16,
                         useNHWC: useNHWC)

        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: source.tensor,
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

        let midBN = BatchNormLayer(graph: graph,
                                   sourceTensor: regularConv.resultTensor,
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

        resultTensor = graph.addition(source.tensor,
                                      finalConv.resultTensor,
                                      name: nil)
    }

    func apply(device: MPSGraphDevice,
               input: UnsafeMutablePointer<Float32>,
               maskPointer: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        let sourceTensorData = MPSGraphTensorData(device: device,
                                                  tensor: source.tensor)!

        let maskTensorData = MPSGraphTensorData(device: device,
                                                tensor: mask.tensor)!

        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)
        maskTensorData.mpsndarray().writeBytes(maskPointer, strideBytes: nil)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        fetch[resultTensor]?.mpsndarray().readBytes(output, strideBytes: nil)
    }
}

class GlobalPoolingLayer {
    let resultTensor: MPSGraphTensor

    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        let hwAxes: [NSNumber]
        let channelAxis: Int

        if useNHWC {
            hwAxes = [1, 2]
            channelAxis = 3
        } else {
            hwAxes = [2, 3]
            channelAxis = 1
        }

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
    }
}

class GlobalPoolingValueLayer {
    let resultTensor: MPSGraphTensor

    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        let hwAxes: [NSNumber]
        let channelAxis: Int

        if useNHWC {
            hwAxes = [1, 2]
            channelAxis = 3
        } else {
            hwAxes = [2, 3]
            channelAxis = 1
        }

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
    }
}

@objc
class SWMatMulLayerDesc: NSObject {
    let inChannels: NSNumber
    let outChannels: NSNumber
    let weights: UnsafeMutablePointer<Float32>

    @objc
    init(inChannels: NSNumber,
         outChannels: NSNumber,
         weights: UnsafeMutablePointer<Float32>) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
}

class MatMulLayer {
    let resultTensor: MPSGraphTensor

    init(graph: MPSGraph,
         descriptor: SWMatMulLayerDesc,
         sourceTensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        let dataType = MPSDataType.float32

        let weightsShape = [descriptor.inChannels,
                            descriptor.outChannels]

        let weightsCount = weightsShape.asShapeCount(of: dataType)
        let weightsData = Data(bytes: descriptor.weights, count: weightsCount)

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: .float32)

        let shape = [-1, descriptor.inChannels]

        let reshapedSource = graph.reshape(sourceTensor,
                                           shape: shape,
                                           name: nil)

        resultTensor = graph.matrixMultiplication(primary: reshapedSource,
                                                  secondary: weightsTensor,
                                                  name: nil)
    }
}

@objc
class SWMatBiasLayerDesc: NSObject {
    let numChannels: NSNumber
    let weights: UnsafeMutablePointer<Float32>

    @objc
    init(numChannels: NSNumber,
         weights: UnsafeMutablePointer<Float32>) {
        self.numChannels = numChannels
        self.weights = weights
    }
}

class MatBiasLayer {
    let resultTensor: MPSGraphTensor

    init(graph: MPSGraph,
         descriptor: SWMatBiasLayerDesc,
         sourceTensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        let dataType = MPSDataType.float32
        let weightsShape = [1, descriptor.numChannels]
        let weightsCount = weightsShape.asShapeCount(of: dataType)
        let weightsData = Data(bytes: descriptor.weights, count: weightsCount)

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: .float32)

        let shape = [-1, descriptor.numChannels]

        let reshapedSource = graph.reshape(sourceTensor,
                                           shape: shape,
                                           name: nil)

        resultTensor = graph.addition(reshapedSource,
                                      weightsTensor,
                                      name: nil)
    }
}

class AddNCBiasLayer {
    let resultTensor: MPSGraphTensor

    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         biasTensor: MPSGraphTensor,
         batchSize: NSNumber,
         numChannels: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        let shape: [NSNumber]

        if useNHWC {
            shape = [batchSize, 1, 1, numChannels]
        } else {
            shape = [batchSize, numChannels, 1, 1]
        }

        let reshaped = graph.reshape(biasTensor, shape: shape, name: nil)
        resultTensor = graph.addition(sourceTensor, reshaped, name: nil)
    }
}

@objc
class SWGlobalPoolingResidualBlockDesc: NSObject {
    let preBN: SWBatchNormLayerDesc
    let preActivation: NSString?
    let regularConv: SWConvLayerDesc
    let gpoolConv: SWConvLayerDesc
    let gpoolBN: SWBatchNormLayerDesc
    let gpoolActivation: NSString?
    let gpoolToBiasMul: SWMatMulLayerDesc
    let midBN: SWBatchNormLayerDesc
    let midActivation: NSString?
    let finalConv: SWConvLayerDesc

    @objc
    init(preBN: SWBatchNormLayerDesc,
         preActivation: NSString?,
         regularConv: SWConvLayerDesc,
         gpoolConv: SWConvLayerDesc,
         gpoolBN: SWBatchNormLayerDesc,
         gpoolActivation: NSString?,
         gpoolToBiasMul: SWMatMulLayerDesc,
         midBN: SWBatchNormLayerDesc,
         midActivation: NSString?,
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

@objc
class GlobalPoolingResidualBlock: NSObject {
    let graph: MPSGraph
    let source: SourceLayer
    let mask: MaskLayer
    let resultTensor: MPSGraphTensor

    @objc
    class func test(descriptor: SWGlobalPoolingResidualBlockDesc,
                    batchSize: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    useFP16: Bool,
                    useNHWC: Bool,
                    input: UnsafeMutablePointer<Float32>,
                    mask: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let layer = GlobalPoolingResidualBlock(graph: MPSGraph(),
                                               sourceTensor: nil,
                                               maskTensor: nil,
                                               maskSumTensor: nil,
                                               maskSumSqrtS14M01Tensor: nil,
                                               descriptor: descriptor,
                                               nnXLen: nnXLen,
                                               nnYLen: nnYLen,
                                               batchSize: batchSize,
                                               useFP16: useFP16,
                                               useNHWC: useNHWC)

        layer.apply(device: device,
                    input: input,
                    maskPointer: mask,
                    output: output)
    }

    init(graph: MPSGraph,
         sourceTensor: MPSGraphTensor?,
         maskTensor: MPSGraphTensor?,
         maskSumTensor: MPSGraphTensor?,
         maskSumSqrtS14M01Tensor: MPSGraphTensor?,
         descriptor: SWGlobalPoolingResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        self.graph = graph

        source = SourceLayer(graph: graph,
                             tensor: sourceTensor,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             numChannels: descriptor.preBN.numChannels,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        mask = MaskLayer(graph: graph,
                         tensor: maskTensor,
                         batchSize: batchSize,
                         nnXLen: nnXLen,
                         nnYLen: nnYLen,
                         useFP16: useFP16,
                         useNHWC: useNHWC)

        let maskSum = MaskSumLayer(graph: graph,
                                   tensor: maskSumTensor,
                                   mask: mask,
                                   useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                             tensor: maskSumSqrtS14M01Tensor,
                                                             maskSum: maskSum,
                                                             useFP16: useFP16)

        let preBN = BatchNormLayer(graph: graph,
                                   sourceTensor: source.tensor,
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

        let gpoolToBiasMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToBiasMul,
                                         sourceTensor: gpoolConcat.resultTensor,
                                         useFP16: useFP16,
                                         useNHWC: useNHWC)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: regularConv.resultTensor,
                                   biasTensor: gpoolToBiasMul.resultTensor,
                                   batchSize: batchSize,
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

        resultTensor = graph.addition(source.tensor,
                                      finalConv.resultTensor,
                                      name: nil)
    }

    func apply(device: MPSGraphDevice,
               input: UnsafeMutablePointer<Float32>,
               maskPointer: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        let sourceTensorData = MPSGraphTensorData(device: device,
                                                  tensor: source.tensor)!

        let maskTensorData = MPSGraphTensorData(device: device,
                                                tensor: mask.tensor)!

        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)
        maskTensorData.mpsndarray().writeBytes(maskPointer, strideBytes: nil)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        fetch[resultTensor]?.mpsndarray().readBytes(output, strideBytes: nil)

#if false // TODO: clean up
        // Debugging
        print("sourceTensor: \(sourceTensor.shape!)")
        input.printAsFloat(24)
        print("maskTensor: \(maskTensor.shape!)")
        mask.printAsFloat(24)
        print("preReLU: \(preReLU.shape!)")
        fetch[preReLU]?.mpsndarray().dumpFloats(name: "preReLU",
                                                length: preReLU.shape!.product().intValue)

        print("gpoolConvTensor: \(gpoolConvTensor.shape!)")
        let gpoolConvLength = gpoolConvTensor.shape!.product().intValue
        fetch[gpoolConvTensor]?.mpsndarray().dumpFloats(name: "gpoolConvTensor",
                                                        length: gpoolConvLength)

        // 2 0 0 0
        // 3 4 0 0
        // 0 5 0 0
        print("gpoolReLU: \(gpoolReLU.shape!)")
        let gpoolReLULength = gpoolReLU.shape!.product().intValue
        fetch[gpoolReLU]?.mpsndarray().dumpFloats(name: "gpoolReLU",
                                                  length: gpoolReLULength)

        // [2, 1, 1, 6]
        // 1.55     0.33
        // 0.11     0.5
        // -1.71111 -0.385017
        // -0.122222 -0.577526
        //       5         1
        //       1         3
        print("gpoolConcatTensor: \(gpoolConcatTensor.shape!)")
        let gpoolConcatLength = gpoolConcatTensor.shape!.product().intValue
        fetch[gpoolConcatTensor]?.mpsndarray().dumpFloats(name: "gpoolConcatTensor",
                                                          length: gpoolConcatLength)
        // Expect
        // 33 16.6742
        print("gpoolToBiasMulTensor: \(gpoolToBiasMulTensor.shape!)")
        let gpoolToBiasMulLength = gpoolToBiasMulTensor.shape!.product().intValue
        fetch[gpoolToBiasMulTensor]?.mpsndarray().dumpFloats(name: "gpoolToBiasMulTensor",
                                                             length: gpoolToBiasMulLength)
#endif
    }
}

@objc
enum BlockKind: Int {
    case ordinary
    case dilated
    case globalPooling
}

@objc
class BlockDescriptor: NSObject {
    let kind: BlockKind
    let ordinary: SWResidualBlockDesc?
    let globalPooling: SWGlobalPoolingResidualBlockDesc?

    @objc
    init(kind: BlockKind,
         ordinary: SWResidualBlockDesc?,
         globalPooling: SWGlobalPoolingResidualBlockDesc?) {
        self.kind = kind
        self.ordinary = ordinary
        self.globalPooling = globalPooling
    }
}

@objc
class SWTrunkDesc: NSObject {
    let version: Int
    let numBlocks: Int
    let trunkNumChannels: NSNumber
    let midNumChannels: NSNumber
    let regularNumChannels: NSNumber
    let dilatedNumChannels: NSNumber
    let gpoolNumChannels: NSNumber
    let initialConv: SWConvLayerDesc
    let initialMatMul: SWMatMulLayerDesc
    let blocks: [BlockDescriptor]
    let trunkTipBN: SWBatchNormLayerDesc
    let trunkTipActivation: String

    @objc
    init(version: Int,
         numBlocks: Int,
         trunkNumChannels: NSNumber,
         midNumChannels: NSNumber,
         regularNumChannels: NSNumber,
         dilatedNumChannels: NSNumber,
         gpoolNumChannels: NSNumber,
         initialConv: SWConvLayerDesc,
         initialMatMul: SWMatMulLayerDesc,
         blocks: [BlockDescriptor],
         trunkTipBN: SWBatchNormLayerDesc,
         trunkTipActivation: String) {
        self.version = version
        self.numBlocks = numBlocks
        self.trunkNumChannels = trunkNumChannels
        self.midNumChannels = midNumChannels
        self.regularNumChannels = regularNumChannels
        self.dilatedNumChannels = dilatedNumChannels
        self.gpoolNumChannels = gpoolNumChannels
        self.initialConv = initialConv
        self.initialMatMul = initialMatMul
        self.blocks = blocks
        self.trunkTipBN = trunkTipBN
        self.trunkTipActivation = trunkTipActivation
    }
}

class Trunk {
    let graph: MPSGraph
    let input: SourceLayer
    let inputGlobal: InputGlobalLayer
    let mask: MaskLayer
    let resultTensor: MPSGraphTensor

    init(graph: MPSGraph,
         descriptor: SWTrunkDesc,
         inputTensor: MPSGraphTensor?,
         inputGlobalTensor: MPSGraphTensor?,
         maskTensor: MPSGraphTensor?,
         maskSumTensor: MPSGraphTensor?,
         maskSumSqrtS14M01Tensor: MPSGraphTensor?,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         numSpatialFeatures: NSNumber,
         numGlobalFeatures: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1
        
        self.graph = graph

        input = SourceLayer(graph: graph,
                            tensor: inputTensor,
                            batchSize: batchSize,
                            nnXLen: nnXLen,
                            nnYLen: nnYLen,
                            numChannels: numSpatialFeatures,
                            useFP16: useFP16,
                            useNHWC: useNHWC)

        inputGlobal = InputGlobalLayer(graph: graph,
                                       tensor: inputGlobalTensor,
                                       batchSize: batchSize,
                                       numGlobalFeatures: numGlobalFeatures,
                                       useFP16: useFP16)

        mask = MaskLayer(graph: graph,
                         tensor: maskTensor,
                         batchSize: batchSize,
                         nnXLen: nnXLen,
                         nnYLen: nnYLen,
                         useFP16: useFP16,
                         useNHWC: useNHWC)

        let maskSum = MaskSumLayer(graph: graph,
                                   tensor: maskSumTensor,
                                   mask: mask,
                                   useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       tensor: maskSumSqrtS14M01Tensor,
                                                       maskSum: maskSum,
                                                       useFP16: useFP16)

        let initialConv = ConvLayer(graph: graph,
                                    sourceTensor: input.tensor,
                                    descriptor: descriptor.initialConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    useFP16: useFP16,
                                    useNHWC: useNHWC)

        let initialMatMul = MatMulLayer(graph: graph,
                                        descriptor: descriptor.initialMatMul,
                                        sourceTensor: inputGlobal.tensor,
                                        useFP16: useFP16,
                                        useNHWC: useNHWC)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: initialConv.resultTensor,
                                   biasTensor: initialMatMul.resultTensor,
                                   batchSize: batchSize,
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
                                             maskTensor: mask.tensor,
                                             descriptor: block.ordinary!,
                                             nnXLen: nnXLen,
                                             nnYLen: nnYLen,
                                             batchSize: batchSize,
                                             useFP16: useFP16,
                                             useNHWC: useNHWC)

                blockInput = ordinary.resultTensor
            default:
                let globalPooling = GlobalPoolingResidualBlock(graph: graph,
                                                               sourceTensor: blockInput,
                                                               maskTensor: mask.tensor,
                                                               maskSumTensor: maskSum.tensor,
                                                               maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
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
                                        maskTensor: mask.tensor,
                                        descriptor: descriptor.trunkTipBN,
                                        nnXLen: nnXLen,
                                        nnYLen: nnYLen,
                                        batchSize: batchSize,
                                        useFP16: useFP16,
                                        useNHWC: useNHWC)

        let trunkTipReLU = graph.reLU(with: trunkTipBN.resultTensor, name: nil)

        resultTensor = trunkTipReLU
    }
}

@objc
class SWPolicyHeadDesc: NSObject {
    let version: Int
    let p1Conv: SWConvLayerDesc
    let g1Conv: SWConvLayerDesc
    let g1BN: SWBatchNormLayerDesc
    let gpoolToBiasMul: SWMatMulLayerDesc
    let p1BN: SWBatchNormLayerDesc
    let p2Conv: SWConvLayerDesc
    let gpoolToPassMul: SWMatMulLayerDesc

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

class PolicyHead {
    let policyTensor: MPSGraphTensor
    let policyPassTensor: MPSGraphTensor

    init(graph: MPSGraph,
         descriptor: SWPolicyHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor?,
         maskSumTensor: MPSGraphTensor?,
         maskSumSqrtS14M01Tensor: MPSGraphTensor?,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {

        let mask = MaskLayer(graph: graph,
                             tensor: maskTensor,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let maskSum = MaskSumLayer(graph: graph,
                                   tensor: maskSumTensor,
                                   mask: mask,
                                   useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       tensor: maskSumSqrtS14M01Tensor,
                                                       maskSum: maskSum,
                                                       useFP16: useFP16)

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
                                  maskTensor: mask.tensor,
                                  descriptor: descriptor.g1BN,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let g1ReLU = graph.reLU(with: g1BN.resultTensor, name: nil)

        let g1Concat = GlobalPoolingLayer(graph: graph,
                                          sourceTensor: g1ReLU,
                                          maskSumTensor: maskSum.tensor,
                                          maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                          useFP16: useFP16,
                                          useNHWC: useNHWC)

        let gpoolToBiasMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToBiasMul,
                                         sourceTensor: g1Concat.resultTensor,
                                         useFP16: useFP16,
                                         useNHWC: useNHWC)

        let added = AddNCBiasLayer(graph: graph,
                                   sourceTensor: p1Conv.resultTensor,
                                   biasTensor: gpoolToBiasMul.resultTensor,
                                   batchSize: batchSize,
                                   numChannels: descriptor.gpoolToBiasMul.outChannels,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let p1BN = BatchNormLayer(graph: graph,
                                  sourceTensor: added.resultTensor,
                                  maskTensor: mask.tensor,
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

        let gpoolToPassMul = MatMulLayer(graph: graph,
                                         descriptor: descriptor.gpoolToPassMul,
                                         sourceTensor: g1Concat.resultTensor,
                                         useFP16: useFP16,
                                         useNHWC: useNHWC)

        policyTensor = p2Conv.resultTensor
        policyPassTensor = gpoolToPassMul.resultTensor
    }
}

@objc
class SWValueHeadDesc: NSObject {
    let version: Int
    let v1Conv: SWConvLayerDesc
    let v1BN: SWBatchNormLayerDesc
    let v2Mul: SWMatMulLayerDesc
    let v2Bias: SWMatBiasLayerDesc
    let v3Mul: SWMatMulLayerDesc
    let v3Bias: SWMatBiasLayerDesc
    let sv3Mul: SWMatMulLayerDesc
    let sv3Bias: SWMatBiasLayerDesc
    let vOwnershipConv: SWConvLayerDesc

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

class ValueHead {
    let valueTensor: MPSGraphTensor
    let scoreValueTensor: MPSGraphTensor
    let ownershipTensor: MPSGraphTensor

    init(graph: MPSGraph,
         descriptor: SWValueHeadDesc,
         sourceTensor: MPSGraphTensor,
         maskTensor: MPSGraphTensor?,
         maskSumTensor: MPSGraphTensor?,
         maskSumSqrtS14M01Tensor: MPSGraphTensor?,
         maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor?,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {

        let mask = MaskLayer(graph: graph,
                             tensor: maskTensor,
                             batchSize: batchSize,
                             nnXLen: nnXLen,
                             nnYLen: nnYLen,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let maskSum = MaskSumLayer(graph: graph,
                                   tensor: maskSumTensor,
                                   mask: mask,
                                   useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                             tensor: maskSumSqrtS14M01Tensor,
                                                             maskSum: maskSum,
                                                             useFP16: useFP16)

        let maskSumSqrtS14M01SquareS01 =
        MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                        tensor: maskSumSqrtS14M01SquareS01Tensor,
                                        maskSumSqrtS14M01: maskSumSqrtS14M01,
                                        useFP16: useFP16)

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
                                  maskTensor: mask.tensor,
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
                                maskSumTensor: maskSum.tensor,
                                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
                                useFP16: useFP16,
                                useNHWC: useNHWC)

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
    }
}

@objc
class SWModelDesc : NSObject {
    let version: Int
    let numInputChannels: NSNumber
    let numInputGlobalChannels: NSNumber
    let numValueChannels: NSNumber
    let numScoreValueChannels: NSNumber
    let numOwnershipChannels: NSNumber
    let trunk: SWTrunkDesc
    let policyHead: SWPolicyHeadDesc
    let valueHead: SWValueHeadDesc

    @objc
    init(version: Int,
         numInputChannels: NSNumber,
         numInputGlobalChannels: NSNumber,
         numValueChannels: NSNumber,
         numScoreValueChannels: NSNumber,
         numOwnershipChannels: NSNumber,
         trunk: SWTrunkDesc,
         policyHead: SWPolicyHeadDesc,
         valueHead: SWValueHeadDesc) {
        self.version = version
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

class Model {
    let graph: MPSGraph
    let version: Int
    let numInputChannels: NSNumber
    let numInputGlobalChannels: NSNumber
    let numValueChannels: NSNumber
    let numScoreValueChannels: NSNumber
    let numOwnershipChannels: NSNumber
    let mask: MaskLayer
    let trunk: Trunk
    let policyHead: PolicyHead
    let valueHead: ValueHead

    init(graph: MPSGraph,
         descriptor: SWModelDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        self.graph = graph
        self.version = descriptor.version
        self.numInputChannels = descriptor.numInputChannels
        self.numInputGlobalChannels = descriptor.numInputGlobalChannels
        self.numValueChannels = descriptor.numValueChannels
        self.numScoreValueChannels = descriptor.numScoreValueChannels
        self.numOwnershipChannels = descriptor.numOwnershipChannels

        mask = MaskLayer(graph: graph,
                         tensor: nil,
                         batchSize: batchSize,
                         nnXLen: nnXLen,
                         nnYLen: nnYLen,
                         useFP16: useFP16,
                         useNHWC: useNHWC)

        let maskSum = MaskSumLayer(graph: graph,
                                   tensor: nil,
                                   mask: mask,
                                   useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       tensor: nil,
                                                       maskSum: maskSum,
                                                       useFP16: useFP16)

        trunk = Trunk(graph: graph,
                      descriptor: descriptor.trunk,
                      inputTensor: nil,
                      inputGlobalTensor: nil,
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
                              maskSumSqrtS14M01SquareS01Tensor: nil,
                              nnXLen: nnXLen,
                              nnYLen: nnYLen,
                              batchSize: batchSize,
                              useFP16: useFP16,
                              useNHWC: useNHWC)
    }

    func apply(device: MPSGraphDevice,
               input: UnsafeMutablePointer<Float32>,
               inputGlobal: UnsafeMutablePointer<Float32>,
               maskPointer: UnsafeMutablePointer<Float32>,
               policy: UnsafeMutablePointer<Float32>,
               policyPass: UnsafeMutablePointer<Float32>,
               value: UnsafeMutablePointer<Float32>,
               scoreValue: UnsafeMutablePointer<Float32>,
               ownership: UnsafeMutablePointer<Float32>) {
        let inputData = MPSGraphTensorData(device: device, tensor: trunk.input.tensor)!

        let inputGlobalData = MPSGraphTensorData(device: device,
                                                 tensor: trunk.inputGlobal.tensor)!

        let maskData = MPSGraphTensorData(device: device, tensor: mask.tensor)!

        inputData.mpsndarray().writeBytes(input, strideBytes: nil)
        inputGlobalData.mpsndarray().writeBytes(inputGlobal, strideBytes: nil)
        maskData.mpsndarray().writeBytes(maskPointer, strideBytes: nil)

        let feeds = [trunk.input.tensor: inputData,
                     trunk.inputGlobal.tensor: inputGlobalData,
                     mask.tensor: maskData]

        let targetTensors = [policyHead.policyTensor,
                             policyHead.policyPassTensor,
                             valueHead.valueTensor,
                             valueHead.scoreValueTensor,
                             valueHead.ownershipTensor]

        let fetch = graph.run(feeds: feeds,
                              targetTensors: targetTensors,
                              targetOperations: nil)

        fetch[policyHead.policyTensor]?.mpsndarray().readBytes(policy,
                                                               strideBytes: nil)

        fetch[policyHead.policyPassTensor]?.mpsndarray().readBytes(policyPass,
                                                                   strideBytes: nil)

        fetch[valueHead.valueTensor]?.mpsndarray().readBytes(value,
                                                              strideBytes: nil)

        fetch[valueHead.scoreValueTensor]?.mpsndarray().readBytes(scoreValue,
                                                                   strideBytes: nil)

        fetch[valueHead.ownershipTensor]?.mpsndarray().readBytes(ownership,
                                                                  strideBytes: nil)
    }
}

@objc
enum SWEnable: Int {
    case False
    case True
    case Auto
}

@objc
class ComputeContext: NSObject {
    static var instance = ComputeContext()
    let nnXLen: NSNumber
    let nnYLen: NSNumber
    let useFP16Mode: SWEnable
    let useNHWCMode: SWEnable

    @objc
    class func createInstance(nnXLen: NSNumber,
                              nnYLen: NSNumber,
                              useFP16Mode: SWEnable,
                              useNHWCMode: SWEnable) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        instance = ComputeContext(nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  useFP16Mode: useFP16Mode,
                                  useNHWCMode: useNHWCMode)
    }

    @objc
    class func getInstance() -> ComputeContext {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        return instance
    }

    private convenience override init() {
        self.init(nnXLen: 19, nnYLen: 19, useFP16Mode: .False, useNHWCMode: .False)
    }

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

@objc
class ComputeHandle: NSObject {
    static var handles: [Int: ComputeHandle] = [:]
    let model: Model

    @objc
    class func createInstance(at gpuIdxForThisThread: Int,
                              descriptor: SWModelDesc,
                              batchSize: NSNumber,
                              serverThreadIdx: Int) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        assert(handles[gpuIdxForThisThread] == nil)

        handles[gpuIdxForThisThread] = ComputeHandle(descriptor: descriptor,
                                                     batchSize: batchSize,
                                                     gpuIdxForThisThread: gpuIdxForThisThread,
                                                     serverThreadIdx: serverThreadIdx)
    }

    @objc
    class func getInstance(at gpuIdxForThisThread: Int) -> ComputeHandle {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        return handles[gpuIdxForThisThread]!
    }

    private init(descriptor: SWModelDesc,
                 batchSize: NSNumber,
                 gpuIdxForThisThread: Int,
                 serverThreadIdx: Int) {

        let context = ComputeContext.getInstance()
        let useFP16: Bool
        let useNHWC: Bool

        NSLog("ComputeHandle:init(gpuIdxForThisThread=\(gpuIdxForThisThread))")

        // TODO: print device and model information here

        switch context.useFP16Mode {
        case .False: useFP16 = false
        default: useFP16 = true
        }

        switch context.useNHWCMode {
        case .False: useNHWC = false
        default: useNHWC = true
        }

        model = Model(graph: MPSGraph(),
                      descriptor: descriptor,
                      nnXLen: context.nnXLen,
                      nnYLen: context.nnYLen,
                      batchSize: batchSize,
                      useFP16: useFP16,
                      useNHWC: useNHWC)
    }
}

@objc
class KataGoGraph: NSObject {
    static let graphs = NSMutableDictionary(capacity: 1)
    let nnXLen: NSNumber
    let nnYLen: NSNumber
    let numInputChannels: NSNumber
    let numInputGlobalChannels: NSNumber
    let device: MTLDevice
    let graph: MPSGraph
    let inputTensor: MPSGraphTensor
    let inputGlobalTensor: MPSGraphTensor
    let symmetriesTensor: MPSGraphTensor
    let includeHistoryTensor: MPSGraphTensor
    let policyOutputTensor: MPSGraphTensor
    let inputTensorData: MPSGraphTensorData
    let inputGlobalTensorData: MPSGraphTensorData

    @objc
    class func getGraph(gpuIndex: NSNumber) -> KataGoGraph {
        return graphs[gpuIndex]! as! KataGoGraph
    }

    @objc
    class func initGraph(gpuIndex: NSNumber,
                         nnXLen: NSNumber,
                         nnYLen: NSNumber,
                         version: NSNumber,
                         numInputChannels: NSNumber,
                         numInputGlobalChannels: NSNumber,
                         numValueChannels: NSNumber,
                         numScoreValueChannels: NSNumber,
                         numOwnershipChannels: NSNumber) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }

        if (graphs[gpuIndex] == nil) {
            graphs[gpuIndex] = KataGoGraph(gpuIndex: gpuIndex,
                                           nnXLen: nnXLen,
                                           nnYLen: nnYLen,
                                           version: version,
                                           numInputChannels: numInputChannels,
                                           numInputGlobalChannels: numInputGlobalChannels,
                                           numValueChannels: numValueChannels,
                                           numScoreValueChannels: numScoreValueChannels,
                                           numOwnershipChannels: numOwnershipChannels)
        }
    }

    private init(gpuIndex: NSNumber,
                 nnXLen: NSNumber,
                 nnYLen: NSNumber,
                 version: NSNumber,
                 numInputChannels: NSNumber,
                 numInputGlobalChannels: NSNumber,
                 numValueChannels: NSNumber,
                 numScoreValueChannels: NSNumber,
                 numOwnershipChannels: NSNumber) {
        // FIXME: Create device with GPU index
        device = MTLCreateSystemDefaultDevice()!
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.numInputChannels = numInputChannels
        self.numInputGlobalChannels = numInputGlobalChannels
        graph = MPSGraph()

        inputTensor = graph.placeholder(shape: [nnXLen,
                                                nnYLen,
                                                numInputChannels],
                                        name: "binInputs")

        let inputArrayDesc = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                  shape: inputTensor.shape!)

        let inputArray = MPSNDArray(device: device, descriptor: inputArrayDesc)

        inputTensorData = MPSGraphTensorData(inputArray)

        inputGlobalTensor = graph.placeholder(shape: [numInputGlobalChannels],
                                              name: "globalInputs")

        let inputGlobalArrayDesc = MPSNDArrayDescriptor(dataType: inputGlobalTensor.dataType,
                                                        shape: inputGlobalTensor.shape!)

        let inputGlobalArray = MPSNDArray(device: device, descriptor: inputGlobalArrayDesc)

        inputGlobalTensorData = MPSGraphTensorData(inputGlobalArray)

        symmetriesTensor = graph.constant(0.0, shape: [3], dataType: .float32)
        includeHistoryTensor = graph.constant(1.0, shape: [5], dataType: .float32)

        // FIXME: The followings are test code, to be removed
        let numInputElements = NSNumber(integerLiteral: nnXLen.intValue * nnYLen.intValue * numInputChannels.intValue)

        let reshaped = graph.reshape(inputTensor,
                                     shape: [1, numInputElements],
                                     name: nil)

        let weightTensor = graph.constant(1.0,
                                          shape: [numInputElements, 1],
                                          dataType: .float32)

        policyOutputTensor = graph.matrixMultiplication(primary: reshaped,
                                                        secondary: weightTensor,
                                                        name: nil)
    }

    @objc
    func run(userInputBuffer: UnsafeMutablePointer<Float32>,
             userInputGlobalBuffer: UnsafeMutablePointer<Float32>,
             policyOutput: UnsafeMutablePointer<Float32>,
             valueOutput: UnsafeMutablePointer<Float32>,
             ownershipOutput: UnsafeMutablePointer<Float32>,
             miscValuesOutput: UnsafeMutablePointer<Float32>,
             moreMiscValuesOutput: UnsafeMutablePointer<Float32>) {
        let feeds = [inputTensor: inputTensorData,
               inputGlobalTensor: inputGlobalTensorData]

        inputTensorData.mpsndarray().writeBytes(userInputBuffer, strideBytes: nil)
        inputGlobalTensorData.mpsndarray().writeBytes(userInputGlobalBuffer, strideBytes: nil)

        let fetch = graph.run(feeds: feeds,
                              targetTensors: [policyOutputTensor],
                              targetOperations: nil)

        fetch[policyOutputTensor]!.mpsndarray().readBytes(policyOutput, strideBytes: nil)

        // TODO: Debugging, to be removed
        policyOutput.printAsFloat(5)
    }
}
