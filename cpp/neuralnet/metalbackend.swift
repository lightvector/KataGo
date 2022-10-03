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
    func dumpFloats(name: String, length: Int) {
        print(name)
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
    let sourceTensor: MPSGraphTensor
    let sourceTensorData: MPSGraphTensorData?
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

        let layer = ConvLayer(device: device,
                              graph: MPSGraph(),
                              sourceTensor: nil,
                              descriptor: descriptor,
                              batchSize: batchSize,
                              nnXLen: nnXLen,
                              nnYLen: nnYLen,
                              useFP16: useFP16,
                              useNHWC: useNHWC)

        layer.apply(input: input, output: output)
    }

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         sourceTensor: MPSGraphTensor?,
         descriptor: SWConvLayerDesc,
         batchSize: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        let sourceShape: [NSNumber]
        let sourceLayout: MPSGraphTensorNamedDataLayout
        let dataType = MPSDataType.float32

        let weightsShape = [descriptor.outChannels,
                            descriptor.inChannels,
                            descriptor.convYSize,
                            descriptor.convXSize]

        if (useNHWC == true) {
            sourceShape = [batchSize,
                           nnYLen,
                           nnXLen,
                           descriptor.inChannels]

            sourceLayout = MPSGraphTensorNamedDataLayout.NHWC
        } else {
            sourceShape = [batchSize,
                           descriptor.inChannels,
                           nnYLen,
                           nnXLen]

            sourceLayout = MPSGraphTensorNamedDataLayout.NCHW
        }

        let convDescriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                                               strideInY: 1,
                                                               dilationRateInX: descriptor.dilationX,
                                                               dilationRateInY: descriptor.dilationY,
                                                               groups: 1,
                                                               paddingStyle: .TF_SAME,
                                                               dataLayout: sourceLayout,
                                                               weightsLayout: .OIHW)!

        self.graph = graph

        if sourceTensor == nil {
            self.sourceTensor = graph.placeholder(shape: sourceShape,
                                                  dataType: dataType,
                                                  name: nil)

            sourceTensorData = MPSGraphTensorData(device: device,
                                                  tensor: self.sourceTensor)!
        } else {
            self.sourceTensor = sourceTensor!
            sourceTensorData = nil
        }

        let weightsData = Data(bytes: descriptor.weights,
                               count: weightsShape.asShapeCount(of: dataType))

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: dataType)

        resultTensor = graph.convolution2D(self.sourceTensor,
                                           weights: weightsTensor,
                                           descriptor: convDescriptor,
                                           name: nil)
    }

    func apply(input: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        sourceTensorData!.mpsndarray().writeBytes(input, strideBytes: nil)

        let fetch = graph.run(feeds: [sourceTensor: sourceTensorData!],
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
    let sourceTensor: MPSGraphTensor
    let sourceTensorData: MPSGraphTensorData?
    let maskTensor: MPSGraphTensor
    let maskTensorData: MPSGraphTensorData?
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

        let layer = BatchNormLayer(device: device,
                                   graph: MPSGraph(),
                                   sourceTensor: nil,
                                   maskTensor: nil,
                                   descriptor: descriptor,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        layer.apply(input: input,
                    mask: mask,
                    output: output)
    }

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         sourceTensor: MPSGraphTensor?,
         maskTensor: MPSGraphTensor?,
         descriptor: SWBatchNormLayerDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        let sourceShape: [NSNumber]
        let maskShape: [NSNumber]
        let meanShape: [NSNumber]
        let dataType = MPSDataType.float32

        if useNHWC {
            sourceShape = [batchSize,
                           nnYLen,
                           nnXLen,
                           descriptor.numChannels]

            maskShape = [batchSize,
                         nnYLen,
                         nnXLen,
                         1]

            meanShape = [1,
                         1,
                         1,
                         descriptor.numChannels]
        } else {
            sourceShape = [batchSize,
                           descriptor.numChannels,
                           nnYLen,
                           nnXLen]

            maskShape = [batchSize,
                         1,
                         nnYLen,
                         nnXLen]

            meanShape = [1,
                         descriptor.numChannels,
                         1,
                         1]
        }

        self.graph = graph

        if sourceTensor == nil {
            self.sourceTensor = graph.placeholder(shape: sourceShape,
                                                  dataType: dataType,
                                                  name: nil)

            sourceTensorData = MPSGraphTensorData(device: device,
                                                  tensor: self.sourceTensor)!
        } else {
            self.sourceTensor = sourceTensor!
            sourceTensorData = nil
        }

        if maskTensor == nil {
            self.maskTensor = graph.placeholder(shape: maskShape,
                                                dataType: dataType,
                                                name: nil)

            maskTensorData = MPSGraphTensorData(device: device,
                                                tensor: self.maskTensor)!
        } else {
            self.maskTensor = maskTensor!
            maskTensorData = nil
        }

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

        let normalized = graph.normalize(self.sourceTensor,
                                         mean: meanTensor,
                                         variance: varianceTensor,
                                         gamma: scaleTensor,
                                         beta: biasTensor,
                                         epsilon: descriptor.epsilon,
                                         name: nil)

        resultTensor = graph.multiplication(normalized,
                                            self.maskTensor,
                                            name: nil)
    }

    func apply(input: UnsafeMutablePointer<Float32>,
               mask: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        sourceTensorData!.mpsndarray().writeBytes(input, strideBytes: nil)
        maskTensorData!.mpsndarray().writeBytes(mask, strideBytes: nil)

        let fetch = graph.run(feeds: [sourceTensor: sourceTensorData!,
                                        maskTensor: maskTensorData!],
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
    let sourceTensor: MPSGraphTensor
    let sourceTensorData: MPSGraphTensorData
    let maskTensor: MPSGraphTensor
    let maskTensorData: MPSGraphTensorData
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

        let layer = ResidualBlock(device: device,
                                  graph: MPSGraph(),
                                  descriptor: descriptor,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  batchSize: batchSize,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        layer.apply(input: input,
                    mask: mask,
                    output: output)
    }

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         descriptor: SWResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        let sourceShape: [NSNumber]
        let maskShape: [NSNumber]
        let dataType = MPSDataType.float32

        if useNHWC {
            sourceShape = [batchSize,
                           nnYLen,
                           nnXLen,
                           descriptor.preBN.numChannels]

            maskShape = [batchSize,
                         nnYLen,
                         nnXLen,
                         1]
        } else {
            sourceShape = [batchSize,
                           descriptor.preBN.numChannels,
                           nnYLen,
                           nnXLen]

            maskShape = [batchSize,
                         1,
                         nnYLen,
                         nnXLen]
        }

        self.graph = graph

        sourceTensor = graph.placeholder(shape: sourceShape,
                                         dataType: dataType,
                                         name: nil)

        sourceTensorData = MPSGraphTensorData(device: device,
                                              tensor: sourceTensor)!

        maskTensor = graph.placeholder(shape: maskShape,
                                       dataType: dataType,
                                       name: nil)

        maskTensorData = MPSGraphTensorData(device: device,
                                            tensor: maskTensor)!

        let preBN = BatchNormLayer(device: device,
                                   graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let preReLU = graph.reLU(with: preBN.resultTensor, name: nil)

        let regularConv = ConvLayer(device: device,
                                    graph: graph,
                                    sourceTensor: preReLU,
                                    descriptor: descriptor.regularConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    useFP16: useFP16,
                                    useNHWC: useNHWC)

        let midBN = BatchNormLayer(device: device,
                                   graph: graph,
                                   sourceTensor: regularConv.resultTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let midReLU = graph.reLU(with: midBN.resultTensor, name: nil)

        let finalConv = ConvLayer(device: device,
                                  graph: graph,
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
    }

    func apply(input: UnsafeMutablePointer<Float32>,
               mask: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)
        maskTensorData.mpsndarray().writeBytes(mask, strideBytes: nil)

        let fetch = graph.run(feeds: [sourceTensor: sourceTensorData,
                                        maskTensor: maskTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        fetch[resultTensor]?.mpsndarray().readBytes(output, strideBytes: nil)
    }
}

class GlobalPoolingLayer: NSObject {
    let graph: MPSGraph
    let sourceTensor: MPSGraphTensor
    let maskSumTensor: MPSGraphTensor
    let resultTensor: MPSGraphTensor

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         sourceTensor: MPSGraphTensor,
         maskSumTensor: MPSGraphTensor,
         maskSumSqrtS14M01Tensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        self.graph = graph
        self.sourceTensor = sourceTensor
        self.maskSumTensor = maskSumTensor

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

@objc
class SWMatMulLayerDesc: NSObject {
    let inChannels: Int
    let outChannels: Int
    let weights: UnsafeMutablePointer<Float32>

    @objc
    init(inChannels: Int,
         outChannels: Int,
         weights: UnsafeMutablePointer<Float32>) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
}

class MatMulLayer {
    let graph: MPSGraph
    let sourceTensor: MPSGraphTensor
    let resultTensor: MPSGraphTensor

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         descriptor: SWMatMulLayerDesc,
         sourceTensor: MPSGraphTensor,
         useFP16: Bool,
         useNHWC: Bool) {
        let dataType = MPSDataType.float32

        self.graph = graph
        self.sourceTensor = sourceTensor

        let weightsShape = [descriptor.inChannels as NSNumber,
                            descriptor.outChannels as NSNumber]

        let weightsCount = weightsShape.asShapeCount(of: dataType)
        let weightsData = Data(bytes: descriptor.weights, count: weightsCount)

        let weightsTensor = graph.constant(weightsData,
                                           shape: weightsShape,
                                           dataType: .float32)

        let shape = [-1, descriptor.inChannels as NSNumber]

        let reshapedSource = graph.reshape(sourceTensor,
                                           shape: shape,
                                           name: nil)

        resultTensor = graph.matrixMultiplication(primary: reshapedSource,
                                                  secondary: weightsTensor,
                                                  name: nil)
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
    let sourceTensor: MPSGraphTensor
    let sourceTensorData: MPSGraphTensorData
    let maskTensor: MPSGraphTensor
    let maskTensorData: MPSGraphTensorData
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

        let layer = GlobalPoolingResidualBlock(device: device,
                                               graph: MPSGraph(),
                                               descriptor: descriptor,
                                               nnXLen: nnXLen,
                                               nnYLen: nnYLen,
                                               batchSize: batchSize,
                                               useFP16: useFP16,
                                               useNHWC: useNHWC)

        layer.apply(input: input,
                    mask: mask,
                    output: output)
    }

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         descriptor: SWGlobalPoolingResidualBlockDesc,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: Bool,
         useNHWC: Bool) {
        // TODO: support useFP16 = 1

        let sourceShape: [NSNumber]
        let maskShape: [NSNumber]
        let hwAxes: [NSNumber]
        let dataType = MPSDataType.float32

        if useNHWC {
            sourceShape = [batchSize,
                           nnYLen,
                           nnXLen,
                           descriptor.preBN.numChannels]

            maskShape = [batchSize, nnYLen, nnXLen, 1]
            hwAxes = [1, 2]

        } else {
            sourceShape = [batchSize,
                           descriptor.preBN.numChannels,
                           nnYLen,
                           nnXLen]

            maskShape = [batchSize, 1, nnYLen, nnXLen]
            hwAxes = [2, 3]
        }

        self.graph = graph

        sourceTensor = graph.placeholder(shape: sourceShape,
                                         dataType: dataType,
                                         name: nil)

        sourceTensorData = MPSGraphTensorData(device: device,
                                              tensor: sourceTensor)!

        maskTensor = graph.placeholder(shape: maskShape,
                                       dataType: dataType,
                                       name: nil)

        maskTensorData = MPSGraphTensorData(device: device,
                                            tensor: maskTensor)!

        let preBN = BatchNormLayer(device: device,
                                   graph: graph,
                                   sourceTensor: sourceTensor,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.preBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let preReLU = graph.reLU(with: preBN.resultTensor, name: nil)

        let regularConv = ConvLayer(device: device,
                                    graph: graph,
                                    sourceTensor: preReLU,
                                    descriptor: descriptor.regularConv,
                                    batchSize: batchSize,
                                    nnXLen: nnXLen,
                                    nnYLen: nnYLen,
                                    useFP16: useFP16,
                                    useNHWC: useNHWC)

        let gpoolConv = ConvLayer(device: device,
                                  graph: graph,
                                  sourceTensor: preReLU,
                                  descriptor: descriptor.gpoolConv,
                                  batchSize: batchSize,
                                  nnXLen: nnXLen,
                                  nnYLen: nnYLen,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

        let gpoolBN = BatchNormLayer(device: device,
                                     graph: graph,
                                     sourceTensor: gpoolConv.resultTensor,
                                     maskTensor: maskTensor,
                                     descriptor: descriptor.gpoolBN,
                                     nnXLen: nnXLen,
                                     nnYLen: nnYLen,
                                     batchSize: batchSize,
                                     useFP16: useFP16,
                                     useNHWC: useNHWC)


        let gpoolReLU = graph.reLU(with: gpoolBN.resultTensor, name: nil)

        let maskSum = graph.reductionSum(with: maskTensor, axes: hwAxes, name: nil)
        let sqrtMaskSum = graph.squareRoot(with: maskSum, name: nil)

        let fourTeen = graph.constant(14.0,
                                      shape: sqrtMaskSum.shape!,
                                      dataType: .float32)

        let subtracted = graph.subtraction(sqrtMaskSum, fourTeen, name: nil)

        let zeroPointone = graph.constant(0.1,
                                          shape: sqrtMaskSum.shape!,
                                          dataType: .float32)

        let maskSumSqrtS14M01 = graph.multiplication(subtracted,
                                                     zeroPointone,
                                                     name: nil)

        let gpoolConcat = GlobalPoolingLayer(device: device,
                                             graph: graph,
                                             sourceTensor: gpoolReLU,
                                             maskSumTensor: maskSum,
                                             maskSumSqrtS14M01Tensor: maskSumSqrtS14M01,
                                             useFP16: useFP16,
                                             useNHWC: useNHWC)

        let gpoolToBiasMul = MatMulLayer(device: device,
                                         graph: graph,
                                         descriptor: descriptor.gpoolToBiasMul,
                                         sourceTensor: gpoolConcat.resultTensor,
                                         useFP16: useFP16,
                                         useNHWC: useNHWC)

        let shape = [batchSize as NSNumber,
                     1,
                     1,
                     descriptor.gpoolToBiasMul.outChannels as NSNumber]

        let reshapedGoolToBiasMul = graph.reshape(gpoolToBiasMul.resultTensor,
                                                  shape: shape,
                                                  name: nil)

        let added = graph.addition(regularConv.resultTensor,
                                   reshapedGoolToBiasMul,
                                   name: nil)

        let midBN = BatchNormLayer(device: device,
                                   graph: graph,
                                   sourceTensor: added,
                                   maskTensor: maskTensor,
                                   descriptor: descriptor.midBN,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC)

        let midReLU = graph.reLU(with: midBN.resultTensor, name: nil)

        let finalConv = ConvLayer(device: device,
                                  graph: graph,
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
    }

    func apply(input: UnsafeMutablePointer<Float32>,
               mask: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)
        maskTensorData.mpsndarray().writeBytes(mask, strideBytes: nil)

        let fetch = graph.run(feeds: [sourceTensor: sourceTensorData,
                                        maskTensor: maskTensorData],
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
