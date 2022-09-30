import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

extension UnsafeMutablePointer<Float32> {
    func printAsFloat() {
        print("data[0]=\(self[0])")
        print("data[1]=\(self[1])")
        print("data[2]=\(self[2])")
        print("data[3]=\(self[3])")
        print("data[4]=\(self[4])")
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

@objc
class ConvLayer: NSObject {
    let graph: MPSGraph
    let sourceType: MPSDataType
    let sourceShape: [NSNumber]
    let sourceLayout: MPSGraphTensorNamedDataLayout
    let sourceTensor: MPSGraphTensor
    let sourceTensorData: MPSGraphTensorData
    let weightsType: MPSDataType
    let weightsTensor: MPSGraphTensor
    let weightsTensorData: MPSGraphTensorData
    let resultTensor: MPSGraphTensor

    @objc
    class func test(convXSize: NSNumber,
                    convYSize: NSNumber,
                    inChannels: NSNumber,
                    outChannels: NSNumber,
                    dilationX: NSNumber,
                    dilationY: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    batchSize: NSNumber,
                    useFP16: NSNumber,
                    useNHWC: NSNumber,
                    weights: UnsafeMutablePointer<Float32>,
                    input: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {
        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let layer = ConvLayer(device: device,
                              graph: MPSGraph(),
                              batchSize: batchSize,
                              convXSize: convXSize,
                              convYSize: convYSize,
                              inChannels: inChannels,
                              outChannels: outChannels,
                              dilationX: dilationX,
                              dilationY: dilationY,
                              nnXLen: nnXLen,
                              nnYLen: nnYLen,
                              useFP16: useFP16,
                              useNHWC: useNHWC,
                              weights: weights)

        layer.apply(input: input, output: output)
    }

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         batchSize: NSNumber,
         convXSize: NSNumber,
         convYSize: NSNumber,
         inChannels: NSNumber,
         outChannels: NSNumber,
         dilationX: NSNumber,
         dilationY: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         useFP16: NSNumber,
         useNHWC: NSNumber,
         weights: UnsafeMutablePointer<Float32>) {
        self.graph = graph
        sourceType = MPSDataType.float32
        weightsType = MPSDataType.float32

        if (useNHWC.boolValue == true) {
            sourceShape = [batchSize.intValue as NSNumber,
                           nnYLen.intValue as NSNumber,
                           nnXLen.intValue as NSNumber,
                           inChannels]

            sourceLayout = MPSGraphTensorNamedDataLayout.NHWC
        } else {
            sourceShape = [batchSize.intValue as NSNumber,
                           inChannels,
                           nnYLen.intValue as NSNumber,
                           nnXLen.intValue as NSNumber]

            sourceLayout = MPSGraphTensorNamedDataLayout.NCHW
        }

        sourceTensor = graph.placeholder(shape: sourceShape,
                                         dataType: sourceType,
                                         name: nil)

        sourceTensorData = MPSGraphTensorData(device: device,
                                              tensor: sourceTensor)!

        let weightsShape = [outChannels,
                            inChannels,
                            convYSize,
                            convXSize]

        weightsTensor = graph.placeholder(shape: weightsShape,
                                          dataType: weightsType,
                                          name: nil)

        weightsTensorData = MPSGraphTensorData(device: device,
                                               tensor: weightsTensor)!

        weightsTensorData.mpsndarray().writeBytes(weights, strideBytes: nil)

        let convDescriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                                               strideInY: 1,
                                                               dilationRateInX: dilationX.intValue,
                                                               dilationRateInY: dilationY.intValue,
                                                               groups: 1,
                                                               paddingStyle: .TF_SAME,
                                                               dataLayout: sourceLayout,
                                                               weightsLayout: .OIHW)!

        resultTensor = graph.convolution2D(sourceTensor,
                                           weights: weightsTensor,
                                           descriptor: convDescriptor,
                                           name: nil)
    }

    func apply(input: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)

        let fetch = graph.run(feeds: [sourceTensor: sourceTensorData,
                                     weightsTensor: weightsTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        fetch[resultTensor]?.mpsndarray().readBytes(output, strideBytes: nil)
    }
}

@objc
class BatchNormLayer: NSObject {
    let graph: MPSGraph
    let sourceType: MPSDataType
    let sourceShape: [NSNumber]
    let sourceLayout: MPSGraphTensorNamedDataLayout
    let sourceTensor: MPSGraphTensor
    let sourceTensorData: MPSGraphTensorData
    let maskType: MPSDataType
    let maskShape: [NSNumber]
    let maskTensor: MPSGraphTensor
    let maskTensorData: MPSGraphTensorData
    let meanType: MPSDataType
    let meanShape: [NSNumber]
    let meanTensor: MPSGraphTensor
    let meanTensorData: MPSGraphTensorData
    let varianceType: MPSDataType
    let varianceTensor: MPSGraphTensor
    let varianceTensorData: MPSGraphTensorData
    let scaleType: MPSDataType
    let scaleTensor: MPSGraphTensor
    let scaleTensorData: MPSGraphTensorData
    let biasType: MPSDataType
    let biasTensor: MPSGraphTensor
    let biasTensorData: MPSGraphTensorData
    let resultTensor: MPSGraphTensor

    @objc
    class func test(numChannels: NSNumber,
                    epsilon: NSNumber,
                    hasScale: NSNumber,
                    hasBias: NSNumber,
                    nnXLen: NSNumber,
                    nnYLen: NSNumber,
                    batchSize: NSNumber,
                    useFP16: NSNumber,
                    useNHWC: NSNumber,
                    mean: UnsafeMutablePointer<Float32>,
                    variance: UnsafeMutablePointer<Float32>,
                    scale: UnsafeMutablePointer<Float32>,
                    bias: UnsafeMutablePointer<Float32>,
                    input: UnsafeMutablePointer<Float32>,
                    mask: UnsafeMutablePointer<Float32>,
                    output: UnsafeMutablePointer<Float32>) {
        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let layer = BatchNormLayer(device: device,
                                   graph: MPSGraph(),
                                   numChannels: numChannels,
                                   epsilon: epsilon,
                                   hasScale: hasScale,
                                   hasBias: hasBias,
                                   nnXLen: nnXLen,
                                   nnYLen: nnYLen,
                                   batchSize: batchSize,
                                   useFP16: useFP16,
                                   useNHWC: useNHWC,
                                   mean: mean,
                                   variance: variance,
                                   scale: scale,
                                   bias: bias)

        layer.apply(input: input,
                    mask: mask,
                    output: output)
    }

    init(device: MPSGraphDevice,
         graph: MPSGraph,
         numChannels: NSNumber,
         epsilon: NSNumber,
         hasScale: NSNumber,
         hasBias: NSNumber,
         nnXLen: NSNumber,
         nnYLen: NSNumber,
         batchSize: NSNumber,
         useFP16: NSNumber,
         useNHWC: NSNumber,
         mean: UnsafeMutablePointer<Float32>,
         variance: UnsafeMutablePointer<Float32>,
         scale: UnsafeMutablePointer<Float32>,
         bias: UnsafeMutablePointer<Float32>) {
        self.graph = graph
        sourceType = MPSDataType.float32
        maskType = MPSDataType.float32
        meanType = MPSDataType.float32
        varianceType = MPSDataType.float32
        scaleType = MPSDataType.float32
        biasType = MPSDataType.float32

        if (useNHWC.boolValue == true) {
            sourceShape = [batchSize.intValue as NSNumber,
                           nnYLen.intValue as NSNumber,
                           nnXLen.intValue as NSNumber,
                           numChannels]

            sourceLayout = MPSGraphTensorNamedDataLayout.NHWC

            meanShape = [1,
                         1,
                         1,
                         numChannels]

            maskShape = [batchSize.intValue as NSNumber,
                         nnYLen.intValue as NSNumber,
                         nnXLen.intValue as NSNumber,
                         1]
        } else {
            sourceShape = [batchSize.intValue as NSNumber,
                           numChannels,
                           nnYLen.intValue as NSNumber,
                           nnXLen.intValue as NSNumber]

            sourceLayout = MPSGraphTensorNamedDataLayout.NCHW

            meanShape = [1,
                         numChannels,
                         1,
                         1]

            maskShape = [batchSize.intValue as NSNumber,
                         1,
                         nnYLen.intValue as NSNumber,
                         nnXLen.intValue as NSNumber]
        }

        sourceTensor = graph.placeholder(shape: sourceShape,
                                         dataType: sourceType,
                                         name: nil)

        sourceTensorData = MPSGraphTensorData(device: device,
                                              tensor: sourceTensor)!

        maskTensor = graph.placeholder(shape: maskShape,
                                       dataType: maskType,
                                       name: nil)

        maskTensorData = MPSGraphTensorData(device: device,
                                            tensor: maskTensor)!

        meanTensor = graph.placeholder(shape: meanShape,
                                       dataType: meanType,
                                       name: nil)

        meanTensorData = MPSGraphTensorData(device: device,
                                            tensor: meanTensor)!

        meanTensorData.mpsndarray().writeBytes(mean, strideBytes: nil)

        let varianceShape = meanShape

        varianceTensor = graph.placeholder(shape: varianceShape,
                                           dataType: varianceType,
                                           name: nil)

        varianceTensorData = MPSGraphTensorData(device: device,
                                                tensor: varianceTensor)!

        varianceTensorData.mpsndarray().writeBytes(variance, strideBytes: nil)

        let scaleShape = meanShape

        scaleTensor = graph.placeholder(shape: scaleShape,
                                        dataType: scaleType,
                                        name: nil)

        scaleTensorData = MPSGraphTensorData(device: device,
                                             tensor: scaleTensor)!

        scaleTensorData.mpsndarray().writeBytes(scale, strideBytes: nil)

        let biasShape = meanShape

        biasTensor = graph.placeholder(shape: biasShape,
                                       dataType: biasType,
                                       name: nil)

        biasTensorData = MPSGraphTensorData(device: device,
                                            tensor: biasTensor)!

        biasTensorData.mpsndarray().writeBytes(bias, strideBytes: nil)

        let normalized = graph.normalize(sourceTensor,
                                         mean: meanTensor,
                                         variance: varianceTensor,
                                         gamma: scaleTensor,
                                         beta: biasTensor,
                                         epsilon: epsilon.floatValue,
                                         name: nil)

        resultTensor = graph.multiplication(normalized,
                                            maskTensor,
                                            name: nil)
    }

    func apply(input: UnsafeMutablePointer<Float32>,
               mask: UnsafeMutablePointer<Float32>,
               output: UnsafeMutablePointer<Float32>) {
        sourceTensorData.mpsndarray().writeBytes(input, strideBytes: nil)
        maskTensorData.mpsndarray().writeBytes(mask, strideBytes: nil)

        let fetch = graph.run(feeds: [sourceTensor: sourceTensorData,
                                        maskTensor: maskTensorData,
                                        meanTensor: meanTensorData,
                                    varianceTensor: varianceTensorData,
                                       scaleTensor: scaleTensorData,
                                        biasTensor: biasTensorData],
                              targetTensors: [resultTensor],
                              targetOperations: nil)

        fetch[resultTensor]?.mpsndarray().readBytes(output, strideBytes: nil)
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

        inputTensor = graph.placeholder(shape: [nnXLen.intValue as NSNumber,
                                                nnYLen.intValue as NSNumber,
                                                numInputChannels.intValue as NSNumber],
                                        name: "binInputs")

        let inputArrayDesc = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                  shape: inputTensor.shape!)

        let inputArray = MPSNDArray(device: device, descriptor: inputArrayDesc)

        inputTensorData = MPSGraphTensorData(inputArray)

        inputGlobalTensor = graph.placeholder(shape: [numInputGlobalChannels.intValue as NSNumber],
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
        policyOutput.printAsFloat()
    }
}
