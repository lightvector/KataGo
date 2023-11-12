import XCTest
import MetalPerformanceShadersGraph

extension MPSNDArray {
    /// Returns the total number of elements in the MPSNDArray.
    func countElements() -> Int {
        // Initialize the range of dimensions from 0 to numberOfDimensions - 1
        let dimensionsRange = 0..<numberOfDimensions

        // Use the reduce function to calculate the total number of elements
        let totalElements = dimensionsRange.reduce(1) { count, dimensionIndex in
            // Multiply the current count by the length of the current dimension
            count * length(ofDimension: dimensionIndex)
        }

        return totalElements
    }
}

final class MPSGraphTest: XCTestCase {

    func testMish() {
        let device = MTLCreateSystemDefaultDevice()!
        let graph = MPSGraph()
        let shape: [NSNumber] = [5]
        let inputTensor = graph.placeholder(shape: shape, name: nil)
        let mishTensor = graph.mish(tensor: inputTensor)

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 5)

        inputPointer[0] = -1
        inputPointer[1] = 0
        inputPointer[2] = 1
        inputPointer[3] = 10.38
        inputPointer[4] = 10.4

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: shape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)

        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [inputTensor: inputTensorData],
                              targetTensors: [mishTensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[mishTensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(mishTensor.shape == shape)
        XCTAssertEqual(buffer[0], -0.30340147018432617, accuracy: 1e-6)
        XCTAssertEqual(buffer[1], 0.0, accuracy: 1e-6)
        XCTAssertEqual(buffer[2], 0.8650983572006226, accuracy: 1e-6)
        XCTAssertEqual(buffer[3], 10.380000114440918, accuracy: 1e-6)
        XCTAssertEqual(buffer[4], 10.4, accuracy: 1e-6)
    }
}

final class InputLayerTest: XCTestCase {

    func testNCHW() {
        let sourceLayer = InputLayer(graph: MPSGraph(),
                                     nnXLen: 5,
                                     nnYLen: 4,
                                     numChannels: 3)

        XCTAssert(sourceLayer.tensor.shape == [-1, 3, 4, 5])
        XCTAssert(sourceLayer.tensor.dataType == .float32)
    }
}

final class InputGlobalLayerTest: XCTestCase {

    func testNilTensor() {
        let inputGlobalLayer = InputGlobalLayer(graph: MPSGraph(),
                                                numGlobalFeatures: 3)

        XCTAssert(inputGlobalLayer.tensor.shape == [-1, 3, 1, 1])
        XCTAssert(inputGlobalLayer.tensor.dataType == .float32)
    }
}

final class MaskLayerTest: XCTestCase {

    func testNilTensor() {
        let graph = MPSGraph()

        let maskLayer = MaskLayer(graph: graph,
                                  nnXLen: 4,
                                  nnYLen: 3)

        XCTAssert(maskLayer.tensor.shape == [-1, 1, 3, 4])
        XCTAssert(maskLayer.tensor.dataType == .float32)
    }
}

final class MaskSumLayerTest: XCTestCase {

    func testTensor() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 1, 1, 1]
        let tensor = graph.constant(12, shape: shape, dataType: .float32)
        let maskSumLayer = MaskSumLayer(tensor: tensor)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumLayer.tensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumLayer.tensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(maskSumLayer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], 12)
        XCTAssertEqual(buffer[1], 12)
    }

    func testNilTensor() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 1, 3, 4]
        let tensor = graph.constant(1, shape: shape, dataType: .float32)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        maskTensor: tensor)

        XCTAssert(maskSumLayer.tensor.shape == [2, 1, 1, 1])

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumLayer.tensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumLayer.tensor]?.mpsndarray().readBytes(buffer)

        XCTAssertEqual(buffer[0], 12)
        XCTAssertEqual(buffer[1], 12)
    }
}

final class MaskSumSqrtS14M01LayerTest: XCTestCase {

    func testTensor() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 1, 1, 1]

        let tensor = graph.constant(-1.053589838486225,
                                     shape: shape,
                                     dataType: .float32)

        let maskSumSqrtS14M01Layer = MaskSumSqrtS14M01Layer(tensor: tensor)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01Layer.tensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01Layer.tensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(maskSumSqrtS14M01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], -1.053589838486225, accuracy: 1e-8)
        XCTAssertEqual(buffer[1], -1.053589838486225, accuracy: 1e-8)
    }

    func testNilTensor() {
        let graph = MPSGraph()

        let shape: [NSNumber] = [2, 1, 3, 4]

        let tensor = graph.constant(1,
                                    shape: shape,
                                    dataType: .float32)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        maskTensor: tensor)

        let maskSumSqrtS14M01Layer = MaskSumSqrtS14M01Layer(graph: graph,
                                                            maskSum: maskSumLayer)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01Layer.tensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01Layer.tensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(maskSumSqrtS14M01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], -1.053589838486225, accuracy: 1e-8)
        XCTAssertEqual(buffer[1], -1.053589838486225, accuracy: 1e-8)
    }
}

final class MaskSumSqrtS14M01SquareS01LayerTest: XCTestCase {

    func testTensor() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 1, 1, 1]

        let tensor = graph.constant(1.010051547761429,
                                    shape: shape,
                                    dataType: .float32)

        let maskSumSqrtS14M01SquareS01Layer = MaskSumSqrtS14M01SquareS01Layer(tensor: tensor)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01SquareS01Layer.tensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01SquareS01Layer.tensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(maskSumSqrtS14M01SquareS01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], 1.010051547761429, accuracy: 1e-8)
        XCTAssertEqual(buffer[1], 1.010051547761429, accuracy: 1e-8)
    }

    func testNilTensor() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 1, 3, 4]

        let tensor = graph.constant(1,
                                    shape: shape,
                                    dataType: .float32)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        maskTensor: tensor)

        let maskSumSqrtS14M01Layer = MaskSumSqrtS14M01Layer(graph: graph,
                                                            maskSum: maskSumLayer)

        let maskSumSqrtS14M01SquareS01Layer =
        MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                        maskSumSqrtS14M01: maskSumSqrtS14M01Layer)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01SquareS01Layer.tensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01SquareS01Layer.tensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(maskSumSqrtS14M01SquareS01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], 1.010051547761429, accuracy: 1e-8)
        XCTAssertEqual(buffer[1], 1.010051547761429, accuracy: 1e-8)
    }
}

final class ConvLayerTest: XCTestCase {

    func testBase() {
        let convXSize = 3
        let convYSize = 3
        let outChannels: NSNumber = 2
        let weightsLength = convXSize * convYSize * outChannels.intValue
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: weightsLength)

        weights[0] = 0
        weights[1] = 1
        weights[2] = 0
        weights[3] = 0
        weights[4] = 0
        weights[5] = 0
        weights[6] = 0
        weights[7] = 0
        weights[8] = 0

        weights[9] = 0
        weights[10] = 0
        weights[11] = 0
        weights[12] = 0
        weights[13] = 0
        weights[14] = 0
        weights[15] = 0
        weights[16] = 1
        weights[17] = 0

        let inChannels: NSNumber = 1

        let descriptor = createSWConvLayerDesc(convYSize: Int32(convYSize),
                                               convXSize: Int32(convXSize),
                                               inChannels: Int32(truncating: inChannels),
                                               outChannels: Int32(truncating: outChannels),
                                               dilationY: 1,
                                               dilationX: 1,
                                               weights: weights)

        let batchSize: NSNumber = 1
        let nnXLen: NSNumber = 3
        let nnYLen: NSNumber = 2

        let inputLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue * inChannels.intValue

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputLength)

        inputPointer[0] = 0
        inputPointer[1] = 1
        inputPointer[2] = 2
        inputPointer[3] = 3
        inputPointer[4] = 4
        inputPointer[5] = 5

        let outputLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue * outChannels.intValue

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputLength)

        testConvLayer(descriptor: descriptor,
                      nnXLen: Int32(truncating: nnXLen),
                      nnYLen: Int32(truncating: nnYLen),
                      batchSize: Int32(truncating: batchSize),
                      input: inputPointer,
                      output: outputPointer)

        XCTAssertEqual(outputPointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[4], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[6], 3, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[8], 5, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[10], 0, accuracy: 1e-8)

        XCTAssertEqual(outputPointer[1], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[5], 2, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[7], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[9], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[11], 0, accuracy: 1e-8)
    }
}

final class BatchNormLayerTest: XCTestCase {

    func testBase() {
        let numChannels: NSNumber = 2
        let length = numChannels.intValue
        let mean = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        mean[0] = 0
        mean[1] = 2

        let variance = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        variance[0] = 3.9
        variance[1] = 0.15

        let scale = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        scale[0] = 0.1
        scale[1] = 1

        let bias = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        bias[0] = 10
        bias[1] = 0

        let descriptor = createSWBatchNormLayerDesc(numChannels: Int32(truncating: numChannels),
                                                    epsilon: 0.1,
                                                    hasScale: true,
                                                    hasBias: true,
                                                    mean: mean,
                                                    variance: variance,
                                                    scale: scale,
                                                    bias: bias)

        let batchSize: NSNumber = 2
        let nnXLen: NSNumber = 5
        let nnYLen: NSNumber = 2

        let inputLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue * numChannels.intValue

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputLength)
        let x = inputPointer

        x[0] = 5; x[2] = 5; x[4] = 4; x[6] = 4; x[8] = 9
        x[10] = 1; x[12] = 1; x[14] = 8; x[16] = 8; x[18] = 9

        x[1] = 0; x[3] = 1; x[5] = 2; x[7] = 3; x[9] = 4
        x[11] = 8; x[13] = 7; x[15] = 6; x[17] = 5; x[19] = 4

        x[20] = 3; x[22] = 0; x[24] = 4; x[26] = 0; x[28] = 5
        x[30] = 0; x[32] = 5; x[34] = 0; x[36] = 6; x[38] = 0

        x[21] = 1; x[23] = 0; x[25] = 0; x[27] = 2; x[29] = 1
        x[31] = 0; x[33] = 2; x[35] = 2; x[37] = 0; x[39] = 2

        let maskLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskLength)
        let m = maskPointer

        m[0] = 1; m[1] = 1; m[2] = 1; m[3] = 1; m[4] = 1
        m[5] = 1; m[6] = 1; m[7] = 1; m[8] = 1; m[9] = 1

        m[10] = 1; m[11] = 1; m[12] = 1; m[13] = 1; m[14] = 1
        m[15] = 1; m[16] = 1; m[17] = 1; m[18] = 1; m[19] = 1

        let outputLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue * numChannels.intValue

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputLength)

        testBatchNormLayer(descriptor: descriptor,
                           nnXLen: Int32(truncating: nnXLen),
                           nnYLen: Int32(truncating: nnYLen),
                           batchSize: Int32(truncating: batchSize),
                           input: inputPointer,
                           mask: maskPointer,
                           output: outputPointer)

        XCTAssertEqual(outputPointer[0], 10.25, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[8], 10.45, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[10], -2.0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[18], 14.0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[19], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[20], 10.15, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[39], 0, accuracy: 1e-8)
    }
}

final class ActivationLayerTest: XCTestCase {

    func testMish() {
        let device = MTLCreateSystemDefaultDevice()!
        let graph = MPSGraph()
        let inputNumber = 6
        let shape: [NSNumber] = [NSNumber(value: inputNumber)]
        let inputTensor = graph.placeholder(shape: shape, name: nil)

        let activationLayer = ActivationLayer(graph: graph,
                                              sourceTensor: inputTensor,
                                              activationKind: ActivationKind.mish)

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputNumber)

        inputPointer[0] = -1e10
        inputPointer[1] = -1
        inputPointer[2] = 0
        inputPointer[3] = 1
        inputPointer[4] = 10.38
        inputPointer[5] = 1e10

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: shape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [inputTensor: inputTensorData],
                              targetTensors: [activationLayer.resultTensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[activationLayer.resultTensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(activationLayer.resultTensor.shape == shape)
        XCTAssertEqual(buffer[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(buffer[1], -0.30340147018432617, accuracy: 1e-6)
        XCTAssertEqual(buffer[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(buffer[3], 0.8650983572006226, accuracy: 1e-6)
        XCTAssertEqual(buffer[4], 10.380000114440918, accuracy: 1e-6)
        XCTAssertEqual(buffer[5], 1e10, accuracy: 1e4)
    }

    func testIdentity() {
        let device = MTLCreateSystemDefaultDevice()!
        let graph = MPSGraph()
        let shape: [NSNumber] = [5]
        let inputTensor = graph.placeholder(shape: shape, name: nil)

        let activationLayer = ActivationLayer(graph: graph,
                                              sourceTensor: inputTensor,
                                              activationKind: ActivationKind.identity)

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 5)

        inputPointer[0] = -10.38
        inputPointer[1] = -1
        inputPointer[2] = 0
        inputPointer[3] = 1
        inputPointer[4] = 10.38

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: shape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [inputTensor: inputTensorData],
                              targetTensors: [activationLayer.resultTensor],
                              targetOperations: nil)

        let length = shape.countElements()
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[activationLayer.resultTensor]?.mpsndarray().readBytes(buffer)

        XCTAssert(activationLayer.resultTensor.shape == shape)
        XCTAssertEqual(buffer[0], inputPointer[0], accuracy: 1e-6)
        XCTAssertEqual(buffer[1], inputPointer[1], accuracy: 1e-6)
        XCTAssertEqual(buffer[2], inputPointer[2], accuracy: 1e-6)
        XCTAssertEqual(buffer[3], inputPointer[3], accuracy: 1e-6)
        XCTAssertEqual(buffer[4], inputPointer[4], accuracy: 1e-6)
    }
}

final class ResidualBlockTest: XCTestCase {

    func testNHWC() {
        let batchSize: NSNumber = 2
        let trunkChannels: NSNumber = 1
        let midChannels: NSNumber = 2
        let nnYLen: NSNumber = 3
        let nnXLen: NSNumber = 4

        let inputLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue * trunkChannels.intValue

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputLength)
        let x = inputPointer

        x[0] = 1; x[1] = 0; x[2]  = 0; x[3]  = 0
        x[4] = 0; x[5] = 2; x[6]  = 2; x[7]  = 0
        x[8] = 0; x[9] = 0; x[10] = 0; x[11] = 1

        x[12] = 0; x[13] = 0; x[14] = 0;  x[15] = 0
        x[16] = 0; x[17] = 3; x[18] = -5; x[19] = 0
        x[20] = 1; x[21] = 1; x[22] = 1;  x[23] = 1

        let maskLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskLength)
        let m = maskPointer

        m[0] = 1; m[1] = 1; m[2]  = 0; m[3]  = 1
        m[4] = 1; m[5] = 1; m[6]  = 1; m[7]  = 1
        m[8] = 1; m[9] = 1; m[10] = 0; m[11] = 1

        m[12] = 1; m[13] = 1; m[14] = 1; m[15] = 1
        m[16] = 1; m[17] = 1; m[18] = 1; m[19] = 0
        m[20] = 1; m[21] = 1; m[22] = 1; m[23] = 1

        let preBN =
        SWBatchNormLayerDesc(numChannels: trunkChannels,
                             epsilon: 0.1,
                             hasScale: true,
                             hasBias: true,
                             mean: UnsafeMutablePointer<Float32>.allocate(capacity: trunkChannels.intValue),
                             variance: UnsafeMutablePointer<Float32>.allocate(capacity: trunkChannels.intValue),
                             scale: UnsafeMutablePointer<Float32>.allocate(capacity: trunkChannels.intValue),
                             bias: UnsafeMutablePointer<Float32>.allocate(capacity: trunkChannels.intValue))

        preBN.mean[0] = 0
        preBN.variance[0] = 0.9
        preBN.scale[0] = 2
        preBN.bias[0] = 0

        let convYSize: NSNumber = 3
        let convXSize: NSNumber = 3
        let capacity = convYSize.intValue * convXSize.intValue * midChannels.intValue

        let regularConv = SWConvLayerDesc(convYSize: convYSize,
                                          convXSize: convXSize,
                                          inChannels: trunkChannels,
                                          outChannels: midChannels,
                                          dilationY: 1,
                                          dilationX: 1,
                                          weights: UnsafeMutablePointer<Float32>.allocate(capacity: capacity))

        let w = regularConv.weights;

        w[0] = 0; w[1] = 1; w[2] = 0
        w[3] = 0; w[4] = 0; w[5] = 0
        w[6] = 0; w[7] = 0; w[8] = 0

        w[9] = 0; w[10] = 0; w[11] = 0
        w[12] = 0; w[13] = 0; w[14] = 0
        w[15] = 0; w[16] = 1; w[17] = 0

        let midBN =
        SWBatchNormLayerDesc(numChannels: midChannels,
                             epsilon: 0.1,
                             hasScale: false,
                             hasBias: false,
                             mean: UnsafeMutablePointer<Float32>.allocate(capacity: midChannels.intValue),
                             variance: UnsafeMutablePointer<Float32>.allocate(capacity: midChannels.intValue),
                             scale: UnsafeMutablePointer<Float32>.allocate(capacity: midChannels.intValue),
                             bias: UnsafeMutablePointer<Float32>.allocate(capacity: midChannels.intValue))

        midBN.mean[0] = 3; midBN.mean[1] = 0
        midBN.variance[0] = 0.9; midBN.variance[1] = 0.9
        midBN.scale[0] = 1; midBN.scale[1] = 1
        midBN.bias[0] = 0; midBN.bias[1] = 0

        let finalConv = SWConvLayerDesc(convYSize: 1,
                                        convXSize: 1,
                                        inChannels: midChannels,
                                        outChannels: trunkChannels,
                                        dilationY: 1,
                                        dilationX: 1,
                                        weights: UnsafeMutablePointer<Float32>.allocate(capacity: 2))

        finalConv.weights[0] = 1; finalConv.weights[1] = 1

        let descriptor = createSWResidualBlockDesc(preBN: preBN,
                                                   preActivation: ActivationKind.relu,
                                                   regularConv: regularConv,
                                                   midBN: midBN,
                                                   midActivation: ActivationKind.relu,
                                                   finalConv: finalConv)

        let outputLength = batchSize.intValue * trunkChannels.intValue * nnYLen.intValue * nnXLen.intValue

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputLength)

        testResidualBlock(descriptor: descriptor,
                          batchSize: Int32(truncating: batchSize),
                          nnXLen: Int32(truncating: nnXLen),
                          nnYLen: Int32(truncating: nnYLen),
                          input: inputPointer,
                          mask: maskPointer,
                          output: outputPointer)

        XCTAssertEqual(outputPointer[0], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[4], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[11], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[12], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[18], -3, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[23], 1, accuracy: 1e-8)
    }

    func testUnity() {
        let batchSize = 2
        let nnXLen = 2
        let nnYLen = 2
        let numChannels = 2

        let unityConvWeights = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels * numChannels)

        unityConvWeights[0] = 1
        unityConvWeights[1] = 0
        unityConvWeights[2] = 0
        unityConvWeights[3] = 1

        let unityConv = SWConvLayerDesc(convYSize: 1,
                                        convXSize: 1,
                                        inChannels: numChannels as NSNumber,
                                        outChannels: numChannels as NSNumber,
                                        dilationY: 1,
                                        dilationX: 1,
                                        weights: unityConvWeights)

        let mean = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        mean[0] = 0
        mean[1] = 0

        let variance = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        variance[0] = 0.9
        variance[1] = 0.9

        let scale = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        scale[0] = 1
        scale[1] = 1

        let bias = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        bias[0] = 0
        bias[1] = 0

        let unityBN = SWBatchNormLayerDesc(numChannels: numChannels as NSNumber,
                                           epsilon: 0.1,
                                           hasScale: false,
                                           hasBias: false,
                                           mean: mean,
                                           variance: variance,
                                           scale: scale,
                                           bias: bias)

        let residualBlock = SWResidualBlockDesc(preBN: unityBN,
                                                preActivation: ActivationKind.relu,
                                                regularConv: unityConv,
                                                midBN: unityBN,
                                                midActivation: ActivationKind.relu,
                                                finalConv: unityConv)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: numChannels as NSNumber)

        let mask = MaskLayer(graph: graph,
                             nnXLen: nnXLen as NSNumber,
                             nnYLen: nnYLen as NSNumber)

        let block = ResidualBlock(graph: graph,
                                  sourceTensor: input.tensor,
                                  maskTensor: mask.tensor,
                                  descriptor: residualBlock,
                                  nnXLen: nnXLen as NSNumber,
                                  nnYLen: nnYLen as NSNumber)

        let inputCount = batchSize * numChannels * nnXLen * nnYLen
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        let maskCount = batchSize * nnXLen * nnYLen
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskCount)

        for i in 0..<maskCount {
            maskPointer[i] = 1
        }

        let device = MTLCreateSystemDefaultDevice()!

        let inputArrayShape = [batchSize, numChannels, nnYLen, nnXLen] as [NSNumber]
        let inputDescriptor = MPSNDArrayDescriptor(dataType: input.tensor.dataType,
                                                   shape: inputArrayShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)

        let maskArrayShape = [batchSize, 1, nnYLen, nnXLen] as [NSNumber]
        let maskDescriptor = MPSNDArrayDescriptor(dataType: mask.tensor.dataType,
                                                  shape: maskArrayShape)

        let maskArray = MPSNDArray(device: device,
                                   descriptor: maskDescriptor)

        maskArray.writeBytes(maskPointer)

        let inputTensorData = MPSGraphTensorData(inputArray)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [block.resultTensor],
                              targetOperations: nil)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        fetch[block.resultTensor]?.mpsndarray().readBytes(outputPointer)

        XCTAssertEqual(outputPointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 2, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 6, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[15], 30, accuracy: 1e-8)
    }
}

final class GlobalPoolingResidualBlockTest: XCTestCase {

    func testNHWC() {
        let batchSize: NSNumber = 2
        let trunkChannels: NSNumber = 1
        let regularChannels: NSNumber = 1
        let gpoolChannels: NSNumber = 2
        let nnYLen: NSNumber = 3
        let nnXLen: NSNumber = 4

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 24)
        let x = inputPointer

        x[0] = 1; x[1] = 2; x[2]  = 0; x[3]  = 0
        x[4] = 0; x[5] = 3; x[6]  = 4; x[7]  = 0
        x[8] = 0; x[9] = 0; x[10] = 5; x[11] = 0

        x[12] = 0; x[13] = 0; x[14] = 0;  x[15] = 0
        x[16] = 0; x[17] = 5; x[18] = -3; x[19] = 0
        x[20] = 0; x[21] = -1; x[22] = 1; x[23] = 1

        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 24)
        let m = maskPointer

        m[0] = 1; m[1] = 1; m[2]  = 1; m[3]  = 0
        m[4] = 1; m[5] = 1; m[6]  = 1; m[7]  = 0
        m[8] = 1; m[9] = 1; m[10] = 1; m[11] = 0

        m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 0
        m[16] = 0; m[17] = 1; m[18] = 1; m[19] = 1
        m[20] = 0; m[21] = 1; m[22] = 1; m[23] = 1

        let preBN =
        SWBatchNormLayerDesc(numChannels: trunkChannels,
                             epsilon: 0.1,
                             hasScale: true,
                             hasBias: true,
                             mean: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                             variance: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                             scale: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                             bias: UnsafeMutablePointer<Float32>.allocate(capacity: 1))

        preBN.mean[0] = 0
        preBN.variance[0] = 0.9
        preBN.scale[0] = 1
        preBN.bias[0] = 0

        let regularConv =
        SWConvLayerDesc(convYSize: 1,
                        convXSize: 1,
                        inChannels: trunkChannels,
                        outChannels: regularChannels,
                        dilationY: 1,
                        dilationX: 1,
                        weights: UnsafeMutablePointer<Float32>.allocate(capacity: 1))

        regularConv.weights[0] = 2

        let convYSize: NSNumber = 3
        let convXSize: NSNumber = 3
        let capacity = convYSize.intValue * convXSize.intValue * gpoolChannels.intValue

        let gpoolConv =
        SWConvLayerDesc(convYSize: convYSize,
                        convXSize: convXSize,
                        inChannels: trunkChannels,
                        outChannels: gpoolChannels,
                        dilationY: 1,
                        dilationX: 1,
                        weights: UnsafeMutablePointer<Float32>.allocate(capacity: capacity))

        let w = gpoolConv.weights;

        w[0] = 0; w[1] = 0; w[2] = 0
        w[3] = 0; w[4] = 0; w[5] = 1
        w[6] = 0; w[7] = 0; w[8] = 0

        w[9] = 0; w[10] = 0; w[11] = 0
        w[12] = 1; w[13] = 0; w[14] = 0
        w[15] = 0; w[16] = 0; w[17] = 0

        let gpoolBN =
        SWBatchNormLayerDesc(numChannels: gpoolChannels,
                             epsilon: 0.1,
                             hasScale: false,
                             hasBias: false,
                             mean: UnsafeMutablePointer<Float32>.allocate(capacity: 2),
                             variance: UnsafeMutablePointer<Float32>.allocate(capacity: 2),
                             scale: UnsafeMutablePointer<Float32>.allocate(capacity: 2),
                             bias: UnsafeMutablePointer<Float32>.allocate(capacity: 2))

        gpoolBN.mean[0] = 0; gpoolBN.mean[1] = 0
        gpoolBN.variance[0] = 0.9; gpoolBN.variance[1] = 0.9
        gpoolBN.scale[0] = 1; gpoolBN.scale[1] = 1
        gpoolBN.bias[0] = 0; gpoolBN.bias[1] = -2

        let gpoolToBiasMul =
        createSWMatMulLayerDesc(inChannels: 6,
                                outChannels: 1,
                                weights: UnsafeMutablePointer<Float32>.allocate(capacity: 6))

        gpoolToBiasMul.weights[0] = 36
        gpoolToBiasMul.weights[1] = 36
        gpoolToBiasMul.weights[2] = 18
        gpoolToBiasMul.weights[3] = 18
        gpoolToBiasMul.weights[4] = 1
        gpoolToBiasMul.weights[5] = 1

        let midBN =
        SWBatchNormLayerDesc(numChannels: 1,
                             epsilon: 0.1,
                             hasScale: false,
                             hasBias: false,
                             mean: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                             variance: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                             scale: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                             bias: UnsafeMutablePointer<Float32>.allocate(capacity: 1))

        midBN.mean[0] = 0
        midBN.variance[0] = 0.9
        midBN.scale[0] = 1
        midBN.bias[0] = 0

        let finalConv =
        SWConvLayerDesc(convYSize: 1,
                        convXSize: 1,
                        inChannels: 1,
                        outChannels: 1,
                        dilationY: 1,
                        dilationX: 1,
                        weights: UnsafeMutablePointer<Float32>.allocate(capacity: 1))

        finalConv.weights[0] = 1

        let descriptor = SWGlobalPoolingResidualBlockDesc(preBN: preBN,
                                                          preActivation: ActivationKind.relu,
                                                          regularConv: regularConv,
                                                          gpoolConv: gpoolConv,
                                                          gpoolBN: gpoolBN,
                                                          gpoolActivation: ActivationKind.relu,
                                                          gpoolToBiasMul: gpoolToBiasMul,
                                                          midBN: midBN,
                                                          midActivation: ActivationKind.relu,
                                                          finalConv: finalConv)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 24)

        testGlobalPoolingResidualBlock(descriptor: descriptor,
                                       batchSize: Int32(truncating: batchSize),
                                       nnXLen: Int32(truncating: nnXLen),
                                       nnYLen: Int32(truncating: nnYLen),
                                       input: inputPointer,
                                       mask: maskPointer,
                                       output: outputPointer)

        let y = UnsafeMutablePointer<Float32>.allocate(capacity: 24)

        y[0] = 3; y[1] = 6; y[2] = 0; y[3] = 0
        y[4] = 0; y[5] = 9; y[6] = 12; y[7] = 0
        y[8] = 0; y[9] = 0; y[10] = 15; y[11] = 0

        y[12] = 0; y[13] = 0; y[14] = 0; y[15] = 0
        y[16] = 0; y[17] = 15; y[18] = -3; y[19] = 0
        y[20] = 0; y[21] = -1; y[22] = 3; y[23] = 3

        for i in 0..<12 {
            y[i] += 56 + (28 * (-11) * 0.1) + 5 + 4 + (2 * (-11) * 0.1) + 1
            y[i] *= m[i]
        }

        for i in 12..<24 {
            let sqrt6: Float32 = sqrt(6)

            y[i] += 12 + (6 * (sqrt6 - 14) * 0.1) + 1 +
            18 + (9 * (sqrt6 - 14) * 0.1) + 3

            y[i] *= m[i]
        }

        XCTAssertEqual(outputPointer[0], y[0], accuracy: 1e-4)
        XCTAssertEqual(outputPointer[3], y[3], accuracy: 1e-4)
        XCTAssertEqual(outputPointer[4], y[4], accuracy: 1e-4)
        XCTAssertEqual(outputPointer[11], y[11], accuracy: 1e-4)
        XCTAssertEqual(outputPointer[12], y[12], accuracy: 1e-4)
        XCTAssertEqual(outputPointer[18], y[18], accuracy: 1e-4)
        XCTAssertEqual(outputPointer[23], y[23], accuracy: 1e-4)
    }
}

final class NestedBottleneckResidualBlockTest: XCTestCase {

    func testFP32() {
        let batchSize = 1
        let nnXLen = 1
        let nnYLen = 1
        let numChannels = 1
        let hasScale = true
        let hasBias = true

        let graph = MPSGraph()

        let source = InputLayer(graph: graph,
                                nnXLen: nnXLen as NSNumber,
                                nnYLen: nnYLen as NSNumber,
                                numChannels: numChannels as NSNumber)

        let mask = MaskLayer(graph: graph,
                             nnXLen: nnXLen as NSNumber,
                             nnYLen: nnYLen as NSNumber)

        let maskSum = MaskSumLayer(graph: graph,
                                   maskTensor: mask.tensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum)

        let preBN = SWBatchNormLayerDesc(numChannels: numChannels as NSNumber,
                                         epsilon: 0.1,
                                         hasScale: hasScale as NSNumber,
                                         hasBias: hasBias as NSNumber,
                                         mean: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                                         variance: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                                         scale: UnsafeMutablePointer<Float32>.allocate(capacity: 1),
                                         bias: UnsafeMutablePointer<Float32>.allocate(capacity: 1))

        preBN.mean[0] = 0
        preBN.variance[0] = 0.9
        preBN.scale[0] = 1
        preBN.bias[0] = 0

        let preActivation = ActivationKind.mish

        let preConv = SWConvLayerDesc(convYSize: 1,
                                      convXSize: 1,
                                      inChannels: numChannels as NSNumber,
                                      outChannels: numChannels as NSNumber,
                                      dilationY: 1,
                                      dilationX: 1,
                                      weights: UnsafeMutablePointer<Float32>.allocate(capacity: 1))

        preConv.weights[0] = 1

        let ordinary = SWResidualBlockDesc(preBN: preBN,
                                           preActivation: preActivation,
                                           regularConv: preConv,
                                           midBN: preBN,
                                           midActivation: preActivation,
                                           finalConv: preConv)

        let nestedBottleneck = createSWNestedBottleneckResidualBlockDesc(preBN: preBN,
                                                                         preActivation: preActivation,
                                                                         preConv: preConv,
                                                                         blockDescriptors: [ordinary],
                                                                         postBN: preBN,
                                                                         postActivation: preActivation,
                                                                         postConv: preConv)

        let descriptor = SWNestedBottleneckResidualBlockDesc(preBN: preBN,
                                                             preActivation: preActivation,
                                                             preConv: preConv,
                                                             blockDescriptors: [nestedBottleneck],
                                                             postBN: preBN,
                                                             postActivation: preActivation,
                                                             postConv: preConv)

        let block = NestedBottleneckResidualBlock(graph: graph,
                                                  sourceTensor: source.tensor,
                                                  maskTensor: mask.tensor,
                                                  maskSumTensor: maskSum.tensor,
                                                  maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                                  descriptor: descriptor,
                                                  nnXLen: nnXLen as NSNumber,
                                                  nnYLen: nnYLen as NSNumber)

        let device = MTLCreateSystemDefaultDevice()!

        let inputArrayShape = InputShape.create(batchSize: batchSize as NSNumber,
                                                numChannels: numChannels as NSNumber,
                                                nnYLen: nnYLen as NSNumber,
                                                nnXLen: nnXLen as NSNumber)

        let inLength = inputArrayShape.countElements()
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inLength)
        inputPointer[0] = 1

        let sourceDescriptor = MPSNDArrayDescriptor(dataType: source.tensor.dataType,
                                                    shape: inputArrayShape)

        let sourceArray = MPSNDArray(device: device,
                                     descriptor: sourceDescriptor)

        sourceArray.writeBytes(inputPointer)
        let sourceTensorData = MPSGraphTensorData(sourceArray)

        let maskArrayShape = InputShape.create(batchSize: batchSize as NSNumber,
                                               numChannels: 1,
                                               nnYLen: nnYLen as NSNumber,
                                               nnXLen: nnXLen as NSNumber)

        let maskLength = maskArrayShape.countElements()
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskLength)
        maskPointer[0] = 1

        let maskDescriptor = MPSNDArrayDescriptor(dataType: mask.tensor.dataType,
                                                  shape: maskArrayShape)

        let maskArray = MPSNDArray(device: device,
                                   descriptor: maskDescriptor)

        maskArray.writeBytes(maskPointer)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [source.tensor: sourceTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [block.resultTensor],
                              targetOperations: nil)

        let outputArray = fetch[block.resultTensor]?.mpsndarray()
        let outLength = outputArray!.countElements()
        let outputFP32 = UnsafeMutablePointer<Float32>.allocate(capacity: outLength)
        outputArray?.readBytes(outputFP32)

        XCTAssertEqual(outputFP32[0], 2.8582418, accuracy: 1e-4)
    }
}

final class MatMulLayerTest: XCTestCase {

    func testFP32() {
        let batchSize = 2
        let nnXLen = 2
        let nnYLen = 1
        let inChannels = 2
        let outChannels = 3
        let weightsCount = inChannels * outChannels
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: weightsCount)

        for i in 0..<weightsCount {
            weights[i] = Float32(i)
        }

        /* weights = {0, 1, 2,
         *            3, 4, 5}
         */

        let descriptor = SWMatMulLayerDesc(inChannels: inChannels as NSNumber,
                                           outChannels: outChannels as NSNumber,
                                           weights: weights)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: inChannels as NSNumber)

        let matMulLayer = MatMulLayer(graph: graph,
                                      descriptor: descriptor,
                                      sourceTensor: input.tensor)

        let inputCount = batchSize * nnXLen * nnYLen * inChannels
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        /* NHWC inputPointer = {0, 1,
         *                      2, 3,
         *
         *                      4, 5,
         *                      6, 7}
         */

        /* outputPointer = {3, 9, 15, 21,
         *                  4, 14, 24, 34,
         *                  5, 19, 33, 47}
         */

        let device = MTLCreateSystemDefaultDevice()!

        let inputArrayShape = [batchSize, inChannels, nnYLen, nnXLen] as [NSNumber]
        let inputDescriptor = MPSNDArrayDescriptor(dataType: input.tensor.dataType,
                                                   shape: inputArrayShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData],
                              targetTensors: [matMulLayer.resultTensor],
                              targetOperations: nil)

        let outputCount = batchSize * nnXLen * nnYLen * outChannels
        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputCount)

        fetch[matMulLayer.resultTensor]?.mpsndarray().readBytes(outputPointer)

        XCTAssertEqual(outputPointer[0], 3, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 5, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 9, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[4], 14, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[5], 19, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[6], 15, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[7], 24, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[8], 33, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[9], 21, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[10], 34, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[11], 47, accuracy: 1e-8)
    }

    func test2D() {
        let batchSize = 2
        let inChannels = 3
        let outChannels = 4
        let weightsCount = inChannels * outChannels
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: weightsCount)

        for i in 0..<weightsCount {
            weights[i] = Float32(i)
        }

        /* weights = {0, 1, 2, 3,
         *            4, 5, 6, 7,
         *            8, 9, 10, 11}
         */

        let descriptor = SWMatMulLayerDesc(inChannels: inChannels as NSNumber,
                                           outChannels: outChannels as NSNumber,
                                           weights: weights)

        let graph = MPSGraph()

        let inputShape = [batchSize as NSNumber,
                          inChannels as NSNumber]

        let inputTensor = graph.placeholder(shape: inputShape,
                                            dataType: .float32,
                                            name: nil)

        let matMulLayer = MatMulLayer(graph: graph,
                                      descriptor: descriptor,
                                      sourceTensor: inputTensor)

        let inputCount = batchSize * inChannels
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        /* inputPointer = {0, 1, 2,
         *                 3, 4, 5}
         */

        /* outputPointer = {20, 23, 26, 29,
         *                  56, 68, 80, 92}
         */

        let device = MTLCreateSystemDefaultDevice()!

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: inputShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [inputTensor: inputTensorData],
                              targetTensors: [matMulLayer.resultTensor],
                              targetOperations: nil)

        let outputCount = batchSize * outChannels
        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputCount)

        fetch[matMulLayer.resultTensor]?.mpsndarray().readBytes(outputPointer)

        XCTAssertEqual(outputPointer[0], 20, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 23, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 26, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 29, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[4], 56, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[5], 68, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[6], 80, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[7], 92, accuracy: 1e-8)
    }

    func testUnity() {
        let batchSize = 2
        let inChannels = 1
        let outChannels = 1
        let weightsCount = inChannels * outChannels
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: weightsCount)

        for i in 0..<weightsCount {
            weights[i] = 1
        }

        /* weights = {1}
         */

        let descriptor = SWMatMulLayerDesc(inChannels: inChannels as NSNumber,
                                           outChannels: outChannels as NSNumber,
                                           weights: weights)

        let graph = MPSGraph()

        let inputShape = [batchSize as NSNumber,
                          inChannels as NSNumber]

        let inputTensor = graph.placeholder(shape: inputShape,
                                            dataType: .float32,
                                            name: nil)

        let matMulLayer = MatMulLayer(graph: graph,
                                      descriptor: descriptor,
                                      sourceTensor: inputTensor)

        let inputCount = batchSize * inChannels
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        /* inputPointer = {0, 1}
         */

        /* outputPointer = {0, 1}
         */

        let device = MTLCreateSystemDefaultDevice()!

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: inputShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [inputTensor: inputTensorData],
                              targetTensors: [matMulLayer.resultTensor],
                              targetOperations: nil)

        let outputCount = batchSize * outChannels
        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputCount)

        fetch[matMulLayer.resultTensor]?.mpsndarray().readBytes(outputPointer)

        XCTAssertEqual(outputPointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 1, accuracy: 1e-8)
    }
}

final class MatBiasLayerTest: XCTestCase {

    func testFP32() {
        let numChannels = 2
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)
        let shape = [8, 2] as [NSNumber]

        weights[0] = 1
        weights[1] = -1

        let descriptor = createSWMatBiasLayerDesc(numChannels: Int32(numChannels),
                                                  weights: weights)

        let graph = MPSGraph()

        let inputTensor = graph.placeholder(shape: [8, 2],
                                            dataType: MPSDataType.float32,
                                            name: nil)

        let matBiasLayer = MatBiasLayer(graph: graph,
                                        descriptor: descriptor,
                                        sourceTensor: inputTensor)

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 16)

        for i in 0..<16 {
            inputPointer[i] = Float32(i)
        }

        let device = MTLCreateSystemDefaultDevice()!

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: shape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [inputTensor: inputTensorData],
                              targetTensors: [matBiasLayer.resultTensor],
                              targetOperations: nil)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 16)

        fetch[matBiasLayer.resultTensor]?.mpsndarray().readBytes(outputPointer)

        XCTAssertEqual(outputPointer[0], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 3, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 2, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[15], 14, accuracy: 1e-8)
    }

    func testUnity() {
        let batchSize = 2
        let numChannels = 1
        let weightsCount = numChannels
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: weightsCount)

        for i in 0..<weightsCount {
            weights[i] = 1
        }

        /* weights = {1}
         */

        let descriptor = SWMatBiasLayerDesc(numChannels: numChannels as NSNumber,
                                            weights: weights)

        let graph = MPSGraph()

        let inputShape = [batchSize as NSNumber,
                          numChannels as NSNumber]

        let inputTensor = graph.placeholder(shape: inputShape,
                                            dataType: .float32,
                                            name: nil)

        let matBiasLayer = MatBiasLayer(graph: graph,
                                        descriptor: descriptor,
                                        sourceTensor: inputTensor)

        let inputCount = batchSize * numChannels
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        /* inputPointer = {0, 1}
         */

        /* outputPointer = {1, 2}
         */

        let device = MTLCreateSystemDefaultDevice()!

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: inputShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let fetch = graph.run(feeds: [inputTensor: inputTensorData],
                              targetTensors: [matBiasLayer.resultTensor],
                              targetOperations: nil)

        let outputCount = batchSize * numChannels
        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputCount)

        fetch[matBiasLayer.resultTensor]?.mpsndarray().readBytes(outputPointer)

        XCTAssertEqual(outputPointer[0], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 2, accuracy: 1e-8)
    }
}

final class TrunkTest: XCTestCase {

    func testUnity() {
        let batchSize = 2
        let nnXLen = 2
        let nnYLen = 2
        let numChannels = 2
        let unityConvWeights = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels * numChannels)

        unityConvWeights[0] = 1
        unityConvWeights[1] = 0
        unityConvWeights[2] = 0
        unityConvWeights[3] = 1

        let unityConv = SWConvLayerDesc(convYSize: 1,
                                        convXSize: 1,
                                        inChannels: numChannels as NSNumber,
                                        outChannels: numChannels as NSNumber,
                                        dilationY: 1,
                                        dilationX: 1,
                                        weights: unityConvWeights)

        let initialMatMulWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: numChannels * numChannels)

        initialMatMulWeights[0] = 1
        initialMatMulWeights[1] = 0
        initialMatMulWeights[2] = 0
        initialMatMulWeights[3] = 1

        let initialMatMul = SWMatMulLayerDesc(inChannels: numChannels as NSNumber,
                                              outChannels: numChannels as NSNumber,
                                              weights: initialMatMulWeights)

        let mean = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        mean[0] = 0
        mean[1] = 0

        let variance = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        variance[0] = 0.9
        variance[1] = 0.9

        let scale = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        scale[0] = 1
        scale[1] = 1

        let bias = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        bias[0] = 0
        bias[1] = 0

        let unityBN = SWBatchNormLayerDesc(numChannels: numChannels as NSNumber,
                                           epsilon: 0.1,
                                           hasScale: false,
                                           hasBias: false,
                                           mean: mean,
                                           variance: variance,
                                           scale: scale,
                                           bias: bias)

        let residualBlock = SWResidualBlockDesc(preBN: unityBN,
                                                preActivation: ActivationKind.relu,
                                                regularConv: unityConv,
                                                midBN: unityBN,
                                                midActivation: ActivationKind.relu,
                                                finalConv: unityConv)

        let gpoolToBiasCount = 3 * numChannels * numChannels
        let gpoolToBiasMulWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: 3 * numChannels * numChannels)

        for i in 0..<gpoolToBiasCount {
            gpoolToBiasMulWeights[i] = 0
        }

        let gpoolToBiasMul = SWMatMulLayerDesc(inChannels: (3 * numChannels) as NSNumber,
                                               outChannels: numChannels as NSNumber,
                                               weights: gpoolToBiasMulWeights)

        let globalPoolingResidualBlock =
        createSWGlobalPoolingResidualBlockDesc(preBN: unityBN,
                                               preActivation: ActivationKind.relu,
                                               regularConv: unityConv,
                                               gpoolConv: unityConv,
                                               gpoolBN: unityBN,
                                               gpoolActivation: ActivationKind.relu,
                                               gpoolToBiasMul: gpoolToBiasMul,
                                               midBN: unityBN,
                                               midActivation: ActivationKind.relu,
                                               finalConv: unityConv)

        let blocks = [residualBlock, globalPoolingResidualBlock]

        let descriptor = createSWTrunkDesc(version: 0,
                                           trunkNumChannels: Int32(numChannels),
                                           midNumChannels: Int32(numChannels),
                                           regularNumChannels: Int32(numChannels),
                                           gpoolNumChannels: Int32(numChannels),
                                           initialConv: unityConv,
                                           initialMatMul: initialMatMul,
                                           blockDescriptors: blocks,
                                           trunkTipBN: unityBN,
                                           trunkTipActivation: ActivationKind.relu)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: numChannels as NSNumber)

        let inputGlobal = InputGlobalLayer(graph: graph,
                                           numGlobalFeatures: numChannels as NSNumber)

        let mask = MaskLayer(graph: graph,
                             nnXLen: nnXLen as NSNumber,
                             nnYLen: nnYLen as NSNumber)

        let maskSum = MaskSumLayer(graph: graph,
                                   maskTensor: mask.tensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum)

        let trunk = Trunk(graph: graph,
                          descriptor: descriptor,
                          inputTensor: input.tensor,
                          inputGlobalTensor: inputGlobal.tensor,
                          maskTensor: mask.tensor,
                          maskSumTensor: maskSum.tensor,
                          maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                          nnXLen: nnXLen as NSNumber,
                          nnYLen: nnYLen as NSNumber,
                          numSpatialFeatures: numChannels as NSNumber,
                          numGlobalFeatures: numChannels as NSNumber)

        let inputCount = batchSize * numChannels * nnXLen * nnYLen
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        let inputGlobalCount = batchSize * numChannels

        let inputGlobalPointer =
        UnsafeMutablePointer<Float32>.allocate(capacity: inputGlobalCount)

        for i in 0..<inputGlobalCount {
            inputGlobalPointer[i] = 1
        }

        let maskCount = batchSize * nnXLen * nnYLen
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskCount)

        for i in 0..<maskCount {
            maskPointer[i] = 1
        }

        let device = MTLCreateSystemDefaultDevice()!

        let inputArrayShape = InputShape.create(batchSize: batchSize as NSNumber,
                                                numChannels: numChannels as NSNumber,
                                                nnYLen: nnYLen as NSNumber,
                                                nnXLen: nnXLen as NSNumber)

        let inputDescriptor = MPSNDArrayDescriptor(dataType: input.tensor.dataType,
                                                   shape: inputArrayShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let inputGlobalArrayShape = InputShape.create(batchSize: batchSize as NSNumber,
                                                      numChannels: numChannels as NSNumber,
                                                      nnYLen: 1,
                                                      nnXLen: 1)

        let inputGlobalDescriptor = MPSNDArrayDescriptor(dataType: inputGlobal.tensor.dataType,
                                                         shape: inputGlobalArrayShape)

        let inputGlobalArray = MPSNDArray(device: device,
                                          descriptor: inputGlobalDescriptor)

        inputGlobalArray.writeBytes(inputGlobalPointer)
        let inputGlobalTensorData = MPSGraphTensorData(inputGlobalArray)

        let maskArrayShape = InputShape.create(batchSize: batchSize as NSNumber,
                                               numChannels: 1,
                                               nnYLen: nnYLen as NSNumber,
                                               nnXLen: nnXLen as NSNumber)

        let maskDescriptor = MPSNDArrayDescriptor(dataType: mask.tensor.dataType,
                                                  shape: maskArrayShape)

        let maskArray = MPSNDArray(device: device,
                                   descriptor: maskDescriptor)

        maskArray.writeBytes(maskPointer)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData,
                                      inputGlobal.tensor: inputGlobalTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [trunk.resultTensor],
                              targetOperations: nil)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        fetch[trunk.resultTensor]?.mpsndarray().readBytes(outputPointer)

        XCTAssertEqual(outputPointer[0], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 8, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 12, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 16, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[15], 64, accuracy: 1e-8)
    }
}

final class PolicyHeadTest: XCTestCase {

    func testUnity() {
        let batchSize = 2
        let nnXLen = 2
        let nnYLen = 2
        let inChannels = 2
        let outChannels = 1

        let unityConvWeights = UnsafeMutablePointer<Float32>.allocate(capacity: inChannels * inChannels)

        unityConvWeights[0] = 1
        unityConvWeights[1] = 0
        unityConvWeights[2] = 0
        unityConvWeights[3] = 1

        let unityConv = SWConvLayerDesc(convYSize: 1,
                                        convXSize: 1,
                                        inChannels: inChannels as NSNumber,
                                        outChannels: inChannels as NSNumber,
                                        dilationY: 1,
                                        dilationX: 1,
                                        weights: unityConvWeights)

        let mean = UnsafeMutablePointer<Float32>.allocate(capacity: inChannels)

        mean[0] = 0
        mean[1] = 0

        let variance = UnsafeMutablePointer<Float32>.allocate(capacity: inChannels)

        variance[0] = 0.9
        variance[1] = 0.9

        let scale = UnsafeMutablePointer<Float32>.allocate(capacity: inChannels)

        scale[0] = 1
        scale[1] = 1

        let bias = UnsafeMutablePointer<Float32>.allocate(capacity: inChannels)

        bias[0] = 0
        bias[1] = 0

        let unityBN = SWBatchNormLayerDesc(numChannels: inChannels as NSNumber,
                                           epsilon: 0.1,
                                           hasScale: false,
                                           hasBias: false,
                                           mean: mean,
                                           variance: variance,
                                           scale: scale,
                                           bias: bias)

        let gpoolToBiasCount = 3 * inChannels * inChannels
        let gpoolToBiasMulWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: 3 * inChannels * inChannels)

        for i in 0..<gpoolToBiasCount {
            gpoolToBiasMulWeights[i] = 0
        }

        let gpoolToBiasMul = SWMatMulLayerDesc(inChannels: (3 * inChannels) as NSNumber,
                                               outChannels: inChannels as NSNumber,
                                               weights: gpoolToBiasMulWeights)

        let p2ConvWeights = UnsafeMutablePointer<Float32>.allocate(capacity: inChannels * outChannels)

        p2ConvWeights[0] = 0.5
        p2ConvWeights[1] = 0.5

        let p2Conv = SWConvLayerDesc(convYSize: 1,
                                     convXSize: 1,
                                     inChannels: inChannels as NSNumber,
                                     outChannels: outChannels as NSNumber,
                                     dilationY: 1,
                                     dilationX: 1,
                                     weights: p2ConvWeights)

        let gpoolToPassCount = 3 * inChannels * outChannels
        let gpoolToPassMulWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: 3 * inChannels * outChannels)

        for i in 0..<gpoolToPassCount {
            gpoolToPassMulWeights[i] = 1
        }

        let gpoolToPassMul = SWMatMulLayerDesc(inChannels: (3 * inChannels) as NSNumber,
                                               outChannels: outChannels as NSNumber,
                                               weights: gpoolToPassMulWeights)

        let descriptor = createSWPolicyHeadDesc(version: 0,
                                                p1Conv: unityConv,
                                                g1Conv: unityConv,
                                                g1BN: unityBN,
                                                g1Activation: ActivationKind.relu,
                                                gpoolToBiasMul: gpoolToBiasMul,
                                                p1BN: unityBN,
                                                p1Activation: ActivationKind.relu,
                                                p2Conv: p2Conv,
                                                gpoolToPassMul: gpoolToPassMul)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: inChannels as NSNumber)

        let mask = MaskLayer(graph: graph,
                             nnXLen: nnXLen as NSNumber,
                             nnYLen: nnYLen as NSNumber)

        let maskSum = MaskSumLayer(graph: graph,
                                   maskTensor: mask.tensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum)

        let policyHead = PolicyHead(graph: graph,
                                    descriptor: descriptor,
                                    sourceTensor: input.tensor,
                                    maskTensor: mask.tensor,
                                    maskSumTensor: maskSum.tensor,
                                    maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                    nnXLen: nnXLen as NSNumber,
                                    nnYLen: nnYLen as NSNumber)

        let inputCount = batchSize * inChannels * nnXLen * nnYLen
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        let maskCount = batchSize * nnXLen * nnYLen
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskCount)

        for i in 0..<maskCount {
            maskPointer[i] = 1
        }

        let device = MTLCreateSystemDefaultDevice()!

        let inputArrayShape = [batchSize, inChannels, nnYLen, nnXLen] as [NSNumber]
        let inputDescriptor = MPSNDArrayDescriptor(dataType: input.tensor.dataType,
                                                   shape: inputArrayShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let maskArrayShape = [batchSize, 1, nnYLen, nnXLen] as [NSNumber]
        let maskDescriptor = MPSNDArrayDescriptor(dataType: mask.tensor.dataType,
                                                  shape: maskArrayShape)

        let maskArray = MPSNDArray(device: device,
                                   descriptor: maskDescriptor)

        maskArray.writeBytes(maskPointer)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [policyHead.policyTensor,
                                              policyHead.policyPassTensor],
                              targetOperations: nil)

        let policyCount = batchSize * outChannels * nnXLen * nnYLen
        let policyPointer = UnsafeMutablePointer<Float32>.allocate(capacity: policyCount)

        fetch[policyHead.policyTensor]?.mpsndarray().readBytes(policyPointer)

        let policyPassCount = batchSize

        let policyPassPointer = UnsafeMutablePointer<Float32>.allocate(capacity: policyPassCount)

        fetch[policyHead.policyPassTensor]?.mpsndarray().readBytes(policyPassPointer)

        XCTAssertEqual(policyPointer[0], 2, accuracy: 1e-8)
        XCTAssertEqual(policyPointer[1], 3, accuracy: 1e-8)
        XCTAssertEqual(policyPointer[2], 4, accuracy: 1e-8)
        XCTAssertEqual(policyPointer[3], 5, accuracy: 1e-8)
        XCTAssertEqual(policyPointer[4], 10, accuracy: 1e-8)
        XCTAssertEqual(policyPointer[5], 11, accuracy: 1e-8)
        XCTAssertEqual(policyPointer[6], 12, accuracy: 1e-8)
        XCTAssertEqual(policyPointer[7], 13, accuracy: 1e-8)
        XCTAssertEqual(policyPassPointer[0], 8.6, accuracy: 1e-4)
        XCTAssertEqual(policyPassPointer[1], 21.4, accuracy: 1e-4)
    }
}

final class ComboLayerTest: XCTestCase {

    func testMatMulBiasLayer() {
        let graph = MPSGraph()
        let inputShape = [3, 2] as [NSNumber]

        let inputTensor = graph.placeholder(shape: inputShape,
                                            dataType: .float32,
                                            name: nil)

        let mulTensor = graph.constant(0,
                                       shape: [2, 1],
                                       dataType: .float32)

        let matMulTensor = graph.matrixMultiplication(primary: inputTensor,
                                                      secondary: mulTensor,
                                                      name: nil)

        let biasTensor = graph.constant(0,
                                        shape: [1, 1],
                                        dataType: .float32)

        let matBiasTensor = graph.addition(matMulTensor,
                                           biasTensor,
                                           name: nil)

        let device = MTLCreateSystemDefaultDevice()!

        let inputDescriptor = MPSNDArrayDescriptor(dataType: inputTensor.dataType,
                                                   shape: inputShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        let inputTensorData = MPSGraphTensorData(inputArray)

        graph.run(feeds: [inputTensor: inputTensorData],
                  targetTensors: [matBiasTensor],
                  targetOperations: nil)

        XCTAssert(matMulTensor.shape! == [3, 1])
        XCTAssert(matBiasTensor.shape! == [3, 1])
    }
}

final class ValueHeadTest: XCTestCase {

    func testZero() {
        let batchSize = 2
        let nnXLen = 2
        let nnYLen = 2
        let inChannels = 1
        let v1OutChannels = 2
        let v2OutChannels = 2
        let v3OutChannels = 1

        let v1ConvCount = inChannels * v1OutChannels
        let v1ConvWeights = UnsafeMutablePointer<Float32>.allocate(capacity: v1ConvCount)

        for i in 0..<v1ConvCount {
            v1ConvWeights[i] = 0
        }

        let v1Conv = SWConvLayerDesc(convYSize: 1,
                                     convXSize: 1,
                                     inChannels: inChannels as NSNumber,
                                     outChannels: v1OutChannels as NSNumber,
                                     dilationY: 1,
                                     dilationX: 1,
                                     weights: v1ConvWeights)

        let mean = UnsafeMutablePointer<Float32>.allocate(capacity: v1OutChannels)

        mean[0] = 0
        mean[1] = 0

        let variance = UnsafeMutablePointer<Float32>.allocate(capacity: v1OutChannels)

        variance[0] = 0.9
        variance[1] = 0.9

        let scale = UnsafeMutablePointer<Float32>.allocate(capacity: v1OutChannels)

        scale[0] = 1
        scale[1] = 1

        let bias = UnsafeMutablePointer<Float32>.allocate(capacity: v1OutChannels)

        bias[0] = 0
        bias[1] = 0

        let v1BN = SWBatchNormLayerDesc(numChannels: v1OutChannels as NSNumber,
                                        epsilon: 0.1,
                                        hasScale: false,
                                        hasBias: false,
                                        mean: mean,
                                        variance: variance,
                                        scale: scale,
                                        bias: bias)

        let v2MulCount = 3 * v1OutChannels * v2OutChannels
        let v2MulWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: v2MulCount)

        for i in 0..<v2MulCount {
            v2MulWeights[i] = 0
        }

        let v2Mul = SWMatMulLayerDesc(inChannels: (3 * v1OutChannels) as NSNumber,
                                      outChannels: v2OutChannels as NSNumber,
                                      weights: v2MulWeights)

        let v2BiasWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: v2OutChannels)

        for i in 0..<v2OutChannels {
            v2BiasWeights[i] = 0
        }

        let v2Bias = SWMatBiasLayerDesc(numChannels: v2OutChannels as NSNumber,
                                        weights: v2BiasWeights)

        let v3MulCount = v2OutChannels * v3OutChannels
        let v3MulWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: v3MulCount)

        for i in 0..<v3MulCount {
            v3MulWeights[i] = 0
        }

        let v3Mul = SWMatMulLayerDesc(inChannels: v2OutChannels as NSNumber,
                                      outChannels: v3OutChannels as NSNumber,
                                      weights: v3MulWeights)

        let v3BiasWeights =
        UnsafeMutablePointer<Float32>.allocate(capacity: v3OutChannels)

        for i in 0..<v3OutChannels {
            v3BiasWeights[i] = 0
        }

        let v3Bias = SWMatBiasLayerDesc(numChannels: v3OutChannels as NSNumber,
                                        weights: v3BiasWeights)

        let sv3Mul = v3Mul
        let sv3Bias = v3Bias

        let vOwnershipConvCount = v1OutChannels * v3OutChannels
        let vOwnershipConvWeights = UnsafeMutablePointer<Float32>.allocate(capacity: vOwnershipConvCount)

        for i in 0..<vOwnershipConvCount {
            vOwnershipConvWeights[i] = 0
        }

        let vOwnershipConv = SWConvLayerDesc(convYSize: 1,
                                             convXSize: 1,
                                             inChannels: v1OutChannels as NSNumber,
                                             outChannels: v3OutChannels as NSNumber,
                                             dilationY: 1,
                                             dilationX: 1,
                                             weights: vOwnershipConvWeights)

        let descriptor = createSWValueHeadDesc(version: 0,
                                               v1Conv: v1Conv,
                                               v1BN: v1BN,
                                               v1Activation: ActivationKind.relu,
                                               v2Mul: v2Mul,
                                               v2Bias: v2Bias,
                                               v2Activation: ActivationKind.relu,
                                               v3Mul: v3Mul,
                                               v3Bias: v3Bias,
                                               sv3Mul: sv3Mul,
                                               sv3Bias: sv3Bias,
                                               vOwnershipConv: vOwnershipConv)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: inChannels as NSNumber)

        let mask = MaskLayer(graph: graph,
                             nnXLen: nnXLen as NSNumber,
                             nnYLen: nnYLen as NSNumber)

        let maskSum = MaskSumLayer(graph: graph,
                                   maskTensor: mask.tensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum)

        let maskSumSqrtS14M01SquareS01 =
        MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                        maskSumSqrtS14M01: maskSumSqrtS14M01)

        let valueHead = ValueHead(graph: graph,
                                  descriptor: descriptor,
                                  sourceTensor: input.tensor,
                                  maskTensor: mask.tensor,
                                  maskSumTensor: maskSum.tensor,
                                  maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                                  maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
                                  nnXLen: nnXLen as NSNumber,
                                  nnYLen: nnYLen as NSNumber)

        let inputCount = batchSize * inChannels * nnXLen * nnYLen
        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float32(i)
        }

        let maskCount = batchSize * nnXLen * nnYLen
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskCount)

        for i in 0..<maskCount {
            maskPointer[i] = 1
        }

        let device = MTLCreateSystemDefaultDevice()!

        let inputArrayShape = [batchSize, inChannels, nnYLen, nnXLen] as [NSNumber]
        let inputDescriptor = MPSNDArrayDescriptor(dataType: input.tensor.dataType,
                                                   shape: inputArrayShape)

        let inputArray = MPSNDArray(device: device,
                                    descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)
        let inputTensorData = MPSGraphTensorData(inputArray)

        let maskArrayShape = [batchSize, 1, nnYLen, nnXLen] as [NSNumber]
        let maskDescriptor = MPSNDArrayDescriptor(dataType: mask.tensor.dataType,
                                                  shape: maskArrayShape)
        
        let maskArray = MPSNDArray(device: device,
                                   descriptor: maskDescriptor)

        maskArray.writeBytes(maskPointer)
        let maskTensorData = MPSGraphTensorData(maskArray)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [valueHead.valueTensor,
                                              valueHead.scoreValueTensor,
                                              valueHead.ownershipTensor],
                              targetOperations: nil)

        let valueCount = batchSize * v3OutChannels
        let valuePointer = UnsafeMutablePointer<Float32>.allocate(capacity: valueCount)

        fetch[valueHead.valueTensor]?.mpsndarray().readBytes(valuePointer)

        let scoreValueCount = batchSize * v3OutChannels
        let scoreValuePointer = UnsafeMutablePointer<Float32>.allocate(capacity: scoreValueCount)

        fetch[valueHead.scoreValueTensor]?.mpsndarray().readBytes(scoreValuePointer)

        let ownershipCount = batchSize * nnXLen * nnYLen * v3OutChannels
        let ownershipPointer = UnsafeMutablePointer<Float32>.allocate(capacity: ownershipCount)

        fetch[valueHead.ownershipTensor]?.mpsndarray().readBytes(ownershipPointer)

        XCTAssertEqual(valuePointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(valuePointer[1], 0, accuracy: 1e-8)
        XCTAssertEqual(scoreValuePointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(scoreValuePointer[1], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[1], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[2], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[3], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[4], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[5], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[6], 0, accuracy: 1e-8)
        XCTAssertEqual(ownershipPointer[7], 0, accuracy: 1e-8)
    }
}

final class SWModelDescTest {

    var unityConvWeights = [Float](repeating: 1, count: 1)
    var unityMatMulWeights = [Float](repeating: 1, count: 1)
    var meanWeights = [Float](repeating: 0, count: 1)
    var varianceWeights = [Float](repeating: 0.9, count: 1)
    var scaleWeights = [Float](repeating: 1, count: 1)
    var biasWeights = [Float](repeating: 0, count: 1)
    var gpoolMatMulWeights = [Float](repeating: 3, count: 3)
    var zeroMatBiasWeights = [Float](repeating: 0, count: 1)

    func createMiniDesc() -> SWModelDesc {
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
                                          gpoolToPassMul: gpoolMatMul)

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

        let modelDesc = createSWModelDesc(version: 0,
                                          name: "test",
                                          numInputChannels: 1,
                                          numInputGlobalChannels: 1,
                                          numValueChannels: 1,
                                          numScoreValueChannels: 1,
                                          numOwnershipChannels: 1,
                                          trunk: trunkDesc,
                                          policyHead: policyHead,
                                          valueHead: valueHead)

        return modelDesc
    }
}

final class ModelTest: XCTestCase {
    let swModelDescTest = SWModelDescTest()

    func createMiniModel() -> Model? {
        let modelDesc = swModelDescTest.createMiniDesc()

        let device = MTLCreateSystemDefaultDevice()!

        let model = Model(device: device,
                          graph: MPSGraph(),
                          descriptor: modelDesc,
                          nnXLen: 1,
                          nnYLen: 1)

        var input = [Float32](repeating: 1, count: 1)
        var inputGlobal = [Float32](repeating: 1, count: 1)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model.apply(input: &input,
                    inputGlobal: &inputGlobal,
                    policy: &policyOutput,
                    policyPass: &policyPassOutput,
                    value: &valueOutput,
                    scoreValue: &scoreValueOutput,
                    ownership: &ownershipOutput,
                    batchSize: 1)

        return model
    }

    func testMiniModel() {
        let model = createMiniModel()
        var input = [Float32](repeating: 1, count: 1)
        var inputGlobal = [Float32](repeating: 1, count: 1)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model?.apply(input: &input,
                     inputGlobal: &inputGlobal,
                     policy: &policyOutput,
                     policyPass: &policyPassOutput,
                     value: &valueOutput,
                     scoreValue: &scoreValueOutput,
                     ownership: &ownershipOutput,
                     batchSize: 1)

        XCTAssertEqual(policyOutput[0], 101.68, accuracy: 1e-4)
        XCTAssertEqual(policyPassOutput[0], 68.88, accuracy: 1e-4)
        XCTAssertEqual(valueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(scoreValueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(ownershipOutput[0], 32.8, accuracy: 1e-4)
    }

    func testMiniModelNHWC() {
        let model = createMiniModel()
        var input = [Float32](repeating: 1, count: 1)
        var inputGlobal = [Float32](repeating: 1, count: 1)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model?.apply(input: &input,
                     inputGlobal: &inputGlobal,
                     policy: &policyOutput,
                     policyPass: &policyPassOutput,
                     value: &valueOutput,
                     scoreValue: &scoreValueOutput,
                     ownership: &ownershipOutput,
                     batchSize: 1)

        XCTAssertEqual(policyOutput[0], 101.68, accuracy: 1e-4)
        XCTAssertEqual(policyPassOutput[0], 68.88, accuracy: 1e-4)
        XCTAssertEqual(valueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(scoreValueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(ownershipOutput[0], 32.8, accuracy: 1e-4)
    }

    func createModelB40C256(batchSize: Int,
                            nnYLen: Int,
                            nnXLen: Int,
                            numInputChannels: Int,
                            numInputGlobalChannels: Int,
                            numValueChannels: Int,
                            numScoreValueChannels: Int,
                            numOwnershipChannels: Int) -> Model? {
        let version = 10
        let convCount = 3 * 3 * 256 * 256
        let normCount = 256
        let randomWeights = UnsafeMutablePointer<Float32>.allocate(capacity: convCount)
        let oneWeights = UnsafeMutablePointer<Float32>.allocate(capacity: normCount)

        for i in 0..<normCount {
            oneWeights[i] = 1
        }

        let initialConv = SWConvLayerDesc(convYSize: 5,
                                          convXSize: 5,
                                          inChannels: 22,
                                          outChannels: 256,
                                          dilationY: 1,
                                          dilationX: 1,
                                          weights: randomWeights)

        let initialMatMul = SWMatMulLayerDesc(inChannels: 19,
                                              outChannels: 256,
                                              weights: randomWeights)

        let preBN = SWBatchNormLayerDesc(numChannels: 256,
                                         epsilon: 1e-20,
                                         hasScale: false,
                                         hasBias: true,
                                         mean: randomWeights,
                                         variance: oneWeights,
                                         scale: randomWeights,
                                         bias: randomWeights)

        let regularConv = SWConvLayerDesc(convYSize: 3,
                                          convXSize: 3,
                                          inChannels: 256,
                                          outChannels: 256,
                                          dilationY: 1,
                                          dilationX: 1,
                                          weights: randomWeights)

        let midBN = SWBatchNormLayerDesc(numChannels: 256,
                                         epsilon: 1e-20,
                                         hasScale: true,
                                         hasBias: true,
                                         mean: randomWeights,
                                         variance: oneWeights,
                                         scale: randomWeights,
                                         bias: randomWeights)

        let finalConv = SWConvLayerDesc(convYSize: 3,
                                        convXSize: 3,
                                        inChannels: 256,
                                        outChannels: 256,
                                        dilationY: 1,
                                        dilationX: 1,
                                        weights: randomWeights)

        let ordinary = SWResidualBlockDesc(preBN: preBN,
                                           preActivation: ActivationKind.relu,
                                           regularConv: regularConv,
                                           midBN: midBN,
                                           midActivation: ActivationKind.relu,
                                           finalConv: finalConv)

        let gRegularConv = SWConvLayerDesc(convYSize: 3,
                                           convXSize: 3,
                                           inChannels: 256,
                                           outChannels: 192,
                                           dilationY: 1,
                                           dilationX: 1,
                                           weights: randomWeights)

        let gpoolConv = SWConvLayerDesc(convYSize: 3,
                                        convXSize: 3,
                                        inChannels: 256,
                                        outChannels: 64,
                                        dilationY: 1,
                                        dilationX: 1,
                                        weights: randomWeights)

        let gpoolBN = SWBatchNormLayerDesc(numChannels: 64,
                                           epsilon: 1e-20,
                                           hasScale: false,
                                           hasBias: true,
                                           mean: randomWeights,
                                           variance: oneWeights,
                                           scale: randomWeights,
                                           bias: randomWeights)

        let gpoolToBiasMul = SWMatMulLayerDesc(inChannels: 192,
                                               outChannels: 192,
                                               weights: randomWeights)

        let gMidBN = SWBatchNormLayerDesc(numChannels: 192,
                                          epsilon: 1e-20,
                                          hasScale: true,
                                          hasBias: true,
                                          mean: randomWeights,
                                          variance: oneWeights,
                                          scale: randomWeights,
                                          bias: randomWeights)

        let gFinalConv = SWConvLayerDesc(convYSize: 3,
                                         convXSize: 3,
                                         inChannels: 192,
                                         outChannels: 256,
                                         dilationY: 1,
                                         dilationX: 1,
                                         weights: randomWeights)

        let globalPooling =
        SWGlobalPoolingResidualBlockDesc(preBN: preBN,
                                         preActivation: ActivationKind.relu,
                                         regularConv: gRegularConv,
                                         gpoolConv: gpoolConv,
                                         gpoolBN: gpoolBN,
                                         gpoolActivation: ActivationKind.relu,
                                         gpoolToBiasMul: gpoolToBiasMul,
                                         midBN: gMidBN,
                                         midActivation: ActivationKind.relu,
                                         finalConv: gFinalConv)

        let blocks: [BlockDescriptor] = [ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         globalPooling,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         globalPooling,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         globalPooling,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         globalPooling,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         globalPooling,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         globalPooling,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         globalPooling,
                                         ordinary,
                                         ordinary,
                                         ordinary,
                                         ordinary]

        assert(blocks.count == 40)

        let trunkTipBN = SWBatchNormLayerDesc(numChannels: 256,
                                              epsilon: 1e-20,
                                              hasScale: false,
                                              hasBias: true,
                                              mean: randomWeights,
                                              variance: oneWeights,
                                              scale: randomWeights,
                                              bias: randomWeights)

        let trunkDesc = SWTrunkDesc(version: version,
                                    trunkNumChannels: 256,
                                    midNumChannels: 256,
                                    regularNumChannels: 192,
                                    gpoolNumChannels: 64,
                                    initialConv: initialConv,
                                    initialMatMul: initialMatMul,
                                    blockDescriptors: blocks,
                                    trunkTipBN: trunkTipBN,
                                    trunkTipActivation: ActivationKind.relu)

        let p1Conv = SWConvLayerDesc(convYSize: 1,
                                     convXSize: 1,
                                     inChannels: 256,
                                     outChannels: 48,
                                     dilationY: 1,
                                     dilationX: 1,
                                     weights: randomWeights)

        let g1Conv = SWConvLayerDesc(convYSize: 1,
                                     convXSize: 1,
                                     inChannels: 256,
                                     outChannels: 48,
                                     dilationY: 1,
                                     dilationX: 1,
                                     weights: randomWeights)

        let g1BN = SWBatchNormLayerDesc(numChannels: 48,
                                        epsilon: 1e-20,
                                        hasScale: false,
                                        hasBias: true,
                                        mean: randomWeights,
                                        variance: oneWeights,
                                        scale: randomWeights,
                                        bias: randomWeights)

        let g1PoolToBiasMul = SWMatMulLayerDesc(inChannels: 144,
                                                outChannels: 48,
                                                weights: randomWeights)

        let p1BN = SWBatchNormLayerDesc(numChannels: 48,
                                        epsilon: 1e-20,
                                        hasScale: false,
                                        hasBias: true,
                                        mean: randomWeights,
                                        variance: oneWeights,
                                        scale: randomWeights,
                                        bias: randomWeights)

        let p2Conv = SWConvLayerDesc(convYSize: 1,
                                     convXSize: 1,
                                     inChannels: 48,
                                     outChannels: 1,
                                     dilationY: 1,
                                     dilationX: 1,
                                     weights: randomWeights)

        let gpoolToPassMul = SWMatMulLayerDesc(inChannels: 144,
                                               outChannels: 1,
                                               weights: randomWeights)

        let policyHead = SWPolicyHeadDesc(version: version,
                                          p1Conv: p1Conv,
                                          g1Conv: g1Conv,
                                          g1BN: g1BN,
                                          g1Activation: ActivationKind.relu,
                                          gpoolToBiasMul: g1PoolToBiasMul,
                                          p1BN: p1BN,
                                          p1Activation: ActivationKind.relu,
                                          p2Conv: p2Conv,
                                          gpoolToPassMul: gpoolToPassMul)

        let v1Conv = SWConvLayerDesc(convYSize: 1,
                                     convXSize: 1,
                                     inChannels: 256,
                                     outChannels: 48,
                                     dilationY: 1,
                                     dilationX: 1,
                                     weights: randomWeights)

        let v1BN = SWBatchNormLayerDesc(numChannels: 48,
                                        epsilon: 1e-20,
                                        hasScale: false,
                                        hasBias: true,
                                        mean: randomWeights,
                                        variance: oneWeights,
                                        scale: randomWeights,
                                        bias: randomWeights)

        let v2Mul = SWMatMulLayerDesc(inChannels: 144,
                                      outChannels: 128,
                                      weights: randomWeights)

        let v2Bias = SWMatBiasLayerDesc(numChannels: 128, weights: randomWeights)
        let v3Mul = SWMatMulLayerDesc(inChannels: 128, outChannels: 3, weights: randomWeights)
        let v3Bias = SWMatBiasLayerDesc(numChannels: 3, weights: randomWeights)
        let sv3Mul = SWMatMulLayerDesc(inChannels: 128, outChannels: 6, weights: randomWeights)
        let sv3Bias = SWMatBiasLayerDesc(numChannels: 6, weights: randomWeights)

        let vOwnershipConv = SWConvLayerDesc(convYSize: 1,
                                             convXSize: 1,
                                             inChannels: 48,
                                             outChannels: 1,
                                             dilationY: 1,
                                             dilationX: 1,
                                             weights: randomWeights)

        let valueHead = SWValueHeadDesc(version: version,
                                        v1Conv: v1Conv,
                                        v1BN: v1BN,
                                        v1Activation: ActivationKind.relu,
                                        v2Mul: v2Mul,
                                        v2Bias: v2Bias,
                                        v2Activation: ActivationKind.relu,
                                        v3Mul: v3Mul,
                                        v3Bias: v3Bias,
                                        sv3Mul: sv3Mul,
                                        sv3Bias: sv3Bias,
                                        vOwnershipConv: vOwnershipConv)

        let modelDesc = SWModelDesc(version: version,
                                    name: "test",
                                    numInputChannels: numInputChannels as NSNumber,
                                    numInputGlobalChannels: numInputGlobalChannels as NSNumber,
                                    numValueChannels: numValueChannels as NSNumber,
                                    numScoreValueChannels: numScoreValueChannels as NSNumber,
                                    numOwnershipChannels: numOwnershipChannels as NSNumber,
                                    trunk: trunkDesc,
                                    policyHead: policyHead,
                                    valueHead: valueHead)

        let device = MTLCreateSystemDefaultDevice()!

        let model = Model(device: device,
                          graph: MPSGraph(),
                          descriptor: modelDesc,
                          nnXLen: nnXLen as NSNumber,
                          nnYLen: nnYLen as NSNumber)

        // warm up to speed up later runs
        let inputCount = batchSize * nnYLen * nnXLen * numInputChannels
        let input = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)
        let inputGlobalCount = batchSize * numInputGlobalChannels
        let inputGlobal = UnsafeMutablePointer<Float32>.allocate(capacity: inputGlobalCount)
        let policyCount = batchSize * nnYLen * nnXLen
        let policyOutput = UnsafeMutablePointer<Float32>.allocate(capacity: policyCount)
        let policyPassCount = batchSize
        let policyPassOutput = UnsafeMutablePointer<Float32>.allocate(capacity: policyPassCount)
        let valueCount = batchSize * numValueChannels
        let valueOutput = UnsafeMutablePointer<Float32>.allocate(capacity: valueCount)
        let scoreValueCount = batchSize * numScoreValueChannels
        let scoreValueOutput = UnsafeMutablePointer<Float32>.allocate(capacity: scoreValueCount)
        let ownershipCount = batchSize * nnYLen * nnXLen * numOwnershipChannels
        let ownershipOutput = UnsafeMutablePointer<Float32>.allocate(capacity: ownershipCount)

        model.apply(input: input,
                    inputGlobal: inputGlobal,
                    policy: policyOutput,
                    policyPass: policyPassOutput,
                    value: valueOutput,
                    scoreValue: scoreValueOutput,
                    ownership: ownershipOutput,
                    batchSize: batchSize)

        return model
    }

    func createBuffers(batchSize: Int,
                       nnYLen: Int,
                       nnXLen: Int,
                       numInputChannels: Int,
                       numInputGlobalChannels: Int,
                       numValueChannels: Int,
                       numScoreValueChannels: Int,
                       numOwnershipChannels: Int) -> (UnsafeMutablePointer<Float32>,
                                                      UnsafeMutablePointer<Float32>,
                                                      UnsafeMutablePointer<Float32>,
                                                      UnsafeMutablePointer<Float32>,
                                                      UnsafeMutablePointer<Float32>,
                                                      UnsafeMutablePointer<Float32>,
                                                      UnsafeMutablePointer<Float32>) {

        let inputCount = batchSize * nnYLen * nnXLen * numInputChannels
        let inputGlobalCount = batchSize * numInputGlobalChannels
        let policyCount = batchSize * nnYLen * nnXLen
        let policyPassCount = batchSize
        let valueCount = batchSize * numValueChannels
        let scoreValueCount = batchSize * numScoreValueChannels
        let ownershipCount = batchSize * nnYLen * nnXLen * numOwnershipChannels

        return (UnsafeMutablePointer<Float32>.allocate(capacity: inputCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: inputGlobalCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: policyCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: policyPassCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: valueCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: scoreValueCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: ownershipCount))
    }

    // Test 40 blocks, 256 channels, 8 batches
    func testB40C256B8() {
        let batchSize = 8
        let nnYLen = 19
        let nnXLen = 19
        let numInputChannels = 22
        let numInputGlobalChannels = 19
        let numValueChannels = 3
        let numScoreValueChannels = 6
        let numOwnershipChannels = 1
        let numEvals = 16
        let iteration: Int = (numEvals + batchSize - 1) / batchSize

        let model = createModelB40C256(batchSize: batchSize,
                                       nnYLen: nnYLen,
                                       nnXLen: nnXLen,
                                       numInputChannels: numInputChannels,
                                       numInputGlobalChannels: numInputGlobalChannels,
                                       numValueChannels: numValueChannels,
                                       numScoreValueChannels: numScoreValueChannels,
                                       numOwnershipChannels: numOwnershipChannels)

        let (input, inputGlobal, policy, policyPass, value, scoreValue, ownership) =
        createBuffers(batchSize: batchSize,
                      nnYLen: nnYLen,
                      nnXLen: nnXLen,
                      numInputChannels: numInputChannels,
                      numInputGlobalChannels: numInputGlobalChannels,
                      numValueChannels: numValueChannels,
                      numScoreValueChannels: numScoreValueChannels,
                      numOwnershipChannels: numOwnershipChannels)

        measure {
            for _ in 0..<iteration {
                model?.apply(input: input,
                             inputGlobal: inputGlobal,
                             policy: policy,
                             policyPass: policyPass,
                             value: value,
                             scoreValue: scoreValue,
                             ownership: ownership,
                             batchSize: batchSize)
            }
        }
    }

    // Test 40 blocks, 256 channels, 16 batches
    func testB40C256B16() {
        let batchSize = 16
        let nnYLen = 19
        let nnXLen = 19
        let numInputChannels = 22
        let numInputGlobalChannels = 19
        let numValueChannels = 3
        let numScoreValueChannels = 6
        let numOwnershipChannels = 1
        let numEvals = 16
        let iteration: Int = (numEvals + batchSize - 1) / batchSize

        let model = createModelB40C256(batchSize: batchSize,
                                       nnYLen: nnYLen,
                                       nnXLen: nnXLen,
                                       numInputChannels: numInputChannels,
                                       numInputGlobalChannels: numInputGlobalChannels,
                                       numValueChannels: numValueChannels,
                                       numScoreValueChannels: numScoreValueChannels,
                                       numOwnershipChannels: numOwnershipChannels)

        let (input, inputGlobal, policy, policyPass, value, scoreValue, ownership) =
        createBuffers(batchSize: batchSize,
                      nnYLen: nnYLen,
                      nnXLen: nnXLen,
                      numInputChannels: numInputChannels,
                      numInputGlobalChannels: numInputGlobalChannels,
                      numValueChannels: numValueChannels,
                      numScoreValueChannels: numScoreValueChannels,
                      numOwnershipChannels: numOwnershipChannels)

        measure {
            for _ in 0..<iteration {
                model?.apply(input: input,
                             inputGlobal: inputGlobal,
                             policy: policy,
                             policyPass: policyPass,
                             value: value,
                             scoreValue: scoreValue,
                             ownership: ownership,
                             batchSize: batchSize)
            }
        }
    }
}

final class ComputeContextTest: XCTestCase {

    func testCreateInstance() {
        let nnXLen: NSNumber = 9
        let nnYLen: NSNumber = 11
        let useFP16Mode: SWEnable = .False
        let useNHWCMode: SWEnable = .False

        createMetalContext(nnXLen: Int32(truncating: nnXLen),
                           nnYLen: Int32(truncating: nnYLen),
                           useFP16Mode: useFP16Mode,
                           useNHWCMode: useNHWCMode)

        let context = MetalComputeContext.getInstance()

        XCTAssert(context.nnXLen == nnXLen)
        XCTAssert(context.nnYLen == nnYLen)
    }

    func testDestroyInstance() {
        let nnXLen: NSNumber = 9
        let nnYLen: NSNumber = 11
        let useFP16Mode: SWEnable = .False
        let useNHWCMode: SWEnable = .False

        MetalComputeContext.createInstance(nnXLen: nnXLen,
                                           nnYLen: nnYLen,
                                           useFP16Mode: useFP16Mode,
                                           useNHWCMode: useNHWCMode)

        destroyMetalContext()

        let context = MetalComputeContext.getInstance()

        XCTAssert(context.nnXLen == MetalComputeContext.defaultNnXLen)
        XCTAssert(context.nnYLen == MetalComputeContext.defaultNnYLen)
    }
}

final class ComputeHandleTest: XCTestCase {
    let swModelDescTest = SWModelDescTest()

    func testCreateInstance() {
        MetalComputeContext.createInstance(nnXLen: 9 as NSNumber,
                                           nnYLen: 11 as NSNumber,
                                           useFP16Mode: .False,
                                           useNHWCMode: .False)

        let swModelDesc = swModelDescTest.createMiniDesc()

        createMetalComputeHandle(descriptor: swModelDesc,
                                 serverThreadIdx: 0)

        let handle = MetalComputeHandle.handle
        let context = MetalComputeContext.getInstance()

        XCTAssert(handle?.model.nnXLen == context.nnXLen)
        XCTAssert(handle?.model.nnYLen == context.nnYLen)
        XCTAssert(handle?.model.version == swModelDesc.version)
        XCTAssert(handle?.model.numInputChannels == swModelDesc.numInputChannels)
        XCTAssert(handle?.model.numInputGlobalChannels == swModelDesc.numInputGlobalChannels)
        XCTAssert(handle?.model.numValueChannels == swModelDesc.numValueChannels)
        XCTAssert(handle?.model.numScoreValueChannels == swModelDesc.numScoreValueChannels)
        XCTAssert(handle?.model.numOwnershipChannels == swModelDesc.numOwnershipChannels)
    }
}

final class MetalBackendTest: XCTestCase {
    let swModelDescTest = SWModelDescTest()

    func testPrintDevices() {
        printMetalDevices()
    }

    func testGetContextXLen() {
        let nnXLen: Int = 9
        let nnYLen: Int = 11

        MetalComputeContext.createInstance(nnXLen: nnXLen as NSNumber,
                                           nnYLen: nnYLen as NSNumber,
                                           useFP16Mode: .False,
                                           useNHWCMode: .False)

        XCTAssert(getMetalContextXLen() == nnXLen)
    }

    func testGetContextYLen() {
        let nnXLen: Int = 9
        let nnYLen: Int = 11

        MetalComputeContext.createInstance(nnXLen: nnXLen as NSNumber,
                                           nnYLen: nnYLen as NSNumber,
                                           useFP16Mode: .False,
                                           useNHWCMode: .False)

        XCTAssert(getMetalContextYLen() == nnYLen)
    }

    func testGetOutput() {
        MetalComputeContext.createInstance(nnXLen: 1 as NSNumber,
                                           nnYLen: 1 as NSNumber,
                                           useFP16Mode: .False,
                                           useNHWCMode: .False)

        let swModelDesc = swModelDescTest.createMiniDesc()

        MetalComputeHandle.createInstance(descriptor: swModelDesc,
                                          serverThreadIdx: 0)
        
        var input = [Float32](repeating: 1, count: 1)
        var inputGlobal = [Float32](repeating: 1, count: 1)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        getMetalHandleOutput(userInputBuffer: &input,
                             userInputGlobalBuffer: &inputGlobal,
                             policyOutput: &policyOutput,
                             policyPassOutput: &policyPassOutput,
                             valueOutput: &valueOutput,
                             ownershipOutput: &ownershipOutput,
                             scoreValueOutput: &scoreValueOutput,
                             batchSize: 1)

        XCTAssertEqual(policyOutput[0], 101.68, accuracy: 1e-4)
        XCTAssertEqual(policyPassOutput[0], 68.88, accuracy: 1e-4)
        XCTAssertEqual(valueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(scoreValueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(ownershipOutput[0], 32.8, accuracy: 1e-4)
    }
}
