import XCTest
import MetalPerformanceShadersGraph

final class InputLayerTest: XCTestCase {

    func testNCHW() {
        let sourceLayer = InputLayer(graph: MPSGraph(),
                                     batchSize: 2,
                                     nnXLen: 5,
                                     nnYLen: 4,
                                     numChannels: 3,
                                     useFP16: false,
                                     useNHWC: false)

        XCTAssert(sourceLayer.tensor.shape == [2, 3, 4, 5])
        XCTAssert(sourceLayer.tensor.dataType == .float32)
    }

    func testTensorNCHW() {
        let graph = MPSGraph()
        let tensor = graph.constant(1, shape: [2, 3, 4, 5], dataType: .float32)

        let sourceLayer = InputLayer(tensor: tensor)

        XCTAssert(sourceLayer.tensor === tensor)
        XCTAssert(sourceLayer.tensor.shape == [2, 3, 4, 5])
        XCTAssert(sourceLayer.tensor.dataType == .float32)
    }

    func testNHWC() {
        let sourceLayer = InputLayer(graph: MPSGraph(),
                                     batchSize: 2,
                                     nnXLen: 5,
                                     nnYLen: 4,
                                     numChannels: 3,
                                     useFP16: false,
                                     useNHWC: true)

        XCTAssert(sourceLayer.tensor.shape == [2, 4, 5, 3])
        XCTAssert(sourceLayer.tensor.dataType == .float32)
    }

    func testFP16() {
        let sourceLayer = InputLayer(graph: MPSGraph(),
                                     batchSize: 2,
                                     nnXLen: 5,
                                     nnYLen: 4,
                                     numChannels: 3,
                                     useFP16: true,
                                     useNHWC: false)

        XCTAssert(sourceLayer.tensor.shape == [2, 3, 4, 5])
        XCTAssert(sourceLayer.tensor.dataType == .float16)
    }
}

final class InputGlobalLayerTest: XCTestCase {

    func testTensor() {
        let graph = MPSGraph()
        let tensor = graph.constant(1, shape: [2, 3, 1, 1], dataType: .float32)
        let inputGlobalLayer = InputGlobalLayer(tensor: tensor)

        XCTAssert(inputGlobalLayer.tensor === tensor)
        XCTAssert(inputGlobalLayer.tensor.shape == [2, 3, 1, 1])
        XCTAssert(inputGlobalLayer.tensor.dataType == .float32)
    }

    func testNilTensor() {
        let inputGlobalLayer = InputGlobalLayer(graph: MPSGraph(),
                                                batchSize: 2,
                                                numGlobalFeatures: 3,
                                                useFP16: false,
                                                useNHWC: false)

        XCTAssert(inputGlobalLayer.tensor.shape == [2, 3, 1, 1])
        XCTAssert(inputGlobalLayer.tensor.dataType == .float32)
    }

    func testFP16() {
        let inputGlobalLayer = InputGlobalLayer(graph: MPSGraph(),
                                                batchSize: 2,
                                                numGlobalFeatures: 3,
                                                useFP16: true,
                                                useNHWC: false)

        XCTAssert(inputGlobalLayer.tensor.shape == [2, 3, 1, 1])
        XCTAssert(inputGlobalLayer.tensor.dataType == .float16)
    }

    func testNHWC() {
        let inputGlobalLayer = InputGlobalLayer(graph: MPSGraph(),
                                                batchSize: 2,
                                                numGlobalFeatures: 3,
                                                useFP16: true,
                                                useNHWC: true)

        XCTAssert(inputGlobalLayer.tensor.shape == [2, 1, 1, 3])
        XCTAssert(inputGlobalLayer.tensor.dataType == .float16)
    }
}

final class MaskLayerTest: XCTestCase {

    func testTensor() {
        let graph = MPSGraph()
        let tensor = graph.constant(1, shape: [2, 1, 3, 4], dataType: .float32)
        let maskLayer = MaskLayer(tensor: tensor)

        XCTAssert(maskLayer.tensor === tensor)
        XCTAssert(maskLayer.tensor.shape == [2, 1, 3, 4])
        XCTAssert(maskLayer.tensor.dataType == .float32)
    }

    func testNilTensor() {
        let graph = MPSGraph()

        let maskLayer = MaskLayer(graph: graph,
                                  batchSize: 2,
                                  nnXLen: 4,
                                  nnYLen: 3,
                                  useFP16: false,
                                  useNHWC: false)

        XCTAssert(maskLayer.tensor.shape == [2, 1, 3, 4])
        XCTAssert(maskLayer.tensor.dataType == .float32)
    }

    func testNHWC() {
        let graph = MPSGraph()

        let maskLayer = MaskLayer(graph: graph,
                                  batchSize: 2,
                                  nnXLen: 4,
                                  nnYLen: 3,
                                  useFP16: false,
                                  useNHWC: true)

        XCTAssert(maskLayer.tensor.shape == [2, 3, 4, 1])
        XCTAssert(maskLayer.tensor.dataType == .float32)
    }

    func testFP16() {
        let graph = MPSGraph()

        let maskLayer = MaskLayer(graph: graph,
                                  batchSize: 2,
                                  nnXLen: 4,
                                  nnYLen: 3,
                                  useFP16: true,
                                  useNHWC: false)

        XCTAssert(maskLayer.tensor.shape == [2, 1, 3, 4])
        XCTAssert(maskLayer.tensor.dataType == .float16)
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

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumLayer.tensor]?.mpsndarray().readBytes(buffer, strideBytes: nil)

        XCTAssert(maskSumLayer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], 12)
        XCTAssertEqual(buffer[1], 12)
    }

    func testNilTensor() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 1, 3, 4]
        let tensor = graph.constant(1, shape: shape, dataType: .float32)
        let useNHWC = false
        let maskLayer = MaskLayer(tensor: tensor)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        mask: maskLayer,
                                        useNHWC: useNHWC)

        XCTAssert(maskSumLayer.tensor.shape == [2, 1, 1, 1])

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumLayer.tensor],
                              targetOperations: nil)

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumLayer.tensor]?.mpsndarray().readBytes(buffer, strideBytes: nil)

        XCTAssertEqual(buffer[0], 12)
        XCTAssertEqual(buffer[1], 12)
    }

    func testNHWC() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 3, 4, 1]
        let tensor = graph.constant(1, shape: shape, dataType: .float32)
        let useNHWC = true
        let maskLayer = MaskLayer(tensor: tensor)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        mask: maskLayer,
                                        useNHWC: useNHWC)

        XCTAssert(maskSumLayer.tensor.shape == [2, 1, 1, 1])

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumLayer.tensor],
                              targetOperations: nil)

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumLayer.tensor]?.mpsndarray().readBytes(buffer, strideBytes: nil)

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

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01Layer.tensor]?.mpsndarray().readBytes(buffer,
                                                                     strideBytes: nil)

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

        let maskLayer = MaskLayer(tensor: tensor)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        mask: maskLayer,
                                        useNHWC: false)

        let maskSumSqrtS14M01Layer = MaskSumSqrtS14M01Layer(graph: graph,
                                                            maskSum: maskSumLayer,
                                                            useFP16: false)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01Layer.tensor],
                              targetOperations: nil)

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01Layer.tensor]?.mpsndarray().readBytes(buffer,
                                                                     strideBytes: nil)

        XCTAssert(maskSumSqrtS14M01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], -1.053589838486225, accuracy: 1e-8)
        XCTAssertEqual(buffer[1], -1.053589838486225, accuracy: 1e-8)
    }

    func testFP16() {
        let graph = MPSGraph()

        let shape: [NSNumber] = [2, 1, 3, 4]

        let tensor = graph.constant(1,
                                    shape: shape,
                                    dataType: .float16)

        let maskLayer = MaskLayer(tensor: tensor)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        mask: maskLayer,
                                        useNHWC: false)

        let maskSumSqrtS14M01Layer = MaskSumSqrtS14M01Layer(graph: graph,
                                                            maskSum: maskSumLayer,
                                                            useFP16: true)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01Layer.tensor],
                              targetOperations: nil)

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float16>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01Layer.tensor]?.mpsndarray().readBytes(buffer,
                                                                     strideBytes: nil)

        XCTAssert(maskSumSqrtS14M01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], -1.053589838486225, accuracy: 1e-4)
        XCTAssertEqual(buffer[1], -1.053589838486225, accuracy: 1e-4)
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

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01SquareS01Layer.tensor]?.mpsndarray().readBytes(buffer,
                                                                              strideBytes: nil)

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

        let maskLayer = MaskLayer(tensor: tensor)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        mask: maskLayer,
                                        useNHWC: false)

        let maskSumSqrtS14M01Layer = MaskSumSqrtS14M01Layer(graph: graph,
                                                            maskSum: maskSumLayer,
                                                            useFP16: false)

        let maskSumSqrtS14M01SquareS01Layer =
        MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                        maskSumSqrtS14M01: maskSumSqrtS14M01Layer,
                                        useFP16: false)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01SquareS01Layer.tensor],
                              targetOperations: nil)

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float32>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01SquareS01Layer.tensor]?.mpsndarray().readBytes(buffer,
                                                                              strideBytes: nil)

        XCTAssert(maskSumSqrtS14M01SquareS01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], 1.010051547761429, accuracy: 1e-8)
        XCTAssertEqual(buffer[1], 1.010051547761429, accuracy: 1e-8)
    }

    func testFP16() {
        let graph = MPSGraph()
        let shape: [NSNumber] = [2, 1, 3, 4]

        let tensor = graph.constant(1,
                                    shape: shape,
                                    dataType: .float16)

        let maskLayer = MaskLayer(tensor: tensor)

        let maskSumLayer = MaskSumLayer(graph: graph,
                                        mask: maskLayer,
                                        useNHWC: false)

        let maskSumSqrtS14M01Layer = MaskSumSqrtS14M01Layer(graph: graph,
                                                            maskSum: maskSumLayer,
                                                            useFP16: true)

        let maskSumSqrtS14M01SquareS01Layer =
        MaskSumSqrtS14M01SquareS01Layer(graph: graph,
                                        maskSumSqrtS14M01: maskSumSqrtS14M01Layer,
                                        useFP16: true)

        let fetch = graph.run(feeds: [:],
                              targetTensors: [maskSumSqrtS14M01SquareS01Layer.tensor],
                              targetOperations: nil)

        let length = Int(truncating: shape.product())
        let buffer = UnsafeMutablePointer<Float16>.allocate(capacity: length)

        fetch[maskSumSqrtS14M01SquareS01Layer.tensor]?.mpsndarray().readBytes(buffer,
                                                                              strideBytes: nil)

        XCTAssert(maskSumSqrtS14M01SquareS01Layer.tensor.shape == [2, 1, 1, 1])
        XCTAssertEqual(buffer[0], 1.010051547761429, accuracy: 1e-4)
        XCTAssertEqual(buffer[1], 1.010051547761429, accuracy: 1e-4)
    }
}

final class ConvLayerTest: XCTestCase {

    func testNHWC() {
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

        let descriptor = SWConvLayerDesc(convYSize: convYSize as NSNumber,
                                         convXSize: convXSize as NSNumber,
                                         inChannels: inChannels,
                                         outChannels: outChannels,
                                         dilationY: 1,
                                         dilationX: 1,
                                         weights: weights)

        let batchSize: NSNumber = 1
        let nnXLen: NSNumber = 3
        let nnYLen: NSNumber = 2
        let useFP16 = false
        let useNHWC = true

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

        ConvLayer.test(descriptor: descriptor,
                       nnXLen: nnXLen,
                       nnYLen: nnYLen,
                       batchSize: batchSize,
                       useFP16: useFP16,
                       useNHWC: useNHWC,
                       input: inputPointer,
                       output: outputPointer)

        XCTAssertEqual(outputPointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[4], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[6], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[8], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[10], 2, accuracy: 1e-8)

        XCTAssertEqual(outputPointer[1], 3, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[5], 5, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[7], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[9], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[11], 0, accuracy: 1e-8)
    }

    func testFP16() {
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

        let descriptor = SWConvLayerDesc(convYSize: convYSize as NSNumber,
                                         convXSize: convXSize as NSNumber,
                                         inChannels: inChannels,
                                         outChannels: outChannels,
                                         dilationY: 1,
                                         dilationX: 1,
                                         weights: weights)

        let batchSize: NSNumber = 1
        let nnXLen: NSNumber = 3
        let nnYLen: NSNumber = 2
        let useFP16 = true
        let useNHWC = false

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

        ConvLayer.test(descriptor: descriptor,
                       nnXLen: nnXLen,
                       nnYLen: nnYLen,
                       batchSize: batchSize,
                       useFP16: useFP16,
                       useNHWC: useNHWC,
                       input: inputPointer,
                       output: outputPointer)

        XCTAssertEqual(outputPointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[4], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[5], 2, accuracy: 1e-8)

        XCTAssertEqual(outputPointer[6], 3, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[7], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[8], 5, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[9], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[10], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[11], 0, accuracy: 1e-8)
    }
}

final class BatchNormLayerTest: XCTestCase {

    func testFP16() {
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

        let descriptor = SWBatchNormLayerDesc(numChannels: numChannels,
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
        let useFP16 = true
        let useNHWC = false

        let inputLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue * numChannels.intValue

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputLength)
        let x = inputPointer

        x[0] = 5; x[1] = 5; x[2] = 4; x[3] = 4; x[4] = 9
        x[5] = 1; x[6] = 1; x[7] = 8; x[8] = 8; x[9] = 9

        x[10] = 0; x[11] = 1; x[12] = 2; x[13] = 3; x[14] = 4
        x[15] = 8; x[16] = 7; x[17] = 6; x[18] = 5; x[19] = 4

        x[20] = 3; x[21] = 0; x[22] = 4; x[23] = 0; x[24] = 5
        x[25] = 0; x[26] = 5; x[27] = 0; x[28] = 6; x[29] = 0

        x[30] = 1; x[31] = 0; x[32] = 0; x[33] = 2; x[34] = 1
        x[35] = 0; x[36] = 2; x[37] = 2; x[38] = 0; x[39] = 2

        let maskLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue
        let maskPointer = UnsafeMutablePointer<Float32>.allocate(capacity: maskLength)
        let m = maskPointer

        m[0] = 1; m[1] = 1; m[2] = 1; m[3] = 1; m[4] = 1
        m[5] = 1; m[6] = 1; m[7] = 1; m[8] = 1; m[9] = 1

        m[10] = 1; m[11] = 1; m[12] = 1; m[13] = 1; m[14] = 1
        m[15] = 1; m[16] = 1; m[17] = 1; m[18] = 1; m[19] = 1

        let outputLength = batchSize.intValue * nnXLen.intValue * nnYLen.intValue * numChannels.intValue

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputLength)

        BatchNormLayer.test(descriptor: descriptor,
                            nnXLen: nnXLen,
                            nnYLen: nnYLen,
                            batchSize: batchSize,
                            useFP16: useFP16,
                            useNHWC: useNHWC,
                            input: inputPointer,
                            mask: maskPointer,
                            output: outputPointer)

        XCTAssertEqual(outputPointer[0], 10.25, accuracy: 1e-2)
        XCTAssertEqual(outputPointer[4], 10.45, accuracy: 1e-2)
        XCTAssertEqual(outputPointer[5], 10.05, accuracy: 1e-2)
        XCTAssertEqual(outputPointer[9], 10.45, accuracy: 1e-2)
        XCTAssertEqual(outputPointer[19], 4, accuracy: 1e-3)
        XCTAssertEqual(outputPointer[20], 10.15, accuracy: 1e-2)
        XCTAssertEqual(outputPointer[39], 0, accuracy: 1e-4)
    }

    func testNHWC() {
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

        let descriptor = SWBatchNormLayerDesc(numChannels: numChannels,
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
        let useFP16 = false
        let useNHWC = true

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

        BatchNormLayer.test(descriptor: descriptor,
                            nnXLen: nnXLen,
                            nnYLen: nnYLen,
                            batchSize: batchSize,
                            useFP16: useFP16,
                            useNHWC: useNHWC,
                            input: inputPointer,
                            mask: maskPointer,
                            output: outputPointer)

        XCTAssertEqual(outputPointer[0], 10.25, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[8], 10.45, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[10], 10.05, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[18], 10.45, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[19], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[20], 10.15, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[39], 0, accuracy: 1e-8)
    }
}

final class ResidualBlockTest: XCTestCase {

    func testFP16() {
        let useFP16 = true
        let useNHWC = false
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

        let descriptor = SWResidualBlockDesc(preBN: preBN,
                                             preActivation: nil,
                                             regularConv: regularConv,
                                             midBN: midBN,
                                             midActivation: nil,
                                             finalConv: finalConv)

        let outputLength = batchSize.intValue * trunkChannels.intValue * nnYLen.intValue * nnXLen.intValue

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputLength)

        ResidualBlock.test(descriptor: descriptor,
                           batchSize: batchSize,
                           nnXLen: nnXLen,
                           nnYLen: nnYLen,
                           useFP16: useFP16,
                           useNHWC: useNHWC,
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

    func testNHWC() {
        let useFP16 = false
        let useNHWC = true
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

        let descriptor = SWResidualBlockDesc(preBN: preBN,
                                             preActivation: nil,
                                             regularConv: regularConv,
                                             midBN: midBN,
                                             midActivation: nil,
                                             finalConv: finalConv)

        let outputLength = batchSize.intValue * trunkChannels.intValue * nnYLen.intValue * nnXLen.intValue

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputLength)

        ResidualBlock.test(descriptor: descriptor,
                           batchSize: batchSize,
                           nnXLen: nnXLen,
                           nnYLen: nnYLen,
                           useFP16: useFP16,
                           useNHWC: useNHWC,
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
        let useFP16 = false
        let useNHWC = false
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
                                                preActivation: nil,
                                                regularConv: unityConv,
                                                midBN: unityBN,
                                                midActivation: nil,
                                                finalConv: unityConv)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               batchSize: batchSize as NSNumber,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: numChannels as NSNumber,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let mask = MaskLayer(graph: graph,
                             batchSize: batchSize as NSNumber,
                             nnXLen: nnXLen as NSNumber,
                             nnYLen: nnYLen as NSNumber,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let block = ResidualBlock(graph: graph,
                                  sourceTensor: input.tensor,
                                  maskTensor: mask.tensor,
                                  descriptor: residualBlock,
                                  nnXLen: nnXLen as NSNumber,
                                  nnYLen: nnYLen as NSNumber,
                                  batchSize: batchSize as NSNumber,
                                  useFP16: useFP16,
                                  useNHWC: useNHWC)

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

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let inputTensorData = MPSGraphTensorData(device: device,
                                                 tensor: input.tensor)!

        inputTensorData.mpsndarray().writeBytes(inputPointer,
                                                strideBytes: nil)

        let maskTensorData = MPSGraphTensorData(device: device,
                                                tensor: mask.tensor)!

        maskTensorData.mpsndarray().writeBytes(maskPointer,
                                               strideBytes: nil)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [block.resultTensor],
                              targetOperations: nil)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        fetch[block.resultTensor]?.mpsndarray().readBytes(outputPointer,
                                                          strideBytes: nil)

        XCTAssertEqual(outputPointer[0], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 2, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 6, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[15], 30, accuracy: 1e-8)
    }
}

final class GlobalPoolingResidualBlockTest: XCTestCase {

    func testFP16() {
        let useFP16 = true
        let useNHWC = false
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
        x[20] = 0; x[21] = -1; x[22] = 1;  x[23] = 1

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

        let inChannels = NSNumber(value: gpoolChannels.intValue * 3)

        let gpoolToBiasMul =
        SWMatMulLayerDesc(inChannels: inChannels,
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
                                                          preActivation: nil,
                                                          regularConv: regularConv,
                                                          gpoolConv: gpoolConv,
                                                          gpoolBN: gpoolBN,
                                                          gpoolActivation: nil,
                                                          gpoolToBiasMul: gpoolToBiasMul,
                                                          midBN: midBN,
                                                          midActivation: nil,
                                                          finalConv: finalConv)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 24)

        GlobalPoolingResidualBlock.test(descriptor: descriptor,
                                        batchSize: batchSize,
                                        nnXLen: nnXLen,
                                        nnYLen: nnYLen,
                                        useFP16: useFP16,
                                        useNHWC: useNHWC,
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

        XCTAssertEqual(outputPointer[0], y[0], accuracy: 2e-2)
        XCTAssertEqual(outputPointer[3], y[3], accuracy: 2e-2)
        XCTAssertEqual(outputPointer[4], y[4], accuracy: 2e-2)
        XCTAssertEqual(outputPointer[11], y[11], accuracy: 2e-2)
        XCTAssertEqual(outputPointer[12], y[12], accuracy: 2e-2)
        XCTAssertEqual(outputPointer[18], y[18], accuracy: 2e-2)
        XCTAssertEqual(outputPointer[23], y[23], accuracy: 2e-2)
    }

    func testNHWC() {
        let useFP16 = false
        let useNHWC = true
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
        SWMatMulLayerDesc(inChannels: 6,
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
                                                          preActivation: nil,
                                                          regularConv: regularConv,
                                                          gpoolConv: gpoolConv,
                                                          gpoolBN: gpoolBN,
                                                          gpoolActivation: nil,
                                                          gpoolToBiasMul: gpoolToBiasMul,
                                                          midBN: midBN,
                                                          midActivation: nil,
                                                          finalConv: finalConv)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 24)

        GlobalPoolingResidualBlock.test(descriptor: descriptor,
                                        batchSize: batchSize,
                                        nnXLen: nnXLen,
                                        nnYLen: nnYLen,
                                        useFP16: useFP16,
                                        useNHWC: useNHWC,
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

final class MatMulLayerTest: XCTestCase {

    func testFP16() {
        let useFP16 = true
        let useNHWC = true
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
                               batchSize: batchSize as NSNumber,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: inChannels as NSNumber,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let matMulLayer = try! MatMulLayer(graph: graph,
                                           descriptor: descriptor,
                                           sourceTensor: input.tensor,
                                           useFP16: useFP16,
                                           useNHWC: useNHWC)

        let inputCount = batchSize * nnXLen * nnYLen * inChannels
        let inputPointer = UnsafeMutablePointer<Float16>.allocate(capacity: inputCount)

        for i in 0..<inputCount {
            inputPointer[i] = Float16(i)
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

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let inputTensorData = MPSGraphTensorData(device: device,
                                                 tensor: input.tensor)!

        inputTensorData.mpsndarray().writeBytes(inputPointer,
                                                strideBytes: nil)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData],
                              targetTensors: [matMulLayer.resultTensor],
                              targetOperations: nil)

        let outputCount = batchSize * nnXLen * nnYLen * outChannels
        let outputPointer = UnsafeMutablePointer<Float16>.allocate(capacity: outputCount)

        fetch[matMulLayer.resultTensor]?.mpsndarray().readBytes(outputPointer,
                                                                strideBytes: nil)

        XCTAssertEqual(outputPointer[0], 3, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[1], 4, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[2], 5, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[3], 9, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[4], 14, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[5], 19, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[6], 15, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[7], 24, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[8], 33, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[9], 21, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[10], 34, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[11], 47, accuracy: 1e-4)
    }

    func testFP32() {
        let useFP16 = false
        let useNHWC = true
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
                               batchSize: batchSize as NSNumber,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: inChannels as NSNumber,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let matMulLayer = try! MatMulLayer(graph: graph,
                                           descriptor: descriptor,
                                           sourceTensor: input.tensor,
                                           useFP16: useFP16,
                                           useNHWC: useNHWC)

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

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let inputTensorData = MPSGraphTensorData(device: device,
                                                 tensor: input.tensor)!

        inputTensorData.mpsndarray().writeBytes(inputPointer,
                                                strideBytes: nil)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData],
                              targetTensors: [matMulLayer.resultTensor],
                              targetOperations: nil)

        let outputCount = batchSize * nnXLen * nnYLen * outChannels
        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: outputCount)

        fetch[matMulLayer.resultTensor]?.mpsndarray().readBytes(outputPointer,
                                                                strideBytes: nil)

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

    func testInvalid() {
        let useFP16 = false
        let useNHWC = false
        let batchSize = 1
        let nnXLen = 2
        let nnYLen = 1
        let inChannels = 1
        let outChannels = 2
        let weightsCount = inChannels * outChannels
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: weightsCount)

        let descriptor = SWMatMulLayerDesc(inChannels: inChannels as NSNumber,
                                           outChannels: outChannels as NSNumber,
                                           weights: weights)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               batchSize: batchSize as NSNumber,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: inChannels as NSNumber,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        XCTAssertThrowsError(try MatMulLayer(graph: graph,
                                             descriptor: descriptor,
                                             sourceTensor: input.tensor,
                                             useFP16: useFP16,
                                             useNHWC: useNHWC))
    }
}

final class MatBiasLayerTest: XCTestCase {

    func testFP16() {
        let useFP16 = true
        let useNHWC = true
        let numChannels = 2
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        weights[0] = 1
        weights[1] = -1

        let descriptor = SWMatBiasLayerDesc(numChannels: numChannels as NSNumber,
                                            weights: weights)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               batchSize: 2,
                               nnXLen: 2,
                               nnYLen: 2,
                               numChannels: 2,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let matBiasLayer = try! MatBiasLayer(graph: graph,
                                             descriptor: descriptor,
                                             sourceTensor: input.tensor,
                                             useFP16: useFP16,
                                             useNHWC: useNHWC)

        let inputPointer = UnsafeMutablePointer<Float16>.allocate(capacity: 16)

        for i in 0..<16 {
            inputPointer[i] = Float16(i)
        }

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let inputTensorData = MPSGraphTensorData(device: device,
                                                 tensor: input.tensor)!

        inputTensorData.mpsndarray().writeBytes(inputPointer,
                                                strideBytes: nil)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData],
                              targetTensors: [matBiasLayer.resultTensor],
                              targetOperations: nil)

        let outputPointer = UnsafeMutablePointer<Float16>.allocate(capacity: 16)

        fetch[matBiasLayer.resultTensor]?.mpsndarray().readBytes(outputPointer,
                                                                 strideBytes: nil)

        XCTAssertEqual(outputPointer[0], 1, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[1], 0, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[2], 3, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[3], 2, accuracy: 1e-4)
        XCTAssertEqual(outputPointer[15], 14, accuracy: 1e-4)
    }

    func testFP32() {
        let useFP16 = false
        let useNHWC = true
        let numChannels = 2
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: numChannels)

        weights[0] = 1
        weights[1] = -1

        let descriptor = SWMatBiasLayerDesc(numChannels: numChannels as NSNumber,
                                            weights: weights)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               batchSize: 2,
                               nnXLen: 2,
                               nnYLen: 2,
                               numChannels: 2,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let matBiasLayer = try! MatBiasLayer(graph: graph,
                                             descriptor: descriptor,
                                             sourceTensor: input.tensor,
                                             useFP16: useFP16,
                                             useNHWC: useNHWC)

        let inputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 16)

        for i in 0..<16 {
            inputPointer[i] = Float32(i)
        }

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let inputTensorData = MPSGraphTensorData(device: device,
                                                 tensor: input.tensor)!

        inputTensorData.mpsndarray().writeBytes(inputPointer,
                                                strideBytes: nil)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData],
                              targetTensors: [matBiasLayer.resultTensor],
                              targetOperations: nil)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: 16)

        fetch[matBiasLayer.resultTensor]?.mpsndarray().readBytes(outputPointer,
                                                                 strideBytes: nil)

        XCTAssertEqual(outputPointer[0], 1, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 0, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 3, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 2, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[15], 14, accuracy: 1e-8)
    }

    func testInvalid() {
        let useFP16 = false
        let useNHWC = false
        let batchSize = 1
        let nnXLen = 2
        let nnYLen = 1
        let numChannels = 2
        let weightsCount = numChannels
        let weights = UnsafeMutablePointer<Float32>.allocate(capacity: weightsCount)

        let descriptor = SWMatBiasLayerDesc(numChannels: numChannels as NSNumber,
                                            weights: weights)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               batchSize: batchSize as NSNumber,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: numChannels as NSNumber,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        XCTAssertThrowsError(try MatBiasLayer(graph: graph,
                                              descriptor: descriptor,
                                              sourceTensor: input.tensor,
                                              useFP16: useFP16,
                                              useNHWC: useNHWC))
    }
}

final class TrunkTest: XCTestCase {

    func testUnity() {
        let useFP16 = false
        let useNHWC = false
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
                                                preActivation: nil,
                                                regularConv: unityConv,
                                                midBN: unityBN,
                                                midActivation: nil,
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
        SWGlobalPoolingResidualBlockDesc(preBN: unityBN,
                                         preActivation: nil,
                                         regularConv: unityConv,
                                         gpoolConv: unityConv,
                                         gpoolBN: unityBN,
                                         gpoolActivation: nil,
                                         gpoolToBiasMul: gpoolToBiasMul,
                                         midBN: unityBN,
                                         midActivation: nil,
                                         finalConv: unityConv)

        let blocks = [
            BlockDescriptor(kind: BlockKind.ordinary,
                            ordinary: residualBlock,
                            globalPooling: nil),
            BlockDescriptor(kind: BlockKind.globalPooling,
                            ordinary: nil,
                            globalPooling: globalPoolingResidualBlock)]

        let descriptor = SWTrunkDesc(version: 0,
                                     numBlocks: blocks.count,
                                     trunkNumChannels: numChannels as NSNumber,
                                     midNumChannels: numChannels as NSNumber,
                                     regularNumChannels: numChannels as NSNumber,
                                     dilatedNumChannels: numChannels as NSNumber,
                                     gpoolNumChannels: numChannels as NSNumber,
                                     initialConv: unityConv,
                                     initialMatMul: initialMatMul,
                                     blocks: blocks,
                                     trunkTipBN: unityBN)

        let graph = MPSGraph()

        let input = InputLayer(graph: graph,
                               batchSize: batchSize as NSNumber,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               numChannels: numChannels as NSNumber,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

        let inputGlobal = InputGlobalLayer(graph: graph,
                                           batchSize: batchSize as NSNumber,
                                           numGlobalFeatures: numChannels as NSNumber,
                                           useFP16: useFP16,
                                           useNHWC: useNHWC)

        let mask = MaskLayer(graph: graph,
                             batchSize: batchSize as NSNumber,
                             nnXLen: nnXLen as NSNumber,
                             nnYLen: nnYLen as NSNumber,
                             useFP16: useFP16,
                             useNHWC: useNHWC)

        let maskSum = MaskSumLayer(graph: graph, mask: mask, useNHWC: useNHWC)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(graph: graph,
                                                       maskSum: maskSum,
                                                       useFP16: useFP16)

        let trunk = try! Trunk(graph: graph,
                               descriptor: descriptor,
                               inputTensor: input.tensor,
                               inputGlobalTensor: inputGlobal.tensor,
                               maskTensor: mask.tensor,
                               maskSumTensor: maskSum.tensor,
                               maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                               nnXLen: nnXLen as NSNumber,
                               nnYLen: nnYLen as NSNumber,
                               batchSize: batchSize as NSNumber,
                               numSpatialFeatures: numChannels as NSNumber,
                               numGlobalFeatures: numChannels as NSNumber,
                               useFP16: useFP16,
                               useNHWC: useNHWC)

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

        let device = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)

        let inputTensorData = MPSGraphTensorData(device: device,
                                                 tensor: input.tensor)!

        inputTensorData.mpsndarray().writeBytes(inputPointer,
                                                strideBytes: nil)

        let inputGlobalTensorData = MPSGraphTensorData(device: device,
                                                       tensor: inputGlobal.tensor)!

        inputGlobalTensorData.mpsndarray().writeBytes(inputGlobalPointer,
                                                      strideBytes: nil)

        let maskTensorData = MPSGraphTensorData(device: device,
                                                tensor: mask.tensor)!

        maskTensorData.mpsndarray().writeBytes(maskPointer,
                                               strideBytes: nil)

        let fetch = graph.run(feeds: [input.tensor: inputTensorData,
                                      inputGlobal.tensor: inputGlobalTensorData,
                                      mask.tensor: maskTensorData],
                              targetTensors: [trunk.resultTensor],
                              targetOperations: nil)

        let outputPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inputCount)

        fetch[trunk.resultTensor]?.mpsndarray().readBytes(outputPointer,
                                                          strideBytes: nil)

        XCTAssertEqual(outputPointer[0], 4, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[1], 8, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[2], 12, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[3], 16, accuracy: 1e-8)
        XCTAssertEqual(outputPointer[15], 64, accuracy: 1e-8)
    }
}
