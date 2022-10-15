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
        let tensor = graph.constant(1, shape: [2, 3], dataType: .float32)
        let inputGlobalLayer = InputGlobalLayer(tensor: tensor)

        XCTAssert(inputGlobalLayer.tensor === tensor)
        XCTAssert(inputGlobalLayer.tensor.shape == [2, 3])
        XCTAssert(inputGlobalLayer.tensor.dataType == .float32)
    }

    func testNilTensor() {
        let inputGlobalLayer = InputGlobalLayer(graph: MPSGraph(),
                                                batchSize: 2,
                                                numGlobalFeatures: 3,
                                                useFP16: false)

        XCTAssert(inputGlobalLayer.tensor.shape == [2, 3])
        XCTAssert(inputGlobalLayer.tensor.dataType == .float32)
    }

    func testFP16() {
        let inputGlobalLayer = InputGlobalLayer(graph: MPSGraph(),
                                                batchSize: 2,
                                                numGlobalFeatures: 3,
                                                useFP16: true)

        XCTAssert(inputGlobalLayer.tensor.shape == [2, 3])
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
