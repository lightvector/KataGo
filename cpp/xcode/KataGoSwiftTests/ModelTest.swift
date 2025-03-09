//
//  ModelTest.swift
//  KataGoSwiftTests
//
//  Created by Chin-Chang Yang on 2024/7/19.
//

import XCTest
import MetalPerformanceShadersGraph

final class SWModelDescTest {

    var unityConvWeights = [Float](repeating: 1, count: 1)
    var unityMatMulWeights = [Float](repeating: 1, count: 1)
    var meanWeights = [Float](repeating: 0, count: 1)
    var varianceWeights = [Float](repeating: 0.9, count: 1)
    var scaleWeights = [Float](repeating: 1, count: 1)
    var biasWeights = [Float](repeating: 0, count: 1)
    var gpoolMatMulWeights = [Float](repeating: 3, count: 3)
    var zeroMatBiasWeights = [Float](repeating: 0, count: 1)
    var gpoolToPassMulWeights = [Float](repeating: 3, count: 9)
    var gpoolToPassBiasWeights = [Float](repeating: 0, count: 3)

    func createMiniDescV15Meta() -> SWModelDesc {
        let version = 15

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

        let zeroMatBias = SWMatBiasLayerDesc(numChannels: 1,
                                             weights: &zeroMatBiasWeights)

        let sgfMetadataEncoder = SWSGFMetadataEncoderDesc(version: version,
                                                          numInputMetaChannels: 1,
                                                          mul1: unityMatMul,
                                                          bias1: zeroMatBias,
                                                          act1: ActivationKind.relu,
                                                          mul2: unityMatMul,
                                                          bias2: zeroMatBias,
                                                          act2: ActivationKind.relu,
                                                          mul3: unityMatMul)

        let trunkDesc = SWTrunkDesc(version: version,
                                    trunkNumChannels: 1,
                                    midNumChannels: 1,
                                    regularNumChannels: 1,
                                    gpoolNumChannels: 1,
                                    initialConv: unityConv,
                                    initialMatMul: unityMatMul,
                                    sgfMetadataEncoder: sgfMetadataEncoder,
                                    blockDescriptors: blocks,
                                    trunkTipBN: unityBatchNorm,
                                    trunkTipActivation: ActivationKind.relu)

        let gpoolToPassMul = SWMatMulLayerDesc(inChannels: 3,
                                               outChannels: 3,
                                               weights: &gpoolToPassMulWeights)

        let gpoolToPassBias = SWMatBiasLayerDesc(numChannels: 3,
                                                 weights: &gpoolToPassBiasWeights)

        let policyHead = createSWPolicyHeadDesc(version: Int32(version),
                                                p1Conv: unityConv,
                                                g1Conv: unityConv,
                                                g1BN: unityBatchNorm,
                                                g1Activation: ActivationKind.relu,
                                                gpoolToBiasMul: gpoolMatMul,
                                                p1BN: unityBatchNorm,
                                                p1Activation: ActivationKind.relu,
                                                p2Conv: unityConv,
                                                gpoolToPassMul: gpoolToPassMul,
                                                gpoolToPassBias: gpoolToPassBias,
                                                passActivation: ActivationKind.relu,
                                                gpoolToPassMul2: gpoolMatMul)

        let valueHead = SWValueHeadDesc(version: version,
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

        let modelDesc = createSWModelDesc(version: Int32(version),
                                          name: "test",
                                          numInputChannels: 1,
                                          numInputGlobalChannels: 1,
                                          numInputMetaChannels: 1,
                                          numValueChannels: 1,
                                          numScoreValueChannels: 1,
                                          numOwnershipChannels: 1,
                                          trunk: trunkDesc,
                                          policyHead: policyHead,
                                          valueHead: valueHead)

        return modelDesc
    }

    func createMiniDescV15() -> SWModelDesc {
        let version = 15

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

        let trunkDesc = SWTrunkDesc(version: version,
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

        let gpoolToPassMul = SWMatMulLayerDesc(inChannels: 3,
                                               outChannels: 3,
                                               weights: &gpoolToPassMulWeights)

        let gpoolToPassBias = SWMatBiasLayerDesc(numChannels: 3,
                                                 weights: &gpoolToPassBiasWeights)

        let policyHead = createSWPolicyHeadDesc(version: Int32(version),
                                                p1Conv: unityConv,
                                                g1Conv: unityConv,
                                                g1BN: unityBatchNorm,
                                                g1Activation: ActivationKind.relu,
                                                gpoolToBiasMul: gpoolMatMul,
                                                p1BN: unityBatchNorm,
                                                p1Activation: ActivationKind.relu,
                                                p2Conv: unityConv,
                                                gpoolToPassMul: gpoolToPassMul,
                                                gpoolToPassBias: gpoolToPassBias,
                                                passActivation: ActivationKind.relu,
                                                gpoolToPassMul2: gpoolMatMul)

        let zeroMatBias = SWMatBiasLayerDesc(numChannels: 1,
                                             weights: &zeroMatBiasWeights)

        let valueHead = SWValueHeadDesc(version: version,
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

        let modelDesc = createSWModelDesc(version: Int32(version),
                                          name: "test",
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
                                    sgfMetadataEncoder: nil,
                                    blockDescriptors: blocks,
                                    trunkTipBN: unityBatchNorm,
                                    trunkTipActivation: ActivationKind.relu)

        let gpoolToPassBias = SWMatBiasLayerDesc(numChannels: 3,
                                                 weights: &gpoolToPassBiasWeights)

        let policyHead = createSWPolicyHeadDesc(version: 0,
                                                p1Conv: unityConv,
                                                g1Conv: unityConv,
                                                g1BN: unityBatchNorm,
                                                g1Activation: ActivationKind.relu,
                                                gpoolToBiasMul: gpoolMatMul,
                                                p1BN: unityBatchNorm,
                                                p1Activation: ActivationKind.relu,
                                                p2Conv: unityConv,
                                                gpoolToPassMul: gpoolMatMul,
                                                gpoolToPassBias: gpoolToPassBias,
                                                passActivation: ActivationKind.relu,
                                                gpoolToPassMul2: gpoolMatMul)

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
                                          numInputMetaChannels: 0,
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

    func createMiniModelV15Meta() -> Model? {
        let modelDesc = swModelDescTest.createMiniDescV15Meta()

        let device = MTLCreateSystemDefaultDevice()!

        let model = Model(device: device,
                          graph: MPSGraph(),
                          descriptor: modelDesc,
                          nnXLen: 1,
                          nnYLen: 1)

        return model
    }

    func createMiniModelV15() -> Model? {
        let modelDesc = swModelDescTest.createMiniDescV15()

        let device = MTLCreateSystemDefaultDevice()!

        let model = Model(device: device,
                          graph: MPSGraph(),
                          descriptor: modelDesc,
                          nnXLen: 1,
                          nnYLen: 1)

        return model
    }

    func testMiniModelV15Meta() {
        let model = createMiniModelV15Meta()
        var input = [Float32](repeating: 1, count: 1)
        var inputGlobal = [Float32](repeating: 1, count: 1)
        var inputMeta = [Float32](repeating: 1, count: 1)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model?.apply(input: &input,
                     inputGlobal: &inputGlobal,
                     inputMeta: &inputMeta,
                     policy: &policyOutput,
                     policyPass: &policyPassOutput,
                     value: &valueOutput,
                     scoreValue: &scoreValueOutput,
                     ownership: &ownershipOutput,
                     batchSize: 1)

        XCTAssertEqual(policyOutput[0], 152.51999, accuracy: 1e-4)
        XCTAssertEqual(policyPassOutput[0], 929.87976, accuracy: 1e-4)
        XCTAssertEqual(valueOutput[0], 190.40402, accuracy: 1e-4)
        XCTAssertEqual(scoreValueOutput[0], 190.40402, accuracy: 1e-4)
        XCTAssertEqual(ownershipOutput[0], 49.199997, accuracy: 1e-4)
    }

    func testMiniModelV15() {
        let model = createMiniModelV15()
        var input = [Float32](repeating: 1, count: 1)
        var inputGlobal = [Float32](repeating: 1, count: 1)
        var inputMeta = [Float32](repeating: 0, count: 0)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model?.apply(input: &input,
                     inputGlobal: &inputGlobal,
                     inputMeta: &inputMeta,
                     policy: &policyOutput,
                     policyPass: &policyPassOutput,
                     value: &valueOutput,
                     scoreValue: &scoreValueOutput,
                     ownership: &ownershipOutput,
                     batchSize: 1)

        XCTAssertEqual(policyOutput[0], 101.68, accuracy: 1e-4)
        XCTAssertEqual(policyPassOutput[0], 619.9198, accuracy: 1e-4)
        XCTAssertEqual(valueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(scoreValueOutput[0], 126.936, accuracy: 1e-4)
        XCTAssertEqual(ownershipOutput[0], 32.8, accuracy: 1e-4)
    }

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
        var inputMeta = [Float32](repeating: 0, count: 0)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model.apply(input: &input,
                    inputGlobal: &inputGlobal,
                    inputMeta: &inputMeta,
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
        var inputMeta = [Float32](repeating: 0, count: 0)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model?.apply(input: &input,
                     inputGlobal: &inputGlobal,
                     inputMeta: &inputMeta,
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
        var inputMeta = [Float32](repeating: 0, count: 0)
        var policyOutput = [Float32](repeating: 1, count: 1)
        var policyPassOutput = [Float32](repeating: 1, count: 1)
        var valueOutput = [Float32](repeating: 1, count: 1)
        var scoreValueOutput = [Float32](repeating: 1, count: 1)
        var ownershipOutput = [Float32](repeating: 1, count: 1)

        model?.apply(input: &input,
                     inputGlobal: &inputGlobal,
                     inputMeta: &inputMeta,
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
                                                      UnsafeMutablePointer<Float32>,
                                                      UnsafeMutablePointer<Float32>) {

        let inputCount = batchSize * nnYLen * nnXLen * numInputChannels
        let inputGlobalCount = batchSize * numInputGlobalChannels
        let inputMeta = 0
        let policyCount = batchSize * nnYLen * nnXLen
        let policyPassCount = batchSize
        let valueCount = batchSize * numValueChannels
        let scoreValueCount = batchSize * numScoreValueChannels
        let ownershipCount = batchSize * nnYLen * nnXLen * numOwnershipChannels

        return (UnsafeMutablePointer<Float32>.allocate(capacity: inputCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: inputGlobalCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: inputMeta),
                UnsafeMutablePointer<Float32>.allocate(capacity: policyCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: policyPassCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: valueCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: scoreValueCount),
                UnsafeMutablePointer<Float32>.allocate(capacity: ownershipCount))
    }

    func createModelB40C256(batchSize: Int,
                            nnYLen: Int,
                            nnXLen: Int,
                            numInputChannels: Int,
                            numInputGlobalChannels: Int,
                            numValueChannels: Int,
                            numScoreValueChannels: Int,
                            numOwnershipChannels: Int) -> Model {
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
                                    sgfMetadataEncoder: nil,
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
                                          gpoolToPassMul: gpoolToPassMul,
                                          gpoolToPassBias: nil,
                                          passActivation: nil,
                                          gpoolToPassMul2: nil)

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
                                    numInputMetaChannels: 0,
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
        let (input, inputGlobal, inputMeta, policy, policyPass, value, scoreValue, ownership) =
        createBuffers(batchSize: batchSize,
                      nnYLen: nnYLen,
                      nnXLen: nnXLen,
                      numInputChannels: numInputChannels,
                      numInputGlobalChannels: numInputGlobalChannels,
                      numValueChannels: numValueChannels,
                      numScoreValueChannels: numScoreValueChannels,
                      numOwnershipChannels: numOwnershipChannels)

        model.apply(input: input,
                    inputGlobal: inputGlobal,
                    inputMeta: inputMeta,
                    policy: policy,
                    policyPass: policyPass,
                    value: value,
                    scoreValue: scoreValue,
                    ownership: ownership,
                    batchSize: batchSize)

        return model
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

        let (input, inputGlobal, inputMeta, policy, policyPass, value, scoreValue, ownership) =
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
                model.apply(input: input,
                            inputGlobal: inputGlobal,
                            inputMeta: inputMeta,
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

        let (input, inputGlobal, inputMeta, policy, policyPass, value, scoreValue, ownership) =
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
                model.apply(input: input,
                            inputGlobal: inputGlobal,
                            inputMeta: inputMeta,
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
