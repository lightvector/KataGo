//
//  coremlmodel.swift
//  KataGo
//
//  Created by Chin-Chang Yang on 2023/11/7.
//

import CryptoKit
import Foundation
import CoreML

class KataGoModelInput: MLFeatureProvider {
    var input_spatial: MLMultiArray
    var input_global: MLMultiArray
    var input_meta: MLMultiArray?

    var featureNames: Set<String> {
        return Set(["input_spatial", "input_global", "input_meta"])
    }

    init(input_spatial: MLMultiArray, input_global: MLMultiArray) {
        self.input_spatial = input_spatial
        self.input_global = input_global
    }

    init(input_spatial: MLMultiArray, input_global: MLMultiArray, input_meta: MLMultiArray) {
        self.input_spatial = input_spatial
        self.input_global = input_global
        self.input_meta = input_meta
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "input_spatial") {
            return MLFeatureValue(multiArray: input_spatial)
        } else if (featureName == "input_global") {
            return MLFeatureValue(multiArray: input_global)
        } else if (featureName == "input_meta"), let input_meta {
            return MLFeatureValue(multiArray: input_meta)
        } else {
            return nil
        }
    }
}

class KataGoModelInputBatch: MLBatchProvider {
    var inputArray: [KataGoModelInput]

    var count: Int {
        inputArray.count
    }

    func features(at index: Int) -> MLFeatureProvider {
        return inputArray[index]
    }

    init(inputArray: [KataGoModelInput]) {
        self.inputArray = inputArray
    }
}

class KataGoModelOutput: MLFeatureProvider {
    var output_policy: MLMultiArray
    var out_value: MLMultiArray
    var out_miscvalue: MLMultiArray
    var out_moremiscvalue: MLMultiArray
    var out_ownership: MLMultiArray

    var featureNames: Set<String> {
        return Set(["output_policy",
                    "out_value",
                    "out_miscvalue",
                    "out_moremiscvalue",
                    "out_ownership"])
    }

    init(output_policy: MLMultiArray,
         out_value: MLMultiArray,
         out_miscvalue: MLMultiArray,
         out_moremiscvalue: MLMultiArray,
         out_ownership: MLMultiArray) {
        self.output_policy = output_policy
        self.out_value = out_value
        self.out_miscvalue = out_miscvalue
        self.out_moremiscvalue = out_moremiscvalue
        self.out_ownership = out_ownership
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "output_policy") {
            return MLFeatureValue(multiArray: output_policy)
        } else if (featureName == "out_value") {
            return MLFeatureValue(multiArray: out_value)
        } else if (featureName == "out_miscvalue") {
            return MLFeatureValue(multiArray: out_miscvalue)
        } else if (featureName == "out_moremiscvalue") {
            return MLFeatureValue(multiArray: out_moremiscvalue)
        } else if (featureName == "out_ownership") {
            return MLFeatureValue(multiArray: out_ownership)
        } else {
            return nil
        }
    }
}

class KataGoModelOutputBatch: MLBatchProvider {
    var outputArray: [KataGoModelOutput]

    var count: Int {
        outputArray.count
    }

    func features(at index: Int) -> MLFeatureProvider {
        return outputArray[index]
    }

    init(outputArray: [KataGoModelOutput]) {
        self.outputArray = outputArray
    }
}

class KataGoModel {
    let model: MLModel

    class func getAppMLModelURL(modelName: String) throws -> URL {
        // Get model package name
        let mlpackageName = "\(modelName).mlpackage"

        // Set the directory for KataGo models
        let directory = "KataGoModels"

        // Get path component
        let pathComponent = "\(directory)/\(mlpackageName)"

        // Get default file manager
        let fileManager = FileManager.default

        // Get application support directory
        // Create the directory if it does not already exist
        let appSupportURL = try fileManager.url(for: .applicationSupportDirectory,
                                                in: .userDomainMask,
                                                appropriateFor: nil,
                                                create: true)

        // Create the URL for the model package file
        let modelURL = appSupportURL.appending(component: pathComponent)

        return modelURL;
    }

    class func compileAppMLModel(modelName: String, useCpuAndNeuralEngine: Bool) -> MLModel? {
        var mlmodel: MLModel?

        do {
            // Get URL of the MLModel at Application Support Directory
            let modelURL = try getAppMLModelURL(modelName: modelName)

            // Check the MLModel is reachable
            let isReachable = try modelURL.checkResourceIsReachable()

            if (isReachable) {
                // Compile MLModel if the MLModel is reachable
                mlmodel = try compileMLModel(modelName: modelName,
                                             modelURL: modelURL,
                                             useCpuAndNeuralEngine: useCpuAndNeuralEngine)
            }
        } catch {
            printError("An error occurred: \(error)")
        }

        return mlmodel;
    }

    class func compileBundleMLModel(modelName: String, useCpuAndNeuralEngine: Bool) -> MLModel? {
        var mlmodel: MLModel?

        do {
            // Set model type name
            let typeName = "mlpackage"

            // Get model path from bundle resource
            // Fallback to create a default model path
            let modelPath = Bundle.main.path(forResource: modelName, ofType: typeName) ?? "\(modelName).\(typeName)"

            // Get model URL at bundle
            let bundleModelURL = URL(filePath: modelPath)

            // Compile MLModel
            mlmodel = try compileMLModel(modelName: modelName,
                                         modelURL: bundleModelURL,
                                         useCpuAndNeuralEngine: useCpuAndNeuralEngine)

            // Get model URL at App Support Directory
            let appModelURL = try getAppMLModelURL(modelName: modelName)

            // Get default file manager
            let fileManager = FileManager.default

            do {
                if try appModelURL.checkResourceIsReachable() {
                    printError("Removing old CoreML model in Application Support directory \(appModelURL)");

                    do {
                        // Remove the old model in Application Support directory
                        try fileManager.removeItem(at: appModelURL)
                    } catch {
                        printError("Unable to remove the old CoreML model in Application Support directory \(appModelURL): \(error)")
                    }
                }
            } catch {
                printError("Unable to check if the old CoreML model is reachable in Application Support directory \(appModelURL)")
            }

            printError("Copying bundle CoreML model to Application Support directory \(appModelURL)")

            // Copy the mlpackage to App Support Directory
            try fileManager.copyItem(at: bundleModelURL, to: appModelURL)
        } catch {
            printError("An error occurred: \(error)")
        }

        return mlmodel;
    }

    private class func getApplicationSupportURL() throws -> URL {
        // Get default file manager
        let fileManager = FileManager.default

        return try fileManager.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask,
                                   appropriateFor: nil,
                                   create: true)
    }

    private class func getDigest(modelURL: URL) throws -> String {
        // Create the URL for the model data file
        let dataURL = modelURL.appending(component: "Data/com.apple.CoreML/model.mlmodel")

        // Get model data
        let modelData = try Data(contentsOf: dataURL)

        // Get SHA256 data
        let hashData = Data(SHA256.hash(data: modelData).makeIterator())

        // Get hash digest
        let digest = hashData.map { String(format: "%02x", $0) }.joined()

        return digest
    }

    private class func checkShouldCompileModel(permanentURL: URL,
                                               savedDigestURL: URL,
                                               modelURL: URL,
                                               digest: String) -> Bool {
        // Model should be compiled if the compiled model is not reachable or the digest changes
        var shouldCompile = true

        // Get saved digest
        do {
            if (try savedDigestURL.checkResourceIsReachable()) {
                let savedDigest = try String(contentsOf: savedDigestURL, encoding: .utf8)

                // Check the saved digest is changed or not
                shouldCompile = digest != savedDigest

                if (shouldCompile) {
                    printError("Saved digest: \(savedDigest)")
                    printError("New digest: \(digest)")
                    printError("Compiling CoreML model because the digest has changed");
                }
            } else {
                printError("Compiling CoreML model because the saved digest URL is not reachable: \(savedDigestURL)")
            }
        } catch {
            printError("Compiling CoreML model because it is unable to get the saved digest from: \(savedDigestURL)")
        }

        if !shouldCompile {
            // Check permanent compiled model is reachable
            do {
                shouldCompile = try (!permanentURL.checkResourceIsReachable())

                if (shouldCompile) {
                    printError("Compiling CoreML model because the permanent URL is not reachable: \(permanentURL)");
                }
            } catch {
                shouldCompile = true

                printError("Compiling CoreML model because it is unable to check the resource at: \(permanentURL)")
            }
        }

        return shouldCompile
    }

    private class func compileAndSaveModel(permanentURL: URL,
                                           savedDigestURL: URL,
                                           modelURL: URL,
                                           digest: String) throws {
        // Get default file manager
        let fileManager = FileManager.default

        printError("Compiling CoreML model at \(modelURL)");

        // Compile the model
        let compiledURL = try MLModel.compileModel(at: modelURL)

        printError("Creating the directory for the permanent location: \(permanentURL)");

        // Create the directory for KataGo models
        try fileManager.createDirectory(at: permanentURL.deletingLastPathComponent(),
                                        withIntermediateDirectories: true)

        printError("Copying the compiled CoreML model to the permanent location \(permanentURL)");

        // Copy the file to the to the permanent location, replacing it if necessary
        try fileManager.replaceItem(at: permanentURL,
                                    withItemAt: compiledURL,
                                    backupItemName: nil,
                                    options: .usingNewMetadataOnly,
                                    resultingItemURL: nil)

        printError("Writing digest to: \(savedDigestURL)")
        printError("Digest: \(digest)")

        // Update the digest
        try digest.write(to: savedDigestURL, atomically: true, encoding: .utf8)
    }

    private class func loadModel(permanentURL: URL, modelName: String, useCpuAndNeuralEngine: Bool) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = useCpuAndNeuralEngine ? .cpuAndNeuralEngine : .all
        configuration.modelDisplayName = modelName
        printError("Creating CoreML model with contents \(permanentURL)")
        return try MLModel(contentsOf: permanentURL, configuration: configuration)
    }

    class func compileMLModel(modelName: String, modelURL: URL, useCpuAndNeuralEngine: Bool) throws -> MLModel {
        let appSupportURL = try getApplicationSupportURL()
        let permanentURL = appSupportURL.appending(component: "KataGoModels/\(modelName).mlmodelc")
        let savedDigestURL = appSupportURL.appending(component: "KataGoModels/\(modelName).digest")
        let digest = try getDigest(modelURL: modelURL)

        let shouldCompileModel = checkShouldCompileModel(permanentURL: permanentURL,
                                                         savedDigestURL: savedDigestURL,
                                                         modelURL: modelURL,
                                                         digest: digest)

        if shouldCompileModel {
            try compileAndSaveModel(permanentURL: permanentURL,
                                    savedDigestURL: savedDigestURL,
                                    modelURL: modelURL,
                                    digest: digest)
        }

        return try loadModel(permanentURL: permanentURL,
                             modelName: modelName,
                             useCpuAndNeuralEngine: useCpuAndNeuralEngine);
    }

    init(model: MLModel) {
        self.model = model
    }

    private func createOutput(from outFeatures: MLFeatureProvider) -> KataGoModelOutput {

        let output_policy = (outFeatures.featureValue(for: "output_policy")?.multiArrayValue)!
        let out_value = (outFeatures.featureValue(for: "out_value")?.multiArrayValue)!
        let out_miscvalue = (outFeatures.featureValue(for: "out_miscvalue")?.multiArrayValue)!
        let out_moremiscvalue = (outFeatures.featureValue(for: "out_moremiscvalue")?.multiArrayValue)!
        let out_ownership = (outFeatures.featureValue(for: "out_ownership")?.multiArrayValue)!

        return KataGoModelOutput(output_policy: output_policy,
                                 out_value: out_value,
                                 out_miscvalue: out_miscvalue,
                                 out_moremiscvalue: out_moremiscvalue,
                                 out_ownership: out_ownership)
    }

    func prediction(from inputBatch: KataGoModelInputBatch,
                    options: MLPredictionOptions) throws -> KataGoModelOutputBatch {

        let outFeaturesBatch = try model.predictions(from: inputBatch, options: options)
        let outputArray = (0..<outFeaturesBatch.count).map { index -> KataGoModelOutput in
            let outFeatures = outFeaturesBatch.features(at: index)
            return createOutput(from: outFeatures)
        }

        return KataGoModelOutputBatch(outputArray: outputArray)
    }
}
