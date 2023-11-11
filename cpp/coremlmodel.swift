//
//  coremlmodel.swift
//  KataGo
//
//  Created by Chin-Chang Yang on 2023/11/7.
//

import CryptoKit
import Foundation
import CoreML
import OSLog

class KataGoModelInput: MLFeatureProvider {
    var input_spatial: MLMultiArray
    var input_global: MLMultiArray

    var featureNames: Set<String> {
        return Set(["input_spatial", "input_global"])
    }

    init(input_spatial: MLMultiArray, input_global: MLMultiArray) {
        self.input_spatial = input_spatial
        self.input_global = input_global
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "input_spatial") {
            return MLFeatureValue(multiArray: input_spatial)
        } else if (featureName == "input_global") {
            return MLFeatureValue(multiArray: input_global)
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

    class func compileAppMLModel(modelName: String) -> MLModel? {
        var mlmodel: MLModel?

        do {
            // Get URL of the MLModel at Application Support Directory
            let modelURL = try getAppMLModelURL(modelName: modelName)

            // Check the MLModel is reachable
            let isReachable = try modelURL.checkResourceIsReachable()

            if (isReachable) {
                // Compile MLModel if the MLModel is reachable
                mlmodel = try compileMLModel(modelName: modelName, modelURL: modelURL)
            }
        } catch {
            Logger().error("An error occurred: \(error)")
        }

        return mlmodel;
    }

    class func compileBundleMLModel(modelName: String) -> MLModel? {
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
            mlmodel = try compileMLModel(modelName: modelName, modelURL: bundleModelURL)

            // Get model URL at App Support Directory
            let appModelURL = try getAppMLModelURL(modelName: modelName)

            // Get default file manager
            let fileManager = FileManager.default

            Logger().info("Removing old CoreML model in Application Support directory \(appModelURL)");

            // Remove the old model in Application Support directory
            try fileManager.removeItem(at: appModelURL)

            Logger().info("Copying bundle CoreML model to Application Support directory \(appModelURL)")

            // Copy the mlpackage to App Support Directory
            try fileManager.copyItem(at: bundleModelURL, to: appModelURL)
        } catch {
            Logger().error("An error occurred: \(error)")
        }

        return mlmodel;
    }

    class func compileMLModel(modelName: String, modelURL: URL) throws -> MLModel {
        // Get compiled model name
        let compiledModelName = "\(modelName).mlmodelc"

        // Set the directory for KataGo models
        let directory = "KataGoModels"

        // Get path component
        let pathComponent = "\(directory)/\(compiledModelName)"

        // Get default file manager
        let fileManager = FileManager.default

        // Get application support directory
        // Create the directory if it does not already exist
        let appSupportURL = try fileManager.url(for: .applicationSupportDirectory,
                                                in: .userDomainMask,
                                                appropriateFor: nil,
                                                create: true)

        // Create the URL for the permanent compiled model file
        let permanentURL = appSupportURL.appending(component: pathComponent)

        // Initialize model
        var model: MLModel

        // Create the URL for the model data file
        let dataURL = modelURL.appending(component: "Data/com.apple.CoreML/model.mlmodel")

        // Get model data
        let modelData = try Data(contentsOf: dataURL)

        // Get SHA256 data
        let hashData = Data(SHA256.hash(data: modelData).makeIterator())

        // Get hash digest
        let digest = hashData.map { String(format: "%02x", $0) }.joined()

        // Set digest path
        let savedDigestPath = "\(directory)/\(modelName).digest"

        // Get digest URL
        let savedDigestURL = appSupportURL.appending(component: savedDigestPath)

        // Get saved digest
        var isChangedDigest = true

        do {
            if (try savedDigestURL.checkResourceIsReachable()) {
                let savedDigest = try String(contentsOf: savedDigestURL, encoding: .utf8)

                // Check the saved digest is changed or not
                isChangedDigest = digest != savedDigest

                if (isChangedDigest) {
                    Logger().info("Compiling CoreML model because the digest has changed");
                }
            } else {
                Logger().info("Compiling CoreML model because the saved digest URL is not reachable: \(savedDigestURL)")
            }
        } catch {
            Logger().warning("Compiling CoreML model because it is unable to get the saved digest from: \(savedDigestURL)")
        }

        // Check permanent compiled model is reachable
        let reachableModel = try permanentURL.checkResourceIsReachable()

        if (!reachableModel) {
            Logger().info("Compiling CoreML model because it is not reachable");
        }

        // Model should be compiled if the compiled model is not reachable or the digest changes
        let shouldCompile = !reachableModel || isChangedDigest;

        if (shouldCompile) {
            Logger().info("Compiling CoreML model at \(modelURL)");

            // Compile the model
            let compiledURL = try MLModel.compileModel(at: modelURL)

            Logger().info("Copying the compiled CoreML model to the permanent location \(permanentURL)");

            // Create the directory for KataGo models
            try fileManager.createDirectory(at: appSupportURL.appending(component: directory),
                                            withIntermediateDirectories: true)

            // Copy the file to the to the permanent location, replacing it if necessary
            try fileManager.replaceItem(at: permanentURL,
                                        withItemAt: compiledURL,
                                        backupItemName: nil,
                                        options: .usingNewMetadataOnly,
                                        resultingItemURL: nil)

            // Update the digest
            try digest.write(to: savedDigestURL, atomically: true, encoding: .utf8)
        }

        // Initialize the model configuration
        let configuration = MLModelConfiguration()

        // Set the compute units to CPU and Neural Engine
        configuration.computeUnits = MLComputeUnits.cpuAndNeuralEngine

        // Set the model display name
        configuration.modelDisplayName = modelName;

        Logger().info("Creating CoreML model with contents \(permanentURL)");

        // Create the model
        model = try MLModel(contentsOf: permanentURL, configuration: configuration)

        let description: String = model.modelDescription.metadata[MLModelMetadataKey.description] as! String? ?? "Unknown"

        Logger().info("Created CoreML model: \(description)");

        // Return the model
        return model;
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
    
    func prediction(from input: KataGoModelInput,
                    options: MLPredictionOptions) throws -> KataGoModelOutput {

        let outFeatures = try model.prediction(from: input, options: options)
        return createOutput(from: outFeatures)
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
