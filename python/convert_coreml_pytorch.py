#!/usr/bin/python3
# Example: python3 convert_coreml_pytorch.py -checkpoint b18c384nbt-uec-20221121b.ckpt -use-swa
import argparse
import torch
from load_model import load_model
import coremltools as ct
import coremlmish

from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpMagnitudePrunerConfig,
    OpPalettizerConfig,
    prune_weights,
    palettize_weights,
    OpLinearQuantizerConfig,
    linear_quantize_weights,
)

description = """
Convert a trained neural net to a CoreML model.
"""

# Print torch version
print(f"torch version: {torch.__version__}")

# Print coremltools version
print(f"coremltools version: {ct.__version__}")

# Print coremlmish function
print(f"Using coremlmish function: {coremlmish.__function__}")


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description=description)

    # Add an argument of checkpoint file
    parser.add_argument("-checkpoint", help="Checkpoint to test", required=True)

    # Add an argument of use swa
    parser.add_argument(
        "-use-swa", help="Use SWA model", action="store_true", required=False
    )

    # Add an argument of position length
    parser.add_argument("-pos-len", help="Position length", type=int, required=False)

    # Add an argument of batch size
    parser.add_argument("-batch-size", help="Batch size", type=int, required=False)

    # Add an argument of 32-bit floating-point
    parser.add_argument(
        "-fp32", help="32-bit floating-point", action="store_true", required=False
    )

    # Add an argument of the number of bits to use for palettizing the weights
    parser.add_argument(
        "-nbits",
        help="Number of bits to use for palettizing the weights",
        type=int,
        required=False,
    )

    # Add an argument of the target sparsity for pruning the weights
    parser.add_argument(
        "-sparsity",
        help="Target sparsity to use for pruning the weights",
        type=float,
        required=False,
    )

    # Parse the arguments
    args = vars(parser.parse_args())

    # Get the argument of checkpoint file
    checkpoint_file = args["checkpoint"]

    # Get the argument of use swa
    use_swa = args["use_swa"]

    # Get the argument of position length
    pos_len = args["pos_len"] if args["pos_len"] else 19

    # Get the argument of batch size
    batch_size = args["batch_size"] if args["batch_size"] else 1

    # Get the argument of 32-bit floating-point
    fp32 = args["fp32"]

    # Get the argument of the number of bits to use for palettizing the weights
    nbits = args["nbits"]

    # Get the argument of the target sparsity for pruning the weights
    sparsity = args["sparsity"] if args["sparsity"] else 0.0

    # Load the model
    model, swa_model, _ = load_model(
        checkpoint_file,
        use_swa,
        device="cpu",
        pos_len=pos_len,
        for_coreml=True,
        verbose=True,
    )

    # Set the model
    func = model if swa_model is None else swa_model

    # Print the model name
    print(f"Using model: {func.__class__.__name__}")

    # Get the meta encoder version
    meta_encoder_version = (
        0
        if model.metadata_encoder is None
        else (
            1
            if "meta_encoder_version" not in model.config["metadata_encoder"]
            else model.config["metadata_encoder"]["meta_encoder_version"]
        )
    )

    # Print the meta encoder version
    print(f"Meta encoder version: {meta_encoder_version}")

    # Get the model version
    version = model.config["version"]

    # Workaround for incorrect model version
    version = max(version, 15) if meta_encoder_version > 0 else version

    # Print the model version
    print(f"Model version: {version}")

    with torch.no_grad():
        # Set the model to eval mode
        func.eval()

        # NCHW
        input_spatial = torch.rand(
            batch_size,
            model.bin_input_shape[0],
            model.bin_input_shape[1],
            model.bin_input_shape[2],
        )

        # NC
        input_global = torch.rand(batch_size, model.global_input_shape[0])

        # NC
        input_meta = (
            torch.rand(batch_size, model.metadata_encoder.c_input)
            if model.metadata_encoder is not None
            else None
        )

        # Set the example inputs
        example_inputs = (
            (input_spatial, input_global, input_meta)
            if input_meta is not None
            else (input_spatial, input_global)
        )

        # Trace the model
        print(f"Tracing model ...")
        traced_model = torch.jit.trace(func, example_inputs)

        # Set the compute precision
        compute_precision = ct.precision.FLOAT16 if not fp32 else ct.precision.FLOAT32

        # Set the input types
        inputs = (
            [
                ct.TensorType(shape=input_spatial.shape),
                ct.TensorType(shape=input_global.shape),
                ct.TensorType(shape=input_meta.shape),
            ]
            if input_meta is not None
            else [
                ct.TensorType(shape=input_spatial.shape),
                ct.TensorType(shape=input_global.shape),
            ]
        )

        # Define the minimum deployment target
        minimum_deployment_target = ct.target.iOS18 if nbits != None else None

        # Convert the model
        print(f"Converting model ...")

        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=inputs,
            compute_precision=compute_precision,
            minimum_deployment_target=minimum_deployment_target,
        )

        # Get the protobuf spec
        spec = mlmodel._spec

        # Rename the input
        ct.utils.rename_feature(spec, "input_1", "input_global")

        # Get input names
        input_names = [input.name for input in spec.description.input]

        # Print the input names
        print(f"Input names: {input_names}")

        # Set output names
        output_names = [
            "output_policy",
            "out_value",
            "out_miscvalue",
            "out_moremiscvalue",
            "out_ownership",
        ]

        # Rename output names
        for i, name in enumerate(output_names):
            # Rename the output
            ct.utils.rename_feature(spec, spec.description.output[i].name, name)

        # Print the output names
        print(f"Output names: {output_names}")

        # Set the compute precision name
        precision_name = "fp16" if not fp32 else "fp32"

        # Set the meta encoder name
        meta_encoder_name = (
            "" if meta_encoder_version == 0 else f"m{meta_encoder_version}"
        )

        if sparsity > 0:
            # Define sparsity configuration
            sparsity_config = OpMagnitudePrunerConfig(target_sparsity=sparsity)

            # Define pruning config
            pruning_config = OptimizationConfig(global_config=sparsity_config)

            # Prune weights
            print(f"Pruning weights with {sparsity} sparsity ...")
            pruned_mlmodel = prune_weights(mlmodel, config=pruning_config)

            # Enable joint compression
            joint_compression = True

            # Sparsity description
            sparsity_description = f"sparsity {sparsity} "
        else:
            # Model without pruning
            pruned_mlmodel = mlmodel

            # Disable joint compression
            joint_compression = False

            # No sparsity description
            sparsity_description = ""

        if nbits != None:
            if nbits == 8:
                # Define weight threshold configuration
                weight_threshold = 2048
                threshold_config = OpLinearQuantizerConfig(
                    mode="linear_symmetric", weight_threshold=weight_threshold
                )

                # Define quantization config
                quantizing_config = OptimizationConfig(global_config=threshold_config)

                # Quantize weights
                print(f"Quantizing weights to 8 bits with the threshold {weight_threshold} ...")
                compressed_mlmodel = linear_quantize_weights(
                    pruned_mlmodel,
                    config=quantizing_config,
                    joint_compression=joint_compression,
                )
            else:
                # Define compressor configuration
                nbits_config = OpPalettizerConfig(nbits=nbits)

                # Define palettization config
                palettizing_config = OptimizationConfig(global_config=nbits_config)

                # Palettize weights
                print(f"Palettizing weights with {nbits} bit(s) ...")
                compressed_mlmodel = palettize_weights(
                    pruned_mlmodel,
                    palettizing_config,
                    joint_compression=joint_compression,
                )

            # Compression description
            compression_description = f"quantization bits {nbits} "
        else:
            # Uncompressed model
            compressed_mlmodel = pruned_mlmodel

            # No compression description for the uncompressed model
            compression_description = ""

        # Set model description
        compressed_mlmodel.short_description = (
            f"KataGo {pos_len}x{pos_len} compute "
            f"precision {precision_name} model version {version} "
            f"{sparsity_description}"
            f"{compression_description}"
            f"meta encoder version {meta_encoder_version} "
            f"converted from {checkpoint_file}"
        )

        # Set model version
        compressed_mlmodel.version = f"{version}"

        # Rebuild the model with the updated spec
        print(f"Rebuilding model with updated spec ...")
        rebuilt_mlmodel = ct.models.MLModel(
            compressed_mlmodel._spec, weights_dir=compressed_mlmodel._weights_dir
        )

        # Set file name
        mlmodel_file = f"KataGoModel{pos_len}x{pos_len}{precision_name}{meta_encoder_name}.mlpackage"

        # Save the model
        print(f"Saving model ...")
        rebuilt_mlmodel.save(mlmodel_file)

        # Print the file name
        print(f"Saved Core ML model at {mlmodel_file}")


if __name__ == "__main__":
    main()
