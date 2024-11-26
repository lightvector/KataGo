#!/usr/bin/python3
"""
Convert a trained PyTorch neural network to a CoreML model.

Example usage:
    python3 convert_coreml_pytorch.py -checkpoint b18c384nbt-uec-20221121b.ckpt -use-swa -nbits 8
"""

import argparse
import sys
from typing import Optional, Tuple

import torch
import coremltools as ct
import coremlmish

from load_model import load_model
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpMagnitudePrunerConfig,
    OpPalettizerConfig,
    prune_weights,
    palettize_weights,
    OpLinearQuantizerConfig,
    linear_quantize_weights,
)


def print_versions():
    """Print versions of torch, coremltools, and coremlmish."""
    print(f"torch version: {torch.__version__}")
    print(f"coremltools version: {ct.__version__}")
    # Assuming coremlmish has an attribute __function__; adjust if necessary
    function_name = getattr(coremlmish, "__function__", "Unknown")
    print(f"Using coremlmish function: {function_name}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a trained neural net to a CoreML model."
    )

    parser.add_argument(
        "-checkpoint",
        required=True,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "-use-swa",
        action="store_true",
        help="Use SWA (Stochastic Weight Averaging) model.",
    )
    parser.add_argument(
        "-pos-len",
        type=int,
        default=19,
        help="Position length (default: 19).",
    )
    parser.add_argument(
        "-batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    parser.add_argument(
        "-fp32",
        action="store_true",
        help="Use 32-bit floating-point precision (default: FLOAT16).",
    )
    parser.add_argument(
        "-nbits",
        type=int,
        choices=[8, 6, 4, 3, 2, 1],
        help="Number of bits for palettizing the weights (e.g., 8).",
    )
    parser.add_argument(
        "-sparsity",
        type=float,
        default=0.0,
        help="Target sparsity for pruning the weights (default: 0.0).",
    )
    parser.add_argument(
        "-output",
        required=False,
        help="Path to the converted Core ML package.",
    )

    return parser.parse_args()


def load_traced_model(
    func: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
) -> torch.jit.ScriptModule:
    """Trace the PyTorch model using TorchScript."""
    print("Tracing model ...")
    traced = torch.jit.trace(func, example_inputs)
    return traced


def prepare_example_inputs(
    model,
    batch_size: int,
) -> Tuple[torch.Tensor, ...]:
    """Prepare example inputs for tracing the model."""
    input_spatial = torch.rand(
        batch_size,
        model.bin_input_shape[0],
        model.bin_input_shape[1],
        model.bin_input_shape[2],
    )
    input_global = torch.rand(batch_size, model.global_input_shape[0])
    input_meta = (
        torch.rand(batch_size, model.metadata_encoder.c_input)
        if model.metadata_encoder
        else None
    )

    if input_meta is not None:
        return (input_spatial, input_global, input_meta)
    return (input_spatial, input_global)


def convert_to_coreml(
    traced_model: torch.jit.ScriptModule,
    model,
    input_shapes: Tuple[torch.Size, ...],
    compute_precision: ct.precision,
    minimum_deployment_target: Optional[ct.target],
) -> ct.models.MLModel:
    """Convert the traced PyTorch model to CoreML format."""
    inputs = [ct.TensorType(shape=shape) for shape in input_shapes]

    print("Converting model ...")
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=inputs,
        compute_precision=compute_precision,
        minimum_deployment_target=minimum_deployment_target,
    )

    return mlmodel


def rename_features(spec, old_name: str, new_name: str):
    """Rename a feature in the CoreML model spec."""
    ct.utils.rename_feature(spec, old_name, new_name)


def apply_optimizations(
    mlmodel: ct.models.MLModel,
    sparsity: float,
    nbits: Optional[int],
    joint_compression: bool,
) -> Tuple[ct.models.MLModel, str]:
    """Apply pruning and quantization optimizations to the CoreML model."""
    spec = mlmodel._spec
    compression_description = ""

    # Apply sparsity pruning if requested
    if sparsity > 0:
        sparsity_config = OpMagnitudePrunerConfig(target_sparsity=sparsity)
        pruning_config = OptimizationConfig(global_config=sparsity_config)

        print(f"Pruning weights with {sparsity} sparsity ...")
        mlmodel = prune_weights(mlmodel, config=pruning_config)
        compression_description += f"sparsity {sparsity} "

    # Apply quantization or palettization if nbits is specified
    if nbits is not None:
        if nbits == 8:
            threshold_config = OpLinearQuantizerConfig(
                mode="linear",
            )
            quantizing_config = OptimizationConfig(global_config=threshold_config)

            print(f"Quantizing weights to {nbits} bits ...")
            mlmodel = linear_quantize_weights(
                mlmodel,
                config=quantizing_config,
                joint_compression=joint_compression,
            )
        else:
            palettizing_config = OptimizationConfig(
                global_config=OpPalettizerConfig(
                    nbits=nbits,
                    mode="kmeans",
                    granularity="per_grouped_channel",
                    group_size=4,
                )
            )

            print(f"Palettizing weights with {nbits} bit(s) ...")
            mlmodel = palettize_weights(
                mlmodel,
                palettizing_config,
                joint_compression=joint_compression,
            )

        compression_description += f"quantization bits {nbits} "

    return mlmodel, compression_description


def update_model_metadata(
    mlmodel: ct.models.MLModel,
    pos_len: int,
    precision_name: str,
    version: int,
    sparsity_description: str,
    compression_description: str,
    meta_encoder_version: int,
    checkpoint_file: str,
) -> None:
    """Update the metadata and description of the CoreML model."""
    description = (
        f"KataGo {pos_len}x{pos_len} compute "
        f"precision {precision_name} model version {version} "
        f"{sparsity_description}"
        f"{compression_description}"
        f"meta encoder version {meta_encoder_version} "
        f"converted from {checkpoint_file}"
    )
    mlmodel.short_description = description
    mlmodel.version = f"{version}"


def save_coreml_model(
    mlmodel: ct.models.MLModel,
    pos_len: int,
    precision_name: str,
    meta_encoder_version: int,
    output_path: str,
) -> str:
    """Save the CoreML model to a file and return the file path."""
    if output_path is None:
        meta_encoder_suffix = f"m{meta_encoder_version}" if meta_encoder_version > 0 else ""
        filename = (
            f"KataGoModel{pos_len}x{pos_len}{precision_name}{meta_encoder_suffix}.mlpackage"
        )
    else:
        filename = output_path

    print("Saving model ...")
    mlmodel.save(filename)
    print(f"Saved Core ML model at {filename}")

    return filename


def main():
    """Main function to convert PyTorch model to CoreML."""
    print_versions()

    args = parse_arguments()

    checkpoint_file = args.checkpoint
    use_swa = args.use_swa
    pos_len = args.pos_len
    batch_size = args.batch_size
    fp32 = args.fp32
    nbits = args.nbits
    sparsity = args.sparsity
    output_path = args.output

    # Load the model
    model, swa_model, _ = load_model(
        checkpoint_file=checkpoint_file,
        use_swa=use_swa,
        device="cpu",
        pos_len=pos_len,
        for_coreml=True,
        verbose=True,
    )

    # Select the appropriate model
    func = swa_model if swa_model is not None else model
    print(f"Using model: {func.__class__.__name__}")

    # Determine meta encoder version
    meta_encoder_version = (
        0
        if model.metadata_encoder is None
        else (
            1
            if "meta_encoder_version" not in model.config["metadata_encoder"]
            else model.config["metadata_encoder"]["meta_encoder_version"]
        )
    )
    print(f"Meta encoder version: {meta_encoder_version}")

    # Determine model version with workaround
    version = model.config.get("version", 0)
    if meta_encoder_version > 0:
        version = max(version, 15)
    print(f"Model version: {version}")

    # Prepare example inputs for tracing
    example_inputs = prepare_example_inputs(model, batch_size)

    with torch.no_grad():
        func.eval()
        traced_model = load_traced_model(func, example_inputs)

    # Determine compute precision
    compute_precision = ct.precision.FLOAT32 if fp32 else ct.precision.FLOAT16

    # Determine minimum deployment target
    minimum_deployment_target = (
        ct.target.iOS18 if sparsity or (nbits and nbits != 8) else 
        ct.target.iOS16 if nbits == 8 else 
        None
    )

    # Convert traced model to CoreML
    mlmodel = convert_to_coreml(
        traced_model=traced_model,
        model=model,
        input_shapes=tuple(input.shape for input in example_inputs),
        compute_precision=compute_precision,
        minimum_deployment_target=minimum_deployment_target,
    )

    # Rename input features
    spec = mlmodel._spec
    rename_features(spec, "input_1", "input_global")
    input_names = [input.name for input in spec.description.input]
    print(f"Input names: {input_names}")

    # Rename output features
    output_names = [
        "output_policy",
        "out_value",
        "out_miscvalue",
        "out_moremiscvalue",
        "out_ownership",
    ]

    for i, new_name in enumerate(output_names):
        old_name = spec.description.output[i].name
        rename_features(spec, old_name, new_name)

    print(f"Output names: {output_names}")

    # Determine precision name
    precision_name = "fp32" if fp32 else "fp16"

    # Apply optimizations
    joint_compression = sparsity > 0
    mlmodel, compression_description = apply_optimizations(
        mlmodel=mlmodel,
        sparsity=sparsity,
        nbits=nbits,
        joint_compression=joint_compression,
    )
    sparsity_description = f"sparsity {sparsity} " if sparsity > 0 else ""

    # Update model metadata
    update_model_metadata(
        mlmodel=mlmodel,
        pos_len=pos_len,
        precision_name=precision_name,
        version=version,
        sparsity_description=sparsity_description,
        compression_description=compression_description,
        meta_encoder_version=meta_encoder_version,
        checkpoint_file=checkpoint_file,
    )

    # Rebuild the model with the updated spec
    print("Rebuilding model with updated spec ...")
    rebuilt_mlmodel = ct.models.MLModel(
        mlmodel._spec,
        weights_dir=mlmodel._weights_dir,
    )

    # Save the CoreML model
    save_coreml_model(
        mlmodel=rebuilt_mlmodel,
        pos_len=pos_len,
        precision_name=precision_name,
        meta_encoder_version=meta_encoder_version,
        output_path=output_path,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
