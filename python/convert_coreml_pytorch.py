#!/usr/bin/python3
# Example: python3 convert_coreml_pytorch.py -checkpoint b18c384nbt-uec-20221121b.ckpt -use-swa
import argparse
import torch
from load_model import load_model
import coremltools as ct
import coremlmish
import coremllogsumexp

description = """
Convert a trained neural net to a CoreML model.
"""

# Print torch version
print(f'torch version: {torch.__version__}')

# Print coremltools version
print(f'coremltools version: {ct.__version__}')

# Print coremlmish function
print(f'Using coremlmish function: {coremlmish.__function__}')

# Print coremllogsumexp name
print(f'Using {coremllogsumexp.__name__}')


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description=description)

    # Add an argument of checkpoint file
    parser.add_argument(
        '-checkpoint', help='Checkpoint to test', required=True)

    # Add an argument of use swa
    parser.add_argument('-use-swa', help='Use SWA model',
                        action="store_true", required=False)

    # Add an argument of position length
    parser.add_argument('-pos-len', help='Position length',
                        type=int, required=False)

    # Add an argument of batch size
    parser.add_argument('-batch-size', help='Batch size',
                        type=int, required=False)

    # Parse the arguments
    args = vars(parser.parse_args())

    # Get the argument of checkpoint file
    checkpoint_file = args["checkpoint"]

    # Get the argument of use swa
    use_swa = args["use_swa"]

    # Get the argument of position length
    pos_len = args['pos_len'] if args['pos_len'] else 19

    # Get the argument of batch size
    batch_size = args['batch_size'] if args['batch_size'] else 1

    # Load the model
    model, swa_model, _ = load_model(
        checkpoint_file,
        use_swa, device="cpu",
        pos_len=pos_len,
        for_coreml=True,
        verbose=True)

    # Set the model
    func = model if swa_model is None else swa_model

    # Print the model name
    print(f'Using model: {func.__class__.__name__}')

    # Get the model version
    version = model.config['version']

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

        # Trace the model
        traced_model = torch.jit.trace(
            func, (input_spatial, input_global))

        # Convert the model
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_spatial.shape),
                    ct.TensorType(shape=input_global.shape)],
        )

        # Get the protobuf spec
        spec = mlmodel.get_spec()

        # Rename the input
        ct.utils.rename_feature(spec, 'input_1', 'input_global')

        # Get input names
        input_names = [input.name for input in spec.description.input]

        # Print the input names
        print(f'Input names: {input_names}')

        # Rename the output
        ct.utils.rename_feature(spec, 'var_2462', 'output_policy')
        ct.utils.rename_feature(spec, 'var_2503', 'out_value')
        ct.utils.rename_feature(spec, 'var_2506', 'out_miscvalue')
        ct.utils.rename_feature(spec, 'var_2509', 'out_moremiscvalue')
        ct.utils.rename_feature(spec, 'var_2514', 'out_ownership')

        # Get output names
        output_names = [output.name for output in spec.description.output]

        # Print the output names
        print(f'Output names: {output_names}')

        # Reload the model with the updated spec
        mlmodel = ct.models.MLModel(spec)

        # Set file name
        mlmodel_file = f'KataGoModel{pos_len}x{pos_len}v{version}.mlmodel'

        # Set model description
        mlmodel.short_description = f'KataGo {pos_len}x{pos_len} model version {version} converted from {checkpoint_file}'

        # Set model version
        mlmodel.version = f'{version}'

        # Save the model
        mlmodel.save(mlmodel_file)

        # Print the file name
        print(f'Saved Core ML model at {mlmodel_file}')


if __name__ == "__main__":
    main()
