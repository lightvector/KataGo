#!/usr/bin/python3
import argparse
import torch
from load_model import load_model
import coremltools as ct
from coremltools import _logger as logger

description = """
Convert a trained neural net to a CoreML model.
"""

# Print coremltools version
print(ct.__version__)

# Parse arguments

parser = argparse.ArgumentParser(description=description)
args = vars(parser.parse_args())


def main(args):
    #logger.setLevel('INFO')
    checkpoint_file = 'b18c384nbt-uec-20221121b.ckpt'  # args["checkpoint"]
    use_swa = True  # args["use_swa"]
    pos_len = 19
    batch_size = 1

    model, swa_model, other_state_dict = load_model(
        checkpoint_file,
        use_swa, device="cpu",
        pos_len=pos_len,
        for_coreml=True,
        verbose=True)

    version = model.config['version']

    with torch.no_grad():
        model.eval()
        if swa_model is not None:
            swa_model.eval()

        # NCHW
        input_spatial = torch.rand(
            batch_size,
            model.bin_input_shape[0],
            model.bin_input_shape[1],
            model.bin_input_shape[2],
        )

        input_global = torch.rand(batch_size, model.global_input_shape[0])

        traced_model = torch.jit.trace(
            swa_model, (input_spatial, input_global))

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_spatial.shape), ct.TensorType(shape=input_global.shape)],
        )

        mlmodel_file = f'KataGoModel{pos_len}x{pos_len}.mlmodel'
        mlmodel.short_description = f'KataGo {pos_len}x{pos_len} model version {version} converted from {checkpoint_file}'
        mlmodel.version = f'{version}'
        mlmodel.save(mlmodel_file)
        print(f'Core ML model saved at {mlmodel_file}')

if __name__ == "__main__":
    main(args)
