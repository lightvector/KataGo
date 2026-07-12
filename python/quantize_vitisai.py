#!/usr/bin/env python3
"""Quantize a KataGo FP32 ONNX graph to INT8 QDQ for the VitisAI (AMD Ryzen AI NPU) execution
provider, for use with katago.exe's ONNX backend (onnxProvider=vitisai).

Offline tool -- katago.exe does NOT invoke this automatically. Requires `amd-quark` (AMD's ONNX
quantizer, providing the `quark.onnx` module, bundled with the Ryzen AI / VitisAI SDK), `onnx`,
and `numpy`.

Typical usage: first export an FP32 ONNX graph from a .bin.gz KataGo model (`katago exportonnx`),
sample calibration data from real games (`katago dumpcalibrationdata`), then run:

    python quantize_vitisai.py --input model-fp32.onnx --calibration calib.npz --output model-int8.onnx
"""
import argparse
import sys

import numpy as np
import onnx


def load_calibration(npz_path):
    data = np.load(npz_path)
    return data["binaryInputNCHW"], data["globalInputNC"]


def classify_input_name(name):
    lname = name.lower()
    if "mask" in lname:
        return "mask"
    if "spatial" in lname:
        return "spatial"
    if "global" in lname:
        return "global"
    if "meta" in lname:
        return "meta"
    return None


class KataGoCalibrationDataReader:
    """Feeds calibration rows sampled by KataGo's dumpcalibrationdata.
    binaryInputNCHW has shape [N, spatialChannels, H, W] where channel 0 is the on-board mask
    (KataGo convention); globalInputNC has shape [N, globalChannels]. These get mapped onto
    whatever the graph's actual input names are (InputSpatial/InputMask/InputGlobal/InputMeta by
    default, but matched generically here by substring in case of custom names).
    """

    def __init__(self, spatial, global_, input_names):
        self.spatial = spatial
        self.global_ = global_
        self.input_names = input_names
        self.idx = 0
        self.n = spatial.shape[0]

    def get_next(self):
        if self.idx >= self.n:
            return None
        i = self.idx
        self.idx += 1
        feed = {}
        if "spatial" in self.input_names:
            feed[self.input_names["spatial"]] = self.spatial[i : i + 1]
        if "mask" in self.input_names:
            feed[self.input_names["mask"]] = self.spatial[i : i + 1, 0:1]
        if "global" in self.input_names:
            g = self.global_[i : i + 1]
            feed[self.input_names["global"]] = g.reshape(g.shape[0], g.shape[1], 1, 1)
        if "meta" in self.input_names:
            feed[self.input_names["meta"]] = np.zeros(self.input_names["meta_shape"], dtype=np.float32)
        return feed

    def rewind(self):
        self.idx = 0


def build_input_map(model):
    input_names = {}
    for inp in model.graph.input:
        role = classify_input_name(inp.name)
        if role is None:
            print(f'[quantize_vitisai] warning: unrecognized graph input "{inp.name}", ignoring', flush=True)
            continue
        input_names[role] = inp.name
        if role == "meta":
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
            input_names["meta_shape"] = tuple(dims)
    return input_names


def quantize(input_path, output_path, reader):
    # Preferred path: quark's high-level ModelQuantizer with the XINT8 preset config, which is
    # AMD's own recommended power-of-2-scale INT8 scheme for Ryzen AI NPU CNN deployment.
    try:
        from quark.onnx import XINT8_QCONFIG, ModelQuantizer

        print("[quantize_vitisai] using quark.onnx ModelQuantizer with XINT8_QCONFIG preset", flush=True)
        quantizer = ModelQuantizer(XINT8_QCONFIG)
        result = quantizer.quantize_model(
            model_input=input_path,
            model_output=output_path,
            calibration_data_reader=reader,
        )
        if result is not None and not _file_nonempty(output_path):
            onnx.save(result, output_path)
        return
    except ImportError as e:
        print(f"[quantize_vitisai] quark.onnx ModelQuantizer/XINT8_QCONFIG unavailable ({e}), "
              f"falling back to quark.onnx.quantize_static", flush=True)

    # Fallback: the lower-level quantize_static function. XINT8_QCONFIG (the preferred path above)
    # is int8 with POWER-OF-2 scaling and MinMSE calibration (quark.onnx.quantization.config.spec.
    # XInt8Spec) -- NOT the same as generic int8 with float32 scaling and MinMax calibration, which
    # is quark's own *default* if calibrate_method/scale type are left unspecified here. Using the
    # wrong scale representation produces a QDQ graph the VitisAI DPU compiler's pattern-matcher
    # does not recognize as fusable, so activation_type/weight_type alone are not sufficient --
    # calibrate_method must explicitly be quark's PowerOfTwoMethod, not onnxruntime's CalibrationMethod.
    reader.rewind()
    import quark.onnx as qo
    from onnxruntime.quantization import QuantFormat, QuantType
    from quark.onnx.calibration import PowerOfTwoMethod

    qo.quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=PowerOfTwoMethod.MinMSE,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        enable_npu_cnn=True,
    )


def _file_nonempty(path):
    try:
        import os

        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input FP32 ONNX model path")
    parser.add_argument("--calibration", required=True, help="Calibration .npz path (binaryInputNCHW, globalInputNC)")
    parser.add_argument("--output", required=True, help="Output INT8 QDQ ONNX model path")
    args = parser.parse_args()

    print(f"[quantize_vitisai] loading model: {args.input}", flush=True)
    model = onnx.load(args.input)

    input_names = build_input_map(model)
    printable = {k: v for k, v in input_names.items() if k != "meta_shape"}
    print(f"[quantize_vitisai] graph inputs mapped: {printable}", flush=True)
    if "spatial" not in input_names or "global" not in input_names:
        print("[quantize_vitisai] ERROR: could not find InputSpatial/InputGlobal in the graph", flush=True)
        sys.exit(1)

    print(f"[quantize_vitisai] loading calibration data: {args.calibration}", flush=True)
    spatial, global_ = load_calibration(args.calibration)
    print(f"[quantize_vitisai] calibration: {spatial.shape[0]} positions", flush=True)

    reader = KataGoCalibrationDataReader(spatial, global_, input_names)

    try:
        quantize(args.input, args.output, reader)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure must be reported, not swallowed
        print(f"[quantize_vitisai] ERROR: quantization failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not _file_nonempty(args.output):
        print(f"[quantize_vitisai] ERROR: expected output {args.output} was not created", flush=True)
        sys.exit(1)

    print(f"[quantize_vitisai] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
