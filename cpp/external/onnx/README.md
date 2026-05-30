# ONNX protobuf schema

`onnx.proto` is the ONNX intermediate representation schema, vendored from the
[ONNX project](https://github.com/onnx/onnx). It is a verbatim copy of
`onnx/onnx-ml.proto` from ONNX version 1.22.0 (the `-ml` variant, to match the
schema that TensorRT's bundled ONNX parser is built against; the message and
field definitions used by KataGo are identical between the `-ml` and non-`-ml`
variants).

KataGo uses this schema only with the TensorRT backend: the backend emits an
`onnx::ModelProto` describing the network and hands the serialized bytes to
TensorRT's `nvonnxparser`, which builds the engine. The schema is compiled to
`onnx.pb.{h,cc}` by `protoc` at build time (see `cpp/CMakeLists.txt`).

Licensed under Apache-2.0; see `LICENSE`.
