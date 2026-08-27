# ONNX Model Files

*Applies to KataGo 1.18.0 and later. Earlier versions have neither the `dumponnx` command nor the ability to load `.onnx` model files.*

The TensorRT and ONNX Runtime backends do not evaluate KataGo's `.bin.gz` model files directly. They translate the model into an [ONNX](https://onnx.ai/) graph in memory and hand that to TensorRT's `nvonnxparser` or to ONNX Runtime. This document covers:

* [Dumping the graph](#dumping-the-graph) that a backend builds, with `katago dumponnx`.
* [Running a `.onnx` file](#running-a-onnx-file) as a model.
* [The model file format](#the-model-file-format), for making a model that KataGo can run from other tooling.
* [Keeping layers out of FP16](#keeping-layers-out-of-fp16) on TensorRT.
* [Checking your model](#checking-your-model).
* [Versioning](#versioning), for extending the format.

Only the TensorRT and ONNX backends can do any of this. The CUDA, OpenCL, Eigen and Metal backends read `.bin.gz` models only.

## Dumping the graph

```
katago dumponnx -model model.bin.gz -out model.onnx
```

Options (the defaults match an ordinary 19x19 run):

| Option | Meaning |
|---|---|
| `-nn-x-len N`, `-nn-y-len N` | Board buffer size the graph is built for. Default 19. Must match the buffer size at run time, normally set by `maxBoardSizeForNNBuffer`. |
| `-require-exact-nnlen` | Build a graph with no board masking, matching `requireMaxBoardSize = true`. Such a graph is only correct for positions that fill the whole buffer. |
| `-transformer-nhwc true\|false` | Run the transformer trunk channel-last, matching `trtTransformerNHWC` / `onnxTransformerNHWC`. Default true. Ignored for models with no transformer blocks. |
| `-skip-scale8` | Skip the 1/8 activation rescaling that keeps convnet activations inside the FP16 range, matching `onnxSkipScale8 = true`. |

The output is byte-for-byte what the backend builds in memory with the same settings, so it is also a way to inspect exactly what KataGo feeds to TensorRT or ONNX Runtime.

## Running a .onnx file

Pass the file to `-model` in place of the `.bin.gz`, on the TensorRT or ONNX backend:

```
katago gtp -model model.onnx -config configs/gtp_example.cfg
```

`.onnx.gz` works too. Loading a graph directly skips the graph-building step at startup, and lets external ONNX tooling sit in between.

Four things are fixed when the graph is built and cannot be changed afterwards: the board buffer size, whether the graph does board masking, the transformer trunk layout, and the scale8 rescaling. The config options for them have no effect on an already-built graph, and KataGo logs a warning if you set one. Board size and masking mode are checked against what the run needs, and a mismatch is an error rather than a wrong evaluation. A masked graph is fine to use in a run where every position happens to fill the buffer, just slightly slower than a graph built for exactly that size.

## The model file format

An ONNX graph says nothing about what KataGo's inputs and outputs mean, which model version's feature encoding to use, or how to turn the outputs into a winrate and a score. Those parameters travel in the ModelProto's `metadata_props` under `katago.` keys. A file without them is refused, since nothing else can supply them.

Any ONNX model that satisfies both halves of the contract below will load and run, whatever produced it. KataGo checks the contract, not the mathematics, so also see [Checking your model](#checking-your-model).

KataGo does not look at the opset or IR version; whether the graph can be parsed is up to TensorRT or ONNX Runtime. For reference, KataGo's own emitter writes IR version 9 and opset 20, uses only standard ONNX operators, and bakes the weights in as initializers.

Below, `X` and `Y` are the board buffer width and height declared in the metadata. They need not be equal, and 19x19 is only the common case. Positions smaller than the buffer are handled by masking, not by a second graph.

### Graph inputs and outputs

All tensors are float32 and NCHW, with a dynamic (symbolic) batch dimension and fixed C, H and W. Graph inputs or outputs beyond these are rejected, since KataGo has nothing to bind to them.

| Input | Shape | Contents |
|---|---|---|
| `InputSpatial` | `[N, numInputChannels, Y, X]` | Per-point features for the model version, as computed by `NNInputs::fillRowV*` in `cpp/neuralnet/nninputs.cpp`. |
| `InputGlobal` | `[N, numInputGlobalChannels, 1, 1]` | Per-position global features, from the same code. |
| `InputMeta` | `[N, numInputMetaChannels, 1, 1]` | SGF metadata features. Present if and only if `metaEncoderVersion > 0`. |
| `InputMask` | `[N, 1, Y, X]` | 1 on-board, 0 off-board. Equal to channel 0 of `InputSpatial`. Must be declared even by a graph that ignores it, since KataGo always binds a buffer to it. |

| Output | Shape | Contents |
|---|---|---|
| `OutputPolicyPass` | `[N, numPolicyChannels, 1, 1]` | Policy logit for the pass move. |
| `OutputPolicy` | `[N, numPolicyChannels, Y, X]` | Policy logits per point. |
| `OutputValue` | `[N, 3, 1, 1]` | Win, loss and no-result logits. |
| `OutputScoreValue` | `[N, numScoreValueChannels, 1, 1]` | Score-related outputs, listed below. |
| `OutputOwnership` | `[N, 1, Y, X]` | Per-point ownership, pre-tanh. |

Declare the inputs in the order `InputSpatial`, `InputGlobal`, `InputMeta` (if present), `InputMask`. This works around a bug in ONNX Runtime's OpenVINO execution provider, present since ORT 1.23.0: the provider builds its name-to-index map skipping graph inputs that no node consumes, but does not adjust the indices it binds tensors by, so every input after such a one gets the wrong buffer. It fails at the first evaluation with

```
can't handle input tensor with name: parameter:InputSpatial, because model input
(shape=[?,22,19,19]) and tensor (shape=[1,1,19,19]) are incompatible
```

The general rule is that an input no node consumes must be declared last. `InputMask` is exactly that in a graph built with `requireExactNNLen`, which does no masking and so never reads it. TensorRT and the other execution providers bind by name and do not care about the order.

Every output is raw: no softmax, tanh or softplus, and no masking of off-board points. KataGo applies all of that itself, along with `outputScaleMultiplier`. Note also:

* Policy channel 0 is the base policy. Channel 1, where present, is the optimism policy, which KataGo blends with the base per query. Models at version 16 and up may have 4 channels.
* `OutputScoreValue` channels, for model version 9 and up, are score mean (pre-scaling), score stdev (pre-softplus), lead (pre-scaling), variance time left (pre-softplus), short-term winloss error (pre-softplus), short-term score error (pre-softplus). Older versions have fewer channels, and the count is checked against the model version at load.
* The graph never sees a symmetry. KataGo transforms the inputs and inverse-transforms the outputs around it, so the graph is a plain function of the board as given.

`NeuralNet::getOutput` in `cpp/neuralnet/onnxbackend.cpp` and the post-processing in `cpp/neuralnet/nneval.cpp` define exactly how each output is consumed.

### Metadata keys

Values are strings, as ONNX metadata always is. Booleans are `true` or `false`.

There are two namespaces:

* **`katago.`** is must-understand. KataGo refuses to load a file carrying a key here that it does not recognize, since such a key is one whose instructions it would be ignoring. Everything that changes how a position is evaluated, an output decoded, or a game adjudicated lives here.
* **`katago.info.`** is safe to ignore. Unknown keys here are skipped, so a file written by a newer KataGo still loads on an older one. Nothing here affects evaluation.

Keys outside `katago.` are ignored entirely and are yours to use, as are ONNX's own conventional `metadata_props`.

Required always:

| Key | Meaning |
|---|---|
| `katago.metadataVersion` | Version of this contract. See [Versioning](#versioning). Currently `1`. |
| `katago.name` | Model name. 1 to 96 characters of `[A-Za-z0-9_-]`, since it is used in cache filenames. |
| `katago.modelVersion` | KataGo model version, which fixes the input encoding and the output decoding. KataGo 1.18.0 accepts 3 through 17. |
| `katago.numInputChannels`, `katago.numInputGlobalChannels`, `katago.numInputMetaChannels` | Input channel counts. Each has exactly one legal value for the model version, and is checked. |
| `katago.numPolicyChannels`, `katago.numValueChannels`, `katago.numScoreValueChannels`, `katago.numOwnershipChannels` | Output channel counts, likewise checked. |
| `katago.build.nnXLen`, `katago.build.nnYLen` | Board buffer size the graph was built for. |
| `katago.build.requireExactNNLen` | `true` if the graph omits board masking, and is therefore only valid for positions that fill the buffer. |

Required for model version 15 and up, where the `.bin.gz` header also carries them, and optional below that, defaulting to `0` and `false`:

| Key | Meaning |
|---|---|
| `katago.metaEncoderVersion` | 0 for a normal model, 1 for a human-style-play model with an `InputMeta` input. |
| `katago.preferPassAliveUnderSuicideRules` | Whether the model expects pass-alive input features computed as if multi-stone suicide were legal. |
| `katago.preferExcludeTerritoryAdjacentToAtari` | Whether the model expects territory scoring with no seki tax to exclude empty points adjacent to a chain in atari, per rules version 3. Affects both its territory input features and how its games are adjudicated. |

Required for model version 13 and up, again matching the `.bin.gz` header, and optional below that, defaulting to 20, 20, 20, 20, 40, 0.25 and 30 respectively:

`katago.postProcess.tdScoreMultiplier`, `katago.postProcess.scoreMeanMultiplier`, `katago.postProcess.scoreStdevMultiplier`, `katago.postProcess.leadMultiplier`, `katago.postProcess.varianceTimeMultiplier`, `katago.postProcess.shorttermValueErrorMultiplier`, `katago.postProcess.shorttermScoreErrorMultiplier`

Optional:

| Key | Default | Meaning |
|---|---|---|
| `katago.postProcess.outputScaleMultiplier` | `1` | Every raw output is multiplied by this before decoding. Only needed by a graph whose activations are deliberately scaled. |
| `katago.fp32Nodes.trunkTipAndHead`, `katago.fp32Nodes.rmsNorm` | empty | Newline-separated node names that TensorRT keeps in FP32. See [Keeping layers out of FP16](#keeping-layers-out-of-fp16). |
| `katago.build.scale8Applied` | `false` | Whether the 1/8 activation rescaling was applied when the graph was built. The compensation for it lives in `outputScaleMultiplier`. |
| `katago.build.transformerNHWC` | `false` | Whether the trunk runs channel-last. TensorRT keys its timing and plan caches on this, so the two layouts do not share cache entries. |
| `katago.info.arch.trunkSpatialConvDepth`, `katago.info.arch.numParameters`, `katago.info.arch.hasAnyTransformerBlocks`, `katago.info.arch.hasAnyNestedBottleneckBlocks` | `0`, `false` | Used only for log lines and for test tolerances. Worth setting: `runnnevalcanarytests` picks its tolerances by model size and applies its most permissive ones to a model that declares no depth. |
| `katago.info.sourceSha256` | empty | The sha256 of the `.bin.gz` the graph was built from, if any. |

### Example

The metadata for a model version 15 net built for a 19x19 buffer, with masking, and with no output scaling or FP32 pinning:

```
katago.metadataVersion              = 1
katago.name                         = my-model
katago.modelVersion                 = 15
katago.numInputChannels             = 22
katago.numInputGlobalChannels       = 19
katago.numInputMetaChannels         = 0
katago.numPolicyChannels            = 2
katago.numValueChannels             = 3
katago.numScoreValueChannels        = 6
katago.numOwnershipChannels         = 1
katago.metaEncoderVersion           = 0
katago.preferPassAliveUnderSuicideRules = false
katago.preferExcludeTerritoryAdjacentToAtari = false
katago.postProcess.tdScoreMultiplier             = 20
katago.postProcess.scoreMeanMultiplier           = 20
katago.postProcess.scoreStdevMultiplier          = 20
katago.postProcess.leadMultiplier                = 20
katago.postProcess.varianceTimeMultiplier        = 40
katago.postProcess.shorttermValueErrorMultiplier = 0.25
katago.postProcess.shorttermScoreErrorMultiplier = 150
katago.build.nnXLen                 = 19
katago.build.nnYLen                 = 19
katago.build.requireExactNNLen      = false
```

The channel counts are the only legal ones for model version 15. The `postProcess` multipliers are not fixed by the version; they are properties of how the net was trained, so take them from the model you are converting rather than from this example.

Adding metadata to an existing graph with the `onnx` Python package:

```python
import onnx

props = {"katago.metadataVersion": "1", "katago.name": "my-model", ...}

model = onnx.load("model.onnx")
for key, value in props.items():
    entry = model.metadata_props.add()
    entry.key, entry.value = key, value
onnx.save(model, "model.onnx")
```

For a complete working example, dump any KataGo model and read its metadata:

```
katago dumponnx -model model.bin.gz -out model.onnx
python -c "import onnx; print(onnx.load('model.onnx').metadata_props)"
```

## Checking your model

KataGo validates the file's structure, not its behavior. A graph that satisfies the contract but computes the wrong thing will load and then play badly, so check a new model against something:

* `katago runnnevalcanarytests -model model.onnx -config configs/gtp_example.cfg` evaluates a handful of known positions and asserts that the top policy move, winrate, score and lead are sane. This quickly catches gross errors such as misrouted inputs, a wrong output channel order, or a wrong `outputScaleMultiplier`. It evaluates a rectangular board alongside the 19x19 ones, so it needs a masked graph.
* `katago testgpuerror -model model.onnx -config configs/gtp_example.cfg` reports how far the backend's outputs drift from its own FP32 outputs, which is the quickest way to see whether a graph is numerically healthy in FP16. Adding `-reference-file ref.bin` compares against saved outputs instead, which is how to check a converted or quantized graph against the model it came from. The reference file can only be written by an Eigen build, by running the same command there with the original model.
* If you converted from a `.bin.gz`, the strongest check is to run the same command against both files and diff the output. They should evaluate identically.

## Keeping layers out of FP16

`katago.fp32Nodes.trunkTipAndHead` and `katago.fp32Nodes.rmsNorm` are newline-separated lists of node names. When TensorRT builds an FP16 engine, it matches them against the network's layer names and pins each match to FP32, as a hard constraint so that TensorRT cannot fuse an FP16 path back in. It logs how many layers it pinned.

KataGo uses them for two separate reasons.

**Accuracy, cheaply.** `trunkTipAndHead` covers the whole region from the trunk tip through the policy and value heads. That region is a small fraction of the total compute but holds the normalizations and small final projections where precision matters most. On the two nets tested, pinning it cut the average winrate error against the same graph's FP32 outputs by 5 to 15 percent and left policy error unchanged. The same trade is usually worth making in a graph of your own.

**Avoiding overflow.** `rmsNorm` covers only the square, reduce and square-root steps of each RMSNorm; the division, scale and mask that follow are elementwise and FP16-safe. A sum of squares over C, H and W at 19x19 reaches about 138000, well past the FP16 maximum of 65504, and overflows to infinity. KataGo hit this in the trunk-tip normalization, and the result was confident but completely wrong evaluations.

Pinning has a limit worth knowing: a per-layer FP32 constraint sets a layer's input and output types, not the internal accumulator of a kernel that TensorRT fuses. In the overflow above the reduction was pinned successfully and still overflowed, because TensorRT fused the square into it and accumulated in FP16. The fix was arithmetic: KataGo now computes every such reduction as a mean rather than a sum, so the value stays small whatever precision it is computed in. If your graph has large FP16 reductions, make the arithmetic safe and treat pinning as a second line of defense.

Both keys are optional:

* Names that match nothing in the built network are an error under FP16. This means the graph was rewritten after the metadata was written, and quietly losing the protection is worse than refusing to run.
* Omitting them is allowed, since your graph may have no such hazard, but TensorRT warns when it builds an FP16 engine for a graph that declares none.

The ONNX Runtime backend ignores these keys. It has no per-node precision control, and precision is up to the execution provider.

## Versioning

`katago.metadataVersion` is the version of the contract on this page. A KataGo build accepts a range of versions:

* A file newer than the build understands is refused, since its keys may not mean what the build thinks they mean.
* A file older than the build's minimum is refused, and can be regenerated with `dumponnx`. KataGo raises that minimum only when it can no longer honor the old semantics, so old files normally keep working.

When extending the format, first decide which namespace the new key belongs in:

* A key that changes evaluation, decoding or adjudication goes under `katago.`, and it should be **required** rather than optional with a default. Silently defaulting a semantic flag off is the failure the must-understand namespace exists to prevent, and a default cannot be inferred from the graph. Adding one bumps the version, as does changing the meaning or units of an existing key, or changing the graph input/output contract.
* A key that only feeds log lines or diagnostics goes under `katago.info.`, with a documented default. Adding one does not bump the version: old readers skip it, and new readers fall back to the default for files that lack it.

Refusing unknown `katago.` keys is the backstop, not the mechanism. Bump the version when the rule says to, so that an older build can report which version it needed instead of reporting a key it has never heard of.

The key list and version constants live in `cpp/neuralnet/onnxmodelbuilder.cpp`, next to a comment pointing back at this document. Changes to one belong with changes to the other.
