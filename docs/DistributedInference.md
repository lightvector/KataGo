# Distributed inference: engine on one machine, GPUs on others

KataGo can run its search/analysis on a machine with no GPU and send every neural net
evaluation over TCP to one or more `katago distserver` processes, each of which loads the
model with a normal local backend (CUDA/TensorRT/OpenCL/Eigen/ONNX/ROCm) and executes it.
The design follows PhoenixGo's `dist_zero_model_server`: the engine keeps all search state,
the servers are stateless evaluators.

This is useful when

* the GPU machines are shared by several engines, or are in a different box from the
  machine running the GTP/analysis frontend,
* the frontend machine has no usable GPU at all, or
* one engine should use GPUs spread over several machines.

## Building

| role | build |
|---|---|
| server | any normal backend build, e.g. `cmake . -DUSE_BACKEND=CUDA`; the `distserver` subcommand is always compiled in |
| client | `cmake . -DUSE_BACKEND=DISTRIBUTED`; needs no GPU and no CUDA toolkit |

Both sides must be built from the same KataGo version. When a client connects, the server
checks the model sha256 and the board size (`nnXLen`/`nnYLen`) and refuses the connection on
a mismatch, so a mixed or misconfigured fleet fails at startup rather than producing wrong
evals.

## Server

```bash
katago distserver -model model.bin.gz -listen 0.0.0.0:17731 \
                  -gpus 0,1 -max-batch 256 -boardsize 19 -timeout 300
```

* `-gpus 0,1` — new connections are assigned round-robin to these device indices. Each
  connection is served by its own thread and gets its own compute handle created on that
  thread; that is what binds it to the device (`cudaSetDevice` on CUDA), so two connections
  on two GPUs really do run on two GPUs.
* `-max-batch` — the largest batch this server will accept. It must be at least the client's
  `nnMaxBatchSize`; a client that declares more is rejected at the handshake.
* `-boardsize` — the `nnXLen`/`nnYLen` this server serves. Clients configured for a larger
  board are rejected; one server binary can serve any board size up to this value.
* `-timeout` — per-connection socket timeout in seconds (`0` = none). With `0`, a client that
  disappears without closing its socket leaves a server thread blocked forever, so setting
  this is recommended in long-running deployments.

One server process per machine is enough. One process per GPU works too and makes per-GPU
logging and tuning simpler; in that case pass a single index (`-gpus 0`) or omit `-gpus`.

Each connection owns its own compute handle, which means its own copy of the model weights on
its GPU. Keep the number of connections per GPU small (1–2); use many search threads on the
client, not many connections, when you want more throughput.

The server currently takes no config file, so backend options that normally come from the
config (`cudaShareModelWeights`, `openclTunerFile`, `useFP16`, …) cannot be set on the server
side. `useFP16 = auto` behaviour of the backend still applies.

## Client

In the usual GTP/analysis config:

```ini
distServers = gpu1:17731,gpu2:17731
numNNServerThreadsPerModel = 4
distTimeoutSec = 600
nnMaxBatchSize = 32
```

* `distServers` — `host:port` list, IPv4/IPv6 or DNS names.
* `numNNServerThreadsPerModel` — number of connections, and therefore of in-flight batches.
  NN server thread `i` connects to server `i % numServers`. Batches are taken from a shared
  queue, so when one machine is slower it simply receives fewer batches.
* `nnMaxBatchSize` — per-connection batch cap. Total batch on a server is the sum over its
  connections, so with several connections per GPU prefer smaller values per connection.
* `distTimeoutSec` — per-request socket timeout, `0` disables it.

The client needs the model file, but only its header is read (shapes and sha256); the weights
are sent nowhere and are only used by the servers.

## Numerical equivalence

Some backends (CUDA, ROCm, OpenCL, TensorRT, ONNX) replace the loaded model with an
activation-scaled variant at context-creation time and compensate by multiplying the
postprocessing scale (`outputScaleMultiplier`) by 8. The client only parses the model
header, so it cannot see that transform. Since protocol v2 the server reports its effective
postprocessing scale in the handshake and the client rescales the received policy/value/
score tensors by `serverScale / clientScale`, which makes a remote evaluation reproduce a
local one bit-for-bit for both convnet and transformer models.

Because the wire format carries this scale, **client and server must be built from the same
KataGo version**; a protocol mismatch is rejected at the handshake.

## Failure behaviour

| event | effect |
|---|---|
| a server is unreachable at startup | the engine exits with `DistProto: could not connect to host:port` |
| a server dies while running | that connection's next eval throws; the engine aborts (`ERROR: NN server thread failed: …`). There is no reconnect or failover |
| a client dies | the server logs the close and keeps serving other connections |
| model / board size / batch cap mismatch | handshake is refused with a message naming the mismatch |

Treat the engine and its fleet as one unit: supervise/restart them together, and put the fleet
behind a load balancer only if you also accept that a dead backend breaks the engines using it.

## Performance

Per-batch cost is `GPU compute + RTT + serialization`. Measured on a loopback link with a
19×19 b12c192 model (V7 features, 22 input planes): **1–11 ms per batch**, with no systematic
growth over batch sizes 1–32, and the transport fraction falling below 1% once a batch takes
~1 s of compute.

Traffic per evaluation is

```
up:   4 * (numInputChannels * XY + numInputGlobalChannels + numInputMetaChannels) bytes
down: 4 * (XY + 1 + 3 + numScoreValueChannels + (XY if the owner map was requested)) bytes
```

i.e. ~32.6 KB up and ~1.5 KB down per eval at 19×19 with V7 features (22 planes). That makes
the network the hard ceiling at sustained high throughput: gigabit Ethernet tops out near
3.7k evals/s, 10 GbE near 37k, before protocol overhead. Small models on fast GPUs need
10 GbE or better; b18/b28-class models on a couple of GPUs are fine on 1 GbE.

Because RTT is paid **per batch**, batching is what hides it. On a 40 ms RTT link a batch-1
stream loses ~40% of its throughput; at batch 8 the same RTT costs only a few percent.

Keep the engine and the GPU machines on the same LAN if possible.

## Validation status and limitations

**Validated** on two rented single-GPU instances (NVIDIA RTX 4080 SUPER 32 GB, CUDA 13.0,
cuDNN 9), one `distserver` and one SakuraLink SSH tunnel per machine:

* from-scratch CUDA build on both machines: 2 m 28 s each (`-j48`);
* client configured with `distServers = 127.0.0.1:<tunnel A>,127.0.0.1:<tunnel B>`;
  both servers logged real `eval batch` work (48 / 37 batches); the remote evaluation is
  **bit-identical** to same-machine direct inference (whiteWin 0.440201, whiteLead 0.550);
* throughput, batch 8: GPU-direct 3127 evals/s, one node over the tunnel 56 evals/s,
  two nodes 62 (2 connections) / 113 (4 connections) evals/s.

Video (2 min, from clean build to benchmark): [media/kata-distributed-verify.mp4](media/kata-distributed-verify.mp4).

**Known limitations**

* Network-bound: one round trip per batch plus 32.6 KB per eval upstream. On a 24 ms RTT
  link with ~10 MB/s upload this caps a two-machine fleet at ~100–350 evals/s, while the
  GPUs themselves can do thousands. Same-host or 10 GbE deployment is required to use the
  GPUs fully.
* Protocol v2 is not backward compatible: client and server must be rebuilt together.
* The server has no config file, so backend options (`cudaShareModelWeights`,
  `openclTunerFile`, FP16 overrides, …) cannot be set on the server side.
* Each connection owns its own model copy on the GPU; many connections per GPU can
  exhaust VRAM.
* No reconnect or failover: a server dying aborts the engine.
* No authentication or TLS on the wire; keep the server bound to loopback behind SSH.
* A same-LAN pair and a multi-machine run with a realistic model are still missing; all
  cross-machine measurements were across the public internet.

## Security

There is no authentication and no encryption. Anyone who can reach a `distserver` port can
submit evaluations and read the results, i.e. consume the GPUs. Bind it to a private
interface, firewall the port, or run it over SSH/WireGuard.

### Single-host two-GPU validation (2 x RTX 4080 SUPER)

With a realistic model (`kata1-tf3-b11c768`, 70.4 M parameter transformer) on a single
machine holding both GPUs, the transport cost becomes small next to the forward pass:

| configuration | batch 8 (evals/s) | batch 32 (evals/s) |
|---|---|---|
| direct, 2 threads (1 per GPU) | 658 | 708 |
| direct, 4 threads (2 per GPU) | 742 | 733 |
| one `distserver -gpus 0,1`, 2 connections | 637 | 695 |
| one `distserver -gpus 0,1`, 4 connections | 734 | 730 |
| two `distserver`s (one GPU each), 2 connections | 642 | 694 |
| two `distserver`s (one GPU each), 4 connections | 737 | 729 |

Distributed reaches **98.8-99.6 % of same-machine direct throughput** at four
connections; the GPU utilisation is the same as for direct inference (~60-70 % at this
model and batch size). Per-batch transport is ~0.3 ms against a 43-174 ms forward pass.
The same run on the small b12c192 convnet (2.5-6 ms per batch) shows 84-91 % instead,
which is the per-batch serialisation cost showing through.

**Precision caveat:** the server has no config file, so it uses backend defaults (FP16 on
this GPU). A client configured with `useFP16 = false` still gets FP16 numbers back
(0.423235 vs 0.423887 win probability on this transformer). This needs a `-config` or
`-useFP16` option on the server, or the precision carried in the handshake.
