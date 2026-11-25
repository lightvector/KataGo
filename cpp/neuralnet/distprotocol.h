#ifndef NEURALNET_DISTPROTOCOL_H_
#define NEURALNET_DISTPROTOCOL_H_

// Wire protocol and minimal cross-platform TCP layer shared by the distributed
// inference client backend (neuralnet/distributedbackend.cpp, USE_BACKEND=DISTRIBUTED)
// and the distributed inference server (command/distserver.cpp).
//
// The architecture follows PhoenixGo's dist/ design: the engine's MCTS runs on a
// machine without any GPU and forwards neural net batches to one or more remote
// "katago distserver" processes, each of which loads the model with a normal local
// backend (CUDA/TensorRT/OpenCL/Eigen/...) and executes the evals.
//
// Protocol v2 ("KGD1"), all integers little-endian, all floats IEEE754 binary32.
// Tensor sizes are implied by the model (sha256-checked in the handshake), so
// requests and responses carry only raw tensor data, like PhoenixGo's ForwardReq
// which only carries the packed input features.
//
// All tensors on the wire are in "canonical" form: the same values that a local
// backend's NeuralNet::getOutput writes into NNOutput. The client applies no
// postprocessing at all - symmetry handling, policy optimism blending, and the
// NHWC/NCHW layout question are all resolved on the server, which owns the model.
//
// v2 adds the server's postprocessing scale to the handshake. Some backends
// (CUDA/ROCm/OpenCL/TensorRT/ONNX) replace the model with an activation-scaled
// variant at context-creation time and multiply postProcessParams.outputScaleMultiplier
// by 8 as part of that transform. The client parses the model header only, so it
// does not know about that transform; with v1 it postprocessed the received logits
// with the unscaled multiplier and produced wrong probabilities (a convnet's 0.4402
// became 0.4925). The server now reports its effective multiplier and the client
// rescales the received tensors before its own postprocessing.
//
//   handshake, client -> server:
//     char[4]   magic = "KGD1"
//     int32     protocolVersion = 2
//     char[64]  model sha256 (lowercase hex, zero padded)
//     int32     nnXLen
//     int32     nnYLen
//     int32     maxBatchSize the client may send
//     int32     inputsUseNHWC of the client (0 = NCHW, 1 = NHWC)
//   handshake, server -> client:
//     int32     status (0 = ok, nonzero = error)
//     if status != 0: int32 errorMessageLen, char[errorMessageLen] (connection then closes)
//     if status == 0: float outputScaleMultiplier the server's model desc ended up with
//
//   eval request, client -> server:
//     int32     batchSize (> 0)
//     per row:
//       int32   symmetry          (as in SymmetryHelpers, 0..7)
//       float   policyOptimism    (as in NNResultBuf)
//       int32   includeOwnerMap   (0/1)
//       float[numInputChannels * nnXLen * nnYLen]   spatial input, CLIENT's layout,
//                                                   canonical orientation
//       float[numInputGlobalChannels]               global input
//       float[numInputMetaChannels]                 meta input (0 channels -> 0 floats)
//   eval response, server -> client:
//     int32     status (0 = ok, nonzero = error)
//     if status != 0: int32 errorMessageLen, char[errorMessageLen] (connection then closes)
//     per row:
//       float[nnXLen * nnYLen + 1]   policyProbs (canonical orientation, includes pass)
//       float[3]                     whiteWinProb, whiteLossProb, whiteNoResultProb
//       float[numScoreValueChannels] score value outputs, in NNOutput field order
//       if includeOwnerMap: float[nnXLen * nnYLen]  whiteOwnerMap (canonical orientation)

#include "../core/global.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace DistProto {

  static const char MAGIC[4] = {'K','G','D','1'};
  static const int32_t PROTOCOL_VERSION = 2;
  static const int32_t SHA256_HEX_LEN = 64;

  // Shape of the tensors exchanged for a model + board size. Must be identical on
  // both ends; guaranteed by the model sha256 check in the handshake.
  struct ModelDims {
    int nnXLen;
    int nnYLen;
    int numInputChannels;
    int numInputGlobalChannels;
    int numInputMetaChannels;
    int numPolicyChannels;
    int numValueChannels;
    int numScoreValueChannels;
    int numOwnershipChannels; // 0 or 1

    size_t spatialElts() const { return (size_t)numInputChannels * nnXLen * nnYLen; }
    size_t policyProbsElts() const { return (size_t)nnXLen * nnYLen + 1; }
    size_t scoreValueElts() const { return (size_t)numScoreValueChannels; }
    size_t ownerElts() const { return (size_t)nnXLen * nnYLen; }
  };

  // Call once per process before any other networking (WSAStartup on Windows).
  void initNetworking();
  void cleanupNetworking();

  // RAII TCP connection wrapping an already-connected socket. Not thread-safe.
  // The public fd ctor exists so the server can wrap accepted sockets; callers should
  // prefer wrapRawSocket which also sets timeouts.
  class Connection {
   public:
    Connection(int fd);
    virtual ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    void writeBytes(const void* data, size_t numBytes);
    void readBytes(void* data, size_t numBytes); // throws StringError on EOF or error

    void writeInt32(int32_t v);
    int32_t readInt32();
    void writeFloat(float f);
    float readFloat();

    // Writes/reads a float buffer as raw little-endian binary32.
    void writeFloats(const float* data, size_t n);
    void readFloats(float* data, size_t n);

   private:
    int sock; // SOCKET on Windows, int elsewhere
  };

  // RAII TCP listener. Not thread-safe.
  class Listener {
   public:
    Listener(const std::string& host, int port);
    ~Listener();

    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;

    // Blocks until a client connects. Throws StringError on failure.
    // The returned value must be released with closeRawSocket.
    int acceptRaw();

   private:
    int listenSock;
  };

  void closeRawSocket(int sock);

  // Opens a new client connection to host:port.
  // timeoutSeconds <= 0 disables send/recv timeouts. Throws StringError on failure.
  std::unique_ptr<Connection> connect(const std::string& host, int port, int timeoutSeconds);

  // Takes ownership of an already-connected socket fd (from acceptRaw).
  // timeoutSeconds <= 0 disables send/recv timeouts.
  std::unique_ptr<Connection> wrapRawSocket(int sock, int timeoutSeconds);

  // Handshake as client. Throws StringError if the server rejects.
  // serverOutputScaleOut receives the server's effective
  // postProcessParams.outputScaleMultiplier (see the protocol comment above); the
  // caller must rescale received tensors by serverOutputScale / its own scale.
  void clientHandshake(
    Connection& conn,
    const std::string& modelSha256,
    int nnXLen,
    int nnYLen,
    int maxBatchSize,
    bool inputsUseNHWC,
    float& serverOutputScaleOut
  );

  // Handshake as server. Returns the client's maxBatchSize and inputsUseNHWC.
  // serverMaxBatchSize bounds what this server will accept; a client that declares a
  // larger maxBatchSize could send a batch this server has to refuse mid-stream, so the
  // handshake fails instead, before any eval is served.
  // serverOutputScale is the server's post-createComputeContext
  // postProcessParams.outputScaleMultiplier (the backends that apply scale8 bump it to 8).
  // Throws StringError after sending an error response if anything mismatches.
  void serverHandshake(
    Connection& conn,
    const std::string& expectedSha256,
    int nnXLen,
    int nnYLen,
    int serverMaxBatchSize,
    float serverOutputScale,
    int& clientMaxBatchSizeOut,
    bool& clientUseNHWCOut
  );

  // Sends an error response (used by the server before dropping the connection).
  void sendError(Connection& conn, const std::string& message);

  // Per-row input for sendEvalRequest.
  struct RowInput {
    int symmetry;          // SymmetryHelpers index, 0..7
    float policyOptimism;  // policy optimism to blend in
    bool includeOwnerMap;
    const float* spatial;  // spatialElts() floats, client's layout, canonical orientation
    const float* global_;  // numInputGlobalChannels floats
    const float* meta;     // numInputMetaChannels floats (may be null if 0 channels)
  };

  void sendEvalRequest(Connection& conn, int batchSize, const ModelDims& dims, const std::vector<RowInput>& rows);

  // Reads the response status. Throws StringError (with the server message) on error.
  void recvEvalResponseHeader(Connection& conn);

  // Per-row NNOutput destination for recvEvalRow. policyProbs must have room for
  // nnXLen*nnYLen+1 floats; ownership may be null if not requested.
  struct RowOutput {
    float* policyProbs; // policyProbsElts() floats
    float* value;       // 3 floats: whiteWinProb, whiteLossProb, whiteNoResultProb
    float* scoreValue;  // scoreValueElts() floats, in NNOutput field order
    float* ownership;   // ownerElts() floats, or null
  };

  void recvEvalRow(Connection& conn, const ModelDims& dims, const RowOutput& row);

  // Writes one row of NNOutput-ish results (server side).
  void sendEvalRow(Connection& conn, const ModelDims& dims, const RowOutput& row);

  // Transposes one C-by-XY tensor between NHWC [y][x][c] and NCHW [c][y][x].
  // srcLayoutIsNHWC describes the layout of src; dst is written in the opposite layout.
  void transposeTensor(const float* src, float* dst, int cSize, int xySize, bool srcLayoutIsNHWC);

  // "host:port" -> (host, port). Throws StringError on malformed input.
  void parseAddress(const std::string& addr, std::string& hostOut, int& portOut);

} // namespace DistProto

#endif // NEURALNET_DISTPROTOCOL_H_
