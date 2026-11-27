#include "../neuralnet/distprotocol.h"

#include "../core/global.h"

#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace std;

#ifdef _WIN32
typedef SOCKET DistSocket;
static const DistSocket DIST_INVALID_SOCKET = INVALID_SOCKET;
static const int DIST_SOCKET_ERROR = SOCKET_ERROR;
static std::string distLastError() {
  int err = WSAGetLastError();
  char* msgBuf = NULL;
  DWORD flags = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT);
  DWORD len = FormatMessageA(
    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL, err, flags, (LPSTR)&msgBuf, 0, NULL);
  std::string msg = std::string("winsock error ") + Global::intToString(err);
  if(len > 0 && msgBuf != NULL) {
    msg += " (";
    msg.append(msgBuf, msgBuf + strnlen(msgBuf, len));
    msg += ")";
    LocalFree(msgBuf);
  }
  return msg;
}
#else
typedef int DistSocket;
static const DistSocket DIST_INVALID_SOCKET = -1;
static const int DIST_SOCKET_ERROR = -1;
static std::string distLastError() {
  return std::string(strerror(errno));
}
#endif

void DistProto::initNetworking() {
#ifdef _WIN32
  WSADATA wsaData;
  int ret = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if(ret != 0)
    throw StringError("DistProto: WSAStartup failed with error " + Global::intToString(ret));
#endif
}

void DistProto::cleanupNetworking() {
#ifdef _WIN32
  WSACleanup();
#endif
}

namespace {

  void setSocketTimeouts(DistSocket sock, int timeoutSeconds) {
#ifdef _WIN32
    DWORD timeoutMs = timeoutSeconds > 0 ? (DWORD)timeoutSeconds * 1000 : 0;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeoutMs, sizeof(timeoutMs));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeoutMs, sizeof(timeoutMs));
#else
    struct timeval tv;
    tv.tv_sec = timeoutSeconds > 0 ? timeoutSeconds : 0;
    tv.tv_usec = 0;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
  }

  void setNoDelay(DistSocket sock) {
    int one = 1;
    ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
  }

  DistSocket connectSocket(const string& host, int port, int timeoutSeconds) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    string portStr = Global::intToString(port);
    struct addrinfo* result = NULL;
    int gaiRet = ::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &result);
    if(gaiRet != 0)
      throw StringError("DistProto: could not resolve host '" + host + "': " + gai_strerror(gaiRet));

    DistSocket sock = DIST_INVALID_SOCKET;
    std::string lastErr = "no address candidates";
    for(struct addrinfo* rp = result; rp != NULL; rp = rp->ai_next) {
      sock = (DistSocket)::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if(sock == DIST_INVALID_SOCKET) {
        lastErr = distLastError();
        continue;
      }
      if(::connect(sock, rp->ai_addr, (socklen_t)rp->ai_addrlen) == 0)
        break;
      lastErr = distLastError();
#ifdef _WIN32
      closesocket(sock);
#else
      close(sock);
#endif
      sock = DIST_INVALID_SOCKET;
    }
    ::freeaddrinfo(result);

    if(sock == DIST_INVALID_SOCKET)
      throw StringError("DistProto: could not connect to " + host + ":" + portStr + ": " + lastErr);

    setNoDelay(sock);
    setSocketTimeouts(sock, timeoutSeconds);
    return sock;
  }

  DistSocket listenSocket(const string& host, int port) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    string portStr = Global::intToString(port);
    struct addrinfo* result = NULL;
    int gaiRet = ::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &result);
    if(gaiRet != 0)
      throw StringError("DistProto: could not resolve bind host '" + host + "': " + gai_strerror(gaiRet));

    DistSocket sock = DIST_INVALID_SOCKET;
    for(struct addrinfo* rp = result; rp != NULL; rp = rp->ai_next) {
      sock = (DistSocket)::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if(sock == DIST_INVALID_SOCKET)
        continue;
      int one = 1;
      ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
      if(::bind(sock, rp->ai_addr, (socklen_t)rp->ai_addrlen) == 0 && ::listen(sock, SOMAXCONN) == 0)
        break;
#ifdef _WIN32
      closesocket(sock);
#else
      close(sock);
#endif
      sock = DIST_INVALID_SOCKET;
    }
    ::freeaddrinfo(result);

    if(sock == DIST_INVALID_SOCKET)
      throw StringError("DistProto: could not bind/listen on " + host + ":" + portStr + ": " + distLastError());

    return sock;
  }

  struct RawConnection final : public DistProto::Connection {
    DistSocket sock;
    explicit RawConnection(DistSocket s) : DistProto::Connection((int)s), sock(s) {}
    ~RawConnection() {
#ifdef _WIN32
      closesocket(sock);
#else
      close(sock);
#endif
    }
  };

}

DistProto::Connection::Connection(int fd) : sock(fd) {}

DistProto::Connection::~Connection() {}

void DistProto::Connection::writeBytes(const void* data, size_t numBytes) {
  const char* p = (const char*)data;
  size_t remaining = numBytes;
  while(remaining > 0) {
#ifdef _WIN32
    int chunk = remaining > (size_t)INT_MAX ? INT_MAX : (int)remaining;
    int sent = ::send((DistSocket)sock, p, chunk, 0);
#else
    ssize_t sent = ::send((DistSocket)sock, p, remaining, MSG_NOSIGNAL);
#endif
    if(sent == DIST_SOCKET_ERROR)
      throw StringError("DistProto: socket send failed: " + distLastError());
    if(sent == 0)
      throw StringError("DistProto: socket send returned 0");
    p += sent;
    remaining -= (size_t)sent;
  }
}

void DistProto::Connection::readBytes(void* data, size_t numBytes) {
  char* p = (char*)data;
  size_t remaining = numBytes;
  while(remaining > 0) {
#ifdef _WIN32
    int chunk = remaining > (size_t)INT_MAX ? INT_MAX : (int)remaining;
    int recvd = ::recv((DistSocket)sock, p, chunk, 0);
#else
    ssize_t recvd = ::recv((DistSocket)sock, p, remaining, 0);
#endif
    if(recvd == 0)
      throw StringError("DistProto: connection closed by peer");
    if(recvd == DIST_SOCKET_ERROR)
      throw StringError("DistProto: socket recv failed: " + distLastError());
    p += recvd;
    remaining -= (size_t)recvd;
  }
}

void DistProto::Connection::writeInt32(int32_t v) {
  writeBytes(&v, sizeof(v));
}

int32_t DistProto::Connection::readInt32() {
  int32_t v;
  readBytes(&v, sizeof(v));
  return v;
}

void DistProto::Connection::writeFloat(float f) {
  writeBytes(&f, sizeof(f));
}

float DistProto::Connection::readFloat() {
  float f;
  readBytes(&f, sizeof(f));
  return f;
}

void DistProto::Connection::writeFloats(const float* data, size_t n) {
  writeBytes(data, n * sizeof(float));
}

void DistProto::Connection::readFloats(float* data, size_t n) {
  readBytes(data, n * sizeof(float));
}

DistProto::Listener::Listener(const string& host, int port) : listenSock((int)listenSocket(host, port)) {}

DistProto::Listener::~Listener() {
#ifdef _WIN32
  closesocket((DistSocket)listenSock);
#else
  close((DistSocket)listenSock);
#endif
}

int DistProto::Listener::acceptRaw() {
  DistSocket s = ::accept((DistSocket)listenSock, NULL, NULL);
  if(s == DIST_INVALID_SOCKET)
    throw StringError("DistProto: accept failed: " + distLastError());
  setNoDelay(s);
  return (int)s;
}

void DistProto::closeRawSocket(int sock_) {
#ifdef _WIN32
  closesocket((DistSocket)sock_);
#else
  close((DistSocket)sock_);
#endif
}

std::unique_ptr<DistProto::Connection> DistProto::connect(const string& host, int port, int timeoutSeconds) {
  return std::make_unique<RawConnection>(connectSocket(host, port, timeoutSeconds));
}

std::unique_ptr<DistProto::Connection> DistProto::wrapRawSocket(int sock_, int timeoutSeconds) {
  setSocketTimeouts((DistSocket)sock_, timeoutSeconds);
  return std::make_unique<RawConnection>((DistSocket)sock_);
}

void DistProto::clientHandshake(
  Connection& conn,
  const string& modelSha256,
  int nnXLen,
  int nnYLen,
  int maxBatchSize,
  bool inputsUseNHWC,
  float& serverOutputScaleOut
) {
  conn.writeBytes(MAGIC, sizeof(MAGIC));
  conn.writeInt32(PROTOCOL_VERSION);
  char shaBuf[SHA256_HEX_LEN];
  memset(shaBuf, 0, sizeof(shaBuf));
  size_t copyLen = modelSha256.size() < (size_t)SHA256_HEX_LEN ? modelSha256.size() : (size_t)SHA256_HEX_LEN;
  memcpy(shaBuf, modelSha256.data(), copyLen);
  conn.writeBytes(shaBuf, sizeof(shaBuf));
  conn.writeInt32(nnXLen);
  conn.writeInt32(nnYLen);
  conn.writeInt32(maxBatchSize);
  conn.writeInt32(inputsUseNHWC ? 1 : 0);

  int32_t status = conn.readInt32();
  if(status != 0) {
    int32_t msgLen = conn.readInt32();
    string msg;
    if(msgLen > 0) {
      msg.resize((size_t)msgLen);
      conn.readBytes(&msg[0], (size_t)msgLen);
    }
    throw StringError("DistProto: server rejected handshake: " + msg);
  }
  serverOutputScaleOut = conn.readFloat();
  if(!(serverOutputScaleOut > 0.0f) || !std::isfinite(serverOutputScaleOut))
    throw StringError("DistProto: server reported an invalid output scale");
}

void DistProto::serverHandshake(
  Connection& conn,
  const string& expectedSha256,
  int nnXLen,
  int nnYLen,
  int serverMaxBatchSize,
  float serverOutputScale,
  int& clientMaxBatchSizeOut,
  bool& clientUseNHWCOut
) {
  char magic[sizeof(MAGIC)];
  conn.readBytes(magic, sizeof(magic));
  if(memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
    sendError(conn, "bad magic: not a KataGo distributed inference client");
    throw StringError("DistProto: bad magic from client");
  }
  int32_t version = conn.readInt32();
  if(version != PROTOCOL_VERSION) {
    sendError(conn, "unsupported protocol version " + Global::intToString(version));
    throw StringError("DistProto: unsupported protocol version");
  }
  char shaBuf[SHA256_HEX_LEN + 1];
  memset(shaBuf, 0, sizeof(shaBuf));
  conn.readBytes(shaBuf, SHA256_HEX_LEN);
  string clientSha(shaBuf); // stops at the first NUL written by memset
  if(clientSha != expectedSha256) {
    sendError(conn, "model sha256 mismatch: server has " + expectedSha256 + ", client requested " + clientSha);
    throw StringError("DistProto: model sha256 mismatch");
  }
  int32_t clientX = conn.readInt32();
  int32_t clientY = conn.readInt32();
  if(clientX != nnXLen || clientY != nnYLen) {
    sendError(conn, "board size mismatch: server serves " + Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen));
    throw StringError("DistProto: board size mismatch");
  }
  clientMaxBatchSizeOut = conn.readInt32();
  if(clientMaxBatchSizeOut <= 0 || clientMaxBatchSizeOut > 65536) {
    sendError(conn, "invalid client maxBatchSize");
    throw StringError("DistProto: invalid client maxBatchSize");
  }
  if(clientMaxBatchSizeOut > serverMaxBatchSize) {
    sendError(
      conn,
      "client maxBatchSize " + Global::intToString(clientMaxBatchSizeOut) +
      " exceeds this server's -max-batch " + Global::intToString(serverMaxBatchSize) +
      "; raise the server's -max-batch to at least the client's nnMaxBatchSize, or lower nnMaxBatchSize"
    );
    throw StringError("DistProto: client maxBatchSize exceeds server max batch");
  }
  clientUseNHWCOut = conn.readInt32() != 0;

  conn.writeInt32(0); // status ok
  conn.writeFloat(serverOutputScale);
}

void DistProto::sendError(Connection& conn, const string& message) {
  try {
    conn.writeInt32(1);
    conn.writeInt32((int32_t)message.size());
    if(!message.empty())
      conn.writeBytes(message.data(), message.size());
  }
  catch(...) {
    // Best effort only.
  }
}

void DistProto::sendEvalRequest(Connection& conn, int batchSize, const ModelDims& dims, const std::vector<RowInput>& rows) {
  conn.writeInt32(batchSize);
  for(int i = 0; i < batchSize; i++) {
    const RowInput& row = rows[i];
    conn.writeInt32(row.symmetry);
    conn.writeFloat(row.policyOptimism);
    conn.writeInt32(row.includeOwnerMap ? 1 : 0);
    conn.writeFloats(row.spatial, dims.spatialElts());
    conn.writeFloats(row.global_, (size_t)dims.numInputGlobalChannels);
    if(dims.numInputMetaChannels > 0)
      conn.writeFloats(row.meta, (size_t)dims.numInputMetaChannels);
  }
}

void DistProto::recvEvalResponseHeader(Connection& conn) {
  int32_t status = conn.readInt32();
  if(status != 0) {
    int32_t msgLen = conn.readInt32();
    string msg;
    if(msgLen > 0) {
      msg.resize((size_t)msgLen);
      conn.readBytes(&msg[0], (size_t)msgLen);
    }
    throw StringError("DistProto: server eval error: " + msg);
  }
}

void DistProto::recvEvalRow(Connection& conn, const ModelDims& dims, const RowOutput& row) {
  conn.readFloats(row.policyProbs, dims.policyProbsElts());
  conn.readFloats(row.value, 3);
  conn.readFloats(row.scoreValue, dims.scoreValueElts());
  if(row.ownership != NULL)
    conn.readFloats(row.ownership, dims.ownerElts());
}

void DistProto::sendEvalRow(Connection& conn, const ModelDims& dims, const RowOutput& row) {
  conn.writeFloats(row.policyProbs, dims.policyProbsElts());
  conn.writeFloats(row.value, 3);
  conn.writeFloats(row.scoreValue, dims.scoreValueElts());
  if(row.ownership != NULL)
    conn.writeFloats(row.ownership, dims.ownerElts());
}

void DistProto::transposeTensor(const float* src, float* dst, int cSize, int xySize, bool srcLayoutIsNHWC) {
  if(srcLayoutIsNHWC) {
    // src: [xy][c] -> dst: [c][xy]
    for(int c = 0; c < cSize; c++) {
      for(int i = 0; i < xySize; i++) {
        dst[(size_t)c * xySize + i] = src[(size_t)i * cSize + c];
      }
    }
  }
  else {
    // src: [c][xy] -> dst: [xy][c]
    for(int c = 0; c < cSize; c++) {
      for(int i = 0; i < xySize; i++) {
        dst[(size_t)i * cSize + c] = src[(size_t)c * xySize + i];
      }
    }
  }
}

void DistProto::parseAddress(const string& addr, string& hostOut, int& portOut) {
  size_t colon = addr.rfind(':');
  if(colon == string::npos || colon == 0 || colon + 1 >= addr.size())
    throw StringError("DistProto: malformed server address (expected host:port): " + addr);
  hostOut = addr.substr(0, colon);
  string portStr = addr.substr(colon + 1);
  int port = Global::stringToInt(portStr);
  if(port <= 0 || port > 65535)
    throw StringError("DistProto: invalid port in server address: " + addr);
  portOut = port;
}
