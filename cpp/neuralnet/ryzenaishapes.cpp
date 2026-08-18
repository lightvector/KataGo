#include "../neuralnet/ryzenaishapes.h"

#include "../neuralnet/desc.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

using std::string;
using std::vector;

namespace RyzenAIShapes {

const char* rowKindName(RowKind kind) {
  switch(kind) {
    case RowKind::Spatial: return "spatial";
    case RowKind::Batch: return "batch";
    case RowKind::AttnScore: return "attn";
  }
  return "?";
}

namespace {

string blockLabel(const string& prefix, size_t idx) {
  std::ostringstream out;
  out << prefix << ".b" << idx;
  return out.str();
}

void addMatMul(vector<GemmUse>& uses, const string& path, const MatMulLayerDesc& d, RowKind rows) {
  // A zero-channel matmul means the layer is absent (e.g. linearGate without SwiGLU).
  if(d.inChannels <= 0 || d.outChannels <= 0)
    return;
  GemmUse u;
  u.path = path;
  u.op = "matmul";
  u.convY = 1;
  u.convX = 1;
  u.inChannels = d.inChannels;
  u.outChannels = d.outChannels;
  u.K = d.inChannels;
  u.N = d.outChannels;
  u.rows = rows;
  uses.push_back(u);
}

void addConv(vector<GemmUse>& uses, const string& path, const ConvLayerDesc& d) {
  if(d.inChannels <= 0 || d.outChannels <= 0)
    return;
  GemmUse u;
  u.path = path;
  std::ostringstream op;
  op << "conv" << d.convYSize << "x" << d.convXSize;
  u.op = op.str();
  u.convY = d.convYSize;
  u.convX = d.convXSize;
  u.inChannels = d.inChannels;
  u.outChannels = d.outChannels;
  // Implicit GEMM: one reduction over every tap of every input channel.
  u.K = d.convYSize * d.convXSize * d.inChannels;
  u.N = d.outChannels;
  u.rows = RowKind::Spatial;
  uses.push_back(u);
}

void addAttnScore(vector<GemmUse>& uses, const string& path, const string& op, int K, int N) {
  GemmUse u;
  u.path = path;
  u.op = op;
  u.convY = 1;
  u.convX = 1;
  u.inChannels = K;
  u.outChannels = N;
  u.K = K;
  u.N = N;
  u.rows = RowKind::AttnScore;
  uses.push_back(u);
}

void walkBlocks(
  vector<GemmUse>& uses,
  const std::vector<std::pair<int, unique_ptr_void>>& blocks,
  const string& prefix,
  int nnXY
) {
  for(size_t i = 0; i < blocks.size(); i++) {
    const int kind = blocks[i].first;
    const void* ptr = blocks[i].second.get();
    const string path = blockLabel(prefix, i);

    if(kind == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* d = (const ResidualBlockDesc*)ptr;
      addConv(uses, path + ".regularConv", d->regularConv);
      addConv(uses, path + ".finalConv", d->finalConv);
    }
    else if(kind == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* d = (const GlobalPoolingResidualBlockDesc*)ptr;
      addConv(uses, path + ".regularConv", d->regularConv);
      addConv(uses, path + ".gpoolConv", d->gpoolConv);
      addMatMul(uses, path + ".gpoolToBiasMul", d->gpoolToBiasMul, RowKind::Batch);
      addConv(uses, path + ".finalConv", d->finalConv);
    }
    else if(kind == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* d = (const NestedBottleneckResidualBlockDesc*)ptr;
      addConv(uses, path + ".preConv", d->preConv);
      walkBlocks(uses, d->blocks, path, nnXY);
      addConv(uses, path + ".postConv", d->postConv);
    }
    else if(kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      const TransformerAttentionDesc* d = (const TransformerAttentionDesc*)ptr;
      addMatMul(uses, path + ".attn.qProj", d->qProj, RowKind::Spatial);
      addMatMul(uses, path + ".attn.kProj", d->kProj, RowKind::Spatial);
      addMatMul(uses, path + ".attn.vProj", d->vProj, RowKind::Spatial);
      if(nnXY > 0) {
        // Per head: scores = Q[nnXY x qHeadDim] * K^T[qHeadDim x nnXY],
        //           ctx    = P[nnXY x nnXY]     * V[nnXY x vHeadDim].
        addAttnScore(uses, path + ".attn.qk", "attn.qk", d->qHeadDim, nnXY);
        addAttnScore(uses, path + ".attn.pv", "attn.pv", nnXY, d->vHeadDim);
      }
      addMatMul(uses, path + ".attn.outProj", d->outProj, RowKind::Spatial);
    }
    else if(kind == TRANSFORMER_FFN_BLOCK_KIND) {
      const TransformerFFNDesc* d = (const TransformerFFNDesc*)ptr;
      addMatMul(uses, path + ".ffn.linear1", d->linear1, RowKind::Spatial);
      if(d->useSwiGLU)
        addMatMul(uses, path + ".ffn.linearGate", d->linearGate, RowKind::Spatial);
      addMatMul(uses, path + ".ffn.linear2", d->linear2, RowKind::Spatial);
    }
    // Unknown kinds are skipped rather than fatal: this is a diagnostic, and the
    // real forward path in reference.cpp is what must reject them.
  }
}

int roundUpTo(int v, int q) {
  return ((v + q - 1) / q) * q;
}

// Right/left aligned fixed-width cells, so the tables line up without pulling in
// iostream manipulator state.
string padLeft(const string& s, size_t width) {
  return s.size() >= width ? s : string(width - s.size(), ' ') + s;
}
string padLeft(int v, size_t width) {
  return padLeft(std::to_string(v), width);
}
string padRight(const string& s, size_t width) {
  return s.size() >= width ? s : s + string(width - s.size(), ' ');
}

}  // namespace

vector<GemmUse> enumerate(const ModelDesc& desc, int nnXY) {
  vector<GemmUse> uses;

  addConv(uses, "trunk.initialConv", desc.trunk.initialConv);
  addMatMul(uses, "trunk.initialMatMul", desc.trunk.initialMatMul, RowKind::Batch);
  if(desc.metaEncoderVersion != 0) {
    const SGFMetadataEncoderDesc& m = desc.trunk.sgfMetadataEncoder;
    addMatMul(uses, "trunk.meta.mul1", m.mul1, RowKind::Batch);
    addMatMul(uses, "trunk.meta.mul2", m.mul2, RowKind::Batch);
    addMatMul(uses, "trunk.meta.mul3", m.mul3, RowKind::Batch);
  }
  walkBlocks(uses, desc.trunk.blocks, "trunk", nnXY);

  const PolicyHeadDesc& p = desc.policyHead;
  addConv(uses, "policy.p1Conv", p.p1Conv);
  addConv(uses, "policy.g1Conv", p.g1Conv);
  addMatMul(uses, "policy.gpoolToBiasMul", p.gpoolToBiasMul, RowKind::Batch);
  addConv(uses, "policy.p2Conv", p.p2Conv);
  addMatMul(uses, "policy.gpoolToPassMul", p.gpoolToPassMul, RowKind::Batch);
  addMatMul(uses, "policy.gpoolToPassMul2", p.gpoolToPassMul2, RowKind::Batch);

  const ValueHeadDesc& v = desc.valueHead;
  addConv(uses, "value.v1Conv", v.v1Conv);
  addMatMul(uses, "value.v2Mul", v.v2Mul, RowKind::Batch);
  addMatMul(uses, "value.v3Mul", v.v3Mul, RowKind::Batch);
  addMatMul(uses, "value.sv3Mul", v.sv3Mul, RowKind::Batch);
  addConv(uses, "value.vOwnershipConv", v.vOwnershipConv);

  return uses;
}

int chooseSingleK(const ModelDesc& desc, int nnXLen, int nnYLen, double maxSpread) {
  const vector<GemmUse> uses = enumerate(desc, nnXLen * nnYLen);

  // Only the spatial layers matter: the head's batch-row matmuls never reach
  // the NPU at all (too few rows to repay a dispatch). And among those, only
  // the ones carrying real arithmetic -- a policy head's 48-channel 1x1 would
  // otherwise drag the spread out by itself while contributing nothing.
  // Weight by arithmetic and aggregate per reduction dim before thresholding:
  // a single instance of even the busiest layer is a fraction of a percent, so
  // the test has to be on the K as a whole, not on one layer.
  std::map<int, double> weightByK;
  double total = 0.0;
  for(size_t i = 0; i < uses.size(); i++) {
    if(uses[i].rows != RowKind::Spatial)
      continue;
    const double macs = (double)uses[i].K * uses[i].N;
    weightByK[uses[i].K] += macs;
    total += macs;
  }
  if(total <= 0.0)
    return 0;

  int minK = 0;
  int maxK = 0;
  for(std::map<int, double>::const_iterator it = weightByK.begin(); it != weightByK.end(); ++it) {
    if(it->second / total < 0.01)
      continue;
    if(minK == 0)
      minK = it->first;
    maxK = it->first;
  }
  if(minK <= 0 || maxK <= 0)
    return 0;
  if((double)maxK / (double)minK > maxSpread)
    return 0;
  return maxK;
}

string report(const ModelDesc& desc, int nnXLen, int nnYLen) {
  const int nnXY = nnXLen * nnYLen;
  const vector<GemmUse> uses = enumerate(desc, nnXY);

  std::ostringstream out;
  out << "RyzenAI shape report for model '" << desc.name << "'"
      << " (version " << desc.modelVersion << ", board " << nnXLen << "x" << nnYLen << ")\n";
  out << "  trunk: " << desc.trunk.numBlocks << " blocks, "
      << desc.trunk.trunkNumChannels << " trunk channels, "
      << desc.trunk.midNumChannels << " mid channels\n";

  // ---- distinct (K,N,rows), with occurrence counts and one example -------------
  struct Agg {
    int count = 0;
    string example;
    string op;
  };
  std::map<std::pair<std::pair<int, int>, int>, Agg> distinct;
  for(size_t i = 0; i < uses.size(); i++) {
    const GemmUse& u = uses[i];
    auto key = std::make_pair(std::make_pair(u.K, u.N), (int)u.rows);
    Agg& a = distinct[key];
    a.count++;
    if(a.example.empty()) {
      a.example = u.path;
      a.op = u.op;
    }
  }

  out << "\n  distinct (K,N) shapes: " << distinct.size()
      << "   (total GEMM sites: " << uses.size() << ")\n";
  out << "         K       N     rows   uses  op          example\n";
  for(auto it = distinct.begin(); it != distinct.end(); ++it) {
    const int K = it->first.first.first;
    const int N = it->first.first.second;
    const RowKind rows = (RowKind)it->first.second;
    out << "  " << padLeft(K, 8) << padLeft(N, 8) << padLeft(rowKindName(rows), 9)
        << padLeft(it->second.count, 7) << "  " << padRight(it->second.op, 10)
        << "  " << it->second.example << "\n";
  }

  // ---- how big M actually gets ------------------------------------------------
  out << "\n  M per dispatch (rows are independent, so M is tiled, not baked into artifacts):\n";
  out << "    spatial: batch * " << nnXY << "\n";
  out << "    batch:   batch\n";
  out << "    attn:    batch * numHeads * " << nnXY << "\n";

  // ---- what a quantized artifact grid would cost ------------------------------
  // K, N and M can all be zero-padded up without changing the result (pad B's
  // rows and A's columns for K, drop the extra output columns for N, drop the
  // extra output rows for M), so a coarse grid covers every model at the price
  // of wasted multiply-accumulates.
  out << "\n  artifact grid if (K,N) are rounded up to a quantum:\n";
  out << "  " << padLeft("quantum", 9) << padLeft("distinct(K,N)", 16)
      << padLeft("padded/useful MACs", 22) << "\n";
  const int quanta[] = {16, 32, 64, 128, 256};
  for(size_t qi = 0; qi < sizeof(quanta) / sizeof(quanta[0]); qi++) {
    const int q = quanta[qi];
    std::set<std::pair<int, int>> padded;
    double useful = 0.0;
    double actual = 0.0;
    for(size_t i = 0; i < uses.size(); i++) {
      const GemmUse& u = uses[i];
      if(u.rows != RowKind::Spatial)
        continue;  // batch/attn rows are a different dispatch story; count the bulk only
      const int Kp = roundUpTo(u.K, q);
      const int Np = roundUpTo(u.N, q);
      padded.insert(std::make_pair(Kp, Np));
      useful += (double)u.K * u.N;
      actual += (double)Kp * Np;
    }
    string waste = "n/a";
    if(useful > 0.0) {
      std::ostringstream w;
      w.precision(3);
      w << (actual / useful) << "x";
      waste = w.str();
    }
    out << "  " << padLeft(q, 9) << padLeft((int)padded.size(), 16) << padLeft(waste, 22) << "\n";
  }
  out << "  (grid counts and waste cover spatial GEMMs only - the bulk of the work)\n";

  return out.str();
}

}  // namespace RyzenAIShapes
