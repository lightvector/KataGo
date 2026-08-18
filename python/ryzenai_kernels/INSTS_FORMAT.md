# `.insts.bin` 指令流格式

IRON/aiecc 产出的 `.insts.bin` 是 NPU 的 **TXN 控制码**：一串 32 位命令，配置 shim DMA 的 buffer descriptor 并把它们压进 DMA 队列。`.xclbin` 提供阵列配置（与 shape 无关），这份指令流则编码了具体的 M/K/N。

M3 的目标是在 C++ 里按 (arch, cols, M, K, N, tile) 现场生成同样的字节流，从而摆脱对每-shape 预编译产物的依赖。

工具：`parse_insts.py`（遍历 `artifacts/manifest.json` 里的全部 golden 并 dump 成可读命令列表）。
参考实现：`tmp/FastFlowLM/src/include/npu_utils/npu_instr_utils.hpp`（735 行，MIT，可商用）。

## 头部（4 个 uint32）

| 字 | 位域 | 含义 |
|---|---|---|
| w0 | `[31:24]` n_rows | 6 |
| | `[23:16]` gen | **3 = npu1/aie2，4 = npu2/aie2p** |
| | `[15:8]` minor | 1 |
| | `[7:0]` major | 0 |
| w1 | `[15:8]` mem_tile_rows | 1 |
| | `[7:0]` num_cols | **物理列数**（npu1=4、npu2=8），不是设计实际用的列数 |
| w2 | — | 命令条数 |
| w3 | — | 总字节数（与文件大小一致，可用于自检） |

移位常量与 FastFlowLM `npu_sequence` 的 `dev_n_row_shift=24 / dev_gen_shift=16 / dev_minor_shift=8 / dev_major_shift=0`、`dev_mem_tile_rows_shift=8 / dev_num_cols_shift=0` 完全一致。

## 命令类型

| op | 名称 | 作用 |
|---|---|---|
| 0 | WRITE32 | 单字写寄存器 |
| 1 | BLOCKWRITE | **写一个 DMA buffer descriptor**（本格式的主体） |
| 3 | MASKWRITE | 带掩码写 |
| 6 | PREEMPT | 抢占级别 |
| 0x80 | WAIT_TCT | 等待 task completion token |
| 0x81 | DDR_PATCH | **把 BD 的地址字段绑定到某个 kernel 参数槽** |
| — | ISSUE_TOKEN / QUEUE_PUSH | 配置 token、把 BD 压进 DMA 通道队列 |

## 主体结构

高度规整，反复出现同一四元组：

```
BLOCKWRITE  col row bd addr len bufoff D0=(size,stride) D1=(size,stride) D2=(size,stride) iter cache lock
DDR_PATCH   col row bd addr arg_idx arg_off        <- arg_idx: 0=A, 1=B, 2=C
[ISSUE_TOKEN col row S2MM/MM2S ch pkt_id mask]
QUEUE_PUSH  col row MM2S/S2MM ch bd repeat issue_token
```

`arg_idx` 与 XRT 的参数槽对应（见 `../manifest.h`：ARG_A=3、ARG_B=4、ARG_C=5，减 3 即为此处的 0/1/2）。

**命令条数只取决于 (arch, cols)，与 M/K/N 无关**——npu2_1col 恒为 35 条，npu1_4col 恒为 136 条。换 shape 只改字段值，不改结构。这对生成器是极有利的性质。

## 字段与 (M, K, N, tile) 的关系式

以 **npu2_1col、M=512、bf16→fp32** 为例，横向对比 K=N∈{384,512,768} 得出（三个 shape 全部吻合）：

| 字段 | 公式 | K=384 | K=512 | K=768 |
|---|---|---|---|---|
| BD0 (C, arg_idx=2) `len` | **M·N/4** | 49152 | 65536 | 98304 |
| BD0 `D1.stride` | **N** | 384 | 512 | 768 |
| BD0 `D0.size` | **tile_n** | 48 | 32 | 32 |
| BD1 (A, arg_idx=0) `len` | **M·K/8** | 24576 | 32768 | 49152 |
| BD1 `D1.stride` | **K/2**（bf16 每字打包 2 个） | 192 | 256 | 384 |
| BD3 `bufoff` | **M·K/4** | 98304 | 131072 | 196608 |
| QUEUE_PUSH `repeat` | **N/tile_n − 1** | 7 | 15 | 23 |

`len` 的单位不是字节也不是字，而是 4 字的块（C 是 fp32：`M·N·4 字节 = M·N 字 = M·N/4 块`；A 是 bf16：`M·K·2 字节 = M·K/2 字 = M·K/8 块`）。

**结论：没有任何字段需要 IRON 的内部布局决策，全部可从 (M, K, N, tile) 推导。** 这是 M3 可行性的关键判据。

## 多列与两代架构

对 K=N=512 横向对比：

| 列数 | 命令数 | C 的 BD `len` | 用到的 col |
|---|---|---|---|
| 1 | 35 | 65536 | 0 |
| 4 | 137 | 16384 = 65536/4 | 0, 1 |
| 8 | 225 | 8192 = 65536/8 | 0, 3 |

**`len` 精确按列数等分**：`len = M·N/4/cols`，即每列负责 N/cols 的输出列块。A 的 BD 同理。

### ★ 命令体与架构无关

npu1 与 npu2 在**相同 (cols, M, K, N, tile)** 下命令体完全相同：

- npu1_1col 与 npu2_1col：均 35 条，BD0 `len`=65536、BD1 `len`=32768，逐字段一致
- npu1_4col 与 npu2_4col：均 137 条，首个 BD `len`=16384，一致

**两代架构的差异只在头部 w0 的 `gen` 字段（npu1=3、npu2=4）。** 生成器因此不需要为两代写两套逻辑，只需在头部填不同的 gen。

## ✅ 多列已解决（2026-08-13，本节推翻下面「关键障碍」一节）

下面那节的统计是在**没有 chunk 结构认知**的前提下做的，把整条流按列计数、混进了 chunk 的重复，因而得出「每列命令数不对称、不可推导」的错误结论。重新分解后：

| 列数 | 每 chunk 描述符数 | 构成 |
|---|---|---|
| 1 | 5 | 1×C + 2×A + 2×B |
| 4 | 20 | 4×C + 8×A + 8×B |
| 8 | 32 | 8×C + 8×A + 16×B |

**唯一不可推导的东西只有「哪个描述符落在哪个物理 shim 列、按什么顺序」**——这是 placer 的输出。而它：

- **只取决于列数**，与 M/K/N/tile 全部无关
- **与架构无关**（npu1 与 npu2 的表逐字段相同）
- **与 dtype 无关**（bf16 与 bfp16 的表相同）

所以把它按列数各存一张表就够了。`kernels/extract_layout.py` 从 golden 里提取并生成 `../sequence_layout.h`（1col 5 条、4col 20 条、8col 32 条，共 108 行），其余全部由 `(M,K,N,tile,cols)` 现场算出。

**每个角色的字段公式**（`cr = chunkRows`、`nA` = 表里 A 描述符个数、`sPerCol = N/(tileN·cols)`）：

```
C: len=cr*N/2/cols  D0=(tileN,1)   D1=(cr/2,N)    D2.stride=tileN*cols
   iter=(2, cr*N/2) bufoff=chunk*cr*N*4 + slot*tileN*4     S2MM repeat=1
A: len=cr*K/2/nA    D0=(tileM,1)   D1=(cr/nA,K/2) D2.stride=tileM
   iter=(1,1)       bufoff=chunk*cr*K*2 + slot*cr*K/cols
B: len=K*tileN/2    D0=(tileN/2,1) D1=(tileK,N/2) D2.stride=tileK*N/2
   iter=(sPerCol, tileN/2*cols)    bufoff=slot*tileN*2（不随 chunk 前进）
MM2S repeat = sPerCol - 1
```

**边界情形 `sPerCol == 1`**（每列正好一个输出 tile，如 8col + N=384/tileN=48）：没有 strip 循环可跑了，IRON 把 chunk 的两个半程折进描述符本身——
`C.len` 翻倍为 `cr*N/cols`、`D2.stride` 变成 `cr*N/2`、`iter=(1,1)`、S2MM `repeat=0`；同时 `B.iter` 的 stride 也归 1（一次迭代的循环没有步长）。两种编码搬运的字节数完全相同。

**WAIT_TCT 不需要单独的表**：它就是 C 描述符的 `(col, ch)` 按表内顺序、每 chunk 重复一遍，`w3 = (ch<<24) | 0x00010100`。发放节奏：chunk0 后 0 条、chunk1 后 2×cols 条、chunk≥2 后各 cols 条。

**BD 号**：chunk0 用集 A、chunk1 用集 B（两者重叠执行），chunk≥2 因为前面必有 wait，一律复用集 A。

### 验收

| 测试 | 结果 |
|---|---|
| 逐字节 vs golden | **35/35**（1/4/8 列 × npu1/npu2 × bf16/bfp16 × M∈{256,512,768,1024} × K∈{384,512,768} × N∈{256,384,512,768} × tileN∈{32,48}） |
| 硬件对拍（生成流，无 golden 的 shape） | **34/34 PASS**，maxAbsErr ≤ 1.2e-05；M 到 5888、N 到 2048 |

约束：`N % (tileN*cols) == 0`。例如 N=1152 在 8col 下不合法（1152/256=4.5），生成器会明确拒绝，运行时降到 4col 或 1col 即可。

---

## ⛔ 关键障碍：多列指令流不可纯推导（**已被上一节推翻，保留作教训**）

按 (列号, 命令类型) 统计 K=N=512：

| 列数 | BLOCKWRITE | DDR_PATCH | QUEUE_PUSH | ISSUE_TOKEN | WAIT_TCT | 列号分布 |
|---|---|---|---|---|---|---|
| 1 | 10 | 10 | 10 | 2 | 2 | `{0:34}` |
| 4 | 40 | 40 | 40 | 8 | 8 | `{0:34, 1:44, 2:34, 3:24}` |
| 8 | 64 | 64 | 64 | 16 | 16 | `{0:12, 1:12, 2:34, 3:44, 4:44, 5:44, 6:34}` |

`ISSUE_TOKEN = WAIT_TCT = 2 × cols`，这条干净。但另外两点不干净：

1. **8 列的设计只用了 7 个列（0–6）**，不是 8 个；每列命令数完全不对称（12/12/34/44/44/44/34）。4 列同样不对称（34/44/34/24）。
2. **BLOCKWRITE 每列数量不是常数**：1col=10、4col=10/列、8col=8/列。

这种不对称说明指令流**不是「按列重复同一段」**，而是 IRON 的 placer 给每列分配了不同角色（某列做广播源、某列只做搬运等）。要逐字节复现就得复现 placer 的启发式决策——**这就是「掺了 IRON 内部布局决策」的风险在多列上的兑现**。

**当时的结论（错误）**：
- 1 列：可纯推导 ✓
- 多列：不可纯推导 ✗ 列角色分配不是 (M,K,N,tile,cols) 的函数

⚠️ **错在哪**：结论的后半句其实是对的——列角色分配确实不是 shape 的函数。错的是从这里跳到「所以多列不可生成」。它**是 cols 的函数**，一张 5/20/32 条的表就能钉死，其余全部可算。

**教训：区分「不可推导」和「不是所求参数的函数」。** 一个量只要维度足够低、取值足够少，列表化就等于解决。当时先按列做直方图、看到不对称就收手了，如果先按 chunk 分解再看，结构立刻是清楚的。

## ★★ M、N、K 的归属已实测分离（2026-08-13）

| 维度 | 在 xclbin 里？ | 判据 |
|---|---|---|
| **K** | **是** | K=512 的 xclbin 驱动 K=768 的 golden 流 → ERT=4 completed 但 maxAbsErr=56.1（静默错误）。xclbin 字节数只随 K 变：K=512→26056、K=768→28872，改 M 或 N 都不变 |
| **M** | 否 | 同一 xclbin 驱动 M=256 / M=768 的 golden 流 → PASS |
| **N** | 否 | 同一 xclbin 驱动 N=256 / N=768 的 golden 流 → PASS；反向用 K=768 的 xclbin 驱动 N=512 与 N=768 也都 PASS |

**结论：xclbin 由 (器件, 列数, dtype, tile, K) 参数化。M 和 N 完全由指令流决定。**

## ★ 指令流的 chunk 结构（M 自由的原因）

以 `chunkRows = tileM × 4 × 2` 行 M 为一个 **chunk**（4 个 AIE 行，×2 是乒乓双缓冲），每 chunk 恒 **17 条命令 / 161 字**：

```
BLOCKWRITE BD(C) + DDR_PATCH(arg=C) + ISSUE_TOKEN + QUEUE_PUSH(S2MM)
2× [ BLOCKWRITE BD(A) + DDR_PATCH(arg=A) + QUEUE_PUSH(MM2S ch0)
     BLOCKWRITE BD(B) + DDR_PATCH(arg=B) + QUEUE_PUSH(MM2S ch1) ]
WAIT_TCT ×1（但 golden 是每两个 chunk 攒着一起发两条）
```

- chunk 之间**除 DDR 偏移外逐字节相同**：实测 M=768 的 chunk0/chunk1 与 M=512 的完全一致
- BD 号在偶/奇 chunk 间交替 `5*(c%2) + {0..4}`，所以 M 再大也不超过 shim tile 的 16 个 BD
- 命令总数 = 17 × M/chunkRows；字数 = 4 + 161 × M/chunkRows（实测 M=256/512/768 → 165/326/487 字）
- IRON 要求 `M % chunkRows == 0`（报错原文：`M/m/n_aie_rows must be even`）

**每 chunk 的字段值（都不含总 M）**：
```
C BD: len=chunkRows*N/2, D1=(chunkRows/2, N), iter=(2, len), bufoff=chunk*chunkRows*N*4
A BD: len=chunkRows*K/4, D1=(chunkRows/2, K/2), iter=(1,1), bufoff=(2*chunk+half)*chunkRows*K
B BD: len=K*tileN/2, D1=(tileK, N/2), D2.stride=tileK*N/2, iter=(N/tileN, tileN/2), bufoff=0
```

## ⛔ 被这个结构咬到的第二个 bug

`sequence.cpp` 最初把 C/A 的字段写成 M 的函数（`lenC = M*N/4`、`D1.size = M/4`）。正确形式是上面的 chunk 常量。当时全部 golden 的 M 都是 512 = 2×chunkRows，两种写法**数值恰好相等**，9 个 golden 全过。

连同 B 的 `D1.stride` 被 K=N 掩盖那次，**同一类错误已经发生两次**。

⚠️ **规则：任何字段公式，必须用「该维度取至少两个不同值」的 golden 验证。** 单点验证等于没验证。

## ✅ 采纳方案（2026-08-13 实测后修订）

**预编译只按 K 一维铺网格（quantum 64）；M、N、列数全部由 `sequence.cpp` 现场生成。**

K 可以零填充抬高（A 补零列、B 补零行），所以一维 K 网格能覆盖**任意**模型，硬约束 C3 完整达成，不需要任何降级。

被否决的：逆向 IRON 的 placer 的**启发式**（工作量大、脆弱、上游一改就崩）。实际做法是不逆向启发式，只把它对每个列数的**输出**列成表——上游若改了 placer，重跑 `extract_layout.py` 即可。

## 实现后的订正（`../sequence.cpp` 已按 golden 逐字节对齐，9/9 PASS）

写生成器时发现上面几处推导有误，**以下为准**：

1. **w1 的 num_cols 不是物理列数**。npu1_1col 的 w1 = `0x00000101`（num_cols=1），npu2_1col 才是 8。对 1 列生成按 golden 直接填：npu1→1、npu2→8。
2. **A 的 BD `bufoff` = `M·K/2 × 序号`**（字节：0、M·K/2、M·K、3M·K/2），上表写的 `M·K/4` 漏算了 bf16 的 2 字节宽度。
3. **「命令体与架构无关」有一个例外**：BLOCKWRITE 的 D1 字里有个 burst 常量，npu1 = `0x80000000`、npu2 = `0xC0000000`（D1 的 size/stride 本身仍然一致）。连同头部 gen，两代共 **2 处**差异。

## ✅ K≠N 歧义已用非方阵 golden 判定（2026-08-13）

最初 9 个 golden 全部是 K=N 方阵，三个字段存在等价歧义。用 `M=512, K=384, N=768` 重新编一个 golden 后全部判定：

| 字段 | 判定 | 被排除的候选 |
|---|---|---|
| B 的 `D1.stride` | **N/2**（=384） | K/2（=192） |
| B 的 `D2.stride` | **tileK·N/2**（=24576） | tileK·K/2 |
| C/A 的 `D1.size` | **M/4**（=128） | — |

**这次验证抓到了一个真实 bug**：`sequence.cpp` 当时把 B 的 `D1.stride` 写成了 `K/2`，而 9 个方阵 golden 因为 K=N 全部通过，掩盖了错误。非方阵一测立刻在 4 个 B 的 BD 上暴露（4/326 个字不同）。

⚠️ **教训：只用方阵验证是不够的。** KataGo 的 attention 投影里 K≠N 是常态（如 QKV 合并投影 K=384、N=1152），这个 bug 若留到 M4 会让 NPU 静默算出错误的 policy 而不报错。今后新增任何字段公式，**必须同时用方阵和非方阵验证**。

## 当前验证状态

| 测试集 | 结果 |
|---|---|
| 9 个方阵 golden（bf16/npu1_1col、bf16/npu2_1col、bfp16/npu2_1col × 3 shape） | **9/9 逐字节一致** |
| 非方阵 M=512 K=384 N=768（npu2_1col bf16） | **逐字节一致** |

测试程序：`C:\Temp\seqtest.cpp`（方阵集）、`C:\Temp\nsqtest.cpp`（非方阵）。非方阵 golden 由
`gemm_bf16.compile_aot('npu2',1,'bf16',512,384,768,32,64,32,...)` 生成，冷编译 30.7 秒。

## 尚未验证的部分

- `iter`、`cache`、`lock` 在已对比样本里未变化，是否与 shape 相关未知。
- 关系式表只用 bf16 推导，bfp16 变体未对比（预期只有 tile 几何不同，因为 BFP16 的微内核是 8×8×8 而非 4×8×8）。
- 1 列情况下 BLOCKWRITE 恒为 10 条，与 M/K/N 无关——需确认更极端的 shape 是否仍然成立（BD 硬件上限是每 shim tile 16 个）。

## 下一步

1. 把上面的对比扩展到 4col / 8col 和 npu1，补全关系式
2. 写 `../sequence.{h,cpp}`（命名空间 `RyzenAISequence`），C++17，只用标准库
3. 对 `artifacts/` 下全部 24 个 golden 逐字节 `memcmp`——**这是唯一的验收标准**
