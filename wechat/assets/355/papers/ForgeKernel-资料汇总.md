# ForgeKernel 论文与国内官方报道对账归档

## 1. 国外 ICLR 2027 双盲评审论文（OpenReview 原件记录）
- **标题**：`ForgeKernel: Self-Evolving Kernels Catching And Surpassing Experts And Agents`
- **OpenReview ID**：`EtV3f5BPD4`
- **论坛链接**：https://openreview.net/forum?id=EtV3f5BPD4
- **PDF 链接**：https://openreview.net/pdf?id=EtV3f5BPD4 （内部存储路径：`https://openreview.net/pdf/22e726ebf7b3734d6d511ae6e1e8c50d0f470ed9.pdf`）
- **关键词**：`GPU kernel optimization; long-horizon autonomous agents; agent harness evaluation; domain-specific orchestration`

### 英文摘要原文（Abstract）
```text
Frontier coding agents now write GPU kernels that rival expert ones: left to run,
even a generic agentic loop beats hand-tuned references on many operators. Yet
every such loop stops at a ceiling of its own making. It settles a kernel’s architecture in its first few attempts and spends the rest of the run tuning beneath it, and
each run starts from the same knowledge as the last and discards what it measured.
We present ForgeKernel, a self-evolving kernel agent built to break that ceiling.
It reads a reference kernel’s architecture from hardware behavior alone — running, timing and profiling a sealed binary, never its source — and turns it into an
architecture-first curriculum that a coding agent implements and that is reopened
whenever measurement shows the architecture, not the tuning, is the bottleneck.
Every accepted kernel is sealed and promoted to the next campaign’s reference,
so optimization continues past the best known implementation, and every finding
certified by a hardware probe enters a knowledge base shared across operators.
Across 27 operators on Hopper and Blackwell, ForgeKernel matches or beats 18 of
22 expert-library kernels, including FlashAttention-4, cuDNN and DeepGEMM,
and all 7 published kernels of three kernel-generation systems, the latter by up to
+170.6% (+39.4% in geometric mean). On DeepSeek-V4.1’s sparse-MLA prefill
and fused attention kernels, released 15 days before this submission with no paper
or second implementation, it comes within 1.3% and 3.7%. With only its profiler
swapped, it leads the best public submission on all 8 level-4 CANN-Bench operators on Ascend, by +33.1% in geometric mean. Promoting the best known kernel
to the reference, a generic loop’s included, gains up to a further +9.9%.
```

---

## 2. 国内面壁智能官方战报与 CNCC 2026 演讲报道（实锤同一系统）
- **CSDN 报道（2026-10-27）**：https://www.csdn.net/article/2026-10-27/169267396 （《面壁智能 RSI 新进展：ForgeKernel 登顶华为、英伟达官方算子榜》）
- **开源中国报道（2026-10-27）**：https://www.oschina.net/news/503257
- **搜狐科技 CNCC 2026 报道（2026-10-23）**：https://m.sohu.com/a/1091803360_104421 （《面壁智能亮相 CNCC 2026：第三届端侧大模型论坛成功举办》，李宇轩演讲《AI 制造 AI：Forge Engineering 驱动端侧大模型的芯片级训推定制》）

### 核心数据对账：
1. **华为昇腾 CANNBench 三榜第一**：
   - `950DT` 榜单：总分 `3903.96` 第一，基准加速比 `1.523×`（约为第二名 `0.305×` 的 5 倍），提交算子 `55/60`；
   - `950PR` 榜单：总分 `3736.44` 第一（领先第二名 `5.2%`），加速比约为第二名的 `1.6` 倍，提交算子 `50/60`；
   - `A3` 榜单：总分 `4338.67` 第一（领先第二名 `7.1%`），加速比约为第二名的 `1.3` 倍，提交算子 `58/60`；
   - 论文摘要中亦明确记载：换用昇腾 Profiler 后在全部 8 个 Level-4 CANN-Bench 算子上以几何平均 `+33.1%` 夺冠。
2. **英伟达榜单与 27 类算子盲测**：
   - 登顶英伟达 `SOL-ExecBench`（`FlashInfer-Bench` 合集），SOL 分数 `0.814101` 第一；
   - Hopper / Blackwell 上 27 类算子打平或超越 `18/22` 个专家库（含 `FlashAttention-4`、`cuDNN`、`DeepGEMM`）；
   - 在提交前 15 天刚发布的 `DeepSeek-V4.1` 的 `sparse-MLA prefill` 和 `fused attention` 算子上逼近至官方手写算子的 `1.3%` 与 `3.7%` 以内。
