# B12X 被 FlashInfer 招安：消费级与工作站 Blackwell（sm_120/121）算子终于转正

**日期：2026-10-20**

---

## 起因：原仓库归档，全建制并入 FlashInfer

今天看了一眼之前在 DGX Spark（`sm_121`）和 RTX 50 系 / RTX PRO 6000（`sm_120`）玩家圈里救命用的开源算子仓库 `b12x`（`local-inference-lab/b12x`），发现原仓库已经挂牌归档（Archived），首页写着一行字：

> **b12x has moved to FlashInfer**
> This repository is archived. Development, bug reports, and pull requests belong in FlashInfer. The b12x inference kernels and their preparation, autotuning, compilation, workspace, and checkpoint loading infrastructure live under `flashinfer/experimental/b12x`.

这支在民间野蛮生长了几个月的“梁山好汉”，正式被推理正规军 **FlashInfer** 招安收编了。

---

## 1. 先记一下：FlashInfer 是什么？

平时大家常听的是 Tri Dao 的 **FlashAttention**，但 FlashAttention 最初主要是为**训练阶段**设计的（输入是一整块规整、连续的张量）。

到了**大模型在线推理阶段（vLLM / SGLang / TensorRT-LLM）**，场景完全变了：
- 同一个批次（Batch）里混着几十个不同用户的请求，有的在跑长文本预填充（Prefill），有的在逐词生成（Decode）；
- 键值缓存（KV Cache）被切成了豆腐块散落在显存各处（PagedAttention 分页缓存），还要处理前缀复用树（RadixCache）、MLA 潜空间压缩、以及 DSA/CSA 稀疏挑选。

陈天奇团队（UW / CMU，即 TVM 与 MLC-LLM 的原班人马，主力作者 Zihao Ye 等）专门针对**“大模型在线推理”**造了一套底层 CUDA/CuTe 算子库，就叫 **FlashInfer**。
- **它在生态里的位置**：如果说 `vLLM` 和 `SGLang` 是负责排班、管显存和调度请求的“总包工头”，那么 **FlashInfer 就是底下真刀真枪拿 CUDA 干脏活的“主力施工队”**（SGLang 从头到脚架在 FlashInfer 上，vLLM 现在的注意力与核心融合算子也重度依赖 FlashInfer）。

---

## 2. B12X 之前在干什么？（被老黄刀法逼出来的民间自救）

`B12X` 这个名字的意思非常直白：**`B`lackwell `sm_12X`**。

老黄在 Blackwell 这一代劈了一刀狠的：
- **数据中心满血卡（B200 / GB200）** 是 **`sm_100`**：每个 SM 有 228 KB 共享内存、专属的 256 KB 张量内存（TMEM），以及第五代矩阵指令 `tcgen05`。DeepSeek 官方的 `DeepGEMM` 和 `FlashAttention-4` 全是绑死在 `sm_100` 上的。
- **消费级与工作站卡** 全被切成了 **`sm_12x`**：RTX 5090、工作站神卡 RTX PRO 6000 Blackwell（96GB 显存）是 **`sm_120`**，DGX Spark（GB10）是 **`sm_121`**。这批卡物理上没有 TMEM、没有 `tcgen05`，每个 SM 的可用共享内存只有 **99 KB**（不到 B200 的一半）。

结果就是：4 月底 DeepSeek V4 Flash（原生 FP8 + MXFP4 混合精度 MoE）刚出的时候，上游大厂的算子在 `sm_120/121` 上不是报指令不存在，就是报 `shared memory overflow`（共享内存溢出）；vLLM 内置的通用 Marlin 算子跑起来还会在第 43 层末尾出乱码，逼得我们当时只能自己手搓“前 42 层 Marlin + 最后 1 层 Triton”的混合后端，关掉 CUDA Graph 跑出 12 token/s。

后来玩多卡 RTX PRO 6000 和 DGX Spark 的硬核社区玩家自己动手，用 NVIDIA 的 `CuTe-DSL` 专门针对 `sm_120/121` 的 99 KB 共享内存和寄存器排布，量体裁衣写出了 **`b12x`**：
- 把 **NVFP4 / MXFP4 的 MoE 路由、动态激活量化、双矩阵乘、SwiGLU 和 Top-K 归约** 全部融进专门调优的 `b12x_fused_moe` 算子里；
- 配合定制的稀疏 MLA 注意力与整图 CUDA Graph，直接把我们在 DGX Spark 上的单流解码速度从 **12 token/s 翻倍到了 24.5 token/s**（开 DSpark 投机解码冲到 83.8 token/s）。

---

## 3. 被招安进 `flashinfer/experimental/b12x` 意味着什么？

以前想在 5090、PRO 6000 或 DGX Spark 上用 `b12x`，非常折腾：因为它游离在官方主线之外，你必须去找社区玩家自己打补丁的私人 vLLM fork，或者去拉民间打包好的野生 Docker 镜像（比如 `aidendle94/sparkrun-vllm-ds4-gb10`），每次上游模型一更新（比如 V4.1 或者 0731 草稿头微调），还得等民间作者更新镜像。

现在 `b12x` 整建制搬进 `flashinfer/experimental/b12x`，并在上游 `vLLM` 里注册成正式后端（如 `flashinfer_b12x_moe`）：

1. **告别野生镜像和私人 Fork**：以后标准安装的 `FlashInfer` + `vLLM` 主线里，直接就自带了这套针对 `sm_120/121` 量体裁衣的 FP4 MoE 与稀疏注意力内核。
2. **消费级/工作站级 Blackwell 成了推理一等公民**：大厂一开始看不上“拿 5090 / PRO 6000 / Spark 跑几百 B 的 MoE 大模型”，结果民间社区用真金白银的出货量和翻倍的算子性能，倒逼上游官方把这套代码收编成了标准基础设施。
