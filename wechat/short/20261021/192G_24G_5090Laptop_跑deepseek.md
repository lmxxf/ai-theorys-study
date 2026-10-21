# 192G 内存 + 24G 显存（RTX 5090 笔记本）跑 DeepSeek-V4.1-Flash-1017 方案调研

**日期：2026-10-21**

---

## 一、核心结论：完全能跑，而且刚好卡在“主干 100% 驻留内存+显存”的黄金甜点上

`DeepSeek-V4.1-Flash-1017` 总参数量高达 **763.2B**（原版权重约 511 GB），拆开看其实是两大块：
1. **主干网络（552B MoE，每层 384 个路由专家选 6 个 + 1 个共享专家 + 注意力层）**
2. **两层 Engram 词组记忆表（第 2 层和第 15 层，共 196B 参数，原版 FP8 占约 189 GiB）**

如果用传统 `llama.cpp` 的笨办法把主干和 Engram 表一股脑全塞进内存，哪怕压成 1-bit（如 `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` 的 `UD-IQ1_S`，主干 217 GiB + Engram 表），192 GB 内存也装不下，一旦触发 SSD 虚拟内存交换专家，速度就会暴跌。

**真正的解法（Strata / Xolotl / Maya / Lucebox 路线）**：
因为 Engram 词组表每个词元只按哈希值查极少数几行，**直接把 160～189 GiB 的 Engram 表留在 NVMe SSD 固态硬盘上按需随机读（read in place），一字节内存都不占**；而把 **552B 的主干网络量化到 142～173 GiB**，让最热的专家常驻在 **24 GB 显存** 里，其余专家 100% 常驻在 **192 GB 内存（pinned RAM arena）** 里，生成时零 SSD 专家交换！

---

## 二、RTX 5090 笔记本（192G RAM + 24G VRAM）首选方案：`Xolotl`（Strata 的 V4.1 分支）

我们在 338 期跑 `Qwen3.8-Flash-Next` 用的原版 `Niko1221/Strata` 衍生出了一个专门跑 `DeepSeek-V4.1-Flash`（以及 `GLM-5.3-Flash`）的加强分支，叫 **`Indras-Mirror/Xolotl`**（基于 `Strata 0.1.44.1` 增加了约 7.1 万行代码）：
- **引擎仓库**：`https://github.com/Indras-Mirror/Xolotl`
- **实测性能参考**：在单张 24 GB 显存的 RTX 4090 + 90 GB DDR4 + 单块 NVMe SSD 上，跑 552B 的 `DeepSeek-V4.1-Flash`（512K 上下文），**Prefill 速度 180～320 tok/s，Decode 生成速度达 26.8 tok/s**（HumanEval 21/21，工具调用 10/10）。在 RTX 5090 上开启多轮会话池（Conversation pool，`--conversation-cache-mib`），30K～60K 上下文的 Agent 轮次切换耗时从 47.5 秒压到了 **3.96 秒**。

### 1. 适配好的三个 `1017` 量化模型仓库（均 < 192 GB 内存上限）

| 模型仓库 | 主干大小（驻 RAM+VRAM） | DSpark 投机头 + 视觉塔 | Engram 表（驻 SSD，不占内存） | 量化特点 |
|---|---:|---:|---:|---|
| **① `kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata`**（首选，未阉割原版对齐） | **162.4 GiB**（两分片 90 + 72.4 GiB，专家预算 153 GiB） | **5.3 GiB**（`Q2_K`） | **188.8 GiB**（L1 94.4 GiB + L14 94.4 GiB） | 用 Agent + 编程轨迹（SWE-rebench / SWE-smith 等）做激活校准，对 30,720 个专家投影按敏感度分配混合精度：28.4% 热门专家用 `Q3_K`（放显存/内存）、38.4% 用 `Q2_0`、33.2% 最冷门用 `1.69 bpw` 三值化（放 CPU 算）。自带 `rank.bin` 热度表（前 8 GiB 热门专家即可覆盖 29.9% 的路由词元）。 |
| **② `IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored`**（Xolotl 官方去审查版） | **V2: 142 GiB**<br>**V3: 152 GiB** | **5.3 GiB** | **约 189 GiB** | 同样基于 antirez 的 GSQ 混合精度管线，V3 针对代码与 Agent 重度负载重新校准，体积更小（142 / 152 GiB），给 192 GB 内存留出 40～50 GB 充裕余量。 |
| **③ `peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF`**（备选，配 `mw00/project-maya` 引擎） | **178.2 GB**（约 **165.9 GiB**，文件 1～4） | **8.0 GB** | **161.3 GB**（`Q6_K`，文件 5～6） | 路由专家用 IST Austria 的 GPTQ 式误差反馈量化（`IQ2_XXS` / `IQ3_XXS`，首尾四层 `IQ2_S` / `IQ3_S`），注意力与共享专家保持满血 `Q8_0`。在留出测试集上与原版满血模型 **Top-1 选词一致率高达 92.6%**（KL 散度仅 `0.120`）。 |

> **硬盘空间准备**：虽然内存 + 显存只需要装 150～175 GiB 的主干，但 NVMe SSD 固态硬盘需要预留 **约 360～380 GB** 空间来存放主干文件 + 约 189 GiB 的 Engram 表文件。

---

## 三、RTX 50 系显卡（Blackwell `sm_120`）编译 `Xolotl` 的必看避坑点

`Indras-Mirror/Xolotl` 原生支持 Linux 与 Windows 11（无需 WSL），但在 **RTX 5090 / 5090 Laptop（`sm_120`）** 上编译时，官方 `docs/WINDOWS.md` 与 Issue `#14` 特别标出了两条硬性要求：

1. **CUDA 版本必须 ≥ 13.1，且架构参数必须写 `120a`（带后缀 `a`）**：
   - CUDA 12.8 无法编译 `sm_120` 下的 MXFP4 块缩放矩阵乘指令（`ptxas` 会报错 `.kind::mxf4 not supported on .target 'sm_120'`）。
   - 必须使用 **CUDA 13.1+**，且 CMake 参数必须传 **`-DCMAKE_CUDA_ARCHITECTURES=120a`**（只写 `120` 同样会失败，后缀 `a` 开启架构专属特性是必需的）：
     ```bash
     git clone https://github.com/Indras-Mirror/Xolotl.git xolotl && cd xolotl
     bash tools/xolotl/fetch-llama.sh
     cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
       -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a \
       -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
       -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON
     cmake --build build-xolotl
     ```
2. **Windows 下编译需指定 VS 2022 Build Tools**：
   - 如果机器上装了更新的预览版 Visual Studio（如版本 18），`nvcc` 会报错不支持宿主机编译器（`C1189`），需显式将 `VCVARS` 指向 **VS 2022** 的 `vcvars64.bat`。

---

## 四、补充对照：如果是 AMD 显卡（ROCm / HIP）跑 `V4.1-Flash-1017` 用什么？

需要注意区分：
- 原版 **`Niko1221/Strata`**（跑 `Qwen3.8-Flash-Next`）同时支持 NVIDIA 和 AMD（RX 9070 XT / RX 7900 / Strix Halo）；
- 但上面跑 `DeepSeek-V4.1-Flash` 的 **`Indras-Mirror/Xolotl` 分支目前只有 NVIDIA CUDA 内核**。
- 如果要在 **AMD 显卡（RDNA 3 / RDNA 4 / Strix Halo）** 上用 192 GB 内存跑 `DeepSeek-V4.1-Flash-1017`，社区对应的专门引擎是 **`Luce-Org/lucebox`**（`https://github.com/Luce-Org/lucebox`），配套模型仓库是 **`Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF`**：
  - 针对 **192 GB 内存** 的专用档位是 **`DeepSeek-V4.1-Flash-1017-ROCMFP2S-MIX10-IM1017.gguf`**：路由专家总大小正好 **`164 GiB`**（前 10 层 `down` 投影为 `fp3`、其余为 `fp2s`，注意力与共享专家为 `Q8_0`），能全部常驻在 192 GB 内存里，Engram 表同样留在 SSD 上原地按需读；
  - 配套的 `DSpark` 投机草稿头专门提供了一个 192 GB 机器用的 **`4.6 GiB` 轻量版**（`DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf`，启动参数 `--profile ds41-gorgon`）。

---

## 五、横向对比：家里现成的「双机 DGX Spark（2 × 128G）」跑 V4.1-Flash-1017 有多快？

社区专门为 **2 台 DGX Spark（`TP=2` over CX7 200Gbps，`sm_121a`）** 做了一套开箱即用的 `vLLM + EXL3` 生产套件，配的是更高精度的 **`2.90 bpw`（主干 196.3 GiB）**：
- **部署套件仓库**：`https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks`
- **1017 权重仓库**：`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`（196.3 GiB，代码困惑度仅比 511 GB 原版高 1.8%，散文仅高 0.4%；每台 Spark 分担 **99.5 GiB** 权重 + **2.5 GiB** KV 池，支持高达 **600K** 上下文；原版第 47、48 分片的 190 GiB FP8 Engram 表走 NVMe 文件直读，不占内存）。

### 双机 DGX Spark vs 192G+24G（5090 笔记本）实测速度对比

| 指标 | 单机 192G RAM + 24G VRAM（5090 笔记本跑 `Xolotl`） | 双机 DGX Spark（`TP=2` 跑 `MiaAI-Lab` vLLM + EXL3） | 差异原因 |
|---|---|---|---|
| **主干量化档位** | `~2.4 bpw`（GSQ 混合精度，**152～162 GiB**） | **`2.90 bpw`**（EXL3 `mul1`，**196.3 GiB**，每节点 99.5 GiB） | 双机 Spark 合计 256 GB 统一内存，装得下接近 3-bit 的高精主干 |
| **Prefill 预填充速度** | **180 ～ 320 tok/s** | **970 ～ 1,338 tok/s**（**快 3～4 倍**；100K 长文仍有 **979 tok/s**，601K 长文 **810 tok/s**） | 笔记本只有 24G 显存，Prefill 时冷专家需走 PCIe 从内存搬运；双机 Spark 的 196 GiB 主干 **100% 常驻 GPU 统一内存**，零 PCIe 搬运 |
| **Decode 单流生成（C=1）** | **~26.8 tok/s**（4090 实测，5090 Laptop 约 **30 tok/s**） | 默认核：散文 **31.6 tok/s** / 代码 **38.8 tok/s**<br>开 `cooperative_moe`：基准 **40.2 tok/s** / 代码 **43.0 tok/s** | 两台 Spark 聚合了 `2 × 273 = 546 GB/s` 内存带宽，配合 DSpark（`k=3`）与 `sm_121a` 定制核 |
| **Decode 双并发总吞吐（C=2）** | 单流为主 | 默认核 **42.5～45.9 tok/s**<br>开 `cooperative_moe` 达 **61.1 tok/s** | 双机 TP=2 在小并发下吞吐扩展性更好 |

> **注**：社区还有 3～4 台 DGX Spark 跑 Zig 编写的 `TensorFold` 引擎 + `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw`（240 GiB）方案（仓库 `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`），单流生成可达 **90～101.8 tok/s**（代码 **134.8 tok/s**），Prefill 达 **2,215 tok/s**，但至少需要 3 台 Spark 才能装下 240 GiB 的 3.5 bpw 权重。

---

## 六、核心洞察：为什么 `Strata` 模式在 552B 的 V4.1 上没能跑赢“慢吞吞”的双机 DGX Spark？

很多人直觉上觉得：DGX Spark 单机只有 273 GB/s 的 LPDDR5x 统一内存带宽，平时看着慢吞吞的；而 RTX 5090 笔记本的 24 GB GDDR7 显存带宽高达 896 GB/s，配合 `Strata` 把热门专家放显存，难道不应该反超 Spark 吗？

算一下两边的 **带宽木桶短板** 就彻底明白了：

1. **在 `24G 显存 + 192G DDR5 内存` 的笔记本上（`Strata / Xolotl` 模式）**：
   - 24 GB 显存扣掉常驻的注意力层、KV 缓存和 5.3 GiB 的 DSpark 投机头之后，剩下的空间只能塞下大约 **8～12 GiB 的最热专家**。
   - 根据 `kaushikvira` 仓库实测的 `rank.bin` 专家热度曲线，**前 8 GiB 热门专家只能覆盖 29.9% 的路由命中**！
   - 这意味着 **剩下约 70% 的专家调用，每走一步都必须落回到笔记本的 DDR5 系统内存（双通道带宽仅约 90 GB/s）或走 PCIe 5.0 总线（约 60 GB/s）去现算现搬**。哪怕 24 GB 显存内部有 896 GB/s 的高带宽，70% 的时间也被死死卡在了 **60～90 GB/s 的 DDR5/PCIe 慢车道** 上（Prefill 阶段要批量激活几乎所有专家，更是全卡在 PCIe 搬运上，所以只有 180～320 tok/s）。

2. **在 `2 × 128G = 256G` 的双机 DGX Spark 上（`TP=2` 模式）**：
   - 单台 Spark 内存带宽是 273 GB/s，两台做 `TP=2` 张量并行聚合起来就是 **`546 GB/s`**。
   - 虽然 546 GB/s 表面上比独立显卡的 GDDR7 峰值低，但胜在 **196.3 GiB 的全部 384 个专家 100% 都平铺在这 546 GB/s 的统一内存池里**！
   - 不管当前词元抽中了最热的专家还是最冷门的专家，全都是按 **546 GB/s** 的速度直接喂给 GPU 核心，**0% 掉进 60～90 GB/s 的慢车道，也完全没有 PCIe 搬运开销**。

### 最终物理分界线（买硬件决策判据）：
- **当模型主干在 `60～80 GiB` 以内时**（如 125B 的 `Qwen3.8-Flash-Next` Q2 约 34 GB 专家）：24 GB 显存能拦住很大比例的热门专家，`Strata` 靠独显高带宽非常香（能跑到 120+ tok/s）；
- **一旦模型主干膨胀到 `150～200 GiB` 级别**（如 552B 主干的 `DeepSeek-V4.1-Flash`）：24 GB 显存连三成专家都兜不住，70% 流量掉进 DDR5 慢内存，这时反而是 **双机 DGX Spark（256 GB 全员 546 GB/s 统一内存）** 在生成速度（40+ tok/s vs 27 tok/s）、预填充速度（1,000+ tok/s vs 250 tok/s）和量化精度（2.9 bpw vs 2.4 bpw）上全面胜出。手里已经有双机 Spark + 3080 游戏本的情况下，完全没必要为了跑本地大模型去买 192G+24G 的重型 5090 笔记本。


