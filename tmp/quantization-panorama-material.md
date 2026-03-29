# 量化全景篇 - 素材汇总

> 朱雀子代理(Claude Code)搜集，2026-03-29

---

## 一、GGUF 格式（llama.cpp 生态）

### 1.1 什么是 GGUF？

GGUF = **GGML Universal File**，是一种二进制格式，将张量数据和元数据存储在单个文件中。设计目标：**可内存映射(memory-mappable)、可扩展(extensible)、自包含(self-contained)**。

### 1.2 GGUF 与 GGML 的关系

| 时间线 | 事件 |
|--------|------|
| 2022年9月 | Georgi Gerganov 开始开发 GGML 库（纯C张量运算库，严格内存管理+多线程） |
| 2023年3月 | llama.cpp 项目启动，基于 GGML，纯C/C++实现LLaMA推理 |
| **2023年8月21日** | **GGUF 格式发布**，替代旧的 GGML 格式 |
| 此后 | GGML 格式不再被 llama.cpp 支持 |

**为什么要从 GGML 升级到 GGUF？**
- GGML 格式缺乏元数据支持，扩展性差
- 随着 llama.cpp 支持越来越多模型架构，需要更好的向后兼容性
- GGUF 支持更好的 tokenization、特殊token、元数据存储

### 1.3 Q4_K_M、Q5_K_S、Q8_0 命名解读

命名格式：`Q{位数}_{量化方法}_{规模变体}`

**位数（Q后面的数字）：**
- Q8 = 8-bit，Q6 = 6-bit，Q5 = 5-bit，Q4 = 4-bit
- 位数越低 → 压缩越大 → 文件越小 → 质量损失越多

**K 代表什么？**
- **K = K-quant**，一种分块量化策略
- 将权重分成若干block，**每个block计算独立的缩放因子**
- 关键创新：**混合精度** —— 识别模型中更关键的权重/层用更高精度量化，不重要的用更低精度
- 相比旧的统一量化（如Q4_0），K-quant在相同位宽下质量大幅提升

**后缀含义：**
- `_0`（如Q8_0、Q4_0）：**Legacy格式**，全局量化，整个张量共享一个scale和zero point
- `_K_S`：K-quant **Small** —— 混合精度偏激进，更小但质量稍低
- `_K_M`：K-quant **Medium** —— 平衡点，最常推荐
- `_K_L`：K-quant **Large** —— 更多层用高精度，更大但质量更高

**实用推荐：**
- Q8_0：最接近全精度，大但质量最好
- Q5_K_M：质量/大小的甜蜜点
- Q4_K_M：4-bit中质量最好的选择，适合内存受限场景
- Q4_K_M 大幅优于 Q4_0（同为4-bit，策略不同，效果天壤之别）

### 1.4 为什么本地部署都用 GGUF？

1. **纯CPU推理**：llama.cpp 不依赖CUDA，Mac/PC/手机都能跑
2. **内存映射**：模型可以mmap加载，启动极快
3. **灵活量化**：从Q2到Q8多种精度可选，适配不同硬件
4. **单文件自包含**：模型+tokenizer+元数据全在一个文件里
5. **生态最大**：Ollama、LM Studio、GPT4All等主流本地工具都基于llama.cpp/GGUF

---

## 二、数据类型对比

### 2.1 主要数据类型一览

| 类型 | 位宽 | 结构 | 精度（有效数字） | 动态范围 | 主要用途 |
|------|------|------|-----------------|----------|----------|
| **FP32** | 32-bit | 1符号+8指数+23尾数 | ~7位十进制 | ~3.4×10^38 | 训练基准/研究 |
| **FP16** | 16-bit | 1符号+5指数+10尾数 | ~3.3位十进制 | ~65504 | 混合精度训练 |
| **BF16** | 16-bit | 1符号+8指数+7尾数 | ~2.4位十进制 | 与FP32相同 | 大规模训练首选 |
| **INT8** | 8-bit | 整数 -128~+127 | 256个离散值 | 固定 | 推理部署 |
| **INT4** | 4-bit | 整数 -8~+7 | 16个离散值 | 固定 | 极致压缩推理 |

**关键洞察：BF16 vs FP16**
- BF16 指数位与FP32相同（8位） → **动态范围与FP32一致**
- FP16 精度更高但范围小 → 容易溢出
- 深度学习对尾数精度不敏感，对指数范围敏感 → **BF16 is the king of training**

**内存效率：**
- FP32 → FP16：内存减半
- FP16 → INT8：再减半
- INT4 相比 FP32：**8倍压缩**
- 即使激进的4-bit量化，模型仍保留约98.1%的推理能力

### 2.2 FP4（NVIDIA Blackwell 的 NVFP4）

**格式规格：E2M1**
- 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits
- 可表示的正值：{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} + 零 + 对应负值
- 值域约 -6 到 +6

**为什么NVIDIA选浮点不选整数？**

核心原因：**动态范围**
- INT4 是均匀间隔的16个整数值（如-8到+7），对所有区间精度相同
- FP4 有指数位，能表示跨越多个数量级的值，**变精度**
- Transformer中的激活和权重分布跨越多个数量级，FP4的浮点语义天然更适合

**NVFP4 的两级缩放策略：**
1. 第一级：每16个元素一组，用 FP8 E4M3 作为缩放因子（比MXFP4的32元素更细粒度）
2. 第二级：FP32 标量做全局缩放
- 更小的block size → 更好适配数据的局部动态范围 → 量化误差更小

**Blackwell 硬件支持：**
- 第五代 Tensor Core 原生支持 NVFP4
- 自动处理微缩放FP4数据的分组、动态缩放和4-bit矩阵运算

### 2.3 NF4（NormalFloat4，QLoRA提出）

**来源论文：** "QLoRA: Efficient Finetuning of Quantized LLMs"
- arXiv: **2305.14314**
- 作者：Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- 发表于 **NeurIPS 2023**

**NF4 的核心思想：**

预训练神经网络的权重近似服从**零均值正态分布 N(0, sigma^2)**。

普通INT4量化用均匀间隔的量化level → 在权重密集的零附近浪费分辨率，在权重稀疏的尾部又不够用。

NF4的做法：
1. 将权重归一化到标准正态分布 N(0,1) 的范围
2. 计算标准正态分布的 **分位数(quantiles)**，将分布等概率切分为16份
3. 每个量化bin内的值数量相等 → **信息论最优**

**直觉解释：**
- 想象正态分布的钟形曲线
- INT4 把x轴等距切16刀 → 中间（值最多的地方）太粗，两边（值少的地方）太细
- NF4 把曲线下面积等分16份 → 值多的地方切得密，值少的地方切得疏
- 这就是"分布感知量化"

**为什么NF4比INT4好？**
- 对正态分布数据是**信息论最优**的4-bit量化
- 经验结果：NF4 在标准LLM任务上接近baseline性能，损失极小
- 由于预训练权重天然近似正态分布，这个假设几乎总是成立

---

## 三、LLM.int8()

### 论文信息

- **全名：** "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
- **arXiv：** **2208.07339**
- **作者：** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **发表：** NeurIPS 2022（第36届神经信息处理系统大会）
- **历史地位：** 第一个正式提出并系统研究大模型**异常值(outlier)问题**的工作

### 核心发现：涌现特征(Emergent Features)

Dettmers发现了一个关键的**相变(phase transition)现象**：

| 模型规模 | 异常值行为 |
|----------|-----------|
| < 2.7B | 异常值不明显 |
| 2.7B ~ 6B | 约60%的层对异常值维度达成共识 |
| **>= 6.7B** | **100%的层使用相同的异常值维度 → 相变发生** |
| 6B → 13B | 异常值维度从约15个增长到约60个 |

这些异常值：
- 幅度远大于其他特征（可达正常值的100倍以上）
- 出现在**固定的特征维度**上
- 对模型的注意力机制和预测性能至关重要
- 如果被INT8截断，模型性能灾难性崩溃

### 混合精度分解方案

**核心方法：**
1. **向量级量化(Vector-wise Quantization)**：矩阵乘法中每个内积使用独立的归一化常数（而非整个张量共享）
2. **异常值隔离**：检测异常值维度（幅度>阈值的特征维度）
3. **混合精度分解**：
   - 异常值维度 → 抽出来用 **FP16** 做矩阵乘法
   - 其余99.9%以上的值 → 用 **INT8** 做矩阵乘法
4. 两部分结果相加得到最终输出

**效果：**
- 支持最大175B参数模型的推理
- 内存减半
- **无性能损失**（这是关键突破）

---

## 四、SmoothQuant

### 论文信息

- **全名：** "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
- **arXiv：** **2211.10438**
- **作者：** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **机构：** MIT + NVIDIA
- **发表：** **ICML 2023**（第40届国际机器学习大会，夏威夷）

### 核心问题

量化LLM的难点不在权重，在**激活值(activations)**：
- 权重分布比较规整，容易量化
- 激活值包含巨大的异常值（outliers），某些channel的值可能是其他的100倍
- 直接对激活值做INT8量化 → 精度灾难

### 核心方法：把难度从激活搬到权重

**数学等价变换：**

原始线性层计算：`Y = X · W`

SmoothQuant 引入对角缩放矩阵 **S**：

```
Y = X · W = (X · diag(s)^{-1}) · (diag(s) · W) = X_smooth · W_smooth
```

其中：
- `X_smooth = X · diag(s)^{-1}` → 激活值被**缩小**（平滑掉异常值）
- `W_smooth = diag(s) · W` → 权重被相应**放大**（承接难度）
- 因为 `diag(s)^{-1} · diag(s) = I`，变换是**严格数学等价的**

**缩放因子 s_j 的计算：**

```
s_j = (max(|X_j|))^α / (max(|W_j|))^{1-α}
```

- `X_j`：第j个channel的激活值
- `W_j`：第j个channel的权重
- `α`：超参数，控制难度在激活/权重间的迁移比例（通常0.5）

**直觉解释：**
- 想象跷跷板：激活太"高"（异常值大），权重太"低"（分布平滑）
- SmoothQuant 就是给跷跷板找平衡点：把激活压下来，把权重抬上去
- 压和抬的幅度由 s 控制，而且严格等价（不丢信息）
- 平衡后，激活和权重**都**变得容易量化 → 实现 W8A8（权重8-bit，激活8-bit）

**效果：**
- 实现 W8A8 量化（之前只能做到 W8A16）
- 训练后量化(PTQ)，不需要重训练
- 通用性强，适用于各种LLM架构

---

## 五、四篇论文的关系脉络（公众号可用）

```
时间线：
2022.08 → LLM.int8()：发现问题（异常值），提出混合精度方案
2022.11 → SmoothQuant：换个思路，把异常值从激活搬到权重
2023.05 → QLoRA/NF4：4-bit量化 + 微调，分布感知量化
2023~25 → GGUF K-quant：工程落地，混合精度思想进入本地部署
2025    → NVFP4(Blackwell)：硬件原生支持4-bit浮点

核心叙事线：
量化从"简单截断"进化到"理解数据分布" → 从"学术方法"落地到"消费级硬件"
```

---

## Sources

- [LLM.int8() - arXiv 2208.07339](https://arxiv.org/abs/2208.07339)
- [LLM.int8() and Emergent Features - Tim Dettmers Blog](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)
- [LLM.int8() - NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf)
- [SmoothQuant - arXiv 2211.10438](https://arxiv.org/abs/2211.10438)
- [SmoothQuant - MIT-IBM Watson AI Lab](https://mitibmwatsonailab.mit.edu/research/blog/smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models/)
- [SmoothQuant - GitHub (MIT Han Lab)](https://github.com/mit-han-lab/smoothquant)
- [QLoRA - arXiv 2305.14314](https://arxiv.org/abs/2305.14314)
- [QLoRA - NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
- [GGUF File Format - llama.cpp](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format)
- [llama.cpp - Wikipedia](https://en.wikipedia.org/wiki/Llama.cpp)
- [Demystifying LLM Quantization Suffixes - Medium](https://medium.com/@paul.ilvez/demystifying-llm-quantization-suffixes-what-q4-k-m-q8-0-and-q6-k-really-mean-0ec2770f17d3)
- [GGUF Quantization Explained - IoTbyHVM](https://iotbyhvm.ooo/gguf-quantization-explained-what-q4_k_m-q5_k_s-and-q8_0-really-mean/)
- [Introducing NVFP4 - NVIDIA Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [FP4 Quantization on Blackwell GPUs - Spheron](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/)
- [NF4 Quantization - Emergent Mind](https://www.emergentmind.com/topics/4-bit-normalfloat-nf4-quantization)
- [FP32, FP16, BF16 & INT8 for AI - DatabaseMart](https://www.databasemart.com/blog/fp32-fp16-bf16-int8)
- [LLM Quantization: BF16 vs FP8 vs INT4](https://research.aimultiple.com/llm-quantization/)
