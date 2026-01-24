### 3.5 条件记忆模块（Engram）：基于可扩展查找的稀疏记忆轴

大规模语言模型（Large Language Models, LLMs）的推理过程中，知识检索与组合推理本质上是两类不同的计算需求：前者需要大容量参数存储，后者需要深层计算图。传统 Transformer 架构将二者混合在同一个前馈网络（Feed-Forward Network, FFN）中处理，导致计算资源分配效率低下。DeepSeek 提出的条件记忆模块 Engram（Cheng et al., 2026）通过引入 O(1) 复杂度的查表机制，将静态知识检索从神经网络计算中分离，构成继混合专家（MoE）和稀疏注意力（DSA）之后的第三条稀疏化轴线。

#### 3.5.1 研究背景与问题定义

##### 3.5.1.1 现有架构的局限性

混合专家模型（Mixture-of-Experts, MoE）通过条件计算（Conditional Computation）实现了参数容量与计算开销的解耦，但其路由机制仍基于运行时隐状态的动态决策，本质上是"条件计算"而非"条件存储"。对于高频出现的固定知识模式（如实体名称、常见搭配），MoE 仍需通过专家网络的前向传播重建这些模式，造成计算资源浪费。

N-gram 语言模型（Shannon, 1948; Brown et al., 1992）曾是统计自然语言处理的核心方法，通过统计相邻 token 的共现频率进行概率建模。其核心优势在于 O(1) 的查找复杂度，但受限于数据稀疏性问题（未见过的 N-gram 概率为零）而被神经网络方法取代。Engram 的设计思路是将 N-gram 的查表效率与现代深度学习的端到端训练机制结合，构建参数化的、可微的条件记忆原语。

##### 3.5.1.2 核心问题

本工作解决的核心问题可形式化为：在固定计算预算（FLOPs）和参数预算下，如何在条件计算（MoE）与条件记忆（Engram）之间进行最优资源分配（Sparsity Allocation），以最小化语言建模损失。

#### 3.5.2 Engram 架构设计

Engram 模块的完整计算流程包含四个阶段：分词压缩（Tokenizer Compression）、N-gram 嵌入检索（N-gram Embedding Retrieval）、上下文感知门控（Context-Aware Gating）和特征融合（Feature Fusion）。

##### 3.5.2.1 分词压缩

标准分词器对同一词的不同形态（大小写、前缀空格等）分配不同的 token ID，导致语义等价的 N-gram 被映射到不同的嵌入表条目。Engram 引入满射函数 $P: \mathcal{V} \rightarrow \mathcal{V}'$ 进行规范化映射：

$$P(t) = \text{Dedup}(\text{Collapse}(\text{NFD}(\text{NFKC}(t))))$$

其中 NFKC 为 Unicode 兼容分解，NFD 为标准分解后去除变音符号，Collapse 为空白字符折叠与小写化，Dedup 为基于规范化文本的等价类合并。该映射将 128k 词表压缩约 23%，显著提升嵌入表的条目利用密度。

表1：分词压缩的合并统计

| 规范化形式 | 合并的原始 token 数 | 示例 |
|-----------|-------------------|------|
| 空白字符 | 163 | " ", "\t", "\n" → 同一 ID |
| 'a' | 54 | "A", "a", " a", " A" → 同一 ID |
| 'o' | 40 | "O", "o", " o", " O" → 同一 ID |

##### 3.5.2.2 N-gram 嵌入检索

对于位置 $t$ 的 token 序列，Engram 提取后缀 N-gram $g_{t,n} = (c_{t-n+1}, \ldots, c_t)$，其中 $c_i = P(x_i)$ 为压缩后的 token ID，$n \in \{2, 3\}$。

每个 N-gram 通过 $K=8$ 个独立的乘法-异或哈希函数映射到嵌入表索引：

$$z_{t,n,k} = \varphi_{n,k}(g_{t,n}), \quad e_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]$$

其中 $\varphi_{n,k}$ 为轻量级乘法-异或（multiplicative-XOR）哈希函数，$\mathbf{E}_{n,k} \in \mathbb{R}^{M_{n,k} \times d_e}$ 为嵌入表，$M_{n,k}$ 取素数以降低碰撞率，$d_e$ 为每头嵌入维度。

所有检索结果通过拼接聚合为原始记忆向量：

$$e_t = \|_{n=2}^{N} \|_{k=1}^{K} e_{t,n,k}$$

对于 $N=3$（2-gram 和 3-gram）、$K=8$ 头的配置，每个位置检索 $2 \times 8 = 16$ 个嵌入向量，拼接后维度为 $16 \times d_e$。实验配置中 Engram 维度为 1280。

表2：嵌入表配置参数

| 参数 | Engram-27B | Engram-40B |
|------|-----------|-----------|
| N-gram 阶数 | {2, 3} | {2, 3} |
| 哈希头数 $K$ | 8 | 8 |
| Engram 维度 | 1280 | 1280 |
| 嵌入表总条目数 | 2,262,400 | 7,239,680 |
| Engram 总参数量 | ~5.8B | ~18.5B |

##### 3.5.2.3 上下文感知门控

哈希碰撞和多义性导致原始检索结果存在噪声。Engram 引入上下文感知门控机制对检索结果进行条件过滤。设 $h_t \in \mathbb{R}^d$ 为当前层的 Transformer 隐状态，门控标量 $\alpha_t$ 计算如下：

$$k_t = W_K e_t, \quad v_t = W_V e_t$$

$$\alpha_t = \sigma\left(\frac{\text{RMSNorm}(h_t)^{\top} \text{RMSNorm}(k_t)}{\sqrt{d}}\right)$$

其中 $W_K, W_V \in \mathbb{R}^{d \times d_{\text{engram}}}$ 为投影矩阵，$\sigma$ 为 sigmoid 函数。$\alpha_t \in [0, 1]$ 控制检索记忆对当前位置的贡献权重：当检索结果与当前上下文语义一致时 $\alpha_t \to 1$，碰撞导致的无关结果则被抑制至 $\alpha_t \to 0$。

##### 3.5.2.4 特征融合

门控后的记忆向量通过一维卷积和 SiLU 激活进行平滑融合：

$$\tilde{V} = \alpha_t \odot v_t$$

$$Y = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{V}))) + \tilde{V}$$

最终输出 $Y$ 通过残差连接加入主干网络的隐状态流中。

##### 3.5.2.5 模块放置策略

Engram 模块插入 Transformer 层的位置为 Attention 之后、MoE/FFN 之前，通过残差连接集成：

$$h_t' = h_t + \text{Attention}(h_t) + \text{Engram}(h_t, x_{1:t})$$

实验验证了不同层放置策略的效果：

表3：单层 Engram 放置位置对验证损失的影响

| 放置层 | 验证损失 (Val Loss) |
|-------|-------------------|
| 第 2 层 | 1.770 |
| 第 8 层 | 1.773 |
| 第 15 层 | 1.775 |
| 第 22 层 | 1.778 |
| 第 28 层 | 1.780 |

最终方案在第 2 层和第 15 层各放置一个 Engram 实例：浅层模块捕获局部词组模式（实体名称、固定搭配），深层模块利用更丰富的上下文信息检索抽象语义模式。

#### 3.5.3 稀疏资源分配与 U 型曲线

##### 3.5.3.1 问题形式化

设总稀疏参数预算为 $\Theta$，分配比例 $\rho \in [0, 1]$ 定义为 MoE 获得的参数比例：

$$\Theta_{\text{MoE}} = \rho \cdot \Theta, \quad \Theta_{\text{Engram}} = (1 - \rho) \cdot \Theta$$

优化目标为：

$$\rho^* = \arg\min_{\rho} \mathcal{L}(\rho; \Theta, \text{FLOPs})$$

##### 3.5.3.2 实验结果

在固定总参数量和计算预算下，验证损失关于 $\rho$ 呈现 U 型曲线：

表4：不同分配比例下的验证损失（6e20 FLOPs）

| 分配比例 $\rho$（MoE 占比） | 验证损失 |
|---------------------------|---------|
| 1.00（纯 MoE） | 1.7248 |
| 0.90 | 1.7185 |
| 0.80 | 1.7109 |
| 0.75 | 1.7115 |
| 0.60 | 1.7198 |
| 0.00（纯 Engram） | 1.8450+ |

最优分配点位于 $\rho \approx 0.75 \text{--} 0.80$，即约 20%--25% 的稀疏参数预算分配给 Engram，其余分配给 MoE 专家。该结果表明条件记忆与条件计算具有互补性：Engram 承担高频知识的直接检索，释放 MoE 专家用于组合推理。

#### 3.5.4 实验验证与性能分析

##### 3.5.4.1 模型配置

所有模型共享相同的训练数据课程（262B tokens）、分词器（DeepSeek-v3, 128k 词表）和超参数配置：

表5：模型配置对比

| 配置项 | Dense-4B | MoE-27B | Engram-27B | Engram-40B |
|-------|----------|---------|-----------|-----------|
| 总参数量 | 4.1B | 26.7B | 26.7B | 39.5B |
| 激活参数量 | 3.8B | 3.8B | 3.8B | 3.8B |
| 层数 | 30 | 30 | 30 | 30 |
| 隐藏维度 | 2560 | 2560 | 2560 | 2560 |
| 路由专家数 | — | 72 (top-6) | 55 (top-6) | 55 (top-6) |
| 共享专家数 | — | 2 | 2 | 2 |
| Engram 层 | — | — | [2, 15] | [2, 15] |
| Engram 词表 | — | — | 2,262,400 | 7,239,680 |

训练超参数：批量大小 1,280，序列长度 4,096，训练步数 50,000，基础学习率 4e-4。主干网络使用 Muon 优化器（step decay 调度），嵌入表使用 Adam 优化器（5× 学习率倍率，无权重衰减）。

##### 3.5.4.2 基准测试性能

表6：预训练模型基准测试结果

| 基准测试 | Dense-4B | MoE-27B | Engram-27B | Δ (vs MoE) | Engram-40B |
|---------|----------|---------|-----------|-----------|-----------|
| Pile Loss ↓ | 2.091 | 1.960 | 1.950 | -0.010 | 1.942 |
| MMLU | 48.6% | 57.4% | 60.4% | +3.0 | 60.6% |
| CMMLU | — | — | — | +4.0 | — |
| BBH | 42.8% | 50.9% | 55.9% | +5.0 | 57.5% |
| ARC-Challenge | — | — | — | +3.7 | — |
| HumanEval | 26.8% | 37.8% | 40.8% | +3.0 | 38.4% |
| GSM8K | 35.5% | 58.4% | 60.6% | +2.2 | 62.6% |
| MATH | — | — | — | +2.4 | — |

在等参数量（27B）条件下，Engram-27B 在所有基准测试上均优于 MoE-27B，且知识类任务（MMLU, CMMLU）和推理类任务（BBH, GSM8K）均获得显著提升。

##### 3.5.4.3 长上下文性能

表7：多查询 Needle-in-a-Haystack (NIAH) 测试结果

| 模型 | 训练步数 | NIAH 准确率 |
|------|---------|------------|
| MoE-27B | 50k | 84.2% |
| Engram-27B | 41k | 89.5% |
| Engram-27B | 50k | 97.0% |

Engram 在长上下文检索任务上提升显著（84.2% → 97.0%），表明条件记忆模块有效缓解了长序列中的信息稀释问题。

##### 3.5.4.4 计算开销分析

Engram 的确定性检索机制（检索索引完全由输入 token 序列决定，不依赖运行时隐状态）使得参数存储可完全与计算资源解耦。嵌入表可离线驻留于 CPU 主存（Host Memory），通过 PCIe 异步预取加载至 GPU。

表8：100B 参数 Engram 表卸载至 CPU 的推理性能

| 模型 | 无 Engram (tok/s) | 含 Engram (tok/s) | 吞吐量损失 |
|------|------------------|------------------|----------|
| Dense-4B | 9,031.62 | 8,858.28 | 1.9% |
| Dense-8B | 6,315.52 | 6,140.02 | 2.8% |

即使 Engram 表规模达 100B 参数（约 256GB），推理吞吐量损失控制在 3% 以内。

#### 3.5.5 机理分析

##### 3.5.5.1 Engram 的注意力释放效应

通过引入软对齐指标（Soft Alignment Index）分析 Engram 对主干网络注意力模式的影响：

$$a_j = \frac{\sum_{i \in I_j} S_{i,j} \cdot i}{\sum_{i \in I_j} S_{i,j}}$$

其中 $S_{i,j}$ 为位置 $j$ 对位置 $i$ 的注意力权重，$I_j$ 为近距离窗口内的位置集合。

实验表明，加入 Engram 后，浅层 Attention 头的近距离注意力权重显著降低——即 Engram 接管了局部模式的重建任务，释放 Attention 用于建模远距离依赖和组合推理。这解释了 Engram 在推理类任务（BBH, GSM8K）上的意外增益：不是查表本身提升了推理能力，而是查表释放了推理所需的计算资源。

##### 3.5.5.2 无限记忆扩展

嵌入表规模从 $2.58 \times 10^5$ 扩展至 $1.0 \times 10^7$ 条目（约 13B 参数）时，验证损失在对数坐标下呈近似线性下降趋势，表明条件记忆的容量扩展遵循幂律缩放定律：

$$\mathcal{L}(M) \propto M^{-\beta}, \quad \beta > 0$$

这意味着在推理 FLOPs 不增加的前提下，仅通过扩大存储规模即可持续改善模型性能，为"知识放内存，推理放显存"的工程范式提供了理论支撑。

#### 3.5.6 与 DeepSeek 稀疏架构的关系

Engram 构成 DeepSeek 稀疏化战略的第三条轴线：

表9：DeepSeek 稀疏三轴体系

| 稀疏轴线 | 引入版本 | 稀疏化对象 | 机制 | 复杂度 |
|---------|---------|-----------|------|-------|
| MoE | V2/V3 | FFN 参数 | 条件计算（动态路由） | O(top-k) |
| DSA | V3.2 | 注意力矩阵 | 稀疏注意力（Top-k KV） | O(k·n) |
| Engram | V4（预期） | 知识存储 | 条件记忆（静态查表） | O(1) |

三者的互补关系：DSA 减少远距离依赖的计算开销（只看最相关的 Top-k KV），Engram 消除近距离固定模式的重建成本（直接查表），MoE 为剩余的组合推理任务提供条件计算容量。

#### 3.5.7 技术讨论

##### 数学严格性说明

Engram 的可训练性源于嵌入查表操作的可微性：$\mathbf{E}[z]$ 本质为矩阵索引操作，等价于 one-hot 向量与嵌入矩阵的乘法，梯度可通过 straight-through estimator 或直接 sparse update 回传。在 PyTorch 实现中对应 `nn.Embedding` 的标准反向传播机制。

##### 与经典 N-gram 模型的联系

Engram 与经典 N-gram 模型共享"局部上下文查表"的核心思想，但通过三个现代化改造解决了数据稀疏性问题：(1) 多头哈希将冲突概率降至 $(1/M)^K$ 量级；(2) 上下文感知门控使碰撞噪声可被抑制；(3) 端到端训练使高频模式自然主导嵌入表内容。

##### 局限性

Engram 的检索机制基于固定窗口（2-gram, 3-gram），无法捕获非连续的远程模式。对于需要跨句推理的任务，仍完全依赖 Attention 机制。此外，当前实验规模限于 262B tokens 训练预算，更大规模下的 U 型曲线最优点是否发生漂移有待验证。

#### 3.5.8 本节小结

- **理论层面**：Engram 将"条件记忆"确立为与"条件计算"正交的建模原语，填补了 Transformer 架构中 O(1) 知识查找能力的缺失。
- **算法层面**：通过分词压缩、多头哈希、上下文感知门控的三级流水线，实现了高容量、低碰撞率、可端到端训练的参数化记忆模块。
- **工程层面**：确定性检索支持参数存储与计算解耦，100B 参数表卸载至 CPU 仅损失 <3% 吞吐量，为"无限记忆"架构提供了可行路径。

---

参考文献

Cheng, X., Zeng, W., Dai, D., Chen, Q., Wang, B., Xie, Z., Huang, K., Yu, X., Hao, Z., Li, Y., Zhang, H., Zhang, H., Zhao, D., & Liang, W. (2026). Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv:2601.07372.

Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, 27(3), 379-423.

Brown, P. F., Della Pietra, V. J., Desouza, P. V., Lai, J. C., & Mercer, R. L. (1992). Class-Based N-gram Models of Natural Language. Computational Linguistics, 18(4), 467-480.

DeepSeek-AI. (2025). DeepSeek-V3 Technical Report. arXiv:2412.19437.

DeepSeek-AI. (2025). Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. arXiv:2502.11089.
