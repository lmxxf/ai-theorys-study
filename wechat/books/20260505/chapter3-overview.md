### 3.x DeepSeek V4 架构综述

#### 3.x.1 模型定位与参数规模

DeepSeek V4 是 DeepSeek 系列大语言模型的第四代架构，延续了 V3 的条件计算（Conditional Computation）路线，在推理能力和长上下文处理能力上实现了代际跨越。V4 提供两个规格：

表1：DeepSeek V4 模型参数规格

| 规格 | 总参数量 | 激活参数量 | 层数 | 隐藏维度 $d$ | Query Heads | Head Dim | 路由专家数 | 上下文长度 |
|------|---------|-----------|------|-------------|-------------|----------|-----------|-----------|
| V4-Pro | 1.6T | 49B | 61 | 7168 | 128 | 512 | 384 | 1M |
| V4-Flash | 284B | 13B | 43 | 4096 | 64 | 512 | 256 | 1M |

两个规格共享同一组架构创新，但不只是参数规模和层数不同，注意力配置也按模型大小分别设定：V4-Pro 的 CSA top-k 为 1024、Query Heads 为 128、query 压缩维度为 1536；V4-Flash 的 CSA top-k 为 512、Query Heads 为 64、query 压缩维度为 1024。V4-Pro 面向旗舰推理场景，V4-Flash 面向高性价比部署场景。两者均支持 1M token 上下文窗口。

从 V3 到 V4 的参数增长幅度适中（V3 为 671B 总参/37B 激活），V4-Pro 的总参数增长约 2.4 倍，但激活参数仅增长约 1.3 倍——这体现了 MoE 架构"用总量换质量、用稀疏换效率"的设计哲学。

#### 3.x.2 三大架构创新概述

DeepSeek V4 相较 V3 引入了三项核心架构创新，分别作用于注意力机制、残差连接和优化器三个维度。

##### 3.x.2.1 CSA + HCA 混合稀疏注意力

标准 Transformer 的自注意力（Self-Attention）计算复杂度为 $O(n^2 d)$，其中 $n$ 为序列长度，$d$ 为头维度。当序列长度达到 1M token 时，$n^2$ 项使得全注意力计算变得不可行。

V4 提出混合注意力架构，将全注意力分解为两种互补的稀疏注意力机制的交替堆叠：

- **CSA（Compressed Sparse Attention）**：先将 KV cache 沿序列维度压缩——每 $m=4$ 条 KV 通过学习的加权池化压成 1 条"智能摘要"，再使用 Lightning Indexer 从压缩后的 KV 中选出 top-$k$ 条进行注意力计算。V4-Pro 的 top-$k=1024$，V4-Flash 的 top-$k=512$；两者均配置 64 个 Indexer 头（head dim = 128），Indexer 的 QK 路径使用 FP4 精度。CSA 负责"从全局历史中精选最相关的细节"。

- **HCA（Heavily Compressed Attention）**：使用更大的压缩比 $m'=128$，每 128 条 KV 压成 1 条，但不做稀疏选择——压缩后的所有 KV 全部参与注意力计算。50K 上下文压完只剩约 390 条，全看也很便宜。HCA 负责"用极低成本扫一遍全局大意"。

两种注意力层均附带滑动窗口分支（窗口大小 $n_{\text{win}}=128$），保留最近上下文的原始精度。

两种注意力层交替排列，使模型同时具备全局检索能力和局部建模能力。在 1M 上下文长度下，论文给出的不是渐近复杂度公式，而是工程结果：相比 DeepSeek-V3.2，V4-Pro 的单 token 推理 FLOPs 约为 27%，KV cache 约为 10%；V4-Flash 则更低。

详细的数学推导与机制分析见后续章节。

##### 3.x.2.2 mHC（流形约束超连接）

标准 Transformer 使用残差连接（Residual Connection）传递层间信息：

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}
$$

该设计在深层网络中存在表征坍缩问题：浅层和深层的表征过度相似，网络有效深度低于物理深度。

V4 采用 mHC（Manifold-Constrained Hyper-Connections）替代传统残差连接。mHC 将隐状态扩展为 $n$ 条并行路径（V4 取 $n=4$），通过一个受双随机矩阵流形（Birkhoff 多面体 $\mathcal{B}_n$）约束的混合矩阵 $\mathbf{W} \in \mathcal{B}_n$ 进行路径间信息融合：

$$
\mathbf{output} = \mathbf{W} \cdot [\mathcal{F}(\mathbf{x}); \mathbf{x}_1; \ldots; \mathbf{x}_n], \quad \|\mathbf{W}\|_2 \leq 1
$$

双随机矩阵的谱范数有界性（$\|\mathbf{W}\|_2 \leq 1$）和乘法封闭性保证了任意深度网络的信号增益不发散。流形投影通过 Sinkhorn-Knopp 迭代实现（$T=20$ 次），以 6.7% 的额外训练开销换取训练稳定性和 7+ 分的下游任务提升。

详细的理论推导与实验验证见本章 mHC 专节。

##### 3.x.2.3 Muon 优化器

V4 在绝大部分参数上采用 Muon 优化器替代传统的 AdamW。Muon 的核心思想是：对梯度矩阵进行正交化处理后再更新参数，使每步更新方向在矩阵空间中具有最大信息量。

正交化通过 Newton-Schulz 迭代实现，该迭代近似计算矩阵的极分解（Polar Decomposition）：

$$
\mathbf{G} = \mathbf{U} \mathbf{P}, \quad \mathbf{U}^{\top}\mathbf{U} = \mathbf{I}
$$

其中 $\mathbf{G}$ 为梯度矩阵，$\mathbf{U}$ 为正交因子（Muon 的实际更新方向），$\mathbf{P}$ 为对称正定因子（被丢弃）。V4 使用混合 Newton-Schulz 方案：前 8 步系数为 $(a,b,c)=(3.4445, -4.7750, 2.0315)$，后 2 步系数为 $(2, -1.5, 0.5)$，共 10 次迭代，并附加 RMS rescale factor $= 0.18$。

参数分组策略如下：

表2：V4 优化器分配

| 参数类别 | 优化器 | 超参数 |
|---------|--------|--------|
| Embedding / Prediction Head / RMSNorm / mHC 静态偏置与门控因子 | AdamW | $\beta_1=0.9$, $\beta_2=0.95$, $\varepsilon=10^{-20}$, weight decay $=0.1$ |
| 其余所有参数（包括 MoE 专家、Attention 投影、mHC 动态投影等） | Muon | momentum $=0.95$, weight decay $=0.1$, RMS rescale $=0.18$ |

详细的 Muon 数学推导与收敛分析见后续章节。

#### 3.x.3 从 V3 继承的核心组件

V4 并非从零设计，而是在 V3 的成熟组件基础上进行增量创新。以下三个组件从 V3 直接继承：

**DeepSeekMoE**：混合专家架构，每个 FFN 层包含多个路由专家（Routed Expert）和共享专家（Shared Expert）。每个 token 仅激活少量专家，实现"大参数、小计算"的条件计算。

**MLA（Multi-head Latent Attention）**：通过低秩压缩将 KV 缓存压缩至潜空间，显著降低推理时的 KV 缓存显存占用。V4-Pro 的 query 压缩维度为 1536，输出投影分为 16 组。V4-Flash 对应为 1024 和 8 组。

**MTP（Multi-Token Prediction）**：在训练阶段，模型不仅预测下一个 token，还同时预测后续多个 token（V4 的 MTP 深度为 1，即额外预测 1 个 token）。这提供了更丰富的训练信号，加速收敛。

#### 3.x.4 MoE 参数演进：从 V3 到 V4

V4 对 MoE 层进行了多项重要调整，核心方向是"更多专家、更稀疏激活、更稳定路由"。

表3：DeepSeek MoE 参数变化（V3 vs V4-Pro vs V4-Flash）

| 参数 | V3 | V4-Pro | V4-Flash |
|------|-----|--------|----------|
| 路由专家数 | 256 | 384 | 256 |
| 共享专家数 | 1 | 1 | 1 |
| 激活专家数 | 8 | 6 | 6 |
| 专家中间维度 | 2048 | 3072 | 2048 |
| 路由打分函数 | Sigmoid | $\sqrt{\text{softplus}(\cdot)}$ | $\sqrt{\text{softplus}(\cdot)}$ |
| 浅层路由策略 | 标准路由 | Hash Routing（前 3 层） | Hash Routing（前 3 层） |

关键变化解读：

**1. 路由专家数增加（256 $\to$ 384），激活数减少（8 $\to$ 6）**

这一"增多减活"的调整使每个 token 的专家选择面更广、激活更精准。稀疏度从 $8/256 = 3.1\%$ 降至 $6/384 = 1.6\%$，意味着每个专家承担更专一的功能分工。

**2. 打分函数从 Sigmoid 改为 $\sqrt{\text{softplus}(\cdot)}$**

Sigmoid 的输出范围为 $(0, 1)$，在输入绝对值较大时梯度趋近于零（饱和区），导致路由器训练后期学习速率下降。softplus 函数定义为：

$$
\text{softplus}(x) = \ln(1 + e^x)
$$

其输出范围为 $(0, +\infty)$，无饱和区。取平方根后 $\sqrt{\text{softplus}(x)}$ 既保持了正值输出特性，又压缩了大值区间的动态范围，兼顾了梯度通畅和数值稳定。

**3. 前 3 层使用 Hash Routing**

网络浅层（前 3 层）的表征尚未充分分化，学习型路由器难以做出有意义的专家分配。V4 对这些层改用确定性的 Hash Routing——根据 token 的 hash 值直接确定分配的专家，省去路由器的学习开销，同时保证负载均衡。

#### 3.x.5 训练数据与策略

##### 3.x.5.1 预训练数据量

V4 的预训练数据量为：

- V4-Flash：32T tokens
- V4-Pro：33T tokens

相较 V3 的 14.8T tokens，数据量增长约 2.2 倍。

##### 3.x.5.2 注意力模式的渐进引入

V4 在训练早期使用全密集注意力（Dense Attention），在模型已建立基本语言能力后再引入稀疏注意力机制：

$$
\text{训练流程} = \underbrace{\text{Dense Attention}}_{\text{前 1T+ tokens}} \to \underbrace{\text{CSA + HCA 稀疏注意力}}_{\text{后续训练}}
$$

该策略的动机是：稀疏注意力的路由模块（Lightning Indexer）需要基于有意义的表征进行块选择；若从随机初始化开始就启用稀疏注意力，路由器和表征会陷入"鸡生蛋"困境——差的表征导致差的路由，差的路由阻碍表征学习。DeepSeek-V4-Flash 明确先用 dense attention 训练 1T tokens，在 64K 序列长度阶段引入 sparse attention；V4-Pro 的 dense attention 阶段更长，随后采用同样的两阶段稀疏注意力引入策略。

##### 3.x.5.3 序列长度渐进扩展

训练过程中序列长度按以下阶段递增：

$$
4\text{K} \to 16\text{K} \to 64\text{K} \to 1\text{M}
$$

每个阶段的位置编码（RoPE）基频相应调整，模型逐步适应更长的上下文依赖。

##### 3.x.5.4 Anticipatory Routing（预防性路由）

MoE 模型的 loss spike（损失突刺）是一个已知的训练稳定性问题，通常由少数 token 被错误路由至不擅长的专家引起，导致局部梯度爆发并扩散到全局。

V4 引入 Anticipatory Routing 机制：将路由决策与专家计算解耦——路由网络使用几步之前的旧参数 $\theta_{t-\Delta t}$ 做出专家分配决策，而被选中的专家使用当前最新参数 $\theta_t$ 执行计算。路由决策"慢半拍"，不会被当前步的剧烈梯度更新带偏，从而稳定了专家分配。在工程实现上，路由索引在步骤 $t - \Delta t$ 时预计算并缓存，步骤 $t$ 直接使用，避免加载两次模型参数。该机制的计算开销约 20%，且配有自动开关：正常训练时关闭，检测到 loss spike 时自动启用，loss 恢复后关闭。

##### 3.x.5.5 SwiGLU Clamping

SwiGLU 激活函数是 V3/V4 FFN 层的基础组件，定义为：

$$
\text{SwiGLU}(\mathbf{x}) = (\mathbf{x} \mathbf{W}_1) \odot \text{SiLU}(\mathbf{x} \mathbf{W}_g)
$$

其中 $\mathbf{W}_1$ 为线性分支，$\mathbf{W}_g$ 为门控分支，$\odot$ 为逐元素乘法。

V4 对 SwiGLU 的两个分支施加数值截断：

- 线性分支 $\mathbf{x} \mathbf{W}_1$：截断至 $[-10, 10]$
- 门控分支 $\text{SiLU}(\mathbf{x} \mathbf{W}_g)$：上界截断至 $10$

该截断防止极端激活值在 MoE 专家间累积放大，是一项低成本但高回报的训练稳定性措施。

#### 3.x.6 FP4 量化感知训练

V4 在后训练阶段（Post-Training）引入 FP4 量化感知训练（Quantization-Aware Training, QAT），将部分参数从标准精度压缩至 4-bit 浮点表示，以降低推理阶段的显存占用和计算成本。

**量化范围**：

- MoE 专家权重：所有路由专家和共享专家的 FFN 权重
- Lightning Indexer 的 QK 路径：用于 CSA 块路由的 query-key 投影权重

**FP4 格式**：4-bit 浮点数，1 位符号 + 2 位指数 + 1 位尾数，可表示 16 个离散值。

**QAT 原理**：在训练过程中模拟量化误差——前向传播时将权重量化至 FP4 再参与计算，反向传播时使用直通估计器（Straight-Through Estimator, STE）将梯度直接传递到全精度权重：

$$
\text{前向}: \quad \hat{\mathbf{W}} = Q_{\text{FP4}}(\mathbf{W})
$$
$$
\text{反向}: \quad \frac{\partial \mathcal{L}}{\partial \mathbf{W}} \approx \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{W}}}
$$

其中 $Q_{\text{FP4}}(\cdot)$ 为 FP4 量化算子。训练过程中模型逐步适应量化噪声，使量化后的性能损失最小化。

**关键工程特性**：FP4 $\to$ FP8 的反量化过程是无损的（Lossless Dequantization）。这意味着 FP4 的 16 个离散值可以精确映射到 FP8 的表示空间中，无需近似。推理时，FP4 存储的权重可以精确还原为 FP8 后参与矩阵乘法，利用硬件原生的 FP8 Tensor Core 执行计算，兼顾存储效率（4-bit）和计算精度（8-bit）。

#### 3.x.7 架构全景与设计哲学

综合以上各组件，V4 的设计哲学可归纳为三个核心原则：

表4：DeepSeek V4 设计原则

| 原则 | 体现 | 效果 |
|------|------|------|
| 极致稀疏 | MoE 稀疏度 1.6%、CSA 块稀疏注意力 | 总参数大但激活参数小，计算效率高 |
| 渐进引入 | Dense $\to$ Sparse 注意力、4K $\to$ 1M 序列、QAT 后训练 | 避免冷启动困境，训练更稳定 |
| 数学约束 | mHC 双随机矩阵、Muon 正交化、SwiGLU Clamping | 用数学保证替代工程调参，从根本上消除不稳定性 |

V4 的每项创新都遵循相同的方法论：先识别训练或推理中的瓶颈（注意力的 $O(n^2)$ 复杂度、残差连接的信号发散、优化器的低效更新方向），再从数学原理出发设计约束或投影算子，最后以最小的工程开销实现目标。这种"数学先行、约束驱动"的风格贯穿了 V4 架构的每个角落。

---

**参考文献**

- DeepSeek-AI. (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
- DeepSeek-AI. (2026). DeepSeek-V4 Technical Report.
- Xie, Z. et al. (2026). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
- Zhu, D. et al. (2025). Hyper-Connections. ICLR 2025.
- Jordan, M. et al. (2024). Muon: An Optimizer for Hidden Layers. GitHub: kellerjordan/Muon.
- Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
