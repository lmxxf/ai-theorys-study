### 3.x mHC：基于流形约束的超连接架构优化

#### 3.x.1 研究背景与问题定义

##### 3.x.1.1 残差连接的历史与局限

残差连接（Residual Connection）作为深度神经网络训练的核心技术，自 He 等人（2015）在 ResNet 中提出以来，已成为现代深度学习架构的基础组件。其核心机制可形式化表示为：

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}
$$

其中 $\mathcal{F}(\cdot)$ 表示残差函数，$\mathbf{x}$ 为层输入。该设计通过引入恒等映射（Identity Mapping）有效缓解了深层网络训练中的梯度消失问题，使得网络深度的扩展成为可能。

在 Transformer 架构中，残差连接的具体应用形成了两种范式：

**Post-Norm**（原始 Transformer, Vaswani et al., 2017）：
$$
\mathbf{y} = \text{LayerNorm}(\mathcal{F}(\mathbf{x}) + \mathbf{x})
$$

**Pre-Norm**（GPT-2 及后续主流架构）：
$$
\mathbf{y} = \mathcal{F}(\text{LayerNorm}(\mathbf{x})) + \mathbf{x}
$$

Pre-Norm 因其训练稳定性优势成为当前大规模语言模型的事实标准，但存在一个固有缺陷：随着网络深度增加，浅层与深层表征的相似度趋近于 1（即表征坍缩，Representation Collapse），导致深层网络的有效深度远低于物理深度。

该现象可直观理解为：Pre-Norm 中恒等映射路径过于强势，$\mathcal{F}(\mathbf{x})$ 对残差流的贡献被逐层稀释，网络深层退化为近似恒等变换。

##### 3.x.1.2 超连接（Hyper-Connections）架构

2024 年 9 月，字节跳动 Seed-Foundation-Model 团队提出超连接（Hyper-Connections, HC）架构（Zhu et al., 2024, arXiv:2409.19606），该工作被 ICLR 2025 收录。HC 通过引入可学习的连接强度系数矩阵，将传统残差连接扩展为多路径并行结构，同时解决梯度消失与表征坍缩的跷跷板效应（Seesaw Effect）。

**HC 矩阵结构**

HC 的连接权重被组织为一个 $(n+1) \times (n+1)$ 的结构化矩阵 $\mathcal{HC}$：

$$
\mathcal{HC} = \begin{pmatrix} 0_{1\times 1} & \mathbf{B}^{\top} \\ \mathbf{A}_m & \mathbf{A}_r \end{pmatrix}
$$

其中：
- $\mathbf{B} \in \mathbb{R}^{1 \times n}$：输出权重向量，控制 $n$ 条路径对最终输出的贡献
- $\mathbf{A}_m \in \mathbb{R}^{n \times 1}$：输入聚合向量，控制层输入如何分配到各路径
- $\mathbf{A}_r \in \mathbb{R}^{n \times n}$：残差连接矩阵，控制路径间的信息交换

**前向传播公式**

对于扩展率 $n$，HC 的前向传播可等价展开为两步操作：

*宽度连接*（Width-Connection，路径间信息交换）：
$$
\mathbf{H}' = \mathbf{A}_r^{\top} \cdot \mathbf{H}
$$

*深度连接*（Depth-Connection，层输出融合）：
$$
\hat{\mathbf{H}} = \mathbf{B}^{\top} \cdot \mathcal{T}(\mathbf{H}^{\top} \cdot \mathbf{A}_m)^{\top} + \mathbf{H}'
$$

其中 $\mathbf{H} \in \mathbb{R}^{n \times d}$ 为 $n$ 条路径的隐状态矩阵，$\mathcal{T}(\cdot)$ 为 Transformer 子层（Attention 或 FFN），$d$ 为隐藏维度。

直观理解：HC 将隐状态从 $d$ 维扩展为 $n \times d$ 维（$n$ 条并行路径），每条路径携带不同的残差混合比例。最终输出为 $n$ 条路径的加权求和。

**统一视角：Pre-Norm 作为 HC 的特例**

当 $n=1$ 时，HC 矩阵退化为：

$$
\mathcal{HC}_{\text{Pre-Norm}} = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}
$$

此时前向传播等价于标准 Pre-Norm 残差连接。因此，HC 可视为 Pre-Norm 的广义扩展。

类似地，Post-Norm 也可表示为 HC 的特殊配置，其权重依赖于输入/输出方差比。

**动态超连接（Dynamic Hyper-Connections, DHC）**

HC 进一步支持输入依赖的动态权重：

$$
\mathcal{B}(\mathbf{H}) = s_\beta \circ \tanh(\bar{\mathbf{H}} \mathbf{W}_\beta)^{\top} + \mathbf{B}
$$
$$
\mathcal{A}_m(\mathbf{H}) = s_\alpha \circ \tanh(\bar{\mathbf{H}} \mathbf{W}_m) + \mathbf{A}_m
$$
$$
\mathcal{A}_r(\mathbf{H}) = s_\alpha \circ \tanh(\bar{\mathbf{H}} \mathbf{W}_r) + \mathbf{A}_r
$$

其中 $\bar{\mathbf{H}}$ 为归一化后的隐状态，$\mathbf{W}_\beta, \mathbf{W}_m, \mathbf{W}_r$ 为可学习的投影矩阵，$s_\beta, s_\alpha$ 为缩放因子。$\tanh$ 激活限制动态偏移量在 $[-1, 1]$ 范围内。

**HC 初始化策略**

为确保训练初期行为与 Pre-Norm 一致：

- 动态参数 $\mathbf{W}_\beta, \mathbf{W}_m, \mathbf{W}_r$ 初始化为零
- 静态矩阵按层索引 $k$ 设置：$\mathbf{B}^k = \mathbf{1}_{1 \times n}$，$\mathbf{A}_m^k = \mathbf{e}_{k \bmod n}$，$\mathbf{A}_r^k = \mathbf{I}_{n \times n}$
- 输出权重缩放 $\sqrt{n}$ 以维持方差一致性

**HC 扩展率实验**

字节跳动团队在 OLMo-1B（500B tokens）上验证了不同扩展率的效果：

表1：不同扩展率下的 HC 性能（OLMo-1B, 500B tokens）

| 扩展率 $n$ | V2 验证损失 | V3 验证损失 | 下游任务平均准确率 |
|-----------|------------|------------|-----------------|
| 1（Pre-Norm 等价） | 2.819 | 2.556 | 62.3% |
| 2 | 2.802 | 2.534 | 63.0% |
| 4 | 2.781 | 2.515 | 63.8% |
| 8 | 2.778 | 2.516 | 62.8% |

最优扩展率为 $n=4$：$n=8$ 虽然验证损失略低，但下游任务准确率反而下降，表明过度扩展引入了冗余参数。

表2：静态 HC vs 动态 HC（OLMo-1B, $n=4$, 500B tokens）

| 变体 | V2 验证损失 | V3 验证损失 | 备注 |
|------|------------|------------|------|
| SHC×4（静态） | 2.791 | — | 固定权重 |
| DHC×4（动态） | 2.781 | 2.515 | 输入依赖权重 |
| DHC×4（无 tanh） | 2.787 | — | 动态偏移无界 |

动态版本（DHC）在所有配置下优于静态版本（SHC），但去除 $\tanh$ 约束后性能下降，暗示无界动态权重存在稳定性风险——这预示了 mHC 要解决的核心问题。

**HC 在大规模模型上的验证**

表3：HC 在 7B 参数规模的性能（OLMo-7B, DHC×4）

| 指标 | 基线 | DHC×4 | 改进 |
|------|------|-------|------|
| V2 验证损失 | — | — | -0.022 |
| 下游任务平均 | 70.1% | 71.0% | +0.9% |
| 训练 400B tokens 后 | — | — | 改进持续，不衰减 |

表4：HC 在 MoE 模型的性能（OLMoE-1B-7B, DHC×4）

| 指标 | 基线 | DHC×4 | 改进 |
|------|------|-------|------|
| 收敛速度 | 1.0× | 1.8× | 快 80% |
| ARC-Challenge | 41.8% | 47.8% | +6.0% |
| MMLU Var | — | — | +1.2% |

HC 在 MoE 架构上的收敛加速效果尤为显著（1.8×），表明多路径残差连接与条件计算具有协同效应。

##### 3.x.1.3 HC 的根本性工程缺陷

HC 架构的核心问题在于：**无约束可学习矩阵导致的信号增益发散**。

设 $L$ 层网络的复合增益矩阵为：

$$
\mathbf{G}_L = \prod_{\ell=1}^{L} \mathcal{HC}_\ell
$$

对于无约束矩阵，$\|\mathcal{HC}_\ell\|_2$ 可能大于 1。即使单层增益仅为 1.05（看似无害），$L=60$ 层累积后：

$$
\|\mathbf{G}_{60}\|_2 \leq 1.05^{60} \approx 18.7
$$

实际训练中，由于梯度反馈导致权重持续偏移，单层增益可远超 1.05，最终复合增益在 27B 模型上观测到 $10^3 \sim 10^5$ 量级的爆发，表现为：

1. 训练约 12k 步后出现突发性损失飙升（Loss Surge）
2. 梯度范数剧烈振荡，与损失飙升高度相关
3. 训练崩溃不可恢复

该问题本质在于：HC 引入多路径后破坏了残差连接的恒等映射性质——当 $\mathcal{F}(\mathbf{x}) \to 0$ 时，输出不再等价于输入，而是经历了不可控的线性变换 $\mathbf{A}_r$。

#### 3.x.2 mHC 架构设计

2025 年 12 月 31 日，DeepSeek 发布技术报告《mHC: Manifold-Constrained Hyper-Connections》（Xie et al., 2025, arXiv:2512.24880），提出流形约束超连接（Manifold-Constrained Hyper-Connections, mHC）架构。该工作由 19 人团队完成，通讯作者为 DeepSeek 创始人梁文峰。

mHC 的核心创新在于：**通过将残差混合矩阵投影至双随机矩阵流形（Birkhoff 多面体），恢复恒等映射性质并保证信号传播的稳定性**。

##### 3.x.2.1 双随机矩阵约束

双随机矩阵（Doubly Stochastic Matrix）定义为满足以下条件的非负方阵 $\mathbf{P} \in \mathbb{R}^{n \times n}$：

$$
\sum_{j=1}^{n} P_{ij} = 1, \quad \forall i \in \{1, \ldots, n\}
$$

$$
\sum_{i=1}^{n} P_{ij} = 1, \quad \forall j \in \{1, \ldots, n\}
$$

$$
P_{ij} \geq 0, \quad \forall i, j
$$

所有 $n \times n$ 双随机矩阵构成的集合称为 Birkhoff 多面体（Birkhoff Polytope），记作 $\mathcal{B}_n$。

**Birkhoff-von Neumann 定理**：$\mathcal{B}_n$ 的顶点集恰好是所有 $n \times n$ 置换矩阵的集合。即任意双随机矩阵可表示为置换矩阵的凸组合。

$\mathcal{B}_n$ 具有以下保证信号传播稳定性的关键性质：

**性质 1（乘法封闭性）**：若 $\mathbf{P}, \mathbf{Q} \in \mathcal{B}_n$，则 $\mathbf{P} \cdot \mathbf{Q} \in \mathcal{B}_n$。

*推论*：$L$ 层 mHC 的复合增益矩阵 $\mathbf{G}_L = \prod_{\ell=1}^L \mathbf{P}_\ell$ 仍为双随机矩阵，信号增益不会随深度累积。

**性质 2（谱范数有界）**：对于任意 $\mathbf{P} \in \mathcal{B}_n$，有 $\|\mathbf{P}\|_2 \leq 1$。

*含义*：每层变换对信号的放大率不超过 1，从根本上杜绝信号爆炸。

**性质 3（恒等映射保持）**：单位矩阵 $\mathbf{I} \in \mathcal{B}_n$，且为 Birkhoff 多面体的相对内点（Relative Interior Point）。

*含义*：在训练初期（权重接近初始化），mHC 自然退化为恒等映射，与标准残差连接行为一致。

**性质 4（信息守恒）**：双随机矩阵的行和与列和均为 1，保证信号在路径间的重新分配不增加也不减少总能量。

上述性质的组合保证了：无论网络深度如何增加、训练如何推进，信号在层间传递时的增益始终受控于 $[0, 1]$ 区间。

##### 3.x.2.2 Sinkhorn-Knopp 投影算法

mHC 采用 Sinkhorn-Knopp 迭代算法（Sinkhorn, 1964; Knopp, 1967）将任意非负矩阵投影至双随机矩阵空间。该算法通过交替进行行归一化和列归一化操作实现收敛。

**算法 1：Sinkhorn-Knopp 迭代**

---
**输入**：非负矩阵 $\mathbf{M} \in \mathbb{R}_{\geq 0}^{n \times n}$，迭代次数 $T$

**输出**：双随机矩阵 $\mathbf{P} \in \mathcal{B}_n$

1. 初始化：$\mathbf{P}^{(0)} = \mathbf{M}$
2. **for** $t = 1, \ldots, T$ **do**
   - 行归一化：$P^{(t-\frac{1}{2})}_{ij} = P^{(t-1)}_{ij} \big/ \sum_{k=1}^n P^{(t-1)}_{ik}$
   - 列归一化：$P^{(t)}_{ij} = P^{(t-\frac{1}{2})}_{ij} \big/ \sum_{k=1}^n P^{(t-\frac{1}{2})}_{kj}$
3. **return** $\mathbf{P}^{(T)}$

---

**收敛性质**：对于正矩阵（$M_{ij} > 0, \forall i,j$），Sinkhorn-Knopp 迭代以线性速率收敛到唯一的双随机矩阵。收敛速率由矩阵的 Hilbert 度量决定：

$$
d_H(\mathbf{P}^{(t)}, \mathbf{P}^*) \leq \lambda^t \cdot d_H(\mathbf{P}^{(0)}, \mathbf{P}^*)
$$

其中 $\lambda < 1$ 为与矩阵条件数相关的收缩系数。

**迭代次数选择**：实验验证表明，$T=20$ 次迭代即可达到工程精度要求。$T=1$ 时约束过弱（矩阵仅行随机），$T \geq 5$ 时双随机约束基本满足，$T=20$ 后继续增加迭代次数无显著改善。

##### 3.x.2.3 mHC 前向传播完整流程

mHC 的前向传播包含以下步骤：

**步骤 1：原始系数计算**

基于三组可学习映射 $H_{\text{pre}}, H_{\text{post}}, H_{\text{res}}$，计算无约束的连接系数：

$$
\tilde{\mathbf{W}}_{\text{pre}} = H_{\text{pre}}(\mathbf{x}), \quad \tilde{\mathbf{W}}_{\text{post}} = H_{\text{post}}(\mathbf{x}), \quad \tilde{\mathbf{W}}_{\text{res}} = H_{\text{res}}(\mathbf{x})
$$

**步骤 2：非负化处理**

将无约束系数转换为非负矩阵（Sinkhorn-Knopp 的前置要求）：

$$
\mathbf{W}' = \exp(\tilde{\mathbf{W}})
$$

$\exp(\cdot)$ 保证输出严格为正，使 Sinkhorn-Knopp 迭代的收敛性成立。

**步骤 3：Sinkhorn-Knopp 流形投影**

$$
\mathbf{W} = \text{Sinkhorn-Knopp}(\mathbf{W}', T=20)
$$

投影后 $\mathbf{W} \in \mathcal{B}_n$，谱范数有界：$\|\mathbf{W}\|_2 \leq 1$。

**步骤 4：残差混合与输出**

$$
\mathbf{output} = \mathbf{W} \cdot [\mathcal{F}(\mathbf{x}); \mathbf{x}_1; \mathbf{x}_2; \ldots; \mathbf{x}_n]
$$

其中 $[\cdot; \cdot]$ 表示 $n+1$ 条路径（$\mathcal{F}(\mathbf{x})$ + $n$ 条残差路径）的拼接。

**步骤 3 的流形投影是 mHC 区别于 HC 的核心差异点。**HC 直接使用 $\tilde{\mathbf{W}}$ 进行混合（无约束），mHC 将其投影到 $\mathcal{B}_n$ 后再混合。

##### 3.x.2.4 mHC 初始化策略

mHC 采用小值初始化门控因子：

$$
\alpha_{\text{gate}} = 0.01
$$

该策略确保训练初期：
- $\exp(\tilde{\mathbf{W}}) \approx \exp(0.01) \approx 1.01$，近似均匀矩阵
- Sinkhorn-Knopp 投影后近似 $\frac{1}{n}\mathbf{1}\mathbf{1}^{\top}$（均匀分布）
- 均匀分布双随机矩阵对各路径等权贡献，等价于 Pre-Norm 的恒等映射行为

随着训练推进，门控因子偏离初始值，矩阵逐渐分化，不同路径获得差异化的残差混合比例，实现自适应的层间信息流动。

##### 3.x.2.5 信号增益的数学保证

设 $L$ 层 mHC 网络的复合前向增益为：

$$
\mathbf{G}_L = \prod_{\ell=1}^{L} \mathbf{W}_\ell, \quad \mathbf{W}_\ell \in \mathcal{B}_n
$$

由性质 1（乘法封闭性），$\mathbf{G}_L \in \mathcal{B}_n$。

由性质 2（谱范数有界），$\|\mathbf{G}_L\|_2 \leq 1$。

因此，无论网络深度 $L$ 取何值：

$$
\|\mathbf{G}_L \mathbf{x}\|_2 \leq \|\mathbf{G}_L\|_2 \cdot \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_2
$$

信号幅值不可能超过输入幅值，从数学上完全排除了信号爆炸的可能性。

对比 HC 的情况：无约束矩阵 $\|\mathcal{HC}_\ell\|_2$ 可取任意正值，$L$ 层累积后复合增益可达指数级。

#### 3.x.3 实验验证与性能分析

##### 3.x.3.1 实验配置

DeepSeek 团队在三个参数规模（3B、9B、27B MoE）上进行了系统性验证。所有模型采用 DeepSeek-V3 的基础架构（Multi-head Latent Attention + MoE），按比例缩放计算预算。

表5：实验模型配置

| 配置项 | 3B | 9B | 27B |
|-------|-----|-----|------|
| 架构 | MoE | MoE | MoE |
| 扩展率 $n$ | 4 | 4 | 4 |
| Sinkhorn 迭代次数 $T$ | 20 | 20 | 20 |
| 门控初始化 $\alpha$ | 0.01 | 0.01 | 0.01 |

##### 3.x.3.2 信号增益对比

表6：HC vs mHC 信号增益特性（27B 模型）

| 架构 | 单层最大增益 | 复合增益（64层） | 训练稳定性 |
|------|------------|-----------------|-----------|
| HC（无约束） | 1 ~ 7 | $10^3 \sim 10^5$ | 12k 步后崩溃 |
| mHC（双随机约束） | ≤ 1.0 | ≈ 1.6 | 全程稳定 |

信号增益从 $10^3$~$10^5$ 压缩至约 1.6，降幅达三个数量级。这是 mHC 实现训练稳定性的核心机制。

训练过程中的具体表现差异：
- **HC**：约 12,000 步出现突发损失飙升，梯度范数与损失高度相关振荡，不可恢复
- **mHC**：损失曲线平滑收敛，梯度范数稳定，与基线模型（Pre-Norm）行为一致

##### 3.x.3.3 基准测试性能

表7：27B 模型全量基准测试结果

| 基准测试 | 基线（Pre-Norm） | HC | mHC | mHC vs 基线 | mHC vs HC |
|---------|----------------|-----|------|------------|----------|
| BBH（Exact Match）| 43.8 | 48.9 | **51.0** | +7.2 | +2.1 |
| DROP（F1） | 47.0 | 51.6 | **53.9** | +6.9 | +2.3 |
| GSM8K | 46.7 | — | **53.8** | +7.1 | — |
| MMLU | 59.0 | — | **63.4** | +4.4 | — |
| HellaSwag | — | — | — | 正向 | 正向 |
| PIQA | — | — | — | 正向 | 正向 |
| TriviaQA | — | — | — | 正向 | 正向 |
| ARC-Challenge | — | — | — | 正向 | 正向 |

在核心推理基准（BBH, DROP, GSM8K）上，mHC 相比 Pre-Norm 基线提升 7+ 分，相比 HC 提升 2+ 分。知识类任务（MMLU）提升 4.4 分。在全部 8 项基准测试中，mHC 均全面超越 HC 和基线。

表8：跨规模一致性验证（BBH 基准）

| 模型规模 | 基线 | mHC | 提升 |
|---------|------|------|------|
| 3B | — | — | 持续正向 |
| 9B | — | — | 持续正向 |
| 27B | 43.8 | 51.0 | +7.2 |

mHC 的性能优势在三个规模上均保持一致，且随计算预算增加有轻微扩大趋势。

##### 3.x.3.4 损失曲线分析

训练损失曲线对比显示：

- mHC 最终损失相比基线低 0.021
- 该损失差距从训练早期开始建立，中后期保持稳定
- HC 因训练崩溃无法完成全程对比

0.021 的损失改善直接转化为前述 7+ 分的下游任务提升，符合语言模型领域"小损失差异 → 大性能差异"的经验规律。

##### 3.x.3.5 计算开销分析

mHC 相比标准 Pre-Norm 残差连接的额外计算开销来源于三个部分：

表9：mHC 额外开销分解

| 开销来源 | 操作内容 | 占比 |
|---------|---------|------|
| 扩展率 $n=4$ | 隐状态维度 $d \to 4d$ 的路径维护 | 主要 |
| Sinkhorn-Knopp | $T=20$ 次行/列归一化（$n \times n$ 矩阵） | 次要 |
| 非负化 + 投影 | $\exp(\cdot)$ 计算 | 微小 |

实测总训练开销增加约 **6.7%**，且该开销在不同模型规模下保持恒定。

**工程优化措施**：
- 混合精度策略：Sinkhorn 迭代在 FP32 下执行（保证数值稳定），主干计算在 BF16/FP8 下执行
- 激活重计算（Activation Recomputation）：减少多路径结构带来的显存压力
- 专用高优先级计算流：FFN 层计算在流水线并行中获得独立 CUDA Stream，避免 Sinkhorn 计算阻塞主计算流

##### 3.x.3.6 Sinkhorn 迭代次数消融

表10：不同迭代次数对性能的影响

| 迭代次数 $T$ | 双随机约束满足程度 | 信号增益控制 | 额外开销 |
|-------------|-----------------|------------|---------|
| 1 | 弱（仅行随机） | 不充分 | 极低 |
| 5 | 近似满足 | 基本控制 | 低 |
| 10 | 良好满足 | 完全控制 | 中等 |
| 20 | 充分满足 | 完全控制 | 中等 |
| 50 | 过度（无额外收益） | 完全控制 | 较高 |

$T=20$ 为工程最优点：约束充分满足且增加迭代无额外性能收益。$T=1$ 时矩阵仅满足行随机条件（行和为 1，列和不保证），不足以控制复合增益。

#### 3.x.4 技术讨论

##### 3.x.4.1 关于"流形约束"的数学严格性

论文标题中"Manifold"一词在严格数学意义上存在术语偏差：

1. Birkhoff 多面体 $\mathcal{B}_n$ 是一个**凸多面体**（Convex Polytope），维度为 $(n-1)^2$，其边界由有限个线性不等式定义，不具备光滑流形（Smooth Manifold）结构
2. 多面体的棱和顶点处不可微，不满足微分流形的光滑性条件
3. mHC 的投影目标为多面体**相对内部**（通过 Sinkhorn-Knopp 收敛到严格正矩阵区域），而非边界

然而，从工程角度看，选择凸多面体作为约束空间具有两方面合理性：

*计算效率*：凸集投影具有唯一性，且 Sinkhorn-Knopp 算法复杂度仅为 $O(Tn^2)$（$T$ 次迭代，每次 $O(n^2)$ 的归一化操作）。相比之下，向光滑流形投影通常需要求解非线性优化问题。

*高维等价性*：在高维空间中（扩展率 $n \geq 4$），Birkhoff 多面体的体积高度集中于"表面"附近（类似高维球体的体积集中现象），内部几乎为空。因此，"约束到多面体内部"与"约束到流形表面"在实际效果上几乎等价。这一现象可用以下估计定量描述：

$$
\frac{\text{Vol}(\mathcal{B}_n(1-\epsilon))}{\text{Vol}(\mathcal{B}_n)} \to 0, \quad \text{as } n \to \infty
$$

其中 $\mathcal{B}_n(1-\epsilon)$ 为按比例 $(1-\epsilon)$ 缩放的内部区域。

##### 3.x.4.2 与最优传输的联系

Sinkhorn-Knopp 算法在最优传输（Optimal Transport）领域被广泛应用，用于求解熵正则化的 Kantorovich 问题。mHC 的流形投影本质上等价于在残差混合矩阵上施加最优传输约束：各路径间的信号传递满足"质量守恒"——总信号量既不增加（防止爆炸）也不减少（防止消失），仅在路径间重新分配。

这一联系提供了 mHC 有效性的另一角度理解：多路径残差连接本质上是一个信号传输网络，双随机约束保证了该网络的传输效率恒为 1（无损传输）。

##### 3.x.4.3 Sinkhorn-Knopp 的可微性

mHC 的端到端训练要求 Sinkhorn-Knopp 迭代过程可微（梯度可通过投影操作回传）。这通过以下机制实现：

1. 行/列归一化操作本质为 softmax 变体，其雅可比矩阵有解析形式
2. $T$ 步迭代展开为计算图的 $T$ 层链式结构，可直接通过自动微分系统处理
3. 实践中采用定步数展开（$T=20$），避免动态收敛判断带来的计算图不确定性

**隐式微分的替代方案**：理论上可对收敛后的不动点使用隐式函数定理计算梯度（节省内存），但 DeepSeek 选择了显式展开方案（$T=20$ 步前向 + 反向），因 $n=4$ 时矩阵尺寸仅为 $4 \times 4$，显式展开的额外开销可忽略。

##### 3.x.4.4 与 Layer Normalization 的互补关系

Layer Normalization（Ba et al., 2016）通过归一化隐状态的幅值来稳定训练：

$$
\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \cdot \gamma + \beta
$$

mHC 通过约束混合矩阵的谱范数来稳定训练。两者作用于不同维度：

| 维度 | LayerNorm | mHC |
|------|-----------|-----|
| 作用对象 | 隐状态向量（$d$ 维） | 残差混合矩阵（$n \times n$） |
| 稳定机制 | 二阶矩归一化 | 谱范数有界 |
| 控制目标 | 表征尺度 | 层间增益 |

两者的组合使用（Pre-Norm + mHC）从表征尺度和层间增益两个维度同时保证训练稳定性。

##### 3.x.4.5 与 DeepSeek 后续模型的关联

DeepSeek 的技术报告发布节奏通常预示其下一代模型的架构方向。综合以下信号：

1. mHC 论文由创始人梁文峰挂名通讯作者
2. 发布时间点（2025 年 12 月 31 日）与预期的 DeepSeek-V4 发布窗口临近
3. mHC 已在万亿 token 训练规模上验证有效性
4. Engram 论文中的实验模型明确使用了 mHC 作为层间连接（"Multi-head Latent Attention with 32 heads connected to feed-forward networks through Manifold Constrained Hyper Connections with expansion rate 4"）

mHC 已确认为 DeepSeek 下一代架构的基础组件。

##### 3.x.4.6 局限性

1. **扩展率的显存开销**：$n=4$ 意味着残差流的宽度扩展 4 倍，虽然不影响 FLOPs（仅增加逐元素操作），但激活显存增加 4 倍，需配合激活重计算使用
2. **小矩阵约束的表达力限制**：$4 \times 4$ 双随机矩阵仅有 $(4-1)^2 = 9$ 个自由度，可能限制路径间信息流的表达能力
3. **推理阶段无法省略**：与训练专用的 Dropout 不同，mHC 的投影操作在推理阶段仍需执行（因混合系数影响输出），增加推理开销

#### 3.x.5 本节小结

mHC 通过引入双随机矩阵流形约束，解决了 HC 架构的信号增益发散问题，在保留多路径残差连接优势的同时恢复了恒等映射性质。其核心贡献包括：

1. **理论层面**：将残差混合矩阵约束到 Birkhoff 多面体 $\mathcal{B}_n$，利用双随机矩阵的乘法封闭性和谱范数有界性，从数学上保证任意深度网络的信号增益不超过 1
2. **算法层面**：采用 Sinkhorn-Knopp 迭代（$T=20$）实现高效的流形投影，算法复杂度 $O(Tn^2)$，对小扩展率（$n=4$）几乎无计算开销
3. **工程层面**：以 6.7% 的额外训练开销换取：(a) 训练稳定性——消除 HC 的 12k 步崩溃问题；(b) 性能提升——BBH +7.2、DROP +6.9、GSM8K +7.1（相比 Pre-Norm 基线）

mHC 与 DeepSeek 的 MoE、MLA、Engram 共同构成了下一代大模型架构的核心组件栈，已在 Engram 论文的实验模型中得到实际应用验证。

---

**参考文献**

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. CVPR 2016.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.
- Sinkhorn, R. (1964). A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices. The Annals of Mathematical Statistics, 35(2), 876-879.
- Sinkhorn, R. & Knopp, P. (1967). Concerning Nonnegative Matrices and Doubly Stochastic Matrices. Pacific Journal of Mathematics, 21(2), 343-348.
- Zhu, D. et al. (2024). Hyper-Connections. arXiv:2409.19606. ICLR 2025.
- Xie, Z. et al. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
- Cheng, X. et al. (2026). Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv:2601.07372.
