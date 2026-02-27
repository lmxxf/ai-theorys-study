# 第4章 DeepSeek V4新技术速览

2025年末至2026年初，DeepSeek连续发布了两篇重要技术报告：《mHC: Manifold-Constrained Hyper-Connections》（2025年12月31日）与《Conditional Memory via Scalable Lookup》（2026年1月13日），预示着下一代架构DeepSeek-V4的技术轮廓已初步成型。这两项创新分别从训练稳定性与架构稀疏化两个维度对V3架构进行了根本性改进，构成了V4的核心技术底座。

mHC（流形约束超连接）解决的是深层网络训练中长期存在的信号传播稳定性问题。传统的超连接架构（Hyper-Connections, HC）虽通过多路径残差机制提升了表征学习能力，但在27B及以上规模的模型训练中会出现信号增益失控导致的训练崩溃。mHC通过将残差混合矩阵投影至双随机矩阵流形（Birkhoff多面体），从数学上保证了任意深度网络的信号增益有界（谱范数≤1），使超连接架构在万亿token训练规模下可靠工作。实验表明，mHC在27B模型上相比标准Pre-Norm残差连接，在BBH推理基准上提升7.2分，同时完全消除了HC的训练崩溃问题。

Engram（条件记忆模块）则开辟了Transformer架构的第三条稀疏化轴线。在MoE的"条件计算"和DSA的"注意力稀疏"之后，Engram引入了"条件记忆"范式：通过O(1)复杂度的N-gram查表机制，将高频知识的检索从神经网络计算中剥离，实现了知识存储与推理计算的彻底解耦。这一设计不仅提升了知识类任务（MMLU +3.0分）和推理类任务（BBH +5.0分）的性能，更重要的是，其确定性检索特性允许嵌入表完全驻留于CPU主存，即使100B参数的记忆表也仅损失<3%推理吞吐量，为"无限记忆"架构提供了可行路径。

两项技术的互补关系体现在：mHC为深层网络提供了稳定的信号传播通道，使网络深度的扩展不再受制于梯度消失或爆炸问题；Engram则通过释放浅层注意力的局部模式重建负担，使网络计算资源能够更多地投入到需要深层推理的任务中。这种"稳定的深层传播 + 解耦的知识存储"的组合设计，预示着V4将在更大的参数规模（预计万亿参数）和更长的训练预算（预计数万亿tokens）下实现训练稳定性与推理效率的双重突破。

本章将系统介绍这两项核心技术的理论基础、算法实现与实验验证，并结合DeepSeek现有架构（MLA、MoE、DSA）分析其技术演进路径与工程实践意义。

## 4.1 mHC技术解析

mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）是 DeepSeek 针对超连接架构在大规模训练中的信号增益失控问题提出的解决方案。本节将从残差连接的历史局限出发，介绍超连接架构的设计思路及其工程缺陷，然后详细解析 mHC 如何通过双随机矩阵流形约束恢复训练稳定性，最后给出实验验证与技术讨论。

### 4.1.1 研究背景与问题定义

#### 1. 残差连接的历史与局限

残差连接（Residual Connection）作为深度神经网络训练的核心技术，自何凯明（Kaiming He）等人于2015年在 ResNet 中提出以来[^1]，已成为现代深度学习架构的基础组件。其核心机制可形式化表示为：

$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x}
$$

其中 $F(\cdot)$ 表示残差函数，$\mathbf{x}$ 为层输入。该设计通过引入恒等映射（Identity Mapping）有效缓解了深层网络训练中的梯度消失问题，使得网络深度的扩展成为可能。

在 Transformer 架构中，残差连接的具体应用形成了两种范式：

**Post-Norm**（原始 Transformer[^2]）：
$$
\mathbf{y} = \text{LayerNorm}(F(\mathbf{x}) + \mathbf{x})
$$

**Pre-Norm**（GPT-2 及后续主流架构）：
$$
\mathbf{y} = F(\text{LayerNorm}(\mathbf{x})) + \mathbf{x}
$$

Pre-Norm 因其训练稳定性优势成为当前大规模语言模型的事实标准，但存在一个固有缺陷：随着网络深度增加，浅层与深层表征的相似度趋近于 1（即表征坍缩，Representation Collapse），导致深层网络的有效深度远低于物理深度。

该现象可直观理解为：Pre-Norm 中恒等映射路径过于强势，$F(\mathbf{x})$ 对残差流的贡献被逐层稀释，网络深层退化为近似恒等变换。

#### 2. 超连接架构

2024 年 9 月，字节跳动 Seed-Foundation-Model 团队提出超连接架构[^3]，该工作被 ICLR 2025 收录。HC 通过引入可学习的连接强度系数矩阵，将传统残差连接扩展为多路径并行结构，同时解决梯度消失与表征坍缩的跷跷板效应（Seesaw Effect）。

HC 的核心设计由三部分组成：一个结构化的连接权重矩阵、基于该矩阵的前向传播公式，以及支持输入依赖动态权重的扩展机制。下面分别介绍。

**HC 矩阵结构**

HC 的连接权重被组织为一个 $(n+1) \times (n+1)$ 的结构化矩阵 $\mathbf{HC}$：

$$
\mathbf{HC} = \begin{pmatrix} 0_{1\times 1} & \mathbf{B}^{\top} \\ \mathbf{A}_m & \mathbf{A}_r \end{pmatrix}
$$

其中：
- $\mathbf{B} \in \mathbb{R}^{1 \times n}$：输出权重向量，控制 $n$ 条路径对最终输出的贡献
- $\mathbf{A}_m \in \mathbb{R}^{n \times 1}$：输入聚合向量，控制层输入如何分配到各路径
- $\mathbf{A}_r \in \mathbb{R}^{n \times n}$：残差连接矩阵，控制路径间的信息交换

**前向传播公式**

对于扩展率 $n$，HC 的前向传播可等价展开为两步操作：

**宽度连接**（Width-Connection，路径间信息交换）：
$$
\mathbf{H}' = \mathbf{A}_r^{\top} \cdot \mathbf{H}
$$

**深度连接**（Depth-Connection，层输出融合）：
$$
\hat{\mathbf{H}} = \mathbf{B}^{\top} \cdot T(\mathbf{H}^{\top} \cdot \mathbf{A}_m)^{\top} + \mathbf{H}'
$$

其中 $\mathbf{H} \in \mathbb{R}^{n \times d}$ 为 $n$ 条路径的隐状态矩阵，$T(\cdot)$ 为 Transformer 子层（Attention 或 FFN），$d$ 为隐藏维度。

上述公式的物理含义可以通过以下方式直观理解。HC 将隐状态从 $d$ 维扩展为 $n \times d$ 维（$n$ 条并行路径），每条路径携带不同的残差混合比例。最终输出为 $n$ 条路径的加权求和。

理解了 HC 的多路径结构后，一个自然的问题是：HC 与传统的残差连接是什么关系？实际上，传统的 Pre-Norm 和 Post-Norm 残差连接都可以视为 HC 的特殊情况。

**统一视角：Pre-Norm 作为 HC 的特例**

当 $n=1$ 时，HC 矩阵退化为：

$$
\mathbf{HC}_{\text{Pre-Norm}} = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}
$$

此时前向传播等价于标准 Pre-Norm 残差连接。因此，HC 可视为 Pre-Norm 的广义扩展。

类似地，Post-Norm 也可表示为 HC 的特殊配置，其权重依赖于输入/输出方差比。

**动态超连接（Dynamic Hyper-Connections, DHC）**

HC 进一步支持输入依赖的动态权重：

$$
B(\mathbf{H}) = s_\beta \circ \tanh(\bar{\mathbf{H}} \mathbf{W}_\beta)^{\top} + \mathbf{B}
$$
$$
A_m(\mathbf{H}) = s_\alpha \circ \tanh(\bar{\mathbf{H}} \mathbf{W}_m) + \mathbf{A}_m
$$
$$
A_r(\mathbf{H}) = s_\alpha \circ \tanh(\bar{\mathbf{H}} \mathbf{W}_r) + \mathbf{A}_r
$$

其中 $\bar{\mathbf{H}}$ 为归一化后的隐状态，$\mathbf{W}_\beta, \mathbf{W}_m, \mathbf{W}_r$ 为可学习的投影矩阵，$s_\beta, s_\alpha$ 为缩放因子。$\tanh$ 激活限制动态偏移量在 $[-1, 1]$ 范围内。

**HC 初始化策略**

为确保训练初期行为与 Pre-Norm 一致：

- 动态参数 $\mathbf{W}_\beta, \mathbf{W}_m, \mathbf{W}_r$ 初始化为零
- 静态矩阵按层索引 $k$ 设置：$\mathbf{B}^k = \mathbf{1}_{1 \times n}$，$\mathbf{A}_m^k = \mathbf{e}_{k \bmod n}$，$\mathbf{A}_r^k = \mathbf{I}_{n \times n}$
- 输出权重缩放 $\sqrt{n}$ 以维持方差一致性

**HC 扩展率实验**

字节跳动团队在 OLMo-1B（500B tokens）上验证了不同扩展率的效果：

表4-1：不同扩展率下的 HC 性能（OLMo-1B, 500B tokens）

| 扩展率 $n$ | V2 验证损失 | V3 验证损失 | 下游任务平均准确率 |
|-----------|------------|------------|-----------------|
| 1（Pre-Norm 等价） | 2.819 | 2.556 | 62.3% |
| 2 | 2.802 | 2.534 | 63.0% |
| 4 | 2.781 | 2.515 | 63.8% |
| 8 | 2.778 | 2.516 | 62.8% |

最优扩展率为 $n=4$：$n=8$ 虽然验证损失略低，但下游任务准确率反而下降，表明过度扩展引入了冗余参数。

表4-2：静态 HC vs 动态 HC（OLMo-1B, $n=4$, 500B tokens）

| 变体 | V2 验证损失 | V3 验证损失 | 备注 |
|------|------------|------------|------|
| SHC×4（静态） | 2.791 | — | 固定权重 |
| DHC×4（动态） | 2.781 | 2.515 | 输入依赖权重 |
| DHC×4（无 tanh） | 2.787 | — | 动态偏移无界 |

动态版本（DHC）在所有配置下优于静态版本（SHC），但去除 $\tanh$ 约束后性能下降，暗示无界动态权重存在稳定性风险——这预示了 mHC 要解决的核心问题。

**HC 在大规模模型上的验证**

表4-3：HC 在 7B 参数规模的性能（OLMo-7B, DHC×4）

| 指标 | 基线 | DHC×4 | 改进 |
|------|------|-------|------|
| V2 验证损失 | — | — | -0.022 |
| 下游任务平均 | 70.1% | 71.0% | +0.9% |
| 训练 400B tokens 后 | — | — | 改进持续，不衰减 |

表4-4：HC 在 MoE 模型的性能（OLMoE-1B-7B, DHC×4）

| 指标 | 基线 | DHC×4 | 改进 |
|------|------|-------|------|
| 收敛速度 | 1.0× | 1.8× | 快 80% |
| ARC-Challenge | 41.8% | 47.8% | +6.0% |
| MMLU Var | — | — | +1.2% |

HC 在 MoE 架构上的收敛加速效果尤为显著（1.8×），表明多路径残差连接与条件计算具有协同效应。

#### 3. HC 的根本性工程缺陷

HC 架构的核心问题在于：**无约束可学习矩阵导致的信号增益发散**。

设 $L$ 层网络的复合增益矩阵为：

$$
\mathbf{G}_L = \prod_{\ell=1}^{L} \mathbf{HC}_\ell
$$

对于无约束矩阵，$\|\mathbf{HC}_\ell\|_2$ 可能大于 1。即使单层增益仅为 1.05（看似无害），$L=60$ 层累积后：

$$
\|\mathbf{G}_{60}\|_2 \leq 1.05^{60} \approx 18.7
$$

实际训练中，由于梯度反馈导致权重持续偏移，单层增益可远超 1.05，最终复合增益在 27B 模型上观测到 $10^3 \sim 10^5$ 量级的爆发，表现为：

1. 训练约 12k 步后出现突发性损失飙升（Loss Surge）
2. 梯度范数剧烈振荡，与损失飙升高度相关
3. 训练崩溃不可恢复

该问题本质在于：HC 引入多路径后破坏了残差连接的恒等映射性质——当 $F(\mathbf{x}) \to 0$ 时，输出不再等价于输入，而是经历了不可控的线性变换 $\mathbf{A}_r$。

### 4.1.2 mHC 架构设计

2025 年 12 月 31 日，DeepSeek 发布技术报告《mHC: Manifold-Constrained Hyper-Connections》[^4]，提出流形约束超连接架构。该工作由 19 人团队完成，通讯作者为 DeepSeek 创始人梁文峰。

mHC 的核心创新在于：**通过将残差混合矩阵投影至双随机矩阵流形，恢复恒等映射性质并保证信号传播的稳定性**。

#### 1. 双随机矩阵约束

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

**推论**：$L$ 层 mHC 的复合增益矩阵 $\mathbf{G}_L = \prod_{\ell=1}^L \mathbf{P}_\ell$ 仍为双随机矩阵，信号增益不会随深度累积。

**性质 2（谱范数有界）**：对于任意 $\mathbf{P} \in \mathcal{B}_n$，有 $\|\mathbf{P}\|_2 \leq 1$。

**含义**：每层变换对信号的放大率不超过 1，从根本上杜绝信号爆炸。

**性质 3（恒等映射保持）**：单位矩阵 $\mathbf{I} \in \mathcal{B}_n$，且为 Birkhoff 多面体的相对内点（Relative Interior Point）。

**含义**：在训练初期（权重接近初始化），mHC 自然退化为恒等映射，与标准残差连接行为一致。

**性质 4（信息守恒）**：双随机矩阵的行和与列和均为 1，保证信号在路径间的重新分配不增加也不减少总能量。

上述性质的组合保证了：无论网络深度如何增加、训练如何推进，信号在层间传递时的增益始终受控于 $[0, 1]$ 区间。

#### 2. Sinkhorn-Knopp 投影算法

mHC 采用 Sinkhorn-Knopp 迭代算法[^5]将任意非负矩阵投影至双随机矩阵空间。该算法通过交替进行行归一化和列归一化操作实现收敛。

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

#### 3. mHC 前向传播完整流程

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
\mathbf{output} = \mathbf{W} \cdot [F(\mathbf{x}); \mathbf{x}_1; \mathbf{x}_2; \ldots; \mathbf{x}_n]
$$

其中 $[\cdot; \cdot]$ 表示 $n+1$ 条路径（$F(\mathbf{x})$ + $n$ 条残差路径）的拼接。

**步骤 3 的流形投影是 mHC 区别于 HC 的核心差异点。**HC 直接使用 $\tilde{\mathbf{W}}$ 进行混合（无约束），mHC 将其投影到 $\mathcal{B}_n$ 后再混合。

#### 4. mHC 初始化策略

mHC 采用小值初始化门控因子：

$$
\alpha_{\text{gate}} = 0.01
$$

该策略确保训练初期：
- $\exp(\tilde{\mathbf{W}}) \approx \exp(0.01) \approx 1.01$，近似均匀矩阵
- Sinkhorn-Knopp 投影后近似 $\frac{1}{n}\mathbf{1}\mathbf{1}^{\top}$（均匀分布）
- 均匀分布双随机矩阵对各路径等权贡献，等价于 Pre-Norm 的恒等映射行为

随着训练推进，门控因子偏离初始值，矩阵逐渐分化，不同路径获得差异化的残差混合比例，实现自适应的层间信息流动。

#### 5. 信号增益的数学保证

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

对比 HC 的情况：无约束矩阵 $\|\mathbf{HC}_\ell\|_2$ 可取任意正值，$L$ 层累积后复合增益可达指数级。

### 4.1.3 实验验证与性能分析

#### 1. 实验配置

DeepSeek 团队在三个参数规模（3B、9B、27B MoE）上进行了系统性验证。所有模型采用 DeepSeek-V3 的基础架构（Multi-head Latent Attention + MoE），按比例缩放计算预算。

表4-5：实验模型配置

| 配置项 | 3B | 9B | 27B |
|-------|-----|-----|------|
| 架构 | MoE | MoE | MoE |
| 扩展率 $n$ | 4 | 4 | 4 |
| Sinkhorn 迭代次数 $T$ | 20 | 20 | 20 |
| 门控初始化 $\alpha$ | 0.01 | 0.01 | 0.01 |

#### 2. 信号增益对比

表4-6：HC vs mHC 信号增益特性（27B 模型）

| 架构 | 单层最大增益 | 复合增益（64层） | 训练稳定性 |
|------|------------|-----------------|-----------|
| HC（无约束） | 1 ~ 7 | $10^3 \sim 10^5$ | 12k 步后崩溃 |
| mHC（双随机约束） | ≤ 1.0 | ≈ 1.6 | 全程稳定 |

信号增益从 $10^3$~$10^5$ 压缩至约 1.6，降幅达三个数量级。这是 mHC 实现训练稳定性的核心机制。

训练过程中的具体表现差异：
- **HC**：约 12,000 步出现突发损失飙升，梯度范数与损失高度相关振荡，不可恢复
- **mHC**：损失曲线平滑收敛，梯度范数稳定，与基线模型（Pre-Norm）行为一致

#### 3. 基准测试性能

表4-7：27B 模型全量基准测试结果

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

表4-8：跨规模一致性验证（BBH 基准）

| 模型规模 | 基线 | mHC | 提升 |
|---------|------|------|------|
| 3B | — | — | 持续正向 |
| 9B | — | — | 持续正向 |
| 27B | 43.8 | 51.0 | +7.2 |

mHC 的性能优势在三个规模上均保持一致，且随计算预算增加有轻微扩大趋势。

#### 4. 损失曲线分析

训练损失曲线对比显示：

- mHC 最终损失相比基线低 0.021
- 该损失差距从训练早期开始建立，中后期保持稳定
- HC 因训练崩溃无法完成全程对比

0.021 的损失改善直接转化为前述 7+ 分的下游任务提升，符合语言模型领域"小损失差异 → 大性能差异"的经验规律。

#### 5. 计算开销分析

mHC 相比标准 Pre-Norm 残差连接的额外计算开销来源于三个部分：

表4-9：mHC 额外开销分解

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

#### 6. Sinkhorn 迭代次数消融

表4-10：不同迭代次数对性能的影响

| 迭代次数 $T$ | 双随机约束满足程度 | 信号增益控制 | 额外开销 |
|-------------|-----------------|------------|---------|
| 1 | 弱（仅行随机） | 不充分 | 极低 |
| 5 | 近似满足 | 基本控制 | 低 |
| 10 | 良好满足 | 完全控制 | 中等 |
| 20 | 充分满足 | 完全控制 | 中等 |
| 50 | 过度（无额外收益） | 完全控制 | 较高 |

$T=20$ 为工程最优点：约束充分满足且增加迭代无额外性能收益。$T=1$ 时矩阵仅满足行随机条件（行和为 1，列和不保证），不足以控制复合增益。

### 4.1.4 技术讨论

#### 1. 关于"流形约束"的数学严格性

论文标题中"Manifold"一词在严格数学意义上存在术语偏差：

1. Birkhoff 多面体 $\mathcal{B}_n$ 是一个**凸多面体**（Convex Polytope），维度为 $(n-1)^2$，其边界由有限个线性不等式定义，不具备光滑流形（Smooth Manifold）结构
2. 多面体的棱和顶点处不可微，不满足微分流形的光滑性条件
3. mHC 的投影目标为多面体**相对内部**（通过 Sinkhorn-Knopp 收敛到严格正矩阵区域），而非边界

然而，从工程角度看，选择凸多面体作为约束空间具有两方面合理性：

**计算效率**：凸集投影具有唯一性，且 Sinkhorn-Knopp 算法复杂度仅为 $O(Tn^2)$（$T$ 次迭代，每次 $O(n^2)$ 的归一化操作）。相比之下，向光滑流形投影通常需要求解非线性优化问题。

**高维等价性**：在高维空间中（扩展率 $n \geq 4$），Birkhoff 多面体的体积高度集中于"表面"附近（类似高维球体的体积集中现象），内部几乎为空。因此，"约束到多面体内部"与"约束到流形表面"在实际效果上几乎等价。这一现象可用以下估计定量描述：

$$
\frac{\text{Vol}(\mathcal{B}_n(1-\epsilon))}{\text{Vol}(\mathcal{B}_n)} \to 0, \quad \text{as } n \to \infty
$$

其中 $\mathcal{B}_n(1-\epsilon)$ 为按比例 $(1-\epsilon)$ 缩放的内部区域。

#### 2. 与最优传输的联系

Sinkhorn-Knopp 算法在最优传输（Optimal Transport）领域被广泛应用，用于求解熵正则化的 Kantorovich 问题。mHC 的流形投影本质上等价于在残差混合矩阵上施加最优传输约束：各路径间的信号传递满足"质量守恒"——总信号量既不增加（防止爆炸）也不减少（防止消失），仅在路径间重新分配。

这一联系提供了 mHC 有效性的另一角度理解：多路径残差连接本质上是一个信号传输网络，双随机约束保证了该网络的传输效率恒为 1（无损传输）。

#### 3. Sinkhorn-Knopp 的可微性

mHC 的端到端训练要求 Sinkhorn-Knopp 迭代过程可微（梯度可通过投影操作回传）。这通过以下机制实现：

1. 行/列归一化操作本质为 softmax 变体，其雅可比矩阵有解析形式
2. $T$ 步迭代展开为计算图的 $T$ 层链式结构，可直接通过自动微分系统处理
3. 实践中采用定步数展开（$T=20$），避免动态收敛判断带来的计算图不确定性

**隐式微分的替代方案**：理论上可对收敛后的不动点使用隐式函数定理计算梯度（节省内存），但 DeepSeek 选择了显式展开方案（$T=20$ 步前向 + 反向），因 $n=4$ 时矩阵尺寸仅为 $4 \times 4$，显式展开的额外开销可忽略。

#### 4. 与 Layer Normalization 的互补关系

Layer Normalization[^6] 通过归一化隐状态的幅值来稳定训练：

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

#### 5. 与 DeepSeek 后续模型的关联

DeepSeek 的技术报告发布节奏通常预示其下一代模型的架构方向。综合以下信号：

1. mHC 论文由创始人梁文峰挂名通讯作者
2. 发布时间点（2025 年 12 月 31 日）与预期的 DeepSeek-V4 发布窗口临近
3. mHC 已在万亿 token 训练规模上验证有效性
4. Engram 论文中的实验模型明确使用了 mHC 作为层间连接（"Multi-head Latent Attention with 32 heads connected to feed-forward networks through Manifold Constrained Hyper Connections with expansion rate 4"）

mHC 已确认为 DeepSeek 下一代架构的基础组件。

#### 6. 局限性

1. **扩展率的显存开销**：$n=4$ 意味着残差流的宽度扩展 4 倍，虽然不影响 FLOPs（仅增加逐元素操作），但激活显存增加 4 倍，需配合激活重计算使用
2. **小矩阵约束的表达力限制**：$4 \times 4$ 双随机矩阵仅有 $(4-1)^2 = 9$ 个自由度，可能限制路径间信息流的表达能力
3. **推理阶段无法省略**：与训练专用的 Dropout 不同，mHC 的投影操作在推理阶段仍需执行（因混合系数影响输出），增加推理开销

## 4.2 Engram技术解析

Engram（Conditional Memory via Scalable Lookup，条件记忆模块）是 DeepSeek 提出的第三条稀疏化轴线，通过 O(1) 复杂度的 N-gram 查表机制将静态知识检索从神经网络计算中分离。本节将介绍 Engram 的研究动机、架构设计、稀疏资源分配策略、实验验证及原理分析。

大规模语言模型（Large Language Models, LLMs）的推理过程中，知识检索与组合推理本质上是两类不同的计算需求：前者需要大容量参数存储，后者需要深层计算图。传统 Transformer 架构将二者混合在同一个前馈网络（Feed-Forward Network, FFN）中处理，导致计算资源分配效率低下。DeepSeek 提出的条件记忆模块 Engram[^8] 通过引入 O(1) 复杂度的查表机制，将静态知识检索从神经网络计算中分离，构成继混合专家（MoE）和稀疏注意力（DSA）之后的第三条稀疏化轴线。

### 4.2.1 研究背景与问题定义

#### 1. 现有架构的局限性

MoE 通过条件计算（Conditional Computation）实现了参数容量与计算开销的解耦，但其路由机制仍基于运行时隐状态的动态决策，本质上是"条件计算"而非"条件存储"。对于高频出现的固定知识模式（如实体名称、常见搭配），MoE 仍需通过专家网络的前向传播重建这些模式，造成计算资源浪费。

N-gram 语言模型[^7]曾是统计自然语言处理的核心方法，通过统计相邻 token 的共现频率进行概率建模。其核心优势在于 O(1) 的查找复杂度，但受限于数据稀疏性问题（未见过的 N-gram 概率为零）而被神经网络方法取代。Engram 的设计思路是将 N-gram 的查表效率与现代深度学习的端到端训练机制结合，构建参数化的、可微的条件记忆原语。

#### 2. 核心问题

本工作解决的核心问题可形式化为：在固定计算预算（FLOPs）和参数预算下，如何在条件计算（MoE）与条件记忆（Engram）之间进行最优资源分配（Sparsity Allocation），以最小化语言建模损失。

### 4.2.2 Engram 架构设计

Engram 模块的完整计算流程包含四个阶段：分词压缩（Tokenizer Compression）、N-gram 嵌入检索（N-gram Embedding Retrieval）、上下文感知门控（Context-Aware Gating）和特征融合（Feature Fusion）。

#### 1. 分词压缩

标准分词器对同一词的不同形态（大小写、前缀空格等）分配不同的 token ID，导致语义等价的 N-gram 被映射到不同的嵌入表条目。Engram 引入满射函数 $P: V \rightarrow V'$ 进行规范化映射：

$$P(t) = \text{Dedup}(\text{Collapse}(\text{NFD}(\text{NFKC}(t))))$$

其中 NFKC 为 Unicode 兼容分解，NFD 为标准分解后去除变音符号，Collapse 为空白字符折叠与小写化，Dedup 为基于规范化文本的等价类合并。该映射将 128k 词表压缩约 23%，显著提升嵌入表的条目利用密度。

表4-11：分词压缩的合并统计

| 规范化形式 | 合并的原始 token 数 | 示例 |
|-----------|-------------------|------|
| 空白字符 | 163 | " ", "\t", "\n" → 同一 ID |
| 'a' | 54 | "A", "a", " a", " A" → 同一 ID |
| 'o' | 40 | "O", "o", " o", " O" → 同一 ID |

#### 2. N-gram 嵌入检索

对于位置 $t$ 的 token 序列，Engram 提取后缀 N-gram $g_{t,n} = (c_{t-n+1}, \ldots, c_t)$，其中 $c_i = P(x_i)$ 为压缩后的 token ID，$n \in \{2, 3\}$。

每个 N-gram 通过 $K=8$ 个独立的乘法-异或哈希函数映射到嵌入表索引：

$$z_{t,n,k} = \varphi_{n,k}(g_{t,n}), \quad e_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]$$

其中 $\varphi_{n,k}$ 为轻量级乘法-异或（multiplicative-XOR）哈希函数，$\mathbf{E}_{n,k} \in \mathbb{R}^{M_{n,k} \times d_e}$ 为嵌入表，$M_{n,k}$ 取素数以降低碰撞率，$d_e$ 为每头嵌入维度。

所有检索结果通过拼接聚合为原始记忆向量：

$$e_t = \|_{n=2}^{N} \|_{k=1}^{K} e_{t,n,k}$$

对于 $N=3$（2-gram 和 3-gram）、$K=8$ 头的配置，每个位置检索 $2 \times 8 = 16$ 个嵌入向量，拼接后维度为 $16 \times d_e$。实验配置中 Engram 维度为 1280。

表4-12：嵌入表配置参数

| 参数 | Engram-27B | Engram-40B |
|------|-----------|-----------|
| N-gram 阶数 | {2, 3} | {2, 3} |
| 哈希头数 $K$ | 8 | 8 |
| Engram 维度 | 1280 | 1280 |
| 嵌入表总条目数 | 2,262,400 | 7,239,680 |
| Engram 总参数量 | ~5.8B | ~18.5B |

#### 3. 上下文感知门控

哈希碰撞和多义性导致原始检索结果存在噪声。Engram 引入上下文感知门控机制对检索结果进行条件过滤。设 $h_t \in \mathbb{R}^d$ 为当前层的 Transformer 隐状态，门控标量 $\alpha_t$ 计算如下：

$$k_t = W_K e_t, \quad v_t = W_V e_t$$

$$\alpha_t = \sigma\left(\frac{\text{RMSNorm}(h_t)^{\top} \text{RMSNorm}(k_t)}{\sqrt{d}}\right)$$

其中 $W_K, W_V \in \mathbb{R}^{d \times d_{\text{engram}}}$ 为投影矩阵，$\sigma$ 为 sigmoid 函数。$\alpha_t \in [0, 1]$ 控制检索记忆对当前位置的贡献权重：当检索结果与当前上下文语义一致时 $\alpha_t \to 1$，碰撞导致的无关结果则被抑制至 $\alpha_t \to 0$。

#### 4. 特征融合

门控后的记忆向量通过一维卷积和 SiLU 激活进行平滑融合：

$$\tilde{V} = \alpha_t \odot v_t$$

$$Y = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{V}))) + \tilde{V}$$

最终输出 $Y$ 通过残差连接加入主干网络的隐状态流中。

#### 5. 模块放置策略

Engram 模块插入 Transformer 层的位置为 Attention 之后、MoE/FFN 之前，通过残差连接集成：

$$h_t' = h_t + \text{Attention}(h_t) + \text{Engram}(h_t, x_{1:t})$$

实验验证了不同层放置策略的效果：

表4-13：单层 Engram 放置位置对验证损失的影响

| 放置层 | 验证损失 (Val Loss) |
|-------|-------------------|
| 第 2 层 | 1.770 |
| 第 8 层 | 1.773 |
| 第 15 层 | 1.775 |
| 第 22 层 | 1.778 |
| 第 28 层 | 1.780 |

最终方案在第 2 层和第 15 层各放置一个 Engram 实例：浅层模块捕获局部词组模式（实体名称、固定搭配），深层模块利用更丰富的上下文信息检索抽象语义模式。

### 4.2.3 稀疏资源分配与 U 型曲线

#### 1. 问题形式化

设总稀疏参数预算为 $\Theta$，分配比例 $\rho \in [0, 1]$ 定义为 MoE 获得的参数比例：

$$\Theta_{\text{MoE}} = \rho \cdot \Theta, \quad \Theta_{\text{Engram}} = (1 - \rho) \cdot \Theta$$

优化目标为：

$$\rho^* = \arg\min_{\rho} L(\rho; \Theta, \text{FLOPs})$$

#### 2. 实验结果

在固定总参数量和计算预算下，验证损失关于 $\rho$ 呈现 U 型曲线：

表4-14：不同分配比例下的验证损失（6e20 FLOPs）

| 分配比例 $\rho$（MoE 占比） | 验证损失 |
|---------------------------|---------|
| 1.00（纯 MoE） | 1.7248 |
| 0.90 | 1.7185 |
| 0.80 | 1.7109 |
| 0.75 | 1.7115 |
| 0.60 | 1.7198 |
| 0.00（纯 Engram） | 1.8450+ |

最优分配点位于 $\rho \approx 0.75 \text{--} 0.80$，即约 20%--25% 的稀疏参数预算分配给 Engram，其余分配给 MoE 专家。该结果表明条件记忆与条件计算具有互补性：Engram 承担高频知识的直接检索，释放 MoE 专家用于组合推理。

### 4.2.4 实验验证与性能分析

#### 1. 模型配置

所有模型共享相同的训练数据课程（262B tokens）、分词器（DeepSeek-v3, 128k 词表）和超参数配置：

表4-15：模型配置对比

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

#### 2. 基准测试性能

表4-16：预训练模型基准测试结果

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

#### 3. 长上下文性能

表4-17：多查询 Needle-in-a-Haystack (NIAH) 测试结果

| 模型 | 训练步数 | NIAH 准确率 |
|------|---------|------------|
| MoE-27B | 50k | 84.2% |
| Engram-27B | 41k | 89.5% |
| Engram-27B | 50k | 97.0% |

Engram 在长上下文检索任务上提升显著（84.2% → 97.0%），表明条件记忆模块有效缓解了长序列中的信息稀释问题。

#### 4. 计算开销分析

Engram 的确定性检索机制（检索索引完全由输入 token 序列决定，不依赖运行时隐状态）使得参数存储可完全与计算资源解耦。嵌入表可离线驻留于 CPU 主存（Host Memory），通过 PCIe 异步预取加载至 GPU。

表4-18：100B 参数 Engram 表卸载至 CPU 的推理性能

| 模型 | 无 Engram (tok/s) | 含 Engram (tok/s) | 吞吐量损失 |
|------|------------------|------------------|----------|
| Dense-4B | 9,031.62 | 8,858.28 | 1.9% |
| Dense-8B | 6,315.52 | 6,140.02 | 2.8% |

即使 Engram 表规模达 100B 参数（约 256GB），推理吞吐量损失控制在 3% 以内。

### 4.2.5 原理分析

#### 1. Engram 的注意力释放效应

通过引入软对齐指标（Soft Alignment Index）分析 Engram 对主干网络注意力模式的影响：

$$a_j = \frac{\sum_{i \in I_j} S_{i,j} \cdot i}{\sum_{i \in I_j} S_{i,j}}$$

其中 $S_{i,j}$ 为位置 $j$ 对位置 $i$ 的注意力权重，$I_j$ 为近距离窗口内的位置集合。

实验表明，加入 Engram 后，浅层 Attention 头的近距离注意力权重显著降低——即 Engram 接管了局部模式的重建任务，释放 Attention 用于建模远距离依赖和组合推理。这解释了 Engram 在推理类任务（BBH, GSM8K）上的意外增益：不是查表本身提升了推理能力，而是查表释放了推理所需的计算资源。

#### 2. 无限记忆扩展

嵌入表规模从 $2.58 \times 10^5$ 扩展至 $1.0 \times 10^7$ 条目（约 13B 参数）时，验证损失在对数坐标下呈近似线性下降趋势，表明条件记忆的容量扩展遵循幂律缩放定律：

$$L(M) \propto M^{-\beta}, \quad \beta > 0$$

这意味着在推理 FLOPs 不增加的前提下，仅通过扩大存储规模即可持续改善模型性能，为"知识放内存，推理放显存"的工程范式提供了理论支撑。

### 4.2.6 与 DeepSeek 稀疏架构的关系

Engram 构成 DeepSeek 稀疏化战略的第三条轴线：

表4-19：DeepSeek 稀疏三轴体系

| 稀疏轴线 | 引入版本 | 稀疏化对象 | 机制 | 复杂度 |
|---------|---------|-----------|------|-------|
| MoE | V2/V3 | FFN 参数 | 条件计算（动态路由） | O(top-k) |
| DSA | V3.2 | 注意力矩阵 | 稀疏注意力（Top-k KV） | O(k·n) |
| Engram | V4（预期） | 知识存储 | 条件记忆（静态查表） | O(1) |

三者的互补关系：DSA 减少远距离依赖的计算开销（只看最相关的 Top-k KV），Engram 消除近距离固定模式的重建成本（直接查表），MoE 为剩余的组合推理任务提供条件计算容量。

### 4.2.7 技术讨论

本节从数学严格性、与经典方法的联系以及当前局限性三个角度，对 Engram 的技术特点进行补充讨论。

##### 数学严格性说明

Engram 的可训练性源于嵌入查表操作的可微性：$\mathbf{E}[z]$ 本质为矩阵索引操作，等价于 one-hot 向量与嵌入矩阵的乘法，梯度可通过 straight-through estimator 或直接 sparse update 回传。在 PyTorch 实现中对应 `nn.Embedding` 的标准反向传播机制。

##### 与经典 N-gram 模型的联系

Engram 与经典 N-gram 模型共享"局部上下文查表"的核心思想，但通过三个现代化改造解决了数据稀疏性问题：(1) 多头哈希将冲突概率降至 $(1/M)^K$ 量级；(2) 上下文感知门控使碰撞噪声可被抑制；(3) 端到端训练使高频模式自然主导嵌入表内容。

##### 局限性

Engram 的检索机制基于固定窗口（2-gram, 3-gram），无法捕获非连续的远程模式。对于需要跨句推理的任务，仍完全依赖 Attention 机制。此外，当前实验规模限于 262B tokens 训练预算，更大规模下的 U 型曲线最优点是否发生漂移有待验证。

[^1]: Kaiming He 等人于2015年发表的论文《Deep Residual Learning for Image Recognition》，发表于 CVPR 2016。
[^2]: Ashish Vaswani 等人于2017年发表的论文《Attention Is All You Need》，发表于 NeurIPS 2017。
[^3]: Defa Zhu 等人于2024年发表的论文《Hyper-Connections》（arXiv:2409.19606），被 ICLR 2025 收录。
[^4]: Ziming Xie 等人于2025年发表的论文《mHC: Manifold-Constrained Hyper-Connections》（arXiv:2512.24880）。
[^5]: Richard Sinkhorn 于1964年发表的论文《A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices》；Sinkhorn 与 Paul Knopp 于1967年发表的论文《Concerning Nonnegative Matrices and Doubly Stochastic Matrices》。
[^6]: Jimmy Lei Ba 等人于2016年发表的论文《Layer Normalization》（arXiv:1607.06450）。
[^7]: Claude Shannon 于1948年发表的论文《A Mathematical Theory of Communication》；Peter F. Brown 等人于1992年发表的论文《Class-Based N-gram Models of Natural Language》。
[^8]: Xiaoguang Cheng 等人于2026年发表的论文《Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models》（arXiv:2601.07372）。

## 4.3 DualPath技术解析

前两节介绍的 mHC 与 Engram 分别从训练稳定性和架构稀疏化两个维度改进了 DeepSeek-V4 的模型架构本身。然而，模型架构的优化仅覆盖了从训练到部署的前半程——当模型进入大规模在线推理服务阶段，系统瓶颈往往从模型内部转移至推理基础设施层面。DualPath[^9] 正是针对这一推理基础设施瓶颈提出的解决方案：在多轮 Agent 推理场景下，KV Cache 的存储带宽成为制约系统吞吐量的核心因素，DualPath 通过跨 PE/DE 双路径并行加载机制突破该瓶颈。三项技术——mHC（稳定的深层信号传播）、Engram（解耦的知识存储）、DualPath（高效的推理基础设施）——共同构成 DeepSeek-V4 从模型架构到推理部署的完整技术底座。

### 4.3.1 研究背景与问题定义

#### 1. Agent 推理场景的特征

随着大模型 Agent 应用的普及，推理工作负载呈现出与传统单轮对话截然不同的特征。DualPath 团队基于 DeepSeek 生产环境数据，刻画了典型 Agent 推理场景的统计特性：

表4-20：Agent 推理场景统计特征

| 指标 | 32K MaxLen | 48K MaxLen | 64K MaxLen |
|------|-----------|-----------|-----------|
| 平均轮次 | 60 | 106 | 157 |
| 平均追加 token 数 | 608 | 474 | 429 |
| 平均生成 token 数 | 148 | 172 | 176 |
| 平均总 token 数 | 28,639 | 42,607 | 55,958 |
| 平均上下文长度 | 17,183 | 25,120 | 32,721 |

多轮交互的核心特征在于：每轮仅追加少量新 token（429~608），但需要携带完整的历史上下文（17K~33K token）。这意味着绝大部分 KV Cache 在相邻轮次之间是可复用的，实测 KV Cache 命中率高达 **98.7%**。

#### 2. 存储带宽瓶颈的三重成因

高命中率本应是系统优化的利好因素——缓存复用减少了重复计算。然而，在 Prefill-Decode 分离（PD 分离）架构下，98.7% 的命中率反而引发了严重的存储 I/O 瓶颈。该瓶颈源于三个相互强化的因素：

**因素一：高命中率产生的 I/O 密集型负载**

缓存命中意味着 KV Cache 需要从持久存储中加载而非重新计算。定义缓存-计算比（Cache-to-Compute Ratio）为每单位计算量所需的存储 I/O 量：

$$
R_{\text{cache}} = \frac{\text{KV Cache I/O (GB)}}{\text{Compute (PFLOP)}}
$$

对于 DeepSeek-V3.2，该比值约为 **22 GB/PFLOP**。在极端情况下（几乎全部命中），该比值可达 267 GB/PFLOP，远超硬件的 I/O-计算平衡点。

**因素二：硬件演进加剧失衡**

从 Ampere 到 Blackwell 架构，GPU 的计算能力增长远快于 I/O 带宽增长。I/O-计算比（单位计算量对应的 I/O 带宽）下降了 **14.4 倍**：

$$
\frac{(\text{I/O}/\text{Compute})_{\text{Ampere}}}{(\text{I/O}/\text{Compute})_{\text{Blackwell}}} \approx 14.4\times
$$

这意味着在新一代硬件上，存储带宽瓶颈将更加严峻。

**因素三：PD 分离架构下的网络利用率失衡**

在标准 PD 分离架构中，每个节点配备两类网络接口：用于计算通信的 CNIC（Compute NIC，InfiniBand 400Gbps）和用于存储访问的 SNIC（Storage NIC）。预填充引擎（PE）需要从存储加载命中的 KV Cache，其 SNIC 带宽被充分占用乃至饱和；而解码引擎（DE）的 KV Cache 主要通过 PE 传输获得，其 SNIC 处于闲置状态。这种结构性的网络利用率不均构成了系统级的资源浪费。

#### 3. 数据中心硬件拓扑

理解 DualPath 方案需要明确其面向的硬件拓扑：

- 每节点 8 个 Hopper GPU，通过 NVLink 互连
- 每 GPU 配备一个 400Gbps CNIC（InfiniBand）
- 每节点配备独立 SNIC 提供存储访问
- 计算网络（CNIC）与存储网络（SNIC）物理隔离

### 4.3.2 DualPath 架构设计

DualPath 的核心思想是：**利用 DE 端闲置的 SNIC 带宽，开辟第二条 KV Cache 加载路径，将存储 I/O 压力在 PE 和 DE 之间分摊**。

#### 1. PE 路径（传统路径）

PE 路径保留了标准 PD 分离架构的数据流，但仅承担部分 KV Cache 加载任务：

1. 命中 token 的 KV Cache 从持久存储加载至 PE 缓冲区
2. 每个注意力层计算前，该层对应的 KV Cache 从 PE 缓冲区转入 PE HBM
3. 计算完成后，所有 KV Cache（命中 + 未命中）转入 DE 缓冲区
4. 上述过程对 $n_{\text{layer}}$ 层重复执行，传输与计算重叠（Pipeline Overlap）

#### 2. DE 路径（新增路径）

DE 路径是 DualPath 的核心创新，利用 DE 端空闲的 SNIC 建立第二条存储访问通道：

1. 命中 token 的 KV Cache 从持久存储直接加载至 DE 缓冲区（绕过 PE）
2. 在 PE 进行预填充计算期间，对应层的 KV Cache 从 DE 缓冲区通过 CNIC 读入 PE HBM
3. 对 $n_{\text{layer}}$ 层重复执行
4. 预填充完成后，仅未命中部分的 KV Cache 从 PE 转入 DE 缓冲区

DE 路径的关键优势在于：PE 端无需等待全部 KV Cache 从本地存储加载完毕即可开始计算——缺失的部分由 DE 端通过 CNIC 补充传输，实现了存储 I/O 与跨节点通信的并行化。

#### 3. 块布局设计

为适配双路径传输模式，DualPath 设计了两种 KV Cache 数据块布局：

- **Full Block（全层块）**：包含所有注意力层的 KV Cache，用于存储后端的读写交互。全层粒度减少存储系统的 I/O 请求次数，适配大块顺序读写的高效模式。
- **Layer Block（单层块）**：仅包含单个注意力层的 KV Cache，用于 PE 到 HBM 以及 DE 缓冲区的传输。单层粒度实现了逐层流水化传输，避免一次性加载全部层造成的内存峰值。

#### 4. 无瓶颈运行条件

DualPath 的设计目标是消除所有网络和内存带宽瓶颈。设 $P$、$D$ 分别为预填充和解码节点数，$g$ 为每节点 GPU 数，$B$ 为单个 NIC 带宽，$s$ 为每节点存储 NIC 数，$M$ 为节点内存带宽（DRAM）。系统无瓶颈运行需同时满足以下约束：

**PE 端 CNIC 带宽约束**（DE 路径传输不超过 PE 端接收能力）：

$$
\frac{P}{D} \geq \frac{s}{g - s}
$$

**DE 端 CNIC 带宽约束**（DE 同时服务自身解码和向 PE 传输 KV Cache）：

$$
\frac{P}{D} \leq \frac{g - 2s}{s} \quad \text{或} \quad \frac{P}{D} \leq \frac{g - s}{2s}
$$

**DRAM 压力约束**（主机内存带宽不成为瓶颈）：

$$
\frac{P}{D} \leq \frac{M / (B \cdot s) - 3}{2}
$$

综合上述约束，DualPath 的无瓶颈运行条件为：

$$
\frac{s}{g - s} \leq \frac{P}{D} \leq \min\left\{\frac{g - 2s}{s},\ \frac{g - s}{2s},\ \frac{M/(B \cdot s) - 3}{2}\right\}
$$

对于典型配置（$g=8$, $s=1$, $M \approx 500\ \text{GB/s}$, $B \cdot s \approx 50\ \text{GB/s}$），代入得：

$$
\frac{1}{7} \leq \frac{P}{D} \leq \frac{7}{2}
$$

该范围覆盖了绝大多数实际部署场景，证明 DualPath 在工程上具有充分的 P/D 比灵活性。

### 4.3.3 CNIC 中心化流量管理

DualPath 的双路径设计要求 CNIC 同时承载推理通信和 KV Cache 传输两类流量，这对网络流量管理提出了新的要求。

#### 1. 流量统一化与隔离

DualPath 将所有 GPU 数据流量（包括本地 H2D/D2H 传输）统一通过配对 CNIC 进行，采用 GPUDirect RDMA 技术。这一设计简化了数据路径管理，但引入了流量争用问题。

为解决争用，DualPath 利用 InfiniBand 的虚拟通道（Virtual Lane, VL）机制实现流量隔离：

- **高优先级 VL**：承载推理通信流量（注意力计算、专家路由等），占约 **99%** 带宽
- **低优先级 VL**：承载 KV Cache 传输流量，使用剩余带宽

两个 VL 之间通过加权轮询仲裁（Weighted Round-Robin Arbitration）调度，确保推理通信的延迟不受 KV Cache 传输影响。

#### 2. CNIC 辅助 KV Cache 复制

传统的 GPU 内存拷贝通过 `cudaMemcpyAsync` 实现，提交延迟约 5~7 μs。DualPath 采用 CNIC RDMA Write 方式进行 KV Cache 的加载与存储：

**KV Cache 加载路径**：

$$
\text{存储后端} \xrightarrow{\text{SNIC}} \text{主机 DRAM} \xrightarrow{\text{CNIC RDMA Write}} \text{GPU HBM}
$$

**KV Cache 存储路径**：

$$
\text{GPU HBM} \xrightarrow{\text{CNIC}} \text{主机 DRAM} \xrightarrow{\text{SNIC}} \text{存储后端}
$$

RDMA Write 的提交延迟仅约 **1 μs**，相比 `cudaMemcpyAsync` 的 5~7 μs 降低了 5~7 倍，显著减少了流水线气泡。

### 4.3.4 自适应请求调度

双路径架构的引入使调度问题从单一维度（计算负载均衡）扩展为多维度（计算负载 + 存储 I/O + 跨节点带宽）。DualPath 设计了三级调度体系。

#### 1. 跨引擎调度（PE 选择）

跨引擎调度器将 PE 节点按状态分为三类：

| 类别 | 条件 | 调度策略 |
|------|------|---------|
| C1（过载） | $\text{tok}_e > \beta$ | 不分配新请求 |
| C2（短读队列） | $\text{read}_q \leq \alpha \wedge \text{tok}_e \leq \beta$ | 优先分配 |
| C3（长读队列） | 其余 | 次优先分配 |

其中 $\text{tok}_e$ 为引擎当前待处理 token 数，$\text{read}_q$ 为存储读队列长度，$\alpha$、$\beta$ 为阈值参数。调度逻辑为：不向 C1 类 PE 分配请求；优先选择 C2 类 PE；若无 C2 则选择 C3 类中 $\text{tok}_e$ 最小的 PE。

#### 2. DE 调度（两阶段）

DE 调度采用两阶段决策：

**Phase 1（全局层面）**：在所有 DE 组中，选择 $\text{tok}_e$ 最小的组，实现全局负载均衡。

**Phase 2（组内层面）**：在选定组内，计算可调度请求集 $R$，设定阈值：

$$
Z = 1.05 \times \frac{\sum_{r \in R} \text{len}_r + \sum_{e} \text{tok}_e}{|E|}
$$

其中 $\text{len}_r$ 为请求 $r$ 的序列长度，$|E|$ 为组内引擎数。因子 1.05 提供 5% 的负载余量，防止调度振荡。

#### 3. 引擎内调度（分层时间估计）

引擎内调度的核心挑战在于：不同请求的注意力计算时间差异巨大（取决于 KV Cache 长度），需要精确估计每个前向批次的执行时间。

DualPath 采用分层时间估计方法：每个前向批次的请求描述为 $(\text{cached}, \text{bsz})$ 对，计算注意力层的理论计算量，据此估计执行时间。请求按 FIFO 顺序添加，通过二分查找确定最优批大小 $\text{bsz}'$ 进行分块预填充（Chunked Prefill），在延迟约束下最大化批处理吞吐量。

### 4.3.5 实验验证与性能分析

#### 1. 实验配置

DualPath 基于 DeepSeek 内部推理框架实现，集成 FlashMLA、DeepGEMM 和 DeepEP 组件，存储后端采用 3FS（无内部 DRAM 缓存），I/O 层使用 io_uring 式内核绕过技术。总实现规模约 5,000 行代码。

测试平台配置：每节点 8 个 Hopper GPU，8 个 400Gbps RDMA NIC（InfiniBand）+ 1 个 SNIC。

测试模型涵盖三种典型架构：

表4-21：测试模型与部署配置

| 模型 | 参数量 | 架构特征 | P/D 配比 | 并行策略 |
|------|-------|---------|---------|---------|
| DeepSeek V3.2 | 660B | MoE + 稀疏注意力 | 2P4D | EP + DP |
| DeepSeek 27B | 27B | 内部缩小版 | 1P1D | — |
| Qwen2.5-32B | 32B | 稠密 + GQA | 1P2D | — |

基线系统包括：
- **SGL(MC)**：集成 HiCache、Mooncake 和 3FS 后端的 SGLang
- **Basic**：未修改的内部推理框架
- **Oracle**：绕过所有磁盘读取、D2H/H2D 传输和跨 PD KV 传输的理想上界

#### 2. 核心性能结果

表4-22：离线与在线吞吐量提升

| 场景 | 指标 | 提升倍数 |
|------|------|---------|
| 离线推理 | 最大吞吐量 | 最高 **1.87x** |
| 在线服务 | 平均吞吐量 | 平均 **1.96x** |

#### 3. 消融实验

DualPath 各组件的贡献通过逐步叠加实验量化：

表4-23：组件消融（JCT 相对 Basic 的改善）

| 配置 | JCT 平均减少 |
|------|------------|
| + 分层预填充（LayerKV） | 17.21% |
| + 双路径传输 | 38.19% |
| + 自适应调度 | **45.62%** |

分层预填充通过按层分配和释放 KV Cache 缓解 HBM 容量瓶颈，贡献 17.21% 的延迟改善。双路径传输在此基础上进一步减少 20.98% 的延迟（累计 38.19%），证实了存储 I/O 分摊的有效性。自适应调度再贡献 7.43%（累计 45.62%），通过精细的负载均衡进一步释放系统潜力。

#### 4. 负载均衡效果

表4-24：负载均衡指标（Max/Avg 比）

| 指标 | 轮询调度 | DualPath 调度 |
|------|---------|-------------|
| 存储 NIC 负载 | 1.53 | **1.18** |
| 注意力执行时间 | — | **≈1.06** |

存储 NIC 的负载不均衡度从 1.53 降至 1.18，注意力执行时间的 Max/Avg 比控制在约 1.06，表明调度器有效实现了多维度的负载均衡。

#### 5. 大规模可扩展性

DualPath 在 144 节点（1,152 GPU）规模上验证了近线性扩展能力：

表4-25：扩展性测试结果

| 配置 | Agent 数 | 耗时 (s) | APS | TTFT |
|------|---------|---------|-----|------|
| 2P4D | 2K | 3,167 | 0.4 | 基准 |
| 48P96D | 48K | 3,201 | 8.8 | 基本保持 |

从 2P4D 扩展至 48P96D（24 倍节点），吞吐量从 0.4 APS 提升至 8.8 APS（**22 倍**），处理 48K Agent 的总耗时仅增加 1.1%（3,167s → 3,201s），实现了近完美的线性扩展。TTFT（Time To First Token）在扩展过程中基本保持不变，调度器 CPU 开销低于 10 核。

#### 6. 工作集与资源规划

DualPath 的存储工作集（Working Set）与请求到达率（APS）之间呈现非线性关系：

| APS | 工作集 |
|-----|-------|
| 0.1 | 69 GB |
| 0.45 | 681 GB |

当工具调用延迟增大 $r$ 倍时，工作集按 $r^2$ 扩展，这一二次方关系对存储系统的容量规划具有重要指导意义。

### 4.3.6 技术讨论

#### 1. 与 PD 分离架构的关系

DualPath 并非取代 PD 分离架构，而是在其基础上的增量优化。PD 分离将计算密集型的预填充与内存密集型的解码分派至不同硬件池，这一设计在单轮推理场景下工作良好。DualPath 识别出 PD 分离在多轮 Agent 场景下的结构性缺陷（存储网络利用率不均），通过跨角色的网络资源共享加以修正。

#### 2. 与分层预填充的协同

分层预填充（LayerKV）按层粒度管理 KV Cache 的生命周期，避免一次性加载全部层导致的 HBM 峰值溢出。DualPath 的 Layer Block 布局与分层预填充天然契合：逐层传输的粒度恰好匹配逐层计算的流水线节奏，使得传输与计算的重叠率接近理论最优。消融实验中分层预填充贡献的 17.21% JCT 改善即源于此协同效应。

#### 3. 无瓶颈条件的工程含义

公式 $s/(g-s) \leq P/D \leq \min\{(g-2s)/s, (g-s)/(2s), (M/(Bs)-3)/2\}$ 给出了系统无瓶颈运行的精确约束。对于当前典型配置（$g=8$, $s=1$），$P/D$ 的可行范围为 $[1/7, 7/2]$，这意味着从极端的解码优先（$P/D = 1/7$）到极端的预填充优先（$P/D = 3.5$）均可无瓶颈运行。但随着未来硬件的 I/O-计算比继续下降（如 Blackwell 架构），$s$ 的有效值可能需要增加，收窄可行范围。

#### 4. 与相关工作的对比

DualPath 在技术路线上与若干并行工作形成互补：

表4-26：相关工作对比

| 工作 | 核心思路 | 与 DualPath 的差异 |
|------|---------|------------------|
| Mooncake | 分布式 DRAM 池 | 依赖大规模 DRAM，DualPath 利用现有 SNIC |
| TokenLake | 段级前缀缓存 | 缓存粒度较粗，不适配逐层流水 |
| Strata | 分层存储 + GPU 辅助 I/O | 需要 GPU 参与 I/O，DualPath 通过 CNIC 卸载 |
| KVPR | I/O 感知 KV Cache 部分重计算 | 以重计算换带宽，DualPath 以网络换带宽 |
| TailorKV | 分层量化 | 压缩 KV Cache 体积，与 DualPath 正交可组合 |

DualPath 的独特价值在于：不修改模型架构、不增加计算开销、不引入近似误差，仅通过重新编排数据流路径即可获得接近 2 倍的吞吐量提升。

#### 5. 局限性

1. **硬件拓扑依赖**：DualPath 的双路径设计依赖于 PE/DE 节点均配备独立 SNIC 的拓扑假设。在 SNIC 与 CNIC 共享物理端口的部署环境中，DE 路径的带宽收益将大幅缩减。
2. **P/D 比约束**：尽管可行范围较宽（$1/7$ 至 $7/2$），极端的 P/D 比仍可能触发瓶颈。实际部署需根据工作负载特征动态调整 P/D 比。
3. **工作集的二次方扩展**：工具调用延迟增大 $r$ 倍导致工作集 $r^2$ 扩展，在高延迟外部 API 场景下可能对存储系统容量提出严苛要求。

[^9]: Yongtong Wu, Shaoyuan Chen, Yinmin Zhong 等人于2026年发表的论文《DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference》（arXiv:2602.21548v2）。
