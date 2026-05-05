### 3.x Muon：基于矩阵正交化的优化器

#### 3.x.1 研究背景：优化器的角色

##### 3.x.1.1 参数更新的基本框架

神经网络训练的核心循环可形式化为：给定损失函数 $L(\theta)$，在每一步 $t$ 根据梯度信息更新参数 $\theta$。优化器（Optimizer）定义了从梯度到参数更新量的映射规则：

$$
\theta_t = \theta_{t-1} - \eta \cdot \mathcal{U}(\nabla_\theta L_t, \text{history})
$$

其中 $\eta$ 为学习率，$\mathcal{U}(\cdot)$ 为优化器定义的更新规则。不同优化器的区别在于 $\mathcal{U}$ 的设计：SGD 直接使用梯度，Adam 引入逐元素自适应，Muon 则在矩阵层面施加几何约束。

模型结构决定"它能表达什么"，训练数据决定"它见过什么"，优化器决定"它怎么从随机初始化走向收敛"。在 1.6T 参数规模的模型中，优化器的选择直接影响训练效率和稳定性。

#### 3.x.2 SGD 与动量法

##### 3.x.2.1 随机梯度下降（SGD）

SGD（Stochastic Gradient Descent）是最基本的参数更新规则（Robbins & Monro, 1951）：

$$
W_t = W_{t-1} - \eta \nabla_W L_t
$$

梯度 $\nabla_W L_t$ 指向损失上升最快的方向，取负号即为下降方向。SGD 的优点是简单直接，缺点是：（1）所有参数共用同一学习率；（2）对单步梯度噪声敏感，更新方向容易振荡。

##### 3.x.2.2 动量法（Polyak, 1964）

Polyak（1964）提出的 heavy ball method 引入动量缓冲区，对历史梯度做指数移动平均：

$$
v_t = \mu v_{t-1} + \nabla_W L_t
$$
$$
W_t = W_{t-1} - \eta v_t
$$

其中 $\mu \in [0, 1)$ 为动量系数（通常取 0.9）。动量法保留历史方向信息，使更新方向更平滑，不易被单次噪声带偏。

Nesterov（1983）进一步改进为 Nesterov Accelerated Gradient（NAG），先沿动量方向"预看一步"再计算梯度，具有更好的收敛速率。后文将看到 Muon 也使用了 Nesterov 动量技巧。

#### 3.x.3 Adam 与 AdamW

##### 3.x.3.1 Adam：逐元素自适应学习率

Adam（Adaptive Moment Estimation, Kingma & Ba, 2014）为每个参数维护两个统计量：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{（一阶矩估计：梯度均值）}
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{（二阶矩估计：梯度均方值）}
$$

经偏差修正后，更新规则为：

$$
W_t = W_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

其中 $\hat{m}_t = m_t / (1 - \beta_1^t)$，$\hat{v}_t = v_t / (1 - \beta_2^t)$。

核心机制：分母 $\sqrt{\hat{v}_t}$ 对梯度波动大的参数施加更大的抑制——梯度稳定的参数步子大，梯度抖动的参数步子小。所有操作均为逐元素（element-wise），即矩阵中每个标量参数独立计算自己的动量和步长。

##### 3.x.3.2 AdamW：解耦权重衰减

Loshchilov & Hutter（2017）指出，原版 Adam 将权重衰减（weight decay）混入梯度再经过自适应步长调制，导致正则化效果被扭曲。AdamW 将权重衰减从自适应更新中解耦：

$$
W_t = W_{t-1} \cdot (1 - \eta\lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

其中 $\lambda$ 为权重衰减系数。解耦后，每个参数被拉向零的力度均匀一致，不受自适应步长影响。

AdamW 自 2017 年以来一直是大模型训练的事实标准优化器（GPT、LLaMA、DeepSeek-V3 均采用），但其根本局限在于：**它将权重矩阵视为一袋独立标量进行逐元素更新，忽略了矩阵作为线性变换的整体几何结构。**

#### 3.x.4 Muon 的核心创新：矩阵级正交化更新

##### 3.x.4.1 从逐元素到矩阵级

Muon（MomentUm Orthogonalized by Newton-Schulz）由 Keller Jordan（2024）提出，先在 NanoGPT speedrun 排行榜上展现优势，随后在 Moonshot Kimi K2（2025）和 DeepSeek V4（2026）等超大规模模型上得到验证。

Muon 的核心思想是：Transformer 的绝大多数参数以二维权重矩阵的形式存在（注意力投影、FFN 升降维、MoE 专家权重）。既然参数本体是矩阵，更新规则也应当在矩阵层面设计，而非将矩阵拆散为独立标量。

##### 3.x.4.2 核心操作流程

Muon 的更新过程分为四步：

1. **计算梯度矩阵** $G_t = \nabla_W L_t(W_{t-1})$
2. **动量平滑** $M_t = \mu M_{t-1} + G_t$
3. **正交化** $M' = \text{Orthogonalize}(\mu M_t + G_t)$（Nesterov 技巧）
4. **更新权重** $W_t = W_{t-1} \cdot (1 - \eta\lambda) - \eta \cdot M'$

被正交化的对象是动量矩阵 $M$（梯度平滑后的结果），而非权重矩阵 $W$。Muon 改造的是"用来更新的工具"的几何形状，不是目标本身。

##### 3.x.4.3 正交化的数学含义

对任意矩阵 $M \in \mathbb{R}^{n \times m}$，其奇异值分解（SVD）为：

$$
M = U \Sigma V^\top
$$

其中 $U \in \mathbb{R}^{n \times r}$，$V \in \mathbb{R}^{m \times r}$ 为正交矩阵（$r = \min(n, m)$），$\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r)$ 为奇异值对角矩阵。

奇异值 $\sigma_i$ 描述矩阵在第 $i$ 个奇异方向上的"强度"。若奇异值分布极度不均（如 $\sigma_1 = 100, \sigma_2 = 0.01$），则更新能量高度集中于第一个方向，其余方向几乎未被更新。

Muon 的正交化操作定义为：

$$
M' = U V^\top
$$

即**保留方向信息 $U, V$，丢弃强度信息 $\Sigma$**。正交化后的矩阵 $M'$ 满足：所有奇异值均为 1，更新能量在各方向上均匀分布。

一句话概括：**相信梯度告诉你的方向，但拒绝梯度给的各方向上的不同强度。**

##### 3.x.4.4 Newton-Schulz 迭代：高效近似正交化

精确 SVD 的计算复杂度为 $O(nm \cdot \min(n,m))$，对大模型训练中的海量矩阵而言代价过高。Muon 采用 Newton-Schulz 迭代（Schulz, 1933）——一种基于矩阵多项式的迭代方法，通过有限次矩阵乘法逼近正交化结果。

设 $M_0 = M / \|M\|_F$（Frobenius 范数归一化），迭代公式为：

$$
M_{k+1} = a M_k + b (M_k M_k^\top) M_k + c (M_k M_k^\top)^2 M_k
$$

其中 $(a, b, c)$ 为预定系数。该迭代使 $M_k$ 的奇异值逐步趋近 1，即 $M_k \to UV^\top$。

核心优势：整个计算过程仅涉及矩阵乘法，而 GPU 硬件对矩阵乘法有极高的吞吐率（GEMM 单元）。这使得 Newton-Schulz 迭代在实际训练中的开销远低于精确 SVD。

#### 3.x.5 DeepSeek V4 的 Muon 实现

##### 3.x.5.1 完整算法（Algorithm 1）

DeepSeek V4 技术报告给出的 Muon 算法如下：

---

**算法 1：Muon 优化器（DeepSeek V4 实现）**

**输入**：学习率 $\eta$，动量系数 $\mu$，权重衰减 $\lambda$，RMS 缩放因子 $\gamma = 0.18$

**for** each training step $t$ **do**

$\quad$ **for** each logically independent weight $W \in \mathbb{R}^{n \times m}$ **do**

$\quad\quad$ $G_t = \nabla_W L_t(W_{t-1})$ $\quad\quad\quad\quad\quad\quad\quad\quad$ // 计算梯度

$\quad\quad$ $M_t = \mu M_{t-1} + G_t$ $\quad\quad\quad\quad\quad\quad\quad\quad\,$ // 累积动量缓冲

$\quad\quad$ $O'_t = \text{HybridNewtonSchulz}(\mu M_t + G_t)$ $\quad$ // Nesterov 技巧 + 混合 NS

$\quad\quad$ $O_t = O'_t \cdot \sqrt{\max(n, m)} \cdot \gamma$ $\quad\quad\quad\quad\,$ // RMS 缩放

$\quad\quad$ $W_t = W_{t-1} \cdot (1 - \eta\lambda) - \eta O_t$ $\quad\quad\quad\quad$ // 权重衰减 + 更新

---

**Nesterov 技巧**：正交化的输入不是 $M_t$ 本身，而是 $\mu M_t + G_t$——这等价于在动量方向上"预看一步"再做正交化，与 Nesterov 加速梯度的思路一致。

**RMS 缩放因子** $\gamma = 0.18$：正交化后矩阵的 RMS（均方根）值为 $1/\sqrt{\max(n,m)}$，乘以 $\sqrt{\max(n,m)} \cdot \gamma$ 后，更新的 RMS 被校准到 $\gamma = 0.18$，统一了不同形状矩阵的更新幅度。

##### 3.x.5.2 Hybrid Newton-Schulz 迭代细节

V4 采用两阶段混合 Newton-Schulz 迭代，共 10 步：

**初始化**：

$$
M_0 = M / \|M\|_F
$$

**迭代公式**：

$$
M_{k+1} = a M_k + b (M_k M_k^\top) M_k + c (M_k M_k^\top)^2 M_k
$$

**两阶段系数**：

| 阶段 | 步数 | $(a, b, c)$ | 目标 |
|------|------|-------------|------|
| 第一阶段 | 8 步 | $(3.4445, -4.7750, 2.0315)$ | 快速收敛，使奇异值接近 1 |
| 第二阶段 | 2 步 | $(2, -1.5, 0.5)$ | 精确稳定，使奇异值恰好为 1 |

两阶段设计的工程考量：第一阶段系数针对快速收敛优化（大步逼近），第二阶段系数保证数值精度（小步锁定）。10 步总迭代在 GPU 上实现为 10 次 GEMM 操作序列，开销可控。

##### 3.x.5.3 参数分组策略

V4 的 1.6T 参数按以下规则分配优化器：

表1：DeepSeek V4 优化器分配策略

| 参数类别 | 优化器 | 占比 | 原因 |
|---------|--------|------|------|
| 注意力投影矩阵（Q/K/V/O） | Muon | — | 二维权重矩阵 |
| FFN / MoE 专家权重 | Muon | — | 二维权重矩阵 |
| mHC 动态投影矩阵 | Muon | — | 二维权重矩阵 |
| **以上合计** | **Muon** | **绝大多数参数** | — |
| Embedding / Prediction Head | AdamW | — | 离散字典 + weight tying |
| RMSNorm 缩放系数 | AdamW | — | 向量参数，无矩阵形状 |
| mHC 静态偏置与门控因子 | AdamW | — | 标量/向量参数 |
| **以上合计** | **AdamW** | **少数参数** | — |

##### 3.x.5.4 关于 QK-Clip

V4 的 Muon 实现**不使用 QK-Clip**。原因在于 V4 架构中已有两层数值保护：

1. **mHC**（流形约束超连接）：将无约束 HC 中可能跨层累乘失控的残差映射压到双随机矩阵集合内；在 mHC 论文实验口径下，整体信号增益从 $10^3 \sim 10^5$ 级别压到约 1.6
2. **RMSNorm on Q/K entries**：对注意力层的 Query 和 Key 做 RMS 归一化，直接控制注意力分数的数值范围

两者从架构端解决了注意力数值失控问题，使优化器端无需额外裁剪。

#### 3.x.6 为什么 Embedding 不适合 Muon

Embedding 矩阵是整个模型中唯一明确使用 AdamW 的二维权重矩阵，原因有两条：

**第一，离散字典结构（不光滑）。** Embedding 矩阵的每一行对应一个 token 的向量表示。行与行之间没有连续关系——"猫"和"狗"的向量可能相近，"猫"和"且"则完全无关。整个矩阵不是一个光滑的空间变换，而是离散点的拼接。Muon 的正交化假设底层映射是光滑的、各方向曲率差别不大。对离散字典而言，该假设不成立。

**第二，两头收梯度（weight tying）。** 多数大模型将 embedding 矩阵与输出预测头共享权重。反向传播时，同一矩阵从两端接收方向完全不同的梯度信号：

$$
\nabla_{\text{total}} = \nabla_{\text{output head}} + \nabla_{\text{input embedding}}
$$

前者来自"预测哪个 token"的分类损失，后者来自"输入表示该如何调整"。若对这两股矛盾信号做正交化（拉平强度），可能抹掉各自有意义的方向偏重。AdamW 的逐元素自适应机制恰好能分别消化来自两端的不同信号。

#### 3.x.7 Muon 的局限性与工程应对

##### 3.x.7.1 正交化抹掉强度信息

Muon 的隐含假设是：**损失地形在各奇异方向上曲率差别不大。** 正交化将所有方向的更新强度拉平为 1，等于假设损失函数是一个各向同性的光滑流形，没有"尖刺"。

若损失地形确实存在少数极其重要的方向（如 Grokking 任务中有效维度从 78 骤降至 8），正交化会拖慢模型发现这些关键方向的速度。

大模型学习自然语言时，有效维度高、变化缓慢（实测幅度约 20-30%），光滑假设近似成立。这解释了 Muon 在大模型上的成功——参数越多、维度越高，损失地形越倾向光滑。

##### 3.x.7.2 Kimi K2 的 MuonClip 经验

Moonshot 在 Kimi K2 训练中（标准 Transformer 架构，无特殊残差设计），使用原版 Muon 时注意力分数最大值飙升至 1000 以上，导致 loss spike 和训练发散。其解法为在优化器端对注意力层 QK 加裁剪（MuonClip），之后万亿参数规模训练实现零 loss spike。

##### 3.x.7.3 V4 的架构端解法

DeepSeek V4 同时使用 Muon 和 HC（多路残差连接），叠加了两个数值风险。V4 的策略是从架构端一次性解决：

- **mHC**：用双随机矩阵约束保证残差映射 $B_l$ 的谱范数 $\leq 1$，避免 HC 式跨层累乘放大；实验口径下整体信号增益约压到 $\sim 1.6\times$
- **RMSNorm on Q/K**：直接控制注意力分数的数值范围

表2：三种方案的对比

| 系统 | 架构 | 优化器 | 数值问题来源 | 解决策略 |
|------|------|--------|------------|---------|
| Kimi K2 | 标准 Transformer | Muon | Muon 正交化 → 注意力数值失控 | 优化器端裁剪（MuonClip） |
| 字节 HC | 多路残差（HC） | — | HC 信号放大 $\sim 3000\times$ | — |
| DeepSeek V4 | mHC | Muon（原版） | Muon + HC 风险叠加 | 架构端压制（mHC + RMSNorm） |

V4 保持原版 Muon 不加任何裁剪，代价是需要在架构设计中预先解决数值稳定性问题。mHC 同时压住了 HC 的放大问题和 Muon 的数值问题。

#### 3.x.8 本节小结

Muon 代表了优化器设计从"逐元素自适应"到"矩阵几何优化"的范式跃迁。其演化线索如下：

| 阶段 | 代表 | 更新粒度 | 核心进步 |
|------|------|---------|---------|
| 1 | SGD（Robbins & Monro, 1951） | 全局统一 | 沿梯度下降 |
| 2 | Momentum（Polyak, 1964） | 全局统一 | 平滑噪声 |
| 3 | Adam/AdamW（Kingma & Ba, 2014; Loshchilov & Hutter, 2017） | 逐元素 | 自适应步长 |
| 4 | Muon（Keller Jordan, 2024） | 矩阵级 | 正交化更新 |

Muon 的核心贡献包括：

1. **算法层面**：对动量矩阵做正交化（$M = U\Sigma V^\top \to M' = UV^\top$），保留方向、拉平强度，使更新能量在各奇异方向均匀分布
2. **工程层面**：用 Newton-Schulz 迭代（纯矩阵乘法）替代精确 SVD，利用 GPU GEMM 单元的高吞吐率，使正交化在大规模训练中可行
3. **系统层面**：在 DeepSeek V4 中，绝大多数矩阵参数使用 Muon，Embedding、Prediction Head、RMSNorm 权重以及 mHC 静态偏置和门控因子保留 AdamW；配合 mHC 和 RMSNorm 从架构端解决数值问题，不需要优化器端的额外裁剪

Muon 的适用条件是明确的：参数以大尺寸二维矩阵为主体，且损失地形近似光滑。大规模语言模型恰好满足这两个条件——这解释了 Muon 在超大模型上的成功，也界定了它不适用的场景（离散字典、低维任务、尖刺地形）。

---

**参考文献**

- Robbins, H. & Monro, S. (1951). A Stochastic Approximation Method. Annals of Mathematical Statistics, 22(3), 400-407.
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5), 1-17.
- Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence $O(1/k^2)$. Doklady AN USSR, 269, 543-547.
- Schulz, G. (1933). Iterative Berechnung der reziproken Matrix. ZAMM, 13(1), 57-59.
- Kingma, D. P. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980. ICLR 2015.
- Loshchilov, I. & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv:1711.05101. ICLR 2019.
- Keller Jordan et al. (2024-2025). Muon: An optimizer for hidden layers in neural networks. https://github.com/KellerJordan/Muon
- DeepSeek V4 Technical Report (2026). https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf
