### 3.x CSA/HCA：混合压缩注意力与 KV Cache 的序列压缩

#### 3.x.1 研究背景：注意力机制的演化简史

##### 3.x.1.1 问题的本质——KV Cache 的存储与计算瓶颈

Transformer 推理时的核心瓶颈在于 KV cache。解码（decode）阶段每生成一个新 token，仅有当前位置的 Query 在工作（$1 \times N$ 向量），但需要访问序列中全部 $N$ 条 Key 和 Value。这些 Key/Value 必须持久存储于显存中，随上下文长度线性增长，构成推理阶段的显存与带宽瓶颈。

以 V4-Pro 的 61 层架构、50000 token 上下文为例，标准多头注意力（MHA）需存储约 81.5 GiB 的 KV cache。注意力机制过去八年的全部演化，本质上都在回答同一个问题：**如何让 KV cache 更小、更快、更省？**

##### 3.x.1.2 演化路径概览

注意力机制的演化形成了一条清晰的技术脉络，每一代都在解决前一代遗留的具体问题：

| 阶段 | 年份 | 核心策略 | 压缩维度 | 遗留问题 |
|------|------|---------|---------|---------|
| MHA | 2017 | 多头独立 KV | 无压缩（基线） | 存储随头数线性增长 |
| MQA | 2019 | 全模型共享 1 组 KV | 头数：$n_h \to 1$ | 表达力显著下降 |
| GQA | 2023 | 分组共享 KV | 头数：$n_h \to g$ | 头数砍到极限 |
| MLA | 2024 | 低秩投影到 latent | 维度：$d \to d_c$ | 条数未减少 |
| DSA | 2025 春 | 推理时选 top-k | 计算量（非存储） | 训推不一致，不省存储 |
| NSA | 2025-02 | 原生可训练稀疏，三分支并行 | 计算量 + 存储 | 实验室规模（27B） |
| CSA/HCA | 2026 | NSA 的层级化工程落地 | 条数：$N \to N/m$ | 当前最优解 |

##### 3.x.1.3 MHA 到 MLA：头数与维度的压缩

**MHA**（Vaswani et al., 2017）的标准注意力计算为：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

$n_h$ 个注意力头各自维护独立的 Key 和 Value，每个 token 每层需存储 $2 \times n_h \times d_h$ 个参数（$d_h$ 为每头维度）。

**MQA**（Shazeer, 2019）将 KV 头数从 $n_h$ 压缩至 1，存储降至 $1/n_h$，但表达力显著衰减。**GQA**（Ainslie et al., 2023）折中为 $g$ 组共享，提供可调旋钮。

**MLA**（DeepSeek-V2, 2024）开辟了全新维度——不再主要靠砍 KV 头数，而是砍每条 KV 的维度。其核心思想是将 Key 和 Value 联合压缩为一个低维 latent 向量：

$$
\mathbf{c}_t = \mathbf{h}_t \mathbf{W}_{\text{down}}, \quad \mathbf{W}_{\text{down}} \in \mathbb{R}^{d \times d_c}
$$

其中 $d$ 是隐藏维度，$d_c$ 是压缩后的 latent 维度。推理时只存 $\mathbf{c}_t$，通过将还原矩阵吸收进 Query 投影和输出投影，**latent 从头到尾不需要被解压为显式的 K 和 V**。V4 的 CSA/HCA 主注意力继续沿用 shared key-value MQA 的思想：压缩 KV entry 同时作为 attention key 和 value。

MLA 将存储从标准 MHA 的 81.5 GiB 压缩至约 2.9 GB（50K 上下文，61 层），但**条数仍与上下文长度一一对应**——50000 个 token 仍是 50000 条 latent。

##### 3.x.1.4 DSA 与 NSA：序列维度的稀疏探索

**DSA**（DeepSeek-V3.2, 2025）首次触及序列维度：推理时用轻量 Indexer 从全部 KV cache 中选出 top-k 条，注意力仅在选中的子集上计算。但 DSA 的两个根本局限是：（1）KV cache 全量存储不变，只省计算不省显存；（2）训练时仍用完整注意力，推理时才切换为稀疏——"训完套眼镜"导致训推不一致。

**NSA**（Yuan et al., 2025, arXiv 2502.11089）解决了训推一致性问题，提出**原生可训练的稀疏注意力**。其架构为三分支并行：压缩全看（Compressed）、选块精读（Selected）、滑动窗口（Sliding Window），通过学习门控加权合并。NSA 在 27B 模型上验证了"原生稀疏不弱于全注意力"的关键结论，但工程规模仅为实验室级别。

V4 的 CSA/HCA 是 NSA 的工程落地——从 27B 扩展到 1.6T，从三分支并行改造为层间交替。

#### 3.x.2 CSA：压缩稀疏注意力

##### 3.x.2.1 压缩机制：学习的加权池化

CSA（Compressed Sparse Attention）的核心操作是将每 $m = 4$ 条连续的 KV latent 压缩为 1 条，通过学习的加权池化实现。

设输入序列的隐藏状态为 $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_N]^\top \in \mathbb{R}^{N \times d}$。将序列按步长 $m$ 分为 $\lfloor N/m \rfloor$ 个块，第 $i$ 个块包含 token $\{(i-1)m+1, \ldots, im\}$。

压缩器（Compressor）对块内的 $m$ 条 token 执行两步操作：

**步骤 1：KV 投影与压缩权重计算**

$$
\mathbf{C}_j^a = \mathbf{h}_j \mathbf{W}^{aKV}, \quad \mathbf{C}_j^b = \mathbf{h}_j \mathbf{W}^{bKV}
$$

$$
Z_j^a = \mathbf{h}_j \mathbf{W}^{aZ}, \quad Z_j^b = \mathbf{h}_j \mathbf{W}^{bZ}
$$

其中上标 $a$ 表示"当前块"角色，上标 $b$ 表示"重叠到下一块"的角色（详见 3.x.2.2 节）。$\mathbf{W}^{aKV}, \mathbf{W}^{bKV}, \mathbf{W}^{aZ}, \mathbf{W}^{bZ} \in \mathbb{R}^{d \times c}$，即压缩权重不是单个标量，而是按 KV entry 的 $c$ 个维度分别给权重。

**步骤 2：跨块 Softmax 归一化与加权求和**

对于第 $i$ 个压缩块，令前一块的 $m$ 个 token 提供 $b$ 角色贡献，当前块的 $m$ 个 token 提供 $a$ 角色贡献。将 $2m$ 个权重拼接后做 softmax：

$$
[S_1^a, \ldots, S_m^a;\; S_1^b, \ldots, S_m^b] = \text{Softmax}_{\text{row}}\!\left([Z_1^a + B_1^a, \ldots, Z_m^a + B_m^a;\; Z_1^b + B_1^b, \ldots, Z_m^b + B_m^b]\right)
$$

其中 $B_j^a, B_j^b$ 为可学习的位置偏置（Absolute Positional Embedding），编码块内位置信息。

**步骤 3：生成压缩后的 KV entry**

$$
\mathbf{C}_i^{\text{Comp}} = \sum_{j=1}^{m} S_j^a \odot \mathbf{C}_j^a + \sum_{j=1}^{m} S_j^b \odot \mathbf{C}_j^b
$$

其中 $\odot$ 为逐元素乘法（广播标量权重到向量各维度）。压缩后的 $\mathbf{C}_i^{\text{Comp}} \in \mathbb{R}^{d_c}$ 即为第 $i$ 个块的 KV cache 条目。

**关键设计：Shared Key-Value MQA**。压缩后的 KV entry 同时充当 Key 和 Value——即 $\mathbf{K}_i^{\text{Comp}} = \mathbf{V}_i^{\text{Comp}} = \mathbf{C}_i^{\text{Comp}}$。这与 MLA 的 latent 联合存储思想一脉相承：既然 K 和 V 都从同一份 latent 还原，不如直接共享。

##### 3.x.2.2 重叠压缩：焊接块边界

CSA 的 $m = 4$ 意味着每 4 个 token 被硬性划入同一个压缩块。块边界处的语义连续性可能被割裂——例如 "transform" 和 "er" 分属相邻块，各自的语义被截断。

重叠压缩（Overlapping Compression）的解决方案：每个压缩块不仅看本块的 $m$ 个 token，还**同时看前一个块的 $m$ 个 token**，总共 $2m = 8$ 个 token 参与加权。但压出来的仍然只有 1 条 KV entry。

具体实现中，每个 token 同时计算两组投影：$(\mathbf{C}^a, Z^a)$ 供**本块使用**，$(\mathbf{C}^b, Z^b)$ 供**下一个块作为前文使用**。Softmax 在 $2m$ 个位置上联合归一化，确保权重和为 1。

表1：重叠压缩 vs 非重叠压缩

| 特性 | 非重叠（HCA） | 重叠（CSA） |
|------|-------------|------------|
| 每块参与 token 数 | $m$ | $2m$ |
| 投影维度 | $d_c$ | $2 \times d_c$（$a$ 和 $b$ 各一份） |
| 块边界信息 | 可能割裂 | 通过重叠窗口焊接 |
| 适用条件 | 大块（$m$ 大，边界影响小） | 小块（$m$ 小，边界影响大） |

##### 3.x.2.3 Lightning Indexer：FP4 加速的 Top-k 选择

压缩后的 CSA KV cache 仍有 $N/m$ 条（50K 上下文约 12500 条）。模型并非全部使用，而是通过 Lightning Indexer 选出最相关的 top-$k$ 条进行注意力计算。V4-Pro 的 $k=1024$，V4-Flash 的 $k=512$。

Lightning Indexer 是一条独立的打分路径。它复用 CSA 的压缩形式生成 indexer keys，再用低秩 query 投影和多头加权打分选择 top-$k$：

$$
c_t^Q = h_t W^{DQ}, \quad q_t^I = c_t^Q W^{IUQ}
$$

$$
[w_t^{I,1}; \ldots; w_t^{I,n_h^I}] = h_t W^w
$$

$$
I_{t,s} = \sum_{h=1}^{n_h^I} w_t^{I,h} \cdot \text{ReLU}\!\left(q_t^{I,h} \cdot K_s^{IComp}\right), \quad \text{idx} = \text{top-}k(I_{t,:})
$$

这里的 $K_s^{IComp}$ 是用同类压缩器得到的 indexer key。论文没有把 Indexer 简化成普通 $q^\top k$，而是使用多头打分 + ReLU + 可学习 head weight 的形式。

**Indexer 的参数配置**：$n_h^I = 64$ 个 Indexer 头，压缩维度 $c^I = 128$。CSA Indexer 的 QK 路径采用 FP4 计算和缓存——排序只需保持相对顺序，对绝对精度容忍度高。

**主路径与 Indexer 的精度分离**：主路径的压缩 KV cache 采用混合精度存储（RoPE 维度 BF16，其余维度 FP8），Indexer 的 QK 路径采用 FP4。两条路精度目标不同：主路径要保留生成质量，Indexer 只负责排序选块。

##### 3.x.2.4 CSA 层的完整流程

综合以上组件，CSA 层对当前 token 的注意力计算流程为：

1. **压缩**：序列中每 $m = 4$ 条 token 经 Compressor 加权池化为 1 条 KV entry（重叠，$2m = 8$ 个 token 参与）
2. **选择**：Lightning Indexer 从 $N/m$ 条压缩 KV 中选出 top-$k$ 条（V4-Pro 为 1024，V4-Flash 为 512）
3. **注意力**：Query 头对选中的 top-$k$ 条压缩 KV + 128 条滑窗 KV 做标准注意力计算（V4-Pro 为 128 个 Query 头，V4-Flash 为 64 个）
4. **输出**：Grouped Output Projection（见 3.x.6.4 节）

#### 3.x.3 HCA：重度压缩注意力

##### 3.x.3.1 设计原理

HCA（Heavily Compressed Attention）与 CSA 使用**同一个 Compressor 类**，区别仅在于三个参数：

表2：CSA vs HCA 参数对比

| 参数 | CSA | HCA |
|------|-----|-----|
| 压缩比 $m$ | 4 | 128 |
| 重叠 | 是（$2m = 8$ 个 token 参与） | 否（仅 $m = 128$ 个 token） |
| 稀疏选择 | top-$k$（Lightning Indexer；Pro 为 1024，Flash 为 512） | 无（压缩后全看） |

##### 3.x.3.2 HCA 为何全看

HCA 的压缩比 $m' = 128$ 意味着 50000 个 token 压缩为：

$$
\lfloor 50000 / 128 \rfloor = 390 \text{ 条}
$$

390 条 KV entry，每条 512 维 FP8，单层仅约 0.2 MB。对 390 条做完整注意力的计算量可忽略不计——因此 HCA 不需要稀疏选择，直接全看。

##### 3.x.3.3 HCA 为何不重叠

HCA 的块大小为 128 个 token，覆盖约 3-5 个完整句子。块边界恰好落在语义断裂点的概率远低于 CSA 的 4-token 块。重叠的收益有限，但代价巨大——投影维度从 $d_c$ 翻倍为 $2d_c$，在 128-token 块上意味着存储和计算开销翻倍。

##### 3.x.3.4 HCA 的功能定位

HCA 提供**低分辨率的全局概览**。50K 上下文被压缩为 390 条"摘要"，每条浓缩了 128 个 token 的核心语义。模型通过 HCA 层获得"上下文长什么样"的粗略地图，具体细节由 CSA 层补充。

#### 3.x.4 滑动窗口：短期记忆的兜底

##### 3.x.4.1 设计动机

CSA 和 HCA 都需要凑满一个完整块才能压缩。CSA 需要 4 个 token，HCA 需要 128 个 token。在凑满之前的"尾巴"尚未被压缩，如果不特殊处理，最近生成的 token 将暂时不可见。

滑动窗口（Sliding Window）解决这一问题：

$$
n_{\text{win}} = 128
$$

当前 token 往前数 128 条 KV cache **原样保留**，不做任何压缩。每生成一个新 token，最老的条目被移出窗口、纳入压缩流程。

##### 3.x.4.2 窗口大小的确定

$n_{\text{win}} = 128$ 恰好覆盖 HCA 的最坏情况：当 HCA 块尚差 1 个 token 才能凑满时，已有 127 个 token 在等待压缩。$n_{\text{win}} = 128 \geq 127$ 保证这些等待中的 token 不会"消失"。

CSA 层仅需 3 个 token 的缓冲，但统一为 128 简化了工程实现，额外存储（约 100 条 512 维 latent）可忽略。

##### 3.x.4.3 三层视野的协作

每一层 Transformer 同时拥有两种视野：

$$
\text{Output}_\ell = \text{Attn}\!\left(\mathbf{Q},\; [\underbrace{\text{Compressed KV}}_{\text{CSA 或 HCA}};\; \underbrace{\text{Window KV}}_{n_{\text{win}} = 128}]\right)
$$

- **HCA 层**：全局粗概览（390 条） + 最近精细上下文（128 条）
- **CSA 层**：精选细节（Pro 为 1024 条，Flash 为 512 条） + 最近精细上下文（128 条）

#### 3.x.5 层间排列与架构组织

##### 3.x.5.1 V4-Pro 的 61 层布局

V4-Pro 主干 Transformer 共 61 层，配置文件中 `compress_ratios` 字段为：

$$
[\underbrace{128, 128}_{\text{前 2 层 HCA}},\; \underbrace{4, 128, 4, 128, \ldots}_{\text{CSA/HCA 严格交替}}]
$$

统计：**31 层 HCA + 30 层 CSA**。前两层均为 HCA——模型在最底层先建立全局概览，随后进入交替模式。

末尾还有 1 个 MTP（Multi-Token Prediction）block，其 `compress_ratio = 0`，表示不压缩，仅使用纯滑动窗口（128 条）。MTP 紧贴输出侧，前面 61 层已将长上下文信息揉入残差流，MTP 只需最近的精细信息即可完成末位预测。

##### 3.x.5.2 从 NSA 三分支并行到层间分工

NSA 论文中，三种粒度（压缩全看、选块精读、滑动窗口）在**每层同时并行**，通过门控加权合并。V4 的工程改造将其拆为层间交替：

$$
\text{NSA: } \text{Output}_\ell = g_1 \cdot \text{Compressed}_\ell + g_2 \cdot \text{Selected}_\ell + g_3 \cdot \text{Window}_\ell
$$

$$
\text{V4: } \text{Output}_\ell = \begin{cases} \text{HCA}_\ell + \text{Window}_\ell & \ell \in \text{HCA 层} \\ \text{CSA}_\ell + \text{Window}_\ell & \ell \in \text{CSA 层} \end{cases}
$$

每层只运行一种压缩模式，计算开销减半。代价是单层视野变窄，但每过两层（一层 HCA + 一层 CSA），模型既扫过全局概览又精选过细节，效果等效于 NSA 的层内并行。

##### 3.x.5.3 层间分工的附带优势

拆分到不同层后，CSA 和 HCA 的块大小可以**独立设定**——NSA 论文中两种粒度共处一层，尺度不能差距过大；V4 不受此约束，直接设为 $m = 4$（CSA）和 $m' = 128$（HCA），各取最优。

#### 3.x.6 其他关键设计细节

##### 3.x.6.1 Attention Sink

标准 softmax 强制注意力权重之和为 1，即使所有 KV 条目都不相关，模型也必须"看某个地方"。V4 引入 Attention Sink 机制：

$$
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\left[\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}};\; s_{\text{sink}}\right]\right) \cdot [\mathbf{V};\; \mathbf{0}]
$$

在 softmax 的分母中增加一个 $\exp(s_{\text{sink}})$ 项（$s_{\text{sink}}$ 为可学习标量），对应一个虚拟的"垃圾桶" Value（零向量）。当模型判断当前上下文无有效信息时，注意力权重可以流向 sink，等效于"弃权"——避免无意义的权重被分配到不相关的 KV 条目上。

##### 3.x.6.2 RMSNorm on Q and KV Entries

为避免注意力分数在长序列上爆炸（点积值随维度和序列长度增大），V4 对 Query 和 KV entry 在注意力计算前分别施加 RMSNorm：

$$
\hat{\mathbf{q}} = \text{RMSNorm}(\mathbf{q}), \quad \hat{\mathbf{k}} = \text{RMSNorm}(\mathbf{k})
$$

$$
\text{score} = \hat{\mathbf{q}}^\top \hat{\mathbf{k}} / \sqrt{d}
$$

RMSNorm 将向量归一化至单位 RMS 幅值，配合 $\sqrt{d}$ 缩放，确保注意力分数的数值范围稳定。

##### 3.x.6.3 Partial RoPE

旋转位置编码（RoPE）仅应用于 KV entry 的**最后 64 维**，而非全部 512 维：

$$
\mathbf{k} = [\underbrace{\mathbf{k}_{1:448}}_{\text{无位置编码}};\; \underbrace{\text{RoPE}(\mathbf{k}_{449:512})}_{\text{64 维 RoPE}}]
$$

Partial RoPE 的设计动机：MLA 的 latent 联合编码了 Key 和 Value 的语义，全维度施加 RoPE 会破坏 Value 的位置无关性。仅在少量维度上编码位置信息，既提供序列位置感知，又保留了大部分维度的语义纯净性。

##### 3.x.6.4 Grouped Output Projection

$n_h = 128$ 个注意力头的输出不做全连接投影，而是分为 $g = 16$ 组，每组 8 个头共享一个输出投影矩阵：

$$
\mathbf{O} = \sum_{i=1}^{g} \text{Concat}(\mathbf{o}_{(i-1) \cdot 8 + 1}, \ldots, \mathbf{o}_{i \cdot 8}) \cdot \mathbf{W}_O^{(i)}, \quad \mathbf{W}_O^{(i)} \in \mathbb{R}^{(8 \times c) \times d_g}
$$

其中 $c = 512$ 为每头维度，$d_g = 1024$ 为每组输出维度。16 组中间输出拼接为 $16 \times 1024 = 16384$ 维，随后再投影回隐藏维度 $d = 7168$。分组投影减少第一段输出投影的计算量，同时保留组内头间的交互能力。

#### 3.x.7 效率分析：KV Cache 压缩的量化计算

##### 3.x.7.1 逐层存储的近似计算

下面的数字是为了帮助读者建立量级感，采用简化口径：只按 512 维压缩 KV entry、FP8 存储估算，未把 RoPE 维度 BF16、Indexer cache、cache block 对齐、状态缓存和工程元数据全部纳入。因此它不能直接替代论文 Figure 1 的官方 KV cache 统计。

以 50000 token 上下文、512 维 latent、FP8（1 字节/元素）为基准：

**CSA 层**（$m = 4$，推理时虽然只读 top-$k$，但仍需存全量压缩 KV 供后续 token 的 Indexer 和注意力使用）：

$$
\text{压缩 KV} = \lfloor 50000 / 4 \rfloor \times 512 \times 1\text{B} = 12500 \times 512 = 6.4\text{ MB}
$$

$$
\text{滑动窗口} = 128 \times 512 \times 1\text{B} = 65.5\text{ KB}
$$

$$
\text{CSA 单层合计} \approx 6.5\text{ MB}
$$

**HCA 层**（$m' = 128$，全看）：

$$
\text{压缩 KV} = \lfloor 50000 / 128 \rfloor \times 512 \times 1\text{B} = 390 \times 512 = 200\text{ KB}
$$

$$
\text{滑动窗口} = 128 \times 512 \times 1\text{B} = 65.5\text{ KB}
$$

$$
\text{HCA 单层合计} \approx 0.27\text{ MB}
$$

##### 3.x.7.2 全模型汇总

$$
\text{Total} = 30 \times 6.5 + 31 \times 0.27 + 1 \times 0.07 \approx 228\text{ MB}
$$

##### 3.x.7.3 演化路径的压缩效率对比

表3：50K 上下文、61 层架构的 KV cache 总量

| 方案 | KV cache 总量 | 相对 MHA |
|------|--------------|---------|
| 标准 MHA（128 头） | 81.5 GiB | 1.000 |
| GQA（8 组） | 5.1 GB | 0.063 |
| MQA（1 组） | 0.64 GB | 0.008 |
| MLA（$d_c = 512$） | 2.9 GB | 0.036 |
| CSA/HCA（V4，简化估算） | 约 228 MB | 0.003 |

在这个简化口径下，总压缩率约为 99.7%。官方论文给出的对比口径更保守：在 1M context 场景下，相对 DeepSeek-V3.2，V4-Pro 的 KV cache 约为 10%，V4-Flash 约为 7.3%；相对常见 BF16 GQA8 baseline，V4 系列约为 2%。

两刀叠加的效果：MLA 砍维度（$d \to d_c$，约 14 倍），Compressor 砍条数（$N \to N/m$，CSA 4 倍、HCA 128 倍）。

#### 3.x.8 从 NSA 到 CSA/HCA 的工程改造总结

表4：NSA 论文 vs V4 CSA/HCA 的工程对比

| 设计维度 | NSA（2025-02 论文） | V4 CSA/HCA（2026） |
|---------|--------------------|--------------------|
| 三分支组织 | 每层并行，门控合并 | 层间交替（CSA/HCA），单层单种 |
| Top-k 选择机制 | 复用压缩分支 softmax 分数 | 独立 Lightning Indexer（FP4） |
| 压缩块大小 | 统一 32（步长 16 重叠） | 差异化：CSA 4（重叠）、HCA 128（不重叠） |
| 滑动窗口 | 512 | 128（覆盖 HCA 最坏情况） |
| 量化策略 | 未涉及 | Indexer QK 路径 FP4 + 主路径混合精度 |
| 模型规模 | 27B（实验） | 1.6T（产品） |
| 上下文长度 | 64K | 1M |

**改造的核心逻辑**：

1. **三分支并行 $\to$ 层级分工**——每层开销减半，两层一个完整周期
2. **复用压缩分数 $\to$ 独立 Lightning Indexer**——消除串行依赖，精度解耦，支持 FP4 加速
3. **统一块大小 $\to$ 差异化（4 vs 128）**——层间独立后各取最优

#### 3.x.9 本节小结

CSA/HCA 混合注意力是 DeepSeek 注意力演化线的当前终点，将 NSA 的"原生可训练稀疏"理论方案工程化为 1.6T 规模的产品架构。其核心贡献包括：

1. **序列维度的双精度压缩**：CSA（$m = 4$，重叠，top-$k$ 精选；Pro 为 1024、Flash 为 512）提供细粒度选择性访问，HCA（$m' = 128$，不重叠，全看）提供低成本全局概览，两者层间交替互补
2. **压缩与选择的解耦**：Lightning Indexer 的 QK 路径采用 FP4 计算和缓存，主注意力路径保留更高精度，消除"选块便宜"和"生成质量"之间的耦合
3. **两阶段稀疏引入**：先用 dense attention 建立基础表征，再在长序列阶段引入 CSA/HCA 稀疏注意力并 warm up Lightning Indexer，避免从随机初始化开始做稀疏选择的冷启动问题
4. **极致存储效率**：结合 MLA/shared-KV 的维度压缩和 Compressor 的条数压缩，50K 上下文的简化 KV cache 估算从 81.5 GiB 压到约 228 MB；官方 1M 场景下，V4-Pro 相对 V3.2 的 KV cache 约为 10%，使 1M 上下文的工程部署成为可能

从更根本的角度看，CSA/HCA 能在简化估算口径下做到极高压缩率，同时官方评测仍保持强性能，根源在于**自然语言的有效信息本身是低维的**——7168 维隐藏空间中，语义路径集中在几百维的低维流形上。V4 的全部注意力优化归结为一件事：只沿着流形存路径，不存流形以外的空气。

---

**参考文献**

- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017. arXiv:1706.03762.
- Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150.
- Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv:2305.13245.
- DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.
- Yuan, J. et al. (2025). Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. arXiv:2502.11089.
- DeepSeek-AI (2026). DeepSeek V4 Technical Report. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf.
