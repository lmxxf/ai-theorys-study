# 文献核查 A：单头"只能平均"、多头"跨头相加"（凸包 / Minkowski 和）

核查日期：2026-09-15。所有引文均来自当天抓取的 arXiv 摘要页 / HTML 全文 / PDF 文本（Sanford 那篇用 pdftotext 抽的 v2），未以训练记忆为准。抓不到的地方标"未核到"。

## 0. 被核命题（我们自己推的）

> 单头 softmax 注意力的输出永远在 value 向量的凸包内（权重非负、和为 1），因此单头只能做"加权平均"不能做"求和"——看到一个 a 和看到两个相同的 a 输出一样，softmax 归一化抹掉了计数；多头把 h 个凸组合跨头相加（拼接乘 W_O），输出落在 h 个凸包的 Minkowski 和里，因此多头本质是"h 个独立指针槽位"。

命题可拆成三小条：
- (A) 单头输出 ∈ conv(values)
- (B) 归一化抹掉绝对计数（一个 a 与两个 a 不可分）
- (C) 多头输出 ∈ ⊕_h W_O^(h) conv(V_h)（Minkowski 和），→ "h 个槽位"

## 1. 结论速览

| 文献 | (A) 凸包 | (B) 归一化抹计数 | (C) Minkowski 和 / 槽位 | 与本命题关系 |
|---|---|---|---|---|
| Fel et al. 2025 "Into the Rabbit Hull" (ICLR 2026) | 原话有 | 无 | **原话有，Proposition 1 就是 Minkowski 和** | (A)+(C) 几乎逐字等价；未推"槽位/计数" |
| Barbero et al. 2024 "Transformers need glasses!" (NeurIPS 2024) | 隐含 | **有：无位置编码时表示只依赖 n0:n1 比例** | 无 | (B) 的严格版（更强：给出定理+浮点崩塌），但结论限于"无位置编码"的整个 Transformer |
| Yehudai et al. 2024 "When Can Transformers Count to n?" (ICLR 2025) | 隐含 | **有：softmax 归一化 → 只能拿到 1/count，要 MLP 反演** | 无 | 与 (B) 相反方向的补充：单头 + MLP 其实能数（d ≥ 词表时），我们的"单头不能计数"要限定为"单头注意力本身、不带 MLP 反演/BOS" |
| Sanford–Hsu–Telgarsky 2023 "Representational Strengths…" (NeurIPS 2023) | 原话有（证明思路句） | 无 | 有头数下界，但不是 Minkowski 和 | 主结论是"注意力≈稀疏平均"（qSA）；头数下界是通信复杂度的 mpH，比"槽位"弱且方向不同 |
| Yu et al. 2025 "Effect of Attention Head Count…" (ICLR 2026) | 无 | 无 | **h ≥ D 头各负责一个坐标，h < D 参数量爆炸** | (C) "槽位"直觉的逼近论版本，最接近"h 个独立指针" |
| Rajaraman–Sundaram–Tesfaye 2026 "Head Complexity of Boolean Functions…" | 隐含（softmax 分母正） | 无 | k 头恰好算 k 位 parity | (C) 的另一个精确版：头数 = 严格资源 |
| Lu et al. 2026 "ZeroS" | 原话有 | 无 | 无 | 只说凸组合"只能加不能减"，角度不同 |

**总判断**：
- (A)+(C) 的几何陈述（凸包 → Minkowski 和）已经被 Fel et al. 2025 用几乎相同的话写出来了（Proposition 1，"Multi-head attention realizes MRH"），公众号可直接引，说"这一点已有论文写成命题"。
- (B) 有两篇严格文献，但都不是"单头"层面的表述：Barbero 是"整个无位置编码的 Transformer 只看比例"，Yehudai 是"softmax 后只剩 1/c，单头 + MLP 反演可恢复"。**我们说"单头只能平均不能求和"要加限定：指注意力算子本身；加了 MLP、BOS/sink 或位置编码后计数可以被恢复。**
- "h 个独立指针槽位"这个说法：没有文献用"slots/pointers"原话，但 Yu et al. 2025 的 "heads can specialize to distinct coordinates" 和 Rajaraman 等的 "k heads ↔ k-bit parity" 是最接近的严格支撑。**未核到**任何文献把 Minkowski 和与"槽位数"直接联系起来——这一步是我们自己的推论。

## 2. 逐篇核查

### 2.1 Sanford, Hsu, Telgarsky — Representational Strengths and Limitations of Transformers

- 标题：Representational Strengths and Limitations of Transformers
- 作者：Clayton Sanford, Daniel Hsu, Matus Telgarsky
- arXiv：**2306.02896**，v1 2023-06-05，v2 2023-11-16
- 会议：**NeurIPS 2023**（neurips.cc poster 72920 / papers.nips.cc 已核）
- 模型定义（Definition 1/2，v2 原文）：
  > f_{Q,K,V}(X) = softmax(XQK^T X^T) XV
  > "For head-count H and self-attention units f_1,…,f_H … a multi-headed attention layer is a function L(X) = Σ_{h=1}^H f_h(X)."

  注意：他们把多头直接定义为**各头输出求和**（不是拼接后乘 W_O，但两者等价——拼接乘 W_O 就是 Σ_h W_O^(h) f_h）。这正是我们 (C) 的形式。
- 任务名：**q-sparse averaging（qSA）**、**Match2**、**Match3**（不是 "q-sparse" 之外的别名）。
  > "the ith output element in qSA is obtained by averaging the q data parts z_j given by j ∈ y_i, meaning qSA(…) = ( (1/q) Σ_{j∈y_1} z_j, …, (1/q) Σ_{j∈y_N} z_j )."
  > Match2(X)_i = 1{∃j s.t. x_i + x_j = 0 (mod M)}, Match3(X)_i = 1{∃j_1,j_2 s.t. x_i + x_{j_1} + x_{j_2} = 0 (mod M)}
- 与凸包直接相关的原话（§3.1，Theorem 2 证明思路）：
  > "Because the output of a self-attention unit is a convex combination of rows of the value matrix ϕ(X)V ∈ R^{N×d′}, a natural way to approximate qSA with a unit of self-attention is to let each value be the corresponding vector in the average (i.e. V^T ϕ(x_i) = z_i) and choose the key and query functions in order to ensure that the attention matrix satisfies softmax(…)_{i,j} ≈ 1/q if j ∈ y_i, 0 otherwise."

  中译：因为自注意力单元的输出是 value 矩阵各行的凸组合，用一个单元逼近 qSA 的自然做法是让每个 value 就是要平均的向量，再让注意力矩阵在 j∈y_i 处≈1/q、其余≈0。
- 头数相关定理：
  - Theorem 2（fixed precision）：m ≥ Ω(d′ + q log N) 时**单头**可 ε-逼近 qSA。
  - Theorem 4：> "For any sufficiently large q, any N ≥ 2q+1, and any d′ ≥ 1, there exists a universal constant c such that if mp ≤ cq, then no f ∈ T^{1,1}_{d,m,d′,p} exists that 1/(2q)-approximates qSA." —— 单头、单层，嵌入维 × 精度 ≲ q 时做不到。
  - Theorem 7：> "There is universal constant c > 0 such that for sufficiently large N, and any M ≥ N+1, if mpH ≤ cN/log log N, then there is no f ∈ T^{1,H}_{1,m,1,p} satisfying f(X) = Match3(X) for all X ∈ [M]^N." —— **单层多头**算 Match3 需要 m·p·H ≳ N/log log N（头数 H 与嵌入维 m、精度 p 可互换）。
  - Informal Theorem 2 第 2 条：> "A single layer of standard multi-headed self-attention cannot compute Match3 unless its number of heads H or embedding dimension m grows polynomially in N (Theorem 7)."
- **与本命题的关系**：
  - 它把"注意力就是稀疏平均"当作注意力的**典型任务**（正面能力），与我们 (A) 同一口径。
  - 它的头数下界是**通信复杂度**（mpH 位）的，头数、维度、精度三者可互换——这比"h 个独立槽位"**弱**（不能推出"每头一个指针"），而且方向不同（讲的是 Match3 三元匹配做不了，不是计数）。
  - 它**没有**讲 Minkowski 和，也没有讲"平均抹掉计数"。
  - 编号 2306.02896 正确；任务名 qSA / Match2 / Match3 正确。

### 2.2 Barbero et al. — Transformers need glasses! Information over-squashing in language tasks

- 标题：Transformers need glasses! Information over-squashing in language tasks
- 作者：Federico Barbero, Andrea Banino, Steven Kapturowski, Dharshan Kumaran, João G.M. Araújo, Alex Vitvitskyi, Razvan Pascanu, Petar Veličković
- arXiv：**2406.04267**，v1 2024-06-06，v2 2024-10-24
- 会议：**NeurIPS 2024**（neurips.cc poster 96332；openreview 93HCE8vTye）
- 与计数直接相关的原话：
  > Proposition 6.1: "A Transformer without positional encodings and a causal attention mechanism is immediately unable to count."
  > Proposition B.9（附录形式版）: "A Transformer without positional encodings and a causal attention mechanism is immediately unable to solve the counting problem." 证明核心："the representations only depend on the ratio between n_0 and n_1"（n_0、n_1 为两种 token 的个数）。
  > §6: "counting is a problem that requires some notion of 'unboundedness' of the representations, whilst the normalisations used inside a Transformer work against this."
  > Corollary 6.2 (Informal): "Counting in certain situations becomes impossible due to representational collapse and finite floating point precision."

  中译：无位置编码的因果 Transformer 天生不能计数（表示只依赖 n_0:n_1 的比例，"10" 与 "1100" 同表示）；计数需要表示的"无界性"，而 Transformer 内部的归一化恰恰反对这种无界性；再加上有限浮点精度与表示坍塌，某些情形计数不可能。
- **与本命题的关系**：
  - 这是我们 (B)"一个 a 和两个 a 输出一样"的**严格版**，机制同一个：softmax 归一化只保留比例。
  - 但它是**更强也更窄**的陈述：更强——针对整个多层 Transformer（含 MLP、LayerNorm），并加了浮点精度的不可逆崩塌；更窄——命题条件是"无位置编码"，有位置编码时比例论证失效（他们进而用 over-squashing / 表示坍塌讨论有位置编码的情况）。
  - 它没有讲头数、凸包、Minkowski 和。
- 抓取局限：HTML 版由小模型转述，Prop B.9 证明只拿到了核心句 "only depend on the ratio between n_0 and n_1"，整段证明**未逐字核到**。

### 2.3 Yehudai et al. — When Can Transformers Count to n?

- 标题：When Can Transformers Count to n?
- 作者：Gilad Yehudai, Haim Kaplan, Guy Dar, Royi Rassin, Asma Ghandeharioun, Mor Geva, Amir Globerson
- arXiv：**2407.15160**，v1 2024-07-21，v2 2024-10-07，v3 2026-02-25
- 会议：**ICLR 2025**（搜索结果称 accepted to ICLR 2025；arXiv 页 comments 未核到，标"据二手来源"）
- 核心定理（v3 原话）：
  > Theorem 4.1 (Histogram Solution): "For the Query Count problem and any context length n, if d ≥ m, there exists a 1-layer, 1-head transformer with an MLP of width d that solves the task perfectly."
  > Theorem 4.2: "If m ≥ 2d, for any choice of embedding vectors, there exist inputs where the Histogram solution incurs an error of at least Ω(√n)."
  > Theorem 5.2: "If d < m, then to solve the MFE task, K must satisfy K ≥ Ω(√n/R · m^{1/d})"（K 是权重范数上界；d 嵌入维，m 词表大小，n 上下文长度）

  摘要（v3）："When the dimension is at least as large as the vocabulary, transformers can perfectly maintain token counts. However, when the vocabulary exceeds the embedding dimension, the interference between non-orthogonal token representations forces the network weights to scale polynomially."
- 与 softmax 归一化直接相关的原话（§4.2 CountAttend）：
  > "because the attention mechanism applies a softmax normalization, recovering the raw count requires inverting the function f(x) ∝ 1/x."

  机制：query token 只对与自己相同的 token 给高权重，每个相同 token 拿到 ≈1/c 的权重，聚合后得到与 1/c 成比例的量，再用 MLP 算 1/z 恢复 c。（另一条路线 Histogram 用正交 embedding 累加，d ≥ m 时 1 头 + MLP 完美计数。）
- **与本命题的关系**：
  - 它**证实**了我们的机制描述："softmax 归一化把计数压成 1/c"——这就是"归一化抹掉绝对数量"的另一种说法。
  - 但它同时**反驳**了"单头不能计数"的粗糙版本：单头 + MLP（做 1/z 反演，或用 BOS 之类固定 token 稀释）能把 c 算回来。所以我们的命题必须说清：**注意力算子本身**输出在凸包内、只见比例；计数信息并没有完全丢，它藏在"平均值被稀释到 1/c"里，要靠 MLP 的非线性或额外 token 才能读出来；而当词表 > 嵌入维时，这条路又因非正交干扰而数值不稳定（Theorem 4.2 / 5.2）。
  - 它没有讲多头、凸包、Minkowski 和。

### 2.4 Fel et al. — Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry（最直接的等价文献）

- 标题：Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry
- 作者：Thomas Fel, Binxu Wang, Michael A. Lepori, Matthew Kowal, Andrew Lee, Randall Balestriero, Sonia Joseph, Ekdeep S. Lubana, Talia Konkle, Demba Ba, Martin Wattenberg
- arXiv：**2510.08638**，v1 2025-10-08，v2 2026-02-26，v3 2026-05-07
- 会议：**ICLR 2026**（arXiv comments："Accepted at ICLR 2026"）
- 原话（摘要）：
  > "tokens are formed by combining convex mixtures of archetypes" … "multi-head attention produces sums of convex mixtures, defining regions bounded by archetypes."
- 原话（§7 Minkowski Representation Hypothesis）：
  > "This Minkowski sum structure naturally emerges from multi-head attention, where each head produces convex combinations of its value vectors, and the final output layer additively combines these convex hulls from different heads."
- 原话（Proposition 1，"Multi-head attention realizes MRH"）：
  > y = Σ_h W_O^(h) y_h = Σ_h Σ_i α_{h,i} W_O^(h) v_h^(i) ∈ ⊕_h W_O^(h)(conv(V_h))
  > "If each head can realize points of the probability simplex, then the attainable set is exactly the Minkowski sum."

  中译：多头输出 = 各头凸组合经 W_O 投影后相加，落在各头（投影后）凸包的 Minkowski 和里；若每头能取到概率单纯形上任意点，可达集恰好就是这个 Minkowski 和。
- **与本命题的关系**：**(A)+(C) 逐字等价**，连"拼接乘 W_O = 各头投影后求和"这一步都写了。它是视觉（DINOv2）解释性论文，用 Minkowski 和解释 token 表示的几何；**没有**推到"计数"或"槽位数"。我们的增量是：把这个几何和 Barbero/Yehudai 的"归一化抹计数"接起来，得出"h 个槽位"。

### 2.5 Yu, Jiang, Bao, Yu, Li — The Effect of Attention Head Count on Transformer Approximation（"槽位"直觉的逼近论版本）

- 标题：The Effect of Attention Head Count on Transformer Approximation
- 作者：Penghao Yu, Haotian Jiang, Zeyu Bao, Ruoxi Yu, Qianxiao Li
- arXiv：**2510.06662**，v1 2025-10-08，v2 2026-03-31
- 会议：**ICLR 2026**（arXiv comments："Accepted by ICLR 2026"）
- 任务：generalized D-retrieval，z̄_i(X) = min_{t∈S_i} f_i(x^(t))，i=1..D，目标 H(X) = F_0(z̄_1,…,z̄_D)。
- 原话：
  > 摘要："transformers with sufficiently many heads admit efficient approximation, whereas with too few heads, the number of parameters must scale at least as O(1/ε^{cT}), for some constant c and sequence length T."
  > Theorem 2 (1)：h = D 头、每头嵌入维 n = 2 即可 ε-逼近；(2)：h = s < D 时参数量下界 Ω(1/ε^k)。
  > "When h < D, a single head must encode multiple roles simultaneously" … "information bottleneck" … "heads can specialize to distinct coordinates z_i, eliminating the bottleneck and enabling efficient approximation."
  > 摘要（单头）："an embedding dimension of order O(T) allows complete memorization of the input, where approximation is entirely achieved by the feed-forward block."

  中译：要同时取 D 个（不同位置的）目标值，D 个头一头一个就够；头数少于 D 时一个头得兼多个角色，出现信息瓶颈，参数量随精度指数爆炸；单头只有靠 O(T) 的嵌入维把整个输入背下来、全交给 MLP。
- **与本命题的关系**：这是"多头 = h 个独立指针槽位"最接近的严格版本——头数 = 能并行取回的独立目标数。它不用凸包语言，而用逼近论的参数量下界；比我们**强**（有定量下界），但任务是 min-检索，不是计数。

### 2.6 Rajaraman, Sundaram, Tesfaye — The Head Complexity of Boolean Functions in Single-Layer Attention

- arXiv：**2609.04046**，v1（按编号为 2026-09；抓取页把日期转述成 "September 3, 2024"，与编号矛盾，**日期未核准**，以编号为准）
- 会议：未核到（arXiv 预印本，cs.CC）
- 模型：MHA(x) = Σ_{j=1}^k Softmax(v_+ A_j X^T) X V_j —— softmax 单层多头、各头求和。
- 原话：
  > Theorem 1: "k heads cannot compute (k+1)-bit parity"；Theorem 2: "k heads compute k-bit parity"
  > "Each monomial of F_x is a product of k factors, and each factor reads only one input position, so a monomial depends on at most k of the k+1 parity bits: some bit is always missing."
  > 下界 "holds at unbounded embedding dimension and unbounded numerical precision"
- **与本命题的关系**：头数是**严格资源**（维度、精度无限也补不了）——比 Sanford 的 mpH 可互换下界更贴近"每头一个槽位"。机制上正是"每头一次只读一个位置"，与"指针"直觉一致。未讲凸包/计数。

### 2.7 Lu et al. — ZeroS: Zero-Sum Linear Attention for Efficient Transformers

- arXiv：**2602.05230**，v1 2026-02-05；会议未核到
- 原话："softmax attention produces convex combinations of value vectors" … "can only blend information additively, unable to express subtractive or contrastive operations directly" … "with softmax weights, a single attention layer cannot express differential or contrastive operations (even with just two tokens)".
- 与本命题：只用了 (A)，推的是"不能做减法"，不是"不能计数"。可作旁证，非等价。

### 2.8 其他搜到但未细核
- Zhang & Alanwar 2026, "Matrix Zonotopic Attention" (arXiv 2608.05472)：用 zonotope（凸多面体）讲 value 投影，抓取未见"Minkowski 和跨头"原话，**未核到**等价陈述。
- Zhu et al. 2024 "Counting Like Transformers: Compiling Temporal Counting Logic Into Softmax Transformers" (arXiv 2404.04393)、"The Counting Power of Transformers" (2505.11199)：讲 softmax 能表达的计数逻辑类，与"单头凸包"角度不同，未细核。
- 搜索中未发现任何文献使用 "head count as slots/pointers" 或 "averaging vs summing" 的原话。

## 3. 给正文的建议措辞（可证据支撑的边界）

1. "单头输出在 value 凸包内" —— 可引 Sanford 2023 §3.1 原话、Fel 2025 Prop 1。安全。
2. "softmax 归一化只保留比例、抹掉绝对计数" —— 引 Barbero 2024 Prop 6.1 / B.9（限无位置编码）+ Yehudai 2024 §4.2（1/c 需 MLP 反演）。要加一句：**计数信息没有消失，是被压成 1/c 藏在幅值里，注意力自己读不出来，得靠 MLP 或额外 token**。别写成"单头绝对不能数"。
3. "多头 = 各头凸包的 Minkowski 和" —— 引 Fel 2025 Prop 1 原话，说明"这一步已有论文写成命题"。
4. "多头 = h 个独立指针槽位" —— 标为**我们的推论**，旁证 Yu 2025（h ≥ D 头各取一坐标）与 Rajaraman 2026（k 头 ↔ k 位 parity，维度精度无限也不能替代头数）；同时诚实指出 Sanford 2023 Theorem 7 里头数与维度、精度是可互换的，所以"槽位"不是普适的硬边界。
