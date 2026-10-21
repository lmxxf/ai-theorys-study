# Audit B：低秩瓶颈定理 + 主流模型头维度表

核查日期：2026-09-15。来源标注：arXiv 原文 PDF（pdftotext 逐行核对）/ HF 官方仓库 config.json（raw）/ 官方论文表格。写"未核到"的一律没查到一手来源，别编。

---

## 1. Bhojanapalli, Yun, Rawat, Reddi, Kumar — "Low-Rank Bottleneck in Multi-head Attention Models"

| 项 | 核查结果 |
|---|---|
| arXiv 编号 | **2002.07028**（v1 2020-02-17）；ICML 2020，PMLR v119 `bhojanapalli20a` |
| 记号 | 序列长度 **n**，embedding 维度 **d**，输入 X ∈ R^{d×n}（**列是 token**，和常见的 n×d 转置）；头数 **h**；每头投影维度 **d_p**（标准做法 d_p = d/h）；单头情形 d_q = d_k = d_v = d |
| Softmax 方向 | 论文里 Softmax 是**按列**算的（"the Softmax here is a columnwise operator computing the attention scores for each query"），所以定理说的是 **column stochastic**，等价于常见写法（X 取 n×d）下的 row-stochastic。公众号按 row-stochastic 说没问题，但引原话时别改词。 |

### 1.1 主定理（Theorem 1，Representation Theorem）原话

> **Theorem 1 (Representation Theorem).** If d_q = d_k = d ≥ n, then given any full column rank matrix X ∈ R^{d×n} and an arbitrary n × n positive column stochastic matrix P, there always exists d × d projection matrices W_q and W_k such that
> Softmax[ (W_k X)^T (W_q X) / √d_k ] = P.   (3)
> If d_q = d_k = d < n, there exist X and P such that (3) does not hold for all W_q and W_k.

中译：若 d_q = d_k = d ≥ n，那么对任意列满秩的 X ∈ R^{d×n} 和任意 n×n **元素全正**的列随机矩阵 P，总存在 d×d 的投影矩阵 W_q、W_k 使 Softmax[(W_k X)^T (W_q X)/√d_k] = P。若 d < n，则存在 X 和 P 使得对所有 W_q、W_k 上式都不成立。

论文紧接着的解读原话：
> "This result shows that the projection dimension d_q = d_k = d needs to be larger than the sequence length n for the attention unit to be able to represent any desired context P."

**注意两点（写文章别说过头）**：
1. 定理是对**单头、投影维度 = d** 写的；推到多头时，每头投影维度是 d/h，所以条件变成 **d/h ≥ n**（§2.3 原话："When the number of heads h is larger than d/n, the attention unit inside each head projects onto a dimension smaller than n, creating a low-rank bottleneck and loses its ability to represent arbitrary context vectors"）。
2. 结论是"**存在**某个 (X, P) 表示不了"，不是"大部分表示不了"，更不是"性能一定掉"。这是一个存在性的表达能力结果，而且是"单个样本序列"（"Even though this result describes a single example sequence case…"）。

### 1.2 证明思路（两三句）

- **d ≥ n 方向（构造）**：X 列满秩 ⇒ 有左逆 X† = (XᵀX)⁻¹Xᵀ，X†X = I_n。令 W_k = W̃_k X†，W_q = W̃_q X†，则 Xᵀ W_kᵀ W_q X = W̃_kᵀ W̃_q =: W̃_kq，把对 X 的依赖消掉；然后取 W̃_kq 的每一列 = log(P 的对应列)（P 全正所以 log 有定义），Softmax 按列还原出 P。
- **d < n 方向（反例）**：取 d = 1, n = 2，X = [1, 0]，那 Xᵀ W_kᵀ W_q X 是 2×2 矩阵 [[W_k W_q, 0],[0, 0]]，第二列恒为 [0,0] → Softmax 后第二列只能是 [0.5, 0.5]，表示不了 P = [[0.5, 0.75],[0.5, 0.25]]。
- **秩链**（这是原理但论文里没有单独成"引理"，是介绍里的表述）："a smaller head size introduces a rank constraint on the projection matrices in each head"——logits 矩阵 Xᵀ W_kᵀ W_q X 的秩 ≤ rank(W_kᵀ W_q) ≤ d_p，当 d_p < n 时它是 n×n 的降秩矩阵，Softmax 不能把任意 n×n 正矩阵的 log 恢复出来（log P 一般满秩）。

### 1.3 Theorem 2（多头版本：FixedMultiHead 严格强于 MultiHead）

> **Theorem 2.** Let n ≥ 2, d ≥ d_p, and h > d/d_p. Consider a FixedMultiHead attention layer g_V(·) with parameters that satisfy the following conditions: V_o × [V_v^1; …; V_v^h] is full rank, and (V_k^i)^T V_q^i = U, for all i = 1, …, h, where U is a rank-d_p matrix. Then, for any f_W ∈ F, there exists X ∈ R^{d×n} such that f_W(X) ≠ g_V(X).

中译：当 h > d/d_p（即头数多到标准做法下每头维度 d/h < d_p）时，存在一个满足简单条件的固定头维度层 g_V，标准多头层 F 里任何函数都无法在全部 X 上复现它——固定头维度的函数类 G 严格包含 F（前面已说明 d_p ≥ d/h 时 F ⊂ G）。

### 1.4 他们提的修法

**FixedMultiHead**：把每头投影维度 d_p 从 d/h 解耦出来，独立于 d 和 h：
```
fixedhead(X)_i = V_v^i X · Softmax[(V_k^i X)^T (V_q^i X)/√d_p] ∈ R^{d_p×n}
FixedMultiHead(X) = Concat[fixedhead(X)_1 … fixedhead(X)_h] ∈ R^{d_p·h×n}
Z = LN(X + V_o · FixedMultiHead(X)),  V_o ∈ R^{d×h·d_p}
```
建议值：**d_p = 输入序列长度 n**（"We propose to set the head size to input sequence length, and independent of the number of heads"）。BERT 实验取 d_p = 128，理由是预训练大多用 128 长度序列（"We choose head size to be 128 for our BERT experiments, as most of the pre-training is done with 128 sequence length data"）。代价：每层参数随头数增长（"the number of parameters per layer increases with the number of heads"）；好处：可以缩小 embedding size、头数不必整除 d。

### 1.5 实验与数字

| 实验 | 设置 | 结果 |
|---|---|---|
| Table 1（动机） | BERT_LARGE，24 层，d=1024，参数固定 336M，头数 8/16/32（标准 d_p = d/h，即 128/64/32） | SQuAD F1 90.89±0.15 / 90.61±0.14 / 90.45±0.08；EM 84.1 / 83.75 / 83.48；MNLI 85±0.2 / 84.5±0.4 / 84.4±0.2。**超过 8 头后下降**（注：d=1024, n=128，8 头恰好 d_p=128=n） |
| LM1B（Fig.1, Fig.3） | 6 层 Transformer，序列 256；baseline d 从 256→512；fixed 模型 d=256, d_p=32，头数 4→70 | fixed d=256 优于 baseline d=512（图，未给数字）；baseline 16 头以上变差，fixed 单调变好 |
| SQuAD/MNLI（Fig.2） | 24 层；baseline BERT_LARGE d 从 512→1024；fixed d=512, d_p=128，头数 8→32，参数量匹配 | "Transformers trained with a fixed head size and 512 embedding size have better performance than the baseline, BERT_LARGE" |
| Table 2(A) 增头数 | fixed d=512, d_p=128，头数 8/12/16/32，参数 168M/193M/218M/319M | SQuAD F1 89.6 / 90.25 / 90.43 / **90.95±0.14**；EM 82.73 / 83.18 / 83.59 / 84.4；MNLI 83.5 / 84.2 / 83.9 / 84.9 |
| Table 2(B) 增头维度 | fixed d=512, 8 头，d_p = 32/64/128/256，参数 130M/142M/168M/218M | SQuAD F1 88.53 / 89.51 / 89.6 / 90.33；EM 81.19 / 82.41 / 82.73 / 83.36；MNLI 82.5 / 83.4 / 83.5 / 83.9 |

注意：Table 2(B) 里 d_p=256 > n=128 仍继续涨，说明"头维度 = 序列长度"只是定理给的下限，实验上更大还有收益（但论文自己也说是 head size / heads / layers 之间的 tradeoff）。**"d=512 fixed 匹配 BERT_LARGE" 的对比里参数量不是更少**——Table 2(A) 32 头 319M vs BERT_LARGE 336M，差不多；论文说的"fewer parameters"主要指 embedding 更小。

---

## 2. 主流模型 d_model / 头数 / d_head 表（一手来源）

| 模型 | d_model | Q 头数 | KV 头数 | d_head（Q/K） | 备注 | 来源 |
|---|---|---|---|---|---|---|
| GPT-3 Small 125M | 768 | 12 | 12 | 64 | | GPT-3 论文 Table 2.1（arXiv 2005.14165） |
| GPT-3 Medium 350M | 1024 | 16 | 16 | 64 | | 同上 |
| GPT-3 Large 760M | 1536 | 16 | 16 | **96** | | 同上 |
| GPT-3 XL 1.3B | 2048 | 24 | 24 | **128**（从此锁定） | | 同上 |
| GPT-3 2.7B | 2560 | 32 | 32 | **80** | **反例：d_head=80** | 同上 |
| GPT-3 6.7B | 4096 | 32 | 32 | 128 | | 同上 |
| GPT-3 13B | 5140 | 40 | 40 | 128 | 论文原文就是 5140（不是 5120） | 同上 |
| GPT-3 175B | **12288** | **96** | 96 | **128** | n_ctx = 2048 | 同上 |
| Llama 3.1 8B | 4096 | 32 | 8 | 128（=4096/32） | GQA | Llama 3 论文 Table 3（arXiv 2407.21783） |
| Llama 3.1 70B | 8192 | 64 | 8 | 128 | GQA | 同上 |
| Llama 3.1 405B | **16384** | **128** | 8 | 128 | GQA；HF config 401（gated）未核到，以论文表为准 | 同上 |
| Qwen3-8B | 4096 | 32 | 8 | 128（config 显式 `head_dim: 128`） | GQA | HF `Qwen/Qwen3-8B` config.json |
| Qwen3-32B | 5120 | 64 | 8 | 128 | **64×128 = 8192 ≠ 5120**——Q 总维度 > d_model，d_head 不是 d_model/heads 算出来的 | HF `Qwen/Qwen3-32B` |
| Qwen3-235B-A22B | 4096 | 64 | 4 | 128 | MoE；64×128 = 8192 ≠ 4096，同上 | HF `Qwen/Qwen3-235B-A22B` |
| Qwen3-Next-80B-A3B | 2048 | 16 | 2 | **256** | **反例：全注意力层 d_head=256**；线性注意力层 key/value head_dim 128 | HF `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| DeepSeek-V3 | 7168 | 128 | 128（MLA，KV 压缩到 kv_lora_rank=512） | **qk = 128 nope + 64 rope = 192**；v_head_dim = 128；q_lora_rank 1536 | MLA | HF `deepseek-ai/DeepSeek-V3` |
| Kimi-K2-Instruct | 7168 | **64** | 64（MLA） | qk = 128 nope + 64 rope = 192；v 128；kv_lora_rank 512 | 同 V3 架构（model_type deepseek_v3），头数减半 | HF `moonshotai/Kimi-K2-Instruct` |
| Kimi-K3 | 7168 | 96 | 96（MLA） | 全注意力层 qk = 128 nope + 64 rope = 192，v 128，kv_lora_rank 512；KDA 线性注意力层 96 头 × head_dim 128 | 93 层，24 层全注意力（每 4 层一层）+ 65 层 KDA | HF `moonshotai/Kimi-K3` config（model_type kimi_k3） |
| DeepSeek-V4-Flash | 4096 | 64 | **1** | config 给 `head_dim: 512`、`qk_rope_head_dim: 64`、`q_lora_rank 1024`、`o_lora_rank 1024`；**无 qk_nope_head_dim / v_head_dim / kv_lora_rank 键** | model_type deepseek_v4；sliding_window 128；config 的 head_dim=512 含义**未核到**（大概率是共享 KV latent 宽度而非常规 d_head，别按 d_head=512 写） | HF `deepseek-ai/DeepSeek-V4-Flash` |
| DeepSeek-V4-Pro | 7168 | 128 | **1** | `head_dim: 512`、`qk_rope_head_dim: 64`、`index_head_dim: 128`、`q_lora_rank 1536` | 同上，语义未核到 | HF `deepseek-ai/DeepSeek-V4-Pro` |
| GLM-4.5 | 5120 | 96 | 8 | 128 | 96×128 = 12288 ≫ 5120（Q 维度是 d_model 的 2.4 倍） | HF `zai-org/GLM-4.5` |
| GLM-5 | 6144 | 64 | 64 | **qk = 192 nope + 64 rope = 256**；v_head_dim **256**；config 另有 `head_dim: 64`（含义不明） | model_type glm_moe_dsa（MLA/DSA 类） | HF `zai-org/GLM-5` |
| GLM-5.x（5.1 等） | 未核到 | | | | | |

### 2.1 "d_head 锁 128、d_model 涨头数跟着涨"——对不对？

- **基本对，但说法要收窄**：从 GPT-3 XL（1.3B）起到 175B、Llama 3 全系、Qwen3 dense/MoE、GLM-4.5，d_head 都是 128；d_model 从 2048 涨到 16384 时头数从 24 涨到 128，head 维度不动。
- **反例（老）**：GPT-3 自己内部就不整齐——Small/Medium 64、Large 96、2.7B **80**。GPT-3 论文明说这些数是"based on computational efficiency and load-balancing in the layout of models across GPU's"，并引 Kaplan 说 loss 对此不敏感。
- **反例（新）**：
  1. MLA 系（DeepSeek-V3 / Kimi K2 / K3）：Q/K 是 **192**（128 内容 + 64 RoPE），V 是 128。
  2. GLM-5：Q/K **256**（192+64），V 256。
  3. Qwen3-Next：全注意力层 **256**。
  4. DeepSeek-V4：config 结构变了（`head_dim: 512`, KV 头数 1），常规 d_head 概念不直接适用，未核到官方解释。
- **另一个别说错的点**：新模型里 **heads × d_head ≠ d_model** 很常见（Qwen3-32B 8192 vs 5120，GLM-4.5 12288 vs 5120，Qwen3-235B 8192 vs 4096）——"d_head = d_model / heads" 这个 Vaswani 式约定已经被打破，这恰恰就是 Bhojanapalli 那篇"把 d_p 与 d 解耦"的做法在工业界落地的形态。
- 趋势判断（我的读法，不是核到的官方表态）：头维度不是"锁死 128"，而是"128 是地板，新架构往 192/256 走"，与 Table 2(B) 的"头维度越大越好"方向一致。

---

## 3. 实验线两篇

| 论文 | 编号 / 会议 | 一句话结论（来自摘要原文） |
|---|---|---|
| Michel, Levy, Neubig, "Are Sixteen Heads Really Better than One?" | **arXiv 1905.10650**（v1 2019-05-25，v3 2019-11-04），NeurIPS 2019 | "even if models have been trained using multiple heads, in practice, a large percentage of attention heads can be removed at test time without significantly impacting performance. In fact, some layers can even be reduced to a single head." 并给出"训练动态在多头收益中起作用"的初步证据（训练时需要多头，推理时可以砍） |
| Voita, Talbot, Moiseev, Sennrich, Titov, "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned" | **arXiv 1905.09418**（v1 2019-05-23，v2 2019-06-07），ACL 2019（ACL Anthology P19-1580） | 最重要、最"自信"的头扮演一致且常可语言学解释的角色（位置头/句法头/罕见词头）；用 L0 松弛的随机门剪枝，专门化的头最后被剪；"on the English-Russian WMT dataset, pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU." |

Bhojanapalli 论文对这两篇的定位（§1.1 原话）："They observe that, during inference, many of the heads in each layer can be pruned away with a little effect on the prediction. However, they still need multiple heads during the training."——可以直接用来衔接"实验线说头能砍"与"理论线说头维度不能小"。

---

## 4. 未核到 / 需注意

- Llama 3.1 405B/70B 的 HF config.json（gated，401），数字来自论文 Table 3；论文表没有单列 head_dim，128 = 16384/128 = 8192/64 推算，与 8B（4096/32）一致。
- DeepSeek-V4 config 里 `head_dim: 512` 和 `num_key_value_heads: 1` 的准确含义（V4 技术报告未查），只列 config 原值。
- GLM-5 config 中 `head_dim: 64` 与 `qk_head_dim: 256` 并存，前者含义未核到。
- GLM-5.1 及以后版本未查。
- Bhojanapalli 的 ICML 正式版（PMLR）定理是否用 d_p 而非 d 陈述——只核了 arXiv v1 PDF，PMLR 版未逐字对比。
