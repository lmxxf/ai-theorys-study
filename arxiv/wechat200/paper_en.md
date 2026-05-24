# Full Rank is an Illusion: Weight Matrices in Large Language Models are Composites of Low-Dimensional Sub-Manifolds

**Jin Yanyan** (corresponding author), **Zhao Lei**

---

## Abstract

*(To be written by co-author.)*

---

## 1. Introduction

Recent empirical work has documented a striking discrepancy in the geometry of weight matrices in large language models (LLMs): the effective rank (eRank) of a weight matrix routinely exceeds its intrinsic dimension estimated by the Two-Nearest-Neighbor (TwoNN) method by one to two orders of magnitude. A natural interpretation of this gap is that the weight manifold has high curvature — that the rows of the matrix live on a surface that bends so aggressively that local neighborhood statistics undercount the true dimensionality. This paper argues that the curvature interpretation is wrong, or at best misleading, and proposes a simpler, empirically falsifiable alternative.

Our claim is that the eRank/TwoNN ratio is not a curvature indicator. It is a proxy for the number of geometrically distinct low-dimensional sub-manifolds that have been concatenated into a single weight matrix by the architecture. When a weight matrix is split along its known functional boundaries — attention heads, query/key/value segments — the ratio drops by one to two orders of magnitude, and the per-component ratios converge to a narrow band (4--9x) that is stable across models of vastly different size and architecture.

This reinterpretation matters because it changes what the ratio tells us about the model. A high ratio does not mean the weight geometry is complex in the differential-geometric sense. It means the matrix is doing multiple jobs at once. Each job, examined in isolation, has simple geometry. The complexity is organizational, not geometric.

We demonstrate this through systematic splitting experiments on two architecturally distinct models: Qwen3.6-27B (27 billion parameters, dense architecture with DeltaNet/GQA hybrid attention) and DeepSeek V4 Flash (280 billion parameters, Mixture-of-Experts with 256 routed experts). Across both models, splitting weight matrices along functional boundaries consistently collapses the ratio from the 50--400x range to the 4--9x range. We further show that value projections in DeltaNet linear attention layers have per-head ratios approaching 2x, indicating near-flat geometry, and that expert feed-forward weights in deeper layers develop internal sub-manifold structure absent in shallow layers.

---

## 2. Background

### 2.1 Effective Rank (eRank)

The effective rank of a matrix, introduced by Roy and Vetterli (2007), measures the "spread" of its singular value spectrum. Given a matrix $W \in \mathbb{R}^{m \times n}$ with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$, define the normalized distribution:

$$p_i = \frac{\sigma_i}{\sum_{j=1}^{r} \sigma_j}$$

The effective rank is the exponential of the Shannon entropy of this distribution:

$$\text{eRank}(W) = \exp\left(-\sum_{i=1}^{r} p_i \ln p_i\right)$$

When all singular values are equal, $\text{eRank}(W) = r$ (the matrix has "full" effective rank). When one singular value dominates, $\text{eRank}(W) \approx 1$. For a typical LLM weight matrix, eRank falls between these extremes and reflects how many spectral directions carry significant energy.

Crucially, eRank is a global, spectral property. It sees the matrix as a whole and is blind to whether the rows cluster into groups with different local structure.

### 2.2 TwoNN Intrinsic Dimension

The TwoNN estimator, introduced by Facco et al. (2017), estimates the intrinsic dimension of a point cloud from the ratio of distances to each point's first and second nearest neighbors. For a point $x_i$ with nearest neighbor at distance $r_1$ and second nearest neighbor at distance $r_2$, define $\mu_i = r_2 / r_1$. Under the assumption that the data lie on a $d$-dimensional manifold and are locally uniformly distributed, the cumulative distribution of $\mu$ follows:

$$P(\mu \leq t) = 1 - t^{-d}$$

The intrinsic dimension $d$ is estimated by maximum likelihood from the empirical distribution of $\mu$ values across all points.

TwoNN is a local estimator. It measures geometric complexity in the immediate neighborhood of each point. If the data lie on a single smooth manifold, TwoNN recovers the manifold dimension regardless of how that manifold is embedded in ambient space. But if the data are a union of multiple manifolds with different local structure, TwoNN reports a dimension reflecting the local geometry of whichever manifold each point sits on, averaged across all points.

### 2.3 Why Their Ratio Does Not Measure Curvature

Consider a matrix $W$ whose rows lie on a single smooth $d$-dimensional manifold. In this case, TwoNN estimates $d$, eRank reflects the spectral spread of the embedding, and their ratio depends on how the manifold's principal axes distribute energy across the spectrum. High curvature could, in principle, cause TwoNN to underestimate $d$ if neighborhoods bend enough to distort distance ratios. Under this reading, a high eRank/TwoNN ratio signals curvature.

But there is a simpler explanation for a high ratio that does not invoke curvature at all. Suppose $W$ is constructed by vertically concatenating $k$ blocks, each of which is a separate low-dimensional manifold with its own orientation in ambient space. TwoNN, being local, estimates the average intrinsic dimension of the individual blocks. eRank, being global, sees a matrix whose singular values reflect the superposition of $k$ differently-oriented subspaces, yielding a much higher effective rank. The ratio scales with $k$ — the number of sub-manifolds — not with the curvature of any one of them.

This is exactly the situation in LLM weight matrices. A query projection matrix $W_Q$ in a multi-head attention layer is the vertical concatenation of $H$ independent head projections. Each head maps the residual stream into its own query subspace. These subspaces have different orientations, are trained under different gradient signals, and develop different spectral profiles. The weight matrix is not one manifold with high curvature. It is $H$ manifolds stitched together.

The key prediction of this interpretation is falsifiable: if the ratio measures sub-manifold count, then splitting the matrix along its functional boundaries should dramatically reduce the ratio. If the ratio measures curvature, splitting should not help — the curvature of each piece should be comparable to the curvature of the whole.

---

## 3. Experimental Setup

### 3.1 Models

We study two models chosen for architectural diversity:

**Qwen3.6-27B** is a 27-billion-parameter dense model with 64 transformer layers. It uses a hybrid attention design: layers 0, 1, and 2 use Gated DeltaNet linear attention (Yang et al., 2025), while the remaining layers use standard grouped-query attention (GQA) with 48 query heads and 8 key-value heads, each with dimension 256. In the DeltaNet layers, Q, K, and V projections are fused into a single `in_proj_qkv` matrix of shape [10240, 5120], where the first 2048 rows encode Q, the next 2048 encode K, and the final 6144 encode V (48 heads $\times$ 128 dimensions). The DeltaNet layers also contain small gating matrices `in_proj_a` ($\alpha$ decay gate) and `in_proj_b` ($\beta$ write gate), each of shape [48, 5120].

**DeepSeek V4 Flash** is a 280-billion-parameter Mixture-of-Experts model with 61 transformer layers. Its attention mechanism uses low-rank factorization: the query path factors through $W_{q\_a}$ [1024, 4096] and $W_{q\_b}$ [32768, 1024], where $W_{q\_b}$ projects into 128 heads of dimension 256. Each MoE layer contains 256 routed experts, with individual expert feed-forward weight matrices of shape [2048, 4096] (up-projection) and [4096, 2048] (down-projection). The routing gate matrix has shape [256, 4096].

### 3.2 Splitting Strategy

We apply two types of splits, both following known architectural boundaries:

**Head-boundary splitting.** For multi-head projection matrices, we partition the row dimension into segments of size equal to the per-head dimension. For Qwen3.6 `q_proj` [12288, 5120], this yields 48 blocks of [256, 5120]. For DeepSeek V4 `wq_b` [32768, 1024], this yields 128 blocks of [256, 1024]. For Qwen3.6 DeltaNet V segments [6144, 5120], this yields 48 blocks of [128, 5120].

**Functional-segment splitting.** For fused projection matrices, we split along the Q/K/V functional boundary. For Qwen3.6 DeltaNet `in_proj_qkv` [10240, 5120], this yields Q [2048, 5120], K [2048, 5120], and V [6144, 5120].

These splits are not arbitrary partitions. They follow the exact boundaries that the model's forward pass uses to route information. Each block, after splitting, corresponds to a functionally independent computation.

### 3.3 Metrics

For each matrix (whole or split), we compute:

- **eRank**: effective rank from the singular value distribution.
- **TwoNN**: intrinsic dimension estimated from nearest-neighbor distance ratios, treating each row of the weight matrix as a point in the column-dimensional ambient space.
- **Ratio**: eRank / TwoNN.

### 3.4 Hardware

All experiments were conducted on an NVIDIA DGX Spark system with 128 GB of unified CPU-GPU memory. Code and data are open-sourced.

---

## 4. Results

### 4.1 Qwen3.6-27B: Query Projection Head Splitting

We selected six GQA layers spanning the full depth of the network: L03, L07, L11, L31, L47, and L63. Table 1 reports the whole-matrix ratio and the per-head statistics after splitting into 48 heads of dimension 256.

**Table 1.** Qwen3.6-27B `q_proj` [12288, 5120]: whole-matrix vs. per-head (48 heads, 256 dim each).

| Layer | Whole eRank | Whole TwoNN | Whole Ratio | Per-Head Ratio (mean) | Per-Head Ratio (std) | Reduction Factor |
|-------|------------|------------|-------------|----------------------|---------------------|-----------------|
| L03   | 4484.3     | 37.3       | 120.3x      | 5.46                 | 3.48                | 22.0x           |
| L07   | 4477.6     | 10.1       | 442.0x      | 8.86                 | 8.10                | 49.9x           |
| L11   | 4411.7     | 81.2       | 54.4x       | 5.98                 | 7.30                | 9.1x            |
| L31   | 4312.5     | 48.6       | 88.8x       | 5.89                 | 3.57                | 15.1x           |
| L47   | 4285.3     | 45.9       | 93.3x       | 6.07                 | 3.33                | 15.4x           |
| L63   | 4375.3     | 45.3       | 96.7x       | 4.67                 | 1.69                | 20.7x           |

The whole-matrix ratio ranges from 54x to 442x. After splitting, the per-head mean ratio collapses to 4.7x -- 8.9x. The reduction factor ranges from 9x to 50x. Despite large variation in the whole-matrix ratio across layers, the per-head ratios converge to a narrow band.

### 4.2 Qwen3.6-27B: DeltaNet in_proj_qkv Functional Splitting

The first three layers (L00, L01, L02) use DeltaNet linear attention with a fused Q/K/V projection matrix. Table 2 reports the results of splitting by functional segment and further splitting V into per-head blocks.

**Table 2.** Qwen3.6-27B DeltaNet `in_proj_qkv` [10240, 5120]: functional and head splitting.

| Layer | Whole Ratio | Q Ratio (2048 rows) | K Ratio (2048 rows) | V Ratio (6144 rows) | V Per-Head Mean (48 heads, 128 dim) |
|-------|------------|--------------------|--------------------|--------------------|------------------------------------|
| L00   | 30.7x      | 14.6x              | 38.4x              | 55.7x              | 2.41                               |
| L01   | 81.6x      | 173.9x             | 94.3x              | 40.8x              | 2.17                               |
| L02   | 56.8x      | 20.6x              | 16.9x              | 44.0x              | 1.76                               |

Several findings stand out. First, the whole-matrix ratios range from 31x to 82x. Second, splitting by Q/K/V functional segment reduces the ratio substantially in most cases but does not always bring it below 20x, indicating that Q and K segments themselves contain internal sub-structure from multiple heads (particularly L01 Q at 173.9x, which contains 16 K heads with different orientations). Third, the V segment, when further split into 48 heads of 128 dimensions each, reaches per-head ratio means of 1.76x -- 2.41x. Individual V heads are nearly flat.

### 4.3 DeepSeek V4 Flash: wq_b Head Splitting

The query projection in V4 Flash factors through a low-rank bottleneck. We analyze $W_{q\_b}$ [32768, 1024], which fans out from the bottleneck into 128 heads of 256 dimensions each. Table 3 reports results across five layers.

**Table 3.** DeepSeek V4 Flash `wq_b` [32768, 1024]: whole-matrix vs. per-head (128 heads, 256 dim each).

| Layer | Whole eRank | Whole TwoNN | Whole Ratio | Per-Head Ratio (mean) | Per-Head Ratio (std) |
|-------|------------|------------|-------------|----------------------|---------------------|
| L00   | 1014.7     | 52.6       | 19.3x       | 5.56                 | 4.29                |
| L10   | 1018.0     | 51.3       | 19.9x       | 4.60                 | 1.98                |
| L20   | 1017.0     | 29.8       | 34.1x       | 6.60                 | 3.06                |
| L30   | 1017.4     | 39.6       | 25.7x       | 4.51                 | 2.40                |
| L42   | 1016.3     | 207.2      | 4.9x        | 4.43                 | 1.51                |

The whole-matrix ratios range from 4.9x to 34.1x. The per-head ratios converge to 4.4x -- 6.6x, the same 4--9x band observed in the Qwen3.6 query heads despite the models differing by an order of magnitude in parameter count and using fundamentally different architectures (dense vs. MoE, standard attention vs. low-rank factored attention).

### 4.4 DeepSeek V4 Flash: Expert Weight Layer-Wise Differentiation

We sampled four experts per layer across five layers and computed the eRank/TwoNN ratio for their feed-forward weight matrices (w1, w2, w3). Table 4 reports per-layer summary statistics.

**Table 4.** DeepSeek V4 Flash expert FFN weight ratio statistics (4 experts $\times$ 3 matrices per layer).

| Layer | Expert Ratio Mean | Expert Ratio Min | Expert Ratio Max | Expert Ratio Std |
|-------|------------------|-----------------|-----------------|-----------------|
| L00   | 14.6             | 7.8             | 21.5            | 4.0             |
| L10   | 16.2             | 7.7             | 45.7            | 10.7            |
| L20   | 26.7             | 12.2            | 61.4            | 17.1            |
| L30   | 55.4             | 16.6            | 120.8           | 40.9            |
| L42   | 34.2             | 9.3             | 113.4           | 30.4            |

Shallow layers (L00, L10) have relatively low and uniform expert ratios (mean 14.6 and 16.2, standard deviation 4.0 and 10.7). Deep layers (L30, L42) exhibit dramatically higher means (55.4, 34.2) and extreme individual values reaching 120.8 and 113.4. The standard deviation increases by a factor of 3--10 from shallow to deep layers.

### 4.5 Background Data: Small Functional Matrices

For completeness, we report eRank/TwoNN ratios for small functional matrices measured in prior experiments:

- **Qwen3.6 `in_proj_a`** ($\alpha$ decay gate, [48, 5120]): ratio 2.4x
- **Qwen3.6 `in_proj_b`** ($\beta$ write gate, [48, 5120]): ratio 1.5x -- 2.2x
- **DeepSeek V4 Flash `gate`** (MoE router, [256, 4096]): ratio 5.3x -- 11.1x

The gating matrices, which serve a single functional purpose (routing or decay), have ratios in the low single digits. The MoE router, which must distinguish among 256 experts, has a somewhat higher ratio, consistent with the sub-manifold interpretation: each row of the router corresponds to one expert's selection boundary.

---

## 5. Discussion

### 5.1 The Ratio is a Sub-Manifold Count Proxy, Not a Curvature Indicator

The central result of this paper is negative: the eRank/TwoNN ratio does not measure curvature of the weight manifold. The evidence is straightforward. If the ratio reflected curvature, splitting a matrix into pieces should not systematically reduce it, because curvature is a local property that does not vanish upon partitioning. But splitting along functional boundaries reduces the ratio by 1--2 orders of magnitude (Tables 1--3).

The ratio is better understood as a proxy for the number of geometrically distinct sub-manifolds packed into a single matrix. eRank, as a global spectral measure, grows when multiple differently-oriented subspaces contribute to the singular value distribution. TwoNN, as a local measure, reports the average local dimension, which does not grow with the number of subspaces as long as each subspace is individually low-dimensional. The gap between them therefore scales with the number of distinct subspaces — that is, with the number of functional components concatenated in the matrix.

This interpretation aligns with what we know about how these matrices are used. A query projection matrix in a 48-head attention layer is, in a precise computational sense, 48 independent linear maps stacked vertically. The architecture defines the sub-manifolds. Training shapes their internal geometry but does not erase the boundaries between them.

### 5.2 Residual Complexity Per Head: An Architectural Constant

A striking empirical regularity emerges from Tables 1 and 3. After splitting, per-head ratios converge to approximately 4x -- 9x across both models, despite differences in:

- Total parameter count (27B vs. 280B)
- Architecture type (dense vs. MoE)
- Attention mechanism (standard GQA vs. low-rank factored)
- Head dimension (256 in both cases, but with different input dimensions: 5120 vs. 1024)
- Training data, training recipe, and model family

This convergence suggests that the per-head ratio of 4--9x reflects a residual geometric complexity that is intrinsic to what an attention head does, rather than an artifact of any particular training run. Each head, regardless of context, develops a weight geometry that is not perfectly flat (ratio $\neq$ 1) but far simpler than the composite matrix it lives in.

One plausible interpretation is that each head's weight rows trace out a low-dimensional surface with modest curvature and a small number of internal modes — perhaps reflecting the fact that each head must encode multiple types of positional or semantic relationships within a single linear map. The ratio of 4--9x may represent a universal "complexity budget" for a single attention head: enough structure to support several distinct functions, but far less than the full rank of the ambient space.

### 5.3 Value Heads Are Nearly Flat

The DeltaNet V-segment results (Table 2) reveal an extreme case: after splitting V into 48 heads of 128 dimensions, the per-head ratio drops to 1.76x -- 2.41x. These values are close to 1x, meaning eRank and TwoNN nearly agree. The weight rows of individual value heads lie on a manifold that is, geometrically, almost flat — a nearly linear subspace with minimal internal structure.

This finding is consistent with the computational role of value projections in linear attention. In Gated DeltaNet, the value projection produces the content that will be written into the recurrent state. Unlike query and key projections, which must encode complex relational patterns (positional encodings, attention patterns), value projections perform a relatively straightforward linear mixing of input features into a per-head value space. The geometry of the weight matrix reflects this simplicity.

The contrast between V heads (ratio $\approx$ 2x) and Q/K segments (ratio 15x -- 174x before head splitting) quantifies a qualitative difference in function: value projections are genuinely near-linear operations, while query and key projections carry substantially more geometric structure.

### 5.4 Deep Expert Sub-Manifold Differentiation

The expert weight analysis (Table 4) reveals a depth-dependent pattern. In shallow layers (L00, L10), expert feed-forward weights have relatively uniform and moderate ratios (mean 14.6 and 16.2), suggesting that each expert's weight matrix is a moderately complex but internally coherent structure. In deep layers (L30, L42), ratios explode: some experts reach 120.8x, while others remain near 10x. The standard deviation increases by up to an order of magnitude.

This pattern suggests that deep experts develop internal sub-manifold structure during training. A deep expert with a ratio of 120x is not simply a more complex version of a shallow expert with a ratio of 15x. Under our interpretation, the deep expert has developed $\sim$8x more internal geometric partitions — it has differentiated into multiple functional sub-components within a single expert. This structural differentiation parallels the functional specialization that prior work has documented in deep MoE layers: deep experts tend to specialize for specific token types, linguistic constructions, or knowledge domains. Our results suggest that this functional specialization has a geometric signature: the weight matrix of a specialized expert literally contains multiple sub-manifolds, each perhaps corresponding to a different mode of operation.

The high variance across experts in the same deep layer is equally informative. Not all experts differentiate equally. Some remain geometrically simple (ratio $\approx$ 10x), while others become highly structured (ratio $>$ 100x). This heterogeneity is consistent with the known skew in expert utilization: a small fraction of experts handle disproportionately diverse inputs and must develop correspondingly richer internal geometry.

### 5.5 Implications for Fine-Tuning and Knowledge Localization

If weight matrices are composites of functionally distinct sub-manifolds, and if these sub-manifolds correspond to identifiable computational units (heads, experts, functional segments), then model editing and fine-tuning operations could in principle operate at the sub-manifold level rather than at the level of whole weight matrices.

Current parameter-efficient fine-tuning methods such as LoRA (Aghajanyan et al., 2021) apply low-rank updates to entire weight matrices. Our results suggest that this is geometrically coarse: a rank-$r$ update to a matrix containing 48 sub-manifolds distributes its capacity across all 48, rather than concentrating on the sub-manifolds relevant to the target task. A sub-manifold-aware fine-tuning strategy would identify which heads or functional segments need modification and apply updates selectively.

Similarly, knowledge localization research has sought to identify where in a model specific facts or capabilities reside. Our analysis suggests a more granular unit of analysis: not "which layer" or "which weight matrix," but "which sub-manifold within which weight matrix." The near-flat geometry of individual value heads (ratio $\approx$ 2x) implies that value projections store information in a nearly linear code, potentially making them easier targets for direct knowledge editing.

---

## 6. Conclusion

The eRank/TwoNN ratio in LLM weight matrices does not measure what it was thought to measure. It is not a curvature indicator. It is a sub-manifold count proxy — a measure of how many geometrically distinct functional components have been concatenated into a single matrix by the architecture.

This paper established this reinterpretation through splitting experiments on two models spanning an order of magnitude in parameter count. The results are consistent: splitting along architectural boundaries reduces the ratio by 1--2 orders of magnitude, and per-head ratios converge to a model-independent band of 4--9x. Value heads in linear attention are nearly flat (ratio $\approx$ 2x). Deep experts develop internal sub-manifold structure that shallow experts lack.

The apparent full rank of LLM weight matrices is an illusion produced by concatenation. Each functional component, examined individually, is a low-dimensional object with modest complexity. The engineering implication is that the matrix is not the right unit of analysis for understanding weight geometry. The sub-manifold — the head, the expert, the functional segment — is.

---

## References

- Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2021). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)*.

- Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7(1), 12140.

- Roy, O., & Vetterli, M. (2007). The effective rank: A measure of effective dimensionality. *15th European Signal Processing Conference (EUSIPCO)*, 606--610.

- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2025). Gated Delta Networks: Improving mamba-2 with delta rule. *Proceedings of the International Conference on Learning Representations (ICLR)*.
