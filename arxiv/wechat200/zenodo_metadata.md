# Zenodo Metadata

**Resource type**: Preprint

**Title**: Full Rank is an Illusion: Weight Matrices in Large Language Models are Composites of Low-Dimensional Sub-Manifolds

**Publication date**: 2026-05-24

**Author 1**: Jin, Yanyan / 独立研究者
**Author 2**: Zhao, Lei / 腾讯

**Description**: We show that the large gap between effective rank (eRank) and TwoNN intrinsic dimension in LLM weight matrices is not evidence of manifold curvature, but of sub-manifold concatenation. When weight matrices are split along known functional boundaries — attention heads, Q/K/V segments — the eRank/TwoNN ratio drops by 1–2 orders of magnitude. Experiments on Qwen3.6-27B and DeepSeek V4 Flash (280B) demonstrate that per-head ratios converge to 4–9x across both models despite 10x difference in parameter count. Value heads in DeltaNet linear attention are nearly flat (ratio ≈ 2x). Deep MoE expert weights develop internal sub-manifold structure absent in shallow layers. Code and data are open-sourced.

**Keywords**: intrinsic dimension, effective rank, weight geometry, sub-manifold, large language models, attention heads, TwoNN

**Languages**: eng

**License**: Creative Commons Attribution 4.0 International
