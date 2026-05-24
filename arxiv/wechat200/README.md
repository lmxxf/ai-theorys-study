# wechat200: Sub-manifold Splitting Experiments

Zenodo DOI: 10.5281/zenodo.20364617

## Core Finding

eRank/TwoNN ratio is not a curvature indicator — it is a proxy for sub-manifold count. Splitting weight matrices along head boundaries reduces ratio by 1-2 orders of magnitude. Per-head ratio converges to 4-9x across Qwen3.6-27B and DeepSeek V4 Flash 280B.

## Experiment Code

### submanifold_split.py (main experiment)

The core script. Splits weight matrices by head / Q/K/V segment, measures eRank and TwoNN before and after.

```bash
# Qwen3.6-27B (9 layers, ~8 min)
python submanifold_split.py --model qwen \
  --ckpt-path /path/to/Qwen3.6-27B \
  --output submanifold_qwen_full.json

# DeepSeek V4 Flash (5 layers + 4 experts/layer, ~2 min)
python submanifold_split.py --model dsv4 \
  --ckpt-path /path/to/deepseek-v4-flash \
  --output submanifold_dsv4_full.json
```

Requires: PyTorch + safetensors + scipy. Runs on any GPU with 8GB+ VRAM (peak <1GB).

### weight_curvature.py (exploratory, not used in paper)

Local PCA tangent-space rotation + Forman-Ricci curvature on kNN graph. Tried as alternative curvature measures but found low discriminative power — most matrices cluster near the random baseline (angle ~ pi/2). Forman-Ricci frac_negative distinguishes embed/head (clustered) from MLP/experts (tree-like), but doesn't differentiate within each group. Kept for reference.

### dsv4_curvature.py (exploratory, not used in paper)

Same as weight_curvature.py but for DeepSeek V4 Flash with FP4/FP8 dequantization.

## Data Files

| File | Description |
|------|-------------|
| submanifold_qwen_full.json | Qwen3.6-27B, 9 layers (L00-L03, L07, L11, L31, L47, L63), q_proj head split + in_proj_qkv Q/K/V + V per-head split + gate_proj chunks |
| submanifold_dsv4_full.json | V4 Flash, 5 layers (L00, L10, L20, L30, L42), wq_b head split + 4 experts/layer w1/w2/w3 |
| submanifold_qwen_L8910.json | Qwen3.6-27B supplementary: L08/L09/L10 in_proj_qkv split (L07-L10 high orthogonality observation) |
| submanifold_split_results.json | Early test run (L00-L03 only), superseded by full runs |
| weight_curvature_test.json | Exploratory tangent rotation + Ricci, L00 only, v1 metrics |
| weight_curvature_test2.json | Same, v2 metrics (mean_angle + grassmann_dist) |
| dsv4_curvature_test.json | Exploratory tangent rotation + Ricci on V4 Flash, L00 only |

## Paper

| File | Description |
|------|-------------|
| paper_en.md / paper_en.pdf | English paper draft |
| paper_zh.md | Chinese paper draft |
| zenodo_metadata.md | Zenodo upload metadata |

## Hardware

NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory, sm_121). Any 8GB GPU works — analysis is per-matrix streaming, peak VRAM <1GB.
