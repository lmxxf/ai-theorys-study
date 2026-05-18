"""
qwen36_intrinsic_dim.py — Measure intrinsic dimensionality of Qwen3.6-27B weights.

Core question: Do GatedDeltaNet (linear attention) layers have flatter weight manifolds
than standard GQA (full attention) layers?

Hypothesis: DeltaNet's state update is strictly linear (no softmax, no nonlinear activation).
If a weight matrix only needs to learn linear operations, the training gradients may not
bend the weight manifold much — eRank/TwoNN ratio should be small (close to 1 = flat).
Standard attention layers with softmax should show large eRank/TwoNN ratio (curved).

Measures both eRank (linear, "bounding box") and TwoNN (manifold, "tree inside the box")
for every weight matrix, then compares linear_attn vs self_attn layers.

Usage:
  python qwen36_intrinsic_dim.py \
    --ckpt-path /home/lmxxf/work/models/Qwen3.6-27B \
    --output-dir /home/lmxxf/work/ai-theorys-study/experiments/qwen36_results \
    --layers 0,1,2,3,7,11,31,47,63
"""

import argparse
import json
import os
import time
from glob import glob
from collections import defaultdict

import torch
import numpy as np
from safetensors import safe_open


def twonn_dimension(X: torch.Tensor, max_points: int = 5000) -> float:
    N = X.shape[0]
    if N < 10:
        return float("nan")
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        X = X[idx]
        N = max_points

    chunk_size = 512
    mus = []

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        Xi = X[i:end_i]
        dists = torch.cdist(Xi, X)
        for j in range(end_i - i):
            dists[j, i + j] = float("inf")
        top2, _ = dists.topk(2, dim=1, largest=False)
        r1 = top2[:, 0]
        r2 = top2[:, 1]
        mu = r2 / r1.clamp(min=1e-30)
        mus.append(mu)

    mus = torch.cat(mus)
    mus = mus[mus > 1.0]
    if len(mus) < 10:
        return float("nan")

    d = float(len(mus) / torch.log(mus).sum())
    return d


def compute_erank(W: torch.Tensor) -> dict:
    sv = torch.linalg.svdvals(W)
    sv_np = sv.cpu().numpy().astype(np.float64)
    sv_np = sv_np[sv_np > 0]

    if len(sv_np) == 0:
        return {"erank": 0, "stable_rank": 0}

    p = sv_np / sv_np.sum()
    H = -np.sum(p * np.log(p))
    erank = float(np.exp(H))
    stable_rank = float((sv_np ** 2).sum() / (sv_np[0] ** 2))

    return {
        "erank": round(erank, 2),
        "stable_rank": round(stable_rank, 2),
        "numerical_rank": int((sv_np > sv_np[0] * 1e-5).sum()),
        "condition_number": float(sv_np[0] / sv_np[-1]) if sv_np[-1] > 0 else float("inf"),
    }


def analyze_weight(W: torch.Tensor, name: str, device: str) -> dict:
    M, N = W.shape
    W_gpu = W.to(device).float()

    t0 = time.time()
    erank_stats = compute_erank(W_gpu)
    svd_time = time.time() - t0

    t0 = time.time()
    twonn_row = twonn_dimension(W_gpu)
    twonn_time = time.time() - t0

    t0 = time.time()
    twonn_col = twonn_dimension(W_gpu.T)
    twonn_col_time = time.time() - t0

    ratio_row = erank_stats["erank"] / twonn_row if twonn_row > 0 and not np.isnan(twonn_row) else float("nan")
    ratio_col = erank_stats["erank"] / twonn_col if twonn_col > 0 and not np.isnan(twonn_col) else float("nan")

    result = {
        "name": name,
        "shape": [M, N],
        "min_dim": min(M, N),
        **erank_stats,
        "twonn_row": round(twonn_row, 2) if not np.isnan(twonn_row) else None,
        "twonn_col": round(twonn_col, 2) if not np.isnan(twonn_col) else None,
        "erank_twonn_ratio_row": round(ratio_row, 2) if not np.isnan(ratio_row) else None,
        "erank_twonn_ratio_col": round(ratio_col, 2) if not np.isnan(ratio_col) else None,
        "svd_time_s": round(svd_time, 2),
        "twonn_time_s": round(twonn_time + twonn_col_time, 2),
    }

    del W_gpu
    torch.cuda.empty_cache()
    return result


def classify_qwen36_weight(key: str, target_layers: set = None):
    """Returns (layer_idx, layer_type, weight_name) or None."""
    if "language_model.layers." not in key:
        if "embed_tokens" in key:
            return (-1, "embed", "embed_tokens")
        if "lm_head" in key:
            return (999, "head", "lm_head")
        return None

    parts = key.split(".")
    idx = parts.index("layers") + 1
    layer_idx = int(parts[idx])

    if target_layers is not None and layer_idx not in target_layers:
        return None

    rest = ".".join(parts[idx + 1:])

    # Linear attention (DeltaNet) layers
    if "linear_attn.in_proj_qkv" in rest:
        return (layer_idx, "linear_attn", "in_proj_qkv")
    if "linear_attn.in_proj_a" in rest:
        return (layer_idx, "linear_attn", "in_proj_a")
    if "linear_attn.in_proj_b" in rest:
        return (layer_idx, "linear_attn", "in_proj_b")
    if "linear_attn.in_proj_z" in rest:
        return (layer_idx, "linear_attn", "in_proj_z")
    if "linear_attn.out_proj" in rest:
        return (layer_idx, "linear_attn", "out_proj")

    # Full attention (GQA) layers
    if "self_attn.q_proj" in rest:
        return (layer_idx, "full_attn", "q_proj")
    if "self_attn.k_proj" in rest:
        return (layer_idx, "full_attn", "k_proj")
    if "self_attn.v_proj" in rest:
        return (layer_idx, "full_attn", "v_proj")
    if "self_attn.o_proj" in rest:
        return (layer_idx, "full_attn", "o_proj")

    # MLP (shared between both layer types)
    if "mlp.gate_proj" in rest:
        return (layer_idx, "mlp", "gate_proj")
    if "mlp.up_proj" in rest:
        return (layer_idx, "mlp", "up_proj")
    if "mlp.down_proj" in rest:
        return (layer_idx, "mlp", "down_proj")

    return None


def collect_small_weights(shard_files: list, target_layers: set, device: str):
    """Collect A_log and dt_bias across all target layers for summary statistics."""
    a_logs = {}
    dt_biases = {}

    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "linear_attn.A_log" in key and "language_model.layers." in key:
                    parts = key.split(".")
                    layer_idx = int(parts[parts.index("layers") + 1])
                    if layer_idx in target_layers:
                        a_logs[layer_idx] = f.get_tensor(key).float().numpy()
                elif "linear_attn.dt_bias" in key and "language_model.layers." in key:
                    parts = key.split(".")
                    layer_idx = int(parts[parts.index("layers") + 1])
                    if layer_idx in target_layers:
                        dt_biases[layer_idx] = f.get_tensor(key).float().numpy()

    return a_logs, dt_biases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,1,2,3,7,11,31,47,63",
                        help="Comma-separated layer indices to analyze")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    target_layers = set(int(x) for x in args.layers.split(","))
    # layer_types from config: linear=0,1,2, full=3, linear=4,5,6, full=7, ...
    linear_layers = set(i for i in range(64) if i % 4 != 3)
    full_layers = set(i for i in range(64) if i % 4 == 3)

    print(f"Target layers: {sorted(target_layers)}", flush=True)
    print(f"  Linear attention: {sorted(target_layers & linear_layers)}", flush=True)
    print(f"  Full attention:   {sorted(target_layers & full_layers)}", flush=True)

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"Found {len(shard_files)} shards", flush=True)

    results = []
    t_start = time.time()

    for shard_idx, shard_file in enumerate(shard_files):
        shard_name = os.path.basename(shard_file)

        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                if not key.endswith(".weight"):
                    continue

                info = classify_qwen36_weight(key, target_layers)
                if info is None:
                    continue

                layer_idx, layer_type, weight_name = info

                tensor = f.get_tensor(key)
                if tensor.dim() != 2:
                    del tensor
                    continue

                full_name = f"L{layer_idx:02d}.{layer_type}.{weight_name}"
                M, N = tensor.shape
                print(f"  [{shard_name}] {full_name} [{M}, {N}]", end="", flush=True)

                stats = analyze_weight(tensor, full_name, args.device)
                stats["layer"] = layer_idx
                stats["layer_type"] = layer_type
                stats["weight_name"] = weight_name
                results.append(stats)

                erank = stats["erank"]
                twonn_r = stats["twonn_row"]
                twonn_c = stats["twonn_col"]
                ratio_r = stats["erank_twonn_ratio_row"]
                ratio_c = stats["erank_twonn_ratio_col"]
                print(f" → eRank={erank:.0f}  TwoNN_row={twonn_r}  TwoNN_col={twonn_c}  "
                      f"ratio_row={ratio_r}  ratio_col={ratio_c}  "
                      f"({stats['svd_time_s'] + stats['twonn_time_s']:.1f}s)", flush=True)

                del tensor

    # Collect A_log and dt_bias
    print("\nCollecting A_log and dt_bias...", flush=True)
    a_logs, dt_biases = collect_small_weights(shard_files, target_layers & linear_layers, args.device)

    elapsed = time.time() - t_start
    print(f"\nDone! {len(results)} matrices in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Save results
    output = {
        "model": "Qwen3.6-27B",
        "dtype": "bfloat16",
        "hidden_size": 5120,
        "num_layers": 64,
        "target_layers": sorted(target_layers),
        "elapsed_s": round(elapsed, 1),
        "weight_results": results,
        "a_log_stats": {
            str(layer): {
                "values": a_logs[layer].tolist(),
                "mean": float(a_logs[layer].mean()),
                "std": float(a_logs[layer].std()),
                "min": float(a_logs[layer].min()),
                "max": float(a_logs[layer].max()),
            } for layer in sorted(a_logs.keys())
        },
        "dt_bias_stats": {
            str(layer): {
                "values": dt_biases[layer].tolist(),
                "mean": float(dt_biases[layer].mean()),
                "std": float(dt_biases[layer].std()),
            } for layer in sorted(dt_biases.keys())
        },
    }

    out_path = os.path.join(args.output_dir, "qwen36_intrinsic_dim.json")
    with open(out_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"Results saved to {out_path}", flush=True)

    # Summary table
    print(f"\n{'='*110}")
    print(f"{'Name':40s} {'Shape':18s} {'eRank':>7s} {'TwoNN_r':>8s} {'TwoNN_c':>8s} {'Ratio_r':>8s} {'Ratio_c':>8s} {'Type':>12s}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: (x["layer"], x["layer_type"], x["weight_name"])):
        tr = r["twonn_row"] if r["twonn_row"] else "N/A"
        tc = r["twonn_col"] if r["twonn_col"] else "N/A"
        rr = r["erank_twonn_ratio_row"] if r["erank_twonn_ratio_row"] else "N/A"
        rc = r["erank_twonn_ratio_col"] if r["erank_twonn_ratio_col"] else "N/A"
        tr_s = f"{tr:.1f}" if isinstance(tr, float) else tr
        tc_s = f"{tc:.1f}" if isinstance(tc, float) else tc
        rr_s = f"{rr:.1f}x" if isinstance(rr, float) else rr
        rc_s = f"{rc:.1f}x" if isinstance(rc, float) else rc
        print(f"{r['name']:40s} {str(r['shape']):18s} {r['erank']:7.0f} {tr_s:>8s} {tc_s:>8s} {rr_s:>8s} {rc_s:>8s} {r['layer_type']:>12s}")

    # Summary by type
    print(f"\n{'='*80}")
    print("SUMMARY: eRank/TwoNN ratio by layer type (higher = more curved)")
    print(f"{'Type':15s} {'Weight':15s} {'Avg Ratio':>10s} {'N':>5s}")
    print("-" * 50)

    type_ratios = defaultdict(list)
    for r in results:
        if r["erank_twonn_ratio_row"] is not None:
            key = f"{r['layer_type']}.{r['weight_name']}"
            type_ratios[key].append(r["erank_twonn_ratio_row"])

    for key in sorted(type_ratios.keys()):
        vals = type_ratios[key]
        lt, wn = key.split(".", 1)
        print(f"{lt:15s} {wn:15s} {np.mean(vals):10.1f}x {len(vals):5d}")

    # A_log summary
    if a_logs:
        print(f"\n{'='*80}")
        print("A_log (decay rate baseline) per layer:")
        for layer in sorted(a_logs.keys()):
            vals = a_logs[layer]
            decay = np.exp(-np.exp(vals))
            print(f"  L{layer:02d}: A_log mean={vals.mean():.3f} std={vals.std():.3f}  "
                  f"→ decay factor mean={decay.mean():.4f} range=[{decay.min():.4f}, {decay.max():.4f}]")


if __name__ == "__main__":
    main()
