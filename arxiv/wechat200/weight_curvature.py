"""
weight_curvature.py — Measure extrinsic bending and intrinsic curvature of Qwen3.6-27B weights.

Continues from 195's eRank/TwoNN by splitting "curvature" into:
  1. Extrinsic bending: local PCA tangent space rotation rate
  2. Intrinsic curvature: Forman-Ricci on kNN graph

Tests the same 59 matrices from 195, at multiple neighborhood scales.

Usage:
  python weight_curvature.py \
    --ckpt-path /home/lmxxf/work/models/Qwen3.6-27B \
    --prev-results /home/lmxxf/work/ai-theorys-study/wechat/assets/195/qwen36_intrinsic_dim.json \
    --output weight_curvature_results.json
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


# ──────────────────────────────────────────────
# kNN utilities
# ──────────────────────────────────────────────

def build_knn_cosine(X: torch.Tensor, k: int):
    """Build kNN graph using cosine distance. Returns (indices, distances).
    X: [N, D], already L2-normalized.
    Returns: indices [N, k], distances [N, k] (cosine distances, 0=identical).
    """
    N = X.shape[0]
    chunk = 512
    all_idx = []
    all_dist = []

    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        sim = X[i:end] @ X.T  # [chunk, N]
        sim[torch.arange(end - i, device=X.device), torch.arange(i, end, device=X.device)] = -2.0
        topk_sim, topk_idx = sim.topk(k, dim=1)
        all_idx.append(topk_idx)
        all_dist.append(1.0 - topk_sim)

    return torch.cat(all_idx, dim=0), torch.cat(all_dist, dim=0)


# ──────────────────────────────────────────────
# Experiment 1: Local PCA tangent space rotation
# ──────────────────────────────────────────────

def local_pca_tangent(X: torch.Tensor, idx: int, knn_indices: torch.Tensor, d: int):
    """Compute d-dim tangent space at point idx from its kNN neighborhood.
    Returns V[:d] (d principal directions), each [D].
    """
    neighbors = X[knn_indices[idx]]  # [k, D]
    Y = neighbors - neighbors.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(Y, full_matrices=False)
    except torch.linalg.LinAlgError:
        return None
    return Vh[:d]  # [d, D]


def principal_angles(T1: torch.Tensor, T2: torch.Tensor):
    """Compute principal angles between two d-dim subspaces.
    T1, T2: [d, D]. Returns angles in radians, shape [d].
    """
    M = T1 @ T2.T  # [d, d]
    try:
        s = torch.linalg.svdvals(M)
    except torch.linalg.LinAlgError:
        return None
    s = s.clamp(-1.0, 1.0)
    return torch.arccos(s)


def extrinsic_bending_stats(X: torch.Tensor, knn_indices: torch.Tensor,
                             knn_dists: torch.Tensor, d: int,
                             n_anchors: int = 1024, seed: int = 20260523):
    """Compute extrinsic bending proxy for anchor points.
    For each anchor, compare its tangent space with each neighbor's tangent space.

    Reports three complementary metrics:
    - mean_angle: average of all principal angles (radians). Direct measure of
      tangent space rotation. Range [0, pi/2]. Higher = more rotation.
    - grassmann_dist: ||sin(theta)||_2 (Grassmann chordal distance). Captures
      total subspace divergence in a single number. Range [0, sqrt(d)].
    - bending_rate: grassmann_dist / cosine_distance. Normalized by neighbor
      distance — but in high-D cosine distances concentrate, so this has less
      discriminative power. Kept for completeness.
    """
    N = X.shape[0]
    k = knn_indices.shape[1]

    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    n_anchors = min(n_anchors, N)
    anchors = torch.randperm(N, generator=rng)[:n_anchors].sort().values

    all_mean_angle = []
    all_grassmann = []
    all_bending = []

    for ai in range(len(anchors)):
        a = anchors[ai].item()
        T_a = local_pca_tangent(X, a, knn_indices, d)
        if T_a is None:
            continue

        for ni in range(k):
            nb = knn_indices[a, ni].item()
            dist = knn_dists[a, ni].item()
            if dist < 1e-8:
                continue

            T_b = local_pca_tangent(X, nb, knn_indices, d)
            if T_b is None:
                continue

            angles = principal_angles(T_a, T_b)
            if angles is None:
                continue

            mean_ang = angles.mean().item()
            grassmann = torch.sin(angles).norm().item()
            bending = grassmann / max(dist, 1e-6)

            all_mean_angle.append(mean_ang)
            all_grassmann.append(grassmann)
            all_bending.append(bending)

    if not all_mean_angle:
        return None

    def stats_dict(arr):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    arr_angle = np.array(all_mean_angle)
    arr_grass = np.array(all_grassmann)
    arr_bend = np.array(all_bending)

    return {
        "mean_angle": stats_dict(arr_angle),
        "grassmann_dist": stats_dict(arr_grass),
        "bending_rate": stats_dict(arr_bend),
        "n_pairs": len(all_mean_angle),
    }


# ──────────────────────────────────────────────
# Experiment 3: Forman-Ricci curvature (fast)
# ──────────────────────────────────────────────

def forman_ricci_stats(knn_indices: torch.Tensor, n_anchors: int = 1024,
                       seed: int = 20260523):
    """Compute Forman-Ricci curvature on kNN graph.
    For edge (i,j): F(i,j) = 4 - deg(i) - deg(j) + 3 * |triangles through (i,j)|
    (unweighted version for speed)
    Returns dict of statistics.
    """
    N, k = knn_indices.shape
    idx_cpu = knn_indices.cpu()

    neighbor_sets = [set(idx_cpu[i].tolist()) for i in range(N)]

    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    n_anchors = min(n_anchors, N)
    anchors = torch.randperm(N, generator=rng)[:n_anchors].sort().values

    all_ricci = []
    deg = k  # all nodes have same degree in kNN

    for ai in range(len(anchors)):
        a = anchors[ai].item()
        for ni in range(k):
            nb = idx_cpu[a, ni].item()
            triangles = len(neighbor_sets[a] & neighbor_sets[nb])
            F = 4 - deg - deg + 3 * triangles
            all_ricci.append(F)

    arr = np.array(all_ricci, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(np.std(arr)),
        "frac_negative": float(np.mean(arr < 0)),
        "n_edges": len(arr),
    }


# ──────────────────────────────────────────────
# Per-matrix analysis
# ──────────────────────────────────────────────

def analyze_curvature(W: torch.Tensor, name: str, device: str,
                      k_list: list, d_list: list,
                      max_rows: int = 8192, n_anchors: int = 1024,
                      seed: int = 20260523):
    M, N = W.shape
    W_gpu = W.to(device).float()

    # L2 normalize rows
    norms = W_gpu.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X = W_gpu / norms

    # subsample if needed
    if M > max_rows:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        perm = torch.randperm(M, generator=rng)[:max_rows]
        X = X[perm]
        sampled = True
    else:
        sampled = False

    N_pts = X.shape[0]
    max_k = max(k_list)

    # build kNN for max_k
    t0 = time.time()
    knn_idx_full, knn_dist_full = build_knn_cosine(X, max_k)
    knn_time = time.time() - t0

    results = {
        "name": name,
        "shape": [M, N],
        "n_points": N_pts,
        "sampled": sampled,
        "knn_time_s": round(knn_time, 2),
        "extrinsic_bending": {},
        "forman_ricci": {},
    }

    # extrinsic bending at multiple (k, d) combos
    for k in k_list:
        knn_idx_k = knn_idx_full[:, :k]
        knn_dist_k = knn_dist_full[:, :k]

        for d in d_list:
            if d >= k:
                continue
            combo = f"k={k},d={d}"
            t0 = time.time()
            stats = extrinsic_bending_stats(X, knn_idx_k, knn_dist_k, d,
                                            n_anchors=n_anchors, seed=seed)
            elapsed = time.time() - t0
            if stats:
                stats["time_s"] = round(elapsed, 2)
                results["extrinsic_bending"][combo] = stats
                ma = stats["mean_angle"]
                print(f"    {combo}: angle_med={ma['median']:.4f} angle_p90={ma['p90']:.4f} grassmann_med={stats['grassmann_dist']['median']:.3f} ({elapsed:.1f}s)", flush=True)
            else:
                print(f"    {combo}: no data", flush=True)

    # Forman-Ricci at multiple k
    for k in k_list:
        knn_idx_k = knn_idx_full[:, :k]
        t0 = time.time()
        stats = forman_ricci_stats(knn_idx_k, n_anchors=n_anchors, seed=seed)
        elapsed = time.time() - t0
        stats["time_s"] = round(elapsed, 2)
        results["forman_ricci"][f"k={k}"] = stats
        print(f"    Forman k={k}: mean={stats['mean']:.1f} frac_neg={stats['frac_negative']:.3f} ({elapsed:.1f}s)", flush=True)

    del X, W_gpu, knn_idx_full, knn_dist_full
    torch.cuda.empty_cache()
    return results


# ──────────────────────────────────────────────
# Weight classification (reused from 195)
# ──────────────────────────────────────────────

def classify_qwen36_weight(key: str, target_layers: set = None):
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

    for prefix, ltype, wname in [
        ("linear_attn.in_proj_qkv", "linear_attn", "in_proj_qkv"),
        ("linear_attn.in_proj_a", "linear_attn", "in_proj_a"),
        ("linear_attn.in_proj_b", "linear_attn", "in_proj_b"),
        ("linear_attn.in_proj_z", "linear_attn", "in_proj_z"),
        ("linear_attn.out_proj", "linear_attn", "out_proj"),
        ("self_attn.q_proj", "full_attn", "q_proj"),
        ("self_attn.k_proj", "full_attn", "k_proj"),
        ("self_attn.v_proj", "full_attn", "v_proj"),
        ("self_attn.o_proj", "full_attn", "o_proj"),
        ("mlp.gate_proj", "mlp", "gate_proj"),
        ("mlp.up_proj", "mlp", "up_proj"),
        ("mlp.down_proj", "mlp", "down_proj"),
    ]:
        if prefix in rest:
            return (layer_idx, ltype, wname)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="/home/lmxxf/work/models/Qwen3.6-27B")
    parser.add_argument("--prev-results", default="/home/lmxxf/work/ai-theorys-study/wechat/assets/195/qwen36_intrinsic_dim.json")
    parser.add_argument("--output", default="/home/lmxxf/work/ai-theorys-study/wechat/assets/195-next/weight_curvature_results.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,1,2,3,7,11,31,47,63")
    parser.add_argument("--k-list", default="16,32,64")
    parser.add_argument("--d-list", default="8,16,32")
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--n-anchors", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    target_layers = set(int(x) for x in args.layers.split(","))
    k_list = [int(x) for x in args.k_list.split(",")]
    d_list = [int(x) for x in args.d_list.split(",")]

    print(f"Target layers: {sorted(target_layers)}", flush=True)
    print(f"k values: {k_list}", flush=True)
    print(f"d values: {d_list}", flush=True)
    print(f"Max rows: {args.max_rows}, anchors: {args.n_anchors}", flush=True)

    # Load previous eRank/TwoNN results for merging
    prev_data = {}
    if os.path.exists(args.prev_results):
        with open(args.prev_results) as f:
            prev = json.load(f)
        for r in prev["weight_results"]:
            prev_data[r["name"]] = r
        print(f"Loaded {len(prev_data)} previous results from {args.prev_results}", flush=True)

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"Found {len(shard_files)} shards", flush=True)

    all_results = []
    t_start = time.time()

    for shard_file in shard_files:
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

                # skip very small matrices (in_proj_a/b have only 48 rows)
                if tensor.shape[0] < 64:
                    full_name = f"L{layer_idx:02d}.{layer_type}.{weight_name}"
                    print(f"\n[{shard_name}] {full_name} [{tensor.shape[0]}, {tensor.shape[1]}] — skipped (< 64 rows)", flush=True)
                    all_results.append({
                        "name": full_name,
                        "shape": list(tensor.shape),
                        "n_points": tensor.shape[0],
                        "sampled": False,
                        "skipped": True,
                        "skip_reason": "too_few_rows",
                        "extrinsic_bending": {},
                        "forman_ricci": {},
                    })
                    del tensor
                    continue

                full_name = f"L{layer_idx:02d}.{layer_type}.{weight_name}"
                print(f"\n[{shard_name}] {full_name} [{tensor.shape[0]}, {tensor.shape[1]}]", flush=True)

                result = analyze_curvature(
                    tensor, full_name, args.device,
                    k_list=k_list, d_list=d_list,
                    max_rows=args.max_rows, n_anchors=args.n_anchors,
                    seed=args.seed,
                )

                # merge previous eRank/TwoNN data
                if full_name in prev_data:
                    p = prev_data[full_name]
                    result["erank"] = p.get("erank")
                    result["twonn_row"] = p.get("twonn_row")
                    result["erank_twonn_ratio_row"] = p.get("erank_twonn_ratio_row")

                result["layer"] = layer_idx
                result["layer_type"] = layer_type
                result["weight_name"] = weight_name
                all_results.append(result)

                del tensor

    elapsed = time.time() - t_start
    print(f"\n{'='*80}", flush=True)
    print(f"Done! {len(all_results)} matrices in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Save
    output = {
        "model": "Qwen3.6-27B",
        "experiment": "weight_curvature",
        "k_list": k_list,
        "d_list": d_list,
        "max_rows": args.max_rows,
        "n_anchors": args.n_anchors,
        "seed": args.seed,
        "elapsed_s": round(elapsed, 1),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"Results saved to {args.output}", flush=True)

    # Summary table
    print(f"\n{'='*140}")
    print("SUMMARY (k=32, d=16)")
    print(f"{'Name':40s} {'Shape':18s} {'Ratio':>7s} {'Angle_med':>10s} {'Angle_p90':>10s} {'Grass_med':>10s} {'Ricci_mean':>11s} {'Ricci_neg%':>11s}")
    print("-" * 140)

    for r in sorted(all_results, key=lambda x: x.get("layer", -1)):
        if r.get("skipped"):
            print(f"{r['name']:40s} {str(r['shape']):18s} {'skip':>7s} {'—':>10s} {'—':>10s} {'—':>10s} {'—':>11s} {'—':>11s}")
            continue

        ratio = r.get("erank_twonn_ratio_row")
        ratio_s = f"{ratio:.1f}x" if ratio else "—"

        bend = r.get("extrinsic_bending", {}).get("k=32,d=16")
        if bend:
            ang_med = f"{bend['mean_angle']['median']:.4f}"
            ang_p90 = f"{bend['mean_angle']['p90']:.4f}"
            grass_med = f"{bend['grassmann_dist']['median']:.3f}"
        else:
            ang_med = ang_p90 = grass_med = "—"

        ricci = r.get("forman_ricci", {}).get("k=32")
        ricci_m = f"{ricci['mean']:.1f}" if ricci else "—"
        ricci_neg = f"{ricci['frac_negative']:.3f}" if ricci else "—"

        print(f"{r['name']:40s} {str(r['shape']):18s} {ratio_s:>7s} {ang_med:>10s} {ang_p90:>10s} {grass_med:>10s} {ricci_m:>11s} {ricci_neg:>11s}")

    # Correlation with eRank/TwoNN ratio
    print(f"\n{'='*80}")
    print("SPEARMAN CORRELATIONS")

    from scipy.stats import spearmanr

    pairs = []
    for r in all_results:
        if r.get("skipped"):
            continue
        ratio = r.get("erank_twonn_ratio_row")
        bend = r.get("extrinsic_bending", {}).get("k=32,d=16")
        ricci = r.get("forman_ricci", {}).get("k=32")
        if ratio and bend and ricci:
            pairs.append({
                "ratio": ratio,
                "angle_median": bend["mean_angle"]["median"],
                "angle_p90": bend["mean_angle"]["p90"],
                "grassmann_median": bend["grassmann_dist"]["median"],
                "ricci_mean": ricci["mean"],
                "ricci_frac_neg": ricci["frac_negative"],
                "twonn": r.get("twonn_row"),
                "erank": r.get("erank"),
                "name": r.get("name"),
            })

    if len(pairs) >= 5:
        import pandas as pd
        df = pd.DataFrame(pairs)
        metrics = ["ratio", "angle_median", "angle_p90", "grassmann_median", "ricci_mean", "ricci_frac_neg"]
        for a, b in [
            ("ratio", "angle_median"),
            ("ratio", "grassmann_median"),
            ("ratio", "ricci_mean"),
            ("ratio", "ricci_frac_neg"),
            ("angle_median", "ricci_mean"),
            ("twonn", "ricci_mean"),
            ("erank", "grassmann_median"),
        ]:
            vals_a = df[a].dropna()
            vals_b = df[b].dropna()
            common = vals_a.index.intersection(vals_b.index)
            if len(common) >= 5:
                rho, p = spearmanr(df.loc[common, a], df.loc[common, b])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {a:20s} vs {b:20s}: rho={rho:+.3f}  p={p:.4f} {sig}")
    else:
        print("  Not enough data for correlation analysis")


if __name__ == "__main__":
    main()
