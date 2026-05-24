"""
dsv4_curvature.py — Measure Forman-Ricci and tangent-space rotation on DeepSeek V4 Flash 280B.

Reuses dequantization from 194's measure_intrinsic_dim.py.
Picks representative layers (0, 10, 20, 30, 42) and samples 4 experts per layer.

Usage (inside Docker container):
  python dsv4_curvature.py \
    --ckpt-path /work/deepseek-v4-flash-deployment/deepseek-v4-flash \
    --output /work/ai-theorys-study/wechat/assets/195-next/dsv4_curvature_results.json
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


# ── Dequantization (from 194) ──

FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def dequant_fp8(weight, scale):
    w = weight.float()
    s = scale.float()
    M, K = w.shape
    bs_m = M // s.shape[0]
    bs_k = K // s.shape[1]
    s_exp = s.repeat_interleave(bs_m, dim=0).repeat_interleave(bs_k, dim=1)
    return w * s_exp


def dequant_fp4(weight_int8, scale):
    raw = weight_int8.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    table = FP4_TABLE.to(raw.device)
    w_low = table[low.long()]
    w_high = table[high.long()]
    w = torch.stack([w_low, w_high], dim=-1).reshape(raw.shape[0], -1)
    s = scale.float()
    N, K = w.shape
    bs_n = N // s.shape[0] if s.shape[0] > 0 else 1
    bs_k = K // s.shape[1] if s.shape[1] > 0 else 1
    s_exp = s.repeat_interleave(bs_n, dim=0).repeat_interleave(bs_k, dim=1)
    return w * s_exp


# ── kNN ──

def build_knn_cosine(X, k):
    N = X.shape[0]
    chunk = 512
    all_idx, all_dist = [], []
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        sim = X[i:end] @ X.T
        sim[torch.arange(end - i, device=X.device), torch.arange(i, end, device=X.device)] = -2.0
        topk_sim, topk_idx = sim.topk(k, dim=1)
        all_idx.append(topk_idx)
        all_dist.append(1.0 - topk_sim)
    return torch.cat(all_idx), torch.cat(all_dist)


# ── Tangent space rotation ──

def local_pca_tangent(X, idx, knn_indices, d):
    neighbors = X[knn_indices[idx]]
    Y = neighbors - neighbors.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(Y, full_matrices=False)
    except torch.linalg.LinAlgError:
        return None
    return Vh[:d]


def principal_angles(T1, T2):
    M = T1 @ T2.T
    try:
        s = torch.linalg.svdvals(M)
    except torch.linalg.LinAlgError:
        return None
    return torch.arccos(s.clamp(-1.0, 1.0))


def tangent_rotation_stats(X, knn_indices, knn_dists, d,
                           n_anchors=1024, seed=20260523):
    N = X.shape[0]
    k = knn_indices.shape[1]
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    n_anchors = min(n_anchors, N)
    anchors = torch.randperm(N, generator=rng)[:n_anchors].sort().values

    all_angle, all_grass = [], []
    for ai in range(len(anchors)):
        a = anchors[ai].item()
        T_a = local_pca_tangent(X, a, knn_indices, d)
        if T_a is None:
            continue
        for ni in range(k):
            nb = knn_indices[a, ni].item()
            if knn_dists[a, ni].item() < 1e-8:
                continue
            T_b = local_pca_tangent(X, nb, knn_indices, d)
            if T_b is None:
                continue
            angles = principal_angles(T_a, T_b)
            if angles is None:
                continue
            all_angle.append(angles.mean().item())
            all_grass.append(torch.sin(angles).norm().item())

    if not all_angle:
        return None

    def s(arr):
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)), "std": float(np.std(arr))}
    return {"mean_angle": s(all_angle), "grassmann_dist": s(all_grass), "n_pairs": len(all_angle)}


# ── Forman-Ricci ──

def forman_ricci_stats(knn_indices, n_anchors=1024, seed=20260523):
    N, k = knn_indices.shape
    idx_cpu = knn_indices.cpu()
    neighbor_sets = [set(idx_cpu[i].tolist()) for i in range(N)]
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    n_anchors = min(n_anchors, N)
    anchors = torch.randperm(N, generator=rng)[:n_anchors].sort().values
    all_ricci = []
    for ai in range(len(anchors)):
        a = anchors[ai].item()
        for ni in range(k):
            nb = idx_cpu[a, ni].item()
            triangles = len(neighbor_sets[a] & neighbor_sets[nb])
            F = 4 - k - k + 3 * triangles
            all_ricci.append(F)
    arr = np.array(all_ricci, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "frac_negative": float(np.mean(arr < 0)), "n_edges": len(arr)}


# ── Per-matrix analysis ──

def analyze_one(W, name, device, k_list, d_list, max_rows=8192, n_anchors=1024, seed=20260523):
    M, N = W.shape
    W_gpu = W.to(device).float()
    norms = W_gpu.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X = W_gpu / norms

    if M > max_rows:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        X = X[torch.randperm(M, generator=rng)[:max_rows]]

    N_pts = X.shape[0]
    max_k = max(k_list)
    knn_idx, knn_dist = build_knn_cosine(X, max_k)

    result = {"name": name, "shape": [M, N], "n_points": N_pts,
              "tangent_rotation": {}, "forman_ricci": {}}

    for k in k_list:
        ki, kd = knn_idx[:, :k], knn_dist[:, :k]
        for d in d_list:
            if d >= k:
                continue
            combo = f"k={k},d={d}"
            t0 = time.time()
            stats = tangent_rotation_stats(X, ki, kd, d, n_anchors, seed)
            elapsed = time.time() - t0
            if stats:
                stats["time_s"] = round(elapsed, 2)
                result["tangent_rotation"][combo] = stats
                ma = stats["mean_angle"]
                print(f"    {combo}: angle_med={ma['median']:.4f} grass_med={stats['grassmann_dist']['median']:.3f} ({elapsed:.1f}s)", flush=True)

        t0 = time.time()
        fr = forman_ricci_stats(ki, n_anchors, seed)
        elapsed = time.time() - t0
        fr["time_s"] = round(elapsed, 2)
        result["forman_ricci"][f"k={k}"] = fr
        print(f"    Forman k={k}: mean={fr['mean']:.1f} frac_neg={fr['frac_negative']:.3f} ({elapsed:.1f}s)", flush=True)

    del X, W_gpu, knn_idx, knn_dist
    torch.cuda.empty_cache()
    return result


# ── Weight classification ──

def classify_weight(key):
    if "experts" in key and "shared" not in key:
        return None
    if not key.startswith("layers."):
        if key == "embed.weight":
            return (-1, "embed", "embed")
        if key == "head.weight":
            return (999, "head", "head")
        return None
    parts = key.split(".")
    layer_idx = int(parts[1])
    rest = ".".join(parts[2:])

    mapping = [
        ("attn.wq_a", "attn", "wq_a"),
        ("attn.wq_b", "attn", "wq_b"),
        ("attn.wo_a", "attn", "wo_a"),
        ("attn.wo_b", "attn", "wo_b"),
        ("ffn.gate.weight", "routing", "gate"),
    ]
    for prefix, cat, sub in mapping:
        if prefix in rest and "norm" not in rest:
            return (layer_idx, cat, sub)
    if "attn.wkv" in rest and "norm" not in rest:
        return (layer_idx, "attn", "wkv")
    return None


def parse_expert_key(key):
    if "experts" not in key or "shared" in key or not key.startswith("layers."):
        return None
    parts = key.split(".")
    try:
        layer_idx = int(parts[1])
        expert_idx = int(parts[4])
        weight_name = parts[5]
        if parts[-1] == "weight":
            return (layer_idx, expert_idx, weight_name)
    except (IndexError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="/work/deepseek-v4-flash-deployment/deepseek-v4-flash")
    parser.add_argument("--output", default="/work/ai-theorys-study/wechat/assets/195-next/dsv4_curvature_results.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,10,20,30,42")
    parser.add_argument("--expert-samples", type=int, default=4)
    parser.add_argument("--k-list", default="16,32,64")
    parser.add_argument("--d-list", default="8,16")
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--n-anchors", type=int, default=1024)
    args = parser.parse_args()

    target_layers = set(int(x) for x in args.layers.split(","))
    k_list = [int(x) for x in args.k_list.split(",")]
    d_list = [int(x) for x in args.d_list.split(",")]

    print(f"Target layers: {sorted(target_layers)}", flush=True)
    print(f"k={k_list}, d={d_list}, max_rows={args.max_rows}", flush=True)

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"Found {len(shard_files)} shards", flush=True)

    expert_sample_set = {}
    scale_buffer = {}
    all_results = []
    t_start = time.time()

    for shard_idx, shard_file in enumerate(shard_files):
        shard_name = os.path.basename(shard_file)

        with safe_open(shard_file, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            for key in keys:
                if key.endswith(".scale"):
                    scale_buffer[key] = f.get_tensor(key)
                    continue
                if not key.endswith(".weight"):
                    continue

                tensor = f.get_tensor(key)

                # shared weights
                info = classify_weight(key)
                if info is not None:
                    layer_idx, category, sub_name = info
                    if layer_idx not in target_layers and layer_idx not in (-1, 999):
                        del tensor
                        continue

                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(args.device), scale.to(args.device))
                    elif tensor.dtype in (torch.bfloat16, torch.float32):
                        W = tensor.to(args.device).float()
                    else:
                        del tensor
                        continue

                    if W.dim() != 2 or min(W.shape) < 64:
                        print(f"  SKIP {key}: shape={list(W.shape)}", flush=True)
                        del W, tensor
                        if scale is not None: del scale
                        torch.cuda.empty_cache()
                        continue

                    full_name = f"L{layer_idx}.{sub_name}"
                    print(f"\n[{shard_name}] {full_name} {list(W.shape)}", flush=True)
                    result = analyze_one(W, full_name, args.device, k_list, d_list,
                                        args.max_rows, args.n_anchors)
                    result["layer"] = layer_idx
                    result["category"] = category
                    all_results.append(result)
                    del W, tensor
                    if scale is not None: del scale
                    torch.cuda.empty_cache()
                    continue

                # expert weights
                expert_info = parse_expert_key(key)
                if expert_info is not None:
                    layer_idx, expert_idx, weight_name = expert_info
                    if layer_idx not in target_layers:
                        del tensor
                        continue
                    if layer_idx not in expert_sample_set:
                        rng = np.random.RandomState(layer_idx * 1000 + 42)
                        expert_sample_set[layer_idx] = set(
                            rng.choice(256, size=min(args.expert_samples, 256), replace=False))
                    if expert_idx not in expert_sample_set[layer_idx]:
                        del tensor
                        continue

                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.int8 and scale is not None:
                        W = dequant_fp4(tensor.to(args.device), scale.to(args.device))
                    elif tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(args.device), scale.to(args.device))
                    else:
                        del tensor
                        continue

                    if W.dim() != 2 or min(W.shape) < 64:
                        del W, tensor
                        if scale is not None: del scale
                        torch.cuda.empty_cache()
                        continue

                    full_name = f"L{layer_idx}.expert{expert_idx}.{weight_name}"
                    print(f"\n[{shard_name}] {full_name} {list(W.shape)}", flush=True)
                    result = analyze_one(W, full_name, args.device, k_list, d_list,
                                        args.max_rows, args.n_anchors)
                    result["layer"] = layer_idx
                    result["category"] = "expert"
                    result["expert_idx"] = expert_idx
                    all_results.append(result)
                    del W, tensor
                    if scale is not None: del scale
                    torch.cuda.empty_cache()
                    continue

                del tensor

        stale = [k for k in scale_buffer if shard_name in k]
        for k in stale:
            del scale_buffer[k]

    elapsed = time.time() - t_start
    print(f"\n{'='*80}", flush=True)
    print(f"Done! {len(all_results)} matrices in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    output = {
        "model": "DeepSeek-V4-Flash-280B",
        "k_list": k_list, "d_list": d_list,
        "max_rows": args.max_rows, "n_anchors": args.n_anchors,
        "elapsed_s": round(elapsed, 1),
        "results": all_results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"Saved to {args.output}", flush=True)

    # Summary
    print(f"\n{'='*130}")
    print(f"{'Name':35s} {'Shape':18s} {'Angle_med':>10s} {'Grass_med':>10s} {'Ricci_mean':>11s} {'Ricci_neg%':>11s}")
    print("-" * 130)
    for r in sorted(all_results, key=lambda x: (x.get("layer", -1), x["name"])):
        tr = r.get("tangent_rotation", {}).get("k=32,d=16") or r.get("tangent_rotation", {}).get("k=32,d=8")
        fr = r.get("forman_ricci", {}).get("k=32")
        ang = f"{tr['mean_angle']['median']:.4f}" if tr else "—"
        grs = f"{tr['grassmann_dist']['median']:.3f}" if tr else "—"
        rm = f"{fr['mean']:.1f}" if fr else "—"
        rn = f"{fr['frac_negative']:.3f}" if fr else "—"
        print(f"{r['name']:35s} {str(r['shape']):18s} {ang:>10s} {grs:>10s} {rm:>11s} {rn:>11s}")


if __name__ == "__main__":
    main()
