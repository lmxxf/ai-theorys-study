"""
submanifold_split.py — Test the "multiple sub-manifolds" hypothesis.

Supports both Qwen3.6-27B and DeepSeek V4 Flash.

Key splits:
  Qwen3.6:
    - in_proj_qkv → Q / K / V segments
    - V segment → per-head (48 heads × 128 dim)
    - q_proj (GQA layers) → per-head
    - gate_proj → 4 equal chunks
  DeepSeek V4:
    - wq_b [32768, 1024] → per-head (128 heads × 256 dim)
    - expert w1/w2/w3 → whole only (baseline)

Usage (inside Docker):
  # Qwen3.6-27B
  python submanifold_split.py --model qwen \
    --ckpt-path /workspace/models/Qwen3.6-27B \
    --output submanifold_qwen.json

  # DeepSeek V4 Flash
  python submanifold_split.py --model dsv4 \
    --ckpt-path /workspace/deepseek-v4-flash-deployment/deepseek-v4-flash \
    --output submanifold_dsv4.json
"""

import argparse
import json
import os
import time
from glob import glob

import torch
import numpy as np
from safetensors import safe_open


# ── Dequantization (V4 Flash) ──

FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

def dequant_fp8(weight, scale):
    w = weight.float()
    s = scale.float()
    M, K = w.shape
    s_exp = s.repeat_interleave(M // s.shape[0], dim=0).repeat_interleave(K // s.shape[1], dim=1)
    return w * s_exp

def dequant_fp4(weight_int8, scale):
    raw = weight_int8.view(torch.uint8)
    low, high = raw & 0x0F, (raw >> 4) & 0x0F
    table = FP4_TABLE.to(raw.device)
    w = torch.stack([table[low.long()], table[high.long()]], dim=-1).reshape(raw.shape[0], -1)
    s = scale.float()
    N, K = w.shape
    s_exp = s.repeat_interleave(N // s.shape[0], dim=0).repeat_interleave(K // s.shape[1], dim=1)
    return w * s_exp


# ── TwoNN ──

def twonn_dimension(X, max_points=5000):
    N = X.shape[0]
    if N < 10:
        return float("nan")
    if N > max_points:
        X = X[torch.randperm(N)[:max_points]]
        N = max_points
    chunk = 512
    mus = []
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        Xi = X[i:end]
        dists = torch.cdist(Xi, X)
        for j in range(end - i):
            dists[j, i + j] = float("inf")
        top2, _ = dists.topk(2, dim=1, largest=False)
        mus.append(top2[:, 1] / top2[:, 0].clamp(min=1e-30))
    mus = torch.cat(mus)
    mus = mus[mus > 1.0]
    if len(mus) < 10:
        return float("nan")
    return float(len(mus) / torch.log(mus).sum())


# ── eRank ──

def compute_erank(W):
    sv = torch.linalg.svdvals(W).cpu().numpy().astype(np.float64)
    sv = sv[sv > 0]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# ── Analyze ──

def analyze(W_gpu, name):
    M, N = W_gpu.shape
    erank = compute_erank(W_gpu)
    twonn = twonn_dimension(W_gpu)
    ratio = erank / twonn if twonn > 0 and not np.isnan(twonn) else float("nan")
    return {
        "name": name, "shape": [M, N],
        "erank": round(erank, 2),
        "twonn": round(twonn, 2) if not np.isnan(twonn) else None,
        "ratio": round(ratio, 2) if not np.isnan(ratio) else None,
    }


def analyze_and_print(W_gpu, name, label=""):
    r = analyze(W_gpu, name)
    print(f"  {label:20s} eRank={r['erank']:<8.0f} TwoNN={str(r['twonn']):>8s}  ratio={str(r['ratio']):>8s}", flush=True)
    return r


def split_by_heads(W_gpu, full_name, n_heads, label_prefix="head"):
    M = W_gpu.shape[0]
    head_dim = M // n_heads
    ratios = []
    eranks = []
    twonns = []
    for h in range(n_heads):
        W_h = W_gpu[h * head_dim:(h + 1) * head_dim]
        r = analyze(W_h, f"{full_name}.{label_prefix}{h}")
        eranks.append(r["erank"])
        if r["twonn"] is not None:
            twonns.append(r["twonn"])
        if r["ratio"] is not None:
            ratios.append(r["ratio"])

    summary = {
        "name": f"{full_name}.per_head_summary",
        "n_heads": n_heads, "head_dim": head_dim,
        "erank_mean": round(float(np.mean(eranks)), 2),
        "twonn_mean": round(float(np.mean(twonns)), 2) if twonns else None,
        "ratio_mean": round(float(np.mean(ratios)), 2) if ratios else None,
        "ratio_min": round(float(np.min(ratios)), 2) if ratios else None,
        "ratio_max": round(float(np.max(ratios)), 2) if ratios else None,
        "ratio_std": round(float(np.std(ratios)), 2) if ratios else None,
    }
    print(f"  {'HEADS':20s} n={n_heads}×{head_dim}  eRank_mean={summary['erank_mean']:.0f}  "
          f"TwoNN_mean={summary['twonn_mean']}  ratio_mean={summary['ratio_mean']}  "
          f"range=[{summary['ratio_min']}, {summary['ratio_max']}]", flush=True)
    return summary


# ══════════════════════════════════════════════
# Qwen3.6-27B
# ══════════════════════════════════════════════

def run_qwen(args):
    target_layers = set(int(x) for x in args.layers.split(","))
    device = args.device

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"Qwen3.6-27B: {len(shard_files)} shards, layers={sorted(target_layers)}", flush=True)

    all_results = []
    t_start = time.time()

    for shard_file in shard_files:
        shard_name = os.path.basename(shard_file)
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                if not key.endswith(".weight") or "language_model.layers." not in key:
                    continue
                parts = key.split(".")
                layer_idx = int(parts[parts.index("layers") + 1])
                if layer_idx not in target_layers:
                    continue
                rest = ".".join(parts[parts.index("layers") + 2:])
                tensor = f.get_tensor(key)
                if tensor.dim() != 2:
                    del tensor
                    continue

                # ── in_proj_qkv: Q/K/V split + V per-head ──
                if "linear_attn.in_proj_qkv" in rest:
                    W = tensor.to(device).float()
                    name = f"L{layer_idx:02d}.in_proj_qkv"
                    print(f"\n[{shard_name}] {name} {list(W.shape)}", flush=True)

                    all_results.append(analyze_and_print(W, f"{name}.whole", "WHOLE"))

                    # Q(2048) + K(2048) + V(6144)
                    for sname, s, e in [("Q", 0, 2048), ("K", 2048, 4096), ("V", 4096, 10240)]:
                        all_results.append(analyze_and_print(W[s:e], f"{name}.{sname}", f"{sname} [{s}:{e}]"))

                    # V segment per-head: 48 V heads × 128 dim
                    W_v = W[4096:10240]  # [6144, 5120]
                    all_results.append(split_by_heads(W_v, f"{name}.V", n_heads=48))

                    del W
                    torch.cuda.empty_cache()

                # ── q_proj: per-head ──
                elif "self_attn.q_proj" in rest:
                    W = tensor.to(device).float()
                    name = f"L{layer_idx:02d}.q_proj"
                    print(f"\n[{shard_name}] {name} {list(W.shape)}", flush=True)

                    all_results.append(analyze_and_print(W, f"{name}.whole", "WHOLE"))
                    # 195.md: q_proj [12288, 5120]. config says 24 Q heads but 12288/24=512.
                    # head_dim for GQA is 256 per 195.md → 12288/256=48 heads? Just compute.
                    n_heads = W.shape[0] // 256  # head_dim=256 for GQA
                    all_results.append(split_by_heads(W, f"{name}", n_heads=n_heads))

                    del W
                    torch.cuda.empty_cache()

                # ── k_proj, v_proj: whole only ──
                elif "self_attn.k_proj" in rest or "self_attn.v_proj" in rest:
                    W = tensor.to(device).float()
                    wn = "k_proj" if "k_proj" in rest else "v_proj"
                    name = f"L{layer_idx:02d}.{wn}"
                    print(f"\n[{shard_name}] {name} {list(W.shape)}", flush=True)
                    all_results.append(analyze_and_print(W, f"{name}.whole", "WHOLE"))
                    del W
                    torch.cuda.empty_cache()

                # ── gate_proj: 4 chunks ──
                elif "mlp.gate_proj" in rest:
                    W = tensor.to(device).float()
                    name = f"L{layer_idx:02d}.gate_proj"
                    print(f"\n[{shard_name}] {name} {list(W.shape)}", flush=True)
                    all_results.append(analyze_and_print(W, f"{name}.whole", "WHOLE"))

                    M = W.shape[0]
                    cs = M // 4
                    for ci in range(4):
                        s, e = ci * cs, (ci + 1) * cs if ci < 3 else M
                        all_results.append(analyze_and_print(W[s:e], f"{name}.chunk{ci}", f"chunk{ci} [{s}:{e}]"))
                    del W
                    torch.cuda.empty_cache()

                del tensor

    elapsed = time.time() - t_start
    print(f"\nDone! {len(all_results)} entries in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    return {"model": "Qwen3.6-27B", "layers": sorted(target_layers),
            "elapsed_s": round(elapsed, 1), "results": all_results}


# ══════════════════════════════════════════════
# DeepSeek V4 Flash
# ══════════════════════════════════════════════

def run_dsv4(args):
    target_layers = set(int(x) for x in args.layers.split(","))
    device = args.device

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"DeepSeek V4 Flash: {len(shard_files)} shards, layers={sorted(target_layers)}", flush=True)

    expert_sample_set = {}
    scale_buffer = {}
    all_results = []
    t_start = time.time()

    for shard_file in shard_files:
        shard_name = os.path.basename(shard_file)
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            for key in keys:
                if key.endswith(".scale"):
                    scale_buffer[key] = f.get_tensor(key)
                    continue
                if not key.endswith(".weight"):
                    continue

                # ── Shared weights ──
                if key.startswith("layers.") and "experts" not in key:
                    parts = key.split(".")
                    layer_idx = int(parts[1])
                    if layer_idx not in target_layers:
                        continue
                    rest = ".".join(parts[2:])
                    tensor = f.get_tensor(key)

                    target_weights = {
                        "attn.wq_b": "wq_b",
                        "attn.wq_a": "wq_a",
                        "attn.wo_a": "wo_a",
                        "attn.wo_b": "wo_b",
                        "attn.wkv": "wkv",
                        "ffn.gate": "gate",
                    }
                    matched = None
                    for prefix, wname in target_weights.items():
                        if prefix in rest and "norm" not in rest:
                            matched = wname
                            break
                    if matched is None:
                        del tensor
                        continue

                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(device), scale.to(device))
                    elif tensor.dtype in (torch.bfloat16, torch.float32):
                        W = tensor.to(device).float()
                    else:
                        del tensor
                        continue

                    if W.dim() != 2 or min(W.shape) < 64:
                        del W, tensor
                        if scale is not None: del scale
                        torch.cuda.empty_cache()
                        continue

                    name = f"L{layer_idx:02d}.{matched}"
                    print(f"\n[{shard_name}] {name} {list(W.shape)}", flush=True)
                    all_results.append(analyze_and_print(W, f"{name}.whole", "WHOLE"))

                    # wq_b: per-head split. [32768, 1024] → 128 heads × 256 dim
                    if matched == "wq_b":
                        n_heads = W.shape[0] // 256
                        all_results.append(split_by_heads(W, name, n_heads=n_heads))

                    del W, tensor
                    if scale is not None: del scale
                    torch.cuda.empty_cache()
                    continue

                # ── Expert weights ──
                if "experts" in key and "shared" not in key and key.startswith("layers."):
                    parts = key.split(".")
                    try:
                        layer_idx = int(parts[1])
                        expert_idx = int(parts[4])
                        weight_name = parts[5]  # w1/w2/w3
                    except (IndexError, ValueError):
                        continue

                    if layer_idx not in target_layers:
                        continue
                    if not key.endswith(".weight"):
                        continue

                    if layer_idx not in expert_sample_set:
                        rng = np.random.RandomState(layer_idx * 1000 + 42)
                        expert_sample_set[layer_idx] = set(
                            rng.choice(256, size=min(args.expert_samples, 256), replace=False))
                    if expert_idx not in expert_sample_set[layer_idx]:
                        continue

                    tensor = f.get_tensor(key)
                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.int8 and scale is not None:
                        W = dequant_fp4(tensor.to(device), scale.to(device))
                    elif tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(device), scale.to(device))
                    else:
                        del tensor
                        continue

                    if W.dim() != 2 or min(W.shape) < 64:
                        del W, tensor
                        if scale is not None: del scale
                        torch.cuda.empty_cache()
                        continue

                    name = f"L{layer_idx:02d}.expert{expert_idx}.{weight_name}"
                    print(f"\n[{shard_name}] {name} {list(W.shape)}", flush=True)
                    all_results.append(analyze_and_print(W, f"{name}.whole", "WHOLE"))

                    del W, tensor
                    if scale is not None: del scale
                    torch.cuda.empty_cache()

        stale = [k for k in scale_buffer if os.path.basename(shard_file) in k]
        for k in stale:
            del scale_buffer[k]

    elapsed = time.time() - t_start
    print(f"\nDone! {len(all_results)} entries in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    return {"model": "DeepSeek-V4-Flash-280B", "layers": sorted(target_layers),
            "elapsed_s": round(elapsed, 1), "results": all_results}


# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen", "dsv4"])
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--expert-samples", type=int, default=4)
    args = parser.parse_args()

    if args.layers is None:
        args.layers = "0,1,2,3,7,11,31,47,63" if args.model == "qwen" else "0,10,20,30,42"

    if args.model == "qwen":
        output = run_qwen(args)
    else:
        output = run_dsv4(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
