#!/usr/bin/env python3
"""
Experiment for WeChat Issue 358:
What do the 128 calibration sentences actually do in AWQ and GPTQ?
1. X-ray of activation channels vs weight channels on Qwen/Qwen3-0.6B:
   - Weight column magnitudes are flat (regularized by AdamW weight decay during pretraining)
   - Input activation X channel magnitudes have extreme salient outliers (top 1% channels are 10x~50x larger!)
2. Hand-by-hand & full-model comparison on Qwen/Qwen3-0.6B (RTX 3080 Laptop):
   - RTN (Round-to-Nearest, data-free naive quantization)
   - AWQ (Activation-aware Weight Quantization: salient channel scaling s = |X|^alpha before rounding)
   - GPTQ (Exact Hessian H = X^T X Cholesky inverse column-by-column error compensation)
   Evaluated across 4-bit (g=128) and 3-bit (g=128) on:
   - Weight SNR (dB) vs Layer Output SNR (dB) (showing GPTQ sacrifices weight SNR to boost output SNR!)
   - End-to-end KL divergence vs BF16 and Perplexity (PPL) on held-out evaluation text
"""
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent


def calc_snr_db(orig: torch.Tensor, recon: torch.Tensor) -> float:
    sig = (orig.float() ** 2).mean().item()
    err = ((orig.float() - recon.float()) ** 2).mean().item()
    if err < 1e-15:
        return 99.0
    return 10.0 * math.log10(sig / err)


def rtn_quantize_group(w: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """
    Symmetric group-wise Round-To-Nearest (RTN) quantization.
    w: [N, K] float32
    """
    qmax = (1 << (bits - 1)) - 1  # 7 for 4-bit, 3 for 3-bit
    N, K = w.shape
    w_g = w.reshape(N, K // group_size, group_size)
    absmax = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = absmax / float(qmax)
    q = (w_g / scale).round().clamp(-qmax, qmax)
    return (q * scale).reshape(N, K)


def awq_quantize_group(
    w: torch.Tensor,
    x_act: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> tuple[torch.Tensor, float]:
    """
    AWQ (Lin et al., MLSys 2024):
    Search optimal per-channel scaling s = (mean(|X|))^alpha with alpha in [0, 1]
    to minimize layer output reconstruction error || X @ W^T - X @ W_hat^T ||_F^2.
    """
    # x_act: [T, K] float32
    s_x = x_act.abs().mean(dim=0).clamp(min=1e-6)  # [K]
    orig_out = x_act @ w.T  # [T, N]

    best_err = float("inf")
    best_alpha = 0.0
    best_w_hat = None

    for i in range(21):
        alpha = i * 0.05
        if alpha == 0.0:
            s = torch.ones_like(s_x)
        else:
            s = s_x.pow(alpha)
            s = s / (s.max() * s.min()).sqrt().clamp(min=1e-6)
        # Scale up salient weight columns before group quantization, then divide back
        w_scaled = w * s.unsqueeze(0)
        w_q = rtn_quantize_group(w_scaled, bits=bits, group_size=group_size)
        w_hat = w_q / s.unsqueeze(0)
        err = ((orig_out - x_act @ w_hat.T) ** 2).mean().item()
        if err < best_err:
            best_err = err
            best_alpha = round(alpha, 2)
            best_w_hat = w_hat

    return best_w_hat, best_alpha


def awq_top1pct_x2_quantize(
    w: torch.Tensor,
    x_act: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    scale_factor: float = 2.0,
) -> torch.Tensor:
    """
    Simplified textbook AWQ: protect the top 1% most salient activation channels by multiplying by 2.
    """
    s_x = x_act.abs().mean(dim=0)
    k_top = max(1, w.shape[1] // 100)
    top_idx = torch.topk(s_x, k=k_top).indices
    s = torch.ones_like(s_x)
    s[top_idx] = scale_factor
    w_scaled = w * s.unsqueeze(0)
    w_q = rtn_quantize_group(w_scaled, bits=bits, group_size=group_size)
    return w_q / s.unsqueeze(0)


def gptq_quantize_group(
    w: torch.Tensor,
    x_act: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    percdamp: float = 0.01,
    actorder: bool = True,
) -> torch.Tensor:
    """
    Standard GPTQ (Frantar et al., ICLR 2023) with Cholesky inverse Hessian and Lazy Batch Updates.
    w: [N, K] float32, x_act: [T, K] float32
    """
    qmax = (1 << (bits - 1)) - 1
    N, K = w.shape
    W = w.clone()

    # 1. Compute group scales on each group of 128 columns (fixed group scales before column updates)
    # Note: In GPTQ with static groups, group scales are computed per 128 contiguous columns
    w_g = W.reshape(N, K // group_size, group_size)
    scales_g = (w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / float(qmax)).repeat(1, 1, group_size).reshape(N, K)

    # 2. Compute Hessian H = (2 / T) * X^T @ X
    H = (x_act.T @ x_act) * (2.0 / x_act.shape[0])
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        scales_g = scales_g[:, perm]
        invperm = torch.argsort(perm)

    # Add damping lambda * I
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=W.device)
    H[diag_idx, diag_idx] += damp

    # Cholesky decomposition of H^-1
    H_inv = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(H_inv)
    H_inv = torch.linalg.cholesky(H_inv, upper=True)

    Q = torch.zeros_like(W)
    blocksize = 128

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = H_inv[i1:i2, i1:i2]

        for i in range(count):
            w_col = W1[:, i]
            d = Hinv1[i, i]
            s_col = scales_g[:, i1 + i]

            q_col = (w_col / s_col).round().clamp(-qmax, qmax) * s_col
            Q1[:, i] = q_col

            err = (w_col - q_col) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        if i2 < K:
            W[:, i2:] -= Err1 @ H_inv[i1:i2, i2:]

    if actorder:
        Q = Q[:, invperm]

    return Q


def main():
    print(f"Loading Qwen/Qwen3-0.6B on {torch.cuda.get_device_name(0)}...")
    model_id = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, local_files_only=True).to(DEVICE)
    model.eval()

    # 16 diverse calibration passages (strictly disjoint from eval_texts)
    calib_texts = [
        "Modern microprocessors employ superscalar out-of-order execution pipelines, branch target buffers, and non-blocking caches to maximize instruction-level parallelism.",
        "Post-training quantization compresses neural network weights from 16-bit floating-point to 4-bit integers without backpropagation or gradient updates.",
        "Activation-aware weight quantization observes that a small fraction of input channels carry significantly larger activation magnitudes across all tokens.",
        "Optimal Brain Surgeon and GPTQ use the inverse Hessian matrix of layer inputs to compensate quantization rounding errors across correlated weight columns.",
        "Graphics processing units execute matrix multiply-accumulate instructions across warps of 32 threads using dedicated tensor core pipelines.",
        "Memory bandwidth bottlenecks single-token autoregressive decoding, because the entire weight matrix must be fetched from VRAM for every generated word.",
        "Transformer attention mechanisms compute query, key, and value projections followed by scaled dot-product attention and output projection.",
        "Grouped-query attention reduces key-value cache footprint by sharing a single key and value head across multiple query heads.",
        "During pretraining, the AdamW optimizer applies decoupled weight decay, which penalizes large weight parameters and produces near-Gaussian weight distributions.",
        "However, residual stream activations accumulate representations across layers and exhibit extreme outlier channels that are dozens of times larger than median channels.",
        "When naive round-to-nearest quantization is applied uniformly, rounding errors on salient channels are amplified by large input activations.",
        "By scaling up the top one percent of salient weight channels before rounding, AWQ reduces effective quantization noise without mixed-precision hardware overhead.",
        "Cholesky decomposition provides a numerically stable way to invert the empirical Hessian matrix and propagate column errors from left to right.",
        "Operating systems manage virtual memory pages, scheduling processes across CPU cores while coordinating asynchronous direct memory access transfers.",
        "Compiler optimization passes perform dead code elimination, loop unrolling, and register allocation to minimize instruction latency and register spilling.",
        "Distributed training frameworks overlap all-reduce gradient communication with backward computation across high-speed interconnects.",
    ]

    # Separate held-out evaluation passages (matching Issue 357 evaluation protocol + additional technical passages)
    eval_texts = [
        "In computer architecture, floating-point representation trades precision for dynamic range, whereas integer arithmetic provides uniform step sizes across the representable interval. When neural network weights follow a near-Gaussian distribution without heavy-tailed outliers, uniform integer grids often achieve higher signal-to-noise ratios than low-mantissa floating-point grids.",
        "Large language models rely on matrix multiplications across attention projections and feed-forward networks. During single-token autoregressive decoding, execution time on graphics processing units is primarily bottlenecked by memory bandwidth rather than arithmetic throughput, making 4-bit weight compression effective.",
        "However, as batch size increases during prefill or parallel decoding, runtime activation quantization and per-group dequantization overhead can erode the bandwidth savings if not carefully fused into the tensor core pipeline.",
    ]

    calib_str = "\n\n".join(calib_texts)
    calib_ids = tok(calib_str, return_tensors="pt").input_ids.to(DEVICE)
    eval_str = "\n\n".join(eval_texts)
    eval_ids = tok(eval_str, return_tensors="pt").input_ids.to(DEVICE)
    print(f"Calibration tokens: {calib_ids.shape[1]}, Eval tokens: {eval_ids.shape[1]}")

    # Step 1: Collect real input activations X across all 28 layers during calibration pass AND eval pass
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layer_inputs = [{} for _ in range(len(model.model.layers))]
    eval_layer_inputs = [{} for _ in range(len(model.model.layers))]
    hooks = []

    for l_idx, layer in enumerate(model.model.layers):
        def make_hook(target_list, li, key):
            def hook(mod, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                target_list[li][key] = x
            return hook

        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(make_hook(layer_inputs, l_idx, "attn_in")))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_inputs, l_idx, "o_in")))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(make_hook(layer_inputs, l_idx, "mlp_in")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_inputs, l_idx, "down_in")))

    with torch.no_grad():
        _ = model(calib_ids)

    for h in hooks:
        h.remove()
    hooks.clear()

    for l_idx, layer in enumerate(model.model.layers):
        def make_eval_hook(target_list, li, key):
            def hook(mod, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                target_list[li][key] = x
            return hook

        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(make_eval_hook(eval_layer_inputs, l_idx, "attn_in")))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_eval_hook(eval_layer_inputs, l_idx, "o_in")))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(make_eval_hook(eval_layer_inputs, l_idx, "mlp_in")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_eval_hook(eval_layer_inputs, l_idx, "down_in")))

    with torch.no_grad():
        _ = model(eval_ids)

    for h in hooks:
        h.remove()

    # Analyze Experiment 1: Channel Salience X-ray (Activation vs Weight channel magnitudes)
    xray_stats = {"attn_in": [], "o_in": [], "mlp_in": [], "down_in": []}
    for l_idx, layer in enumerate(model.model.layers):
        pairs = [
            ("attn_in", layer.self_attn.q_proj.weight.data.float()),
            ("o_in", layer.self_attn.o_proj.weight.data.float()),
            ("mlp_in", layer.mlp.gate_proj.weight.data.float()),
            ("down_in", layer.mlp.down_proj.weight.data.float()),
        ]
        for key, w in pairs:
            x = layer_inputs[l_idx][key]  # [T, K]
            act_ch_mean = x.abs().mean(dim=0)  # [K]
            w_ch_mean = w.abs().mean(dim=0)    # [K]

            act_med = act_ch_mean.median().item()
            act_max = act_ch_mean.max().item()
            k_top1 = max(1, x.shape[1] // 100)
            act_top1 = torch.topk(act_ch_mean, k=k_top1).values.mean().item()

            w_med = w_ch_mean.median().item()
            w_max = w_ch_mean.max().item()
            w_top1 = torch.topk(w_ch_mean, k=k_top1).values.mean().item()

            # Also check Hessian diagonal H_jj = mean(x_j^2): max / median ratio!
            h_diag = (x ** 2).mean(dim=0)
            h_max_med = (h_diag.max() / h_diag.median()).item()

            # Under naive 4-bit RTN, what % of total per-channel squared output error ||x_j * dw_j||^2 comes from top 1% channels?
            w_rtn = rtn_quantize_group(w, bits=4, group_size=128)
            dw_sq = ((w - w_rtn) ** 2).mean(dim=0)  # [K]
            ch_out_err_energy = h_diag * dw_sq       # [K]
            top1_idx = torch.topk(act_ch_mean, k=k_top1).indices
            top1_err_share_pct = (ch_out_err_energy[top1_idx].sum() / ch_out_err_energy.sum()).item() * 100.0

            xray_stats[key].append({
                "layer": l_idx,
                "act_max_over_median": round(act_max / act_med, 2),
                "act_top1pct_over_median": round(act_top1 / act_med, 2),
                "hessian_diag_max_over_median": round(h_max_med, 1),
                "weight_max_over_median": round(w_max / w_med, 2),
                "weight_top1pct_over_median": round(w_top1 / w_med, 2),
                "top1pct_ch_err_energy_share_pct": round(top1_err_share_pct, 1),
            })

    xray_summary = {}
    for key, lst in xray_stats.items():
        xray_summary[key] = {
            "avg_act_max_over_median": round(sum(x["act_max_over_median"] for x in lst) / len(lst), 2),
            "peak_act_max_over_median": max(x["act_max_over_median"] for x in lst),
            "avg_act_top1pct_over_median": round(sum(x["act_top1pct_over_median"] for x in lst) / len(lst), 2),
            "avg_hessian_diag_max_over_median": round(sum(x["hessian_diag_max_over_median"] for x in lst) / len(lst), 1),
            "peak_hessian_diag_max_over_median": max(x["hessian_diag_max_over_median"] for x in lst),
            "avg_weight_max_over_median": round(sum(x["weight_max_over_median"] for x in lst) / len(lst), 2),
            "avg_weight_top1pct_over_median": round(sum(x["weight_top1pct_over_median"] for x in lst) / len(lst), 2),
            "avg_top1pct_ch_err_energy_share_pct": round(sum(x["top1pct_ch_err_energy_share_pct"] for x in lst) / len(lst), 1),
        }

    print("\n=== Experiment 1: Channel X-Ray Summary across 28 Layers ===")
    print(json.dumps(xray_summary, indent=2))

    # Pick a representative layer (e.g., Layer 14 mlp_in / down_in) to show top-5 channels vs median
    rep_layer = 14
    x_rep = layer_inputs[rep_layer]["mlp_in"]
    w_rep = model.model.layers[rep_layer].mlp.up_proj.weight.data.float()
    act_ch = x_rep.abs().mean(dim=0)
    h_diag_rep = (x_rep ** 2).mean(dim=0)
    h_diag_med = h_diag_rep.median().item()
    w_ch = w_rep.abs().mean(dim=0)
    top5_vals, top5_idx = torch.topk(act_ch, k=5)
    rep_detail = {
        "layer": rep_layer,
        "input_type": "mlp_in (up_proj / gate_proj)",
        "num_channels": x_rep.shape[1],
        "median_act_abs": round(act_ch.median().item(), 4),
        "median_hessian_diag": round(h_diag_med, 4),
        "top5_channels": [
            {
                "ch_idx": int(idx.item()),
                "act_abs_mean": round(val.item(), 4),
                "ratio_to_median": round(val.item() / act_ch.median().item(), 2),
                "ratio_sq_to_median": round((val.item() / act_ch.median().item()) ** 2, 1),
                "hessian_diag_ratio_to_median": round((h_diag_rep[idx].item() / h_diag_med), 1),
                "weight_abs_mean": round(w_ch[idx].item(), 5),
                "weight_ratio_to_median": round(w_ch[idx].item() / w_ch.median().item(), 2),
            }
            for val, idx in zip(top5_vals, top5_idx)
        ],
    }
    print("\nRepresentative Layer 14 mlp_in Top-5 Salient Channels:")
    print(json.dumps(rep_detail, indent=2))

    # Step 2: Baseline BF16 evaluation
    with torch.no_grad():
        out_bf16 = model(eval_ids, labels=eval_ids)
        ppl_bf16 = math.exp(out_bf16.loss.item())
        logits_bf16 = out_bf16.logits.float()

    orig_state = {k: v.clone() for k, v in model.state_dict().items() if "layers." in k}

    def run_full_quantization_benchmark(method: str, bits: int, group_size: int = 128):
        t0 = time.perf_counter()
        weight_snrs = []
        calib_output_snrs = []
        eval_output_snrs = []
        alphas = []

        with torch.no_grad():
            for k, v in orig_state.items():
                model.state_dict()[k].copy_(v)

            for l_idx, layer in enumerate(model.model.layers):
                projs = [
                    ("q_proj", layer.self_attn.q_proj, "attn_in"),
                    ("k_proj", layer.self_attn.k_proj, "attn_in"),
                    ("v_proj", layer.self_attn.v_proj, "attn_in"),
                    ("o_proj", layer.self_attn.o_proj, "o_in"),
                    ("gate_proj", layer.mlp.gate_proj, "mlp_in"),
                    ("up_proj", layer.mlp.up_proj, "mlp_in"),
                    ("down_proj", layer.mlp.down_proj, "down_in"),
                ]
                for pname, mod, in_key in projs:
                    w = mod.weight.data.float()
                    x_in = layer_inputs[l_idx][in_key]
                    x_eval_in = eval_layer_inputs[l_idx][in_key]
                    if method == "rtn":
                        w_hat = rtn_quantize_group(w, bits=bits, group_size=group_size)
                    elif method == "awq_top1_x2":
                        w_hat = awq_top1pct_x2_quantize(w, x_in, bits=bits, group_size=group_size, scale_factor=2.0)
                    elif method == "awq":
                        w_hat, best_a = awq_quantize_group(w, x_in, bits=bits, group_size=group_size)
                        alphas.append(best_a)
                    elif method == "gptq":
                        w_hat = gptq_quantize_group(w, x_in, bits=bits, group_size=group_size, actorder=True)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # Record Weight SNR and Layer Output SNR on sampled layers
                    if l_idx in [0, 7, 14, 21, 27]:
                        w_snr = calc_snr_db(w, w_hat)
                        c_snr = calc_snr_db(x_in @ w.T, x_in @ w_hat.T)
                        e_snr = calc_snr_db(x_eval_in @ w.T, x_eval_in @ w_hat.T)
                        weight_snrs.append(w_snr)
                        calib_output_snrs.append(c_snr)
                        eval_output_snrs.append(e_snr)

                    mod.weight.data.copy_(w_hat.to(torch.bfloat16))

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            out_eval = model(eval_ids, labels=eval_ids)
            ppl = math.exp(out_eval.loss.item())
            logits = out_eval.logits.float()
            log_p = F.log_softmax(logits_bf16, dim=-1)
            log_q = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(log_q, log_p, log_target=True, reduction="batchmean").item()

        res = {
            "method": method,
            "bits": bits,
            "group_size": group_size,
            "avg_weight_snr_db": round(sum(weight_snrs) / len(weight_snrs), 2),
            "avg_calib_output_snr_db": round(sum(calib_output_snrs) / len(calib_output_snrs), 2),
            "avg_eval_output_snr_db": round(sum(eval_output_snrs) / len(eval_output_snrs), 2),
            "eval_kl_div": round(kl, 4),
            "eval_ppl": round(ppl, 2),
            "quant_time_sec": round(elapsed, 2),
        }
        if alphas:
            res["avg_awq_alpha"] = round(sum(alphas) / len(alphas), 3)
        return res

    print("\n=== Experiment 2: Running RTN vs AWQ vs GPTQ on Qwen3-0.6B ===")
    benchmarks = {
        "bf16_baseline": {
            "method": "bf16",
            "bits": 16,
            "avg_weight_snr_db": float("inf"),
            "avg_output_snr_db": float("inf"),
            "eval_kl_div": 0.0,
            "eval_ppl": round(ppl_bf16, 2),
        }
    }

    configs = [
        ("w4_g16_rtn", "rtn", 4, 16),
        ("w4_g128_rtn", "rtn", 4, 128),
        ("w4_g128_awq_top1_x2", "awq_top1_x2", 4, 128),
        ("w4_g128_awq", "awq", 4, 128),
        ("w4_g128_gptq", "gptq", 4, 128),
        ("w3_g128_rtn", "rtn", 3, 128),
        ("w3_g128_awq_top1_x2", "awq_top1_x2", 3, 128),
        ("w3_g128_awq", "awq", 3, 128),
        ("w3_g128_gptq", "gptq", 3, 128),
    ]

    for label, m, b, g in configs:
        res = run_full_quantization_benchmark(m, bits=b, group_size=g)
        benchmarks[label] = res
        print(
            f"  {label}: Weight SNR={res['avg_weight_snr_db']} dB, "
            f"Calib Out SNR={res['avg_calib_output_snr_db']} dB, "
            f"Eval Out SNR={res['avg_eval_output_snr_db']} dB, "
            f"KL={res['eval_kl_div']}, PPL={res['eval_ppl']} ({res['quant_time_sec']}s)"
        )

    #Restore original weights
    with torch.no_grad():
        for k, v in orig_state.items():
            model.state_dict()[k].copy_(v)

    final_json = {
        "gpu": torch.cuda.get_device_name(0),
        "model": model_id,
        "calib_tokens": int(calib_ids.shape[1]),
        "eval_tokens": int(eval_ids.shape[1]),
        "xray_channel_summary": xray_summary,
        "layer14_mlp_in_top5": rep_detail,
        "benchmarks": benchmarks,
    }

    out_path = OUT_DIR / "exp_rtn_awq_gptq.json"
    out_path.write_text(json.dumps(final_json, indent=2) + "\n")
    print(f"\nSaved complete results to {out_path}")


if __name__ == "__main__":
    main()
