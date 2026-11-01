#!/usr/bin/env python3
"""
Experiment 1 & 2 for WeChat Issue 357:
Verifying secemp's HINT4 (Hardware-Invariant NVFP4 via INT8 Tensor Cores):
1. PTX / SASS census across sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 (proving sm_90/100/120 drop native .s4 IMMA)
2. Dual-Grid (INT4 vs FP4*2={0,1,2,3,4,6,8,12} fitting in INT8, flag in E4M3 sign bit)
3. Two-Level (TL: 2x16 weights sharing 1 E4M3 scale + 1..8 integer sub-scale for K=32 mma.s8)
4. Kurtosis & Hadamard Rotation test on real Qwen3-0.6B weights (why near-Gaussian weights don't need rotation)
5. Layer sensitivity & W8 o_proj recovery on real Qwen3-0.6B forward / perplexity / KL
"""
import json
import math
import subprocess
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent

# Positive E2M1 magnitudes: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEVICE)
E2M1_GRID = torch.cat([-E2M1_POS.flip(0)[:-1], E2M1_POS])  # 15 unique values
# Doubled E2M1: 0, 1, 2, 3, 4, 6, 8, 12 -> exact integers!
E2M1_X2_GRID = E2M1_GRID * 2.0
# Symmetric INT4 grid: -7 .. +7
INT4_GRID = torch.arange(-7, 8, dtype=torch.float32, device=DEVICE)


def round_to_e4m3_with_global(x: torch.Tensor, global_scale: float, round_up: bool = False) -> torch.Tensor:
    """
    NVFP4 / HINT4 uses a per-tensor FP32 global scale + per-block E4M3 scale!
    Without the per-tensor FP32 scale, small weights ~0.01 underflow E4M3's minimum normal 2^-6.
    """
    scaled = x / global_scale
    x_clamped = torch.clamp(scaled, min=2**-9, max=448.0)
    s_fp8 = x_clamped.to(torch.float8_e4m3fn).to(torch.float32)
    if round_up:
        mask = s_fp8 < x_clamped
        exp = torch.floor(torch.log2(torch.clamp(s_fp8, min=2**-6)))
        ulp = torch.pow(2.0, exp - 3.0)
        s_up = torch.clamp(s_fp8 + ulp, max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
        s_fp8 = torch.where(mask, s_up, s_fp8)
    return torch.clamp(s_fp8 * global_scale, min=1e-12)


def quantize_blocks_grid(w_blocks: torch.Tensor, grid: torch.Tensor, max_val: float, search_scale: bool = False, round_up: bool = False):
    absmax = w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    global_s = (absmax.max().item() / max_val) / 224.0  # Map tensor peak to E4M3 ~224.0
    if global_s < 1e-12:
        global_s = 1.0
    if not search_scale:
        s = round_to_e4m3_with_global(absmax / max_val, global_s, round_up=round_up)
        norm = w_blocks / s
        idx = (norm.unsqueeze(-1) - grid).abs().argmin(dim=-1)
        q = grid[idx]
        w_hat = q * s
        return q, s, w_hat
    else:
        best_err = torch.full((w_blocks.shape[0], 1), float("inf"), device=w_blocks.device)
        best_q = torch.zeros_like(w_blocks)
        best_s = torch.ones((w_blocks.shape[0], 1), device=w_blocks.device)
        for ratio in [0.82, 0.86, 0.90, 0.94, 0.97, 1.0, 1.03]:
            s_cand = round_to_e4m3_with_global((absmax * ratio) / max_val, global_s, round_up=True)
            norm = w_blocks / s_cand
            idx = (norm.unsqueeze(-1) - grid).abs().argmin(dim=-1)
            q_cand = grid[idx]
            w_cand = q_cand * s_cand
            err = ((w_blocks - w_cand) ** 2).sum(dim=-1, keepdim=True)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q_cand, best_q)
            best_s = torch.where(better, s_cand, best_s)
        return best_q, best_s, best_q * best_s


def quantize_dual_grid(w_blocks: torch.Tensor, search_scale: bool = True):
    q_int4, s_int4, hat_int4 = quantize_blocks_grid(w_blocks, INT4_GRID, 7.0, search_scale=search_scale, round_up=True)
    q_fp4x2, s_fp4x2, hat_fp4x2 = quantize_blocks_grid(w_blocks, E2M1_X2_GRID, 12.0, search_scale=search_scale, round_up=True)
    err_int4 = ((w_blocks - hat_int4) ** 2).sum(dim=-1, keepdim=True)
    err_fp4 = ((w_blocks - hat_fp4x2) ** 2).sum(dim=-1, keepdim=True)
    use_fp4 = err_fp4 < err_int4
    w_hat = torch.where(use_fp4, hat_fp4x2, hat_int4)
    q_int8 = torch.where(use_fp4, q_fp4x2, q_int4)
    s_final = torch.where(use_fp4, s_fp4x2, s_int4)
    return q_int8, s_final, w_hat, use_fp4.squeeze(-1)


def quantize_tl_two_level(w_blocks_32: torch.Tensor):
    """
    HINT4 TL (Two-Level):
    Each pair of 16-weight groups (total 32 along K) shares ONE E4M3 scale,
    and each 16-weight sub-block gets an integer multiplier m in {1..8}.
    """
    sub_absmax = w_blocks_32.abs().amax(dim=-1)  # [N_32, 2]
    pair_absmax = sub_absmax.amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [N_32, 1]
    global_s = (pair_absmax.max().item() / 96.0) / 224.0
    if global_s < 1e-12:
        global_s = 1.0

    # Shared E4M3 scale for the 32-weight pair (using 12*8=96 denominator; for INT4 blocks we scale grid or test candidate shared scales)
    s_shared = round_to_e4m3_with_global(pair_absmax / 56.0, global_s, round_up=True)

    best_hat = torch.zeros_like(w_blocks_32)
    for sub_idx in range(2):
        w_sub = w_blocks_32[:, sub_idx, :]
        best_e = torch.full((w_sub.shape[0], 1), float("inf"), device=w_sub.device)
        best_w = torch.zeros_like(w_sub)
        # Integer multiplier m in 1..8 for INT4 grid, or FP4*2 grid (multiplier 0 marker in header)
        for m in range(1, 9):
            for grid in [INT4_GRID, E2M1_X2_GRID]:
                # Note: q * m is in [-96, 96] <= 127, exact signed byte!
                s_eff = s_shared * float(m) * (7.0 / 12.0 if len(grid) == 15 and grid.max() == 12.0 else 1.0)
                # Wait: in hardware, both sub-blocks share the exact same FP32/E4M3 scale!
                # So s_eff is strictly s_shared * m! Let's use strictly s_shared * m:
                s_eff_hw = s_shared * float(m)
                norm = w_sub / s_eff_hw
                idx = (norm.unsqueeze(-1) - grid).abs().argmin(dim=-1)
                q = grid[idx]
                q_widened = (q * float(m)).clamp(-127.0, 127.0)  # exact INT8 operand fed to mma.s8!
                w_c = q_widened * s_shared
                e = ((w_sub - w_c) ** 2).sum(dim=-1, keepdim=True)
                better = e < best_e
                best_e = torch.where(better, e, best_e)
                best_w = torch.where(better, w_c, best_w)
        best_hat[:, sub_idx, :] = best_w
    return best_hat


def hadamard_matrix(n: int, device=DEVICE) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


def calc_snr_db(orig: torch.Tensor, recon: torch.Tensor) -> float:
    sig = (orig.float() ** 2).mean().item()
    err = ((orig.float() - recon.float()) ** 2).mean().item()
    if err < 1e-15:
        return 99.0
    return 10.0 * math.log10(sig / err)


def calc_kurtosis(w: torch.Tensor) -> float:
    wf = w.float().reshape(-1)
    mean = wf.mean()
    diff = wf - mean
    var = (diff ** 2).mean()
    fourth = (diff ** 4).mean()
    return (fourth / (var ** 2)).item()


def run_ptx_sass_census():
    ptxas = "/home/lmxxf/.local/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas"
    cuobjdump = "/home/lmxxf/.local/lib/python3.10/site-packages/triton/backends/nvidia/bin/cuobjdump"

    ptx_s4 = """
.version 8.7
.target {target}
.address_size 64

.visible .entry kernel_mma_s4(
    .param .u64 out_ptr,
    .param .u32 a0, .param .u32 a1,
    .param .u32 b0,
    .param .u32 c0, .param .u32 c1, .param .u32 c2, .param .u32 c3
)
{{
    .reg .u64 %ptr;
    .reg .b32 %ra<2>, %rb<1>, %rc<4>, %rd<4>;
    ld.param.u64 %ptr, [out_ptr];
    ld.param.u32 %ra0, [a0];
    ld.param.u32 %ra1, [a1];
    ld.param.u32 %rb0, [b0];
    ld.param.u32 %rc0, [c0];
    ld.param.u32 %rc1, [c1];
    ld.param.u32 %rc2, [c2];
    ld.param.u32 %rc3, [c3];

    mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32
        {{%rd0, %rd1, %rd2, %rd3}},
        {{%ra0, %ra1}},
        {{%rb0}},
        {{%rc0, %rc1, %rc2, %rc3}};

    st.global.v4.u32 [%ptr], {{%rd0, %rd1, %rd2, %rd3}};
    ret;
}}
"""

    ptx_s8 = """
.version 8.7
.target {target}
.address_size 64

.visible .entry kernel_mma_s8(
    .param .u64 out_ptr,
    .param .u32 a0, .param .u32 a1, .param .u32 a2, .param .u32 a3,
    .param .u32 b0, .param .u32 b1,
    .param .u32 c0, .param .u32 c1, .param .u32 c2, .param .u32 c3
)
{{
    .reg .u64 %ptr;
    .reg .b32 %ra<4>, %rb<2>, %rc<4>, %rd<4>;
    ld.param.u64 %ptr, [out_ptr];
    ld.param.u32 %ra0, [a0];
    ld.param.u32 %ra1, [a1];
    ld.param.u32 %ra2, [a2];
    ld.param.u32 %ra3, [a3];
    ld.param.u32 %rb0, [b0];
    ld.param.u32 %rb1, [b1];
    ld.param.u32 %rc0, [c0];
    ld.param.u32 %rc1, [c1];
    ld.param.u32 %rc2, [c2];
    ld.param.u32 %rc3, [c3];

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {{%rd0, %rd1, %rd2, %rd3}},
        {{%ra0, %ra1, %ra2, %ra3}},
        {{%rb0, %rb1}},
        {{%rc0, %rc1, %rc2, %rc3}};

    st.global.v4.u32 [%ptr], {{%rd0, %rd1, %rd2, %rd3}};
    ret;
}}
"""
    results = {}
    targets = ["sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        for t in targets:
            results[t] = {}
            for kind, tpl in [("s4_m16n8k32", ptx_s4), ("s8_m16n8k32", ptx_s8)]:
                ptx_file = tmp_p / f"{kind}_{t}.ptx"
                cubin_file = tmp_p / f"{kind}_{t}.cubin"
                ptx_file.write_text(tpl.format(target=t))
                cp = subprocess.run([ptxas, f"-arch={t}", str(ptx_file), "-o", str(cubin_file)], capture_output=True, text=True)
                if cp.returncode != 0:
                    results[t][kind] = {"error": cp.stderr.strip()}
                    continue
                sass = subprocess.check_output([cuobjdump, "-sass", str(cubin_file)], text=True)
                imma_only = [line.split("*/")[-1].strip() for line in sass.splitlines() if "IMMA" in line]
                lop3_count = sum(1 for line in sass.splitlines() if "LOP3" in line)
                shf_count = sum(1 for line in sass.splitlines() if "SHF" in line)
                results[t][kind] = {
                    "imma_instructions": imma_only,
                    "imma_count": len(imma_only),
                    "lop3_count": lop3_count,
                    "shf_count": shf_count,
                }
    return results


def main():
    print("1. Running PTX -> SASS Census across sm_80, sm_86, sm_89, sm_90, sm_100, sm_120...")
    census = run_ptx_sass_census()
    print(json.dumps(census, indent=2))

    print("\n2. Loading Qwen/Qwen3-0.6B on RTX 3080 Laptop (sm_86)...")
    model_id = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, local_files_only=True).to(DEVICE)
    model.eval()

    kurtosis_by_proj = {"q_proj": [], "k_proj": [], "v_proj": [], "o_proj": [], "gate_proj": [], "up_proj": [], "down_proj": []}
    snr_stats = {
        "nvfp4_r2n": [],
        "int4_r2n": [],
        "hint4_dual_r2n": [],
        "hint4_dual_opt": [],
        "hint4_tl_k32": [],
        "nvfp4_rot": [],
        "int4_rot": [],
        "fp4_win_ratio": [],
    }

    H64 = hadamard_matrix(64, DEVICE)
    torch.manual_seed(42)
    rand_sign = torch.randint(0, 2, (64,), device=DEVICE).float() * 2.0 - 1.0
    RH64 = H64 * rand_sign.unsqueeze(0)

    for layer_idx, layer in enumerate(model.model.layers):
        projs = {
            "q_proj": layer.self_attn.q_proj.weight.data,
            "k_proj": layer.self_attn.k_proj.weight.data,
            "v_proj": layer.self_attn.v_proj.weight.data,
            "o_proj": layer.self_attn.o_proj.weight.data,
            "gate_proj": layer.mlp.gate_proj.weight.data,
            "up_proj": layer.mlp.up_proj.weight.data,
            "down_proj": layer.mlp.down_proj.weight.data,
        }
        for name, w in projs.items():
            kurtosis_by_proj[name].append(calc_kurtosis(w))
            if layer_idx in [0, 7, 14, 21, 27]:
                wf = w.float()
                blocks16 = wf.reshape(-1, 16)
                _, _, hat_nvfp4 = quantize_blocks_grid(blocks16, E2M1_GRID, 6.0, search_scale=False, round_up=False)
                _, _, hat_int4 = quantize_blocks_grid(blocks16, INT4_GRID, 7.0, search_scale=False, round_up=False)
                _, _, hat_dual_r2n, use_fp4_r2n = quantize_dual_grid(blocks16, search_scale=False)
                _, _, hat_dual_opt, use_fp4_opt = quantize_dual_grid(blocks16, search_scale=True)

                blocks32 = wf.reshape(-1, 2, 16)
                hat_tl = quantize_tl_two_level(blocks32)

                w_rot64 = (wf.reshape(-1, 64) @ RH64).reshape(-1, 16)
                _, _, hat_nvfp4_rot = quantize_blocks_grid(w_rot64, E2M1_GRID, 6.0, search_scale=False, round_up=False)
                _, _, hat_int4_rot = quantize_blocks_grid(w_rot64, INT4_GRID, 7.0, search_scale=False, round_up=False)
                hat_nvfp4_unrot = (hat_nvfp4_rot.reshape(-1, 64) @ RH64.T).reshape_as(wf)
                hat_int4_unrot = (hat_int4_rot.reshape(-1, 64) @ RH64.T).reshape_as(wf)

                snr_stats["nvfp4_r2n"].append(calc_snr_db(wf, hat_nvfp4.reshape_as(wf)))
                snr_stats["int4_r2n"].append(calc_snr_db(wf, hat_int4.reshape_as(wf)))
                snr_stats["hint4_dual_r2n"].append(calc_snr_db(wf, hat_dual_r2n.reshape_as(wf)))
                snr_stats["hint4_dual_opt"].append(calc_snr_db(wf, hat_dual_opt.reshape_as(wf)))
                snr_stats["hint4_tl_k32"].append(calc_snr_db(wf, hat_tl.reshape_as(wf)))
                snr_stats["nvfp4_rot"].append(calc_snr_db(wf, hat_nvfp4_unrot))
                snr_stats["int4_rot"].append(calc_snr_db(wf, hat_int4_unrot))
                snr_stats["fp4_win_ratio"].append(use_fp4_opt.float().mean().item())

    summary_kurtosis = {k: round(sum(v) / len(v), 3) for k, v in kurtosis_by_proj.items()}
    summary_snr = {k: round(sum(v) / len(v), 3) for k, v in snr_stats.items()}
    print("\nSummary Kurtosis across 28 layers (Gaussian = 3.0):")
    print(json.dumps(summary_kurtosis, indent=2))
    print("\nSummary Reconstruction SNR (dB) across 35 sampled matrices:")
    print(json.dumps(summary_snr, indent=2))

    # Synthetic Heavy-Tailed vs Gaussian Tensor Test
    torch.manual_seed(123)
    gauss_t = torch.randn(1024, 1024, device=DEVICE) * 0.02
    outlier_t = torch.randn(1024, 1024, device=DEVICE) * 0.02
    outlier_t[:, ::64] *= 12.0

    synth_results = {}
    for label, t in [("gaussian_kurt_3", gauss_t), ("outlier_channels", outlier_t)]:
        b16 = t.reshape(-1, 16)
        _, _, h_fp4 = quantize_blocks_grid(b16, E2M1_GRID, 6.0, search_scale=False)
        _, _, h_i4 = quantize_blocks_grid(b16, INT4_GRID, 7.0, search_scale=False)
        _, _, h_dual, win_fp4 = quantize_dual_grid(b16, search_scale=True)
        t_rot = (t.reshape(-1, 64) @ RH64).reshape(-1, 16)
        _, _, h_fp4_r = quantize_blocks_grid(t_rot, E2M1_GRID, 6.0, search_scale=False)
        _, _, h_i4_r = quantize_blocks_grid(t_rot, INT4_GRID, 7.0, search_scale=False)
        synth_results[label] = {
            "kurtosis": round(calc_kurtosis(t), 2),
            "nvfp4_snr_db": round(calc_snr_db(t, h_fp4.reshape_as(t)), 2),
            "int4_snr_db": round(calc_snr_db(t, h_i4.reshape_as(t)), 2),
            "hint4_dual_snr_db": round(calc_snr_db(t, h_dual.reshape_as(t)), 2),
            "nvfp4_rotated_snr_db": round(calc_snr_db(t, (h_fp4_r.reshape(-1, 64) @ RH64.T).reshape_as(t)), 2),
            "int4_rotated_snr_db": round(calc_snr_db(t, (h_i4_r.reshape(-1, 64) @ RH64.T).reshape_as(t)), 2),
            "fp4_grid_win_pct": round(win_fp4.float().mean().item() * 100, 1),
        }
    print("\nSynthetic Gaussian vs Outlier Tensor Test:")
    print(json.dumps(synth_results, indent=2))

    # Whole-Model Perplexity & KL Divergence on Qwen3-0.6B
    eval_texts = [
        "In computer architecture, floating-point representation trades precision for dynamic range, whereas integer arithmetic provides uniform step sizes across the representable interval. When neural network weights follow a near-Gaussian distribution without heavy-tailed outliers, uniform integer grids often achieve higher signal-to-noise ratios than low-mantissa floating-point grids.",
        "Large language models rely on matrix multiplications across attention projections and feed-forward networks. During single-token autoregressive decoding, execution time on graphics processing units is primarily bottlenecked by memory bandwidth rather than arithmetic throughput, making 4-bit weight compression effective.",
        "However, as batch size increases during prefill or parallel decoding, runtime activation quantization and per-group dequantization overhead can erode the bandwidth savings if not carefully fused into the tensor core pipeline.",
    ]
    full_text = "\n\n".join(eval_texts)
    input_ids = tok(full_text, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        out_bf16 = model(input_ids, labels=input_ids)
        ppl_bf16 = math.exp(out_bf16.loss.item())
        logits_bf16 = out_bf16.logits.float()

    orig_state = {k: v.clone() for k, v in model.state_dict().items() if "layers." in k}

    def eval_quantized_model(mode: str, protect_o_proj: bool = False, act_int8: bool = False):
        with torch.no_grad():
            for k, v in orig_state.items():
                model.state_dict()[k].copy_(v)
            for layer in model.model.layers:
                for pname in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                    mod = getattr(layer.self_attn, pname) if pname in ["q_proj", "k_proj", "v_proj", "o_proj"] else getattr(layer.mlp, pname)
                    w = mod.weight.data.float()
                    if protect_o_proj and pname == "o_proj":
                        b16 = w.reshape(-1, 16)
                        gs = (b16.abs().max().item() / 127.0) / 224.0
                        s = round_to_e4m3_with_global(b16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0, max(gs, 1e-12), round_up=True)
                        q = (b16 / s).round().clamp(-127, 127)
                        mod.weight.data.copy_((q * s).reshape_as(w).to(torch.bfloat16))
                        continue
                    b16 = w.reshape(-1, 16)
                    if mode == "nvfp4":
                        _, _, hat = quantize_blocks_grid(b16, E2M1_GRID, 6.0, search_scale=False)
                    elif mode == "int4":
                        _, _, hat = quantize_blocks_grid(b16, INT4_GRID, 7.0, search_scale=False)
                    elif mode == "hint4_dual":
                        _, _, hat, _ = quantize_dual_grid(b16, search_scale=True)
                    elif mode == "hint4_tl":
                        hat = quantize_tl_two_level(w.reshape(-1, 2, 16))
                    elif mode == "int4_rot":
                        w_r = (w.reshape(-1, 64) @ RH64).reshape(-1, 16)
                        _, _, h_r = quantize_blocks_grid(w_r, INT4_GRID, 7.0, search_scale=False)
                        hat = (h_r.reshape(-1, 64) @ RH64.T)
                    mod.weight.data.copy_(hat.reshape_as(w).to(torch.bfloat16))

        hooks = []
        if act_int8:
            def quant_act_hook(module, args):
                x = args[0]
                xf = x.float()
                amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                s = amax / 127.0
                x_q = (xf / s).round().clamp(-127, 127) * s
                return (x_q.to(x.dtype),)

            for layer in model.model.layers:
                for pname in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                    mod = getattr(layer.self_attn, pname) if pname in ["q_proj", "k_proj", "v_proj", "o_proj"] else getattr(layer.mlp, pname)
                    hooks.append(mod.register_forward_pre_hook(quant_act_hook))

        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            ppl = math.exp(out.loss.item())
            logits = out.logits.float()
            log_p = F.log_softmax(logits_bf16, dim=-1)
            log_q = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(log_q, log_p, log_target=True, reduction="batchmean").item()

        for h in hooks:
            h.remove()
        return {"ppl": round(ppl, 4), "kl_vs_bf16": round(kl, 5)}

    model_eval = {"bf16": {"ppl": round(ppl_bf16, 4), "kl_vs_bf16": 0.0}}
    for name, cfg in [
        ("nvfp4_w4a16", ("nvfp4", False, False)),
        ("int4_w4a16", ("int4", False, False)),
        ("int4_rot_w4a16", ("int4_rot", False, False)),
        ("hint4_dual_w4a16", ("hint4_dual", False, False)),
        ("hint4_dual_w4a8", ("hint4_dual", False, True)),
        ("hint4_tl_w4a8", ("hint4_tl", False, True)),
        ("hint4_tl_w4a8_oproj_w8", ("hint4_tl", True, True)),
    ]:
        model_eval[name] = eval_quantized_model(*cfg)
        print(f"  {name}: {model_eval[name]}")

    all_out = {
        "gpu": torch.cuda.get_device_name(0),
        "ptx_sass_census": census,
        "qwen3_0_6b_kurtosis": summary_kurtosis,
        "qwen3_0_6b_snr_db": summary_snr,
        "synthetic_test": synth_results,
        "qwen3_0_6b_eval": model_eval,
    }
    (OUT_DIR / "exp1_numerics_and_census.json").write_text(json.dumps(all_out, indent=2) + "\n")
    print("\nSaved to exp1_numerics_and_census.json")


if __name__ == "__main__":
    main()
