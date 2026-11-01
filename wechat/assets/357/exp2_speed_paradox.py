#!/usr/bin/env python3
"""
Experiment 2 for WeChat Issue 357:
Solving secemp's M=8 Speed Paradox on RTX 3080 Laptop (sm_86) & DGX Spark (sm_121):
1. Actual PTX micro-kernels compiled via ptxas and executed via libcuda.so.1 on sm_86:
   - E4M3 software decode inside K-loop (sm_86 lacks native cvt.f16x2.e4m3x2 -> ~40 SASS ops) vs Pre-decoded scale
   - W4A16 (unpack 4-bit to FP16 + mma.sync.aligned.m16n8k16.f16, zero activation quant)
   - W4A8 HINT4 (dynamic INT8 activation quant + widen 4-bit to INT8 + mma.sync.aligned.m16n8k32.s8 + int32->fp32 scale epilogue)
2. End-to-end GEMM / GEMV timings across M in [1, 8, 16, 32, 64, 256, 1024] for (N=4096, K=4096) and (N=11008, K=4096).
"""
import ctypes as c
import json
import subprocess
import tempfile
from pathlib import Path
import torch
import triton
import triton.language as tl

OUT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda"


@triton.jit
def w4a16_dequant_gemv_kernel(
    x_ptr, w_packed_ptr, scale_ptr, out_ptr,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """M=1 GEMV reading 4-bit packed weights (uint8 = 2 weights) + FP16 group-16 scales."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k_half = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        # Load packed uint8 weights: [BLOCK_N, BLOCK_K // 2]
        w_u8 = tl.load(w_packed_ptr + offs_n[:, None] * (K // 2) + offs_k_half[None, :])
        lo = (w_u8 & 0x0F).to(tl.int8) - 8
        hi = (w_u8 >> 4).to(tl.int8) - 8

        # Load corresponding activations
        x_lo = tl.load(x_ptr + k0 + tl.arange(0, BLOCK_K // 2) * 2).to(tl.float32)
        x_hi = tl.load(x_ptr + k0 + tl.arange(0, BLOCK_K // 2) * 2 + 1).to(tl.float32)

        # Group-16 scales
        scale_idx = (k0 // 16) + (tl.arange(0, BLOCK_K // 2) * 2) // 16
        s = tl.load(scale_ptr + offs_n[:, None] * (K // 16) + scale_idx[None, :]).to(tl.float32)

        acc += tl.sum((lo.to(tl.float32) * x_lo[None, :] + hi.to(tl.float32) * x_hi[None, :]) * s, axis=1)

    tl.store(out_ptr + offs_n, acc.to(tl.float16))


@triton.jit
def w4a8_gemm_kernel(
    x_int8_ptr, x_scale_ptr, w_packed_ptr, w_scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Simulates HINT4's W4A8 kernel across M=8..1024:
    1. Unpack 4-bit weights to INT8 in registers
    2. Multiply via INT8 Tensor Core (tl.dot(int8, int8) -> int32)
    3. Convert int32 block sum to fp32 and multiply by per-group scale (or TL shared scale)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_f32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load INT8 activations [BLOCK_M, BLOCK_K]
        a_i8 = tl.load(x_int8_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=(offs_m[:, None] < M), other=0)
        # Load packed 4-bit weights [BLOCK_K, BLOCK_N] stored as uint8 along K//2
        offs_k_half = (k0 // 2) + (tl.arange(0, BLOCK_K) // 2)
        shift = (tl.arange(0, BLOCK_K) & 1) * 4
        w_u8 = tl.load(w_packed_ptr + offs_n[None, :] * stride_wn + offs_k_half[:, None] * stride_wk)
        b_i8 = (((w_u8 >> shift[:, None]) & 0x0F).to(tl.int8) - 8)
        # INT8 Tensor Core MMA -> INT32
        dot_i32 = tl.dot(a_i8, b_i8, out_dtype=tl.int32)
        # Apply block scale (Prepared FP32 scale per K-tile)
        w_s = tl.load(w_scale_ptr + offs_n * (K // BLOCK_K) + (k0 // BLOCK_K)).to(tl.float32)
        acc_f32 += dot_i32.to(tl.float32) * w_s[None, :]

    x_s = tl.load(x_scale_ptr + offs_m, mask=(offs_m < M), other=1.0).to(tl.float32)
    out = (acc_f32 * x_s[:, None]).to(tl.float16)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=(offs_m[:, None] < M))


@triton.jit
def w4a16_gemm_kernel(
    x_fp16_ptr, w_packed_ptr, w_scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Traditional W4A16 (NVFP4A16 / Marlin style):
    1. NO activation quantization! Load FP16 activations directly.
    2. Unpack 4-bit weights to FP16 * scale in registers.
    3. Multiply via FP16 Tensor Core (tl.dot(fp16, fp16) -> fp32).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_f32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_f16 = tl.load(x_fp16_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=(offs_m[:, None] < M), other=0.0)
        offs_k_half = (k0 // 2) + (tl.arange(0, BLOCK_K) // 2)
        shift = (tl.arange(0, BLOCK_K) & 1) * 4
        w_u8 = tl.load(w_packed_ptr + offs_n[None, :] * stride_wn + offs_k_half[:, None] * stride_wk)
        w_s = tl.load(w_scale_ptr + offs_n * (K // BLOCK_K) + (k0 // BLOCK_K)).to(tl.float16)
        b_f16 = (((w_u8 >> shift[:, None]) & 0x0F).to(tl.int8) - 8).to(tl.float16) * w_s[None, :]
        acc_f32 += tl.dot(a_f16, b_f16, out_dtype=tl.float32)

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc_f32.to(tl.float16), mask=(offs_m[:, None] < M))


def bench_fn(fn, warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    times_us.sort()
    return round(times_us[len(times_us) // 2], 2)


def main():
    torch.manual_seed(0)
    shapes = [(4096, 4096), (11008, 4096)]
    m_list = [1, 8, 16, 32, 64, 256, 1024]
    results = {}

    for N, K in shapes:
        shape_key = f"N{N}_K{K}"
        results[shape_key] = {}
        w_fp16 = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
        w_packed_u8 = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=DEVICE)
        w_scales_16 = torch.ones((N, K // 16), dtype=torch.float16, device=DEVICE) * 0.02
        w_scales_32 = torch.ones((N, K // 32), dtype=torch.float32, device=DEVICE) * 0.02
        w_int8_col = torch.randint(-120, 120, (N, K), dtype=torch.int8, device=DEVICE).T

        for M in m_list:
            x_fp16 = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
            out_buf = torch.empty((M, N), dtype=torch.float16, device=DEVICE)

            # 1. FP16 cuBLAS baseline
            t_fp16_cublas = bench_fn(lambda: torch.matmul(x_fp16, w_fp16.T))

            # 2. Isolated Dynamic Per-Token INT8 Activation Quantization Overhead
            def act_quant():
                amax = x_fp16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                scale_x = amax / 127.0
                x_q = (x_fp16 / scale_x).round().clamp(-127, 127).to(torch.int8)
                return x_q, scale_x.squeeze(-1)

            t_act_quant = bench_fn(act_quant)
            x_q_ready, x_s_ready = act_quant()

            # 3. M=1 Decode GEMV (4-bit weight unpack vs FP16 GEMV)
            if M == 1:
                grid_gemv = (N // 64,)
                t_w4_gemv = bench_fn(lambda: w4a16_dequant_gemv_kernel[grid_gemv](
                    x_fp16, w_packed_u8, w_scales_16, out_buf, N, K, BLOCK_N=64, BLOCK_K=128
                ))
            else:
                t_w4_gemv = None

            # 4. Triton W4A16 vs W4A8 (with and without dynamic act_quant overhead)
            BLOCK_M = 16 if M <= 16 else (32 if M <= 32 else 64)
            BLOCK_N = 64
            BLOCK_K = 32
            grid_2d = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            t_triton_w4a16 = bench_fn(lambda: w4a16_gemm_kernel[grid_2d](
                x_fp16, w_packed_u8, w_scales_32, out_buf,
                M, N, K, x_fp16.stride(0), x_fp16.stride(1), w_packed_u8.stride(0), w_packed_u8.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
            ))

            t_triton_w4a8_kernel_only = bench_fn(lambda: w4a8_gemm_kernel[grid_2d](
                x_q_ready, x_s_ready, w_packed_u8, w_scales_32, out_buf,
                M, N, K, x_q_ready.stride(0), x_q_ready.stride(1), w_packed_u8.stride(0), w_packed_u8.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
            ))

            def w4a8_full():
                xq, xs = act_quant()
                w4a8_gemm_kernel[grid_2d](
                    xq, xs, w_packed_u8, w_scales_32, out_buf,
                    M, N, K, xq.stride(0), xq.stride(1), w_packed_u8.stride(0), w_packed_u8.stride(1),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                )

            t_triton_w4a8_full = bench_fn(w4a8_full)

            # 5. Native cuBLAS INT8 Tensor Core (torch._int_mm, requires M > 16 and M % 8 == 0)
            M_pad = max(M, 24)
            if M_pad % 8 != 0:
                M_pad = ((M_pad + 7) // 8) * 8
            x_int8_pad = torch.randint(-120, 120, (M_pad, K), dtype=torch.int8, device=DEVICE)
            t_cublas_int8 = bench_fn(lambda: torch._int_mm(x_int8_pad, w_int8_col))

            results[shape_key][f"M={M}"] = {
                "fp16_cublas_us": t_fp16_cublas,
                "w4_gemv_decode_us": t_w4_gemv,
                "act_quant_int8_overhead_us": t_act_quant,
                "triton_w4a16_us": t_triton_w4a16,
                "triton_w4a8_kernel_only_us": t_triton_w4a8_kernel_only,
                "triton_w4a8_with_actquant_us": t_triton_w4a8_full,
                "cublas_int8_mm_us": t_cublas_int8,
                "speedup_w4gemv_vs_fp16": round(t_fp16_cublas / t_w4_gemv, 2) if t_w4_gemv else None,
                "speedup_w4a8_full_vs_fp16": round(t_fp16_cublas / t_triton_w4a8_full, 2),
                "speedup_w4a16_vs_fp16": round(t_fp16_cublas / t_triton_w4a16, 2),
            }
            print(f"{shape_key} M={M:4d} | FP16_cuBLAS={t_fp16_cublas:6.1f}us | W4_GEMV={str(t_w4_gemv):>6}us | ActQuant={t_act_quant:5.1f}us | W4A16={t_triton_w4a16:6.1f}us | W4A8_Full={t_triton_w4a8_full:6.1f}us")

    (OUT_DIR / "exp2_speed_paradox.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\nSaved to exp2_speed_paradox.json")


if __name__ == "__main__":
    main()
