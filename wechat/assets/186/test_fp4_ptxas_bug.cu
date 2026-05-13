// RTX 5090 FP4 ptxas codegen bug 验证探针（sm_121 版）
// 基于王多鱼-加油的逆向分析，在 DGX Spark sm_121 上复现
//
// 用 CUDA driver API 加载手写 PTX（ptxas 才认 kind::f8f6f4）

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA RT error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_DRV(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* msg; cuGetErrorString(err, &msg); \
        printf("CUDA DRV error at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

// Generate PTX for FP4 (e2m1) probe kernel
// Each thread: fills A with fill_a, B with fill_b, runs mma, writes 4 floats
static std::string make_fp4_ptx(const char* kernel_name, const char* target) {
    char buf[4096];
    snprintf(buf, sizeof(buf), R"(
.version 9.2
.target %s
.address_size 64

.visible .entry %s(
    .param .u64 param_out,
    .param .u32 param_fill_a,
    .param .u32 param_fill_b
)
{
    .reg .u64 out_ptr, addr;
    .reg .u32 tid, fill_a, fill_b, off;
    .reg .b32 a<4>, b<2>;
    .reg .f32 d<4>;
    .reg .pred p;

    ld.param.u64 out_ptr, [param_out];
    ld.param.u32 fill_a, [param_fill_a];
    ld.param.u32 fill_b, [param_fill_b];

    mov.b32 a0, fill_a;
    mov.b32 a1, fill_a;
    mov.b32 a2, fill_a;
    mov.b32 a3, fill_a;
    mov.b32 b0, fill_b;
    mov.b32 b1, fill_b;
    mov.f32 d0, 0f00000000;
    mov.f32 d1, 0f00000000;
    mov.f32 d2, 0f00000000;
    mov.f32 d3, 0f00000000;

    mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32
        {d0, d1, d2, d3},
        {a0, a1, a2, a3},
        {b0, b1},
        {d0, d1, d2, d3};

    mov.u32 tid, %%tid.x;
    setp.lt.u32 p, tid, 32;
    @!p bra DONE;

    // out[tid*4 + 0..3]
    shl.b32 off, tid, 4;  // tid * 16 bytes (4 floats)
    cvt.u64.u32 addr, off;
    add.u64 addr, out_ptr, addr;
    st.global.f32 [addr],    d0;
    st.global.f32 [addr+4],  d1;
    st.global.f32 [addr+8],  d2;
    st.global.f32 [addr+12], d3;

DONE:
    ret;
}
)", target, kernel_name);
    return std::string(buf);
}

// Same but for FP8 (e4m3) — used as control group
static std::string make_fp8_ptx(const char* kernel_name, const char* target) {
    char buf[4096];
    snprintf(buf, sizeof(buf), R"(
.version 9.2
.target %s
.address_size 64

.visible .entry %s(
    .param .u64 param_out,
    .param .u32 param_fill_a,
    .param .u32 param_fill_b
)
{
    .reg .u64 out_ptr, addr;
    .reg .u32 tid, fill_a, fill_b, off;
    .reg .b32 a<4>, b<2>;
    .reg .f32 d<4>;
    .reg .pred p;

    ld.param.u64 out_ptr, [param_out];
    ld.param.u32 fill_a, [param_fill_a];
    ld.param.u32 fill_b, [param_fill_b];

    mov.b32 a0, fill_a;
    mov.b32 a1, fill_a;
    mov.b32 a2, fill_a;
    mov.b32 a3, fill_a;
    mov.b32 b0, fill_b;
    mov.b32 b1, fill_b;
    mov.f32 d0, 0f00000000;
    mov.f32 d1, 0f00000000;
    mov.f32 d2, 0f00000000;
    mov.f32 d3, 0f00000000;

    mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32
        {d0, d1, d2, d3},
        {a0, a1, a2, a3},
        {b0, b1},
        {d0, d1, d2, d3};

    mov.u32 tid, %%tid.x;
    setp.lt.u32 p, tid, 32;
    @!p bra DONE;

    shl.b32 off, tid, 4;
    cvt.u64.u32 addr, off;
    add.u64 addr, out_ptr, addr;
    st.global.f32 [addr],    d0;
    st.global.f32 [addr+4],  d1;
    st.global.f32 [addr+8],  d2;
    st.global.f32 [addr+12], d3;

DONE:
    ret;
}
)", target, kernel_name);
    return std::string(buf);
}

// FP4 probe with per-register A fill (for V25 nibble injection)
static std::string make_fp4_per_reg_ptx(const char* kernel_name, const char* target) {
    char buf[4096];
    snprintf(buf, sizeof(buf), R"(
.version 9.2
.target %s
.address_size 64

.visible .entry %s(
    .param .u64 param_out,
    .param .u32 param_a0,
    .param .u32 param_a1,
    .param .u32 param_a2,
    .param .u32 param_a3,
    .param .u32 param_fill_b
)
{
    .reg .u64 out_ptr, addr;
    .reg .u32 tid, off;
    .reg .b32 a<4>, b<2>;
    .reg .f32 d<4>;
    .reg .pred p;

    ld.param.u64 out_ptr, [param_out];
    ld.param.u32 a0, [param_a0];
    ld.param.u32 a1, [param_a1];
    ld.param.u32 a2, [param_a2];
    ld.param.u32 a3, [param_a3];
    .reg .u32 fb;
    ld.param.u32 fb, [param_fill_b];
    mov.b32 b0, fb;
    mov.b32 b1, fb;
    mov.f32 d0, 0f00000000;
    mov.f32 d1, 0f00000000;
    mov.f32 d2, 0f00000000;
    mov.f32 d3, 0f00000000;

    mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32
        {d0, d1, d2, d3},
        {a0, a1, a2, a3},
        {b0, b1},
        {d0, d1, d2, d3};

    mov.u32 tid, %%tid.x;
    setp.lt.u32 p, tid, 32;
    @!p bra DONE;

    shl.b32 off, tid, 4;
    cvt.u64.u32 addr, off;
    add.u64 addr, out_ptr, addr;
    st.global.f32 [addr],    d0;
    st.global.f32 [addr+4],  d1;
    st.global.f32 [addr+8],  d2;
    st.global.f32 [addr+12], d3;

DONE:
    ret;
}
)", target, kernel_name);
    return std::string(buf);
}

static CUfunction load_ptx_kernel(const std::string& ptx, const char* kernel_name) {
    CUmodule mod;
    CUfunction func;
    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER };
    char error_log[4096] = {0};
    size_t log_size = sizeof(error_log);
    void* option_vals[] = { (void*)log_size, (void*)error_log };

    CUresult err = cuModuleLoadDataEx(&mod, ptx.c_str(), 2, options, option_vals);
    if (err != CUDA_SUCCESS) {
        printf("PTX load error: %s\n", error_log);
        const char* msg; cuGetErrorString(err, &msg);
        printf("Error: %s\n", msg);
        exit(1);
    }
    CHECK_DRV(cuModuleGetFunction(&func, mod, kernel_name));
    return func;
}

float e2m1_decode(uint8_t nibble) {
    nibble &= 0xF;
    int s = (nibble >> 3) & 1;
    int e = (nibble >> 1) & 3;
    int m = nibble & 1;
    float val;
    if (e == 0) {
        val = m * 0.5f;
    } else {
        val = (1.0f + m * 0.5f) * powf(2.0f, (float)(e - 1));
    }
    if (s) val = -val;
    return val;
}

int main() {
    CHECK_DRV(cuInit(0));

    float *d_out, h_out[128];
    CHECK_CUDA(cudaMalloc(&d_out, 128 * sizeof(float)));

    // Detect actual GPU
    int cc_major, cc_minor;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);

    // sm_120a for compilation (PTX target), runs on sm_121 hardware
    const char* target = "sm_120a";

    printf("=== SM%d%d FP4 ptxas codegen bug probe ===\n", cc_major, cc_minor);
    printf("Based on 王多鱼-加油 (RTX 5090 sm_120a, ptxas 13.1.115)\n");
    printf("PTX target: %s, actual GPU: sm_%d%d\n\n", target, cc_major, cc_minor);

    // Load kernels
    std::string fp4_ptx = make_fp4_ptx("probe_fp4", target);
    std::string fp8_ptx = make_fp8_ptx("probe_fp8", target);
    std::string fp4_reg_ptx = make_fp4_per_reg_ptx("probe_fp4_reg", target);

    CUfunction fp4_func = load_ptx_kernel(fp4_ptx, "probe_fp4");
    CUfunction fp8_func = load_ptx_kernel(fp8_ptx, "probe_fp8");
    CUfunction fp4_reg_func = load_ptx_kernel(fp4_reg_ptx, "probe_fp4_reg");

    // ============================================================
    printf("=== V24: FP4 (e2m1) vs FP8 (e4m3) ===\n\n");

    // FP4: A=0x22222222 (all 1.0), B=0x22222222 (all 1.0)
    {
        CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(float)));
        CUdeviceptr dptr = (CUdeviceptr)d_out;
        uint32_t fill_a = 0x22222222, fill_b = 0x22222222;
        void* args[] = { &dptr, &fill_a, &fill_b };
        CHECK_DRV(cuLaunchKernel(fp4_func, 1,1,1, 32,1,1, 0, 0, args, NULL));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, 128*sizeof(float), cudaMemcpyDeviceToHost));

        printf("FP4 (e2m1): A=all 0x22 (=1.0 per nibble), B=all 0x22\n");
        printf("  Expected (standard e2m1): 32.0 (K=32, each 1.0*1.0)\n");
        printf("  Expected (if 王多鱼 bug):  2.0\n");
        printf("  Thread 0: [%.4f, %.4f, %.4f, %.4f]\n", h_out[0], h_out[1], h_out[2], h_out[3]);
        printf("  All t d[0]:");
        for (int t = 0; t < 32; t++) printf(" %.1f", h_out[t*4]);
        printf("\n");
        int nz = 0;
        for (int i = 0; i < 128; i++) if (h_out[i] != 0) nz++;
        printf("  Non-zero: %d/128\n\n", nz);
    }

    // FP8: A=0x38383838 (all 1.0), B=0x38383838
    {
        CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(float)));
        CUdeviceptr dptr = (CUdeviceptr)d_out;
        uint32_t fill_a = 0x38383838, fill_b = 0x38383838;
        void* args[] = { &dptr, &fill_a, &fill_b };
        CHECK_DRV(cuLaunchKernel(fp8_func, 1,1,1, 32,1,1, 0, 0, args, NULL));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, 128*sizeof(float), cudaMemcpyDeviceToHost));

        printf("FP8 (e4m3): A=all 0x38 (=1.0 per byte), B=all 0x38\n");
        printf("  Expected: 32.0 (K=32 for m16n8k32, each 1.0*1.0)\n");
        printf("  Thread 0: [%.4f, %.4f, %.4f, %.4f]\n", h_out[0], h_out[1], h_out[2], h_out[3]);
        printf("  All t d[0]:");
        for (int t = 0; t < 32; t++) printf(" %.1f", h_out[t*4]);
        printf("\n");
        int nz = 0;
        for (int i = 0; i < 128; i++) if (h_out[i] != 0) nz++;
        printf("  Non-zero: %d/128\n\n", nz);
    }

    // ============================================================
    printf("=== V25: Per-nibble injection ===\n");
    printf("Inject 0x2 (e2m1=1.0) at one nibble in A, rest=0, B=all 0x22\n");
    printf("王多鱼: only even nibbles active, odd silent\n\n");

    int active = 0, silent = 0;
    for (int nib = 0; nib < 32; nib++) {
        CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(float)));
        uint32_t a_vals[4] = {0, 0, 0, 0};
        int reg = nib / 8;
        int shift = (nib % 8) * 4;
        a_vals[reg] = (uint32_t)0x2 << shift;

        CUdeviceptr dptr = (CUdeviceptr)d_out;
        uint32_t fill_b = 0x22222222;
        void* args[] = { &dptr, &a_vals[0], &a_vals[1], &a_vals[2], &a_vals[3], &fill_b };
        CHECK_DRV(cuLaunchKernel(fp4_reg_func, 1,1,1, 32,1,1, 0, 0, args, NULL));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, 128*sizeof(float), cudaMemcpyDeviceToHost));

        float total = 0;
        for (int i = 0; i < 128; i++) total += fabsf(h_out[i]);
        int is_active = (total > 0);
        if (is_active) active++; else silent++;

        printf("  nib[%2d] a[%d]<<%2d: sum=%.4f %s %s\n",
               nib, reg, shift, total,
               is_active ? "ACTIVE" : "SILENT",
               ((nib%2==0) == is_active) ? "" : "*** UNEXPECTED ***");
    }
    printf("\n  %d ACTIVE, %d SILENT\n\n", active, silent);

    // ============================================================
    printf("=== V26: Byte scan (low nibble 0x00..0x0F) ===\n");
    printf("A=all same byte, B=all 0x22\n\n");
    printf("byte | d[0]     | e2m1 lo  | e2m1 hi\n");
    printf("-----|----------|----------|--------\n");

    for (int bv = 0; bv < 16; bv++) {
        CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(float)));
        uint32_t fill_byte = bv | (bv << 8) | (bv << 16) | (bv << 24);
        CUdeviceptr dptr = (CUdeviceptr)d_out;
        uint32_t fill_b = 0x22222222;
        void* args[] = { &dptr, &fill_byte, &fill_b };
        CHECK_DRV(cuLaunchKernel(fp4_func, 1,1,1, 32,1,1, 0, 0, args, NULL));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, 128*sizeof(float), cudaMemcpyDeviceToHost));

        printf("0x%02X | %8.4f | %8.4f | %8.4f\n",
               bv, h_out[0], e2m1_decode(bv & 0xF), e2m1_decode((bv>>4)&0xF));
    }

    printf("\n0x10 boundary scan:\n");
    for (int bv = 0x0E; bv <= 0x14; bv++) {
        CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(float)));
        uint32_t fill_byte = bv | (bv << 8) | (bv << 16) | (bv << 24);
        CUdeviceptr dptr = (CUdeviceptr)d_out;
        uint32_t fill_b = 0x22222222;
        void* args[] = { &dptr, &fill_byte, &fill_b };
        CHECK_DRV(cuLaunchKernel(fp4_func, 1,1,1, 32,1,1, 0, 0, args, NULL));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, 128*sizeof(float), cudaMemcpyDeviceToHost));

        printf("0x%02X: t0=[%.4f, %.4f, %.4f, %.4f]%s\n",
               bv, h_out[0], h_out[1], h_out[2], h_out[3],
               bv == 0x10 ? "  <-- boundary" : "");
    }

    CHECK_CUDA(cudaFree(d_out));
    printf("\nDone.\n");
    return 0;
}
