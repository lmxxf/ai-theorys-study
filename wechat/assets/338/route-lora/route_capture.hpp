// Offline research capture only. Disabled unless STRATA_CAPTURE_ROUTES is set.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
namespace strata_route_capture {
class Writer {
    std::FILE* f_ = nullptr;
    uint64_t window_ = 0, kept_ = 0;
    int stride_ = 4, cap_ = 2048, expected_ = 0, rows_ = 0;
    bool take_ = false;
    void put(const void* p, size_t bytes) {
        if (std::fwrite(p, 1, bytes, f_) != bytes) {
            std::fprintf(stderr, "route capture: write failed\n"); std::abort();
        }
    }
public:
    Writer() {
        const char* path = std::getenv("STRATA_CAPTURE_ROUTES");
        if (!path || !*path) return;
        if (const char* s = std::getenv("STRATA_CAPTURE_STRIDE")) stride_ = std::max(1, std::atoi(s));
        if (const char* s = std::getenv("STRATA_CAPTURE_MAX_WINDOWS")) cap_ = std::max(1, std::atoi(s));
        f_ = std::fopen(path, "wb");
        if (!f_) { std::fprintf(stderr, "route capture: cannot open output\n"); std::abort(); }
        std::setvbuf(f_, nullptr, _IOFBF, 4 << 20);
        const char magic[8] = {'S','T','R','A','R','T','E','1'};
        const uint32_t header[6] = {1, 48, 2560, 512, 10, (uint32_t)stride_};
        put(magic, 8); put(header, sizeof header); std::fflush(f_);
        std::fprintf(stderr, "route capture: f32 input + actual top10 + resident mask; stride=%d max_windows=%d\n", stride_, cap_);
    }
    ~Writer() { if (f_) std::fclose(f_); }
    void record(int layer, int rows, int k, const float* x, const int32_t* ids, const int32_t* resident) {
        if (!f_) return;
        if (layer == 0) {
            if (expected_ != 0) { std::fprintf(stderr, "route capture: incomplete window\n"); std::abort(); }
            ++window_; rows_ = rows;
            take_ = ((window_ - 1) % stride_ == 0) && kept_ < (uint64_t)cap_;
            if (take_) ++kept_;
        }
        if (layer != expected_ || rows != rows_ || rows < 1 || rows > 8 || k != 10) {
            std::fprintf(stderr, "route capture: layer/window shape mismatch %d/%d rows=%d/%d k=%d\n", layer, expected_, rows, rows_, k);
            std::abort();
        }
        expected_ = (layer + 1) % 48;
        if (!take_) return;
        const uint32_t rec[4] = {(uint32_t)layer, (uint32_t)rows, 2560, (uint32_t)k};
        put(&window_, sizeof window_); put(rec, sizeof rec);
        put(x, (size_t)rows * 2560 * sizeof(float));
        put(ids, (size_t)rows * k * sizeof(int32_t));
        uint8_t mask[512];
        for (int i = 0; i < 512; ++i) mask[i] = resident && resident[layer * 512 + i] >= 0 ? 1 : 0;
        put(mask, sizeof mask);
        if (layer == 47) std::fflush(f_);
    }
};
inline void record(int l, int n, int k, const float* x, const int32_t* ids, const int32_t* resident) {
    static Writer writer;
    writer.record(l, n, k, x, ids, resident);
}
}
