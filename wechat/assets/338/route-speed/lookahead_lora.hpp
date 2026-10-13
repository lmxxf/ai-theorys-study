#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "strata/kernels/cpu/kq_avx2.hpp"
namespace strata::core {
class LookaheadLora {
    struct Layer { std::vector<uint16_t> a, b; };
    std::vector<Layer> layers_;
    bool loaded_ = false;
    int active_ = 0;
    std::atomic<uint64_t> calls_{0}, micros_{0};
public:
    bool load(int64_t n, int64_t e, int64_t layers, std::string& err) {
        if (loaded_) return true;
        loaded_ = true;
        const char* path = std::getenv("STRATA_LOOKAHEAD_LORA");
        if (!path || !*path) return true;
        std::ifstream f(path, std::ios::binary);
        char magic[8]; uint32_t h[5];
        if (!f.read(magic, 8) || std::memcmp(magic,"STRALR01",8) || !f.read((char*)h,sizeof h) ||
            h[0] != layers || h[1] != n || h[2] != e || h[3] != 16 || n != 2560 || e != 512) {
            err="lookahead LoRA: invalid header"; return false;
        }
        layers_.resize((size_t)layers);
        for (int64_t l=0;l<layers;++l) {
            uint32_t on=0;
            if (!f.read((char*)&on,4) || on>1) {err="lookahead LoRA: invalid layer flag";return false;}
            if (!on) continue;
            Layer& v=layers_[(size_t)l]; v.a.resize(16*n);v.b.resize(e*16);
            if (!f.read((char*)v.a.data(),v.a.size()*2) || !f.read((char*)v.b.data(),v.b.size()*2)) {
                err="lookahead LoRA: truncated weights";return false;
            }
            ++active_;
        }
        if (active_!=(int)h[4] || f.peek()!=std::char_traits<char>::eof()) {err="lookahead LoRA: invalid length/count";return false;}
        std::fprintf(stderr,"lookahead LoRA: %d active next-layer corrections, rank 16, BF16 CPU AVX2\n",active_);
        // Optional numerical gate before the timed experiment; input/output are private fixture files.
        if (const char* fixture=std::getenv("STRATA_LOOKAHEAD_LORA_FIXTURE")) {
            std::ifstream in(fixture,std::ios::binary); uint32_t nt=0;
            if (!in.read((char*)&nt,4) || nt<1 || nt>8) {err="LoRA fixture header";return false;}
            std::ofstream out(std::string(fixture)+".out",std::ios::binary);
            std::vector<float> x(nt*2560),z(nt*512);
            for (int64_t l=0;l<layers;++l) {
                if (!in.read((char*)x.data(),x.size()*4)) {err="LoRA fixture truncated";return false;}
                std::fill(z.begin(),z.end(),0.f);apply(l,x.data(),nt,z.data());out.write((char*)z.data(),z.size()*4);
            }
            if (!out) {err="LoRA fixture write failed";return false;}
            calls_=0;micros_=0;
        }
        return true;
    }
    void apply(int64_t target, const float* x, int nt, float* logits) {
        if (target<0 || target>=(int64_t)layers_.size() || layers_[(size_t)target].a.empty()) return;
        const auto t0=std::chrono::steady_clock::now();
        const Layer& v=layers_[(size_t)target];float hidden[8*16],delta[8*512];
        strata::kernels::cpu::bf16_rows_dot_multi(v.a.data(),16,2560,x,nt,hidden);
        strata::kernels::cpu::bf16_rows_dot_multi(v.b.data(),512,16,hidden,nt,delta);
        for (int i=0;i<nt*512;++i) logits[i]+=delta[i];
        calls_.fetch_add(1,std::memory_order_relaxed);
        micros_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0).count(),std::memory_order_relaxed);
    }
    std::string report() const {
        return "LoRA active="+std::to_string(active_)+" calls="+std::to_string(calls_.load())+" us="+std::to_string(micros_.load());
    }
};
inline LookaheadLora& lookahead_lora() { static LookaheadLora value; return value; }
}
