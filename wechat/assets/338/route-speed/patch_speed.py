from pathlib import Path
import shutil,sys
root=Path(sys.argv[1]);here=Path(__file__).resolve().parent
shutil.copyfile(here/'lookahead_lora.hpp',root/'src/core/lookahead_lora.hpp')
p=root/'include/strata/core/expert_source.hpp';s=p.read_text();old='    void enable_stats(int64_t n_layers);';assert old in s;s=s.replace(old,old+'\n    std::string lora_report() const;');p.write_text(s)
p=root/'src/core/expert_source.cpp';s=p.read_text();s='#include "lookahead_lora.hpp"\n'+s
old='    routers_ = std::move(routers);';assert s.count(old)==1;s=s.replace(old,'    if (!lookahead_lora().load(n_embd,n_expert,(int64_t)routers.size(),err)) return false;\n'+old)
old='        const int kmax = stats_on_ ?';assert s.count(old)==1
s=s.replace(old,'''        if (dj == 0) lookahead_lora().apply(layer, x_.data(), (int)nt, logits.data());
        static const bool uncached = [] { const char* p=std::getenv("STRATA_LOOKAHEAD_UNCACHED");return p && std::atoi(p)!=0; }();
        if (uncached && host_res)
            for (int64_t t=0;t<nt;++t) for (int64_t e=0;e<n_expert_;++e)
                if (host_res[layer*n_expert_+e]>=0) logits[t*n_expert_+e]=-std::numeric_limits<float>::infinity();
'''+old)
s+='\nstd::string strata::core::RouterLookahead::lora_report() const { return strata::core::lookahead_lora().report(); }\n';p.write_text(s)
p=root/'src/core/verify.cpp';s=p.read_text();old='    g_diag_verifier.store(this);';assert s.count(old)==1;s=s.replace(old,'    if (const char* p=std::getenv("STRATA_LOOKAHEAD_FRESH_INPUT"); p && std::atoi(p)!=0) always_publish_=true;\n'+old);p.write_text(s)
p=root/'src/program/generate.cpp';s=p.read_text();old='            if (drive.d.fs) std::fprintf(stderr, "strata serve: %s\\n", drive.d.fs->report().c_str());';assert s.count(old)==1
s=s.replace(old,old+'''\n            if (drive.d.lookahead) std::fprintf(stderr,"strata lookahead totals: predicted_experts=%lld skipped=%lld busy_ms=%.3f %s\\n",(long long)lookahead.predicted(),(long long)lookahead.skipped(),lookahead.busy_ms(),lookahead.lora_report().c_str());''');p.write_text(s)
print('speed integration patched')
