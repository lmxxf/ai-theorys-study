from pathlib import Path
import sys,shutil
root=Path(sys.argv[1]); here=Path(__file__).resolve().parent
shutil.copyfile(here/'route_capture.hpp',root/'src/program/route_capture.hpp')
p=root/'src/program/generate.cpp';s=p.read_text()
if '#include "route_capture.hpp"' not in s:
 s='#include "route_capture.hpp"\n'+s
 old='    t->d.layers = layer;\n    const Clock::time_point a = Clock::now();'
 new='    strata_route_capture::record((int)layer, (int)n_tok, (int)k, x_f, ids, t->d.host_res);\n'+old
 assert s.count(old)==1;s=s.replace(old,new)
 p.write_text(s)
p=root/'src/core/verify.cpp';s=p.read_text()
old='    g_diag_verifier.store(this);'
new='''    // Research capture: every layer must publish fresh host inputs, even on a full cache hit.
    if (const char* cap = std::getenv("STRATA_CAPTURE_ROUTES"); cap && *cap) {
        if (g.n_embd != 2560 || g.n_expert != 512 || g.n_layers != 48) {
            err = "route capture supports only 48x512/2560 model geometry"; return false;
        }
        always_publish_ = true;
    }
'''+old
if 'route capture supports only' not in s:
 assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
print('capture patch applied',root)
