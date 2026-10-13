from pathlib import Path
import sys
p=Path(sys.argv[1])/'src/program/generate.cpp';s=p.read_text()
old='''                    if (!ver.commit(T, e) || (use_mtp && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e))) return false;
                    q += T;'''
new='''                    if (!ver.commit(T, e) || (use_mtp && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e))) return false;
                    // run() already synchronized the expert reads. Advance swap reuse/budget for compatibility prefill too.
                    if (drive.d.fs) drive.d.fs->completed.fetch_add(1, std::memory_order_release);
                    q += T;'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
