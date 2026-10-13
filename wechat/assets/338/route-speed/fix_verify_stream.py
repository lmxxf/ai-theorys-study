from pathlib import Path
import sys
p=Path(sys.argv[1])/'src/core/expert_source.cpp';s=p.read_text()
old='''            const bool same = cudaMemcpy(back.data(), (const void*) sl.dptr, (size_t) sl.bytes, cudaMemcpyDeviceToHost) ==
                                  cudaSuccess && std::memcmp(back.data(), sl.src, (size_t) sl.bytes) == 0;'''
new='''            // The main graph is waiting for this host callback: never synchronize the legacy default stream here.
            cudaStream_t check_stream = (cudaStream_t)card_stream[(size_t)layer_card[(size_t)layer]];
            const bool same = cudaMemcpyAsync(back.data(), (const void*)sl.dptr, (size_t)sl.bytes,
                                               cudaMemcpyDeviceToHost, check_stream) == cudaSuccess &&
                              cudaStreamSynchronize(check_stream) == cudaSuccess &&
                              std::memcmp(back.data(), sl.src, (size_t)sl.bytes) == 0;'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
