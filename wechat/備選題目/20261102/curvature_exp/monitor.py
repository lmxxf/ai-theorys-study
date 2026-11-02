import time,json
from pathlib import Path
pid=2157833
with open("/home/lmxxf/work/interp-curvature-exp/resource_monitor.jsonl","w") as out:
 while Path(f"/proc/{pid}").exists():
  m={s.split(":")[0]:s.split(":")[1].strip() for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith(("MemAvailable:","SwapFree:"))}
  m["time"]=time.time()
  out.write(json.dumps(m)+"\n");out.flush();time.sleep(1)
