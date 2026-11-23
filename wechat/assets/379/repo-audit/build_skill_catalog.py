"""List every tracked SKILL.md without loading or executing repository instructions."""
from pathlib import Path
import json, subprocess
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
repos=json.loads((HERE/'inventory_summary.json').read_text())['repositories']
out=['# 八仓 SKILL.md 完整路径目录','', '本表列出当前快照全部 SKILL.md 路径；镜像、覆盖版、内置/参考技能均保留，不宣称每条都是独立或默认启用的能力。文件行数包括 frontmatter 与空行。','']
rows=[]
for repo in repos:
 name=repo['repo']; root=BASE/name
 paths=subprocess.check_output(['git','ls-files','-z'],cwd=root,text=True).split('\0')
 skills=sorted(p for p in paths if Path(p).name.lower()=='skill.md')
 out += [f'## {name}：{len(skills)} 条','', '| SKILL 路径 | 行数 |','|---|---:|']
 for p in skills:
  lines=len((root/p).read_text().splitlines())
  out.append(f'| [{p}](../{name}/{p}) | {lines:,} |')
  rows.append({'repo':name,'path':p,'lines':lines})
 out.append('')
(HERE/'skill_catalog.md').write_text('\n'.join(out)+'\n')
(HERE/'skill_catalog.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(f'{len(rows)} SKILL paths')
