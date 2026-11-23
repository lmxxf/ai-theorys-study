"""Verify audit artifacts and unchanged, ignored source clones; no upstream code execution."""
from pathlib import Path
import subprocess, json, re, hashlib, gzip, csv, datetime
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
MAIN=BASE/'八大科研Skill仓库三分类深度解剖报告.md'
snapshots=json.loads((HERE/'inventory_summary.json').read_text())['repositories']
failures=[]; sources=[]
for repo in snapshots:
 name=repo['repo'];root=BASE/name
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
 ignored=subprocess.run(['git','check-ignore','-q',str(root)],cwd=BASE,check=False).returncode==0
 report=HERE/(name+'.md')
 item=dict(repo=name,unchanged_head=head==repo['head'],clean=not status,ignored=ignored,report_exists=report.exists())
 sources.append(item)
 if not all(v for k,v in item.items() if k!='repo'):failures.append(name)
for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',MAIN.read_text()):
 if target=='repo-audit/verification.json':continue
 if '://' not in target and not (MAIN.parent/target).exists():failures.append('missing link: '+target)
with gzip.open(HERE/'inventory_files.tsv.gz','rt',encoding='utf-8') as stream:
 manifest=list(csv.DictReader(stream,delimiter='\t'))
expected=sum(r['counts']['tracked_regular_files'] for r in snapshots)
if len(manifest)!=expected:failures.append('manifest row count')
catalog=json.loads((HERE/'skill_catalog.json').read_text())
if len(catalog)!=sum(r['counts']['skill_md_files'] for r in snapshots):failures.append('catalog count')
artifacts=[dict(path=str(p.relative_to(BASE)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='verification.json']
result=dict(checked_at=datetime.datetime.now().isoformat(),scope='Static artifact and clone-state checks only; no upstream tests, API or runtime validation',sources=sources,manifest_rows=len(manifest),skill_paths=len(catalog),artifacts=artifacts,main_bytes=MAIN.stat().st_size,failures=failures)
(HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(sources=sources,manifest_rows=len(manifest),skill_paths=len(catalog),failures=failures),ensure_ascii=False,indent=2))
raise SystemExit(bool(failures))
