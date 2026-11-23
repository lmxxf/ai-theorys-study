"""Read-only reproducible inventory of tracked files; no repository code is imported."""
from pathlib import Path
import subprocess,json,collections,hashlib,csv,datetime,gzip
BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
NAMES=['scientific-agent-skills','academic-research-skills','nature-skills','Auto-claude-code-research-in-sleep','HERO-Anti-OverDefense','Anti-Autoresearch','AutoResearchClaw','AI-Research-SKILLs']
CODE={'.py','.sh','.bash','.js','.mjs','.cjs','.ts','.tsx','.jsx','.go','.rs','.c','.h','.cpp','.java','.r','.R','.jl','.ps1','.bat','.sql'}
rows=[];summaries=[]
for name in NAMES:
 root=BASE/name;files=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0');files=[x for x in files if x];counts=collections.Counter();exts=collections.defaultdict(lambda:dict(files=0,lines=0,nonblank=0));totaltext=0;links=[]
 for f in files:
  p=root/f
  if p.is_symlink():links.append(dict(path=f,target=str(p.readlink())));continue
  if not p.is_file():continue
  data=p.read_bytes();size=len(data);ext=p.suffix;parts=p.parts;low=f.lower();iscode=ext in CODE
  try:text=data.decode('utf-8');binary='\x00'in text
  except UnicodeDecodeError:text='';binary=True
  lines=len(text.splitlines())if not binary else 0;nonblank=sum(bool(x.strip())for x in text.splitlines())if not binary else 0
  if not binary:totaltext+=lines
  if any(x in low.split('/')for x in ['node_modules','vendor','external','third_party','third-party','dist','build']):role='vendor_or_external'
  elif any(x in low.split('/')for x in ['tests','test','__tests__'])or p.name.startswith('test_')or p.name.endswith('.test.ts'):role='test'
  elif any(x in low.split('/')for x in ['examples','example','demo','demos','experiments']):role='example_or_experiment'
  elif any(x in low.split('/')for x in ['docs','website','frontend-legacy']):role='docs_or_web'
  else:role='candidate_runtime'
  counts['tracked_regular_files']+=1;counts['bytes']+=size
  if p.name.lower()=='skill.md':counts['skill_md_files']+=1
  if ext.lower()=='.md':counts['markdown_files']+=1;counts['markdown_lines']+=lines
  if iscode and not binary:
   exts[ext]['files']+=1;exts[ext]['lines']+=lines;exts[ext]['nonblank']+=nonblank;counts['code_files']+=1;counts['code_lines']+=lines;counts['code_nonblank']+=nonblank;counts['code_'+role+'_files']+=1;counts['code_'+role+'_lines']+=lines
  rows.append(dict(repo=name,path=f,bytes=size,extension=ext,text_lines=lines,nonblank_lines=nonblank,code_extension=iscode,heuristic_code_role=role if iscode else'',sha256=hashlib.sha256(data).hexdigest()))
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
 summaries.append(dict(repo=name,head=head,tracked_path_count=len(files),counts=dict(counts),text_lines=totaltext,code_extensions=dict(exts),symlinks=links,worktree_status=dirty,role_note='Directory heuristics only; candidate_runtime is not a claim of usable research automation. Generated/minified/front-end distinctions require manual audit. Lines count text splitlines, includes comments/docstrings; nonblank excludes empty lines, not comments. Markdown fences never count as executable code.'))
(OUT/'inventory_summary.json').write_text(json.dumps(dict(generated_at=datetime.datetime.now().isoformat(),repositories=summaries),ensure_ascii=False,indent=2))
with gzip.open(OUT/'inventory_files.tsv.gz','wt',encoding='utf-8',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
print(json.dumps([dict(repo=x['repo'],head=x['head'][:12],**x['counts'])for x in summaries],ensure_ascii=False,indent=2))
