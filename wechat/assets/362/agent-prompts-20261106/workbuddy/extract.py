#!/usr/bin/env python3
"""Read-only extraction; offsets are zero-based UTF-8 bytes, ranges [start,end)."""
from pathlib import Path
import json,hashlib,shutil,struct
ROOT=Path('/mnt/c/Users/lmxxf/AppData/Local/Programs/WorkBuddyAI/resources')
SRC=ROOT/'app.asar.unpacked'; OUT=Path(__file__).resolve().parent
manifest={'offset_unit':'zero-based UTF-8 bytes; end exclusive','sources':{},'prompts':[],'fragments':[]}
def sha(b):return hashlib.sha256(b).hexdigest()
for name in ['cli/product.json','cli/dist/codebuddy-lite-wb.mjs','node_modules/@genie/agent-harness/lib/session/compact.js']:
 p=SRC/name;b=p.read_bytes();manifest['sources'][name]={'path':str(p),'size':len(b),'sha256':sha(b)}
p=SRC/'cli/product.json';b=p.read_bytes();d=json.loads(b)
for i,t in enumerate(d['prompts']):
 name=t['name']
 if any(x in name for x in ['compact','summary','skill-description','skillmanage']):
  text=t['template'];o=OUT/'prompts'/f'{name}.txt';o.parent.mkdir(exist_ok=True);o.write_bytes(text.encode())
  marker=json.dumps(name).encode();pos=b.find(marker);ts=b.find(b'"template"',pos);start=b.find(b'"',ts+len(b'"template"'))
  j=start+1
  while j<len(b):
   if b[j]==92:j+=2;continue
   if b[j]==34:break
   j+=1
  assert json.loads(b[start:j+1])==text
  manifest['prompts'].append({'name':name,'json_pointer':f'/prompts/{i}/template','name_offset':pos,'string_start':start,'string_end':j+1,'length_decoded_bytes':len(text.encode()),'sha256_decoded':sha(text.encode()),'output':str(o.relative_to(OUT))})
(OUT/'product-selection.json').write_text(json.dumps({k:d.get(k)for k in ['commit','date','genieVersion','tokenUsageThresholds','productFeatures','commands','agents','disabledBuiltinSkills']},ensure_ascii=False,indent=2))
# Archive exact source slices, not beautified reconstructions.
p=SRC/'cli/dist/codebuddy-lite-wb.mjs';b=p.read_bytes()
sections=[('thresholds','let tI="compact-prompt"','function isPositiveSafeInteger'),('runner-shouldCompact','async shouldCompact(t,i)','isContextWindowOverflowRisk(t,i,n,s)'),('pre-message-trigger','let n=await this.agentManager.getDefault(),s=await ModelUtils.resolveModelForTokenLimits(this.agentManager,i,t.agent.name||n.name)','var context_compact_summarizer_decorate'),('summary-router','async compactWithResultOwned(t,i)','let aP=new Set'),('engineering-compactor','let aP=new Set','let ax='),('strategy-base','let CompactStrategyBase=class','var pre_message_compact_strategy_decorate'),('pre-message-strategy','let aD=class extends CompactStrategyBase','var max_token_compact_strategy_decorate'),('max-token-strategy','let aN=eo=class extends CompactStrategyBase','var blocking_compact_strategy_decorate'),('blocking-strategy','let aL=class extends CompactStrategyBase','let a$=Symbol')]
for name,left,right in sections:
 start=b.find(left.encode());end=b.find(right.encode(),start+len(left)) if start>=0 else -1
 if start<0 or end<0:
  manifest['fragments'].append({'name':name,'error':'anchor missing','start':start,'end':end});continue
 raw=b[start:end];dest=OUT/f'{name}.js';dest.write_bytes(raw)
 manifest['fragments'].append({'name':name,'source':str(p),'start':start,'end':end,'length':len(raw),'sha256':sha(raw),'output':dest.name})
skill=SRC/'resources/plugins/workbuddy-builtin/skills/skill-creator';shutil.copytree(skill,OUT/'skill-creator',dirs_exist_ok=True)
manifest['skill_files']=[]
for p in sorted(skill.rglob('*')):
 if p.is_file():manifest['skill_files'].append({'source':str(p),'offset':0,'size':p.stat().st_size,'sha256':sha(p.read_bytes()),'output':str((OUT/'skill-creator'/p.relative_to(skill)).relative_to(OUT))})
# ASAR package metadata + physical location of packed package.json.
p=ROOT/'app.asar';f=p.open('rb');h=f.read(16);a,c,_,n=struct.unpack('<4I',h);tree=json.loads(f.read(n));v=tree['files']['package.json'];base=8+c;off=base+int(v['offset']);f.seek(off);raw=f.read(v['size']);(OUT/'app-package.json').write_bytes(raw)
manifest['app_package']={'source':str(p),'asar_entry':'/package.json','absolute_byte_start':off,'size':len(raw),'sha256':sha(raw),'version':json.loads(raw).get('version')}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'prompts':len(manifest['prompts']),'fragments':[(x['name'],x.get('start'),x.get('end'),x.get('error'))for x in manifest['fragments']],'version':manifest['app_package']['version']},ensure_ascii=False,indent=2))
