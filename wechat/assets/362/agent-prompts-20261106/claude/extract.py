#!/usr/bin/env python3
"""Extract static strings only; no installed product code or model is run."""
from pathlib import Path
import hashlib,json,subprocess,shutil
P=Path('/home/lmxxf/.local/share/claude/versions/2.1.311');D=P.read_bytes();O=Path(__file__).resolve().parent
ranges=[]
def raw(name,start,end):
 (O/name).write_bytes(D[start:end]);ranges.append(dict(file=name,source=str(P),offset=start,offset_hex=hex(start),end_exclusive=end,length=end-start,sha256=hashlib.sha256(D[start:end]).hexdigest()))
def region(name,start,end):
 a=D.index(start,240000000);b=D.index(end,a);raw(name,a,b);return D[a:b].decode()
s=region('compact-builders.source.js',b'var L3e="an <analysis>',b'var N3e=')
# This isolated slice consists only of string constants, pure template builders and an object of those builders.
js='const vm=require("vm");const s='+json.dumps(s)+';const ctx={};vm.runInNewContext(s+`\nglobalThis.out={variants:Object.fromEntries(Object.keys(Hen).map(k=>[k,NJ("",k,true)])),partial_from:Uen("","from"),partial_up_to:Uen("","up_to"),resumeGuidance:Hen.short_v2.resumeGuidance,anti_spoof:DJ,security:LJ};`,ctx,{timeout:1000});process.stdout.write(JSON.stringify(ctx.out));'
x=json.loads(subprocess.check_output(['node','-e',js]))
for n,t in x.pop('variants').items():(O/f'compact-{n}.txt').write_text(t)
for n,t in x.items():(O/f'{n}.txt').write_text(t)
# Independent SDK legacy compactionControl constant, not CLI /compact.
a=D.index(b'od=`You have been working',240000000);b=D.index(b'`;var ke,',a);raw('sdk-legacy.source.js',a,b+1)
# Decode this static template by JavaScript expression evaluation without evaluating surrounding SDK.
expr=D[a+3:b+1].decode();(O/'sdk-legacy.txt').write_text(subprocess.check_output(['node','-e','process.stdout.write('+expr+')']).decode())
# Init has old/new static prompt templates. Keep full new-init body because Phase 6 creates skills.
a=D.index(b'h4o=()=>`',248000000);b=D.index(b'`,y4o={type:"prompt",name:"init"',a);raw('new-init.source.js',a,b+1)
expr=D[a+8:b+1].decode();(O/'new-init.txt').write_text(subprocess.check_output(['node','-e','const yHe=()=>false; process.stdout.write('+expr+')']).decode())
for n,a,b in [('compact-command.source.js',249244248,249244630),('compact-selection.source.js',250553383,250554900),('threshold.source.js',249994624,250003000),('compact-execution.source.js',250843000,250860000),('restore-wrapper.source.js',250550287,250553000),('auto-failure.source.js',250871867,250876000),('compact-settings.source.js',248135000,248136000)]:raw(n,a,b)
# Locate and record semantically useful anchors, zero-based byte offsets.
anchors={}
for q in ['function NJ(','function Uen(','function HJ(','function G3e(','function Con(','function iQ(','function US(','function UQ(','name:"compact"','name:"autocompact"','name:"skills"','name:"learn"','## Phase 6: Suggest and create skills','skill-creator@claude-plugins-official','This session is being continued from a previous conversation']:
 p=D.find(q.encode(),248000000 if q=='function US(' else 240000000);anchors[q]={'offset':p,'hex':hex(p) if p>=0 else None}
plugin=Path('/home/lmxxf/.claude/plugins/marketplaces/claude-plugins-official/plugins/skill-creator')
if plugin.exists():shutil.copytree(plugin,O/'official-skill-creator',dirs_exist_ok=True)
manifest={'product':'Claude Code','version':'2.1.311','binary':str(P),'binary_sha256':hashlib.sha256(D).hexdigest(),'binary_size':len(D),'offset_units':'zero-based bytes in installed binary; ends exclusive','ranges':ranges,'anchors':anchors,'plugin_source':str(plugin),'plugin_is_separate_from_binary':True,'files':{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='manifest.json'}}
(O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'sha':manifest['binary_sha256'],'anchors':anchors,'artifacts':len(manifest['files'])},ensure_ascii=False,indent=2))
