import json,hashlib,sys
from pathlib import Path
from transformers import AutoTokenizer
from run_basepost import setup,make_plan,render,MODELS,ROOT,STIMULI,GUIDE
setup()
a=Path('/home/lmxxf/work/models')/MODELS[0];b=Path('/home/lmxxf/work/models')/MODELS[1]
ca=json.loads((a/'config.json').read_text())['text_config'];cb=json.loads((b/'config.json').read_text())['text_config'];assert ca==cb
base=AutoTokenizer.from_pretrained(a,local_files_only=True);post=AutoTokenizer.from_pretrained(b,local_files_only=True)
checks=[]
for c in make_plan(['raw','qa','chat']):
 if c.get('history_mode')=='natural':continue
 history=None
 if c.get('history_mode')=='shared':history=dict(content=(ROOT/STIMULI[c['history_stimulus']]).read_text()+'\n'+GUIDE,reply='我已读。')
 text=render(base,c['format'],c['text'],svg=c.get('svg',False),history=history)
 ia=base(text,add_special_tokens=False).input_ids;ib=post(text,add_special_tokens=False).input_ids
 assert ia==ib,c['id'];checks.append(dict(id=c['id'],tokens=len(ia),sha256=hashlib.sha256(text.encode()).hexdigest()))
plans={}
for name,formats in [(MODELS[0],['raw','qa','chat']),(MODELS[1],['raw','qa','chat'])]:
 cases=make_plan(formats);plans[name]=dict(formats=formats,total=len(cases),generated_cases=sum(c['max_new']>0 for c in cases));(ROOT/(name+'-plan.json')).write_text(json.dumps(dict(cases=cases,total=len(cases)),ensure_ascii=False,indent=2))
protocol=dict(text_configs_identical=True,tokenizer_checks=checks,base_has_chat_template=bool(base.chat_template),post_has_chat_template=bool(post.chat_template),plans=plans,total=sum(p['total'] for p in plans.values()),generated_tokens_limit=160,common_eos=[248044,248046],input_format_note='Base and Post share raw/QA/chat prompt bytes and token IDs for all unprimed and fixed-shared-history cases. Both official tokenizers expose the same embedded chat template.',natural_history_note='Uses each model own first answer; mediator text differs, so this is a total prompt-history effect, not a same-prefix weight-only comparison.',shared_history_note='Raw only, all three SVGs, fixed teacher-forced intermediate reply 我已读。',svg_guide=GUIDE,router_audit='Full router entropy, top-8 weight entropy and selected expert counts; word entropy is not a direct dimensionality measurement.',source_hashes={str(k):hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in STIMULI.items()})
(ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in protocol.items() if k not in ['tokenizer_checks','source_hashes']},ensure_ascii=False))
