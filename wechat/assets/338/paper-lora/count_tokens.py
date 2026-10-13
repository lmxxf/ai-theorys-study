import sys,pathlib,json
root=pathlib.Path(r'C:\Strata-Lab');sys.path.insert(0,str(root/'Strata/tools'))
import strata_tokenizer as ST
p=root/'Strata-data/packs/q2_0/tokenizer';v=json.loads((p/'vocab.json').read_text(encoding='utf-8'));tokens=[None]*len(v)
for token,i in v.items():tokens[i]=token
tok=ST.Tokenizer(tokens,(p/'merges.txt').read_text(encoding='utf-8').split('\n'),json.loads((p/'token_type.json').read_text()))
m=json.loads((root/'papers-study/prompts.json').read_text(encoding='utf-8'))
counts={s:0 for s in ['train','val','test']};maximum=0
for c in m['chunks']:
 n=len(tok.encode(c['prompt']));c['content_tokens']=n;counts[c['split']]+=n;maximum=max(maximum,n)
assert maximum<3900,maximum
(root/'papers-study/prompts.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'papers-study/token-counts.json').write_text(json.dumps({'content_tokens':counts,'max_chunk_tokens':maximum,'chunks':len(m['chunks'])},indent=2))
print(counts,'max',maximum,flush=True)
