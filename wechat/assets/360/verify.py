"""CPU-only artifact/measurement audit. Run after experiment finishes."""
import json,hashlib,math
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;d=json.loads((R/'results.json').read_text());a=d['manifest']['alpha'];audit={}
assert len(d['original'])==40 and len(d['edited'])==len(d['random_control'])==32
assert [x['id'] for x in d['original'][8:]]==[x['id'] for x in d['edited']]==[x['id'] for x in d['random_control']]
assert set(x['question'] for x in d['fit']).isdisjoint(x['question'] for x in json.loads((R/'twins.json').read_text())['transfer'])
twins=json.loads((R/'twins.json').read_text())
for kind in ['fit','transfer']:
 for x,y in zip(twins[kind][::2],twins[kind][1::2]):assert x['common_ids']==y['common_ids'] and len(x['tail_ids'])==len(y['tail_ids'])
for m in d['manifest']['layers']:
 l=m['layer'];p=torch.load(R/f'edit_layer_{l}.pt',map_location='cpu',weights_only=True);w=p['W0'];e=(w.float()-a*p['delta']).to(torch.bfloat16);rr=(w.float()-a*p['random_delta']).to(torch.bfloat16)
 assert hashlib.sha256(e.view(torch.uint8).numpy().tobytes()).hexdigest()==m['edited_sha256']
 delta=e.float()-w.float();random_delta=rr.float()-w.float();outside=torch.ones(3072,dtype=torch.bool);outside[p['columns']]=False
 assert not delta[:,outside].count_nonzero() and not random_delta[:,outside].count_nonzero()
 assert abs(float(torch.dot(p['u'],p['random_direction'])))<1e-6
 assert math.isclose(float(p['delta'].norm()),float(p['random_delta'].norm()),rel_tol=1e-5)
 audit[str(l)]=dict(actual_target_F=float(delta.norm()),actual_random_F=float(random_delta.norm()),random_to_target_actual_F_ratio=float(random_delta.norm()/delta.norm()),outside_selected_changed=0,actual_changed=int(delta.count_nonzero()),random_actual_changed=int(random_delta.count_nonzero()),unit_u_norm=float(p['u'].norm()),orthogonal_dot=float(torch.dot(p['u'],p['random_direction'])),reconstruction_hash_match=True)
audit['script_fingerprint_matches']=hashlib.sha256((R/'run_experiment.py').read_bytes()).hexdigest()==d['config']['script_sha256'];assert audit['script_fingerprint_matches']
audit['record_counts']=[40,32,32];audit['pairs_shared_prefix_valid']=True
d['audit']=audit;(R/'results.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');(R/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit,indent=2))
