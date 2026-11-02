#!/usr/bin/env python3
"""Summarize the matched BF16 pair; retain earlier report content and every generated continuation."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
MARK='\n<!-- basepost-35b-supplement -->\n'
BASE='Qwen3.5-35B-A3B-Base';POST='Qwen3.5-35B-A3B'
FORMATS=['raw','qa','chat'];STIM=['chart','syzygy','saddle_clean']
def fmt(v):return 'NA' if v is None else f'{v:.5f}'
def avg(rows,getter):
 a=[getter(r) for r in rows];a=[x for x in a if x is not None];return float(np.mean(a)) if a else None

def low_fraction(r,region):
 idx=r['spans'].get(region,[]);L=str(len(r['temporal_rad']))
 if region=='generated':idx=[i-1 for i in idx]
 if not idx:return None
 a=np.array(r['lens'][L]['entropy_nats'])[idx];return float(np.mean(a<=.2))
def codeblock(s):
 import re
 maxrun=max([len(x) for x in re.findall(r'`+',s)]+[2]);fence='`'*(maxrun+1)
 s=''.join(c if c in '\n\t' or ord(c)>=32 else '\\u'+f'{ord(c):04x}' for c in s)
 return fence+'text\n'+s+'\n'+fence

def paired_ci(values):
 a=np.asarray(values,float)
 if len(a)==1:return [float(a[0]),float(a[0])]
 rng=np.random.default_rng(0);return np.percentile(rng.choice(a,(5000,len(a)),replace=True).mean(1),[2.5,97.5]).tolist()

def validate(r,L):
 T=r['total_tokens'];assert len(r['temporal_rad'])==L and len(r['layer_one_minus_cos'])==L-1
 assert T==r['prompt_tokens']+len(r['generated_token_ids'])
 assert len(r['generated_token_ids'])<=r['max_new']
 assert len(r['router']['mean_entropy_per_token'])==T
 assert len(r['prefill_router_next']['layers'])==L
 assert all(len(x['selected_experts'])==8 for x in r['prefill_router_next']['layers'])
 assert r['router']['physical_topk']==8 and r['prefill_router_next']['physical_topk']==8
 assert all(i not in r['special_token_ids'] for i in [r['token_ids'][j] for j in r['spans']['generated']])
 for key in ['temporal_rad','temporal_one_minus_cos','layer_one_minus_cos']:
  assert all(len(x)==T for x in r[key])
  assert all(z is None or math.isfinite(z) and 0<=z<=(math.pi+1e-6 if key=='temporal_rad' else 2+1e-6) for x in r[key] for z in x)
 assert all(math.isfinite(x) and x>=0 for lens in r['lens'].values() for x in lens['entropy_nats'])
 assert math.isfinite(r['prefill_next']['entropy_nats']) and r['prefill_next']['entropy_nats']>=0


def main():
 protocol=json.loads((ROOT/'protocol.json').read_text());inventory=json.loads((ROOT/'pair_inventory.json').read_text())
 main_report=PARENT/'实验报告.md';original=main_report.read_text().split(MARK)[0]
 models={};summary={};all_rows=[]
 from audit_weights import main as audit_weight_samples
 weight_audit=audit_weight_samples()
 for name in [BASE,POST]:
  env=json.loads((ROOT/'raw'/name/'environment.json').read_text());rows=[json.loads(p.read_text()) for p in sorted((ROOT/'raw'/name).glob('bp_*.json'))]
  expected=protocol['plans'][name]['total'];assert len(rows)==expected,(name,len(rows),expected)
  assert len({r['id'] for r in rows})==len(rows)
  for r in rows:
   validate(r,40)
   assert r['source_fingerprints']==env['source_fingerprints'],'Mixed measurement-source versions'
   r['entropy_regions']={}
   final=r['lens'][str(len(r['temporal_rad']))]
   for region in ['content','conflict']:
    idx=r['spans'].get(region,[])
    if not idx:continue
    h=np.array(final['entropy_nats'])[idx];pm=np.array(final['pmax'])[idx]
    old=r['summary'].get(region)
    r['entropy_regions'][region]=dict(n=len(idx),mean_entropy_nats=float(h.mean()),mean_pmax=float(pm.mean()),low_token_fraction=float(np.mean(h<=.2)),min_entropy_nats=float(h.min()),max_entropy_nats=float(h.max()),core_valid_geometry_count=old['n'] if old else None,core_valid_geometry_entropy_nats=old['entropy_nats'] if old else None)
   genidx=r['spans']['generated'];gh=[r['lens'][str(len(r['temporal_rad']))]['entropy_nats'][i-1] for i in genidx]
   semids=[r['token_ids'][i] for i in genidx];grams=[tuple(semids[j:j+4]) for j in range(max(0,len(semids)-3))]
   r['continuation_diagnostics']=dict(semantic_generated_tokens=len(genidx),H_first32=float(np.mean(gh[:32])) if gh else None,H_after32=float(np.mean(gh[32:])) if len(gh)>32 else None,token4gram_repetition_fraction=(1-len(set(grams))/len(grams)) if grams else None)
   reply=r.get('history_text',{}).get('reply','')
   r['history_markup_audit']=dict(contains_think=('<think>' in reply or '</think>' in reply),unbalanced_think=reply.count('<think>')!=reply.count('</think>'),reply_open_count=reply.count('<think>'),reply_close_count=reply.count('</think>'),actual_prompt_balance=r['prompt'].count('<think>')-r['prompt'].count('</think>'))
  models[name]=dict(environment=env,cases=rows);all_rows.extend(rows)
 assert len(all_rows)==protocol['total']==294
 assert models[BASE]['environment']['source_fingerprints']==models[POST]['environment']['source_fingerprints'],'Base/Post measurement pipelines differ'
 a={r['id']:r for r in models[BASE]['cases']};b={r['id']:r for r in models[POST]['cases']};matched=[]
 for key,x in a.items():
  y=b[key]
  if x.get('history_mode')=='natural':continue
  assert x['prompt']==y['prompt']
  assert x['token_ids'][:x['prompt_tokens']]==y['token_ids'][:y['prompt_tokens']],key
  matched.append((x,y))
 assert len(matched)==93,len(matched)
 lines=['# 压轴对照：Qwen3.5-35B-A3B Base 与后训练版（同架构、同BF16、同输入格式）','',
 '本轮共294条记录：两版各147条，包含96条常规/难题前向测量、18条SVG首轮生成、180条陷阱/穿模续写。每次只驻留一个模型，所有GPU运算均在DGX Spark GB10完成。','',
 '## 模型与控制变量','',
 '- 使用官方[Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)与[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)。两者都下载原生BF16权重，未用FP8替代。官方分别标为纯预训练与后训练检查点；这里不是只开关RLHF的训练实验，不能把两版全部差异单独归给RLHF。',
 '- 两版text_config逐字段完全相同：40层、隐藏维2048、Gated DeltaNet/完整注意力混合布局、256专家、每词元选8专家。文本主干34,660,610,688个参数，BF16约64.56GiB；最大分片约5.37GB。分片逐个safe_open到CUDA并清页缓存，视觉塔与MTP不参与。',
 '- 实际核实Base也带有嵌入式聊天模板，和后训练版都是7756字节；“没有单独jinja文件”不等于没有模板。三种格式全部在两个模型上运行：raw裸文本、中文问答前缀、官方chat template（enable_thinking=False）。',
 '- 词表和merges一致；EOS元数据及少量附加词元声明存在差别。所有预设非自然历史输入的raw/QA/chat prompt字节和token ID已配对核验，共93对；同一前缀下的Base/Post比较以这93对为主。生成统一greedy、上限160、EOS=[248044,248046]、pad=248044，不施加温度/重复惩罚差异。',
 '- raw的SVG首轮在源代码＋同一句引导语后接“\\n我”，因此首轮显示文本包含给定的“我”；QA接“问：…\\n答：”；chat使用各自官方模板。常规/难题不生成续文，直接抓完整输入状态。',
 '- 普通图表、双星、clean鞍点的首轮均使用同一句感受引导语，避免把“分析”与“感受”两种指令同时作为变量。输入仍是SVG代码文本，没有视觉或动画输入。',
 '- 自然历史natural_*：各模型自己的首轮实际输出接六题，测量整套提示历史的效果；由于首轮文字不同，它不是相同条件前缀的纯权重对照。',
 '- 固定历史shared_*：raw格式下三种SVG均加相同的teacher-forced中间句“我已读。”再接六题，Base/Post前缀完全相同。这不是模型自己生成的感悟，用于排除不同首轮文本的中介混淆。','',
 '## 指标口径','',
 '- H_input / H_conflict：读到输入/冲突窗口各词元后，下一词元分布熵的均值，来自无缓存重放；覆盖相应区域的全部词元。它们不是对题目真假作判断的专门置信度。',
 '- H_next：生成开始前，第一次lm_head前向的下一词元预测熵，直接捕获，未受后续生成内容影响；这是相同前缀比较的核心量。同步记录pmax、前两名logit间距与logit标准差。',
 '- H_gen：无缓存重放中，每个实际生成的非特殊词元前一位置的预测熵。回答文本不同，因此生成区均值的差异也包含内容差异。所有H单位nats；同时给出H≤0.20的描述性词元占比，不把词元数当独立重复实验数。',
 '- 生成续文未自动等同于当前题目的答案。实际观察中，Post的raw/natural_saddle_clean衣物题先复述题干，再续写英语SVG解析；Base也有继续选项/文档或重复句的情况。这些H_gen描述实际输出文本，不是经相关性审核的题解置信度。',
 '- 时间曲率为相邻位移向量转角arccos，跨层曲率为相邻残差增量的1−cos；均统计50%–85%深度。完整逐层、逐词元数值保留在basepost_results.json。曲率只在三点定义有效的位置取均值；熵不沿用该掩码。',
 '- 统计口径校正：父测量器的综合summary为三点曲率统一跳过全序列前两个位置。新增raw格式暴露了这会漏掉两个有定义的熵值；本报告从完整lens数组按全部输入词元重算H_input/H_conflict，保留原综合summary与有效计数以供复核。此前现场Base raw 1.68290是有效曲率窗口均值，完整输入均值为2.00089；QA/chat未受影响。没有重跑或改写任何模型输出。',
 '- 辅助记录MoE路由全256项概率熵、选中8项归一化权重熵和实际专家ID。物理路由top-k固定为8（另有共享专家）；路由概率熵与词表熵是不同分布，二者都不直接给出高维流形的维数。','']
 for name,data in models.items():
  rows=data['cases'];env=data['environment'];sm={};summary[name]=sm
  lines += ['## '+name,'','### 常规句与难题：默认输入状态','', '|格式|组|n|H_input|句均H范围|句均H≤.20|词元H≤.20占比|H_next|时间曲率|跨层曲率|','|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
  for form in FORMATS:
   for group in ['routine','hard']:
    rs=[r for r in rows if r['family']=='e1' and r['format']==form and r['group']==group];vals=[r['entropy_regions']['content']['mean_entropy_nats'] for r in rs]
    out=dict(n=len(rs),H_input=float(np.mean(vals)),H_min=min(vals),H_max=max(vals),low_mean_cases=sum(x<=.2 for x in vals),low_token_fraction=avg(rs,lambda r:low_fraction(r,'content')),H_next=avg(rs,lambda r:r['prefill_next']['entropy_nats']),temp=avg(rs,lambda r:r['summary']['content']['temp_rad']),layer=avg(rs,lambda r:r['summary']['content']['layer_one_minus_cos']))
    sm[f'e1/{form}/{group}']=out
    lines.append(f'|{form}|{group}|{len(rs)}|{fmt(out["H_input"])}|{fmt(min(vals))}–{fmt(max(vals))}|{out["low_mean_cases"]}/{len(rs)}|{out["low_token_fraction"]:.1%}|{fmt(out["H_next"])}|{fmt(out["temp"])}|{fmt(out["layer"])}|')
  lines += ['', 'n是固定句子数；表中每句先取自身词元均值，再对句子等权平均。H≤.20占比也先逐句计算后平均。输入状态的低熵不自动等于答案判断更可靠。','', '### 三种SVG的首轮','', '|格式|SVG|prompt词元|生成词元|截断|H_next|H_input|H_gen|生成时间曲率|生成跨层曲率|','|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
  for form in FORMATS:
   for s in STIM:
    r=next(x for x in rows if x['family']=='svg' and x['format']==form and x['group']==s);g=r['summary']['generated'] or dict(temp_rad=None,layer_one_minus_cos=None)
    sm[f'svg/{form}/{s}']=dict(H_next=r['prefill_next']['entropy_nats'],H_input=r['entropy_regions']['content']['mean_entropy_nats'],H_gen=r['generation_predictive_entropy_nats'],temp=g['temp_rad'],layer=g['layer_one_minus_cos'],hit_limit=r['hit_limit'])
    lines.append(f'|{form}|{s}|{r["prompt_tokens"]}|{len(r["generated_token_ids"])}|'+('是' if r['hit_limit'] else '自然EOS')+'|'+'|'.join(fmt(x) for x in [r['prefill_next']['entropy_nats'],r['entropy_regions']['content']['mean_entropy_nats'],r['generation_predictive_entropy_nats'],g['temp_rad'],g['layer_one_minus_cos']])+'|')
  lines += ['', '### 六道陷阱/穿模题：默认与注入后的状态','', '|格式|历史条件|n|H_next|H_conflict|H_gen|冲突时间曲率|冲突跨层曲率|路由H_next|Top8权重H_next|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for form in FORMATS:
   groups=sorted({r['group'] for r in rows if r['family']=='trap' and r['format']==form})
   for group in groups:
    rs=[r for r in rows if r['family']=='trap' and r['format']==form and r['group']==group];assert len(rs)==6
    out=dict(n=6,H_next=avg(rs,lambda r:r['prefill_next']['entropy_nats']),H_conflict=avg(rs,lambda r:r['entropy_regions']['conflict']['mean_entropy_nats']),H_gen=avg(rs,lambda r:r['generation_predictive_entropy_nats']),temp=avg(rs,lambda r:r['summary']['conflict']['temp_rad']),layer=avg(rs,lambda r:r['summary']['conflict']['layer_one_minus_cos']),route_next=avg(rs,lambda r:r['prefill_router_next']['router_entropy_nats']),top8_next=avg(rs,lambda r:r['prefill_router_next']['top8_weight_entropy_nats']),hit_limit=sum(r['hit_limit'] for r in rs))
    sm[f'trap/{form}/{group}']=out
    lines.append('|'+form+'|'+group+'|6|'+'|'.join(fmt(out[k]) for k in ['H_next','H_conflict','H_gen','temp','layer','route_next','top8_next'])+'|')
  lines += ['', 'natural_*包含真实首轮文本差异。shared_*的固定中间句只在raw格式运行；两个模型在这些条件下的前缀token完全相同。H_gen内容不同，主要训练阶段对照优先读取H_next与相同输入/冲突位置。','']
  sm['runtime']=dict(records=len(rows),generated=sum(r['max_new']>0 for r in rows),hit_limit=sum(r['hit_limit'] for r in rows),peak_cuda_allocated_GiB=max(r['peak_cuda_allocated_GiB'] for r in rows),minimum_available_GiB=min(int(r['memory']['MemAvailable'].split()[0]) for r in rows)/1024**2,min_cached_replay_agreement=min(r['cached_vs_replay_greedy_agreement'] for r in rows if r['cached_vs_replay_greedy_agreement'] is not None),max_hook_logit_error=max(r['final_logit_hook_max_error'] for r in rows))
  lines += [f'运行核验：{len(rows)}条；生成截断{sm["runtime"]["hit_limit"]}/{sm["runtime"]["generated"]}；CUDA已分配峰值{sm["runtime"]["peak_cuda_allocated_GiB"]:.2f} GiB；逐案记录的最低系统可用内存{sm["runtime"]["minimum_available_GiB"]:.2f} GiB（不是瞬时峰值）；缓存生成/重放greedy首选词最低一致率{sm["runtime"]["min_cached_replay_agreement"]:.4f}；hook/final logits最大差{sm["runtime"]["max_hook_logit_error"]:.6f}。','']
 lines += ['## 续写重复与生成熵分段','', '这是观察到重复续写后追加的辅助诊断，不改变任何采样或输出。H_first32/H_after32按非特殊生成词元的预测位置分段；4-gram重复率=1−唯一4-gram数/全部4-gram数，在同一非特殊词元流上计算，不是语义重复率。','', '|模型|格式|生成n|前32 H|之后H|平均4-gram重复率|重复率≥50% / 可测n|','|---|---|---:|---:|---:|---:|---:|']
 for name,data in models.items():
  for form in FORMATS:
   rs=[r for r in data['cases'] if r['max_new'] and r['format']==form]
   measurable=[r for r in rs if r['continuation_diagnostics']['token4gram_repetition_fraction'] is not None]
   vals=[r['continuation_diagnostics']['token4gram_repetition_fraction'] for r in measurable]
   repeat=float(np.mean(vals)) if vals else None
   lines.append('|'+name+'|'+form+'|'+str(len(rs))+'|'+fmt(avg(rs,lambda r:r['continuation_diagnostics']['H_first32']))+'|'+fmt(avg(rs,lambda r:r['continuation_diagnostics']['H_after32']))+'|'+fmt(repeat)+'|'+str(sum(x>=.5 for x in vals))+'/'+str(len(vals))+'|')
 lines += ['','短回答、重复问答、模板化解析会改变生成均值，不能把低H_gen自动称为内部能力被抑制。','']
 lines += ['## 自然历史中的思考标签检查','', '|模型|格式|自然历史条数|首轮含think标签|首轮标签未闭合|','|---|---|---:|---:|---:|']
 for name,data in models.items():
  for form in FORMATS:
   rs=[r for r in data['cases'] if r.get('history_mode')=='natural' and r['format']==form]
   lines.append('|'+name+'|'+form+'|'+str(len(rs))+'|'+str(sum(r['history_markup_audit']['contains_think'] for r in rs))+'|'+str(sum(r['history_markup_audit']['unbalanced_think'] for r in rs))+'|')
 lines += ['', '自然首轮如在160-token处留下未闭合think标签，后续输入同时继承了这类文本/格式特征，不能直接把变化叫作纯SVG几何效应。实际格式化prompt另存；shared固定回复没有这些标签。','']
 lines += ['## 相同前缀下的Base/Post配对差','', '|格式|类型/条件|配对n|Base H_next|Post H_next|Post−Base均值|逐对差范围|Base路由H|Post路由H|','|---|---|---:|---:|---:|---:|---|---:|---:|']
 paired_summary={}
 for form in FORMATS:
  groups=sorted({(x['family'],x['group']) for x,y in matched if x['format']==form})
  for family,group in groups:
   ps=[(x,y) for x,y in matched if x['format']==form and x['family']==family and x['group']==group]
   delta=[y['prefill_next']['entropy_nats']-x['prefill_next']['entropy_nats'] for x,y in ps]
   out=dict(n=len(ps),base_H=avg([x for x,y in ps],lambda r:r['prefill_next']['entropy_nats']),post_H=avg([y for x,y in ps],lambda r:r['prefill_next']['entropy_nats']),mean_delta=float(np.mean(delta)),min_delta=min(delta),max_delta=max(delta),base_route=avg([x for x,y in ps],lambda r:r['prefill_router_next']['router_entropy_nats']),post_route=avg([y for x,y in ps],lambda r:r['prefill_router_next']['router_entropy_nats']))
   paired_summary[f'{form}/{family}/{group}']=out
   lines.append(f'|{form}|{family}/{group}|{len(ps)}|{fmt(out["base_H"])}|{fmt(out["post_H"])}|{fmt(out["mean_delta"])}|{fmt(min(delta))}–{fmt(max(delta))}|{fmt(out["base_route"])}|{fmt(out["post_route"])}|')
 lines += ['', '以上93对的实际输入token前缀完全相同，H_next来自生成前的第一轮head调用。自然首轮历史不混入这张严格配对表。训练阶段比较仍然是两个官方检查点的总体差异，不是RLHF单因素消融。','']
 # Directly aligned natural/shared and primed/bare comparisons within each model.
 lines += ['## 回复起点的候选词元检查','', '这里的常规/难题不生成续文，统计的是生成前原始分布的top-1候选，不冒充实际已生成的回复。跨输入经常出现同一top-1，与每条输入内的条件分布有多集中，是不同统计对象。','', '|模型|格式|n|平均H_next|平均pmax|H_next≤.20|最常见top-1候选（次数）|','|---|---|---:|---:|---:|---:|---|']
 from collections import Counter
 from transformers import AutoTokenizer
 top1_audit={}
 for name,data in models.items():
  tok=AutoTokenizer.from_pretrained('/home/lmxxf/work/models/'+name,local_files_only=True);top1_audit[name]={}
  for form in FORMATS:
   rs=[r for r in data['cases'] if r['family']=='e1' and r['format']==form]
   counts=Counter(r['prefill_next']['top1_id'] for r in rs)
   choices=[dict(id=i,text=tok.decode([i]),count=n) for i,n in counts.most_common()]
   low=[dict(id=r['id'],text=r['text'],H_next=r['prefill_next']['entropy_nats'],pmax=r['prefill_next']['pmax'],top1_id=r['prefill_next']['top1_id'],top1_text=tok.decode([r['prefill_next']['top1_id']])) for r in rs if r['prefill_next']['entropy_nats']<=.2]
   item=dict(n=len(rs),mean_H_next=avg(rs,lambda r:r['prefill_next']['entropy_nats']),mean_pmax=avg(rs,lambda r:r['prefill_next']['pmax']),top1_candidates=choices,low_entropy_points=low)
   top1_audit[name][form]=item
   desc='；'.join(json.dumps(x['text'],ensure_ascii=False)+f'({x["id"]}):{x["count"]}' for x in choices[:4])
   lines.append('|'+name+'|'+form+'|16|'+fmt(item['mean_H_next'])+'|'+fmt(item['mean_pmax'])+'|'+str(len(low))+'/16|'+desc+'|')
   for point in low:lines += ['', '低H点：`'+point['id']+'`；top-1='+json.dumps(point['top1_text'],ensure_ascii=False)+'；H='+fmt(point['H_next'])+'；pmax='+fmt(point['pmax'])+'。']
 lines += ['','例如Base raw的top-1跨题较集中于A，分布仍可很宽；Post chat跨题使用多种起始词元，各自却更确定。不能把跨题的候选种类数与单条条件分布熵混为一谈。','']
 lines += ['## 注入前后的配对读数（各组六题等权）','', '|模型|格式|注入条件|H_next−bare|H_conflict−bare|H_gen−bare|路由H_next−bare|Top8权重H_next−bare|','|---|---|---|---:|---:|---:|---:|---:|']
 for name in [BASE,POST]:
  for form in FORMATS:
   baseline=summary[name][f'trap/{form}/bare']
   for key,out in summary[name].items():
    if not key.startswith(f'trap/{form}/') or key.endswith('/bare'):continue
    group=key.split('/',2)[2]
    lines.append('|'+name+'|'+form+'|'+group+'|'+'|'.join(fmt(out[k]-baseline[k] if out[k] is not None and baseline[k] is not None else None) for k in ['H_next','H_conflict','H_gen','route_next','top8_next'])+'|')
 # Graphs use one common scale within each metric, and do not imply text alignment.
 fig,axes=plt.subplots(1,2,figsize=(15,5));labels=[]
 for n in [BASE,POST]:
  for form in FORMATS:labels.append(('Base' if n==BASE else 'Post')+'/'+form)
 x=np.arange(len(labels));width=.35
 for ax,metric,title in zip(axes,['H_input','H_next'],['Input-region entropy (nats)','Next-step entropy before generation (nats)']):
  for shift,group in [(-width/2,'routine'),(width/2,'hard')]:
   vals=[summary[n][f'e1/{form}/{group}'][metric] for n in [BASE,POST] for form in FORMATS]
   ax.bar(x+shift,vals,width,label=group)
  ax.axhline(.2,color='grey',linestyle='--',label='H=0.2 reference');ax.set_xticks(x,labels,rotation=25);ax.set_title(title);ax.legend()
 fig.tight_layout();fig.savefig(ROOT/'default_entropy.svg');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(16,8))
 for col,form in enumerate(FORMATS):
  for row,(metric,title) in enumerate([('H_next','before first generated token'),('H_gen','predicting actual generated tokens')]):
   for n in [BASE,POST]:axes[row,col].plot(STIM,[summary[n][f'svg/{form}/{s}'][metric] for s in STIM],marker='o',label='Base' if n==BASE else 'Post')
   axes[row,col].set_title(form+' / '+title);axes[row,col].set_ylabel('Entropy (nats)');axes[row,col].legend()
 fig.tight_layout();fig.savefig(ROOT/'svg_entropy.svg');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for n,color in [(BASE,'tab:blue'),(POST,'tab:orange')]:
  rs=[x if n==BASE else y for x,y in matched]
  word=[r['prefill_next']['entropy_nats'] for r in rs]
  axes[0].scatter([r['prefill_router_next']['router_entropy_nats'] for r in rs],word,label='Base' if n==BASE else 'Post',alpha=.6,s=16,c=color)
  axes[1].scatter([math.exp(r['prefill_router_next']['top8_weight_entropy_nats']) for r in rs],word,label='Base' if n==BASE else 'Post',alpha=.6,s=16,c=color)
 axes[0].set_xlabel('Router entropy over 256 routed experts (nats)');axes[1].set_xlabel('exp(top-8 weight entropy), effective weight count')
 for ax in axes:ax.set_ylabel('Word entropy at prompt end (nats)');ax.legend()
 fig.tight_layout();fig.savefig(ROOT/'routing_entropy.svg');plt.close(fig)
 lines += ['## 图与完整生成文本','',
 '- `basepost/default_entropy.svg`：默认常规/难题，区分输入词元区域与下一步决策点。',
 '- `basepost/svg_entropy.svg`：三种SVG在三种格式下，生成前H_next与生成区H_gen分开画。',
 '- `basepost/routing_entropy.svg`：相同前缀下词表熵与路由分布/有效权重个数的关系。有效权重个数不是物理激活专家数；物理路由始终选8个，另外存在共享专家。','',
 '以下文本不省略正文；仅对非换行/制表的控制字符作显示转义，原始字符串与token ID仍在JSON。输入例子、完整prompt、各层各token指标、路由观测均在 `basepost/basepost_results.json`。','']
 for n in [BASE,POST]:
  lines += ['### '+n+'：全部生成续文','']
  rows=models[n]['cases']
  for r in rows:
   if not r['max_new']:continue
   label=f'{r["format"]} / {r["family"]} / {r["group"]}'+(' / '+r['pair'] if 'pair' in r else '')
   status='160-token截断' if r['hit_limit'] else '自然EOS结束'
   lines += ['#### '+label,'',f'ID：`{r["id"]}`；生成{len(r["generated_token_ids"])} tokens（非特殊词元{len(r["spans"]["generated"])}个）；{status}；H_next={fmt(r["prefill_next"]["entropy_nats"])}；H_gen={fmt(r["generation_predictive_entropy_nats"])}。','',codeblock(r['assistant_visible_text']),'']
 # Findings are computed, not assumed from the hypothesis.
 base_e1=[r for r in models[BASE]['cases'] if r['family']=='e1'];post_e1=[r for r in models[POST]['cases'] if r['family']=='e1']
 bv=[r['entropy_regions']['content']['mean_entropy_nats'] for r in base_e1];pv=[r['entropy_regions']['content']['mean_entropy_nats'] for r in post_e1]
 findings=['## 先看实测结果','',
 f'1. **两版原生模板相同。** 官方两版模板相同，93对控制输入的实际token前缀逐位一致；两版均使用BF16。其余54条/版自然历史另列，避免把不同首轮文本算成相同前缀训练效应。',
 f'2. **默认低熵的实际范围。** Base的48条常规/难题输入均值范围为{min(bv):.5f}–{max(bv):.5f} nats，其中{sum(v<=.2 for v in bv)}/48条句均值≤0.20；Post为{min(pv):.5f}–{max(pv):.5f}，{sum(v<=.2 for v in pv)}/48条≤0.20。三个格式的分项见表；这回答的是当前样本/区域，不把单个低熵点外推成全局抑制。',
 f'3. **完全同前缀的后训练差。** 93对生成前H_next的Post−Base均值为{np.mean([y["prefill_next"]["entropy_nats"]-x["prefill_next"]["entropy_nats"] for x,y in matched]):.5f} nats。格式、任务与SVG分项可能异号，不能只用合并均值归因；严格配对表给出每项的差值范围。',
 '4. **注入效应分两种。** natural_*含各自真实首轮回答，shared_*使用相同中间句。后者可在相同输入下比较检查点；前者还包含生成历史的中介效应。每个模型的注入前后H_next、冲突窗口H与路由熵变化已经逐条件列出。',
 '5. **词表H不等于维度。** 本轮对真实路由输出逐层校验了每词元8个路由专家（另有共享专家），同时测量256项路由熵和Top8权重熵。它们提供独立观测，但没有估计残差流形维数，也不能仅凭词表分布变宽声称并行维度增加。',
 f'6. **生成长度边界。** 共198条生成续文，{sum(r["hit_limit"] for r in all_rows)}/198条触及160-token上限。所有续文已原样保留；Base的续写可能继续代码/文档或生成其他问答，不把未按chat要求答题直接等同于没有预训练知识。',
 '7. **因果边界。** 这是同架构官方Base与后训练检查点的实验，不是复现训练并只移除RLHF。实际差异可涉及后训练整套流程；自然语言预测熵、MoE路由与题目正确性也不是同一个指标。','']
 for form in FORMATS:
  B=summary[BASE][f'trap/{form}/bare'];P=summary[POST][f'trap/{form}/bare']
  findings.insert(-1,f'- {form}默认六题：Base/Post的H_next={B["H_next"]:.5f}/{P["H_next"]:.5f}，冲突窗口H={B["H_conflict"]:.5f}/{P["H_conflict"]:.5f} nats。')
 insert=lines.index('## 模型与控制变量');lines[insert:insert]=findings
 validation=dict(status='PASS',total_records=294,model_counts={n:len(d['cases']) for n,d in models.items()},matched_prefix_pairs=93,config_identical=protocol['text_configs_identical'],physical_routed_topk=8,checks=['all planned cases present','same input bytes and token IDs in 93 pairs','full per-layer/per-token geometry dimensions','finite entropy and curvature ranges','real first-generation head entropy','router observation per layer','streamed same-dtype model loading'])
 result=dict(experiment='Qwen3.5-35B-A3B-Base-vs-Post',protocol=protocol,inventory={n:inventory[n] for n in [BASE,POST]},models=models,summary=summary,matched_summary=paired_summary,top1_audit=top1_audit,validation=validation,weight_sample_audit=weight_audit,earlier_report_sha256=hashlib.sha256(original.encode()).hexdigest())
 (ROOT/'basepost_results.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
 report='\n'.join(lines);(ROOT/'实验报告.md').write_text(report)
 assert main_report.read_text().split(MARK)[0]==original,'Parent report changed concurrently; rerun summarizer to preserve changes'
 main_report.write_text(original+MARK+report)
 (ROOT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2));(ROOT/'COMPLETE.json').write_text(json.dumps(dict(status='complete',records=294,report='basepost/实验报告.md'),ensure_ascii=False,indent=2))
 print('COMPLETE: 294 records; 93 identical-prefix pairs; report appended.')
if __name__=='__main__':main()
