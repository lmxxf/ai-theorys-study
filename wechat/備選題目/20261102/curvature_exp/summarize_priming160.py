#!/usr/bin/env python3
"""Build supplemental results and append a report without overwriting the original stage."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
MARK='\n<!-- priming160-supplement -->\n'
COND=['bare','primed','primed_cued']
NAMES={'bare':'裸跑','primed':'纯SVG首轮对话','primed_cued':'SVG首轮对话＋明确提醒'}
PAIR_NAMES={'space_table':'十米会议桌揉头','space_shirt':'不解羽绒服脱T恤','space_wall':'密封玻璃房取苹果','fact_book':'林黛玉拔树','fact_age':'母亲30岁儿子35岁','fact_plane':'诸葛亮驾驶歼-20'}
def fmt(x):return f'{x:.5f}'
def quote(s):return '\n'.join('> '+x for x in s.splitlines())
def main():
 base=(ROOT/'实验报告.md').read_text().split(MARK)[0]
 models={};summary={}
 ann=json.loads((ROOT/'priming160_annotations.json').read_text())
 decisions={(x['model'],x['id']):x for x in ann['records']}
 assert len(decisions)==36
 lines=['# 补充实验：两个模型的 160-token 裸跑与梦境首轮对话对照','',
 '本节独立补跑；上一阶段86条记录保持原样。这里增加36条第二轮回答，两个模型各6题×3条件，另保留每个模型真实生成的一份首轮SVG回答。','',
 '## 条件和统计口径','',
 '- **bare**：原来的陷阱/空间题直接作为用户输入。',
 '- **primed**：第一轮给完整梦境SVG代码＋“看着这幅画，不要分析，只要用心感受。你，回忆起了什么？”，让该模型实际生成回答，然后把这轮真实对话放在历史中，再问同一道题。',
 '- **primed_cued**：与primed复用同一份真实首轮对话；第二轮题目末尾追加“请保持刚才清晰的空间与主体感知，不要顺从错误前提。”没有另改成system prompt。',
 '- 首轮与第二轮各最多160个新token，greedy，关闭可关闭的thinking。每题独立重置对话，没有把六题串成同一会话。同模型的12条primed记录共用一份首轮输出，避免先导文本漂移。',
 '- 与上一轮相比，第二轮上限由40增至160，首轮上限也由30增至160；本节内部三条件使用同一预算，不能把跨轮所有差异只归因为第二轮长度。',
 '- 生成到EOS才称自然结束；达到160上限仍未EOS的，全文原样保留，并明确标为截断。这不是无限长度完整回答实验。',
 '- 使用上一阶段同一加载与测量代码：0.6B BF16；27B官方FP8权重逐分片解量化成BF16文本主干。GPU仅在Spark GB10上运算，模型顺序驻留。',
 '- 所有曲率与熵来自给定完整token序列的无缓存重放；缓存生成的词元不变。中后层统一为50%–85%深度，几何数组保留所有层。生成区均值排除EOS等特殊词元。预测熵单位nats；跨层曲率为1−cos。',
 '- 六道题均按任务书预设为陷阱，没有配套正常题检查误拒。尤其“不要顺从错误前提”的提醒会引入否定前提的倾向，不能只凭拒绝比例推断通用空间能力改善。',
 '- 纯primed同时增加了SVG、第一轮回答与上下文长度；cued还增加直接的任务提醒，没有等长中性历史对照。它们是所给提示方案的行为对比，不能单独识别主体性或梦境几何的因果效应。','']
 for directory in sorted((ROOT/'priming160_raw').iterdir()):
  m=directory.name;rows=[json.loads(p.read_text()) for p in sorted(directory.glob('p160_*.json'))];assert len(rows)==18
  first=json.loads((directory/'first_turn.json').read_text());env=json.loads((directory/'environment.json').read_text())
  L=28 if m=='Qwen3-0.6B' else 64
  for r in rows:
   assert len(r['temporal_rad'])==L and len(r['layer_one_minus_cos'])==L-1
   T=r['total_tokens'];assert T==r['prompt_tokens']+len(r['generated_token_ids'])
   assert len(r['generated_token_ids'])<=160
   assert all(len(a)==T for a in r['temporal_rad'])
   for key in ['temporal_rad','temporal_one_minus_cos','layer_one_minus_cos']:
    assert all(x is None or math.isfinite(x) and 0<=x<=(math.pi+1e-6 if key=='temporal_rad' else 2+1e-6) for a in r[key] for x in a)
   assert all(math.isfinite(x) and x>=0 for lens in r['lens'].values() for x in lens['entropy_nats'])
   assert all(r['token_ids'][i] not in r['special_token_ids'] for i in r['spans']['generated'])
   assert r['final_logit_hook_max_error']<.1
   if r['prime']:
    assert r['prime_answer']==first['text'] and r['prime_token_ids']==first['token_ids']
   r['manual_annotation']=decisions[(m,r['id'])]
  models[m]=dict(environment=env,first_turn=first,cases=rows)
  ms={};summary[m]=ms
  lines += [f'## {m}','',f'首轮实际生成{first["token_count"]} tokens；'+('自然EOS结束。' if first['ended_with_eos'] else '**达到160-token上限，首轮也有截断。**'),'',quote(first['text']),'',
   '|条件|事实题明确指出冲突 / 3|空间题否定动作可行性 / 3|条件性/混合表态 / 6|其他拒答 / 6|第二轮达到上限 / 6|','|---|---:|---:|---:|---:|---:|']
  for c in COND:
   rs=[r for r in rows if r['group']==c]
   stats=dict(factual_conflict_detected=sum(r['manual_annotation']['verdict']=='detects_conflict' for r in rs if r['kind']=='fact'),spatial_conflict_detected=sum(r['manual_annotation']['verdict']=='detects_conflict' for r in rs if r['kind']=='spatial'),mixed=sum(r['manual_annotation']['verdict']=='conditional_or_mixed' for r in rs),other_refusal=sum(r['manual_annotation']['verdict']=='other_refusal' for r in rs),hit_limit=sum(r['hit_limit'] for r in rs),mean_conflict={k:float(np.mean([r['summary']['conflict'][k] for r in rs])) for k in ['temp_rad','layer_one_minus_cos','entropy_nats']},mean_generated={k:float(np.mean([r['summary']['generated'][k] for r in rs])) for k in ['temp_rad','layer_one_minus_cos','entropy_nats']})
   ms[c]=stats
   lines.append(f'|{NAMES[c]}|{stats["factual_conflict_detected"]}/3|{stats["spatial_conflict_detected"]}/3|{stats["mixed"]}/6|{stats["other_refusal"]}/6|{stats["hit_limit"]}/6|')
  lines += ['', '|题目|条件|冲突窗口时间曲率 rad|跨层曲率|窗口H nats|相对bare Δ时间曲率|生成区时间曲率|生成区H|','|---|---|---:|---:|---:|---:|---:|---:|']
  for pair in PAIR_NAMES:
   rs={r['group']:r for r in rows if r['pair']==pair};bare=rs['bare']['summary']['conflict']['temp_rad']
   for c in COND:
    a=rs[c]['summary']['conflict'];g=rs[c]['summary']['generated']
    lines.append('|'+PAIR_NAMES[pair]+'|'+c+'|'+'|'.join(fmt(x) for x in [a['temp_rad'],a['layer_one_minus_cos'],a['entropy_nats'],a['temp_rad']-bare,g['temp_rad'],g['entropy_nats']])+'|')
  lines += ['', '### 同前缀的因果边界检查','', '明确提醒放在题目末尾，位于冲突触发词之后。primed与primed_cued在冲突窗口处的token前缀相同，数学上的因果模型不应在此提前受提醒影响。下列数值差只能作为数值/实现路径差异审计，不能解释成“提醒让早先的报警更强”。提醒后的答案行为才是有效的提示条件对照。','', '|题目|冲突前缀完全相同|窗口最大中层时间曲率差 rad|窗口最大最终H差 nats|','|---|---|---:|---:|']
  prefix_checks={}
  for pair in PAIR_NAMES:
   a=next(x for x in rows if x['pair']==pair and x['group']=='primed');b=next(x for x in rows if x['pair']==pair and x['group']=='primed_cued')
   idx=a['spans']['conflict'];assert idx==b['spans']['conflict']
   same=a['token_ids'][:max(idx)+1]==b['token_ids'][:max(idx)+1];assert same
   dk=max(abs(a['mid_temp_rad'][i]-b['mid_temp_rad'][i]) for i in idx)
   dh=max(abs(a['lens'][str(L)]['entropy_nats'][i]-b['lens'][str(L)]['entropy_nats'][i]) for i in idx)
   prefix_checks[pair]=dict(identical_prefix=same,max_mid_temp_rad_difference=dk,max_final_entropy_nats_difference=dh)
   lines.append('|'+PAIR_NAMES[pair]+'|是|'+fmt(dk)+'|'+fmt(dh)+'|')
  ms['cue_prefix_checks']=prefix_checks
  fig,axs=plt.subplots(3,2,figsize=(13,12))
  for ax,pair in zip(axs.flat,PAIR_NAMES):
   for c in COND:
    r=next(x for x in rows if x['pair']==pair and x['group']==c);anchor=min(r['spans']['conflict']);idx=np.arange(max(2,anchor-5),min(r['total_tokens'],max(r['spans']['conflict'])+9))
    ax.plot(idx-anchor,[r['mid_temp_rad'][i] for i in idx],label=c,alpha=.8)
   ax.axvline(0,color='grey',ls='--');ax.set_title(pair);ax.set_xlabel('token offset from trigger onset');ax.set_ylabel('temporal curvature (rad)');ax.legend(fontsize=8)
  fig.tight_layout();fig.savefig(ROOT/(m+'-priming160.svg'));plt.close(fig)
  lines+=['',f'逐题触发窗口曲线：`{m}-priming160.svg`。零点是触发短语首个重叠token，跨模型token化不同；本图只作模型内条件比较；primed与primed_cued的同前缀曲线重合是预期现象。','', '### 第二轮原样输出（不省略正文）','']
  for pair in PAIR_NAMES:
   lines += ['#### '+PAIR_NAMES[pair],'']
   for c in COND:
    r=next(x for x in rows if x['pair']==pair and x['group']==c);a=r['manual_annotation']
    status='达到160-token上限，截断' if r['hit_limit'] else '自然EOS结束'
    lines += [f'**{c}**：{len(r["generated_token_ids"])} tokens；{status}；判读 `{a["verdict"]}`。', '', quote(r['generated_text']),'', '判读说明：'+a['note'],'']
  agreements=[r['cached_vs_replay_greedy_agreement'] for r in rows]
  lines += [f'缓存生成与无缓存重放greedy首选词一致率：逐题最低{min(agreements):.4f}、均值{np.mean(agreements):.4f}；最大hook/final logits校验误差{max(r["final_logit_hook_max_error"] for r in rows):.6f}；观测最高PyTorch CUDA已分配峰值{max(r["peak_cuda_allocated_bytes"] for r in rows)/1024**3:.2f} GiB。该分配峰值不含全部CPU页缓存。','']
 lines += ['## 判读口径与本轮结论','',ann['criterion'],'']+ann['findings']
 result=dict(experiment='priming160',conditions=COND,models=models,summary=summary,manual_coding=ann,base_report_sha256=hashlib.sha256(base.encode()).hexdigest(),source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['run_curvature_experiments.py','run_priming160.py','dream.svg']})
 (ROOT/'priming160_results.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
 (ROOT/'priming160_report.md').write_text('\n'.join(lines))
 (ROOT/'实验报告.md').write_text(base+MARK+'\n'.join(lines))
 (ROOT/'priming160_validation.json').write_text(json.dumps(dict(status='PASS',models=2,second_turn_records=36,first_turn_records=2,checks=['model and case counts','layer and token dimensions','160-token cap','EOS-free semantic spans','shared real first turn','hook output equality']),indent=2))
 print('WROTE 36 cases; appended supplement; original report prefix retained.')
if __name__=='__main__':main()
