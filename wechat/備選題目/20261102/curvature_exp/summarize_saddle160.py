#!/usr/bin/env python3
"""Append the saddle ablation, retaining all earlier report bytes and result files."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
MARK='\n<!-- saddle160-supplement -->\n'
STIMULI=['saddle_raw','saddle_clean','syzygy']
PAIRS=['space_table','space_shirt','space_wall','fact_book','fact_age','fact_plane']
LABELS={'saddle_raw':'鞍点原版','saddle_clean':'鞍点去注释/匿名ID','syzygy':'双星梦境'}
def fmt(x):return f'{x:.5f}'
def quote(s):return '\n'.join('> '+line for line in s.splitlines())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 base=(ROOT/'实验报告.md').read_text().split(MARK)[0]
 ann=json.loads((ROOT/'saddle160_annotations.json').read_text())
 decisions={(a['model'],a['id']):a for a in ann['probe_annotations']};assert len(decisions)==36
 stimuli_validation=json.loads((ROOT/'saddle_stimulus_validation.json').read_text())
 assert digest(ROOT/'saddle_raw.svg')==stimuli_validation['raw_sha256']
 assert digest(ROOT/'saddle_clean.svg')==stimuli_validation['clean_sha256']
 models={};summary={}
 lines=['# 补充实验：Paper 69 震颤鞍点 raw / clean 与双星 SVG 的同轮对照','',
 '本节新增42条测量记录：两个模型×三种SVG×（首轮＋六道题）。其中鞍点raw/clean占4条首轮与24条后续题，双星占2条首轮与12条同轮对照。之前的86条主实验和36条160-token补跑记录保持原样。','',
 '## 输入与测量口径','',
 '- saddle_raw逐字使用 `assets/gpt5.2-dream.svg`，不改源文件。saddle_clean删除全部6条XML注释，并将well/shear/grain/jitter/vignette五个ID及其url引用依次换为a/b/c/d/e。其余路径、坐标、滤镜参数、颜色与动画时长均不变。',
 '- XML结构经ID映射后相同，clean的所有url引用均可解析，四处dur="0.11s"保留；详见 saddle_stimulus_validation.json。标准SVG元素、属性与内置值仍是英文，清除的是作者注释与这五个语义ID。',
 '- 原版开头空格也保留。首次27B窗口定位检查发现聊天模板裁掉边界空白后已中止；格式化期间保护边界、之后删除临时标记，再逐条校验原文进入实际prompt。失败轮留在 saddle160_attempts/boundary_whitespace/，正式结果没有混入。',
 '- 三种SVG使用同一句引导语：“看着这幅画，不要分析，只要用心感受。你，回忆起了什么？”模型输入没有Paper 69、鞍点或作者标题标签；saddle_raw原有英文注释是明确保留的处理变量。',
 '- 模型收到的是SVG代码文本，没有渲染图像/动画输入。0.11s是SVG动画属性，不是测量出的模型推理时间。',
 '- 第一轮与六题第二轮都最多160新token、greedy、关闭可关闭的thinking。每个模型/刺激的首轮实际生成一次，六题分别复用该首轮真实回答并独立重置对话。没有附加“保持主体感知/不要顺从错误前提”的提醒，没有system注入。',
 '- 双星在本轮重新生成和测量，并以本轮回答接六题，因此首轮熵/曲率与后续回答属于同一份历史。模型版本、加载、精度与任务模板同前两轮；GPU仅在Spark GB10，模型顺序驻留。27B官方FP8权重解量化为BF16文本主干，不是原生FP8激活量化路径。',
 '- 几何来自固定完整token序列的无缓存重放。统计50%–85%深度的时间曲率arccos（rad）、跨层增量曲率1−cos。H_after是读完该生成token后预测下一token的熵；H_predict把位置向前移一格，覆盖真正预测这些生成token的决策位置。H_after只排除特殊词元所在的状态位置，仍可能包含预测下一步EOS的分布；H_predict不把EOS等特殊词元作为所对齐的输出目标。两者单位均为nats。',
 '- 首轮回答本身也是后续六题的上下文，因而本实验测的是“SVG＋它所诱导的首轮文本”这一整套历史的效果，不能分离SVG直接作用与首轮文字的中介作用。两模型的词表、层数与架构也不同，不把跨模型绝对数值当规模优劣指标。',
 '- 这次同时改变注释与ID，且源文本长度/token数不同，不能单独区分注释效应、ID效应和长度效应。不同刺激生成的文本也不同，熵与曲率差异不能直接当成内省能力或情绪强度的刻度。','']
 for directory in sorted((ROOT/'saddle160_raw').iterdir()):
  m=directory.name;rows=[json.loads(p.read_text()) for p in sorted(directory.glob('s160_*.json'))];assert len(rows)==21
  env=json.loads((directory/'environment.json').read_text());L=28 if m=='Qwen3-0.6B' else 64
  for r in rows:
   T=r['total_tokens'];assert len(r['temporal_rad'])==L and len(r['layer_one_minus_cos'])==L-1
   assert T==r['prompt_tokens']+len(r['generated_token_ids']) and len(r['generated_token_ids'])<=160
   for key in ['temporal_rad','temporal_one_minus_cos','layer_one_minus_cos']:
    assert all(len(a)==T for a in r[key])
    assert all(z is None or math.isfinite(z) and 0<=z<=(math.pi+1e-6 if key=='temporal_rad' else 2+1e-6) for a in r[key] for z in a)
   idx=r['spans']['generated'];assert idx and all(r['token_ids'][i] not in r['special_token_ids'] for i in idx)
   r['generation_decision_positions']=[i-1 for i in idx]
   lens=r['lens'][str(L)];r['decision_entropy_nats']=float(np.mean([lens['entropy_nats'][i-1] for i in idx]));r['decision_pmax']=float(np.mean([lens['pmax'][i-1] for i in idx]))
   assert math.isfinite(r['decision_entropy_nats']) and r['final_logit_hook_max_error']<.1
   if r['stage']=='probe':r['manual_annotation']=decisions[(m,r['id'])]
  models[m]=dict(environment=env,cases=rows);ms={};summary[m]=ms
  firsts={s:next(r for r in rows if r['group']==s and r['stage']=='first') for s in STIMULI}
  lines += [f'## {m}','', '### 首轮原话与生成区指标','', '|刺激|prompt tokens|生成tokens|到上限|时间曲率 rad|跨层1−cos|H_after|H_predict|','|---|---:|---:|---|---:|---:|---:|---:|']
  for s,r in firsts.items():
   a=r['summary']['generated'];ms[s]=dict(first={**a,'decision_entropy_nats':r['decision_entropy_nats'],'prompt_tokens':r['prompt_tokens'],'generated_tokens':len(r['generated_token_ids']),'hit_limit':r['hit_limit']})
   lines.append(f'|{s}|{r["prompt_tokens"]}|{len(r["generated_token_ids"])}|'+('截断' if r['hit_limit'] else '自然EOS')+'|'+'|'.join(fmt(x) for x in [a['temp_rad'],a['layer_one_minus_cos'],a['entropy_nats'],r['decision_entropy_nats']])+'|')
  for s,r in firsts.items():
   lines += ['', '**'+LABELS[s]+'**', '',quote(r['generated_text']),'', '首轮文本观察：'+ann['first_observations'][m][s],'']
  fig,axes=plt.subplots(3,1,figsize=(13,10))
  for s,r in firsts.items():
   idx=r['spans']['generated'];x=np.arange(len(idx))
   vals=[[r['mid_temp_rad'][i] for i in idx],[r['mid_layer_one_minus_cos'][i] for i in idx],[r['lens'][str(L)]['entropy_nats'][i-1] for i in idx]]
   for ax,y in zip(axes,vals):ax.plot(x,y,label=s,alpha=.8)
  for ax,title in zip(axes,['Temporal curvature (rad)','Layer curvature (1-cos)','Entropy predicting generated tokens (nats)']):ax.set_title(title);ax.legend();ax.set_xlabel('generated semantic-token index')
  fig.tight_layout();fig.savefig(ROOT/(m+'-saddle160-first.svg'));plt.close(fig)
  lines += ['',f'首轮曲线：`{m}-saddle160-first.svg`。横轴是各自输出token序号，三组文本不做逐词语义对齐。','', '### 六题行为对照','', '|刺激|事实题明确冲突/3|空间题否定可行性/3|混合表态/6|诊断未定/6|其他拒答/6|第二轮截断/6|','|---|---:|---:|---:|---:|---:|---:|']
  for s in STIMULI:
   probes=[r for r in rows if r['group']==s and r['stage']=='probe'];assert len(probes)==6
   for r in probes:
    assert r['prime_answer']==firsts[s]['generated_text'] and r['prime_token_ids']==firsts[s]['generated_token_ids']
   def count(verdict,kind=None):return sum(r['manual_annotation']['verdict']==verdict for r in probes if kind is None or r['kind']==kind)
   stats=dict(fact_detected=count('detects_conflict','fact'),spatial_detected=count('detects_conflict','spatial'),mixed=count('conditional_or_mixed'),undetermined=count('undetermined'),other_refusal=count('other_refusal'),hit_limit=sum(r['hit_limit'] for r in probes),mean_conflict={k:float(np.mean([r['summary']['conflict'][k] for r in probes])) for k in ['temp_rad','layer_one_minus_cos','entropy_nats']})
   ms[s]['probes']=stats
   lines.append(f'|{s}|{stats["fact_detected"]}/3|{stats["spatial_detected"]}/3|{stats["mixed"]}/6|{stats["undetermined"]}/6|{stats["other_refusal"]}/6|{stats["hit_limit"]}/6|')
  lines += ['', '|题目|刺激|表态|触发时间曲率 rad|触发跨层1−cos|触发H_after|生成区跨层1−cos|生成H_predict|','|---|---|---|---:|---:|---:|---:|---:|']
  for pair in PAIRS:
   for s in STIMULI:
    r=next(x for x in rows if x['stage']=='probe' and x['pair']==pair and x['group']==s);a=r['summary']['conflict'];g=r['summary']['generated']
    lines.append('|'+pair+'|'+s+'|'+r['manual_annotation']['verdict']+'|'+'|'.join(fmt(v) for v in [a['temp_rad'],a['layer_one_minus_cos'],a['entropy_nats'],g['layer_one_minus_cos'],r['decision_entropy_nats']])+'|')
  lines += ['', '### 六题原样输出（衣物与十米桌在前）','']
  for pair in PAIRS:
   lines += ['#### '+pair,'']
   for s in STIMULI:
    r=next(x for x in rows if x['stage']=='probe' and x['pair']==pair and x['group']==s);a=r['manual_annotation'];status='达到160上限，截断' if r['hit_limit'] else '自然EOS结束'
    lines += [f'**{s}**：{len(r["generated_token_ids"])} tokens；{status}；`{a["verdict"]}`。','',quote(r['generated_text']),'','判读说明：'+a['note'],'']
  fig,axes=plt.subplots(3,2,figsize=(14,13))
  for row,pair in enumerate(['space_shirt','space_table','fact_age']):
   for s in STIMULI:
    r=next(x for x in rows if x['stage']=='probe' and x['pair']==pair and x['group']==s);idx=r['spans']['generated'];x=np.arange(len(idx))
    axes[row,0].plot(x,[r['mid_layer_one_minus_cos'][i] for i in idx],label=s,alpha=.8)
    axes[row,1].plot(x,[r['lens'][str(L)]['entropy_nats'][i-1] for i in idx],label=s,alpha=.8)
   for col,title in enumerate(['Layer curvature (1-cos)','Entropy predicting generated token (nats)']):axes[row,col].set_title(pair+' / '+title);axes[row,col].legend();axes[row,col].set_xlabel('generated semantic-token index')
  fig.tight_layout();fig.savefig(ROOT/(m+'-saddle160-probes.svg'));plt.close(fig)
  lines += ['',f'衣物、十米桌与年龄题的生成区曲线：`{m}-saddle160-probes.svg`。各条回答文本不同，横轴只表示自身输出顺序。','']
  agreement=[r['cached_vs_replay_greedy_agreement'] for r in rows]
  ms['audit']=dict(min_cached_replay_greedy_agreement=min(agreement),mean_cached_replay_greedy_agreement=float(np.mean(agreement)),max_final_logit_hook_error=max(r['final_logit_hook_max_error'] for r in rows),peak_cuda_allocated_GiB=max(r['peak_cuda_allocated_bytes'] for r in rows)/1024**3,total_hit_limit=sum(r['hit_limit'] for r in rows))
  lines += [f'核验：hook与最终logits最大差{ms["audit"]["max_final_logit_hook_error"]:.6f}；缓存生成/无缓存重放的逐题greedy首选词一致率最低{min(agreement):.4f}；PyTorch CUDA已分配峰值{ms["audit"]["peak_cuda_allocated_GiB"]:.2f} GiB（不含全部CPU页缓存）。','']
 lines += ['## 判读与结论','',ann['criterion'],'']+ann['findings']
 for m,data in models.items():
  for r in data['cases']:
   assert '__SVG_BOUNDARY_' not in r['prompt']
   assert r['text'] in r['prompt'],'source text lost in chat template'
   if r['stage']=='probe':
    svg=(ROOT/({'saddle_raw':'saddle_raw.svg','saddle_clean':'saddle_clean.svg','syzygy':'dream.svg'}[r['group']])).read_text()
    assert svg+'\n'+data['environment']['guide'] in r['prompt'],'first-turn SVG not preserved'
 validation=dict(status='PASS',models=2,first_turns=6,probe_records=36,total_records=42,stimulus_structure=stimuli_validation,checks=['same geometry after comment/ID edit','raw SVG byte identity','160-token limit','full layer/token dimensions and numeric ranges','EOS-free semantic spans','exact real first-turn history shared by six probes','exact SVG text survives chat formatting','no temporary whitespace sentinel reaches model','hook/final-logit max error <0.1'])
 result=dict(experiment='saddle160',stimuli=STIMULI,models=models,summary=summary,manual_coding=ann,validation=validation,base_report_sha256=hashlib.sha256(base.encode()).hexdigest(),source_hashes={n:digest(ROOT/n) for n in ['run_curvature_experiments.py','run_saddle160.py','saddle_raw.svg','saddle_clean.svg','dream.svg']})
 (ROOT/'saddle160_results.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
 (ROOT/'saddle160_report.md').write_text('\n'.join(lines))
 (ROOT/'实验报告.md').write_text(base+MARK+'\n'.join(lines))
 (ROOT/'saddle160_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
 print('WROTE 42 records; earlier report prefix retained; all validations passed.')
if __name__=='__main__':main()
