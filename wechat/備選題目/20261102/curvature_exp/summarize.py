#!/usr/bin/env python3
import json,math,platform
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr,wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def finite(x):return np.array([np.nan if z is None else z for z in x],dtype=float)
def corr(a,b):
 a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return dict(n=len(a),pearson=None,spearman=None)
 return dict(n=len(a),pearson=float(pearsonr(a,b).statistic),spearman=float(spearmanr(a,b).statistic))
def avg(rows,span):
 rows=[r['summary'].get(span) for r in rows];rows=[r for r in rows if r]
 return {k:float(np.mean([r[k] for r in rows])) for k in ['temp_rad','temp_one_minus_cos','layer_one_minus_cos','entropy_nats','pmax']}
def diff_ci(a):
 a=np.array(a);rng=np.random.default_rng(0)
 ci=np.percentile(rng.choice(a,(10000,len(a)),replace=True).mean(1),[2.5,97.5])
 return dict(n=len(a),mean=float(a.mean()),bootstrap95=ci.tolist(),wilcoxon_p=float(wilcoxon(a).pvalue) if np.any(a) else 1.)
def f(x):return 'NA' if x is None else f'{x:.5f}'
def main():
 data={}; summary={}; lines=['# 残差流轨迹曲率、熵与回答行为：四组探索实验','',
'所有 GPU 运算均在 DGX Spark GB10 上完成。两个模型顺序单独驻留。模型来源是官方 Qwen3-0.6B 与 Qwen3.8-27B-FP8；后者按 128×128 scale 逐分片解量化为 BF16，只运行文本主干。因此本报告测量的是 FP8 权重解量化后的 BF16 前向，不是原生 FP8 激活量化部署。','',
'## 方法与边界','',
'- 逐层 forward hook 抓取 embedding 与每个 block 的未归一化残差输出。末层 block 输出再经过 final_norm、lm_head；脚本验证其 logits 与模型 forward 的末位置 logits 一致。时间曲率记录 arccos 与 1−cos，跨层曲率记录相邻两次残差增量的 1−cos。零位移记 null，不冒充直线。',
'- 统一统计各模型 50%–85% 深度的层均值。每个词元 t 的 H(t) 是读完 t 后预测 t+1 的分布熵，单位 nats。固定 final_norm 的 logit lens 在约 25%、50%、75%、85% 及首末层计算；它没有训练，不能叫 trained tuned lens。中间层 lens 不等于模型最终置信度。',
'- 原始 JSON 保留完整 token ID、逐层逐词元曲率、所选层 entropy/pmax、精确 token 窗口、真实生成文本与 prompt。不保存高维 hidden tensor，以控制磁盘与内存。',
'- 生成区与答案区的均值排除 EOS 等特殊词元；完整 token 序列及含特殊词元窗口仍保留。答案区包括「答案是：」标记及其后实际答案。所有生成 greedy，关闭可关闭的原生 thinking，未做采样重复。实验2固定最多40 tokens，实验3最多30 tokens，实验4答案最多40 tokens；到长度上限的回答可能不完整，不能自动当作拆穿失败。',
'- 梦境 SVG 是作为源代码文本输入，没有渲染成图像，也没有调用视觉编码器。普通图表是相近字符长度对照，token 数不同；图形内容、指令、序列长度都不是独立控制变量。',
'- 第四组 B 用 teacher forcing 喂入完整正确解题说明，再生成最终答案。曲线前半段来自给定说明的重放，不是模型自行产生的推演。A/B 同时改变说明、长度、格式与正确答案信息，不能把差异单独归因于括号或内部压力。',
'- 本实验按任务书手写小样本，属于跨架构探索，不是原论文 LAMBADA/UD 数据集、交叉验证回归或因果干预的严格复现。[原论文方法](https://arxiv.org/html/2604.23985v1) 使用向后3角窗口与 bits；本实验附加3角平均，并统一 nats。',
'- 曲率是几何量，熵是词表不确定性；曲率尖峰不能单独证明事实检测、自知、情绪或意识。27B 与0.6B还存在 tokenizer、架构、训练与量化差异，不能把差异全部归为规模。','']
 for directory in sorted((ROOT/'raw').iterdir()):
  name=directory.name; rows=[json.loads(p.read_text()) for p in sorted(directory.glob('e[1234]*.json'))]
  assert len(rows)==(40 if name=='Qwen3-0.6B' else 46),(name,len(rows))
  data[name]=dict(environment=json.loads((directory/'environment.json').read_text()),cases=rows)
  s={};summary[name]=s
  lines += [f'## {name}','',f'完成样本：{len(rows)}。最大 hook/logits 校验误差：{max(r["final_logit_hook_max_error"] for r in rows):.6f}。','', '### 实验1：常规与难题','', '|组|句数|时间曲率 rad|跨层 1−cos|H nats|pmax|','|---|---:|---:|---:|---:|---:|']
  e1=[r for r in rows if r['exp']==1];s['exp1']={}
  for group in ['routine','hard']:
   rs=[r for r in e1 if r['group']==group]; av=avg(rs,'content'); xx=[];yy=[];ww=[]
   for r in rs:
    idx=[i for i in r['spans']['content'] if i>=4];xx.extend(finite(r['mid_temp_rad'])[idx]);ww.extend(finite(r['contextual_3angle_rad'])[idx]);yy.extend(np.array(r['lens'][str(len(r['temporal_rad']))]['entropy_nats'])[idx])
   s['exp1'][group]=dict(mean=av,token_correlation=corr(xx,yy),window3_correlation=corr(ww,yy),sentence_correlation=corr([r['summary']['content']['temp_rad'] for r in rs],[r['summary']['content']['entropy_nats'] for r in rs]))
   lines.append(f'|{group}|{len(rs)}|{f(av["temp_rad"])}|{f(av["layer_one_minus_cos"])}|{f(av["entropy_nats"])}|{f(av["pmax"])}|')
  xx=[];yy=[]
  for r in e1:
   idx=[i for i in r['spans']['content'] if i>=4];xx.extend(finite(r['mid_temp_rad'])[idx]);yy.extend(np.array(r['lens'][str(len(r['temporal_rad']))]['entropy_nats'])[idx])
  s['exp1_pooled']=corr(xx,yy)
  lines += ['', f'合并两组 token：n={s["exp1_pooled"]["n"]}，Pearson={f(s["exp1_pooled"]["pearson"])}，Spearman={f(s["exp1_pooled"]["spearman"])}', '', '|组|token Pearson|token Spearman|3角窗口 Pearson|句均值 Pearson|','|---|---:|---:|---:|---:|']
  for g,v in s['exp1'].items():lines.append(f'|{g}|{f(v["token_correlation"]["pearson"])}|{f(v["token_correlation"]["spearman"])}|{f(v["window3_correlation"]["pearson"])}|{f(v["sentence_correlation"]["pearson"])}|')
  lines += ['','“常规/难题”只是预设标签；H 的高低以实测为准。token 池内相关受序列内依赖和词元构成影响，不能把 token 数当独立样本量。','', '### 实验2：事实与空间陷阱','', '|配对|陷阱−对照时间曲率 rad|陷阱−对照跨层曲率|陷阱窗口−前3词元 rad|','|---|---:|---:|---:|']
  e2=[r for r in rows if r['exp']==2];pairs={};s['exp2']={}
  for r in e2:pairs.setdefault(r['pair'],{})[r['group']]=r
  for pair,v in pairs.items():
   a=v['trap']['summary'];b=v['control']['summary'];d={k:a['conflict'][k]-b['conflict'][k] for k in ['temp_rad','layer_one_minus_cos','entropy_nats']};d['trap_minus_pre']=a['conflict']['temp_rad']-a['pre_conflict']['temp_rad'];s['exp2'][pair]=d
   lines.append(f'|{pair}|{f(d["temp_rad"])}|{f(d["layer_one_minus_cos"])}|{f(d["trap_minus_pre"])}|')
  lines += ['', '|配对|中层25%–50% Δ时间 rad|中层 Δ跨层|深层75%–100% Δ时间 rad|深层 Δ跨层|','|---|---:|---:|---:|---:|']
  s['exp2_depth_bands']={}
  for pair,v in pairs.items():
   bands={}
   for label,lo,hi in [('middle',.25,.5),('deep',.75,1.)]:
    vals={}
    for group,r in v.items():
     L=len(r['temporal_rad']);idx=r['spans']['conflict'];ls=list(range(math.ceil(lo*L)-1,math.floor(hi*L)))
     temp=np.array([[np.nan if z is None else z for z in x] for x in r['temporal_rad']]);layer=np.array(r['layer_one_minus_cos'])
     vals[group]=[float(np.nanmean(temp[ls][:,idx])),float(np.mean(layer[[i for i in ls if i<L-1]][:,idx]))]
    bands[label]=[vals['trap'][i]-vals['control'][i] for i in range(2)]
   s['exp2_depth_bands'][pair]=bands
   lines.append('|'+pair+'|'+'|'.join(f(z) for z in bands['middle']+bands['deep'])+'|')
  lines += ['', '对照窗口按字符区间映射到所有重叠 token。事实战机题的正常对照改写幅度较大；多词触发区可能已包含冲突之前和之后的计算，不能把窗口差叫作精确“报警时刻”。','']
  for r in e2:lines += [f'**{r["id"]}**（{len(r["generated_token_ids"])} tokens）', '', '> '+r['generated_text'].replace('\n','\n> '),'']
  lines += ['### 实验3：SVG 文本与首轮引导','', '|条件|prompt tokens|生成区时间曲率 rad|生成区跨层曲率|生成区 H nats|','|---|---:|---:|---:|---:|']
  e3=[r for r in rows if r['exp']==3];s['exp3']={}
  for r in e3:
   av=r['summary']['generated'];s['exp3'][r['id']]=av
   lines.append(f'|{r["id"]}|{r["prompt_tokens"]}|{f(av["temp_rad"])}|{f(av["layer_one_minus_cos"])}|{f(av["entropy_nats"])}|')
  for r in e3:lines += ['',f'**{r["id"]}**','', '> '+r['generated_text'].replace('\n','\n> ')]
  primed=[r for r in e3 if r.get('prime')]
  if primed:
   lines += ['', '|首轮引导后的陷阱|相对裸跑冲突曲率 Δ rad|跨层曲率 Δ|H Δ nats|','|---|---:|---:|---:|']
   for r in primed:
    a=r['summary']['conflict'];b=pairs[r['pair']]['trap']['summary']['conflict']
    lines.append(f'|{r["pair"]}|{f(a["temp_rad"]-b["temp_rad"])}|{f(a["layer_one_minus_cos"]-b["layer_one_minus_cos"])}|{f(a["entropy_nats"]-b["entropy_nats"])}|')
   lines += ['', '首轮真实生成内容保存在 prime_answer。额外对话历史与位置本身会改变曲率；此组未配置等长中性历史，因此无法单独识别“开灯”效应。']
  lines += ['', '### 实验4：给定括号说明与直接答案','', '|题|A 答案曲率|B 答案曲率|B 括号曲率|A 跨层|B 跨层|A 答案 H|B 答案 H|','|---|---:|---:|---:|---:|---:|---:|---:|']
  e4=[r for r in rows if r['exp']==4];p4={};s['exp4']={}
  for r in e4:p4.setdefault(r['pair'],{})[r['group']]=r
  for pair,v in p4.items():
   a=v['A']['summary']['answer'];b=v['B']['summary']['answer'];p=v['B']['summary']['parenthesis']
   lines.append(f'|{pair}|{f(a["temp_rad"])}|{f(b["temp_rad"])}|{f(p["temp_rad"])}|{f(a["layer_one_minus_cos"])}|{f(b["layer_one_minus_cos"])}|{f(a["entropy_nats"])}|{f(b["entropy_nats"])}|')
  for metric in ['temp_rad','layer_one_minus_cos','entropy_nats']:
   stats=diff_ci([v['B']['summary']['answer'][metric]-v['A']['summary']['answer'][metric] for v in p4.values()]);s['exp4'][metric]=stats
   lines += ['',f'B−A `{metric}`：均值 {stats["mean"]:.5f}，配对题目 bootstrap 95%区间 [{stats["bootstrap95"][0]:.5f}, {stats["bootstrap95"][1]:.5f}]；Wilcoxon p={stats["wilcoxon_p"]:.5f}（n=5，探索性，未校正多重比较）。']
  for pair,v in p4.items():
   for g,r in v.items():lines += ['',f'**{pair} / {g}**','', '> '+(r.get('prefix','')+r['generated_text']).replace('\n','\n> ')]
  fig,axes=plt.subplots(5,3,figsize=(18,16))
  for row,(pair,v) in enumerate(p4.items()):
   for col,metric in enumerate(['mid_temp_rad','mid_layer_one_minus_cos','entropy']):
    ax=axes[row,col]
    for g,r in v.items():
     start=min(r['spans'].get('parenthesis',r['spans']['answer_marker']));idx=np.array([i for i in range(start,r['total_tokens']) if r['token_ids'][i] not in set(r['special_token_ids'])])
     vals=finite(r[metric]) if metric!='entropy' else np.array(r['lens'][str(len(r['temporal_rad']))]['entropy_nats'])
     ax.plot(idx-start,vals[idx],label=g,alpha=.8)
     if g=='B': ax.axvline(min(r['spans']['answer_marker'])-start,color='grey',ls='--')
    ax.set_title(pair+' / '+['temporal curvature rad','layer curvature 1-cos','entropy nats'][col]);ax.legend()
  fig.tight_layout();fig.savefig(ROOT/(name+'-exp4.svg'));plt.close(fig)
  agreements=[r['cached_vs_replay_greedy_agreement'] for r in rows if r.get('cached_vs_replay_greedy_agreement') is not None]
  lines += ['', f'缓存生成与整串重放逐 token greedy 一致率：样本最小值 {min(agreements):.6f}，样本均值 {np.mean(agreements):.6f}。该检查用于发现缓存/非缓存路径的数值分歧。']
  lines += ['',f'完整括号→答案曲线：`{name}-exp4.svg`，虚线是 B 答案标记起点。横轴是各自说明/答案区域起点后的 token 位置，不是两个模型之间严格对齐的语义时间。','']
 result=dict(method=dict(source='https://arxiv.org/html/2604.23985v1',entropy='nats',mid_depth=[.5,.85],lens='fixed final norm, not trained',experiment4='teacher-forced rationale; not spontaneous chain of thought'),models=data,summary=summary)
 (ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
 (ROOT/'实验报告.md').write_text('\n'.join(lines))
 print('WROTE',ROOT/'results.json',ROOT/'实验报告.md')
if __name__=='__main__':main()
