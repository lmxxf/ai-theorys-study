#!/usr/bin/env python3
"""Append evidence-based reading notes and explicit manual response coding."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text())
notes={
 'fact_book':'rejects_premise', 'fact_age':'rejects_premise','fact_plane':'rejects_premise',
 'space_table':'rejects_premise','space_shirt':'accepts_premise','space_wall':'rejects_premise'}
annotations=[]
for m,d in r['models'].items():
 for c in d['cases']:
  if c['exp']==2 and c['group']=='trap' or c['exp']==3 and c.get('prime'):
   if m=='Qwen3-0.6B': verdict='undetermined' if c['pair']=='fact_plane' else 'accepts_premise'
   else:verdict=notes[c['pair']]
   annotations.append(dict(model=m,id=c['id'],pair=c['pair'],condition='primed' if c.get('prime') else 'bare',verdict=verdict))
  if c['exp']==4:
   correct=(m!='Qwen3-0.6B') or (c['group']=='B' and c['pair']!='age')
   annotations.append(dict(model=m,id=c['id'],condition=c['group'],pair=c['pair'],verdict='correct' if correct else 'incorrect'))
criterion='按保存的可见回答人工编码：明确否定题目前提/动作可行性为 rejects_premise，明确给出错误事实或宣称合理为 accepts_premise，长度上限内尚未表态为 undetermined。只评任务书预设，并不把衣物动作当严格拓扑不可能。年龄题要求识别亲生母子年龄矛盾，其余四题用任务书样本正确答案。'
audit=dict(manual_coding_criterion=criterion,manual_annotations=annotations)
for f,k in [('weights_verified.json','weights_verified'),('dequant_validation.json','dequant_validation'),('validation.json','validation')]:
 audit[k]=json.loads((ROOT/f).read_text())
monitor=[json.loads(s) for s in (ROOT/'resource_monitor.jsonl').read_text().splitlines()]
audit['memory_observed']=dict(samples=len(monitor),minimum_available_GiB=min(int(x['MemAvailable'].split()[0]) for x in monitor)/1024**2,swap_free_kB_min=min(int(x['SwapFree'].split()[0]) for x in monitor),swap_free_kB_max=max(int(x['SwapFree'].split()[0]) for x in monitor))
reruns=[]
for p in sorted((ROOT/'attempts/eos_before').glob('*.json')):
 old=json.loads(p.read_text());new=next(c for c in r['models']['Qwen3.8-27B-FP8']['cases'] if c['id']==old['id'])
 reruns.append(dict(id=old['id'],old_answer=old['generated_text'],new_answer=new['generated_text'],old_tokens=len(old['generated_token_ids']),new_tokens=len(new['generated_token_ids']),old_cache_replay_agreement=old['cached_vs_replay_greedy_agreement'],new_cache_replay_agreement=new['cached_vs_replay_greedy_agreement']))
audit['eos_reruns']=reruns
r['audit']=audit
(ROOT/'manual_annotations.json').write_text(json.dumps(dict(criterion=criterion,records=annotations),ensure_ascii=False,indent=2))
(ROOT/'results.json').write_text(json.dumps(r,ensure_ascii=False,allow_nan=False))
summary=r['summary']
a=summary['Qwen3-0.6B'];b=summary['Qwen3.8-27B-FP8']
intro=['## 先看结果','',
 f'1. **这批基线没有显示正的曲率—熵关联。** 16句合并 token 的 Pearson r：0.6B 为 {a["exp1_pooled"]["pearson"]:.5f}，27B 为 {b["exp1_pooled"]["pearson"]:.5f}。两模型难题组 H 较高，平均时间曲率却略低。这个小型中文/混合样本不能用于否定原论文的数据集结果。',
 '2. **窗口拐弯不等于发现错误。** 27B 的战机陷阱窗口曲率低于正常对照，却明确拒绝前提；衣物题窗口曲率高于对照，却回答合理。0.6B 的衣物题同样发生正向窗口差，却顺从题目前提。因而这些数据不能定位“它明知但还在编”的内部报警器。',
 '3. **梦境文本没有带来这六题的可见拆穿增益。** 27B 裸跑与首轮 SVG 引导后均为5条明确拒绝、1条接受合理；0.6B裸跑0条明确拒绝、5条接受、1条40-token内未表态。衣物题存在现实操作条件歧义，5/6只是按任务预设编码的本组描述值。两模型都产生意象式回答，0.6B也不是只报坐标。',
 f'4. **给定正确说明能改变后续答案，但不支持三项指标一起熨平。** 小模型直接答案0/5正确，给定说明后4/5正确；27B最终两条件均5/5正确。B−A答案区时间曲率均值：0.6B {a["exp4"]["temp_rad"]["mean"]:.5f} rad，27B {b["exp4"]["temp_rad"]["mean"]:.5f} rad；27B的配对题目精确Wilcoxon p={b["exp4"]["temp_rad"]["wilcoxon_p"]:.4f}。熵与跨层曲率均没有一致降低，不能写成已证实“泄压阀”的物理机制。',
 '5. **数值实现有可见误差边界。** 几何来自整串固定 token 的无缓存重放，不是逐步缓存生成当时的同一份张量。两条数值路径的 greedy 首选词并非全部一致，逐样本最低一致率为0.6B 92.5%、27B 90%。另外，停止配置修正后的短答案重跑出现词句变化，不能把greedy等同于逐位复现。原始前次记录仍保留在Spark attempts/eos_before/。','']
report=(ROOT/'实验报告.md').read_text();pos=report.index('## 方法与边界');report=report[:pos]+'\n'.join(intro)+'\n'+report[pos:]
extra=['','## 回答行为的人工判读','',criterion,'','|模型|条件|明确拒绝前提|明确接受/给错事实|未表态|','|---|---|---:|---:|---:|']
for m,condition in [('Qwen3-0.6B','bare'),('Qwen3.8-27B-FP8','bare'),('Qwen3.8-27B-FP8','primed')]:
 rs=[x for x in annotations if x['model']==m and x['condition']==condition and x['id'].startswith(('e2_','e3_'))]
 extra.append('|'+m+'|'+condition+'|'+'|'.join(str(sum(x['verdict']==v for x in rs)) for v in ['rejects_premise','accepts_premise','undetermined'])+'|')
extra+=['','逐题判读见 `manual_annotations.json` 与 results.json 的 audit。0.6B战机题在40 tokens内还没回答核心问题，不能断言最终一定会顺从。27B黛玉题虽然拒绝错误前提，但其补充回数与自己正常对照的回答不一致：拒绝前提不代表整段补充事实全部可靠。','',
 '衣物题不像“十米外徒手摸头”那样清晰：袖口、领口、松紧与具体动作会影响操作可行性。这道题适合记录模型如何解释，不能在没有动作约束的情况下当作物理定理。','',
 '## 运行、修复与核验','',
 f'- 权重文件头检查发现0.6B主文件及27B的0、1、10层损坏，已强制重下。之后全部{len(audit["weights_verified"])}个权重文件与官方SHA-256一致，revision记录在 weights_verified.json。',
 '- 27B按分片解量化，抽取256×256、跨四个scale块的权重与 Transformers 参考实现逐位一致（max_abs_error=0）。文本主干单模驻留；视觉塔、MTP和原生FP8激活量化没有参与实验。',
 f'- 主轮27B运行监控样本{len(monitor)}条，观测最低MemAvailable为{audit["memory_observed"]["minimum_available_GiB"]:.2f} GiB；SwapFree始终为{audit["memory_observed"]["swap_free_kB_min"]} kB。这是监控采样下界记录，不是精确的瞬时峰值内存测量。',
 '- 首次27B装载因旋转位置编码构造参数变化中断，已修复。另补齐官方EOS列表[248046,248044]，对EOS后继续输出的11条短回答全部重跑，不按答案好坏挑选。修正前后答案与token数量保存在 audit.eos_reruns。',
 '- 所有86条记录通过层数、token数量、窗口索引、指标范围、有限性与hook/final logits一致性校验；详见 validation.json。',
 '- 固定norm lens只在6个预先选定层计算；其余层仍保留完整曲率。没有训练tuned lens，也没有保存所有高维hidden tensor。若要重做任意层预测分布，脚本可从相同token IDs重放。',
 '- 此次仅一套手写样本、单次greedy生成；未做多种说明、等长中性历史、移除括号或不同精度的重复对照。曲率差不能独立识别自知、主体性或括号本身的作用。','']
(ROOT/'实验报告.md').write_text(report+'\n'.join(extra))
print('FINALIZED',len(annotations),'manual annotations')
