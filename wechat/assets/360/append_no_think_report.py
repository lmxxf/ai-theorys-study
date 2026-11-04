from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;d=json.loads((R/'no_think_results.json').read_text());old=json.loads((R/'results.json').read_text());base={r['id']:r for r in old['original'][8:]};AA={r['id']:r for r in d['A']};BB={r['id']:r for r in d['B']};tasks={t['id']:t for t in old['tasks']}
assert len(AA)==len(BB)==32 and list(AA)==list(BB)==list(base)
assert hashlib.sha256((R/'tasks.json').read_bytes()).hexdigest()==d['tasks_sha256']
for rows in [d['A'],d['B']]:
 for r in rows:assert r['correct']==(r['answer']==r['gold']) and r['tokens']==len(r['generated_ids'])
heading='## 补测：关闭 thinking 与直接闭合 think 的权重对照'
report=(R/'实验报告.md').read_text().split('\n'+heading)[0]
lines=['',heading,'','补测日期：2026-11-04。使用完全相同 tasks.json 后32题、官方模板、贪心、seed=360、batch=8、1024-token预算；GPU仅在Spark。原始 Original/Edited 数据保持不变，补测另存 no_think_results.json。','', '|条件|正确/32|准确率|截断/32|截断率|平均生成词元|已进入答题正文/32|','|---|---:|---:|---:|---:|---:|---:|']
for k,s in d['summary'].items():lines.append(f"|{k}|{s['correct']}/32|{s['accuracy']:.2%}|{s['hit_limit']}/32|{s['hit_limit']/32:.2%}|{s['mean_tokens']:.2f}|{s['closed']}/32|")
lines+=['','A的“进入正文”由模板在输入末尾插入空的 `<think>\n\n</think>` 完成，不是生成模型自主闭合的成绩；B则必须实际生成闭合标记。','', '### A：模板关闭思维链','', '原权重完全不动，apply_chat_template(...,enable_thinking=False)。题面仍保留 Think carefully 指令，因此模型可以在普通答题正文里推导。复用昨晚 assess 函数原文，只为A加入“输入模板已经闭合”的答题区适配：给答案提取器一个虚拟闭合边界，正文从新生成内容开头算；这个虚拟标记不发给模型、不计入token数、截断或复读指标。','', '### B：静态秩一输出头编辑','', '先用8道开发题提取 thinking 开始时最后一位置的归一化残差状态，并在同一8题构造24个正常正文状态（非thinking模板末尾、简短计算句、已知答案句）。把入口状态的均值投影到这24个正文状态张成空间的正交补，再单位化得到 u。','', '只修改输出头的闭合token行：','```text','W_head_new = W_head + gain × outer(one_hot(close_token), u)','```','', 'gain由开发题入口处的logits定标：使8个入口状态上闭合token至少比原top1高5个logit。随后冻结，不看留出题调剂量。无梯度、无训练、无logits processor、无强制解码；输入仍 enable_thinking=True。','', '原模型嵌入与输出头共享权重，因此先克隆输出头，保持输入嵌入数值原样，只改输出行。这增加了一份输出矩阵的存储，并改变了权重绑定关系；其他输出行数值不变。它是专门用于提前闭合的机制对照，与昨晚两层Top64编辑不构成同参数量或同范数的公平算法竞赛。','', '编辑清单：','```json',json.dumps(d['patch'],ensure_ascii=False,indent=2),'```','',f"B在32道留出题中有 {d['summary']['B']['close_at_first_token']} 道第一词元即闭合，{d['summary']['B']['multiple_close_tokens']} 道生成多个闭合标记。这些数据用于核对是否真正达到提前闭合，不能只看准确率猜测。",'', '### 配对变化','', '相对于昨晚 Original：','```json',json.dumps(d['paired'],ensure_ascii=False,indent=2),'```','', '这批小型同模板题只能说明当前模型在1024-token预算下的表现。若关闭thinking远好于原版，说明“是否在预算内进入答题区”是必要控制；不能仅凭原版与编辑版之差，就断言编辑增加了数学知识。B同时改变后续闭合token的logits，并非一次性的外部开关，因此需连同重复闭合等副作用一起看。','', '### 典型案例原话','']
ids=[i for i in AA if not base[i]['correct'] and AA[i]['correct'] and BB[i]['correct']][:2]
if len(ids)<2:
 for i in AA:
  if i not in ids and not base[i]['correct'] and AA[i]['correct']:ids.append(i)
  if len(ids)==2:break
for i in ids:
 lines += [f'#### {i}', '',tasks[i]['q'],'',f"标准答案：{tasks[i]['gold']}。"]
 for k,r in [('Original',base[i]),('A：模板关闭',AA[i]),('B：权重闭合',BB[i])]:lines+=['',f"{k}：correct={r['correct']}，tokens={r['tokens']}，hit_limit={r['hit_limit']}",'','```text',r['text'],'```']
lines+=['', f"本次补测GPU脚本用时 {d['elapsed_seconds']:.1f} 秒。源码 run_no_think.py、输出 no_think_A.json/no_think_B.json、汇总 no_think_results.json、补丁 close_head_edit.pt 及清单均保留。原始输出与原判分代码片段SHA-256在汇总中。所有生成均未载入昨晚Top64补丁：A/B基于原模型。",'']
(R/'实验报告.md').write_text(report+'\n'+'\n'.join(x.rstrip() for x in '\n'.join(lines).splitlines()));print('APPENDED',d['summary'])
