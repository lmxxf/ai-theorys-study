# 316 核查记录（2026-09-21）

保留“产品实用性不等于架构新范式”的论点与原章节顺序。本轮核查公开材料，未付费调用 Jev，未重做业务测试。落款9月22日沿用次日发布安排。

## 官方接口和宣传

- [API](https://docs.typesafe.ai/api)：原稿请求/响应字段和值与示例一致。注释块改为 JSONC，检查去注释后可解析。choice最多255项；[score](https://docs.typesafe.ai/api) 为有序档位概率的加权平均，不只是选一个档。受约束的是输出题型，state和问题说明可嵌套。
- [Models](https://docs.typesafe.ai/models)：一次读入state，问题并行评估；没有证据把一个query写成一次forward。输出免费是计费规则；删除34 output tokens恒定与输出无成本断言。
- [发布博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)：宣称新架构、并行采样和RLCD，但未提供可复现结构。首页倍数来自自建workflow评测，官方自己称可能处于真实收益高端。大模型概率输出适配器比只给离散答案更慢更贵。
- [首页](https://typesafe.ai/) 的折叠FAQ承认可以选错：保证shape，不保证每个decision正确。[完整FAQ模块](https://framerusercontent.com/sites/43bTeC8cU9jZO20XvdK79t/dGJLAdQuVoUx6UaFHNr_eoAZeRAF2Z5vA5nm8BY6Eao.BiQ9PASq.mjs)包含被网页抽取漏掉的回答。正文改为“大字零幻觉、小字承认选错”，不再暗示官方从未承认。
- [Confidence](https://docs.typesafe.ai/confidence)及[原始MDX](https://docs.typesafe.ai/confidence.md)：三选项公式是交互demo的approximation，不是生产API公式。0.88代入得0.82，并不等于API示例0.81，故不把二者硬接成实际算法。
- [Classification Using Confidence](https://docs.typesafe.ai/cookbooks/classification_using_confidence)：60份年报、75类；jev-1.12，2026-08-12。confidence≥0.9为27/30正确，低于为12/30；改报更粗类别后21/30。纠正“没有任何量化验证”。这是阈值分流证据，不是完整可靠性曲线或跨任务ECE。
- [AI Primer](https://docs.typesafe.ai/introduction/machine-learning-primer)：RLCD为Reinforcement Learning for Calibrated Decisions，说明目标与输出契约，不能据此恢复内部架构。

## 倍数与采用率

- [Workflow Evals](https://evals.typesafe.ai/)：四工作流等权平均；参考为GPT-6 Astra与Claude Fable 5.1高思考回答的平均，其余模型用提供商默认设置。比较的是相对模型共识，不是独立人工金标准。
- [开源适配器](https://github.com/typesafe-ai/system-one-adapter-python)：可选概率/离散答案及原生结构化输出。厂商不是完全没给复核材料；但workflow条件不能升级成任意任务的领先倍数。
- [反刷榜博客](https://typesafe.ai/blog/antibenchmaxxing)：实际题名Lies, Damned Lies, and Benchmarks，2026-09-11，反对标准榜单优化，提出公开内部评测及局限。
- [Vercel采用率](https://vercel.com/blog/ai-gateway-jev-model-launch)、[上架公告](https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway)：13%及2倍比较各模型上线首24小时、Gateway付费团队采用比例，上线有免费促销；不是全市场份额或当前总调用量。

## 第三方原始证据

- [Vercel Pranit 原推及附图](https://x.com/fazxes/status/2100300097695232164)：70案例×3=210判定；Jev207/210、98.6%，Luna203/210、96.7%；中位312/1458ms，p95为374/6583ms。5–18倍对应不同统计量。附图已目视核对，撤回“只有两句话没有数字”。UTC 9月16日19:05，CST为9月17日03:05。
- [Bryo首帖](https://x.com/nikhilmudholkar/status/2100604560335139083)、[数据构成](https://x.com/nikhilmudholkar/status/2100605143993503771)：1565=1201真实+364合成，10类，以德语为主；不是全真实生产邮件。
- [准确率](https://x.com/nikhilmudholkar/status/2100605383966396442)：Jev96.4%、Gemini3.5 Flash-Lite97.5%、Gemini3.8 Flash98.5%。
- [置信度分组](https://x.com/nikhilmudholkar/status/2100605870409195985)：737个≥99%的预测全部匹配参考标签，低于70%近半错；[成本](https://x.com/nikhilmudholkar/status/2100606298630816018)每千封0.08/0.80/1.79美元；[局限](https://x.com/nikhilmudholkar/status/2100606801028764054)未处理附件。发布为CST9月17日23:15起。
- [TechCrunch报道](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/)仅用于对照宣传与转述措辞，性能数据回原帖核。

## HN与履历

- [主讨论](https://news.ycombinator.com/item?id=49717558)：抓取时1930分。逐条核了[速度比较](https://news.ycombinator.com/item?id=49718120)、[裸数字](https://news.ycombinator.com/item?id=49720601)、[noul释义](https://news.ycombinator.com/item?id=49718407)、[随机森林回应](https://news.ycombinator.com/item?id=49718767)、[类型安全评论](https://news.ycombinator.com/item?id=49719001)。
- [创始人下一条回复](https://news.ycombinator.com/item?id=49719080)明确同意类型安全不等于事实正确，正文补入。
- [架构不公开](https://news.ycombinator.com/item?id=49718824)、[encoder猜测](https://news.ycombinator.com/item?id=49718313)、[公布分数的讽刺](https://news.ycombinator.com/item?id=49718242)、[校准覆盖率请求](https://news.ycombinator.com/item?id=49742459)可核。原稿“最平衡的评论”整段未找到对应原话，改成作者分析；没有把未找到升级成断言不存在。
- [InstructGPT](https://arxiv.org/pdf/2203.02155)确认Almeida第四作者、Primary authors；[2017论文](https://arxiv.org/abs/1706.03741)、[2019语言偏好训练](https://arxiv.org/abs/1909.08593)、[2020摘要](https://arxiv.org/abs/2009.01325)构成更早谱系，不能以InstructGPT为RLHF首次用于语言模型。
- [公司发布新闻稿](https://www.businesswire.com/news/home/20260915525333/en/TypeSafe-AI-Emerges-From-Stealth-With-$40M-in-Funding-With-New-Model-for-Composable-AI)、[本人发布推文](https://x.com/CompleteSkeptic/status/2099925682726002904)：官方与本人都称共同发明ChatGPT。撤回“官方从没这么写，是媒体加码”。

## 验证

原稿两条正文URL保留且不重复；JSONC去注释可解析；粗体、代码围栏、摘要长度与`git diff --check`检查。正文不动暂存区、不提交。其他工作区修改未处理。
