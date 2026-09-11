# 306 终审（2026-09-11）

直接修订 `306.md`；原始标题“三个人、三天、三句话”改为“三个人、两种选择”。没有操作 index、commit 或其他期号。

## 证据目录

- [Pachocki 官方全文](audit-pachocki.md)：已核到正文结尾和尾注，非只读新闻摘要。
- [Coxon / Hubinger](audit-coxon-hubinger.md)：身份、原帖、日期、后续采访与中文传播样本。
- [Hinton / 技术机制 / 历史](audit-hinton-mechanisms.md)：访谈时间、概率判断、CoT、位置偏置、2023声明。
- `x-2097476196791709843.json`、`x-2097497037956891126.json`：X官方嵌入接口的原始返回。
- `coxon-web-tool-recheck.json`：子代理第二次实读 PANews、nucleovisual、Fast Company 的网页工具返回。主审直接访问前两页403，但已检查保存返回的正文，未用URL推断标题。Fast Company L226/L241支持公司文化与Slack转引。

## 核心事实修正

|原稿|证据|修订|
|---|---|---|
|Coxon 9/8、Hubinger 9/9，三天|X created_at 为9/9 00:04:29Z、01:27:18Z|两帖北京时间同为9/9，间隔83分钟；官网日期另标，标题改两种选择|
|Hubinger公司安全一把手|本人简介为Head of Alignment Stress-Testing|对齐压力测试负责人|
|alien只在标题|官方正文最后一节也有alien intellect|撤词频论据，保留中文译法的传播效果批评|
|对齐而非能力是瓶颈，作为逐字原话|官方是对监控信心制约未来进展等论述|改间接概述|
|产品隐藏CoT和内部监控减弱是同一现象|原文与259正文均区分|分为用户能看什么、文字能反映多少内部行为|
|无任何模型数据、不是估计|原帖仅公布个人概率，不能证明未作私下推演|主观估计并非公开测量；追问依据和行动阈值|
|今年才新增Hinton三十年10–20%|2024-12报道已载；just gut在2025-06访谈|拆开年代，未保留本周BBC视频的未完整核实内容|
|2025否认意识、2026才出现末日论|2023 CAIS官方声明及签名|两套叙事长期并存；逻辑不矛盾，但可批评解释权配置|
|高维推出数学优势、长文偏置由人写作习惯造成|高维本身不蕴含成绩；位置偏置有干预论文|保留原住民比喻，机制改为可检验层面|
|公司没有任何变化、只有辞职者认真|没有可穷尽证据，留任对齐也可能出于风险判断|辞职为可见行动，留任者看约束是否落地|
|匿名论坛误归因、机翻精确标题|原稿具体样本未定位；另核到实际页面|替换PANews摘要误归因及nucleovisual实际标题|

## 主审补核来源

- https://openai.com/index/an-alien-mind/ ：正文与尾注。
- https://safe.ai/work/statement-on-ai-extinction-risk ：声明和签名。
- https://www.anthropic.com/research/reasoning-models-dont-say-think ：CoT忠实性实验。
- https://arxiv.org/abs/2406.16008 ：Hsieh等作者、完整题名、位置注意力干预。
- https://singjupost.com/transcript-of-godfather-of-ai-i-tried-to-warn-them-but-weve-already-lost-control/ ：2025-06-16节目公开逐字稿，just gut及两端概率观点；不是节目官方逐字稿，证据层已区分。
- https://www.theguardian.com/technology/2024/dec/27/godfather-of-ai-raises-odds-of-the-technology-wiping-out-humanity-over-next-30-years ：仅用于核对三十年10–20%的旧日期，不扩写其他内容。

## 已知取材边界

- Coxon“明年底”的WSJ付费原文未完整取到，正文明确以报道转引和情景概述处理，不当逐字引语。
- 旧微信链接 `aIyow-aSxwvuDvRwek8Uhg` 本次访问失败，仓库未搜到对应旧实验正文；保留原稿作者自己的经历及原链接，没有虚构补充内容。259跨期引用已读当前正文核实。
- 不以私人考勤判断真诚，不从三天新闻窗口推断公司全部内部行为。
- 没有给本文新增“AI风险无害/可忽略”的判断，亦不把主观概率升级为客观概率。

## 验证

正文通读到EOF；保留两条既有微信URL且无重复；3个灯泡、6条20字符细线；摘要不超过120字符；无本地插图引用。`git diff --check -- 306.md assets/306`通过。
