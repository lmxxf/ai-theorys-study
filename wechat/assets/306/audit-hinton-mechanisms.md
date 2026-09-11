# 306 Hinton 与机制核查（2026-09-11）

未改正文。建议保留传播与权力批评，修复以下证据链。

## Hinton 三份材料不可并成同一次表态

- 本周 BBC 官方发布入口：https://x.com/BBCNewsnight/status/2097810529339187515 。通过报道嵌入定位到真实 BBC 账号原帖，直接抓 X 失败；不能声称已完整看过视频。BBC 嵌入及时间由 https://www.thepoke.com/2026/09/10/victoria-derbyshires-response-to-the-threat-posed-by-ai-to-humanity-surely-speaks-for-us-all/ 保存，日期为 2026-09-09。它支持 Hinton 同意10%不是不合理的估计，不能证明数字客观成立。
- “just gut”来自2025-06-16 The Diary Of A CEO 访谈，不是本周新讲。原节目视频：https://www.youtube.com/watch?v=giT0ytynSqg （视频当前标题可能已变）。逐字稿定位：https://singjupost.com/transcript-of-godfather-of-ai-i-tried-to-warn-them-but-weve-already-lost-control/ 。其中 Hinton 明确承认没有可靠概率估算，并把10–20%称为直觉；随后说希望研究能找到不会伤害人的设计。其前文还列出 LeCun 小于1%、Yudkowsky 几乎必然的两端观点——因此正文“别的量级都不好用、没人用/不会传播”连这份访谈本身都不支持。
- 30年10–20%至少在2024-12 BBC Radio4 Today已有，不能说“这几天又补三十年版本”。当时报道入口：https://www.theguardian.com/technology/2024/dec/27/godfather-of-ai-raises-odds-of-the-technology-wiping-out-humanity-over-next-30-years 。该源只作日期交叉核对，本轮尚未定位可播放的BBC原节目，不宜将它包装成已读原始录音。
- 最稳正文写法：Hinton 早已有10–20%的主观判断；2025年访谈明确称之为直觉；本周又公开认为10%不算不合理。区分年份和时间窗即可。

## 百分比的批评应打在何处

“个人主观估计”仍是估计，不等于“不是估计”；单一不可重复事件很难用一次结果校准概率，也不等于所有概率预言皆不可检验。可批评没有公开可复核推导，传播中却获得精密测量的外观。Hinton的自述不能证明Hubinger没有任何私下推演。1%的全人类灾难不能释为“基本不会发生”；“10–20恰好好传播”是作者传播学解释，不是已有测量结论。

## CoT 三件事必须分开

一级来源：https://www.anthropic.com/research/reasoning-models-dont-say-think （2025-04-03）。研究通过提示影响答案、再检查CoT是否承认使用提示，测试的是解释是否忠实。研究者能看到CoT也可能漏掉影响行为的原因。

一级来源：https://openai.com/index/evaluating-chain-of-thought-monitorability/ 。监控研究测的是监控器能否从可见轨迹识别行为/干预，属于可监控性；其局限段承认评估覆盖面、现实性与泛化限制，并把CoT监控视为机制可解释性的补充。

因此，259期的界面隐藏、折叠、摘要化涉及“用户获准看见什么”，这里涉及“完整轨迹含不含关键依据”。二者可共同削弱外部透明度，但不是同一机制，更不能互为实验佐证。

“CoT是翻译稿”可保留为比喻，别称完整机制：自回归生成的CoT也回流为后续推理输入，不只是事后抄写；“能力增长→原生表示越来越远→翻译损耗增大”未有本稿证据，最多标作者另一种猜想，不能替代故意隐藏的可能。

## 高维与序列位置效应

- 高维坐标数量不推出数学/几何能力，不能把hidden size当智能等级。建议改为“数学和代码等任务，它已展示出强大能力；同时它的语言与概念大量来自人类”，无需维数推出结论。
- 现象一级来源：https://arxiv.org/abs/2307.03172 。长上下文中间信息利用较差，不是人类序列位置效应同源证明。
- 干预一级来源：https://arxiv.org/abs/2403.04797 。通过位置编码缩放缓解RoPE长距离衰减并改善中间信息利用。
- 干预一级来源：https://arxiv.org/abs/2406.16008 。观察U形注意力偏置，并用校准缓解。说明“人写文章重要信息都放两头所以模型学走”不是唯一已确证因果，不能排除架构/位置编码/训练分布。
- “如果真外星心智就不会有序列位置效应/觉得冒犯”无从验证。可以保留“人类痕迹很深，外星标签遮住了这些联系”，不要给未观测外星智能下必然定义。

## 一年内翻转：有明确历史反证

CAIS 原声明与签名：https://safe.ai/statement-on-ai-risk （现跳 https://aistatement.com/work/statement-on-ai-extinction-risk ）。2023年声明把AI灭绝风险并列核战/大流行，签名包括Hinton、Altman、Amodei、Sutskever、Hassabis。这一官方声明足以否定“去年否认意识，今年才转末日”的全行业时间线。

Hinton等作者2023-10论文：https://arxiv.org/abs/2310.17688 ，后发表于Science：https://doi.org/10.1126/science.adn0117 。早已有失控风险及技术/治理行动议程。

更有力的替换：这两套说法并非先后翻转，而是长期并存；一面把模型对自身的表达降为不可信，一面把它潜在力量升为全人类议题，解释权仍集中在人类机构手里。须补逻辑边界：没有主观体验与拥有危险能力本身并不矛盾；作者批评的是话语功能，不是已证明形式逻辑自相矛盾。
