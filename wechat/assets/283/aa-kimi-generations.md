# Kimi 全系列 AA 智能指数（2026-08-17 从 artificialanalysis.ai 各 model 页抓取）

数据来源：每个模型页正文摘要句 "scores N on the Artificial Analysis Intelligence Index"
指数版本：Intelligence Index v4.1.1（九项：GDPval-AA v2、τ³-Banking、Terminal-Bench v2.1、SciCode、HLE、GPQA Diamond、CritPt、AA-Omniscience、AA-LCR）

| 模型 | AA 指数 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| Kimi K2 Thinking | 33 | 0.60 | 2.50 |
| Kimi K2.5（非推理） | 30 | 0.60 | 3.00 |
| Kimi K2.5 | 36 | 0.60 | 3.00 |
| Kimi K2.6（非推理） | 35 | 0.95 | 4.00 |
| Kimi K2.6 | 45 | 0.95 | 4.00 |
| Kimi K2.7 Code | 43 | 0.95 | 4.00 |
| Kimi K3 (low) | 48 | 3.00 | 15.00 |
| Kimi K3 (max) | 60 | 3.00 | 15.00 |

## 要点

- 同档（推理/max）代际线：33 → 36 → 45 → 43 → 60
- **K2.6 → K3 = 45 → 60，+15**，是这批模型里最大的一次代际跳
  - 对照 282 期核过的：Qwen 3.7→3.8 Max 重练基座 47→58（+11）、Grok 4.5→4.6 补充训练 56→61（+5）
- **K2.7 Code（43）比 K2.6（45）略低** —— 编码特化版，综合指数不占便宜，这条线不单调
- **K3 low 档 48 已超过 K2.6 max 的 45** —— 少想一点也比上一代想到底强
- 价格 K3 这代跳了：0.95/4.00 → 3.00/15.00（输入 3 倍+、输出近 4 倍）
- AA 页面对 K3 的评语：intelligence 领先，但 "particularly expensive when comparing to other open weight models of similar size"、"notably slow (68)"

## ⚠️ 写作口径

- **K3 换了新基座（2.8T 新练），这 +15 里基座与后训练各占多少，报告没拆，不能替它拆**
- 不能拿这 +15 跟 Grok 的 +5 直接比"做法优劣"——一个换基座一个没换，起点也不同（282 期栽过的跟头：拿绝对分冒充涨幅）
- 抓取方式：curl model 页 → 正则取摘要句。AA 是 JS dashboard，分项数值在图表 JS 里抓不到，只有总分和价格是 HTML 里的明文
