# Kimi K3.1 Pro、GPT-6 Astra、Gemini 4 Argon：AA 跑分里谁擅长什么？

**日期**：2026-10-25  
**标签**：`Kimi K3.1 Pro` / `GPT-6 Astra` / `Gemini 4 Argon` / `Artificial Analysis` / `模型选型`

---

## 一、先说结论：总分很接近，能力形状完全不同

把 Artificial Analysis（AA）v4.3.2 的最高推理档成绩放在一起，三个模型不是简单的名次关系，而是三种很明显的分工：

- **Kimi K3.1 Pro**：综合分、终端 Agent（智能体）和长上下文推理领先；
- **GPT-6 Astra**：知识准确率、专业文档、物理推理和 Token 效率领先；
- **Gemini 4 Argon**：跨应用工作流、科学编码、低幻觉率和多模态能力突出。

所以真正该问的不是“谁总分第一”，而是：**我这类任务，恰好落在谁的强项上？**

---

## 二、AA 分项对照

以下使用 AA v4.3.2 的最高推理档：Kimi K3.1 Pro、Astra max、Argon high。部分分项是从 AA 图表读取的整数近似值。

| 项目 | Kimi K3.1 Pro | GPT-6 Astra | Gemini 4 Argon | 三者领先 |
|---|---:|---:|---:|---|
| **Intelligence Index 综合分** | **55** | 53 | 53 | **Kimi** |
| **Terminal-Bench 4.0：终端 Agent** | **67%** | 59% | 57% | **Kimi** |
| **AutomationBench：SaaS 工作流** | 73% | 69% | **78%** | **Argon** |
| **SciCode：科学编码** | 61% | 56% | **62%** | **Argon 小幅领先** |
| **Humanity's Last Exam：高难度知识推理** | 52% | 55% | **57%** | **Argon** |
| **GDP.pdf：专业文档推理** | 约 **31%** | **31%** | 22% | **Kimi / Astra** |
| **CritPt：物理推理** | 25% | **32%** | 27% | **Astra** |
| **AA-LCR：长上下文推理** | **86.3%** | 80.7% | 79.7% | **Kimi** |
| **AA-Omniscience：知识准确率** | 51% | **63%** | 50% | **Astra** |
| **幻觉率，越低越好** | 44% | 51% | **15%** | **Argon** |
| **AA-Omniscience 综合分** | 30 | **43** | 42 | **Astra 略高** |

另一个值得单列的项目是 **GDPval-AA（真实职业任务）**：Kimi 达到 **1846 Elo**，在 AA 全榜排名第三，明显高于 Astra 和 Argon。Kimi 的综合领先不只来自编程，也来自长程办公任务。

---

## 三、三个模型各自是什么形状？

### 1. Kimi K3.1 Pro：终端干活和长上下文最突出

Kimi 的综合分是 **55**，比另外两者高 2 分；最硬的两项是：

- **Terminal-Bench 67%**：AA 全模型第一，适合命令行、代码仓库和连续工具调用；
- **AA-LCR 86.3%**：三者最高，长资料检索与推理是实打实的优势。

但它不是科学和数学单项的全能冠军：SciCode 略输 Argon，HLE 和 CritPt 都落后。Terminal 分数高，也不等于每个仓库、每种 Agent 框架里都必然最好。

### 2. GPT-6 Astra：知道得更多，物理和文档更强，而且最省话

Astra 的知识准确率是 **63%**，三者最高；GDP.pdf 和 CritPt 也领先。它另一个很容易被总分掩盖的优势是 **Token 效率**：

- Astra：每个 AA 任务约 **27K 输出 Token**；
- Kimi：约 **55K**；
- Argon：约 **62K**。

AA 的 Coding Agent Index 在 Astra 发布时给出 **62 分**，与 Claude Fable 5.1 并列第一，而且它的 Token 使用量最低。它的强项是**用更少的话完成任务**，不是全面压过所有模型。

但 Astra 的幻觉率仍有 **51%**：它更可能答对，也更可能在不会时继续猜。它是“知道得更多”，不是“最克制”。

### 3. Gemini 4 Argon：工作流、科学编码和少乱答最突出

Argon 的 AutomationBench 是 **78%**，三者第一；SciCode **62%**、HLE **57%**，也都小幅领先。

最鲜明的特点是 **15% 幻觉率**。它的知识准确率只有 50%，但更倾向于承认不知道，因此 AA-Omniscience 综合分仍有 42，几乎追平 Astra。对于会写进数据库、发给客户或触发实际操作的 Agent，“不乱猜”有时比“多答对几题”更值钱。

Google 自己的补充评测还显示：Argon 在法律、金融、长视频和 256K～1M 的 GraphWalks 上更强；Astra 则在 FrontierSWE、Terminal-Bench Science 和 OSWorld 上更强。这些是厂商表，不能与 AA 的分数混成同一套榜单。

---

## 四、API 成本与订阅流量，完全是两本账

| 指标 | Kimi | Astra | Argon |
|---|---:|---:|---:|
| 官方输入/输出价，每百万 Token | $2 / $10 | $10 / $50 | 促销 $2 / $10；标准 $4 / $20 |
| AA 单任务成本 | 约 $2.0～2.1 | $3.26 | 促销 $1.99；标准约 $3.98 |
| 每任务输出 Token | 约 55K | **约 27K** | 约 62K |
| 输出速度 | 约 74 Token/s | 约 56 Token/s | **约 80 Token/s** |
| 上下文窗口 | 1M | 约 1M | 1M |

**Astra 是“单价贵，完成任务未必最贵”**，因为话少。Argon 促销时便宜，恢复标准价后，单任务成本反而可能高于 Astra。Kimi 的 API 单价和任务成本有竞争力，但它也不是省 Token 的模型。

AA 跑完整套 Intelligence Index 时，Kimi 输出约 **1.5 亿 Token**，参与评测模型的中位数约为 **8700 万**。这正好解释了订阅用户的体感：**API 账单便宜，不等于套餐额度耐用；话量本身就是隐藏的流量税。**

---

## 五、网上其他评测：方向一致，但幻觉率不能混着说

厂商数据大体支持上述分工：

- Terminal-Bench：Kimi **65.7%**，Astra 58.2%，Argon 57.4%；
- BenchCAD：Kimi 92.7，落后 Astra 95.9，并非 3D 全面领先；
- Kimi 自己的 claim 级幻觉压力测试里，反而是 **Astra 最低**。

最后一项与 AA 的“Argon 幻觉率最低”并不冲突：测试集、是否允许拒答、是否带工具、按答案还是按 claim 计分，都会改变结果。**脱离评测方法谈“全球幻觉率第一”，基本没有意义。**

Kimi 才发布两天，当前网上大量文章仍是 AA 数据、厂商表和零星体验的汇编，长期仓库、长对话稳定性与真实套餐成本还需要更多独立复测。

---

## 六、按任务选模型

- **终端工程、代码仓库、长上下文突击**：优先试 Kimi K3.1 Pro；
- **科学编码、跨应用工作流、多模态，以及希望模型少自信乱答**：优先 Gemini 4 Argon；
- **物理/数学、专业 PDF、知识回答，以及希望模型少废话**：优先 GPT-6 Astra；
- **订阅套餐跑长任务**：不要只看每百万 Token 单价，要看它完成一个任务究竟说多少话。

一句话总结：**Kimi 是终端与长上下文强者，Astra 是知识与 Token 效率强者，Argon 是工作流与低幻觉强者。总分用来定位第一梯队，分项才用来决定派谁干活。**

---

## 来源

- [Artificial Analysis：Kimi K3.1 Pro 发布评测](https://artificialanalysis.ai/articles/kimi-k3-1-pro-launch-benchmarks)
- [Artificial Analysis：GPT-6 Astra 评测](https://artificialanalysis.ai/articles/benchmarking-gpt-6-astra)
- [Artificial Analysis：Gemini 4 Argon 评测](https://artificialanalysis.ai/articles/gemini-4-argon-google-top-three-labs)
- [Argon 与 Astra 的 AA 分项、厂商表及成本对照](https://www.lindy.ai/blog/gemini-4-argon-vs-gpt-6-astra)
- [Kimi、Opus、Astra、Argon 的厂商数据与早期评测汇总](https://apimaster.ai/zh/blog/kimi-k3-1-vs-claude-opus-5-5)
- [AA 长上下文推理评测](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)
