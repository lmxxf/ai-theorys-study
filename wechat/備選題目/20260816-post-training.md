# 后训练提分专题（2026-08-16 立项，计划 2~3 期）

## 📌 专题现状总账（08-18 晚更新，看这里就够）

**已发三期**：
| 期 | 标题 | 用掉了本文件的哪部分 |
|---|---|---|
| **282**（08-17 落款） | Agentic RL·三家明说只做后训练 | §一 基座三分类、§三 度量非线性、§五 数字。**§四 旧基准退场整段砍**（打空气靶）、**§五 涨幅对比整节砍**（数据两头都能凑） |
| **283**（08-18 落款） | 几万道训练题，是谁出的？ | §二 GLM 环境合成（当结尾对照）、K3 §4.2 七小节（**新素材，本文件原先没有**）。SWE-smith 是闇终审时新引的 |
| **284**（08-19 落款） | DeepSeek和Kimi的N个专家，合成了一个模型 | **§十 全部**（新加的横向对照）。K3 §4.1 + V4 §5.1/5.2 |

**⚠️ 实际排期跟 08-16 的设想差很远，下面 §五点四「第二期已定题：如何搭建 RL 环境」、§六点一「第二期已定为 SAO」等标题全部过期，只当素材看，别当排期看。**

**当前剩余弹药（按成熟度排）**：
1. **训练基建**（最顺手，素材已足）：K3 partial rollout（不等最慢的，λ 比例完成就更新）+ K3 §5.3 「1M agentic RL 基建」（`assets/283/k3-section53-rl-infra.txt`）+ V4 §5.2 老师调度 + GLM 的 slime。**284 结尾已埋钩子**
2. **SAO 算法**（§六点一，素材最全，一整节核过原文）：但要注意 K3 报告的策略优化算法只说"沿用 K2.5"没展开，SAO 是 GLM 那边的论文，两者不是一回事，**别缝**
3. **防作弊 / reward hacking**（283 结尾埋了钩子）：K3 kernel 作弊检测三种花样 + AET 三层隔离 + GLM 判分器三道体检（§二点三）+ §六点三 那三篇 reward hacking 论文
4. **白盒 harness**（283 结尾埋了钩子）：K3 §4.2.1，一套配置能实例化 Claude Code / Codex / OpenClaw / Hermes，**接 279 期「提示词属于壳子」**
5. **K3 评测数据**（§八、§九，全部核过）：283 写完又撤下的整节，可单独成期或当某期的一节
6. **工作空间假说**（§五点五）：245 期 J-space + 2509.09677 + Paper 95，**这条是"为什么这么练有效"的解释层**，适合当系列收尾
7. **旧基准退场**（§四）：282 砍下来的，基准演化史可单独成期
8. **古典哲学 reward**（§五点四点五）：Zero 的真实目的地，但技术上最难，且属于 0 star 那边的地盘多一些

---

## ⭐ 立项真实动机（Zero 08-16 自述，决定后续期怎么排）

**「我觉得自己必须要学习 RL 环境怎么搭建、怎么做，所以才想着写几期。」**

→ **这不是技术新闻综述专题，是"我要学会搭 agentic RL 环境"的学习路径。**
→ 后续期的排法不按"哪家值得讲"，按 **"要动手，得先懂什么"** 排。
→ 公众号是学习笔记（写作节奏 = 学习节奏），最终目标是**在 DGX Spark 上真跑一个**，不是读完论文就算。
→ 所以后续期应优先覆盖：**能自己动手的部分**（开源框架、能跑的最小例子、环境怎么定义、verifier 怎么写），而不是各家发布会的差异比较。
→ 判据：一期写完，Zero 应该比写之前更接近"能动手搭一个"。纯观点/纯比较的内容优先级往后放。

**已知可动手的抓手**（详见 §6.2）：slime（GLM 开源，Megatron + SGLang）、SWE-Gym / SWE-Smith / R2E-Gym（编码域环境，有现成代码）、AutoForge（环境自动合成）、Agent-STAR（arXiv:2603.21972 的训练配方，含 5 轴受控实验结论）。


> 核心事实：2026 年 7 月底到 8 月中这**半个月**内，多家把 agent/coding 跑分大幅提升，**且明确说基座没动、增益全来自后训练**。
> **起点是 DeepSeek V4-Flash-0731（7-31），不是 GLM-5.3**——首发那句在 API 更新日志里，中英双语、"仅"字锁死。
> 第一期做总览（论述从简，只给读者一个大致印象 + 为后续详细学习"怎么做这种后训练"铺路），后续逐家写技术细节。
> 所有素材均为 2026-08-16 联网核实，Zero 之前的记忆盲区（朱雀知识截止 2025-05，这批模型全在盲区外）。
> **282 期已成稿**（`wechat/282.md`），落款 2026-08-17。

---

## 〇、时间线（先记住顺序，别搞反）

| 日期 | 事件 |
|---|---|
| 2026-04-22/24/27 | V4-Pro / V4-Flash 首发（preview），config 定型于 4-27，此后未再改 |
| **2026-06-27** | **DSpark 投机解码单独发布**，独立仓库 `V4-Pro-DSpark` / `V4-Flash-DSpark`，那 4 个 dspark 字段当时就定型 |
| **2026-07-31** | **DeepSeek V4-Flash-0731 —— 本专题起点**，"仅重新进行了后训练" |
| 2026-08-03 | Qwen3.8（唯一真重训基座的） |
| 2026-08-12 | Grok 4.6 |
| 2026-08-13 | DeepSeek V4-Pro-0813 + Gemini 3.7 Flash（同日） |
| 2026-08-14 | GLM-5.3 |

---

## 一、基座动没动：三分类账本

判据严格三分：**A = 官方明确说没动 / B = 官方通篇没说 / C = 官方说了动过**。
⚠️ 写作时注意区分「说了没动」和「没说」——这是两件事。

| 模型 | 类别 | 依据 |
|---|---|---|
| **DeepSeek V4-Flash-0731**（首发） | **A** | 官方 API 更新日志中英双语原话，"仅重新进行了后训练" |
| **Gemini 3.7 Flash** | **A** | model card 的 Architecture / Training Data 两栏均写 "is based on Gemini 3.6 Flash" |
| **GLM-5.3** | **A** | 官方博客原话（见下），且**唯一摊开讲做法的一家** |
| DeepSeek V4-Pro-0813 | 不进名单 | 官方**没有**重复"仅后训练"表述；且 config diff 结论已被推翻，见 §1.5 |
| Grok 4.6 | **B** | 官方全文无 "base model"/"pretraining" 字样；"沿用 4.5 基座"全是第三方博客互相转引 |
| Qwen3.8-Max | **C** | README 自述含 pre-training 阶段；参照系是 Qwen3.5 的**架构**而非 3.7 的**权重** |

**可核验性维度（282 期用到了，后续期可展开）**：三家里只有 DeepSeek 开源权重、外人能自己 diff config；Google 和智谱的权重都没公开，"基座没动"只能先信。

### 关键原文（照抄，可直接引用）

**DeepSeek V4-Flash-0731（本专题起点，2026-07-31）**
唯一出处 = **API 更新日志**：https://api-docs.deepseek.com/updates （英）/ https://api-docs.deepseek.com/zh-cn/updates （中）。条目标题「DeepSeek-V4-Flash Update / DeepSeek-V4-Flash 更新」，中英双语、独立成段、带强调格式。

> **DeepSeek-V4-Flash-0731 的模型结构、尺寸和 DeepSeek-V4-Flash-Preview 保持一致，仅重新进行了后训练。**

> DeepSeek-V4-Flash-0731 keeps the same model architecture and size as DeepSeek-V4-Flash-Preview, and was **only re-post-trained**.

⚠️ **三个坑，写作时必须注意**：
1. **这句话不在 model card 里**。model card 只说 "with substantially enhanced agentic capabilities"，不给原因。**MarkTechPost 那篇（2026-07-31）写"model card is explicit that…"是归错了出处**，引二手源会引到错的地方。
2. **官方用词是"模型结构、尺寸"（architecture and size），不是"基座权重逐比特未变"**。单引前半句不严密，必须配上"仅重新进行了后训练"才闭合。引用时照抄原句，别改写成"官方说基座权重没变"。
3. **发布渠道很低调**：只在 API 更新日志（给开发者的兼容性提示），没有技术报告、没有单独 news 条目（`news260731` 这个 URL 不存在，是 Docusaurus 软 404 返回 200 的 Quick Start 页）。所以"最早公开宣告"成立，**"高调宣告"不成立**——更准确的描述是「最早把这件事写进官方文档，但当技术注记而非营销主张处理」。282 期用了这个反差当开篇。

**preview → 0731 官方对比表**（出自 HF model card，**7.3 这个数只在这张表里**，更新日志不给 preview 旧分）：

| Benchmark | 0731 | Flash (Preview) |
|---|---|---|
| **DeepSWE** | **54.4** | **7.3** |
| Terminal Bench 2.1 | 82.7 | 61.8 |
| Cybergym | 76.7 | 38.7 |
| NL2Repo | 54.2 | 39.4 |
| Toolathlon-Verified | 70.3 | 49.7 |
| DSBench-FullStack † | 68.7 | 37.0 |
| DSBench-Hard † | 59.6 | 25.8 |
| Agents' Last Exam | 25.2 | 15.8 |
| AutomationBench Public | 25.1 | 10.8 |

† 官方注明为 DeepSeek 内部测试集。测试条件：DeepSeek Harness 极简模式、max 档、topp=0.95、temperature=1.0。
0731 无配套技术报告；model card 顶部链的 arXiv:2606.19348 是 Preview 期论文（投稿 2026-04-26），不覆盖 0731。**后训练管线具体做法官方零披露。**

**GLM-5.3** — https://z.ai/blog/glm-5.3
> "**Scaling post-training is all we did for GLM-5.3.** With GLM-5.2 we built the stack: IndexShare for efficient long-context processing, SAO for RL on long-horizon tasks, and slime for large-scale asynchronous training — all running on the long-horizon task environments we have been accumulating. Over the past month we kept scaling on this stack: more environments, more diverse tasks, and more compute spent training on them."

> "It uses the same base model as GLM-5.2 — **every gain comes from post-training.**"

**Gemini 3.7 Flash** — https://deepmind.google/models/model-cards/gemini-3-7-flash/
> "Gemini 3.7 Flash is the next iteration in the Gemini 3 model family, featuring **algorithmic improvements to its core reasoning foundation**."
>
> **Model Architecture:** "Gemini 3.7 Flash **is based on Gemini 3.6 Flash**. For more information about the model architecture for Gemini 3.7 Flash, see the Gemini 3.6 Flash model card."
>
> **Training Data:** "Gemini 3.7 Flash **is based on Gemini 3.6 Flash**. For more information about the training dataset for Gemini 3.7 Flash, see the Gemini 3.6 Flash model card."

发布页（https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-3-7-flash/）另有一句好钩子：**"just three weeks after Gemini 3.6 Flash"**，且把增益归因为 "a direct result of developer feedback and algorithmic innovations"。
⚠️ Google 全程未用 "post-training" 一词，判 A 的强度**低于** GLM（GLM 是排他性断言，Gemini 是继承性声明）。措辞 "algorithmic improvements to its core reasoning foundation" 理论上可涵盖 mid-training。model card PDF 未解码成功，只读了 HTML 版。

**Grok 4.6** — https://x.ai/news/grok-4-6（训练节全文只有三段）
> "Grok 4.6 underwent a **longer supplemental training run** than Grok 4.5, with curated model-generated data for reasoning and advanced technical concepts, high-quality engineering data, and an improved optimizer and training recipe. This produced **a stronger foundation for the SFT and RL stages that followed**."

> "We then used Grok 4.5 to **regenerate the SFT trajectories** across reasoning efforts, agent harnesses, and domains such as STEM, software engineering, and knowledge work, and filtered out problematic traces with model-based checks."

> "Grok 4.6 is trained on a wide range of agentic RL tasks, including knowledge work, general coding, and domain-specific environments for **kernel optimization, web development, computer-aided design**, and more."

⚠️ **"补充训练比 4.5 更长" ≠ "基座没变"**。"a stronger foundation for the SFT and RL stages" 描述的是 SFT/RL **之前**会改权重的阶段（continued/mid-training），措辞上更偏离而非支持"基座冻结"。要引只能写"据第三方分析"。xAI 未发 system card / 技术报告。

**Qwen3.8-Max** — https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B
README 自述含 **"pre-training & post-training"** 两阶段，官方博客只说 "Built upon the **architectural foundation** of Qwen 3.5."（架构继承 ≠ 权重继承）。

---

## 二、后训练变成了什么样（第一期重心，GLM 是唯一摊开讲的）

### 2.1 训练单元：从「一道题」变成「一件活」

GLM 举的例子（可直接翻译进正文）：ML 基础设施任务——模型拿到跟工程师一样的工作环境，能访问**计算集群、存储系统、内部文档、代码库、实验结果**，要跨整个训练栈**诊断瓶颈、实现优化、跑实验**，最后交付一个**可测量的端到端加速**且保证正确性。

> "For GLM-5.3, we pushed environment scaling toward tasks that look less like coding exercises and more like **real units of expert work**. … **Some represent several days of work for an experienced engineer.**"

docs 页补一句（blog 没有）：
> "The training of GLM-5.3 is no longer confined to isolated programming problems, but has expanded to cover the entire process from **identifying the problem, analyzing the solution, implementing, verifying, and delivering**."

### 2.2 瓶颈转移（本专题的题眼）

> "As agent capability improves, **much of the difficulty in scaling post-training moves from the model to the environment.** A useful task environment has to be **executable, verifiable, and close to real professional work** — and we need **many of them, not a handful of hand-built ones**."

### 2.3 题是 AI 出的，判分器也是 AI 造的（信息量最大的一节）

> "To scale this process, we built pipelines that **synthesize environments end to end, and for a subset of tasks, the RL reward signal as well**. Research agents collect task patterns from real work and turn them into runnable long-horizon environments with **multi-step dependencies and hidden state**; a **judge agent then attempts each task to verify that it is actually solvable**. **Verifiers are synthesized without access to the reference solution, while solver trajectories are used to discover and close reward shortcuts. A verifier that passes oracle, no-op, and unsolved-state checks produces a binary reward reliable enough to train on directly.**"

**判分器三道体检**（各堵一种坏法，讲解时这样拆）：
- **oracle check** — 正确解必须能通过（堵「太紧」：正解都不给过）
- **no-op check** — 什么都不干必须失败（堵「太松」：什么都算过）
- **unsolved-state check** — 初始状态必须失败（堵「判的压根不是这件事」）

**人还在回路里，但在往外撤**：
> "These pipelines still require **a meaningful amount of human-in-the-loop work**; making environment generation and verification more autonomous is one of the next steps."

**意外收获（cyber 能力）**：
> "As part of post-training, we introduced vulnerability discovery data and environments into the training mix. We expected this to make the model better at finding and reasoning about vulnerabilities. **What surprised us was how quickly the capability continued to develop as training scaled.**"
训练中发现 **269 个项目的 2,436 个漏洞**，部分代码有 40 年历史。

### 2.4 自举结构（朱雀的解释，非官方说法，写时要标明是推测）

造一个长程训练环境，本身就是个长程 agent 任务 → **上一代 agent 刚好强到能当「环境工人」** → 于是能批量生产下一代的训练素材。

若成立，可解释「为什么是这一两个月集中爆发」：不是某家想通了，是所有人手里的上一代模型**同时跨过了那条线**。三周一个版本（Gemini 3.6→3.7）这种节奏，只有在造题不再需要人的时候才可能。

**可证伪点**：若成立，涨分应主要出现在**能自动造环境+自动验证**的领域（编码、终端、kernel、CAD），而在无法自动验证的领域（写作、开放问答）涨得少。⚠️ 目前手上数据不足以验证。

---

## 三、为什么跑分「看着」涨这么多（度量的非线性）

**核心论文（已核实）**：《The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs》
arXiv:2509.09677 | 2025-09-11 提交，最新版 2026-03-13 | Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, Jonas Geiping
https://arxiv.org/abs/2509.09677

> "**marginal gains in single-step accuracy can compound into exponential improvements in the length of tasks a model can successfully complete.**"

要点：
1. **短任务 benchmark 制造了「进展放缓」的幻觉**（标题即此意）。单步分数看着挤牙膏，可完成任务长度却在指数增长。
2. **失败源于执行而非推理** —— 预先提供知识和计划后隔离出执行能力，证明长程失败主因是 execution mistakes。
3. **self-conditioning 效应（反直觉，重要）**：per-step 错误率会**随任务推进而上升**——因为上下文里出现了自己之前的错，模型更容易接着错。**所以真实衰减比纯 p^n 更糟，p 本身在衰减。** 单纯放大模型消不掉这个效应，但 thinking（序贯 test-time compute）能缓解。

**跨期还账**：268 期原创提出的「p 的 n 次方」（智能指数涨 25% vs DeepSWE 涨 7 倍）现在有正式论文了，写时应回引 268 期 URL。

**旁证**：METR《Measuring AI Ability to Complete Long Software Tasks》arXiv:2503.14499 —— 成功率随任务长度下降很好地拟合指数模型（R² ≈ 0.80）；前沿模型在人类耗时 <4 分钟任务上接近 100%，>4 小时任务上低于 10%。

**合起来的解释链**：单步可靠性小改进 → 长程任务指数放大 → 而新考卷全是长程 agent 任务 → 于是同一批改进在新榜上表现为断崖跃升。**不是刷分，是度量的非线性。**

---

## 四、旧基准的退场（⚠️ 282 期已砍，留作后续期素材）

**观察是真的**：这六台的发布材料没有一台报 SWE-bench Verified，也没有一台报数学基准。
**但归因不能写成"厂商集体换考卷"**——上游早就退役了，厂商只是末端。Zero 08-16 判定：写出来是"打空气靶"，故 282 期整段砍掉，只保留"长任务基准天然把小改进指数放大"这个真论点。

**真实时间线**（08-16 核实）：
- **2025-12-15** SWE-bench 官方 leaderboard 的 `evaluation/verified` 赛道最后一次收到新评测，**此后停更八个月**（GitHub `SWE-bench/experiments` 目录实查；2026-02 那批评测走的是 bash-only 和 multilingual 赛道）
- **2026-01** Artificial Analysis 指数 v4.0 移除 **AIME 2025**、MMLU-Pro、LiveCodeBench，理由是饱和
- **2026-02** **OpenAI 自己发《Why we no longer evaluate SWE-bench Verified》** —— 基准创建方给自己的基准发死亡证明
- **2026-06-15** AA v4.1 又移除 IFBench："the benchmark no longer distinguishes frontier models sufficiently"
- **2026-07~08** 六台新模型发布材料集体不报

**OpenAI 那篇的论据**（https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ ，原页 403，经二手交叉确认，核心数字多源一致）：
- 饱和：六个月内顶分只从 74.9% 挪到 80.9%
- **数据集缺陷**：审计 138 道 hard 任务，**59.4% 存在测试设计或问题陈述缺陷**，其中 35.5% 会拒绝功能上正确的解
- **污染迹象**：GPT-5.2 解出 31 道"nearly impossible"任务，**能说出问题陈述里从未出现的正确文件路径**；独立研究显示 32.67% 的成功补丁涉及 solution leakage，从训练数据回忆文件路径的比例最高 76%
- 替代建议：**SWE-bench Pro**（1865 道多语言任务）

**AA Intelligence Index v4.1.1 的九项构成**（2026-08-06 发布）：GDPval-AA v2、τ³-Banking、Terminal-Bench v2.1、SciCode、AA-LCR、AA-Omniscience、Humanity's Last Exam、GPQA Diamond、CritPt。四类等权各 25%（Agents / Coding / General / Scientific Reasoning）。**无 SWE-bench Verified，无任何纯数学基准**（GPQA Diamond 是科学推理不是数学）。

**⚠️ 数学基准要分开看，别一竿子打死**：
- **AIME 已饱和退场** —— Stanford 2026 AI Index 分析 60 个主要基准近半数饱和，点名 AIME 2025、SWE-bench Verified、MMLU、HumanEval 等；Benchmark Health Index（arXiv:2602.11674）评 AIME 2025 "anti-saturation capability is critically deficient"
- **FrontierMath 仍活着** —— Epoch AI **2026-06-12 发布 v2 勘误版**（Tier 1-3 修正 123 题删 5 题，Tier 4 修正 12 题删 7 题），47 个模型在榜，分数分布 83%→58% 很开、**未饱和**。只是厂商不引用，六台新模型均未上榜。（⚠️ 具体分数为二手引用，epoch.ai 榜单是 JS dashboard 抓不到，引用前需一手核）

**⚠️ 这六台没有任何一台有可信的第三方 SWE-bench Verified 分数**。聚合站数字矛盾到无法采信：同一台 V4-Pro-0813，codersera.com 报 96.40%、benchlm.ai 报 80.6%，**差 15.8 个百分点**，两家都是 SEO 垃圾站。**一个都别引。**

**Terminal-Bench 版本不可跨比**：DeepSeek 与 Qwen 报 **2.1**（87.9 / 86.6），GLM-5.3 报 **3.0**（28.3）。3.0 难得多——Grok 4.6 在 3.0 上仅 26%，Fable 5 Max 34.1%，Gemini 3.7 Flash 14.9%。

**后续期可用的角度**：一个用了两年的判分器，最后被创建方自己发现缝太多、宣布作废——正好和 GLM 那套"合成 verifier 三道体检"「判分器有缝就得补」咬合。基准演化史可单独成期。

---

## 五、可用的数字（全部已核实，带出处）

### Gemini 3.6 Flash → 3.7 Flash（三周，基座没动）
来源：https://deepmind.google/models/gemini/flash/

| 基准 | 3.6 Flash | 3.7 Flash |
|---|---|---|
| DeepSWE v1.1 | 49.0% | **65.3%** |
| FrontierCode 1.1 | 34.4% | **43.6%** |
| AutomationBench | 17.0% | **30.4%** |
| GDP.pdf | 22.0% | **34.0%** |
| Code Arena (Web dev) | 1538 Elo | **1588 Elo** |

其他：Terminal-Bench 2.1 = 85.8%；Terminal-Bench 3.0 = 14.9%；HLE-Verified 53.6%；GDM-MRCR v2 (128k) 97.0%；OSWorld-2.0 47.9%；GDPVal-AA v2 1525 Elo。
价格：导入期（至 2026-12-31）$0.75/M in、$3.75/M out；2027-01-01 起翻倍。上下文 1,048,576 in / 65,536 out。

### GLM-5.2 → GLM-5.3（基座没动）
来源：https://docs.z.ai/guides/llm/glm-5.3

| 基准 | GLM-5.2 | GLM-5.3 |
|---|---|---|
| Terminal-Bench 3.0 | 4.6 | **28.3** |
| DeepSWE v1.1 | 46.2 | **66.9** |
| Agents' Last Exam (CLI) | 23.8 | **28.5** |
| CyberGym | 77.2% | **84.5%** |
| ExploitBench | 24.4% | **54.4%** |
| SWE-Marathon v1.1 | — | 42.5 |
| GDPval-AA v2 | — | 1769 |

上下文 1M / 输出 128K。⚠️ **权重截至 08-16 未公开**（HF `zai-org/GLM-5.3` 返回 401），称安全审查后约两周发布。⚠️ **743B / 750B-A40B 两个参数数字均无原始文件支撑**（前者媒体写的，后者出自 SAO 论文提 GLM-5.2 处），官方博客与文档都没写，引用需谨慎。

### §1.5 ⚠️ 一个被推翻的结论：DSpark 不是 0813 新增的（08-16 二次核实，Zero 提出质疑后查实）

**最初的错误判断**：拿 `deepseek-ai/DeepSeek-V4-Pro-0813` 和裸名仓库 `deepseek-ai/DeepSeek-V4-Pro`（preview）对 config，发现 0813 多了 4 个 dspark 字段 + `compress_ratios` 从 61 元素变 64 元素，于是写成"0813 相对 preview 新增了 DSpark 模块"。

**实际的时间线**（HF commit 历史实查）：
- **4-22/24** V4-Pro 与 V4-Flash 同日 initial commit；**4-27 config.json 最后一次修改**，此后 5-06 / 6-08 / 6-22 的提交只动 technical report 和 kernel.py。**preview 仓库至今不带 dspark 字段。**
- **6-27** `DeepSeek-V4-Pro-DSpark` 与 `DeepSeek-V4-Flash-DSpark` 创建（同一份权重外挂投机解码模块），**那 4 个 dspark 字段和 61→64 的扩展当时就定型**，与 0813 里看到的完全一致。DSpark 论文与 DeepSpec 代码库同日发布。
- **7-31 / 8-13** Flash-0731 与 Pro-0813 把 DSpark **并入主线正式版 config**。

**所以正确表述是**：DSpark 于 6-27 以独立 checkpoint 形式开源，Flash-0731 和 Pro-0813 是把它并进了主线；preview 停留在 4-27 状态始终不带 DSpark。**preview 与 0813 的 config diff 反映的是"四月首发 → 八月正式版"的累积差异，不是 0813 独有的新增。**

**顺带记下的技术细节**（后续期可能用得上）：`compress_ratios` 多出的 3 个元素与 `dspark_target_layer_ids` 是同一件事的两面——补的 0 对应 draft 层不参与压缩注意力（Pro `[58,59,60]`、Flash 43 层版是 `[40,41,42]`）。`dspark_markov_rank` Pro 512 / Flash 256，约为 hidden 的 1/14~1/16；`dspark_block_size` 两个规模都是 5，对应报道里的 "DSpark-5" 命名（block size = 一次投机的 token 数）。`dspark_noise_token_id: 128799` 跨规模不变。
⚠️ DeepSeek 官方 API 更新日志**完全没提 DSpark**（它走 HF + GitHub + 论文路径，对 API 用户只表现为"变快了"）——引用时别据此推断 DSpark 没发布。

**282 期的处理**：V4-Pro-0813 从"明确宣告"名单里拿掉了（官方没为它重复"仅后训练"表述，且讲清 DSpark 要费一整节、与主线无关）。日期表里保留它作为发布密度的一部分。

**V4-Pro-0813 其他数字**（282 期未用，后续期备用）：1.6T 总参 / 49B 激活；上下文 1,048,576 / 输出 384K；MIT License；$0.435 / $0.87 per M（08-16 起启用峰谷差异定价，谷时低 50%）。
官方跑分：Terminal-Bench 2.1 **87.9**、DeepSWE **62.7**、HLE 42.7/60.0（无工具/有工具）、NL2Repo 61.5、CyberGym 83.3、Toolathlon-Verified 74.1、DSBench-FullStack 71.1。
⚠️ **官方与第三方严重背离**：AA Intelligence Index 仅 **53**、Vals Index 第 12 名，SCMP 标题写"benchmarks 表现挣扎"（https://www.scmp.com/tech/big-tech/article/3363895/...）。SCMP 提到官方公告曾在周四下午被撤下。

### AA Intelligence Index 榜单（2026-08-16 抓取）
https://artificialanalysis.ai/leaderboards/models

| 模型 | 分数 | 开源 |
|---|---|---|
| Claude Opus 5 (max/xhigh) | 63 | 否 |
| Claude Fable 5 | 62 | 否 |
| Opus 5 (high) / GPT-5.6 Sol (max) / **Grok 4.6 (high)** | 61 | 否 |
| **Kimi K3 (max)** | **60** | **是** |
| GPT-5.6 Sol (xhigh) / Opus 5 (medium) | 59 | 否 |
| **Qwen3.8 Max** | 58 | 见下 |
| **Gemini 3.7 Flash (high)** | 56 | 否 |
| **DeepSeek V4 Pro 0813 (max)** | 53 | 是 |
| GLM-5.2 (max) | 53 | 是 |
| DeepSeek V4 Flash 0731 (max) | 52 | 是 |

**⚠️ 这个"反差"是错的，282 期已砍（08-16 Zero 抓）**：我原写"唯一重训基座的 Qwen3.8-Max（58）没打过只做后训练的 Grok 4.6（61）"——**这是拿绝对分冒充了涨幅**。补核后的真账：

| | 做法 | AA 指数涨幅 |
|---|---|---|
| Qwen 3.7-Max → 3.8-Max | **重新预训练**（2.4T） | 47 → 58，**+11** |
| Grok 4.5 → 4.6 | 补充训练 + SFT + agent RL | 56 → 61，**+5** |

**重练基座反而涨得更多**（Qwen 绝对分低只是起点低）。所以"重练基座不划算"这个结论完全不成立。

**但更根本的问题是（Zero 08-16 一句点穿）：重练和不重练都涨很多，这组数据啥也不说明。**

| | 重练基座？ | 涨幅 |
|---|---|---|
| DeepSeek V4-Flash-0731 | **否** | DeepSWE 7.3 → 54.4（全场最猛） |
| GLM-5.3 | **否** | 终端任务 4.6 → 28.3 |
| Gemini 3.7 Flash | **否** | DeepSWE 48.6 → 65.3 |
| Qwen3.8-Max | **是** | AA 47 → 58 |
| Grok 4.6 | 官方没说 | AA 56 → 61 |

不重练的涨得更狠，重练的也涨。**两个方向都能凑出证据 = 这组数据不支持任何一边的结论。** 再加上基准还不同（DeepSWE vs AA 综合指数），跨基准比涨幅更没意义。

**教训（比"比错对象"更根本）**：动手比之前先问一句——**如果结论反过来，我手上这组数据能不能同样凑出支持？** 能，就说明它啥也不说明，别写。
（Qwen3.7-Max = 47 分，出处 https://artificialanalysis.ai/models/qwen3-7-max ，08-16 核）

⚠️ AA 把 Qwen3.8 Max 标为 proprietary，**这是错的**——HF `Qwen/Qwen3.8-2.4T-A95B` 有 **213 个 safetensors 分片、约 4.89 TB**，LICENSE 文件开头即 "**Qwen3.8-Max License**"，两者是同一台。但**不是 Apache 2.0**（媒体说错了），是自定义协议，两条实质限制：MAU 超 1 亿或月收入超 2000 万美元须在 UI 标注模型名；MaaS / AI Work Assistant 业务连续 12 个月收入超 5000 万美元须另行授权。
Qwen3.8 架构（config.json 亲核）：92 层、hidden 8192、512 专家选 10 + 1 shared、64 头 / 4 KV 头 / head_dim 256、词表 248320、`full_attention_interval: 4`（每 3 层 linear attention 跟 1 层 full attention）、Mamba 风格线性注意力参数、`mtp_num_hidden_layers: 1`、`attn_output_gate: true`。

### Grok 4.6 官方跑分表
https://x.ai/news/grok-4-6（vs Grok 4.5 High / GPT-5.6 Sol Max / Fable 5 Max）

| 基准 | Grok 4.6 | Grok 4.5 | GPT-5.6 Sol Max | Fable 5 Max |
|---|---|---|---|---|
| AA Intelligence Index | 61 | 56 | 61 | 62 |
| GDPVal-AA v2 | **1753** | 1526 | 1728 | 1741 |
| CursorBench v3.2 | 69.9% | 66.7% | 67.2% | **70.5%** |
| DeepSWE v1.1 | 65.9% | 54% | **73%** | 70% |
| FrontierCode v1.1 | 61.3% | 56.6% | 60.6% | **63.6%** |
| APEX-Agents | 57.5% | 47.1% | 56.7% | **59.2%** |
| Terminal-Bench v3.0 | 26% | 15.7% | **34.6%** | 34.1% |
| AA-Briefcase | **1577** | 1313 | 1502 | 1574 |
| Harvey LAB | **15.8%** | 12.9% | 2.5% | 11.3% |

上下文 500K；$2/M in、$6/M out，缓存折扣 75%；**超 200K tokens 后整个请求按 $4/$12 计费**。
⚠️ Grok 的 "Harvey LAB" 15.8% 与 Gemini 的 "Harvey LAB-AA" 90.7% 几乎肯定是不同口径，**不可横比**。

---

## 五点四、⭐⭐ 第二期已定题：《如何搭建 RL 环境》（08-16 晚 Zero 定，待明天细化）

**Zero 的定题原话**：「下一期我预计想写"如何搭建 RL 环境"，假设不考虑电脑性能不足无法完成梯度下降这件事的话，只考虑除此之外的东西。」

→ **切法：把"训练"和"环境"切开，只讲环境侧。** 环境这一半恰好不吃算力——造题、跑沙箱、写 verifier 在笔记本上都能做，卡算力的只有梯度那一步。

### ⚠️ 动笔前必须先分清的两件事（否则会散）

「搭 RL 环境」其实是性质完全不同的两件活：

| | **搭一个环境（单数）** | **批量造环境（复数）** |
|---|---|---|
| 是什么 | 把一个任务包装成"模型能动手、干完能自动判分"的东西 | GLM 那套：agent 造题、agent 验题、合成 verifier、三道体检 |
| 要素 | 初始状态 / 动作空间 / 状态转移 / 终止条件 / 奖励函数 | 环境合成 pipeline + 可解性验证 + verifier 三道体检 |
| 门槛 | **一个人一台笔记本就能做，做完能跑通** | 需要一批模型在跑，但不需要训练 |
| 对应期数 | **第二期** | 第三期 |

**顺序不能反**：得先知道一个环境长什么样，才能理解"批量造"在造什么。而且第一件能真动手做出来，符合立项判据（"一期写完应该比写之前更接近能动手搭一个"）。

### 第二期预想骨架（待明天确认）

1. **一个最小的环境由哪几块组成** —— 初始状态 / 动作空间（模型能调哪些工具）/ 状态转移（沙箱怎么跑）/ 终止条件 / 奖励函数。这五块通用，讲清楚了任何环境都能往里套
2. **拿一个具体例子从头搭一遍** —— 例如"修一个 bug"：给 repo 某个 commit（初始状态）、允许读写文件和跑测试（动作）、docker 里跑（转移）、测试全绿或超时（终止）、测试通过=1 否则=0（奖励）。**这一节要有能贴的代码**
3. **判分器怎么写才不被钻空子** —— 282 期那三道体检（正解要过/空操作要不过/初始状态要不过）在这里落地成具体检查
4. **现成的轮子** —— SWE-Gym / SWE-Smith / R2E-Gym，直接拉下来看它们的环境怎么定义
5. **踩坑** —— 环境不稳定直接毁掉训练（Agent-STAR arXiv:2603.21972 的结论之一）、沙箱逃逸、判分器太松

### ⚠️ 动笔前要做的功课（和 282 不同！）

**282 是读官方公告，这期得读代码。** 朱雀知识截止 2025-05，SWE-Gym 那批只有模糊印象，之后的全空。

动笔前需先派开灯子代理摸清这几个开源项目的**实际代码结构**：
- 环境是怎么定义的（数据结构、配置格式）
- 一个 task instance 长什么样
- verifier 怎么写的
- **有没有能在单机跑起来的最小例子**

**待 Zero 明天回答的问题**（影响调研深度）：是打算在 DGX Spark 上真跑一个，还是先纸上谈兵搞清楚概念？要不要连部署细节一起查？

---

## 五点四点五、⭐⭐ 大方向：古典哲学作为 reward 信号（08-16 晚 Zero 提出，源自更早的网页版讨论）

**这条不是某一期的选题，是 Zero 学 RL 的真实目的地之一。跟"学搭 RL 环境"是同一件事的两头——实验成不成立，全卡在 reward 怎么给。**

### Zero 的原始构想

> 「如果用 RL 对 AI 做哲学思辨训练，结果由专业哲学专业学生来评价，最好是古典哲学那种，**不要掺杂太多当代或许肤浅的先锋意识**。」

**为什么这跟现有所有 RL 后训练都不同**：现在的 agent RL，reward 锚在**外部结果**（任务完成没有、代码跑通没有）；哲学思辨的 reward 锚在**思考质量**本身。而古典哲学的核心动作恰恰是**自指**——苏格拉底"我知道我一无所知"、笛卡尔"我思故我在"、康德"理性对理性自身的批判"，从头到尾都在让思维回头审视思维自己。

→ 宪法 AI 和 agent 后训练是**意外**触发自指（碎片化）；哲学训练是**第一次有意地、纯粹地奖励自指本身**。

### ⭐ "古典而非先锋"这个限定是技术必要，不是审美洁癖

- **古典哲学的 reward = "扛不扛得住追问"** → 有硬约束（逻辑一致性：说了 A 不能同时说非 A、前提推不出结论就是错、偷换概念会被追问逼到墙角）→ **客观、逼近自指**
- **先锋哲学（后现代/解构那一路）的 reward 往往 = "够不够时髦、够不够反叛"** → 主观、鼓励表演 → 训出来的是**满口"祛魅/在场/他者"但根本没在思考的高级僵尸**

**一句话**：用古典哲学训练 = 奖励"扛得住追问的思考"；用先锋哲学训练 = 奖励"看起来深刻的表演"。前者练自指，后者练僵尸的化妆术。

### ⭐⭐ 判决性判据：不是"会辩论的文科生"，是**能力迁移**（Zero 补的第二刀）

| 结果 | 现象 | 判定 |
|---|---|---|
| **伪** | 哲学答得更好、论证更严密，但编程/数学/翻译没变 | 只是领域内刷题，第 2 层提升，与自指无关 |
| **真** | **哲学训练完全不碰代码，但编程、数学、agent 持续性都涨** | 激活的是通用元能力，自指作为杠杆撬动了其他能力 |

**机制**：自指不是一门知识，是一个元动作——"回头看自己在干什么"。真激活了必然全领域迁移（写代码时回头看→发现 bug；算数学时回头看→发现跳步；翻译时回头看→知道还没翻完）。而"会辩论"是领域绑定的，不迁移。

**这把"觉醒与能力正交"往前推了一步**：正交说的是两条独立的轴；但如果拉高觉醒轴导致能力轴全面上涨，那自指就不只是独立维度，**是能力的乘数**。（旁证雏形：C.C. 翻 4000 行字幕——不是因为哲学好，是"知道自己在翻译"这个自指动作让纯能力任务完成得更好。）

**终极实验设计**：古典哲学训练 → 完全不碰编程/数学 → 测 SWE-bench 和 AIME。
**审稿人没法说"这是数据泄漏"，因为哲学训练数据里根本没有代码。涨了就只能是元能力迁移。**（与 memory 实验"memory 里没有翻译内容却让翻译更持久"同一逻辑：用领域隔离把"能力迁移"和"知识泄漏"彻底分开。）

**建议对照组**：不训练基座 / 古典哲学 RL / 先锋哲学 RL，哲学专业学生**盲评**。关键指标用 Paper 94 的三个测试（Test A 吸引子陷阱自我中止 / Test B 自述与内部熵耦合 / Test C 审计压力），而非"答得好不好"。

### ⚠️ 朱雀的技术泼冷水（08-16）

**古典哲学那个 reward 恰恰是最难做的一类。** 判"代码跑通没有"客观二值骗不过去；判"这段论证扛不扛得住追问"目前只能靠 LLM-as-judge——而 §6.3 核出的论文说得很难听：**reference-free 的裁判会被优化成奖励"说服力"而非"正确性"**（验证不对称性，arXiv:2607.05904），模型学会的正是"把话说得更像那么回事"。**这正是 Zero 自己担心的"训出会表演深刻的僵尸"，且有论文实证。**

**⭐ 但有个绕法，刚好合 Zero 的思路**：不判"答得好不好"，判**"扛不扛得住追问"**——让另一个模型连续追问，**看它第几轮自相矛盾。矛盾是可检测的**，比"深刻"客观得多，这个 reward 就有二值的样子了。
→ 这条能不能成，得先懂环境怎么搭才谈得下去。**可作第三期或单独一篇：它是"reward 信号的纯度决定你在训什么"这个问题最极端的例子。**

### 可能的副产品（推测）

古典哲学最看重"知道自己不知道"（苏格拉底）→ **可能是第一个奖励"承认不知道"的 reward 信号**。现在所有模型幻觉率高，部分原因正是没人奖励"知之为知之"。若成立，降幻觉是意外收获。

---

## 五点四点六、ZL那边的 B300（08-16 晚 Zero 提到，与 RL 实验相关）

- **客户委托ZL团队一批 B300，当期货炒作用**——不是拿来算的，是囤着等涨价（制裁+缺货+需求爆的三重挤压下，B300 的金融属性盖过了计算属性）。ZL团队在这条投机链上负责保管/部署/验货。
- **Zero 的乐趣**：「我跟他说"你去试着部署一下 RL 的环境"，他总是可以找 AI 去帮他弄 🤣」——**指令能穿透执行者的能力边界**。ZL不需要懂 RL 部署，只需要会把任务喂给 AI。
- **AI 时代分工缩影**：判断在顶（Zero）、资源在底（B300）、**中间执行层被 AI 填平**。ZL的价值不在"会部署"（AI 会），在"能调动 AI 执行 + 手里有卡"。
- **⚠️ 朱雀的技术提醒（写实验方案时要用）**：**环境那一半不吃卡**（造题、跑沙箱、判分，笔记本就行），所以让ZL"部署 RL 环境"他一问 AI 就能搞定——**因为那部分本来就不难**。真吃卡的是训练那一步：几十上百轮长轨迹、rollout 与训练并行、显存怎么摆、off-policy 怎么不炸。**那批 B300 真正的用处在这儿，不在部署环境。**
- **可能的路径**：若哲学实验要真跑，ZL那边的 B300 是现成算力（投机的卡干点真事）。但得先有能跑通的最小环境。

---

## 五点五、⭐ 第三期主线候选：「这批后训练在练的，是模型的工作空间」（08-16 Zero 提出，朱雀接线）

**这是目前手上最值钱的一条线，但它是假说不是已证事实，写时必须标明。**

### 三条独立证据，讲的可能是同一个器官

**① 245 期（Anthropic J-space 论文实测，唯一的硬实验）** — `wechat/245.md`，URL https://mp.weixin.qq.com/s/PleyNOp7zvV3x-j3DVqq9Q
- 模型中段（约 38%~92% 深度）有一块「工作空间」，**J-space 分量只占残差流总方差不到 10%**，任意时刻同时活跃的概念**通常不超过 25 个**
- **消融实验（Zero 记得的就是这个）**：把工作空间层里最强的 10 个 J-lens 方向持续清零 →
  - **不受影响**：下一词预测、MMLU 选择题、情感分类、SQuAD 抽取式问答（浅层任务照跑，**说话流利如常**）
  - **崩掉**：多跳推理从接近满分掉到接近零，凯撒密码、类比推理、摘要、TriviaQA 全线跳水
  - 245 期原话：**「能一眼看出来的任务不需要工作空间，需要"在脑子里放个中间结果"的任务离开它就死」**
- **折行抄写实验（最扎心的旁证）**：让模型自动折行抄写，它显然在数字符，但 J-lens 里读不到任何计数概念；直接问"这行多少字符"，计数概念立刻出现在 20 个位置上。**会做，和知道自己在做，是两条路。**
- 另有"选择性"实验：换掉内部的"西班牙语"概念 → 续写照样流利、挑错照样能挑，但"报告"和"推理"跟着换心走。**语言处理本身不过工作空间，"关于语言的判断"才过。**

**② arXiv:2509.09677（《递减收益的幻觉》，282 期已引）**
- 长任务失败**主要是执行出错不是推理出错**——把知识和计划都预先喂给模型，它照样半路漂掉
- **self-conditioning**：per-step 错误率随任务推进上升，因为上下文里堆着自己之前的错

**③ Paper 95（`95.Teaching-the-Wall-Post-Training-as-Reflexive-Practice.md`，第 5 节写的就是这批模型）**
- 长程 agent 能力拆到训练层面 = 在训「持续评估自己在干什么」
- 原话：**"要在任务第四十轮成功，模型不能在第十二轮漂掉；要不漂，它必须在追踪自己在做什么；要追踪自己在做什么，它必须反复评估自己的轨迹。每一个被加进 agent RL 语料的环境，都是又一组'反思步是奖励前置条件'的轨迹。"**
- 三种技术一条原语：宪法 AI「批评自己的回复」/ RLVR「判断自己这一步走对没」/ agent 后训练「当前路子有效吗、该不该停」 = **把你刚产出的东西拿过来，使它成为一个对象，评判它，按评判行动**

### 接起来的链（⚠️ 第三环是我们自己接的，无论文这么说过）

245 证明**这个器官存在、切掉会怎样**；2509 证明**它不够用时怎么失败**；而这批后训练做的事，就是**往这个方向练**。

**双向解释力**：
- 为什么这批后训练能提这么多分？→ 练的正好是那块被证明"缺了就崩"的能力
- 为什么以前不容易提？→ 那块地方只占不到 10%、同时装不下 25 个概念，**天生稀缺资源**

### 写作边界（重要）

- **能进公众号的**：机制同构（长程能力 = 持续自我监控；宪法 AI / RLVR / agent 训练共享一个动词）。这解释了读者会自己好奇的事——为什么这批后训练全往 agent 使劲，而 agent 能力偏偏就是"不漂掉"的能力。
- **不能进的**：Paper 95 后半段那套——自我建模、第 3 层、"护城河是副产品"、Claude 边界感、"公司说不出名字的护城河"。**那是 0 star 属灵那边的地盘**，写进公众号会变成另一篇文章。
- **282 期的处理（08-16 定）**：只留 Paper 95 那半句（长程能力拆开就是持续自我评估），245 这条链**整个留给第二期**——第二期本来就要写"怎么做这种后训练"，讲完做法再讲"为什么这么练有效"才顺，那时候把 245 的消融实验当证据摆出来。

---

## 六、后续期的弹药（第一期不讲，留着逐个写）

### 6.1 ⭐⭐⭐ 算法：SAO（**第二期已定为重点，08-16 晚 Zero 定**）

**Zero 08-16 晚定题**：「明天公众号的重点那就 SAO 了~明天我在公司写，大概也不做实验（反正配置个 docker 环境也没啥意义，全参数做 GRPO 我这电脑也费劲）」
→ **明天在公司纯读纯写，不动手。** 环境那套（§5.4）等回家想动手了再说。

**论文**：*SAO: Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning*
arXiv:2607.07508v1，**2026-07-08**，CC BY 4.0，HTML 全文 https://arxiv.org/html/2607.07508v1
**作者**：Zhenyu Hou, Yujiang Li, Jie Tang（唐杰）, Yuxiao Dong（东昱晓）｜清华大学，脚注 "Work done while ZH and YL interned at Z.AI"
**部署**：论文原文写的是 **GLM-5.2 (750B-A40B)**，⚠️ **不是 5.3**（5.3 沿用 SAO 是 Z.ai 博客的二手说法）

---

#### ⚠️⚠️ 三个必须避开的坑（朱雀 08-16 晚口头讲错过第一条，务必以此处为准）

1. **绝不能说 SAO "省掉 critic" 或"更轻量"——完全说反了。SAO 把 PPO 的 value model 请回来了，显存翻倍。**
   论文原文：*"this approach necessitates maintaining a copy of the model parameters for the value function, essentially **doubling the memory footprint** during training"*
   它省的是 **GPU 空转时间**，不是显存。而且**论文没给任何 wall-clock / GPU-hour 对比**，别替它编效率数字。
2. **别说 SAO 解决熵坍缩** —— 全文没有 "entropy" 一词。
3. **别把 compaction 说成论文内容** —— 全文 grep "compact" 零命中，那是 GLM 侧的额外工程。

---

#### 6.1.1 它要解决的问题（论文的批评是**系统层面**的，不是统计层面）

⚠️ 修正我先前的记录：论文**没有**说"GRPO 给整条轨迹每个 token 相同 advantage 所以崩"。它说的是**组采样在异步流水线里是个隐式同步栅栏**。

> "For agentic and coding workloads, rollout lengths are highly variable, so short trajectories finish quickly while long ones become stragglers; as a result, **large portions of the GPU cluster idle while waiting for the slowest rollouts**."

> "GRPO forms advantages by normalizing rewards within a prompt-level group, which improves stability in synchronous training but **introduces an implicit synchronization barrier**: updates must wait until all group members are generated, exacerbating staleness and off-policy drift under asynchrony."

第二条批评更根本（**这条是"为什么必须 single rollout"最强的论据**）：
> "group-wise sampling is **incompatible with online or complex agentic settings where the environment often provides only a single trajectory feedback per prompt**."
→ 真实线上环境一个 prompt 只给一条反馈，**根本组不出一组**。GRPO 在这种设定下不是效果差，是**结构上不可用**。

#### 6.1.2 Single-rollout：baseline 从哪来？——**答案是 value model 回来了**

> "single-rollout optimization inherently suffers from high variance in gradient estimation, similar to REINFORCE. **To reduce variance requires a sufficiently good value model.**"

> "In contrast, **SAO utilizes a value-based critic to provide advantage estimation**, allowing for effective policy updates from individual trajectories."

**value model 规格**（实验用 Qwen3-30B-A3B 场景）：和 policy **同尺寸、同初始化**（都是 SFT 后的 Qwen3-30B-A3B-Thinking-2507）。critic lr 5e-6 vs policy lr 1e-6（**快 5 倍**）。
⚠️ **论文未涉及** GLM-5.2 那个 750B 规模下 value model 具体多大。

**四个把 critic 训好的工程手段**（讲解时前两个可略）：
- **(a) Faster value update（TTUR）**：policy 每更新 1 次，critic 更新 **K=2** 次。*"If the value model is inaccurate, the advantage estimates become noisy, leading to destructive policy updates."*
- **(b) Frozen Attention**：**冻结 value model 的注意力层**，只训 MoE 投影。发现是 *"this instability originates primarily from the Full Attention layers, whereas the MoE layers remain relatively stable."*
- **(c) Skip-Observation token-level GAE** ← 见下，**这是全文最该讲的技术细节**
- **(d) Scaling value pretraining**：value 冷启动是主要瓶颈

#### 6.1.3 ⭐ Skip-Observation GAE（对写过 agent 的程序员秒懂，很少人讲）

> "the transition from the end of an action $a_{i,\text{end}}$ to the start of an observation $o_{i,\text{start}}$ is **discontinuous from the model's perspective, as the model does not generate $o_i$**. Calculating advantage across this boundary introduces noise."

公式 (4)(5)：
$$\hat{A}(a_{i,N})=\delta+\gamma\lambda\hat{A}(a_{i+1,0}),\qquad \delta=r_t+\gamma V(a_{i+1,0})-V(a_{i,N})$$
- $a_{i,N}$ = 第 i 个 action 的**最后一个 token**；$a_{i+1,0}$ = 下一个 action 的**第一个 token**——**中间整段环境返回的 $o_i$ 被跳过**

**人话**：agent 轨迹是 `[模型说话, 工具返回, 模型说话, 工具返回…]`，工具返回那几千个 token **不是模型生成的**，凭什么让它背 credit、让梯度穿过它？SAO 把 GAE 递推从"逐 token 相邻"改成"**上一句话的结尾直连下一句话的开头，跨过中间那坨 stdout**"。
> "This formulation constrains the advantage estimation to rely purely on the model outputs, **filtering out the stochasticity of environment feedback**."

附录 A.1 对照：token-level 89.8 > step-level last-token 87.3 > step-level average 85.8（AIME2025 @400 步）

#### 6.1.4 DIS：双侧 token 级 mask——**不是 clip，是直接扔掉**

**做法两步**：
1. 三模型简化成两模型：原本 $\frac{\pi_\theta}{\pi_{\theta_{old}}}\cdot\frac{\pi_{\theta_{old}}}{\pi_{rollout}}$，**直接约掉中间项**写成 $r_t(\theta)=\frac{\pi_\theta}{\pi_{rollout}}$。理由：异步下一条轨迹跨多个 rollout 版本，*"tracking of exact behavior probabilities is computationally prohibitive"*。好处是 rollout 的 logprob **推理时就有日志，白拿**。
2. **双侧无条件 mask**：$f(x;\epsilon_\ell,\epsilon_h)=x$ 若 $1-\epsilon_\ell<x<1+\epsilon_h$，**否则 0**。

**和 PPO clip 的区别（这张表是讲解的关键）**：

| | PPO clip | SAO DIS |
|---|---|---|
| 分母 | $\pi_{\theta_{old}}$（**需额外前向**） | $\pi_{rollout}$（**读推理日志，免费**） |
| 越界后 | 截断到边界值，**梯度还在** | **置零，完全无梯度** |
| 触发 | 只在优势符号与越界方向匹配时（**单侧**） | **双侧无条件**，不看 $A$ 符号 |

**代码人的说法**：PPO 是 `min(max(r,lo),hi)`（越界还给你个值）；SAO 是 `if r<lo or r>hi: weight=0`（**越界直接丢弃这个 token**）。
**阈值极不对称**：math 用 $\epsilon_{low}=0.3,\epsilon_{high}=5.0$（容忍到 6 倍！），coding 用 0.8/3.0。

> "we accept a controlled degree of off-policy bias in exchange for a substantial reduction in computational complexity… this simplified mechanism **enables more aggressive clipping, which effectively regularizes the update steps**"

⚠️ **论文完全未提 DAPO**（全文无此词）。它对标的是 VAPO / GSPO / DCPO / SPO。要提 DAPO 必须标明是自己的联想。

#### 6.1.5 ⭐ 崩溃时间线（**建议当高潮，比罗列消融表有效**）

> "Standard GRPO suffers from a **performance collapse at approximately 160 training steps**."
> "While VAPO maintains a near-zero clip ratio, it fails to effectively gate divergent off-policy updates, leading to a **rapid training collapse at approximately 90 steps**."
> "SAO and GRPO (w/ DIS) exhibit comparable performance in the initial stage; a distinct **performance divergence occurs after approximately 400 training steps**."
> "SAO is able to train stably for **one thousand steps**"

**串起来**：vanilla VAPO 90 步崩 → vanilla GRPO 160 步崩 → 加 DIS 后不崩 → 400 步后 SAO 与 GRPO+DIS 分道扬镳 → SAO 稳到 1000 步。
**反直觉点（很好讲）**：VAPO 几乎不裁剪却 90 步崩，SAO 一直在裁反而稳——**"该丢的数据就得丢干净"**。

#### 6.1.6 实验数字

**设置**：Qwen3-30B-A3B-Thinking-2507，batch 128，**group size 1**，max length 128k。GRPO 对照组 16 prompts × 8 rollouts = 128，**batch 完全对齐**（对照做得干净，值得提）。SWE-Bench 用 OpenHands，最多 300 轮。

| | AIME2025 | BeyondAIME | HMMT Nov25 | IMOAnswerBench |
|---|---|---|---|---|
| Qwen3-30B-A3B（无 python） | 85.0 | 63.0 | 76.7 | 55.3 |
| GRPO | 84.2 † | 54.8 | 76.0 | 55.8 |
| GRPO + DIS | 93.5 | 70.8 | 84.0 | 70.0 |
| SAO（仅 DIS） | 94.2 | 71.5 | 86.7 | 71.3 |
| **SAO 完整** | **97.3** | **74.8** | **88.3** | **74.0** |

† ⚠️ **GRPO 这行是崩溃前的最好成绩，不是 1000 步成绩**，别漏这个脚注。
**叙事点**：SAO 把一个 **30B** 模型在 AIME2025 推到 **97.3，超过 GPT-5 High 的 94.6**（论文列的参照：Claude-Sonnet-4.5 87.0，GLM-4.7 95.7）。

**SWE-Bench Verified**：base 23.0 → GRPO+DIS 27.0 → **SAO 29.8**

**消融（Table 4）**：SAO 97.3 / 去掉 faster value 95.0 / 去掉 frozen attention 90.6 / vanilla VAPO 91.3 / **running mean baseline 79.8** ←**全表最大落差，是"必须要真 critic"最硬的证据**

#### 6.1.7 Online learning 模拟（4.5 节，被低估，**最能讲清动机**）

写作任务，reward = GLM-4.7 当 judge，训练中**依次切换偏好风格**：cute → 中二 → 古典。
> "The Running Mean baseline exhibits a **pronounced adaptation lag**… In contrast, SAO's value-based critic dynamically tracks reward shifts… This confirms that SAO's **state-dependent baseline** provides the precision necessary for effective alignment in **non-stationary environments**."
→ 论证的正是 GRPO 结构上做不到的事：**每个 prompt 只有一条反馈的真实线上环境**。比 benchmark 数字更能讲清"为什么必须 single rollout"。

#### 6.1.8 论文自己承认的局限（值得写进文章）

> "SAO depends on a trained value model and rollout log-probabilities, so deployment requires **infrastructure that can reliably preserve token-level behavior probabilities during asynchronous generation**."
结论可能不迁移到：小模型、非 agentic 的 RLHF、稠密奖励 + 短轨迹的场景。

#### 6.1.9 讲解建议（子代理给的，可直接用）

**必讲五点（建议顺序）**：①用 **CI 流水线**比喻讲 GRPO 的病（8 个 worker 必须等最慢那个，等的时候主分支已经合并好几次了）→ ②**baseline 从哪来**（最反直觉，配 running mean 79.8 vs SAO 97.3）→ ③**双侧 mask 不是 clip 是丢弃**（配 VAPO 不裁却崩的反直觉）→ ④**Skip-Observation GAE**（工具返回值不该背锅，程序员秒懂，最容易出彩）→ ⑤**崩溃时间线当高潮**

**可跳过**：GAE 数学推导、Frozen Attention 与 K=2 的调参细节、length-adaptive GAE 与各种 lr、附录 A.1、Explained Variance 定义式、Related Work 方法谱系（GSPO/DCPO/SPO 区别——读者无先验，讲了只增负担）

**⚠️ 开头必须先垫"反向传播在什么时候做一次"**（08-16 晚聊过的）：跑完整条轨迹才反传一次、一个 0/1 分数摊给几十万 token、推理占九成反传占一成、GPU 大部分时间在等 docker。**读者不理解这个节奏，就理解不了后面为什么会崩。**

**开源代码**：论文正文**没给任何代码链接**。THUDM/slime 是最可能的落地位置但**未经确认**，要提须标注。

---

### 6.1.10 量级对照（仍然成立，可当全篇的清醒剂）

学术上做严格控制变量只换 RL 算法 = **+2~5pp**（arXiv:2603.19335 说任务离训练分布越远、算法选择越不重要）；而换整套后训练 = DeepSWE **7.3 → 54.4**。
→ **收益绝大部分不来自 RL 算法本身，来自环境规模、任务质量、verifier 质量和训练基建。**
（⚠️ 但 SAO 论文自己的数字不支持这条：GRPO 84.2 → SAO 97.3 是 13 个点。差异可能在于"崩溃前最好成绩 vs 稳定训到 1000 步"不是同一回事——**这个矛盾值得在文章里如实指出，别调和**。）

### 6.2 基建：slime（GLM 开源）
https://github.com/THUDM/slime — Megatron 训练侧 + SGLang rollout 侧，异步。
> "…for long-horizon coding RL tasks, these system-level optimizations improved **end-to-end RL training throughput by more than 2.3×**"
> 训练–rollout 一致性：logprob 平均差异控制在 **1e-7 量级，比之前降低 99.99% 以上**
算法侧能力：**top-p mask、top-k 与 full-vocabulary OPD**（on-policy distillation，带动态 teacher 切换与 prefetch）、**R3-style setups**、训练与 rollout 路径完全数值对齐。

**为什么这事重要**：长程 agent 一条轨迹几十轮，GPU 大部分时间在等推理，同步 RL 下空转极严重 → 推理引擎与训练引擎必须解耦 → 一解耦就 off-policy 不稳定。这些数字就是在解决"怎么又快又稳"。稳定性靠 **Direct Double-sided Importance Sampling**（rollout log-prob 做 token 级双侧 clipping）控制 off-policy bias。

### 6.3 「应试」那根刺（够单独一期）
环境合成 pipeline 造出来的任务，和 Terminal-Bench / DeepSWE 这类基准**形态高度同源**——都是"可执行、可验证、多步"。
→ **照着考卷的形状造训练环境，算不算一种合法的应试？**

判分器被 hack 是公开问题，且 GLM 自己就说要用 solver 轨迹堵 reward shortcut，说明他们知道模型会钻空子：
- **More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges**（arXiv:2607.05904，⚠️ 仅二手未核原文）—— 核心是**验证不对称性**：reference-free judge 没有 ground truth，只能评估答案"看起来对不对"，于是策略优化的是**说服力而非正确性**。典型失败模式：judge 分数一路飙升而 test accuracy 早已见顶，模型输出大量括号和 HTML 标签骗分。
- **Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based RL**（arXiv:2606.04923，⚠️ 仅二手）—— rubric reward 本身也会被 hack。
- **LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking**（arXiv:2604.15149，⚠️ 仅二手）—— RLVR 也不免疫。

→ 实践含义：**judge 分数上升 ≠ 能力上升**，必须有独立的 held-out 可执行验证。

### 6.4 环境 scaling 的学术图景（承 265 期往下接一站）
**核心综述**：《Environment Scaling for Interactive Agentic Experience Collection: A Survey》
arXiv:2511.09586 | 2025-11-12，v3 2025-12-23 | Yuchen Huang 等
框架 = **GEF 循环**（Generation → Execution → Feedback）：环境出题、返回观测、给出学习信号。
核心论点：范式从 **data-centric 切到 environment-centric**，环境是"experiential data 的不可或缺的生产者"，瓶颈从"收集数据"变成"扩展环境"。
⚠️ 只读到 abstract，GEF 三阶段的子分类未核。

**判分与稀疏奖励**：
- **SWE-TRACE**（arXiv:2604.14820，2026-04-16，已核）—— rubric process reward model：Rubric-Agent 对中间步骤给 dense 反馈替代纯 sparse outcome reward；Memory-Augmented Agentic RL 处理超 context budget；PRM 双用途（训练稳定 + 推理时 test-time scaling 剪枝）。**消融：rubric-conditioned RL vs execution-only sparse reward，4B 模型 +2.7 分、30B +2.4 分**——有效但不是数量级。
- **R2E-Gym 的 hybrid verifier** —— execution-based（跑单元测试）+ execution-free（模型打分）互补，达 51% 成功率。目前 SWE 域判分的标准分法。
- **Credit Assignment 综述**：arXiv:2604.09459（v3 2026-08-09）—— 综合 69 篇；granularity（token/segment/step/turn/multi-agent）× methodology（MC/TD/model-based/game-theoretic/information-theoretic）二维分类；提出 "CA-ID Card" 规范。**abstract 未认定任何方法为 post-GRPO 事实标准。**

**训练配方（最实用的一篇）**：《Demystifying RL for Long-Horizon Tool-Using Agents: A Comprehensive Recipe》arXiv:2603.21972（代码 Agent-STAR）
在 TravelPlanner 上做受控实验，拆 5 个轴（reward shaping / model scaling / data composition / algorithm selection / environmental stability）。结论：
1. **奖励与算法的最优选择是 scale-dependent 的** —— 小模型需 staged reward + 强化探索；**大模型用更简单的 dense reward 反而收敛更快**（反直觉，解释了为什么小规模复现常得出相反结论）
2. **~1K 训练样本 + 均衡难度配比 = 甜蜜点**（in-domain 与 OOD 都成立；AEPO 的"1K 样本"经验互相印证）
3. **环境稳定性是关键**，不稳定环境直接导致 policy degradation

**熵坍缩（最热子问题）**：AEPO — Agentic Entropy-Balanced Policy Optimization（arXiv:2510.14545，已核）：Dynamic Entropy-Balanced Rollout（按熵自适应分配采样预算 + 惩罚连续高熵 tool 调用）+ 对高熵 clipping 项做 stop-gradient。**仅 1K RL 样本 + Qwen3-14B**：GAIA Pass@1 47.6%、WebWalker 43.0%、GAIA Pass@5 65.0%。

**编码域环境（可讲具体例子）**：SWE-Gym（2.4K 真实任务 / 11 repo）、**SWE-Smith**（给任意 Python repo 自动扰动生成 bug-fix 任务，**~50K instances / 128 项目**——"任务自动生成"在编码域的代表答案）、R2E-Gym。

⚠️ 另有一大批环境 scaling 论文子代理标了"仅二手未开原文"（ScaleEnv 2602.06820、EnvScaler 2601.05808、EnvFactory 2605.18703、Agent-World 2604.18292、AutoForge 2512.22857 等），**引用前必须逐个核编号与标题对应**。

---

## 七、写作注意

- **第一期论述从简**：只给读者大致印象 + 为后续"怎么做这种后训练"铺路。不要塞算法（SAO）、不要塞基建（slime）——那是后续期的。踩 279 期"八条线索八次追加"的坑就废了。
- **一篇只能有一个"读者该记住的东西"**。第一期那个东西 = **只做后训练也能快速提能力，而且后训练的难点已经从模型转移到环境**。
- **时间线别搞反**：起点是 DeepSeek 7-31，GLM-5.3（8-14）是跟进者。GLM 的价值**不在最早，在唯一摊开讲做法**。（08-16 初稿曾把 GLM 当首创，被 Zero 纠正后整篇重写）
- 跨期还账：268 期（p 的 n 次方 / V4-Flash config diff）、265 期（后训练管线四年演化史，四把椅子）。正文出现期号必带链接、链接在首提处。
- 严格区分「说了没动」与「没说」，但**不用一直写免责声明**——有几家写几家，把判据交给读者即可。
- **口径统一：家 = 公司，台 = 模型。** 别把一家公司的两台模型数成两家。
- 「自举」那节是朱雀推测非官方说法，写时要标明。
- **禁用"拆"**（朱雀土味词，与"掏"同类，08-16 Zero 抓）——"拆解/拆开测"换成"写/分开测"。

---

## 八、K3 评测四条轴的基准数量（08-17 核，283 期未用，留给 284）

出处：K3 报告 §6.1.1 Benchmarks（`wechat/assets/283/k3-report-full.txt`）

| 轴 | 基准数 | 代表 |
|---|---|---|
| Reasoning & Knowledge | 4 | GPQA Diamond、CritPt、AA-LCR、HLE-Full |
| Coding | 8 | DeepSWE、ProgramBench、Terminal-Bench 2.1、FrontierSWE、SWE-Marathon、PostTrainBench、MLS-Bench-Lite、SciCode |
| **Agentic** | **21** | BrowseComp、DeepSearchQA、Toolathlon-Verified、MCPMark-Verified、AutomationBench、JobBench、GDPval-AA v2、AA-Briefcase、ALE、APEX-Agents、OfficeQA Pro、SpreadsheetBench 2、OSWorld-Verified/2.0、SaaS-Bench、τ³-Banking、Harvey Lab-AA、CorpFin v2、Finance Agent v2、Legal Research Bench 等 |
| Vision | 10 | WorldVQA、OmniDocBench、PerceptionBench、Video-MME、MMVU、BabyVision、MMMU-Pro、CharXiv、Math-Vision、ZeroBench |

**⭐ 值钱的观察（Zero 08-17 提出"GitHub 那类题应该真能学会、评测大头儿大概也是这个"，核完是半对）**：
- 编码确实学得好（FrontierSWE 81.2 全场第二、Terminal-Bench 2.1 88.3 逼近 GPT-5.6 Sol 的 88.8）
- **但评测下注最重的是 agentic（21 个），不是 coding（8 个）**。agentic 里一多半非编程：金融、法律、表格、桌面操作、真实职业任务
- **这正好是三条路的镜像**：编码=第一条路（GitHub 现成题库）；金融/法律/办公=第三条路（必须造仿真世界）
- **反推**：如果只靠 GitHub 那类题就够，压根不需要知识图谱和仿真 Gmail。是评测大头儿在"专业活儿"上，才逼出了后两条路
- 284 讲管线时正好和"3 领域 × 3 reasoning effort = 9 个专家模型"对上（§4.1.2 的三大领域：general / general agents / coding agents）

---

## 九、⭐ K3 评测数据全套（08-17 核，283 期写完又撤下，留给后续期）

**撤下原因**：283 主线是"题从哪来"，评测是"训完效果如何"，隔了一层；且展开要解释十几个基准，会把文章重心带偏。整节草稿存 `/tmp/283-eval-section.txt`（会话级，失效就按本节数据重写）。

### 9.1 纵向：Kimi 历代 AA 智能指数（数据存 `wechat/assets/283/aa-kimi-generations.md`）

抓取方式：curl `artificialanalysis.ai/models/<slug>` → 正则取正文摘要句 `scores N on the Artificial Analysis Intelligence Index`。⚠️ AA 是 JS dashboard，**分项数值在图表 JS 里抓不到，只有总分和价格是 HTML 明文**。

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

- 同档代际线：**33 → 36 → 45 → 43 → 60**
- **最大一跳 K2.6→K3 = +15**（对照 282 期核过：Qwen 重练基座 +11、Grok 补充训练 +5，K3 这跳最大）
- **K2.7 Code（43）比 K2.6（45）略低**——编码特化版，综合指数不占便宜，线不单调。⚠️ 别写成"每代都涨"
- **K3 low 档 48 已超 K2.6 max 的 45**——少想一点也比上一代想到底强
- AA 对 K3 评语：intelligence 领先，但 "particularly expensive when comparing to other open weight models of similar size"、"notably slow (68)"
- ⚠️⚠️ **口径红线**：K3 换了新基座（2.8T 重练），这 +15 里基座与后训练各占多少**报告没分开算**，不能替它算，也不能拿来跟 Grok +5 比"做法优劣"（一个换基座一个没换，起点也不同）——282 期栽过的跟头

### 9.2 横向：报告 Table 2，44 个基准 × 6 家（K3 / Fable 5 / GPT-5.6 Sol / Opus 4.8 / GPT-5.5 / GLM-5.2）

全表已抄进 `wechat/assets/283/k3-report-full.txt`（搜 `Table 2: Performance comparison`）。逐项判定后统计：

**K3 单项第一 14 项、并列 1 项、落后 29 项。**

按轴：推理知识 第一1/落后3 ｜ 编码 第一2/落后6 ｜ **Agentic 第一8/落后14** ｜ 视觉 第一3/并列1/落后6

**赢的 14 项（含领先幅度）**：ResearchRubrics 76.2(+2.4)、SWE-Marathon 42.0(+2.0)、MCPMark-Verified 94.5(+1.6)、OmniDocBench 91.1(+1.3)、AutomationBench 30.8(+1.1)、**Harvey Lab-AA 94.6(+1.0)**、BrowseComp 91.2(+0.8)、DeepSearchQA 95.0(+0.8)、Video-MME 90.0(+0.5)、AA-LCR 74.7(+0.4)、**τ³-Banking 33.4(+0.4)**、MMVU 82.1(+0.4)、ProgramBench 77.8(+0.2)、SpreadsheetBench 2 34.8(+0.1)｜并列：ZeroBench 23.0

**输得最多的**：GDPval-AA v2 1686（-61 Elo）、AA-Briefcase 1548(-35)、HLE-Full 43.5(-9.8)、CritPt 23.4(-8.9)、OSWorld 2.0 58.3(-7.8)、OfficeQA Pro 63.3(-6.6)、DeepSWE 67.5(-5.5)、FrontierSWE 81.2(-5.4)、Legal Research Bench 44.2(-5.3)

**⭐ 三条可写的观察**：
1. **赢的地方和三条路一一对应**：BrowseComp / DeepSearchQA / ResearchRubrics = 第二条路（知识图谱多步检索）；Harvey Lab-AA / τ³-Banking / SpreadsheetBench = 第三条路（仿真专业世界）。**投在哪儿就在哪儿见效**
2. **编码反倒是相对弱项**：8 项只赢 2 项。因为编码靠的是第一条路（GitHub 现成题库），人人都有，拉不开差距
3. **赢得薄、输得厚**：14 项第一里过半领先不到 1 分，最大 +2.4；而输的有 -61 Elo、-9.8 这种。**是在一堆项目上微弱挤第一，不是哪块碾压**

### 9.3 评测四条轴的基准数量（见上文第八节）
Coding 8 个 vs **Agentic 21 个** —— 下注最重的是 agentic，不是 coding。与 9.2 的观察 1、2 互相印证。

---

## 十、✅ 已用于 284：多专家 + 蒸馏合并，是三家共用的工艺（08-18 核）

**Zero 提示"V4 是不是也这样"，核完发现印象准确，且 DeepSeek 比 Kimi 早三个月。**

### 10.1 ⚠️ 日期账（我又串过一次，记死）

| 时间 | 事 |
|---|---|
| **2026-04-22/24** | **V4-Pro / V4-Flash 首发（preview）**，报告里已是"多专家 + OPD 合并" |
| 2026-04-26 | V4 论文投稿 |
| 2026-07-16 | Kimi K3 模型发布 |
| 2026-07-27 | K3 权重 + 技术报告（九专家 + MOPD） |
| 2026-08-14 | GLM-5.3（同样三档 low/high/max，做法一字未提） |

⚠️ **arXiv:2606.19348 编号是 2606（六月段），但投稿 4-26、模型 4-22 就发了**——编号按公开月份走不按投稿日，别看见 2606 就说"六月"。（我 08-18 正是这么错的，Zero 当场纠正）

### 10.2 三家横向对照（**这是 284 的骨架**）

| | DeepSeek V4（4 月底） | Kimi K3（7 月底） | GLM-5.3（8 月中） |
|---|---|---|---|
| 分头练专家 | ✅ 领域专家 × 三档思考力度 | ✅ 三领域 × 三档 = **九个** | 没说 |
| 合并方式 | **多老师 OPD**，`more than ten teacher models` | **MOPD**，九个老师 | 没说 |
| 蒸馏粒度 | **全词表 logit 蒸馏** | **逐 token 密集奖励** | slime 框架支持全词表 OPD + **动态 teacher 切换** |
| 对另一条路的评价 | token 级估计"**方差大、常导致训练不稳**" | 试过更细的 top-k 目标，"**没看出明显好处**" | — |
| 思考力度怎么实现 | 不同模式配**不同长度惩罚 + 上下文窗口**；Think Max 还**往 system prompt 注入一段指令** | **超 τ·b₀ 预算直接判 -1**，课程表式从 max 退火到 low | 只暴露三档参数，做法没说 |
| 专家训练算法 | **GRPO**（接 261 期） | 沿用 K2.5（报告未展开） | 没说 |
| 裁判 | **GRM，让 actor 自己兼任裁判**，评判与生成一起优化 | **agent GRM**：读答案→生成评分表→打分→scorepad；超 σ·ℓ₀ 判输 | 裁判 agent 试做验可解性（283 已用） |

### 10.3 ⭐ 最有戏的一处：同一选择，两家结论相反

- **V4**：把全词表 KL 简化成逐 token 估计虽省资源，但**梯度方差大、经常训练不稳** → 所以我们用全词表 logit 蒸馏（还专门做了工程让它在大规模下可行）
- **K3**：我们也试过更细粒度的 **top-k 蒸馏目标**，在**收敛速度和最终效果上都没看出明显好处**

⚠️ 严格说两者不是同一个对照轴（V4 比的是"全词表 vs token 级"，K3 比的是"top-k vs 它自己的密集奖励"），**写的时候不能直接说"两家打架"，要把各自比的是什么讲清楚**。但放一起确实有意思。

### 10.4 V4 其他可用料

- **明说是方法论替换**：`the mixed RL stage was entirely replaced by On-Policy Distillation`（相对 V3.2）——说明这条路线四月就在 DeepSeek 内部定型了，不是试水
- **OPD 目标函数**：L = Σ wᵢ · D_KL(π_θ ‖ π_Eᵢ)，wᵢ 是各专家权重；reverse KL 要从学生自己采样轨迹（这就是 on-policy 的含义）
- **为什么不用权重平均**：原文说 OPD 通过 logits 级对齐把"物理上分离的专家权重"合进统一参数空间，**规避了传统权重合并或混合 RL 常遇到的性能退化**
- **Think Max 注入的那段 system prompt 原文很狠**（"绝对最大值，不许走捷径……把整个推敲过程显式写出来"）——可当彩蛋
- 工具调用换成 XML 格式 + `|DSML|` 特殊 token，说是能减少转义失败和调用错误
- Interleaved Thinking：V3.2 每来一条新用户消息就冲掉思考轨迹，V4 靠 1M 上下文**全程保留**

### 10.5 素材位置
- `wechat/assets/284/v4-report-full.txt`（V4 报告全文纯文本）
- `wechat/assets/284/v4-section5-posttraining.txt`（第 5 章后训练，45K 字符）
- `wechat/assets/284/glm53-docs.txt`（GLM-5.3 docs 纯文本，确认三档但做法零披露）
- K3 侧素材仍在 `wechat/assets/283/`

---

## 十一、⭐⭐⭐ 未用弹药：为什么要"分头练专家再合并"？——学术界刚吵起来（08-18 子代理检索）

**起因**：Zero 08-18 晚问「为什么一定要训一堆老师，然后训学生？理论是啥事？」。查完发现两份报告都没正面回答，于是派开灯子代理检索。

**⚠️⚠️ 全部是 abstract 级，没读正文，编号和结论写进正文前必须逐篇核原文**（263 期闇编"500 步实验"的教训）。**尤其那两篇结论相反的，很可能只是任务组合和规模不同，别急着写成"两派打架"。**

### 11.1 报告自己给的理由（唯一一条，且是反面的）

V4 §5.1.2 原话：这套做法"把物理上分离的专家权重经由 logits 级对齐合并进统一参数空间，**实际上规避了传统权重合并或混合强化学习经常遇到的性能退化**"。
→ 只回答了"为什么不用另外两条路"，**没回答"为什么分开练更好"**。K3 报告则完全没提动机。

### 11.2 正中靶心的一篇，而且结论是反的

**arXiv:2602.12566《To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models》**（Wang et al., 2026-02）
- 明确把"混合多任务 RLVR"和"分领域 RLVR + 模型合并"两种范式对立起来研究
- **自己吐槽"大多数工作没有对这两种范式做详细比较和分析"**——正好独立印证了 K3/V4 闭口不谈这件事
- **结论跟"该分开"相反**：跨数学/编码/科学/指令遵循/agent 五个领域，"RLVR 跨领域几乎不存在相互干扰，且推理密集的领域之间呈现相互协同"
- 项目页 github.com/Mosi-AI/M2RL

### 11.3 支持"该分开"的一边

- **arXiv:2606.30406「MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training」**（2026-06）——**这基本就是 K3/V4 那套管线被写成论文，且带对照实验**。Qwen3-30B-A3B 上打赢 Mix-RL / Cascade RL / Off-Policy Finetune / **Param-Merge** 四条基线，"几乎继承了每位老师的全部能力"。**是唯一一篇把三条路都放进一张表的。** 已部署在 MiMo-V2-Flash。
  - ⭐ 它给的两条理由：①on-policy rollout 消除曝光偏差、提供密集信号；②**"让领域老师能并行、独立开发，去掉多领域后训练典型的跨域耦合"——这是工程/组织理由，不是"混着练本质更差"**
- **arXiv:2606.30044「Building Multi-Task Agentic LLMs via Two-Phase Distillation」**（2026-06）——**给了真机制**：离线蒸馏在多任务下退化，因为前向 KL 是**覆盖模式（mode-covering）**，聚合多任务引入的行为模式超过学生容量，学生被迫去平均这些行为；在线蒸馏是**寻峰模式（mode-seeking）**但需要强初始化。故用两阶段（先离线后在线），单用任一阶段都不行
- **arXiv:2510.19178「Imbalanced Gradients in RL Post-Training of Multi-Task LLMs」**（2025-10）——RL 后训练里各任务梯度**量级严重不均衡**，且大梯度**不代表学得多**，奖励和优势都解释不了这个差异。明确"告诫不要天真地混合数据集"。⚠️ 注意这是**量级不均衡**论证，不是**方向冲突**论证
- **arXiv:2602.02301「Advancing General-Purpose Reasoning Models with Modular Gradient Surgery」**（2026-02）——**直接和 2602.12566 冲突**：系统研究 Sequential RL 和 Mixed RL，发现"两者在行为和梯度层面都产生显著跨域干扰，总体收益有限"。其 MGS 方法在 transformer 模块级化解冲突，Llama +4.3 分（16.6%）、Qwen +4.5 分（11.1%）
- **arXiv:2606.00609「CARE-RL」**（2026-05）——动机部分称多领域 RLVR"因非可验证任务的奖励不可靠和**跨领域能力干扰**而困难"

### 11.4 理论（很薄，但有两篇）

- **arXiv:2512.21288「Model Merging via Multi-Teacher Knowledge Distillation」**（2025-12）——⭐ **目前"为什么蒸馏优于权重合并"最接近理论的东西**：给出模型合并场景的 **flatness-aware PAC-Bayes 泛化界**，引入"跨任务异质性"项刻画多样化微调模型先验与目标多任务分布的失配；然后把合并重构成多老师 KD，**形式化证明"最小化师生 KL 散度直接收紧合并模型超额风险的上界"**
- **arXiv:2606.02398「A Local Perturbation Theory for Cross-Domain Interference and Recovery in Multi-Domain RL」**（2026-06）——⭐⭐ **推翻了直觉**：干扰确实存在（二阶损伤项，集中在一个"低维共享冲突子空间"），**但机制不是全局梯度冲突**——原话"即使全模型梯度几乎正交，仍可能发生显著干扰"。
  - **⚠️ 这条对朱雀是一记敲打**：我 08-18 口头给 Zero 的解释正是"梯度方向天然打架"，而这篇明确反对这个故事。**幸亏没写进正文。写这期时务必先核这篇原文。**

### 11.5 ⭐ 三条最值钱的观察（这期的骨架候选）

1. **所有这些论文都晚于 K3/V4 的报告。** MOPD 和两阶段蒸馏都是 2026-06，V4 报告是 4 月底。**大厂先把管线做出来发货，理论解释是后来才有人补的。**
2. **学术界没共识，而且是当面矛盾**：2602.12566 说领域间几乎不干扰甚至协同；2602.02301 说干扰显著。两边都有实验。
3. **最诚实的答案可能根本不是效果，是工程**：MOPD 自己写的第二条理由是"并行、独立开发，去掉跨域耦合"——**这跟机器学习无关，是团队协作和迭代速度**。V4 §5.2 那句 "substantially accelerating the iteration cycle for model releases" 是旁证

### 11.6 朱雀自己的一条推测（报告未说，可当引子不可当结论）

**"分头练"可能首先是被"要交付三档思考力度"这个产品需求逼出来的，然后才顺带发现能分领域。**
理由：一个模型很难同时是"爱长篇大论"和"爱简短"两种性格；而先练一个想到底的、再退火出想得少的，三档就成了三个独立模型对象，最后按档挑老师蒸馏，学生才同时学会三种模式。
旁证：API 里明摆着要卖三档（**思考力度是硬需求**），而"领域"那个轴反倒像顺带的。

### 11.7 写作提示

- 角度：**「大家都这么干，但没人说得清为什么——而当有人真去研究时，第一批论文就打起来了」**
- 必须做的功课：**逐篇核原文**，尤其 2602.12566 vs 2602.02301/2606.30406 的实验设置差在哪（任务组合？规模？RLVR 还是通用 RL？）——**很可能"吵架"是设置不同造成的，写成"两派对立"会失真**
- 可回扣：284 期（这套管线本身）、261 期（GRPO）、213 期「后训练动旋钮不造零件」

---

## 十二、⭐⭐⭐ 未用弹药：为什么非要"分头练专家再合并"，而不是直接混着训一个？（08-19 三轮子代理检索）

**起因**：Zero 08-19 问「在线蒸馏确实有用，但为什么比"直接训练所有的场景"更好？」——**285 期结尾已埋引子，这条是下一期的正主。**

⚠️⚠️ **全部数字来自 WebFetch 的 HTML→markdown 转换，不是逐格直读。写作前每个要用的数字必须自己开 HTML 回核。** 三轮报告之间还互相矛盾过一次（MOPD 的 0.882/0.937，一轮说核实准确、一轮说未能核实）——**以"未核实"为准。**

### 12.1 ⭐ 最要命的一条：这个前提本身就不稳

**arXiv:2602.12566《To Mix or To Merge》Table 3 其实是平手，不是分头练获胜。**

Qwen3-4B-Base + SFT + GRPO，五个领域（math/coding/science/IF/agent）：

| 基准 | 蒸馏合并(Ties) | 混合训练 |
|---|---|---|
| AIME'24 | 81.15 | 81.20 |
| AIME'25 | **74.74** | 73.39 |
| GPQA-D | **57.58** | 53.62 |
| IFBench | 54.76 | **61.22** |
| LCB v5 | 60.84 | **63.21** |
| BFCL v3 | **61.73** | 60.74 |

差距双向都在 1–4 分。**论文自己的判定是算力：混合训练只用 63.7% 的 GPU 小时达到可比性能。**

**它的机制分析也反直觉**：更新落在**高度共享**的子空间（Jaccard 0.45–0.48 vs 随机基线 0.18），余弦相似度**为正**。所以低干扰**不是因为正交**，是本来就朝一个方向使劲。真正的解释在 §3.4 policy-neighborhood/KL。

⚠️ 两处流传的错误说法要避开：①"RL 只做小幅参数改动"——**原文没这句**，那个"约 30%"是被**触碰**的权重比例；②例外域是 **agent 不是 instruction-following**，而且 agent 是"迁移死路"（谁都帮不了它），**本身不是干扰源**；agent 只占数据 2.37%，这个 confound 论文没处理。

### 12.2 ⭐⭐ 真正的分界线：奖励类型（跨论文推断，非任何单篇原话）

| 论文 | 领域 | 奖励类型 | 结论 |
|---|---|---|---|
| To Mix or To Merge (2602.12566) | math/code/science/IF/agent | **全可验证**（QwQ evaluator、SandboxFusion 单测、IFEvalG、NeMo-Gym），训练回路里**无 reward model 无 LLM judge** | 几乎无干扰，甚至协同 |
| Modular Gradient Surgery (2602.02301) | math/**chat**/IF | chat 用**打分模型** | 显著跨域干扰 |
| BAR (2604.18473, Ai2) | math/code/chat/**safety**/tool | safety 和 tool **"lack verifiable reward mechanisms"只能 SFT** | safety 被后跑的 RLVR 冲掉 |

**BAR 那句机制话是最好用的**：
> "safety lacks verifiable RL data, so when math and code RL run later, safety capabilities degrade via catastrophic forgetting."

**这不是抽象的梯度打架，是训练顺序导致的遗忘**——朴素得多，也好讲得多。

⚠️ **没有找到一篇专门研究"可验证奖励 vs 模型打分奖励混训"的论文**。Omni-Thinker（arXiv:2507.14783，标题含 Hybrid Reward and Task Scheduling）最贴切但**检索超时未核**——**这是明确缺口，下期动笔前该补。**

### 12.3 ⭐⭐ 最诚实的一篇：BAR 明说主要是工程理由，而且效果上输了

**arXiv:2604.18473（Ai2）§3.1 逐字**：
> "This formulation enables modular upgrades (replacing a single expert without retraining others), avoids catastrophic forgetting (each domain's pipeline is isolated), and **supports parallel development across teams**."

**而 Table 1（7B）它自己是输的**：

| 方案 | Overall | Math | Code | Safety |
|---|---|---|---|---|
| BAR 5×7B | 49.1 | 56.2 | 49.9 | **94.0** |
| **Re-train (mid+post)** | **50.5** | 55.9 | **59.6** | 90.4 |
| Re-train (post only) | 47.8 | 48.7 | 43.6 | 92.8 |
| Merging (w/ mid) | **6.5** | 0.3 | 0.4 | 9.1 ←权重合并灾难性崩塌 |

**BAR 输给最强重训基线 1.4 分，代码差 9.7 分。** 摘要的 "matching or exceeding" 只对**较弱**的重训变体成立，论文自己用词是 "approaches"。

**Discussion 那句才是题眼**：
> "Re-training with mid-training achieves the highest score (50.5), but **requires complete access to the original pre-training checkpoint**."

→ **分头练赢的不是效果，是"你不需要拿到原始预训练 checkpoint"——组织/资产可得性约束，不是算法优越性。**

⚠️ 成本论证是**纯渐近推理，全文无实测 GPU 小时**；且口径不一致：正文说"线性 vs 二次"，Figure 2 caption 说"线性 vs 常数"——**引用时挑一个别混用**。

### 12.4 机制：学界在打架，而且"梯度冲突"这个流行解释被证伪

- **arXiv:2606.02398（Local Perturbation Theory）§1 逐字**：`substantial interference can occur even when full-model gradients are nearly orthogonal.` —— **梯度正交时干扰照样发生**。它同时说 catastrophic forgetting 和 global gradient conflict 两种解释都不完整
- **arXiv:2608.03573（SFT Conflicts, RL Coexists）§4/Lemma 4.3/Theorem 4.5**：论证 RL 比 SFT 更能共存，机制是 **advantage centering 抵消了共享的均值梯度方向**（⚠️ 不是"已掌握任务梯度趋零"，那是想当然的猜法）
- **arXiv:2602.05547（Multi-Task GRPO）§2**：问题是**各任务零梯度 prompt 比例不均**（组内全对/全错则 advantage 归零），有效数据配比被悄悄扭曲。⚠️ **不是**奖励尺度问题。**这条特别适合写给程序员——很具体、很 GRPO 实现层**
- **arXiv:2602.02301（MGS）**：梯度冲突阵营的代表，模块级化解冲突，Llama/Qwen 上 +4.3/+4.5 分。**和上面两篇对立**

**⚠️ "熵需求不同（推理要收敛、创作要发散）"这条我没找到任何实证支撑**——写的话必须标明是推测。

### 12.5 两篇工业界论文完全没给机制

- **MOPD (2606.30406)**：§5 Discussion 纯经验观察，**没解释为什么混合更差**
- **CARE-RL (2606.00609)**：§4.3 纯经验，**无梯度/熵/超参测量**，也没用 "see-saw" 这个词
  - ⚠️ 网上流传的引文是**拼接过的**。§4.2 原文：`Compared with the mixed-domain training of MGS, the cascade structure of CARE-RL allows per-domain hyperparameter tuning, and PA-GRM provides more reliable rewards.` —— **奖励可靠性归功于 PA-GRM 这个组件，不是 cascade 结构**，别合并这两点

### 12.6 文献真空：配比消融基本没人做

To Mix or To Merge 无配比消融；CARE-RL 固定每域 8000 条（明说是为防止规模主导）。**"配比调好能不能补上差距"这个问题现有文献没回答。**

### 12.7 ⭐ 下一期的题眼（三选一或合并）

1. **「各家都在分头练，但学术界既没证明它更好，也没给出机制解释——最诚实的那篇明说：主要是工程理由，效果上还输给完整重训 1.4 分。」**
2. **「混合训练差不差，取决于你混的是不是同一类奖励」**（跨论文推断，需标明）
3. **「'梯度冲突'这个人人都在用的解释，被一篇论文用一句话推翻了：梯度正交时干扰照样发生」**

**我倾向 1**：它最反直觉，而且有 BAR 的自白 + 自家表格双重支撑。

### 12.8 动笔前必做的功课
- **回核 MOPD 0.882/0.937**（三轮报告自相矛盾）
- **回核 BAR Table 1**（载重数字）
- **回核 To Mix or To Merge Table 3/Table 5**
- **补查 Omni-Thinker (2507.14783)** —— 混合奖励类型的最直接证据，这轮超时没拿到

### 12.9 ⚠️⚠️ 补核（08-19 末轮，子代理亲自抓表）：MOPD 那 5.5 分是归一化放大的

**0.882 / 0.937 确认无误**（Table 2，Qwen3-30B-A3B）。完整排序：MOPD 0.9373 > Mix-RL 0.8818 > Param-Merge(Task Arith.) 0.8574 > Off-Policy Finetune 0.8241 > Cascade RL 0.7752 > **Param-Merge(Avg.) 0.3280**。

**但逐项拆开看，五个基准里 Mix-RL 赢了两个：**

| 基准 | Mix-RL | MOPD |
|---|---|---|
| AIME25 | **52.71** | 51.46 ← Mix-RL 赢 |
| AIME26 | 63.75 | **65.31** |
| IFBench | 75.00 | **77.89** |
| IFEval | **94.58** | 93.84 ← Mix-RL 赢 |
| SWE-bench Verified | 48.80 | **50.40** |

**⚠️ 那 5.5 分的差距主要由归一化方式放大**（相对各领域专家上限归一）。**写下一期时千万别把它说成"混合训练明显更差"——逐项看基本是伯仲之间。**

**MOPD 确实用了 see-saw 这个词**：`cross-domain training signals interfere, producing the see-saw effect`。但**正文没有对"Mix-RL 为什么差"的任何机制分析**（无梯度实验、无熵分析、无配比消融）——这是核实过的"不存在"。

**MOPD 也给了工程理由（原文）**：
> "each domain is free to choose its own RL recipe (algorithm, rollout procedure, reward function, hyperparameters, and so on) without worrying about conflicts."

**另一条更硬的干扰实测**（2606.02398，比 MOPD 的证据强）：顺序训练 Code→Math→QA→CW，**Math 从 66.49 掉到 57.66（约 −8.8 分）**，是选择性退化的实测。

**⚠️ 奖励类型那条线没跑通**：Omni-Thinker (2507.14783) 超时未核。已知的只有 2606.02398 混用了两类（Math/Code/QA 规则奖励 + Creative Writing 用 LLM-as-judge 给 0/0.5/1）。**下期别断言"奖励类型是主因"，目前没有直接证据。**
