# 后训练提分专题（2026-08-16 立项，计划 2~3 期）

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

## 四、考卷也换了（可单独成节或并入上节）

**这六家没有一家报 SWE-bench Verified，也没有一家报数学基准（AIME/FrontierMath 全部缺失）。**

统一换成：**DeepSWE v1.1 / Terminal-Bench 2.1 & 3.0 / FrontierCode / APEX / GDPVal-AA / AutomationBench**。

⚠️ **Terminal-Bench 版本不可跨比**：DeepSeek 与 Qwen 报 **2.1**（87.9 / 86.6），GLM-5.3 报 **3.0**（28.3）。3.0 难得多——Grok 4.6 在 3.0 上仅 26%，Fable 5 Max 34.1%，Gemini 3.7 Flash 14.9%。

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

**可用的反差**：唯一动了基座重新预训练的 Qwen3.8-Max（58），**没打过在旧基座上加后训练的 Grok 4.6（61）**。花大钱重训 2.4T 的，输给了只做后训练的。

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

## 六、后续期的弹药（第一期不讲，留着逐个拆）

### 6.1 算法：GRPO 崩在长轨迹上 → SAO
**SAO: Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning** — arXiv:2607.07508
> "we replace group-wise sampling with **single-rollout sampling**, that is, using one rollout per prompt… we introduce a **strict double-side token-level clipping strategy**. SAO is able to train stably for one thousand steps and consistently outperform GRPO and its variants…"

GLM-5.3 用的是 "**SAO with compaction**"（compaction 处理长轨迹上下文，未展开定义）。

**要讲清楚的故障机制**：GRPO / REINFORCE 这类 episode-level 方法**给轨迹里每个 token 相同的 advantage**——短轨迹是可接受的近似，但 agent 交互动辄 **10–100+ 轮、100K–500K+ tokens**，近似就崩了。这是「为什么单纯搬 GRPO 到 agent 上不 work」的一句话答案。

**量级对照（很重要的一条）**：学术上做严格控制变量只换 RL 算法 = **+2~5pp**（arXiv:2603.19335 甚至说任务离训练分布越远、算法选择越不重要，通用推理 benchmark 上没有任何方法偏离 base model 均值超过 0.29pp）；而换整套后训练 = DeepSWE **7.3 → 54.4**。
→ **收益绝大部分不来自 RL 算法本身，来自环境规模、任务质量、verifier 质量和训练基建。**

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
