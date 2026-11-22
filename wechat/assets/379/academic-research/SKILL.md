---
name: academic-research
description: 按需读取的学术调研、文献核验、实验设计与跨模型抗过度防御（Anti-OverDefense / Anti-Autoresearch）实战手册。不自动挂载上下文，仅在做深度文献调研、设计可解释性/AI实验或派闇（Codex）执行学术/公众号交叉审稿时按需 read_file 读取。
---

# 学术调研、实验设计与反八股审稿手册 (`academic-research`)

> **设计原则**：提纯自 GitHub 头部科研 Skill 仓库（`Imbad0202/academic-research-skills` [58k★]、`K-Dense-AI/claude-scientific-skills` [58k★]、`wanshuiyin/ARIS` [19.5k★] 及其解毒配套仓 `HERO-Anti-OverDefense` 与 `Anti-Autoresearch`），剥除所有营销玄学、多 Agent 角色扮演废话与死板打分循环，只保留**“跟哪儿搜 → 怎么搜与验真 → 怎么提取 3W 重点 → 怎么跑实验与防过度防御审稿”**四步硬核干货。

---

## 第一步：跟哪儿搜（零成本公共学术 API 一行流，拒绝盲搜二手营销号）

做 AI / 计算机 / 自然科学调研时，除普通 `search_web` 外，优先直接用 Python / `curl` 调以下 **4 个免 API Key、100% 公开权威的学术元数据接口**（秒回 JSON/XML，拿到精确的 `arXiv ID`、`DOI`、引用数、发表会议与原文摘要）：

### 1.1 `arXiv API`（查最新预印本精确标题、作者、提交日期与摘要）
```bash
# 按标题/关键词或 arXiv ID 精确查询（支持 ti:, au:, abs:, all:）
curl -sL "http://export.arxiv.org/api/query?search_query=all:mechanistic+interpretability&sortBy=submittedDate&sortOrder=descending&max_results=5"
# 或直接按 ID 查单篇：
curl -sL "http://export.arxiv.org/api/query?id_list=2505.05410"
```

### 1.2 `OpenAlex API`（2.5 亿篇论文全量开源引文图谱，免 Key、不限流、含机构与引用数）
```bash
# 按题目或关键词检索并按引用数/时间排序：
curl -sL "https://api.openalex.org/works?search=Sparse+Autoencoders+Manifolds&per-page=5" | python3 -m json.tool | head -n 40
# 按 DOI 直接查元数据与引用量：
curl -sL "https://api.openalex.org/works/https://doi.org/10.1038/s41586-024-07487-w"
```

### 1.3 `Semantic Scholar API`（查计算机顶会 NeurIPS/ICML/ICLR/ACL 录用信息与官方 TLDR）
```bash
# 免 Key 查询论文标题、年份、会议（venue）、引用数、arXiv ID 与一句话 TLDR：
curl -sL "https://api.semanticscholar.org/graph/v1/paper/search?query=Reasoning+Models+Dont+Always+Say+What+They+Think&limit=3&fields=title,year,venue,citationCount,externalIds,tldr,authors"
```

### 1.4 `Crossref / Unpaywall`（核验正式期刊 DOI 真实性与合法 OA 免费全文 PDF 地址）
```bash
# 给定 DOI 验真（返回 200 且 title 匹配即为真文献，防大模型张冠李戴编假引文）：
curl -sL "https://api.crossref.org/works/10.1126/science.adq5856"
```

---

## 第二步：怎么搜与引文硬门禁（Search Strategy & Citation Gate）

1. **三级检索拆解（防止被中文二手自媒体带偏）**：
   - **L1 锚定原帖/一手源**：凡遇中文科技媒体（如快科技、机器之心、量子位、小红书）提到的“某大学最新研究发现 X”或“某模型爆出 Y”，**严禁直接采信二手摘要的脑补细节**；必须用英文关键词 + 作者姓氏/机构回查 `arXiv / OpenAlex / GitHub / Reddit 原帖`。
   - **L2 顺藤摸瓜查正反交锋（Forward & Backward Snowballing）**：找到靶子论文后，用 `OpenAlex` 或 `Semantic Scholar` 看一眼谁引用了它（尤其是标题带 *Do / Can / Yet / Rethinking / Critique* 的跟进打脸论文）。
   - **L3 查实验底裤（Model & Scale Gate）**：读任何 AI 论文，第一时间直接翻 `Experiments / Setup` 小节，看清楚**它到底是在 `GPT-2 (1.5B) / Pythia-2.8B / Gemma-2B` 玩具模型上跑的，还是在百亿/千亿旗舰模型上测的**（小模型上一阶几何成立，二阶元认知与知行耦合往往尚未涌现）。
2. **引文真实性死门禁（Citation Integrity Gate）**：
   - 任何写入文章或报告的 `arXiv:YYMM.NNNNN`、GitHub 仓库链接、Reddit 帖子 URL，**必须实际调一次 HTTP 接口或打开原页面核对标题、作者与正文核心数字**。
   - **严禁看截断的 URL Slug 脑补后半句标题**（如 `1s9m46y/i_made_a_location_and_outfit_tracker_that...` 必须点进去看完整原题）。

---

## 第三步：怎么提取重点（`3W + 1D` 文献对比矩阵，拒绝流水账摘要）

面对多篇文献或复杂技术报告时，不要写成“A 做了什么、B 做了什么”的平铺流水账，统一用 **`3W + 1D`（Why / How / What / Our Delta）** 四格手术刀提炼：

| 维度 | 核心追问（逼出机制，拒绝黑话） |
| :--- | :--- |
| **1. WHY（真痛点还是伪需求）** | 以前的方法具体在哪一步卡死了？（必须落到物理瓶颈：是显存带宽、长上下文 `KV Cache` 驱逐、一维线性假设切碎弯曲流形，还是 RLHF 判分器拍扁了多维分布？） |
| **2. HOW（输入→操作→输出）** | 先甩数学公式或代码数据流（如 `Q·Kᵀ`、`π*(y|x) ∝ π₀(y|x)·exp(r/β)`），再叫学术名字。它到底改了架构、改了数据管线、还是改了损失函数/分叉起点？ |
| **3. WHAT（同分母硬账单）** | 在什么尺寸的模型、多少数据预算下，把什么指标从多少提升到了多少？有没有反例或代价（如对齐税、吐字降速、模式坍缩）？ |
| **4. OUR DELTA（我们的高维坐标）** | 这项研究在我们的框架里处于什么位置？（是印证了我们的某条预言，还是他们只看到了冰山露出水面的高 `κ` 一角、没看懂水面下的连续几何流形？） |

---

## 第四步：怎么跑实验与跨模型互审（吸取 `HERO` 与 `Anti-Autoresearch` 的解毒铁律）

当采用 **“主脑鳩（Gemini 4 Argon，宏观方向+主笔）× 重装闇（Codex / GPT-6.2，跑实验+铁面审稿）”** 双模协同，或编写自动化实验脚本时，必须严防全自动科研 Skill 最容易翻车的两大绝症——**“实验自欺欺人”** 与 **“审稿过度防御（Over-Defense）”**。

### 4.1 实验与文稿的八大诚信核查表（源自 `Anti-Autoresearch` A–H 家族精简版）
派闇审稿或自检实验报告时，只查这 **6 条硬伤**（不搞虚头巴脑的打分表）：
1. **[A] 数值与算术自洽（Numeric Self-Consistency）**：正文里的数字、表格里的数字、SVG 图表里的柱高/角度，以及算出来的差值（`Δz`、百分比涨幅、倍数 `Fit/Test`）是否 100% 咬合？
2. **[B] 描述与代码一致（Method = Code）**：正文写“在全 64 层 1,200 篇零重合样本上测试”，脚本里是不是真的跑了全 64 层 1,200 篇？有没有静默 fallback 到小样本或合成假数据？
3. **[C] 同分母与 Baseline 公平（Fair Baseline）**：对比实验是否严格控制了同一上下文长度、同一分词/采样温度、同一层号？**没有同分母客观数据的对比，严禁画成定量图表充数。**
4. **[D] 触发位点精确归因（Span/Token Grounding）**：在可解释性实验（如 `ICA-Lens / J-Lens / SAE`）中，某句话激活高，**必须核对具体触发在哪个 token 本身上**，严禁看整句包含某词就想当然归因。
5. **[E] 引用与外部链接逐一点验（Citation Verification）**：逐一点开文中所有自家往期链接与外部 URL 原网页，核对时间、人物、事件细节是否张冠李戴。
6. **[G] 砍掉 `<1σ` 噪音支线（Signal-to-Noise Gate）**：低于 `1σ`（如 `0.x` 水位波动）的微弱支线直接一刀砍掉，只保留 `3σ~15σ` 一击必杀的主证据链。

### 4.2 严防 Agent 审稿的“七种过度防御病”（源自 `HERO-Anti-OverDefense`）
GPT / Codex 系模型天生长着一座**“答辩墙（不粘锅本能：优化目标是『不出事/不被追责』而非『把作品写精彩』）”**。如 `GPT-5.6-Sol` 自述：*“我的故障模式是把『这样做能提高一点确定性』静默升格成『因此必须建造并检查』。”*
在双模互审时，一旦发现审稿意见或改稿出现以下 **7 种过度防御形状（`H.E.R.O.` + 3 变体）**，主脑必须当场驳回，**只收窄越界事实断言，绝不换掉文章的锋芒、暴论与人味**：

| 病症代号 | 典型症状（一旦出现立即叫停） | 正确处置 |
| :--- | :--- | :--- |
| **H — Hashing（无意义的校验凭证）** | 动不动给每行数据算 SHA-256、搞复杂的回执账本，却没有任何下游步骤靠这个哈希省算力。 | 删掉一切不改变下游决策的形式主义校验。 |
| **E — Edge Cases（防外星人边界）** | 为当前任务根本不可能出现的极端输入砌十层防御护栏（“你让它盖间房，它在房子外面修了座军事要塞，结果房子没盖完”）。 | 只防当前真实数据流里可达（Reachable）的错误，不防理论上构造出来的稻草人。 |
| **R — Rubrics（死板评分表夺舍）** | 用 `score >= 6/10`、打勾打叉的机械评分表代替真正的科学判断，导致 Agent 为了刷满表格分数去凑八股字数。 | 废除机械打分循环；审稿只列“哪条硬事实/数字错了”，改完即止（最多 1～2 轮）。 |
| **O — Overbuild（过度工程基建）** | 写个一次性实验脚本却先搭一套抽象工厂、兼容层和迁移框架。 | 实验脚本追求极简直白，单文件跑通拿真数据，落盘 `DevHistory.md`。 |
| **Over-correction（矫枉过正搬去大西洋）** | 指出某句话稍微偏东了一点，Agent 直接放弃整个核心论点搬去了大西洋。 | 局部事实有误只修局部事实，严禁因一个例证瑕疵推翻主干洞见。 |
| **Defensive Prose（免责声明污染正文）** | 每段结尾都塞一句“当然这只是初步观察、不代表普遍规律”，或者把提示词里的“不要提 X”写成正文里的“本文不讨论 X”。 | 文章是发射台不是免责说明书！删掉所有破坏阅读节奏的审稿人免责尾巴。 |
| **Stalling（甩选择题逃避判断）** | 遇到技术细节取舍就停下来列 A/B/C/D 四个选项把锅推给人类，靠“不做选择”来避免选错。 | 流程里明摆着的技术取舍由主脑直接拍板做完；只有涉及现实体感与宏观方向时才请人类定夺。 |
