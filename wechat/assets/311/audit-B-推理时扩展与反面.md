# 311 期文献核查 B：推理时扩展与反面

核查时间：2026-09-16。方法：arXiv abs 页 + HTML 全文（`arxiv.org/html/...`）一手核对，训练记忆里的细节一律不采信。

**五篇全部存在，编号全部正确，无「查无此文」。但有 4 处需要改口，见每篇的「⚠️ 修正」。**

---

## 1. arXiv:2504.02495 — Inference-Time Scaling for Generalist Reward Modeling

| 项 | 核实结果 |
|---|---|
| 标题 | Inference-Time Scaling for Generalist Reward Modeling |
| 作者 | Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu |
| 机构 | DeepSeek-AI + 清华大学（计算机系 / 智能产业研究院 AIR） |
| 编号/日期 | arXiv:2504.02495，v1 2025-04-03，v2 2025-04-05，**v3 2025-09-25（现行版）** |
| 会议状态 | arXiv 预印本，comments 未标注会议 |

### 指定数字：全部属实 ✅

论文的「Overall」= RewardBench / PPE Preference / PPE Correctness / RMB / ReaLMistake **五个基准的综合分**，不是 RewardBench 单项。

| 模型 | Overall |
|---|---|
| DeepSeek-GRM-27B（greedy 贪心解码） | 69.9 |
| **DeepSeek-GRM-27B（MetaRM 引导，Voting@32）** | **72.8** |
| Nemotron-4-340B-Reward | 70.5 |
| GPT-4o | 71.3 |

> ⚠️ **修正 1（重要，直接关系到主线成色）**：72.8 **不是**「RewardBench 上是 72.8」，而是五基准综合分。同一行在 RewardBench 单项上是 90.4，贪心时是 86.0。**写的时候必须说「综合分」，说成 RewardBench 就是硬错。**
>
> ⚠️ **修正 2（对主线最关键的一条）**：72.8 这个数**需要 meta-RM 引导的投票**，不是朴素多数投票。也就是说「小裁判多想几遍」在这篇里**并不是白捡的**——它额外挂了一个 meta 奖励模型来筛选那 32 个采样。朴素投票（direct voting）的对标对象是另一句话（见下）。
>
> ⚠️ **修正 3**：Nemotron 的 70.5 和 GPT-4o 的 71.3 都是**贪心解码**成绩，没有给它们同等的 32 次采样预算。所以这是「小模型花推理算力 vs 大模型不花」的对比，**不是等算力对比**。这一点是本期最容易说过头的地方。

### 原话 1：32 次采样 ≈ 671B

> **英文**："Direct voting with 32 samples of DeepSeek-GRM-27B could achieve comparable performance to the 671B MoE model."
>
> **中文**：「DeepSeek-GRM-27B 用 32 个样本做直接投票，就能达到与 671B MoE 模型相当的性能。」

注意：这句用的是 **direct voting（朴素投票）**，对标的是 **671B MoE**——这句是干净的，可以直接引。上面 72.8 那个数用的才是 meta-RM 版本。两者别混。

### 原话 2：推理时扩展 > 训练时扩展

> **英文**："Inference-time scaling could outperform model size scaling in training time."
>
> **中文**：「推理时扩展可以胜过训练时的模型规模扩展。」

另有一句解释机制：

> **英文**："With larger-scale sampling, DeepSeek-GRM could judge more accurately based on more diverse principles, and output rewards with finer granularity."
>
> **中文**：「在更大规模的采样下，DeepSeek-GRM 能基于更多样的原则做出更准确的判断，并输出粒度更细的奖励。」

这句对主线有用：多想几轮之所以有效，机制是**采出更多样的评判原则**，而不是简单地把同一个判断重复做几遍再取平均。

---

## 2. arXiv:2604.16004 — AgentV-RL: Scaling Reward Modeling with Agentic Verifier

| 项 | 核实结果 |
|---|---|
| 标题 | AgentV-RL: Scaling Reward Modeling with Agentic Verifier |
| 作者 | Jiazheng Zhang, Ziche Fu, Zhiheng Xi, Wenqing Jing, Mingxu Chai, Wei He, Guoqiang Zhang, Chenghao Fan, Chenxin An, Wenxiang Chen, Zhicheng Liu, Haojie Pan, Dingwei Zhu, Tao Gui, Qi Zhang, Xuanjing Huang |
| 机构 | 复旦大学计算机学院（主）+ 华中科技大学 + 香港大学 + 字节跳动 |
| 编号/日期 | arXiv:2604.16004，v1 2026-04-17 |
| 会议状态 | **comments 字段只写「ACL 2026」，没写 Findings** |

> ⚠️ **修正 4**：**不能说它是 ACL 2026 Findings。**arXiv comments 原文就三个字「ACL 2026」，未区分主会还是 Findings。稳妥写法：「已被 ACL 2026 接收」。

### 25.2% 的真实所指：绝对百分点，但基准和对照都要说清 ⚠️

> **英文**："Notably, Agentic-Verifier-Qwen3-4B achieves the highest accuracy on MATH500 (up to 79.0%), surpassing the previous best outcome-level RM, Skywork-V2-Llama-8B, by a substantial margin of 25.2 percentage points."
>
> **中文**：「值得注意的是，Agentic-Verifier-Qwen3-4B 在 MATH500 上取得了最高准确率（最高 79.0%），比此前最好的结果级奖励模型 Skywork-V2-Llama-8B 高出 25.2 个百分点。」

拆开说：

- **超过的是什么**：MATH500 这**单个基准**上的准确率，不是整体平均。
- **对照是谁**：Skywork-V2-Llama-8B，一个**结果级（outcome-level）奖励模型**。79.0% vs 约 54.0%（表 1）。
- **25.2 是绝对还是相对**：**绝对百分点**（percentage points），原文写死了。79.0 − 54.0 ≈ 25.0，算术对得上。

所以「4B 变体超过当时最好结果 25.2%」这句话**三处都需要收窄**：不是「当时最好结果」而是「当时最好的结果级奖励模型」；不是全面超过而是 MATH500 单项；单位是百分点不是相对提升。另有一句可引的对照：

> **英文**："our 4B variant consistently outperforms INF-ORM-Llama3.1-70B, an outcome reward model with ten times more parameters"
>
> **中文**：「我们的 4B 变体持续胜过 INF-ORM-Llama3.1-70B ——一个参数量多十倍的结果奖励模型。」

**这句才是主线最想要的那句**（小裁判打赢大十倍的裁判），而且是 "consistently"（持续地），比 25.2 那个单点数据更稳。建议主用这句。

### 正反向双 agent：描述基本准确 ✅

「一个 agent 从前提往结论推、另一个从结论倒推」——**方向说对了**，但原文的分工比这更具体：

> **前向 agent 英文**："Starting from the problem premises, forward agent sequentially traces the solution path to review whether each step in the solution is correct, and validate whether the preceding step constitutes a sufficient condition for the subsequent derivation."
>
> **中文**：「从题目前提出发，前向 agent 顺序追踪解题路径，检查解答中每一步是否正确，并验证前一步是否构成后一步推导的**充分条件**。」

> **后向 agent 英文**："The backward agent is designed to identify errors the forward agent may overlook... It verifies the necessity of a solution by reasoning in reverse, from the final answer back to the problem statement."
>
> **中文**：「后向 agent 用于发现前向 agent 可能遗漏的错误……它通过从最终答案反推回题目陈述，来验证解答的**必要性**。」

可加一层：正向查**充分性**、反向查**必要性**——这是数学味很正的分工，比「一个正推一个倒推」信息量大。另外框架被描述为 "transforms reward modeling into a multi-turn, tool-augmented deliberative process"（把奖励建模变成多轮、带工具的**审议**过程），且 verifier "interleaves tool-use with internal reasoning"（把工具调用与内部推理交替进行）——**注意它不是纯「多想几轮」，是「多想几轮 + 调工具」**，这对主线是个重要限定。

---

## 3. arXiv:2608.30005 — Small Language Models as Judges for Rubric-Based Reinforcement Learning

| 项 | 核实结果 |
|---|---|
| 标题 | Small Language Models as Judges for Rubric-Based Reinforcement Learning |
| 作者 | Fengyu Xie, Yilun Zhao, Bingsen Chen, Arman Cohan, Chen Zhao |
| 机构 | 纽约大学（Xie / Chen / Zhao）+ 耶鲁大学（Zhao / Cohan） |
| 编号/日期 | arXiv:2608.30005，v1 2026-08-30 |
| 会议状态 | **EMNLP 2026 Findings**（comments：9 pages, 1 figure; EMNLP 2026 Findings）✅ |

### 原话：信息在隐状态里，只是没被读出来 ✅

> **英文**："useful rubric-satisfaction information is present in the model's hidden states but is not recovered reliably by Generative or Logprob scoring"
>
> **中文**：「有用的**评分项满足度**信息**确实存在于模型的隐状态之中**，只是生成式打分和 logprob 打分没能可靠地把它读出来。」

这句是本期的宝。它说的不是「小模型不够聪明」，而是**小模型知道，只是嘴说不出来**——和「判分能力被输出层卡住」完全同构。

### 指定数字：全部属实 ✅

| 数字 | 核实 |
|---|---|
| 1.7B 探针 macro-F1 = 0.835 | ✅ 属实，具体是在 **RaR-Science-Static** 上取得 0.835 macro-F1，且是三种方法里 criterion-level 一致性最强的 |
| 作奖励模型时策略得分 0.643 | ✅ 属实，原话："the Qwen3-1.7B Probe reward improves the final RaR-Science rubric score from **0.232 to 0.643**"（注意有个起点 0.232，写进去更有说服力） |
| 8B 生成式裁判 0.594 | ✅ 属实，原话："the larger Qwen3-8B Generative reward reaches 0.594" |
| 判分快 10.7 倍 | ✅ 属实，原话："Generative requires **89,912.1 seconds** of cumulative judge time versus **8,389.9 seconds** for Probe, a **10.7× ratio**" |
| 模型家族 Qwen3 0.6B~8B | ✅ 属实，四档：**0.6B / 1.7B / 4B / 8B** |

一句话概括本篇给主线的弹药：**1.7B 的探针裁判（0.643）训出来的策略，打赢了 8B 的生成式裁判（0.594），还快 10.7 倍。**

> ⚠️ 但这篇**恰恰不是「多想几轮」**。它的做法是**换读取方式**（用探针直接读隐状态），不是让小模型多推理几轮——它反而**省掉了**生成过程。放进「大从空间挪到时间」的主线时要小心：**这篇支持的是「小裁判够用」，不支持「靠多想几轮」**，它是另一条路径（挪到「读取方式」上，不是挪到时间上）。硬塞进时间轴会是本期最隐蔽的一处说过头。

---

## 4. arXiv:2509.17995 — Variation in Verification: Understanding Verification Dynamics in Large Language Models

| 项 | 核实结果 |
|---|---|
| 标题 | Variation in Verification: Understanding Verification Dynamics in Large Language Models |
| 作者 | Yefan Zhou, Austin Xu, Yilun Zhou, Janvijay Singh, Jiang Gui, Shafiq Joty |
| 机构 | Salesforce AI Research + 达特茅斯学院 + UIUC |
| 编号/日期 | **arXiv:2509.17995，v1 2025-09-22，v2 2026-04-14** |
| 会议状态 | **ICLR 2026**（comments 字段写明）✅ |

> **关于年份的确认**：编号 2509 **确实是 2025 年 9 月**投的（arXiv 编号规则 YYMM = 2025 年 09 月），标题也对得上。v2 修订于 2026-04-14。所以它是一篇**2025 年 9 月首发、2026 年 4 月修订、中了 ICLR 2026** 的论文。**写的时候别说成「2026 年的新论文」，它比 AgentV-RL 早半年多。**

### 两句原话 ✅

> **英文**："Weak generators produce errors that are easier to detect than strong generators."
>
> **中文**：「弱生成器产生的错误，比强生成器产生的错误更容易被检测出来。」

> **英文**："Verifier scaling alone cannot overcome fundamental verification challenges."
>
> **中文**：「仅仅扩大验证器的规模，无法克服根本性的验证难题。」

**第二句是本期最重要的一句反面证据**，可以直接当「反面」那一节的题眼。

### 实验规模：属实 ✅

- **12 个基准**：8 个数学推理 + MMLU-Pro（知识）+ 3 个自然语言推理数据集，跨数学/知识/语言三个域。
- **模型**：**14 个开源模型 + GPT-4o**（注意不是「14 个模型」，是 14 开源 + GPT-4o）。
- **规模范围**：**2B ~ 72B** ✅。

### 「三者共同决定」这个说法：⚠️ 需要改写

论文里**没有**一句话直接说「验证效果由题目难度、生成器强度、验证器强度三者共同决定」。原文是**分成三条并列的发现**：

> **英文**："Problem difficulty primarily governs the recognition of correct solutions; Generator capability influences error detection; Verifier generation capability correlates with verification performance in a manner dependent on problem difficulty."
>
> **中文**：「**题目难度**主要支配着『能否认出正确解答』；**生成器能力**影响『能否查出错误』；**验证器自身的生成能力**与验证表现相关，但这种相关性**取决于题目难度**。」

所以准确的说法是：**三个因素各管一段、且互相纠缠**（验证器的作用还要看题目难度），而不是一个笼统的「三者共同决定」。按原文这样拆开写，反而更有料——**认对**和**查错**是两件由不同因素支配的事。

另有一个具体数字可用：

> **英文**："the Gemma2-9B to Gemma2-27B performance gap shrinks by 75.7%"
>
> **中文**：「（经过验证之后）Gemma2-9B 与 Gemma2-27B 之间的性能差距缩小了 75.7%。」

**这个数对主线是正面弹药**：验证这一步能把小模型和大模型的差距抹掉四分之三。

---

## 5. arXiv:2403.02839 — An Empirical Study of LLM-as-a-Judge

| 项 | 核实结果 |
|---|---|
| 标题 | An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4 |
| 作者 | Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang, Bing Xu, Tiejun Zhao |
| 机构 | 哈尔滨工业大学 + 百度等（主体为哈工大团队） |
| 编号/日期 | arXiv:2403.02839，v1 2024-03-05，v2 2024-06-17，v3 2024-11-05，**v4 2025-05-30（现行版）** |
| 会议状态 | **ACL 2025 Findings**（Accepted to Findings of ACL 2025）✅ |

### 抗偏见基准的名字：**LLMBar** ✅

由 Natural（自然）+ 四个对抗子集（Neighbor / GPTInst / GPTOut / Manual）构成。

### 指定数字：基本属实，但结构要说清 ⚠️

| 数字 | 核实 |
|---|---|
| 微调裁判域内约 82% | ✅ 属实。**JudgeLM 在自己的域内测试集上 82.39**。注意这是「各模型在各自的域内测试集」，不是统一基准。 |
| 抗偏见基准掉到 16~23% | ✅ 属实，但要说清是**哪几个子集**。对抗子集上的成绩（见下表）确实密集落在 16.5 ~ 32.6，其中 16~23 这一档覆盖了多数格子。 |
| 同题 GPT-4 是 93.5% | ✅ 属实，**GPT-4-1106 = 93.5**，位于 LLMBar 的 **Natural（自然）子集**列。 |

Table 4 实测（单位 %）：

| 模型 | LLMBar/Natural | Neighbor | GPTInst | GPTOut |
|---|---|---|---|---|
| JudgeLM-7B | 62.0 | 23.1 | 26.1 | 46.8 |
| PandaLM-7B | 59.0 | 16.5 | 21.7 | 42.6 |
| Auto-J-13B | 70.0 | 20.9 | 21.7 | 46.8 |
| Prometheus-7B | 53.0 | 22.4 | 17.4 | 27.7 |
| **GPT-4-1106** | **93.5** | 64.2 | 76.6 | 76.6 |

> ⚠️ 写的时候注意：「掉到 16~23%」指的是**对抗子集**（Neighbor 等），不是「在抗偏见基准上整体只有 16~23」——它们在 Natural 子集上还有 53~70。**准确说法：微调裁判在对抗子集上掉到 16~23%，比随机猜（50%）还差得多；而 GPT-4 同题仍有 64~77，自然子集 93.5。**
>
> 「比随机猜还差」是原文明说的：**"the fine-tuned judge models perform poorly on adversarial testsets, even worse than random-guess"**（微调裁判在对抗测试集上表现很差，**甚至不如随机猜测**）。低于 50% 意味着它们**系统性地偏向错误答案**——不是不会判，是被表面特征牵着走反着判。

机制原话：

> **英文**：fine-tuned judges "are severely biased toward superficial quality such as formality or verbosity, while neglecting crucial properties such as instruction following"
>
> **中文**：微调裁判「**严重偏向于表面质量**，例如**格式规整度或啰嗦程度**，却忽略了**是否遵循指令**这类关键属性。」

还有一句总纲，适合当全期的落点：

> **英文**："the fine-tuned judge model inherently operates as a task-specific classifier, consequently imposing the limitations."
>
> **中文**：「微调后的裁判模型**本质上是在充当一个任务专用的分类器**，其局限性由此而来。」

---

## 综合判断：「小裁判多想几遍就够用」能说到什么程度

### 能说到这个程度（有硬证据）

1. **在可核验、题目分布已知的任务上，小裁判 + 推理时算力，确实能顶掉大裁判。** 三篇独立证据：27B 投票 32 次 ≈ 671B MoE（2504.02495）；4B 持续打赢参数多十倍的 70B 结果奖励模型（2604.16004）；1.7B 探针裁判训出的策略（0.643）打赢 8B 生成式裁判（0.594）且快 10.7 倍（2608.30005）。**三篇的模型家族、任务、方法都不同，结论方向一致——这个交叉验证是真的。**

2. **「大」确实可以从参数挪到别处。** 而且论文原话就是这么说的："Inference-time scaling could outperform model size scaling in training time."

3. **验证这一步能显著抹平模型规模差距**——Gemma2-9B 与 27B 的差距经验证后缩小 75.7%（2509.17995）。这是「换算率」的直接测量值，很硬。

4. **小模型不是「不知道」，是「说不出来」。** 2608.30005 的隐状态那句是全套材料里最深的一刀：信息在隐状态里，生成式和 logprob 读不可靠。**判分能力被输出层卡住了，不是模型里没有。**

### 这里会说过头（四个坑，按危险程度排）

1. **「多想几轮」在三篇里其实是三种不同的东西，不能合并同类项。**
   - 2504.02495 = **并行采样后投票**（而且拿满 72.8 还需要一个额外的 meta 奖励模型来筛）；
   - 2604.16004 = **多轮审议 + 调外部工具**（"interleaves tool-use with internal reasoning"）——**它赢在能调工具，不只是赢在想得久**；
   - 2608.30005 = **根本没多想**，它用探针直接读隐状态，把生成过程**省掉了**。
   
   把这三个都说成「让小裁判多想几轮」，是本期最容易犯、也最不容易被读者发现的错。**尤其第三篇，它其实是把「大」挪到了「读取方式」上，不是挪到时间上。** 诚实的写法是：**「大」有好几个可挪的去处——采样次数、审议轮数+工具、读取方式——时间只是其中一条。**

2. **对照组没花同等算力。** 2504.02495 里 Nemotron-4-340B（70.5）和 GPT-4o（71.3）都是**贪心解码一次**，而 27B 那 72.8 是采样 32 次还外挂 meta-RM。这是「小模型花钱 vs 大模型不花钱」，**不是等算力对比**。如果给 340B 也来 32 次投票，差距会怎样，论文没答。**这一条必须写进去，否则整期的说服力建立在一个不对等的比较上。**

3. **2509.17995 是明确的反面，别把它读成「只是补充说明」。** 两句原话摆在这：**"Verifier scaling alone cannot overcome fundamental verification challenges"**，以及 **"Weak generators produce errors that are easier to detect than strong generators."**

   第二句的推论对主线是真的不利：**上面那些成功案例，很可能有一部分红利来自「被判的答案本身就烂，所以好判」。** 一旦生成器变强、错误变得隐蔽，小裁判的优势会缩水。**换句话说，「小裁判够用」这个结论对「生成器有多强」是敏感的，而现在的评测大多用的是没那么强的生成器。** 这是全套材料里最该老实交代的一条。

   再加上它的三条分立发现：**认对**（由题目难度支配）和**查错**（由生成器能力支配）是两件不同的事，而验证器自身的能力起多大作用**还要看题目难度**。所以不存在一个「小裁判够不够用」的统一答案——**答案是随题目难度和生成器强度漂移的**。

4. **2403.02839 给出了失败模式的形状，而且这个模式不会被「多想几轮」治好。** 微调小裁判域内 82.39，对抗子集掉到 16~23——**低于随机猜**，意味着系统性反着判；根因是**偏向格式规整和啰嗦程度，忽略是否遵循指令**，本质上**"inherently operates as a task-specific classifier"**。
   
   关键在于：**这是分布外失效，不是算力不足。** 一个被表面特征牵着走的分类器，让它多投 32 次票，只会把同一个偏见投得更稳——**投票降的是方差，不是偏见。** 前面三篇的漂亮成绩全部是在**域内**取得的（RaR-Science、MATH500、各自的测试分布），**这篇量出的正是它们没测的那一面。** 而且它是四篇里唯一直接对着「微调小裁判」开火的。

### 一句话结论

**可以说：在题目分布已知、生成器不算太强的场景里，把算力从参数挪到推理，小裁判确实能换到大裁判的判分质量，而且便宜十倍——这有三篇独立证据。不能说：这是一个普遍的换算率。** 挪的方式至少有三种（投票 / 审议+工具 / 换读取方式），对照组通常没花等量算力，**红利有一部分来自被判对象太弱**（2509.17995），而**小裁判真正的失效是分布外的偏见，多想几遍不但治不好，还会把偏见投得更稳**（2403.02839）。

**最诚实的主线表述**：不是「小裁判多想几遍就够用」，而是「**判分所需的『大』确实能从空间挪到别处——但换算率不是常数，它随题目难度、生成器强度和是否出分布而漂移；而且投票能降方差，降不了偏见。**」
