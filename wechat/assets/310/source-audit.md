# 310 期核查记录

核查日期：2026-09-14
方式：三组开灯子代理，全部回一手源（arXiv HTML 正文 / 官方 PDF / 会议 virtual 站点）逐字提取。

---

## 一、会议接收状态（截至 2026-09-14）

一手源：`iclr.cc` / `icml.cc` 的 virtual 站点 oral 列表页（服务端渲染，curl 可抓全，每篇页面 `<title>` 自带 "ICLR Oral" / "ICML Oral"）；ACL 走官方 best_papers 页 + ACL Anthology。OpenReview 全站 challenge 验证（WebFetch 与 curl 均 403），未用。

**主题命中量（标题含 self-evolving / self-improving）**：ICML 约 20 篇、ICLR 24 篇、ACL 21 篇、COLM 17 篇 ≈ 八十多篇。

**拿到 oral / 奖的共 6 篇：**

| # | 论文 | 会议/档次 | 机构 | arXiv | 属哪一支 |
|---|---|---|---|---|---|
| 1 | Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine | ICLR 2026 **Oral** | KAUST（Schmidhuber 组，他本人末位作者） | 2510.21614 | agent 改自己代码。核心贡献 = 指出"自我改进潜力"与 benchmark 分数错配（Metaproductivity-Performance Mismatch），改用后代成绩聚合指标 CMP 选分支 |
| 2 | In-Place Test-Time Training | ICLR 2026 **Oral**（Session 3C） | ByteDance Seed + 北大 | 2604.06169（comments 字段自述 "ICLR 2026 Oral Presentation"） | TTT / 推理时权重更新 |
| 3 | Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs | ICLR 2026 **Oral** | 华科 + 阿里（Jieping Ye 在列，机构属推断） | 2603.02556（comments 写 "accepted to ICLR 2026 (oral)"） | self-training / STaR 系 |
| 4 | Neon: Negative Extrapolation From Self-Training Improves Image Generation | ICLR 2026 **Oral** | UT Austin ECE + Rice（Baraniuk） | 2510.03597 | self-training 反向用：把自训练退化当信号，梯度反向外推。图像生成非 LLM |
| 5 | Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning | ICML 2026 **Oral** | UNC-Chapel Hill（Huaxiu Yao 组） | 2511.19900 | self-evolving agent + 自我批判，Solver/Verifier 双角色 |
| 6 | CURE: Critique-Driven Unified Reinforcement Learning for Test-Time Self-Improvement | ACL 2026 **Outstanding Paper**（全主题唯一的奖） | 人民大学（Yankai Lin 末位，机构属推断） | 未核到 | 测试时无 ground truth 自改进 |

**最佳论文：这个主题一篇都没有。** ICLR 2026 两篇 Outstanding = Transformers are Inherently Succinct / LLMs Get Lost In Multi-Turn Conversation；ICML 2026 两篇 = 均为 diffusion。

**尚未到时候的三个会：**
- **NeurIPS 2026**：决定未公布，作者通知日 2026-09-24，会议 12/11–13。
- **EMNLP 2026**：8/20 已通知（5252 篇），会议 10/24–29 布达佩斯未开，ACL Anthology 无 2026 卷（404）。
- **COLM 2026**：7 月初已出 856 篇接收名单（`colm.eventhosts.cc`，非 colmweb.org），但 **oral 名单未分配**，日程里 6 个 Oral Session 全是空壳，当前一律标 poster。筐内 18 篇，含 Agent0（2511.16043）、Self-Evolving Curriculum（2505.14970，含 Bengio）、Scaling Self-Play with Self-Guidance（Stanford）、Recursive Agent Optimization（CMU, Neubig）、CORAL（MIT）。

**⚠️ 三条防坑（写作时差点栽）：**
1. **Darwin Gödel Machine（arXiv:2505.22954，UBC+Vector+Sakana）在 ICLR 2026 是 poster 不是 oral。** 名气最大、最像"自进化"字面义的工作，六个会里零 oral。
2. **Agent0（非 VL 版）的 GitHub README 标 "[ICML'26 & COLM'26]"，但 ICML 全量 6628 篇标题比对查无此篇**（只有 Agent0-VL）。COLM 那半句是真的。README 徽章不可靠。
3. **`2026.emnlp.org` 的论文列表页当前返回的是 EMNLP 2025 的论文**（抽查五篇全部落在 2025）。从那页抓的东西一篇都不能当 2026 用。
4. **spotlight 档无法核实**：ICLR 确有此层级，但 virtual 站点只暴露 oral 和 poster 两种页面类型，poster 页不区分 spotlight。所以凡"非 oral"的只说"不是 oral"，不替它判档。

---

## 二、三篇泼冷水论文（回 arXiv HTML 正文逐条核）

**共同点：三篇全是 v1 单版本预印本，全部未被任何会议接收，无同行评审记录。**

### 2606.21090 — Self-Improvement Can Self-Regress

- 作者：**Jianzhe Lin，单人独作**；机构 **Meta AI**（HTML 标题页脚注 "Work done at Meta"，abs 页无），jianzhelin@meta.com
- 提交 2026-06-17 v1，cs.AI
- 实验（§5.1）：Qwen-2.5-3B/7B/32B-Instruct（32B 作冻结基线）+ Gemma-3-4B pilot；competitive programming，CodeGrader 二值 reward，题源 HumanEval/MBPP/APPS；REINFORCE post-training，50 步一个 campaign 串联（C1→C2→C3）
- 崩塌（§1 + Fig 1 caption 原文）：`pass@1 rises from 25% to 81% within the first ≈50 steps and then collapses to near-zero by step 200`；§5.2 `All methods collapse within every campaign`；Table 2 caption `continuing from the collapsed checkpoint yields pass@1 == 0.00`
- **KL/EWC（§5.2 原话）**：`KL regularization is counterproductive. EWC (KL=0.05) and adaptive KL both consistently end near zero (0.08–0.10 in C2/C3). The KL penalty anchors the model to each campaign's degraded end-state.`
  - ⚠️ **此文的 "EWC" 不是真 Fisher-matrix EWC**，是论文自定义的 "Strong KL penalty to reference policy (EWC-style constraint)"（§5.1 baseline 定义）。正文未用 EWC 一词，只说"对着原始模型算 KL 惩罚"。
- **CARE**（跨 campaign 记忆，Table 6 §5.6）：3B naive **4.9% [2.1, 9.5]** → CARE **9.5% [6.3, 12.7]**，paired bootstrap 95% CI 不含零。7B 11.8%→13.8%，**CI 重叠不显著**。正文未引此数（避免过度解读）。
- **ES（早停+回滚）**（Table 9 §5.8）：Qwen-2.5-7B **22.2% [14.1, 28.0]**。正文引用。
- **GRPO**（Table 10 §5.9）：naive 20.7% [15.7, 25.1]，+CARE 20.4%。§1 原话 `GRPO raises the floor, but does not remove the cliff.` 后接：GRPO 的 7B 收益来自 between-campaign carryover，不是 within-campaign 稳定；per-campaign peak-to-end gap 两者均约 17 pt。
- **机制解释**（§1）：`This is within-task policy over-optimization: early REINFORCE sharpens useful behaviours already present in the pretrained model, but continued optimization narrows the policy around brittle reward-correlated patterns, reducing solution diversity and overwriting broad code-generation priors.` 并明确排除任务切换归因。

### 2606.28438 — When AI Reviews Its Own Code

- 作者：Xinyuan Song, Zekun Cai, Liang Zhao；Song/Zhao = **Emory University**，Cai = **东京大学 + LocationMind**
- 提交 2026-06-26 v1
- 设置（§3）：SantaCoder 1.1B / StarCoder2-3B / Qwen2.5-Coder-1.5B / Code Llama-7B；HumanEval/MBPP/LiveCodeBench + HumanEval+/MBPP+；**5 轮**，每轮固定 prompt 池生成、门选 5000 条、微调 3000 步（5 轮 = 15k 步）
- 三 regime：
  - Vanilla 无审查最快塌：SantaCoder MBPP+ 0.294→0.019（**−93.5%**）
  - Human-gate（编译+静态）：`Compile filtering improves Qwen2.5-Coder at R5 relative to Vanilla on both HumanEval+ (0.098 vs. 0.043) and MBPP+ (0.127 vs. 0.082)` — 减缓不阻止
  - AI-self-gate：早期最高（R5 HumanEval+ SantaCoder 0.1037 / StarCoder2-3B 0.122 / Qwen2.5-Coder 0.122），后期失去过滤力
- **rubber-stamp 是原文用词**（三处）：摘要/§4 `The binary classifier is the clearest AI-self-gate failure case: acceptance scores rise while benchmark correctness falls, indicating a self-confirming rubber-stamp regime.`；§2.3.1 定义 `the rubber-stamp regime: on the code that the current model is likely to generate, the AI reviewer assigns essentially the same acceptance score.`
- ⚠️ **"通过率上升"与自己的表有张力，正文已据实修正**：Table 2 里 **PPL 门**通过率 0.167(R1)→0.235(R5)（原文 `indicating gradual degradation of the self-calibration signal`）；**binary 门**通过率 5 轮恒定 **0.250**，并未上升。正确率两门都在掉（binary: HumanEval+ 0.110→0.104，MBPP+ 0.061→0.019）。**正文采用"接受信号逐渐失去区分度，正确率持续下滑"的说法，并在文中公开纠正了自己最初的转述。**
- ⚠️ **"数学证明"必须收窄**：是 **Theorem 2.3（Degeneracy to ungated recursion）**，前提为 **Assumption 2.2（Self-confirming acceptance）**——存在可测函数 κ:𝒳→(0,1] 使 rθt(x,c)=κ(x) 对 pθt(·|x)-几乎所有 c 成立，**即已假设审查者对本模型输出给同一分**。结论 mtA = mt^ungated (a.e.)。**这是把橡皮图章形式化，不是证明 AI 自审必然崩。正文已如实写明。**
- 结论（§4 / 摘要）：`Stable recursive code LLM training therefore needs verification that remains outside the model's own preference distribution` / `requires exogenous verification rather than model-coupled self-review`

### 2607.04277 — Self-Reference in Large Language Models

- 标题原文写全称 **Large Language Models**（非 LLMs 缩写）
- 作者：Jiang Zhang（**北京师范大学系统科学学院 + 集智 Swarma Research**）、Bing Yuan、Qian Zhang（均 Swarma Research）
- 提交 2026-07-05 v1，**arXiv 分类 physics.soc-ph**（非 cs.AI/cs.CL），标题页自标 "A Preprint"
- **纯理论 + 文献综述，无任何原创实验。** 摘要里的 "empirical review" 指综述他人 2022–2026 工作，不是自己做实验。
- quasi-introspection 定义（§4.6 逐字）：`fragmentary, approximate, and domain-specific self-knowledge that resembles introspection in some dimensions but lacks the completeness, reliability, and causal grounding required by our formal definition.`
- 三个结构性障碍（§5.1 小节标题）：①No Reflexivity: The Absence of Complete Self-Access ②Feedforward Architecture and the Impossibility of Self-Simulation ③Self-Reports Without Causal Grounding
  - ⚠️ 摘要版第三条写的是 "computational class constraints that prevent fixed-point iteration"，与正文 §5.1.3 标题不一致。**引用时择一，正文只用了前两条 + 冯·诺依曼阈值论。**
- 冯·诺依曼原话（§3.1.1）：`There is a minimum number of parts below which complication is degenerative, in the sense that if one automaton makes another the second is less complex than the first, but above which it is possible for an automaton to construct other automata of equal or higher complexity.`
- Kleene 第二递归定理（§3.2.2）：`For any total computable function f:ℕ→ℕ, there exists an index (source code) e such that φₑ ≃ φ_f(e).` **仅用于存在性论证**（内省程序理论上不被排除），不证明 LLM 有内省、不证明可达到。正文未引此条。

---

## 三、厂商官方原话（14 条全核到，0 条未核到）

### 智谱 GLM-5.3 官方博客

来源 https://z.ai/blog/glm-5.3 。⚠️ 页面是 React SPA，`curl` 拿到的 HTML 只有 598 字节空壳，**正文打包在 JS 里**：`https://z.ai/blog/assets/glm-5.3-B8Uy_aqG.js`（30KB），引文均从该 bundle 逐字提取，可复现。

全部核实一致的句子：
1. `Scaling post-training is all we did for GLM-5.3.`（**开篇第一句，加粗**）
2. `It uses the same base model as GLM-5.2 — every gain comes from post-training.`
3. `As agent capability improves, much of the difficulty in scaling post-training moves from the model to the environment.`
4. `A useful task environment has to be executable, verifiable, and close to real professional work — and we need many of them, not a handful of hand-built ones.`
5. `a judge agent then attempts each task to verify that it is actually solvable`
6. `Verifiers are synthesized without access to the reference solution, while solver trajectories are used to discover and close reward shortcuts.`
7. `A verifier that passes oracle, no-op, and unsolved-state checks produces a binary reward reliable enough to train on directly.`

**人工介入那句（正文主证据之一）：**
> These pipelines still require a meaningful amount of human-in-the-loop work; making environment generation and verification more autonomous is one of the next steps.

⚠️ **位置修正**：**不在博客结尾**（282 期曾这么写，实为误记）。它在"环境合成"一节末尾，紧接基准分数句（Terminal-Bench 3.0 4.6→28.3、DeepSWE v1.1 46.2→66.9、Agents' Last Exam 23.8→28.5）之后；后文还有 Z.ai Code Bench、Emergent Cyber Capability、slime 等数节。主语是 `These pipelines`（环境+verifier 合成流水线），**不是整套 post-training**。原文把它框成 `one of the next steps`（待办），不是纯坦白。

**⭐ 漏洞数字（本期最有力的证据，比上一条好用）：**
> We then tested whether these capabilities transfer beyond controlled benchmarks. Since GLM-5.2, we have been working with several security teams in China to run our models against real-world codebases. **After expert review, screening, and deduplication,** the model identified 2,436 vulnerabilities across 269 projects, including 1,097 medium-to-high severity issues. The findings span system kernels, operating systems, browser engines, open-source infrastructure, web applications, and network protocols. Many had remained unnoticed for years or even decades, with the oldest dating back roughly 40 years.

"人筛过"这句就长在战绩数字的主句前面，不是免责声明。

另：`What surprised us was how quickly the capability continued to develop as training scaled.`（Emergent Cyber Capability 节，前句 `We expected this to make the model better at finding and reasoning about vulnerabilities.`）

⚠️ **博客自身不一致**：正文写漏洞最老 `roughly 40 years`，页面统计卡片写 `45 YEARS OF IMPACT`（卡片另有 2,436 FINDINGS TRACKED / 53 PUBLICLY DISCLOSED / 2,383 UNDER EMBARGO / 1,097 CRITICAL & HIGH / 269 OSS PROJECTS）。**引哪个都行，别两个一起引。正文用了 40 年。**

### Kimi K3 技术报告

来源 https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf （官方 PDF 逐字提取）

| 处 | 原文 | 位置 |
|---|---|---|
| 网页开发 | `We construct a diverse suite of expert-curated web development tasks covering typical scenarios.` | §4.2.7 |
| SFT 数据 | `Specifically, we synthesize data trajectories using domain-specialized models from the prior Kimi series, followed by multi-stage verification and human-in-the-loop annotation.` | §4.1.1 |
| 推理长度旋钮 τ | `The adjustment of τ is configured per domain under human-in-the-loop guidance.` | §4.1.2 |
| 个人助理任务 | `Building on these mock applications, we design complex tasks inspired by real-world professional workflows in scenarios like human resources, legal services, and finance.` | §4.2.5 |

⚠️ **第 4 条须收窄**：同节后文 `The initial workspace is constructed by agents that autonomously search the web...`——**任务由人设计，环境由 agent 填充**。说"这块全靠人"会过头。正文已按此写。

### DeepSeek V4 生成式奖励模型

来源 https://arxiv.org/html/2606.19348v1 （*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，2026-04-26），位置 **§5.1.1 Specialist Training，小标题 "Generative Reward Model."**

完整原文：
> **Generative Reward Model.** Typically, easy-to-verify tasks can be effectively optimized using simple rule-based verifiers or test cases. In contrast, hard-to-verify tasks traditionally rely on Reinforcement Learning from Human Feedback (RLHF), which necessitates extensive human annotation to train a scalar reward model. In the post-training phase of DeepSeek-V4 series, however, we dispense with these conventional scalar-based reward models. Instead, to address hard-to-verify tasks, we curate rubric-guided RL data and employ a Generative Reward Model (GRM) to evaluate policy trajectories. Crucially, we apply RL optimization directly to the GRM itself. In this paradigm, the actor network natively functions as the GRM, enabling the joint optimization of the model's evaluative (judging) proficiency alongside its standard generative capabilities. By unifying these roles, the model's internal reasoning capabilities are inherently fused into its evaluative process, resulting in highly robust scoring. Furthermore, this approach achieves superior performance with only a minimal set of diverse human annotations, as the model leverages its own logic to generalize across complex tasks.

三点校准（正文均已按此写）：
1. 扔掉的是 **scalar（标量）奖励模型**，不是所有独立 RM。
2. **判分能力本身也在被 RL 训练**（`we apply RL optimization directly to the GRM itself`）——284 期转述时漏了这点。
3. ⚠️ **适用范围仅限 hard-to-verify tasks**，easy-to-verify 仍走 rule-based verifier / test case。**不能写成"V4 全部奖励都自己判"。**
4. `a minimal set of **diverse** human annotations` — 是"少而多样"，diverse 不是凑数形容词。

---

## 四、写作过程中自我纠正的三处（已写进正文）

1. **橡皮图章"通过率上升"挂错了门**（binary 恒定 0.250，上升的是 PPL 门 0.167→0.235）。正文第 311 行公开纠正。
2. **"数学证明 AI 自审必然崩"夸大了**，定理前提已假设橡皮图章。正文第 313 行说明。
3. **内省阈值那篇不是实证研究**，纯理论、physics.soc-ph 分类、无同行评审。正文已标明。

另：**282 期写"智谱博客结尾承认需要人工介入"位置有误**（实为环境合成节末尾），本期已更正，该期不回改。

---

## 五、可复用的通用教训（已同步到写作规矩）

- **`2026.emnlp.org` 论文列表页当前返回 2025 年论文** —— 会议官网的"当年列表"未必是当年的，抓之前抽样验证年份。
- **GitHub README 的会议徽章不可靠**（Agent0 自称 ICML'26，全量比对查无此篇）。
- **React SPA 官方博客 curl 拿不到正文**，要去 JS bundle 里提（z.ai 即此类）。
- **spotlight 档次在 ICLR/ICML virtual 站点无法区分**，只能确证"是不是 oral"。
