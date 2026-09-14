# 311 期核查记录：判分模型该多大

核查日期：2026-09-15
方式：开灯子代理，回 arXiv 原文 / ar5iv 解析逐字提取。

---

## 一、那个 6B 的真实出处（本期地基）

**InstructGPT，arXiv:2203.02155（Ouyang et al., OpenAI, 2022-03），§3.5 Models，Reward modeling (RM) 小节**，逐字：

> Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward. In this paper we only use 6B RMs, **as this saves a lot of compute**, and we found that **175B RM training could be unstable** and thus was less suitable to be used as the value function during RL

来源：https://ar5iv.labs.arxiv.org/html/2203.02155

**⭐ 关键定性（本期主线）**：这是**一句工程备注，不是研究结论**。没有规模消融、没有曲线、没讨论 6B 是不是最优。理由只有两条——省算力、175B 训不稳。**整个 RLHF 范式"判分模型比策略小 29 倍"的依据就是这一句顺带提及的话，全行业照抄四年。**

⚠️ 310 期正文里写的"四年前那套流程里，就已经不是'越强越会判'了"——**措辞要改**：不是"四年前就懂"，是"四年前顺手写了一句，没人验过"。

---

## 二、擦着问题过去的三篇

### Gao, Schulman & Hilton《Scaling Laws for Reward Model Overoptimization》

arXiv:2210.10760（2022，ICML 2023 正式发表）

- 代理 RM 规模 3M → 3B，金标准 RM 固定 6B
- 拟合规律：Best-of-N `R(d) = d(α − β·d)`；RL `R(d) = d(α − β·log d)`（d = 优化后策略与初始策略的 KL 散度）
- RM 大小的角色：α、β 随 RM 参数量平滑变化，更大的 RM 对过优化更鲁棒（金分峰值更高、扛得更久）
- **反直觉一条**：策略变大（1.2B→6B）**并没有**更快过优化，两者在相近 KL 处见顶

**⚠️ 认知校准**：这篇研究的是 **RM 作为"被优化的靶子"有多耐操**，不是"RM 需要多大才够"。不回答本期核心问题。

### Anthropic《A General Language Assistant as a Laboratory for Alignment》

arXiv:2112.00861（含 Askell、Kaplan、Amodei）。研究 preference model 的 scaling trend，结论是"ranked preference modeling 比 imitation learning 好得多，且随规模扩展更好"。**这是"训练目标哪个更好扩展"，不是"PM 相对策略要多大"。只核到摘要。**

**Anthropic 侧未核到专门研究 RM 规模需求的论文**（是"未找到"，非"确认不存在"）。

### Weak-to-strong generalization

arXiv:2312.09390，Collin Burns 等 11 人，OpenAI，2023-12。GPT-4 家族，跨 7 个数量级预训练算力。

PGR（performance gap recovered）：

| 任务 | 朴素微调 PGR | 加技巧后 |
|---|---|---|
| NLP 任务 | **20–50%** | 辅助置信度损失 → 近 **80%** |
| 国际象棋 | 差距小时 >40%；**差距大时 ≈ 0** | bootstrapping 有改善 |
| ChatGPT 奖励建模 | **仅约 10%**，很少超过 20% | 生成式微调 +10–20%；配合真值早停 → 30–40% |

**⭐ 直接打到本期问题上**：三个任务里，**奖励建模恰恰是弱监督效果最差的那一个（~10%）**。"弱的能不能监督强的"，在别的任务上勉强行，**偏偏在"判分"这件事本身上最不行**。

后续线：arXiv:2502.01458（能力与局限）、2412.03881（数据视角）、2508.17018（refinement 方法的局限）。OpenAI 另有 1000 万美元资助计划。

---

## 三、⭐ 唯一正面立靶子的：《Does RLHF Scale?》

arXiv:2412.06000，Zhenyu Hou 等，**清华大学 + 智谱 AI**，2024-12

- 测试矩阵：RM **9B / 32B / 200B** × 策略 **9B → 200B**
- 逐字结论：
  > larger reward models can effectively boost performance, but the improvement still significantly falls behind the gains in Best-of-N evaluation of the reward model.
- **量化**：用 32B RM 时，**策略从 9B 涨到 200B，平均性能增益从 4.4% 掉到 1.9%**
- 另一条：**数据翻 2 倍 > 模型规模翻 4 倍**
- 总判断：RLHF 扩展效率**低于**预训练，收益递减

**人话**：把 RM 做大，收益在 RM 自己的评测上看着挺好，**传导到策略身上就衰减掉了**；而且策略越大，RM 能给的增益越小。

---

## 四、2025-2026：问题被绕过去了，不是被回答了

### DeepSeek-GRM：27B 采样 32 次 ≈ 671B

《Inference-Time Scaling for Generalist Reward Modeling》arXiv:2504.02495（DeepSeek + 清华，SPCT 方法）

- DeepSeek-GRM-27B（基于 Gemma-2-27B）
- **RewardBench 总分：27B + meta RM 引导投票@32 = 72.8**，对比 Nemotron-4-340B-Reward **70.5**、GPT-4o **71.3**
- 逐字：`direct voting with 32 samples of DeepSeek-GRM-27B could achieve comparable performance with the 671B MoE model`
- 逐字：`inference-time scaling could achieve better performance compared to training-time scaling on model sizes`

### AgentV-RL：4B + 工具审议超 SOTA 25.2%

arXiv:2604.16004（ACL 2026 Findings）。把奖励建模改造成多轮、带工具的审议过程（前向 agent 从前提推结论，后向 agent 反查结论对不对得上前提）。**4B 变体超过 SOTA ORM 达 25.2%**。⚠️ 只核到摘要。

### 小 judge 直接读隐藏状态

《Small Language Models as Judges for Rubric-Based Reinforcement Learning》arXiv:2608.30005

- Qwen3 0.6B/1.7B/4B/8B 对标 GPT-4o
- **Qwen3-1.7B "Probe" judge macro-F1 0.835**；当 RL 奖励模型用时策略得 **0.643，反超 8B 生成式 judge 的 0.594**，**判分耗时少 10.7 倍**
- 机制逐字：`useful rubric-satisfaction information is present in the model's hidden states but is not recovered reliably by Generative or Logprob scoring`
- **⭐ 探针直接读残差流，绕过 language head——"信息在隐藏状态里有，是'说出来'这一步把它丢了"**

### SWE-RM：激活 3B 判分

软件工程领域奖励模型，**MoE 架构 30B 总参数、推理时激活 3B**，直接用作 agentic RL 奖励信号。⚠️ **只核到二手转述，未核到原始技术报告 URL，引用需存疑或先补核。**

---

## 五、必须带的反面（防止说过头）

### 被判的越强，越难判

《Variation in Verification: Understanding Verification Dynamics in LLMs》arXiv:2509.17995，12 个基准，模型 2B–72B：

1. 逐字：`weak generators produce errors that are easier to detect than strong generators`
2. 量化：`some weak generators can nearly match stronger ones in post-verification TTS performance (e.g., the Gemma2-9B to Gemma2-27B performance gap shrinks by 75.7%)`
3. 逐字：`verifier scaling alone cannot overcome fundamental verification challenges`

**这篇还质疑了"最强模型当 verifier"这个默认假设**，并指出 verification 效果由三个维度共同决定：问题难度、生成器能力、验证器能力。**不是"验证天然更容易"这么简单。**

### 小 judge 域内够用，域外崩塌

《An Empirical Study of LLM-as-a-Judge: Fine-tuned Judge Model is not a General Substitute for GPT-4》arXiv:2403.02839

- 测 JudgeLM-7B、PandaLM-7B、Auto-J-13B、Prometheus-7B/13B
- 逐字：`the fine-tuned judge model cannot serve as a general substitute for GPT-4 in terms of LLM evaluation`
- **量化崩塌**：域内 JudgeLM ~82% 准确率 → LLMBar（抗偏见）掉到 **16–23%**，同题 **GPT-4 是 93.5%**

### 概念层面

- 正式命名：**verification asymmetry** / **Generation-Verification Gap (GV-Gap)**，已形式化为可测指标
- 学理谱系明确追溯到：认知科学的 **recall vs recognition**、理论 CS 的 **P vs NP**（找解 vs 验解）
- 专门工作：《Weaver: Shrinking the Generation-Verification Gap with Weak Verifiers》arXiv:2506.18203（NeurIPS 2025）——聚合一堆弱 verifier 顶一个强的

---

## 六、总体判断（本期主线）

**"判分模型该多大"这个问题，被拆成碎片研究过，但没有一篇论文正面立过靶子。**

三层：

1. **最著名的那个事实没有研究支撑**——6B 是一句施工便条，全行业抄了四年。
2. **相关研究都在回答隔壁那个问题**——Gao 研究 RM 当靶子多耐打，Anthropic 研究训练目标哪个更好扩展，weak-to-strong 研究弱监督能激发多少能力。唯一正面撞上的《Does RLHF Scale?》是 2024-12 才出，结论是负面的。
3. **2025 年之后这个问题被绕过去了，不是被回答了**——答案从"判分模型要多大"变成"判分要想多久"，判分者的"大"从空间挪到了时间上。**这恰好是 P vs NP 类比的正确读法：验证之所以便宜，不在于验证者更聪明，而在于验证可以反复来。**

**但这条路在跟一个变难的趋势赛跑**：被判者越强越难判，而 weak-to-strong 里奖励建模恰恰是弱监督最差的任务。

**⭐ 一句话概括**：判分模型该多大，全行业抄了一句 2022 年的施工便条；等到有人正经去量，得到的答案是"做大没用"；于是 2025 年之后大家改成让小模型想久一点——而这条路能走多远，取决于"验证比生成容易"这个假设在被判者越来越强时还成不成立，**而现有证据说它正在变弱**。

---

## 七、未核到 / 存疑

- **SWE-RM 原始技术报告 URL 未核到**（激活 3B 那条只有二手转述），引用前需补核或标明。
- 若干规模趋势数字仅二手（7B→72B RM 数学高约 3%、视觉生成 1B→26B 正相关、Qwen2.5 0.5B–14B judge 准确率单调上升且生成式 judge 偏见随规模下降）——**未逐条核到原文，不要直接引用。**
- Anthropic 无专门研究 RM 规模需求的论文：是"未找到"，非"确认不存在"。
