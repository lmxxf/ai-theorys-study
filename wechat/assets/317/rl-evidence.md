# 317 独立 RL 证据核查

核查日期：2026-09-22。只核来源，不修改正文。

## 1. Yue et al. 2504.13837

- 一手全文：https://arxiv.org/html/2504.13837v5
- 元数据：https://arxiv.org/abs/2504.13837
- 标题、八位作者、清华 LeapLab / 上海交通大学、2025-11-24 v5、NeurIPS 2025 Oral 均与稿中相符；两个 Yang Yue 确实是不同作者（乐洋、乐阳）。
- §3.1 / Figure 2：所测设置出现低 k 时 RL 优、高 k 时 base 追上反超。数学采用 Qwen2.5 7B/14B/32B、Qwen2.5-Math、LLaMA3.1-8B 等，另有代码、视觉任务。
- §4.1 Table 2 就存在 base 未过、RL 通过的 MATH500 题：1.0%；作者说的是近似子集。不能把有限采样集合直接称为数学支撑集。
- §4.2 蒸馏对照支持该实验中教师传入推理模式。论文结论反复限定 current RLVR / current training setup，不能转述为一切 RL 不可能产生新能力。
- 核查判断：pass@k 交叉可说明给定提示、解码与预算下覆盖改变，不能证明 π0=0，也不能证明参数化模型内部没学到新东西。

## 2. ProRL 2505.24864

- 一手全文：https://arxiv.org/html/2505.24864v1
- 正确全名：ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models。
- 作者：Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong；NVIDIA；2025-05-30。
- **必须改归属**：稿中“Korbak 那篇对此的回应”其实是本篇 §2.3.1 的观点：去 KL 的工作常从未 SFT 基座开始，他们从 DeepSeek-R1-Distill-Qwen-1.5B 开始，保留 KL 有助稳定与熵。不能挂到 2022 Korbak 名下。
- 同节有周期性替换参考策略并重置优化器，不符合一个初始参考分布固定到底的简单叙述。
- §4 报告逻辑等任务上高 k 仍有增益；本文的主旨正是支持持续 RL 扩展可观测推理边界，不能只拿来支持 KL 稳定性，却在下一节说支持新能力的一方没有实证。
- 这些也不是对严格数学零概率的证明；合适写法是“已有扩大有限预算下解题覆盖的实测，不能宣布 RL 只能激发”。

## 3. Interplay 2512.07783

- 一手全文：https://arxiv.org/html/2512.07783v1
- 作者 Charlie Zhang, Graham Neubig, Xiang Yue；Carnegie Mellon University / Language Technologies Institute；2025-12-08。
- §2.4 / Appendix A.3：从头训练 Qwen2.5 架构 100M 模型、10B 合成算术 token；过程和最终答案共同验证；后训练 GRPO，KL 系数 1e-3。
- 摘要、§3：RL 数据处于能力边缘、预训练留有空间时，pass@128 外推泛化可以提升。§4：基本操作少量预训练暴露后，RL 可组合成更复杂任务。
- §5：固定算力比较 mid-training 与 RL 配比；较难外推任务受益于保留适量 mid-training、加大 RL 预算。
- **未核到**：“目前没有可靠指标事前预测强化学习潜力”“中期训练提升在基座跑分看不出、仅 RL 后显形”这两个稿中归属。全文 §5 不如此表述，不宜保留为本文结论。
- 可替代：一个基座的当前跑分不足以单独回答后训练空间；在这一合成任务实验里，预训练覆盖、后训练难度和算力分配共同影响结果。避免推广为已证明大模型选型无指标。

## 4. 稿中未署名的 pass@k 反驳

- 伪阳性来源已确认：Xumeng Wen et al., Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs；2025-06-17；微软亚洲研究院、北大、港中文、UCLA 作者。
- 全文：https://arxiv.org/html/2506.14245v1
- §4 / Figure 2：Qwen2.5-32B 与 DAPO-Qwen-32B，改用 CoT-Pass@K，检查推理过程和答案；AIME2024/2025 中至 k=1024 仍保留 RL 优势。评审模型为 DeepSeek-R1-0528-Qwen3-8B，有人工抽查。Math500/AMC23 效应较弱、Minerva 未改善，因此不能写所有交叉一概消失。
- “多数模型对不显著、8K 无交叉而 32K 有”未定位到可信的一手原出处，建议删除具体 8K/32K 例子，保留通用“比较应统一并报告生成长度、提示词与解码条件”。
- 查到一篇相关而非完全匹配的统计批评：Pass@k Is Not a Property of a Model: Prompt Format, Decoding, and Statistical Power in Evaluations of RL for Reasoning，https://www.preprints.org/manuscript/202608.1495 。其 §5 的统计结果是一个基座规模、一种算法、三个种子；并非“多数模型对”。其对 Yue Table 2 的复核中 MATH500 差异显著 (p=0.0106)，AIME24 不显著 (p>=0.125)。不要将它误写成普遍否定原论文。

## 必改结论

“RL 不可能学会基座不会的东西”“没有哪一方拿出了新能力硬证据”“网友那 25% 大概率不在基座里”均不能由上述来源推出。保留先验影响学习难易和探索成本这条主线，将数学支撑集、有限预算可达性、知识与技能学习三者分开。
