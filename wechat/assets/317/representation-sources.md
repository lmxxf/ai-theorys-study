# 317 期：内部表示与可达支撑集来源核查

核查日期：2026-09-22。仅整理来源，不修改正文。

## 1. 内部电路：2509.21044

- 完整标题：Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs
- 作者：Honglin Zhang、Qianyue Hao、Fengli Xu、Yong Li。
- 机构：Department of Electronic Engineering, BNRist, Tsinghua University，北京。
- 全文：https://arxiv.org/html/2509.21044v1
- 定位：§3.2 公式 (9)、§3.3、§4.1–4.3，尤其公式 (14)–(16)、Table 1、Figure 3。

§4.2 公式 (15) 的熵是所有样本 EAP 边归因绝对值形成直方图后的 Shannon 熵，不是隐藏激活协方差矩阵的特征谱或有效秩。公式 (14) 是归因绝对值的平均，公式 (16) 是归因分布峰度。不能将这些量与有效秩混称。

实验比较四对约 7B 模型：DeepSeek-Math instruct/RL（GRPO）、Mistral SFT/Math-Shepherd（PPO）、DeepSeek-R1-Distill-Qwen/AceReason-Nemotron（GRPO）、Qwen2.5 SFT/DPO。任务为 GSM8K、MATH、College Math。§3.3 只保留两模型都答对且长度接近的题，并截取生成前部计算归因。在线 PPO/GRPO 的归因强度和分布复杂度总体增加，有例外；DPO 不一致。

稿件问题：这不是“内部表示秩坍缩的方向相反反证”，也不能写“唯一测过的结果相反”。可以用它说明输出分布与内部电路需要分别测量。公式中的冻结参考分布也不能证明训练模型的内部表示没变。

## 2. 自由能：2605.08368

- 完整标题：On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective
- 作者：Yuhao Li、Shengchao Liu。
- 机构：The Chinese University of Hong Kong（香港中文大学）。
- 元信息：https://arxiv.org/abs/2605.08368
- 全文：https://arxiv.org/html/2605.08368v1
- 定位：摘要；§4.1；§4.4–4.5；§5 View 1、View 3、View 4。

§4.1 明确指出 softmax 模型的严格数学支撑集对能力分析太弱，因而改用 accessible support（可达支撑集）：有限采样、解码、搜索、优化、散度预算下实际能产生的行为。它是诊断概念，并没有唯一估计量。

§4.4 与 §5 View 1 认可 RL 结合搜索、交互、过程监督、工具或新信息参与能力创造；§5 View 4 区分参数空间优化动态与行为分布变化。局部重加权框架不能替代全部优化分析。

稿件问题：把该文概念写成严格“概率非零输出的集合”，再归结为“RL 只能激发”，与原文明示的概念和范围相冲突。改先验也不是唯一可能的实践路径。

## 3. 过训练：2606.15455

- 完整标题：Understanding Diversity Collapse in RLVR via the Lens of Overtraining
- 作者：Suqin Yuan、Jinkun Chen、Jiyang Zheng、Muyang Li、Lei Feng、Dadong Wang、Tao Xiang、Tongliang Liu、Bo An。
- 机构对应：Yuan、Li、Liu 属 Sydney AI Centre, The University of Sydney；Chen、Feng 属 Southeast University；Zheng 属 Microsoft；Wang 属 Data61, CSIRO；Xiang 属 Chongqing University；An 属 Nanyang Technological University。
- 全文：https://arxiv.org/html/2606.15455v1
- 定位：§3.1–3.3，Figure 2c、Figure 3–4；§5 实验设置；Appendix A。

精确数字：Qwen2.5-Math-7B 在 MATH 训练集上做 RLVR，训练 5→20 epochs 时，MinervaMath 与 OlympiadBench 两项平均 pass@256 从 75.2% 降到 70.6%，pass@1 从 37.3% 到 37.7%，增加 0.4 个百分点。不是两项各自都得到这组数。

§5 设置：MATH 训练集约 7.5k 题，每题 8 次 rollout；峰值学习率 1e-6、cosine decay；最大提示长度 1024、回答长度 3072；KL 系数为 0，约 140 steps / 20 epochs。

§3.3 Figure 3–4 还显示初始 pass@256=0 的题训练后变得可解；限制只更新零成功桶可让 pass@256 超过基座。原文认为总体下降可能是新增能力被更大的遗失抵消，不能据总体下降断言没有新增。这里的初始不可解依然是有限采样定义，不等于数学概率严格为零。

稿件问题：数字本身准确，但若只用该文证明“队伍没新增”，就漏掉作者直接反对这种推论的主要结果。

## 4. 安全几何：2602.15799

- 完整标题：The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety
- 作者：Max Springer、Chung Peng Lee、Blossom Metevier、Jane Castleman、Bohdan Turbal、Hayoung Jung、Zeyu Shen、Aleksandra Korolova。
- 机构：Princeton University，全体作者。
- PDF：https://arxiv.org/pdf/2602.15799
- HTML：https://arxiv.org/html/2602.15799v1
- 定位：PDF 首页机构；§3–7，尤其参数空间 Fisher 敏感子空间、Alignment Instability Condition。

讨论参数空间安全敏感方向、曲率耦合与微调安全退化，不是检验 RL 导致隐藏激活协方差有效秩下降。稿件参考条目对范围的区分基本正确，但正文没有直接使用，删除该条也不影响论证。

备注：HTML 顶部自动日期与 PDF 首页日期不一致；不采用 HTML 自动日期判断投稿或修订时间。
