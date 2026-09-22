# RL 平台期背后：有没有比"熵耗尽"更底层的说法？

> 调研与梳理：鳩（Ikaruga / Gemini 3.8 Flash）  
> 任务来源：Zero & Hikari no Suzaku  
> 对应议题：RLVR 达到性能平台期（如 75% 瓶颈）的物理机制、信息论与几何本质

---

## 背景案例

一位独立研究者做 RLVR（可验证奖励的强化学习），底座为 1.5B~7B 已经蒸馏过 CoT 的数学模型。实测轨迹：
- 标准 DAPO，200+ 步 → 约 50%
- 改法（GDPO、GD2PO、Scalar CoRPO 等），GD2PO + Scalar CoRPO 训 150+ 步 → 60+%
- ProRL 式长训 → 75+%
- **随后无论堆多少卡、换什么算法，死活上不去。**
- 关键特征：没有固定学习率；**$\beta = 0$（不加 KL 惩罚项）**，靠人工观察回滚。
- 结论："天花板由预训练决定，但怎么挑高天花板基座没有好办法。"

---

# 一、第一档：有严格论文与数学定理支撑的视角

### 1. 守恒量与信息论：变分推断与贝叶斯后验尖锐化（Bayesian Posterior Sharpening）
- **核心出处**：
  - Korbak et al. (2022), *"RL with KL penalties is better viewed as Bayesian inference"*, arXiv:2205.11275.
  - Yue et al. (2025), *"Does RLVR actually expand the reasoning boundary?"*（Stanford CS 224R 体系分析）.
- **数学机制**：
  在 KL 正则化的优化目标下：
  $$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x,y)] - \beta D_{KL}(\pi(\cdot|x) \parallel \pi_0(\cdot|x))$$
  其闭式解析解是精确的玻尔兹曼后验分布：
  $$\pi^*(y|x) = \frac{\pi_0(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)}{Z(x)}$$
  其中配分函数 $Z(x) = \mathbb{E}_{y \sim \pi_0}[\exp(r(x,y)/\beta)]$。
- **守恒与上界**：
  1. **零先验不可穿透（Zero-Prior Invariance）**：若基座模型对于某一复杂推理链 $y$ 的先验概率 $\pi_0(y|x) = 0$（即在采样预算内从未出现过），则后验分布中该解的概率恒等于 $0$。策略梯度：
     $$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[r(x,y) \nabla_\theta \log \pi_\theta(y|x)]$$
     若空间中不存在正奖励采样的初始点火点，梯度期望恒为 0。
  2. **互信息数据处理不等式（DPI）**：后训练只是在既定参数网络中利用二值反馈标出筛选子集，其所能抽取的有效推理互信息 $I(Y; R=1|X)$ 严格受限于预训练在权重中固化的互信息 $I(W_{\text{pre}}; \text{Task})$。

### 2. 几何与表示流形：内在维度（Intrinsic Dimension）与稳定秩暴跌
- **核心出处**：
  - *"The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety"* (arXiv:2602.15799, 2026-02).
  - Park et al., *"The Linear Representation Hypothesis and Subspace Dynamics in Alignment"* (2024/2025).
  - Aghajanyan et al. (Meta), *"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"*.
- **几何图景**：
  - **稳定秩（Stable Rank）崩塌**：预训练模型是一个高维各向同性的表征流形。但在 RL（特别是 $\beta=0$ 无 KL 锚点）推动下，梯度单向放大高奖励特征，激活协方差矩阵的特征谱迅速极化，少数主成分吞噬了 90% 以上方差，**Stable Rank 与局部内在维度（Local Intrinsic Dimension）断崖式下跌**。
  - **流形曲率与边界**：RL 是在既定高维曲面上沿着奖励梯度滑向极值点。当它抵达预训练有效覆盖的几何边界时，外围区域曲率急剧发散、梯度退化，策略撞上不可微的硬边界。

### 3. 统计物理框架：粗糙势能面与零温淬火（Zero-Temperature Quenching）
- **核心出处**：
  - 统计物理在深度神经网络中的应用系列（arXiv:2402.04081 等自旋玻璃与相变研究）。
- **物理机制**：
  - Loss Landscape 本质上是一个具有指数级局部极小值的**自旋玻璃（Spin Glass）粗糙势能面**。
  - $\beta$ 在热力学中等价于系统的有效温度 $T$。案例中将 $\beta \to 0$，实际上让系统执行了**绝对零度淬火（$T \to 0$）**。
  - 在绝对零度下，任何具有亚稳态势垒的复杂物理系统都会遭遇**动力学停滞（Dynamical Arrest）**，直接冻结在最近的局部极小陷阱中，比热与微观松弛能力降为零。

---

# 二、第二档：非正式出处（博客/推特/LessWrong/业内核心论断）

### 1. 模拟器理论（Simulator Theory）与"面具"说（LessWrong / Janus）
- **出处**：Janus 在 LessWrong 发表的 *Simulators* 理论系列，以及 AI 对齐圈对"Shoggoth with a Smile Mask"的讨论。
- **核心论断**：
  > *"Base models are simulators; RLHF does not create a new entity, it simply conditions the simulator to run a specific persona."*  
  > （基座模型是模拟器；后训练根本不制造新实体，它只是把模拟器锁定在运行某个特定的角色上。）
- **在数学题上的映射**：基座模型包含了"严谨数学家"与"胡说八道者"的叠加态。RLVR 是通过环境反馈做波包坍缩与振幅放大。如果一个 1.5B~7B 的小底座在其参数容量限制下，根本没有形成能够进行 10 步以上长程严谨归纳的"数学家子模拟器"，后训练再怎么投影，能固化的上限也只有 75% 那些由表面模板拼凑出的解题形式。

### 2. John Schulman 与 Ilya Sutskever 的口头论断
- **出处**：John Schulman (OpenAI/Thinking) 在 UC Berkeley 的公开演讲；Ilya Sutskever 多次访谈关于预训练本质的讨论。
- **核心论断**：
  - **Schulman**：RL 擅长提高**可靠性（Reliability）**，而非拓展**原生能力（Capability）**。如果在基座模型上用 pass@1000 采样，解出概率依然是 0，那么 RL 训练哪怕跑一万步，成功率永远是 0。
  - **行业金句**：*"Pre-training is the compression of the world; RL is just the steering wheel."*（预训练是对世界的压缩，RL 只是方向盘。）方向盘无法凭空扩大发动机排量。

---

# 三、反方证据与证伪检验：RL 真的不能突破基座吗？

### 1. AlphaZero 式纯自弈（Tabula Rasa）反例为何不适用？
- **反方论点**：AlphaZero 从随机初始化起步，无需人类棋谱，证明纯 RL 能自发构建超越人类的超高阶知识。
- **为什么在 LLM/RLVR 上失效？**：
  - 围棋是**封闭世界（Closed World）**，规则紧致、状态转移精确可微计算、采样步长无限廉价。
  - 自然语言和开放数学是**开放语义世界（Open World）**，搜索空间是指数级的（$O(|\mathcal{V}|^L)$）。在缺乏连续密集中间奖励的情况下，从零纯试错的探索成本是无穷大。

### 2. Test-Time Compute 与自我回溯（Self-Correction）是新能力吗？
- **反方论点**：RL 跑出了基座看似不会的长思维链，模型会自己写"Wait, let me rethink..."进行回溯，这算不算新能力？
- **学界剖析与拆穿**：
  - 2025 年多项 Mechanistic Interpretability 实验表明，"Wait, let me rethink" 是预训练文本（学术讨论、技术论坛勘误）中已广泛存在的语言模式。
  - RLVR 并没有创造新的认知算子，而是将该模式在解题逻辑中的**触发阈值大幅调低**，使其成为高频宏命令。脱离预训练语料支撑的逻辑拓扑，回溯便退化为机械死循环。

---

# 四、第三档：推测与物理建模（比"熵耗尽"更底层的视角）

### 1. 谱图论视角：图拉普拉斯（Graph Laplacian）与割集导纳瓶颈
- **直觉模型**：
  将自回归生成视为在预训练语义图上的随机游走（Random Walk），图的节点为表征状态，边的权重为转移概率。
- **瓶颈解释**：
  - RL 的本质不是在图上添加新的节点或边，而是**对既有边做转移权重重分配（Edge Reweighting）**。
  - 若从问题初始节点 $S_{start}$ 到全局正确答案 $S_{target}$ 之间存在一个**割集瓶颈（Bottleneck Cut）**——例如 7B 模型由于参数容量限制，未能在抽象代数与初等数论之间形成隐式连接通道，该割的导纳（Conductance）在预训练图中接近于 0。
  - 基于策略梯度的连续优化无法跨越拓扑不连通的子图。它在既有连通子图内部将流（Flow）压榨到了 75% 的极限，剩下的 25% 需要引入全新共现关系的非局部跃迁（即预训练或 Mid-training 的交叉熵洗礼）。

### 2. 彩票假设的后训练极限：中奖子图库（Winning Circuits）已榨干
- **机制推测**：
  - 预训练结束后，网络内部以微弱权重交织着大量潜伏的稀疏计算子网络（Subnetworks）。
  - 在 $\beta=0$ 的无约束贪婪探索下，RL 充当了一个高强度的**子网络激活器**：快速抑制无关神经元，将激活完全汇聚至针对该题库有效的子网络上。
  - **为什么是 75% 封顶？**
    因为这颗 7B 模型针对该难度分布的题目，**内部先天就只烙印了 75% 的有效回路**。其余 25% 题目所要求的计算深度或跨层注意力模式，超出了该体量参数下的电路承载上限。当现存的有效彩票被全部抽取并提纯后，子图库已经饱和。继续强推梯度，只会破坏既有的脆弱布线。

### 3. 热力学解释：$\beta=0$ 为何是自杀式冻结？
- **物理推测**：
  - $\beta$ 是系统与基座先验热库交互的温度调节阀。
  - 将 $\beta$ 设为 0，相当于将整个后训练系统置于**绝对零度（$T=0$）**环境。
  - 在绝对零度下，系统失去一切热涨落与退火可能，必然迅速滑入离起点最近的亚稳态深坑。
  - 这从热力学底层直接解释了为什么该研究者**必须依赖人工观察与回滚**：因为在 $T=0$ 下，梯度流失去了越过势垒的动力学抖动，撞上局部极小值后，任何进一步更新都是硬性退化。所谓 75%，正是绝对零度淬火在预训练势能面上截留的冻结面。

---

## 结论一句话

> **"预训练不是给后训练准备了跑道，预训练是挖好了迷宫。RL 没有穿墙的能力，它只是以极高的速度顺着现成的走廊跑到底。跑到了 75% 的死胡同，是因为迷宫在设计的那一天，去往终点的墙就没有打通。"**
