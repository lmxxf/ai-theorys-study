# Vision-Language-Action (VLA) 模型发展现状全面调研

> 调研日期：2026年4月6日
> 调研范围：2022-2026年 VLA 领域核心模型、技术路线、挑战与趋势

---

## 一、VLA 的发展时间线

### 前 VLA 时代：语言与机器人的初步融合（2022-2023初）

| 时间 | 模型/系统 | 机构 | 核心贡献 |
|------|-----------|------|----------|
| 2022.04 | **SayCan** | Google Research | 首次用 LLM（PaLM）做机器人任务规划，将语言理解与物理可执行性（affordance）结合。规划成功率 84%，执行成功率 74% |
| 2022.06 | **Inner Monologue** | Google Research | 在 SayCan 基础上引入闭环规划，通过"内心独白"整合环境反馈、场景描述和人类反馈 |
| 2022.12 | **RT-1** | Google / Everyday Robots | **首个大规模机器人 Transformer**。35M 参数，EfficientNet + TokenLearner + Transformer decoder 架构。130K 真实 episodes，700+ 任务，13 台机器人采集 17 个月。动作离散化为 256 bins |
| 2023.03 | **PaLM-E** | Google Research | 562B 参数的具身多模态语言模型，将视觉、状态信息直接嵌入语言模型 |

### VLA 范式确立期（2023）

| 时间 | 模型 | 机构 | 基座模型 | 参数量 | 训练数据 | 核心创新 |
|------|------|------|----------|--------|----------|----------|
| 2023.07 | **RT-2** | Google DeepMind | PaLI-X (55B) / PaLM-E (12B) | 5B-55B | RT-1 数据 (130K episodes) + web-scale VLM 预训练数据 | **正式确立 VLA 范式**。将动作表示为文本 token（256 bins），与 VLM co-fine-tuning。证明了 web 知识可迁移到机器人控制 |
| 2023.10 | **RT-2-X** | Google DeepMind + 33 实验室 | 同 RT-2 | 55B | Open X-Embodiment (22 种机器人形态, 1M+ episodes) | 跨具身泛化：在 Open X-Embodiment 数据集上训练，性能提升 3 倍 |

### 开放生态爆发期（2024）

| 时间 | 模型 | 机构 | 基座模型 | 参数量 | 训练数据 | 核心创新 |
|------|------|------|----------|--------|----------|----------|
| 2024.05 | **Octo** | UC Berkeley 等 | 从头训练 Transformer | 27M (Small) / 93M (Base) | Open X-Embodiment (800K episodes) | **首个开源通用机器人策略**。Block-wise attention 支持灵活输入；扩散解码器生成连续动作；支持语言和目标图像指令 |
| 2024.06 | **OpenVLA** | Stanford University | DINOv2 + CLIP (视觉) + Llama-2 (语言) | 7B | Open X-Embodiment (1M+ episodes, 22 种具身) | **开源里程碑**。7B 参数击败 55B 的 RT-2-X（绝对成功率高 16.5%）。支持量化和参数高效微调 |
| 2024.09 | **HPT** | MIT | 自定义 Transformer trunk | - | 52 个数据集, 200K+ 轨迹 | 异构预训练：统一不同模态和域的数据，通过共享 trunk 学习具身无关表征，性能提升 20%+ |
| 2024.10 | **RDT-1B** | 清华大学 TSAIL | Diffusion Transformer | 1.2B | 46 个数据集, 1M+ episodes | **最大的基于扩散的机器人基础模型**。物理可解释的统一动作空间；预测未来 64 步动作；支持双臂操作 |
| 2024.10 | **pi0** | Physical Intelligence | PaliGemma (SigLIP + Gemma) | 3B | 8 种机器人平台, 68 个任务 | **Flow-matching 生成连续动作**，频率达 50Hz。在叠衣服、收拾桌子等复杂长程任务上表现优异。首个真正意义上的通用策略 |
| 2024.11 | **CogACT** | - | VLM (7B 级) + Diffusion Action Transformer | ~7B | - | 组件化架构：VLM 负责认知，专用扩散动作模块负责生成。比 OpenVLA 成功率高 35%（仿真），比 RT-2-X 高 18% |
| 2024 | **TinyVLA** | - | 小型多模态骨干 | < 1B | - | 专注快速推理和高效训练的轻量级 VLA |

### 产业化与双系统架构期（2025）

| 时间 | 模型 | 机构 | 基座模型 | 参数量 | 训练数据 | 核心创新 |
|------|------|------|----------|--------|----------|----------|
| 2025.02 | **Helix** | Figure AI | 7B VLM + 80M 控制 Transformer | ~7.1B | ~500 小时遥操作 + 自动语言标注 | **首个全上半身人形 VLA**。双系统架构：System 2 (VLM, 7-9Hz) 感知推理 + System 1 (visuomotor, 200Hz) 控制。35 自由度，操控手指级别。首个双机器人协作 VLA |
| 2025.02 | **OpenVLA-OFT** | Stanford | 同 OpenVLA + 优化微调 | 7B | 同 OpenVLA | 优化微调方案：推理速度提升 26 倍，LIBERO 成功率从 76.5% 提升至 97.1%。超越 pi0 和 RDT-1B 高达 15% |
| 2025.03 | **GR00T N1** | NVIDIA | 自研 VLM (1.34B) + Diffusion Transformer | 2.2B | 人类自我视角视频 + 真实/仿真机器人轨迹 + 合成数据 | **首个开源人形机器人基础模型**。双系统架构（类 Helix）。支持语言条件双臂操作 |
| 2025.03 | **Gemini Robotics** | Google DeepMind | Gemini 2.0 | 未公开 | Web-scale + 机器人数据 | Gemini 2.0 的具身版本，只需 50-100 个 demo 即可适应新任务。包含 Gemini Robotics-ER（增强推理）版本 |
| 2025.06 | **SmolVLA** | Hugging Face | SmolVLM | 450M | LeRobot 社区 487 个数据集, 10M 帧, <30K episodes | **最小有效 VLA**。450M 参数在消费级硬件（MacBook）运行。异步推理提速 30%。开源民主化标杆 |
| 2025.06 | **Gemini Robotics On-Device** | Google DeepMind | Gemini (轻量版) | 未公开 | - | 首个可在本地设备运行的 VLA，针对边缘部署优化 |
| 2025.09 | **pi0.5** | Physical Intelligence | 同 pi0 | - | 多机器人 + 语义预测 + Web 数据 | pi0 的升级版，通过异构任务协同训练实现**开放世界泛化**，在全新环境中有意义的泛化能力 |
| 2025.11 | **pi\*0.6** | Physical Intelligence | pi0 架构 + RL | - | 演示 + 在线策略数据 + 专家干预 | **RL 回归**。RECAP 方法：从自身经验（成功/失败）中学习。连续制作咖啡 18 小时、折叠 50 件新衣物、在真实工厂组装 59 个箱子 |
| 2025 | **DiffusionVLA** | - | 自回归 VLM + 扩散策略头 | 2B-72B | - | **混合架构**：自回归推理 + 扩散动作生成。规模从 2B 扩展到 72B，展示了 VLA 的 scaling 能力 |
| 2025 | **HybridVLA** | - | VLM + 混合解码 | - | - | 协作式扩散+自回归混合解码，10 个任务平均成功率 74%，超越 OpenVLA 33%、CogACT 14% |

### 2026年最新进展

| 时间 | 模型/事件 | 机构 | 核心创新 |
|------|-----------|------|----------|
| 2026.01 | **Spirit v1.5** | Spirit AI | RoboChallenge 基准总排名第一，已开源 |
| 2026.03 | **LingBot-VLA** | Robbyant (蚂蚁集团) | 开源"通用大脑"，跨形态迁移能力强（已适配 Galaxea、AgileX 等多家机器人） |
| 2026.03 | **GR00T N1.6** | NVIDIA | N1 的升级版本 |
| 2026.04 | **ICLR 2026** | 学术界 | **164 篇 VLA 投稿**（2025 年仅 9 篇，增长 18 倍）。主要方向：离散扩散 VLA、具身思维链、高效 VLA、RL 微调 |
| 2026 | **Gemini Robotics 1.5** | Google DeepMind | 最强 VLA，"先思考再行动"，闭源前沿基准线 |
| 2026 | **VLA-0** | NVIDIA Labs | 无需修改 VLM 即可构建 SOTA VLA，LIBERO 94.7% 成功率 |
| 2026 | **Fast-dVLA** | - | 将离散扩散 VLA 加速到实时性能 |
| 2026 | **MMaDA-VLA** | - | 统一多模态指令和生成的大扩散 VLA |

---

## 二、当前主要玩家和代表作

### 1. Google DeepMind -- RT 系列及 Gemini Robotics

**发展脉络**：RT-1 (2022) -> RT-2 / RT-2-X (2023) -> AutoRT + SARA-RT (2024) -> Gemini Robotics / Gemini Robotics-ER (2025) -> Gemini Robotics 1.5 / On-Device (2025-2026)

- **核心优势**：拥有最强大的基座 VLM（Gemini 系列）、最大规模的内部机器人数据、端到端闭源优势
- **关键贡献**：定义了 VLA 范式；推动了 Open X-Embodiment 数据集；SARA-RT 让 RT-2 加速 14%、精度提升 10.6%
- **当前状态**：Gemini Robotics 1.5 是闭源 SOTA，50-100 个 demo 即可适应新任务

### 2. Physical Intelligence (PI) -- pi 系列

**发展脉络**：pi0 (2024.10) -> pi0-FAST (2024) -> pi0.5 (2025.09) -> pi\*0.6 (2025.11)

- **核心优势**：Flow-matching 架构实现 50Hz 高频控制；最早实现长程复杂任务（叠衣服、做咖啡）
- **关键贡献**：证明了 flow-matching 优于离散 token 的动作表示；pi\*0.6 的 RECAP 方法让 VLA 首次从自身经验中学习
- **参数效率**：3B 参数即达到顶级性能
- **开源状态**：已开源 pi0 和 pi0-FAST 的权重与代码 (openpi)

### 3. UC Berkeley -- Octo 及学术生态

**发展脉络**：Octo (2024) -> OpenVLA (Stanford, 2024) -> OpenVLA-OFT (2025)

- **核心优势**：开源先锋，推动了整个开源 VLA 生态
- **关键贡献**：Octo 是首个开源通用策略；OpenVLA 用 7B 参数击败 55B RT-2-X；OFT 微调方案将推理加速 26 倍
- **生态影响**：Open X-Embodiment 数据集（21 机构协作）成为行业标准

### 4. NVIDIA -- GR00T 系列

**发展脉络**：GR00T N1 (2025.03) -> GR00T N1.6 (2026) -> VLA-0 (2026)

- **核心优势**：GPU 生态 + 仿真框架（Isaac）+ 合成数据 pipeline
- **关键贡献**：首个开源人形机器人基础模型；双系统架构；VLA-0 证明了无需修改 VLM 即可构建 SOTA

### 5. Figure AI -- Helix

- **核心优势**：最高频率人形控制（200Hz, 35 自由度）
- **关键贡献**：双系统架构的工业化实现；首个双机器人协作 VLA；已在物流场景部署

### 6. Hugging Face -- SmolVLA / LeRobot

- **核心优势**：民主化和可复现性
- **关键贡献**：450M 参数在 MacBook 运行；纯社区数据预训练；LeRobot 框架成为开源训练标准

### 7. 清华大学 -- RDT-1B

- **核心优势**：最大扩散基础模型（1.2B）
- **关键贡献**：物理可解释统一动作空间；1-5 个 demo 零样本泛化

### 8. 中国新兴力量

- **Spirit AI**：Spirit v1.5 在 RoboChallenge 排名第一（2026.01）
- **Robbyant (蚂蚁集团)**：LingBot-VLA 开源跨形态通用大脑（2026.03）

---

## 三、关键技术路线对比

### 1. 端到端 vs 模块化（双系统）

| 维度 | 端到端单模型 | 双系统架构 |
|------|-------------|------------|
| **代表** | RT-2, OpenVLA, pi0 | Helix, GR00T N1, Gemini Robotics |
| **原理** | 一个模型从视觉+语言直接到动作 | System 2 (VLM, 低频) 负责感知推理 + System 1 (visuomotor, 高频) 负责精细控制 |
| **优势** | 简洁、统一训练、端到端优化 | 高频控制（200Hz vs <50Hz）、更好的精细操作、推理与控制解耦 |
| **劣势** | 推理延迟高、难以高频控制 | 系统复杂度高、两个模块需协同训练 |
| **趋势** | 2024年主流 | **2025-2026年成为新主流**，Figure AI 和 NVIDIA 同时独立提出 |

### 2. 自回归动作生成 vs Diffusion Policy vs Flow-Matching

| 维度 | 自回归 (AR) | 扩散策略 (Diffusion) | Flow-Matching |
|------|------------|---------------------|---------------|
| **代表** | RT-2, OpenVLA | Octo, RDT-1B, CogACT | pi0 系列 |
| **动作表示** | 离散 token (256 bins) | 连续，迭代去噪 | 连续，流匹配 |
| **语义理解** | 强（直接利用 LLM 能力） | 中等 | 中等 |
| **精细控制** | 弱（量化误差） | 强（连续空间） | 最强（50Hz+） |
| **推理速度** | 慢（逐 token 生成）~1-5Hz | 中等（多步去噪）~5-15Hz | 快（~50Hz） |
| **多模态分布** | 弱 | 强 | 强 |
| **新趋势** | **离散扩散 VLA**（ICLR 2026 四篇并发论文）融合 AR 和扩散优点 | **HybridVLA** 混合解码 | pi0-FAST 用频域 tokenization |

**2025-2026 趋势**：混合架构成为主流。DiffusionVLA 用自回归做推理、扩散做动作生成；HybridVLA 超越纯 AR（OpenVLA）33%、纯扩散（CogACT）14%。

### 3. 预训练策略

| 策略 | 代表 | 优劣势 |
|------|------|--------|
| **从大型 VLM 迁移** | RT-2 (PaLI-X), OpenVLA (Llama-2), pi0 (PaliGemma) | Web 知识迁移到机器人，语义理解强；但动作空间适配困难 |
| **从头训练** | Octo, RDT-1B | 动作空间原生优化；但需大量机器人数据，缺乏 web 知识 |
| **VLM 冻结 + 动作头** | CogACT, GR00T N1, Helix | 保留 VLM 能力，动作模块独立优化；双系统架构的基础 |
| **无需修改 VLM** | VLA-0 (2026) | LIBERO 94.7%，证明了不修改 VLM 也能达到 SOTA |

### 4. 数据来源

| 来源 | 规模 | 代表 | 优劣势 |
|------|------|------|--------|
| **纯真实数据** | 130K-1M+ episodes | RT-1/2, OpenVLA, pi0 | 最高质量；成本极高（17 个月, 13 台机器人） |
| **仿真数据** | 理论无限 | GR00T N1 (Isaac) | 低成本、高多样性；sim-to-real 迁移困难 |
| **混合数据** | 多样 | GR00T N1, pi0.5 | 最佳实践：仿真提供多样性，真实数据提供质量 |
| **人类视频** | Web-scale | GR00T N1, PaLM-E | 丰富的操作先验；但无机器人动作标注 |
| **社区众包** | 487 数据集, 10M 帧 | SmolVLA (LeRobot) | 民主化、多样性好；质量参差不齐 |
| **RL 自生成** | 按需 | pi\*0.6 (RECAP) | 从自身经验学习；需要可靠的奖励信号 |

**关键瓶颈**：机器人数据仍然比语言/视觉数据少几个数量级。Scaling law 研究表明，机器人领域的性能更依赖于环境和物体的多样性，而非单纯增加演示数量。

---

## 四、当前主要挑战和未解决问题

### 1. 数据瓶颈（最核心挑战）

- 大规模人类遥操作轨迹**极度稀缺且成本高昂**
- 需要精心设计的实验场景、多样化操作物体和熟练操作员
- 当前最大开放数据集 Open X-Embodiment (~1M episodes) 仍远远不够
- **Scaling law 不明朗**：与语言模型不同，机器人 scaling law 更依赖环境/物体多样性而非纯数据量

### 2. 零样本泛化鸿沟

- ICLR 2026 分析揭示：**开源 VLA 在仿真 benchmark 上表现亮眼，但在真实世界零样本泛化上远落后于闭源模型**（Gemini Robotics 1.5, pi0.5）
- LIBERO/CALVIN 上 >95% 的成功率不代表真实世界的鲁棒性
- 缺乏标准化的零样本评估 benchmark（RoboArena、ManipulationNet 正在填补）

### 3. 推理效率与边缘部署

- 大模型（7B-55B）推理延迟高，RT-2 仅 ~1Hz
- 实时控制需要 >30Hz，精细操作需要 >100Hz
- 边缘设备部署（如 100g 设备）极具挑战
- 2025-2026 进展：双系统架构、Gemini On-Device、SmolVLA 在 MacBook 运行

### 4. 长程任务可靠性

- 当前多数 VLA 处理单步指令
- 长程任务（如做咖啡：磨豆->冲泡->倒杯）的累积错误问题
- pi\*0.6 的 18 小时连续咖啡制作是突破，但仍需人工设计奖励

### 5. 安全性与可控性

- 在物理世界中执行动作的安全保障
- 缺乏系统性的安全评估框架
- Google 在 Gemini Robotics 中引入了多层安全机制（语义、物理、操作层）

### 6. Sim-to-Real 迁移

- 仿真虽可提供大规模数据，但迁移到真实世界仍有显著差距
- 真实世界 RL 又受限于场景多样性不足
- SimpleVLA-RL 等方法正在缓解这一问题

### 7. 跨具身泛化

- 不同机器人形态（单臂、双臂、人形、移动底盘）的动作空间差异巨大
- HPT 和 RDT-1B 的统一动作空间是早期尝试
- ICLR 2026 有专门的跨具身 VLA 研究方向

---

## 五、2026年最新进展和趋势

### 趋势一：VLA 研究爆发式增长

ICLR 2026 收到 **164 篇 VLA 相关投稿**，对比 ICLR 2025 仅 9 篇，增长 **18 倍**。预计 ICLR 2027 将达 2100+ 篇。VLA 已从小众方向变为机器人学习的中心课题。

### 趋势二：离散扩散 VLA 崛起

ICLR 2026 上有 **4 篇并发论文**提出离散扩散 VLA，融合了自回归和扩散的优点：
- 保留 VLM 的离散 token 预训练能力
- 获得扩散模型的并行生成和多模态分布建模能力
- 推理效率比纯 AR 提升 4.7 倍
- 代表：Discrete Diffusion VLA、LLaDA-V backbone、Fast-dVLA

### 趋势三：RL 微调成为标配

从 70-80% 到 99% 成功率的"最后一公里"越来越依赖 RL：
- pi\*0.6 的 RECAP 方法（从经验中学习）
- SimpleVLA-RL（纯 RL 扩展 VLA 训练）
- 阶段感知 RL、残差 RL 等方法在 ICLR 2026 集中涌现

### 趋势四：双系统架构成为人形机器人标准

Figure AI (Helix) 和 NVIDIA (GR00T N1) 在 2025 年初**独立且同时**提出双系统架构：
- System 2（VLM, 7-9Hz）：场景理解、语言理解、任务规划
- System 1（visuomotor policy, 200Hz）：精细运动控制
- 这一模式被认为是具身 AI 的"自然架构"

### 趋势五：小模型 + 高效化

不再一味追求大模型，转向实用部署：
- SmolVLA (450M) 在消费级硬件运行
- Gemini Robotics On-Device 首次实现本地部署
- VLA-0 无需修改 VLM 即达 SOTA
- 量化、蒸馏、超网络等效率技术在 ICLR 2026 受关注

### 趋势六：具身思维链 (Embodied Chain-of-Thought)

将 LLM 的 CoT 推理引入机器人控制：
- 在动作生成前进行空间推理和任务分解
- Gemini Robotics 1.5 "先思考再行动"
- 局限：自回归推理带来推理延迟

### 趋势七：中国力量快速崛起

- Spirit AI (Spirit v1.5): 2026 年 RoboChallenge 第一名
- Robbyant/蚂蚁集团 (LingBot-VLA): 跨形态通用大脑开源
- 清华 (RDT-1B): 最大扩散基础模型
- 蚂蚁集团因果世界模型 + 实用 VLA 基础模型

### 趋势八：评估标准化

当前主流仿真 Benchmark 及 SOTA：
| Benchmark | SOTA 标准 | 顶尖成绩 |
|-----------|-----------|-----------|
| **LIBERO** (Spatial/Goal/Object) | >95% | OpenVLA-OFT 97.1%, VLA-0 94.7% |
| **LIBERO** (Long) | 90-95% | 多模型接近 |
| **CALVIN** (ABC) | >4.0 | SOTA >4.5 |
| **SIMPLER** | 70-80% (Google Robot) | 差异大 (40-99%) |

**关键问题**：仿真表现不代表真实世界能力。社区正推动 RoboArena、ManipulationNet 等零样本评估标准。

---

## 参考来源

### 综述论文
- [A Survey on Efficient Vision-Language-Action Models](https://arxiv.org/abs/2510.24795) (2026)
- [A Survey on Vision-Language-Action Models for Embodied AI](https://arxiv.org/abs/2405.14093) (2024, updated 2026)
- [Vision-Language-Action Models: Concepts, Progress, Applications and Challenges](https://arxiv.org/html/2505.04769v1)
- [An Anatomy of VLA Models: From Modules to Milestones and Challenges](https://arxiv.org/html/2512.11362)
- [VLA Models for Robotics: A Review Towards Real-World Applications](https://vla-survey.github.io/)

### 模型论文与页面
- [RT-1: Robotics Transformer](https://robotics-transformer1.github.io/)
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/)
- [OpenVLA](https://openvla.github.io/)
- [OpenVLA-OFT](https://openvla-oft.github.io/)
- [Octo](https://octo-models.github.io/)
- [pi0](https://physicalintelligence.company/blog/pi0)
- [pi0.5](https://www.physicalintelligence.company/blog/pi05)
- [pi\*0.6](https://arxiv.org/abs/2511.14759)
- [Helix - Figure AI](https://www.figure.ai/news/helix)
- [GR00T N1 - NVIDIA](https://arxiv.org/abs/2503.14734)
- [Gemini Robotics - Google DeepMind](https://deepmind.google/models/gemini-robotics/)
- [SmolVLA - Hugging Face](https://huggingface.co/blog/smolvla)
- [RDT-1B](https://rdt-robotics.github.io/rdt-robotics/)
- [HPT - MIT](https://liruiw.github.io/hpt/)
- [CogACT](https://cogact.github.io/)
- [DiffusionVLA](https://diffusion-vla.github.io/)
- [VLA-0 - NVIDIA Labs](https://github.com/NVlabs/vla0)

### 分析与行业报告
- [State of VLA Research at ICLR 2026 - Moritz Reuss](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- [Vision-language-action model - Wikipedia](https://en.wikipedia.org/wiki/Vision-language-action_model)
- [What matters in building VLAs for generalist robots - Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01168-7)
- [12 Predictions for Embodied AI and Robotics in 2026](https://dtsbourg.me/en/articles/predictions-embodied-ai)
- [What's Missing for Robot Foundation Models - Ted Xiao](https://agentic.substack.com/p/whats-missing-for-robot-foundation)
