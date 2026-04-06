# AI/ML 调研素材库（2026-04-06）

## 一、arxiv 近期热门论文 Top 15

### 自我改进 / Reward Model / Judge

1. **Reinforcement Learning from Self-Feedback (RLSF)** — arxiv.org/abs/2507.21931
   - LLM 用自身置信度当内在奖励，无需人类标注，纯自我改进

2. **J1: Incentivizing Thinking in LLM-as-a-Judge via RL** — arxiv.org/abs/2505.10320 (ICLR 2026)
   - 用 RL 激励 Judge 深度思考再评判

3. **Recursive Rubric Decomposition (RRD)** — arxiv.org/abs/2602.05125
   - 递归分解评分标准，Judge 偏好精度大幅提升，奖励信号稳定性 +10-20%

4. **Meta-Rewarding: Self-Improving via LLM-as-a-Meta-Judge** — arxiv.org/abs/2407.19594
   - 三层自我改进：Actor → Judge → Meta-Judge，全程无监督，AlpacaEval 22.9% → 39.4%

5. **SPELL: Self-Play RL for Evolving Long-Context LLMs** — arxiv.org/abs/2509.23863
   - 三角色自博弈（提问/回答/验证），无标注数据持续提升长上下文推理

### MoE 相关

6. **Mixture of Universal Experts (MoUE)** — arxiv.org/abs/2603.04971
   - "虚拟宽度"新维度，跨层通用专家池，超越基线 1.3-4.2%

7. **Optimal Expert-Attention Allocation in MoE** — arxiv.org/abs/2603.10379
   - MoE 的神经缩放律，专家与注意力的最优分配遵循幂律

8. **The Expert Strikes Back** — arxiv.org/abs/2604.02178 (2026.04)
   - MoE 在专家层面具有内在可解释性

9. **Speculating Experts** — arxiv.org/abs/2603.19289
   - 用内部表示预测未来专家，推理加速 14%

### AI 意识 / 潜空间

10. **Just Aware Enough: Artificial Awareness under Consciousness Uncertainty** — arxiv.org/abs/2601.14901
    - 从"意识"转向"人工觉知"，定义四个评估标准

11. **The Latent Space: Foundation, Evolution, Mechanism** — arxiv.org/abs/2604.02029 (2026.04)
    - 潜空间全面综述，HuggingFace 123 upvotes

12. **AI Consciousness is Inevitable** — arxiv.org/abs/2403.17101
    - 从理论计算机科学角度论证机器意识不可避免

### Transformer / 架构

13. **Geometric and Dynamic Scaling in Deep Transformers** — arxiv.org/abs/2601.01014
    - 流形几何 Transformer (MGT)，强制几何有效性避免秩坍缩
    - 两个核心机制：mHC（流形约束超连接，控制方向）+ DDL（深度 Delta 学习，控制大小和符号）
    - Delta 算子 A(β,k) = I − β·k·kᵀ，三种模式：β→0 恒等，β→1 擦除，β→2 反射
    - 完整更新：X_{l+1} = X_l + β ⊙ (V_mHC − α·X_l)
    - ⚠️ 纯理论，没有实验结果
    - 核心论点：几何而非深度是瓶颈

14. **MemOS: A Memory Operating System for AI** — arxiv.org/abs/2507.03724
    - 166 upvotes，40+ 人团队
    - 三种记忆：Plaintext（外存）、Activation（KV-cache）、Parameter（权重）
    - 三种记忆可互相转换：高频文本→KV-cache，稳定知识→LoRA，冷知识→外存
    - MemCube 统一抽象：payload + metadata（时间戳、来源、版本链、语义指纹）
    - 生命周期：Generated → Activated → Merged → Archived → Expired
    - 三层架构类比 OS：接口层/操作层/基础设施层
    - vs RAG：RAG 是无状态补丁，MemOS 是有生命周期的资源管理
    - 偏工程设计，数学深度不够

### Agent / 其他

15. **SKILL0: In-Context Agentic RL for Skill Internalization** — arxiv.org/abs/2604.02268
    - 智能体通过 RL 把上下文技能内化为永久能力

### 补充

| 论文 | 链接 | 一句话 |
|------|------|--------|
| Hourglass FFN | arxiv.org/pdf/2602.06471 | 宽-窄-宽子 MLP 替代传统 FFN |
| Generative World Renderer | arxiv.org/abs/2604.02329 | 生成式世界渲染器 |
| OpenResearcher | arxiv.org/abs/2603.20278 | 完全开源的长视野深度研究代理 |
| Omni-WorldBench | arxiv.org/abs/2603.22212 | 世界模型综合评测基准 |

---

## 二、学术顶会一览（2026）

### 时间线

| 会议 | 日期 | 地点 | 状态 |
|------|------|------|------|
| AAAI 2026 | 1月20-27日 | 新加坡 | ✅ 已结束 |
| EACL 2026 | 3月24-29日 | 摩洛哥拉巴特 | ✅ 已结束 |
| AAAI Spring Symposium | 4月7-9日 | 旧金山 | 即将开始 |
| ICLR 2026 | 4月23-27日 | 巴西里约 | 🔥 5300+篇，17天后 |
| CVPR 2026 | 6月3-7日 | 丹佛 | 审稿中 |
| ACL 2026 | 7月2-7日 | 圣地亚哥 | 投稿中 |
| ICML 2026 | 7月6-11日 | 首尔 | 投稿/审稿中 |
| NeurIPS 2026 | 12月6-12日 | 悉尼 | 未开始征稿 |

### AAAI 2026 获奖论文

29,000 篇投稿，录用 4,167 篇，录用率 17.6%

**Outstanding Paper Awards（5篇）**

1. **ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver**
   - 港科大广州/西湖大学 | arxiv.org/abs/2508.10333
   - VLA 模型视觉注意力散 → 用 diffusion transformer 重建注视区域图像引导注意力
   - 隐式 grounding > 显式 grounding > CoT grounding（CoT 在机器人上直接崩）
   - 架构：SigLIP + Qwen2-7b + 冻结 VAE + Diffusion Transformer denoiser
   - 损失：L = L_action + L_visual，L_visual = E[‖D(z_t; h_r, t) − ε‖²]
   - 数据：100k+ 轨迹，200 万样本（BridgeData V2 + LIBERO + CALVIN）
   - CALVIN 5/5: 64.1%（baseline 49.0%），叠积木 79.5%（baseline 59.3%）
   - 真实机器人未见目标 ~80% 成功率，baseline ~0%
   - 消融：注视区域重建 >> 整图重建（+10pp）

2. **LLM2CLIP: Powerful Language Model Unlocks Richer Cross-Modality Representation**
   - arxiv.org/abs/2411.04997
   - LLM 嵌入 CLIP 做后训练，长文本/复杂 caption 能力暴涨，超越 EVA02 和 SigLIP-2

3. **Causal Structure Learning for Dynamical Systems (CaDyT)**
   - arxiv.org/abs/2512.14361
   - 连续动力系统上的因果发现，规则/不规则采样都优于 SOTA

4. **Model Change for Description Logic Concepts**
   - arxiv.org/abs/2603.05562
   - 描述逻辑模型修改理论，证明 revision 不可分解为 eviction+reception

5. **High-pass Matters (Sheaflet HNN)**
   - 证明超图神经网络同时需要低通+高通，提出 sheaflet 架构

**其他热门**
- VLA-Adapter：0.5B 参数轻量 VLA，GitHub 1600+ stars
- Blue Teaming Function-Calling Agents：评估函数调用 LLM 对攻击的鲁棒性
- LLM and Brain Similarity：LLM 嵌入 vs fMRI 脑活动模式
- Slum Detection with MoE (GRAM)：MoE 做跨洲贫民窟卫星图分割

### EACL 2026 获奖论文

700+ 篇论文

**Best Paper**
- **Humans and Transformer LMs: Abstraction Drives Language Learning** — Jasper Jian, Christopher D. Manning（斯坦福）
  - arxiv.org/abs/2603.17475
  - Transformer 和人类语言学习轨迹对比，抽象类级行为先于词汇级行为涌现

**其他值得关注**
- Teams of LLM Agents can Exploit Zero-Day Vulnerabilities — LLM 团队能发现并利用真实零日漏洞
- Understanding Jailbreak Success: A Study of Latent Space Dynamics — 从潜空间动力学分析越狱机制
- CrossThink: Scaling Self-Learning beyond Math Reasoning (NVIDIA) — 自我学习扩展到通用推理
- NP-Hard Lower Bound for Semantic Self-Verification — LLM 自我验证是 NP-Hard
- Barriers to Discrete Reasoning with Transformers — Transformer 推理基本限制综述
- Out of Style: RAG's Fragility to Linguistic Variation — RAG 在语言风格变化下崩溃

---

## 三、VLA（Vision-Language-Action）发展现状

### 一个数字

ICLR 2026 收到 164 篇 VLA 投稿，2025 年是 9 篇。**18 倍增长**。

### 发展时间线

| 时间 | 模型 | 机构 | 参数量 | 核心创新 |
|------|------|------|--------|---------|
| 2022.12 | RT-1 | Google | 35M | 首个大规模真实机器人 Transformer 策略 |
| 2023.07 | RT-2 | Google DeepMind | 55B | 首次 VLM→VLA 迁移，动作编码为 token |
| 2023.10 | RT-X | 合作联盟 | - | 跨机器人数据集标准化（Open X-Embodiment） |
| 2024.06 | Octo | UC Berkeley | 93M | 首个开源通用机器人策略，diffusion head |
| 2024.06 | OpenVLA | Stanford/Berkeley | 7B | 7B 击败 55B RT-2-X |
| 2024.10 | π₀ | Physical Intelligence | 3B | Flow-matching 动作生成，50Hz |
| 2024.11 | RDT-1B | - | 1B | Diffusion Transformer 双臂操控 |
| 2025.01 | π₀.5 | Physical Intelligence | - | 双系统：高层 VLM + 低层 flow policy |
| 2025.02 | Helix | Figure AI | - | 双系统 200Hz，人形机器人 |
| 2025.03 | GR00T N1 | NVIDIA | - | 双系统人形专用，仿真→真实 |
| 2025.03 | Gemini Robotics | Google DeepMind | - | Gemini 2.0 直接做机器人 |
| 2025.05 | OpenVLA-OFT | Stanford | 7B | OpenVLA + RL fine-tuning |
| 2025 | SmolVLA | HuggingFace | 450M | MacBook 上能跑的 VLA |
| 2025 | π₀.6 | Physical Intelligence | - | 扩展泛化能力 |
| 2026 | ReconVLA | 港科大广州/西湖 | 7B | 隐式 grounding，AAAI Outstanding Paper |

### 主要玩家

| 团队 | 代表作 | 路线 |
|------|--------|------|
| Google DeepMind | RT-1/2/X → Gemini Robotics | VLM 直接做机器人 |
| Physical Intelligence | π₀ / π₀.5 / π₀.6 | Flow-matching，融了 $2.4B |
| Figure AI | Helix | 双系统 200Hz，人形赛道 |
| NVIDIA | GR00T N1 | 人形专用，卖铲子 |
| Stanford/Berkeley | OpenVLA, Octo | 开源主力，学术标杆 |
| HuggingFace | SmolVLA | 边缘/民主化路线 |

### 五大技术路线分歧

**1. 端到端 vs 双系统**

端到端：图像+语言 → 一个模型 → 动作（RT-2, OpenVLA, ReconVLA）
双系统：图像+语言 → 高层 VLM → 低层策略 200Hz 控制（π₀.5, Helix, GR00T N1）

趋势：2025-26 人形机器人标配双系统。原因：VLM 推理 ~5Hz，精细操控需要 50-200Hz。

**2. 动作生成方式**

| 方式 | 代表 | 特点 |
|------|------|------|
| 自回归 | RT-2, OpenVLA | 简单，但动作是连续的不是离散的 |
| Diffusion Policy | Octo, RDT-1B | 连续动作轨迹，适合精细操控 |
| Flow-matching | π₀ 系列 | Diffusion 优化版，更快更稳 |
| 混合 | 2026 趋势 | 高层自回归+低层 diffusion/flow |

**3. 预训练策略**

| 策略 | 做法 | 代表 |
|------|------|------|
| 从 VLM 迁移 | 拿现成 VLM 加动作头 | OpenVLA, ReconVLA（主流）|
| 从头训练 | 专门架构 | RT-1 |
| VLM 冻结+适配器 | VLM 不动，只训适配层 | SmolVLA |

**4. 数据来源**

- 纯真实：贵、慢、稀缺
- 纯仿真：便宜但 sim-to-real gap
- 混合：当前最佳实践

核心瓶颈：机器人数据比语言数据少几个数量级。

**5. 大模型 vs 小模型**

- RT-2 走过 55B 弯路 → OpenVLA 7B 证明小模型够用 → SmolVLA 450M MacBook 能跑
- 趋势：越来越小，越来越快

### 当前主要挑战

1. **数据瓶颈**：机器人数据远不够，Open X-Embodiment 在努力
2. **零样本泛化鸿沟**：仿真 SOTA ≠ 真实世界
3. **推理效率**：VLM ~5Hz vs 操控需要 50-200Hz → 逼出双系统
4. **长时序任务**：串 5+ 任务成功率急剧下降
5. **安全性**：物理世界出错 = 砸东西

### 2026 趋势

1. 双系统成为人形标准
2. RL 微调 VLA（OpenVLA-OFT）
3. 边缘部署（SmolVLA 450M）
4. 离散扩散 VLA（ICLR 2026 新方向）
5. 跨具身泛化（一个模型控制不同形态机器人）

---

## 四、中国机器人产业调研（2025-2026）

> 调研时间：2026年4月 | 数据截至2026年Q1

### 1. 宇树科技 (Unitree Robotics)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2016年 |
| **创始人** | 王兴兴（1991年生，浙江宁波人，浙江理工大学本科） |
| **融资情况** | 完成10轮融资，投资方含美团、腾讯、阿里、红杉中国、经纬创投、深创投等；2025年6月C轮估值127亿元；2026年3月IPO申报获受理（上交所科创板），拟募资42.02亿元，初始发行市值约420亿元 |
| **主要产品** | 四足机器人（Go系列）、人形机器人（G1/H1）、机械臂 |
| **技术路线** | 全电驱技术 + 国产化供应链（国产化率超90%）；强运动控制（H1跑速超5m/s，全球首例全尺寸电驱人形后空翻/侧空翻）；端到端控制正在推进 |
| **最新进展** | 2025年人形机器人出货量超5,500台（全球第一）；2025年营收约17.08亿元（同比+335%），连续2年盈利；2026年春晚合作伙伴；G1售价$16,000（3.99万元起）；IPO进行中 |
| **核心团队** | 创始人王兴兴为浙江理工大学本科，曾获多项机器人竞赛奖；团队以硬件和运动控制见长 |

### 2. 智元机器人 (AgiBot)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2023年2月 |
| **创始人** | 彭志辉（稚晖君），1993年生，前华为"天才少年"，华为昇腾AI芯片研究员 |
| **融资情况** | 成立不到3年融资数十亿；2025年3月B轮由腾讯领投估值150亿元；5月京东加入B+轮；7月LG电子、正大集团战略投资；计划2026年Q3前完成港股/科创板上市 |
| **主要产品** | 远征A系列人形机器人（A2/A3）、G系列灵巧操作平台 |
| **技术路线** | 全栈自研：启元大模型GO-1（全球首个通用具身基座大模型，2025年3月发布）；AgiBot Digital World仿真框架；Genie Studio一站式开发平台；AGIBOT WORLD开源数据集（百万级真机/仿真数据） |
| **最新进展** | 2025年出货超5,100台（Omdia报告全球市占率39%，排名第一）；2026年3月第10,000台远征A3下线；2026年2月进入德国市场；主办ICRA 2026 AGIBOT WORLD挑战赛 |
| **核心团队** | CEO华为前副总裁邓泰华操盘运营；CTO稚晖君负责技术；团队融合华为AI+互联网背景 |

### 3. 银河通用 (Galbot)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2023年5月 |
| **创始人** | 王鹤（北京大学助理教授，清华本科，斯坦福博士，师从三院院士Leonidas Guibas） |
| **融资情况** | 累计融资超25亿元（居中国具身智能领域首位）；2025年6月宁德时代领投11亿元（当时具身智能单笔最大融资）；2026年初再融资25亿元，估值达$30亿+ |
| **主要产品** | Galbot G1通用机器人 |
| **技术路线** | 具身多模态大模型驱动；自研VLA模型（TrackVLA导航模型、GroceryVLA零售场景模型）；独特数据策略：99%合成数据+1%真实数据训练VLA；全球首台搭载NVIDIA Thor芯片 |
| **最新进展** | 2026年春晚指定"具身大模型机器人"（完成盘核桃、拣碎片、叠衣等任务）；与百达精工签署1,000台部署协议；计划年内开设100家零售合作门店 |
| **核心团队** | 创始人王鹤为计算机视觉/3D理解领域顶级学者，北大教授；核心团队以学术背景为主 |

### 4. 优必选 (UBTech)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2012年 |
| **创始人** | 周剑 |
| **融资情况** | 已于港交所上市（2023年12月），累计融资超数十亿美元 |
| **主要产品** | Walker S2工业人形机器人、消费级教育机器人 |
| **技术路线** | 全栈人形机器人技术（运控+感知+决策）；与华为合作接入CloudRobo平台和盘古具身智能大模型；模块化与模型化并行 |
| **最新进展** | Walker S2量产交付中，2025年累计订单超8亿元，交付超500台；2026年产能目标5,000台/年，2027年目标10,000台；与空客签署Walker S2飞机装配协议；第1,000台工业人形机器人已下线 |
| **核心团队** | 深耕人形机器人超10年，团队规模最大的中国人形机器人公司之一 |

### 5. 傅利叶智能 (Fourier Intelligence)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2015年 |
| **创始人** | 顾捷（上海张江） |
| **融资情况** | 2025年完成E轮融资；2025年7月改为股份公司（筹备IPO） |
| **主要产品** | GR-3人形机器人（165cm/71kg/55自由度）、Fourier N1开源人形机器人、FDH-6灵巧手、康复机器人系列 |
| **技术路线** | "双轮驱动"——医疗康复+具身智能；开源策略（N1为首款开源人形机器人）；"1+3+X"应用生态布局 |
| **最新进展** | 康复机器人已进入40+国家3,000+医院，服务超100万患者；CES 2026展出Care-bot GR-3；2025年4月发布开源N1（38kg/23自由度/2小时续航） |
| **核心团队** | 医疗康复机器人出身，具有深厚的医工交叉背景 |

### 6. 星动纪元 (RobotEra)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2023年8月 |
| **创始人** | 陈建宇（清华大学交叉信息研究院助理教授，UC Berkeley博士，师从Masayoshi Tomizuka院士） |
| **融资情况** | 2025年7月A轮近5亿元；2025年11月A+轮近10亿元（吉利资本领投、北汽产投战略投资）；清华大学唯一持股的机器人企业 |
| **主要产品** | 星动L7（全尺寸双足人形）、星动Q5（轮式服务机器人）、XHAND 1/1 Lite灵巧手 |
| **技术路线** | ERA-42端到端原生机器人大模型（全球首次融合世界模型的VLA）；"具身大脑+本体+灵巧手"一体化；语音命令端到端控制全身灵巧操作 |
| **最新进展** | 累计订单突破5亿元；海外业务占比50%（北美/欧洲/中东/日韩）；全球市值TOP10科技公司中9家为客户；CES 2026展出完整产品矩阵 |
| **核心团队** | 清华交叉信息研究院孵化，学术能力突出（AI+控制融合） |

### 7. 小鹏机器人 (Xpeng Robotics)

| 项目 | 详情 |
|------|------|
| **成立时间** | 小鹏汽车内部项目，2024-2025年独立运营 |
| **创始人** | 何小鹏（小鹏汽车CEO） |
| **融资情况** | 依托小鹏汽车（NYSE: XPEV）资金支持 |
| **主要产品** | IRON人形机器人（82自由度，骨骼-肌肉-皮肤仿生设计） |
| **技术路线** | 第二代VLA大模型（720亿参数基座模型，视觉到动作端到端直接生成）；3颗图灵AI芯片；VLT+VLA+VLM高阶大小脑融合；固态电池（行业首创）；自动驾驶技术平移至机器人 |
| **最新进展** | 2025年11月科技日发布全新IRON；计划2026年底实现高阶人形机器人规模量产；与宝钢集团合作（工厂巡检）；引发"真人扮演"争议后持续技术验证 |
| **核心团队** | 四位关键人物带队，融合自动驾驶+机器人双领域经验 |

### 8. 小米机器人 (Xiaomi Robotics)

| 项目 | 详情 |
|------|------|
| **成立时间** | 北京小米机器人技术有限公司（小米集团子公司） |
| **创始人** | 雷军（小米集团CEO） |
| **融资情况** | 小米集团内部投入，不独立融资 |
| **主要产品** | CyberOne（铁大）人形机器人，已迭代至第三代 |
| **技术路线** | 仿生手技术突破（体积压缩60%，64%自由度提升，8200mm2触觉传感覆盖）；汽车工厂场景优先落地；累计240+机器人相关专利 |
| **最新进展** | 第三代CyberOne完成著作权登记；在北京亦庄汽车工厂产线分阶段试用（1-2个工位装配/质检）；持续招聘具身智能人才 |
| **核心团队** | 小米集团总裁卢伟冰统筹；专项团队规模持续扩张 |

### 9. 乐聚机器人 (LeJu Robotics)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2016年 |
| **创始人** | 冷晓琨、常琳、安子威（均为哈尔滨工业大学博士） |
| **融资情况** | 2025年10月完成近15亿元Pre-IPO轮融资；已启动科创板上市辅导（东方证券），预计2026年H1完成 |
| **主要产品** | KUAVO系列人形机器人（含4Pro讲解机器人）、夸父系列 |
| **技术路线** | Model-Based与RL算法融合的"小脑"运控系统；与华为云合作"盘古具身智能大模型+夸父人形机器人"技术路线；与腾讯、阿里云、火山引擎等40+生态伙伴合作 |
| **最新进展** | 2025年Q1订单同比+200%，全年预计交付千台级；与华为、中国移动等广泛合作；2025年8月发布KUAVO 4Pro |
| **核心团队** | 哈工大机器人实验室出身，10年+人形机器人研发积累 |

### 10. 逐际动力 (LimX Dynamics)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2022年 |
| **创始人** | 深圳团队 |
| **融资情况** | 已完成多轮融资（具体金额待披露） |
| **主要产品** | Oli全尺寸人形机器人（165cm/31自由度）、TRON 2多形态机器人（4.98万元起） |
| **技术路线** | LimX COSA（Cognitive OS of Agents）——物理世界原生具身Agentic OS，融合高阶认知与全身运动控制；三形态切换设计（TRON 2） |
| **最新进展** | 2026年1月发布LimX COSA系统；2025年12月发布TRON 2；与上汽北京合作共建具身智能实验室；WRC 2025获奖 |
| **核心团队** | 以运动控制和强化学习见长 |

### 11. 加速进化 (Booster Robotics)

| 项目 | 详情 |
|------|------|
| **成立时间** | 清华大学机器人控制实验室孵化 |
| **创始人** | 清华火神机器人足球队核心成员 |
| **融资情况** | 完成A轮和A+轮融资 |
| **主要产品** | Booster T1（成人尺寸人形）、Booster K1（儿童尺寸人形） |
| **技术路线** | 强运动控制（足球竞技级别）；面向开发者的开放平台定位 |
| **最新进展** | 2025年RoboCup世界杯AdultSize组金银牌（T1）、KidSize组金银牌（K1）；世界人形机器人运动会AI足球赛包揽奖牌（2金2银2铜）；K1切入STEM教育市场 |
| **核心团队** | 清华火神机器人足球队出身，20年人形机器人技术积累 |

### 12. 达闼科技 (CloudMinds)

| 项目 | 详情 |
|------|------|
| **成立时间** | 2015年 |
| **创始人** | 黄晓庆（前中国移动研究院院长） |
| **融资情况** | 累计融资6.1亿美元；曾计划赴美IPO但受美国制裁被迫撤回 |
| **主要产品** | 云端智能机器人（迎宾、安保巡逻、清洁、零售、讲解等服务机器人） |
| **技术路线** | "云端大脑"架构——机器人本体为轻量执行端，AI推理在云端完成；注重网络安全 |
| **最新进展** | 2025年1月与天津津南签署具身智能战略合作；700名员工；服务机器人持续商业部署 |
| **核心团队** | 创始人有深厚电信行业背景，侧重云-端架构 |

### 13. 华为 (Huawei) -- 平台赋能者

| 项目 | 详情 |
|------|------|
| **定位** | 不做机器人本体，做具身智能基础设施平台 |
| **核心产品** | CloudRobo具身智能平台（2025年6月发布）；盘古具身智能大模型 |
| **技术能力** | 三大核心模型：具身多模态生成模型、具身规划模型、具身执行模型；80%训练样本可合成生成 |
| **合作伙伴** | 优必选（2025年5月签署全面合作）、乐聚机器人等 |
| **战略意义** | 类似Android之于手机的角色——为机器人企业提供AI底座 |

---

### 技术路线对比

#### VLA vs 传统控制

| 技术路线 | 代表企业 | 特点 |
|----------|----------|------|
| **端到端VLA** | 星动纪元（ERA-42）、小鹏（第二代VLA，720亿参数）、银河通用（TrackVLA/GroceryVLA） | 视觉-语言-动作端到端，泛化能力强，但对算力要求极高 |
| **VLA + 世界模型** | 星动纪元（ERA-42，全球首个融合世界模型的VLA）、智元（GO-1） | 加入预测能力，可对物理世界进行推演 |
| **具身基座大模型** | 智元（GO-1启元大模型）、银河通用 | 训练通用具身基座，再微调到具体场景 |
| **Model-Based + RL** | 乐聚、逐际动力、宇树 | 传统控制理论与强化学习融合，运控能力扎实 |
| **云端大脑** | 达闼（CloudMinds） | 本体轻量化，推理在云端 |
| **平台化大模型** | 华为（盘古具身智能大模型）| 提供通用AI底座，合作伙伴做垂直应用 |

#### 大模型方案

| 方案 | 企业 | 详情 |
|------|------|------|
| **完全自研** | 智元（GO-1）、星动纪元（ERA-42）、银河通用（TrackVLA）、小鹏（第二代VLA） | 自建模型+自建训练基础设施 |
| **接入国产大模型** | 优必选（华为盘古）、乐聚（华为盘古+腾讯+阿里云+火山引擎） | 用华为CloudRobo等平台提供的大模型能力 |
| **开源+自研混合** | 傅利叶（开源N1生态+自研GR-3）、加速进化（面向开发者） | 部分开源促进生态，核心自研 |

#### 硬件自研 vs 采购

| 策略 | 企业 | 详情 |
|------|------|------|
| **高度自研** | 宇树（全电驱，国产化率>90%）、智元、小鹏（固态电池首创）、小米 | 关节电机、驱动器、传感器等核心部件自研 |
| **核心自研+部分采购** | 银河通用（搭载NVIDIA Thor芯片）、星动纪元、傅利叶 | 芯片等用国际供应，本体自研 |
| **平台型采购** | 乐聚、逐际动力 | 更侧重软件和算法，硬件部分外采 |

**核心零部件国产化现状：** 中国人形机器人核心零部件本地化率已超75%，关键突破领域包括高精密减速器、力矩传感器、灵巧手、关节模组等。

#### 仿真平台

| 平台 | 使用企业 |
|------|----------|
| **NVIDIA Isaac Sim / Isaac Lab** | 宇树、傅利叶、银河通用、小米等（主流选择） |
| **MuJoCo** | 多数科研型企业（开源免费） |
| **智元 AgiBot Digital World** | 智元自研，百万级仿真数据生成 |
| **银河通用自研仿真** | 银河通用（99%合成数据训练策略的核心基础设施） |

> 注：目前国产仿真平台尚未出现与Isaac Sim/MuJoCo抗衡的通用替代品，大多数企业仍依赖国际平台，部分头部企业开发了专用仿真工具。

---

### 政策和产业环境

#### 国家政策支持

| 政策/事件 | 时间 | 内容 |
|-----------|------|------|
| **"具身智能"首次写入政府工作报告** | 2025年3月 | 与生物制造、量子技术、6G并列为核心未来产业 |
| **"十五五"规划建议** | 2025年 | 具身智能列为重点未来产业方向 |
| **国家人工智能产业投资基金** | 2025-2026年 | 600亿元国家AI基金，覆盖机器人领域 |
| **国家级创业投资引导基金** | 2026年 | 计划20年内吸引近1万亿元（$1,380亿）资本，聚焦机器人/AI/前沿创新 |
| **工信部"机器人+"行动** | 持续 | 推动供需对接和推广应用 |
| **央地共建创新中心** | 2025-2026年 | 北京、上海建设具身智能/人形机器人创新中心，开发"青龙""天工"开源公版机和"开物"操作系统 |

#### 地方政策

| 城市 | 政策亮点 |
|------|----------|
| **北京** | 《北京具身智能科技创新与产业培育行动计划(2025-2027)》；亦庄"具身智能机器人十条"；2年内释放超万台机器人应用机会；人形机器人整机企业超30家 |
| **深圳** | 《深圳市具身智能机器人技术创新与产业发展行动计划(2025-2027)》；重点支持高精密关节模组、多模态传感器、仿生灵巧手、机器人AI芯片研发 |
| **上海** | 融合国际资本与高端产业场景；智元、傅利叶等龙头企业总部所在地 |
| **雄安新区** | 《机器人产业创新发展三年行动计划(2024-2026)》；目标引进100家机器人企业，核心产业收入达10亿+ |

#### 产业集群分布

- **广东：** 55家人形机器人企业（全国最多），以深圳为核心
- **北京：** 16家，科研资源最强（清华/北大/中科院）
- **江苏：** 15家
- **上海：** 12家，国际化程度最高
- 全国人形机器人相关企业约150-320家（不同口径统计）

#### 与美国的差距分析

| 维度 | 中国优势 | 美国优势 |
|------|----------|----------|
| **量产能力** | 全球唯一大规模量产人形机器人的国家；2025年全球出货1.3-1.8万台中87-90%来自中国 | Tesla Optimus尚未量产（预计2027年+） |
| **价格** | 宇树G1仅$16,000 | Tesla Optimus估计$30,000+ |
| **专利** | 过去5年中国企业申请7,705件人形机器人专利 | 美国企业仅1,561件 |
| **企业数量** | 320+家具身智能企业，150+家人形机器人公司 | 50+家主要企业 |
| **AI基础模型** | 追赶中，VLA模型能力逐步接近 | Physical Intelligence pi0.5等仍属前沿，但Spirit v1.5已超越 |
| **软件/AI** | 大模型能力快速提升 | **仍有优势**：OpenAI/Google等基础大模型生态更成熟 |
| **芯片** | 依赖NVIDIA（Thor等），国产芯片在追赶 | **明显优势**：NVIDIA垄断机器人AI芯片 |
| **供应链** | **明显优势**：硬件制造和供应链成本远低于美国 | 高端传感器/精密器件有优势 |

> **关键判断：** 中国在人形机器人的量产能力、性价比、供应链上已全面领先；但在基础AI模型（尤其是通用大模型底座）和高端AI芯片上仍依赖或追赶美国。2026年a16z发文警告"美国不能输掉机器人竞赛"，反映了美国对中国领先地位的焦虑。

---

### 值得关注的技术亮点

#### 星动纪元 ERA-42：全球首个融合世界模型的VLA
- 端到端原生机器人大模型，集视觉、理解、预测、行动为一体
- 同一个VLA模型控制全身灵巧操作（含五指灵巧手）
- 语音命令即可完成上百种复杂操作

#### 银河通用的"99%合成数据"策略
- 训练VLA大模型时使用99%合成数据+1%真实数据
- 大幅降低真机遥操作数据采集成本
- 如果策略可规模化验证，将彻底改变机器人训练范式

#### 宇树G1的极致性价比
- 3.99万元人民币（约$16,000）的全尺寸人形机器人
- 全电驱+国产化率>90%的供应链控制
- 2025年已实现盈利，复制了"中国制造"在消费电子的成本优势

#### 智元AGIBOT WORLD全栈开源生态
- 开源百万级真机/仿真数据集
- GO-1通用具身基座大模型
- 仿真平台AgiBot Digital World
- 目标是成为机器人领域的"Android"

#### 小鹏第二代VLA的跨域迁移
- 720亿参数基座模型同时服务自动驾驶、Robotaxi、飞行汽车和人形机器人
- "去掉语言转译环节，视觉信号到动作指令端到端直接生成"
- 汽车产业资金和数据优势向机器人迁移

#### Spirit AI v1.5基准测试超越美国
- 中国AI创业公司Spirit AI的具身智能基座模型
- 在RoboChallenge真实世界机器人基准测试中以66.09总分、50.33%任务成功率排名全球第一
- 超越了美国Physical Intelligence的pi0.5模型

#### 中国机器人工厂的"30分钟一台"产能
- 中国已建成每30分钟下线一台人形机器人的工厂
- 年产能1万台级别的专用产线已投产
- 量产经验是目前其他国家完全不具备的

#### 华为CloudRobo平台的产业角色
- 不做本体，做"机器人的Android"
- 80%训练样本可合成生成
- 三大核心模型覆盖感知-规划-执行全链路
- 如果生态成功，将极大降低机器人企业的AI门槛

---

### 市场规模关键数据汇总

| 指标 | 数据 |
|------|------|
| 2025年全球人形机器人出货量 | 1.3-1.8万台 |
| 中国占全球出货比例 | 87-90% |
| 2025年中国具身智能市场规模 | 约915亿元 |
| 2026年预计突破 | 1万亿元 |
| 2030年预计全球市场 | $770亿（IDC预测） |
| 2025年行业融资总额 | 约380亿元 |
| 国内具身智能企业总数 | 230+家 |
| 国内人形机器人企业数 | 140-150家 |
| 2025年发布人形机器人产品数 | 330+款 |
| 中国人形机器人专利占全球比例 | 68% |
| 核心零部件国产化率 | >75% |
| 宇树IPO拟募资 | 42.02亿元 |
| 银河通用累计融资 | 居具身智能领域首位（>25亿元） |

---

### Sources

- [2026中国人形机器人产业全景 - IT之家](https://www.ithome.com/0/925/903.htm)
- [2026中国人形机器人企业潜力排行 - 新浪财经](https://finance.sina.com.cn/roll/2026-02-10/doc-inhmihnp5626088.shtml)
- [IDC 2026年中国机器人与具身智能市场十大趋势](https://www.idc.com/resource-center/blog/%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%AD%A3%E5%9C%A8%E8%BF%9B%E5%8C%96-2026%E5%B9%B4%E4%B8%AD%E5%9B%BD%E6%9C%BA%E5%99%A8%E4%BA%BA%E4%B8%8E%E5%85%B7%E8%BA%AB%E6%99%BA/)
- [Unitree $610M Shanghai IPO - Rest of World](https://restofworld.org/2026/unitree-china-humanoid-robot-shanghai-ipo/)
- [Why China's humanoid robot industry is winning - TechCrunch](https://techcrunch.com/2026/02/28/why-chinas-humanoid-robot-industry-is-winning-the-early-market/)
- [China leads the humanoid robot race - Rest of World](https://restofworld.org/2026/china-tesla-robot-race/)
- [Embodied AI: China's Big Bet - Carnegie Endowment](https://carnegieendowment.org/research/2025/11/embodied-ai-china-smart-robots)
- [America Cannot Lose the Robotics Race - a16z](https://a16z.com/america-cannot-lose-the-robotics-race/)
- [智元机器人Omdia全球出货量第一](https://www.zhiyuan-robot.com/article/188/detail/121.html)
- [银河通用再融资25亿元](https://tech.cnr.cn/techgd/20260302/t20260302_527540956.shtml)
- [星动纪元A+轮近10亿元融资](https://www.qbitai.com/2025/11/354404.html)
- [宇树科技估值9年暴增1270倍](https://finance.sina.com.cn/wm/2026-03-22/doc-inhrwckv4922220.shtml)
- [稚晖君冲进科创板](https://www.qbitai.com/2025/07/306495.html)
- [银河通用创始人王鹤对话](https://www.bianews.com/news/details?id=214653)
- [星动纪元ERA-42端到端模型](https://blog.csdn.net/2401_89760565/article/details/145574456)
- [小鹏科技日 IRON人形机器人](http://www.news.cn/auto/20251106/a49e7e6cdd38419ab363605c1c528ae4/c.html)
- [UBTech Walker S2 Mass Production](https://www.prnewswire.com/news-releases/ubtech-humanoid-robot-walker-s2-begins-mass-production-and-delivery-with-orders-exceeding-800-million-yuan-302616924.html)
- [傅利叶开源N1人形机器人](https://www.fftai.cn/open-source/18)
- [逐际动力LimX COSA系统发布](https://finance.sina.com.cn/tech/roll/2026-01-12/doc-inhfziuv6945504.shtml)
- [乐聚机器人Pre-IPO轮融资近15亿](https://www.stcn.com/article/detail/3395865.html)
- [华为云CloudRobo具身智能平台发布](https://www.guancha.cn/economy/2025_06_20_780175.shtml)
- [Chinese AI startup tops global benchmark - People's Daily](https://en.people.cn/n3/2026/0114/c90000-20413808.html)
- [China to Invest 1 Trillion Yuan in Robotics - IFR](https://ifr.org/news/china-to-invest-1-trillion-yuan-in-robotics-and-high-tech-industries/)
- [深圳市具身智能机器人行动计划(2025-2027)](https://stic.sz.gov.cn/xxgk/tzgg/content/post_12052515.html)
- [北京具身智能行动计划(2025-2027)](https://www.beijing.gov.cn/zhengce/zhengcefagui/202503/t20250304_4024579.html)
- [中国信通院具身智能发展报告(2025)](https://www.caict.ac.cn/kxyj/qwfb/bps/202601/P020260130541978285206.pdf)
