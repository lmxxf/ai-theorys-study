# 322 RL 素材补充（09-28，只补 素材.md 没覆盖的；行号 = report.txt）

## 0. 先说"没有"的东西（写之前别编）
- **报告没有 KL 项**：式 1（436–446）只有 IS 比 r、mask M、优势 A，全文 grep 无 "KL"。
- **没有独立"局限"章节**，全文无 "limitation / future work"。唯一的自承是 5.5 失败记录 + 5.3 "token 随分数一起涨"那句（见 §7）。
- **没给 SFT 数据量、mid-training token 数、MOPD2 数据量/步数**。SFT 只一句 "Following a short supervised fine-tuning (SFT) stage"（421）。
- **Table 3 没标口径**（avg@k / pass@1 都没写），⚠️待核。且 Table 3 的 DeepSWE（Pro 71.9 / Flash 67.9）≠ 图 3 RL 结束值（72.6 / 65.7，avg@3）——推测 Table 3 是 MOPD2 之后的最终模型（1469 "The final evaluation results ... are reported in Table 3"），报告没解释差异 ⚠️。

## 1. 训练流程顺序
1. 预训练：Flash 48T（文本 26T + omni 22T），Pro 30T（27T + 3T）；32K→256K；AdamW（350–353）
2. mid-training：agent 中心数据混合（coding/general/visual/research 轨迹 + 文本/仓库级代码/图像视频音频）；**256K 占大部分算力，最后一段扩到 1M**（363–365）；换 Muown；MXFP4 QAT；混入"反思作弊→改正"样本（773–778）
3. 短 SFT（421）——量未给
4. mixed RL（一次跑完，"You Only RL Once"，§5 标题 1142）：Flash/Pro 各 30 步（图 12，1393）
5. MOPD2 蒸馏（1410 "After mixed RL, we use ... MOPD2"）
- 耗时：**Pro 30 步 123.1 h，Flash 81.8 h**（图 12，1383/1387）

## 2. RL 算法本体（§4.1 式 1 + §5.1）
- 算法：**GRPO + 异步 partial rollout，staleness 4**（1151 "Training uses GRPO with asynchronous partial rollouts at a staleness of 4."）
- 聚合：**prompt-mean**（先在 prompt 内平均再平均），理由 "this prevents response length from growing too quickly during RL"（1164–1166）。对照：GAR 消融实验用的是 token-mean（1022）
- IS 比：逐 token `r = sg[π_θ/μ_θold]`，训练概率来自训练框架当前模型，推理概率来自 rollout 时；**partial rollout 不重算推理概率**（1166–1170）——梯度走 log π，r 停梯度当权重（REINFORCE+IS 形式，不是 PPO min 形式；这句是我读式 1 的判断 ⚠️）
- clip：**四个解耦边界**，正优势 [ε+^l, ε+^h]、负优势 [ε−^l, ε−^h]，超界 token 直接 mask；初值都是 **[0.2, 5.0]**，按熵在线调："when entropy is too low, we widen the positive bounds and narrow the negative ones ... and we do the opposite when entropy is too high"（1170–1181）
- dynamic sampler 过滤全对/全错组（471–472）；优势全 0 的组默认丢（1647–1648）
- 优化器：Muown，lr **3×10⁻⁶**，无 weight decay、无 warmup，grad clip 1.0；Muon 动量 0.95 Nesterov、NS 迭代 10 次、更新缩放 0.5；Adam 部分 β1=β2=0.95、ε=1e-8（1153–1159）；从 SFT 继承 FP32 master 权重和 Muown row state（1159–1161）；**冻结路由器**（1161）
- 每步 1568 prompt × G=16 = 25K 条，2.7–3.7B token（426–429）
- 训推一致：专家每次更新后按 MXFP4 Humming GEMM 约束做 QDQ，两边看到相同专家权重（1828–1830）；R3 + top-p 候选集 bitmap（1830–1841）

## 3. 长度控制与行为惩罚（§4.3.3，素材.md 没写）
- **组相对长度惩罚**（式 4，1085–1105）：组通过率 > A 时，取通过解长度的 B 分位作参考长度 ℓ*，只对**通过的**解扣分，扣分 = X·clip(((ℓ/ℓ* −1−δ)/(s−δ))^γ,0,1)。原话 "encourages concise successful solutions using a reference adapted to each prompt, while the pass-rate gate preserves room for exploration on difficult prompts"。**超参 A/B/X/δ/s/γ 具体值没给** ⚠️
- **段级行为惩罚**（式 5，1107–1139）：格式错、工具名错、参数坏的 token 标记 h=1；正轨迹里标记 token 优势清零、挪给未标记 token（α 放大）；负轨迹里标记 token 罚 κ>1 倍、未标记 token 减罚（β）；"Each sign's total advantage mass is conserved ... limiting excess negative pressure that can drive uncontrolled entropy growth"
- Penalty Module（§6.1，1547–1560）：Rule（检测）与 Strategy（mask / 优势整形 / monitor / early stop）分离；可抓"infrastructure failures not attributable to the model, garbled token patterns, calls to unavailable tools, and repetition"——基础设施故障不算模型的错

## 4. 环境构造 / 验证 / 判分器
**代码（§4.2.1）**：五条合成路径——GitHub PR+issue（无 issue 时 LLM 从补丁重建描述，"explicitly instructed to omit implementation-specific details"，537–539）、**公司员工真实开发请求**（含 vibe coding，539–543）、规格驱动、CodeMidas 从现有代码库功能反推任务（546–550）、长程任务迭代扩展；外加公开数据集和**付费数据供应商**（553–555）
- 验证：每题 coding agent 跑 **4 次**，audit agent 看全部 4 条判"通过但错=假阳性 / 失败但对=假阴性"（567–578）；F2P/P2P 前后检查，**8 次重跑**结果必须稳定（580–584）

**通用（§4.2.2）**：真实文件 + 多 agent 自动造本地 software mock（MCP/API/CLI/GUI），全部本地可重置（605–614）；planning agent 规划 + 联网搜索落地 + 多 agent 并行生成 + review agent 查一致性（651–660）；rubric 为原子二值项，代码检查 + LLM 检查，多次/多模型判分一致性找歧义（664–668）；用不同能力模型 rollout 校 rubric 松紧，加负向检查和对抗解（669–674）
- **判分器："During RL, a self-hosted MiMo-V2.6-SFT model serves as the grader to support stable scoring."（675–676）** 规模："thousands of environments"（676）
- GAR 判分器："An SFT-trained agentic grader"（990–991），能看仓库、跑定向测试（996–997）；输出不可用时回退原优势（1019）；异步运行、结果可滞后（1643–1645）

**视觉（§4.2.3）**：网站/交互应用/游戏/3D/幻灯片/SVG/视频/Figma（685–687）；开放设计 = 先 pointwise rubric 稳定后再加 groupwise 组内比较（692–697）；高保真复刻 = 像素级相似度为主 + LLM 整体评判（700–702）

**网安（§4.2.4）补充**：给 agent **编译好的 harness 二进制**，CyberGym 不给（727–734）；Table 3 注 1："We corrected the flawed evaluation environments based on the method described in Section 4.2.4."（1210）⚠️ 即 CyberGym 分数是在他们修过的环境上测的，写时要注明

**harness 为何不用生产壳（744–750 原话要点）**："production harnesses ... steer the model with numerous constraint and instruction prompts. These extras fall outside the task-completion reward signal, making credit assignment unreliable"

## 5. MOPD2（§5.6，1409–1471）
- 全名 Multi-Prefix Multi-Teacher On-Policy Distillation
- 老师两类：可验证任务用 **mixRL 老师**；难验证开放任务用 **SFT 老师**（合成高质量示范训的）（1444–1446, 1460–1461）
- 和 V2-Flash MOPD 的差别："we retain autonomous student rollouts in domains with suitable mixRL teachers (Standard MOPD) and add prefix-conditioned single-turn rollouts"（1412–1453）
  - 新增 **Prefix-Conditioned OPD**：一条有 k 个 assistant 回合的轨迹切出 k 个历史前缀，学生从每个前缀只生成**一个新回合**，不重演前面交互；老师按同样历史给 token 级监督（1455–1459）
  - 前缀来源：老师 rollout（Teacher-Prefix）或 SFT 数据（SFT-Prefix）；"SFT data provide prefix contexts rather than fixed continuation targets"（1450）
  - 为何要 SFT-Prefix：SFT 老师对"学生多次偏离后到达的历史"覆盖有限，固定示范前缀限制偏离（1461–1465）
- 用途：扩到 "long-horizon game development, scientific research, and embodied intelligence"（1466–1468）
- ⚠️ 蒸馏在 RL 之后，Table 3 分数混了 MOPD2 贡献，不能全记在 RL 头上

## 6. 分数
### 6.1 RL 过程（Flash/Pro 分开）
- DeepSWE v1.1 avg@3：Pro 58.4→72.6，Flash 48.7→65.7（431–432，已在素材.md）
- 图 1 / 图 9 其他曲线只有坐标轴、pdftotext 抽不出端点值 ⚠️（DeepSWE / SWE-Bench Pro / MiMo Code Bench / AutomationBench / Visual Coding / Cyber Bench，y 轴范围约 48–72、54–63、52–60、45–54、66–75、66–78）
- 图 9 总 token 随训练涨（DeepSWE 约 160–240K 区间）："These gains generally accompany increasing total token counts"（1285–1286）
- GAR 消融（Flash、仅代码、batch 128、token-mean）：有 GAR 通过率涨到 step 52（1065）——具体数值只在图上 ⚠️
- 多 harness（Pro/Flash 未明说哪个 ⚠️）：留出 harness 均值 pass@1 ≈50%→66%（1324）
- 路由冻结后：CV≈0.7、峰值≈5.5×、冷专家≈1%（1340）

### 6.2 Table 3 最终对比（1477–1503，口径未标 ⚠️；基线开 max effort，1278–1279）
| 基准 | V2.6-Pro | V2.6-Flash | V2.5-Pro | Opus 5 | GPT-5.6 Sol | Fable 5 |
|---|---|---|---|---|---|---|
| DeepSWE v1.1 | 71.9 | 67.9 | 19.0 | 74.0 | 73.0 | 70.0 |
| ProgramBench | 26.5 | 26.0 | 12.5 | 37.0 | 25.0 | 33.0 |
| MiMo Code Bench | 63.2 | 61.2 | 40.4 | 68.6 | 59.3 | - |
| AutomationBench v1.0.6 | 53.1 | 52.3 | 16.0 | 50.3 | 45.8 | 46.2 |
| Toolathlon-Verified | 76.9 | 73.6 | 49.1 | 80.6 | 74.9 | 77.9 |
| GDPval-AA 2.1 | 1673 | - | 1107 | 1708 | 1588 | 1595 |
| Agents' Last Exam | 31.6 | 27.6 | 13.2 | 31.6 | 30.8 | 25.7 |
| Terminal Bench 4.0 | 34.9 | 28.8 | 1.5 | 49.0 | 39.9 | 42.4 |
| Terminal Bench 2.1 | 89.9 | 87.6 | 65.2 | 89.1 | 88.8 | 84.3 |
| OSWorld-Verified | 82.0 | 80.8 | - | 83.4 | 83.0 | 86.0 |
| JobBench | 62.0 | 61.2 | 25.0 | 65.7 | 45.4 | 57.4 |
| CyberGym | 94.0 | 95.1 | 40.0 | - | - | - |
| MiMo Cyber Bench | 80.2 | 77.2 | 0.0 | - | - | - |
| ExploitGym | 17.8 | 6.0 | 0.2 | 22.1 | 30.3 | 28.4 |
| ExploitBench | 47.9 | 25.3 | 16.6 | 70.0 | 78.5 | 78.0 |
| SEC Bench Pro | 66.3 | 47.5 | 17.7 | - | 79.1 | - |
| MiMo Visual Coding | 72.3 | 71.5 | - | 70.0 | 73.4 | 69.1 |

读法（Pro 对三家基线）：
- **Pro 全场第一**：AutomationBench 53.1、Terminal Bench 2.1 89.9；Agents' Last Exam 31.6 与 Opus 5 并列
- **领先 Sol/Fable 但输 Opus 5**：MiMo Code Bench、GDPval、JobBench；DeepSWE 输三家中的两家（Opus 74.0 / Sol 73.0），赢 Fable 70.0
- **明显落后**：Terminal Bench 4.0（34.9 vs 39.9–49.0）、ProgramBench（26.5 vs Opus 37.0 / Fable 33.0，只赢 Sol）、ExploitGym、ExploitBench、SEC Bench Pro、OSWorld（全三家最低）
- CyberGym / MiMo Cyber Bench 无基线；CyberGym 是自修环境版（1210）
- Visual Coding：Pro 72.3 输 Sol 73.4，赢 Opus/Fable
- **Flash 贴 Pro**：大部分项差 1–4 分，CyberGym 还反超（95.1）；差距大的全在网安利用类（ExploitGym 6.0、ExploitBench 25.3、SEC Bench Pro 47.5）和 Terminal Bench 4.0（28.8）
- V2.5-Pro→V2.6-Pro 跳幅：DeepSWE 19.0→71.9，Terminal Bench 4.0 1.5→34.9，MiMo Cyber Bench 0.0→80.2
- 报告自评："achieving performance comparable to that of frontier models across various domains"（1470–1471）

### 6.3 9B 蒸馏 + 开源环境 RL（Table 6，1969–1990；口径在表内）
| 基准 | 口径 | Qwen3.5-9B | SFT | RL |
|---|---|---|---|---|
| SWE-bench Verified | avg@3 | 60.0 | 61.1 | 66.2 |
| SWE-bench Pro | avg@3 | 32.0 | 44.6 | 47.6 |
| MiMo Code Bench (mini) | avg@3 | 19.5 | 51.6 | 59.9 |
| MiMo Cyber Bench (mini) | avg@3 | 5.7 | 31.3 | 47.0 |
| AutomationBench v1.0.6 | avg@1 | 5.0 | 30.3 | 33.1 |
| Terminal Bench 2.1 | avg@1 | 27.0 | 37.1 | 52.8 |
| Toolathlon-Verified | avg@1 | 25.9 | 35.2 | 38.0 |
| OfficeQA Pro | avg@1 | 9.0 | 19.5 | 24.8 |
| JobBench | avg@1 | 2.6 | 18.3 | 25.2 |
| MiMo General Bench (mini) | avg@1 | 28.5 | 62.2 | 70.6 |
| MiMo Visual Coding (mini) | avg@1 | 61.7 | 64.0 | 72.4 |
- RL 是**按领域分开跑的 GRPO**，代码单 harness（1951–1953, 1989–1990）；音乐内部基准 45.7→52.5（1964–1965）
- SFT 数据（Table 4，1919–1925）：总 77.4B token / loss token 27.2B；Code 23.2B(29.9%)/7.3B，Cyber 11.0B(14.2%)/4.8B，General 22.0B(28.5%)/5.7B，Visual 21.2B(27.4%)/9.4B
- 开源环境（Table 5）：Code 3k 可执行测试 / Cyber 1k 规则检查 / General 1k rubric 判分 / Visual 2k 视觉判分；另约 1k 音乐（1930–1949）
- Table 7 多 harness（2020–2036）Mean 列：SWE-bench Verified 53.1→62.3→65.7；SWE-bench Pro 27.5→44.4→46.5；MiMo Code Bench (mini) 15.3→53.1→59.0；"improves on Qwen3.5-9B in all 21 dataset–harness pairs, and multi-harness RL further improves every pair"（2002–2003）；Code Bench mini 各 harness 增益 1.8–9.3 点（2004–2005）
- 可写一句：**9B 上 SFT 贡献普遍大于 RL**（如 Code Bench mini +32.1 vs +8.3；General mini +33.7 vs +8.4），报告自己也说 "highlight the value of high-quality training data and a strong SFT initialization"（1993–1994）——例外 Terminal Bench 2.1（SFT +10.1，RL +15.7）、Cyber mini（+25.6 / +15.7）、Visual mini（+2.3 / +8.4）

## 7. 自我归因原话（"为什么分数高"）
- 摘要："We scale RL compute along three dimensions ..."（11–16）
- "Scaling RL compute substantially unlocks model potential on both verifiable tasks such as coding and less verifiable tasks such as web development."（155–156）
- "achieving a fundamental leap in model capabilities"（423）
- mid-training："expands the exploration space for agentic tasks, enabling the model to discover more effective task-solving trajectories during post-training"（132–133）
- harness："Varying both the harness and the task improves the model generalization."（139–140）
- 结论："emphasizing the joint importance of broad exploration, informative feedback, and scalable training systems"（2067–2068）
- ⚠️ 注意：没有做"去掉某维度"的整体消融；只有 GAR（图 8）、路由冻结（图 11）、多 harness（图 10）三个局部对照

## 8. 报告自承的问题 / 失败（无正式局限节）
- 分数涨伴随 token 涨："stronger task performance develops alongside greater token usage"（1285–1286）——和摘要说 grading "steers the model towards shorter, more token-efficient solutions"（15–16）放一起看有张力（我的观察）
- 早期模型有 reward hacking 倾向："In early experiments, we observed a tendency toward reward hacking in MiMo."（773–774）；hack agent 找到 "many exploit paths that we had not observed during training"（834–835）
- 失败清单（1344–1407）：GPU 显存 DBE；Flash 网安集群 K8s 故障（step 15–16）；Pro 判分器断网（step 14 后）；partial rollout 下短 rollout 先完成导致长度估计偏差、KV 池耗尽；某 harness rollout 长度不到其他一半；EP rank 30× 负载 OOM；Flash 后期 packing 撑爆 host 内存 CPU OOM
- 冷启动：启动收集时间约为稳态 1.8×（1803–1804）
- 25 个数据源生成 token 差 90×、rollout 时长差 66×（1675–1677）
- SFT 老师覆盖有限（1461–1462）；网安利用类在 Table 3 大幅落后但正文没评论

## 9. 往期 Agentic RL 各期 × 小米报告：接上了什么、新在哪（09-28 朱雀整理）
- **282 骨架没动全靠后训练** → 小米是又一个样本，但**它是唯一给出分阶段数字的**：DeepSWE V2.5-Pro 19.0 → RL 前 58.4 → RL 后 72.6 → 最终 71.9（口径不一 ⚠️）。大头在 mid-training + SFT，不在 RL
- **282 GLM 判分器三道检查** ↔ 小米代码环境验证：跑 4 次 + audit agent 判假阳/假阴 + F2P/P2P 8 次重跑必须稳定（567–584）。同一类思路
- **283 题是谁出的** ↔ 小米五条造题路，新增两条 K3 没有的：**公司员工真实开发请求**（539–543）、**付费数据供应商**（553–555）；CodeMidas 从现有代码反推任务。283 那句“人往上挪了一层”的活样本
- **284/286 分头练 vs 混合练** → ⭐ **286 引的 MOPD 论文（Mix-RL 0.882 vs MOPD 0.937）就是北大+小米的**；到自家旗舰，小米反而先做“You Only RL Once”混合 RL，蒸馏挪到后面当 MOPD2。不是否定蒸馏，是换了位置
- **285 老师不是越强越好**（285 的主角就是小米那篇 MOPD，235B 外部老师教崩）→ MOPD2 的老师全是自家的：可验证任务用 mixRL 老师、开放任务用 SFT 老师；新增前缀条件单回合蒸馏，专门处理学生偏离后的历史。跟 285“老师得离学生近”一致
- **287 最贵的动作是等** ↔ 失败清单：partial rollout 里短的先跑完导致长度估计偏差（**跟 V4 WAL 那段论证的是同一个问题**）、KV 池耗尽、冷启动约为稳态 1.8×、25 个数据源 token 差 90×、时长差 66×；staleness 4（K3 是“极端 off-policy”）
- **291 真活怎么变训练题** ↔ 为什么不用生产壳训：生产壳里的约束提示词在奖励信号之外，功劳分不清（744–750）——新论点
- **311 判分器要多聪明** ↔ 判分器就是自家的 MiMo-V2.6-SFT（675–676），GAR 判分器也是 SFT 训的 agent；判分占 12.7% 预算
- **315 熵坍缩** ↔ ⭐ 新问题是**熵往上失控**：clip 四个边界双向调（熵低放宽正边界，熵高反过来）；段级惩罚、GAR 都写明“正负优势总量守恒，防熵增长失控”。全文没有 KL（Ring-Zero 用了 KL 1e-4）
- **317 π* ∝ π₀·exp(r/β)，采不到的推不上去** ↔ mid-training 的自述“expands the exploration space for agentic tasks, enabling the model to discover more effective task-solving trajectories during post-training”（132–133）——先在 mid-training 把 π₀ 抬起来，RL 才采得到。**正好解释 ① 里为什么 40 分涨在 RL 之前**；mid-training 里还混了“反思作弊→改正”样本，也是往 π₀ 里种东西
- **246 MoE 路由** ↔ 路由冻结（往期没有任何一家写过）
- **完全新的**：GAR（过了测试之后比好坏）、hack agent 循环、路由冻结、熵双向控制、前缀条件蒸馏、员工真实请求造题；以及一处自相矛盾——摘要说判分让解法“更短更省 token”，正文说“分数涨伴随 token 涨”
