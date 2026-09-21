# Jin Yanyan (靳岩岩)

**AI Agent Systems · On-Device AI · GPU Kernel & Inference Optimization**

19 years in software engineering — embedded (MTK/Nokia), iOS (Meituan L8 / Tantan T7), CI/CD, and OS-level development. The last three years have been AI: building an AI agent subsystem into an operating system, deploying and optimizing LLM inference, and reverse-engineering and porting neural networks down to the GPU kernel level.

Based in Beijing. Available from October 2026.
📧 lmxxf@hotmail.com · GitHub [@lmxxf](https://github.com/lmxxf) · ORCID [0009-0008-0169-0409](https://orcid.org/0009-0008-0169-0409)

*[中文版见下方](#靳岩岩--简历) / Chinese version below*

---

## What I Do

**AI agent systems.** Led the design and delivery of a system-level AI agent subsystem on OpenHarmony, from scratch: multi-agent orchestration (ReAct, function calling, a three-tier delegation chain, A2A, an MCP tool ecosystem), RAG long-term memory (vector retrieval + knowledge graph), and multimodal interaction (VLM / ASR / gesture / intent recognition). Plenty of people can wire up an agent against an API; far fewer have put one inside an operating system.

**LLM deployment and inference optimization.** Deployed Qwen3.6-27B privately on an 8×V100 cluster (vLLM + AWQ + tensor parallelism) serving 70+ engineers daily; diagnosed and fixed a CUDA graph concurrency defect for a 2.6× throughput gain at 8-way concurrency. Built a two-node DGX Spark deployment image for DeepSeek-V4 (280B) — 10K+ pulls on Docker Hub.

**GPU kernels and neural network inference.** Reverse-engineered NVIDIA's DLSS 5 network (71 modules, 152 compute layers) out of a closed-source DLL and ported it to AMD RDNA4, with two backends — D3D12/HLSL and HIP — validated bit-for-bit throughout. Brought a frame from 576 ms down to 13 ms; 56–57 FPS at 900p in real gameplay. Found and fixed an unhandled FP8 numerical defect along the way (hardware E4M3 conversion doesn't saturate — values beyond ±448 become NaN, which `clamp` then washes into black blocks on screen); that fix is the direct reason the community reports better image quality from this implementation than from comparable closed-source ones. 60★ and climbing.

**Working across unfamiliar territory.** I use Claude Code / Codex as my primary development method, on a self-built two-node DGX Spark cluster (128 GB × 2, 200 Gbps direct link) for interpretability experiments and 280B inference and fine-tuning. When a domain is new to me — GPU kernels, FP8 numerical behavior, graphics pipelines — I get to a working result in weeks by holding to bit-exact comparison, recomputable evidence, and small reversible steps. The DLSS 5 port is the full worked example: every optimization round left behind per-frame bit-exact RGB comparisons and ABBA timing records, and any gain without a control was not adopted.

**Writing and research.** I publish a daily AI technical newsletter (315+ posts, 6,000+ subscribers), verifying technical reports and reproducing results by hand — model architectures, the full post-training chain (SFT/RLHF/RLVR, online distillation, agentic RL infrastructure), and interpretability (grokking, activation subspaces, SAE). Five independent research papers on interpretability, archived with DOIs on Zenodo.

**Languages:** Python (PyTorch training/inference), C/C++, ArkTS, Objective-C, Shell. Comfortable with Linux systems programming, SELinux, cross-compilation, and embedded porting.

---

## Selected Projects

### DLSS 5 Neural Network on AMD GPUs · 2026/08 – present · creator, principal developer
[github.com/lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting) · 60★

NVIDIA's DLSS 5 denoising/upscaling network shipped inside a driver DLL with no public documentation. I recovered the full network structure from that 158 MB binary — a six-stage U-shaped vision network, 71 modules, 152 compute layers, FP8 stored and computed in two layers — and reimplemented it on an AMD RX 9070 XT (RDNA4) across two backends, then wired it into three injection paths (OptiScaler / Magpie / REFramework) for playable use in commercial games.

- 576 ms → 13 ms per frame, located round by round by reading disassembly: matrix instructions, memory coalescing, eliminating divergent branches and redundant reads. 56–57 FPS at 900p in real gameplay.
- Diagnosed and fixed an unhandled hardware FP8 (E4M3) saturation defect producing NaN beyond ±448, surfacing as black blocks on screen. Community testing reports image quality and frame rate ahead of comparable closed-source implementations.
- Acceptance throughout was per-frame bit-exact RGB comparison plus ABBA timing; no gain without a control was ever adopted. Releases ship full per-file SHA256 manifests.

### System-Level AI Agent Subsystem on OpenHarmony · 2026/03 – 2026/09 · project lead
Built end-to-end AI agent capability into OpenHarmony 6.0: a C++ system ability managing a Python 3.12 multi-agent runtime and pluggable AI capabilities, exposed to applications through a unified NAPI/IPC interface for agents, intent tools, gesture and speech recognition.

- Delivered end-to-end on two hardware platforms (RK3568 / RK3588), supporting project acceptance.
- Three-tier agent delegation, multimodal interaction, cross-device A2A orchestration, and session/long-term memory all running on real hardware.
- Full-chain security closure under SELinux enforcing mode, unattended from boot.

### Tantan App CI / Binary Build System · 2021/09 – 2023/03 · project lead
Automated merges, packaging, and submission for the Tantan iOS app, plus an OCLint/SwiftLint checking system reporting into GitLab MR comments. One release per week, shipped on time every time; the recurring classes of production bugs stopped happening.

### Meituan MTFlexbox · 2016/12 – 2018/06 · iOS lead
A Flexbox-based dynamic UI framework, first used on the Meituan home page. From initial design through first release and two years of iteration, it spread to main search results, user profiles, and other sections — reaching effectively the whole app's DAU, and becoming Meituan's primary component for dynamic UI.

---

## Experience

**China North Vehicle Research Institute (No. 208 Research Institute)** · Engineer · 2023/07 – 2026/09

For the past year I led the design and development of a system-level AI agent subsystem — brought in by the business unit for my AI background to build a complete on-device agent stack on OpenHarmony from nothing, covering multi-agent orchestration, LLM deployment, RAG memory, multimodal interaction, and a secure agent execution environment. (The two years before that: OH system application development in ArkTS/C++, working across the application framework, system services, and whole-device integration.)

1. **Multi-agent orchestration.** A three-tier personal → functional → platform delegation chain (ReAct loop + function calling + agent-as-tool, communicating over A2A), concurrent sub-agent dispatch, dynamic tool discovery, and an MCP tool ecosystem; distributed orchestration across three devices. *(Python, C++)*
2. **Agent memory.** Session memory persisted across restarts, plus RAG long-term memory (SQLite + vector retrieval + knowledge graph expansion, isolated per agent). *(Python)*
3. **LLM deployment and inference optimization.** Qwen3.6-27B inference on the company's 8×V100 cluster (vLLM, AWQ quantization + tensor parallelism), serving 70+ people in the OS group daily. Diagnosed a concurrency defect on V100 — CUDA graph capture sizes only covered batch=1/2, so concurrent requests fell back to the slow per-kernel launch path — for a 2.6× gain in total throughput at 8-way concurrency. *(vLLM, CUDA)*
4. **Multimodal AI services.** A pluggable AI capability system service integrating visual understanding (VLM), speech recognition (Zipformer/SenseVoice with dynamic engine switching and streaming model upload), gesture recognition (MindSpore Lite), and an intent recognition framework (InsightIntent). *(C++, Python)*
5. **Agent runtime at the system level.** Ported the AgentScope multi-agent framework into an OpenHarmony system service (C++ SA + NAPI/IPC + UDS), and designed a "whole musl + compatibility shim" approach that brought Python 3.12 and native dependencies like numpy and grpc onto the device with zero cross-compilation. Delivered end-to-end on RK3568/RK3588. *(C++, Python, ABI)*
6. **Secure agent execution.** A capability-separated Alpine chroot command sandbox for agent tool execution (isolated over an SSH channel), with a purpose-built SELinux domain and file labels closing the chain under enforcing mode. *(SELinux, Linux)*
7. **Hard-problem debugging.** Led several system-level investigations — a company-wide daily build hanging at boot (bisected across images, then swapped parts on the board, down to a single line of permission configuration), an inference service DDoSing itself, SELinux startup races — and wrote up the methodology each time.

**Tantan** · iOS Technical Expert (T7) · 2021/09 – 2023/03
CI/CD for app releases: OCLint/SwiftLint checks on MRs reported into GitLab comments (20 custom OCLint rules, ~100 automated comments weekly); full automation of merge gating, release, and App Store submission; a binary build system from scratch that took weekly Jenkins builds from 20 to 10 minutes and local clean builds from 20 to 12; internal tooling including the Tofu commit tool and a binary CocoaPods plugin. *(Python, C++, Objective-C, Ruby, Shell)*

**Meituan-Dianping** · Senior Software Engineer (L8) · 2016/08 – 2021/09
Search module performance and feature work, raising instant-open rate from 30% to 60% (2020.2–2021.9); iOS lead for the unified web container Titans, clearing accumulated WebView crashes and substantially lowering the app's crash rate, completing the UIWebView → WKWebView migration and adding offline pages, long-lived connections and WebP (2018.6–2020.2); designed and single-handedly built MTFlexbox (2016.12–2018.6); platform business group — accounts, search, home category sections (2016.8–2017.2).

**SOHO China** · Mobile Development Lead · 2014/09 – 2016/07
Front-end development for the SOHO3Q internet project across iOS, Android and H5 — the app, the website, online property payments, WeChat payments — plus coordination of outsourced projects and front-end support for traditional project departments.

**Microsoft, Devices & Services** · Senior Software Engineer · 2013/08 – 2014/09
Nokia IME (S40NG/Android): new language support and low-level grammar rules across dozens of languages worldwide, including smaller ones such as Burmese; automated regression tests; code review for the Teleca team in Chengdu. *(C/C++, Java)*

**Guangzhou Chujian Culture, Beijing Branch** · iOS Development Lead · 2011/02 – 2013/07
7 Days Inn and Chujian lifestyle iOS clients from 1.0 through 3.0 — overall architecture, core modules (networking, XMPP chat, caching), and team hiring and training. *(Objective-C)*

**Harris (Beijing) Communications** · Senior Development Engineer · 2009/06 – 2011/02
Maintained the ADC Air-Client / Media-Client broadcast playout software used by US broadcasters and TVB Hong Kong — reliability work on nearly two decades of legacy code. *(Object Pascal, Delphi)*

**MediaTek (MTK) Beijing** · Software Engineer · 2007/02 – 2009/06
UI development and maintenance for networking modules on the MTK platform — Wi-Fi, IPSec, email, dial-up, provisioning — within a thousand-person cross-national team. *(C)*

---

## Education

**Beijing University of Technology** — M.S., Computer Application Technology · 2004/09 – 2007/07
**Beijing University of Technology** — B.S., Computer Science and Technology · 2000/09 – 2004/07

## Also

**Patents.** Public-key encryption over finite field GF(2^m) conic curves (CN1920841); a mobile device security audit and alerting method based on device-cloud collaboration and user behavior baselines (passed Beijing pre-examination, late 2025); an LLM optimization system that adaptively selects inference mode based on task characteristics (filed with patent counsel); others pending.

**Papers.** Four cryptography papers during my master's (*Computer Engineering*, *Acta Electronica Sinica*, and others). Five independent 2026 papers on AI cognition and interpretability — grokking and manifold geometry, learnability boundaries of neural networks, activation subspace structure in LLMs — all formally archived with Zenodo DOIs.

**Newsletter.** A daily AI technical newsletter (315+ posts, 6,000+ subscribers) along three lines: deep reads of model architectures (DeepSeek MLA/MoE, Qwen, Kimi linear attention — verifying every technical report's numbers and running the experiments myself), the full post-training chain, and AI interpretability.

**Open source.** [@lmxxf](https://github.com/lmxxf), 200+ stars total: DLSS 5 on AMD GPUs (60★), an open implementation of Agentic Context Engineering (45★), LLM reasoning-enhancement prompt experiments (43★), two-node DGX Spark deployment for DeepSeek-V4 (25★), OpenHarmony on-device AI porting. The companion image `lmxxf/vllm-deepseek-v4-dgx-spark` has 10K+ pulls on Docker Hub.

**Languages.** English (CET-6), Japanese (conversational), Mandarin, Hokkien.

Fifteen-plus years of training every day, which is where the steadiness comes from.

---
---

# 靳岩岩 · 简历

**AI Agent 系统 · 端侧智能 · GPU 内核与推理优化**

19 年软件开发经历，横跨嵌入式（MTK/Nokia）、iOS（美团 L8 / 探探 T7）、CI/CD 与操作系统级开发。近三年转向 AI：把 AI Agent 做进操作系统，做大模型部署与推理优化，以及把神经网络逆向还原并一路优化到 GPU 内核层。

坐标北京，2026 年 10 月起可到岗。
📧 lmxxf@hotmail.com · GitHub [@lmxxf](https://github.com/lmxxf) · ORCID [0009-0008-0169-0409](https://orcid.org/0009-0008-0169-0409)

---

## 我做什么

**AI Agent 系统开发。** 牵头 OpenHarmony 系统级 AI 智能体子系统的设计与交付，从零构建：多 Agent 编排（ReAct、Function Calling、三层委派链、A2A、MCP 工具生态）、RAG 长期记忆（向量检索 + 知识图谱）、多模态交互（VLM / ASR / 手势 / 意图识别）。会调 API 攒 Agent 的人多，能把 Agent 做进操作系统的少。

**大模型部署与推理优化。** 在 8×V100 集群私有化部署 Qwen3.6-27B（vLLM + AWQ + 张量并行），服务 70+ 工程师日常使用；定位并修复 CUDA graph 并发缺陷，8 路并发总吞吐提升 2.6 倍。自制 DeepSeek-V4（280B）双机 DGX Spark 部署镜像，Docker Hub 下载 10K+。

**GPU 内核与神经网络推理。** 把 NVIDIA DLSS 5 的神经网络（71 模块 152 计算层）从闭源 DLL 中逆向还原，移植到 AMD RDNA4，D3D12/HLSL 与 HIP 两条后端全程逐位一致验证。单帧从 576 毫秒优化到 13 毫秒，实玩 900P 56–57 FPS。过程中定位并修复了一个上游未处理的 FP8 数值缺陷——硬件 E4M3 转换不饱和，超过 ±448 变成 NaN，再被 `clamp` 洗成画面上的黑块——这个修复正是社区反馈本实现画质优于同类闭源方案的直接原因。开源 60★，仍在增长。

**在不熟悉的领域里推进。** 日常以 Claude Code / Codex 为主力开发方式，自建 DGX Spark 双机集群（128G × 2，200Gbps 直连）跑可解释性实验与 280B 模型推理微调。面对不熟的领域——GPU 内核、FP8 数值行为、图形管线——靠逐位对照、可复算的证据、小步可回退的验证，在周级别拿到可用结果。DLSS 5 移植就是这套方法的完整样本：每一轮优化都留下逐帧 RGB 逐位比对与 ABBA 计时记录，没有对照的收益一律不采用。

**写作与研究。** 独立运营 AI 技术公众号（日更 315+ 篇，6000+ 订阅），逐篇核实技术报告并动手复现——主流模型架构、后训练全链条（SFT/RLHF/RLVR、在线蒸馏、Agentic RL 基建）、AI 可解释性（Grokking、激活子空间、SAE）。可解释性方向独立研究论文 5 篇，均有 Zenodo DOI 存档。

**常用语言：** Python（PyTorch 训练/推理）、C/C++、ArkTS、Objective-C、Shell；熟悉 Linux 系统编程、SELinux、交叉编译与嵌入式移植。

---

## 代表项目

### DLSS 5 神经网络移植到 AMD 显卡 · 2026/08 – 至今 · 发起者、主要开发
[github.com/lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting) · 60★

NVIDIA DLSS 5 的降噪/超分神经网络随驱动 DLL 泄露，但没有任何公开文档。我从这个 158MB 的二进制里还原出完整网络结构——U 形视觉网络六站、71 模块 152 计算层、FP8 存算两层——在 AMD RX 9070 XT（RDNA4）上用两条后端重新实现，并接入 OptiScaler / Magpie / REFramework 三条注入路径，在多款商业游戏中实玩可用。

- 单帧 576 毫秒 → 13 毫秒，靠读汇编逐轮定位：矩阵指令、访存合并、消除发散分支与重复读取。实玩 900P 56–57 FPS。
- 定位并修复上游未处理的硬件 FP8（E4M3）不饱和缺陷：超过 ±448 产生 NaN，在画面上表现为黑块。社区实测反馈画质与帧率优于同类闭源实现。
- 全程以逐帧 RGB 逐位比对 + ABBA 计时作为验收，没有对照的收益一律不采用。发行包附完整的逐文件 SHA256 清单。

### OpenHarmony 系统级 AI 智能体子系统 · 2026/03 – 2026/09 · 项目负责人
在 OpenHarmony 6.0 上从零构建系统级 AI Agent 能力：系统侧 C++ SA 管理 Python 3.12 多智能体运行时及插件化 AI 能力，应用侧通过统一 NAPI/IPC 接口使用 Agent、意图工具、手势与语音识别。

- RK3568 / RK3588 两个硬件平台端到端交付，支撑项目验收。
- 三层 Agent 委派、多模态交互、跨设备 A2A 编排、会话与长期记忆全部真机跑通。
- SELinux Enforcing 模式下无人值守开机、全链路安全闭环。

### 探探 App 持续集成与二进制化 · 2021/09 – 2023/03 · 项目负责人
维护探探 App 日常合入、打包、提审自动化，以及 OCLint/SwiftLint 检查体系（结果输出到 GitLab MR comment）。每周一个版本稳定发布、未出现过版本延迟；历史常见的那几类线上 bug 不再发生。

### 美团 MTFlexbox 动态化 UI 框架 · 2016/12 – 2018/06 · iOS 端负责人
基于 Flexbox 的动态化 UI 视图框架，最初用在美团首页。从设计到第一版上线并持续改进两年，扩展到主搜索结果、个人主页等各区块，覆盖面基本等同美团 App DAU，成为美团 UI 动态化方向的主要组件。

---

## 工作经历

**兵器装备 208 研究所** · 工程师 · 2023/07 – 2026/09

近一年牵头系统级 AI 智能体子系统的设计与开发——因 AI 技术背景被业务部门点将，从零构建 OpenHarmony 上完整的端侧 Agent 栈，覆盖多 Agent 编排、大模型部署、RAG 记忆、多模态交互与 Agent 安全执行环境。（此前两年负责 OH 系统应用开发，ArkTS/C++，熟悉应用框架、系统服务与整机集成。）

1. **多 Agent 编排体系。** 设计实现"个人 → 职能 → 平台"三层 Agent 委派链（ReAct 循环 + Function Calling + Agent-as-Tool，A2A 协议通信）、子 Agent 并发派发、动态工具发现与 MCP 工具生态；支持三台设备跨设备分布式编排。*(Python, C++)*
2. **Agent 记忆系统。** 会话记忆持久化（跨重启恢复上下文）+ RAG 长期记忆（SQLite + 向量检索 + 知识图谱扩展，按 Agent 隔离）。*(Python)*
3. **大模型私有化部署与推理优化。** 在公司 8×V100 集群部署 Qwen3.6-27B 推理服务（vLLM，AWQ 量化 + 张量并行），服务鸿蒙系统组 70+ 员工日常使用；定位修复推理框架在 V100 上的并发性能缺陷——CUDA graph 捕获尺寸只覆盖 batch=1/2，并发时掉回逐 kernel 发射的慢路径——8 路并发总吞吐提升 2.6 倍。*(vLLM, CUDA)*
4. **多模态 AI 能力服务。** 插件化 AI 能力系统服务，集成视觉理解（VLM 看图）、语音识别（Zipformer/SenseVoice 双引擎动态切换、流式模型上传）、手势识别（MindSpore Lite）与意图识别框架（InsightIntent）。*(C++, Python)*
5. **Agent 运行时的系统级落地。** 将 AgentScope 多智能体框架移植为 OpenHarmony 系统服务（C++ SA + NAPI/IPC + UDS），设计"整块 musl + 兼容 shim"方案，零交叉编译引入 Python 3.12 及 numpy/grpc 等 native 依赖，在 RK3568/RK3588 多款硬件端到端交付。*(C++, Python, ABI)*
6. **Agent 安全执行环境。** 为 Agent 工具执行设计权能分离的 Alpine chroot 命令沙盒（SSH 通道隔离）；自建 SELinux 域与文件标签，Enforcing 模式下全链路安全闭环。*(SELinux, Linux)*
7. **疑难问题攻坚。** 多次主导系统级疑难排查——全公司 dailybuild 开机卡死（镜像二分 + 板上换件，定位到单行权限配置）、推理服务自我 DDoS、SELinux 启动竞态等——并产出排查方法论文档。

**探探科技** · iOS 技术专家（T7）· 2021/09 – 2023/03
负责版本发布持续集成：MR 提测期的 OCLint/SwiftLint 检查并输出到 GitLab comment（自研 OCLint 规则 20 条，每周自动化 comment 约 100 条）；合版准入、版本发布、上传 App Store 全流程自动化；从零构建二进制化方案，Jenkins 每周平均编译 20 分钟缩短至 10 分钟、本地 clean build 20 分钟缩短至 12 分钟；开发代码提交工具 Tofu、二进制化 cocoapods 插件等。*(Python, C++, Objective-C, Ruby, Shell)*

**美团点评** · 高级软件工程师（L8）· 2016/08 – 2021/09
搜索模块性能优化及业务开发，搜索秒开率 30% → 60%（2020.2–2021.9）；iOS 统一 web 容器 Titans 的 iOS 侧负责人，解决大量 webview 积累 crash、大幅降低 App crash 率，完成 UIWebView 向 WKWebView 迁移，实现离线化 web 页面、长连接与 webp 支持（2018.6–2020.2）；设计并独立开发 MTFlexbox（2016.12–2018.6）；平台业务组账户、搜索、首页品类区等模块开发（2016.8–2017.2）。

**SOHO 中国** · 移动端开发负责人 · 2014/09 – 2016/07
SOHO3Q 互联网项目前端相关开发（iOS、Android、H5）：SOHO3Q App、H5 网站、物业在线收费、微信收费等；负责外包项目协调统筹，为传统项目部门提供前端技术支持。

**微软 · 设备与服务部门** · 高级软件工程师 · 2013/08 – 2014/09
诺基亚输入法（NokiaIME，S40NG/Android）：新语言支持与底层语法规则支持，覆盖全球几十种语言（含缅甸语等小语种）；自动化回归测试用例编写；审核成都 Teleca 代码提交。*(C/C++, Java)*

**广州初见文化传播 · 北京分公司** · iOS 开发负责人 · 2011/02 – 2013/07
七天连锁酒店 / 初见吃喝玩乐 iOS 客户端，从 1.0 到 3.0 持续迭代：整体架构、核心模块（网络、xmpp 聊天、缓存）、团队招聘与培训。*(Objective-C)*

**HARRIS（北京）通讯技术** · 高级开发工程师 · 2009/06 – 2011/02
维护 ADC Air-Client / Media-Client 电视播控软件（美国及香港 TVB 等电视台采用），近二十年历史遗留代码的可靠性维护。*(Object Pascal, Delphi)*

**联发博动科技（北京）/ MTK** · 软件工程师 · 2007/02 – 2009/06
MTK 平台网络相关模块 UI 开发维护（Wi-Fi、IPSec、电子邮件、拨号上网、Provisioning 等），千人级跨国团队协作。*(C)*

---

## 教育经历

**北京工业大学** — 计算机应用技术 硕士 · 2004/09 – 2007/07
**北京工业大学** — 计算机科学与技术 本科 · 2000/09 – 2004/07

## 其他

**专利。**《基于有限域 GF(2^m) 的圆锥曲线公钥加密方法和装置》（公告号 CN1920841）；《一种基于端云协同与用户行为基线的移动设备安全审计预警方法及系统》（2025 年底通过北京预审）；《一种基于任务特点自适应选择推理模式的大语言模型优化系统》（已递交专利代理人）；另有多项在申请中。

**论文。** 硕士阶段密码学论文 4 篇（《计算机工程》《电子学报》等）；2026 年 AI 认知与可解释性方向独立研究论文 5 篇（Grokking 流形几何、神经网络可学习性边界、LLM 激活子空间结构等），均有 Zenodo DOI 正式存档。

**公众号。** AI 技术公众号日更 315+ 篇、6000+ 订阅，三大主线：主流模型架构深度解析（DeepSeek MLA/MoE、Qwen、Kimi 线性注意力等，逐篇核实技术报告的数据并动手做实验）、后训练全链条、AI 可解释性。

**开源。** [@lmxxf](https://github.com/lmxxf)，累计 200+ star：DLSS 5 移植到 AMD 显卡 60★、Agentic Context Engineering 开源实现 45★、LLM 推理增强提示词实验 43★、DeepSeek-V4 双机 DGX Spark 部署 25★、OpenHarmony 端侧 AI 移植等；配套 Docker 镜像 `lmxxf/vllm-deepseek-v4-dgx-spark` 在 Docker Hub 下载 10K+。

**语言。** 英语（CET-6）、日语（基本听说读写）、普通话、闽南语。

连续 15 年以上每天健身——冷静大概就是从这儿来的。
