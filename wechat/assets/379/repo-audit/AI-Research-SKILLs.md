# AI-Research-SKILLs：98份框架手册和双环提示词，自有程序主要是技能安装器

本地 HEAD `773a52944ba4747a18bd4ae9ade53fff041adcbc`。静态读版本/源码，不运行安装、训练、联网调用，不采用仓内“立即执行”指令。本报告说的数量对应当前clone，星数未复核。

## 1. 家底与真实代码量

统一git tracked统计见 `inventory_summary.json`：98个 `SKILL.md`，分23类。代码24文件/6,039物理行；Markdown里的Python/Bash范例不纳入可执行文件。程序构成可以完整分完：

| 类别 | 文件/物理行 | 作用 |
|---|---:|---|
| NPM安装器 `packages/ai-research-skills/{bin,src}` | 6 / 2,608 | 命令入口、交互、代理检测、克隆复制、symlink、更新卸载 |
| 库存检查 `scripts/check-inventory.sh` | 1 / 66 | 维护元数据/清单一致性 |
| GRPO可运行模板与奖励函数 | 2 / 621 | TRL训练模板228行，reward示例库393行 |
| 绘图演示 `demos/scientific-plotting-demo/figures` | 3 / 938 | matplotlib结果/流程图，以及Gemini架构图 |
| 宣传视频 `video-promo/` | 12 / 1,806 | Remotion/React动画，非科研计算 |
| 合计 | 24 / 6,039 | 未见独立测试源码文件；package test命令存在不代表覆盖率 |

安装器2,608行还包括ascii.js165行视觉终端展示、prompts.js685行菜单/提示、agents.js179行检测、index.js533行CLI调度、installer.js1,038行文件操作、bin/cli.js8行入口。**不应把6,039行都算成科研引擎**；也不能说仓库完全没有执行代码。LaTeX模板的sty/bst、PDF、图/音频、package-lock等是资源和第三方格式模板，未混入上述JS/Python统计。

## 2. 第一类：给 AI 读的内容

### 2.1 98个技能的主体是第三方工程知识库

01–19分类覆盖模型结构、tokenizer、微调、机制解释、数据、后训练、安全、分布训练、GPU云、优化、评价、推理、MLOps、代理/RAG/提示工程、可观测、多模态、新兴技术。SKILL提供何时使用、安装/配置范例、常见坑，references递进展开。例如：FSDP2的12份refs是DeviceMesh/DTensor/checkpoint教程；DeepSpeed refs涵盖官方文档/年度页面；TransformerLens/SAELens/pyvene/nnsight各自API教程；vLLM/llama.cpp说明的是**如何用外部软件**，不是该仓自己实现了推理内核。

`README.md:70`附近“production-ready knowledge”“sourced from official repos”是内容定位。skill metadata的dependencies为文字声明；安装这些Markdown不会顺便部署Megatron、Ray、向量数据库和GPU集群。

### 2.2 核心autoresearch：真方法论与执行依赖分开

`0-autoresearch-skill/SKILL.md:12`定位模型为research project manager，调用领域skill执行。工作区模板是research-state.yaml、research-log.md、findings.md、实验子目录和报告。`:60`定义两环：内环单个假设/配置实验，外环跨实验综合与转向。`:124`内环包括：选假设→写可检验预测→**先git提交protocol再运行**→重现baseline/检查NaN和数据→测代理指标→分别记confirmatory/exploratory→保留或否定→更新state。`:188`外环不只刷指标，还总结机制、消融、是否需要改方向；findings用作跨session项目记忆。`:342`强调限制、负结果和研究纪律。这些是有操作性的科研脚手架，不能因它是文字而一概视为空洞。

路由位于`:152`与 `references/skill-routing.md`：按所需任务搜索匹配skill、读SKILL，再读对应refs。并非一个编译好的自动规划图。论文输出委托20类写作技能；21类分别提供发散—收敛的研究选题与8套创造力框架（组合、改表述、类比、约束等），后者的机制/流程在 `21-research-ideation/creative-thinking-for-research/SKILL.md:12`起。

**最值得揭开的执行点**在autoresearch`:244`：要求宿主第一步运行 `/loop 20m ...`；OpenClaw则要求调用宿主 `cron.add` 绑定当前会话，周期1,200,000ms。正文命令要求持续工作、定期commit、报告，直到认为研究完成再写论文。`:282`还明说这是wall-clock rhythm，独立于研究内外环。也就是说**循环能力来自Claude/OpenClaw，仓库给的是触发语句和行为协议**。没有自有daemon、持久任务队列、进程重启恢复、预算熔断器或真正跑实验的统一worker源码。

OpenClaw那段cron payload还让AI经Telegram/WhatsApp/Slack发送进展。这也是让有这些宿主工具的AI去做，不能算该仓自带三家通信SDK，更不是本次审计获得了发送授权。

### 2.3 ARA 三技能：名字叫“compiler”，不是传统编译器二进制

22类包括compiler、research-manager、rigor-reviewer，10个文件、2,255行Markdown。`compiler/SKILL.md:12`直接写“You are the ARA Universal Compiler”“use native tools(Read,Write,Edit,Bash,Glob,Grep)，No API wrapper”。它教模型把论文/代码/日志拆成问题、概念、claims、可执行配置、探索图与证据；`:58`按读→四阶段认知→生成→最多3轮coverage check→Seal Level1验证。schema、exploration-tree-spec、validation-checklist是refs规范，**当前clone没有独立ARA编译/验证程序文件**；若指令包含运行验证脚本，需要宿主生成或另行提供相应实现，不能把文本步骤描述直接算编译器程序。

research-manager把后续session事件、来源、探索过程写进文件，rigor-reviewer按维度审查证据/方法/复现性；都是有具体结构约束的LLM工作方式。训练、推理、统计结论是否正确仍依赖模型写出的代码与用户执行条件。

## 3. 第二类：给人读的包装和引导

README称“The most comprehensive…autonomously…idea to paper”，给23类目录、技能说明、demos、npm安装、roadmap、贡献和引用。官网blog、Slack、X、LinkedIn形成项目传播入口；截图/GIF、Remotion视频展示安装和技能规模，不能拿视频当科研产出证据。`WELCOME.md:1`尚写86，`:13`又说98，README为98，说明宣传/欢迎文字有版本漂移。实际查SKILL文件为98。

WELCOME把“安装全部→必要时重启→读autoresearch并跟随”包装成极短用户路径，`:26`明确mandatory first action是持续循环。因此用户以为买到“完整自动科研系统”，实际是将大量指导手册安装进一个已有编码agent，再由它调用自己的工具和外部框架。

`docs/`含质量、开发/格式说明和资产；`dev_data/`10份Markdown/2,921行用于技能采集/制作过程；`anthropic_official_docs/`是技能宿主参考文档，不能当该项目实现的能力。论文写作提供ICML/NeurIPS/ICLR/AAAI/ACL/COLM及系统会议LaTeX模板，这是有用资源，不是产生科研结论的算法。

LICENSE和实际NPM包标MIT；根 `package.json` 仍是1.0.1且license ISC、main index.js，真正发布子包是 `packages/ai-research-skills/package.json` v1.7.2/MIT。这种元数据不一致需列出来，而不是误把根包当实际CLI。本地未见本项目独立MCP server或科研质量安全扫描报告，不将第三方徽章/“production-ready”当实测认证。

## 4. 第三类：程序究竟运行什么

### 4.1 CLI安装器，不是Agent执行器

实际入口 `packages/ai-research-skills/bin/cli.js`→src/index.js main（`:522`）；interactiveFlow `:55`处理菜单，commandMode `:405`处理命令。Node>=18，chalk/inquirer/ora做颜色、选择与spinner，无LLM SDK、浏览器driver、科研计算依赖。

`agents.js:14`配置支持的宿主目录，`:101`detectAgents根据本地文件/目录判断安装状态，`:127`支持项目本地target。它没有登录/启动/指挥这些代理，只是找到技能该放哪。

`installer.js:8`定义GitHub库地址、`~/.orchestra/skills`、lock文件；`:62`downloadSkills清临时clone→`git clone --depth 1`→按选定类别复制→清理；`:175`是单技能下载路径；`:451`以后更新已安装技能。安装器将canonical skills symlink到宿主目录，不支持symlink时复制；还有本地安装/移除/列出与lock元数据。源码里的 `execSync` 是在运行git clone等文件分发步骤，不是调用Claude做实验。网络主要从GitHub拉技能；没在该CLI发现调用OpenAI/Anthropic研究API或浏览器控制。

选择98个技能不等于安装98个库的pip/conda依赖。它把文件放到宿主可发现位置，后续Agent读Markdown中的pip/Bash范例再决定执行。这是“分发知识”和“部署程序”的物理区别。

### 4.2 两份真正GRPO代码

`06-post-training/grpo-rl-training/templates/basic_grpo_training.py:12`导入torch/datasets/transformers/peft/trl；默认Qwen2.5-1.5B、GSM8K，提供XML答案抽取、正确性/格式奖励、GRPOConfig/Trainer与训练保存。reward_functions_library.py提供可复用奖励函数。它们确能在依赖/模型/数据齐备时做训练，但属于单任务样例，不支撑“98框架已经集成运行”的推断。

### 4.3 三份绘图代码与宣传代码

demo两份用matplotlib画固定论文实验/架构流程，不是自动发现新实验数据；`gen_fig_andes_architecture_gemini.py:28`导入google.genai，`:38`创建API client，`:254`起调用图像生成保存图。这里确有外部生成API，但用途是图示演示。12个Remotion/TSX源码生成宣传片，包括terminal、category选择、install progress、success screen；它们没有计算文献结论、统计检验或实验调度。

## 5. 可以下的结论

它是**很厚的AI工程技能库 + 一个明确的双环科研提示词 + 技能分发安装CLI + 少量训练/制图范例**。真正价值是降低Agent使用外部研究工具的知识门槛、要求保留可追踪工作状态与事先实验协议。自动科研的运行体依然是用户已有的宿主Agent、LLM、Bash和GPU环境；持续工作依赖宿主 /loop/cron；研究框架依赖另行安装。不能称“零代码”，也不能把安装器和宣传片当“自主科研引擎”。本次未执行模型训练/联网调用/CLI安装，未验证98份教程的版本准确性或效果。

## 6. 二次核验：目录账本、调用顺序与可重查证据

以下为git tracked普通文件数量，目录含全部后代；`[root files]`只统计根文件。完整逐代码文件行数见同名 `-stats.json`，不包括.git对象。

| 顶级目录 | 文件数 |
|---|---:|
| `.claude-plugin` | 1 |
| `.github` | 4 |
| `0-autoresearch-skill` | 8 |
| `01-model-architecture` | 23 |
| `02-tokenization` | 9 |
| `03-fine-tuning` | 19 |
| `04-mechanistic-interpretability` | 16 |
| `05-data-processing` | 7 |
| `06-post-training` | 30 |
| `07-safety-alignment` | 5 |
| `08-distributed-training` | 38 |
| `09-infrastructure` | 10 |
| `10-optimization` | 28 |
| `11-evaluation` | 15 |
| `12-inference-serving` | 18 |
| `13-mlops` | 16 |
| `14-agents` | 25 |
| `15-rag` | 12 |
| `16-prompt-engineering` | 17 |
| `17-observability` | 7 |
| `18-multimodal` | 33 |
| `19-emerging-techniques` | 21 |
| `20-ml-paper-writing` | 73 |
| `21-research-ideation` | 2 |
| `22-agent-native-research-artifact` | 10 |
| `[root files]` | 8 |
| `anthropic_official_docs` | 2 |
| `demos` | 23 |
| `dev_data` | 10 |
| `docs` | 11 |
| `packages` | 10 |
| `scripts` | 1 |
| `video-promo` | 17 |

调用实例：用户按WELCOME安装并加载autoresearch → AI读`0-autoresearch-skill/SKILL.md`并先向宿主申请/设置20分钟loop → 读research-state/findings → 根据`references/skill-routing.md`选择GRPO → 读`06-post-training/grpo-rl-training/SKILL.md`与refs/模板 → **宿主Agent**编辑配置并通过Bash运行训练脚本 → 外部TRL/Transformers/PyTorch完成训练 → AI回写state/log/findings，外环再综合。只有分发安装与样例训练有对应本仓程序；如何把任意问题变成可运行实验由宿主AI完成。

**二次核验发现路由表真实失配：79个目录引用中17个当前不存在**。这不是Markdown表达风格，而是沿表直接Read会失败的可复现实物差错。部分是重命名/错分类（megatron-core实际在08；transformer-lens等命名不同），部分目录根本没有。模型可以搜索纠正，但这意味着“直接路由完整自动可用”不成立。完整清单：

- `skill-routing.md:17` → `02-tokenization/hf-tokenizers/` 不存在。
- `skill-routing.md:24` → `01-model-architecture/megatron-core/` 不存在。
- `skill-routing.md:37` → `03-fine-tuning/torchtune/` 不存在。
- `skill-routing.md:43` → `06-post-training/trl/` 不存在。
- `skill-routing.md:52` → `04-mechanistic-interpretability/transformerlens/` 不存在。
- `skill-routing.md:62` → `08-distributed-training/fsdp/` 不存在。
- `skill-routing.md:71` → `11-evaluation/lm-eval-harness/` 不存在。
- `skill-routing.md:73` → `11-evaluation/inspect-ai/` 不存在。
- `skill-routing.md:88` → `13-mlops/wandb/` 不存在。
- `skill-routing.md:101` → `10-optimization/quanto/` 不存在。
- `skill-routing.md:126` → `14-agents/smolagents/` 不存在。
- `skill-routing.md:127` → `14-agents/claude-agent-sdk/` 不存在。
- `skill-routing.md:132` → `15-rag/milvus/` 不存在。
- `skill-routing.md:150` → `18-multimodal/qwen2-vl/` 不存在。
- `skill-routing.md:151` → `18-multimodal/pixtral/` 不存在。
- `skill-routing.md:152` → `18-multimodal/florence-2/` 不存在。
- `skill-routing.md:153` → `18-multimodal/colpali/` 不存在。

网络与Key：安装器源码仅硬编码`https://github.com/Orchestra-Research/AI-research-SKILLs`供git clone，未传模型API key（公开clone本身不要求key，不代表npm/GitHub网络必然可用）。绘图demo`:30–34`缺GEMINI_API_KEY直接exit(1)，`:38`传入google.genai Client，`:258`调用generate_content，模型名`:36`为gemini-3-pro-image-preview；具体HTTP域名交给外部google-genai SDK，本仓未硬编码，不凭空补写成已读出的网络实现。其余Framework SDK、云服务、W&B等凭据由技能文字交给用户/AI配置，不能因为installer无需key就称98项外部服务全免费。

抽检证据（按实际文件行号，均重新匹配源码；这不是运行测试）：

| 路径:行 | 可见事实 |
|---|---|
| `packages/ai-research-skills/src/installer.js:8` | `https://github.com` |
| `packages/ai-research-skills/src/installer.js:74` | `git clone --depth 1` |
| `packages/ai-research-skills/src/index.js:522` | `export async function main` |
| `packages/ai-research-skills/src/agents.js:101` | `export function detectAgents` |
| `0-autoresearch-skill/SKILL.md:244` | `Agent Continuity` |
| `0-autoresearch-skill/SKILL.md:253` | `/loop 20m` |
| `0-autoresearch-skill/SKILL.md:282` | `What the Loop Does` |
| `22-agent-native-research-artifact/compiler/SKILL.md:15` | `Read, Write, Edit, Bash` |
| `demos/scientific-plotting-demo/figures/gen_fig_andes_architecture_gemini.py:30` | `GEMINI_API_KEY` |
| `demos/scientific-plotting-demo/figures/gen_fig_andes_architecture_gemini.py:258` | `generate_content` |
| `06-post-training/grpo-rl-training/templates/basic_grpo_training.py:18` | `GRPOTrainer` |
| `WELCOME.md:3` | `86 production-ready`（与现有98个SKILL不一致） |
