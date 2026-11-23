# 八大科研 Skill 仓库三分类深度解剖报告

审计日期：2026-11-23。对象是 `wechat/assets/379/` 下八个本地 Git 快照。三名子代理并发拆仓，主代理统一计数、追踪执行链、抽检关键实现并汇总。本文按 Zero 指定的三个视角回答：**谁给 AI 下指令，谁给人介绍产品，哪些程序真正落到电脑上执行。**

先给最重要的判断：这八个项目不是同一种东西。既有专科科学软件工具箱、写作与审稿协议、文档生产工作站，也有模型调用桥、行为提醒补丁、论文取证工具和真正的 Python 科研流水线。**“全是 Markdown 嘴炮”不符合本地实物；“装上 Skills 就部署了完整自动科学家”同样不符合实物。** 必须继续问：谁负责研究判断，谁运行实验，谁提供外部数据和模型，哪一段只是结构校验。

## 一、范围、证据与计数规则

### 1. 本次做了什么

- 对八仓 `git ls-files` 逐文件建立路径、字节数、文本行数、扩展名、SHA-256 账本；记录 origin、HEAD 与工作树状态。
- 分仓阅读 SKILL、references、角色和状态协议、README、安装与安全材料；追踪有代表性的 CLI → 函数 → 外部调用 → 输出文件链。
- 将测试、例程、安装器、排版工具、科研计算、模型/API 桥分开解释；核对宣传文字与实现存在的差距。
- 补做 Python AST 静态解析，确认程序不只是换后缀的纯文章；**AST 成功只说明可解析，不说明依赖齐全或功能正确**。

没有安装依赖、运行这八仓的安装器、启动浏览器/daemon/MCP、请求模型/API、训练或复现作者论文。本文的“真实程序”指源码中有具体可追踪实现，不能改写成“本次已经跑通”。用户提供的星数是选题背景，本次未联网核验当前星数；也不把星数当科研能力指标。

三类是**用途视角，不是互斥扩展名分区**。Python 里的长 prompt 属于给 AI 读的语义内容；SKILL 中的 Bash/Python 代码块可能由宿主执行，但不是仓内独立程序；README 同时可能约束 AI。源码行数包含空行、注释、docstring、prompt 常量和测试，不等于原创算法行数，也不等于科研效果。

### 2. 快照身份

| 本地仓库 | 本地 HEAD 前12位 | 定位 |
|---|---|---|
| scientific-agent-skills / K-Dense-AI | `ec59a29a03d5` | 177个专科科学技能及配套工具 |
| academic-research-skills / Imbad0202 | `d169ec5c4beb` | 文献—写作—评审协作协议与验证工程 |
| nature-skills / Yuan1z0825 | `c6701dbea29a` | 论文读写、检索下载与文档生产 |
| Auto-claude-code-research-in-sleep / ARIS | `4ba94ae2a1c1` | 编码 Agent 主持的研究工作流及模型/工具接线 |
| HERO-Anti-OverDefense | `778582680e29` | 减少空转的行为规则、交接及推进提醒 |
| Anti-Autoresearch | `0abae59aa060` | 论文诚信取证协议、算术/统计检查与裁决 |
| AutoResearchClaw / aiming-lab | `be4ba4755bf1` | Python 实现的多阶段科研流水线 |
| AI-Research-SKILLs / Orchestra-Research | `773a52944ba4` | AI工程技能知识库、双环科研提示词及安装器 |

完整 origin 和40位提交号见 [repository_roots.json](repo-audit/repository_roots.json)。不得用 README 中别的分支、远端二进制版本或未来路线图补齐当前快照缺失的实现。

### 3. 统一文件与源码账本

| 仓库 | tracked普通文件 | Markdown文件 / 行 | SKILL.md路径数 | 程序文件 / 物理行 | 其中测试行 | 非空程序行 |
|---|---:|---:|---:|---:|---:|---:|
| K-Dense scientific | 2,780 | 1,538 / 329,700 | 177 | 776 / 249,102 | 79,276 | 218,126 |
| academic-research | 2,882 | 1,159 / 202,649 | 5 | 506 / 292,474 | 148,831 | 256,635 |
| nature | 897 | 443 / 41,255 | 24 | 273 / 56,855 | 15,016 | 49,373 |
| ARIS | 768 | 431 / 159,890 | 197 | 176 / 64,975 | 20,695 | 56,249 |
| HERO | 22 | 13 / 2,427 | 2 | 2 / 707 | 397 | 641 |
| Anti-Autoresearch | 65 | 25 / 12,399 | 12 | 18 / 4,493 | 1,347 | 3,877 |
| AutoResearchClaw | 756 | 95 / 20,995 | 41 | 423 / 130,658 | 40,790 | 112,070 |
| Orchestra | 529 | 400 / 188,697 | 98 | 24 / 6,039 | 0* | 5,290 |

\* 指统一目录/文件名规则未识别独立测试源码，不是证明“完全没有任何自检”。程序扩展包括 Python、Shell、JS/TS、R 等；Markdown 代码块、LaTeX 样式、PDF、图片、锁文件不混入源码。测试分组使用路径启发式，某些 fixture、eval、内嵌自测仍需看分仓解释。

这里最容易读错三件事：

1. **ARIS 的197个 SKILL 是87个主技能、87个 Codex镜像、8个Claude-review覆盖和15个Gemini-review覆盖，不是197种独立科研方法。**
2. academic-research 的29万行程序约一半是测试；119个 `scripts/check_*.py` 又有55,870行，是大量契约、资产与回归检查，不能叫29万行科研算法。
3. Orchestra 的6,039行包含2,608行安装器、1,806行宣传视频、938行绘图demo和621行GRPO模板。安装、展示和研究运行要分账。

完整556条 SKILL 路径见 [skill_catalog.md](repo-audit/skill_catalog.md)；ARIS镜像、Claw内置/参考等全部原样列出。统计实现见 [inventory_repos.py](repo-audit/inventory_repos.py)、[inventory_summary.json](repo-audit/inventory_summary.json)；Python结构补充见 [python_logic_summary.json](repo-audit/python_logic_summary.json)。不得将这些结构计数换算成科学质量排名。

## 二、K-Dense scientific-agent-skills：专科工具箱，不是一个自治总控

详见 [完整177技能与源码解剖](repo-audit/scientific-agent-skills.md)。本地 manifest 标记2.72.0；origin 与本地名称一致，不能凭旧印象擅自改称其他仓名。

### A. 给 AI 读：按需使用科学包、数据库与窄工作流

典型文件结构是 `skills/<name>/SKILL.md + references/ + scripts/ + assets/`。SKILL 的描述决定何时加载，reference 提供更深的操作知识，scripts 提供可选执行工具。仓内 `AGENTS.md:11–23` 明确拒绝宽泛总控型技能，强调专科边界。这是有意设计，并非“缺一个万能科研prompt”。

内容覆盖分子药物、组学、质谱、材料、物理模拟、统计/优化、数据库、实验室平台，以及论文、图、海报与PPT。它教的是选择上游库、准备数据、调用API、检查假设和解释输出；很多算法实际属于 SciPy、statsmodels、Scanpy、RDKit、PyDESeq2 等上游项目。

方法论有实质内容：检索先固定实体/物种/参考基因组与过滤条件、分页限流和来源；实验设计区分实验单位、随机化、重复、区组和混杂；写作将数字/主张连到证据ID；假说生成要求替代解释、区分性预测与决策记录。`database-lookup` 有83个文件但零独立程序，是检索合同与API配方；这不等于没用，也不等于本仓实现了80个数据库服务器。

`arbor/scripts/tree.py` 确实维护假设树、baseline、证据、传播、剪枝和合并；真正提出改动、训练与评估仍由Agent/外部评价器做。brainstorming的脚手架和验证器明确写“No network or LLM calls”。**保存科研状态和检查字段，不等于自己产生科研发现。**

### B. 给人读：安装教学、示例、商业工作台与扫描材料

README介绍177技能、选择性安装、常见领域组合、依赖限制；贡献规范、CITATION、SECURITY 和测试工作流是治理材料。首页突出 K-Dense BYOK、教程、社交与商业生态；另一个工作台/托管产品不能算成本仓随附程序。当前本仓MIT，不能因为商业引流就抹掉实有代码。

安全材料尤其不该简化成“全绿”：`docs/security-report.md:3–11` 记载177技能、472条finding、32 critical、21 high、159/177标safe，本次0重扫、177沿用历史结果；scanner版本和模型另有注明。这是上游扫描器的分类结果，既不是本次确认的472个漏洞，也不是对159个技能永久担保。

### C. 本机真程序：有科学计算，也有大量校验和产物工具

776程序文件249,102行；测试79,276行，明显例程2,139行，其余候选167,687行。不是零代码，也不是16万行统一自动科学家。

| 代表程序 | 真正操作 | 依赖/边界 |
|---|---|---|
| statistical-power `power.py`、`simulate_power.py` | 样本量/功效/MDE求解；模拟数据→拟合检验→拒绝比例与区间 | statsmodels/SciPy；效应大小和模型假设由研究者负责 |
| statistical-analysis `assumption_checks.py` | Shapiro、方差、回归诊断、VIF等 | 诊断不能替代判断实验单位和因果设计 |
| pydeseq2 `run_deseq2_analysis.py` | 读计数、检查设计、拟合、对比、LFC收缩、导出图表 | 算法由外部PyDESeq2执行 |
| Scanpy/RDKit/pyOpenMS等脚本 | QC、聚类、分子描述符、质谱处理 | 分别安装领域环境；不是根安装器一键全装 |
| research-lookup、paper-lookup、ARAX客户端 | 实际API/CLI请求、分页、解析与来源封套 | Parallel/OpenRouter等账户与费用另计 |
| 图像、PPTX、论文/临床材料检查器 | 外部图像模型调用、组装与格式/字段校验 | 格式正确不能证明科学事实 |
| scan_skills、generate_skill_image | 扫描、汇总、维护仓库展示 | 不能算进科研推理核心 |

完整调用链是：用户问题→宿主选技能→加载reference→配置依赖与凭据→本地脚本/外部软件/API执行→宿主解释→记录/产物检查→下一步。根包不提供全局研究daemon，也不因装插件就获得实验室或已登录浏览器。

`.github/workflows/skill-tests.yml:80起` 明确CI不运行全部隔离科学环境，涉及约100套torch/qiskit/scanpy等环境且部分需CUDA/JDK/MATLAB。**有测试工程属实；绿色CI不表示177套科学依赖全部集成跑通。**

## 三、academic-research-skills：人机协作协议，外加很厚的验证工程

详见 [分仓报告](repo-audit/academic-research-skills.md)。README明确是 copilot，不是 pilot；这是理解它的正确入口。

### A. 给 AI 读：五个技能与审查门

五个技能是 deep-research、academic-paper、academic-paper-reviewer、academic-pipeline、sr-screener。角色、references、共享schema与命令提示词把检索、构架、起草、审稿、修订、筛选分开；47个agents路径包含共享和镜像，不代表47个常驻Agent进程。

`docs/ARCHITECTURE.md:43` 的主链是 RESEARCH→WRITE→2.5诚信检查→REVIEW→REVISE→RE-REVIEW→必要的再修→4.5最终诚信检查→FINALIZE→过程总结；**阶段用户确认是设计中心**。它和“睡觉时无人值守自动投稿”不是同一个产品。

实质方法有 corpus-first/缺口检索、Material Passport保存出处和阅读状态、Argument Map、材料缺口明确标记、逐条回复/修改记录、评审维度盲承诺、证据覆盖分母与未知状态。它试图区分“文献存在”“这句话获原文支持”“抽样检查过”和“全量证明”。关键语义判断仍由模型/人做，Markdown状态机不等于Python引擎强制执行。

### B. 给人读：多语言教学、测量说明与使用边界

README、QUICKSTART、SETUP、MODE_REGISTRY、CONTROL_AVAILABILITY分别解释安装、模式与不同宿主能力。纯聊天导入与CLI hook不能混称；README明确 Python可选、核心prompt-driven。赞助、DOI和生态列举属于传播，许可证CC BY-NC 4.0，不是MIT式无限商用。

评估文档有重要自我披露：25条合成goldset曾用返回gold标签的stub judge验证工具链，不代表真实线上语义judge可靠。ISO42001文档是“spirit gap assessment”，不是ISO认证。安全、治理、测量材料很厚，必须区分规则测试、stub、实模型测量。

### C. 真程序：引用核验、综述筛选、留痕和宿主约束

506程序文件292,474行，测试148,831行；其余143,643行也有大量检查器、eval、迁移和资产维护。119个 `check_*.py` 的55,870行是其工程重心之一。

- **真实检索/核验**：Semantic Scholar、OpenAlex、Crossref、arXiv客户端，DOI/标题匹配、多源退路、SQLite缓存、verify_citation/verify_passport。当前 arxiv_client 默认是HTTP `export.arxiv.org`，其他路径另有HTTPS，不能笼统改写。元数据匹配证明存在性，不证明引用支持论断。
- **真实筛选产物链**：sr-screener的六个Python与两个JS模板3,156行，解析记录→去重/分批→生成宿主工作流→`agent(...)`判断/第二审阅/争议处理→合并→XLSX或CSV、RIS、PRISMA计数。LLM判断与确定性文件处理各占一段；JS模板需要相应宿主接口，不是任意Node直接跑。
- **真实约束与收据**：scope guard处理PreToolUse输入、限制特定分派角色的Write/Edit/MultiEdit/Bash；main session及未覆盖工具不在此围栏。缺Python/守卫故障有pass-through路径，不能称操作系统级隔离。run ledger、证据行、receipt、提交包manifest验证是真程序；构建投稿包不是自动投稿。
- **真实外部模型接线**：Codex/Claude/agy/Gemini相关transport与面板调用；检索/judge依赖注入必须接真实调用方。`ARS_CLAIM_AUDIT` 默认关闭；模块存在不能推成所有稿件都自动全量审查。

可选key分支不等于服务永久免费。四索引元数据、中文resolver、明确同意后的外部模型请求和更新检查，分别见数据流文档。未见自有通用浏览器接管后端或MCP server；网页搜索主要委托宿主工具。

## 四、nature-skills：读写、全文获取和文档生产工作站

详见 [完整24技能报告](repo-audit/nature-skills.md)。这里的 Nature 是项目命名和投稿/写作取向，不能当成 Nature 出版集团官方认证或论文录用保证。

### A. 给 AI 读：多轴加载与方法边界

nature-writing要求先读manifest，再按 task、paper_type、section、language、journal选择reference。写作、润色、回复、模拟评审、深读、图、文献卡等职责分离；shared是依赖层，不是直接接任务的独立研究员。

这些路由主要由模型阅读与执行，没有Python统一解释器把所有写作分支强制锁死。方法内容包括主张链、Results/Discussion分工、术语账、材料缺口、实验单位/n、效应与区间、多重比较、技术/生物重复。**nature-statistics零独立Python/R，定位是统计报告审查；不能写成内置统计检验引擎。** nature-data也主要做可用性声明/FAIR元数据，不是数据清洗后端。

“daily cron→30候选→六维评分→top5→飞书/Telegram→归档”的 nature-literature-pipeline 有12个文档/配置文件、零独立程序。权重与工作流很具体，但定时、搜索、推送来自Hermes/宿主，`production-tested`是作者叙述，不能凭这句话称仓内有独立文献daemon。

### B. 给人读：商业入口很明确，运行依赖也有明示

README前部直接介绍GPT/Claude代充/成品号、微信客服、知识星球及商务合作；这是可见商业引流，不需要推测。另一面，它也写清安装的是技能文件，Python/R、PDF/PPTX工具、浏览器、MCP和key另配。不能因商业入口否定工具，也不能把README的安装命令当“全能力已部署”。

有双语README、教程、主页、安装wrapper、自动更新说明。SessionStart更新、每小时节流和失败跳过涉及真实本机行为；更新器是软件分发功能，不是科研算法。没有看到与K-Dense同等的全技能scanner汇总，测试/validate不能转述成独立安全认证。

### C. 真程序：MCP、认证浏览器下载、图形审计、PPTX

273程序文件56,855行，测试15,016行。非测试代码重心在检索下载、图形/PPT、专利/排版和文件状态管理。

| 子系统 | 本地实现 | 谁提供关键能力 |
|---|---|---|
| academic-search | FastMCP注册工具→sources→Crossref/PubMed/arXiv/Scopus/ScienceDirect→统一JSON | 外部索引与访问权限 |
| nature-citation | 2,388行CLI，切段、检索、去重、作者完整性、引用映射、RIS/BibTeX等导出、checkpoint | 候选检索是真；全文支持关系需AI/人复核 |
| downloader | provider路由、OA/API、学校配置、PDF验证；CDP认证页面fetch与本地写入 | 已登录Chrome与外部web-access CDP proxy；不自带机构权限 |
| figure | 绘图模板、字体/尺寸/格式、PDF/SVG文字、碰撞/面板对齐审计 | 外部绘图库；几何QA不证明数据真 |
| image2ppt | manifest/对象/资产状态、PPTX构建与校验，约万行程序 | 模型负责图像/对象理解，Python负责文件和状态 |
| paper-trans | 905行页任务/锁/预约/审阅/合成管理 | 翻译与图像化页面来自宿主/外部模型 |
| paper-to-patent / paper2word | 技术材料与DOCX辅助、版式检查 | 专利判断需人；Word核心依赖另装md-to-word引擎 |

最该纠正的细节是 **claim-to-source mapping 的实现**：`nature_citation.py:1196–1246` 是段落生成query→Crossref相关候选→期刊范围筛选→取前N→绑定citation marker，没在这条链看到全文蕴含验证。命中了相关论文与“该论文足以支持这句话”之间，还有阅读判断这一步。

图像PPT模块源码还会读取Codex认证并调用特定ChatGPT backend图像路径，也支持API后端；这不是完全离线生成。本审计未读取或输出任何实际凭据、未调用服务。

## 五、ARIS：编码 Agent 当总控，程序负责工具接线与状态

详见 [分仓逐文件报告](repo-audit/Auto-claude-code-research-in-sleep.md)。197条SKILL路径必须先去掉平台镜像的数量错觉。

### A. 给 AI 读：从选题到实验、评审和成果生产

主链 `research-pipeline/SKILL.md:94` 是 idea-discovery→experiment-bridge→auto-review-loop→可选paper-writing。技能覆盖文献/知识库、可证伪选题、实验计划/消融、算力部署、监视分析、result-to-claim、写作/引用、海报/幻灯、rebuttal，以及证明/Lean、专利等分支。references、templates和共享门禁规定文件契约与失败处理。

auto-review-loop最多四轮，按medium/hard/nightmare选择模型桥、评审记忆或外部编码CLI读仓；score≥6且ready/almost是模型评审终止条件，**不是学术录用或科学正确性证明**。某些兼容路线还要求外部finalizer，caller自报身份仍注明unverified。

默认 `AUTO_PROCEED=true, HUMAN_CHECKPOINT=false, CODE_REVIEW=true, AUTO_WRITE=false, ARXIV_DOWNLOAD=false`。所以“默认一觉出完整论文并下载全部原文”不准确；默认论文写作还需要显式启用与venue。总体流程的执行者首先是宿主Agent，不是一个Python main包办全部研究。

### B. 给人读：大量安装/供应商教程与另一产品发布叙事

README、中英SETUP_GUIDE、各宿主适配、API供应商/GPU/断线恢复/Overleaf等文档很厚，模板/截图/社区论文提供使用路径。微信社区、模型平台教程、案例构成传播入口；未据此发现本仓内建付费订阅计费系统。

重大版本边界：README前部宣传 **ARIS-Code独立CLI、Rust crates、沙箱与流式API**；当前checkout没有 `.rs`/Cargo源码。可以报告README在宣传另一个发行物，不能把那套Rust实现或其测试数字算进本次本地源码审计。

安全材料包括自家策略审计、prompt注入卫生和threat_scan；scanner的clean只表示没命中特定规则，不是语义免疫或独立安全认证。

### C. 真程序：模型桥、检索、GPU队列、恢复门禁与可视化

176程序文件64,975行；测试20,695行，非测试44,280行。路径上包括tools、mcp-servers、技能内脚本、monitor和安装模板；逐字节副本及改写镜像还会扩大总量。

- **MCP/模型桥**：codex-exec、claude-review、antigravity/grok/gemini、model-worker通过CLI子进程或SDK；llm-chat走OpenAI兼容HTTP；manual-review提供人工回填界面；这些是实实现，不是只写“找另一个模型审稿”。
- **真实检索**：arXiv、S2、OpenAlex、Crossref核验，以及Exa/Firecrawl/Keenable/DeepXiv等外部服务封装；默认元数据不是全文深读证明。
- **研究状态与知识库**：research_wiki约1,745行维护节点/边/claims/ideas/experiments；review_gate、run_state、forensics/evidence/provenance等记录stop/continue/escalate、done/accepted/provisional。记录正确不意味着科学主张正确。
- **真实GPU队列**：canonical queue_manager433行，manifest→nvidia-smi→screen启动→日志/OOM→阶段依赖/state。两个tools转发小文件不应重复当调度器；未来Modal/Vast说明不能当本地已完成后端。
- **监视与通知**：watchdog明确detect-only；桌面monitor扫描session，飞书桥发卡片/poll回复。提醒、检测和科学验收不是同一功能。
- **成果工具与安装器**：HTML、poster、图片/PDF资产、版式/风格/gate、LaTeX/Overleaf及各平台安装更新占较多行。它们有用，但不宜统称科研算法。

典型实际过程：人给brief→宿主加载pipeline→模型桥交叉评审/工具检索→宿主改实验代码→队列或Bash部署→看日志/分析→门禁写状态→可选论文与图。它的工程强项是接线、恢复与留证，而非凭模板保证新发现。

## 六、HERO-Anti-OverDefense：减少空转的规则与提醒进程

详见 [分仓报告](repo-audit/HERO-Anti-OverDefense.md)。本仓只有22文件，但不能再写“完全没程序”。

### A. 给 AI 读

H/E/R/O针对无收益Hash、脱离项目的边界防御、清单代替判断、过度脚手架。要求检查能改变下一步行动、持续推进已授权目标，案例只供人反问，不宜整包塞上下文。规则明确不能覆盖用户和项目要求的审查；不是“取消必要验证”的越权指令。

两技能：session-handoff是续会话实物核对、决定/失败/待办/所有权/阅读顺序；goal-pulse是目标推进节奏。作者承认只是helps a little，宿主高优先级/更具体流程可能压过规则。

### B. 给人读

中英README、RULES、host配置落点、案例、量化/ML示范提示词和共享社区入口。配置安装主要是复制规则，不部署科学模型；也没有证据可由案例推出普遍提速百分比。

### C. 真程序

2个Python共707行：生产 `goal_pulse.py`310，测试397。watcher连接已有Codex daemon的Unix socket，JSON-RPC initialize→读取Goal/线程/turn→在同目标active、turn运行中及冷却满足时 `turn/steer` 注入预设提醒；有锁、停止文件/信号和后台进程管理。Claude路径依赖宿主Cron工具，无另一套自有定时服务。

它确实会向正在运行的Agent送消息；不创建研究Goal、不读完整推理、不执行统计、不写实验、不重新裁判科研质量。**小型真工程，服务行为节奏。**

## 七、Anti-Autoresearch：论文取证，不是另一套写论文工具

详见 [11技能+1编排器及全部程序](repo-audit/Anti-Autoresearch.md)。

### A. 给 AI 读：证据账本与分维审计

总流程是ingest→claim/span账本→分角色审计→memo→Python裁决→报告；11支线覆盖一致性、实验取证、baseline公平、评测泄漏、引用、推导证明、展示信号、风格观感、反方反驳与创新/重复建议。每条finding要求锚定账本，不因“像AI写的”直接判造假。

风格观感零诚信权重，创新建议memo-only；PDF/源码/代码观测级分开；覆盖不全不能输出无条件CLEAN；明确不鼓励定时反复重判未变化论文。流程主要由宿主模型扇出角色，Python不是自带所有LLM调用的总控。

### B. 给人读：取证边界、安装陷阱与社区

README/DESIGN/SECURITY/limitations强调不是学术不端定罪，也不是实验复现；承认误报与提取失败。安装Shell只复制技能/workflow，**不复制Python tools**，所以安装技能后仍要定位原仓工具、schema和references；外部reviewer另配。微信社区与说明文档不是程序能力。

### C. 真程序与文档—实现冲突

18程序文件4,493行，测试1,347行；非测试中还有272行合成eval。重要程序：312行claim ledger用regex提取源文/数字/表格/caption/span，183行算术一致性，530行统计检查（GRIM、方差、statcheck重算z/t/F/chi²/r），893行裁决，另有manifest、presentation、countercheck/refutation。t/F等部分依赖可选SciPy，缺失应跳过，不能当通过。

**最关键的源码结论：确定性聚合不等于LLM不影响严重性。** DESIGN仍说观测不足自动降info；当前 `adjudicate_findings.py:361–446` 中，锚点无效会降级，但观测级不足、误报风险、待外核和有依据的反驳主要是加标记/contested，不自动改变原severity。critical/major/minor的起点来自auditor，随后固定映射HARD_FLAGS/SOFT_FLAGS。

因此相同findings必得相同裁决，不代表findings真实；span存在也不证明指控成立。零权重分类和覆盖控制有实际约束，但不能从“确定性”三个字推出结论独立于模型。

## 八、AutoResearchClaw：真Python流水线，但要沿每条分支辨真伪

详见 [分仓报告](repo-audit/AutoResearchClaw.md)、[模块代码账](repo-audit/AutoResearchClaw_code_breakdown.json)。这是八仓中一个明显不同的执行体：有CLI、阶段函数表、状态/重试/实验后端与服务实现，不能和仅由宿主解释SKILL的系统混称。

### A. 给 AI 读：技能与长prompt嵌在真正引擎里

本地41条SKILL路径包括内置及其他参考/演示，不宜全当默认启用；loader读frontmatter、body、enabled和元数据，registry/matcher按阶段/关键词找技能。与此同时Python里的prompts模块约4,815行，本质仍是交给外部模型的指令，不是这4,815行自身会做科学推断。

23阶段：topic_init→problem_decompose→search_strategy→literature_collect→literature_screen→knowledge_extract→synthesis→hypothesis→design→code_generation→resource→experiment_run→iterative_refine→analysis→decision→outline→draft→review→revision→quality_gate→knowledge_archive→export→citation_verify。**有阶段枚举与真实函数分派；语义选题、代码生成、解释与写作仍依赖LLM。**

### B. 给人读：idea-to-paper、一键与展示

README宣传聊天idea到论文、23阶段、测试数字和样例论文；wizard、配置、quickstart、co-pilot、自动批准等降低启动成本。README中的测试通过数和paper showcase不是本次实测；`--auto-approve`也不应解释成科学质量已验收。项目metadata还列出本快照不存在的某些包路径，不能据此凭空补实现或未经构建就断言整包不可安装。

### C. 真程序：调度、执行、统计、联网都存在

423程序文件130,658行；测试40,790行，例程6,810行，候选运行82,032行，另有external与docs/web。pipeline约22,684行、experiment8,402、hitl7,835、agents6,834；不是一个Markdown安装器。

- CLI→execute_pipeline→23阶段function map，保存checkpoint/heartbeat/events、控制重试/修订/pivot。这是可追踪程序调度。
- 检索有arXiv SDK、S2、OpenAlex、Crossref/DataCite等；Tavily有key时调用，失败可退DuckDuckGo HTML；Google Scholar用外部scholarly。
- crawler可用Crawl4AI/headless browser或urllib。与Nature的已认证Chrome/CDP不是同一种浏览器接入，不能说它默认接管用户学校登录态。
- 实验factory支持本地subprocess、Docker、SSH及若干适配器；外部coding CLI、远端/云依赖各自环境。Colab Drive路径可生成worker供人跑，不是凭配置就自动获得云GPU。**Docker不可用的某条路径会退回本机subprocess，不能默认始终容器隔离。**
- analysis对多seed值计算均值/std/bootstrap95%区间、成对比较t检验等；这是实际数值计算，但baseline选择、种子独立性、对照设计不由公式自动保证。
- verified_registry可核数字是否匹配注册值/容差；它不能证明数字对应语境、实验未造假或结论有因果效力。

### D. 四个必须写出的执行边界

**第一，显式模拟模式存在。** `ExperimentConfig.mode`数据类和缺失字段解析默认是simulated；但示例配置明确是sandbox，`cmd_init`复制该模板并保留实验mode，不能把字段缺省扩大为普通安装用户都跑模拟。`_execution.py` 的simulated分支明确写status/notes，并按固定公式生成指标。必须读最终有效配置与产物状态，不能把示范数值当真实实验结果。

**第二，文献失败有placeholder分支。** 所有搜索失效时会生成 `[Placeholder] Study…`、example.org、`is_placeholder:true`；BibTeX生成会跳过placeholder，结果还记录real_search状态。因此不能说它悄悄将全假引用当真引用，也不能因为阶段返回完成就认定真实文献已获得。

**第三，质量门有强制推进路径。** 达到pivot上限/连续空指标等条件时，程序可能强制PROCEED并写quality_warning。它有失败标记与门禁，不代表任何坏结果都会永久阻断后续写作；这些推进路径也不等于后面的stage20/23门禁全部被跳过。论文成品可生成与科研证据足够，是两个不同条件。

**第四，MCP名义覆盖大于当前方法实现。** server中run_pipeline返回mcp-stub标识，literature/review也有stub；读取已有状态/产物是真的。当前`cmd_mcp --start`只执行置标志/日志的start随后返回，没有接入transport消费循环；SSE路径明确未实现。FastAPI serve是另一套真实HTTP路径，不该一起抹成stub；但其FastAPI/uvicorn依赖又未列入错误提示推荐的当前`[web]`组，安装后能否启动仍需验证。逐方法而不是按目录名判能力。

## 九、Orchestra AI-Research-SKILLs：很厚的工程手册，执行体主要在宿主

详见 [98技能、安装器和路由失配报告](repo-audit/AI-Research-SKILLs.md)。

### A. 给 AI 读：98份知识与双环科研

23类覆盖模型/分词/微调/解释/数据/后训练/安全/分布式/GPU云/优化/评测/推理/MLOps/Agent/RAG/提示/可观测/多模态/新兴技术/写作/选题/ARA。大量内容是如何正确使用第三方工程框架，非本仓重新实现这些框架。

autoresearch内环：选假设→可检验预测→**先commit protocol再运行**→复现baseline/检查数据与NaN→测指标→区分confirmatory/exploratory→记录接受/否定→更新state。外环跨实验综合、消融、转向、findings与论文。这是有内容的科研纪律，不能因载体是文字就说没有方法。

但持续执行的关键是 `SKILL.md:244起` 要求宿主 `/loop 20m`；OpenClaw调用宿主 `cron.add`。当前仓没有自己的研究daemon/统一worker/持久队列。向Telegram/Slack等发进展也是给有对应工具的宿主下指令，不是本仓带了通信SDK。

ARA“compiler”明确写“You are the compiler”，让模型用Read/Write/Edit/Bash把材料生成结构化研究资产；schemas/checklist约束格式。没有独立ARA编译器程序。不能将角色命名当传统compiler实现。

### B. 给人读：production-ready、短安装路径与传播

README、WELCOME、官网/博客/Slack/X/LinkedIn、GIF/截图、Remotion宣传片构成展示。WELCOME还残留86与98的版本漂移。LaTeX模板是实用资源，不是研究发现算法。根包ISC与实际子包MIT/版本不同，应按真实发布包判断，不凭根manifest概括。

### C. 真程序的完整分账

| 文件组 | 文件 / 行 | 做什么 |
|---|---:|---|
| npm CLI | 6 / 2,608 | 检测宿主目录、菜单、git clone/复制/symlink、更新卸载与lock |
| inventory检查 | 1 / 66 | 清单维护 |
| GRPO模板/奖励 | 2 / 621 | TRL/Transformers/PyTorch执行示范训练 |
| 绘图demo | 3 / 938 | matplotlib固定展示与Gemini架构图生成 |
| Remotion宣传片 | 12 / 1,806 | 安装/分类/终端等视频画面 |
| 合计 | 24 / 6,039 | 不能全算自动科研执行器 |

安装98技能不安装98软件框架；安装器本身无模型研究SDK/浏览器driver，主要从GitHub分发文件。训练模板确能在环境齐备时跑一个真实任务，但不是98框架统一集成。

**路由实物有失配**：`references/skill-routing.md` 79个目录引用中17个本地不存在，包括部分重命名/错分类和缺目录。逐条见分仓报告。模型可以搜索修正，但沿表直接Read会失败；这比“它只是文字”更具体，也更可复核。

## 十、八仓放在同一张执行账本上

以下“有”表示找到实际源码路径；不表示本次运行通过。“宿主”指仓内主要提供让外部Agent执行的协议；“局部”不能外推所有功能。

| 仓库 | 研究总控在哪里 | 自有联网/模型接线 | 真实数值/实验代码 | 浏览器 | 自有周期/队列 | 主要产物 |
|---|---|---|---|---|---|---|
| K-Dense | 宿主按专科技能串联 | 有多客户端/生成API；另有纯文档配方 | 有统计、组学、分子/物理包装等 | 部分技能委托宿主 | 无统一研究daemon | 数据分析、图、证据/材料包 |
| academic | 宿主+人阶段确认 | 有四索引与外部模型transport | 筛选/验证/留痕为主，非通用实验器 | 主要宿主 | 非统一无人值守引擎 | 文献矩阵、稿件、审稿/修订、PRISMA |
| nature | 宿主manifest路由 | 有FastMCP、来源客户端/图像后端 | 文档/图形工程为主；统计审查无程序 | 有CDP下载，依赖外部proxy/登录 | 文献cron主要宿主 | 引用、原文、图、PPTX/DOCX/PDF |
| ARIS | 宿主读研究pipeline | 有多个MCP/CLI/API桥 | 队列部署/监视/分析工具；研究代码由Agent按项目写 | 部分外部服务/宿主路线 | 有GPU队列、监视/状态 | 实验与证据日志、可选论文/海报 |
| HERO | 行为规则/交接 | 本机Codex RPC，不是研究API | 无科学计算目标 | 无 | 真watcher；Claude用宿主Cron | 推进提醒、交接材料 |
| Anti | 宿主分维审计+Python裁决 | Python工具无直接LLM API | 有算术/统计取证，不做原实验 | 无自有后端 | 明确避免未变化材料定时重判 | claims/findings/裁决报告 |
| Claw | 真Python23阶段引擎+LLM | 有LLM、检索、爬虫；部分MCP方法stub | 真subprocess/Docker/远端与统计分析，也有显式模拟 | Crawl4AI/headless | 真阶段/重试/恢复 | 实验、稿件、引用与导出 |
| Orchestra | 宿主双环prompt | 安装器GitHub；图demo有Gemini | 少量GRPO/奖励/绘图示例 | 无自有后端 | `/loop`/cron委托宿主 | state/findings、外部实验、论文模板 |

### 1. “真实方法论”与“强制执行机制”是两层

protocol先提交、分清探索/验证、baseline公平、来源定位、覆盖分母、反例/消融、原始数据/主张对应，都可以是有用方法论。写进SKILL让宿主更容易遵守，是软约束；程序明确检查、拒绝、记录或限制工具，是硬一些的约束。两者都不能自动保证模型对科学问题理解正确。

因此不能仅按“Markdown多不多”判价值，也不能按“Python行多不多”判自动化。应问一个具体动作：**谁触发它，输入来自哪里，源码哪一函数真正做，失败会怎样，输出如何验收。**

### 2. 多模型审稿不等于独立证据

多个角色、不同CLI、甚至不同供应商，可以减少某些盲点；但模型可能共享错误、读同一虚假材料，且severity/评分依赖模型判断。ARIS记录reviewer来源，academic要求评审维度与收据，Anti约束证据锚点，都改善可追踪性；它们不把模型评分变成外部真值。

### 3. 引用链要分四级

| 级别 | 实际问题 | 代表实现/边界 |
|---|---|---|
| 找到候选 | 有没有相关标题/DOI/摘要 | 索引检索、段落query |
| 核对身份 | 这篇论文/作者/年份是否存在 | DOI/标题多源核验、缓存 |
| 核对支持 | 全文这段是否支持稿中主张、范围是否一致 | 多数仍依赖AI/人阅读；Nature映射不等于蕴含验证 |
| 核对真实性 | 原始实验是否真的这样做、数据是否可靠 | 通常超出这些工具可观测输入；需数据/代码/复现实验 |

不能拿第一/二级命中率宣传成第四级科研真实性。

### 4. 统计也要分四种

K-Dense确有功效/诊断/领域分析程序；Claw会对seed指标做统计汇总；Anti重算报告中的算术/检验一致性；Nature统计技能主要审查文字报告。**计算新分析、汇总实验、检查已报告数字、指导模型选检验，四者不能揉成“内置统计引擎”。**

### 5. API、账户、浏览器与科研硬件另算

公开元数据API、按key收费检索、机构订阅全文、外部模型、GPU/SSH、实验室下单是不同资源。源码允许keyless分支只证明支持无key请求，不证明外部服务当前免费或可用。安装技能不提供Scopus/ScienceDirect学校访问权、已登录Chrome、模型订阅、Docker daemon、GPU或机器人。

商业入口也应实话实说：Nature首页直接做代充/社区引流；K-Dense导向BYOK/工作台；ARIS生态有社区和供应商教学；Orchestra有品牌/社交和宣传视频；academic有赞助。它们是产品传播事实，不能单靠它们判项目有效或无效。

## 十一、379期可直接采用的事实修正表

| 容易写出的说法 | 本次实物支持的写法 |
|---|---|
| 八个都是提示词包，没真代码 | 各仓差异很大：有领域数值工具、MCP/模型桥、Python引擎，也有宿主驱动协议与安装器 |
| 有几十万行Python，所以它能自主科研 | 先拆测试/检查器/安装/排版/prompt与真正实验；自动判断与证据仍多依赖LLM/人 |
| ARIS有197种独立技能 | 87主技能+平台镜像/覆盖，共197条SKILL路径 |
| ARIS这个clone含Rust独立科研CLI | README宣传相关发行物，但当前checkout无Rust/Cargo源码 |
| HERO纯Markdown | 核心是规则；另有310行Codex watcher，实发turn/steer提醒 |
| Anti最终严重性完全不受模型影响 | 裁决映射确定，severity仍继承auditor；多项保护现在只注释不降级 |
| Nature内置统计检验、每天自己跑文献daemon | 统计/文献管线主要是宿主执行协议；检索MCP/认证下载/图形/PPT则有真程序 |
| claim-to-source就是论文全文证实 | 当前Nature程序是候选检索与段落绑定；全文支持需再核 |
| Orchestra自带自动科研后台 | 持续执行依赖宿主`/loop`/cron；自有CLI主要分发技能 |
| 安装98技能就装好98框架 | 只分发Markdown与配套资源，框架/模型/账户另装另配 |
| Claw所有MCP工具都接通流水线 | 部分方法是stub，真实CLI/阶段/HTTP服务需分别看 |
| Claw产了指标就有真实实验 | 检查最终mode与status，显式simulated分支生成公式指标 |
| Claw有质量门，所以弱证据不可能成稿 | 上限/空结果等路径可能强制推进并写warning |
| K-Dense绿CI说明177技能全面跑通 | CI明确不运行所有隔离科学依赖套件 |
| 扫描报告safe等于没有风险 | 是某版本、某规则/模型和覆盖下的分类；本次未重扫或认证 |

这些项目不需要被整齐归为“骗局”或“自动科学家”。更有信息量的解剖是：**它们把原先由人记忆的工作方法写进上下文，把可机械化部分写进程序，再借用户已有Agent、模型、账户、软件与硬件拼出科研流程。各仓的拼接深度、实执行比例和失败处理差别很大。**

## 十二、分仓明细与复核材料

| 报告/文件 | 覆盖内容 |
|---|---|
| [scientific-agent-skills.md](repo-audit/scientific-agent-skills.md) | 177技能完整库存、代码/测试、依赖、CI/扫描、数值/API代表链 |
| [academic-research-skills.md](repo-audit/academic-research-skills.md) | 五技能、模式/角色、验证/筛选、数据流、目录与证据表 |
| [nature-skills.md](repo-audit/nature-skills.md) | 24技能、manifest路由、商业入口、MCP/CDP/PPT/引用实现 |
| [Auto-claude-code-research-in-sleep.md](repo-audit/Auto-claude-code-research-in-sleep.md) | 主技能与镜像、桥/队列/门禁/安装器、逐代码文件账 |
| [HERO-Anti-OverDefense.md](repo-audit/HERO-Anti-OverDefense.md) | 行为规则、交接、watcher调用链、全部代码 |
| [Anti-Autoresearch.md](repo-audit/Anti-Autoresearch.md) | 11维审计、全部工具、统计与裁决文档差异 |
| [AutoResearchClaw.md](repo-audit/AutoResearchClaw.md) | 23阶段引擎、实验后端、模拟/回退/stub等边界 |
| [AI-Research-SKILLs.md](repo-audit/AI-Research-SKILLs.md) | 98技能、双环/ARA、程序完整分账、17条失效路由 |
| [skill_catalog.md](repo-audit/skill_catalog.md) | 全部556条SKILL路径，保留镜像/覆盖/参考项 |
| [inventory_summary.json](repo-audit/inventory_summary.json) | 八仓统一文件/行数与快照状态 |
| [repository_roots.json](repo-audit/repository_roots.json) | origin、完整HEAD、顶层目录 |
| [inventory_repos.py](repo-audit/inventory_repos.py) | 不导入上游代码的可复算库存脚本 |
| [python_logic_metrics.py](repo-audit/python_logic_metrics.py) | AST只读辅助统计，非运行测试 |
| [verification.json](repo-audit/verification.json) | 附件/链接、清单条数、clone HEAD/工作树与忽略状态复核 |

分仓报告的源码证据路径相对相应clone，带行号以便本机逐行复查。完整逐文件manifest压缩存储于 `repo-audit/inventory_files.tsv.gz`，可由库存脚本重建。clone在本机被 `.gitignore` 排除：报告链接能在本机追源码；未将八仓复制进本项目Git，因此未来公开阅读不能假定相对clone链接会存在，复核时应按origin+HEAD获取相同快照。本次未执行 `git add`、未改动预先存在的379稿件或其他用户文件。
