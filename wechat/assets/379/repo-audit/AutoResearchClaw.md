# AutoResearchClaw：23阶段科研程序的三分类深度解剖

本地HEAD `be4ba4755bf1b52220f25e13b2293b5956590070`；2026-11-23静态审计。未安装、未执行上游程序、未连接API、未验证README“2699 tests passed”徽章。以下路径相对本仓；引用当前实现，不将宣传视为运行证据。

## 完整家底及统计口径

中央inventory：756个tracked regular files，约52.6MB；95个Markdown共20,995行，41个SKILL.md。423个代码文件130,658物理行（含空行/注释/docstring）：403.py共128,957行、10.sh共933行、10.js共768行。按目录启发式：运行候选280文件82,032行，测试105文件40,790行，experiments27文件6,810行，external1文件258行，frontend-legacy10文件768行。候选数不是可用功能认证；Python中的提示词字符串、适配层/配置/未接通stub照样占代码行。

核心模块代码细账见同目录 `AutoResearchClaw_code_breakdown.json`；所有仓逐文件账在`inventory_files.tsv.gz`。重要部分：pipeline22,684行、experiment8,402行、hitl7,835行、prompts4,815行、literature3,295行、mcp494行。文件包括研究执行、代理角色、LLM接线、人机协作、Web服务、知识/记忆、文献检索、技能库、配置与CLI，而不只是SKILL包装。

顶层researchclaw为主包；tests、experiments/arc_bench为测试及跨域benchmark脚本；external/agents含外部科研代理说明和技能（不代表外部系统全部实现都已vendor进来）；frontend-legacy、website、image为前端/展示；docs有中文及另外八种译本、HITL、integration、论文showcase等；scripts有部署/维护入口。pyproject wheel列researchclaw/sibyl/arc，但列名不应直接当成当前目录都存在的证明。

## 第一类：给AI读——技能、内嵌prompt和角色

这里AI指令不止Markdown：`researchclaw/prompts`的Python提示词与agents角色通过程序进入LLM上下文，SKILL只是可插拔的额外方法论。41份SKILL分为20份主包内置、9份`.claude/skills`（宿主入口/选用集）、12份external/agents里的生物/统计配套技能；不能合计为41个独立科研领域。

内置20份完整名称：pytorch-training、distributed-training、mixed-precision、data-loading；nlp-pretraining、quantum-qiskit、cv-detection、nlp-alignment、rl-policy-optimization、chemistry-rdkit、cv-classification、biology-biopython；literature-search、scientific-visualization、hypothesis-formulation、systematic-review、meta-analysis、statistical-reporting、scientific-writing、experimental-design。它们给训练/领域库与科研规范提示，并不把PyTorch/Qiskit/RDKit/Biopython安装实现都包进SKILL。

`skills/loader.py:18-100`读取SKILL frontmatter及body、支持enabled=false、保留source_dir；:117起支持旧YAML/JSON，SKILL优先。`skills/matcher.py:24-94`按阶段适用性、关键词重叠、描述回退(0.5折扣)、priority排序取top_k，:97起把body裁到预算注入prompt。这是**词项路由**，不是另训练的科研语义router。Markdown指导实验设计/统计写作，Python状态机确保阶段有顺序；领域结论仍由LLM及真实实验承担。

## 23阶段真正调用链

`pyproject.toml`注册researchclaw→`researchclaw.cli:main`；`cli.py:156`的cmd_run加载配置、处理resume/HITL/自动批准，:426调用execute_pipeline；`pipeline/runner.py:431`入口及:483后循环调各阶段implementation，写产物、状态与heartbeat。`pipeline/stages.py:23`起枚举：

|阶段|动作|主要性质|
|---|---|---|
|1 TOPIC_INIT / 2 PROBLEM_DECOMPOSE|建题目、拆子问题|LLM+结构化文件|
|3 SEARCH_STRATEGY / 4 LITERATURE_COLLECT|检索式、文献API收集|LLM规划+真网络客户端|
|5 LITERATURE_SCREEN / 6 KNOWLEDGE_EXTRACT|筛选、抽知识|门禁+LLM|
|7 SYNTHESIS / 8 HYPOTHESIS_GEN|综合缺口、生成假设|LLM|
|9 EXPERIMENT_DESIGN / 10 CODE_GENERATION / 11 RESOURCE_PLANNING|设计、生成代码、资源计划|门禁+LLM/外部coding agent|
|12 EXPERIMENT_RUN / 13 ITERATIVE_REFINE|执行、迭代修复|依mode为真实程序或显式模拟|
|14 RESULT_ANALYSIS / 15 RESEARCH_DECISION|统计、决定proceed/refine/pivot|真计算+LLM/规则|
|16 PAPER_OUTLINE / 17 PAPER_DRAFT / 18 PEER_REVIEW / 19 PAPER_REVISION|大纲、成稿、互评、修改|LLM生成与模型审稿|
|20 QUALITY_GATE / 21 KNOWLEDGE_ARCHIVE / 22 EXPORT_PUBLISH / 23 CITATION_VERIFY|质量门、归档、导出、引用核验|混合规则/LLM/工具|

stages.py将5/9/20设gate，拒绝有回退映射；stage15最多2次pivot/refine。这已是能脱离Claude解释SKILL独立跑的Python状态机，明显区别于Markdown清单。但“23阶段”本身不证明每阶段都输出真实科学证据。

### 一：模拟缺省与示例实际默认必须分清

`config.py:585`数据类ExperimentConfig.mode='simulated'，:1345解析缺失mode也回退simulated。`stage_impls/_execution.py:449-476`显式模拟分支按task序号造primary=0.3+idx*0.03、secondary=0.6-idx*0.04，写status='simulated'、notes='Simulated run result'。这些不是运行模型/算法得来的实验数据，不能用于宣称实测科研成功。

**但不能因此写成普通安装用户默认必然跑模拟。** `config.researchclaw.example.yaml:94-99`明确mode='sandbox'；`cli.py:820-840`的cmd_init寻找并复制该示例，后续向导主要改provider/key/model，保留实验mode。cmd_run不把所有输入配置强改sandbox；省略mode的手写/第三方配置仍会落simulated。README所示full-auto是project.mode的阶段批准策略，和experiment.mode不是同一个开关；CLI --mode又是HITL模式，不可混为一谈。

### 二：pivot上限是终止机制，不是质量达标证明

`runner.py:699-815`有真实回退/版本保留；连续refine无指标会强制PROCEED，最多2次pivot后仍强制PROCEED，可写quality_warning.txt，明说“Paper will be written but may have significant issues”。还会promote最佳stage14结果。这里允许进入写作阶段，不等于后续stage20/23所有门禁也被跳过；不过只看“成稿产出”会高估研究成功率。质量门文案与证据验证能力必须分开。

### 三：检索失败占位有显式标签，也不是实文献

`stage_impls/_literature.py:568-584`所有来源失败后造至少20条占位，title带[Placeholder]、URL是example.org、is_placeholder=true。:592-594生成BibTeX时跳过；:650附近search_meta记录real_search等来源，不把这批统一冒充真实检索。但即便能维持流水线形状，占位不是“真的找到20篇相关论文”，下游筛选/引用门仍须识别来源。本审计不将标签机制等同所有下游永远不误用。

## 第二类：给人类读——产品包装、教程与依赖

README以“One topic in, one paper out”、23阶段、paper showcase、2699测试、ACP多宿主/OpenClaw消息平台、HITL六模式等说明产品（README:27,74,121,243,344）。docs多语言说明、HITL_GUIDE与integration、image/website/前端、benchmark脚本和论文展示帮助安装、展示与社区传播。Discord入口在README:30/46；没有据此确认内建付费订阅/计费业务。第三方API/云算力费用仍由用户供应商产生，社区展示不是独立效果审计。

`pyproject.toml`基础Python>=3.11，核心仅pyyaml/rich/arxiv/numpy；httpx、scholarly/crawl4ai/tavily、PyMuPDF、HF/matplotlib/scipy是可选包。**serve实际还需要FastAPI/uvicorn，当前[web]列项并不包含这两项**；cmd_serve的“pip install researchclaw[web]”提示不能代替依赖齐备证明。需真实安装验证，本文没有运行。

## 第三类：真实程序做了哪些实事

- **LLM/API与ACP**：llm/client.py:115定义OpenAI兼容endpoint，:128-175从配置/env取key、选provider/bridge；:187起reviewer可指定独立provider/model，否则可复用主provider/key，不能自动宣称模型评审独立；:538-573 urllib HTTP请求与fallback。anthropic_adapter.py:127走/v1/messages；acp_client维持外部agent会话；模型预设有OpenAI/OpenRouter/DeepSeek/Anthropic/AtlasCloud/MiniMax/Ollama等。API接通是生成能力来源，不是本地训练科研模型。
- **文献与网页**：literature/arxiv_client、semantic_scholar、openalex_client、cache/search/novelty/verify；verify实际调用arXiv/Crossref/DataCite/OpenAlex等核元信息。web模块包装scholarly、Tavily、crawler；可选依赖未装会影响路径。文献真实性核验不等于核验论文实验真实性；没有把本仓泛称“全功能接管浏览器”，源码支持的是具体检索/抓取路径。
- **实验本地/远程执行**：experiment/sandbox.py:314-339及:428-444用subprocess.run、cwd/env/超时捕获、stdout/stderr及指标解析；不是Markdown假装运行。factory.py:18起按sandbox/docker/ssh_remote/colab_drive/外部biology/stat/collider模式选后端。Docker不可达时warning后退回本机subprocess(:28-34)，隔离保证会改变；agentic专用工厂:120后则Docker不可达直接raise，不能混说所有后端都静默降级。SSH需host/connectivity，Colab需Drive可用及另跑worker。sandbox命名不代表操作系统强隔离，静态检查/超时和容器是不同保护层。
- **真统计**：stage_impls/_analysis.py:341-377对至少3个seed值算mean/stdev、固定Random(42)做1000次bootstrap 95% CI，异常时退mean±1.96SE；:386-430按共同seed配对比较，SciPy t尾概率；缺SciPy时正态近似fallback。该部分是真数值程序，但统计正确性还依赖独立seed、实验设计、指标输入与模型生成代码是否真实。
- **HITL/服务/附属工程**：hitl约7,835行含六类干预策略、审批/暂停/成本界限、适配器与状态；server是真FastAPI项目API/Chat/WebSocket，cli.py:702-728实际create_app+uvicorn.run；Overleaf有Git同步；memory/knowledge、assessor/agents、dashboard/calendar等是支撑程序。功能存在不代表每个外部平台已端到端验证。

### 四：MCP公开入口尚有stub，不能与真实Web服务混写

`researchclaw/mcp/server.py:72-81`run返回mcp-stub run_id；:104-118search返回空结果及stub消息、review是stub；:136-139 start仅置_running并log，不消费transport。`cli.py:642-655`mcp --start asyncio.run(server.start())后return，当前调用链不会变成常驻MCP服务。transport.py:76-94 SSE明确stub；client.py:85也写stub。

这并不否认Web服务：`cmd_serve:702`是真uvicorn运行；另有HITL MCP adapter也不能拿同名概念替代这个对外research MCP工具面的缺口。README/API的“支持MCP”必须精确到已实现的那部分，不能以存在tools schema就判完整互联。

## 最终判读

此仓是八仓中具有完整独立科研运行时的一类：真CLI、23阶段状态机、模型API、文献检索、代码生成/子进程实验、统计、交互与论文输出。不能贬成只写Markdown；也不能按篇数/阶段数宣布科学可靠性。显式模拟数据、占位文献、有限次回退后继续、MCP未接通、Docker回落和可选依赖都是必须记录的真实执行边界。标签/门禁/日志增加可审计性；它们既不把模拟变实测，也不证明LLM主张成立。
