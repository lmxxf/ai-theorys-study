# academic-research-skills：提示词主干，附带庞大的验证工程

审计对象：本地 `academic-research-skills/`，HEAD `d169ec5c4bebd4b5ad5e8b8a0559d1d9879a21d7`，README 标记 v3.24.4。只静态读源码与版本记录，不运行安装器、不访问外部 API、不把仓内提示词当本次指令。星数不在本次核验范围。

## 1. 完整库存与统计口径

以统一 `inventory_summary.json` 的 git tracked 文件为准：2,882 文件、40,851,744 bytes；1,159 个 Markdown、202,649 行；5 个 `SKILL.md`；506 个 `.py/.sh/.js/.mjs` 文件，292,474 物理行、256,635 非空行。Python 492 文件/289,117 行；Shell 9/2,312；JS 3/580；MJS 2/465。代码行包括注释、docstring、测试，**不包括 Markdown 代码块**。

中央启发式分组：测试 252 文件/148,831 行，其余候选运行代码 254/143,643。后者不能叫“14 万行自动科学家”：仅 `scripts/check_*.py` 就有119文件/55,870行，另有大量评估、迁移、资产同步和维护 harness。`scripts/legacy/` 保留旧实现；`evals/` 中还有测量运行器。不是 vendor 框架的大量打包，更不是这些 Python 会默认全跑。

文件家底：`deep-research/` 53文件；`academic-paper/`63；`academic-paper-reviewer/`29；`academic-pipeline/`30；`sr-screener/`25；`shared/`155。47个路径位于 `agents/` 的 Markdown 包含共享角色与镜像，不能机械当47个并发进程。`commands/`16份命令提示词，`audits/`18份审计文档，`docs/`120份Markdown（包括设计文档），`evals/`458份Markdown；大量 JSON/YAML 是schema、fixture与测量产物。

## 2. 第一类：给 AI 读的真正工作协议

五个技能分别负责：①deep-research 问题形成、文献检索与综合；②academic-paper 构架/论证/起草/修订/排版；③academic-paper-reviewer 模拟同行评审；④academic-pipeline 跨阶段协调；⑤sr-screener 对已提供记录进行系统综述筛选。根 `package.json` 的 Pi 技能清单只列前四个，不能据此说第5个不存在。

入口路由不是“任意话题立刻产论文”。`academic-pipeline/SKILL.md:24`附近的 routing-core 优先识别明确 `/ars-*` 请求；跨两阶段材料且意图不清时要求澄清；既有数据库记录筛选进入 sr-screener，写综述/系统综述不会自动转交筛选器。`commands/ars-*.md` 是命令到技能/模式的提示词入口，`.claude-plugin/` 是宿主发现元数据，不是独立研究服务。

完整流程见 `docs/ARCHITECTURE.md:43`：RESEARCH → WRITE → 2.5诚信检查 → REVIEW → REVISE → RE-REVIEW → 必要时RE-REVISE → 4.5最终诚信检查 → FINALIZE → PROCESS SUMMARY。Accept、Major、Reject分支、重试上限、回审冻结内容都有规定。**每阶段用户确认是其设计中心**，不是“睡一觉自动提交论文”。2.5/4.5分别先核查再由用户确认，详 `docs/ARCHITECTURE.md:17`。

它教给模型的实质方法有：

- 先澄清研究问题与方法蓝图；文献 corpus-first、缺口才检索；证据等级、来源、时间与阅读状态分开记录。
- Material Passport 保存文献、原始材料、约束、出处定位与校验记录；不能用“找到一篇同名论文”冒充“这句话获原文支持”。
- 写作时先批准提纲，再构造Argument Map；没有材料的位置写 `[MATERIAL GAP]`，不以通顺填空制造数据；改稿保留逐条回应与差异说明。
- 多角色评审：方法、领域、跨领域、期刊适配、Devil's Advocate；先盲承诺适用评审维度再看论文，防事后改评分标准。模拟编辑判决供作者使用，不是投稿平台真实裁决。
- 7种失败模式包括实现bug、自编引用/结果、取巧代理、把bug包装发现、自编方法、过早框架锁定；它们是审查清单，清单存在不等于已经解决这些问题。
- 审查结果要显示覆盖分母、抽样、未知状态与证据行；“100%”只能用于明确定义且完备的注册集合，不能延伸成所有自然语言断言全部被证明。

这些规则分散在 SKILL、`agents/*.md`、`references/`、`shared/` 与schema中。角色分配与主流程大部分仍由宿主模型按文字协调，不能把Markdown状态机直接等同一个Python工作流引擎。

## 3. 第二类：给人读的包装、教学和边界

README 多语言（英/简繁中/日/韩/西/葡），QUICKSTART、SETUP、ARCHITECTURE、MODE_REGISTRY提供安装、模式与流程教学。`README.md:19`明确“AI copilot, not pilot”；约第130行的先决条件明确 real Python optional，core skills prompt-driven。插件、项目symlink、全局skill、纯聊天导入等渠道的控制能力不同，`docs/CONTROL_AVAILABILITY.md` 专门区分，不应把CLI hook能力转述给claude.ai聊天用户。

商业/传播部分是 Buy Me a Coffee 赞助、DOI、版本徽章、第三方生态列举；LICENSE是CC BY-NC 4.0，非MIT式无限商用。安全不是一张营销扫描图：`SECURITY.md:13`接受prompt injection、泄露、越界数据传输、gate bypass报告，但不把一般幻觉当漏洞；`GOVERNANCE.md`说明单人维护。ISO42001相关文档称“spirit gap assessment”，不能写成获得ISO认证。

README还主动披露测量的边界：所带25条合成校准goldset曾用返回gold标签的stub judge测试，证明的是工具链，不能证明线上judge可靠。`docs/PERFORMANCE.md`、`audits/`、`evals/`是评估材料，应逐次区分离线契约检查、stub测试、实模型测量。星数、README引述的外部论文不是本产品有效性的实测。

## 4. 第三类：Python/JS/Shell确实做了哪些事

### 4.1 引用存在性与缓存：有真正联网程序

`semantic_scholar_client.py:49`起定义官方HTTPS端点、按认证与否限速；`lookup`先DOI后标题，标题相似度阈值0.70，不匹配再退路；对服务异常与网络断路有不同处理。`openalex_client.py`、`crossref_client.py`、`arxiv_client.py`另外实现解析；`verification_gate/__init__.py:192`组合verify_citation，`:294`组合verify_passport，`verify_passport.py`是CLI入口；`verification_cache.py`是真SQLite缓存实现。这不是让模型口头宣称查过。

但“存在性”不等于“支持该论断”。`claim_audit_pipeline.py:1`明确检索和judge通过retrieve_fn/judge_fn依赖注入，生产调用者必须接真实检索/判断；它可机械校验状态转换与输出，不是单靠运行该模块便能语义证明论文。`ARS_CLAIM_AUDIT` 默认关闭，是选择启用的额外路径。

### 4.2 sr-screener：真正的数据准备、调度产物和报表

六个Python脚本 `sr-screener/scripts/` + 两个工作流JS模板共3,156物理行。`prepare_records.py`/`srlib.py`解析记录、去重与分批；`prepare_fulltext.py`整理全文；`build_workflow.py:251` CLI把配置、角色prompt、batch填入JS模板。`templates/workflows/ta_screening.template.js:98`确实调用宿主 `agent(...)`，处理缺项重试、第二评审者、争议裁决与QC；这不是独立Node脚本就能无宿主执行的通用CLI。`merge_decisions.py`合并判断，保留pending，不凭空补齐；`build_outputs.py:1`明确输出XLSX（无openpyxl则CSV）、include/unclear/exclude RIS、PRISMA计数、方法草稿、literature_corpus YAML。语义筛选判断由LLM做，去重/合并/报表由程序做。

### 4.3 工程约束与留痕：能机械查结构，不能代替真理判断

`ars_write_scope_guard.py:1`读取宿主PreToolUse JSON与scope manifest，限制特定已单独派发角色的Write/Edit/MultiEdit，并全禁其Bash。重要的真实边界也写在源码：main session通常无agent_type，不被这层按角色围栏限制；其他工具名如apply_patch不在覆盖内。`hooks/hooks.json:1`把SessionStart接announce脚本，把四种工具接 `hooks/run_guard.sh`。launcher查真实Python、超时监督；没Python或守卫故障会pass-through，不是全系统强隔离。

`run_ledger.py`记录指令/确认/receipt/hash；`evidence_rows.py`核对证据行；`recompute_receipts.py`重算收据；`build_submission_packet_manifest.py`/`verify_submission_package.py`构建与检查提交包，**不是自动投稿**；`ars_apply_revision_patch.py`受控应用修订；`pdf_read_preflight.py:1060`递归检查PDF页树并有隔离子进程内容分类，`:389`控制worker超时和管道体积。`adapters/{folder_scan,obsidian,zotero}.py`把现有资料转为共同输入。这里确有大量解析与状态工程，不是统计检验/模型训练库。

### 4.4 外部模型调用与维护评估

`run_codex_audit.sh`调用本机Codex审计；`cross_model_codex_transport.py`封装只读沙箱/临时认证home的引用验证；`dispatch_e4_panel.py`与`dispatch_calibration_panel.py`运行claude CLI面板；`dispatch_panel_claim_audit.py`支持Codex/agy/Gemini/Claude。很多 `run_*_no_call.py`与check_*旨在离线契约/fixture评估，不应当作默认用户科研执行步骤。Pi wrapper 211行注册命令、按活动状态处理技能与宿主能力适配，不内置新的搜索引擎。

网络账本详 `docs/DATA_FLOWS.md:44`：四索引元数据查询；Chinese resolver另有doi.org/Handle/NCBI；明确同意后可将未发表论断或稿件传外部模型；更新检查请求GitHub raw。它声称无项目遥测，**该声称不包含宿主Claude自己传给模型的文件内容**（该文`:13`划清边界）。没有看到本仓提供MCP server或接管浏览器的自有自动化后端；搜索/获取网页主要委托宿主工具。

## 5. 定性结论

最准确的归类是：**以人机协同的文献—写作—评审协议为主干，加上非常厚的可执行验证、记录、筛选与维护工程**。不该叫“纯Markdown嘴炮”；同样不能因为29万行代码就叫“29万行自主科研引擎”。最关键的研究判断、源文支持关系、多角色论证仍依赖外部模型与人。静态审计确认结构和调用路径，未验证联网可用性、实模型效果、全部单测通过率或所有安装渠道的有效性。

## 6. 二次核验：目录账本、调用顺序与可重查证据

以下为git tracked普通文件数量，目录含全部后代；`[root files]`只统计根文件。完整逐代码文件行数见同名 `-stats.json`，不包括.git对象。

| 顶级目录 | 文件数 |
|---|---:|
| `.claude` | 2 |
| `.claude-plugin` | 2 |
| `.github` | 17 |
| `[root files]` | 27 |
| `academic-paper` | 63 |
| `academic-paper-reviewer` | 29 |
| `academic-pipeline` | 30 |
| `agents` | 4 |
| `audits` | 18 |
| `commands` | 16 |
| `deep-research` | 53 |
| `docs` | 125 |
| `evals` | 1146 |
| `examples` | 45 |
| `hooks` | 2 |
| `pi` | 5 |
| `plugin-evals` | 38 |
| `plugin-evals-citation-check` | 61 |
| `plugin-evals-citation-locale` | 15 |
| `scripts` | 654 |
| `shared` | 155 |
| `sr-screener` | 25 |
| `tests` | 343 |
| `tools` | 6 |

具体AI加载/运行实例：`/ars-lit-review` → `commands/ars-lit-review.md:8`要求调用Skill工具加载`academic-paper`，传lit-review模式 → `academic-paper/SKILL.md:133`/`:164`确定literature_strategist → 按mode guide与Phase 1 references阅读检索协议 → 宿主提供搜索/HTTP能力（或让Python resolver实际联网） → 输出文献矩阵/综述。命令摘要明确禁止替代缺失的技能文件。程序验证路径另是 `verify_passport.py` → verification_gate.verify_passport → verify_citation → 四resolver/SQLite；**这条程序调用链不等于整篇稿件主流程都由Python驱动**。

凭据与域名必须分开说：S2 `api.semanticscholar.org` 的x-api-key在client`:186`条件添加；OpenAlex `api.openalex.org` 的api_key在`:127`条件添加；Crossref `api.crossref.org`只可选polite email；arXiv默认client`:54`实为 **HTTP** `export.arxiv.org`，claim_standing_discovery.py:315另用HTTPS。同一项目不同路径不可混称。上述是源码支持无Key分支，不保证2026外部服务当下额度、可用性或商业免费。Chinese resolver`:84–87`列doi.org、hdl.handle.net、eutils.ncbi.nlm.nih.gov，NCBI在`:1102`硬要求有效email，`:1108`才可选加key。跨模型OpenAI/Gemini需要配置相应key，`shared/cross_model_verification.md:459–460`与`:500`给出api.openai.com与generativelanguage.googleapis.com带凭据调用；Codex transport可用已登录订阅认证替代API key，仍非无账户免费。部分调用只在Markdown中供宿主执行，不能归入自有HTTP客户端。

抽检证据（按实际文件行号，均重新匹配源码；这不是运行测试）：

| 路径:行 | 可见事实 |
|---|---|
| `commands/ars-lit-review.md:8` | `First invoke` |
| `scripts/semantic_scholar_client.py:53` | `https://api.semanticscholar.org` |
| `scripts/semantic_scholar_client.py:186` | `if self._api_key` |
| `scripts/openalex_client.py:98` | `api_key: str | None = None` |
| `scripts/openalex_client.py:127` | `if self._api_key` |
| `scripts/crossref_client.py:47` | `https://api.crossref.org` |
| `scripts/arxiv_client.py:54` | `http://export.arxiv.org/api/query` |
| `scripts/chinese_literature_client.py:1102` | `if not self._ncbi_email` |
| `scripts/verification_gate/__init__.py:192` | `def verify_citation` |
| `scripts/verification_gate/__init__.py:294` | `def verify_passport` |
| `scripts/claim_audit_pipeline.py:8` | `Retrieval and judge invocation` |
| `scripts/ars_write_scope_guard.py:9` | `WHEN THE FENCE ENGAGES` |
| `sr-screener/templates/workflows/ta_screening.template.js:98` | `agent(` |
| `sr-screener/scripts/build_outputs.py:8` | `Writes into` |
| `pi/wrapper.js:79` | `export default` |
