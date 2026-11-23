# nature-skills：三类受众与实际执行体深度解剖

本报告是 **2026-11-23 本地 Git 快照的静态源码审计**。没有安装依赖、执行仓内程序、调用收费 API、重跑测试或验证论文质量；函数存在不等于运行成功，仓内“tested/production-tested”是作者陈述。星数不由 Git 快照证明。行数按 git tracked 文件文本 `splitlines()`，包含注释、docstring、空行，Markdown 中代码块不计可执行源码。测试按目录/文件名识别，候选运行代码仍含校验器、模板及例程，不能全称“自主科研核心”。统一账本见 `inventory_summary.json`、`inventory_files.tsv.gz`；本报告附的逐技能清单另按技能目录人工可复核统计，包含例程，不与中央 runtime 类强行等同。

## 1. 本地快照与代码账本

- origin：`https://github.com/Yuan1z0825/nature-skills.git`；HEAD：`c6701dbea29afa7c596d029b957b7edc92130ea5`；24个技能；897个tracked文件，工作树干净。`.codex-plugin/plugin.json`声明0.1.0、Apache-2.0、skills目录，由Codex/Claude等宿主发现。
- Markdown **443文件 / 41,255行**；源码 **273文件 / 56,855物理行 / 49,373非空行**。
- 源码按语言：Python **225 / 50,297**；MJS **39 / 5,142**；Shell **7 / 987**；R **2 / 429**。
- 测试 **114文件 / 15,016行**；非测试候选程序 **159 / 41,839**。其中约 **39,287行**在技能目录或RDA演示等、约2,552行仓库工具；候选程序含模板、图形例程、检查器、CLI支持，不都是自动科研循环。
- 首页 `index.html` 2,832行是人类网页，不能偷偷算科研运行代码。RDA-results有R分析与示例图/表，是示范产物，不是自主科研的普遍实证。

## 2. 给 AI 的：路由化写作/文献流程，而非单一万能系统

### 2.1 真正的路由规则

`nature-writing/SKILL.md:7起`要求先读manifest与always_load，再按 task / paper_type / section / language / journal选择片段；`manifest.yaml:9–93`具体映射 manuscript/submission-package、research/methods/hypothesis/algorithmic/review、摘要/引言/方法/实验/讨论等、en/zh-to-en、Nature/Nature Communications/NMI/其他期刊。references按需加载。

这是一套**让模型自己执行的文本路由协议**，不是Python解释器硬执行状态机。manifest让知识模块化、减少整仓灌入上下文；宿主仍负责判断轴值、Read文件和输出。`nature-shared`是共享参考层，不独立接用户任务；安装单技能不会自动补齐其共享依赖（README:444）。

### 2.2 方法论的硬内容

- 写作/润色/回复/审稿各有边界：writing从作者证据组织论证，polishing保事实和术语改语言，response处理审稿意见，reviewer给证据支撑的模拟评审；不是每个任务都“扮演Nature编辑”胡编一遍。
- shared保存读者工作流、论文类型、术语账、伦理、main-text discipline、Results/Discussion边界、引言知识缺口、摘要主张链；缺材料明确标记，而不补造实验。
- `nature-statistics/SKILL.md:7–55`自称**统计报告/审查技能**：独立实验单位与n、技术/生物重复、效应与区间、假设、纠正、多重比较、层级数据；:96起禁止捏造p、样本量、功效。它**零Python/R脚本**，不能称“内置统计检验引擎”；若用户给数据要计算，仍需Agent另写代码/调用统计包。
- nature-data也主要写Data/Code Availability、FAIR metadata和仓储计划，**不是数据清洗计算管线**。
- reader / paper-card / paper-xray要求文本/图表/方程溯源、方法到主张的证据链、区分abstract/full-text/metadata；真正理解仍由宿主LLM承担，附带数学/文件准备检查器不等于语义正确性验证器。
- social-science提供选题→知识问题→证据路线→反例/新颖性审计→可行性；不是社科调查自动开展器。

### 2.3 最值得拆开的“自动文献管线”

`nature-literature-pipeline/SKILL.md:16–44`画daily cron→30候选→六维评分→top5精读→飞书/Telegram推送→归档；权重35/20/15/10/10/10，提供去重和只写raw不改wiki等规则。**该技能12个文件、没有一份独立可执行源码**；engine/application是references的组织名，cron设置、检索、推送由Hermes/宿主外部工具实现。`production-tested`是作者陈述，不是仓里有一个可以独立启动的文献daemon。

## 3. 给人类的：教程、安装工具和直接商业引流

- README:54–92先介绍运营者，自营GPT/Claude代充/成品号、微信客服、知识星球、商务合作；这是明确可见的商业/社区漏斗，不必靠猜测。它与工具代码可并存，不能由商业入口直接推出代码虚假。
- README:128–468分npx、Claude Code wrapper/复制、Codex、Chatbox安装，说明本地依赖与凭据另配。中英双README、各技能中英README、docs与index网页主要服务使用者。
- README:266–306和394–429详细说明SessionStart自动更新、每小时节流、失败跳过、目标目录状态与hook信任。是真本机更新行为，不是单纯Markdown。
- `scripts/autoupdate-skills.sh`/`update-codex-skills.sh`是安装同步/更新器，有对应Shell回归测试，执行时会访问上游、写宿主技能目录；用户必须把这与“研究时做了什么”分开。
- README:196,370,444,468已经坦承：安装的是技能和配套文件，Python/R、PDF/PPTX、浏览器、MCP与API keys不是自动配齐。`PUBMED_EMAIL`、Scopus/ScienceDirect等凭据及OpenRouter等服务是实际门槛。
- 本仓没有看到与K-Dense同样的全技能Cisco scanner汇总报告；tests/validate脚本证明结构/回归工程存在，不等于第三方安全认证。

## 4. 可执行源码的真实链路

### 4.1 文献检索与引用：真正有MCP server、网络请求和导出

`nature-academic-search/mcp-server/academic_search_server.py:14–33`建立FastMCP及CrossRef、PubMed、arXiv、Scopus、ScienceDirect单例；:87–107通过asyncio.to_thread包装同步source；:163起注册工具。sources目录实现不同服务请求、错误、元数据归一。输入query/DOI/PMID/arXiv→source API→统一JSON；另有检索CLI和RIS/BibTeX/EndNote等转换。**这是本地可启动服务，非只告诉AI“去搜”。**不等于自建搜索引擎，索引/访问权来自外部服务。

`nature-citation/scripts/nature_citation.py`2388行：:643分割稿件、:711查Crossref、:886取PubMed、:904去重、:923作者完整性、:955/1001/1040多格式导出、:1196搜索段落、:1229建立mapping、:1298批次处理、:2269解析CLI、:2305主入口。输出参考文献、claim-to-source映射、TSV/JSON/HTML报告及checkpoint。**尤其要注意 `search_segment:1196–1226` 只是将段落和fallback query提交Crossref，按期刊范围筛选、去重后取前per_segment；`build_mapping:1229–1246`把这些候选与段落绑定，并生成citation marker。没有在这条源码链看到全文语义蕴含验证。** 候选匹配和元数据核对是实操作，不能仅凭命中/导出就断言引用真的支持因果主张；阅读核验仍属于AI/人。

### 4.2 全文下载：真正通过已认证浏览器，但不自带无条件访问权

`nature-downloader`有Python配置向导、学校/访问路线/健康检查；MJS实现metadata、provider registry、CNKI、OA、出版社、manifest、PDF验证。

`browser_pdf_downloader.mjs:1–12`引入CDP proxy封装，:56主入口，:66健康检查，:72起创建/导航标签，:88在已认证页面执行 `fetch(url, credentials: include)`，分块取字节→校验PDF→本地写入。它依赖**外部web-access CDP proxy与已登录Chrome**，不是自己绕过登录/CAPTCHA/paywall；脚本usage:19–21就声明这个边界。OA/API下载走对应provider；学校权限、出版商凭据不由安装技能提供。

### 4.3 图与PPT：主要工程体量在这里

- nature-figure非测试程序约8千行，含原创模板、绘图、源码审计、PDF/SVG文本与图形碰撞、多面板对齐、示意图API辅助，另有reference/examples。`validate_figure.py:72–692`检查语法/字体/大小/配色/可编辑文字/向量输出/抽样/排除/面板门等，部分是regex/AST启发式；`main:815`生成审计结论。`audit_panel_alignment.py`、`audit_figure_collisions.py`处理渲染后几何关系，确有图形QA；不能证明数据真实或统计推断正确。
- `plot_templates.py:91,105,111,215`读CSV/画图/保存bundle与QA；演示数据明确隔离。R例程只是可运行画图模板，不是所有统计检验完整实现。
- nature-image2ppt约1万行程序：主CLI `runtime/main.py:55–207`有doctor/config/setup/prepare/backend/status，:207 next、:300 dispatch、:309 record、:325 page-build、:342 validate、:359 finalize。它组织任务/对象manifest/资产→可编辑PPTX构建→验证；`build_pptx_from_manifest.py`1143行，`validate_pptx.py`1024行，是实质文档工程。
- 同模块 `runtime/image_gen.py:277–350`读取Codex认证并调用受限的 `https://chatgpt.com/backend-api/codex/images/...` 路径，也支持外部API后端。这是具体凭据/外部图像生成依赖，不能把图像智能算为本地算法；不应在报告公开凭据值。本次只看源码，未调用。
- nature-paper2ppt的435行脚本做素材/组边界与交付检查，论文→报告内容编排仍由Agent；paper2word的199行reference_layout是引用格式/版式辅助，Word成品依赖**另装的 `md-to-word-docx` 包和 `md-to-word` CLI**，`nature-paper2word/SKILL.md:9–15`明确先检查引擎，禁止拿手写python-docx替代。不能把外部排版引擎的功能全部记在本仓199行脚本名下。

### 4.4 翻译、专利与实验记录

- `nature-paper-trans/scripts/pages.py`905行是**任务与页资产状态管理**：:367 prepare、:428 reserve、:476 record、:535 review、:554 select、:587 composition、:620 compose_pages等，包含原子写/锁/源文件digest/逐页审阅/合成PDF。真正翻译和图像化页面由宿主模型/图像工具完成；不是905行代码内置机器翻译模型。
- nature-paper-to-patent约4.6千行：DOCX生成、结构/材料整理、技术交底/专利文档包；源材料到技术法律判断/新颖性由AI与用户负责，不是自动授予专利。
- nature-experiment-log零独立脚本：把图片/语音/文本组织成YAML Markdown，飞书CLI/Obsidian依赖宿主；不是独立实验管理服务器。
- nature-paper-card的prepare_paper.py/file工具处理源文件、图及卡片准备；nature-reader validate_reader_math.py是数学输出标记检查，不能保证推导成立。

## 5. 真实完整流程及关键边界

用户指定任务→宿主路由具体Skill→按manifest/reference获取方法→检索MCP/CLI获取元数据→下载provider/CDP取原文→宿主读/写/作图或生成manifest→配套脚本做结构/版式/文件/数学格式检查→导出PDF/PPTX/DOCX/引用包→用户审核。这是一套**科研文献与传播工作站配方，加上多个实际工具子系统**，不是统一调度、自动做实验、统计、训练、发表的自治科学家。

统计规范非常具体，实际统计计算却不是它的主要代码重心；编排文本依赖宿主判断，确定性审计器只能覆盖自己编码的检查项；浏览器、模型API和付费数据库可用性不由代码行数保证。代码有tests和输入验证，但本次静态审计不宣称运行成功。

## 6. 全24技能库存

|技能目录|文件数|非测试程序文件 / 物理行|第一层职责（frontmatter，原文截取）|
|---|---:|---:|---|
|`nature-academic-search`|55|17 / 4,487|Search literature across sources, verify or manage citations, and build MeSH strategies or citation-impact audits. Use for 文献检索、引文核对、参考文献管理、严格他引 and evidence-backed citer profiles; not for translating a paper or drafting|
|`nature-citation`|17|1 / 2,388|Find and verify Nature/CNS-family literature supporting manuscript claims, with claim-to-source mapping and reference-manager export. Use for Nature系列引用、CNS支撑文献、分段补引用 when this journal scope is requested; use broader lit|
|`nature-data`|15|0 / 0|Draft or audit manuscript Data/Code Availability statements, dataset access routes, repository plans, and FAIR metadata. Use for 数据可用性声明、数据共享、数据仓库选择 and dataset citations; not general data cleaning or statistical analysi|
|`nature-downloader`|71|30 / 5,409|Use when a user needs lawful academic full text, CNKI institutional access, English OA retrieval, publisher API access, institutional browser fallback, or supporting information downloads.|
|`nature-experiment-log`|11|0 / 0|标准化实验日志记录——直接上传或读取本地图片、语音和文字，产出带 YAML frontmatter 的 Markdown；可选集成飞书 CLI 与 Obsidian。|
|`nature-figure`|130|37 / 8,008|Create, revise, audit, and export manuscript scientific figures in Python or R.   Use for 论文配图、科研绘图、多面板图 and submission-ready plots, or explicitly   requested AI-generated graphical abstracts and mechanism schematics. No|
|`nature-image2ppt`|91|37 / 10,615|Reconstruct slide images, screenshots, scanned PDFs, or image-only PPTX files as object-level editable PowerPoint. Use for 图片转可编辑PPT、截图还原PPT and diagram reconstruction; not authoring a new deck from research notes.|
|`nature-literature-pipeline`|12|0 / 0|Complete automated literature discovery pipeline: multi-source search → six-dimension scoring → fine reading → formatted delivery → archival.   Combines a configurable engine with daily cron-driven application layer. Wor|
|`nature-paper-card`|22|2 / 951|Build a structured deep-reading Paper Card for one scientific paper, analysing methods, experiment-to-claim evidence, limitations, and research ideas. Use for 论文精读卡、方法拆解、证据链分析; not full-paper bilingual translation or for|
|`nature-paper-to-patent`|69|17 / 4,599|Turn research papers or inventor materials into evidence-grounded Chinese invention patent drafts and technical disclosures. Use for 技术交底书、专利撰写、现有技术对比 and Chinese DOCX patent packages; not general manuscript writing.|
|`nature-paper-trans`|13|1 / 905|将英文论文 PDF 的指定页面或全文逐页翻译为简体中文图片型 PDF，尽量保留原文的版式、分栏、图表和公式；最终仅交付中文 PDF。用于论文阅读翻译，不用于可编辑 Word 重建或逐字审校。|
|`nature-paper-xray`|7|0 / 0|Read one supplied paper closely enough to reconstruct how its authors actually got there, rather   than restating what the abstract claims. Use for 精读论文, 讲透这篇论文, 把这篇读懂, 作者为什么这么设计,   这个公式怎么推出来的, paper x-ray, and deep sing|
|`nature-paper2ppt`|19|1 / 435|Create or improve a Chinese academic PPTX from a scientific paper or research reading notes, with source figures and speaker notes. Use for 论文做PPT、文献汇报、组会PPT and paper-based conference or defense presentations.|
|`nature-paper2word`|10|1 / 199|Compile an academic Markdown manuscript into a journal- or school-compliant Word (.docx): declared typography, native OMML equations, three-line tables, cross-references, per-section page numbers, CSL bibliographies (GB/|
|`nature-polishing`|33|0 / 0|Polish, translate, or tighten existing academic prose while preserving facts, terminology, and evidence boundaries. Use for 论文润色、学术翻译、正文精简, or manuscript LaTeX layout fixes. Use nature-writing when the main task is draft|
|`nature-proposal-writer`|36|1 / 240|Compose, revise, or audit research proposals, opening reports, and research plans from supporting evidence. Use for 研究计划、开题报告、科研项目申请书. Invoked as researchwrite for compatibility; use nature-writing for general manuscript|
|`nature-reader`|20|1 / 463|Create source-grounded Chinese-English paper readers with aligned text, figures, tables, and equations. Use for 全文翻译、中英文对照、论文精读 or source-linked questions about a paper; respect a requested excerpt or question without ge|
|`nature-ref-verifier`|6|0 / 0|对学术文献逐条执行多源交叉验证，逐字段对比作者、标题、年份、卷期、页码，   标记卷年/DOI年冲突、作者顺序异常、页码偏差等问题，输出结构化验证报告。   可批量处理整篇论文/开题报告的参考文献列表，也可单条校验，支持与 Zotero 同步修正。|
|`nature-response`|38|1 / 316|Draft, audit, or revise responses to peer review, revision cover letters, and   marked-manuscript or LaTeX revision packages. Use for 审稿意见回复、逐点回复、返修信、   rebuttals and edits to existing response drafts. Initial-submission|
|`nature-reviewer`|21|0 / 0|Provide evidence-grounded mock peer review of scientific manuscripts or excerpts, covering significance, validity, and major/minor concerns. Use for 模拟审稿、投稿前自审、审稿人视角评估; not author rebuttal drafting.|
|`nature-shared`|21|1 / 272|Internal shared-reference support package for installed Nature Skills, including nature-writing, nature-polishing, nature-response, nature-reader, and nature-paper2ppt. Do not invoke it as a standalone user workflow. Loa|
|`nature-social-science`|14|0 / 0|社会科学及相邻学科学术写作与选题论证工作台，覆盖选题构思、文献梳理、论证组织、学术表达四大要素。Use for 找论文题目/研究问题、把兴趣或宽泛主题变成可讨论的研究方向、文献综述与领域版图、区分"研究少"与真正未决的知识问题、设计证据路线与主张边界、新颖性审计与"审稿人2"式攻击、博士/论文可行性评估、导师沟通话术、或检查论文中的过度断言与概念混用。也适用于开题报告、研究计划书、学位论文的选题阶段与论证自检。|
|`nature-statistics`|11|0 / 0|Audit or improve manuscript statistical reporting, including experimental units, replication, uncertainty, tests, and figure statistics. Use for 统计审查、统计方法小节、图注统计 and reviewer concerns; compute new analyses only when requ|
|`nature-writing`|77|0 / 0|Draft or restructure scientific manuscript arguments, sections, and initial-submission materials from author-provided evidence. Use for 论文写作、章节起草、论证重构、正文压缩、首次投稿材料. Use nature-polishing for language-only edits to existing|

## 7. 最大源码文件（测试明确可见，避免把行数全算运行核心）

|路径|物理行|
|---|---:|
|`skills/nature-citation/scripts/nature_citation.py`|2,388|
|`skills/nature-image2ppt/tests/test_multi_agent_backend.py`|2,031|
|`skills/nature-image2ppt/cli/image2ppt/runtime/build_pptx_from_manifest.py`|1,143|
|`skills/nature-image2ppt/cli/image2ppt/runtime/image_gen.py`|1,093|
|`skills/nature-image2ppt/cli/image2ppt/runtime/validate_pptx.py`|1,024|
|`skills/nature-paper-to-patent/scripts/disclosure/md_to_docx.py`|1,010|
|`skills/nature-figure/scripts/audit_panel_alignment.py`|935|
|`skills/nature-paper-trans/scripts/pages.py`|905|
|`skills/nature-image2ppt/cli/image2ppt/runtime/main.py`|872|
|`skills/nature-academic-search/mcp-server/tests/test_sources.py`|870|
|`skills/nature-figure/scripts/validate_figure.py`|849|
|`skills/nature-downloader/scripts/batch_download.mjs`|843|
|`scripts/generate-star-history.py`|805|
|`skills/nature-downloader/scripts/lib/anti-bot.mjs`|791|
|`skills/nature-figure/scripts/audit_figure_collisions.py`|756|
|`skills/nature-academic-search/mcp-server/academic_search_server.py`|651|
|`skills/nature-paper-trans/tests/test_pages.py`|644|
|`skills/nature-figure/scripts/plot_templates.py`|614|
|`skills/nature-academic-search/scripts/converters.py`|602|
|`skills/nature-downloader/src/wizard.py`|582|
|`skills/nature-academic-search/scripts/format-converter.py`|543|
|`scripts/tests/test_nature_figure_panel_alignment.py`|516|
|`skills/nature-academic-search/mcp-server/sources/scopus.py`|510|
|`skills/nature-paper-card/scripts/prepare_paper.py`|495|
|`skills/nature-academic-search/scripts/academic_search.py`|483|
|`skills/nature-image2ppt/scripts/postprocess_manifest_arrows.py`|479|
|`skills/nature-reader/scripts/validate_reader_math.py`|463|
|`skills/nature-paper-card/scripts/audit_paper_card.py`|456|
|`skills/nature-paper-to-patent/scripts/disclosure/mermaid_render.py`|454|
|`skills/nature-image2ppt/cli/image2ppt/runtime/runtime_env.py`|448|

## 8. 给379期可直接使用的判断

Nature的写作、统计审查、文献推送很多是**给AI的方法合同**；文献MCP、带认证下载、图形审计、PPTX生成与页任务管理则是**本地真程序**。首页商业引流很靠前，但不能因此抹掉4万多行非测试候选程序。反过来，也不能把“六维评分”“角色”“状态”和“自动化”这些词都当成仓里已有可独立执行的研究Agent。
