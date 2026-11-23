# ARIS：Auto-claude-code-research-in-sleep 本地源码解剖

审计日期：2026-11-23；本地 HEAD `4ba94ae2a1c1fbbb14858750dd8764cabf6c28bd`。仅静态审计；不安装、不调用模型、不运行上游程序。路径均相对此仓。代码行数为物理行（含空行、注释、docstring），不是“有效算法行”，不能当能力评分。

## 家底与正确计数

本快照 768 个文件、431 个 Markdown、197 个 SKILL.md。197 不是197种独立科研方法：`skills/<name>` 有 **87** 个主技能；`skills/skills-codex` 87 个镜像；Claude-review 覆盖8个、Gemini-review覆盖15个。镜像会改调用惯例与模型路由，仍不能按独立技能加总。目录共有157个.py、17个.sh、2个.ps1；中央 inventory 统计176个候选代码文件、**64,975行**，其中83个测试/支持文件20,695行，非tests部分44,280行（与中央inventory一致；下列按read_bytes splitlines细分在换行处理上合计差2行）。按路径：tools 49文件21,199行；mcp-servers 13文件8,529行；skills 24文件13,269行；aris-monitor 6文件1,196行；templates 1文件85行；tests 83文件20,695行。逐字节去重可消除3份副本，剩63,146行，但还有改写镜像与同文件嵌入自测，不能把这个数叫原创生产代码。完整文件账见同目录 inventory_files.tsv.gz.gz。

还有56个HTML、16个tex、7个bst、6个sty、3个cls；论文模板、渲染产物不是新增科研执行器。community_papers含8.5MB PDF；assets/docs有大量截图、幻灯片与示例。

## 第一类：给 AI 读的真正方法与控制流

主入口 `skills/research-pipeline/SKILL.md:94` 明示 `/idea-discovery → /experiment-bridge → /auto-review-loop → /paper-writing(optional)`。这条“流程图”的调度者首先是宿主编码agent，**不是一个统一Python main函数**。每个技能把具体动作、工具名、输入文件、失败处理、产物契约写进上下文。

1. 选题：research-lit/各文献源检索、research-wiki沉淀；idea-creator/idea-discovery生成候选；novelty-check/refine评估可证伪问题、基线、创新点。communication领域另有comm-lit-review来源政策/venue分级参考。
2. 实验桥：experiment-plan、ablation-planner定义对照/指标；experiment-bridge按项目环境写或改代码；run-experiment部署；monitor-experiment/training-check看启动与日志；analyze-results将真实结果汇总；result-to-claim限定论文主张。
3. 评审循环：`skills/auto-review-loop/SKILL.md:25`最多4轮；medium使用MCP评审，hard增加reviewer memory与辩论，nightmare让外部编码CLI直接读仓（:390/:452/:513）。review→解析Score/Verdict→门禁→修实验/等结果→逐轮记录。停止条件是政策允许的reviewer给出 `score>=6`且ready/almost；这只是模型评审阈值，不是顶会录用证明。Copilot compatibility路线正评不能自行终结，须外部Codex/manual finalizer；原文明确caller-declared身份仍unverified（:237、:713）。
4. 写作：paper-plan/paper-write/paper-writing，图表、公式、引用、claim/integrity审计、LaTeX编译、slides/poster/talk/rebuttal/resubmit。专利支线含prior-art、claims、jurisdiction、specification与patent-pipeline；证明支线含proof-writer/checker/orchestrator及Lean工作循环。grant-proposal、GPU/Vast/Modal、Overleaf也有专项技能。

`tools/skill-groups.tsv:25`起把技能按文献、选题、评审、理论、实验算力、论文主线等分组，并记录安装依赖；`agents/aris-fanout-leaf.md`规定并行叶节点责任；`skills/shared-references/`集中接受门禁、外部节奏、可恢复运行、提示注入约束等。它不是随意一句“帮我科研”，而是一套有文件契约的研究操作手册；但模型仍负责选方向、写算法、理解科学结果，模板不会自动提供领域发现。

默认值值得写清：pipeline :57-68是AUTO_PROCEED=true、HUMAN_CHECKPOINT=false、CODE_REVIEW=true、AUTO_WRITE=false、ARXIV_DOWNLOAD=false。所以“默认睡一觉直接出完整论文”不准确：默认到实验叙事报告，论文写作需显式启用且给venue；默认文献主要拉元数据而非全量PDF。:77-92规定自动选点继续，飞书interactive是显式阻塞例外。

## 第二类：给人类的包装、安装与支持

README.md约241KB、README_CN约203KB，远超一个技能清单：安装方式、工作流、模型组合、展示案例、社区成果、更新史、截图、API服务教程。SETUP_GUIDE中英双份；docs有Codex/Gemini/Claude/Copilot/Antigravity/Trae/OpenClaw适配、多模型供应商接线、GPU、断线恢复、manual-review、watchdog、项目文件解释、Overleaf等。templates提供research brief/contract、实验计划/日志、findings、paper plan、专利和叙事报告骨架。

关键边界：README前部大段 **ARIS-Code独立CLI** v0.4.x发布日志声称Rust crates、流式API、沙箱、权限等功能（README:39起）。**本 checkout 无.rs/Cargo源码**，docs/ARIS-Code-README仅指向相应分支/资料；这些发布说明不能按本地已审计Rust实现计入。本地能证实的是Python桥与脚本，不能用README的CLI测试数代替当前源码验证。

微信社区/二维码、ARIS品牌、社区论文展示、模型/API平台教程形成传播与使用入口；本次没有发现可据以断言的内建付费订阅计费代码，第三方API/GPU仍可能收费。安全材料如CODEX_POLICY_AUDIT、注入卫生与threat_scan是自家规则/审计材料，不能等同独立安全认证。

## 第三类：真正运行的程序——不是只有Markdown

| 本地组件 | 实干与证据 | 不该夸大的部分 |
|---|---|---|
| 模型桥 | `mcp-servers/codex-exec/server.py:119,180,337`构造CLI参数、Popen、线程续接、stdio MCP；claude-review/antigravity/grok/gemini-review走各CLI/SDK；model-worker/server.py:310起spawn子进程；llm-chat/server.py用httpx对OpenAI兼容chat/completions，minimax-chat接MiniMax | 桥依赖已安装CLI/账号；不是训练新科研模型 |
| 人工审稿与绘图 | manual-review/server.py 910行提供本地人工回填页面/响应桥；codex-image2 893行、paper-illustration包装图像调用 | 手动回填仍需人；调用图像模型不等于验证科学图 |
| 文献抓取 | arxiv_fetch 335行走export.arxiv；S2 438行；OpenAlex326行；verify_papers692行跨arXiv/Crossref/S2核参考；exa425、firecrawl417、keenable355行分别接搜索/抓取API；deepxiv213行封装外部CLI | 可检索核身份/元信息，不会自动证明论文结论 |
| 研究知识库 | research_wiki.py 1,745行，节点/边/claims/ideas/experiments入库与元数据；capture_filter筛选；provenance/iteration_log留记录 | 知识图谱结构正确不代表内容科学正确 |
| 队列 | canonical `skills/experiment-queue/scripts/queue_manager.py`433行；读manifest、nvidia-smi找GPU、screen启动、日志识别OOM、按阶段依赖排队与写state（:55,104,112,194起） | tools/experiment_queue两个44行文件只是execv转发；“Modal/Vast future support”写在模块头，不能说canonical调度器已实现全云后端 |
| 门禁与恢复 | review_gate559行持有stop/continue/escalate表，run_state399行区分done/accepted/provisional并保留reviewer/verdict ID；forensics_gate669、evidence_check212、idea_discovery_gate256、provenance294行 | 验收凭证是记录/匹配，不是数学上证明reviewer独立或结论真 |
| 监视与通知 | watchdog509行监测session、GPU、下载/输出、循环活性（:294明确detect-only）；aris-monitor1,196行扫描session并用桌面widget/focus辅助；feishu-bridge240行通过lark-oapi发卡片及poll回复 | watchdog不是自动科研判官，飞书不等于不需用户配置 |
| 文档/视觉产物 | render-html1008行、海报posterly/asset/style/render/preflight/gates一组约6千行，figure_renderer799行，PDF图抽取666行、Overleaf脚本 | 视觉与排版工程会显著拉高代码量，不能都叫科研推理代码 |
| 安装更新 | install_aris.sh1760、Codex1049、Copilot1052、PowerShell安装器，smart_update及skill_picker238行选择安装、链接/镜像、依赖/配置处理 | 安装器代码不等于科学算法 |

安全scanner `tools/threat_scan.py:43-198`是regex匹配、范围分级与quarantine，不是“语义免疫”；模块头:14已经承认clean只是没命中已知坏字串。meta_opt/trigger_eval.py对触发查询调用模型评估技能路由，是工具使用优化，不是基于现实科研收益的无限RSI。

## 结论

ARIS本地版是“**大量可执行工具支撑的、由编码agent解释Markdown来调度的研究工作流套件**”。既不能贬成零代码嘴炮，也不能把完整自动科学发现、论文接受率、独立CLI发布版源码都归到此快照。其最强工程内容是多模型评审路由、可恢复状态/证据门禁、工具接线与成果生产；科学质量依赖研究问题、数据、真实实验与模型判断。静态审计未证实任何宣称的成功率，也未运行账号/API/远程实验。

## 附：本地代码文件逐项账（含注释空行）

| 路径 | 文本行数 | 归类 |
|---|---:|---|
| `aris-monitor/focus-tty.sh` | 156 | 执行候选（见正文功能边界） |
| `aris-monitor/focus.py` | 93 | 执行候选（见正文功能边界） |
| `aris-monitor/run.sh` | 49 | 执行候选（见正文功能边界） |
| `aris-monitor/scanner.py` | 445 | 执行候选（见正文功能边界） |
| `aris-monitor/ticker.py` | 81 | 执行候选（见正文功能边界） |
| `aris-monitor/widget.py` | 372 | 执行候选（见正文功能边界） |
| `mcp-servers/antigravity-exec/server.py` | 636 | 执行候选（见正文功能边界） |
| `mcp-servers/claude-review/run_with_claude_aws.sh` | 49 | 执行候选（见正文功能边界） |
| `mcp-servers/claude-review/server.py` | 782 | 执行候选（见正文功能边界） |
| `mcp-servers/codex-exec/server.py` | 480 | 执行候选（见正文功能边界） |
| `mcp-servers/codex-image2/server.py` | 893 | 执行候选（见正文功能边界） |
| `mcp-servers/feishu-bridge/server.py` | 240 | 执行候选（见正文功能边界） |
| `mcp-servers/gemini-review/server.py` | 1856 | 执行候选（见正文功能边界） |
| `mcp-servers/grok-exec/server.py` | 651 | 执行候选（见正文功能边界） |
| `mcp-servers/llm-chat/server.py` | 742 | 执行候选（见正文功能边界） |
| `mcp-servers/manual-review/server.py` | 910 | 执行候选（见正文功能边界） |
| `mcp-servers/minimax-chat/server.py` | 381 | 执行候选（见正文功能边界） |
| `mcp-servers/model-worker/client.py` | 189 | 执行候选（见正文功能边界） |
| `mcp-servers/model-worker/server.py` | 720 | 执行候选（见正文功能边界） |
| `skills/citation-crosscheck/scripts/cx_verify.py` | 1572 | 执行候选（见正文功能边界） |
| `skills/experiment-queue/scripts/build_manifest.py` | 142 | 执行候选（见正文功能边界） |
| `skills/experiment-queue/scripts/queue_manager.py` | 433 | 执行候选（见正文功能边界） |
| `skills/figure-spec/scripts/figure_renderer.py` | 799 | 执行候选（见正文功能边界） |
| `skills/paper-illustration-image2/scripts/paper_illustration_image2.py` | 255 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/__init__.py` | 0 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/canvas.py` | 200 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/measure.py` | 588 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/polish.py` | 498 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/preflight.py` | 489 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/render.py` | 215 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/textutil.py` | 16 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/_posterly/verify_final.py` | 171 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/asset_check.py` | 897 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/extract_pdf_figures.py` | 666 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/poster_check.py` | 251 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/preprocess_figures.py` | 238 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/render_preview.py` | 217 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/run_gates.py` | 556 | 执行候选（见正文功能边界） |
| `skills/paper-poster-html/scripts/style_check.py` | 1324 | 执行候选（见正文功能边界） |
| `skills/render-html/scripts/render_html.py` | 1008 | 执行候选（见正文功能边界） |
| `skills/skills-codex/citation-crosscheck/scripts/cx_verify.py` | 1572 | 镜像/平台适配副本 |
| `skills/skills-codex/paper-illustration-image2/scripts/paper_illustration_image2.py` | 255 | 镜像/平台适配副本 |
| `skills/skills-codex/render-html/scripts/render_html.py` | 909 | 镜像/平台适配副本 |
| `templates/claude-hooks/corpus_write_guard.py` | 85 | 执行候选（见正文功能边界） |
| `tests/__init__.py` | 0 | 测试/测试支持 |
| `tests/_feishu_bridge_helpers.py` | 82 | 测试/测试支持 |
| `tests/_llm_chat_helpers.py` | 179 | 测试/测试支持 |
| `tests/_minimax_helpers.py` | 189 | 测试/测试支持 |
| `tests/conftest.py` | 16 | 测试/测试支持 |
| `tests/test_antigravity_exec_bridge.py` | 300 | 测试/测试支持 |
| `tests/test_argument_hint_lint.py` | 183 | 测试/测试支持 |
| `tests/test_arxiv_fetch.py` | 298 | 测试/测试支持 |
| `tests/test_audit_codex_invocation.py` | 114 | 测试/测试支持 |
| `tests/test_auto_proceed_contract.py` | 106 | 测试/测试支持 |
| `tests/test_bounded_fanout_contract.py` | 242 | 测试/测试支持 |
| `tests/test_capture_filter.py` | 86 | 测试/测试支持 |
| `tests/test_citation_crosscheck_gate.py` | 213 | 测试/测试支持 |
| `tests/test_claude_review_artifact_tools.py` | 177 | 测试/测试支持 |
| `tests/test_claude_review_generator.py` | 51 | 测试/测试支持 |
| `tests/test_claude_review_overlay.py` | 64 | 测试/测试支持 |
| `tests/test_claude_review_parse.py` | 388 | 测试/测试支持 |
| `tests/test_codex_exec_bridge.py` | 306 | 测试/测试支持 |
| `tests/test_codex_image2_server.py` | 424 | 测试/测试支持 |
| `tests/test_codex_install_update.py` | 575 | 测试/测试支持 |
| `tests/test_codex_skill_mirror.py` | 547 | 测试/测试支持 |
| `tests/test_copilot_install.py` | 1202 | 测试/测试支持 |
| `tests/test_copilot_native_evidence.py` | 562 | 测试/测试支持 |
| `tests/test_corpus_write_guard.py` | 76 | 测试/测试支持 |
| `tests/test_deepxiv_fetch.py` | 180 | 测试/测试支持 |
| `tests/test_evidence_check.py` | 189 | 测试/测试支持 |
| `tests/test_exa_search.py` | 220 | 测试/测试支持 |
| `tests/test_feishu_bridge_server.py` | 289 | 测试/测试支持 |
| `tests/test_firecrawl_search.py` | 649 | 测试/测试支持 |
| `tests/test_forensics_gate.py` | 565 | 测试/测试支持 |
| `tests/test_grok_exec_bridge.py` | 290 | 测试/测试支持 |
| `tests/test_http_reviewer_fallback_contract.py` | 43 | 测试/测试支持 |
| `tests/test_http_reviewer_provenance_alignment.py` | 25 | 测试/测试支持 |
| `tests/test_http_reviewer_review_gate.py` | 112 | 测试/测试支持 |
| `tests/test_idea_creator_query_pack_scan.py` | 238 | 测试/测试支持 |
| `tests/test_idea_discovery_gate.py` | 400 | 测试/测试支持 |
| `tests/test_install_aris_agent_profiles.py` | 111 | 测试/测试支持 |
| `tests/test_install_aris_ps1.py` | 816 | 测试/测试支持 |
| `tests/test_install_aris_replace_link.py` | 119 | 测试/测试支持 |
| `tests/test_install_aris_selective.py` | 559 | 测试/测试支持 |
| `tests/test_install_aris_tools_symlink.py` | 336 | 测试/测试支持 |
| `tests/test_iteration_log.py` | 85 | 测试/测试支持 |
| `tests/test_keenable_search.py` | 335 | 测试/测试支持 |
| `tests/test_llm_chat_converter.py` | 61 | 测试/测试支持 |
| `tests/test_llm_chat_request_compatibility.py` | 409 | 测试/测试支持 |
| `tests/test_llm_chat_review_fallback.py` | 516 | 测试/测试支持 |
| `tests/test_llm_chat_server.py` | 434 | 测试/测试支持 |
| `tests/test_mainline_large_payload_prompts.py` | 72 | 测试/测试支持 |
| `tests/test_manual_review.py` | 804 | 测试/测试支持 |
| `tests/test_manual_review_integration.py` | 305 | 测试/测试支持 |
| `tests/test_minimax_chat_server.py` | 464 | 测试/测试支持 |
| `tests/test_minimax_integration.py` | 241 | 测试/测试支持 |
| `tests/test_model_worker.py` | 403 | 测试/测试支持 |
| `tests/test_model_worker_client.py` | 78 | 测试/测试支持 |
| `tests/test_openalex_fetch.py` | 197 | 测试/测试支持 |
| `tests/test_plugin_manifest.py` | 40 | 测试/测试支持 |
| `tests/test_probe_codex_invocation.py` | 48 | 测试/测试支持 |
| `tests/test_proof_suite_integration.py` | 66 | 测试/测试支持 |
| `tests/test_provenance.py` | 284 | 测试/测试支持 |
| `tests/test_readme_navigation.py` | 48 | 测试/测试支持 |
| `tests/test_research_wiki_claims.py` | 93 | 测试/测试支持 |
| `tests/test_research_wiki_encoding.py` | 74 | 测试/测试支持 |
| `tests/test_research_wiki_experiments.py` | 88 | 测试/测试支持 |
| `tests/test_research_wiki_fetch_arxiv_metadata.py` | 242 | 测试/测试支持 |
| `tests/test_research_wiki_helper_resolution.py` | 253 | 测试/测试支持 |
| `tests/test_research_wiki_ideas.py` | 113 | 测试/测试支持 |
| `tests/test_review_gate.py` | 449 | 测试/测试支持 |
| `tests/test_reviewer_pins.py` | 90 | 测试/测试支持 |
| `tests/test_reviewer_route.py` | 127 | 测试/测试支持 |
| `tests/test_reviewer_trace.py` | 230 | 测试/测试支持 |
| `tests/test_run_experiment_sync.py` | 68 | 测试/测试支持 |
| `tests/test_run_state.py` | 313 | 测试/测试支持 |
| `tests/test_skill_groups.py` | 111 | 测试/测试支持 |
| `tests/test_skill_picker.py` | 207 | 测试/测试支持 |
| `tests/test_skills_inventory_allowed_tools.py` | 52 | 测试/测试支持 |
| `tests/test_threat_scan.py` | 139 | 测试/测试支持 |
| `tests/test_trigger_eval.py` | 171 | 测试/测试支持 |
| `tests/test_verify_paper_audits.py` | 295 | 测试/测试支持 |
| `tests/test_verify_papers.py` | 210 | 测试/测试支持 |
| `tests/test_verify_papers_s2.py` | 113 | 测试/测试支持 |
| `tests/test_watchdog.py` | 541 | 测试/测试支持 |
| `tests/test_watchdog_loop.py` | 148 | 测试/测试支持 |
| `tests/test_web_debug_search.py` | 157 | 测试/测试支持 |
| `tools/arxiv_fetch.py` | 335 | 执行候选（见正文功能边界） |
| `tools/audit_codex_invocation.py` | 115 | 执行候选（见正文功能边界） |
| `tools/capture_filter.py` | 126 | 执行候选（见正文功能边界） |
| `tools/check_skills_inventory.py` | 303 | 执行候选（见正文功能边界） |
| `tools/convert_skills_to_llm_chat.py` | 285 | 执行候选（见正文功能边界） |
| `tools/copilot_native_evidence.py` | 818 | 执行候选（见正文功能边界） |
| `tools/deepxiv_fetch.py` | 213 | 执行候选（见正文功能边界） |
| `tools/evidence_check.py` | 212 | 执行候选（见正文功能边界） |
| `tools/exa_search.py` | 425 | 执行候选（见正文功能边界） |
| `tools/experiment_queue/build_manifest.py` | 44 | 执行候选（见正文功能边界） |
| `tools/experiment_queue/queue_manager.py` | 44 | 执行候选（见正文功能边界） |
| `tools/extract_paper_style.py` | 560 | 执行候选（见正文功能边界） |
| `tools/figure_renderer.py` | 69 | 执行候选（见正文功能边界） |
| `tools/firecrawl_search.py` | 417 | 执行候选（见正文功能边界） |
| `tools/forensics_gate.py` | 669 | 执行候选（见正文功能边界） |
| `tools/generate_codex_claude_review_overrides.py` | 343 | 执行候选（见正文功能边界） |
| `tools/idea_discovery_gate.py` | 256 | 执行候选（见正文功能边界） |
| `tools/install_aris.ps1` | 1750 | 执行候选（见正文功能边界） |
| `tools/install_aris.sh` | 1760 | 执行候选（见正文功能边界） |
| `tools/install_aris_codex.sh` | 1049 | 执行候选（见正文功能边界） |
| `tools/install_aris_copilot.sh` | 1052 | 执行候选（见正文功能边界） |
| `tools/iteration_log.py` | 143 | 执行候选（见正文功能边界） |
| `tools/keenable_search.py` | 355 | 执行候选（见正文功能边界） |
| `tools/lint_skills_helpers.sh` | 84 | 执行候选（见正文功能边界） |
| `tools/meta_opt/check_ready.sh` | 80 | 执行候选（见正文功能边界） |
| `tools/meta_opt/log_event.sh` | 91 | 执行候选（见正文功能边界） |
| `tools/meta_opt/trigger_eval.py` | 280 | 执行候选（见正文功能边界） |
| `tools/openalex_fetch.py` | 326 | 执行候选（见正文功能边界） |
| `tools/overleaf_audit.sh` | 104 | 执行候选（见正文功能边界） |
| `tools/overleaf_setup.sh` | 150 | 执行候选（见正文功能边界） |
| `tools/paper_illustration_image2.py` | 62 | 执行候选（见正文功能边界） |
| `tools/probe_codex_invocation.py` | 97 | 执行候选（见正文功能边界） |
| `tools/provenance.py` | 294 | 执行候选（见正文功能边界） |
| `tools/research_wiki.py` | 1745 | 执行候选（见正文功能边界） |
| `tools/review_gate.py` | 559 | 执行候选（见正文功能边界） |
| `tools/reviewer_route.py` | 148 | 执行候选（见正文功能边界） |
| `tools/run_state.py` | 399 | 执行候选（见正文功能边界） |
| `tools/save_trace.sh` | 529 | 执行候选（见正文功能边界） |
| `tools/semantic_scholar_fetch.py` | 438 | 执行候选（见正文功能边界） |
| `tools/skill_picker.py` | 238 | 执行候选（见正文功能边界） |
| `tools/smart_update.ps1` | 521 | 执行候选（见正文功能边界） |
| `tools/smart_update.sh` | 591 | 执行候选（见正文功能边界） |
| `tools/smart_update_codex.sh` | 419 | 执行候选（见正文功能边界） |
| `tools/smart_update_copilot.sh` | 605 | 执行候选（见正文功能边界） |
| `tools/threat_scan.py` | 222 | 执行候选（见正文功能边界） |
| `tools/verify_paper_audits.sh` | 497 | 执行候选（见正文功能边界） |
| `tools/verify_papers.py` | 692 | 执行候选（见正文功能边界） |
| `tools/verify_wiki_coverage.sh` | 176 | 执行候选（见正文功能边界） |
| `tools/watchdog.py` | 509 | 执行候选（见正文功能边界） |

## 附：87个主技能完整名称（镜像不重复算）

`ablation-planner`、`alphaxiv`、`analyze-results`、`arxiv`、`auto-paper-improvement-loop`、`auto-review-loop`、`auto-review-loop-llm`、`auto-review-loop-minimax`、`citation-audit`、`citation-crosscheck`、`claims-drafting`、`comm-lit-review`、`deepxiv`、`dse-loop`、`embodiment-description`、`exa-search`、`experiment-audit`、`experiment-bridge`、`experiment-plan`、`experiment-queue`、`feishu-notify`、`figure-description`、`figure-spec`、`firecrawl-search`、`formula-derivation`、`gemini-search`、`grant-proposal`、`idea-creator`、`idea-discovery`、`idea-discovery-robot`、`integrity-forensics`、`interview-cheatsheet`、`invention-structuring`、`jurisdiction-format`、`keenable-search`、`kill-argument`、`lean-formalize`、`mermaid-diagram`、`meta-apply`、`meta-optimize`、`monitor-experiment`、`novelty-check`、`openalex`、`overleaf-sync`、`paper-claim-audit`、`paper-compile`、`paper-figure`、`paper-illustration`、`paper-illustration-image2`、`paper-plan`、`paper-poster`、`paper-poster-html`、`paper-slides`、`paper-talk`、`paper-write`、`paper-writing`、`patent-novelty-check`、`patent-pipeline`、`patent-review`、`pixel-art`、`prior-art-search`、`proof-checker`、`proof-orchestrator`、`proof-writer`、`qzcli`、`rebuttal`、`render-html`、`research-implement-feature`、`research-lit`、`research-pipeline`、`research-refine`、`research-refine-pipeline`、`research-review`、`research-wiki`、`resubmit-pipeline`、`result-to-claim`、`run-experiment`、`semantic-scholar`、`serverless-modal`、`slides-polish`、`specification-writing`、`system-profile`、`training-check`、`vast-gpu`、`web-debug-search`、`wiki-enrich`、`writing-systems-papers`。

附录说明：上述源码账按文件文本读取splitlines，中央inventory_summary.json为统一总表口径；镜像、安装器、内嵌自测与渲染工程仍保留在原路径，避免把它们悄悄充当新增科研能力。
