# Anti-Autoresearch：研究诚信取证工作流与确定性算术脊柱

静态审计日期2026-11-23；本地HEAD `0abae59aa060e2bcfbd8ef88fd48efaa1be242ee`。65文件、25个Markdown/12,399行、12个SKILL.md（11技能+1编排器）；17.py+1.sh共4,493物理行。tools11文件2,874行、eval2文件272行、tests5文件1,347行。非测试共3,146行，其中eval是合成回归而非生产审稿。另有13个JSON（schema、fixture期望等）、1个tex、1张社区图片；不是大规模论文训练集。代码统计与同目录inventory_summary.json一致。

## 第一类：给AI读——从写论文转成查论文

`workflows/anti-autoresearch/SKILL.md`1,072行是总入口；:45-80给出证据账本→分维审计→备忘录→固定规则裁决→给人报告。先ingest论文源码/PDF与可用repo，构建claims.json，所有finding必须指回账本中的claim/span。随后由宿主agent扇出不同审阅角色，提议finding，合并供Python裁决。不是一个Python总控自动发起全部LLM请求。

11个技能的完整职责：

| 技能 | 行数 | 真正关注 |
|---|---:|---|
| evidence-ledger |637|源文抽取、逐span落账，限定审计输入|
| consistency-audit |666|数字与正文/表格、主张一致性|
| experiment-forensics |790|实现/实验报告能否对上、占位结果、造数形状|
| baseline-comparison-audit |946|对照与预算公平、消融归因|
| eval-design-forensics |945|泄漏、题目/指标设计、数据选择|
| citation-forensics |800|引用是否支撑语境、文献可核验|
| proof-derivation-forensics |821|证明/推导在源文中的逻辑缺口|
| presentation-signals |647|重复表、流水线残留等表面信号|
| ai-style-impressions |773|风格印象单列，零诚信裁决权重|
| adversarial-case-builder |745|反方memo及反驳审计，非裁决来源|
| novelty-duplication-advisory |1106|创新/重复与外部文献对照，memo-only|

references/hack-pattern-taxonomy.md873行组织检测模式与误报条件；observability-levels、reviewer-independence与integrity-forensics-contract定义L0(PDF)/L1源文/L2代码等证据层级、模型独立性及产物契约。总流程:13-20明令不把裁决放定时器反复跑：只有paper/repo/ledger变化才重审，外部下载/检索可等待。:869-918给无模型fallback，但语义维度没跑时不得装作全CLEAN，报告REVIEW_UNAVAILABLE。

## 第二类：给人读——它自称的边界与现实接入

README中英、DESIGN、SECURITY、CONTRIBUTING、docs/positioning和limitations；README:494有共享ARIS微信入口。叙事是“forensics而非AI文本检测”，并明确不是学术不端定罪、不是实验复现。风格AIGC观感单独展示；不能把一篇写得像LLM的文章判作造假。docs/limitations承认提取失败、误报、可游戏化和模型错误。

安装器`tools/install_anti_autoresearch.sh:12-17`只把skills/workflows复制进目标技能目录；:8-9特意说明Python tools **不复制**。因此不是复制SKILL就能处处跑：技能需定位原仓工具与schemas/references，调用外部Codex reviewer也要另装配置（安装器:24）。本仓没有自带通用科研模型server，没有本地LLM权重，没有浏览器自动化服务。

## 第三类：程序到底检查什么

| 程序 | 行数 | 静态确认行为 |
|---|---:|---|
| build_claim_ledger.py |312|stdlib regex从LaTeX/文本提取数字、对比、scope语言、引用、caption、表单元格，附span/行号/claim ID；非深层语义解析|
| build_manifest.py |112|收集可用论文/源码/代码证据，写artifact manifest|
| select_primary_pdf.py |97|在候选PDF选主文件，避免混淆补充材料|
| check_numeric_consistency.py |183|核“from a to b提升x%”的算术；headline数字是否出现在表单元格，后者误报风险较高|
| check_stat_consistency.py |530|GRIM计数比例可达性、GRIMMER/有界方差检查、statcheck重算z/t/F/chi2/r报告p值；z用NormalDist，t/F/chi2/r依赖可选SciPy（:41、:361-433），没装就跳过而非伪装通过|
| check_presentation.py |228|重复/近同表签名、精确流水线/助手残留字串|
| check_ai_style.py |154|规则找风格印象，不把它当诚信证据|
| countercheck.py |193|对数字推断尝试反核，现实现仅报告，不允许自动降级|
| attach_refutation.py |144|把反方反驳挂回finding|
| adjudicate_findings.py |893|校验账本锚点、区分零权重、汇总severity及覆盖状态，输出report.json/REPORT.md|
| eval/corrupt.py+run_eval.py |272|从小型clean tex造合成缺陷，跑提取/规则/裁决，比较required/forbidden patterns和预期verdict；非真实世界论文benchmark|

这些本地Python工具没有直接调用模型API；语义工作由SKILL要求宿主/外部reviewer做。文献创新支线能指导外部检索，但不能把整个仓说成“Python自动爬全网、自动判学术造假”。

## 最关键的版本陷阱：确定性裁决≠LLM不影响结论

`DESIGN.md:21-36`仍说LLM不评分、证据层不足自动降info；但**当前代码不能照这段宣传概括**。

`tools/adjudicate_findings.py:361-379`自己的注释已明确：真正能移动severity的计算主要是ANCHOR。:388-396没在账本中找到可核span才降info。:398-410观测级不足只标 `_observability_met=false`；:412-423 false_positive_risk仅注释；:441-446 needs_external_check也只注释；:330-352反驳有账本依据只标contested，仍不改变severity。:425-433对AIS/备忘录按类别强制零权重，这部分是实打实的规则。

:455-460 `critical→HARD_FLAGS`、`major/minor→SOFT_FLAGS`；而critical/major/minor起点来自auditor提议。**Python固定映射能保证相同findings输入得相同输出，不能保证LLM提议的严重性正确**。同样“span真实存在”只证明引用文字存在，不证明对这段文字的指控成立。模块中 `_surface_signal`标记也不应按早期docstring自动理解为所有surface finding都已被强制cap；应以当前adjudicate函数为准。

这是设计文档与现实现的重要差异，不能简单吹成“最终结论完全不受模型影响”。仓库对覆盖不全时阻止无条件CLEAN的处理有价值，但依然需要人阅读每条finding及其证据/观测级。

## 判断

Anti不是另一个自动写论文工具，而是**LLM语义审计手册+证据账本+真实统计/算术检测器+确定性报告裁决器**。它能查出可计算的不一致，也能强迫指控引用实物；不能从少量可观测材料证明不存在造假，更不能由可重现的聚合规则推出发现一定真实。此处未跑上游测试/模型，也没有重现其README宣称的真实论文效果。

## 附：本地代码文件逐项账（含注释空行）

| 路径 | 文本行数 | 归类 |
|---|---:|---|
| `eval/corrupt.py` | 92 | 合成评估 |
| `eval/run_eval.py` | 180 | 合成评估 |
| `tests/test_adjudicator.py` | 784 | 测试/测试支持 |
| `tests/test_ai_style.py` | 144 | 测试/测试支持 |
| `tests/test_countercheck.py` | 187 | 测试/测试支持 |
| `tests/test_embedded_python.py` | 74 | 测试/测试支持 |
| `tests/test_select_primary_pdf.py` | 158 | 测试/测试支持 |
| `tools/adjudicate_findings.py` | 893 | 执行候选（见正文功能边界） |
| `tools/attach_refutation.py` | 144 | 执行候选（见正文功能边界） |
| `tools/build_claim_ledger.py` | 312 | 执行候选（见正文功能边界） |
| `tools/build_manifest.py` | 112 | 执行候选（见正文功能边界） |
| `tools/check_ai_style.py` | 154 | 执行候选（见正文功能边界） |
| `tools/check_numeric_consistency.py` | 183 | 执行候选（见正文功能边界） |
| `tools/check_presentation.py` | 228 | 执行候选（见正文功能边界） |
| `tools/check_stat_consistency.py` | 530 | 执行候选（见正文功能边界） |
| `tools/countercheck.py` | 193 | 执行候选（见正文功能边界） |
| `tools/install_anti_autoresearch.sh` | 28 | 执行候选（见正文功能边界） |
| `tools/select_primary_pdf.py` | 97 | 执行候选（见正文功能边界） |
