# scientific-agent-skills：三类受众与实际执行体深度解剖

本报告是 **2026-11-23 本地 Git 快照的静态源码审计**。没有安装依赖、执行仓内程序、调用收费 API、重跑测试或验证论文质量；函数存在不等于运行成功，仓内“tested/production-tested”是作者陈述。星数不由 Git 快照证明。行数按 git tracked 文件文本 `splitlines()`，包含注释、docstring、空行，Markdown 中代码块不计可执行源码。测试按目录/文件名识别，候选运行代码仍含校验器、模板及例程，不能全称“自主科研核心”。统一账本见 `inventory_summary.json`、`inventory_files.tsv.gz`；本报告附的逐技能清单另按技能目录人工可复核统计，包含例程，不与中央 runtime 类强行等同。

## 1. 身份与定量家底

- origin：`https://github.com/K-Dense-AI/scientific-agent-skills.git`；HEAD：`ec59a29a03d5c40f73ab40878df04860bc00e876`；工作树干净。
- `pyproject.toml:1–17` 标明 scientific-agent-skills **2.72.0**；`plugin.json:2–14` 品牌为 K-Dense Inc.。本地名、origin、manifest一致；没有从本快照找到旧名 `claude-scientific-skills` 的直接重命名声明，不能凭印象替它补历史。
- **2,780** 个 tracked 文件；**177** 个 `skills/*/SKILL.md`；**1,538** 个 Markdown、**329,700** 行。
- 源码扩展 **776 文件 / 249,102 行 / 218,126 非空行**：Python 771/248,635，JS 2/223，Shell 3/244。
- 其中测试 **237 文件 / 79,276 行**；明显 examples **8 / 2,139**；其余候选运行程序 **531 / 167,687**。再按位置看，技能内（不含 examples）的程序约 **528 / 165,717**，仓库工具3/1,970。这里绝不是“177份说明书而已”，但也不是一个16万行的统一科学家Agent。
- tracked 总体积约290.9 MB，含文档图片、夹具与资源；体积不等于逻辑代码量。没有 node_modules/外部依赖整包混入源码统计；`_common.py` 等重复支持模块、脚手架仍逐文件计数，未做相似代码去重。

## 2. 给 AI 读的：专科知识库与可选工具箱

### 2.1 加载机制与职责边界

`AGENTS.md:11–23` 明确拒收 broad orchestrator：每个技能对应一个科学包、数据库、平台或窄工作流。`AGENTS.md:27–53` 规定 `SKILL.md + references/ + scripts/ + assets/`，测试放顶层tests。177技能通过 frontmatter 的 name/description触发，由宿主Agent决定调用；根 `plugin.json` 是分发清单，不内置模型服务、MCP server、全局自治循环或浏览器。

第一层读 SKILL.md 的任务边界/工作流；第二层按问题打开 references；第三层调用 bundled scripts 或由Agent依据文档编写 Python/CLI。这是“教会宿主科学软件怎么正确用”，不是“安装后一个本地守护进程自动发论文”。根 `pyproject.toml` 只装scanner、pytest、dotenv；不安装177套科研依赖（README:305–311也明确说了）。

### 2.2 真方法论在哪里

- **检索不是猜答案**：`skills/database-lookup/SKILL.md:13–42` 收录80数据库检索合同；先实体/物种/参考基因组/日期/筛选条件，再选权威库，明确服务器过滤与本地过滤、分页、期望计数、限流、来源记录；API返回不能当外部指令。该技能有83文件、零独立程序，API样例属于让Agent执行的配方。
- **写作先绑证据**：`scientific-writing/SKILL.md:11–79` 分开起草、事实核验、投稿批准；每个事实/数字连到证据ID，人工核对定位与元数据；不凭流畅度生成结果。它不是自动发现事实的LLM内核；配套脚本做登记表与结构一致性检查。
- **实验设计先于数据**：experimental-design提供随机化、区组、因子、混杂/单位判断；statistical-power提供功效/样本量；statistical-analysis提供检验选择和前提诊断；scientific-critical-thinking提供偏倚/证据等级。这些规范能约束宿主，但正确性仍要由设计、数据及统计假设负责。
- **假说与审稿不是神秘人格**：hypothesis-generation、scientific-brainstorming、peer-review以证据、替代解释、可区分预测、评价rubric、decision log组织输出；配套JSON/schema/验证脚本主要检验字段与证据账目，不能证明新假说正确。`scientific-brainstorming/scripts/validate_register.py:592–598`与`session_scaffold.py:163–169`甚至明确声明“No network or LLM calls”：它们搭账本和校验，不自己生成科学洞见。
- **Arbor是一个局部迭代方法**：`skills/arbor/scripts/tree.py:139,184,213,242,274,297,319,364,496` 实现init/add-node/baseline/evidence/propagate/prune/merge/cycle/validate持久化树。它保存假设、评估分数与状态；没有在该脚本找到自动调用模型提出改动→训练→跑评估的完整闭环。Agent或外部评估器负责真正实验，不能把tree.py当自动训练器。
- consciousness-council、dhdna-profiler、what-if-oracle等是思维组织或反思rubric，无独立执行程序；不应把名称读成意识测量装置。

### 2.3 它覆盖的科学工作

分子/药物（RDKit、DeepChem、DiffDock、MedChem、PyTDC），组学（Scanpy、PyDESeq2、scVI、pysam、QIIME2），质谱（pyOpenMS），材料（pymatgen、pycalphad）、物理（Cantera、PyBaMM、FluidSim、QuTiP/Qiskit）、统计/优化（PyMC、pymoo、statsmodels）、科研通信（文献、引用、论文、图表、海报、PPT）及实验平台（Opentrons、PyLabRobot、OMERO、DNAnexus、Benchling等）。**覆盖面不能替代同一个任务的端到端验证**；许多技能只教如何调用上游包，其算法实现来自这些上游项目。

## 3. 给人类读的：安装、使用演示、商业产品与真实安全报告

- README:29–37说明177技能与curated documentation定位；:166–256有 npx/gh skill/Agent Plugins/宿主路径安装；:343起给药物发现、单细胞、多组学等拼接示例。它们是用户向Agent下需求的例子，不是对应已注册的流水线CLI。
- README:20–25首页突出 **K-Dense BYOK**（另一个仓库）、教程视频、社交账号；:689起博客/托管产品比较，引导到自家研究工作台及商业生态。本仓MIT；BYOK工作台、云计算Modal、托管产品不包含在本仓，也没有理由把商业入口算作恶意功能。
- CONTRIBUTING、AGENTS、CODE_OF_CONDUCT、SECURITY、CITATION和GitHub Actions属于贡献治理/可信度建设，而非科研执行核心。README:97仍有“181 skills”的旧更新叙述，当前177应按实物清单，而非挑营销数字。
- **安全报告不是全绿勋章**：`docs/security-report.md:3–11`记录2026-11-16，177技能、472条finding、32 critical、21 high、159/177被标safe；本次0重扫、177沿用历史结果，scanner 2.3.2 / claude-opus-5。这是LLM scanner分类，不是本审计确认的漏洞，也不是159个技能完全安全的证明。
- `scan_skills.py`与`scan_pr_skills.py`是真扫描驱动/汇总代码，不做科研；README:264–286也明确有限审核、选择性安装、查看贡献记录、扫描复查。把它宣传成“官方审核100%放心”与仓内文字相反。

## 4. 真正在本机运行的程序：入口→调用→产出

|程序/入口证据|真实操作|产出与边界|
|---|---|---|
|`skills/statistical-analysis/scripts/assumption_checks.py:62,126,251`|SciPy Shapiro、方差齐性；statsmodels回归诊断、Breusch–Pagan、Durbin–Watson、VIF|诊断统计与图；不是只打印统计话术，也不是代替实验单位判断|
|`skills/statistical-power/scripts/power.py:34,88,279,320,344,392`|统一封装statsmodels power求解器，相关/回归功效用SciPy分布及brentq|样本量、功效、MDE与功效曲线；结果依赖效应大小和模型假设|
|`skills/statistical-power/scripts/simulate_power.py:92,142,184,202,230,269`|重复生成数据→拟合/检验→拒绝比例→Wilson区间→搜索样本量；含两组差、logistic、cluster、重复测量例子|蒙特卡洛功效估计，不是LLM猜数字|
|`skills/pydeseq2/scripts/run_deseq2_analysis.py:58,76,122,178,217,278,354`|读CSV、计数验证、设计/对比设置，拟合DeseqDataSet/DeseqStats、LFC收缩、导出、画图|差异表达表与图；依赖真实pydeseq2/numpy/pandas，不内置DESeq算法|
|`skills/scanpy/scripts/`、`skills/rdkit/scripts/`、`skills/pyopenms/scripts/`|上游库QC、聚类/分子描述符/质谱流程包装与数据验证|可执行领域任务；不等于算法由本仓原创|
|`skills/research-lookup/scripts/research_lookup.py:133,213–236,273,746,802,971,1193`|宿主执行Parallel CLI或显式Parallel Chat API、OpenRouter Perplexity；管理请求/错误/来源封套|检索JSON/可读结果；需要对应CLI及密钥，付费检索不是离线能力|
|`skills/paper-lookup/scripts/`、`skills/ncats-arax/scripts/arax_client.py`|文献API及ARAX知识图谱客户端；后者2109行含请求、响应与验证逻辑|返回外部数据库证据，质量/可用性依赖服务，不是本地知识库全量复制|
|`skills/generate-image/scripts/`、scientific-schematics、infographics|模型发现、图像API请求、迭代审阅或排版辅助|真实图像文件与日志；依赖OpenRouter/图像模型费用，外部模型承担生成|
|`skills/pptx-posters/scripts/_pptx.py`、scientific-slides|读写PPTX/素材、检查版式、组装报告|演示文件不是新实验数据；需要python-pptx等依赖|
|exploratory-data-analysis、genomic-coordinates、uncertainty-and-units等scripts|文件读取/统计摘要、坐标及单位转换、数值不确定性传播与质量审计|确定性数据处理及检查；不是自治研究Agent|
|`scripts/generate_skill_image.py`，scan_*|为文档生成流程示意图、扫描风险/生成报告|仓库维护工具，应从科研核心拆开|

**浏览器控制也要分清**：fictiv、ginkgo-cloud-lab等SKILL告诉宿主在已登录浏览器报价/下单，其中CAD检查等脚本是本地辅助；不是根插件自带浏览器或机器人。adaptyv、benchling等有的仅文档/SKD调用样例，执行与凭据交给宿主和上游客户端。不要把所有API名称都当本仓实现了专用SDK。

## 5. 依赖、运行条件与完整科研流程

典型使用是：用户问题→宿主选窄技能→读适用reference→检查CPU/GPU/数据约束→安装目标领域依赖→调用本地脚本或上游API→Agent解释输出→证据与格式检查→人工决定下一步。它可以被外层Agent串成研究流程，但没有单一“sleep research”调度器负责这一切。

- 本地数值技能依赖SciPy、statsmodels、Scanpy/PyDESeq2/RDKit等各自环境，可能需要GPU、模型权重、系统CLI。
- 数据平台/实验平台依赖账户、URL、权限；Exa、Parallel、OpenRouter、Benchling、DNAnexus、Modal等按技能配置，不能因装了Skill就凭空获得。
- 真工作可能昂贵：模型训练、蛋白折叠、云模拟、实验室订购。文档中的批准门、版本固定、只读默认等是指令约束；是否被宿主严格执行另需运行验证。
- 顶层tests有真实断言和fixtures，例如 `tests/statistical-power/test_scripts.py:45–61`按Cohen经典效应d=0.5断言64/组、r=0.30断言85，以及往返/单调性测试；:32–34使用importorskip，缺依赖会跳过，不代表科学计算已覆盖。证明作者投入了测试工程；本次未运行，不能报告“79k行测试全部通过”。测试常对错误路径、结构、mock API和数值边界验证，不能换算成论文成功率。更关键的是 `.github/workflows/skill-tests.yml:80起`明确CI刻意不运行所有技能的隔离依赖套件：完整运行会建立约100个含torch/qiskit/scanpy的环境，部分还需CUDA/JDK/MATLAB；CI选择标准库技能及meta检查。因此绿色CI徽章也不表示177套科学软件全部集成验证。

## 6. 全177技能库存（非测试程序含例程；0表示无独立源码文件，不表示无可执行文档示例）

|技能目录|文件数|非测试程序文件 / 物理行|第一层职责（frontmatter，原文截取）|
|---|---:|---:|---|
|`13c-metabolic-flux`|16|3 / 692|Estimates intracellular metabolic fluxes from steady-state carbon-13 isotope-tracing measurements using validated atom maps, mfapy isotope simulation, constrained multistart fitting, and flux-profile diagnostics. Use for|
|`adaptyv`|2|0 / 0|Uses the Adaptyv Bio Foundry API and Python SDK to design protein characterization experiments, estimate costs, submit sequences, monitor laboratory progress, and retrieve results. Applies to Adaptyv Foundry, its target |
|`aeon`|12|0 / 0|This skill should be used for time series machine learning tasks including classification, regression, clustering, forecasting, anomaly detection, segmentation, and similarity search. Use when working with temporal data,|
|`alphagenome`|8|4 / 1,497|Looks up precomputed AlphaGenome Atlas effects for any GRCh38 single-nucleotide variant (AVI score with Phred and 18 SHAP feature attributions, plus raw and quantile scores for RNA-seq, DNase, ATAC, ChIP-TF, ChIP-histone|
|`analytical-method-validation`|17|8 / 3,363|Plans, executes, and documents validation, verification, and transfer of analytical procedures under the governing framework - ICH Q2(R2) and Q14, USP <1220>/<1225>/<1226>, ICH M10 bioanalytical, CLSI EP, or ISO/IEC 1702|
|`anndata`|6|0 / 0|Handles annotated matrices in single-cell analysis, .h5ad and Zarr files, and integration with the scverse ecosystem. This is the data format skill—for analysis workflows use scanpy; for probabilistic models use scvi-too|
|`arbor`|6|1 / 607|Applies Arbor Hypothesis Tree Refinement to research artifacts with repeatable evaluators, including model training, agent harnesses, data synthesis and benchmark optimization. Uses persistent hypotheses, isolated experi|
|`arboreto`|5|1 / 104|Infers candidate gene regulatory networks from bulk or single-cell expression data using AertsLab Arboreto GRNBoost2 and GENIE3. Use for transcription factor-target association ranking, compatible Dask execution, sparse |
|`astropy`|9|0 / 0|Core Python library for astronomy and astrophysics workflows that need Astropy APIs, including units/quantities, coordinates, FITS I/O, tables, time systems, WCS, and cosmology. Use when implementing or debugging astrono|
|`autoskill`|16|10 / 836|Analyzes user-requested Screenpipe history windows to detect repeated research workflows, match existing scientific skills, and stage new skill drafts or composition recipes for review. Requires a reachable Screenpipe HT|
|`benchling-integration`|6|0 / 0|Benchling Python SDK and REST API integration for registry entities, inventory, ELN entries, workflows, Benchling Apps, and Data Warehouse queries. Use when automating lab data with benchling-sdk or the v2 API.|
|`bgpt-paper-search`|1|0 / 0|Searches BGPT scientific papers by topic or DOI and retrieves claim-level evidence extracted from full text, including experiments, reported statistics, scope, limitations, and provenance. Use for literature reviews, evi|
|`bids`|8|1 / 101|Organizes, queries, validates, and converts Brain Imaging Data Structure (BIDS) datasets. Supports   organizing neuroscience and biomedical data (MRI, EEG, MEG, iEEG, PET, microscopy,   NIRS, motion capture, EMG, MR spec|
|`biopython`|9|0 / 0|Provides Biopython workflows for sequence manipulation, file parsing (FASTA/GenBank/PDB), phylogenetics, and programmatic NCBI/PubMed access (Bio.Entrez). Supports batch processing, custom molecular-biology pipelines, BL|
|`bioservices`|8|4 / 1,447|Provides a Python interface to bioinformatics services including UniProt, KEGG, ChEMBL, Reactome, QuickGO, and UniChem. Used for cross-database protein annotation, pathway retrieval, chemical identifier mapping, and inte|
|`bulk-rnaseq`|8|3 / 580|Prepares bulk RNA-seq FASTQ, Salmon, STAR or featureCounts output for gene-level differential expression. Covers nf-core/rnaseq and standalone quantification, biological replication, strandedness, reference provenance, v|
|`cantera`|4|1 / 198|Runs Cantera homogeneous chemical reactors and evaluates ignition delay with mechanism provenance, conservation checks, and numerical refinement. Use for combustion kinetics, closed adiabatic ideal-gas constant-volume or|
|`cellprofiler`|4|1 / 216|Runs reproducible CellProfiler microscopy pipelines for nuclear segmentation, cell counts, and per-object fluorescence measurements. Supports image/channel manifests, headless batch execution, segmentation overlays, and |
|`cellxgene-census`|5|0 / 0|Queries the CZ CELLxGENE Census programmatically for versioned public single-cell and spatial transcriptomics data. Use when you need population-scale cell metadata, gene expression slices, Census summary counts, source |
|`cirq`|8|0 / 0|Google quantum computing framework. Use when targeting Google Quantum AI hardware, designing noise-aware circuits, or running quantum characterization experiments. Best for Google hardware, noise modeling, and low-level |
|`citation-management`|22|8 / 3,381|Comprehensive citation management for academic research. Search OpenAlex, PubMed, and Google Scholar for papers, extract accurate metadata, validate citations, and generate properly formatted BibTeX entries. This skill s|
|`clinical-decision-support`|28|8 / 2,287|Prepares and validates research-only clinical decision-support evaluation, evidence-profile, cohort, survival, biomarker/model, privacy, and governance artifacts. Supports aggregate or synthetic research documentation an|
|`clinical-reports`|36|9 / 2,893|Creates safety-bounded draft structures and runs local deterministic checks for clinical case, diagnostic, trial, safety, and aggregate research reports. Use only with synthetic, de-identified, or aggregate inputs and ve|
|`cobrapy`|3|0 / 0|Performs constraint-based metabolic modeling with COBRApy, including FBA, pFBA, FVA, gene knockouts, flux sampling, growth media, production envelopes, gap filling, and SBML model validation for systems biology and metab|
|`consciousness-council`|2|0 / 0|Structures a multi-perspective council exercise for decisions, research trade-offs, and creative challenges. Simulates thinking archetypes, separates evidence from assumptions and values, and synthesizes a conditional re|
|`dask`|8|0 / 0|Scales pandas, NumPy, and custom Python research workflows beyond memory or across clusters with Dask. Covers DataFrames, Arrays, Bags, Futures, chunking, schedulers, and distributed diagnostics. Use for partitioned file|
|`database-lookup`|83|0 / 0|Queries documented public database APIs with explicit endpoints, filters, pagination, and provenance. Used when a scientific, regulatory, financial, or other database-backed fact must be retrieved reproducibly from a nam|
|`datalad`|4|0 / 0|Retrieves, versions, and publishes scientific datasets with DataLad and git-annex, and captures computational provenance with datalad run, rerun, and containers-run. Use when cloning or fetching data from OpenNeuro, DAND|
|`datamol`|10|0 / 0|Pythonic wrapper around RDKit with simplified interface and sensible defaults. Preferred for standard drug discovery including SMILES parsing, standardization, descriptors, fingerprints, clustering, 3D conformers, parall|
|`deepchem`|10|4 / 893|Builds molecular property prediction and MoleculeNet workflows with DeepChem, including SMILES featurization, scaffold or grouped holdouts, masked labels, graph models and explicit pretrained encoder transfer. Used for A|
|`deepspot-m`|3|0 / 0|Generates transcriptome-wide virtual spatial transcriptomics from H&E histology with DeepSpot-M. Used for predicted log1p-CPM expression from 224x224 tiles at about 20x, querying the released protein-coding gene panel by|
|`deeptools`|10|2 / 798|NGS analysis toolkit. BAM to bigWig conversion, QC (correlation, PCA, fingerprints), heatmaps/profiles (TSS, peaks), for ChIP-seq, RNA-seq, ATAC-seq visualization.|
|`depmap`|3|1 / 201|Retrieves and analyzes Cancer Dependency Map (DepMap) release data, including CRISPR Chronos gene effects, cancer model annotations, omics biomarkers, and PRISM drug sensitivity. Supports cancer-selective dependency, co-|
|`dhdna-profiler`|2|0 / 0|Applies the DHDNA framework as an exploratory rubric for reasoning and writing patterns in supplied text. Used for explicit requests for DHDNA, cognitive-style reflection, a thinking-pattern profile, or comparisons of te|
|`diffdock`|9|3 / 922|Predicts protein-small-molecule binding poses with DiffDock and DiffDock-L from PDB or sequence plus SMILES/SDF/MOL2. Covers batch docking, pose triage, confidence interpretation, and validation. Use for molecular dockin|
|`dnanexus-integration`|12|2 / 1,174|Builds and operates reproducible genomics workloads on DNAnexus with the dx CLI, dxpy, apps/applets, native workflows, dxCompiler, and Nextflow. Supports DNAnexus data transfers, dxapp.json development, execution monitor|
|`esm`|8|1 / 106|Uses the Biohub esm Python SDK for ESM3 protein generation, ESMC embeddings, and ESMFold2 all-atom folding. Applies to local model inference and Biohub hosted clients, including former Forge workflows; distinguishes the |
|`etetoolkit`|9|2 / 949|Analyzes, manipulates, compares, annotates, and visualizes phylogenetic or other hierarchical trees with ETE 4. Supports Newick/Nexus tree I/O, topology edits and pattern matching, Robinson-Foulds comparisons, gene-tree |
|`exa-search`|5|2 / 317|Searches scientific and technical web content with Exa and extracts page or PDF text from URLs in batches. Supports scholarly discovery with the publication category and academic domain filters. Applies to requests to se|
|`experimental-design`|7|2 / 455|Designs experiments and studies BEFORE data is collected — choosing a design, randomizing, blocking, and laying out treatment combinations so results are interpretable. Use whenever someone is planning a study, asks how |
|`exploratory-data-analysis`|21|13 / 3,850|Performs bounded, local exploratory analysis of explicitly supported scientific files. Supports redacted CSV/TSV/JSON profiles; optional NumPy, HDF5, FASTA/FASTQ, and basic image metadata inspection; missingness/leakage |
|`fictiv`|10|3 / 488|Operates Fictiv (app.fictiv.com), the on-demand manufacturing platform, end to end in the user's browser. Covers uploading CAD parts, configuring process, material, finish, threads, tolerances and inspections, getting in|
|`flowio`|7|1 / 465|Reads, inspects, and writes Flow Cytometry Standard (FCS) 2.0, 3.0, and 3.1 files with FlowIO. Use for low-level FCS metadata and channel inspection, NumPy event extraction, multi-dataset files, table export, and FCS 3.1|
|`flowkit`|4|1 / 204|Analyzes flow cytometry data with FlowKit, including spillover compensation, logicle and biexponential transforms, hierarchical gating, GatingML strategies, and supported FlowJo 10 workspaces. Use for reproducible gate c|
|`fluidsim`|17|10 / 3,585|Plans, configures, inspects, restarts, and analyzes bounded FluidSim computational-fluid-dynamics simulations with explicit numerical-validity and HPC safety checks. Use for FluidSim solver selection, parameter review, F|
|`folklore-variant-evidence`|2|0 / 0|Retrieves ClinGen gene-disease validity assertions for a public gene or disease, and reviews source-linked public evidence and literature for one supported GRCh38 germline nuclear SNV or simple indel through Folklore Cli|
|`generate-image`|3|1 / 802|Generates or edits images with AI models through the OpenRouter Image API (Gemini, Seedream, Recraft, GPT-Image, Riverflow). Use for photos, illustrations, artwork, concept art, visual assets, logos, and image editing or|
|`geniml`|14|8 / 2,671|Supports audited local Geniml genomic-interval workflows: validate BED and universe contracts, plan Region2Vec or scEmbed runs, inspect model/tokenizer compatibility, and assess consensus universes.|
|`genomic-coordinates`|10|5 / 1,798|Converts genomic intervals between coordinate conventions, normalises and compares variant representations, and detects assembly or contig-naming mismatches before they corrupt an analysis. Used whenever coordinates cros|
|`genomic-intelligence`|6|0 / 0|Predicts regulatory features, gene structure, and expression directly from DNA sequence using Genomic Intelligence's hosted transformer DNA language models — no local GPU or model weights. Six tasks over a REST API and a|
|`geomaster`|18|1 / 153|Supports geospatial research workflows for remote sensing, vector and raster GIS, spatial statistics, terrain and network analysis, and machine learning for Earth observation. Use when processing satellite imagery, align|
|`geopandas`|14|7 / 2,116|Guidance and local audit tools for Python workflows that directly use GeoPandas GeoSeries, GeoDataFrame, spatial operations, or vector-data I/O.|
|`get-available-resources`|9|5 / 3,060|Detects host inventory and effective CPU, memory, disk, scheduler, container, and accelerator limits when a user asks for resource-aware planning or before a clearly resource-sensitive local workload. Produces a redacted|
|`gget`|9|3 / 649|Queries 20+ bioinformatics resources through CLI/Python. Supports quick lookups of gene info, BLAST/BLAT, viral sequence downloads, PDB/mmCIF structures, G2P residue annotations, enrichment analysis, OpenTargets, COSMIC,|
|`ginkgo-cloud-lab`|19|0 / 0|Guides protocol selection, input preparation, pricing checks, and browser ordering on Ginkgo Bioworks Cloud Lab (cloud.ginkgo.bio). Applies to cell-free, E. coli, and Pichia protein expression; HiBiT, A280, and LabChip r|
|`glycoengineering`|5|2 / 190|Analyzes and engineers protein glycosylation by scanning canonical N-glycosylation sequons, describing S/T-rich regions, checking curated glycan evidence, and preparing NetNGlyc, NetOGlyc and GlycoSHIELD workflows. Use f|
|`gtars`|15|8 / 2,183|Supports Gtars for local genomic interval models and set algebra, overlaps and counts, consensus and coverage, tokenization, fragment processing, and refget/BEDbase planning across Python, Rust, and the CLI.|
|`histolab`|8|0 / 0|Extracts and preprocesses whole-slide histology image tiles with Histolab. Use for WSI inspection, tissue masks, random/grid/score-based tile extraction, H&E stain normalization, and tile dataset preparation. For multipl|
|`hugging-science`|8|1 / 382|Discovers and evaluates scientific datasets, models, methodology posts, and Spaces through the Hugging Science catalog. Used when selecting scientific ML resources in biology, chemistry, genomics, materials, climate, phy|
|`hypogenic`|18|7 / 2,582|Plans and audits use of ChicagoHAI HypoGeniC/HypoRefine for LLM-assisted hypothesis generation from labeled text datasets. Use for the `hypogenic` package, its task configs, hypothesis banks, or HypoBench datasets—not fo|
|`hypothesis-generation`|27|8 / 3,402|Formulates evidence-bounded scientific questions, candidate hypotheses, rival explanations, causal or associational claims, discriminating predictions, measurements, and preregistration-ready analysis plans. Used when tu|
|`imaging-data-commons`|15|1 / 138|Queries and downloads public cancer imaging data from NCI Imaging Data Commons. Supports IDC collection discovery, DICOM access, radiology (CT, MR, PET) and pathology AI datasets, metadata SQL, visualization, licensing, |
|`infographics`|8|2 / 1,667|Creates and reviews infographics with Nano Banana 2 via OpenRouter. Use for statistical summaries, timelines, comparisons, processes, and visual explanations with supplied data or optional Sonar research. Supports ten la|
|`iso-standards-readiness`|32|10 / 2,332|Prepares and structurally reviews readiness evidence for ISO management-system and laboratory-competence standards - ISO 13485 medical device QMS, ISO 14971 device risk management, ISO/IEC 17025 testing and calibration l|
|`lab-hardware-cad`|13|4 / 1,892|Designs custom laboratory hardware as parametric build123d models and exports fabrication artifacts as STEP, STL, and DXF files - microfluidic chips and molds, optomechanical mounts and breadboard adapters, cuvette and m|
|`labarchive-integration`|8|3 / 1,099|Integrates with the official LabArchives ELN REST-like API and Inventory API v1. Supports regional endpoint selection, signed-request construction, user authorization and UID flows, local LA container validation, and ver|
|`lamindb`|8|0 / 0|Manages biological datasets and models with LaminDB, including artifact registration, lineage tracking, schema validation, Bionty ontology annotation, query/search, collections, branches, storage, and workflow integratio|
|`latchbio-integration`|11|1 / 292|Builds, registers, debugs, and operates bioinformatics workflows on Latch using the Python SDK, CLI, Latch Data and Registry, Nextflow, Snakemake, programmatic execution, and Latch MCP. Use when authoring or deploying La|
|`latex-posters`|17|3 / 1,202|Creates research posters in LaTeX using beamerposter, tikzposter, or baposter. Use for conference posters, academic presentations, multi-column scientific layouts, figure integration, typography, compilation, and PDF pre|
|`liteparse`|7|1 / 189|Local document and PDF parsing that returns spatial text with bounding boxes. Use for extracting text from PDFs, DOCX, Office files, and images; running OCR on scans; producing layout-preserved JSON for RAG; batch-ingest|
|`literature-review`|12|5 / 1,629|Conducts systematic, scoping, and narrative literature reviews using PubMed, arXiv, bioRxiv, Semantic Scholar, and other appropriate sources. Use for research synthesis, reproducible literature searches, screening, citat|
|`mageck`|4|1 / 259|Analyzes pooled CRISPR screen FASTQ reads and guide-count matrices with MAGeCK, validates guide libraries and contrasts, measures replicate and library QC, and produces gene hit rankings with effect sizes and FDR. Use fo|
|`marine-carbonate-chemistry`|4|1 / 222|Solves seawater carbonate chemistry with PyCO2SYS for chemical oceanography, ocean acidification, and marine carbon-cycle research. Use for paired total alkalinity, dissolved inorganic carbon, pH, or seawater pCO2/fCO2 m|
|`markdown-mermaid-writing`|38|0 / 0|Writes scientific Markdown documentation and Mermaid diagrams for workflows, relationships, timelines, and schemas. Provides syntax references, document templates, accessibility guidance, and version-aware rendering chec|
|`market-research-reports`|26|8 / 2,543|Builds evidence-traceable market research reports and assumption-driven market sizing or forecast scenarios. Use for market definition, industry and customer evidence, competitive landscapes, TAM/SAM/SOM reconciliation, |
|`markitdown`|11|3 / 921|Converts heterogeneous documents and selected URIs to Markdown with Microsoft MarkItDown for text analysis, search, and LLM/RAG ingestion. Covers safe local conversion, streams, Office/PDF/data formats, batch workflows, |
|`matchms`|8|1 / 624|Processes, cleans, compares, and searches tandem mass spectra with matchms. Use for MS/MS file I/O, metadata harmonization, peak filtering, spectral similarity, library matching, score matrices, and molecular-similarity |
|`matlab`|20|8 / 2,245|Builds, reviews, migrates, and plans MATLAB or GNU Octave numerical workflows. Use for arrays, tabular/time data, tests, projects, graphics, MAT files, and explicit Python interoperability.|
|`matplotlib`|7|2 / 841|Creates and customizes scientific plots with Matplotlib. Used for fine-grained control over plot elements, novel plot types, and scientific workflows. Export to PNG/PDF/SVG for publication. For quick statistical plots us|
|`medchem`|4|1 / 330|Applies medicinal chemistry filters for compound triage, using drug-likeness rules (Lipinski, Veber, CNS), structural alert catalogs (PAINS, NIBR, ChEMBL), complexity metrics, and the medchem query language for library f|
|`modal`|13|0 / 0|Modal is a serverless cloud platform for running Python on demand, including on-demand GPUs. Use when deploying or serving AI/ML models, running GPU-accelerated workloads (training, fine-tuning, inference), serving web e|
|`molecular-dynamics`|3|0 / 0|Runs and analyzes molecular dynamics simulations with OpenMM and MDAnalysis. Sets up protein/small molecule systems, defines force fields, runs energy minimization and production MD, and analyzes trajectories (RMSD, RMSF|
|`molfeat`|5|0 / 0|Featurizes small molecules with Molfeat for QSAR/QSPR, chemical similarity, virtual screening, and molecular ML. Covers ECFP/MACCS fingerprints, RDKit descriptors, pharmacophores, pretrained embeddings, configuration per|
|`ncats-arax`|4|1 / 2,109|Queries the NCATS Translator ARAX production API for bounded, typed, provenance-rich one-hop and endpoint-pinned two-hop biomedical knowledge-graph relationships. Use for Biolink-constrained RTX-KG2 lookup, explicit sele|
|`networkx`|7|0 / 0|Creates, analyzes, and visualizes complex networks and graphs in Python with NetworkX. Use when working with network/graph data structures, computing graph algorithms (shortest paths, centrality, clustering), detecting c|
|`neurokit2`|20|7 / 2,449|Builds and audits reproducible NeuroKit2 research workflows for physiological time-series preprocessing, event/interval analysis, multimodal alignment, variability, and complexity. Use when code imports neurokit2 or need|
|`neuropixels-analysis`|19|8 / 1,371|Analyzes Neuropixels extracellular recordings end-to-end with SpikeInterface. Covers loading SpikeGLX/Open Ephys/NWB data, preprocessing, drift/motion correction, Kilosort4 (and CPU) spike sorting, quality metrics, and u|
|`nextflow`|8|0 / 0|Builds, runs, and debugs Nextflow DSL2 pipelines and nf-core workflows. Use for Nextflow, nf-core, .nf files, nextflow.config, processes/channels/operators, samplesheets, nf-test, modules/subworkflows, container and exec|
|`nmrglue`|4|1 / 190|Processes calibrated one-dimensional complex NMR free-induction decays with nmrglue into phased spectra, peak candidates, and signed integration regions. Use for raw 1D NMR processing, ppm-axis verification, apodization,|
|`nwb-conversion`|5|1 / 249|Converts neuroscience acquisition data to Neurodata Without Borders files with NeuroConv and PyNWB, preserves metadata and timebases, checks evidence-based clock alignment, and produces schema validation, NWB Inspector f|
|`omero-integration`|15|5 / 1,911|Inspects and automates microscopy data workflows against OMERO.server with omero-py, BlitzGateway, OMERO CLI, tables, annotations, ROIs, rendering, and documented OMERO.web APIs. Use this skill for scoped OMERO inventory|
|`onekgpd`|6|2 / 1,329|Queries the 1000 Genomes Project dataset (3,202 whole-genome-sequenced   individuals, GRCh38) at the level of individual participants.   Use when a question is about individuals or variants in the 1000 Genomes|
|`ontology-term-resolution`|12|7 / 1,901|Resolves free-text scientific labels to ontology term IDs and validates existing CURIEs against the EBI Ontology Lookup Service (OLS4). Also looks up prefixes in Bioregistry, resolves compact identifiers via Identifiers.|
|`open-notebook`|9|4 / 354|Organizes research with the self-hosted Open Notebook alternative to NotebookLM. Supports source ingestion (PDFs, web pages, audio, video, and Office documents), cited document chat, text and vector search, notes, custom|
|`openpiv`|6|4 / 500|Performs Particle Image Velocimetry (PIV) analysis with OpenPIV. Use when extracting velocity fields from PIV image pairs, analyzing fluid dynamics or flow visualization experiments, cross-correlating interrogation windo|
|`opentrons-integration`|16|6 / 592|Authors, reviews, migrates, simulates, and troubleshoots official Opentrons Python Protocol API v2 protocols for Flex and OT-2 robots. Use for robot-specific liquid handling, deck and labware setup, pipettes, modules, ru|
|`optimize-for-gpu`|16|0 / 0|GPU-accelerates scientific Python on NVIDIA hardware and verifies that the result is correct and faster. Use for CUDA/GPU optimization; CPU-bound NumPy, SciPy, pandas, scikit-learn, NetworkX, scikit-image, vector-search,|
|`pacsomatic`|7|1 / 891|Prepares and launches nf-core/pacsomatic matched tumor-normal PacBio HiFi genomics workflows from unaligned BAM inputs. Supports samplesheet generation, pinned Nextflow launch artifacts, local checks, LSF/Slurm/PBS Pro/S|
|`paper-lookup`|24|5 / 1,446|Searches 18 scholarly APIs for papers, preprints, citations, open-access full text, repository records, and journal OA status, and returns results with reproducible provenance. Covers PubMed, PMC, Europe PMC, bioRxiv, me|
|`paperclip`|7|0 / 0|Searches and reads biomedical papers, FDA/PMDA/EMA documents, clinical trials, and protein records with the GXL Paperclip CLI and Python SDK. Supports source-scoped search, full-text grep, metadata SQL, map/reduce extrac|
|`paperzilla`|2|0 / 0|Reads projects, searches project feeds, and retrieves recommendations and canonical papers in Paperzilla through the pz CLI. Supports recent recommendations, paper details, markdown-based summaries, recommendation feedba|
|`parallel-web`|8|0 / 0|Uses Parallel CLI for web search, URL extraction, deep research, structured data enrichment, entity discovery, and recurring web monitoring. Best for requests that explicitly need current web evidence, academic-source di|
|`pathml`|13|6 / 2,128|Supports local computational pathology research with PathML: slide loading and tiling, preprocessing and QC, h5path storage, multiplex quantification, spatial graphs, and bounded model inference. Use for whole-slide H&E,|
|`pathogen-variant-surveillance`|9|5 / 1,720|Queries public GenSpectrum LAPIS data for pathogen genomic surveillance, current lineage nomenclature, weekly sequence proportions, reporting delays, and descriptive mutation frequencies. Use for variant surveillance, Pa|
|`pathway-enrichment`|6|1 / 266|Performs pathway and gene-set enrichment analysis on gene lists or ranked gene data and interprets the results. Used when the user has a set of genes (differentially expressed genes from PyDESeq2/Scanpy, CRISPR-screen hi|
|`peer-review`|24|8 / 2,302|Prepares evidence-bounded, constructive peer-review drafts and structured manuscript assessments. Supports authorized review of scientific manuscripts, protocols, preprints, or research proposals; reporting-guideline sel|
|`pennylane`|8|0 / 0|Builds and differentiates PennyLane quantum circuits, hybrid PyTorch or JAX models, molecular VQE and QAOA workflows. Use for variational quantum algorithms, quantum machine learning, simulator validation, and moving val|
|`phylogenetics`|3|1 / 279|Builds and analyzes phylogenetic trees using MAFFT multiple sequence alignment, IQ-TREE maximum likelihood with ModelFinder and branch support, and FastTree approximate inference. Uses ETE3 for tree summaries and visuali|
|`pi-agent`|36|0 / 0|Builds with and operates Pi, the minimal terminal coding harness. Use for installing Pi, configuring providers/models/settings/environment variables, creating Pi skills/extensions/packages/themes/prompt templates, embedd|
|`pkpd-modeling`|28|11 / 4,811|Pharmacokinetic and pharmacodynamic modelling and simulation - non-compartmental analysis, compartmental and population PK, PK/PD and exposure-response, TMDD, PBPK orientation, bioequivalence, allometric scaling and firs|
|`polars`|8|0 / 0|High-performance DataFrame library for Python ETL, analytics, and pandas migration. It supports expression-based data manipulation with lazy query optimization, parallel execution, streaming out-of-core processing, Arrow|
|`polars-bio`|7|0 / 0|Performs genomic interval overlap, nearest, merge, coverage, complement and subtraction on Polars DataFrames, and reads or writes BED, VCF, BCF, BAM, CRAM, GFF, GTF, FASTA and FASTQ data. Use for coordinate-aware genomic|
|`pptx-posters`|21|10 / 4,515|Creates and audits editable scientific posters in macro-free PowerPoint (.pptx) from author-approved local content and assets. Used when the requested deliverable is a PowerPoint research/conference poster and exact phys|
|`primekg`|3|1 / 214|Queries a pinned Precision Medicine Knowledge Graph (PrimeKG) CSV for typed gene, drug, disease, and phenotype nodes, direct associations, disease context, and one- or two-hop paths. Use for PrimeKG reproducibility, biol|
|`primer-design`|15|5 / 1,107|Designs and audits PCR and RT-qPCR primers with Primer3, explicit thermodynamic conditions, reference-based off-target amplification searches, and traceable sequence coordinates. Use for designing primer pairs, checking |
|`protocolsio-integration`|15|7 / 2,551|Reads, validates, and safely exports protocols.io data with current official REST/MCP contracts, or creates non-executing mutation plans. The bundled client makes bounded official-host GET requests only with explicit --e|
|`pufferlib`|16|9 / 2,131|Version-aware guidance for PufferLib reinforcement-learning environments, vectorization, policies, PuffeRL training, evaluation, and safe checkpoint review. Covers the native 5.0 build and environment API, published 3.0.|
|`pybamm`|4|1 / 214|Simulates lithium-ion battery charge, discharge and rest experiments with PyBaMM, records parameter-set provenance, checks mesh and solver sensitivity, and compares predicted voltage curves with measured cycling data. Us|
|`pycalphad`|5|1 / 184|Computes finite-temperature CALPHAD equilibria, phase fractions, and phase compositions from thermodynamic TDB databases using pycalphad. Use for alloy phase stability, equilibrium temperature sweeps, tie lines, lever-ru|
|`pydeseq2`|6|1 / 466|Performs bulk RNA-seq differential expression analysis with PyDESeq2, including count validation, formula designs, explicit contrasts, Wald tests, FDR correction, coefficient-matched LFC shrinkage, and result visualizati|
|`pydicom`|13|10 / 4,046|Reads, inspects, writes, transforms, and preflights local DICOM datasets and pixel data. Applies to DICOM metadata, transfer syntaxes, compression plugins, frames, private elements, JSON, and bounded de-identification re|
|`pyhealth`|8|1 / 108|Builds and validates PyHealth clinical machine-learning pipelines for EHR, signals, imaging, and medical codes. Use for PyHealth dataset loading, MIMIC-III/IV, eICU or OMOP prediction tasks, patient-level evaluation, mor|
|`pylabrobot`|16|7 / 1,264|Develops and reviews PyLabRobot lab-automation resources, liquid-handling plans, offline simulations, and supported-device integrations. Supports PyLabRobot protocols and API questions; keep physical execution behind an |
|`pymatgen`|15|9 / 2,607|Analyzes, validates, converts, and transforms materials structures and computed materials data with pymatgen. Use for local phase diagrams, symmetry sensitivity, electronic-structure I/O, and bounded Materials Project qu|
|`pymc`|10|4 / 820|Builds and checks Bayesian models with PyMC, including hierarchical models, NUTS MCMC, variational inference, mutable-data predictions, posterior predictive checks, diagnostics, and PSIS-LOO model comparison. Use for pro|
|`pymoo`|14|5 / 569|Solves and validates single-, multi-, and many-objective optimization with pymoo, including NSGA-II, NSGA-III, MOEA/D, constraints, Pareto approximations, reference directions, and ZDT/DTLZ benchmarks for engineering and|
|`pyopenms`|24|17 / 1,778|Processes mass spectrometry data with pyOpenMS. Supports proteomics and metabolomics workflows—feature detection, peptide/protein identification, label-free quantification, adduct/accurate-mass annotation, and complex LC|
|`pysam`|14|4 / 1,598|Provides Python/HTSlib workflows for genomic files. Used when reading, querying, filtering, or writing SAM/BAM/CRAM, VCF/BCF, FASTA/FASTQ, or tabix data with pysam, including pileup, coverage, indexing, and CRAM referenc|
|`pytdc`|11|6 / 1,739|Provides Therapeutics Data Commons workflows through PyTDC for registry discovery, dataset access, task-aware splits, evaluator metrics, benchmark groups, and bounded molecular-oracle scoring. Use when working with TDC t|
|`pytorch-lightning`|11|3 / 873|Deep learning framework (PyTorch Lightning / lightning package). Organize PyTorch code into LightningModules, configure Trainers for multi-GPU/TPU, implement data pipelines, callbacks, logging (W&B, TensorBoard, MLflow),|
|`pyzotero`|15|0 / 0|Manages Zotero reference libraries using the pyzotero Python client: retrieves, creates, updates, and deletes items, collections, tags, and attachments via the Zotero Web API v3 or local API. Applies when working with Zo|
|`qiime2-amplicon`|3|1 / 272|Processes paired-end 16S amplicon reads into QIIME 2 ASVs and taxonomy with retained artifact provenance. Checks paired FASTQ manifests, primer orientation diagnostics, predicted post-trimming overlap, sample IDs, runtim|
|`qiskit`|15|3 / 684|Builds, simulates, transpiles, and executes quantum circuits with Qiskit and IBM Quantum Runtime. Use for Qiskit 2.x circuits and operators, V2 Sampler or Estimator primitives, target-aware transpilation, local or noisy |
|`qutip`|13|7 / 2,430|Simulate and audit closed and open quantum-system models with QuTiP 5, including deterministic, trajectory, steady-state, spectral, and phase-space workflows. Use for local quantum-dynamics work where physical assumption|
|`rdkit`|11|4 / 972|Cheminformatics toolkit for fine-grained molecular control. SMILES/SDF parsing, descriptors (MW, LogP, TPSA), fingerprints, substructure search, 2D/3D generation, similarity, reactions. For standard workflows with simple|
|`relion`|4|1 / 260|Validates and executes RELION single-particle cryo-EM refinement and half-map postprocessing. Supports STAR optics/acquisition checks, particle-stack consistency, gold-standard half sets, soft-mask validation, diagnostic|
|`relsa-severity-assessment`|9|4 / 2,048|Supports multivariate severity assessment and exploratory endpoint-time score forecasting for laboratory animal studies using the RELSA (RELative Severity Assessment) score and ARIMA-based foRcast forecasting. Use when c|
|`research-grants`|16|0 / 0|Supports research proposal preparation and review for NSF, NIH, DOE, DARPA, and Taiwan NSTC, including opportunity-specific requirements, aims, review criteria, budgets, broader impacts, forms, and resubmissions. Use for|
|`research-lookup`|5|2 / 2,029|Compiles current scholarly evidence for a scientific manuscript or research brief when the user explicitly asks to gather literature, references, background evidence, competing findings, or a manuscript research packet. |
|`rowan`|6|0 / 0|Rowan is a cloud-native molecular modeling and medicinal-chemistry workflow platform with a Python API. Use for pKa and macropKa prediction, conformer and tautomer ensembles, docking and analogue docking, protein-ligand |
|`scanpy`|26|16 / 1,768|Performs Scanpy single-cell RNA-seq QC, normalization, HVG selection, PCA/UMAP/t-SNE, clustering, exploratory marker ranking, pseudobulk preparation, visualization, and Seurat or SingleCellExperiment RDS conversion to h5|
|`scholar-evaluation`|19|8 / 2,362|Provides qualitative-first, evidence-traceable developmental review of scholarly works and audit low-stakes research-assessment rubrics with optional local quality controls. Never use for ranking people or consequential |
|`scientific-brainstorming`|10|4 / 1,754|Facilitates evidence-aware scientific ideation with independent generation, structured discussion, explicit assumptions, transparent evaluation, adversarial review, and decision logs. Use for early-stage research brainst|
|`scientific-critical-thinking`|9|0 / 0|Evaluates scientific claims and evidence quality. Applies to experimental design validity, biases and confounders, statistical interpretation, evidence grading frameworks (GRADE, Cochrane Risk of Bias), and teaching crit|
|`scientific-schematics`|6|3 / 1,150|Generates scientific diagram drafts using Nano Banana 2 AI with smart iterative refinement. Uses Gemini 3.7 Flash for quality review. Refines when the review requests improvement, with at most two generations. Specialize|
|`scientific-slides`|24|7 / 2,939|Builds slide decks and presentations for research talks. Used for making PowerPoint slides, conference presentations, seminar talks, research presentations, thesis defense slides, or any scientific talk. Provides slide s|
|`scientific-visualization`|18|8 / 3,434|Creates and audits truthful, accessible, publication-ready scientific figures with Matplotlib, Seaborn, or Plotly. Use it for figure design, multi-panel layouts, uncertainty and missing-data displays, color/contrast revi|
|`scientific-writing`|31|9 / 2,462|Drafts, revises, and audits scientific manuscripts or reports with explicit evidence provenance, reporting-guideline coverage, authorship accountability, confidentiality controls, and local consistency checks. Use for ma|
|`scikit-bio`|3|0 / 0|Biological data toolkit. Sequence analysis, alignments, phylogenetic trees, diversity metrics (alpha/beta, UniFrac), ordination (PCoA), PERMANOVA, FASTA/Newick I/O, for microbiome analysis.|
|`scikit-learn`|11|2 / 647|Supports machine learning in Python with scikit-learn. Applies when working with supervised learning (classification, regression), unsupervised learning (clustering, dimensionality reduction), model evaluation, hyperpara|
|`scikit-survival`|13|6 / 2,163|Builds, evaluates, and audits right-censored or competing-risk survival workflows with scikit-survival, including leakage-safe preprocessing, model selection, probability prediction, and censoring-aware metrics.|
|`scvelo`|3|1 / 251|Performs RNA velocity analysis with scVelo from spliced and unspliced single-cell RNA counts. Fits deterministic or dynamical models, examines gene phase portraits, builds velocity graphs, estimates relative latent time,|
|`scvi-tools`|9|0 / 0|Fits probabilistic models for single-cell omics, including scVI batch integration, scANVI annotation, totalVI CITE-seq, MultiVI RNA/ATAC integration, and posterior differential expression. Use for generative modeling, re|
|`seaborn`|8|0 / 0|Creates Seaborn statistical visualizations with pandas integration for distributions, relationships, categorical comparisons, regression displays, pair plots, and heatmaps. Supports function and objects interfaces with e|
|`shap`|10|1 / 348|Explain and audit machine-learning predictions with SHAP. Use for selecting SHAP explainers and maskers, computing and validating feature attributions, handling multi-output explanations, and producing local or global SH|
|`simpy`|16|7 / 2,141|Builds, inspects, tests, and analyzes bounded process-based discrete-event simulations with SimPy. Use for event scheduling, resource queues, interrupts, monitoring, independent replications, warm-up, and reproducible ou|
|`stable-baselines3`|8|3 / 516|Trains and evaluates single-agent reinforcement learning with Stable Baselines3 (PPO, SAC, DQN, TD3, DDPG, A2C), Gymnasium custom environments, vectorized rollouts, callbacks, and checkpoint normalization. Applies to rep|
|`statistical-analysis`|7|1 / 352|Guided statistical analysis for research data - test selection, assumption checking, effect sizes, power analysis, Bayesian alternatives, and APA-formatted reporting. Use whenever a user wants to compare groups, test a h|
|`statistical-power`|6|2 / 767|Calculates sample sizes and statistical power for study planning. Applies when someone asks "how many subjects/samples/replicates do I need", wants an a priori power analysis, a minimum detectable effect (MDE), a power c|
|`statsmodels`|10|0 / 0|Fits and diagnoses Python statistical models including OLS, GLM, discrete and mixed models, ARIMA and SARIMAX. Supports coefficient inference, marginal effects, model comparison and time series forecasting with explicit |
|`sympy`|8|0 / 0|Performs exact symbolic mathematics with SymPy for algebra, calculus, equation solving, symbolic linear algebra, physics, and lambdify or LaTeX code generation. Use when a task needs symbolic results, explicit assumption|
|`tamarind`|5|0 / 0|Provides access to a collection of open-source molecular design and structural biology tools on the Tamarind Bio platform, via its REST API or MCP server — no local GPUs required. Tamarind bundles popular open-source mod|
|`tellurium`|5|1 / 209|Simulates biochemical kinetic models from SBML or Antimony with Tellurium and libRoadRunner, checks model units, compares deterministic parameter perturbations, and exports and replays SBML plus SED-ML COMBINE archives. |
|`tiledbvcf`|2|0 / 0|Stores and retrieves genomic variant calls with TileDB-VCF. Use for indexed single-sample VCF/BCF ingestion, incremental cohorts, region and sample queries, streaming results, allele statistics, QC, and VCF/BCF export lo|
|`timesfm-forecasting`|21|10 / 2,872|Performs zero-shot time-series forecasting with Google's TimesFM, including regular-grid CSV preparation, quantile forecasts, XReg covariates, and held-out evaluation. Uses the Apache-licensed TimesFM 2.5 checkpoint by d|
|`torch-geometric`|8|0 / 0|Supports PyTorch Geometric (PyG) graph neural networks — node/link/graph classification, message passing (GCN, GAT, GraphSAGE, GIN), heterogeneous graphs, neighbor sampling, and custom datasets. Use when working with tor|
|`torchdrug`|10|0 / 0|Builds and troubleshoots TorchDrug 0.2.1 workflows for molecular graphs, property prediction, self-supervised pretraining, molecule generation, retrosynthesis, protein representation learning, and knowledge graph reasoni|
|`transformers`|7|0 / 0|Hugging Face Transformers for loading Hub models, running pipeline inference, text generation, and Trainer fine-tuning on NLP, vision, audio, and multimodal tasks. Applies when working with AutoModel, pipelines, tokenize|
|`treatment-plans`|23|8 / 3,015|Formats and structurally validates local treatment-plan documentation after clinical decisions have already been supplied and verified by authorized licensed professionals. Use for source traceability, clinician-authored|
|`umap-learn`|2|0 / 0|Applies UMAP-learn to nonlinear dimensionality reduction, 2D/3D embeddings, clustering preprocessing, supervised or semi-supervised UMAP, DensMAP, AlignedUMAP, and Parametric UMAP workflows.|
|`uncertainty-and-units`|15|7 / 3,876|Tracks physical units and propagates measurement uncertainty in scientific calculations using pint and uncertainties. Use for unit conversion and dimensional checking, GUM uncertainty budgets, Type A and Type B evaluatio|
|`usfiscaldata`|9|0 / 0|Queries the U.S. Treasury Fiscal Data REST API for federal financial data. No API key required. Use for national debt (Debt to the Penny), Daily Treasury Statements, Monthly Treasury Statements, Treasury securities aucti|
|`vaex`|8|0 / 0|Processes large tabular scientific datasets with Vaex expressions, filtered views, streamed statistics, binned visualizations, and file conversion. Use for larger-than-RAM HDF5, Arrow, CSV, or Parquet analysis, virtual f|
|`venue-templates`|32|3 / 754|Prepares journal manuscripts, conference papers, research posters, and grant documents using venue-specific formatting guidance and bundled LaTeX scaffolds. Supports selecting an official template, checking current page |
|`waypoint-bio`|8|2 / 777|Supports work with Outpost Bio's open microbiome foundation models - the Waypoint checkpoints (Waypoint-6m, Waypoint-45m, Waypoint-170m), the Atlas pretraining corpus, the Compass eight-task benchmark, or the `waypoint` |
|`what-if-oracle`|2|0 / 0|Supports structured what-if scenario analysis for research planning, experimental contingencies, and scientific project decisions. Explores favorable, reference, adverse, wild-card, contrarian, and second-order scenarios|
|`zarr-python`|8|0 / 0|Stores and queries chunked N-D scientific arrays with Zarr-Python 3, including codecs, sharding, S3/GCS storage, and NumPy/Dask/Xarray integration. Use for array layout, bounded I/O, format migration, or scientific metad|

## 7. 最大源码文件，可供后续逐段复核

|路径|物理行|
|---|---:|
|`skills/ncats-arax/scripts/arax_client.py`|2,109|
|`skills/get-available-resources/scripts/detect_resources.py`|1,888|
|`tests/ncats-arax/test_scripts.py`|1,635|
|`skills/pptx-posters/scripts/_pptx.py`|1,484|
|`tests/onekgpd/test_scripts.py`|1,373|
|`skills/infographics/scripts/generate_infographic_ai.py`|1,364|
|`skills/pptx-posters/scripts/_manifest.py`|1,324|
|`skills/hypogenic/scripts/_common.py`|1,312|
|`skills/research-lookup/scripts/research_lookup.py`|1,253|
|`skills/treatment-plans/scripts/_common.py`|1,191|
|`scripts/generate_skill_image.py`|1,084|
|`tests/uncertainty-and-units/test_scripts.py`|1,079|
|`skills/hypothesis-generation/scripts/validate_hypothesis_schema.py`|1,078|
|`tests/pyopenms/test_scripts.py`|1,072|
|`tests/pathogen-variant-surveillance/test_scripts.py`|1,067|
|`tests/pkpd-modeling/test_scripts.py`|1,053|
|`tests/bioservices/test_scripts.py`|1,026|
|`tests/analytical-method-validation/test_scripts.py`|1,011|
|`skills/scholar-evaluation/scripts/_common.py`|1,002|
|`skills/analytical-method-validation/scripts/_common.py`|1,000|
|`tests/etetoolkit/test_scripts.py`|996|
|`tests/ontology-term-resolution/test_scripts.py`|993|
|`tests/scientific-slides/test_scripts.py`|972|
|`skills/pydicom/scripts/_common.py`|916|
|`skills/uncertainty-and-units/scripts/check_plausibility.py`|911|
|`skills/exploratory-data-analysis/scripts/_tabular.py`|905|
|`skills/fluidsim/scripts/_schema.py`|904|
|`skills/pacsomatic/scripts/run_pacsomatic.py`|891|
|`skills/scientific-slides/scripts/generate_schematic_ai.py`|857|
|`skills/scientific-schematics/scripts/generate_schematic_ai.py`|857|

## 8. 审计结论

这是规模很大的**科学软件使用知识库 + 领域脚本/客户端 + 校验/测试工程**。价值既在“给AI的正确操作配方”也在实际可执行数值与数据处理工具；既不能叫纯Markdown空壳，也不能把库的广度、API数量和scanner徽章等价成端到端自主科研能力。最适合把已知科研任务交给有终端/文件/联网权限的宿主Agent，最薄弱的是跨领域方法正确性和最终科学判断仍无独立验证器。
