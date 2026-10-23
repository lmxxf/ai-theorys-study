# CodeGraph（及 AirLLM）调研笔记：75k Stars 的“AI 代码知识图谱”到底装了什么？

**日期：2026-10-23**  
**调研对象**：[`colbymchenry/codegraph`](https://github.com/colbymchenry/codegraph)（v1.6.2，~75k Stars） & `lyogavin/airllm`（~40k Stars）

---

## 一、核心结论：现代版 `ctags` + SQLite 全文检索 + 跑分会计魔术

`CodeGraph` 在 GitHub 首页打出 **“Supercharge Claude Code, Cursor, Codex... with Semantic Code Intelligence”**、**“The fastest complete code graph · surgical context”**、**“少调 88% 工具 · 省 62% token · 快 3.6 倍”** 的巨幅海报，狂揽 75k Stars，并在顶部挂载 `getcodegraph.com` 的商业 SaaS 候补名单（Waitlist）。

但直接扒开其仓库源码（`package.json`、`src/mcp/server-instructions.ts`、`src/db/schema.sql` 及 `docs/benchmarks/residual-context-occupancy.md`）后发现：
1. **它既没有向量模型（Embedding），也没有编译器级的语义/类型分析（LSP / Clangd / SCIP）**；
2. 它的底层完完全全就是 **`tree-sitter`（语法树切词）+ 本地 `SQLite FTS5`（字面词法全文检索）**；
3. 它外宣大吹 **“Semantic（语义智能）”**，却在注入给 AI 的系统提示词里亲口承认自己是 **“Lexical（纯字面词法匹配，不含语义）”**；
4. 它宣称的“省 62% token”，是靠**单问一题立刻关会话（One-shot）**算出来的吞吐账；而在真实多轮开发会话中，它自己补测的数据承认：**它会让残留在上下文窗口里的垃圾暴涨 80%～270%（VS Code 仓从 18k 暴涨到 67k tokens）**。

---

## 二、技术底牌拆解：那三样零件到底是什么？

翻开 `package.json` 与架构说明，CodeGraph 的全部工作流只有三步：

### 1. 解析层：`tree-sitter` 静态语法树（非编译器 AST/LSP）
- 用 Rust 内核（编译进了 20 种语言的 `tree-sitter` 语法规则）或 `web-tree-sitter`（WASM 兜底其余语言）扫描代码库，提取两张表：
  - **节点（Nodes）**：函数名、类名、方法名、组件名；
  - **边（Edges）**：`calls`（谁调了谁）、`imports`、`extends`、`implements`，外加作者手写的一堆框架启发式规则（如 React/Vue 路由、React Native 桥接、甚至手写了鸿蒙 `ArkTS` 的 `@Component` 装饰器规则）。
- **硬伤**：`tree-sitter` 只是单文件语法解析器，**没有真正的编译器符号表与类型推导系统**。遇到跨文件多态、C++ 模板/宏重载、或者代码里到处重名的通用方法（如 `init()`、`run()`、`build()`、`handle()`），它根本分不清运行时到底调的是哪一个。

### 2. 存储与检索层：`.codegraph/codegraph.db`（SQLite + FTS5）
- 所有抠出来的符号和边，全部塞进本地一个 SQLite 数据库文件，开启 `FTS5` 全文检索。
- **外宣 vs 内宣的当场打脸**：
  - **README 首页标题**：`Semantic Code Intelligence`（语义代码智能）
  - **`src/mcp/server-instructions.ts`（注入给 Agent 的系统提示词第 63 行）原文**：
    > `Explore matches names and indexed code words lexically, not by meaning; an empty result reports word matches and may suggest indexed candidate names to retry with codegraph_explore.`
    > **（Explore 只是按字面词法匹配名字和代码单词，根本不是按语义匹配！）**

### 3. 接口层：单一 MCP 工具 `codegraph_explore` 的“全家桶一勺烩”
- 默认只向 Agent 暴露一个工具：`codegraph_explore`。
- 当 Agent 传入一个问题或符号名时，CodeGraph 从 SQLite 里捞出匹配的符号，把**所有匹配符号的完整带行号源码（verbatim source）+ 调用链（call paths）+ 爆炸半径（blast radius，谁依赖它）**，一次性拼成一个巨大的文本包（Payload）塞回给 Agent。
- 遇到同名/重载符号怎么办？`server-instructions.ts` 原话：
  > `For an overloaded name it returns every matching definition's body in one call.`
  > **（遇到重载/重名的名字，它会在一次调用里把所有匹配到的函数定义源码全部吐出来！）**

---

## 三、“省 62% Token、少调 88% 工具”是怎么算出来的？（跑分魔术）

README 首页最抓眼球的是 7 个开源仓（VS Code、Excalidraw、Django、Tokio 等）的 A/B 跑分表：

| 仓库 | 工具调用次数（WITH vs WITHOUT） | 耗时 | 读文件次数 | 处理 Token 数 | 账单成本 |
|:---|:---|:---|:---|:---|:---|
| **VS Code** (~11k 文件) | **2 vs 28** | 58s vs 2m10s | **0** vs 12 | -77% | -71% ($0.53 vs $1.80) |
| **Excalidraw** (~640 文件) | **2 vs 43** | 45s vs 2m42s | **0** vs 18 | -84% | -78% ($0.54 vs $2.43) |
| **Tokio** (~790 文件) | **3 vs 29** | 1m03s vs 2m43s | **0** vs 19 | -65% | -64% ($0.66 vs $1.83) |

**这套漂亮数字背后的两个机关**：

1. **用系统提示词硬性“封死”原生工具**：
   在 `src/mcp/server-instructions.ts` 中，CodeGraph 通过 MCP `initialize` 往 Agent 的系统提示词里塞了强指令：
   - `One tool: codegraph_explore — use it instead of reading files`（用它代替读文件，读文件次数当然变成 0 了！）
   - `Trust codegraph's results — don't re-verify them with grep.`（相信它的结果，严禁用 `grep` 二次核实！）
2. **只测“单问一题立刻关窗（One-shot）”的吞吐量（Throughput）**：
   测试脚本跑的是 `claude -p "How does the extension host communicate..."`——**只问一道宏观架构检索题，回答完立刻退出进程结账**。
   - **不用 CodeGraph 的对照组**：Claude Code 需要调 28～43 次 `grep` 和 `Read` 顺藤摸瓜。因为每次工具调用都是一个新的 assistant turn，要把前面的上下文重发一遍（即 241 期算过的“多轮工具调用重发税”），43 轮累加下来的总处理 token（Throughput）高达 670k～991k。
   - **用了 CodeGraph 的实验组**：`codegraph_explore` 一把将十几个相关文件的函数源码和调用图全部打包塞进 1～2 次返回里。因为只有 2 个 turn，没有 40 次重发税，单题总吞吐降到了 155k。

---

## 四、被用户骂出来的致命死穴：多轮上下文窗口直接撑爆 2～3.7 倍

单题跑分看着省钱，但用户真装进 Cursor / Claude Code 干活时，立刻炸锅了（见仓库 **Issue #1500**）：**为什么用了 CodeGraph 之后，聊没几句上下文窗口就爆了？**

原因很简单：
- 原生 Agent 用 `grep` 和小范围 `Read` 摸路时，虽然调用轮数多，但每次读进来的片段很小、很精准；
- 而 `codegraph_explore` 为了追求“一枪命中（1～2 次调用答完）”，每次都把一整盆“全家桶源码 + 调用链 + 爆炸半径”直接扣进上下文窗口里。
- **在单题测试里，答完题进程就退出了，这盆几十 KB 的死肉不用付后续的账；但在真实的多轮写代码会话里，这盆死肉会永远钉死在上下文窗口里！**

在用户质疑下，作者不得不在 `docs/benchmarks/residual-context-occupancy.md`（残留上下文占用率）和 README 小字警告里承认了这个事实：

> *“Across the same seven repos in multi-turn sessions, CodeGraph's responses leave about **80% more retrieval context resident** at the end of a session than a file-reading agent's do — on VS Code, **67k tokens against 18k**.”*

- **VS Code 仓库实测**：原生 `grep + Read` 跑完，留在上下文窗口里的代码只有 **18k tokens**；用了 CodeGraph 跑完，留在上下文窗口里的内容高达 **67k tokens（膨胀了 3.7 倍）**！
- 这意味着：在接下来的第 2 轮、第 3 轮……第 20 轮改代码过程中，**你每一轮说话都要为这多出来的 49,000 个 token 付一遍缓存费，同时还要承受几万字无关重载代码对模型注意力的严重稀释**。

---

## 五、同类样本对照：GitHub 高星榜的“爽文经济学”（AirLLM vs Strata vs CodeGraph）

把今天顺手看的两个高星项目放在一起，能看清当前 GitHub Star 的流量密码：

| 项目 | GitHub Stars | 宣传爽点（标题党） | 物理/工程真相 | 真实可用性 |
|:---|:---|:---|:---|:---|
| **`AirLLM`** (`lyogavin/airllm`) | **~40k ★** | “4GB 显存无损跑 70B，8GB 显存跑 405B，无需量化！” | 针对 **Dense 稠密模型**，把模型按层切片存在 SSD 上，每算一层从 SSD 读进显存、算完扔掉。**每吐 1 个字，SSD 就要完整顺序读取一遍 35GB（70B 4-bit）权重**；回一段 300 字的话要读盘 **10.5 TB**，耗时半小时以上 | **0 实用价值**（约 `0.05~0.2 tok/s`，纯固态硬盘寿命毁灭器；对照 338 期 `Strata` 利用稀疏 MoE 驻留 RAM + SSD 每次只查 5KB 词表跑出 `57.5 tok/s`，差出几百倍） |
| **`CodeGraph`** (`colbymchenry/codegraph`) | **~75k ★** | “Rust 内核语义知识图谱，让 Claude Code 少调 88% 工具、省 62% 成本、快 3.6 倍！” | **`tree-sitter` + `SQLite FTS5` 字面词法检索**，包一层 MCP。靠单题即走（One-shot）避开多轮重发税刷出 62% 省钱海报；真实多轮会话下因为“一勺烩”返回全家桶源码，**残留上下文反倒膨胀 80%～270%（18k→67k）** | **SaaS 引流钩子**（只适合刚拉下陌生大仓库问一句“某流程怎么走”就关窗口；真正在长会话里写代码，不如 Agent 自己跑两行精准 `grep` 干净） |

### 为什么这种东西能刷到 40k / 75k Stars？

1. **精准收割大众的“免思考焦虑”**：
   - 手里只有 8GB 显卡的人不想算 PCIe/SSD 带宽除法，看到“4GB 跑 70B”就点 Star；
   - 每天嫌 Claude Code 账单贵、工具调用慢的 Vibe Coder 不想看源码，看到“一行命令装个 Rust 知识图谱立省 62%”就点 Star。
2. **硅谷增长黑客（Growth Hacking）的标准漏斗**：
   满屏徽章（把市面上所有 Agent 名字全挂上蹭搜索）+ 精心挑选口径的对比柱状图 + X/Reddit 爆款推文引流 → 冲上 GitHub Trending 滚雪球收割几万“赛博松鼠党” → 顶部挂一个 `Join the waitlist (getcodegraph.com)` 拿 75k Stars 的数据去找 VC 融资、卖企业版 PR 审查平台。
3. **底层老黄牛 vs 顶层套壳客的倒挂**：
   真正出力干活的 `tree-sitter`、`sqlite`、`ggml` 吭哧吭哧写底层，星星不及上面包一层壳子、写几句营销话术的项目零头。
