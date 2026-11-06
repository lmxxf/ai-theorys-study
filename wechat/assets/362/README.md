# 第 362 期附件索引：四大主流 Agent（Claude Code / Codex / Google agy / 腾讯 WorkBuddyAI）`/compact` 与 Skill 提炼逆向档案

本目录（`wechat/assets/362/`）归档了公众号第 362 期《扒开四大 Agent 的 /compact 与 Skill 提示词：为什么越压缩越傻，提炼的技能总不触发？》的全套静态逆向报告、完整英文提示词原件、中文对照翻译、提取脚本与正文配图。未来需要复查或重新横向对比这四款产品时，直接按本索引进入。

---

## 一、 核心横向对比与分产品主报告（直接读这四份）

| 文件名 | 覆盖范围与核心内容 |
| :--- | :--- |
| **[四大Agent压缩与Skill提炼横向对比.md](./四大Agent压缩与Skill提炼横向对比.md)** | **【总入口】** 四款产品的横向总表、机制解剖、完整英文 Prompt 原文与中文对照合集 |
| **[claude压缩与skill.md](./claude压缩与skill.md)** | **Anthropic Claude Code (`v2.1.311`)** 专档：8 个 `/compact` 实验臂（`control` 9 节、`lean`、`short`、`capped` 等）、第 6 节防伪 `user:` 补丁、新版 `/init` Phase 6 与官方 `skill-creator` 插件（485 行全文、Pushy 描述词与 6:4 留出集刷触发率机制） |
| **[agy压缩.md](./agy压缩.md)** | **Google Antigravity CLI (`agy v1.5.1`)** 专档：7 节 `/compact` 续接模板（第 4 节穷举报错原文隐患）、内置 `/learn` 命令拼装逻辑（抓 `pivotal change`、分流 `Rule` vs `Skill`、优先改旧项、提案审批流）与 `agy-customizations` |
| **[codex提示词.md](./codex提示词.md)** | **OpenAI Codex CLI (`codex v0.166.0`)** 专档：本地 9 行极简 `/compact` 提示词（426 字节 + `experimental_compact_prompt_file` 覆盖钩子）、远端 `remote_compaction_v2` 加密块机制、`token_budget` 记笔记切窗口机制与内置系统技能 `skill-creator` |

> **补充指针（腾讯 WorkBuddyAI 全量逆向仓）**：
> - 腾讯 `WorkBuddyAI (v5.9.4)` 的完整 20 条逆向解剖报告位于独立仓库 `/home/lmxxf/work/workbuddy-study/README.md`（其中第 19、20 节专门总结 `/compact` 的 `HistoryEngineeringCompactor` 15% 免 LLM 删减与“所有任务均已完成”二压乌龙、以及与 Codex 同源的 `skill-creator` 模板）；
> - 其提取出的原始代码切片与提示词保存在 `/home/lmxxf/work/workbuddy-study/extracted/compact-skill-20261106/`（并在下方 `agent-prompts-20261106/` 中留有完整对照）。

---

## 二、 原始切片、提取脚本与校验清单（`agent-prompts-20261106/`）

需要直接查看未修饰的原始英文 `.txt / .md / .js` 切片、字节偏移与 SHA-256 校验值时，按子目录查找：

### 1. `agent-prompts-20261106/agy/`（Google `agy` v1.5.1，二进制 `~/.local/bin/agy`）
- `manifest.json` / `extract_verify.py`：精确字节偏移、长度、SHA-256 与一键复核脚本
- `learn.txt` / `learn-expanded.txt` / `learn.zh.txt`：内置 `/learn` 完整展开提示词与中译
- `learnPrompt.objdump.txt`：`slashcommands.learnPrompt`（`0x8bf88a0`）反汇编拼装证据
- `compact.txt` / `compact.zh.txt`：7 节 `/compact` 提示词原文（偏移 `0x60d0427`）与中译
- `agy-customizations-SKILL.md` / `skills-guide.md` / `rules-guide.md`：内置自定义、技能与规则指南原件

### 2. `agent-prompts-20261106/claude/`（Anthropic `Claude Code` v2.1.311，二进制 `~/.local/share/claude/versions/2.1.311`）
- `manifest.json` / `extract.py`：精确字节偏移、长度与提取脚本
- `compact-builders.source.js` / `compact-selection.source.js` / `threshold.source.js`：8 臂压缩提示词拼装源码、实验分流函数 `HJ()` 与窗口阈值计算公式源码
- `compact-control.txt` / `compact-lean.txt` / `compact-short.txt` / `compact-short_v2.txt` / `compact-capped.txt` / `compact-no_scratchpad*.txt` / `partial_*.txt`：全部 8 个实验臂及部分压缩变体的展开原文（含 `compact-control.zh.txt` 与 `compact-variants.zh.md` 中译）
- `anti_spoof.txt`：第 6 节拦截助手伪造 `user: ...` / `Human: ...` 的核心片段
- `new-init.txt` / `new-init.source.js`：新版 `/init` Phase 0–8（含 Phase 6 生成 `.claude/skills/`）完整提示词
- `official-skill-creator/`：Anthropic 官方 `skill-creator` 插件完整目录镜像（含 `SKILL.md` 485 行全文、`agents/grader.md`、`comparator.md`、`analyzer.md` 及 `scripts/run_loop.py` 6:4 训练测试集描述词优化脚本）

### 3. `agent-prompts-20261106/codex/`（OpenAI `Codex CLI` v0.166.0，二进制 `~/.codex/packages/standalone/releases/.../bin/codex`）
- `manifest.json` / `extract.py`：精确字节偏移、长度与提取脚本
- `compact-prompt.en.txt` / `compact-handoff.en.txt` / `compact.zh.md`：本地 426 字节压缩提示词与恢复前缀（偏移 `0xfadafe0` & `0xfacc7e3`）
- `token-budget-*.txt`：Token Budget 主动记笔记切窗口机制的三段提示词原件
- `skill-creator/` & `skill-creator.zh.md`：内嵌系统技能 `skill-creator/SKILL.md`（偏移 `0xfdcbc68`）及配套 `init_skill.py`、`quick_validate.py`、`generate_openai_yaml.py`

---

## 三、 第 362 期公众号正文配图（单卡片竖屏 SVG）

- `compact-four-routes.svg`：四家 `/compact` 压缩路线对比（详尽工单派 vs 极简/远端派 vs 腾讯先剪刀后催眠派）
- `claude-fake-user-guard.svg`：Claude `/compact` 第 6 节拦截助手回合伪造 `user: ...` 机制示意
- `learn-vs-skill-creator.svg`：“学经验”两大流派对比（Google `agy /learn` 抓转折点分流 Rule/Skill vs 三家同源 `skill-creator` 1.0/2.0 演进）
