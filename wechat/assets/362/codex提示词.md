# OpenAI Codex CLI (`codex`) 上下文压缩机制与提示词逆向记录

> **逆向时间**：2026-10-30  
> **目标二进制**：`/home/lmxxf/.nvm/versions/node/v24.12.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-arm64/vendor/aarch64-unknown-linux-musl/bin/codex`（Rust 静态链接 ELF 二进制，版本 `0.165.0`）  
> **关联源码模块**：
> - `core/src/compact.rs`（本地提示词压缩与历史重建）
> - `core/src/compact_remote_v2.rs` / `compact_remote_v2_attempt.rs` / `compact_remote_v2_installation.rs` / `compact_remote_history.rs`（OpenAI 远端加密压缩 v2）
> - `core/src/compact_token_budget.rs` / `ext/history-notes/src/extension.rs`（基于 `notes` + `history` + `functions.new_context` 的窗口预算软重置机制）
> - `core/src/session/async_compaction.rs` / `core/src/tasks/compact.rs`

---

## 一、核心结论：为什么 Codex 的压缩比 `agy` 干净得多？

通过逆向 `codex` 二进制并扫描本地 `~/.codex/sessions/` 下的 **182 个会话文件（共 127 次真实压缩记录）**，我们发现 Codex 实际上并存着 **三代（三套）上下文压缩机制**，且它完全避开了 `agy` 那种“7 段式工单 + 穷举复读报错导致提纯毒药”的蠢设计：

1. **第一套（本地/通用 LLM 提示词压缩，`core/src/compact.rs`）极度克制**：
   - 生成摘要的提示词只有短短 **9 行（4 点要求）**，只要求写清“当前进度与关键决策、重要上下文/约束/用户偏好、下一步待办、继续工作所需的关键数据或引用”。
   - **完全没有 `agy` 第 4 节那种“必须穷举所有已解决和未解决的报错原文（specific error messages）”的死命令**，因此不会把偶发乱码或报错死循环提纯置顶。
   - 此外，Codex 支持在 `config.toml` 中通过 **`experimental_compact_prompt_file`** 直接用自定义文件覆盖这套默认压缩提示词！
2. **第二套（OpenAI 官方模型默认走的远端加密压缩，`compact_remote_v2.rs`）不走明文摘要**：
   - 在我们本地全部 **127 次** `gpt-6.1-sol` / `gpt-6-astra` 真实压缩记录中，`payload.message` 全部为空字符串 `""`，取而代之的是一个 `type: "compaction"`、ID 形如 `cmp_...` 的块，内含约 75KB 的 Fernet 加密载荷 **`encrypted_content: "gAAAAAB..."`**（带 `compaction_model_hash: "3000"`）。
   - 同时，它的 `replacement_history` 会**原样保留历次关键的 User 原始消息（含 `AGENTS.md` 全文注入与用户的任务指令）**，而把中间庞大的工具调用输出与助手过程压缩进服务端加密的 `compaction` 块中。这就是为什么闇（Codex）在长程逆向 DLSS 5 时经历多次自动压缩后依然不丢 `AGENTS.md` 身份、也不丢用户指令细节。
3. **第三套（内嵌在 `models_cache.json` 中的 `token_budget` + `notes/history` 软切换机制）**：
   - 当启用 `features.token_budget` 时，系统会在剩余 `6144` tokens 时注入 `<context_window_reminder>`，让模型自己调 `notes` 工具写检查点笔记，再调 `functions.new_context` 开启新窗口，并在新窗口用只读 `history` 工具按需回查旧窗口的 `window_id` + `item_id`。

---

## 二、机制一：本地提示词压缩（`core/src/compact.rs`）完整原文与译文

当使用非远端加密压缩通道（或触发本地 `/compact` 提示词流）时，`codex` 二进制（偏移量 `212815936` 与 `212899659`）硬编码了以下两段成对的提示词：

### 1. 摘要生成提示词（Compaction Prompt）

```text
You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
```

#### 中文对照翻译：
```text
你正在执行一次上下文检查点压缩（CONTEXT CHECKPOINT COMPACTION）。请为即将接手恢复该任务的另一个大语言模型创建一份交接摘要（handoff summary）。

需包含：
- 当前进展与已做出的关键决策
- 重要的上下文、约束条件或用户偏好
- 剩余待办事项（清晰的下一步计划）
- 继续工作所需的任何关键数据、示例或引用参考

请保持简明、结构化，并专注于帮助下一个大语言模型无缝继续这项工作。
```

### 2. 压缩后新窗口的恢复前缀（Handoff Prefix，偏移量 `212804921` & `212899271`）

当模型生成上述摘要后，`codex` 在重建上下文时会在摘要正文中加上这段引导前缀：

```text
Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:
```

#### 中文对照翻译：
```text
另一个大语言模型已经开始解决这个问题，并生成了一份其思考过程的摘要。你还可以访问该语言模型曾使用过的工具状态。请在此已完成工作的基础上继续推进，避免重复劳动。以下是另一个语言模型生成的摘要，请利用其中的信息辅助你自己的分析：
```

### 3. 本地自定义覆盖钩子（`experimental_compact_prompt_file`）

在 `codex` 二进制的 `ConfigToml` 解析结构中（偏移量 `210672980`），预留了一个未公开的配置项：
- **配置键名**：`experimental_compact_prompt_file`
- **作用**：可在 `~/.codex/config.toml` 中指定一个本地 Markdown/文本文件路径，直接替换掉上面那段默认的 `You are performing a CONTEXT CHECKPOINT COMPACTION...` 提示词。

---

## 三、机制二：远端加密压缩 v2（`compact_remote_v2.rs`，GPT-6.1-Sol 默认路径）

在 `~/.codex/sessions/2026/10/29/rollout-...jsonl` 等真实会话日志中，当上下文接近模型窗口阈值（`gpt-6.1-sol` 在一次真实压缩时的 `input_tokens` 为 **`789,167`**，接近 `max_context_window: 872,000`）时，Codex 触发 `remote_compaction_v2`，写入一条 `type: "compacted"` 记录，其 `replacement_history` 结构如下：

```json
{
  "type": "compacted",
  "payload": {
    "message": "",
    "window_number": 2,
    "first_window_id": "01a182ba-a909-7e92-b6e2-8c4591ae10e7",
    "previous_window_id": "01a184b4-225f-7111-bfc1-073a10d4ba5b",
    "window_id": "01a186a2-b7dd-7290-a351-750474c032fb",
    "compaction_response_id": "resp_0b73bb693935a6f9016ae2f9ba49608191a730de814a15067d",
    "replacement_history": [
      { "type": "additional_tools", "role": "developer", "tools": ["..."] },
      { "type": "message", "role": "developer", "content": "You are Codex, an agent based on GPT-6..." },
      { "type": "message", "role": "user", "content": "<保留的历次关键 User 消息原文 1..N>" },
      { "type": "message", "role": "developer", "content": "<skills_instructions> / <multi_agent_role> / <multi_agent_mode>" },
      { "type": "message", "role": "user", "content": "# AGENTS.md instructions for /home/lmxxf/work/...\n<完整的 AGENTS.md 原文>" },
      { "type": "message", "role": "user", "content": "<最新一轮 User 指令原文>" },
      {
        "type": "compaction",
        "id": "cmp_0b73bb693935a6f9016ae2f9c0e93c8191b3d7031f9f9e1e0c",
        "encrypted_content": "gAAAAABq4vqNruToRgXUKmWtk-Y0R3RAvnJd5RaazrJ_OWwkbqcK54V79206ytDZ...（约 75 KB 服务端加密状态）"
      }
    ]
  }
}
```

**关键设计亮点**：
1. **`AGENTS.md` 与历次 User 核心指令全量免损保留**：在 `replacement_history` 的第 `2..19` 项中，用户发出的任务单和完整的 `AGENTS.md` 被原封不动保留在明文历史里，只把中间几十万 token 的命令行输出和过程推理折叠进了末尾的 `type: "compaction"`（`encrypted_content`）中。
2. **服务端同构状态压缩（`comp_hash: "3000"`）**：在模型目录配置中，每个 GPT-5.6 / GPT-6 / GPT-6.1 模型都带有 `"comp_hash": "3000"`，服务端直接对生成历史做约 `6,732` output tokens 的加密高密度状态摘要（Fernet token `gAAAAAB...`），下次请求时原样传回 OpenAI 后端解密恢复。

---

## 四、机制三：Token Budget 主动记笔记切窗口提示词（`compact_token_budget.rs`）

在二进制内嵌的模型配置表（偏移量 `213977610`）以及 `~/.codex/models_cache.json` 的 `model_messages.token_budget` 字段中，定义了第三套由模型自主调用 `notes` + `functions.new_context` 切换窗口的提示词模板（当前默认 `"enabled": false`）：

### 1. 系统常驻引导提示词（`guidance_message`）

```text
For tasks that may span context windows, use `notes` to maintain a concise checkpoint of the goal, decisions, progress, learnings and next steps. Include the window ID and item ID for every relevant user request you are currently solving as well as important actions/tool calls. You can use `history` tool to look up details with the references later. Note that every non-assistant item, such as user, developer, tool response, has an item id `[id: ...]` that is immediately after its item content. Relative note paths belong to the current thread; absolute paths may read other threads' notes, but writes are limited to the current thread.

It is a good idea to take incremental notes while you work so that you do not miss any important info. You can also use `get_context_remaining` tool to find the remaining token budget for better planning. Once the token budget is exhausted, you will lose access to the current window and continue in a fresh context window and you can only recover through `notes` and `history` tools. So be careful not to over-run the context window without any documentation.

If Previous context window id is present in `<context_window>`, it means a context reset occurred and this is a new window. After a reset, read the checkpoint and use the read-only `history` tool to recover any missing details. When a window ID and item ID are known, prefer `read_item` directly; when they are missing or uncertain, use `list_items`, or `search_contents` to locate the item first.

Treat notes and history as internal bookkeeping. Do not mention them in user-facing messages.
```

### 2. 窗口即将耗尽提醒模板（`reminder_message_template`，剩余 `6144` tokens 触发）

```xml
<context_window_reminder>
Your current context window is nearly exhausted; only {n_remaining} tokens remain. Before starting a new context window, save concise progress notes with the `notes` tool with the goal, decisions, progress, learnings, next steps, and the window ID and item ID of every relevant user request still being solved, as well as important actions/tool calls for future reference. Note that every non-assistant item, such as user, developer, tool response, has an item id `[id: ...]` that is immediately after its item content. You should write or append notes in a way to best help you recover in a new context window. It is also a good idea to clean up your old notes if they become obsolete or irrelevant. Future context windows will not automatically include the current conversation. After saving your state, call `functions.new_context` to continue in a fresh context window.
</context_window_reminder>
```

### 3. 窗口彻底耗尽强制兜底提示词（`auto_compact_fallback_prompt`，预留 `16384` buffer tokens）

```xml
<context_window_reminder>
The current context window is exhausted. Do not continue the task or give a final answer in this window. The next window will not automatically include this conversation. Make exactly one write or append call to `notes` now to save a concise checkpoint with the goal, decisions, progress, learnings, next steps, and the window ID and item ID of every relevant user request still being solved, as well as important actions/tool calls for future reference. Note that every non-assistant item, such as user, developer, tool response, has an item id `[id: ...]` that is immediately after its item content. After the notes result returns, call `functions.new_context`; do not use any tools other than `notes` and `functions.new_context`.
</context_window_reminder>
```

---

## 五、Codex 内嵌模型窗口与压缩参数一览表（来自 `0.165.0` 二进制与 `models_cache.json`）

| 模型代号 (`slug`) | 显示名称 (`display_name`) | 默认推理档位 (`default_reasoning_level`) | 常规上下文窗口 (`context_window`) | 最大上下文窗口 (`max_context_window`) | 远端压缩哈希 (`comp_hash`) | 子代理推理档位 (`multi_agent_reasoning_effort`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gpt-6.1-sol` | `GPT-6.1-Sol` | `low` | `272,000` | **`872,000`** | `"3000"` | `xhigh` |
| `gpt-6.1-terra` | `GPT-6.1-Terra` | `medium` | `500,000` | **`872,000`** | `"3000"` | `xhigh` |
| `gpt-6-astra` | `GPT-6-Astra` | `medium` | `272,000` | **`872,000`** | `"3000"` | `xhigh` |
| `gpt-6-sol` | `GPT-6-Sol` | `low` | `272,000` | **`872,000`** | `"3000"` | `xhigh` |
| `gpt-6-luna` | `GPT-6-Luna` | `low` | `272,000` | **`872,000`** | `"3000"` | `high` |
| `gpt-5.6-sol` / `terra` / `luna` | `GPT-5.6-*` | `low` / `medium` | `272,000` | **`872,000`** | `"3000"` | `high` / `xhigh` |
| `gpt-daybreak-red-latest` | `GPT-Daybreak-Red` | `medium` | `372,000` | `372,000` | `"3000"` | `high` |
| `gpt-5.5` | `GPT-5.5` | `medium` | `272,000` | `272,000` | `"3000"` | `high` |

---

## 六、`agy` vs `codex` 压缩设计横向对比

| 维度 | Antigravity CLI (`agy`) | OpenAI Codex CLI (`codex`) |
| :--- | :--- | :--- |
| **本地摘要提示词风格** | 7 段式死板工单模板（强制 `Errors & Fixes` 穷举所有具体报错原文） | 4 点极简交接说明（只记进度/决策、上下文/偏好、下一步、关键引用） |
| **对生成乱码/报错的免疫力** | **极差（毒药提纯器）**：把乱码/报错当成 `specific error messages` 摘抄置顶，压缩后直接引爆自回归死循环 | **高**：不要求穷举报错原文；且官方模型默认走服务端 `remote_compaction_v2` |
| **身份与用户指令保留** | 只保留最近 10 条 User Request（`max_user_requests: 10`） | `replacement_history` 原样保留完整 `AGENTS.md` 与历次关键 User 指令，过程折叠为服务端加密 `compaction` 块 |
| **是否支持自定义压缩提示词** | 二进制内虽有 `session_summary_prompt_override` protobuf 字段，但未向用户配置文件暴露 | 原生支持在 `~/.codex/config.toml` 配置 `experimental_compact_prompt_file` |


## 七、2026-11-06补充：当前0.166.0 x86_64与完整Skill Creator

本节是当前现场核验，前文0.165.0 ARM64与旧会话统计作为历史记录保留，**不可混用构建偏移，也不可直接当今日默认配置**。

- 当前版本为 `codex-cli 0.166.0`，二进制 `/home/lmxxf/.codex/packages/standalone/releases/0.166.0-x86_64-unknown-linux-musl/bin/codex`；319,346,888字节，SHA-256 `b8a11fedf24b2dfb59a70c9e52b030f9a508dcd49cd2dd0d41956f4c785156b8`。
- 本地摘要提示词仍为同一份426字节原文，现偏移263040992（`0xfadafe0`）；恢复前缀偏移262981603（`0xfacc7e3`）。
- **Skill Creator完整英文、逐节中文、所有捆绑脚本与引用的精确偏移**见[本轮完整档案](agent-prompts-20261106/codex/findings.md)，[SKILL.md原件](agent-prompts-20261106/codex/skill-creator/SKILL.md)，[完整中译](agent-prompts-20261106/codex/skill-creator.zh.md)，[机器可读manifest](agent-prompts-20261106/codex/manifest.json)。SKILL.md为15,311字节，二进制266124392（`0xfdcbc68`）处与安装文件逐字节一致，属于真正内置系统技能，不是固定 `/learn` 命令。当前命中字面 `/learn` 的均是官方文档URL，未发现核心 `/learn` 命令声明。
- 当前creator没有固定“把本次对话转换成技能”的专门prompt；它是通用创建／更新指南，要求保留用户范围、只留有用的非显然经验、避免把单次失败升级成普遍要求，按需附scripts／references／assets，验证格式并在确有必要时验证行为。
- **更正前文证据强度：** 远端v2内部提示词并未提取到；不能把本地426字节prompt当作远端实际prompt。`gAAAAAB...`前缀不足以确认一定使用Fernet、内部一定是纯自然语言摘要或“同构状态”恢复。旧日志保留的用户和AGENTS消息是样本事实，不是所有会话全部用户消息无损保留的保证。未要求穷举报错也不意味着免疫乱码／复读。
- 当前gpt-6.1-sol缓存context_window为500000、max_context_window为872000、effective_context_window_percent为95，不能继续把旧表272000当当前值；notes/history token-budget路径当前缓存仍为enabled=false、use_history_notes_extension=false，6144提醒和16384兜底buffer只是配置存在，不是开启证明。
