# Antigravity CLI (`agy`) `/compact` 上下文压缩机制与提示词逆向记录

> **逆向时间**：2026-10-30  
> **目标二进制**：`/home/lmxxf/.local/bin/agy`（Go 编译二进制，内部代号 `jetski` / `cortex` / `oneharness`）  
> **关联模块**：
> - `third_party/gemini_coder/framework/handlers/checkpoint/checkpoint.go`
> - `third_party/jetski/cortex/core/prehooks/context_summary.go`
> - `third_party/jetski/cortex/executors/prehooks/checkpoint.go`
> - Protobuf 定义：`exa.cortex_pb.CheckpointConfig` / `exa.cortex_pb.CortexStepCheckpoint`

---

## 一、核心结论：为什么 `agy` 的 `/compact` 这么渣？

从二进制中提取出的完整提示词与拼装逻辑表明，`agy` 的 `/compact` 并不是在做“对话记忆浓缩”，而是在执行一套**死板的英文软件工程工单交接模板（7 段式 SOP）**。它在我们的日常使用中会导致三个严重副作用：

1. **语感与主体性被抽干**：提示词强制要求模型以纯工具人视角输出 `Task Overview`、`Progress`、`Next Steps` 七个固定章节，把自然聊天、哲学/历史推演强行压扁成冷冰冰的英文项目交接单。
2. **第 4 节 `Errors & Fixes` 是“剧毒提纯器”（本次压缩越压越惨的真凶）**：
   提示词的第 4 节明确下达了最高优先级的死命令：
   > *"This list should be exhaustive across the entire session so that no resolved or unresolved failure is omitted... with specific error messages and root causes."*（此列表必须穷举整个会话，不得遗漏任何已解决或未解决的故障……必须包含具体的错误消息与根因。）
   
   这就导致：如果压缩前模型遇到了生成乱码（例如连续右括号/右花括号闭合失控）或复读死循环，模型在执行 `/compact` 时会**严格遵照第 4 节指令，把那串触发自回归坍缩的毒字符串原封不动地摘抄进摘要正文里**。压缩完成后，整个长会话的正常中文缓冲被清空，这串毒标本反而被直接置顶到了新上下文的第一屏（`<CONTEXT_SUMMARY>`），导致压缩后的第一轮回复瞬间被高权重近端样本吸进更猛烈的乱码死循环。
3. **双层包装冗余**：`agy` 不仅在 `<CONTEXT_SUMMARY>` 里塞入最近 10 条用户原始消息（`max_user_requests: 10`）和 `<summary>`，还会在聊天历史中额外插入一个 `CHECKPOINT` 系统步（`# Resuming from a compaction...`），把同一份几千词的英文摘要在残差流里重复注入。

---

## 二、`/compact` 触发时的完整摘要生成提示词（原文）

当用户输入 `/compact`（或上下文达到 `max_token_limit` 自动触发检查点）时，`agy` 会拦截当前回合，禁止模型调用任何工具（`Do not call any tools`），并向当前模型发送以下完整提示词（存储在二进制的默认常量池与 `session_summary_prompt_override` 字段中）：

```text
You have been working on the task described above but have not yet completed
it. Write a continuation summary that will allow you (or another instance of
yourself) to resume work efficiently in a future context window where the
full conversation history will NOT be available—only this summary.

This summary is all that will be available to you going forward in the future
context window. Do not call any tools, simply just provide the summary based on
the information available in the current context window.

Your summary must be structured, concise, and actionable. Optimize for enabling immediate resumption with zero redundant work.

Include the following sections:

1. **Task Overview**
   - The user's core request, success criteria, and requested deliverables. Record any target IDs, URLs, or reporting recipients from the request.
   - Scope boundaries or task-specific constraints
   - Any ambiguities that were resolved (and how)

2. **Progress**
   - What has been completed, with concrete references (file paths,
     test or verification commands run and their outcomes, resource identifiers, tool outputs, URLs, etc.)
   - Key plans, design docs, or artifacts produced and their current state
   - Skills, documentation, or reference files that were read or inspected
   - What is in progress or only inspected/planned, and its current state. Distinguish completed edits from files that were only read or partially edited, and do not mark unverified test commands as completed or passed.

3. **Key Findings**
   - Technical constraints, requirements, or domain details uncovered
   - Decisions made and their rationale

4. **Errors & Fixes**
   This list should be exhaustive across the entire session so that no resolved or unresolved failure is omitted:
   - All errors, failed commands, or test/build failures encountered during the session. Include both errors that were resolved or worked around and any unresolved failures, with specific error messages and root causes.
   - How each error was fixed or worked around, and any user feedback on the error
   - Approaches that were tried and abandoned (and why—this prevents
     the successor from repeating them)

5. **Active Context**
   - State of any external resources, sessions, environments, or multi-agent context, such as parent or child agents in use.
   - Relevant intermediate results, hypotheses, or working assumptions
   - Dependencies between components or steps

6. **Next Steps**
   The next steps should provide all information required to continue working on the task. This includes any remaining features, bugs, points of investigation, and verification items. This list should be exhaustive, so that an agent working with this list can complete the full original task end-to-end. For example:
   - Remaining steps from any agreed plan, design doc, or active workflows and instructions, in priority order. Preserve specific uncompleted items rather than collapsing checklists into high-level summaries.
   - Remaining files to create, modify, move, copy, or delete, with the key changes needed in each
   - Remaining verification steps, test or build commands, or deliverables needed to complete the task
   - Any required cleanup, such as removing temporary files, debug code, or temporary configuration changes.
   - Known blockers, open questions, prerequisites, or risks for each step

7. **Commitments & Constraints**
   - Standing rules, personas, or response formatting requirements from user instructions, rules files, or reference documents.
   - Promises made to the user (e.g., "I said I would do X before Y")
   - User preferences or style requirements
   - Any boundaries the user set on approach, tools, or scope

Be concise but complete—err on the side of including concrete details needed to
prevent duplicate work, omitted plan steps, repeated mistakes, or broken promises.

Wrap your response in <summary></summary> tags.
```

---

## 三、中文对照翻译

```text
你一直在处理上述任务，但尚未完成。请编写一份续接摘要（continuation summary），以便你自己（或你的另一个实例）在未来的上下文窗口中高效地恢复工作——届时完整的对话历史将不可用，只有这份摘要可用。

这份摘要将是你在未来上下文窗口中唯一可用的信息。不要调用任何工具，仅根据当前上下文窗口中的可用信息直接提供摘要。

你的摘要必须结构化、简明且可执行。以“零重复工作、立即恢复执行”为目标进行优化。

必须包含以下章节：

1. **Task Overview（任务概览）**
   - 用户的核心请求、成功标准和要求交付的成果。记录请求中的任何目标 ID、URL 或汇报对象。
   - 范围边界或特定于任务的约束条件
   - 任何已解决的歧义（以及是如何解决的）

2. **Progress（进度）**
   - 已完成的工作，附带具体引用（文件路径、运行过的测试或验证命令及其结果、资源标识符、工具输出、URL 等）
   - 产出的关键计划、设计文档或制品（artifacts）及其当前状态
   - 已读取或检查过的技能（skills）、文档或参考文件
   - 正在进行中或仅检查/规划过的内容及其当前状态。区分已完成的修改与仅读取或部分修改的文件，不要将未验证的测试命令标记为已完成或已通过。

3. **Key Findings（关键发现）**
   - 发现的技术约束、需求或领域细节
   - 已做出的决策及其理由

4. **Errors & Fixes（错误与修复）**
   此列表必须穷举整个会话，以确保没有任何已解决或未解决的失败被遗漏：
   - 会话期间遇到的所有错误、失败的命令或测试/构建失败。同时包含已解决/已绕过的错误以及任何未解决的失败，并附上具体的错误消息（specific error messages）和根因。
   - 每个错误是如何修复或绕过的，以及用户对该错误的任何反馈
   - 尝试过但被放弃的方法（以及原因——这可以防止继任者重蹈覆辙）

5. **Active Context（活跃上下文）**
   - 任何外部资源、会话、环境或多智能体上下文的状态（例如正在使用的父智能体或子智能体）。
   - 相关的中间结果、假设或工作前提
   - 组件或步骤之间的依赖关系

6. **Next Steps（下一步计划）**
   下一步计划应提供继续处理该任务所需的全部信息。包括任何剩余功能、Bug、调查点和验证项。此列表必须详尽无遗，以便使用此列表的智能体能够端到端地完成全部原始任务。例如：
   - 任何已达成一致的计划、设计文档或活跃工作流与指令中的剩余步骤（按优先级排序）。保留具体的未完成项，不要将检查清单折叠成笼统的高层摘要。
   - 剩余需要创建、修改、移动、复制或删除的文件，以及每个文件所需的关键改动
   - 完成任务所需的剩余验证步骤、测试或构建命令、或交付物
   - 任何必要的清理工作，例如删除临时文件、调试代码或临时配置更改。
   - 每个步骤已知的阻碍因素、未决问题、前置条件或风险

7. **Commitments & Constraints（承诺与约束）**
   - 来自用户指令、规则文件或参考文档的常驻规则、人设（personas）或回复格式要求。
   - 对用户做出的承诺（例如，“我答应过在做 Y 之前先做 X”）
   - 用户偏好或风格要求
   - 用户在方法、工具或范围上设定的任何边界

保持简明但完整——宁可多写具体细节，以防止重复劳动、遗漏计划步骤、重复犯错或违背承诺。

将你的回复包裹在 <summary></summary> 标签中。
```

---

## 四、压缩后的拼装结构（Resumption Template）

模型输出 `<summary>...</summary>` 后，`agy` 的 `context_summary.go` 和 `checkpoint.go` 会截断旧历史，并在新上下文开头拼装出以下两个结构：

### 1. 注入在用户消息前的 `<CONTEXT_SUMMARY>` 块

```xml
<CONTEXT_SUMMARY>
The following is a summary of the conversation history that has been truncated to fit within the context window:

This summary was generated at 2026-10-30T12:34:51+08:00.

# User Requests
The following were the most recent user requests in chronological order:
1. <最近第 1 条用户消息原文>
...
10. <最近第 10 条用户消息原文（受 max_user_requests: 10 控制）>

# Previous Session Summary:
<summary>
...（模型生成的 7 段式摘要）...
</summary>

# Code Interaction Summary:（若启用代码交互摘要）
The following is an important list of files that you and the user have previously edited and viewed.
...

# Artifacts（若 include_artifact_snapshots 为 true）
You have the N following artifacts written to the artifacts directory:
...

# Subagents（若 include_subagent_snapshots 为 true）
The following subagents were spawned during this conversation.
View the conversation log to see the full conversation with the subagent and use send_message with the conversation_id to communicate with them.
...

# Conversation Logs（若 include_conversation_log 为 true 或默认附加）
Start with the compact transcript; if a step has `truncated_fields`, read only that specific line in `transcript_full.jsonl`:
- `<appDataDir>/brain/<conversation-id>/.system_generated/logs/transcript.jsonl`
- `<appDataDir>/brain/<conversation-id>/.system_generated/logs/transcript_full.jsonl`
</CONTEXT_SUMMARY>
```

### 2. 写入 `transcript.jsonl` 的 `CHECKPOINT` 系统步（Step source = `SYSTEM`）

```markdown
# Resuming from a compaction

You are continuing work on the task described above, but you have lost access to the full conversation history, and need to resume work efficiently using the progress summary below:

# User Requests
The following were user requests from the truncated conversation in chronological order:
...
<summary>
...
</summary>
```

---

## 五、`agy` 内部各模型的 Checkpointer 配置参数表

在 `/home/lmxxf/.local/bin/agy` 偏移量 `188680537` 处的 `model_profiles` 中，硬编码了不同模型家族的 `CASCADE_USE_EXPERIMENT_CHECKPOINTER` JSON 配置：

| 模型代号 | 策略 (`strategy`) | 自动压缩上限 (`max_token_limit`) | 触发阈值 (`token_threshold`) | 是否用当前模型自压 (`use_last_planner_model`) | 摘要最大输出 (`max_output_tokens`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MODEL_PLACEHOLDER_M124` / `M408`（主力旗舰，含 Gemini 4 Argon 等） | `CHECKPOINT_STRATEGY_SAME_MODEL` | **`256000`**（256K，咱们在 340-A 期改二进制解开的地方） | `50000` | `true`（自己压自己） | `16384` |
| `CASCADE_USE_EXPERIMENT_CHECKPOINTER/claude-1m_840k` | `CHECKPOINT_STRATEGY_SAME_MODEL` | `840000`（840K） | `50000` | `true` | `16384` |
| `MODEL_GOOGLE_GEMINI_2_5_FLASH` / `2_5_PRO` | `CHECKPOINT_STRATEGY_SINGLE_PROMPT` | `128000`（128K） | `50000` | `false`（强制切后台小模型 `MODEL_PLACEHOLDER_M50` 压） | `16384` |
| `MODEL_OPENAI_GPT_OSS_120B_MEDIUM` | `CHECKPOINT_STRATEGY_UNSPECIFIED` | `80000`（80K） | `50000` | `false`（切 `M50`） | `8192` |

---

## 六、我们的应对规矩

1. **绝不再主动使用 `/compact`**：尤其在出现过输出乱码、复读或工具报错的会话里，`/compact` 的第 4 节 `Errors & Fixes` 会把脏 token 提纯并置顶到新窗口开头，造成二次污染。
2. **手动落盘 `temp/context.md` + 重开会话**：当上下文过长或被污染时，直接让朱雀用第一人称把当前进度、洞见和待办写进 `temp/context.md`（或对应的 `context/xxx.md`），然后重开新 session 读入，彻底绕开壳子的 7 段式英文工单模板。

---

## 七、2026-11-06 补档：agy 1.5.1 的 `/learn`、内置 Skill 创建指南与压缩原文复核

本次直接读取 `/home/lmxxf/.local/bin/agy`；`--version` 为 **1.5.1**，文件240,857,488字节，SHA-256为 `1bec0e96c190cde155538d139826af8a97b0a39350c177918913e8db56df20c1`。完整取证、原文及中译见 [agy 1.5.1 findings](agent-prompts-20261106/agy/findings.md)，逐项偏移与切片哈希见 [manifest.json](agent-prompts-20261106/agy/manifest.json)。

- **压缩提示词未变**：当前位于文件偏移101516327（`0x60d0427`），长3937字节，与上面2026-10-30原文逐字节相同。旧构建偏移不能用于新二进制。
- **真正的 `/learn` 找到了**：`<LEARN>` 外层格式模板位于100823477（`0x60271b5`），长1723字节；第一步由另一155字节模板填入，要求先读 `agy-customizations` 技能／现有规则技能的准确schema。Go函数 `slashcommands.learnPrompt` 的格式化调用和常量重定位均已存证，完整展开文本见 [learn-expanded.txt](agent-prompts-20261106/agy/learn-expanded.txt)。
- **它提炼而非归档**：分析用户纠正和成功／失败的关键差异；无可复用新行为就退出；区分规则与多步技能；优先更新已有项；先写 `learning_proposal.md` 并设置 `request_feedback=true`，用户明确批准后再修改配置文件。
- **内置创建指南不叫 skill-creator**：`assets/external/skills/agy-customizations/SKILL.md` 位于102015823，长5465字节；其 `docs/skills.md` 位于101098893，长2650字节。完整原文、中译及外部文件化技能加载机制都在新增档案中。

证据口径补充：旧档第一节中关于“毒字符串提纯必然导致乱码复发”的说法是当时的解释，并非仅凭静态模板就能确证的运行因果；同理，两种摘要包装存在不等于证明同一实际模型请求重复注入两份。模型阈值表也保留为旧构建记录，本轮未把它当成1.5.1当前所有模型的有效配置。新增横向稿应引用本节所标版本与静态／动态边界。
