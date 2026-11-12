# agy 1.5.7 上下文拼装与三层数据形态核实

核实时间：2026-11-12 深夜至 11-13 凌晨，Asia/Taipei。靶标为本机正在使用的程序及指定活跃会话。本文是逆向报告，供 371 期取材；**确定事实、静态可达能力、尚未观测到的行为分别表述**。

## 0. 先给结论

1. **SQLite 存的是带字段和类型的 Protobuf 消息；发往当前 Google 接口的是 Protobuf 对象经过 `protojson` 序列化的 HTTP JSON，请求流式响应走 SSE。** 不是把整个聊天窗口先拼成一大段纯文本再发出去，也不是因为存在 gRPC stub 就能判定当前走 gRPC。
2. **XML 包装有相当一部分在客户端已经完成。** 例如用户请求、时间、设置变化、系统提醒、压缩摘要；但工具声明、工具调用、工具结果、思考标志与签名在请求中还有独立结构。
3. 本构建未检出 `SPECIAL INSTRUCTION: think silently`、`EFFORT LEVEL:`、`declaration:default_api:` 的字面字符串。结合结构化请求构建链，**支持这些具体表面形式在客户端之外渲染的解释**；不能仅凭负搜索就“100% 实锤”是哪台云端服务、哪段代码生成，更不能拿模型口述代替抓包。
4. **不是所有会话固定 11 个系统区块。** 本会话配置的有序候选为 14 项，其中沙箱、MCP、子代理提醒等可按条件不输出。另有更多已注册但本会话没选的区块。
5. **当前 `executor_config.use_core_direct=1`。** 应以 OneHarness `DefaultSerializer` 为当前主路径；旧 `TrajectoryChatConverter` 仍在二进制里，不能看到旧函数就认作现场执行路径。两条路径共享若干文本模板。
6. **44,133 的服务端 input 数确认了，但四项精确分账无法从这份持久化记录恢复。** 本库没有保存该次完整 system prompt、message prompts、工具 schema 和分项 token breakdown。首轮本地估算是 53,019，不能把它当成 44,133 的分项账本。

可复核附件：

- [实盘字段与协议 schema 摘录](agy-context-evidence.json)：版本/hash、表结构、首两次生成统计、实际配置、原始首条用户文本与转换后文本、相关 Protobuf 字段。
- [静态调用与字符串交叉引用](agy-context-static-evidence.txt)：当前构建函数地址、反汇编调用目标、常量引用；指令中的常量后可能紧接其他 Go 字符串，**只取经长度参数或明确模板边界确认的部分**。

## 1. 取证边界与对象

| 项目 | 本次观测 |
|---|---|
| 程序 | `/home/lmxxf/.local/bin/agy` |
| `--version` | `1.5.7` |
| 文件大小 | 280,564,136 bytes |
| SHA-256 | `3cac18bd597d4c6e75b27556ec4ec4cf285a2ca7d63884f7a68cbcf41acc6ddf` |
| 架构 | ELF x86-64，Go 函数表和原始 proto descriptor 可提取 |
| 会话 | `87600377-5184-4a58-9b89-cd51f6c29dde` |
| 实盘模型 | `plan_model=1438`；usage 同号；`response_model=gemini-4-carbon`；display `Gemini 4 Carbon (Low)` |
| 当前窗口配置 | `max_context_tokens=786432`；checkpoint `max_token_limit=786432` |

该 SHA 与项目 `wechat/assets/340-A/x86-linux/agy-x86-patches.json` 的 **patched_sha256** 一致：审的是本地已打补丁的 1.5.7，不能把 786,432 窗口写成原版默认值。manifest 与跳板源码还说明规则/定制预算路径被改到 131,072；因此下文持久化的 20,000 配置值尤其不能直接当作最终执行预算。这里只读核对既有补丁，没有应用或重打补丁。

所有 SQLite 连接均使用 `sqlite3.connect('file:…?mode=ro', uri=True)`，随后 `PRAGMA query_only=ON`；需要一致读取时开启只读事务。**没有执行 DB 写入、checkpoint、VACUUM，没有复制覆盖活跃库，也没有修改程序。** 活跃进程仍可自行追加记录，因此全库文件 hash 变化不是审计写入的证据；本报告固定引用首两次生成和早期 step。

方法：解析 `.gopclntab` 定位函数，`objdump` 查看调用和常量交叉引用；从文件描述符取消息字段号，按 wire format 解包数据库。注解依赖不全时不强行假造完整 descriptor pool，实盘字段使用原始 descriptor 的字段号/类型递归解析。未识别字段保留原值，不靠猜名称填空。首条 step 的公共头和 payload 分支另按字段号解包，并与 transcript 双向核对。

**没有 TLS 抓包、运行期内存 dump、Google 后端源码或最终 tokenizer 输入。** 下文“请求结构”由实际接口日志、请求构建函数与协议 schema 共同支持，不等于拿到了该次完整 HTTP body。

## 2. 三层数据形态

### 2.1 本地状态：数据库不是聊天纯文本

现场表结构如下（附件记录时为 33 steps / 13 generations / 3 executors，后续可增长）：

| 表 | 作用 / 数据形态 |
|---|---|
| `trajectory_meta` | trajectory/cascade ID、类型和 source 等关系字段 |
| `steps` | `idx, step_type, status, has_subtrajectory` + `metadata, error_details, permissions, task_details, render_info, step_payload` 等 blob，以及 `step_format` |
| `gen_metadata` | 每次生成的 `idx, data, size`；data 可按 `CortexStepGeneratorMetadata` 解包 |
| `executor_metadata` | 执行批次的 data；可按 `ExecutorMetadata` 解包，含完整配置与 loaded customizations |
| `parent_references` | 父引用 blob |
| `trajectory_metadata_blob` | 会话元信息 blob |
| `battle_mode_infos` | battle 模式信息 blob |

初次读取时有 26 个 steps、11 个 gen_metadata、2 个 executor_metadata。`step_format=0` 的现场 step_payload 是 Protobuf wire 数据，**不是 JSON，也不是压缩后的聊天 Markdown**。例如 step 0：field 1=14、field 4=3，field 5 是公共 metadata，field 19 是 `CortexStepUserInput`；step 1 的 field 20 是 `CortexStepPlannerResponse`。

首条用户 payload 保存的是裸请求（130 UTF-8 bytes）以及 `items.text`、`active_user_state`、`user_config` 等结构，**没有把 `<USER_REQUEST>` 原封不动存进 `user_response`**。而 transcript 首行已经有 XML 和时间/设置变化，证明包装发生在客户端的投影/序列化阶段。

首个 planner response 的工具调用独立保存：

```text
name = read_file
arguments_json = {"Path":"…/context.md"}
id = call_12741555034289484863
thinking_signature = 138 bytes
```

签名是二进制字段，不是把一行“签名文字”写入正文。该步 usage 还记录 thinking output 2 tokens；**这不等于 DB 保存了那两个 token 的可读思考原文**。

### 2.2 Transcript 是投影，不能当作实际请求全文

相关函数（包名前缀均为 `google3/third_party/jetski/`）：

- `cortex/artifacts/artifacts.WriteTranscriptChunks`：`0x79eade0`。
- `…ReconstructTranscripts`：`0x79eb160`。
- `…reconstructTranscript`：`0x79eb300`。

现场 `transcript.jsonl` 与 `transcript_full.jsonl` 是逐步 JSON 行，含 source/type/status/content/tool_calls/token usage 等用于回看和恢复的内容。配置明确有 compact transcript 字节裁剪：`max_bytes_per_step=4096`、`max_bytes_per_tool_arg=2048`；完整日志是另一份投影。

**“full”指这套日志的完整版本，不保证包括完整系统提示词、全部工具 schema、所有隐藏字段或最终模型输入。** 首轮 system prompt 不在 transcript 开头；不能据此说它没发，也不能把 transcript 拼起来就宣称还原了模型 prompt。

### 2.3 当前网络路径：HTTP JSON + SSE

当前 cli.log 中可读到：

```text
https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse
```

二进制调用链：

| 函数（`google3/third_party/jetski/language_server/code_assist_client/codeassistclient`） | 地址 | 硬证据 |
|---|---:|---|
| `BuildGenerateContentRequest` | `0x9490f00` | 构造结构化 contents、tool calls/results、tools/config 等 |
| `marshalCCPARequest` | `0x9492f60` | 调 `protojson.MarshalOptions.marshal`（`0x949301a`） |
| `(*CodeAssistClient).streamGenerateContentFromCCPA` | `0x9493040` | 先 marshal；设置 `alt=sse`、方法名 `streamGenerateContent`；调用 `net/http.NewRequestWithContext`，method `POST` |
| `(*ccpaConverter).ToRequestPayload` | `0x949fd00` | 转入同一个 `marshalCCPARequest` |

因此：**客户端内部 Request 类型来自 Protobuf，跨网卡时这里是 JSON 编码。** 二进制也包含 Vertex、BusinessAI、AICode、其他提供商及 gRPC 服务桩；它们的存在不能改变本次接口实证。

外层 descriptor：`google.internal.cloud.code.v1internal.GenerateContentRequest`：

| 字段号 | 字段 | 形态 |
|---:|---|---|
| 1 | `project` | string |
| 2 | `request_id` | string |
| 3 | `request` | `google.cloud.aiplatform.master.GenerateContentRequest` |
| 4 | `model` | string |
| 5–8 | `user_prompt_id, user_agent, request_type, enabled_credit_types` | 标识、请求类型、额度枚举等 |

内层关键字段：`contents` #2；`generation_config` #4；`model` #5；`tools` #6；`tool_config` #7；**`system_instruction` #8**；`cached_content` #9；`session_id` #13；`implicit_cache_config` #17；以及 labels、safety 等可选字段。**schema 有某字段不表示本次每个字段都非空。**

结构可画成下面这样（语义示意，非截获的原包，省略可选字段）：

```json
{
  "model": "…",
  "request": {
    "systemInstruction": {"parts": [{"text": "<identity>…</identity>…"}]},
    "tools": [{"functionDeclarations": [{"name": "read_file", "parameters": {}}]}],
    "contents": [
      {"role": "user", "parts": [{"text": "<USER_REQUEST>…</USER_REQUEST>…"}]},
      {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {}}, "thoughtSignature": "…"}]},
      {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {}}}]}
    ],
    "generationConfig": {"thinkingConfig": {}}
  }
}
```

`Content` 独立存 `role` 与 `parts`。`Part` 的关键字段号是 text #1、function_call #5、function_response #6、thought #10、thought_signature #11；还有媒体与新式 tool_call/tool_response 等分支。`BuildGenerateContentRequest` 明确调用 `processToolCalls`、`processUserOrContextMessage`、`processToolMessage`，没有把所有工具交互当普通 Markdown。

`GenerationConfig.ThinkingConfig` 包含 `include_thoughts` #1、`thinking_budget` #3、`thinking_level` #4、`include_raw_thoughts` #5、`enable_thinking` #6。**本次没有捕获该包里 Low 档对应的具体 budget/level 值；更不能由模型口述的 0.05 倒填该数值。** 选择模型枚举本身也可能参与服务端配置。

### 2.4 模型端视图的边界

| 内容 | 本次能定位到哪一层 |
|---|---|
| `<USER_REQUEST>`、`<ADDITIONAL_METADATA>`、`<USER_SETTINGS_CHANGE>` | 本地 serializer 已格式化；实盘 transcript 同时验证 |
| `<identity>` 等系统区块 | 本地 PromptBuilder 格式化为 XML；本会话区块配置可见，完整渲染结果未留存 |
| `<SYSTEM_MESSAGE>`、`<CONTEXT_SUMMARY>` | 本地模板和 serializer 中存在 |
| function declarations / calls / responses | 客户端 Request 中仍为结构化字段 |
| thought 标志、签名字节 | 客户端/协议中独立字段；首个 call 的签名现场存在 |
| `SPECIAL INSTRUCTION…0.05`、`declaration:default_api:…` | 本机构建未检出指定字面串；不能给出后端具体 renderer 实现 |
| 最终 token ID、控制 token 排序、嵌入及残差流 | 本地数据库/程序未提供该层观测 |

准确说法是：**客户端已经把业务规则和提醒的一部分渲染成文本，但仍把角色、工具、媒体和控制参数作为结构化请求送给服务端。服务端再转换为推理引擎所需输入。** 最终输入也未必是“所有 JSON 字段按键顺序变成的一段纯文本”：有角色边界、专用 token、媒体编码；generation 参数里有些只控制解码，不必全变成 prompt token。

模型所述“努力档 → 工具声明 → 11 块 XML”可以作为待解释的观测，**本次不能独立确认前两者最终 token 顺序，也不能把 JSON 对象键顺序当作模型 token 顺序**。

## 3. 系统提示词的组装顺序

### 3.1 当前配置的严格有序候选

直接来自首批 `executor_metadata.cascade_config.planner_config.declarative_mixin_config.prompt_sections`，首条 user_config 中也有相同列表：

| 序号 | builtin_name / 常规 XML 名称 | 当前可确认的条件或内容 |
|---:|---|---|
| 1 | `identity` | 身份/职责模板 |
| 2 | `terminal_sandbox` | 沙箱说明；本会话 `enable_terminal_sandbox=0`，不能把它算作必然输出 |
| 3 | `artifacts` | 产物目录及产物工作约定 |
| 4 | `conversation_transcript` | 本地日志回看说明 |
| 5 | `guidelines` | 工作指南 |
| 6 | `communication_style` | 沟通风格 |
| 7 | `slash_commands` | 斜杠命令说明 |
| 8 | `messaging` | 消息机制 |
| 9 | `mcp_servers` | 依当前可用 MCP 生成；不是无条件 `<mcp>` |
| 10 | `subagents` | 子代理目录/使用约定 |
| 11 | `skills` | 技能目录与使用约定；不是所有技能全文无条件灌入 |
| 12 | `user_rules` | 加载的用户/项目规则；本会话载入项目 `AGENTS.md` |
| 13 | `user_information` | 用户/环境信息 |
| 14 | `subagent_reminder` | 子代理情境专用提醒；不能算主会话必然有的第 14 块 |

**14 是配置候选数，11 是模型口述的可见块数，二者可以因条件跳过而同时成立；本库没有渲染后全文，不能逐块确证哪三块为空。**

### 3.2 拼装实现与模板来源

- `google3/third_party/jetski/cortex/core/systemprompt/systemprompt.Construct`：`0x8e62460`。
- 同包 `Build`：`0x8e62700`，调用 `mixins.(*PromptBuilder).AddSections` → `Build`。
- `google3/third_party/jetski/cortex/mixins/mixins.(*PromptBuilder).Build`：`0x8e4fb60`；处理 section 的包含条件与内容，并应用定制。
- 同类型 `processContent`：`0x8e51480`；`formatContent`：`0x8e51600`，后者调用 `utils.FormatWithXMLTag`（调用点 `0x8e5174a`），并先 `strings.TrimSpace`。
- 模板/内容可经 `resolveContent`、`loadContentFromTemplate`、`loadConfigTemplateOverride`、`resolveInlineTemplates` 解析。源码路径字符串包含 `third_party/jetski/cortex/mixins/prompt_builder.go`。
- 内嵌模板提供器：`google3/third_party/jetski/prompt/template_provider/templateprovider.(*TemplateProvider).GetSystemPromptStringWithData`：`0x8690400`。
- `IdentitySection.Content`、`GuidelinesSection.Content`、`CommunicationSection.Content` 等提供模板来源；`UserRulesSection.Content`（`0x8e43680`）等还读取运行期规则。**内容不全是同一种固定常量**。

注册表 `systemprompt.init`（`0x8e61440`）还注册：`web_application_development, ephemeral_message, customizations, workflows, flows, featured_plugins, browser_safety, chrome_default_profile, plan_execute_protocol, swarm_worker, knowledge_items, persistent_context, agent_api, sidecars, planning_mode, planning_mode_artifacts, plugins` 等。它们不在上述 14 项里，不能说当前主会话把这些全部注入了。

因此：**本会话区块顺序是配置数组定义的，输出是有条件的；不是全产品永远硬编码成同一段 11 块文本。** `memory` 也不是本次列表中一个独立必需标签，记忆类内容可以经 rules/knowledge/persistent context 路径进入。

## 4. Token 账本：已核实的数与无法倒推的数

### 4.1 首两次生成实盘对账

| 项目 | 第一次生成（step 1） | 第二次生成（step 3） |
|---|---:|---:|
| gen_metadata 行 idx | 0 | 1 |
| 服务 usage `input_tokens` | 44,133 | 6,255 |
| 服务 usage `cache_read_tokens` | 未设置 / transcript 记 0 | 44,134 |
| 服务 usage `output_tokens` | 193 | 162 |
| `thinking_output_tokens` | 2 | 2 |
| `response_output_tokens` | 191 | 160 |
| 本地 `estimated_tokens_used` | 53,019 | 58,959 |
| `max_context_tokens` | 786,432 | 786,432 |

第二次 input 与 cache read 两项相加为 **50,389**。不能把 6,255 写成“模型只看了 6,255 tokens”；该接口把缓存读取另列。也不要把本地 58,959 与服务端 50,389 混为同一个分词结果。

44,134 与首轮 44,133 接近，支持稳定前缀获得大量缓存复用；**差 1 的确切来源不能由汇总 usage 确定**。不能断言是 EOS、某个角色分隔符或工具 token。缓存统计也不能证明每层 KV 的具体物理存储方式。

### 4.2 四项精确拆分的可恢复性

| 要求的分项 | 本次能给的证据 | 精确 Carbon token 数 |
|---|---|---|
| 系统提示词（扣除 rules） | 有 section 配置和模板链；缺首轮渲染全文 | 无法恢复 |
| `AGENTS.md` / user_rules | loaded_customizations 证实加载该路径；当前文件 123,402 bytes / 57,867 字符 | 无法恢复历史实际注入与该模型分词数 |
| system tools | 有 16 个配置候选名称；缺最终筛选后的 schema | 无法恢复 |
| 首条用户消息 | 裸文 130 bytes；转换后完整文本 549 bytes / 537 字符 | 有字节数，没有 Carbon tokenizer 分项实测 |

首轮 `ChatModelMetadata` 的 descriptor 确实设计了 `system_prompt` #1、`message_prompts` #2、`tools` #8、`prompt_sections` #16；但 **现场 gen_metadata-0 未保存这些字段**。`context_window_metadata` 也只有估计总量和最大窗口，未保存 `token_breakdown` #3。缺失字段不是“0 tokens”。

`tokens.GetSystemPromptTokenGroups`（`0x87a0060`）与 `GetToolTokenGroups`（`0x87a04a0`）表明 `/context` 可用分组计数；前者调用 `tokenizer.GetDefaultTokenCounter`，后者对工具结构计数。**面板的本地估算不是服务端精确分账的承诺**。本次没有触发活跃窗口命令来改变现场，也没有用其他 tokenizer 假装精确 Carbon tokenizer。

额外注意：配置中 `rules_token_budget=20000`、`customization_token_budget=20000` 是配置值，**不等于已测得本轮规则恰好 20,000 tokens，也不足以单独判断补丁后的实际裁剪行为**。

16 个配置候选工具：`read_file, run_command, manage_task, send_message, schedule, invoke_subagent, define_subagent, manage_subagents, write_file, edit_file, search_web, swarm_manage, swarm_post_message, swarm_read_board, swarm_wait, ask_question`。**配置候选数不等于最终 16 个声明；本次未独立验证模型口述的 12 个声明数。**

## 5. 多轮顺序、角色与缓存

本次可复核的常规逻辑顺序是：

1. 本地生成系统 sections；工具声明独立组装。
2. 按会话历史顺序转换用户、模型回复、工具调用与结果等。
3. 用户 step 先变成带请求/metadata/设置变更的文本。
4. 模型回复可同时包含文本、thinking/raw thinking、签名和工具调用；不是所有字段都有内容。
5. 工具调用保留 name、arguments、id 等结构；工具结果与调用关联后变成 function response。
6. 新消息、hook、ephemeral 内容可能添加 step，也可能按配置并入临近消息/工具结果。
7. 到达压缩/裁剪条件时可缩减旧历史；此时不能继续假设整个前缀逐字不变。

当前 OneHarness 核心函数：

| `google3/third_party/oneharness/internal/core/serializers/serializers` | 地址 |
|---|---:|
| `serializeUserInput` | `0x8cfca60` |
| `DefaultSerializer.userInputToMessage` | `0x8cf7ba0` |
| `DefaultSerializer.systemMessageToMessage` | `0x8cf7d60` |
| `DefaultSerializer.ephemeralMessageToMessage` | `0x8cf7e60` |
| `DefaultSerializer.formatSystemMessage` | `0x8cf8420` |
| `DefaultSerializer.mergeSystemMessagesIntoPrecedingToolResult` | `0x8cf6020` |
| `DefaultSerializer.formatCheckpointWrapper` | `0x8cf8ce0` |
| `DefaultSerializer.checkpointToMessage` | `0x8cf8fa0` |

现场配置：`disable_system_message_merging=0`、`append_ephemeral_to_previous_tool_result=1`、`enable_context_role=0`、`use_structured_tool_responses=1`、`use_raw_user_message=0`、`disable_system_message_tag=0`。

**数据库的 source=SYSTEM，不等于 Gemini wire role=system。** 系统提醒在这里是客户端生成的内容，可经普通历史消息承载，也可合并进工具结果；真正系统指令另有 `system_instruction`。在当前配置下，不能把每个 `<SYSTEM_MESSAGE>` 都画成新的顶层 system role，也不能保证每个都独立占一个 user turn。

缓存命中的常见原因是前部系统文本/工具/历史保持稳定，新内容追加到后面；签名等 opaque 字段也被保存以便历史回传。**但“为了 KV Cache 所以所有动态变化必定只追加且从不动前缀”是过度概括**：条件 sections、工具集改变、模型切换、压缩、裁剪、消息合并都可能影响前缀。

## 6. 动态注入与提醒清单

先分两类：**hook 是执行控制接口，未必向模型写字；有文本写入的 hook 才是提醒。** 当前显式 pre-invocation 列表只有 `message_delivery → swarm_board_head → context_summary`。还加载一个名为 `herdr` 的用户 hook。

下表的“条件”以实盘或对应函数/常量可确认的范围为准；标“候选”表示存在实现，不表示当前会话已触发。函数简名后的包说明见本节末尾。

| 标签/名称 | 函数与触发条件 | 位置 / 关键原文 / 现场状态 |
|---|---|---|
| `USER_REQUEST` | OneHarness `serializeUserInput`；正常用户输入转换 | 本轮 user 文本首部。实盘已见；源 DB 保存裸文 |
| `ADDITIONAL_METADATA` / 时间 | 同上；取 step 时间并按本地时区格式化，格式串 `2006-01-02T15:04:05Z07:00` | 接在请求块后；`The current local time is: %s.`。不是每次重发历史都给旧消息换成“现在时间” |
| 用户身份、@提及、媒体说明 | 同上；相应字段非空时 | 同轮附加 metadata/媒体说明：`The user is: %s.`、`The user has mentioned some items in the form @[ITEM]…`、上传音频/视频/图片的说明；媒体本体仍可为结构化 Part |
| IDE 当前状态 | 同上调用状态转换；active state 非空 | `The user's current state is as follows:`；旧共享转换函数 `stateSnapshotToCascadeContext` 可格式化 `Active Document: %s (%s)`、`Cursor is on line: %d`、`Other open documents:`、`Running terminal commands:`。**首条现场 active_user_state={}，不能说本轮暗中灌入了这些 IDE 内容** |
| `USER_SETTINGS_CHANGE` | `serializeUserInput` 比较 user_config / last_user_config；旧路径 `UserInputStringConverter.buildOutputStringForConfigChange` 同类实现 | metadata 后。首条现场：`Model Selection` 从 None 改到 Carbon Low，并附 `No need to comment on this change…`。不是模型自己从对话猜到的设置 |
| `SYSTEM_MESSAGE` 通用包装 | `DefaultSerializer.formatSystemMessage`；有系统消息 step 时 | 本地模板：`The following is a <SYSTEM_MESSAGE> not actually sent by the user. It is provided by the system as important information to pay attention to.` 然后 XML。可与前工具结果合并 |
| 子代理/异步消息投递 | `prehooks.newMessageDeliveryHook` → `utils.DeliverMessages`（`0x8502d20`） | 生成投递内容进入后续历史，具体外壳依消息/serializer。当前 hook 已启用；不是所有消息都由模型主动调用 read 才能看到 |
| Swarm 看板头 | `nativeswarm.NewHeadHook`（`0x8cbc760`）；已加入看板且需告知状态时 | 系统提醒候选：`[board] the latest message is #%d (you have read up to #%d). Read anything you have missed, and wait for messages when you have nothing left to do.`；空板有 `nothing has been posted yet…`。hook 在当前列表，未证明本会话已加入 board |
| 目录新规则/技能发现 | `prehooks.newDirectoryCustomizationsHook`、`collectNewDirPaths`、`collectEmittedCustomizations`、`newDirectoryCustomizationsStep` | 新 step，去重已发项目；模板：`The following customizations were discovered from directories you recently accessed. Review the referenced files if they are relevant to your task:`。注册项 `directory_customizations`，**不在当前显式 prehook 列表** |
| 外部文件变化 / diff | `framework/stringconverters.(*FileChangeStringConverter).ToOutputString`（`0x86918c0`） | 可区分 `the USER` / 工具；`The following changes were made by %s to: %s.` + `[diff_block_start]…[diff_block_end]`。证明能渲染文件变更 step；**不证明本 CLI 当前无条件监视所有外部写入并自动发提醒** |
| 临时 ephemeral message | `executors/prehooks.NewEphemeralMessageHook` 与核心 `ephemeralMessageToMessage` | 可作为临时内容进入序列；当前允许附到前一工具结果。与永久改 system prompt 不是一回事 |
| `CONTEXT_SUMMARY` | `prehooks.NewContextSummaryHook` → OneHarness hooks；自动阈值或手动 compaction | 替代旧历史的 checkpoint/摘要，经 `formatCheckpointWrapper` 包装；详见下一节 |
| 强制继续 | `posthooks.NewForceInvocationHook` / `forceContinueStep`（`0x8ce8a20`） | 候选新增继续 step；模板开头 `You did not call any tools. You should continue calling tools until you've completed your task.`。依配置/计数触发，不是任意普通文本回复都必被催 |
| 空输出、待投递消息、终止步、次数上限 | `NewEmptyOutputContinuationCheckHook`、`NewMessageContinueCheckHook`、`NewTerminalStepCheckHook`、`NewMaxGeneratorInvocationsHook`、`NewChainCheckHook`、`NewNoToolCallCheckHook` | 当前均配置了相应项；有的只返回继续/停止控制。不能把 debug log 如 `undelivered messages found, continuing execution` 全当作模型可见文本 |
| goal 继续提醒 | `stophooks.NewGoalStopHook`、`buildReminderMessage`（`0x8d53180`） | 有未完成 goal 且满足 stop 条件时继续；模板 `You are still working toward the user's goal. Do not stop until the task is fully complete.`，含时间、continuation 次数与核对要求。当前配置有 goal_stop，**不等于已创建目标** |
| 子代理空闲/Swarm 停驻保护 | `stophooks.NewIdleSubagentGuardStopHook`；`nativeswarm.NewParkGuardStopHook` | stop hook 控制/候选提醒。当前配置了 `idle_subagent_guard, swarm_park_guard`；有对应状态才生效 |
| 产物请求反馈后停手 | `stophooks.NewRequestArtifactFeedbackStopHook`（`0x8d53800`）及产物相关流程 | 常量：`Note: You have just created an artifact and requested user feedback. Stop calling tools to end your turn…`。这是条件行为，不是每写一个文件都触发 |
| 工具时间戳 | `tooldecorators` 的 `tool_timestamp` | 工具结果添加 Created/Completed At；首个 read_file 的现场 transcript 已见。当前启用 |
| 工具输出截断 | `tooldecorators.truncateOutput`（`0x8d14300`） | **模型可见**工具结果提示：`The tool's output was truncated due to being too long. You can find the full output in a file at %s.`；中段标记 `<truncated %d of %d bytes from the middle of the output>`。当前启用 output_truncation |
| 编辑错误 / lint | FileChangeStringConverter 等工具结果；lint 配置开启 | 编辑不精确时提醒检查文件；配置 `auto_fix_lints_config.enabled=1`，通知开头 `As IDE feedback, the following lint errors may be related to your recent edits…`。这是工具/IDE 反馈能力，当前 CLI 是否每次实际得到 lint 另需运行观测 |
| 规划与 coordinator | 注册项 `plan_mode_review, plan_mode_handoff`；`owl/hooks.CoordinatorReminderBody` / `NewRemindPipelineCoordinatorRoleHook` | 特定 Plan/流水线角色的 hook/section，不在本次主会话显式 hook 列表，不能泛称所有轮都有 |
| 浏览器感知 | `browser/hooks.NewMcpBrowserPageSnapshotHook`、`NewWebviewBrowserPerceptionHook` 等 | 浏览器会话/工具条件下的截图、快照或临时反馈；本次主会话没有触发证据 |
| 用户 JSON hooks | `customization/hooks/jsonhook` + `hooks.BuildInjectedSteps` | 可构造 user/system/ephemeral/error/checkpoint/tool step。**现场 herdr hook 的脚本固定输出 `{}`，仅向 Herdr 上报会话，不注入提示词** |

关键包：`prehooks` = `google3/third_party/jetski/cortex/core/prehooks/prehooks`；`posthooks`、`stophooks` 同级；`nativeswarm` = `…/core/native_swarm/nativeswarm`；`tooldecorators` = `…/core/tooldecorators/tooldecorators`；`framework/stringconverters` 位于 `google3/third_party/gemini_coder/`。

### 6.1 完整注册面与当前启用面的区别

当前配置的 hooks/decorators **完整列表**：

| 阶段 | 顺序 / 项目 |
|---|---|
| pre invocation | `message_delivery, swarm_board_head, context_summary` |
| post invocation | `chain_check, message_continue_check, terminal_step_check, max_generator_invocations_check, empty_output_continuation_check, force_invocation, no_tool_call_check, swarm_wait_check` |
| pre tool | `battle_mode_tool_args, command_assessor, auto_run_safety_gate` |
| post tool | `command_assessor` |
| stop | `goal_stop, idle_subagent_guard, swarm_park_guard` |
| tool decorators | `tool_timestamp, output_truncation` |
| 用户自定义 | `herdr`，PreInvocation，命令模式，timeout=10；脚本只读检查，未执行 |

`force_disable_fundamental_components=1` 也在现场 declarative config 中。**不能把程序注册表中的全部旧默认组件再自动叠加到这份清单上。**

额外 prehook 注册项包括 `directory_customizations, plan_mode_review, battle_mode_max_turns, remind_pipeline_coordinator_role, mcp_browser_page_snapshot, webview_browser_perception, step_compaction`，以及兼容注册项 `checkpoint, checkpoint_validation, conversation_history, directory_rules_discovery, ephemeral_message, knowledge_artifacts`。额外 posthook 注册面还有 `command_assessor, plan_mode_handoff, checkpoint, conversation_log, ki_insertion, knowledge_generation, knowledge_timestamp`；stophook 注册面还有 `request_artifact_feedback_stop, mcp_browser_recording_stitch, browser_subagent_host_facts, pipeline_blackboard_updater`。

这是本构建所查核心注册面和当前配置的盘点，**不是宣称穷尽所有插件、所有工具错误文字、所有实验 variant、所有未来服务端提醒**。扩展 hook 能输出任意内容，因而不存在一个跨配置、跨服务器版本永远封闭的“全部提醒字符串表”。

### 6.2 负面问题明确回答

- **Git 状态是否每轮隐式塞入？** 本次没有证据证明存在通用的每轮 Git 状态注入。二进制含 Git 工具/状态逻辑，不足以证明它进了 prompt；首轮 active state 为空。
- **有无循环警告？** 有工具/编辑/lint 反馈中的避免打转措辞，也有继续/次数检查控制；没有证据证明每次循环都自动插一条统一“循环警告”。
- **有无错误重试？** usage 含 retry_infos、执行配置有 retry config；网络重试本身不等于向模型发一条提醒。工具错误则可作为模型可见结果。
- **有无预算逼近提醒？** 找到预算与 checkpoint 控制；**未确认本会话有统一“剩余 N tokens”模型可见倒计时**。
- **`truncation_reason` 是否自动通知模型？** 它是 ContextWindowMetadata 的结构化字段。仅见此字段不能断言有文本通知；工具输出截断则另有明确可见模板。
- **Todo / Artifact / Plan / Swarm 是否同一种机制？** 不是。任务可通过 `manage_task`/工具结果体现，Artifact 有反馈停手约定，Plan/Swarm 有专用 section 和 hook。不能统一画成“每轮追加一个 todo system-reminder”。
- **后台命令结束、定时任务、服务重启是否全有同一格式？** 通用系统消息/后台任务/调度机制确实存在，具体事件还取决于 producer；本次未制造这些事件，**不把所有事件的具体模板和 role 写成已实测**。

## 7. 压缩会改什么

当前 descriptor 确认 `strategy=1` 对应 `CHECKPOINT_STRATEGY_SAME_MODEL`（该枚举值标记 deprecated，不表示此路径不存在）；现场有 `max_token_limit=786432`、`token_threshold=50000`、`max_output_tokens=32768`、`use_model_reported_token_count=1`。**50,000 不能脱离策略解释成“到 50K 就立刻压缩”；窗口上限也不是旧文的 256K。** 具体 numeric strategy 值见证据附件，不沿用旧构建偏移表。

当前主路径 `core/prehooks.NewContextSummaryHook` 委托 OneHarness hooks；相关源码路径字符串 `third_party/oneharness/internal/core/hooks/context_summary.go`。手动触发时能追加：

```text
ON-DEMAND COMPACTION:
The user triggered this compaction. Follow any instructions in their latest message, if present.
```

`DefaultSerializer.formatCheckpointWrapper` 有两种本地格式：

```text
<CONTEXT_SUMMARY>
%s
</CONTEXT_SUMMARY>
```

或带解释的：

```text
<CONTEXT_SUMMARY>
The following is a summary of the conversation history that has been truncated to fit within the context window:

%s
</CONTEXT_SUMMARY>
```

`checkpointToMessage` 还能加入 User Requests、Code Interaction Summary、Conversation Logs、Running Background Tasks、Subagents 等恢复内容，具体依 checkpoint 数据及配置。二进制中也有 `# Resuming from a compaction` 展示/旧流程文字。

**两种字符串都存在，不等于当前一次请求重复注入同一份摘要两次。** 本会话初期 `checkpoint_index=-1`（wire uint64 解包呈 `18446744073709551615`），没有当前会话压缩后的请求样本，不能据此验证重复注入，也不能把摘要模板“可能复述错误”直接写成造成后续乱码的因果结论。

## 8. 公众号可直接使用的硬干货

| 容易混淆的两件事 | 可直接使用的说法 |
|---|---|
| 数据库 vs 聊天日志 | 数据库保存带类型的状态；日志是供人回看的投影，完整日志也不是模型请求的原包 |
| Protobuf vs JSON | Protobuf 定义消息结构；这个接口把对象序列化成 JSON 经 HTTP 发出，响应用 SSE |
| XML vs system role | XML 是文本约定，不会因名字叫 SYSTEM_MESSAGE 就自动获得顶层 system role |
| 工具说明 vs 工具调用 | 工具声明、调用、结果有结构化字段，后端再转成模型的输入格式 |
| Low 档 vs 那句 0.05 | 客户端有模型选择和 thinking 配置通道；0.05 的最终渲染来源本次未拿到后端证据 |
| 11 块 vs 固定产品协议 | 当前配置有 14 个有序候选，条件过滤后才得到实际系统文本 |
| input vs cache read | 第二次 6,255 是另计缓存后的 input 项；另有 44,134 cache read，不能说模型只看了 6K |
| 本地 token 估计 vs 服务 usage | 首轮分别为 53,019 和 44,133；不应用本地面板估计伪造服务端精确分账 |
| 动态提醒 vs 全局改系统词 | 可以追加 step，也可以并进工具结果，还可以是只影响执行的 hook；压缩则会替换旧历史 |
| 模型口述 vs 外部取证 | 口述可以提出线索；客户端逆向、DB 和日志分别验证能验证的边界，最终 token 串不能凭口述定案 |

建议主线：**App 管结构和业务文本，API 管消息协议，推理后端把协议变成模型输入。三层彼此相关，但不是同一份数据换三个名字。**

## 9. 未决项与可交付范围

本次已经核实程序版本/函数链、只读解包早期真实会话、系统候选顺序、当前 hook 全配置、用户包装、结构化工具签名、HTTP JSON/SSE 接口、首两轮 token/cache 统计和可见/不可见边界。

以下没有从证据中补造：首轮四项精确 token 分账、完整历史 system prompt、最终 12 个工具声明的确切 schema、Low→0.05 的后端映射、模型内部最终控制 token 的精确顺序，以及所有模式下每一种事件的运行触发。要再缩小这些缺口，需要未来单独的请求正文/运行期 serializer 取证和服务端 tokenizer/renderer 观测；不是再搜几遍 `strings` 就能解决。
