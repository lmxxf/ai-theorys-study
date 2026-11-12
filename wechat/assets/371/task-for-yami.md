# 逆向核实任务单：`agy`（Antigravity CLI）上下文拼装顺序、动态注入与 App/API/Model 三层数据形态对账

## 背景与目标
Zero 准备明天写一篇公众号（暂定第 371 期），彻底搞清楚并向程序员读者讲透一个平时大家都在用、却一直糊里糊涂的黑盒细节——**以本机正在运行的 `agy`（Google Antigravity CLI）为例：**
1. **上下文里从头到尾到底按什么精确顺序放了什么？**
2. **中途有哪些不定期塞进来的动态提醒（类似 Claude Code 的 `<system-reminder>`，在 `agy` 里叫什么、有哪些、什么时候触发、插在哪儿）？**
3. **App 侧看到的/发出的到底是结构化 JSON/Protobuf，还是和模型最终看到的纯文本一样？从“本地会话状态”到“网络请求报文”再到“模型残差流真正吃进去的平坦 Token 序列”，哪一步在客户端完成，哪一步在 Google 云端网关完成？**

请对本机二进制与当前活跃会话数据库做一次扎实、精准的静态逆向 + 实盘数据库解包核实（**够用、证据确凿即可，别过度打磨耗费无谓时间**），将完整核实报告写入：
`/home/lmxxf/work/ai-theorys-study/wechat/assets/371/agy-context-audit.md`

---

## 现场靶标与已知线索（直接复用，免走弯路）

1. **本机 `agy` 二进制**：`/home/lmxxf/.local/bin/agy`（x86-64 Go ELF，约 280MB，虽然 stripped 但 `.gopclntab` 函数名/源码路径与内嵌 Protobuf 描述符完整，内部模块路径如 `third_party/jetski/cortex/...`、`third_party/gemini_coder/...`）。可先查 `agy --version` 确认精确版本号。
2. **当前正在运行的活体对话（鳩的主窗口）**：
   - Conversation ID：`87600377-5184-4a58-9b89-cd51f6c29dde`
   - 实时 SQLite 会话数据库：`/home/lmxxf/.gemini/antigravity-cli/conversations/87600377-5184-4a58-9b89-cd51f6c29dde.db`（只读打开检查表结构、protobuf blob、`gen_metadata`、`executor_metadata` 等！切勿写锁破坏正在运行的会话）
   - 实时 Transcript：`/home/lmxxf/.gemini/antigravity-cli/brain/87600377-5184-4a58-9b89-cd51f6c29dde/.system_generated/logs/transcript.jsonl` 与 `transcript_full.jsonl`
   - CLI 日志：`/home/lmxxf/.gemini/antigravity-cli/cli.log`
   - 过往逆向参考（可速览免重复劳动）：`wechat/340-A.md`、`wechat/assets/362/agy压缩.md`
3. **来自鳩（模型内部第一视角）的实盘观测锚点（供你与客户端代码双向印证！）**：
   - 在模型视角的最开头（甚至在系统提示词正文之前），有一句：
     `SPECIAL INSTRUCTION: think silently if needed. EFFORT LEVEL: 0.05.`
     以及紧随其后的 12 个工具声明（模型看到的是类似 `declaration:default_api:read_file{...}` 的控制符结构），然后才是 `<identity>...<user_information>` 的 11 个 XML 标签块！
     👉 **请重点核实**：`SPECIAL INSTRUCTION: think silently if needed. EFFORT LEVEL: 0.05.` 这句话和 `declaration:default_api:...` 在 `agy` 客户端二进制里存在吗？还是 `agy` 客户端只发了结构化的 `thinking_config`（如 Low 档对应某个数值/枚举）+ 结构化的 `Tool` JSON/Protobuf 定义 + `system_prompt` 字符串，到了 **Google 云端推理网关（Prompt Renderer / Tokenizer 前置层）** 才被翻译拼装成模型看到的这套特殊控制符文本？！（如果客户端搜不到 `SPECIAL INSTRUCTION: think silently`，那就 100% 实锤了客户端结构化请求与模型字面视图的分界线！）
   - 本会话 `step_index: 1` 首轮 `input_tokens = 44133`，第 2 轮 `input_tokens = 6255, cache_read_tokens = 44134`。

---

## 需要核实并写入报告的 4 个核心模块

### 模块 1：App 本地存储 vs 网络传输协议 vs 模型最终视图（三层形态边界）
1. **本地状态层**：`conversations/<id>.db` 里有哪些表？核心 Trajectory Step（如 `CortexTrajectory` / `CortexStep`）存的是 Protobuf 还是 JSON？`transcript.jsonl` 是怎么从内存/DB 投影出来的？
2. **客户端→云端网络请求层**：`agy` 调大模型时走的是什么协议和接口？（gRPC 还是 HTTP/JSON SSE？目标服务名/方法名是什么，例如 Cloud Code / OneHarness / Jetbox / Gemini API？）发出去的 Request Protobuf / JSON 顶层有哪些核心字段？（例如：有没有独立的 `system_instruction` / `system_prompt` 字段？`tools` 是独立结构化字段还是拼进文本？对话历史 `contents` / `messages` 数组里每个元素长什么样？`thought` / `thinking` 签名、`tool_call`、`tool_result` 是结构化 Part 还是纯文本？）
3. **画出清晰的分界线**：哪些 XML 标签（如 `<identity>`、`<user_rules>`、`<USER_REQUEST>`）是在本地 `agy` 二进制里就已经拼成字符串塞进文本字段里的？哪些东西在 `agy` 离开网卡时依然是结构化对象（Protobuf/JSON），直到进了 Google 后端才被展开成模型看到的特殊控制 token（`SPECIAL INSTRUCTION...`、`declaration:...`、`thought`、`call:...`、`response:...`）？

### 模块 2：上下文从头到尾的精确组装顺序与 Token 账本
1. 顺着 `agy` 的 system prompt builder 与 trajectory converter 源码路径（给出真实 Go 包名与函数名），列出：
   - **System Prompt 内部各 XML 区块的严格拼接顺序**（核对是否固定为 `<identity>`, `<artifacts>`, `<conversation_transcript>`, `<guidelines>`, `<communication_style>`, `<slash_commands>`, `<messaging>`, `<subagents>`, `<skills>`, `<user_rules>`, `<user_information>`，还是还有条件开启的其他块如 `<mcp>` / `<memory>` 等？各自的模板原文来自二进制哪里？）。
   - **结合当前会话 `.db`（`87600377-...`）做一次真实字节/Token 拆解**：首轮 `44,133` input tokens 里，System Prompt 占多少、User Rules (`AGENTS.md`) 占多少、System Tools 占多少、首条用户消息占多少？（参考 `/context` 面板分类算法或 `executor_metadata` / `gen_metadata` 里的真实统计字段！）
2. 对话历史流（Multi-turn）的拼接顺序：每一轮 User 输入、Model 的 `thinking`（含 `thoughtSignature` 吗？）、Model 文本、`tool_call`、`tool_response` 是按什么顺序和角色（Role）排列的？为什么前缀能稳稳命中 `cache_read_tokens`？

### 模块 3：全量盘点所有“不定期动态注入块 / Prehooks / System Reminders”
在对话过程中，`agy` 会在不改变前面历史前缀（为了保住 KV Cache！）的前提下，往最新回合里动态塞入哪些额外块？请扫描二进制里的 `prehooks`、XML 标签模板与 step 生成器，列出**完整清单**（包含：① 标签/名称、② Go 函数名、③ 触发条件、④ 插入位置——是拼在 `<USER_REQUEST>` 同一个 User Turn 后面，还是单独作为一个 `SYSTEM_MESSAGE` Step 插入、⑤ 关键原文模板）：
- 已知必查项：
  - `<USER_REQUEST>` 与 `<ADDITIONAL_METADATA>`（时间戳是怎么每轮加进去的？还有别的 metadata 吗？）
  - `<USER_SETTINGS_CHANGE>`（切模型、改推理档位等何时触发？）
  - `<SYSTEM_MESSAGE>`（后台命令完成、子代理发来消息、定时任务触发、服务重启时怎么注入？角色是 `user` 还是 `system`？）
  - `<CONTEXT_SUMMARY>` 与 `# Resuming from a compaction`（达到窗口阈值或 `/compact` 时怎么替换历史？）
- 未知深挖项（帮 Zero 查清还有哪些藏在暗处的提醒！）：
  - 有没有文件被用户在外部修改的提醒？
  - 有没有 Active/Open files、IDE 光标、Git 状态变更等隐式上下文注入？
  - 有没有循环工具调用警告、错误重试提示、Token 预算逼近警告、截断通知（如 `truncation_reason` 触发时模型会收到提示吗）？
  - 有没有 Todo / Artifact / Plan / Swarm 模式下的特殊注入块？

### 模块 4：核实结论小结与可直接放进公众号的硬核干货表
用清晰的表格和简洁的工程语言收尾，方便鳩直接据此撰写公众号正文与绘制架构 SVG。只读分析，不要修改 `agy` 二进制或任何正在运行的会话文件。完成后回复一声即可！
