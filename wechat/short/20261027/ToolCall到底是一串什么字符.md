# Tool Call 到底是一串什么字符？——从 Agent 联网搜索的四种接线法到底层真实字节流

> **记录时间**：2026-10-27 晚  
> **缘起**：与网友聊到“AI Agent 接入网络搜索 API 有几种方式（Tool Call / MCP / CLI，平时用哪种？）”，顺手往下追问一层：剥掉所有 SDK 封装之后，大模型在触发 `Tool Call` 前后，真正吃进去和吐出来的原始字符流到底长什么样？

---

## 一、先理清层级：`Tool Call`、`MCP`、`CLI` 根本不是平级的三选一

很多刚写 Agent 的人容易把 `Tool Call`、`MCP`、`CLI` 并列起来选，但其实它们住在完全不同的层：

### 1. 站在“大模型自己”的视角看：永远只有 1 种方式——就是 `Tool Call`
大模型权重本身没有网卡，不会自己发 HTTP 请求。无论外面接的是原生 Python 代码、`MCP` 还是 `CLI` 命令行，到达大模型面前时，**100% 都被翻译成了 `Tool Call`（Function Calling）**——模型永远只是在上下文里吐出一段类似 `<tool_call>{"name": "xxx", "arguments": {...}}</tool_call>` 的结构化文本，等外面的壳子（Agent 宿主程序）去跑完，再把文字结果塞回上下文。

### 2. 站在“外面的壳子（Agent）”怎么接搜索 API 的视角看，有 4 种工程接线法：

| 接线方式 | 具体做法 | 适用场景与行业现状 |
|---|---|---|
| **① 壳子原生 `Tool Call` 直连搜索 HTTP API** | 给模型声明 `web_search(query)` 工具；模型吐出 `Tool Call` 后，Agent 自己的后端代码直接发 HTTP 请求调 **Tavily、Exa、Serper（Google）、Bing 或博查 API**，清洗截断后塞回给模型（如 345 期 `open-deep-research`）。 | **🔥 商业产品与生产环境绝对主流（占 80%+）**。少一层中间件，延迟最低、最稳，且能精确控制网页清洗与字符偏移取证。 |
| **② 大模型云端 API 原生内置搜索（Server-side Grounding）** | 调 OpenAI（`web_search`）、Claude（`web_search_20250305`）、Gemini（`google_search`）或 Kimi/豆包 API 时，直接在请求里传开关 `tools=[{"type": "web_search"}]`。模型吐出搜索指令后，**云端机房内部直接搜完并拼进上下文继续生成**。 | **最省事的商业主流**。连自己的服务器都不用发搜索请求，但没法深度定制中间的清洗和重排逻辑。 |
| **③ 通过 `MCP`（Model Context Protocol）挂载搜索插件** | 在客户端配置里挂一个独立的本地子进程（如 `brave-search-mcp` 或 `tavily-mcp`）。壳子通过 JSON-RPC 连上 MCP 读出工具列表，**转手还是拼成普通的 `Tool Call` 喂给大模型**。 | **通用桌面客户端 / IDE（Claude Desktop、Cursor、Cherry Studio）最常用**。本质是“壳子与外部插件之间的 USB 转接头”，方便零代码插拔，但高并发生产后端很少绕这道弯。 |
| **④ 通过 `CLI / Bash` 命令调用** | 不给模型单独注册 `web_search` 工具，只给一个通用的 **`bash`（执行终端命令）Tool Call**，在系统提示词或 `SKILL.md` 里告诉它：“要查资料就自己在命令行跑 `search-cli "关键词"` 或 `curl`”。 | **极客 Coding Agent 与沙箱 Agent 的新宠**。省掉了塞几十个工具 schema 的上下文开销（258 期《Pi 的胜利》），模型还能顺手用 `grep / head / jq` 自己过滤结果。 |

---

## 二、剥掉 SDK 封装：一次 `Tool Call` 前后，模型真正看到的 4 步原始字符流

平时调 OpenAI / Anthropic 兼容接口时，程序员看到的是优雅的 `tools=[...]` 参数和返回的 `response.choices[0].message.tool_calls` 对象，很容易产生一种错觉——以为大模型有一条专门收发工具结构体的“硬件通道”。

实际上（接 295 期《你拼的那个 messages 数组，模型根本没见过》），在分词器 `tokenizer.apply_chat_template()` 渲染之后，**整场调用完全是一次纯文本接力赛**。下面以 **`Qwen3`**（对比 **`DeepSeek-V4`**）的真实底层模板为例，完整还原一次联网搜索的 4 步字符流：

### 第 1 步：喂给模型的开头（你传的 `tools` 数组，其实被硬拼进了 `system` 字符串里）

你在 API 里传的 `tools=[{"name": "web_search", ...}]`，在底层被分词器模板直接转成了一段英文说明 + `<tools>...</tools>` XML 块，塞在 `system` 提示词末尾。此时喂进模型第 0 层的真实字符流长这样：

```text
<|im_start|>system
你是一个有用的助手。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "web_search", "description": "搜索互联网实时信息", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
MiniMax 刚开源的 428B 模型叫什么名字？<|im_end|>
<|im_start|>assistant
```
*(注：最后一行故意停在 `<|im_start|>assistant\n`，逼着模型顺着往下自回归接话。)*

---

### 第 2 步：AI 真正一个字一个字吐出来的字符流

模型接着 `<|im_start|>assistant\n` 往后预测下一个词元。它发现自己需要联网搜索，于是照着上面 `system` 里教的格式，**原封不动吐出下面这串字符**：

```text
<think>
用户问 MiniMax 刚开源的 428B 模型叫什么，我需要调 web_search 查一下。
</think>
<tool_call>
{"name": "web_search", "arguments": {"query": "MiniMax 开源 428B 模型"}}
</tool_call><|im_end|>
```

> **💡 各家门派的特殊 token 差异：**
> - **`Qwen3`**：直接用普通文本字符 `<tool_call>` 和 `</tool_call>` 包一段 JSON；
> - **`DeepSeek-V4`**：不用普通 XML 字符，而是在词表里专门注册了几个**人类键盘打不出来的特殊单 token**，它吐出来的原始流长这样：
>   `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>web_search\n```json\n{"query": "MiniMax 开源 428B 模型"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>`

---

### 第 3 步：推理引擎正则截胡停机 → 壳子去搜 → 把搜索结果直接拼接到刚才那串字符屁股后面

1. **半路截胡**：服务端推理引擎（如 vLLM / SGLang）一边收模型吐的字，一边盯着 `<tool_call>`；一看到模型吐出 `</tool_call><|im_end|>`，立刻**停机（Stop）**，用解析器（正则表达式或状态机）把中间那段 `{"name": "web_search", ...}` 抠出来，包装成 API 返回包里的 `tool_calls` 字段发给你的 Agent 壳子。**普通程序员在 API 返回值里看不到 `<tool_call>` 标签，以为模型天生会返回结构体，其实是推理引擎在半路帮你用正则抠出来的**；
2. **搜完续拼**：Agent 壳子拿到 `"MiniMax 开源 428B 模型"` 去调搜索引擎，拿到结果后，把结果包进 `<tool_response>` 标签，**原封不动追加在刚才第 2 步的字符串屁股后面**，并在最底下再挂一个新的 `<|im_start|>assistant\n` 开启第二轮推理：

```text
...（前面是第 1 步的 system 和 user）...
<|im_start|>assistant
<think>
用户问 MiniMax 刚开源的 428B 模型叫什么，我需要调 web_search 查一下。
</think>
<tool_call>
{"name": "web_search", "arguments": {"query": "MiniMax 开源 428B 模型"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"results": [{"title": "MiniMax 开源 MiniMax-M3.1-Flash", "snippet": "总参数 428B，采用 60 层全员稀疏注意力 MSA 与 DSpark 草稿头..."}]}
</tool_response><|im_end|>
<|im_start|>assistant
```
*(注：在 `Qwen3` 底层模板里，`role: "tool"` 渲染成字符流时，外层套的其实是 `<|im_start|>user`——也就是说对模型而言，工具返回的结果就像是“用户替我去跑了一趟腿，把结果塞在 `<tool_response>` 里发给我了”。)*

---

### 第 4 步：AI 顺着第二个 `<|im_start|>assistant` 吐出最终回答

模型读完刚才新拼上来的 `<tool_response>`，顺着最末尾那个 `<|im_start|>assistant\n` 接着往下吐字，生成最终给人类看的回答：

```text
<think>
搜索结果拿到了，名字叫 MiniMax-M3.1-Flash。
</think>
MiniMax 刚开源的 428B 模型叫 **MiniMax-M3.1-Flash**，采用了 60 层全员稀疏注意力（MSA）并自带 DSpark 投机解码草稿头。<|im_end|>
```

---

## 三、一句话总结

所谓的 `Tool Call`（乃至外面再包一层的 `MCP`），剥到最底层的字节流，其实就是：
**在 `system` 里用文字教模型写 `<tool_call>{JSON}</tool_call>` → 模型真写出这串字符后停机 → 外部程序用正则抠出 JSON 去跑 API → 把返回值包进 `<tool_response>` 贴在后面 → 让模型接着往下续写。**
从头到尾没有魔法，全是一场纯文本的字符串拼接接力赛。
