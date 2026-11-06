# WorkBuddyAI 5.9.4：压缩与技能创建静态逆向（2026-11-06）

这是安装包静态证据，不是一次触发真实会话的网络抓包。没有修改安装文件、更新配置或启动模型请求。可逐字复提取的脚本：[extract.py](extract.py)；精确UTF-8字节偏移、长度和SHA-256：[manifest.json](manifest.json)。偏移从0起算，结束位置不包含在片段内。

## 版本与证据文件

安装根：`/mnt/c/Users/lmxxf/AppData/Local/Programs/WorkBuddyAI/`。

- Electron `app.asar` 的 `/package.json`：版本5.9.4；ASAR内绝对字节132165580开始，长度10158（以manifest为准）。
- Skill插件元数据给出精细构建号 `5.9.4-wb.42593613.gfa0a0901.hd4338c001c60`。
- CLI运行逻辑：`resources/app.asar.unpacked/cli/dist/codebuddy-lite-wb.mjs`。
- Prompt配置：`resources/app.asar.unpacked/cli/product.json`，不是一定有单独compact.tpl文件；解JSON后才是完整英文模板。
- 技能：`resources/app.asar.unpacked/resources/plugins/workbuddy-builtin/skills/skill-creator/SKILL.md`。
- ACP层的 `node_modules/@genie/agent-harness/lib/session/compact.js` 只定义`session/compact`请求、事件校验以及manual／auto元数据，不负责生成摘要。

## 机制：三种策略，至少三套压缩正文

路由按优先级PreMessage(3000)、MaxToken(2000)、Blocking(1000)选择。**不能把包里的compact-prompt单独说成所有路径实际发送的prompt。**

### 1. 手动／消息前的PreMessage：先工程压缩，再按需LLM总结

`aD.canHandle`接受`pre-message-auto`和`user-command`。先由HistoryEngineeringCompactor从最后压缩边界提取历史：

- 保留用户文字、助手可见文本、工具名和参数；工具结果替为`<omitted />`。
- 移除reasoning、function_call_result、旧标题／话题／summary／file-history-snapshot等类别；对工具参数中的content、widget_code、new_string字段也替换成`<omitted>`，图片转换为描述性占位。
- 外层包`<cb_summary>`，token粗估使用`ceil(JS字符串长度/4)`，不是精确tokenizer。
- 若结果≤有效上下文预算×15%，直接安装这份工程压缩结果，**不调LLM**。
- 若仍太大，调用无工具contextSummary agent，用`context-summary-agent-prompt`作为系统说明、`context-summary-prompt`作为总结正文。
- 二次总结失败则truncate-then-compact，仍失败才用原工程压缩；写入`isCompacted/isCompactInternal/isSummary`等标记并切掉之前模型可见历史。

**15%是这条路径判断“工程结果是否够小”的阈值，不是保证LLM输出恰好为原历史15%。** 常量可被`CODEBUDDY_ENGINEERING_COMPACT_SUFFICIENCY_PCT`覆盖。

这条LLM模板有一个值得单独引用的风险：要求摘要写上“All tasks described below are already completed / DO NOT re-run…”，但输入可能确有未完成任务，模板本身没有提供逐任务状态判别的可靠保证。不能把这句话当实际任务已经完成。

### 2. EMERGENCY_AUTO：MaxToken恢复

停止当前流，取压缩边界后历史，去系统警告并修补工具调用配对，交contextSummary agent，**正文切换为context-summary-max-token-prompt**。失败后将最后一批工具输出截断再试；再失败走确定性截断降级。成功后外包`<conversation_history_summary>`，替换模型上下文、恢复原agent并按需自动续跑。还有连续触发次数限制，避免压缩死循环。

这个九节模板要求保留待办和当前工作，与上面的“全部已完成”模板语义不同。

### 3. Blocking兼容／兜底

上述策略不匹配时使用Blocking。它调用名为compact的agent，系统模板是`compact-agent-prompt`，命令正文是`compact-prompt`；该正文是详尽九节模板，要求所有文件、所有错误和用户消息，强调完整代码／准确路径。先要求analysis思考，末尾又要求返回前删除analysis，只交带summary的XML。SummaryUtils会检查摘要结构，不是无条件接受任意输出。

包内compact agent虽然列有Read／ViewImage，模板却明确禁止调用工具；这是**提示约束与工具列表不完全一致**，不能写成该路径技术上绝无工具能力。contextSummary agent则明确tools=[]。

### 4. 恢复文本与工程摘要前缀

工程前缀原文：

```text
Summary of the conversation so far:
The conversation is between an AI agent and a user.
Use this to get up to speed, and continue helping the user as the AI agent.
Some contents may be omitted, indicated by "<omitted />".
```

中译：此前对话的摘要；对话发生在AI agent与用户之间；用它熟悉进度并继续帮助用户；部分内容可能省略，以`<omitted />`标记。

普通恢复原文：

```text
Please continue with the conversation based on the summarized context above. Maintain the same level of detail and helpfulness as before the summarization.
```

子Agent恢复模板：

```text
Please continue based on the summarized context above. Your original task was: "${t.originalUserIntent}" Maintain the same approach and level of detail.
```

中译：根据上方摘要继续，保持之前的详细程度和帮助水平；子Agent额外附上存储的原任务，并要求维持相同方法和详尽程度。

## 自动阈值：更正旧README第18节的过度结论

product.json确实有`compact.emergency=.4 / compact.modelOverrides.deepseek=.5 / summary.emergency=.15`，但本次追踪到的活动运行链读取的是：

- Runner.shouldCompact：`inputTokens.emergency`，包内为**0.9**；`CODEBUDDY_AUTOCOMPACT_PCT_OVERRIDE`可覆盖。冷却30秒，还要求有有意义的新历史。
- 消息前检查：`inputTokens.preMessage`，包内为**0.5**；受会话`preMessageCompactPct`、环境变量、功能开关、缓存过期、模型切换等条件影响。
- 有效预算来自受支持窗口／会话覆盖／默认窗口／maxInputTokens；再与autoCompactWindow取较小值后乘比例。无已知预算时另走100K–1M范围内的窗口兜底，并非所有情况简单R×A。
- 安装包productFeatures中未设置`enablePreMessageCompact`或`ContextSummaryAgent`，函数默认不启用单纯阈值路径；但缓存过期与切换模型分支仍可触发，远程产品配置也可能覆盖本地默认。

因此旧README“达到40%就一定强制压缩、全部压到15%”**不能由这些常量推出，应以本节为准**。若有效预算300K且使用包内0.9阈值，对应270K；若消息前0.5路径实际开启则150K。没有运行时有效配置和日志，不能声称某个用户必然在120K失忆。

## 技能创建：一个随包技能，不是单独训练

安装包有真正的skill-creator原件（英文14318字节）和init_skill.py、quick_validate.py、package_skill.py。调用入口由Skill工具/技能命令解析加载SKILL.md；不是在内核找到另一个固定`/learn`。包内commands数组未注册learn，不能排除用户额外安装同名技能。

创建链：具体例子→规划复用脚本／资料／资源→初始化SKILL.md→写内容→校验打包ZIP→真实使用迭代。写用户级`~/.codebuddy/skills/`或项目级`.codebuddy/skills/`，不改变模型权重，也不自动保留整段会话。

腾讯的具体扩展：

- 要求`agent_created: true`用于后续管理权限。
- 编辑市场技能，保存后在`_skillhub_meta.json`或`_knot_meta.json`写`userModified=true`以免更新静默覆盖；本地新建技能跳过。
- 三级按需加载：name/description常驻、SKILL.md触发加载、资源按需。
- 文档声称打包校验结构、引用和质量，**实际quick_validate.py仅检查前言/name/description、命名和描述中的尖括号等最小规则**；没有行为评测，也没有完整资源引用审计。init_skill模板中也没有自动填agent_created，要求最终写作者补齐；不能把prompt要求等同脚本已落实。
- productFeatures.SkillManage=false，尽管产品里仍含管理工具描述；存在模板不等于当前开启。

## Prompt边界与比较注意

所有文件是本地版本的默认模板。`promptRenderer`会合并动态上下文变量、语言设置；PreCompact hooks能追加自定义指令，插件可接管／跳过压缩；产品配置可更新。这里提供完整静态原文和已追踪的调用关系，不假装掌握任意账号服务端配置、展开后的私人对话或不可见服务端system prompt。

完整英文原文及中文对照附在主README本次增补；本目录保留独立原件，便于引用而不用再逆向一次。
