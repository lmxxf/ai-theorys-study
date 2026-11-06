from pathlib import Path
import json
O=Path(__file__).resolve().parent;B=O.parent.parent;manifest=json.loads((O/'manifest.json').read_text());P='agent-prompts-20261106/claude/'
def link(n):return f'[{n}]({P}{n})'
intro='''# Claude Code 2.1.311：上下文压缩与 Skill 创建逆向档案

取证日期：2026-11-06。只读取本地安装文件、官方插件市场缓存；没有启动模型会话，没有修改安装文件。本文件讨论提示词与路由的静态实现，不把字符串存在当作当前账号必然命中某实验臂的证据。

## 1. 样本身份与最重要结论

- 目标：`/home/lmxxf/.local/share/claude/versions/2.1.311`，`~/.local/bin/claude` 指向它。
- 二进制SHA-256：`86dc1fcc69498159eaf15f446613e847010f9036bfbbdffd4c49350be8489ada`；长度293,776,752字节。
- 下文偏移均为**该文件零起算字节偏移**，不是虚拟地址。区间结束不包含末字节。原始source及生成后展开文本分开存储，`manifest.json`给出区间与哈希。
- CLI `/compact` 不是SDK那段“你还没完成任务”的短提示词。真正CLI实现有**八个实验臂**，共同禁止工具；control为九节详尽摘要；还存在lean、short、2000词上限等精简路线。静态回退值control，环境与远程实验配置可改变实际命中。
- `/learn` 没有在本版本静态命令注册中找到。`/skills`只列技能；创建技能至少有两个独立入口：**受门控的新版内置`/init` Phase6**，以及**单独安装的官方skill-creator插件**。不要把“官方插件”写成“二进制自带命令”。
- 本地存在官方市场目录中的skill-creator全量文件，已复制保存；未据此声称它已经在当前会话安装或启用。

## 2. 精确证据入口

|实现|十进制字节偏移|十六进制|意义|
|---|---:|---|---|
|`name:"compact"`|249244268|0xedb2a6c|本地slash命令，支持可选摘要指令|
|`NJ`|250549998|0xeef16ee|拼装完整压缩提示词|
|`Uen`|250549856|0xeef1660|部分压缩from/up_to提示词|
|`HJ`|250553531|0xeef24bb|读取环境／实验配置选择臂|
|`iQ`|249994624|0xee69d80|正常自动压缩阈值|
|`US`|249997688|0xee6a978|有效压缩窗口来源解析|
|`UQ`|250000828|0xee6b5bc|窗口扣预留输出|
|`G3e`|250855717|0xef3c125|执行层拒绝工具调用|
|`Con`|250855915|0xef3c1eb|压缩子请求、共享前缀与回退|
|`/autocompact`|250140715|0xee8d82b|调整自动压缩窗口的命令|
|`/skills`|250183586|0xee97fa2|列出技能，非学习命令|
|新版init Phase6|250163107|0xee92fa3|把用户偏好队列转换为SKILL.md|
|官方插件安装建议|250169613|0xee9490d|明确要求另装skill-creator插件|
|恢复引导句|243651860|0xe85d514|“从超出上下文的旧会话继续”|

原始[manifest](agent-prompts-20261106/claude/manifest.json)、[可重跑提取器](agent-prompts-20261106/claude/extract.py)、[完整拼装源码区间](agent-prompts-20261106/claude/compact-builders.source.js)。提取器只对已截出的常量和纯模板函数进行字符串求值，不执行整份安装代码。原文中的动态用户指令没有替人编造；下面展开的是无额外指令、完整转录路径可用时的模板。

## 3. `/compact` 与 auto-compact 的实际机制

### 3.1 命令、阈值和选择

`/compact`注册为local命令，`DISABLE_COMPACT`可使它不可用，参数是optional custom summarization instructions。它加载`/$bunfs/root/chunk-8vpkewhx.js`。自动压缩又受`autoCompactEnabled`设置与`DISABLE_AUTO_COMPACT`控制。

不能写死“Claude在95%就压缩”。本版本可读代码给出的基本公式是：

```text
W = US(model, config).window        # 不超过模型contextWindow
E = W - min(model.maxOutputTokens, 20000)
T = E - 13000
若有效 CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = p (0<p<=100)：
    T = min(floor(E*p/100), E-13000)
```

W可能取环境`CLAUDE_CODE_AUTO_COMPACT_WINDOW`、模型/默认设置、clientdata、实验、模型默认值或自动窗口；`US`负责来源优先级与钳制。提前预计算还有独立buffer fraction；不能把它当成用户可见实际清空阈值。还存在收到prompt-too-long后反应式压缩、保留近期分组、预计算摘要复用与连续失败熔断路径，3次连续失败可停止本会话后续尝试。

`HJ()`读取`CLAUDE_CODE_CURRIED_TRINKET`，否则读取实验`tengu_curried_trinket`，默认control；不在八种枚举中的值回退control。本次没有读取账号远程实验结果，因此不能声称所有用户实际都在跑control。

### 3.2 请求和恢复

正常完整压缩路径`_Nr`会先触发PreCompact hook，把用户自定义摘要指令与hook新增指令拼到一起，选择臂，再由`NJ`组装：

```text
Nen(replyShape)                     # 开头禁工具、规定analysis/summary形状
+ selectedArm.instructions(hasTranscriptPath)
+ [若有自定义要求："Additional Instructions:\n" + 要求]
+ Ben(replyShape)                   # 末尾再禁工具
```

`Con`优先fork现有上下文以共享prompt cache，单次摘要maxTurns:1；使用主循环模型及允许的fallback，不是这里硬编码一个小模型。`G3e`在执行层返回deny，理由“compaction agent should only produce text summary”，所以“禁止工具”不只靠提示词自觉。缓存共享失败还有其他摘要请求路线，输入太长时可丢弃部分头部重试，不能保证所有原始输入都进了总结。

完整压缩成功后写compact boundary，用summary及恢复附件重建上下文，`messagesToKeep:[]`；部分／反应式压缩可保留近期消息。恢复端`aZo`去掉analysis块，把summary标签内容改成`Summary:`，`$J`加恢复引导、工具信息与可用的全文转录路径；若头部曾被裁掉，会提示摘要不覆盖最早部分，不要猜。恢复时也可附“不要再问用户，直接继续最后任务”；short_v2另加验证交付物的持续工作要求。

系统、项目规则、记忆、计划、已加载技能、后台状态与近期文件由恢复/附件通道重建；lean提示词明确告知最多重读最近五个文件（长文件截短）。这不等于保留整个历史，也不等于用户消息被程序逐字原样留在新窗口：多数用户文字在摘要里由模型转述，精度依提示词和模型决定。

PreCompact与PostCompact属于程序钩子扩展面；用户配置可以改变摘要输入或恢复内容。因此“二进制默认模板”与“这个用户当次实际API输入”必须区分。

## 4. control 完整英文提示词及全文中译

这是静态回退臂的完整无自定义指令展开文本，非断言当前账号一定命中。

'''
parts=[intro,'### 英文原文\n\n```text\n'+(O/'compact-control.txt').read_text()+'\n```\n','### 中文全文对照\n\n```text\n'+(O/'compact-control.zh.txt').read_text()+'\n```\n']
parts+=['\n## 5. 其他臂与部分压缩的完整原文\n\n八种臂及两种部分压缩均已展开，以下不是从文案猜测出来的名称。共同模板源码和manifest能复核插值来源。\n\n']
for n in ['no_scratchpad','no_scratchpad_capped','no_scratchpad_keep_asks','lean','short','short_v2','capped']:
 parts+=['### '+n+'\n\n原始文件：'+link('compact-'+n+'.txt')+'。\n\n```text\n'+(O/f'compact-{n}.txt').read_text()+'\n```\n']
for n in ['partial_from','partial_up_to']:
 parts+=['### '+n+'\n\n原始文件：'+link(n+'.txt')+'。\n\n```text\n'+(O/f'{n}.txt').read_text()+'\n```\n']
parts+=['### 变体中文差异与精简正文中译\n\n'+(O/'compact-variants.zh.md').read_text()+'\n']
parts+=['''## 6. 不要认错：SDK旧版compactionControl不是CLI默认

二进制243772570字节处的“ You have been working on the task described above ... ”属于内嵌Anthropic SDK的toolRunner。附近代码`rd=1e5`、`contextTokenThreshold??rd`和`summaryPrompt??od`只说明该SDK旧分支默认100000 token、可覆盖模型和摘要提示词。旁边已写明compactionControl deprecated、建议server-side compaction；不能把100K套给Claude Code自动压缩阈值，更不能拿此旧模板代表CLI的八种臂。

原始文件：'''+link('sdk-legacy.source.js')+'、'+link('sdk-legacy.txt')+'。\n\n```text\n'+(O/'sdk-legacy.txt').read_text()+'\n```\n']
parts+=['''## 7. 技能创建：新版 `/init` 内置分支

`m4o()`由`CLAUDE_CODE_NEW_INIT`或`tengu_slate_harbor_experiment`开启新init；否则老init只以创建CLAUDE.md为主。新版完整提示词不是一句“从工作里学经验”：它是Phase0–8引导向导，先查已有CLAUDE.md、问要建项目还是个人规则、是否skills/hooks，探索仓库、收集用户偏好并提案，再创建文件。

Phase6把Phase3收集的skill条目变成`.claude/skills/<name>/SKILL.md`；正文应使用用户访谈原话与仓库调查得到的命令、格式、部署目标。重复工作流或专门参考知识也可建议技能；已有技能先查，不覆盖。默认用户和模型均可调用；有副作用工作流加`disable-model-invocation: true`，可用`$ARGUMENTS`传参。它没有独立“自动把每次会话变成skill”的后台事件。

完整英文模板原件：'''+link('new-init.txt')+'；带精确字节区间、保留条件表达式的源码：'+link('new-init.source.js')+'''。`new-init.txt`展开了`yHe()=false`的环境条件，省去的是检测到Codex/Gemini配置时的迁移建议；完整条件仍保留在source文件，不能把展开视为所有环境的固定输入。

Phase6中文逐项对照：

1. 技能按需增加能力，不膨胀每个会话。
2. 首先消费Phase3偏好队列的skill条目，按用户偏好命名，以用户原话和Phase2发现写正文；若已有捆绑技能覆盖同类工作，项目技能只补用户约束，告知原技能仍在。
3. 偏好信息不足时简短追问。
4. 再建议特定任务参考知识、可重复工作流；每项给名称、一行用途、适合仓库的原因。
5. 先审查已存在`.claude/skills/`，只建议互补技能，不覆盖。
6. 写入标准YAML name/description及模型指令。
7. 默认模型和用户可调用；有副作用流程加`disable-model-invocation: true`，用`$ARGUMENTS`接输入。

## 8. 官方 skill-creator 插件：完整英文与中译

本机读取源：`/home/lmxxf/.claude/plugins/marketplaces/claude-plugins-official/plugins/skill-creator/skills/skill-creator/SKILL.md`，485行，文件偏移0起全量。插件manifest作者Anthropic；本缓存未提供可复核Git提交，不凭名字捏造版本号。所有原始文件SHA在本档案manifest中。

二进制新版init在250169613字节明确建议先`/plugin install skill-creator@claude-plugins-official`再`/skill-creator <skill-name>`。市场缓存可读不证明安装启用；本次不进行安装也不运行它的流程。这里把它作为研究对象，不把它的提问、测试、开浏览器等指令当作本次逆向任务要执行的指令。

它确实明确覆盖“把当前对话流程固化成技能”：Capture Intent要求先从历史提取工具、步骤、用户纠正和输入输出格式，用户填缺口；不是只教空白写模板。

原件整套：'''+link('official-skill-creator/skills/skill-creator/SKILL.md')+'''；额外的grader/comparator/analyzer提示词及schemas、描述优化脚本、打包脚本也按原目录完整存档。主技能可展开如下。

### 英文全文

````markdown
'''+(O/'official-skill-creator/skills/skill-creator/SKILL.md').read_text()+'\n````\n\n### 中文按节对照\n\n'+(O/'skill-creator.zh.md').read_text()+'\n']
parts+=['''## 9. 设计解剖与可引用边界

- Claude的control同样要求列全部错误、完整代码、ALL user messages；“错误细节过度保留”不是agy独有特征。提示词可能造成冗长或保存噪声，但静态文本不能证明会必然诱发乱码、复读或人格改变。
- control九节与分析块有利于交接细节，也消耗摘要预算；no_scratchpad、lean和short说明开发者正在权衡这一点，不能以单一旧模板概括全部Claude用户。
- 伪用户回合鉴别是明确设计：助手编出的“user:”不算授权；保留用户约束、引用最新任务，防止摘要把自己的计划升格为用户要求。这与单纯压缩字数不同。
- skill-creator强调流程提取、用户纠正、泛化、确定性脚本复用和有技能／无技能比较；它的学习是把说明与资源写进文件，**不更新模型权重**。
- 独立测试基线与留出触发测试比“生成一个SKILL.md就宣称学会”更可验，但成本也更高。脚本能跑、缓存存在、提示词写了评估，均不等于本次已验证其真实效果。
- `/init`、插件和自动压缩分别是不同路径；不要把自动记忆、CLAUDE.md、创建skill和压缩摘要混作一个“长期学习”机制。

## 10. 复核

保存了完整原文、源片段、哈希清单与可重跑提取器；没有改二进制，没有触发实际compact，没有消耗模型请求。英文展开文件可由提取器重建；中文是本次翻译，不冒充官方中文提示词。新init的环境条件与远程实验命中仍应按实际运行证据区分。
''']
(B/'claude压缩与skill.md').write_text('\n'.join(parts))
print(B/'claude压缩与skill.md')
