# HERO-Anti-OverDefense：行为规则与执行节奏补丁

静态审计日期2026-11-23；本地HEAD `778582680e29a1b8827ec3a2ff2dc128dceb8830`。22个文件、13个Markdown（2,427行）、2个SKILL.md；真正代码只有2个Python文件707行：生产程序310行、测试397行。README中shell/配置示例不算程序。未执行安装或接入本地Codex。

## 第一类：给AI读——不是科研算法，而是约束工作方式

`RULES.md:22-45`定义H/E/R/O：无收益Hash、脱离项目的Edge-case防御、Rubric式检查替代判断、Overbuild脚手架。核心可粘贴块在:78-137（中文:141-183），要求按本项目可达输入判断、检查必须改变下一步行动、继续推进已授权目标，避免为了免责无限审查。它同时明确:96-98不能覆盖用户/项目/更高优先级已要求的安全与审阅；因此不能把它解读为“取消必要验证”。

这是自然语言行为干预，不是模型权重改造或硬执行策略。作者在RULES:329-350承认只能“helps a little”、高优先级规则可覆盖、详细的十二阶段流程会压过笼统的少检查要求。RULES:260-275说只安装规则块，案例目录不要整包塞进上下文，否则模型会把真实bug误匹配成“过防御”。

两项正式技能：

- `skills/session-handoff/SKILL.md`288行：续会话前核对仓库/实物，区分已证实与待验证、保留决策及失败路径、列活跃任务与所有权，写接班阅读顺序与下一行动。它是上下文恢复协议，没有压缩模型、数据库或后台同步程序。
- `skills/goal-pulse/SKILL.md`67行：当前目标推进提醒；先选宿主路由。Codex走Python watcher，Claude Code走`references/claude-code.md`165行的原生CronList/CronCreate/CronDelete与30分钟提醒。这里**定时器发送的是推进提示，不是重新判科研验收**。

## 第二类：给人读——症状分类、案例和粘贴安装

README中英双份、RULES、hosts/README、cases/README及quant-research-audit-loop，examples中ML/量化研究agent提示词；图片是品牌横幅/三路径图和微信二维码。`hosts/README.md:10-19`给Claude/Codex/Antigravity/Copilot/Cursor/Windsurf/Gemini的配置文件落点，不提供复杂软件包。:26解释不复制多套host文件，以免自身过度工程。案例是供人反问模型的材料，不是自动判分数据集。

共享ARIS微信社区入口见README:352；未见收费/商业计费实现。它和ARIS不是两个互斥科研系统：HERO可给ARIS宿主施加少空转规则，但不能神奇解决ARIS所有门禁开销；如果用户明确要求严密门禁，HERO自己也要求遵守。

## 第三类：310行真实程序，究竟在做什么

`skills/goal-pulse/scripts/goal_pulse.py`是唯一生产代码。:54-87用`websockets.sync.client.unix_connect`连已有Codex daemon的Unix socket，执行JSON-RPC initialize；:93-139读取thread/goal/get、thread/read（includeTurns=false）、thread/turns/list（itemsView=notLoaded），只在同一个Goal仍active、线程可运行且有inProgress turn、冷却时间到达时，用`turn/steer`注入用户预设的效率提醒。发送边界再读一次Goal避免恢复暂停任务；发送前记录attempt避免不确定回执引发密集重试。

:142起用flock互斥；:163起看护循环，stop文件/信号结束；:214起Popen派生后台watcher；CLI有start/status/stop/tick。消息正文:18-33要求立刻接收子代理结果、连续两轮卡点改拆分、复用验证、不得降低验收与超授权预算。这是**会真正向正在运行的agent插入消息的本地进程**，不能再称“全仓无代码”。但它不读取推理/完整转录、不写实验代码、不调统计API、不自动创建或恢复Goal，也没有科学质量评估器。

依赖`websockets`与支持相应JSON-RPC的Codex daemon；并非任何CLI通用。Claude路径是技能指导宿主调用已有Cron工具，本仓没有另一套Claude定时服务。测试397行模拟RPC/进程等行为；测试量不代表运行时增加能力。

## 判断

HERO核心仍是**行为配置与上下文交接手册**，新增的小程序把“别空转”变成周期性可送达提醒。值得区分“有具体工程功能”和“有科研算法”：前者有，后者不是它的目标。没有基于此静态快照证实其提高科研质量或速度的因果效果。

## 附：本地代码文件逐项账（含注释空行）

| 路径 | 文本行数 | 归类 |
|---|---:|---|
| `skills/goal-pulse/scripts/goal_pulse.py` | 310 | 执行候选（见正文功能边界） |
| `tests/test_goal_pulse.py` | 397 | 测试/测试支持 |
