# agy ARM64 补丁维护操作说明

更新：2026-10-18。此目录只保留当前修改版；用于 agy 自动升级后恢复长上下文、完整规则注入和关闭自动 checkpoint。按用户指定保留 `readme.md` 作为操作入口，供后续 agent/skill 读取，不自动安装到技能目录。

## 当前产物

| 文件 | 用途 |
|---|---|
| `agy.gz` | 已修改的 Linux ARM64 1.4.2 可执行文件，解压后 210981096 字节 |
| `agy.sha256` | 解压后的执行文件校验值 |
| `agy-patches.json` | 原版/修改版 SHA256、每处文件偏移和修改前后完整字节 |
| `fix-argon-context.py` | 此构建的完整幂等补丁；支持 `--target`，默认修改 `~/.local/bin/agy` |
| `verify-request.py` | 只读检查指定会话数据库中的请求窗口、规则全文注入 |
| `verification.json` | 当前版本已完成的小请求验证记录；不是新版本的验证证据 |

原版 SHA256：`842764a2ada5c71084254ea792a40edc988929e5dee490f3224c0b77b6b7d0eb`

修改版 SHA256：`faa76d5f6d24f43555149d090566cf4fc4265d61cba3ff52d152c8e24a6eddc5`

目录不存原版二进制。安装位置保留 `agy-1.4.2-original.bak`；补丁脚本创建备份且不会覆盖不同内容的备份。旧版 README、封面及归档已按要求移除，x86-linux 不在本次维护范围。

## 已知版本：恢复或重新打补丁

先检查 `command -v agy`、`file`、`agy --version` 和 SHA256。同为 1.4.2 也可能重新构建，以哈希为准。

对匹配原版哈希的文件执行：

```bash
python3 fix-argon-context.py
```

已是修改版则报告 Already patched；本脚本只接受当前构建原版或修改版哈希。未知哈希直接退出，不能删校验硬套。脚本在同目录写临时文件、保留执行权限，再原子替换，避免对运行中的文件原地写入。

解包归档用于检查或恢复（当前目录已有解包文件时先确认它的用途）：

```bash
gzip -dk agy.gz
chmod +x agy
sha256sum -c agy.sha256
```

需要覆盖安装时，先备份当前安装版，再把校验通过的解包文件通过临时文件原子替换到安装路径。恢复归档会回到本目录的版本；用户要求适配新版本时，走下一节，不用旧归档冒充升级适配。

## 新版本：重新定位后再修改

目标保持四项：Argon Low/Medium/High 三档的最终窗口预算均为 1000000；规则/skill 单文件上限 1048576 字节；rules/customizations 总预算各 131072 tokens；自动 checkpoint 谓词返回 false,nil。其他模型窗口不扩改。先确认新版本是否仍有这些限制；已被上游修好的部分不重复补。

### 1. 找函数，别沿用偏移

读取 ELF 程序头/节头，建立虚拟地址→文件偏移映射。此次代码范围两者相等，数据段并非如此，不能推广到整文件或下一版。

Go stripped 二进制仍可从 pclntab 找函数：搜索 Go 1.20+ ARM64 头字节 `f1 ff ff ff 00 00 04 08`，验证计数、边界、名称及函数代码。文件里可能存在嵌套二进制/第二套表，不要只靠 magic 命中就采用。

pcHeader 的 +8 是 nfunc；+24 是 textStart；+32 是 funcnameOffset；+64 是 pclnOffset。后两项相对表头。函数表每项是两枚 uint32：entryOff、funcOff；下一项 entryOff 给出结束位置。函数记录在 pclnOffset+funcOff，+4 的 int32 是名称偏移，指向 funcname 表。PIE 文件的 textStart 可能是零，要结合模块初始化/重定位和已知函数反汇编定位 Go text 起点，并用多个 getter 交叉核验。

1.4.2 的表头在 `0x1517550`，Go text 起点为 `0x61440c0`，这些只是本次定位结果。

找这些名称：

- `config.applyFeatureProviderCheckpointOverrideConfig`
- `hooks.checkpointNeeded`
- `customizations.(*Manager).readGlobalRules`
- `customizations.(*Manager).parseRuleFile`
- `customizations.convertSkillPath`
- `customizations.(*Manager).computeTruncatedCustomizations`
- `(*CascadeConfig).GetPlannerConfig`
- `(*CascadePlannerConfig).GetPlanModel`
- `(*CheckpointConfig).GetMaxTokenLimit`

用 `aarch64-linux-gnu-objdump` 或 `llvm-objdump` 查看各函数有限范围；不要把整份约 200MB 二进制的反汇编灌进上下文。

### 2. 在最终配置合并出口设窗口

仅修改早期 GetCheckpointConfig 常量、内嵌实验字符串或 UI contextWindow，可能再次被特征配置覆盖。1.4.2 只补最终合并出口即可，小请求已验证，无需复制 1.3.5 的早期试错改动。

逐项核实：模型名对应的枚举值；CascadeConfig 的 planner 指针偏移；planner 的 plan_model 偏移；CheckpointConfig 的 max_token_limit 字段宽度和偏移；出口现场寄存器与返回路径。

本构建：Argon High=1417、Medium=1418、Low=1419，planner 指针在 +8，plan_model 为32位字段 +64，max_token_limit 为32位字段 +16。出口 `0x75da010` 原指令为 `str x1,[x0,#16]`；x0 是外层配置，x1 是最终 checkpoint 配置，x3 可作临时寄存器。

补丁跳到 `0x7b404d0`：检查 planner 非空、模型编号在1417到1419之间（减1417后按无符号数比较是否≤2）；匹配时写1000000；随后无论匹配与否都补做原指令，跳回 `0x75da014`。精确机器码见 JSON，生成分支时校验4字节对齐和 ARM64 B 的范围。

### 3. 自动 checkpoint 与跳板空间

本构建 `checkpointNeeded` 入口 `0x7b404b0`，返回签名 `(bool,error)`。入口写：

```text
mov x0,xzr
mov x1,xzr
mov x2,xzr
ret
```

只清 x0 会把残留 x1/x2 当作非空 error 接口，之前真实崩溃过。新版本须从调用方重新确认返回约定。

入口短路后，后方函数体才可作为不可达跳板空间；检查不存在其他入口指向被占用区域。跳板不调用函数、不改栈、不改变文件大小，回到原执行点。不要把“关闭自动 checkpoint”写成“所有裁剪都关闭”；手动 compact 需另测才可宣称可用。

### 4. 两层规则限制一起处理

单文件24000字节截断与总预算20000 tokens是两件事。只放开字节数会让整份文件超过总预算，最后完全不注入。

本构建三处 `mov x2,#24000`（`0xd28bb802`）改为 `0xd2a00202`：

| 文件偏移 | 函数 |
|---|---|
| `0x7b82ccc` | readGlobalRules |
| `0x7b839c0` | parseRuleFile |
| `0x7b86e44` | convertSkillPath |

总预算要修改实际消费的配置读取路径和默认路径，两项共四处：

| 偏移 | 原指令 | 新指令 |
|---|---|---|
| `0x7b87fc4` | `0xb98000e7` | `0xd2a00047` |
| `0x7b87fd4` | `0xd289c407` | `0xd2a00047` |
| `0x7b88024` | `0xb98000c6` | `0xd2a00046` |
| `0x7b88034` | `0xd289c406` | `0xd2a00046` |

即 x7/x6 都设131072。函数前部还存在用于判断是否配置预算的读取，不要误认成最终消费点；沿控制流追到汇合及预算参数传递处。

### 5. 制作补丁与验证

先保存原版哈希和独立备份，再生成每处原字节/新字节清单。写入前校验整个构建哈希及每处原指令，修改后校验文件长度、指令和预期哈希。未知版本、结构不符或找不到安全跳板时停止写入，继续定位；不得靠试偏移覆盖可执行文件。

先检查 `--version` 能启动和重复运行脚本幂等，再发一条不调用工具的小请求；启动成功不代表补丁生效。在包含实际 AGENTS.md 的项目根目录开新会话：

```bash
agy --model gemini-4-argon-low --print '只回复 OK，不调用工具。' --print-timeout 45s --output-format json
```

取输出的 conversation_id，运行：

```bash
python3 verify-request.py <conversation_id> --agents /实际项目/AGENTS.md
```

该脚本读取 `~/.gemini/antigravity-cli/conversations/<id>.db`。它按当前 protobuf 字段号检查 gen_metadata：ChatModelMetadata→ChatStartMetadata→ContextWindowMetadata.max_context_tokens；prompt_sections 中 user_rules 必须逐字包含本地文件，并且没有预算排除提示。新版本字段变化时先从二进制 protobuf descriptor 核对，不能把解析失败直接当补丁失败。

同时核对 executor_metadata 中运行配置的 checkpoint 上限。当前 schema：executor 根字段10→CascadeConfig 字段2→CheckpointConfig 字段5；gen 根字段1→chat 字段9→start 字段10→window 字段4。脚本当前检查后者，前者用字段解析核对。

本次真实结果：模型回复 OK，实际请求上限983040（1048576减65536输出预留）；三档的user_rules均为118811字节，逐字包含AGENTS.md全部118430字节。1.4.2 **未做超过256K的长会话压力测试**；1.3.5 曾达到474.8K，不能当新版验收。需要完整长会话验收时交用户运行，再检查历史是否保留及是否裁剪。

## 每次更新本目录

确认本机补丁验证通过后，更新同名文件，不堆 v1/v2。以当前安装文件生成 `agy.gz`，同步哈希、补丁脚本、字节清单与验证记录；解压逐字节比对安装版，并将补丁清单逆向恢复的原版哈希对上，以免归档和脚本来自不同构建。

更新本文的版本、大小、定位和验证边界。操作完成后提示重启正在运行的 agy；新会话可重新加载完整规则，旧会话已丢的内容不会自动恢复。不改 git index，不自动提交或发布。

## 10月18日：1.4.2 升级后重新适配

自动升级覆盖了补丁。本次从新版 Go pclntab 重新定位并核对 getter、最终合并出口、checkpoint 函数与规则预算消费点，重新生成跳转和逐处字节清单。三档真实请求均回复 OK，模型编号仍为 High=1417、Medium=1418、Low=1419；executor 与 gen 两处实际上限均为983040。三档均包含本地 AGENTS.md 全部118430字节，user_rules各118811字节。

已验证启动、补丁幂等、归档解压与安装版逐字节一致、逆向补丁恢复原版哈希。直接分支扫描确认借用函数体只有新插入跳转引用。尚未重跑超过256K长会话，也未验证手动 compact；不能用旧版压力测试替代本版验证。

10月17日旧版只补 Low 导致切 Medium 后回到256K的教训继续适用：每次升级都须验证三档，不能只测默认 Low。
