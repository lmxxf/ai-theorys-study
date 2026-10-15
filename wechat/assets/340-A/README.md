# agy 修改版存档与续改说明

更新：2026-10-16。维护一份当前可用版本，后续直接覆盖更新，不堆 v1/v2；不保存原版执行文件或上游源码。cover.svg 是文章封面。

## 文件与当前状态

- `agy.gz`：完整修改版执行文件的 gzip 压缩归档，通过普通 Git 保存；解压得到 Linux ARM64 ELF PIE `agy`，v1.3.5，204820664 字节。本机目录保留了解压后的 agy，Git 忽略它以免重复存储。
- `agy.sha256`：当前执行文件的 SHA256。
- `agy-patches.json`：与最初原版备份逐字节比较生成的全部改动；含原版/修改版哈希、文件偏移、修改前后字节。原版执行文件本身未复制。
- 本说明：问题机制、修改位置、实测结果和续改入口。

解压及校验（在本目录执行）：

```bash
gzip -dk agy.gz
chmod +x agy
sha256sum -c agy.sha256
```

已有 agy 时，先确认是否要覆盖再解压。LFS 上传链路过慢，本次改用约 62 MiB 的 gzip 归档；已验证解压内容与安装版逐字节一致。

当前修改版 SHA256：

```text
d64feb0f3c39825b5d20a2bcf11c90624c89da8862942da8c6380950e130a40f
```

## 问题与实测结论

面板显示 1M，实际请求却被限制到 256000 tokens。关闭自动 checkpoint 后，发送请求前仍会裁剪历史；数据库记录过 `255921 → 247567 → 241167`，每次实际 `max_context_tokens=256000`，回落请求的 `truncation_reason=2`。不能把每次回落都解释成生成了摘要。

AGENTS.md 有两层限制：单文件 24000 字节，以及规则/自定义内容各 20000 tokens 总预算。只解开第一层后，完整文件超过总预算，会整份排除注入，只留文件路径和“需要时查看”的提示。这就是“连 AGENTS.md 都不读”的原因。

最终验证：

- 小请求正常回复 OK，执行配置及实际请求的上限均为 **983040 tokens**，等于 `1048576 - 65536`，与输出预留 64K 的逻辑一致。
- AGENTS.md 的 **118908 字节完整原文**逐字出现在请求的 user_rules 中；该 section 共 119283 字节，没有超预算排除提示。
- **2026-10-15 23:33，Zero 提供长会话面板：474.8K / 1.0M（45.3%），已跨过旧 256K 门槛。** 其中工具调用 414.1K、系统提示 43.2K。这是用户现场反馈；请求上限和全文注入另经数据库检查。
- 尚未测试接近 983040 的边界行为。手动 `/compact` 未做功能验证，不宣称已经验证保留。

## 实际修改

以下偏移仅适用于当前这一构建。完整差异以 agy-patches.json 为准。

### 1. 单文件规则限制：24000 字节 → 1048576 字节

三处 `mov x2, #24000`，指令字 `0xd28bb802 → 0xd2a00202`：

| 函数 | 文件偏移 |
|---|---|
| `customizations.(*Manager).readGlobalRules` | `0x77395dc` |
| `customizations.(*Manager).parseRuleFile` | `0x773a2d0` |
| `customizations.convertSkillPath` | `0x773d724` |

### 2. 早期窗口补丁：保留，但单独不生效

`utils.GetCheckpointConfig`：

- `0x703bfc8..0x703bfcc`、`0x703bfd4..0x703bfd8`：`0xd29d0002 + 0xf2a00062 → 0xd2884802 + 0xf2a001e2`。
- `0x703c298..0x703c29c`：`0xd29d0003 + 0xf2a00063 → 0xd2884803 + 0xf2a001e3`。

即 256000 改为 1000000。前两组参与 `min(模型可用额度, 上限)`，并非无条件赋值。后续配置覆盖仍能压回 256K。

Argon Low 模型编号是 **1419 / MODEL_PLACEHOLDER_M419**，内嵌配置映射继承 **M408**。M408 的 `CASCADE_USE_EXPERIMENT_CHECKPOINTER` 含 `max_token_limit="256000"`。已改成 1000000，并删掉一处缩进空格保持整个二进制长度不变；但单改这里，重启后实际请求仍是 256000。

### 3. 真正生效：修改最终合并配置的出口

函数 `config.applyFeatureProviderCheckpointOverrideConfig` 合并配置后，仅当 `planner_config.plan_model == 1419`，把最终 `CheckpointConfig.max_token_limit` 写为 **1000000**。

- `0x718d094`：原 `str x1, [x0, #16]` 改为跳到补丁代码。
- `0x76f7ca0`：补丁代码。x0 是 CascadeConfig 指针，x1 是合并后的 CheckpointConfig 指针，x3 是临时寄存器。
- CascadeConfig 的 planner_config 指针在 `+8`；planner 的 plan_model 在 `+64`；CheckpointConfig 的 max_token_limit 在 `+16`。
- 检查模型编号，匹配则写 1000000；执行被替换的 `str x1, [x0, #16]`，跳回 `0x718d098`。其他模型的窗口配置不改。
- 不新增调用、不改栈布局，不改变文件长度。后续窗口/输出预留处理将实际上限收为 983040。

补丁代码借用了已禁用的 `hooks.checkpointNeeded` 函数体：入口 `0x76f7c80` 直接返回 **false, nil**，指令为 `x0=x1=x2=0; ret`，后面的空间才可复用。

**Go 多返回值坑：不能只清 x0 就 ret。** x1/x2 是 error 接口返回寄存器，未清零会被当成错误对象；试验版曾因此崩溃，已回退修正，最终版请求通过。原先 `0x76f7e84` 和 `0x76f7f38` 两处 true→false 的改动也仍在，但当前入口直接返回，不会执行到它们。

### 4. 规则总预算：两项均为 131072 tokens

`customizations.(*Manager).computeTruncatedCustomizations` 中，同时改配置读取路径和默认路径，防止显式配置继续覆盖默认值：

| 偏移 | 原指令字 | 新指令字 | 用途 |
|---|---|---|---|
| `0x773e864` | `0xb98000e7` | `0xd2a00047` | 自定义内容预算读取路径，x7=131072 |
| `0x773e874` | `0xd289c407` | `0xd2a00047` | 自定义内容预算默认路径 |
| `0x773e8c4` | `0xb98000c6` | `0xd2a00046` | 规则预算读取路径，x6=131072 |
| `0x773e8d4` | `0xd289c406` | `0xd2a00046` | 规则预算默认路径 |

这两项预算修改不限 Argon；规则正文真正进入请求的实测已通过。

## 下次怎么继续改

1. 先校验本目录 agy.sha256，核对运行中进程实际加载的文件、inode 和状态。补丁后重启；不要把残留的暂停进程误认成当前窗口。
2. 读本说明和 agy-patches.json。字节清单按“连续变化的字节”分组，可能从指令中间开始；分析 ARM64 时按四字节边界读取。
3. 同一个原版构建可以重放清单：核验原版完整哈希和大小，验证所有 before_hex，写入 after_hex，最后核验 patched_sha256。不要在原版哈希不匹配时套用。
4. 升级后重新定位 Go 函数、结构字段和配置合并路径。旧脚本写死 text_start=0x5da40c0，还把虚拟地址当文件偏移；本构建恰好一致，不代表新版也成立。须根据 ELF sections/segments 做地址映射。
5. 修改后先做一个只回复 OK 的小请求，检查实际配置，再检查规则全文注入，最后让长会话跨过旧门槛。`--version` 正常和面板显示 1M 都不能替代请求检查。
6. 更新本目录 agy，并用 `gzip -n -1 -c agy > agy.gz` 更新归档，同时更新哈希、字节清单及本文；直接维护同一份，不堆版本文件。

安装时先退出 agy，将本目录执行文件复制到目标旁的临时文件、保留执行权限，再 rename 替换目标。避免原地改写正在执行的文件。此文件只适用于 Linux ARM64，不是 x86-64 构建。

## 本机工作路径与验证入口

本机增量补丁脚本：`~/.local/share/agy-patches/fix-argon-context.py`。它只负责最终配置出口和规则总预算，要求已有早期补丁；有哈希/原指令检查，**不是升级通用脚本，也不是原版一键安装器**。本目录的完整字节清单才包含全部修改。

本机回退点（未复制到本目录，不假设换机器仍存在）：

- `~/.local/bin/agy.orig.bak`：最初原版。
- `~/.local/bin/agy.before-argon-context.bak`：修改 M408 内嵌配置之前。
- `~/.local/bin/agy.before-argon-runtime.bak`：修改最终配置出口之前。
- `~/.local/bin/agy.before-rules-budget.bak`：修改规则总预算之前。

请求记录：`~/.gemini/antigravity-cli/conversations/<会话ID>.db`，SQLite 只读打开。`executor_metadata.data` 是 protobuf，检查其中 CascadeConfig 的 checkpoint_config；`gen_metadata.data` 是 CortexStepGeneratorMetadata，检查 chat_model.chat_start_metadata.context_window_metadata 与 prompt_sections 中的 user_rules。

- 上限验证会话：`6e2f9941-c7d2-442e-a39a-46f3c09f6ef2`。
- AGENTS.md 完整注入验证会话：`9f7e03a3-b6e0-41b3-be2b-e516d0151d14`。
- 临时检查工具曾放在 `/tmp/agy_inspect.py`，可能被清理，不应依赖它持久存在。protobuf 字段可从二进制内嵌 descriptor 提取。

内置 contextWindow=900K 已试过，实际仍为 256000，该设置已撤回。恢复旧会话不会自动补回先前已丢失的上下文；验证启动规则注入使用新会话。
