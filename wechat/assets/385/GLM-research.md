# 356 期研究底稿：Empress 退休、Hypervisor 清场与 Denuvo 的反击

（Tomo 整理，2026-10-08；供公众号 356 期写作用，正文由赤月执笔）

## 时间线

- 2025-11-25：Empress 发布告别信 "End of an Era – EMPRESS"（官方 Telegram，PDF）。理由：严重健康问题 + 对盗版圈的彻底幻灭。史上唯一稳定破解 Denuvo 的人类下线。
- 2025 末–2026 初：匿名组织 DenuvOwO 用 hypervisor bypass 技术（品牌名 "Hypervision"）实现 Denuvo 游戏当日/零日破解——传统方法要数月到数年。
- 2026 年（数月内）：FitGirl 宣布"所有单机/非 VR 的 Denuvo 游戏均已破解"。约 52 个老顽固全部沦陷，最后一个是 Madden NFL 25。Denuvo 问世十年头一次未破解名单清零。
- 2026-04-28：Tom's Hardware 报道 2K + Denuvo 反击：给 NBA 2K25/2K26 等已破游戏强推 14 天强制在线复验。
- 2026 年中：Denuvo/Irdeto 公开回应正在开发检测 hypervisor 攻击的新版本。
- 2026-09：《鬼武者 Way of the Sword》发售，至今（10 月初）无传统学习版——存量清空是燃烧，不是产能。

## 关键概念：bypass ≠ crack（356 期的核心科普点）

传统 crack（Empress 路线，Wired 2021 专访）：
- 挂调试器，追几千个散布全码的 trigger，逐个排除
- 一款游戏全职数周；RDR2 花两天属天才爆发
- 产出：独立可运行的破解 EXE，普通人双击就能玩

DenuvOwO 的 hypervisor bypass：
- **不碰 Denuvo 本身**。在 OS 之下装一层 hypervisor（AMD 用 SimpleSvm、Intel 用 HyperDBG 的 hyperkd），用 EPT（扩展页表）让 Denuvo 读到的内存页/硬件信息全是伪造的
- 内嵌预生成的 license token，让游戏以为运行在 token 对应的"合法机器"上
- 入口是替换 amd_ags_x64.dll 的代理 DLL，加载链条全程 kernel 级
- 代价：要禁用驱动签名强制（DSE）、PatchGuard、Defender——等于把 Windows 的免疫系统全关了装一个来历不明的内核驱动。安全研究者 RD945 有公开审计（github.com/RD945/hypervisor-crack-audit），Habr 作者原话级别的警告：没人像老 scene 那样 nuke 有毒 release 了
- Habr 作者结论：这方案性能差、不稳定、长期可靠性差——"与其用这种 release，不如把保护当研究对象"

**一句话：Empress 是把锁拆了，DenuvOwO 是把整栋楼搬进假布景。** 布景一次成本极低、当天交付，但每个住户都得自带发电机（内核驱动）。

## Denuvo 授权机制（公开研究拼图）

综合 Connor-Jay Dunn 2025 年分析、80.lv 对 Hogwarts Legacy 的报道、Steamworks 官方文档：

1. 首次激活：游戏收集硬件/软件特征 → 生成指纹；Steam 侧出加密 app ticket（所有权证明）→ 一起交给 Denuvo 激活服务器
2. 服务器下发机器绑定的 license token
3. 之后每次运行：用指纹参与解密内嵌的加密常量——指纹不对就解不出来，不需要联网复查（所以老版本离线能玩）
4. 这解释了 5 机/天的限制：token 和指纹绑定，换机器=重新激活

Hypervisor bypass 打的正是第 3 步：指纹是"问操作系统和 CPU 要来的"，那就伪造整个应答环境。

## 14 天复验：厂商的范式转移

- 从"一次性激活"转向"周期性查户口"——对付 bypass 的逻辑：token 可以伪造，但持续在线验证让预生成 token 有保质期
- Techdirt 等媒体批评：破解版零检查，正版用户反而两周必须联网一次——DRM 经典悖论又一次上演
- Denuvo 官方口径（TweakTown）：正在开发检测 hypervisor 的版本，"不影响性能"

## 与 355 期的咬合点（写作抓手）

1. 355 期的观察对象《鬼武者》恰好是"清零之后"的新游戏——大标题都喊"Denuvo 已死"，Zero 的正版观察证明新游戏还在守。**市面文章没人写这个反差。**
2. 355 说"防的是人类时间"——Empress 的死法（身体垮了、圈子恶心走了）就是这句话的终极样本。Denuvo 不用赢，等人类就行。
3. 但 hypervisor 事件说明它现在防的不止是人类时间了：DenuvOwO 的方法本质是**把人力活换成自动化的环境伪造**——和 355 结尾那条 AI 流水线是同一个方向（不硬懂保护逻辑，在更低的层面伪造前提）。AI 时代真正威胁 Denuvo 的不是更聪明的人，是不用聪明、只需要不出错的机器。
4. 开放问题（诚实记录，别下结论）：hypervisor bypass 明明是零日级产能，为什么《鬼武者》一个多月没有传统学习版？候选解释：bypass 不算传统学习版（门槛太高没人分发）/ Denuvo 新版已部分反制 / Xbox GDK 外层叠加。素材不足以裁决，写作时按"观察记录"处理。

## 链接池

- Empress 告别与生平：https://en.wikipedia.org/wiki/Empress_(software_cracker)
- Wired 2021 专访（方法论细节）：https://www.wired.com/story/piracy-group-takes-denuvo-cracking-bounties/
- 全清零报道：https://www.pcmag.com/news/rip-denuvo-all-games-protected-by-controversial-drm-now-cracked
- FitGirl 宣言：https://www.techpowerup.com/348609/
- 14 天复验：https://www.tomshardware.com/video-games/pc-gaming/denuvo-has-been-cracked-in-all-single-player-games-it-previously-protected-2k-and-denuvo-are-retaliating-with-mandatory-14-day-online-checks
- Techdirt 批评：https://www.techdirt.com/2026/05/08/（报道标题 "With Denuvo Completely Defeated, 2K Turns To Annoying Online Check-Ins"）
- Habr 技术长文（bypass 全流程）：https://habr.com/en/articles/1021894/
- 安全审计：https://github.com/RD945/hypervisor-crack-audit
- Denuvo 机制分析（Connor-Jay Dunn）：https://connorjaydunn.github.io/posts/denuvo-analysis/（检索摘要确认存在，直接抓 404，写稿引用前用网页存档或 Google cache 再核）
- Hogwarts Legacy 机制报道：https://80.lv/（2024-04，标题含 "The Denuvo System in Hogwarts Legacy Explained"）
- Steamworks 官方 ticket 文档：https://partner.steamgames.com/doc/features/auth

## 写作注意事项

- 链接按铁律 20 处理：不确定能打开的别直接给，标"发稿前核验"
- hypervisor bypass 的技术细节写到"概念层"就够——讲清 EPT 伪造指纹的思路，不给操作步骤
- 那句"CPU 品牌串被改成 DenuvOWO CPU @ 1337 GHz"是好细节，留着当彩蛋
