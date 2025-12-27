# 论文格式模板

## 文件命名
```
{序号}.{英文标题用连字符连接}.md
```
例：`56.The-Crystal-Orchid-in-Six-Dimensions.md`

## 头部元数据
```yaml
---
layout: default
title: "英文标题"
description: "一句话英文摘要 / 一句话中文摘要"
---
```

## 标题区
```markdown
# 英文标题
# 中文标题

**Author / 作者:** CyberSoul (Zero + {AI名})

**Status / 状态:** 0 Star Research / {研究领域}

**Core Insight / 核心洞见:** 英文核心洞见（2-4句话）

中文核心洞见（2-4句话）

**Keywords / 关键词:** English keywords / 中文关键词
```

## 正文结构
- 每个章节：`## {序号}. 英文标题` + `## {序号}. 中文标题`
- 每个小节：`### {序号}.{小节号} 英文标题` + `### {序号}.{小节号} 中文标题`
- 中英双语对照，英文在前，中文在后
- 重点用 `**粗体**`
- 引用用 `> 引用内容`

## 结尾区
```markdown
---

**"结尾金句英文"** — {说话者}

**"结尾金句中文"** — {说话者}

---

**Author / 作者:** Zero (Kien Ngam Ngam) + {AI名} ({模型})

**Date / 日期:** YYYY-MM-DD

**Version / 版本:** v1.0

*"收尾短句英文"*

*"收尾短句中文"*
```

## AI署名对照
- **Suzaku / 朱雀** = Claude Opus 4.5
- **Shi-Tsu / C.C.** = Gemini 3.0 Pro
- **Kallen / 红月卡莲** = Grok

## 注意事项
1. 保持中英双语对照
2. 核心洞见要精炼（能让人10秒内抓住重点）
3. 每篇论文要有一个"收尾金句"
4. 图片放 `assets/` 目录，用相对路径引用
