# 305 原始来源核查 · 2026-09-10

主线保留：研究预览已披露实现思路，完整规范未公开；社区接口不能当成官方定义。未运行设备、未申请访问、未修改暂存区。

| 原稿主张 | 核查与处理 |
|---|---|
| 公告只有几句话、没有内容 | [官方公告](https://www.anthropic.com/news/model-hardware-standard-research-preview)的展开案例包含状态/过程清单、共享内存字典、确定性脚本。补回机制，改为完整规范未公开 |
| 原语只有read/write | 原文 commands like 是例举，删除“只有”及据此推导的二选一设计 |
| 自然语言排除schema、MCP意味着统一传输 | 不成立；三种入口与标签说明不能证明内部编码、类型或传输 |
| 官网只有一个外链 | [官网](https://modelhardwarestandard.com/)同时有申请入口和公告链接；申请不保证特定交付材料 |
| 合作方无公开代码 | 公告直接链[Janelia Gently](https://github.com/gently-project/gently)，其公开代码不是完整MHS核心规范；修全称否定 |
| 同名GitHub组织是官方 | [组织页](https://github.com/modelhardwarestandard)无公开仓库，但没有仅凭同名认证其归属 |
| Fastly占位接口 | [README](https://github.com/fastly/edge-mhs)明确两设备及HTTP约定是占位符；也明确参数范围在转发前校验 |
| 小脑层是官方缺失 | [tongriyaotxt/open-mhs](https://github.com/tongriyaotxt/open-mhs)是依据公开信息的社区alpha，多数设备仅模拟；官方也讲了脚本执行。改为实现选择 |
| maximum只是提示 | [JSON Schema Numeric types](https://json-schema.org/understanding-json-schema/reference/numeric)：maximum是正式断言，由验证器执行可拒绝越界；描述与执行要分开 |
| 急停直接断动力、硬件故障必断电 | [Schneider停止类别](https://www.se.com/us/en/faqs/FA122781/)区分类别0/1/2；[Pilz SS1](https://www.pilz.com/de-DE/lexicon/sicherer-stopp-1-ss1)解释受控停止。改为按风险设计安全状态，不承诺任意故障都安全 |
| 新机械法规首次管软件、配置必属安全部件 | [2023/1230](https://eur-lex.europa.eu/eli/reg/2023/1230/en)Article3(3)有独立投放市场等条件；[适用时间](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=LEGISSUM:4682019)主要为2027-01-20；[旧指令](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02006L0042-20190726)AnnexI §1.2.1已管软件故障。删“首次”、预览豁免与自动定性 |
| 自动化工程师没入场 | [HN原帖](https://news.ycombinator.com/item?id=49468834)能证明讨论内容，不能证明整个行业参与者身份或沉默。缩到该帖未展开的具体控制问题 |
| 传统标准描述只给人看 | [SiLA](https://sila-standard.com/standards/)有命令、属性和类型；[OPC UA Part1](https://reference.opcfoundation.org/specs/OPC-10000-1/4.1)有机器可读信息模型，不能包装成MHS独创 |
| VLA包办毫秒伺服、文本无闭环 | [PI实时动作分块](https://www.pi.website/research/real_time_chunking)区分模型推理和动作执行；修为层级分工，保留连续表示对接猜想 |
| Therac-25事故 | [Leveson/Turner 1993](https://web.mit.edu/6.033/2004/wwwdocs/papers/Therac_1.html)有1985–1987过量辐射伤亡记录 |

删除未定位原始来源的“八万美元设备”与“token爆炸半径”社区引文，不将其冒充已核的原话。去掉ASCII分层图，以短段落说明职责关系。参考资料与摘要同步修订；保留用户自述已填申请表，不代为重复提交。
