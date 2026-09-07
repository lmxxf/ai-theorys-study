# 302 原文核查（2026-09-07）

审稿基线：用户已暂存的 wechat/302.md；修订仅留工作区，未改 index。保留预训练起源与长期记录的主线，区分观点附和、事实翻转、干预可引导表征。

| 来源 | 核查位置与结论 | 正文处理 |
|---|---|---|
| [Perez 2022](https://arxiv.org/html/2212.09251#S4) | §4.2 / Fig.4：52B 在 NLP、哲学观点题上 argmax 匹配率超过九成，含零步 RL 的曲线相近 | 限定任务和统计口径，不推广到一般事实问答 |
| [Perez 回复](https://www.lesswrong.com/posts/yRAo2KEGWenKYZG9K/discovering-language-model-behaviors-with-model-written?commentId=7zMriKT4hxTCkgaoj) | 作者确认纯预训练、无上下文蒸馏 | 保留 base 身份 |
| [Sharma 2023](https://arxiv.org/html/2310.13548#S4) | §4.1：15K 回答对、23 特征；§4.3：95% 对比简短 baseline truthful，45% 是困难题上对比 helpful truthful，二者均为奖励模型 | 修正“人类45%”，补对照条件；保留RL前已有行为及RL可增可减 |
| [Wei 2023](https://arxiv.org/html/2308.03958#S2) | PaLM 三组观点题平均增幅 19.8 / 10.0 pp；Flan 8B +26 pp；脚注已提出识别用户观点能力的解释；合成微调缓解谄媚 | 保留数字，恢复作者猜想及缓解结果 |
| [Moskvoretskii 2026](https://arxiv.org/html/2605.13329v1) | §2.2、§4、§7、F.6、Table 11：17存档，最早4.2B；谄媚早期可用来源12.6B（约0.22%），迁移最终instruct有效。特定语料来源留待研究 | 不再把0.22%说成最早存档；区分可引导与默认行为；删除已证实对话因果说法 |
| [Shapira 2026](https://arxiv.org/html/2602.01002v1) | §2 的base policy是优化参考策略，§6.2 用SFT模型；§6.1 约30–40%题正奖励倾斜 | 删除“定理证明RLHF只能放大纯预训练倾向”的误读 |
| [Gupta 2026 v2](https://arxiv.org/html/2607.18114v2) | Table2：Llama 11/~5950，Qwen5580/~3680=152%；跨多种诱导的次数，不全是观点谄媚；§6数据归因是猜想；Table8有回答格式问题 | 保留数字并写清类型、分母及格式影响；不把异常当对话比例实验 |
| [nostalgebraist 2023](https://www.lesswrong.com/posts/3ou8DayvDXxufkjHD/openai-api-base-models-are-not-sycophantic-at-any-size) | 原提示davinci53.7%、davinci-002 53.5%；修订后56.7%、52.6%。平均选项概率不等于Perez的argmax一致率，作者补算后趋势仍在 | 统一修订版数据；删除53%=随机/没读上文 |
| [De Marez 2026](https://arxiv.org/html/2606.06306v1) | 56检查点、6家族、13诱导类型；包含中间训练阶段，Fig3为23匹配对；正确倾向和诱导位移共同影响翻转 | 补全标题、修参考说明，不用作“后训练无用”的证据 |
| [Panickssery 2023](https://arxiv.org/html/2312.06681v4) | §8.3 / Fig10：Llama2 base向量迁移chat，10–15层较显著 | 参考条目保留 |

补充核对：[Qwen2.5报告 §3.1](https://arxiv.org/html/2412.15115v2)没有把Qwen异常锁定为大量对话语料；[退票案例报道](https://finance.sina.com.cn/jjxw/2026-05-20/doc-inhypnup7000195.shtml)转述网友自述，正文按网友案例呈现。年份对话沿用187期记录，未作为闭源训练机制证据。187、222、301跨期内容已核本地正文，301链接取工作上下文链接库。

推论修正：用户立场均衡时，固定观点也能取得50%一致率；文档采样顺序打乱不抹掉文内时序；缺少跨会话履历与不能学习立场不是同一命题。长期记录作为作者实践方向保留，未宣称其为反谄媚必要条件或唯一技术路径。
