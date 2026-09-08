# 303：TTT / Titans / Nested Learning 核查

核查日期：2026-09-08。正文由主审统一修改。按主审追加任务，已修订两张 SVG 并用 cairosvg 渲染、逐张检查，校验 PNG 仅存 /tmp。

## 一手来源

- TTT v4 全文：https://arxiv.org/html/2407.04620v4
- 官方 PyTorch：https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py
- Titans v1 全文：https://arxiv.org/html/2501.00663v1
- Nested Learning：https://arxiv.org/html/2512.24695v1
- Google 2025-11-07 官方介绍：https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

## 必须修改

1. TTT 作者机构是 Stanford、UCSD、UC Berkeley、Meta AI，没有 CMU。
2. “清零”应统一改成“回到外层训练学到的共同初始状态”。TTT §2.7 的 W0 可学习；官方代码 531–552 行新建 TTTCache 时复制各层 W1/b1 等初值，清零的只是梯度缓冲。922 行 W1 是正态初始化的 nn.Parameter；动态副本才是推理隐状态。因此“在代码里不是参数”的绝对句也不准。
3. “每 16 个 token 更新一次”会误导因果性。§2.4：同批梯度都在批初状态计算，可以并行；每个位置仍累计截至自身的梯度形成自己的状态。官方代码 979–1007 行用 tril 实现因果前缀。图中改“16 个 token 一批，因果更新”，不能写“16 个词”。
4. 不用猜小模型宽度。官方代码 614–615 行 head_dim = width // num_heads；配置 31–55 行给 768/12、1024/16、1536/16、2048/32，因此分别为 64、64、96、64。922 行 W1 形状为 heads × head_dim × head_dim；MLP 两层见 1076–1078 行。正文可写每个头一个小模型，举 64×64。
5. “每个 Transformer 块注意力换成 TTT”适合作接法示意，不宜说论文全部模型就是这一种骨架。§3 明确默认采用 Mamba backbone，另有 Transformer backbone 消融。两种都可作为序列建模组件；不能把示意扩大成唯一架构。
6. 线性注意力严格等价条件补齐：§2.6 Theorem 1 为纯线性 f(x)=Wx、W0=0、η=1/2、全批梯度，得到无 softmax 的最简线性注意力。实验版另有 LN、残差、可学习初值、mini-batch，并不直接等价。非参数情形也需指定指数核加权估计，不是任意非参数方法。
7. “真正的门只有学习率”“门永远是0到1系数”均过满。官方实现还支持输出门（907–910 行）；论文学习率为基础率乘 sigmoid，MLP 基础率 0.1。写“这里的学习率也可以看作一道门”即可。
8. LoRA 与 TTT 区别不能归结为“存不存盘”。保存隐状态是工程选择，LoRA 也可继续在线训练。应落到本文方案的使用方式：前者把巩固结果作为后续使用的参数补丁，后者在序列内更新动态状态；学习目标、更新机制与生命周期均不同。
9. TTT 快一成要限定 §3.3 的 TPU v5e-256、2K context、TTT-Linear 0.27s vs Transformer 0.30s。长文本超过16K Mamba趋平是该文 Books3 / 匹配计算实验结果，不是所有 Mamba 的定律。
10. Titans 数字发生跨列拼接：Table 2 的80.2是 LMM / S-NIAH-N /16K，Mamba2 同列0.0；5.4来自 S-NIAH-PK。建议改 MAC / S-NIAH-PK /16K 的98.4对 Mamba2 5.4。340M Wiki困惑度25.43/25.07/24.69和TTT27.44可保留，对照应标 Transformer++。
11. MAC 是分段窗口内全因果注意力，见 Figure 3(a)，不是 MAG 的逐token滑动窗口。MAC 不能叫所有任务“效果最好”，可叫“长上下文表现较好的一种”。
12. Titans 图漏了最终输出：§4.1 Eq21–25 为旧记忆检索→拼接→注意力 y→更新记忆→用更新后的记忆再读取 y，与 y 门控合成输出。现图直接从注意力连输出不完整。守门员解释有原文依据，可保留，但不是保证不溢出。
13. Titans “会挑”不是相对 TTT 独有：TTT §2.1 本已解释大梯度写入更多，且学习率依输入。Titans 区别在动量、数据相关衰减、与注意力组合等。
14. 不宜把“TTT清零、Titans全文没说”抬成两者核心区别。后者未展示跨独立用户会话的持久记忆实验，也未给此部署协议；可直接写两者展示的是序列内记忆，不是跨天个人记忆。状态可保存不等于训练目标和干扰管理就已解决。

## Nested Learning

“不同更新频率的记忆谱系”准确。更准确的关系是 Nested Learning 为框架，Hope 是用该框架构造的 Titans 变体。参考作者应 Behrouz et al.，共有 Behrouz / Razaviyayn / Zhong / Mirrokni；2025-11是 Google 官方介绍日期，该 arXiv 版本首次提交是2025-12-31。两日期不要混写为 arXiv 11月稿。

## SVG 对应清单

- lora-vs-ttt.svg：两处清零/用完扔措辞、16个词、底部“差别不在怎么挂，在存不存盘”、LoRA部署后一律冻住、纯 Transformer 骨架示意需限定。
- titans-vs-ttt.svg：清零、16个词、MAC最好、滑动窗口、少最终记忆读取和门控、底部Titans才会挑、论文没有说清零作为中心结论，均同步改。

## 核查边界

核到了 TTT 作者官方代码。Titans v1末尾仍写代码将发布；本轮搜索没有找到可由作者页面确认的正式仓库，未拿第三方复现当作者实现，也不以缺少reset关键词证明跨会话协议不存在。
