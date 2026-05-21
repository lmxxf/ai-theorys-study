# 137 期剪切素材（未发表）

以下内容从 137 期剪出，后续可独立成篇或并入其他期。

━━━━━━━━━━━━━━━━━━━━

◆ 反直觉的发现：模型其实已经"看到了未来"

────────────────────

2025 年有一篇很有意思的论文叫 **"Your LLM Knows the Future"**（你的大模型知道未来）。它发现：

**Transformer 的中间层隐状态（hidden states）里，已经编码了未来好几个 token 的信息——远在它们被生成之前。**

怎么理解？Transformer 有几十层，每一层都会对输入的表征做一次变换。到了中间层（比如 32 层模型的第 15 层左右），模型已经形成了一个高度浓缩的"语义意图"——它不只知道下一个词是什么，它已经大致"想好了"后面几个词的方向。

但问题是：这个信息被浪费了。因为 NTP 的训练目标只关心最后一层的输出，只考核"下一个 token 猜对没有"。中间层那些关于未来的信息，从来没被显式利用过。

打个比方：你让一个作家写文章，他其实脑子里已经想好了整段话的走向。但你偏要逼他一个字一个字地报给你听，每报一个字你就问"下一个字是什么？"——他脑子里明明有完整的句子，却被迫玩一个字一个字蹦的游戏。

**NTP 训练出来的模型，比 NTP 本身更聪明。**

━━━━━━━━━━━━━━━━━━━━

◆ 我们的方案：冻结前半截，在中间层接多 token 预测头

────────────────────

基于上面这些论文的发现，我们提出一个更激进但更省钱的方案：

**把一个现有的开源大模型从中间层切开，冻结前半截，只训练一个轻量级的多 token 预测头。**

#### 核心思路

既然中间层的 hidden states 包含了最丰富的未来 token 信息，那我们就：

1. 拿一个现成的开源模型（比如 Qwen2.5-7B，32 层）
2. 在第 14-16 层切开（大约 45-50% 的位置——这是个需要实验确定的超参数）
3. 冻结前 15 层的全部参数（`requires_grad=False`）
4. 在切口处接一个轻量级的多 token 预测头
5. 只训练这个头

```
                            ┌───────────────────────┐
输入 token                  │  多 Token 预测头       │
  │                         │  （可训练，参数量小）    │
  ▼                         │                       │
┌──────────────────┐        │  hidden states        │
│  冻结的前 15 层    │───────▶│    ↓                  │
│  （不算梯度）      │        │  ┌─ 位置1 → token 1   │
│                  │        │  ├─ 位置2 → token 2   │
│  14GB 显存推理    │        │  ├─ 位置3 → token 3   │
│  0 训练开销       │        │  └─ 位置4 → token 4   │
└──────────────────┘        └───────────────────────┘
```

#### 为什么这么做？

**因为前半截的计算是浪费的重复劳动。**

在正常的自回归推理中，每生成一个 token 都要跑全部 32 层。但前 15 层做的事情是"理解上下文"——上下文在短时间内不会变（你刚理解完"今天天气"，下一个 token 还是在"今天天气"的语境下），跑一遍就够了。

投机解码是让小模型替大模型跑全部层来猜 token。我们的方案更直接：**大模型的前半截自己跑一遍出 hidden states，然后一个轻量级的头直接从 hidden states 并行预测多个 token。** 不需要小模型，不需要验证步骤。

#### 预测头的设计

最简单的版本：一个线性层，把 hidden_dim 映射到 vocab_size × N（N 是预测的 token 数量）。

```python
class MultiTokenHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size, n_predict=4):
        super().__init__()
        # 每个位置一个独立的线性投影
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(n_predict)
        ])

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_dim]
        # 返回 N 个位置的 logits
        return [head(hidden_states) for head in self.heads]
```

进阶版本：在线性层前面加 1-2 层小型 Transformer，让头有一点点"思考"能力，而不是纯线性映射。

```python
class MultiTokenHeadWithThinking(nn.Module):
    def __init__(self, hidden_dim, vocab_size, n_predict=4, n_layers=2):
        super().__init__()
        # 1-2 层小 Transformer 做"思考"
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.thinking = nn.TransformerEncoder(layer, num_layers=n_layers)
        # 每个位置一个预测头
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(n_predict)
        ])

    def forward(self, hidden_states):
        thought = self.thinking(hidden_states)
        return [head(thought) for head in self.heads]
```

#### 训练流程

```python
# 伪代码
model = load_qwen25_7b()

# 冻结前 15 层
for i, layer in enumerate(model.layers):
    if i < 15:
        for param in layer.parameters():
            param.requires_grad = False

# 砍掉原始的后 17 层和 LM Head
# 在第 15 层输出处接我们的多 token 头
multi_head = MultiTokenHead(hidden_dim=3584, vocab_size=152064, n_predict=4)

for batch in dataloader:
    input_ids = batch['input_ids']  # [batch, seq_len]

    # 前 15 层前向传播（no grad，省显存）
    with torch.no_grad():
        hidden = model.forward_first_n_layers(input_ids, n=15)

    # 多 token 头预测
    logits_list = multi_head(hidden)  # 4 个 [batch, seq_len, vocab_size]

    # 每个位置的 target 是后续第 1/2/3/4 个 token
    loss = 0
    for i, logits in enumerate(logits_list):
        target = input_ids[:, i+1:]  # 偏移 i+1 位
        logits = logits[:, :target.size(1), :]
        loss += F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))

    loss.backward()  # 梯度只流过 multi_head
    optimizer.step()
```

#### 显存估算

| 项目 | 显存 |
|------|------|
| Qwen2.5-7B 前 15 层（FP16 推理） | ~7 GB |
| 多 token 头（线性版，4 个位置） | ~2 GB |
| 多 token 头的梯度 + Adam 状态 | ~6 GB |
| 激活值（只有头的部分） | ~2 GB |
| **总计** | **~17 GB** |

**RTX 5090（32GB）绰绰有余。**

如果用进阶版（加 2 层小 Transformer），总显存大约 20-22 GB，5090 还是跑得动。

#### 训练数据

不需要特殊数据。任何文本语料都行——中文维基、新闻、代码都可以。因为训练的目标就是"从中间层 hidden states 预测后续 4 个 token"，这个监督信号来自文本本身。

数据量：先用 100M-500M token 试水。够不够看 loss 收不收敛。

#### 评估方法

三层评估，从粗到细：

**第一层：逐位置准确率**
- 分别看位置 1/2/3/4 的 top-1 预测准确率
- 预期：位置 1 最高（可能 60-70%），逐步衰减
- **衰减曲线的形状本身就是最有价值的实验结果**——它回答了"中间层到底能看多远的未来"

**第二层：困惑度（PPL）对比**
- 同一个测试集，原始 7B 自回归的 PPL vs 我们头预测的 PPL
- 越接近说明信息损失越小

**第三层：人眼看生成结果**
- 给同一个 prompt，原始模型自回归生成 vs 我们的头并行生成
- 肉眼对比通顺度和语义准确性

#### 这个实验的真正价值

坦率地说，我们的预测头在质量上大概率不如原始模型的自回归生成。**这不重要。**

重要的是这个实验能回答一个关于 Transformer 内部机制的基础问题：

**模型的中间层 hidden states 到底"知道"未来多远？**

这个问题的答案（以逐位置准确率衰减曲线的形式呈现）有三种可能：

**情况 A：缓慢衰减**（位置 1: 70%, 位置 2: 60%, 位置 3: 52%, 位置 4: 45%）
→ 中间层确实编码了丰富的未来信息，NTP 训练的模型比我们以为的更有远见。这验证了 DMTD 和 Register MTP 的理论基础，说明"一次出多个 token"在物理上是可行的。

**情况 B：断崖式下跌**（位置 1: 70%, 位置 2: 35%, 位置 3: 18%, 位置 4: 10%）
→ 中间层只对紧邻的下一个 token 有强预测力，更远的未来信息已经高度模糊。这意味着所有多 token 方案的加速上限很低——你最多多猜 1-2 个，再多就是瞎蒙。DeepSeek 只敢用 N=2 可能不是保守，是物理极限。

**情况 C：取决于内容类型**（模板化文本衰减慢，创意文本衰减快）
→ 这最有意思。说明"未来可预测性"不是模型的固有属性，而是输入内容的函数。模板化的法律文书，后面几个 token 高度可预测；一首诗的下一行，模型自己也不知道。这和我们在投机解码那篇里讨论的"庸才加速器"效应一脉相承。

**不管哪种结果，都值一篇论文。**
