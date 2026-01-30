# 悬链线模型实验验证

验证论文《The Catenary of Cognition》的核心假说：U 型 attention 反映的是**人类语言本身的信息分布结构**。

**论文链接**：[Zenodo](https://zenodo.org/records/18334266)

---

## 核心假说（修正版）

**原始表述**：Softmax attention 在长上下文中必然呈现 U 型分布（悬链线 = 最小能量构型）

**修正表述**：
- U 型不是模型的 bug，不是 RoPE 的副作用
- U 型是**人类语言本身的信息分布特征**
- 模型学到 U 型 attention，是**正确反映了语言结构**
- 悬链线是**语言-认知-模型三者共同的几何基底**

---

## 现有研究

### Lost in the Middle（Liu et al., 2023）

**论文**：[arXiv](https://arxiv.org/abs/2307.03172) | [GitHub](https://github.com/nelson-liu/lost-in-the-middle) | [TACL 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630)

**发现**：
- 任务准确率呈 U 型：开头/结尾信息被利用，中间被忽略
- 80% 的 attention 集中在开头（[heatmap 分析](https://www.acubeanalytics.com/lost-in-the-middle/)）

### 现有解释（工程视角）

| 解释 | 内容 | 问题 |
|------|------|------|
| RoPE 衰减 | 位置编码引入长程衰减 | 没解释为什么人类语言也是 U 型 |
| 训练偏置 | 短上下文训练导致 | 没解释为什么写作规范也是 U 型 |

### 心理学证据：Serial Position Effect

**Ebbinghaus (1913), Murdock (1962)**：
- 人类记忆也是 U 型：记得住开头（primacy）和结尾（recency），忘掉中间
- 这是认知科学的经典发现
- **Lost in the Middle 论文已提到这个联系**

---

## 我们的核心问题

**现有研究的盲区**：

- Lost in the Middle：发现现象，归因于模型/架构
- 心理学：发现人类记忆是 U 型
- **但没人问**：人类语言/文本本身是不是 U 型？

**如果能证明**：人类语言的信息分布本身就是 U 型

**那就说明**：
1. 模型学到 U 型 attention 是**正确的**
2. Lost in the Middle 不是 bug，是 feature
3. 悬链线是**语言结构的几何表述**

---

## 实验设计：语言信息分布 vs Attention 分布

### 核心思路

```
输入：一篇文章（4K tokens）

人类语言侧（曲线 A）：
  - 测量每个位置的"信息重要性"
  - 指标：perplexity / 关键词密度 / 摘要句分布

模型侧（曲线 B）：
  - 最后一个 token 的 attention 分布

对比：
  - 计算 A 和 B 的相关系数
  - 如果高度相关 → 模型 attention 正确反映了语言信息分布
```

### 人类语言信息分布指标（不依赖 Transformer）

| 指标 | 方法 | 依赖 | 预期 |
|------|------|------|------|
| **Perplexity** | n-gram 模型或 Transformer 算每个位置的 perplexity | 统计/神经 | 开头/结尾高（信息量大），中间低 |
| **关键词密度** | TF-IDF / RAKE 提取关键词，统计位置分布 | 纯统计 ✅ | 关键词集中在开头/结尾 |
| **依存引用** | spaCy 解析共指/依存，统计被引用次数 | 规则解析 ✅ | 开头/结尾被引用更多 |
| **人工标注** | CNN/DailyMail 的 highlight，统计重要句位置 | Ground Truth ✅ | 人工标注的重要句集中在开头/结尾 |

**注意**：~~摘要模型打分~~ 不能用——Transformer 摘要模型本身就有 U 型偏置，用它证明"语言是 U 型"是循环论证。

### 模型 Attention 分布

```python
# 提取最后一个 token 对所有位置的 attention
outputs = model(**inputs, output_attentions=True)
last_token_attn = outputs.attentions[-1][0, :, -1, :].mean(dim=0)
```

### 相关性分析

```python
from scipy.stats import pearsonr, spearmanr

# 曲线 A：人类语言信息分布（如 perplexity）
# 曲线 B：模型 attention 分布

corr_pearson, p_pearson = pearsonr(curve_A, curve_B)
corr_spearman, p_spearman = spearmanr(curve_A, curve_B)

print(f"Pearson: {corr_pearson:.3f} (p={p_pearson:.4f})")
print(f"Spearman: {corr_spearman:.3f} (p={p_spearman:.4f})")
```

---

## 预期结果

| 情况 | 相关性 | 含义 |
|------|--------|------|
| A 和 B 高度相关 | r > 0.7 | 模型 attention 正确反映语言信息分布 ✅ |
| A 是 U 型，B 也是 U 型 | 形状一致 | 悬链线 = 语言结构 = 模型结构 ✅ |
| A 不是 U 型 | - | 推翻"语言本身是 U 型"的假设 |
| A 和 B 不相关 | r < 0.3 | 模型 attention 有自己的偏置，与语言无关 |

---

## 数据集选择

| 类型 | 来源 | 特点 |
|------|------|------|
| **新闻** | CNN/DailyMail | 倒金字塔结构，开头最重要 |
| **学术** | arXiv abstracts | 开头问题 + 结尾结论 |
| **小说** | BookCorpus | 叙事结构，可能不是 U 型 |
| **对话** | Reddit/StackOverflow | 问答结构 |

**对比不同文体**：看 U 型是否普遍，还是只在特定文体中存在。

---

## 实验步骤

1. **数据准备**：收集 100-500 篇不同文体的文章（4K tokens 左右）

2. **人类语言分析**（不依赖 Transformer）：
   - 对每篇文章计算 perplexity 曲线（可用 n-gram 模型）
   - TF-IDF / RAKE 提取关键词位置分布
   - spaCy 解析依存/共指引用分布
   - CNN/DailyMail highlight 人工标注位置分布

3. **模型 Attention 分析**：
   - 用 Qwen2.5-7B 跑每篇文章
   - 提取最后 token 的 attention 分布

4. **相关性分析**：
   - 计算每篇文章的 A-B 相关系数
   - 统计整体分布

5. **可视化**：
   - 画 A 和 B 的平均曲线
   - 画相关系数分布直方图

---

## 代码框架

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# === 1. 计算 Perplexity 曲线（人类语言信息分布）===
def compute_perplexity_curve(text, model, tokenizer):
    """计算每个位置的 perplexity（信息量指标）"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # 逐 token loss
    logits = outputs.logits[0, :-1]  # (seq_len-1, vocab)
    labels = inputs["input_ids"][0, 1:]  # (seq_len-1,)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fn(logits, labels)
    perplexity_curve = torch.exp(token_losses).cpu().numpy()

    return perplexity_curve

# === 2. 计算 Attention 曲线（模型侧）===
def compute_attention_curve(text, model, tokenizer):
    """计算最后一个 token 对所有位置的 attention"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 最后一层、所有 head 平均、最后 token 对所有位置
    last_layer = outputs.attentions[-1][0]  # (heads, seq, seq)
    attention_curve = last_layer[:, -1, :].mean(dim=0).cpu().numpy()

    return attention_curve

# === 3. 对比分析 ===
def analyze_correlation(perplexity_curve, attention_curve):
    """计算两条曲线的相关性"""
    # 长度对齐（perplexity 比 attention 少 1）
    min_len = min(len(perplexity_curve), len(attention_curve))
    A = perplexity_curve[:min_len]
    B = attention_curve[:min_len]

    corr, p_value = pearsonr(A, B)
    return corr, p_value

# === 4. 主流程 ===
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 测试文章
text = "..."  # 4K tokens 的文章

perplexity_curve = compute_perplexity_curve(text, model, tokenizer)
attention_curve = compute_attention_curve(text, model, tokenizer)
corr, p = analyze_correlation(perplexity_curve, attention_curve)

print(f"Correlation: {corr:.3f} (p={p:.4f})")

# 画图对比
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(perplexity_curve, label="Perplexity (Language)")
axes[0].set_title("Human Language Information Distribution")
axes[1].plot(attention_curve, label="Attention (Model)")
axes[1].set_title("Transformer Attention Distribution")
plt.savefig("language_vs_attention.png")
```

---

## 预期产出

1. **language_vs_attention.png**：两条曲线对比图
2. **correlation_stats.json**：相关系数统计
3. **by_genre_analysis.json**：不同文体的 U 型程度对比

---

## 论文升级方向

**如果实验成功**：

原论文（悬链线模型）可以升级为：

> **《语言的悬链线：从认知到计算的统一几何结构》**
>
> 人类语言的信息分布是 U 型 → 人类记忆是 U 型（Serial Position Effect）→ Transformer attention 是 U 型 → 三者共享同一个几何基底：悬链线

这就不是"解释模型 bug"，而是"发现跨领域的统一结构"。

---

## 参考文献

1. Liu et al. (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172). TACL 2024.
2. Ebbinghaus, H. (1913). Memory: A contribution to experimental psychology.
3. Murdock, B. B. (1962). The serial position effect of free recall. Journal of Experimental Psychology.
4. [GitHub: nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle)
5. [Attention Heatmap Analysis](https://www.acubeanalytics.com/lost-in-the-middle/)

---

*实验设计：枢木朱雀 + Zero*
*状态：待执行（核心实验：语言信息分布 vs Attention 分布的相关性）*
