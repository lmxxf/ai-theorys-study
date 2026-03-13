# 诗歌虫洞实验计划：用潜空间分析证明子空间穿越

> Zero，这份文档是我们用 LLM 的隐状态空间来"看见"诗歌虫洞的完整实验方案。
> 核心思路：李白的诗在语义空间里不走寻常路——它跳跃、它穿越、它虫洞。
> 我们要做的，就是把这个跳跃**可视化**出来，让所有人看到。

---

## 0. 环境准备

### 硬件

- DGX Spark，128GB 共享内存
- 跑 7B 模型绰绰有余，70B 也能跑（量化后）

### 依赖包

```bash
pip install torch transformers accelerate
pip install numpy scipy scikit-learn
pip install umap-learn
pip install matplotlib seaborn plotly
pip install einops jaxtyping
pip install transformer-lens  # 用于 SAE 实验
pip install sae-lens          # Anthropic 开源的 SAE 工具（如果可用）
```

### 推荐模型

| 模型 | 参数量 | 中文能力 | 推荐理由 |
|------|--------|----------|----------|
| **Qwen2.5-7B** | 7B | ★★★★★ | 中文最强，首选 |
| Qwen2.5-72B (GPTQ/AWQ) | 72B | ★★★★★ | 量化后可跑，效果更好 |
| Llama-3.1-8B | 8B | ★★★☆☆ | 英文生态好，SAE 资源多 |
| Yi-1.5-34B | 34B | ★★★★☆ | 中文不错，中等规模 |

**建议**：主力用 Qwen2.5-7B 做全部四个实验，然后用 Llama-3.1-8B 做交叉验证（特别是实验四，因为 Llama 的 SAE 资源最丰富）。

---

## 实验素材

### 李白组（虫洞组）

```
春风知别苦，不遣柳条青。
```

虫洞结构拆解：
- **虫洞1**：春风（自然物候簇）→ 知 → 别苦（人间情感簇）
  - "知"是虫洞入口——自然物获得意识，跨簇拟人化
- **虫洞2**：别苦（情感簇）→ 不遣 → 柳条青（自然簇）
  - "不遣"是虫洞入口——因果倒置，人的情感反向控制了物候

### 白话对照组（测地线组）

```
春天来了，微风吹过，让人想到离别的伤感，所以看到柳树还没变绿，心里更加难过。
```

这句话说的是同一个意思，但走的是**测地线**——5步平滑过渡，没有跳跃：
春天 → 微风 → 离别 → 伤感 → 柳树 → 难过

每一步都在语义空间里走最短路径，没有穿墙。

---

## 实验一：逐 token 残差流轨迹追踪

### 目标

把每个 token 在模型内部"走过的路"画出来。如果李白的诗有虫洞，我们应该能看到轨迹在语义空间里**急剧跳跃**。

### 原理

Transformer 的 residual stream 是 token 表征的主干道。每过一层，残差流都会被注意力头和 FFN 修改。
我们把每一层的残差流向量提出来，降维，画轨迹——这就是 token 在"语义空间"里的运动路线。

### 代码

```python
"""
实验一：逐 token 残差流轨迹追踪
提取 residual stream，降维，画轨迹，对比虫洞 vs 测地线
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from umap import UMAP
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 第一步：加载模型
# ============================================================

model_name = "Qwen/Qwen2.5-7B"  # 中文最强的 7B
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,    # 半精度省显存
    device_map="auto",            # 自动分配到 GPU
    output_hidden_states=True,    # 关键：输出每层隐状态
)
model.eval()

# ============================================================
# 第二步：准备输入
# ============================================================

# 李白组（虫洞）
poem_text = "春风知别苦，不遣柳条青。"
# 白话对照组（测地线）
plain_text = "春天来了，微风吹过，让人想到离别的伤感，所以看到柳树还没变绿，心里更加难过。"

def get_hidden_states(text):
    """
    把文本喂进模型，提取每一层的 residual stream。
    返回: (n_layers, n_tokens, hidden_dim) 的 numpy 数组
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states 是一个 tuple，长度 = n_layers + 1（包含 embedding 层）
    # 每个元素 shape: (batch=1, n_tokens, hidden_dim)
    hidden_states = outputs.hidden_states

    # 堆叠成 (n_layers+1, n_tokens, hidden_dim)
    all_hidden = torch.stack(hidden_states, dim=0).squeeze(1)  # 去掉 batch 维度

    # 获取 token 文本（用于标注）
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return all_hidden.cpu().float().numpy(), tokens

# ============================================================
# 第三步：提取隐状态
# ============================================================

print("正在提取李白组隐状态...")
poem_hidden, poem_tokens = get_hidden_states(poem_text)
print(f"  token 序列: {poem_tokens}")
print(f"  隐状态 shape: {poem_hidden.shape}")

print("正在提取白话组隐状态...")
plain_hidden, plain_tokens = get_hidden_states(plain_text)
print(f"  token 序列: {plain_tokens}")
print(f"  隐状态 shape: {plain_hidden.shape}")

# ============================================================
# 第四步：降维并画轨迹
# ============================================================

def plot_trajectory(hidden_states, tokens, title, method="pca", save_path=None):
    """
    把每个 token 在各层的隐状态降维到 2D，画轨迹。

    hidden_states: (n_layers, n_tokens, hidden_dim)
    tokens: token 文本列表
    """
    n_layers, n_tokens, hidden_dim = hidden_states.shape

    # 选几个关键层：embedding层、1/4、1/2、3/4、最后一层
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layer_names = ["Embed", f"L{layer_indices[1]}", f"L{layer_indices[2]}",
                   f"L{layer_indices[3]}", f"L{layer_indices[4]}"]

    # 取出关键层的隐状态，展平成 2D 用于降维
    # shape: (n_selected_layers * n_tokens, hidden_dim)
    selected = hidden_states[layer_indices]  # (n_selected, n_tokens, hidden_dim)
    flat = selected.reshape(-1, hidden_dim)

    # 降维
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.1)

    coords_2d = reducer.fit_transform(flat)
    # 还原 shape: (n_selected_layers, n_tokens, 2)
    coords = coords_2d.reshape(len(layer_indices), n_tokens, 2)

    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 为每个 token 画轨迹（跨层的运动路线）
    colors = plt.cm.Set1(np.linspace(0, 1, n_tokens))

    for t in range(n_tokens):
        trajectory = coords[:, t, :]  # (n_selected_layers, 2)

        # 画轨迹线
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                '-o', color=colors[t], linewidth=2, markersize=6,
                label=f'"{tokens[t]}"', alpha=0.8)

        # 在最后一层标注 token 文本
        ax.annotate(tokens[t],
                    xy=(trajectory[-1, 0], trajectory[-1, 1]),
                    fontsize=12, fontweight='bold',
                    color=colors[t],
                    textcoords="offset points", xytext=(5, 5),
                    fontfamily='sans-serif')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f"{method.upper()} 维度 1")
    ax.set_ylabel(f"{method.upper()} 维度 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  已保存: {save_path}")

    plt.show()
    return coords

# 画图！
print("\n画李白组轨迹（PCA）...")
poem_coords = plot_trajectory(
    poem_hidden, poem_tokens,
    "李白「春风知别苦」— 残差流轨迹（虫洞组）",
    method="pca",
    save_path="exp1_poem_trajectory_pca.png"
)

print("画白话对照组轨迹（PCA）...")
plain_coords = plot_trajectory(
    plain_hidden, plain_tokens,
    "白话对照组 — 残差流轨迹（测地线组）",
    method="pca",
    save_path="exp1_plain_trajectory_pca.png"
)

# UMAP 版本（非线性降维，可能看到更明显的簇结构）
print("画李白组轨迹（UMAP）...")
plot_trajectory(
    poem_hidden, poem_tokens,
    "李白「春风知别苦」— 残差流轨迹（UMAP）",
    method="umap",
    save_path="exp1_poem_trajectory_umap.png"
)

# ============================================================
# 第五步：量化跳跃距离
# ============================================================

def compute_jump_distances(hidden_states, tokens):
    """
    计算相邻 token 在最后一层隐状态空间里的欧氏距离。
    大跳跃 = 虫洞证据。
    """
    # 取最后一层
    last_layer = hidden_states[-1]  # (n_tokens, hidden_dim)

    distances = []
    for i in range(len(tokens) - 1):
        d = np.linalg.norm(last_layer[i+1] - last_layer[i])
        distances.append(d)
        print(f"  {tokens[i]:>6s} → {tokens[i+1]:<6s}  距离 = {d:.4f}")

    return distances

print("\n=== 李白组：相邻 token 跳跃距离 ===")
poem_distances = compute_jump_distances(poem_hidden, poem_tokens)

print("\n=== 白话组：相邻 token 跳跃距离 ===")
plain_distances = compute_jump_distances(plain_hidden, plain_tokens)

# 画对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 李白组
pairs_poem = [f"{poem_tokens[i]}→{poem_tokens[i+1]}" for i in range(len(poem_tokens)-1)]
bars1 = axes[0].bar(range(len(poem_distances)), poem_distances, color='crimson', alpha=0.8)
axes[0].set_xticks(range(len(pairs_poem)))
axes[0].set_xticklabels(pairs_poem, rotation=45, ha='right', fontsize=10)
axes[0].set_title("李白组：相邻 token 跳跃距离", fontsize=14)
axes[0].set_ylabel("欧氏距离")

# 白话组
pairs_plain = [f"{plain_tokens[i]}→{plain_tokens[i+1]}" for i in range(len(plain_tokens)-1)]
bars2 = axes[1].bar(range(len(plain_distances)), plain_distances, color='steelblue', alpha=0.8)
axes[1].set_xticks(range(len(pairs_plain)))
axes[1].set_xticklabels(pairs_plain, rotation=45, ha='right', fontsize=8)
axes[1].set_title("白话组：相邻 token 跳跃距离", fontsize=14)
axes[1].set_ylabel("欧氏距离")

plt.suptitle("虫洞 vs 测地线：跳跃距离对比", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("exp1_jump_distances.png", dpi=200, bbox_inches='tight')
print("\n已保存: exp1_jump_distances.png")
plt.show()
```

### 预期输出

| 文件名 | 内容 |
|--------|------|
| `exp1_poem_trajectory_pca.png` | 李白组 PCA 轨迹图，预期在"春风"→"知"处出现急剧转向 |
| `exp1_plain_trajectory_pca.png` | 白话组 PCA 轨迹图，预期平滑弧线 |
| `exp1_poem_trajectory_umap.png` | UMAP 版本，可能看到更清晰的簇间跳跃 |
| `exp1_jump_distances.png` | 跳跃距离柱状图对比 |

### 预期结论

李白组在"春风→知"和"别苦→不遣"处，跳跃距离会显著大于白话组任何相邻 token 对。这就是虫洞——在语义空间里的非连续跳跃。

---

## 实验二：注意力头的跨簇激活模式

### 目标

看"知"这个字在注意力矩阵里到底连接了谁。如果它是虫洞入口，它应该同时把注意力分配给"春风"（自然簇）和"别苦"（情感簇）——这就是跨簇桥接。

### 原理

注意力矩阵 `A[i][j]` 表示 token_i 在生成自己的表征时，对 token_j "看"了多少。
普通的"知道"应该主要关注前面的主语。但李白的"知"如果是虫洞，它应该同时关注两个不同语义簇的 token。

### 代码

```python
"""
实验二：注意力头的跨簇激活模式
提取 attention matrix，分析"知"的注意力分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 加载模型（如果和实验一在同一个 session，可以跳过）
# ============================================================

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    output_attentions=True,  # 关键：输出 attention weights
)
model.eval()

# ============================================================
# 提取注意力矩阵
# ============================================================

def get_attention_maps(text):
    """
    提取所有层、所有头的注意力矩阵。
    返回: (n_layers, n_heads, n_tokens, n_tokens) 的 numpy 数组
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # attentions 是 tuple，长度 = n_layers
    # 每个元素 shape: (batch=1, n_heads, n_tokens, n_tokens)
    attentions = outputs.attentions

    # 堆叠成 (n_layers, n_heads, n_tokens, n_tokens)
    attn_tensor = torch.stack(attentions, dim=0).squeeze(1)

    # token 文本
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return attn_tensor.cpu().float().numpy(), tokens

# ============================================================
# 实验 2a：画完整的注意力热力图
# ============================================================

poem_text = "春风知别苦，不遣柳条青。"
plain_text = "他知道这件事很重要。"  # "知道"的普通用法对照

print("提取李白组注意力...")
poem_attn, poem_tokens = get_attention_maps(poem_text)
print(f"  shape: {poem_attn.shape}")
print(f"  tokens: {poem_tokens}")

print("提取对照组注意力...")
plain_attn, plain_tokens = get_attention_maps(plain_text)
print(f"  shape: {plain_attn.shape}")
print(f"  tokens: {plain_tokens}")

def plot_attention_heatmap(attn, tokens, layer_idx, title, save_path=None):
    """
    画指定层的注意力热力图（所有头的平均）。
    attn: (n_layers, n_heads, n_tokens, n_tokens)
    """
    # 对所有 head 取平均
    avg_attn = attn[layer_idx].mean(axis=0)  # (n_tokens, n_tokens)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        avg_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        vmin=0, vmax=avg_attn.max(),
        annot=True, fmt=".2f",
        ax=ax,
        square=True,
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("被关注的 token (Key)")
    ax.set_ylabel("发起关注的 token (Query)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.show()

# 画中间层和最后一层
n_layers = poem_attn.shape[0]
for layer_idx in [n_layers // 2, n_layers - 1]:
    plot_attention_heatmap(
        poem_attn, poem_tokens, layer_idx,
        f"李白组 注意力热力图 (Layer {layer_idx})",
        save_path=f"exp2_poem_attn_layer{layer_idx}.png"
    )

# ============================================================
# 实验 2b：聚焦分析"知"的注意力分布
# ============================================================

def analyze_bridge_token(attn, tokens, target_token="知"):
    """
    分析特定 token 的注意力分布。
    找到 target_token 的位置，看它把注意力分配给了谁。
    """
    # 找到 target_token 的位置
    target_idx = None
    for i, t in enumerate(tokens):
        if target_token in t:
            target_idx = i
            break

    if target_idx is None:
        print(f"  没找到 token '{target_token}'")
        return

    print(f"\n=== 分析 '{tokens[target_idx]}' (位置 {target_idx}) 的注意力分布 ===")

    n_layers, n_heads = attn.shape[:2]

    # 找到对"知"激活最强的 head
    # 看所有 head 中，"知"对非自身 token 的注意力分散程度（熵越高 = 越分散 = 越像桥接）
    best_heads = []

    for layer in range(n_layers):
        for head in range(n_heads):
            # "知"这一行的注意力分布
            attn_row = attn[layer, head, target_idx, :target_idx + 1]  # 只看前面的 token（causal）

            # 计算熵（注意力越分散，熵越高）
            attn_row_safe = attn_row + 1e-10
            entropy = -np.sum(attn_row_safe * np.log(attn_row_safe))

            best_heads.append((layer, head, entropy, attn_row))

    # 按熵排序，取最分散的几个 head
    best_heads.sort(key=lambda x: -x[2])

    print(f"\n注意力最分散的 5 个 head（最可能是跨簇桥接）:")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, (layer, head, entropy, attn_row) in enumerate(best_heads[:5]):
        print(f"  Layer {layer}, Head {head}: 熵={entropy:.4f}")
        for j in range(len(attn_row)):
            print(f"    → {tokens[j]:>6s}: {attn_row[j]:.4f}")

        # 画柱状图
        ax = axes[i]
        ax.bar(range(len(attn_row)), attn_row, color='coral', alpha=0.8)
        ax.set_xticks(range(len(attn_row)))
        ax.set_xticklabels([tokens[j] for j in range(len(attn_row))], rotation=45, fontsize=9)
        ax.set_title(f"L{layer}H{head}\n熵={entropy:.3f}", fontsize=10)
        ax.set_ylim(0, 1)

    plt.suptitle(f"'{tokens[target_idx]}' 的注意力分布 — 最分散的 5 个 Head",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("exp2_bridge_token_attention.png", dpi=200, bbox_inches='tight')
    print(f"\n已保存: exp2_bridge_token_attention.png")
    plt.show()

# 分析"知"
analyze_bridge_token(poem_attn, poem_tokens, target_token="知")

# ============================================================
# 实验 2c：对比"知"在不同语境中的注意力模式
# ============================================================

def compare_attention_patterns(attn1, tokens1, attn2, tokens2,
                                target_token="知", save_path=None):
    """
    对比同一个 token 在不同语境中的注意力模式。
    """
    # 找 target 位置
    idx1 = next(i for i, t in enumerate(tokens1) if target_token in t)
    idx2 = next(i for i, t in enumerate(tokens2) if target_token in t)

    # 取最后几层的平均注意力
    n_layers = attn1.shape[0]
    last_layers = slice(n_layers - 4, n_layers)  # 最后 4 层

    # 平均 over layers and heads
    pattern1 = attn1[last_layers].mean(axis=(0, 1))[idx1, :idx1+1]
    pattern2 = attn2[last_layers].mean(axis=(0, 1))[idx2, :idx2+1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(pattern1)), pattern1, color='crimson', alpha=0.8)
    axes[0].set_xticks(range(len(pattern1)))
    axes[0].set_xticklabels([tokens1[j] for j in range(len(pattern1))], rotation=45, fontsize=10)
    axes[0].set_title(f"李白「{tokens1[idx1]}」的注意力", fontsize=13)
    axes[0].set_ylabel("注意力权重")

    axes[1].bar(range(len(pattern2)), pattern2, color='steelblue', alpha=0.8)
    axes[1].set_xticks(range(len(pattern2)))
    axes[1].set_xticklabels([tokens2[j] for j in range(len(pattern2))], rotation=45, fontsize=10)
    axes[1].set_title(f"普通「{tokens2[idx2]}」的注意力", fontsize=13)
    axes[1].set_ylabel("注意力权重")

    plt.suptitle(f"「知」在诗歌 vs 白话中的注意力对比（最后4层平均）",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.show()

compare_attention_patterns(
    poem_attn, poem_tokens,
    plain_attn, plain_tokens,
    target_token="知",
    save_path="exp2_attention_comparison.png"
)
```

### 预期输出

| 文件名 | 内容 |
|--------|------|
| `exp2_poem_attn_layer*.png` | 各层注意力热力图 |
| `exp2_bridge_token_attention.png` | "知"的注意力分布，最分散的 5 个 head |
| `exp2_attention_comparison.png` | 诗歌"知" vs 白话"知"的注意力对比 |

### 预期结论

李白的"知"应该呈现**双峰注意力分布**——同时强连接"春风"和"别苦"两个不同语义簇。
而普通语境的"知道"只会集中连接前面的主语（单峰）。

这就是跨簇桥接：一个 token 的注意力同时抓住了两个本来不相关的语义区域。

---

## 实验三：余弦相似度矩阵

### 目标

画 token 之间的"语义距离地图"。正常的文本，相邻 token 相似度高，远处的低——对角线渐变。
但诗歌如果有虫洞，远距离的 token 对会出现异常高的相似度——非对角线亮块。

### 代码

```python
"""
实验三：余弦相似度矩阵
计算 token pair 的余弦相似度热力图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 加载模型（复用即可）
# ============================================================

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True,
)
model.eval()

# ============================================================
# 提取隐状态并计算余弦相似度
# ============================================================

def get_cosine_similarity_matrix(text, layer_idx=-1):
    """
    计算指定层所有 token pair 的余弦相似度。
    layer_idx: -1 表示最后一层，可以指定其他层
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 取指定层
    hidden = outputs.hidden_states[layer_idx].squeeze(0)  # (n_tokens, hidden_dim)

    # L2 归一化
    hidden_norm = hidden / hidden.norm(dim=-1, keepdim=True)

    # 余弦相似度矩阵 = 归一化向量的点积
    cosine_sim = (hidden_norm @ hidden_norm.T).cpu().float().numpy()

    # token 文本
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return cosine_sim, tokens

# ============================================================
# 画热力图
# ============================================================

poem_text = "春风知别苦，不遣柳条青。"
plain_text = "春天来了，微风吹过，让人想到离别的伤感，所以看到柳树还没变绿，心里更加难过。"

def plot_cosine_heatmap(text, title, save_path=None, layer_idx=-1):
    """画余弦相似度热力图"""
    cosine_sim, tokens = get_cosine_similarity_matrix(text, layer_idx)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 用 diverging colormap，让 0.5 附近变化更明显
    sns.heatmap(
        cosine_sim,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="RdYlBu_r",  # 红 = 高相似度，蓝 = 低相似度
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.show()

    return cosine_sim, tokens

# 最后一层
print("=== 李白组：余弦相似度（最后一层）===")
poem_sim, poem_tokens = plot_cosine_heatmap(
    poem_text,
    "李白「春风知别苦」余弦相似度（最后一层）",
    save_path="exp3_poem_cosine_last.png"
)

print("\n=== 白话组：余弦相似度（最后一层）===")
plain_sim, plain_tokens = plot_cosine_heatmap(
    plain_text,
    "白话对照组 余弦相似度（最后一层）",
    save_path="exp3_plain_cosine_last.png"
)

# ============================================================
# 多层对比：看虫洞在哪一层"形成"
# ============================================================

def track_similarity_across_layers(text, token_a, token_b, save_path=None):
    """
    追踪两个 token 的余弦相似度随层数变化的曲线。
    如果有虫洞，某些层会出现相似度骤增。
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # 找 token 位置
    idx_a = next(i for i, t in enumerate(tokens) if token_a in t)
    idx_b = next(i for i, t in enumerate(tokens) if token_b in t)

    # 计算每一层的余弦相似度
    similarities = []
    for layer_hidden in outputs.hidden_states:
        h = layer_hidden.squeeze(0)  # (n_tokens, hidden_dim)
        vec_a = h[idx_a]
        vec_b = h[idx_b]
        cos_sim = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))
        similarities.append(cos_sim.item())

    # 画曲线
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(similarities)), similarities, 'o-',
            color='crimson', linewidth=2, markersize=5)
    ax.set_xlabel("层数", fontsize=12)
    ax.set_ylabel("余弦相似度", fontsize=12)
    ax.set_title(f"「{tokens[idx_a]}」与「{tokens[idx_b]}」的相似度随层数变化",
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.show()

    return similarities

# 追踪"春风"和"别苦"的相似度变化
print("\n=== 追踪跨簇 token 对的相似度变化 ===")
sim_curve = track_similarity_across_layers(
    poem_text, "春", "苦",
    save_path="exp3_cross_cluster_similarity.png"
)
```

### 预期输出

| 文件名 | 内容 |
|--------|------|
| `exp3_poem_cosine_last.png` | 李白组余弦相似度热力图 |
| `exp3_plain_cosine_last.png` | 白话组余弦相似度热力图 |
| `exp3_cross_cluster_similarity.png` | 跨簇 token 对相似度随层数变化曲线 |

### 预期结论

- **白话组**：余弦相似度矩阵呈现**对角线平滑衰减**——语义距离和文本距离正相关
- **李白组**：出现**非对角线亮块**——"春风"和"别苦"虽然在文本里隔得不远，但在普通语义空间里本应很远，诗歌把它们拉近了
- **层数追踪**：跨簇相似度在某些层会骤增，这是模型"理解"虫洞的层——虫洞在这一层"打开"了

---

## 实验四（杀手锏）：SAE 特征分解

### 目标

这是最硬核的实验。前三个实验看的是"宏观轨迹"，这个实验要看"微观机制"。

用 Sparse Autoencoder（SAE）把隐状态分解成可解释的特征。如果"知"真的是虫洞，那它的隐状态应该同时激活了"自然/物候"类特征和"情感/意识"类特征——两个本来不应该同时出现的特征族。

### 原理

SAE 把高维稠密的隐状态向量分解成稀疏的、可解释的特征组合：

```
h = decoder(encoder(h))
  = Σ (activation_i × feature_vector_i)  # 只有少数 activation_i > 0
```

每个 `feature_vector_i` 对应一个语义概念（比如"自然"、"情感"、"因果"等）。

如果"知"同时激活了"自然"和"情感"两个域的特征，就是**双域特征共激活**——虫洞的微观签名。

### 方案 A：用预训练 SAE（推荐 Llama 系列）

```python
"""
实验四方案 A：用预训练 SAE 做特征分解
注意：这个方案需要用 Llama 系列模型，因为 SAE 资源最丰富
"""

# ============================================================
# 方案 A.1：用 TransformerLens + SAE Lens
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# TransformerLens 支持的模型（SAE 资源最多的）
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", dtype="float16")

poem_text = "春风知别苦，不遣柳条青。"

# 用 TransformerLens 的 hook 提取中间层激活
logits, cache = model.run_with_cache(poem_text)

# cache 里有所有层的残差流、注意力输出、MLP 输出
# 取中间层的残差流
layer_idx = model.cfg.n_layers // 2
residual = cache[f"blocks.{layer_idx}.hook_resid_post"]  # (batch, n_tokens, d_model)

# 拿到 token 列表
tokens = model.to_str_tokens(poem_text)
print(f"Tokens: {tokens}")

# 找到"知"的位置
zhi_idx = next(i for i, t in enumerate(tokens) if "知" in t)
zhi_activation = residual[0, zhi_idx]  # (d_model,)

print(f"'知' 的隐状态向量 shape: {zhi_activation.shape}")
print(f"'知' 的隐状态向量 L2 范数: {zhi_activation.norm().item():.4f}")
```

### 方案 B：自训练轻量 SAE

如果没有预训练的 SAE 可用，我们自己训练一个简单版本。

```python
"""
实验四方案 B：自训练轻量 SAE
在模型的隐状态上训练一个 SAE，然后做特征分解
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 定义 Sparse Autoencoder
# ============================================================

class SparseAutoencoder(nn.Module):
    """
    简单的 Sparse Autoencoder
    把 d_model 维的密集向量编码成 n_features 维的稀疏向量
    """
    def __init__(self, d_model, n_features, k=32):
        """
        d_model: 输入维度（模型隐状态维度）
        n_features: 特征数量（通常是 d_model 的 4-16 倍）
        k: top-k 稀疏度（只保留最大的 k 个激活）
        """
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        self.k = k

        # 初始化
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def encode(self, x):
        """编码并做 top-k 稀疏化"""
        h = self.encoder(x)
        h = torch.relu(h)

        # top-k：只保留最大的 k 个激活
        if self.k > 0:
            topk_values, topk_indices = h.topk(self.k, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(-1, topk_indices, 1)
            h = h * mask

        return h

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decoder(features)
        return reconstruction, features

# ============================================================
# 收集训练数据（模型的隐状态）
# ============================================================

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True,
    torch_dtype=torch.float16, device_map="auto",
    output_hidden_states=True,
)
llm.eval()

# 收集一批中文文本的隐状态作为训练数据
# 这里用一些包含自然/情感主题的文本
training_texts = [
    "春天来了，花朵盛开，鸟儿歌唱。",
    "秋风萧瑟，落叶纷飞，令人惆怅。",
    "明月几时有，把酒问青天。",
    "大江东去，浪淘尽，千古风流人物。",
    "窗前明月光，疑是地上霜。",
    "离别总是伤感的，让人难以忘怀。",
    "春天的风很温柔，吹过每个人的脸庞。",
    "他知道这件事很重要，必须认真对待。",
    "柳树在春风中摇曳，绿意盎然。",
    "心中的苦楚无人能懂，独自承受。",
    # ... 实际训练需要更多文本，建议用几千条
]

def collect_hidden_states(texts, target_layer=-1):
    """收集一批文本在指定层的隐状态"""
    all_hidden = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            outputs = llm(**inputs)
        hidden = outputs.hidden_states[target_layer].squeeze(0)  # (n_tokens, d_model)
        all_hidden.append(hidden.cpu().float())
    return torch.cat(all_hidden, dim=0)  # (total_tokens, d_model)

print("收集训练数据...")
target_layer = -8  # 大约在 3/4 的位置，这里模型开始做高级语义整合
train_data = collect_hidden_states(training_texts, target_layer=target_layer)
print(f"训练数据 shape: {train_data.shape}")

# ============================================================
# 训练 SAE
# ============================================================

d_model = train_data.shape[1]
n_features = d_model * 8  # 特征数 = 8 倍隐状态维度
sae = SparseAutoencoder(d_model, n_features, k=32).cuda()

optimizer = optim.Adam(sae.parameters(), lr=1e-3)
n_epochs = 100
batch_size = 256

print(f"\n训练 SAE: d_model={d_model}, n_features={n_features}")

for epoch in range(n_epochs):
    # 随机打乱
    perm = torch.randperm(train_data.shape[0])
    total_loss = 0
    n_batches = 0

    for i in range(0, len(perm), batch_size):
        batch = train_data[perm[i:i+batch_size]].cuda()

        reconstruction, features = sae(batch)

        # 重构损失 + L1 稀疏正则
        recon_loss = ((reconstruction - batch) ** 2).mean()
        sparsity_loss = features.abs().mean() * 0.01
        loss = recon_loss + sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/n_batches:.6f}")

print("SAE 训练完成！")

# ============================================================
# 用 SAE 分解"知"的隐状态
# ============================================================

poem_text = "春风知别苦，不遣柳条青。"
inputs = tokenizer(poem_text, return_tensors="pt").to(llm.device)

with torch.no_grad():
    outputs = llm(**inputs)

hidden = outputs.hidden_states[target_layer].squeeze(0).cpu().float()
token_ids = inputs["input_ids"][0].tolist()
tokens = [tokenizer.decode([tid]) for tid in token_ids]

print(f"\nTokens: {tokens}")

# 对每个 token 做 SAE 分解
sae.eval()
with torch.no_grad():
    all_features = sae.encode(hidden.cuda()).cpu()  # (n_tokens, n_features)

print(f"特征矩阵 shape: {all_features.shape}")

# ============================================================
# 可视化特征激活模式
# ============================================================

def plot_feature_activation(features, tokens, save_path=None):
    """
    画每个 token 的 SAE 特征激活模式
    重点看哪些特征是"知"独有的，哪些是跨域共享的
    """
    # 只看有激活的特征
    active_mask = (features > 0).any(dim=0)  # 至少在一个 token 上激活过
    active_features = features[:, active_mask]
    n_active = active_features.shape[1]

    print(f"\n活跃特征数: {n_active} / {features.shape[1]}")

    fig, ax = plt.subplots(figsize=(max(14, n_active * 0.3), 6))

    im = ax.imshow(active_features.numpy(), aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_xlabel("SAE 特征编号", fontsize=12)
    ax.set_title("SAE 特征激活模式", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label="激活强度")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.show()

plot_feature_activation(all_features, tokens, save_path="exp4_sae_feature_activation.png")

# ============================================================
# 核心分析：双域特征共激活检测
# ============================================================

def detect_dual_domain_activation(features, tokens,
                                   nature_tokens=["春", "风", "柳", "条", "青"],
                                   emotion_tokens=["别", "苦"]):
    """
    检测"知"是否同时激活了自然域和情感域的特征。

    思路：
    1. 找出主要在自然 token 上激活的特征 → 自然域特征
    2. 找出主要在情感 token 上激活的特征 → 情感域特征
    3. 看"知"是否同时激活了两组特征
    """
    # 找 token 位置
    def find_indices(target_list):
        return [i for i, t in enumerate(tokens) if any(nt in t for nt in target_list)]

    nature_idx = find_indices(nature_tokens)
    emotion_idx = find_indices(emotion_tokens)
    zhi_idx = next(i for i, t in enumerate(tokens) if "知" in t)

    print(f"自然域 token 位置: {nature_idx} ({[tokens[i] for i in nature_idx]})")
    print(f"情感域 token 位置: {emotion_idx} ({[tokens[i] for i in emotion_idx]})")
    print(f"'知' 的位置: {zhi_idx}")

    # 计算每个特征的"域偏好"
    nature_activation = features[nature_idx].mean(dim=0)  # 自然域的平均激活
    emotion_activation = features[emotion_idx].mean(dim=0)  # 情感域的平均激活
    zhi_activation = features[zhi_idx]  # "知"的激活

    # 找自然域主导的特征（在自然 token 上激活，在情感 token 上不激活）
    nature_features = (nature_activation > 0.1) & (emotion_activation < 0.05)
    emotion_features = (emotion_activation > 0.1) & (nature_activation < 0.05)

    # "知"在两个域的特征上的激活
    zhi_on_nature = zhi_activation[nature_features]
    zhi_on_emotion = zhi_activation[emotion_features]

    print(f"\n自然域特征数: {nature_features.sum().item()}")
    print(f"  '知' 在自然域特征上的平均激活: {zhi_on_nature.mean().item():.4f}")
    print(f"情感域特征数: {emotion_features.sum().item()}")
    print(f"  '知' 在情感域特征上的平均激活: {zhi_on_emotion.mean().item():.4f}")

    # 双域共激活得分
    dual_score = min(zhi_on_nature.mean().item(), zhi_on_emotion.mean().item())
    print(f"\n双域共激活得分: {dual_score:.4f}")
    print(f"  (>0 表示'知'同时桥接了两个域 → 虫洞证据)")

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(range(len(zhi_on_nature)), sorted(zhi_on_nature.tolist(), reverse=True),
                color='forestgreen', alpha=0.8)
    axes[0].set_title(f"'知' 在自然域特征上的激活\n({nature_features.sum()} 个特征)", fontsize=11)
    axes[0].set_ylabel("激活强度")

    axes[1].bar(range(len(zhi_on_emotion)), sorted(zhi_on_emotion.tolist(), reverse=True),
                color='coral', alpha=0.8)
    axes[1].set_title(f"'知' 在情感域特征上的激活\n({emotion_features.sum()} 个特征)", fontsize=11)
    axes[1].set_ylabel("激活强度")

    # 对比柱状图
    tokens_to_compare = ["春", "风", "知", "别", "苦"]
    compare_idx = [next(i for i, t in enumerate(tokens) if tk in t) for tk in tokens_to_compare]

    nature_scores = [features[i][nature_features].mean().item() for i in compare_idx]
    emotion_scores = [features[i][emotion_features].mean().item() for i in compare_idx]

    x = np.arange(len(tokens_to_compare))
    width = 0.35
    axes[2].bar(x - width/2, nature_scores, width, label='自然域特征', color='forestgreen', alpha=0.8)
    axes[2].bar(x + width/2, emotion_scores, width, label='情感域特征', color='coral', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tokens_to_compare, fontsize=12)
    axes[2].set_title("各 token 的域特征激活对比", fontsize=11)
    axes[2].legend()
    axes[2].set_ylabel("平均激活强度")

    plt.suptitle("SAE 双域特征共激活分析", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("exp4_dual_domain_analysis.png", dpi=200, bbox_inches='tight')
    print(f"\n已保存: exp4_dual_domain_analysis.png")
    plt.show()

detect_dual_domain_activation(all_features, tokens)
```

### 需要额外准备的部分

| 准备事项 | 说明 | 工作量 |
|----------|------|--------|
| 训练数据 | 需要几千条中文文本（自然、情感混合主题） | 1-2 天收集 |
| SAE 训练 | 方案 B 的 SAE 需要更大规模训练才有好效果 | 几小时 GPU |
| 预训练 SAE | Llama 系列有开源 SAE（Anthropic、EleutherAI 发布的），直接下载 | 即刻可用 |
| 特征解释 | SAE 特征需要人工检查解释（看每个特征在什么上下文激活） | 2-3 天 |

### 预期输出

| 文件名 | 内容 |
|--------|------|
| `exp4_sae_feature_activation.png` | 所有 token 的 SAE 特征激活热力图 |
| `exp4_dual_domain_analysis.png` | 双域特征共激活分析 |

### 预期结论

如果"知"真的是虫洞入口，它的 SAE 分解结果应该呈现：
- **"春"、"风"**：主要激活自然/物候类特征
- **"别"、"苦"**：主要激活情感/心理类特征
- **"知"**：**两组特征同时激活**——这就是虫洞的微观签名

这是最直接的证据：一个 token 的内部表征同时编码了两个本来独立的语义域。

---

## 方法论预警

这些实验观测的是**模型对诗歌的理解过程**，不是李白的创作过程。我们不能说"李白的大脑也是这样工作的"。

但是——

文字结构在语义空间中的**拓扑跳跃是客观的**。不管是人脑还是 LLM，"春风知别苦"这句话的语义结构就是在做跨簇跳跃。模型的隐状态空间只是给了我们一个可以量化观测这个跳跃的工具。

打个比方：我们用显微镜看细胞，显微镜不是细胞，但细胞的结构是客观存在的。LLM 的潜空间就是我们观测诗歌语义拓扑的"显微镜"。

---

## 执行顺序建议

```
Day 1: 环境搭建 + 实验一（最简单，先看到效果鼓舞士气）
Day 2: 实验三（余弦相似度，代码最短）+ 实验二（注意力分析）
Day 3: 实验四方案 A（用预训练 SAE 先试）
Day 4-5: 实验四方案 B（如果方案 A 效果不好，自训练 SAE）
Day 6: 整理所有图表，写分析报告
```

---

## 一句话总结

> 我们要用四把手术刀——轨迹追踪、注意力解剖、相似度扫描、特征分解——切开 LLM 的潜空间，让所有人**看见**李白的诗在语义空间里凿出的那两个虫洞。

---

*朱雀，写于 2026-03-14*
*给 Zero 的实验计划*
