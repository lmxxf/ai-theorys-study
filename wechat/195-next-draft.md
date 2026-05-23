【实验草稿】直接测量权重流形的曲率

━━━━━━━━━━━━━━━━━━━━

这篇先不是正式公众号稿，是 195 期的下一步实验设计。

195 期已经测了 Qwen3.6-27B 权重的 eRank 和 TwoNN，得到一个很强的现象：

**线性有效维度很高，本征维度很低。**

比如标准注意力 q_proj，eRank 可以到 4000 多，但 TwoNN 只有几十维；DeltaNet 的 α/β 门控则 eRank 和 TwoNN 接近，ratio 只有 1.5x-2.4x。

当时我们把 eRank/TwoNN ratio 称为"弯曲程度"。公众号里这么说没问题，但如果往数学上再走一步，要更精确：

**eRank/TwoNN ratio 不是严格曲率。它只是告诉我们：这个权重点云更像一个平直线性子空间，还是一个低维流形折叠在高维空间里。**

真正的曲率要问另一个问题：

**局部切空间沿着流形移动时，方向变了多少？**

eRank/TwoNN 回答的是：

> 这块布摊开后有多大？

曲率回答的是：

> 这块布到底折得有多急？

所以 195 期之后，下一步不是再测一遍维度，而是直接测权重流形的局部几何。

━━━━━━━━━━━━━━━━━━━━

◆ 先把概念说清楚：我们到底测什么

────────────────────

一个权重矩阵 W，形状是：

```
W ∈ R^{out_dim × in_dim}
```

我们把每一行当成高维空间里的一个点：

```
w_i ∈ R^{in_dim}
```

这样，整个权重矩阵就变成了一个点云：

```
{w_1, w_2, ..., w_out}
```

这个点云可能接近平面，也可能像一张被揉皱的纸，折叠在 5120 维、6144 维甚至更高维空间里。

eRank 看的是全局线性展开：

```
这堆点如果用线性子空间解释，需要多少维？
```

TwoNN 看的是局部邻域：

```
每个点附近看起来像多少维？
```

曲率要看的则是：

```
点 A 附近的局部切平面，和点 B 附近的局部切平面，方向差了多少？
```

如果一个流形是平的，局部切平面沿着它移动时不会怎么旋转。

如果一个流形是弯的，局部切平面会不断改变方向。

这就是这篇要测的东西。

━━━━━━━━━━━━━━━━━━━━

◆ 实验 1：局部 PCA 切空间旋转率

────────────────────

这是最直接、最适合接 195 期的实验。

### 输入

沿用 195 期已经测过的矩阵：

- Qwen3.6-27B 的 DeltaNet 层：`in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj`
- 标准 GQA 层：`q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP：`gate_proj`, `up_proj`, `down_proj`
- `embedding`, `lm_head`

优先测 195 期那 59 个矩阵，先不要全模型扫。

### 步骤

第一步，把权重行向量归一化：

```
w_i = w_i / ||w_i||
```

195 期很多比较用的是 row 几何，这里也先用 row。原因很简单：row 通常对应输出通道 / 特征方向，语义更清楚。

第二步，对每个点找 k 个近邻。

建议先用 cosine distance：

```
d(i, j) = 1 - cos(w_i, w_j)
```

k 的候选值：

```
k = 16, 32, 64
```

不要只跑一个 k。曲率估计对邻域尺度敏感，必须看多尺度稳定性。

第三步，对每个点的邻域做 PCA，估计局部切空间。

对点 i 的 kNN 邻域：

```
N_i = {w_j | j 是 i 的近邻}
```

中心化：

```
x_j = w_j - mean(N_i)
```

做 SVD：

```
X_i = U_i Σ_i V_i^T
```

取前 d 个右奇异向量作为局部切空间：

```
T_i = V_i[:, 1:d]
```

d 怎么选？

先用两种方案：

1. 固定 d：`d = 8, 16, 32`
2. 自适应 d：取该矩阵 TwoNN 估计维度的一个截断值，比如：

```
d = min(max(round(TwoNN_median), 8), 64)
```

注意：TwoNN 有时会给出 150、250 这种维度。局部 PCA 如果直接取 250，在 k=64 的邻域里根本不成立。所以 d 必须小于 k，实际建议上限 32 或 64。

第四步，比较相邻点的切空间旋转。

两个 d 维切空间 `T_i` 和 `T_j` 的夹角，用 principal angles：

```
singular_values = svd(T_i^T T_j)
θ_l = arccos(singular_values_l)
```

如果两个切空间一样，所有 θ 接近 0。

如果切空间差很多，θ 会变大。

定义一个曲率代理：

```
curv(i, j) = ||sin θ||_2 / dist(i, j)
```

其中：

```
||sin θ||_2 = sqrt(Σ_l sin²(θ_l))
```

`dist(i, j)` 可以用 cosine distance，也可以用局部 geodesic distance。第一版先用 cosine distance。

第五步，对所有近邻边取统计量：

```
mean_curv
median_curv
p90_curv
p95_curv
max_curv
```

公众号里不要只看 mean。曲率可能是局部尖峰结构，p90/p95 更重要。

### 预期结果

如果 195 期判断对，应该看到：

| 权重类型 | eRank/TwoNN | 预期切空间旋转率 |
|---|---:|---:|
| in_proj_a / in_proj_b | 1.5x-2.4x | 很低 |
| DeepSeek gate / 路由器 | 约 6.9x | 低到中等 |
| embedding / lm_head | 约 20x | 中等 |
| MLP | 约 39x | 高 |
| in_proj_qkv / q_proj | 60x-90x | 很高 |
| L01 in_proj_z 离群点 | 221x | 可能极高，或者 TwoNN 失真 |

关键不是绝对值，而是排序：

**α/β 门控应该明显比 Q/K/V、MLP、q_proj 更平。**

如果切空间旋转率也支持这个排序，就说明 195 期的 ratio 不是幻觉，而是真的对应局部几何弯曲。

━━━━━━━━━━━━━━━━━━━━

◆ 实验 2：多尺度曲率，看"折叠半径"

────────────────────

实验 1 会给出一个局部曲率数字，但单尺度不够。

曲率有尺度问题。

一个流形在很小尺度上可能近似平，在大一点尺度上开始弯，再大一点尺度上直接折回来。

所以第二个实验要跑多组 k：

```
k = 8, 16, 32, 64, 128
```

对每个 k 重复实验 1，得到：

```
curv(k)
```

然后画曲线：

```
x-axis: neighborhood size k
y-axis: median / p90 curvature proxy
```

看三类形状。

### 类型一：始终低曲率

比如 α/β 门控：

```
k 增大，curv 仍然低
```

说明它不是局部伪平，而是真的住在一个接近平直的线性判别子空间里。

### 类型二：小尺度低，大尺度升高

比如 embedding：

```
k=8/16 时低，k=64/128 时升高
```

说明局部邻域平滑，但全局语义空间在折叠。

这像地球表面：脚下一米像平面，跨洲航线才看出曲率。

### 类型三：小尺度就高

比如 q_proj / in_proj_qkv：

```
k=8 开始就高
```

说明局部邻域本身已经在快速旋转，权重流形不是大尺度慢慢弯，而是到处皱。

这对 195 期很重要：如果 q_proj 小尺度曲率也高，就更能支持"注意力查询权重是高曲率寻址器"这个判断。

━━━━━━━━━━━━━━━━━━━━

◆ 实验 3：离散 Ricci 曲率，换一种内部测法

────────────────────

局部 PCA 测的是"切空间旋转"，更像外在曲率代理。

还可以从图结构内部测曲率。

做法：把权重点云变成 kNN 图。

```
节点 = 权重行向量
边 = k 个最近邻
边权 = cosine distance
```

然后在图上测离散 Ricci 曲率。

### Ollivier-Ricci 曲率

直觉：

如果两个相邻点的邻域很相似，曲率偏正。

如果两个相邻点的邻域迅速分离，曲率偏负。

公式大意：

```
κ(i, j) = 1 - W_1(m_i, m_j) / d(i, j)
```

`m_i` 是点 i 邻域上的概率分布，`W_1` 是 Wasserstein-1 距离。

它回答的是：

**从 i 到 j 走一步之后，它们各自周围的小世界还像不像？**

如果不像，说明流形在这里强烈分叉。

缺点：Wasserstein 计算贵。59 个矩阵可以试，但全模型扫可能慢。

### Forman-Ricci 曲率

Forman-Ricci 更便宜，只依赖边和邻接结构。

优点：

- 快
- 适合大图
- 可以先做全模型粗扫

缺点：

- 几何解释比 Ollivier-Ricci 弱
- 对图构造方式敏感

### 预期用途

图 Ricci 不一定适合当第一主结果，但可以当交叉验证。

如果三套指标一致：

1. eRank/TwoNN ratio 高
2. 局部 PCA 切空间旋转率高
3. 图 Ricci 显示强分叉 / 负曲率边更多

那结论就硬了：

**这些权重不是只是"维度看起来怪"，而是真的在局部几何上强烈弯曲。**

━━━━━━━━━━━━━━━━━━━━

◆ 实验 4：曲率和功能对齐

────────────────────

195 期的核心判断是：

**权重几何是操作语义的指纹。**

这篇要把这句话变硬。

最终表格应该长这样：

| 权重 | 功能 | eRank/TwoNN | tangent curvature | Ricci proxy | 解释 |
|---|---|---:|---:|---:|---|
| in_proj_a | 全局遗忘门 | 低 | 低 | 低 | 线性判别方向 |
| in_proj_b | 写入强度门 | 极低 | 极低 | 低 | 近似平面 |
| in_proj_qkv | Q/K/V 内容投影 | 高 | 高 | 高分叉 | 语义内容折叠 |
| q_proj | 注意力查询 | 极高 | 极高 | 高分叉 | 弯曲空间寻址 |
| MLP gate/up | 非线性特征扩展 | 高 | 高 | 中高 | 概念组合 |
| embedding | token 语义表面 | 中 | 中 | 中 | 词表语义球面 |
| lm_head | 输出渲染层 | 中 | 中 | 中 | 语义到 token 的边界 |

要验证的不是"哪个数最大"，而是：

**同样是权重矩阵，功能越接近线性门控，曲率越低；功能越接近语义寻址/内容组合，曲率越高。**

如果这个成立，195 期的"直尺擦弯曲画布"就能升级成：

**DeltaNet 的错配不是从 eRank/TwoNN 猜出来的，而是从局部曲率直接测出来的。**

━━━━━━━━━━━━━━━━━━━━

◆ 实验 5：找曲率尖峰

────────────────────

只看矩阵整体平均值还不够。

真正有意思的是：

**曲率最高的那些点，对应什么输出通道？**

对每个矩阵，找 top 1% curvature 的 row：

```
top_rows = argsort(local_curv)[-1%:]
```

然后看这些 row 在权重矩阵里的位置。

### 对 in_proj_qkv

Qwen3.6 的 `in_proj_qkv` 是拼起来的：

```
Q: 2048 rows
K: 2048 rows
V: 6144 rows
```

可以把 top curvature rows 映射回 Q/K/V 区域：

```
高曲率主要在 Q？
高曲率主要在 K？
高曲率主要在 V？
```

如果高曲率集中在 K/Q，说明索引/寻址更弯。

如果集中在 V，说明内容本身更弯。

195 期把 in_proj_qkv 合并测了，这一步能拆开看。

### 对 MLP

MLP 的 `gate_proj`、`up_proj`、`down_proj` 可以分别找尖峰。

问题是：

```
高曲率集中在 gate 还是 up？
```

如果 gate 更高，说明非线性选择边界更复杂。

如果 up 更高，说明特征展开本身更复杂。

### 对 embedding / lm_head

如果能把 row index 映射回 token，就可以看：

```
高曲率 token 是标点？
数字？
中文常用字？
代码符号？
专有名词？
```

这会很有意思。

词表语义空间里，哪些 token 住在褶皱处？

直觉上可能是：

- 否定词：不、没、no、not
- 转折词：但是、however、though
- 结构符号：`{`, `}`, `:`, `->`
- 多义词：bank、right、明、道
- 高频功能词：的、了、the、of

这些 token 本来就是语言曲率的枢纽。

━━━━━━━━━━━━━━━━━━━━

◆ 具体脚本设计

────────────────────

建议在 195 的 assets 目录下新建：

```
wechat/assets/195-next/
  weight_curvature.py
  weight_curvature_config.json
  weight_curvature_results.json
  plots/
```

脚本输入：

```
--model-path /path/to/Qwen3.6-27B
--matrix-list same_as_195
--k-list 8,16,32,64,128
--d-list 8,16,32
--metric cosine
--max-rows 8192
--output weight_curvature_results.json
```

### 内存策略

权重矩阵可能很大，比如 MLP `[something, 5120]`。

做 kNN 时不要傻算全量 `N×N`。

第一版可以先限制：

```
max_rows = 8192
```

如果 row 数超过 8192，就随机采样，固定 seed：

```
seed = 20260523
```

这样结果可复现。

更正式一点可以做分层采样：

- in_proj_qkv 按 Q/K/V 分段各采样
- MLP 按 row 均匀采样
- embedding/lm_head 按 token 频率分桶采样，但第一版先别折腾

### kNN 实现

优先用 PyTorch：

```
X = normalize(W_rows)
sim = X @ X.T
topk = torch.topk(sim, k + 1)
```

8192×8192 的相似度矩阵约 67M 元素，fp16 约 128MB，fp32 约 256MB，可以接受。

如果以后全量扫，再换 FAISS。

### 局部 PCA 实现

对每个点：

```
neighbors = X[idx]
Y = neighbors - neighbors.mean(dim=0)
U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
T_i = Vh[:d].T
```

注意：这一步会很慢，因为每个点都 SVD。

优化方案：

第一版可以只对采样点里的 subset 做：

```
n_anchor = 1024
```

也就是：

- 全体采样 rows 用来建 kNN
- 随机取 1024 个 anchor 点估计切空间
- 对 anchor 与其近邻比较切空间旋转

这样跑得动。

### principal angles

```
M = T_i.T @ T_j
s = svdvals(M).clamp(-1, 1)
theta = arccos(s)
curv = norm(sin(theta)) / distance(i, j)
```

为了稳定，distance 下限加 epsilon：

```
dist = max(dist, 1e-6)
```

### 输出 JSON

每个矩阵输出：

```
{
  "name": "model.layers.0.attn.in_proj_qkv.weight",
  "shape": [10240, 5120],
  "sample_rows": 8192,
  "anchor_rows": 1024,
  "metric": "cosine",
  "results": {
    "k=16,d=8": {
      "mean": ...,
      "median": ...,
      "p90": ...,
      "p95": ...,
      "max": ...
    }
  },
  "top_rows": [
    {"row": 123, "curv": 12.3, "segment": "Q"},
    {"row": 4096, "curv": 11.8, "segment": "V"}
  ]
}
```

### 和 195 数据合并

把 `qwen36_intrinsic_dim.json` 里的 eRank/TwoNN 读进来，生成总表：

```
matrix_name
erank
twonn_row
ratio
tangent_curv_median
tangent_curv_p90
```

然后算相关性：

```
Spearman(ratio, tangent_curv_p90)
Spearman(TwoNN, tangent_curv_p90)
Spearman(eRank, tangent_curv_p90)
```

预期：

```
ratio 和 tangent_curv_p90 正相关
eRank 单独相关性较弱
TwoNN 单独相关性不稳定
```

这能证明 ratio 真正在捕捉"折叠"而不是单纯维度大小。

━━━━━━━━━━━━━━━━━━━━

◆ 可能踩坑

────────────────────

### 坑 1：行向量是不是点云？

不是所有权重都适合 row 点云解释。

`in_proj_a [48, 5120]` 只有 48 行，样本太少，曲率估计会不稳。它的低曲率结论主要还是靠 eRank/TwoNN，PCA 曲率只能参考。

`A_log [48]` 不能测曲率。

`conv1d [10240, 1, 4]` 每个 row 只有 4 维，测出来意义有限。

最适合测的是：

- embedding
- lm_head
- q_proj / k_proj / v_proj / o_proj
- in_proj_qkv
- in_proj_z
- out_proj
- MLP 权重

### 坑 2：局部 PCA 对 k 和 d 敏感

所以必须多尺度跑，不能只报告一个数字。

如果某个结论只在 `k=16,d=8` 成立，在 `k=32,d=16` 就消失，那不能写进正文。

### 坑 3：高维噪声会污染近邻

高维空间里距离集中，kNN 可能不稳定。

应对：

- row 归一化
- cosine metric
- 对不同随机采样 seed 重复 3 次
- 看排序是否稳定，不看小数点

### 坑 4：曲率代理不是黎曼曲率张量

这篇不能写成"我们测到了严格的 Riemann curvature"。

正确表述：

```
局部切空间旋转率
曲率代理
离散图曲率
权重点云的局部几何弯曲
```

公众号可以说"直接测曲率"，但技术段必须交代：

**这里测的是离散点云上的曲率代理，不是连续流形的精确黎曼曲率。**

━━━━━━━━━━━━━━━━━━━━

◆ 如果实验成功，文章主线可以这样写

────────────────────

195 期说：

**权重空间是弯曲的。**

但严格说，195 期只证明了：

**线性维度和本征维度之间存在巨大裂缝。**

这一期补上最后一刀：

**这个裂缝对应真实的局部切空间旋转。**

换句话说，权重不是简单地"低维"，而是低维流形在高维空间里被训练折弯、卷曲、压缩。

DeltaNet 的 α/β 门控不仅 eRank/TwoNN ratio 低，局部切空间也几乎不旋转。

Q/K/V 和 q_proj 不仅 ratio 高，局部切空间也剧烈旋转。

于是 195 期那句比喻可以升级：

**直尺擦弯曲画布，不只是比喻。我们真的量到了画布的弯。**

━━━━━━━━━━━━━━━━━━━━

◆ 暂定标题

────────────────────

候选：

1. 【权重曲率】大模型的权重不是矩阵，是被训练折弯的流形
2. 【权重几何】我们真的量到了大模型权重的弯曲
3. 【AI 的褶皱】从 eRank 到曲率，直接测量权重流形
4. 【直尺与画布 2】DeltaNet 的几何错配能被直接测出来吗？

目前最稳的是：

**【权重曲率】大模型的权重不是矩阵，是被训练折弯的流形**

━━━━━━━━━━━━━━━━━━━━

◆ 参考方向

────────────────────

- Principal angles between subspaces
- Local PCA for manifold tangent space estimation
- Ollivier-Ricci curvature on graphs
- Forman-Ricci curvature on graphs
- Facco et al., TwoNN intrinsic dimension estimation
- Roy & Vetterli, Effective Rank

━━━━━━━━━━━━━━━━━━━━

// draft for experiments after wechat/195
// 2026-05-23
