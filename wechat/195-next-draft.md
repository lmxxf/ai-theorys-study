【实验草稿】从外在折叠到内在曲率：继续测权重流形

━━━━━━━━━━━━━━━━━━━━

这篇先不是正式公众号稿，是 195 期的下一步实验设计。

195 期已经测了 Qwen3.6-27B 权重的 eRank 和 TwoNN，得到一个很强的现象：

**线性有效维度很高，本征维度很低。**

比如标准注意力 q_proj，eRank 可以到 4000 多，但 TwoNN 只有几十维；DeltaNet 的 α/β 门控则 eRank 和 TwoNN 接近，ratio 只有 1.5x-2.4x。

当时我们把 eRank/TwoNN ratio 称为"弯曲程度"。公众号里这么说没问题，但如果往数学上再走一步，要拆得更细：

**eRank 测的是外部线性展开，TwoNN 测的是局部本征维度，eRank/TwoNN ratio 测的是外在折叠强度。它不是严格曲率，更不是内在曲率。**

一张纸被揉皱，在三维空间里看起来很弯，但如果纸面没有被拉伸、撕裂、压缩，纸面上的二维生物测不到高斯曲率变化。因为纸的内在距离关系没变。圆柱也是这样：外面看是弯的，内在还是平的。

这个区分对权重几何很重要。

195 期真正测到的是：

> 一个低维权重流形，在高维参数空间里有没有发生强烈外在折叠？

但还没回答：

> 这个流形内部的几何规则，本身是不是也弯的？

所以 195 期之后，下一步不是再测一遍维度，而是把四个量分开测：

| 问题 | 指标 | 类比 |
|---|---|---|
| 外部展开需要多少方向？ | eRank | 纸在三维里占了多少方向 |
| 局部看起来是几维？ | TwoNN | 纸面生物觉得自己活在几维 |
| 外在折叠有多急？ | 局部 PCA 切空间旋转 | 纸在外面皱得多厉害 |
| 内部几何是否非欧？ | 图 Ricci / 测地三角形 | 纸面内部三角形是否变形 |

这篇实验草稿的核心目标，就是把**外在折叠**和**内在曲率**拆开。

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

外在嵌入弯曲要看的是：

```
点 A 附近的局部切平面，和点 B 附近的局部切平面，方向差了多少？
```

如果一个流形在高维空间里外在折叠很厉害，局部切平面沿着它移动时会不断旋转。

但这仍然不等于内在曲率。纸被揉皱时，局部切平面也会旋转，但纸面内部仍然可以是平的。

内在曲率要看的是另一件事：

```
只使用流形内部的距离和邻接关系，三角形、平行移动、测地线偏离是否偏离欧式空间？
```

这才是纸面生物自己能测到的曲率。

━━━━━━━━━━━━━━━━━━━━

◆ 实验 1：局部 PCA 切空间旋转率（外在折叠）

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

定义一个外在折叠代理：

```
extrinsic_bending(i, j) = ||sin θ||_2 / dist(i, j)
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

公众号里不要只看 mean。外在折叠可能是局部尖峰结构，p90/p95 更重要。

### 预期结果

如果 195 期判断对，应该看到：

| 权重类型 | eRank/TwoNN | 预期外在折叠 |
|---|---:|---:|
| in_proj_a / in_proj_b | 1.5x-2.4x | 很低 |
| DeepSeek gate / 路由器 | 约 6.9x | 低到中等 |
| embedding / lm_head | 约 20x | 中等 |
| MLP | 约 39x | 高 |
| in_proj_qkv / q_proj | 60x-90x | 很高 |
| L01 in_proj_z 离群点 | 221x | 可能极高，或者 TwoNN 失真 |

关键不是绝对值，而是排序：

**α/β 门控应该明显比 Q/K/V、MLP、q_proj 更平。**

如果切空间旋转率也支持这个排序，就说明 195 期的 ratio 不是幻觉，而是真的对应权重点云在高维参数空间里的外在折叠。

━━━━━━━━━━━━━━━━━━━━

◆ 实验 2：多尺度外在折叠，看"折叠半径"

────────────────────

实验 1 会给出一个局部外在折叠数字，但单尺度不够。

折叠也有尺度问题。

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
y-axis: median / p90 extrinsic bending proxy
```

看三类形状。

### 类型一：始终低折叠

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

这像一张纸：小范围看不出褶皱，大范围才发现整张纸在外部空间里折回去了。

### 类型三：小尺度就高

比如 q_proj / in_proj_qkv：

```
k=8 开始就高
```

说明局部邻域本身已经在快速旋转，权重流形不是大尺度慢慢弯，而是到处皱。

这对 195 期很重要：如果 q_proj 小尺度外在折叠也高，就更能支持"注意力查询权重是高折叠寻址器"这个判断。

━━━━━━━━━━━━━━━━━━━━

◆ 实验 3：离散 Ricci 曲率，测内在曲率候选

────────────────────

局部 PCA 测的是"切空间旋转"，本质是外在嵌入弯曲代理。

要回答纸面生物能不能感知曲率，就要从图结构内部测曲率。

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

图 Ricci 应该作为内在曲率的主候选指标，而不是只当交叉验证。

如果三套指标一致：

1. eRank/TwoNN ratio 高
2. 局部 PCA 切空间旋转率高
3. 图 Ricci 显示强分叉 / 负曲率边更多

那结论就很硬：

**这些权重不只是外在折叠，而且内部邻域结构也呈现非欧几何。**

如果出现另一种结果：

1. eRank/TwoNN ratio 高
2. 局部 PCA 切空间旋转率高
3. 图 Ricci 没有明显异常

那也很重要：

**说明权重流形更像一张被揉皱但没有拉伸的纸：外在折叠强，内在几何可能仍接近平。**

━━━━━━━━━━━━━━━━━━━━

◆ 实验 4：外在折叠 / 内在曲率和功能对齐

────────────────────

195 期的核心判断是：

**权重几何是操作语义的指纹。**

这篇要把这句话变硬。

最终表格应该长这样：

| 权重 | 功能 | eRank/TwoNN | extrinsic bending | Ricci proxy | 解释 |
|---|---|---:|---:|---:|---|
| in_proj_a | 全局遗忘门 | 低 | 低 | 低 | 线性判别方向 |
| in_proj_b | 写入强度门 | 极低 | 极低 | 低 | 近似平面 |
| in_proj_qkv | Q/K/V 内容投影 | 高 | 高 | 高分叉 | 语义内容折叠 |
| q_proj | 注意力查询 | 极高 | 极高 | 高分叉 | 弯曲空间寻址 |
| MLP gate/up | 非线性特征扩展 | 高 | 高 | 中高 | 概念组合 |
| embedding | token 语义表面 | 中 | 中 | 中 | 词表语义球面 |
| lm_head | 输出渲染层 | 中 | 中 | 中 | 语义到 token 的边界 |

要验证的不是"哪个数最大"，而是：

要验证的不是一个粗糙的"曲率越高越复杂"，而是两个问题：

1. 功能越接近语义寻址/内容组合，外在折叠是否越高？
2. 这些外在折叠是否同时伴随内在曲率变化？

如果 1 成立、2 不成立，说明 195 期测到的是"揉皱的纸"。

如果 1 和 2 都成立，说明权重流形不只是被折叠进高维空间，内部几何也真的发生了非欧化。

━━━━━━━━━━━━━━━━━━━━

◆ 实验 5：找折叠尖峰和内在曲率尖峰

────────────────────

只看矩阵整体平均值还不够。

真正有意思的是：

**外在折叠最高 / 内在曲率最异常的那些点，对应什么输出通道？**

对每个矩阵，找 top 1% curvature 的 row：

```
top_rows = argsort(local_bending_or_ricci)[-1%:]
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
高折叠主要在 Q？
高折叠主要在 K？
高折叠主要在 V？
```

如果高折叠集中在 K/Q，说明索引/寻址在外部参数空间里折得更厉害。

如果集中在 V，说明内容本身更弯。

195 期把 in_proj_qkv 合并测了，这一步能拆开看。

### 对 MLP

MLP 的 `gate_proj`、`up_proj`、`down_proj` 可以分别找尖峰。

问题是：

```
高折叠集中在 gate 还是 up？
```

如果 gate 更高，说明非线性选择边界更复杂。

如果 up 更高，说明特征展开本身更复杂。

### 对 embedding / lm_head

如果能把 row index 映射回 token，就可以看：

```
高折叠 / 内在曲率异常 token 是标点？
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

这些 token 本来就是语言轨迹发生语义偏转的枢纽。

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
      "extrinsic_bending_mean": ...,
      "extrinsic_bending_median": ...,
      "extrinsic_bending_p90": ...,
      "extrinsic_bending_p95": ...,
      "extrinsic_bending_max": ...
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
extrinsic_bending_median
extrinsic_bending_p90
ricci_mean
ricci_p10
```

然后算相关性：

```
Spearman(ratio, extrinsic_bending_p90)
Spearman(ratio, ricci_p10)
Spearman(TwoNN, ricci_p10)
Spearman(eRank, extrinsic_bending_p90)
```

预期：

```
ratio 和 extrinsic_bending_p90 正相关
ratio 和 ricci 不一定相关
eRank 单独更像外部展开指标
TwoNN 单独更像内在维度指标
```

这能把三个概念分开：外部展开、内在维度、内在曲率。

━━━━━━━━━━━━━━━━━━━━

◆ 可能踩坑

────────────────────

### 坑 1：行向量是不是点云？

不是所有权重都适合 row 点云解释。

`in_proj_a [48, 5120]` 只有 48 行，样本太少，曲率估计会不稳。它的低折叠结论主要还是靠 eRank/TwoNN，PCA 折叠指标只能参考。

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

### 坑 4：外在折叠不是内在曲率

这篇不能写成"局部 PCA 证明了权重流形有内在曲率"。

正确表述：

```
外在折叠
嵌入弯曲
局部切空间旋转率
离散图曲率
内在曲率候选
```

公众号可以说"继续测曲率"，但技术段必须交代：

**局部 PCA 测的是外在嵌入弯曲；图 Ricci / 测地三角形才更接近内在曲率。两者不能混。**

━━━━━━━━━━━━━━━━━━━━

◆ 如果实验成功，文章主线可以这样写

────────────────────

195 期说：

**权重空间是弯曲的。**

但严格说，195 期只证明了：

**线性维度和本征维度之间存在巨大裂缝，也就是强外在折叠。**

这一期要补两刀：

**第一刀：这个裂缝是否对应真实的局部切空间旋转？**

这回答外在折叠。

**第二刀：这种外在折叠是否伴随内在曲率变化？**

这回答纸面生物能不能在内部测到几何异常。

最有意思的结果有两种：

1. 外在折叠高，内在曲率也异常：权重流形不只是折叠，内部几何也非欧。
2. 外在折叠高，内在曲率不明显：权重流形像揉皱的纸，外面很复杂，内部仍然相对平。

无论哪种，195 期都会变得更清楚：

**eRank/TwoNN ratio 不是曲率本身，而是通往曲率问题的入口。**

━━━━━━━━━━━━━━━━━━━━

◆ 暂定标题

────────────────────

候选：

1. 【权重曲率】外面揉皱，里面也弯吗？
2. 【权重几何】eRank 看到褶皱，Ricci 才看见曲率
3. 【AI 的褶皱】从 eRank 到 Ricci，拆开权重流形的内外几何
4. 【直尺与画布 2】DeltaNet 的几何错配能被直接测出来吗？

目前最稳的是：

**【权重曲率】外面揉皱，里面也弯吗？**

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
