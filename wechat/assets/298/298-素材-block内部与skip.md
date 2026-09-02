# 298 素材:block 内部三形状 + skip vs cross-attention(闇 2026-09-01 讲解)

## 核心区分:block ≠ layer ≠ 权重记录
- block0-70 = **71 个 block**(建筑平面图的房间数)
- 运行时实际 = **152 个计算 Layer**(房间里摆的机器数)
- 权重包 = **153 条权重记录**(比152多1:block70一个Layer同时有主权重+blend_scale两条)

## 完整六站 block 分配(闇精确版,注意512回程是39-47共9个)
```
输入
32通道   block 0-4    5个
64通道   block 5-8    4个
128通道  block 9-14   6个
256通道  block 15-22  8个
512通道  block 23-30  8个
1024通道 block 31-38  8个(瓶颈)
512通道  block 39-47  9个(回程,含block39拐弯)
256通道  block 48-55  8个
128通道  block 56-61  6个
64通道   block 62-65  4个
32通道   block 66-69  4个
block 70 输出RGB+原图混合
= 5+4+6+8+8+8+9+8+6+4+4+1 = 71
```

## block 内部三种形状(越深拆得越细)

**形状1:外围 32/64/128/256 通道 = 1个block融合成1层**
一个融合 Swin,内部数学:
- cosine attention:QKV → 8×8 窗口注意力 → 投影
- 两次 residual(原输入加回来)
- 例 128通道block:FFN 128→160→128 / 4个注意力头每头看16维 / 投影64→128 / 输出仍128
- 运行时融合成1个GPU kernel,所以算1 block=1 Layer,但数学里有FFN+注意力+残差

**形状2:512 通道 = 拆成 4 个 Layer**(显卡不融合这么宽的)
- Layer1:FFN展开
- Layer2:FFN投影回来
- Layer3:QKV+attention
- Layer4:attention投影
- block23-30、block40-47 各8个block但内部计算层多
- block30 还负责 512→1024 送进瓶颈

**形状3:1024 瓶颈 = 拆成 5 个 Layer**(最深最重)
- Layer1:FFN Expand
- Layer2:FFN Contract
- Layer3:生成Q/K/V
- Layer4:全局/一维Attention
- Layer5:Projection
- 8 block×5层 = 40个运行时Layer,网络最重的脑子在这

**回程5个特殊"升采样+接线"block:**
```
block39:1024→512,接block30 skip
block48:512→256,接block22 skip
block56:256→128,接block14 skip
block62:128→64,接block8 skip
block66:64→32,接block4 skip
block70:接block69主干+block0最浅特征,输出RGB
```

## skip 是什么(比 cross-attention 简单太多)

**一句话:前面存一份,后面直接拿过来。**

普通主干 A→B→C→D:A的信息必须经过B、C才到D,中间压缩丢了D拿不回。
skip:A多拉一根线直接送D(跳过B、C),所以叫skip。

**为什么需要(头发比喻,极好用):**
- 原图有根细头发,一路缩小1920→960→480→...
- 缩到底层,模型知道"这里是脑袋",但头发精确位置没了
- 右边放大只凭"脑袋"抽象信息,只能猜头发画哪
- 于是左边头发还清楚时偷偷存一份,右边放大到同尺寸时直接拿回来
- 底层信息=画什么(这里该生成头发),skip信息=画在哪(那根细线具体在这)
- 两份一起用:既知道画什么,也知道画在哪

**skip vs cross-attention(金比喻):**
- **skip**:同事把整份旧文件直接塞给你(拼接/相加/一起喂下一层)
- **cross-attention**:你拿问题去档案库检索,只挑相关段落回来(算QKᵀ、softmax、加权读V)
- 这网络的skip接近第一种,CPU构图恢复出的是真正双输入
- 一句话:cross-attention="带着问题去查另一份资料",skip="怕你忘了,直接把旧稿塞回来"

## 298 可用的两个金句
1. "71个block是建筑平面图,152个Layer是里面实际摆着的机器"
2. "Cross-attention是带着问题去查另一份资料,skip是怕你忘了直接把旧稿塞回来"

## 298 待核(写时确认)
- block56内部到底是相加/拼接后卷积/特制融合——看具体kernel数学,闇说执行图层面已确定双输入,内部融合方式待定
- 512回程 block39-47、256回程block48-55 = 闇版对。**已用 network-graph.json 权威核准(2026-09-01):block宽度归属 0-4/32,5-8/64,9-14/128,15-22/256,23-30/512,31-38/1024,39-47/512,48-55/256,56-61/128,62-65/64,66-69/32,70/output。297第一章表格已按此改正。**
