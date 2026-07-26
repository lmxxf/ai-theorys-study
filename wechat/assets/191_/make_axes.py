#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""191_ 期地基图：横向=生成过程，纵向=层；每层一条轨迹，测出一个数，再按层排开"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12.0, 6.3))
ax = fig.add_axes([0.04, 0.05, 0.60, 0.83])
ax2 = fig.add_axes([0.72, 0.20, 0.26, 0.55])

BLUE, ORANGE, GREEN, GREY = '#2f6fb5', '#e08a3c', '#2e9e6b', '#8b949e'

TOK = ['答', '案', '是', '.', '3']
LAYERS = ['第 1 层', '第 2 层', 'DOTS', '第 42 层', '第 43 层']
ys = [4.2, 3.3, 2.45, 1.55, 0.65]

ax.set_xlim(-1.85, 6.4)
ax.set_ylim(-0.55, 5.5)
ax.axis('off')

# 顶部：token 列（横向 = 生成过程）
for j, t in enumerate(TOK):
    ax.text(0.75 + j * 1.05, 5.02, t, fontsize=13, ha='center',
            color='#333', weight='bold')
ax.add_patch(FancyArrow(0.35, 4.72, 4.55, 0, width=0.012, head_width=0.13,
                        head_length=0.22, length_includes_head=True,
                        color=GREY, zorder=3))
ax.text(2.65, 4.86, '横向：模型一个字一个字往外蹦', fontsize=11.5,
        ha='center', color=GREY)

# 每层一行点
for i, (lb, y) in enumerate(zip(LAYERS, ys)):
    if lb == 'DOTS':
        ax.text(-0.35, y, '. . .', fontsize=13, ha='right', va='center',
                color=GREY, rotation=90)
        for j in range(5):
            ax.text(0.75 + j * 1.05, y, '. . .', fontsize=12, rotation=90,
                    ha='center', va='center', color=GREY)
        continue
    ax.text(-0.35, y, lb, fontsize=11.5, ha='right', va='center', color='#333')
    deep = i >= 3
    col = BLUE if not deep else GREEN
    ax.add_patch(FancyBboxPatch((0.28, y - 0.29), 4.72, 0.58,
                                boxstyle='round,pad=0.02,rounding_size=0.1',
                                facecolor=col, alpha=0.09,
                                edgecolor=col, lw=1.0, zorder=1))
    for j in range(5):
        ax.plot(0.75 + j * 1.05, y, 'o', ms=9, color=col, zorder=4)
    ax.plot([0.75, 0.75 + 4 * 1.05], [y, y], color=col, lw=1.4,
            alpha=0.55, zorder=2)
    # 右侧：这一行测出来的数
    val = ['ID ≈ 48', 'ID ≈ 31', '', 'ID ≈ 6', 'ID ≈ 4'][i]
    ax.text(5.42, y, val, fontsize=12, va='center', color=col, weight='bold')

# 纵向箭头
ax.add_patch(FancyArrow(-1.42, 4.45, 0, -3.95, width=0.012, head_width=0.13,
                        head_length=0.22, length_includes_head=True,
                        color=GREY, zorder=3))
ax.text(-1.62, 2.5, '纵向：一层一层往下算', fontsize=11.5, rotation=90,
        va='center', ha='center', color=GREY)

ax.text(2.65, -0.30,
        '每一行是一条轨迹：同一层上，随着字一个个蹦出来，状态走过的路线\n'
        '量这条轨迹占了几个方向 → 每层得到一个数（ID）',
        fontsize=11.5, ha='center', color='#333', linespacing=1.65)

ax.text(5.42, 5.30, '每层测出\n一个数', fontsize=10.5, ha='center', color='#555', linespacing=1.5)

# ---- 右图：把这些数按层排开 ----
import numpy as np
xx = np.linspace(0, 1, 120)
yy = 48 * np.exp(-xx / 0.16) + 4
ax2.plot(xx, yy, color=BLUE, lw=2.4)
ax2.scatter([0, 0.03, 0.95, 1.0], [48, 31, 6, 4],
            color=[BLUE, BLUE, GREEN, GREEN], s=52, zorder=5)
ax2.set_xlabel('层深（浅 → 深）', fontsize=11)
ax2.set_ylabel('内在维度 ID', fontsize=11)
ax2.set_title('把每层那个数按层排开', fontsize=12.5, pad=10)
ax2.grid(alpha=0.22, lw=0.6)
ax2.tick_params(labelsize=9)
for s in ax2.spines.values():
    s.set_linewidth(0.7)
ax2.text(0.52, 30, '这条曲线\n就是论文的主角', fontsize=11,
         ha='center', color='#444', linespacing=1.5)

fig.suptitle('先分清两个方向：横向是生成过程，纵向是层——测维度在横向，看下降在纵向',
             fontsize=15, y=0.965)
fig.text(0.5, 0.005,
         '注意：向量本身一直是 4096 维，从第 1 层到第 43 层都没变。'
         '降的是"这条轨迹实际占了几个方向"，不是向量变短了',
         ha='center', fontsize=10.5, color='#b3541e')

out = '/home/lmxxf/work/ai-theorys-study/wechat/assets/191_/two-axes.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved:', out)
