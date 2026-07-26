#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""191_ 期配图：每一层都有自己的一套 attention + FFN，各算各的"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE, ORANGE, GREEN, GREY = '#2f6fb5', '#e08a3c', '#2e9e6b', '#8b949e'

fig = plt.figure(figsize=(10.6, 7.4))
ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

LAYER_Y = [7.9, 6.15, 4.4, 2.42]
LABELS = ['第 1 层', '第 2 层', '第 3 层', '第 43 层']
CX = 4.15          # 层块中心
W, H = 5.0, 1.32


def layer_box(y, label, col, dim=False):
    a = 0.30 if not dim else 0.16
    ax.add_patch(FancyBboxPatch((CX - W / 2, y - H / 2), W, H,
                                boxstyle='round,pad=0.05,rounding_size=0.13',
                                facecolor=col, alpha=0.085,
                                edgecolor=col, lw=1.5, zorder=2))
    ax.text(CX - W / 2 - 0.28, y, label, fontsize=12, ha='right',
            va='center', color='#333')
    # 内部两个零件
    ax.add_patch(FancyBboxPatch((CX - 2.28, y - 0.42), 2.16, 0.84,
                                boxstyle='round,pad=0.03,rounding_size=0.1',
                                facecolor=ORANGE, alpha=a,
                                edgecolor=ORANGE, lw=1.1, zorder=3))
    ax.text(CX - 1.20, y, 'Attention', fontsize=11.5, ha='center',
            va='center', color='#8a4a12', weight='bold', zorder=4)
    ax.add_patch(FancyBboxPatch((CX + 0.12, y - 0.42), 2.16, 0.84,
                                boxstyle='round,pad=0.03,rounding_size=0.1',
                                facecolor=GREEN, alpha=a,
                                edgecolor=GREEN, lw=1.1, zorder=3))
    ax.text(CX + 1.20, y, 'FFN', fontsize=11.5, ha='center',
            va='center', color='#1c6b48', weight='bold', zorder=4)
    ax.plot([CX - 0.12, CX + 0.12], [y, y], color=GREY, lw=1.0, zorder=3)
    # 右侧：这一层留下的隐状态
    ax.plot(CX + W / 2 + 0.62, y, 'o', ms=13, color=col, zorder=5)
    ax.text(CX + W / 2 + 1.05, y, '这一层的隐状态', fontsize=11,
            va='center', color=col)


for y, lb in zip(LAYER_Y[:3], LABELS[:3]):
    layer_box(y, lb, BLUE)
layer_box(LAYER_Y[3], LABELS[3], BLUE)

# 层间箭头
for i in range(3):
    y0, y1 = LAYER_Y[i] - H / 2, LAYER_Y[i + 1] + H / 2
    if i == 2:
        continue
    ax.add_patch(FancyArrow(CX, y0 - 0.04, 0, y1 - y0 + 0.10, width=0.015,
                            head_width=0.16, head_length=0.14,
                            length_includes_head=True, color=GREY, zorder=1))

# 省略号
ax.text(CX, 3.44, '. . .', fontsize=17, ha='center', va='center',
        color=GREY, rotation=90)

# 输入输出
ax.text(CX, 9.24, '输入："答案是"', fontsize=12.5, ha='center', color='#333',
        weight='bold')
ax.add_patch(FancyArrow(CX, 9.02, 0, -0.36, width=0.015, head_width=0.16,
                        head_length=0.14, length_includes_head=True,
                        color=GREY, zorder=1))
ax.add_patch(FancyArrow(CX, LAYER_Y[3] - H / 2 - 0.04, 0, -0.28, width=0.015,
                        head_width=0.16, head_length=0.14,
                        length_includes_head=True, color=GREY, zorder=1))
ax.text(CX, 1.16, '输出下一个字："3"', fontsize=12.5, ha='center',
        color='#333', weight='bold')

# 右侧竖向大括号说明
ax.annotate('', xy=(9.52, 8.55), xytext=(9.52, 2.10),
            arrowprops=dict(arrowstyle='-', color=BLUE, lw=1.6))
ax.text(9.72, 5.2, '43 层 = 43 套零件\n43 个隐状态', fontsize=11.5,
        rotation=90, va='center', ha='center', color=BLUE, linespacing=1.6)

ax.text(5.0, 0.36,
        '每一层都有自己独立的一套 Attention 和 FFN 权重，参数完全不共享。\n'
        '生成一个字，要把这 43 套零件从上到下全跑一遍。',
        fontsize=12, ha='center', color='#333', linespacing=1.7)

fig.suptitle('Attention 不是只算一次——每一层都有自己的一套，各算各的',
             fontsize=15.5, y=0.955)

out = '/home/lmxxf/work/ai-theorys-study/wechat/assets/191_/layers.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved:', out)
