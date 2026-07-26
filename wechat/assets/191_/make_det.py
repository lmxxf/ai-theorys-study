#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""191_ 期配图：行列式 = 体积，det=0 = 塌了"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5))
fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.22, left=0.06, right=0.97)

BLUE, ORANGE, GREY = '#2f6fb5', '#e08a3c', '#9aa3ad'


def arrow(ax, dx, dy, color, label, lx, ly):
    ax.add_patch(FancyArrow(0, 0, dx, dy, width=0.055, head_width=0.26,
                            head_length=0.32, length_includes_head=True,
                            color=color, zorder=5))
    ax.text(lx, ly, label, fontsize=12.5, color=color, weight='bold', zorder=6)


def frame(ax, title, sub, subcolor):
    ax.set_xlim(-0.9, 5.2)
    ax.set_ylim(-1.15, 4.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='#ccd2d8', lw=0.9, zorder=0)
    ax.axvline(0, color='#ccd2d8', lw=0.9, zorder=0)
    ax.set_title(title, fontsize=14, pad=12)
    ax.text(2.15, -0.92, sub, fontsize=13, color=subcolor,
            ha='center', weight='bold')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ---- 左：撑开的平行四边形 ----
ax1.add_patch(Polygon([[0, 0], [2, 0], [2, 3], [0, 3]],
                      closed=True, facecolor=BLUE, alpha=0.16,
                      edgecolor=BLUE, lw=1.3, ls='--', zorder=1))
ax1.text(1.0, 1.45, '面积 = 6', fontsize=15, color=BLUE,
         ha='center', weight='bold', zorder=7)
arrow(ax1, 2, 0, BLUE, 'a = (2, 0)', 2.15, -0.42)
arrow(ax1, 0, 3, ORANGE, 'b = (0, 3)', 0.16, 3.42)
frame(ax1, '两个方向不同的向量，张开一个平行四边形',
      'det = 6　撑得开，分得清', BLUE)

# ---- 右：共线塌成一条线 ----
ax2.plot([0, 4], [0, 0], color=GREY, lw=9, alpha=0.30,
         solid_capstyle='butt', zorder=1)
arrow(ax2, 4, 0, ORANGE, 'b = (4, 0)', 3.05, 0.42)
arrow(ax2, 2, 0, BLUE, 'a = (2, 0)', 0.55, -0.55)
ax2.text(2.0, 1.5, '被压扁了\n面积 = 0', fontsize=15, color='#c0392b',
         ha='center', weight='bold', zorder=7)
frame(ax2, '两个方向相同的向量，压成一条线',
      'det = 0　塌了，区分不出来', '#c0392b')

fig.suptitle('行列式量的是"这堆向量撑开了多大地盘"——塌了就是 0',
             fontsize=15.5, y=0.945)

out = '/home/lmxxf/work/ai-theorys-study/wechat/assets/191_/determinant.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved:', out)
