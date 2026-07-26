#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""191_ 期配图：三片点云 → 特征值 → 维度和体积"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrow

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE, ORANGE, RED, GREEN = '#2f6fb5', '#e08a3c', '#c0392b', '#2e9e6b'

CASES = [
    dict(title='扁云', pts=[(2, 0), (0, 1), (-2, 0)],
         l1=8, l2=1, col=BLUE,
         note='横向铺得开，纵向窄', dim='2 个方向都在用', det='体积 = 8 × 1 = 8'),
    dict(title='塌成一条线', pts=[(2, 0), (0, 0), (-2, 0)],
         l1=8, l2=0, col=RED,
         note='纵向一点都没铺开', dim='实际只有 1 个方向', det='体积 = 8 × 0 = 0'),
    dict(title='方方正正', pts=[(2, 2), (2, -2), (-2, 2), (-2, -2)],
         l1=16, l2=16, col=GREEN,
         note='两个方向一样长', dim='2 个方向都在用', det='体积 = 16 × 16 = 256'),
]

fig, axes = plt.subplots(1, 3, figsize=(12.4, 6.6))
fig.subplots_adjust(top=0.82, bottom=0.40, wspace=0.24, left=0.05, right=0.97)

for ax, c in zip(axes, CASES):
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.6, 3.6)
    ax.set_aspect('equal')
    ax.axhline(0, color='#d5dae0', lw=0.9, zorder=0)
    ax.axvline(0, color='#d5dae0', lw=0.9, zorder=0)

    # 椭圆示意铺开范围
    import math
    w = 2 * math.sqrt(c['l1']) * 0.86
    h = 2 * math.sqrt(c['l2']) * 0.86 if c['l2'] > 0 else 0.16
    ax.add_patch(Ellipse((0, 0), w, h, facecolor=c['col'], alpha=0.13,
                         edgecolor=c['col'], lw=1.4, ls='--', zorder=1))

    for (px, py) in c['pts']:
        ax.plot(px, py, 'o', ms=13, color=c['col'], zorder=5)

    ax.set_title(c['title'], fontsize=14, pad=10, color=c['col'])
    ax.text(0, -4.35, c['note'], fontsize=11, ha='center', color='#555')
    ax.text(0, -5.25, f"λ1 = {c['l1']}    λ2 = {c['l2']}", fontsize=13.5,
            ha='center', color=c['col'], weight='bold')
    ax.text(0, -6.1, c['dim'], fontsize=11, ha='center', color='#333')
    ax.text(0, -6.9, c['det'], fontsize=11, ha='center', color='#333')

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

axes[1].text(0, 4.55, '只要有一个 λ 是 0，体积就归零', fontsize=11.5,
             ha='center', color=RED, weight='bold')

fig.suptitle('特征值 λ = 这片点云在各个方向上分别铺开了多少',
             fontsize=16, y=0.945)
fig.text(0.5, 0.085,
         '数一数有几个 λ 明显大于 0 → 这就是内在维度；把所有 λ 乘起来 → 这就是体积（行列式）',
         ha='center', fontsize=12, color='#333')
fig.text(0.5, 0.028,
         '真实情况只是把 2 维换成几千维：几百个点、几千个 λ，其中绝大多数接近 0，'
         '真正明显大于 0 的只有两三个到七八个',
         ha='center', fontsize=10, color='#888')

out = '/home/lmxxf/work/ai-theorys-study/wechat/assets/191_/eigen.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved:', out)
