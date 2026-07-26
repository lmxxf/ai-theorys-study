#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""191_ 期配图：八个 benchmark 上，只看维度 / 加信息体积 / 三项合一 的相关性对比
数据来自 arXiv 2605.08142 Figure 5B 的柱子标签（论文印刷值）"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 论文 Figure 5B 印刷值
DATA = [
    ('AIME\'25',            0.23, 0.68, 0.76),
    ('Arena-Hard',          0.69, 0.79, 0.88),
    ('AutoLogi',            0.26, 0.69, 0.92),
    ('BFCL v3',             0.46, 0.74, 0.89),
    ('Creative Writing v3', 0.75, 0.88, 0.94),
    ('GPQA-Diamond',        0.29, 0.73, 0.91),
    ('LiveBench',           0.31, 0.71, 0.90),
    ('LiveCodeBench v5',    0.32, 0.68, 0.89),
]

GREY, BLUE, NAVY = '#c9ced4', '#5fa2dd', '#1f3f66'

fig, ax = plt.subplots(figsize=(11.6, 5.9))
fig.subplots_adjust(top=0.80, bottom=0.20, left=0.07, right=0.985)

x = np.arange(len(DATA))
w = 0.26
v1 = [d[1] for d in DATA]
v2 = [d[2] for d in DATA]
v3 = [d[3] for d in DATA]

b1 = ax.bar(x - w, v1, w, color=GREY, edgecolor='#a8afb7', lw=0.8,
            label='只看维度（几何压缩）')
b2 = ax.bar(x, v2, w, color=BLUE, edgecolor='#3d84c4', lw=0.8,
            label='维度 + 信息体积')
b3 = ax.bar(x + w, v3, w, color=NAVY, edgecolor='#132a45', lw=0.8,
            label='三项合一（完整的 H）')

for bars, vals in ((b1, v1), (b2, v2), (b3, v3)):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.022, f'{v:.2f}',
                ha='center', fontsize=9.5, color='#333')

ax.axhline(0.9, color='#c0392b', lw=1.2, ls='--', alpha=0.75, zorder=0)
ax.text(7.62, 0.915, '0.9', fontsize=10, color='#c0392b', ha='left')

ax.set_ylim(0, 1.10)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel('与 benchmark 成绩的相关性（Spearman ρ）', fontsize=11.5)
ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in DATA], fontsize=10.5)
ax.grid(axis='y', alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_linewidth(0.8)
ax.legend(fontsize=10.5, frameon=False, ncol=3, loc='upper center',
          bbox_to_anchor=(0.5, 1.10))

# 重点标注
ax.annotate('最难的数学题上，\n光看降维几乎等于瞎猜',
            xy=(-0.26, 0.23), xytext=(0.42, 0.40),
            fontsize=10.5, color='#b3541e', ha='left', linespacing=1.5,
            arrowprops=dict(arrowstyle='->', color='#b3541e', lw=1.3))

fig.suptitle('降维一项撑不起来，加上信息体积才立得住',
             fontsize=16, y=0.965)
fig.text(0.5, 0.055,
         '灰柱到深蓝柱的跃升，就是这篇论文的全部论点：'
         '只看"降到几维"（0.23~0.75）忽高忽低，三项合一才稳定在 0.9 上下',
         ha='center', fontsize=11, color='#333')
fig.text(0.5, 0.012,
         '数据为 arXiv 2605.08142 Figure 5B 的标注值。'
         '注意八项里只有四项真正达到 0.90，AIME\'25 只有 0.76',
         ha='center', fontsize=9.5, color='#888')

out = '/home/lmxxf/work/ai-theorys-study/wechat/assets/191_/spearman.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved:', out)
