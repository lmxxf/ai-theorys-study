#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
191_ 期配图：四个模型系列的逐层内在维度曲线（中文版）
数据来源：arXiv 2605.08142 Figure 1，按原图曲线形态与坐标轴量程复刻。
注意：这是形状示意图，不是原始数据点——图中已标注说明。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(7)
x = np.linspace(0, 1, 220)


def jitter(y, amp):
    """加低频抖动，模拟原图曲线的震荡质感"""
    n = len(y)
    noise = rng.normal(0, 1, n)
    k = np.ones(9) / 9.0
    noise = np.convolve(noise, k, mode='same')
    return y + noise * amp


# ---------- Qwen2.5：浅层高位断崖 + 长尾震荡衰减 ----------
def qwen25(start, peak_h, peak_x, decay):
    base = start * np.exp(-decay * x)
    bump = peak_h * np.exp(-((x - peak_x) ** 2) / (2 * 0.06 ** 2))
    y = base + bump
    y = jitter(y, start * 0.035)
    return np.clip(y, 2, None)


# ---------- Qwen3：断崖 + 低位长震荡平台 ----------
def qwen3(start, plateau, mid_bump):
    drop = (start - plateau) * np.exp(-x / 0.055) + plateau
    bump = mid_bump * np.exp(-((x - 0.55) ** 2) / (2 * 0.09 ** 2))
    y = drop + bump
    y = jitter(y, 1.4)
    return np.clip(y, 1.2, None)


# ---------- Gemma3：浴缸形（先升、触底、后半程回升）----------
def gemma3(top, bottom, rise_end):
    early = top + 0.22 * np.exp(-((x - 0.06) ** 2) / (2 * 0.045 ** 2))
    dip = (top - bottom) * np.exp(-((x - 0.27) ** 2) / (2 * 0.115 ** 2))
    recover = (rise_end - bottom) * np.clip((x - 0.30) / 0.70, 0, 1) ** 1.15
    y = early - dip + recover
    return jitter(y, 0.045)


# ---------- DeepSeek-R1-Distill：中层大驼峰 ----------
def deepseek(start, hump_h, hump_x, end):
    base = (start - end) * np.exp(-x / 0.5) + end
    hump = hump_h * np.exp(-((x - hump_x) ** 2) / (2 * 0.11 ** 2))
    y = base + hump
    return np.clip(jitter(y, 6.5), 3, None)


fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.86, bottom=0.13,
                    left=0.075, right=0.985)

# ===== 面板 1：Qwen2.5 =====
ax = axes[0, 0]
reds = plt.cm.Reds(np.linspace(0.35, 0.95, 7))
cfg = [(330, 0, 0.02, 9.0), (150, 55, 0.10, 5.0), (120, 40, 0.13, 4.2),
       (95, 45, 0.16, 3.6), (80, 35, 0.20, 3.0), (68, 30, 0.24, 2.6),
       (58, 26, 0.30, 2.2)]
labels25 = ['0.5B', '1.5B', '3B', '7B', '14B', '32B', '72B']
for (s, ph, px, dc), c, lb in zip(cfg, reds, labels25):
    ax.plot(x, qwen25(s, ph, px, dc), color=c, lw=1.5, label=lb)
ax.set_ylim(0, 350)
ax.set_title('Qwen2.5 系列', fontsize=13, pad=8)
ax.legend(fontsize=7.5, ncol=2, frameon=False, loc='upper right')

# ===== 面板 2：Gemma3 =====
ax = axes[0, 1]
greens = plt.cm.Greens(np.linspace(0.45, 0.95, 4))
for (t, b, r), c, lb in zip([(4.62, 3.12, 3.30), (4.70, 3.18, 3.26),
                             (4.78, 3.20, 3.62), (4.85, 3.05, 4.02)],
                            greens, ['1B', '4B', '12B', '27B']):
    ax.plot(x, gemma3(t, b, r), color=c, lw=1.6, label=lb)
ax.set_ylim(3.0, 5.3)
ax.set_title('Gemma3 系列', fontsize=13, pad=8)
ax.legend(fontsize=8, ncol=2, frameon=False, loc='upper right')

# ===== 面板 3：Qwen3 =====
ax = axes[1, 0]
blues = plt.cm.Blues(np.linspace(0.4, 0.95, 6))
for (s, p, m), c, lb in zip([(45, 6, 4.5), (38, 5.5, 6.0), (33, 5, 3.5),
                             (30, 4.5, 5.0), (26, 4, 3.0), (22, 3.5, 4.0)],
                            blues, ['0.6B', '1.7B', '4B', '8B', '14B', '32B']):
    ax.plot(x, qwen3(s, p, m), color=c, lw=1.5, label=lb)
ax.set_ylim(0, 46)
ax.set_title('Qwen3 系列', fontsize=13, pad=8)
ax.legend(fontsize=7.5, ncol=2, frameon=False, loc='upper right')

# ===== 面板 4：DeepSeek-R1-Distill =====
ax = axes[1, 1]
purples = plt.cm.Purples(np.linspace(0.45, 0.9, 3))
for (s, h, hx, e), c, lb in zip([(150, 55, 0.33, 12), (95, 100, 0.37, 8),
                                 (52, 85, 0.40, 6)],
                                purples, ['1.5B', '14B', '32B']):
    ax.plot(x, deepseek(s, h, hx, e), color=c, lw=1.6, label=lb)
ax.set_ylim(0, 200)
ax.set_title('DeepSeek-R1-Distill-Qwen 系列', fontsize=13, pad=8)
ax.legend(fontsize=8, frameon=False, loc='upper right')

for ax in axes.flat:
    ax.set_xlabel('相对层深（0 = 第一层，1 = 最后一层）', fontsize=9.5)
    ax.set_ylabel('内在维度', fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=8.5)
    for s in ax.spines.values():
        s.set_linewidth(0.7)

fig.suptitle('四个模型系列的逐层内在维度：起点差两个数量级，中途各走各的，终点都收进个位数',
             fontsize=14.5, y=0.955)
fig.text(0.5, 0.037,
         '⚠ 注意四个面板的纵轴量程完全不同（Gemma3 只有 3~5.3，Qwen2.5 到 350）——看形状可以，横向比数值不行',
         ha='center', fontsize=9.5, color='#b3541e')
fig.text(0.5, 0.007,
         '按 arXiv 2605.08142 Figure 1 的曲线形态与坐标轴量程复刻，用于说明趋势，非原始数据点',
         ha='center', fontsize=8, color='#888888')

out = '/home/lmxxf/work/ai-theorys-study/wechat/assets/191_/id-layerwise.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved:', out)
