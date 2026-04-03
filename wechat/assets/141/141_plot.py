"""
141 配图：同调不确定性加权的均衡点分析
展示 L = exp(-s)*L_CE + α*s 中第一项和第二项的拉锯
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

# 尝试加载中文字体
_zh_font = None
for _fname in ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Serif CJK SC']:
    try:
        _zh_font = FontProperties(family=_fname)
        break
    except:
        continue

matplotlib.rcParams['axes.unicode_minus'] = False
if _zh_font:
    matplotlib.rcParams['font.family'] = _zh_font.get_name()

L_CE = 2.0  # 假设交叉熵损失值
s = np.linspace(-2, 4, 500)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

fp = dict(fontproperties=_zh_font) if _zh_font else {}

# ========== 图1：两项的拉锯（α=1） ==========
ax = axes[0]
term1 = np.exp(-s) * L_CE
term2 = s
total = term1 + term2

# 均衡点
s_eq = np.log(L_CE)
L_eq = np.exp(-s_eq) * L_CE + s_eq

ax.plot(s, term1, 'b-', linewidth=2.5, label='Term 1: exp(-s)$\\times$L_CE')
ax.plot(s, term2, 'r-', linewidth=2.5, label='Term 2: s')
ax.plot(s, total, 'k-', linewidth=3, label='Total: L(s)')
ax.plot(s_eq, L_eq, 'ko', markersize=12, zorder=5)
ax.annotate(f'Equilibrium\ns={s_eq:.2f}', xy=(s_eq, L_eq), xytext=(s_eq+0.8, L_eq+1.5),
            fontsize=12, ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('s = log($\\sigma^2$)', fontsize=13)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('Term 1 vs Term 2  ($\\alpha$=1, L_CE=2.0)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 8)
ax.grid(True, alpha=0.3)

# ========== 图2：策略2——乘λ，均衡点不动 ==========
ax = axes[1]
offsets_lam = {1: (1.0, 1.5), 5: (0.6, 1.8), 10: (-2.0, 1.5)}
for lam, color, ls in [(1, '#2196F3', '-'), (5, '#FF9800', '--'), (10, '#F44336', ':')]:
    total_lam = lam * np.exp(-s) * L_CE + s
    s_eq_lam = np.log(lam * L_CE)
    L_eq_lam = lam * np.exp(-s_eq_lam) * L_CE + s_eq_lam
    ax.plot(s, total_lam, color=color, linewidth=2.5, linestyle=ls, label=f'$\\lambda$={lam}')
    ax.plot(s_eq_lam, L_eq_lam, 'o', color=color, markersize=10, zorder=5)
    # 标注均衡点处的等效权重
    eff = lam * np.exp(-s_eq_lam) * L_CE
    ox, oy = offsets_lam[lam]
    ax.annotate(f'$\\lambda$$\\cdot$e$^{{-s}}$$\\cdot$L = {eff:.1f}', xy=(s_eq_lam, L_eq_lam),
                xytext=(s_eq_lam+ox, L_eq_lam+oy),
                fontsize=10, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('s = log($\\sigma^2$)', fontsize=13)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('Strategy 2: scale loss by $\\lambda$ (product always = 1)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 8)
ax.grid(True, alpha=0.3)

# ========== 图3：策略3——改α，均衡点真的动了 ==========
ax = axes[2]
offsets_alpha = {1: (1.0, 1.5), 2: (-1.8, 1.8), 3: (-2.0, 1.2)}
for alpha, color, ls in [(1, '#2196F3', '-'), (2, '#4CAF50', '--'), (3, '#9C27B0', ':')]:
    total_alpha = np.exp(-s) * L_CE + alpha * s
    s_eq_a = np.log(L_CE / alpha)
    L_eq_a = np.exp(-s_eq_a) * L_CE + alpha * s_eq_a
    ax.plot(s, total_alpha, color=color, linewidth=2.5, linestyle=ls, label=f'$\\alpha$={alpha}')
    ax.plot(s_eq_a, L_eq_a, 'o', color=color, markersize=10, zorder=5)
    # 标注均衡点处的等效权重
    eff_a = np.exp(-s_eq_a) * L_CE
    ox, oy = offsets_alpha[alpha]
    ax.annotate(f'e$^{{-s}}$$\\cdot$L = {eff_a:.1f}', xy=(s_eq_a, L_eq_a),
                xytext=(s_eq_a+ox, L_eq_a+oy),
                fontsize=10, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('s = log($\\sigma^2$)', fontsize=13)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('Strategy 3: change $\\alpha$ (equilibrium shifts!)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 8)
ax.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/141_equilibrium.png', dpi=150, bbox_inches='tight')
plt.close()
print("done")
