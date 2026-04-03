"""
141 配图2：等效权重随 λ 和 α 的变化
核心论点的最直接可视化：λ 改不了等效权重，α 能
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False

L_CE = 2.0

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# λ 从 0.1 到 20
lam = np.linspace(0.1, 20, 500)
# 均衡点处：exp(-s)*L_CE = 1，所以 λ*exp(-s)*L_CE = 1，等效权重恒为 1
eff_lam = np.ones_like(lam)

# α 从 0.1 到 20
alpha = np.linspace(0.1, 20, 500)
# 均衡点处：exp(-s)*L_CE = α，等效权重 = α
eff_alpha = alpha

ax.plot(lam, eff_lam, '#F44336', linewidth=3, linestyle='--',
        label='Strategy 2:  effective weight vs $\\lambda$  (flat!)')
ax.plot(alpha, eff_alpha, '#4CAF50', linewidth=3,
        label='Strategy 3:  effective weight vs $\\alpha$  (linear)')

# 标注几个关键点
for v in [1, 5, 10]:
    ax.plot(v, 1, 'o', color='#F44336', markersize=10, zorder=5)
    ax.plot(v, v, 'o', color='#4CAF50', markersize=10, zorder=5)

# 标注文字
ax.annotate('$\\lambda$=10, but weight still = 1\n"Spring bounces back"',
            xy=(10, 1), xytext=(13, 3),
            fontsize=11, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))

ax.annotate('$\\alpha$=10 $\\Rightarrow$ weight = 10\n"Stiffer spring"',
            xy=(10, 10), xytext=(13, 13),
            fontsize=11, color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

ax.set_xlabel('$\\lambda$  or  $\\alpha$', fontsize=14)
ax.set_ylabel('Effective weight at equilibrium:  exp($-$s) $\\cdot$ L', fontsize=13)
ax.set_title('Why $\\lambda$ does nothing but $\\alpha$ works', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.grid(True, alpha=0.3)

# 加一条 y=x 的参考淡线
ax.plot([0, 20], [0, 20], color='gray', linewidth=0.5, linestyle=':')

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/assets/141/141_lambda_vs_alpha.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("done")
