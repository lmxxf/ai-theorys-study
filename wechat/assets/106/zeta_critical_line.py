"""
黎曼 Zeta 函数在临界线 Re(s)=1/2 上的截面图
横轴：虚部 Im(s)，纵轴：|ζ(1/2 + it)|
零点就是曲线掉到 0 的地方
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 计算临界线上的 |ζ(1/2 + it)|
t_vals = np.linspace(0, 36, 2000)
zeta_abs = []

print("正在计算临界线上的 ζ 值...")
for t in t_vals:
    zval = complex(mpmath.zeta(complex(0.5, t)))
    zeta_abs.append(abs(zval))
print("计算完成！")

zeta_abs = np.array(zeta_abs)

# 已知零点虚部
known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351]

fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

# 主曲线
ax.plot(t_vals, zeta_abs, color='red', linewidth=2)

# 零点标注
for z_im in known_zeros:
    ax.plot(z_im, 0, 'o', color='gold', markersize=12,
            markeredgecolor='black', markeredgewidth=1.2, zorder=10)
    ax.annotate(f'{z_im:.2f}', xy=(z_im, 0), xytext=(z_im, -0.3),
                fontsize=10, fontweight='bold', ha='center', va='top',
                color='black')

# 零线
ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)

ax.set_xlabel('Im(s)  虚部', fontsize=14)
ax.set_ylabel('|ζ(1/2 + it)|', fontsize=14)
ax.set_title('临界线上的 ζ 函数——曲线掉到零的地方就是零点', fontsize=16, fontweight='bold')
ax.set_xlim(0, 36)
ax.set_ylim(-0.5, max(zeta_abs) * 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/images/zeta-critical-line.png',
            bbox_inches='tight', facecolor='white')
print("图片已保存!")
