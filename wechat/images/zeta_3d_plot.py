"""
黎曼 Zeta 函数在复平面上的三维可视化
用于公众号科普插图
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 网格参数
re_min, re_max = -1, 2
im_min, im_max = 0, 35
n_re, n_im = 120, 250

re_vals = np.linspace(re_min, re_max, n_re)
im_vals = np.linspace(im_min, im_max, n_im)
RE, IM = np.meshgrid(re_vals, im_vals)

# 计算 zeta 函数值
print("正在计算 zeta(s) 值...")
Z = np.zeros_like(RE)
for i in range(n_im):
    if i % 50 == 0:
        print(f"  进度: {i}/{n_im}")
    for j in range(n_re):
        s = complex(RE[i, j], IM[i, j])
        try:
            zval = complex(mpmath.zeta(s))
            Z[i, j] = np.log(abs(zval) + 1e-30)
        except:
            Z[i, j] = np.nan
print("计算完成！")

# 裁剪极端值
Z_clipped = np.clip(Z, -6, 5)

# 绘图
fig = plt.figure(figsize=(14, 10), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 曲面图 - 降低透明度让标记更可见
surf = ax.plot_surface(RE, IM, Z_clipped, cmap='coolwarm',
                       alpha=0.75, rstride=1, cstride=1,
                       linewidth=0, antialiased=True,
                       vmin=-5, vmax=4)

# 临界线 Re(s) = 1/2 —— 画在曲面上
re_crit = 0.5
im_crit_fine = np.linspace(im_min, im_max, 500)
z_crit = []
for t in im_crit_fine:
    try:
        zval = complex(mpmath.zeta(complex(re_crit, t)))
        z_crit.append(np.log(abs(zval) + 1e-30))
    except:
        z_crit.append(np.nan)
z_crit = np.clip(np.array(z_crit), -6, 5)

ax.plot([re_crit] * len(im_crit_fine), im_crit_fine, z_crit,
        color='red', linewidth=3, label='临界线 Re(s) = 1/2', zorder=10)

# 已知零点
known_zeros_im = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351]

# 零点标记：用竖线从谷底到曲面上方，更醒目
for z_im in known_zeros_im:
    zval = complex(mpmath.zeta(complex(0.5, z_im)))
    z_val_log = np.clip(np.log(abs(zval) + 1e-30), -6, 5)
    # 竖直虚线从谷底到上方
    ax.plot([0.5, 0.5], [z_im, z_im], [z_val_log, 3.0],
            color='gold', linewidth=2, linestyle='--', zorder=20)
    # 顶部金色大圆点
    ax.scatter([0.5], [z_im], [3.0], color='gold', s=100, zorder=25,
               edgecolors='black', linewidth=1.2, depthshade=False)
    # 在圆点旁标注虚部数值
    ax.text(0.5, z_im, 3.5, f'{z_im:.2f}i',
            fontsize=9, color='black', fontweight='bold',
            ha='center', va='bottom', zorder=30)

# 在图的空白处标注零点数值
zero_lines = ["非平凡零点（临界线上）:"]
for idx, z_im in enumerate(known_zeros_im):
    zero_lines.append(f"  {idx+1}.  s = 1/2 + {z_im:.2f}i")
zero_text = "\n".join(zero_lines)
fig.text(0.02, 0.35, zero_text, fontsize=11,
         fontfamily='WenQuanYi Micro Hei',
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                   edgecolor='goldenrod', alpha=0.9))

# 坐标轴标签
ax.set_xlabel('Re(s)  实部', fontsize=13, labelpad=12)
ax.set_ylabel('Im(s)  虚部', fontsize=13, labelpad=12)
ax.set_zlabel(r'$\log|\zeta(s)|$', fontsize=13, labelpad=10)

ax.set_title('黎曼 ζ 函数的零点地形', fontsize=20, fontweight='bold', pad=25)

# 视角
ax.view_init(elev=30, azim=-60)

# 颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=15, pad=0.1)
cbar.set_label(r'$\log|\zeta(s)|$', fontsize=12)

# 图例
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/images/zeta-3d.png',
            bbox_inches='tight', facecolor='white')
print("图片已保存!")
