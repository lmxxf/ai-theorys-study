"""
用黎曼零点叠加出素数分布
使用完整的黎曼显式公式（包含所有修正项）
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 计算前 200 个零点
N_ZEROS = 200
zeros_im = []
print(f"正在计算前 {N_ZEROS} 个零点...")
for i in range(1, N_ZEROS + 1):
    z = mpmath.zetazero(i)
    zeros_im.append(float(z.imag))
    if i % 100 == 0:
        print(f"  已计算 {i} 个")
print("零点计算完成！")

# 素数计数函数 π(x) 的真实值
def prime_count(x):
    if x < 2:
        return 0
    count = 0
    for n in range(2, int(x) + 1):
        if all(n % d != 0 for d in range(2, int(n**0.5) + 1)):
            count += 1
    return count

def riemann_R(x):
    """黎曼 R 函数: R(x) = Σ_{n=1}^{∞} μ(n)/n * Li(x^{1/n})
    用 mpmath 的 riemannr 直接算"""
    if x <= 1:
        return 0.0
    return float(mpmath.riemannr(x))

def approx_prime_count(x, n_zeros):
    """
    完整的黎曼显式公式:
    π(x) ≈ R(x) - Σ_ρ R(x^ρ)
    其中 R 是黎曼 R 函数，ρ 是非平凡零点
    零点成对出现: ρ = 1/2 + it 和 ρ* = 1/2 - it
    一对零点的贡献: R(x^ρ) + R(x^{ρ*}) = 2 * Re(R(x^ρ))
    """
    if x <= 1:
        return 0.0
    # 主项
    result = riemann_R(x)
    # 零点修正项
    logx = mpmath.log(x)
    for k in range(min(n_zeros, len(zeros_im))):
        t = zeros_im[k]
        # x^ρ = x^(1/2 + it) = sqrt(x) * x^(it) = sqrt(x) * e^(it*logx)
        xrho = mpmath.exp((0.5 + 1j * t) * logx)
        r_val = mpmath.riemannr(xrho)
        # 一对共轭零点的贡献 = 2 * Re(R(x^ρ))
        result -= 2 * float(r_val.real)
    # 常数修正项: -log(2)
    result -= float(mpmath.log(2))
    return float(result)

# 计算
x_range = np.linspace(2, 100, 2000)
print("正在计算真实素数个数...")
y_true = [prime_count(x) for x in x_range]

print("正在计算主项 R(x)...")
y_approx_0 = [riemann_R(x) for x in x_range]

configs_calc = [5, 10, 20, 30, 100, 200]
y_approx = {}
for nz in configs_calc:
    print(f"正在计算 {nz} 个零点的逼近...")
    y_approx[nz] = [approx_prime_count(x, nz) for x in x_range]

print("计算完成！开始绘图...")

# 绘图：2x4（上面一排小的，下面一排大的）
fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)

# 合并下面一排的后两格为一个大图不好做，改用 2x4
plot_configs = [
    (axes[0, 0], y_approx_0, '0 个零点（主项 R(x)）', '#888888'),
    (axes[0, 1], y_approx[5], '5 个零点', '#e67e22'),
    (axes[0, 2], y_approx[10], '10 个零点', '#f39c12'),
    (axes[0, 3], y_approx[20], '20 个零点', '#e74c3c'),
    (axes[1, 0], y_approx[30], '30 个零点', '#c0392b'),
    (axes[1, 1], y_approx[100], '100 个零点', '#8e44ad'),
    (axes[1, 2], y_approx[200], '200 个零点', '#27ae60'),
]

# 最后一格放说明文字
axes[1, 3].axis('off')
axes[1, 3].text(0.5, 0.5,
    '黑色阶梯 = 真实素数个数\n'
    '彩色曲线 = 零点叠加的逼近\n\n'
    '零点越多\n曲线越像阶梯\n\n'
    '→ 零点是"密码"\n→ 素数是"音乐"\n→ 密码控制音乐',
    transform=axes[1, 3].transAxes,
    fontsize=14, ha='center', va='center',
    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
              edgecolor='goldenrod', alpha=0.9))

for ax, y_data, title, color in plot_configs:
    ax.step(x_range, y_true, where='post', color='#2c3e50',
            linewidth=2.5, label='真实素数个数 π(x)', zorder=5)
    ax.plot(x_range, y_data, color=color, linewidth=1.8,
            label='零点逼近', alpha=0.9, zorder=4)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('x 以下的素数个数', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(2, 100)
    ax.set_ylim(-2, 30)
    ax.grid(True, alpha=0.3)

fig.suptitle('零点如何"雕刻"出素数分布\n——加入的零点越多，曲线越像阶梯',
             fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/images/prime-from-zeros.png',
            bbox_inches='tight', facecolor='white')
print("图片已保存!")
