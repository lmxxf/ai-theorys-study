"""
用黎曼零点叠加出素数分布——公众号 106 期配图
====================================================

最终公式（黎曼显式公式的 Möbius 反演形式）：

  π(x) ≈ R(x) - Σ_ρ 2·Re(Ei(ρ·ln(x)))

  其中：
    R(x) = Σ_{n=1}^{∞} μ(n)/n · Ei(ln(x)/n)   （黎曼 R 函数，μ 是莫比乌斯函数）
    ρ = 1/2 + it                                 （Zeta 函数非平凡零点）
    Ei                                            （指数积分函数）

踩坑记录（2026-03-06，Suzaku）：
----------------------------------------------------

v1（简化公式）：
  用 Li(x) 做主项 + 简化的余弦修正项
  结果：主项偏移大，曲线和阶梯之间距离远
  原因：Li(x) 精度不如 R(x)，修正项系数是简化近似

v2（riemannr 公式，第一次"修正"）：
  改用 mpmath.riemannr(x) 做主项 + mpmath.riemannr(x^ρ) 做零点修正
  还多减了一个 log(2) 常数项
  结果：主项 R(x) 很准，但加 5 个零点后曲线剧烈震荡，比不加还差
  原因：
    1. riemannr(x^ρ) 对复数参数有分支切割（branch cut）问题，
       内部是 Gram 级数展开，不能直接塞复数零点
    2. log(2) 项在 R 函数形式里已被 Möbius 反演吸收，不需要单独减
  教训：R(x) 单独算没问题 ≠ R(x^ρ) 也没问题，复数域里分支切割是杀手

v3（正确公式，当前版本）：
  主项：R(x) = Σ μ(n)/n · Ei(ln(x)/n)  手动用莫比乌斯函数展开
  零点修正：对每个零点 ρ = 1/2+it，减去 2·Re(Ei(ρ·ln(x)))
  关键：用 Ei(ρ·ln(x)) 而非 li(x^ρ) 或 riemannr(x^ρ)
       ln(x) 是实数对数，ρ 是复数，Ei 在这个组合下没有分支切割问题
  结果：正确！零点递增，曲线平稳逼近阶梯，无震荡

参考：
  - Daniel Hutama, Riemann Explicit Formula for Primes (GitHub)
  - mpmath 文档: ei(), zetazero()
  - Wolfram MathWorld: Riemann Prime Counting Function
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 计算前 30 个零点
N_ZEROS = 100
zeros_im = []
print(f"正在计算前 {N_ZEROS} 个零点...")
for i in range(1, N_ZEROS + 1):
    z = mpmath.zetazero(i)
    zeros_im.append(float(z.imag))
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

def R_main(x, N_terms=50):
    """
    黎曼 R 函数主项：R(x) = Σ_{n=1}^{N} μ(n)/n * li(x^{1/n})
    用 Ei(log(x)/n) 计算 li(x^{1/n})
    """
    if x <= 1:
        return 0.0
    logx = mpmath.log(x)
    # 莫比乌斯函数前 50 项
    mu = [0, 1, -1, -1, 0, -1, 1, -1, 0, 0, 1,
          -1, 0, -1, 1, 1, 0, -1, 0, -1, 0,
          1, 1, -1, 0, 0, 1, 0, 0, -1, -1,
          -1, 0, 1, 1, 1, 0, -1, 1, 1, 0,
          -1, -1, -1, 0, 0, -1, 0, -1, 0, 0]
    result = mpmath.mpf(0)
    for n in range(1, min(N_terms, len(mu))):
        if mu[n] != 0:
            result += mpmath.mpf(mu[n]) / n * mpmath.ei(logx / n)
    return float(result)

def approx_prime_count(x, n_zeros):
    """
    π(x) ≈ R(x) - Σ_ρ [用 Ei(ρ·log(x)) 计算的零点修正]

    每个非平凡零点 ρ = 1/2 + it 的贡献：
    用 Möbius 反演：-Σ_{n: μ(n)≠0} (μ(n)/n) * 2*Re(Ei(ρ·log(x)/n))
    简化（只取 n=1 主项）：-2*Re(Ei((1/2+it)·log(x)))
    """
    if x <= 1:
        return 0.0
    result = R_main(x)
    logx = mpmath.log(x)
    for k in range(min(n_zeros, len(zeros_im))):
        t = zeros_im[k]
        rho = mpmath.mpc(0.5, t)
        ei_val = mpmath.ei(rho * logx)
        # 一对共轭零点的贡献
        result -= 2 * float(ei_val.real)
    return float(result)

# 计算
x_range = np.linspace(2, 60, 1200)
print("正在计算真实素数个数...")
y_true = [prime_count(x) for x in x_range]

print("正在计算主项 R(x)...")
y_approx_0 = [R_main(x) for x in x_range]

configs_calc = [10, 50, 100]
y_approx = {}
for nz in configs_calc:
    print(f"正在计算 {nz} 个零点的逼近...")
    y_approx[nz] = [approx_prime_count(x, nz) for x in x_range]

print("计算完成！开始绘图...")

# 绘图：竖排 4 格（手机友好）
fig, axes = plt.subplots(4, 1, figsize=(10, 20), dpi=150)

plot_configs = [
    (axes[0], y_approx_0, '0 个零点（只有平滑主项）', '#888888'),
    (axes[1], y_approx[10], '10 个零点', '#e67e22'),
    (axes[2], y_approx[50], '50 个零点', '#e74c3c'),
    (axes[3], y_approx[100], '100 个零点', '#27ae60'),
]

for ax, y_data, title, color in plot_configs:
    ax.step(x_range, y_true, where='post', color='#2c3e50',
            linewidth=2.5, label='真实素数个数 π(x)', zorder=5)
    ax.plot(x_range, y_data, color=color, linewidth=2,
            label='零点逼近', alpha=0.9, zorder=4)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('x 以下的素数个数', fontsize=13)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xlim(2, 60)
    ax.set_ylim(-2, 20)
    ax.grid(True, alpha=0.3)
    # 标注 x=46 和 x=47
    ax.axvline(x=46, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.axvline(x=47, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.text(46, -1.5, '46', fontsize=9, ha='center', color='gray')
    ax.text(47, -1.5, '47', fontsize=9, ha='center', color='#e74c3c', fontweight='bold')

fig.suptitle('零点如何"雕刻"出素数分布\n——加入的零点越多，曲线越像阶梯',
             fontsize=20, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/assets/106/prime-from-zeros.png',
            bbox_inches='tight', facecolor='white')
print("图片已保存!")
