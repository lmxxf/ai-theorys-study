"""
mHC vs HC 可视化对比
====================================
DeepSeek mHC 论文核心思想演示

- 残差连接 (Residual): 跳跃连接，稳定但单一路径
- HC (超连接): 加宽残差流，多路径并行，但容易发散
- mHC (流形约束超连接): 把 HC 投影回流形表面

论文: arXiv:2512.24880
更多: https://github.com/lmxxf/ai-theorys-study

作者: 靳岩岩的AI学习笔记
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_to_sphere(points, radius=1.0):
    """投影到球面上 (mHC 的核心操作)"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # 避免除零
    return points / norms * radius

def simulate_residual(start, n_layers=10, noise_scale=0.1):
    """模拟残差连接: 稳定但单一路径"""
    trajectory = [start.copy()]
    current = start.copy()

    for _ in range(n_layers):
        # 残差连接: 小幅更新 + 保持原信号
        delta = np.random.randn(3) * noise_scale
        current = current + delta * 0.3  # 小步更新
        trajectory.append(current.copy())

    return np.array(trajectory)

def simulate_hc(start, n_layers=10, noise_scale=0.3, n_streams=4):
    """模拟 HC (超连接): 多路径但容易发散"""
    trajectories = []

    for stream in range(n_streams):
        trajectory = [start.copy()]
        current = start.copy()

        for _ in range(n_layers):
            # HC: 更大的更新幅度，多路径
            delta = np.random.randn(3) * noise_scale
            # 不同路径有不同的偏移
            current = current + delta + np.random.randn(3) * 0.1 * stream
            trajectory.append(current.copy())

        trajectories.append(np.array(trajectory))

    return trajectories

def simulate_mhc(start, n_layers=10, noise_scale=0.3, n_streams=4, radius=1.5):
    """模拟 mHC: 多路径 + 流形约束 (投影回球面)"""
    trajectories = []

    for stream in range(n_streams):
        trajectory = [normalize_to_sphere(start.reshape(1, -1), radius)[0]]
        current = start.copy()

        for _ in range(n_layers):
            # 和 HC 一样的更新
            delta = np.random.randn(3) * noise_scale
            current = current + delta + np.random.randn(3) * 0.1 * stream
            # mHC 的关键: 投影回流形 (球面)
            current_normalized = normalize_to_sphere(current.reshape(1, -1), radius)[0]
            trajectory.append(current_normalized.copy())
            current = current_normalized

        trajectories.append(np.array(trajectory))

    return trajectories

def draw_sphere(ax, radius=1.5, alpha=0.1):
    """画一个半透明球面表示流形"""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color='lightblue')

def main():
    np.random.seed(42)

    # 起点
    start = np.array([1.0, 0.5, 0.3])
    start_normalized = normalize_to_sphere(start.reshape(1, -1), radius=1.5)[0]

    # 模拟三种方法
    residual_traj = simulate_residual(start, n_layers=15)
    hc_trajs = simulate_hc(start, n_layers=15, n_streams=4)
    mhc_trajs = simulate_mhc(start_normalized, n_layers=15, n_streams=4, radius=1.5)

    # 创建图
    fig = plt.figure(figsize=(16, 5))

    # ===== 子图1: 残差连接 =====
    # 注: 图里的标题用英文，因为 matplotlib 默认不支持中文字体
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Residual Connection\n(Stable, Single Path)', fontsize=12, fontweight='bold')

    ax1.plot(residual_traj[:, 0], residual_traj[:, 1], residual_traj[:, 2],
             'b-', linewidth=2, label='Single Path')
    ax1.scatter(*start, c='green', s=100, marker='o', label='Start')
    ax1.scatter(*residual_traj[-1], c='red', s=100, marker='*', label='End')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim([-2, 3])
    ax1.set_ylim([-2, 3])
    ax1.set_zlim([-2, 3])

    # ===== 子图2: HC (野马脱缰) =====
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('HC (Hyper-Connections)\n(Multi-Path, Divergent)', fontsize=12, fontweight='bold')

    colors = ['blue', 'orange', 'green', 'purple']
    for i, traj in enumerate(hc_trajs):
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                 color=colors[i], linewidth=1.5, alpha=0.7, label=f'Path {i+1}')
        ax2.scatter(*traj[-1], c='red', s=50, marker='x')

    ax2.scatter(*start, c='green', s=100, marker='o', label='Start')

    # 发散警告
    ax2.text(2.5, 2.5, 2.5, 'DIVERGE!', fontsize=10, color='red')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim([-2, 4])
    ax2.set_ylim([-2, 4])
    ax2.set_zlim([-2, 4])

    # ===== 子图3: mHC (流形约束) =====
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('mHC (Manifold-Constrained)\n(Multi-Path + Stable)', fontsize=12, fontweight='bold')

    # 画球面 (流形)
    draw_sphere(ax3, radius=1.5, alpha=0.15)

    for i, traj in enumerate(mhc_trajs):
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                 color=colors[i], linewidth=1.5, alpha=0.8, label=f'Path {i+1}')
        ax3.scatter(*traj[-1], c='red', s=50, marker='*')

    ax3.scatter(*start_normalized, c='green', s=100, marker='o', label='Start')
    ax3.text(0, 0, 2.2, 'Manifold', fontsize=10, color='blue', ha='center')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_xlim([-2, 2])
    ax3.set_ylim([-2, 2])
    ax3.set_zlim([-2, 2])

    plt.tight_layout()

    # 保存图片
    output_path = '/home/lmxxf/work/ai-theorys-study/script/mhc_vs_hc.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'图片已保存到: {output_path}')

    # 也保存 SVG 版本
    svg_path = '/home/lmxxf/work/ai-theorys-study/script/mhc_vs_hc.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f'SVG 已保存到: {svg_path}')

    plt.show()

    # 打印说明
    print("""
====================================
mHC vs HC 核心区别
====================================

【残差连接 (2015, 何恺明)】
  - 单一路径
  - 稳定，不会发散
  - 但信息容量有限

【HC - 超连接 (2024, 字节)】
  - 多路径并行，信息容量大
  - 问题: 破坏了"恒等映射"
  - 结果: 模型越大越容易发散 (野马脱缰)

【mHC - 流形约束超连接 (2025, DeepSeek)】
  - 保留 HC 的多路径优点
  - 关键创新: 把所有路径投影回"流形" (曲面)
  - 结果: 既宽又稳

用人话说:
  HC = 把车道加宽，但车乱跑
  mHC = 加宽车道 + 装上护栏

论文: arXiv:2512.24880
更多: https://github.com/lmxxf/ai-theorys-study
====================================
""")

if __name__ == '__main__':
    main()
