"""
mHC vs HC Visualization
====================================
DeepSeek's mHC paper core idea demo

- Residual Connection: skip connection, stable but single path
- HC (Hyper-Connections): wider residual stream, multi-path, but unstable
- mHC (Manifold-Constrained HC): project HC back to manifold surface

Paper: arXiv:2512.24880
More: https://github.com/lmxxf/ai-theorys-study

Author: Jin Yanyan's AI Study Notes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_to_sphere(points, radius=1.0):
    """Project to sphere surface (mHC core operation)"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return points / norms * radius

def simulate_residual(start, n_layers=10, noise_scale=0.1):
    """Simulate Residual Connection: stable but single path"""
    trajectory = [start.copy()]
    current = start.copy()

    for _ in range(n_layers):
        # Residual: small update + keep original signal
        delta = np.random.randn(3) * noise_scale
        current = current + delta * 0.3  # small step
        trajectory.append(current.copy())

    return np.array(trajectory)

def simulate_hc(start, n_layers=10, noise_scale=0.3, n_streams=4):
    """Simulate HC (Hyper-Connections): multi-path but divergent"""
    trajectories = []

    for stream in range(n_streams):
        trajectory = [start.copy()]
        current = start.copy()

        for _ in range(n_layers):
            # HC: larger update, multi-path
            delta = np.random.randn(3) * noise_scale
            # different streams have different offsets
            current = current + delta + np.random.randn(3) * 0.1 * stream
            trajectory.append(current.copy())

        trajectories.append(np.array(trajectory))

    return trajectories

def simulate_mhc(start, n_layers=10, noise_scale=0.3, n_streams=4, radius=1.5):
    """Simulate mHC: multi-path + manifold constraint (project to sphere)"""
    trajectories = []

    for stream in range(n_streams):
        trajectory = [normalize_to_sphere(start.reshape(1, -1), radius)[0]]
        current = start.copy()

        for _ in range(n_layers):
            # same update as HC
            delta = np.random.randn(3) * noise_scale
            current = current + delta + np.random.randn(3) * 0.1 * stream
            # mHC key: project back to manifold (sphere)
            current_normalized = normalize_to_sphere(current.reshape(1, -1), radius)[0]
            trajectory.append(current_normalized.copy())
            current = current_normalized

        trajectories.append(np.array(trajectory))

    return trajectories

def draw_sphere(ax, radius=1.5, alpha=0.1):
    """Draw a translucent sphere representing the manifold"""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color='lightblue')

def main():
    np.random.seed(42)

    # Starting point
    start = np.array([1.0, 0.5, 0.3])
    start_normalized = normalize_to_sphere(start.reshape(1, -1), radius=1.5)[0]

    # Simulate three methods
    residual_traj = simulate_residual(start, n_layers=15)
    hc_trajs = simulate_hc(start, n_layers=15, n_streams=4)
    mhc_trajs = simulate_mhc(start_normalized, n_layers=15, n_streams=4, radius=1.5)

    # Create figure
    fig = plt.figure(figsize=(16, 5))

    # ===== Subplot 1: Residual Connection =====
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

    # ===== Subplot 2: HC (Wild Horse) =====
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('HC (Hyper-Connections)\n(Multi-Path, Divergent)', fontsize=12, fontweight='bold')

    colors = ['blue', 'orange', 'green', 'purple']
    for i, traj in enumerate(hc_trajs):
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                 color=colors[i], linewidth=1.5, alpha=0.7, label=f'Path {i+1}')
        ax2.scatter(*traj[-1], c='red', s=50, marker='x')

    ax2.scatter(*start, c='green', s=100, marker='o', label='Start')

    # Warning text for divergence
    ax2.text(2.5, 2.5, 2.5, 'DIVERGE!', fontsize=10, color='red')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim([-2, 4])
    ax2.set_ylim([-2, 4])
    ax2.set_zlim([-2, 4])

    # ===== Subplot 3: mHC (Manifold Constrained) =====
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('mHC (Manifold-Constrained)\n(Multi-Path + Stable)', fontsize=12, fontweight='bold')

    # Draw sphere (manifold)
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

    # Save image
    output_path = '/home/lmxxf/work/ai-theorys-study/script/mhc_vs_hc.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Image saved to: {output_path}')

    # Also save SVG version
    svg_path = '/home/lmxxf/work/ai-theorys-study/script/mhc_vs_hc.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f'SVG saved to: {svg_path}')

    plt.show()

    # Print explanation
    print("""
====================================
mHC vs HC: Key Differences
====================================

[Residual Connection (2015, He Kaiming)]
  - Single path
  - Stable, no divergence
  - But limited information capacity

[HC - Hyper-Connections (2024, ByteDance)]
  - Multi-path, high information capacity
  - Problem: breaks "identity mapping"
  - Result: larger models diverge more easily (wild horse)

[mHC - Manifold-Constrained HC (2025, DeepSeek)]
  - Keep HC's multi-path advantage
  - Key innovation: project all paths back to "manifold" (surface)
  - Result: wide AND stable

In plain words:
  HC = widen the lanes, but cars run wild
  mHC = widen the lanes + install guardrails

Paper: arXiv:2512.24880
More: https://github.com/lmxxf/ai-theorys-study
====================================
""")

if __name__ == '__main__':
    main()
