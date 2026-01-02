"""
Birkhoff 多面体可视化
====================================
n=3 的双随机矩阵空间

什么是双随机矩阵？
- 每行加起来 = 1
- 每列加起来 = 1
- 所有元素 ≥ 0

蛋糕比喻：3 个人分 3 块蛋糕，每人分到的总量=1，每块蛋糕被分完=1

Birkhoff 多面体：
- 顶点 = 排列矩阵（每人拿整块蛋糕，不切）
- 边 = 两个排列只差一次对换
- 内部 = 切蛋糕分着吃

n=3 时：6 个顶点（3!=6），12 条边

作者: 靳岩岩的AI学习笔记
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import permutations

def get_permutation_matrices():
    """获取所有 3x3 排列矩阵"""
    perms = list(permutations([0, 1, 2]))
    matrices = []
    labels = []
    for perm in perms:
        m = np.zeros((3, 3))
        for i, j in enumerate(perm):
            m[i, j] = 1
        matrices.append(m)
        # 标签：谁拿了哪块蛋糕
        labels.append(f"({perm[0]+1},{perm[1]+1},{perm[2]+1})")
    return matrices, labels

def matrix_to_3d(m):
    """
    把 3x3 双随机矩阵投影到 3D

    用 4 个独立元素，线性组合到 3D
    投影矩阵经过调整，让 6 个顶点分布均匀
    """
    v = np.array([m[0, 0], m[0, 1], m[1, 0], m[1, 1]])
    proj = np.array([
        [1, -0.5, -0.5, 0],
        [0, 0.866, -0.866, 0],
        [0.5, 0.5, 0.5, -1.5]
    ])
    return proj @ v

def get_edges(matrices):
    """计算相邻的排列矩阵（差一次对换）"""
    edges = []
    n = len(matrices)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(np.abs(matrices[i] - matrices[j]))
            # 差一次对换 = 4 个元素变化
            if diff == 4:
                edges.append((i, j))
    return edges

def main():
    # 获取 6 个排列矩阵
    matrices, labels = get_permutation_matrices()
    coords = np.array([matrix_to_3d(m) for m in matrices])
    edges = get_edges(matrices)

    print("=" * 50)
    print("Birkhoff 多面体：3x3 双随机矩阵的几何")
    print("=" * 50)
    print("\n6 个顶点（排列矩阵）：")
    print("标签含义：(人1拿蛋糕?, 人2拿蛋糕?, 人3拿蛋糕?)")
    print()
    for i, (m, label, coord) in enumerate(zip(matrices, labels, coords)):
        print(f"顶点 {i}: {label}")
        print(f"  矩阵:\n{m.astype(int)}")
        print(f"  3D坐标: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
        print()

    print(f"边数: {len(edges)}")
    print("边（差一次对换的排列对）:")
    for i, j in edges:
        print(f"  {labels[i]} <-> {labels[j]}")

    # 创建图
    fig = plt.figure(figsize=(14, 6))

    # ===== 左图：多面体结构 =====
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Birkhoff Polytope (n=3)\n6 Vertices, 12 Edges', fontsize=14, fontweight='bold')

    # 画边
    for i, j in edges:
        ax1.plot([coords[i, 0], coords[j, 0]],
                 [coords[i, 1], coords[j, 1]],
                 [coords[i, 2], coords[j, 2]],
                 'b-', alpha=0.6, linewidth=2)

    # 画顶点
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                c='red', s=150, marker='o', alpha=1.0, zorder=10)

    # 标注顶点
    for i, (coord, label) in enumerate(zip(coords, labels)):
        ax1.text(coord[0] + 0.1, coord[1] + 0.1, coord[2] + 0.1,
                 label, fontsize=9, color='darkred')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-2.0, 1.5])

    # ===== 右图：蛋糕比喻 =====
    ax2 = fig.add_subplot(122)
    ax2.set_title('Cake Analogy\n(3 People, 3 Cakes)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 画一个示例双随机矩阵
    example_matrix = np.array([
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])

    # 画表格
    cell_size = 0.15
    start_x, start_y = 0.25, 0.55

    # 表头
    ax2.text(start_x - 0.08, start_y + 0.12, 'Person\\Cake', fontsize=10, ha='center', fontweight='bold')
    for j in range(3):
        ax2.text(start_x + j * cell_size + cell_size/2, start_y + 0.12,
                 f'Cake {j+1}', fontsize=10, ha='center', fontweight='bold')

    for i in range(3):
        ax2.text(start_x - 0.08, start_y - i * cell_size - cell_size/2,
                 f'Person {i+1}', fontsize=10, ha='center', fontweight='bold')
        for j in range(3):
            # 画格子
            rect = plt.Rectangle((start_x + j * cell_size, start_y - (i+1) * cell_size),
                                  cell_size, cell_size,
                                  facecolor=plt.cm.Blues(example_matrix[i, j]),
                                  edgecolor='black', linewidth=1)
            ax2.add_patch(rect)
            # 写数值
            ax2.text(start_x + j * cell_size + cell_size/2,
                     start_y - i * cell_size - cell_size/2,
                     f'{example_matrix[i, j]:.1f}', fontsize=11, ha='center', va='center')

    # 行和列约束标注
    ax2.text(start_x + 3.3 * cell_size, start_y - 0.5 * cell_size, '= 1.0', fontsize=10)
    ax2.text(start_x + 3.3 * cell_size, start_y - 1.5 * cell_size, '= 1.0', fontsize=10)
    ax2.text(start_x + 3.3 * cell_size, start_y - 2.5 * cell_size, '= 1.0', fontsize=10)

    ax2.text(start_x + 0.5 * cell_size, start_y - 3.3 * cell_size, '= 1.0', fontsize=10, ha='center')
    ax2.text(start_x + 1.5 * cell_size, start_y - 3.3 * cell_size, '= 1.0', fontsize=10, ha='center')
    ax2.text(start_x + 2.5 * cell_size, start_y - 3.3 * cell_size, '= 1.0', fontsize=10, ha='center')

    # 说明文字
    explanation = """
Doubly Stochastic Matrix:
• Each row sums to 1 (each person gets total = 1)
• Each column sums to 1 (each cake is fully distributed)
• All entries ≥ 0 (no negative cake!)

Birkhoff Polytope:
• Vertices = Permutation matrices (whole cakes, no cutting)
• Interior = Cut and share
• mHC constrains weights to this polytope
    """
    ax2.text(0.05, 0.05, explanation, fontsize=10, family='monospace',
             verticalalignment='bottom', transform=ax2.transAxes)

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # 保存
    output_path = '/home/lmxxf/work/ai-theorys-study/script/birkhoff_polytope.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存到: {output_path}")

    svg_path = '/home/lmxxf/work/ai-theorys-study/script/birkhoff_polytope.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"SVG 已保存到: {svg_path}")

    plt.show()

if __name__ == '__main__':
    main()
