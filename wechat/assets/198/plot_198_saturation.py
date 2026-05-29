import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
cjk_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['font.sans-serif'] = [cjk_name]
plt.rcParams['axes.unicode_minus'] = False

# k(N) = K_max / (1 + N0/N)
K_max = 1.0
N0 = 5.0  # 半饱和点（单位 B），示意值

N = np.linspace(0.3, 80, 500)
k = K_max / (1 + N0 / N)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)

# 曲线
ax.plot(N, k, color='#2c6fbb', linewidth=2.8, zorder=3)

# 理论上限 K_max 虚线
ax.axhline(K_max, color='#c0392b', linestyle='--', linewidth=1.4, alpha=0.8, zorder=2)
ax.text(78, K_max - 0.04, 'K_max（学习效率理论上限）',
        ha='right', va='top', color='#c0392b', fontsize=11)

# 实验里的参数点
points = [0.5, 1.5, 3, 7, 14, 32, 72]
for p in points:
    kp = K_max / (1 + N0 / p)
    ax.scatter(p, kp, s=55, color='#2c6fbb', zorder=4, edgecolors='white', linewidths=1.2)

# 标注关键点
def annotate(p, label, dy):
    kp = K_max / (1 + N0 / p)
    ax.annotate(label, (p, kp), textcoords='offset points',
                xytext=(8, dy), fontsize=10.5, color='#333')

annotate(0.5, '0.5B', 8)
annotate(7, '7B', -16)
annotate(32, '32B', -18)
annotate(72, '72B', -18)

# 32B 之后饱和区域阴影
ax.axvspan(32, 80, color='#f0ad4e', alpha=0.12, zorder=1)
ax.text(54, 0.30, '32B 之后\n增长几乎停滞', ha='center', va='center',
        fontsize=12, color='#a06a00', fontweight='bold')

# 前段陡增箭头说明
ax.annotate('小模型阶段：模型越大，RL 越高效',
            xy=(3, K_max / (1 + N0 / 3)), xytext=(12, 0.40),
            fontsize=11, color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.3))

ax.set_xlabel('模型参数量 N（十亿 / B）', fontsize=12)
ax.set_ylabel('RL 学习效率 k(N)', fontsize=12)
ax.set_title('RL 后训练的学习效率：先陡后平的饱和曲线\n' +
             r'$k(N)=K_{max}\,/\,(1+N_0/N)$',
             fontsize=14, pad=12)

ax.set_xlim(0, 80)
ax.set_ylim(0, 1.08)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/198_saturation.svg', format='svg', bbox_inches='tight')
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/198_saturation.png', format='png', bbox_inches='tight', dpi=150)
print('saved svg + png')
