import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
cjk_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['font.sans-serif'] = [cjk_name]
plt.rcParams['axes.unicode_minus'] = False

C = np.linspace(0.05, 10, 600)  # 算力（任意单位）

# 预训练 power law：收益随 log(算力) 持续上升，无天花板
power = 0.32 * np.log(C / 0.05)

# RL sigmoid：R(C)-R0 = (A-R0)/(1+(C_mid/C)^B)，有天花板 A
A = 1.0
C_mid = 1.2
B = 1.6
sigmoid = A / (1 + (C_mid / C) ** B)

fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)

# power law 曲线
ax.plot(C, power, color='#2c6fbb', linewidth=2.8, zorder=3,
        label='预训练（power law）：无天花板')

# sigmoid 曲线
ax.plot(C, sigmoid, color='#c0392b', linewidth=2.8, zorder=3,
        label='RL 后训练（sigmoid）：有天花板')

# 天花板虚线
ax.axhline(A, color='#c0392b', linestyle='--', linewidth=1.3, alpha=0.7, zorder=2)
ax.text(9.8, A + 0.03, '天花板 A（砸多少算力都过不去）',
        ha='right', va='bottom', color='#c0392b', fontsize=11)

# power law 不封顶箭头
ax.annotate('一直涨，没有上限',
            xy=(9.2, 0.32 * np.log(9.2 / 0.05)), xytext=(5.0, 1.45),
            fontsize=11.5, color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.3))

# sigmoid 趋平注释
ax.annotate('趋于平坦：再加算力也不涨',
            xy=(7.0, A / (1 + (C_mid / 7.0) ** B)), xytext=(3.2, 0.55),
            fontsize=11.5, color='#922b21',
            arrowprops=dict(arrowstyle='->', color='#922b21', lw=1.3))

ax.set_xlabel('RL 训练算力 C（对数尺度，任意单位）', fontsize=12)
ax.set_ylabel('能力提升 / 奖励', fontsize=12)
ax.set_title('power law vs sigmoid：预训练无界，RL 后训练有天花板',
             fontsize=14, pad=12)

ax.set_xscale('log')
ax.set_xlim(0.05, 10)
ax.set_ylim(0, 1.7)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/198_sigmoid_vs_powerlaw.svg', format='svg', bbox_inches='tight')
plt.savefig('/home/lmxxf/work/ai-theorys-study/wechat/198_sigmoid_vs_powerlaw.png', format='png', bbox_inches='tight', dpi=150)
print('saved')
