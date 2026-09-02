import numpy as np, matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']='Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus']=False

def act(x):
    t=np.clip(x,-4,4)
    g=0.447265625-0.055908203125*np.abs(t)
    g=0.89453125+t*g
    return t*g
silu=lambda x: x/(1+np.exp(-x))
x=np.linspace(-6,6,1201)

fig,ax=plt.subplots(figsize=(8,5),dpi=130)
ax.axhline(0,color='#bbb',lw=.8); ax.axvline(0,color='#bbb',lw=.8)
ax.plot(x,np.maximum(x,0),'--',color='#999',lw=1.4,label='ReLU（参照）')
ax.plot(x,silu(x),'--',color='#2a6fd6',lw=1.6,label='SiLU（参照）')
ax.plot(x,act(x),color='#d9822b',lw=2.6,label='DLSS 5 的多项式激活')
for v in (-4,4):
    ax.axvline(v,color='#d9822b',ls=':',lw=1,alpha=.6)
ax.annotate('x < -4：彻底归零',xy=(-4,0),xytext=(-5.9,2.4),fontsize=10,color='#d9822b',
            arrowprops=dict(arrowstyle='->',color='#d9822b',lw=1))
ax.annotate('x > 4：封顶在 7.16',xy=(4.6,7.16),xytext=(1.1,8.4),fontsize=10,color='#d9822b',
            arrowprops=dict(arrowstyle='->',color='#d9822b',lw=1))
ax.annotate('负半轴留一个小坑\n（最低约 -0.5）',xy=(-1,-0.503),xytext=(-5.9,-2.6),fontsize=10,color='#555',
            arrowprops=dict(arrowstyle='->',color='#888',lw=1))
ax.set_xlim(-6,6); ax.set_ylim(-3.5,9.5)
ax.set_xlabel('输入'); ax.set_ylabel('输出')
ax.set_title('DLSS 5 的激活函数：形状像 SiLU，但正半轴放大、两端截断')
ax.legend(loc='upper left',frameon=False,fontsize=10)
ax.grid(alpha=.25,lw=.6)
fig.tight_layout(); fig.savefig('activation.png')
print('saved')
