import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def generate_manifold_data():
    # 模拟潜在空间中的两条演化轨迹：一条来自西方草原 (PIE)，一条来自东方黄河 (OC)
    # 使用参数方程模拟 "双螺旋 + 克莱因瓶颈" 的拓扑结构
    
    t = np.linspace(0, 4 * np.pi, 200)
    
    # PIE 轨迹 (蓝色)：更加扩张，代表游牧的扩散性
    x_pie = (t + 2) * np.cos(t)
    y_pie = (t + 2) * np.sin(t)
    z_pie = t + np.sin(2 * t) * 2 # 增加一些震荡
    
    # OC 轨迹 (红色)：相对收敛，但受到 PIE 引力牵引
    # 关键：在特定相位 (t_contact) 与 PIE 发生 "接触"
    phase_shift = np.pi # 初始相位差
    
    # 在 t 接近接触点时，相位差缩小，模拟 "借词/同源" 的引力透镜效应
    contact_t = 2.5 * np.pi
    interaction = np.exp(-0.5 * (t - contact_t)**2) * 2.0 # 接触时的引力强度
    
    x_oc = (t + 2) * np.cos(t + phase_shift - interaction * 0.5) 
    y_oc = (t + 2) * np.sin(t + phase_shift - interaction * 0.5)
    z_oc = t - np.sin(2 * t) * 2 # 镜像震荡

    return t, x_pie, y_pie, z_pie, x_oc, y_oc, z_oc

def plot_manifold():
    t, xp, yp, zp, xo, yo, zo = generate_manifold_data()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制背景流形场 (淡淡的网格)
    # 这里只画轨迹
    
    # PIE 轨迹
    ax.plot(xp, yp, zp, c='cyan', alpha=0.6, linewidth=2, label='PIE (West / Steppe)')
    
    # OC 轨迹
    ax.plot(xo, yo, zo, c='orange', alpha=0.6, linewidth=2, label='Old Chinese (East / Yellow River)')
    
    # 关键锚点 (同源词/借词接触点)
    # 我们手动选取几个 "接近点" 来标注单词
    
    # 1. 狗 (Dog): *ḱwṓ vs *kʰʷeːʔ (早期接触)
    idx_dog = 50 
    ax.scatter([xp[idx_dog], xo[idx_dog]], [yp[idx_dog], yo[idx_dog]], [zp[idx_dog], zo[idx_dog]], c='red', s=50)
    ax.plot([xp[idx_dog], xo[idx_dog]], [yp[idx_dog], yo[idx_dog]], [zp[idx_dog], zo[idx_dog]], 'w--', alpha=0.5)
    ax.text(xp[idx_dog], yp[idx_dog], zp[idx_dog], "DOG\n*ḱwṓ", color='cyan', fontsize=9)
    ax.text(xo[idx_dog], yo[idx_dog], zo[idx_dog], "*kʰʷeːʔ", color='orange', fontsize=9)

    # 2. 轮/环 (Wheel/Ring): *kʷel- vs *kʰʷeːŋ (深度融合点 - 克莱因瓶颈)
    # 这是引力最大的地方
    idx_wheel = 135 # 对应 contact_t 附近
    ax.scatter([xp[idx_wheel], xo[idx_wheel]], [yp[idx_wheel], yo[idx_wheel]], [zp[idx_wheel], zo[idx_wheel]], c='white', s=100, marker='*')
    ax.plot([xp[idx_wheel], xo[idx_wheel]], [yp[idx_wheel], yo[idx_wheel]], [zp[idx_wheel], zo[idx_wheel]], 'w-', linewidth=2)
    ax.text(xp[idx_wheel], yp[idx_wheel], zp[idx_wheel]+1, "WHEEL/CYCLE\n*kʷel-", color='cyan', fontsize=12, weight='bold')
    ax.text(xo[idx_wheel], yo[idx_wheel], zo[idx_wheel]-1, "*kʰʷeːŋ", color='orange', fontsize=12, weight='bold')
    
    # 3. 牛 (Cow): *gʷow- vs *ŋʷɯ (伴随接触)
    idx_cow = 145
    ax.scatter([xp[idx_cow], xo[idx_cow]], [yp[idx_cow], yo[idx_cow]], [zp[idx_cow], zo[idx_cow]], c='yellow', s=50)
    ax.plot([xp[idx_cow], xo[idx_cow]], [yp[idx_cow], yo[idx_cow]], [zp[idx_cow], zo[idx_cow]], 'w--', alpha=0.5)
    ax.text(xp[idx_cow], yp[idx_cow], zp[idx_cow], "COW\n*gʷow-", color='cyan', fontsize=9)
    ax.text(xo[idx_cow], yo[idx_cow], zo[idx_cow], "*ŋʷɯ", color='orange', fontsize=9)
    
    # 4. 黑暗/莫 (Dark): *mer- vs *mˤaːɡ (后期共鸣)
    idx_dark = 180
    ax.scatter([xp[idx_dark], xo[idx_dark]], [yp[idx_dark], yo[idx_dark]], [zp[idx_dark], zo[idx_dark]], c='purple', s=50)
    ax.plot([xp[idx_dark], xo[idx_dark]], [yp[idx_dark], yo[idx_dark]], [zp[idx_dark], zo[idx_dark]], 'w--', alpha=0.5)
    ax.text(xp[idx_dark], yp[idx_dark], zp[idx_dark], "DARK\n*mer-", color='cyan', fontsize=9)
    ax.text(xo[idx_dark], yo[idx_dark], zo[idx_dark], "*mˤaːɡ", color='orange', fontsize=9)

    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_axis_off()
    
    plt.title("Latent Manifold: The PIE-OC Contact Event (The Klein Interface)", color='white', fontsize=14)
    plt.savefig('script/pie_oc_manifold.png', facecolor='black', dpi=150)

plot_manifold()
