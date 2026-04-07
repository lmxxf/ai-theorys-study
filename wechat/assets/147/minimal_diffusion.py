"""
最小扩散模型：用 PyTorch 从零实现 DDPM，学习生成 2D 螺旋线上的点。

基于 tiny-diffusion (https://github.com/tanelp/tiny-diffusion) 验证通过的代码。
核心逻辑和 Push-T 的 Diffusion Policy 完全一致，只是：
- 数据从 32 维轨迹换成了 2 维的 (x, y) 点
- 网络从 1D U-Net 换成了 MLP（2D 点太简单，MLP 就够）

运行：python minimal_diffusion.py
输出：spiral_diffusion_result.png（左边原始数据，右边生成结果）
耗时：CPU 约 2-3 分钟
依赖：torch, matplotlib, numpy（无其他依赖）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 超参数
# ============================================================
T = 50           # 加噪总步数（泼多少遍噪声）
EPOCHS = 500     # 训练轮数
BATCH = 32       # 每批多少个点
LR = 1e-3        # 学习率
N_POINTS = 8000  # 螺旋线上的点数（训练数据量）
N_GEN = 1000     # 推理时生成多少个点

# ============================================================
# 第一段：准备数据和工具
# ============================================================

# ---- 数据：生成螺旋线上的点 ----
# 螺旋线在 2D 平面上，每个点是 (x, y) 坐标
# 这就是我们的"训练数据分布"——模型要学会从噪声中生成这个分布的样本
def make_spiral(n_points=N_POINTS):
    theta = torch.linspace(0, 4 * math.pi, n_points) + torch.randn(n_points) * 0.1
    r = theta / (4 * math.pi)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    data = torch.stack([x, y], dim=1) * 4  # 放大到 [-4, 4] 范围
    return data


# ---- Sinusoidal 位置嵌入 ----
# 把一个标量变成高维向量，和 Transformer 的位置编码是同一个东西
# 时间步用它，输入坐标也用它（让网络能感知高频弯曲细节）
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size=128, scale=1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


# ---- 残差块 ----
class Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.ff(x))  # 残差连接：输出 = 输入 + 变换


# ---- 神经网络 ----
# 关键：x 和 y 坐标各自做 sinusoidal 嵌入（128维），时间步也做（128维）
# 拼接后 = 128*3 = 384 维输入
class NoisePredictor(nn.Module):
    def __init__(self, hidden=128, n_layers=3, emb_size=128):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(emb_size, scale=1.0)
        self.input_emb_x = SinusoidalEmbedding(emb_size, scale=25.0)
        self.input_emb_y = SinusoidalEmbedding(emb_size, scale=25.0)

        layers = [nn.Linear(emb_size * 3, hidden), nn.GELU()]
        for _ in range(n_layers):
            layers.append(Block(hidden))
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t):
        x_emb = self.input_emb_x(x_t[:, 0])   # x 坐标 → 128 维
        y_emb = self.input_emb_y(x_t[:, 1])   # y 坐标 → 128 维
        t_emb = self.time_emb(t)                # 时间步 → 128 维
        inp = torch.cat([x_emb, y_emb, t_emb], dim=-1)  # 拼成 384 维
        return self.net(inp)


# ---- 噪声调度器 ----
# 预计算所有需要的系数，训练和推理时直接查表
class NoiseScheduler:
    def __init__(self, num_timesteps=50, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)         # 累乘
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

        # 正向加噪用
        self.sqrt_alpha_bars = self.alpha_bars ** 0.5
        self.sqrt_one_minus_alpha_bars = (1 - self.alpha_bars) ** 0.5

        # 反向去噪用
        self.sqrt_inv_alpha_bars = torch.sqrt(1.0 / self.alpha_bars)
        self.sqrt_inv_alpha_bars_minus_one = torch.sqrt(1.0 / self.alpha_bars - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_mean_coef2 = (1. - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1. - self.alpha_bars)

    def add_noise(self, x0, eps, t):
        """正向加噪：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps"""
        s1 = self.sqrt_alpha_bars[t].reshape(-1, 1)
        s2 = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1)
        return s1 * x0 + s2 * eps

    def step(self, eps_pred, timestep, x_t):
        """反向去噪一步：从 x_t 和预测的噪声，算 x_{t-1}"""
        t = timestep
        # 先从预测的噪声反推 x_0
        s1 = self.sqrt_inv_alpha_bars[t].reshape(-1, 1)
        s2 = self.sqrt_inv_alpha_bars_minus_one[t].reshape(-1, 1)
        x0_pred = s1 * x_t - s2 * eps_pred

        # 用 q_posterior 算 x_{t-1} 的均值
        c1 = self.posterior_mean_coef1[t].reshape(-1, 1)
        c2 = self.posterior_mean_coef2[t].reshape(-1, 1)
        mean = c1 * x0_pred + c2 * x_t

        # 加随机扰动（t=0 时不加）
        if t > 0:
            var = (self.betas[t] * (1. - self.alpha_bars_prev[t]) / (1. - self.alpha_bars[t])).clamp(min=1e-20)
            mean = mean + (var ** 0.5) * torch.randn_like(x_t)

        return mean

    def __len__(self):
        return self.num_timesteps


# ============================================================
# 第二段：训练（出题 → 答题 → 对答案）
# ============================================================

data = make_spiral()
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=True)
scheduler = NoiseScheduler(num_timesteps=T)
model = NoisePredictor()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

snapshots = []  # 记录不同轮数的生成结果

print(f"开始训练：{EPOCHS} 轮，batch={BATCH}，加噪 {T} 步")

for epoch in range(EPOCHS):
    model.train()
    for (batch,) in dataloader:
        eps = torch.randn(batch.shape)                                 # 掷随机数
        t = torch.randint(0, scheduler.num_timesteps, (batch.shape[0],)).long()  # 随机选时间步
        x_t = scheduler.add_noise(batch, eps, t)                       # 正向加噪（出题）
        eps_pred = model(x_t, t)                                       # 网络猜噪声（答题）
        loss = F.mse_loss(eps_pred, eps)                               # 对答案

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 50 == 0:
        print(f"  第 {epoch+1} 轮，Loss = {loss.item():.4f}")

    # 每隔一定轮数，用当前模型生成一次，记录学习过程
    if (epoch + 1) in [1, 10, 50, 100, 200, 500]:
        model.eval()
        with torch.no_grad():
            sample = torch.randn(N_GEN, 2)
            for t_step in reversed(range(T)):
                t_batch = torch.from_numpy(np.repeat(t_step, N_GEN)).long()
                eps_pred = model(sample, t_batch)
                sample = scheduler.step(eps_pred, t_batch[0], sample)
        snapshots.append((epoch + 1, sample.numpy()))
        model.train()

print("训练完成！")


# ============================================================
# 第三段：推理（最终生成）
# ============================================================

print(f"开始生成：从 {N_GEN} 个纯随机点出发，去噪 {T} 步")

model.eval()
with torch.no_grad():
    x = torch.randn(N_GEN, 2)

    for t_step in reversed(range(T)):
        t_batch = torch.from_numpy(np.repeat(t_step, N_GEN)).long()
        eps_pred = model(x, t_batch)
        x = scheduler.step(eps_pred, t_batch[0], x)

generated = x.numpy()
print("生成完成！")


# ============================================================
# 画图 1：最终结果（左边原始数据，右边最终生成）
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

original = data.numpy()
ax1.scatter(original[:, 0], original[:, 1], s=3, alpha=0.5, c='#2a7fff')
ax1.set_title('Original Spiral (Training Data)', fontsize=14)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

ax2.scatter(generated[:, 0], generated[:, 1], s=3, alpha=0.5, c='#ff6644')
ax2.set_title('Generated Spiral (From Pure Noise)', fontsize=14)
ax2.set_xlim(-6, 6)
ax2.set_ylim(-6, 6)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spiral_diffusion_result.png', dpi=150, bbox_inches='tight')
print("结果已保存到 spiral_diffusion_result.png")


# ============================================================
# 画图 2：学习过程（不同训练轮数的生成效果）
# ============================================================

n_snap = len(snapshots)
fig, axes = plt.subplots(n_snap + 1, 1, figsize=(4, 4 * (n_snap + 1)))

# 第一格：原始数据
axes[0].scatter(original[:, 0], original[:, 1], s=2, alpha=0.4, c='#2a7fff')
axes[0].set_title('Training Data', fontsize=12)
axes[0].set_xlim(-6, 6); axes[0].set_ylim(-6, 6)
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.2)

# 后面每格：对应轮数的生成结果
for i, (ep, pts) in enumerate(snapshots):
    axes[i + 1].scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, c='#ff6644')
    axes[i + 1].set_title(f'Epoch {ep}', fontsize=12)
    axes[i + 1].set_xlim(-6, 6); axes[i + 1].set_ylim(-6, 6)
    axes[i + 1].set_aspect('equal'); axes[i + 1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('spiral_learning_process.png', dpi=150, bbox_inches='tight')
print("学习过程已保存到 spiral_learning_process.png")
