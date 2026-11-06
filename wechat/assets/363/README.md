# 公众号第 363 期实验资产：顺着流形表皮走的几何（表皮曲率、测地快慢与切空间方向有效维度）

本目录归档了公众号第 363 期《【AI可解释性】为什么前文 MIT 的残差流曲率只测出 1%？别拿穿球肚子的弦量球面，顺着流形表皮走才看得见快慢与分叉》的核心实验脚本、实验报告与纯白底正文配图。

## 一、 核心文件清单

- [`run_manifold_exp.py`](./run_manifold_exp.py)：主实验脚本（含球面大圆平行移动、去均值锥投影、真实测地曲率 `kappa_geo`、球面测地步长 `s_geo`、5×5 Gram 矩阵切空间有效维度 `erank_dir`、垂直跨层表皮角速度 `v_layer`、10 折跨篇章分组交叉验证与球面非欧几何单元测试）。
- [`run_case_studies.py`](./run_case_studies.py)：微观案例分析脚本（顺口溜 `ROUTINE` vs 烧脑逻辑题 `HARD` 及冲突对 `PAIRS` 对比）。
- [`plot_results.py`](./plot_results.py)：实验统计汇总作图辅助脚本。
- [`实验报告.md`](./实验报告.md)：DGX Spark（`spark-3a10`）上的全量 32 项指标与 10 折交叉验证详细实验报告。

## 二、 正文纯白底 SVG 配图

1. [`sphere-triangle-60-vs-90.svg`](./sphere-triangle-60-vs-90.svg)：为什么球面上的等边三角形是 90° 而不是 60°（穿球肚子平直弦 120° vs 流形表皮大圆弧线 90° 与赤道直行 0°）。
2. [`chord-vs-tangent-cancellation.svg`](./chord-vs-tangent-cancellation.svg)：为什么 MIT 只测出 1%（平直弦转角把正相关的表皮真转角与负相关的球皮向心折角混在一起互相抵消）。
3. [`true-geodesic-curvature.svg`](./true-geodesic-curvature.svg)：把表皮转角除以球面步长还原真测地曲率 `κ_geo = θ_surf / s_geo`（单指标解释力提升 4 至 80 倍）。
4. [`surface-speed-step.svg`](./surface-speed-step.svg)：顺着流形表皮走的“快慢”（胸有成竹大步流星 vs 拿不准急刹迈小碎步）。
5. [`surface-direction-erank.svg`](./surface-direction-erank.svg)：顺着流形表皮走的“方向多寡”（顺口溜钟摆低维往返 vs 烧脑题多向正交分叉）。
6. [`manifold-full-r2-jump.svg`](./manifold-full-r2-jump.svg)：10 折跨篇章交叉验证解释力 `R²` 阶梯跃升图（从 MIT 平直弦约 1% 跃升至单层 23%–41%、全层 54.8%–67.4%）。
