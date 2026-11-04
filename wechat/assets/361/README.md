# 公众号第 361 期配图与独立复现脚本目录 (`wechat/assets/361/`)

本目录保留公众号第 361 期《【AI可解释性】大模型犹豫时残差流在“急转弯”？我们在 DGX Spark 上复现了 MIT 这篇 ICML 论文，顺手测出了它没看见的真相》（`wechat/361.md`）的正文配图、主复现脚本与中文复现报告：

## 1. 本目录文件清单
- `curvature-concept-and-subspaces.svg`：残差流横向时间轨迹曲率手算原理与五类子空间干预示意图；
- `observations.svg`：`GPT-2 XL` 与 `Pythia-2.8B` 在 `LAMBADA` 语料上的逐层平均转角及预测熵相关系数曲线；
- `interventions.svg`：`GPT-2 XL` 与 `Pythia-2.8B` 在中层五类子空间下的 32 位置主实验因果干预结果；
- `our-three-truths-35b-27b.svg`：我们在 DGX Spark 上对 `Qwen3.5-35B-A3B Base/Post`、`Qwen3.8-27B` 与 `Qwen3-0.6B` 测出的三组结构性现象汇总图；
- `run_reproduction.py`：对齐 `arXiv:2604.23985` 公式与主要控制条件的核心复现脚本；
- `复现报告.md`：`GPT-2 XL` 与 `Pythia-2.8B` 独立复现报告；
- `审稿意见.md`：第 361 期终审核查记录。

## 2. 完整实验过程数据归档位置
为保持公众号主仓库轻量，本轮复现的大体积逐词元 `.npz` 数组、语料文件、中间 JSON 与补充验证脚本均已归档至独立数据仓目录 `/home/lmxxf/work/grocery-haven/wechat-361/`（DGX Spark 源运行目录为 `192.168.31.198:/home/lmxxf/work/curvature-icml2026-361/`）；后半篇 836 条现代模型前向测量记录归档于 `/home/lmxxf/work/grocery-haven/paper99-curvature-exp-20261102/`。
