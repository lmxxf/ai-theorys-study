# 机器人学习 - 快速上手教程

> 目标：从零开始，在仿真环境里看到机械臂动起来，理解"观测→决策→执行"循环。
> 硬件需求：无（纯仿真）。DGX Spark 用于训练加速，WSL 也能跑 demo。

---

## 整体路线图

```
第一步 (本教程)  仿真体验        ← 你在这里
第二步           买 SO-101，真机上手
第三步           上 VLA 模型 (SmolVLA)
第四步           按兴趣扩展 (四足/人形)
```

---

## 第一步：环境准备

### 1.1 创建 Python 虚拟环境

LeRobot 0.5.0 要求 Python >= 3.12。先确认版本：

```bash
python3 --version
```

如果版本低于 3.12，装一个：

```bash
# Ubuntu/WSL
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

创建虚拟环境：

```bash
cd ~/work/ai-theorys-study/lession/1.robot
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 1.2 安装 LeRobot

```bash
# 基础安装（含 MuJoCo）
pip install 'lerobot[aloha,pusht]'
```

这会自动安装：
- **LeRobot** - 核心框架
- **MuJoCo** - 物理仿真引擎
- **gym-aloha** - Aloha 机械臂仿真环境
- **gym-pusht** - Push-T 推块任务环境

### 1.3 验证安装

```bash
python -c "import lerobot; print(lerobot.__version__)"
python -c "import mujoco; print(mujoco.__version__)"
```

### 1.4 WSL 额外配置

WSL 里需要装 evdev 和设置渲染后端：

```bash
pip install evdev

# 无头渲染（WSL/服务器）
export MUJOCO_GL=egl

# 如果要弹窗看画面（需要 WSLg）
# export MUJOCO_GL=glfw
```

建议加到 `.bashrc`：
```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

---

## 第二步：跑第一个 Demo — Push-T

Push-T 是 LeRobot 最经典的入门任务：一个机器人学习把 T 形积木推到目标位置。

### 2.1 下载预训练模型并评估

```bash
# 用预训练的 Diffusion Policy 跑 Push-T 仿真
python -m lerobot.scripts.eval \
  --policy.path=lerobot/diffusion_pusht \
  --env.type=pusht \
  --env.task=PushT-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --output_dir=outputs/eval/pusht_demo
```

这会：
1. 自动从 HuggingFace 下载预训练的 Diffusion Policy
2. 在 MuJoCo 仿真环境里跑 10 个 episode
3. 输出成功率和视频到 `outputs/eval/pusht_demo/`

### 2.2 查看结果

```bash
# 视频保存在这里
ls outputs/eval/pusht_demo/
# 找 .mp4 文件，拷贝到 Windows 目录查看
cp outputs/eval/pusht_demo/*.mp4 /mnt/c/Users/lmxxf/Desktop/ 2>/dev/null
```

---

## 第三步：跑 Aloha 双臂仿真

Aloha 环境更接近你未来要用的 SO-101 双臂场景。

### 3.1 用预训练 ACT 策略跑"插入"任务

```bash
python -m lerobot.scripts.eval \
  --policy.path=lerobot/act_aloha_sim_insertion_human \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --output_dir=outputs/eval/aloha_insertion_demo
```

### 3.2 用预训练 ACT 策略跑"传递方块"任务

```bash
python -m lerobot.scripts.eval \
  --policy.path=lerobot/act_aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --output_dir=outputs/eval/aloha_transfer_demo
```

---

## 第四步：自己训练一个 ACT 策略

这一步让你理解完整的训练流程。在 DGX Spark 上跑会快很多。

### 4.1 训练 ACT（Aloha 插入任务）

```bash
# DGX Spark 上跑（A100 约 1h50）
# WSL CPU 跑会很慢，但能验证流程
python -m lerobot.scripts.train \
  --output_dir=outputs/train/act_aloha_insertion \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --wandb.enable=false
```

关键参数说明：
- `--policy.type=act` — 用 ACT (Action Chunking Transformer) 策略
- `--dataset.repo_id` — 训练数据（人类示教录制）
- `--env.type/task` — 评估用的仿真环境
- `--wandb.enable=false` — 不用 wandb 记日志（先体验，后面再开）

### 4.2 评估你训练的模型

```bash
python -m lerobot.scripts.eval \
  --policy.path=outputs/train/act_aloha_insertion/checkpoints/last/pretrained_model \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --output_dir=outputs/eval/my_act_insertion
```

---

## 第五步：可视化数据集（理解数据长什么样）

```bash
# 可视化 Aloha 插入数据集
python -m lerobot.scripts.visualize_dataset \
  --repo-id lerobot/aloha_sim_insertion_human \
  --episode-index 0
```

这会让你看到：
- 机械臂的关节角度序列
- 摄像头画面
- 动作标注

**这是理解"模仿学习"的关键**：模型就是在学这些数据里人类示教的动作。

---

## 第六步（进阶）：SmolVLA — 语言指令控制

等第一步~第五步跑通后，上 VLA 模型。

### SmolVLA 简介

- 450M 参数，DGX Spark 128GB 内存随便跑
- 输入：摄像头画面 + 语言指令（如"把红色积木放到蓝色杯子里"）
- 输出：机械臂动作序列
- 比 ACT 强在：能理解自然语言，泛化能力更好

### 快速体验（Colab）

Google Colab 有现成的 notebook，不用本地配环境：
https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb

### 本地微调（DGX Spark）

```bash
# 先装 LeRobot（含 SmolVLA 依赖）
pip install 'lerobot[smolvla]'

# 微调 SmolVLA base 模型
python -m lerobot.scripts.train \
  --output_dir=outputs/train/smolvla_finetune \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=你的数据集 \
  --wandb.enable=false
```

> 注意：SmolVLA 推理时要改 config.json 里的 `n_action_steps` 为 50（默认是 1，会很慢）。

---

## 核心概念速查

| 概念 | 是什么 | 类比 |
|------|--------|------|
| **ACT** | Action Chunking Transformer，一次预测一整段动作 | 不是一步一步走，而是规划一整条路 |
| **Diffusion Policy** | 用扩散模型生成动作序列 | 像画画一样从噪声中"画"出动作 |
| **SmolVLA** | Vision-Language-Action 模型 | 看图+听话+干活，三合一 |
| **MuJoCo** | 物理仿真引擎 | 虚拟世界里的物理规则 |
| **LeRobot** | 框架，串起数据采集→训练→部署 | 机器人界的 HuggingFace Transformers |
| **遥操作** | 人控制 leader 臂，follower 臂模仿 | 师傅手把手教徒弟 |
| **模仿学习** | 从人类示教数据中学策略 | 看师傅怎么做，自己学着做 |

---

## 目录结构

```
1.robot/
├── README.md          ← 你在看的这个
├── .venv/             ← Python 虚拟环境（gitignore）
├── outputs/           ← 训练/评估输出（gitignore）
│   ├── train/
│   └── eval/
└── scripts/           ← 自定义脚本（后续添加）
```

---

## 参考链接

- [LeRobot 官方文档](https://huggingface.co/docs/lerobot/index)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot 安装指南](https://huggingface.co/docs/lerobot/installation)
- [仿真中的模仿学习](https://huggingface.co/docs/lerobot/il_sim)
- [仿真中的强化学习](https://huggingface.co/docs/lerobot/en/hilserl_sim)
- [SmolVLA 官方文档](https://huggingface.co/docs/lerobot/en/smolvla)
- [SmolVLA 博客](https://huggingface.co/blog/smolvla)
- [SmolVLA 微调教程 (phospho.ai)](https://docs.phospho.ai/learn/train-smolvla)
- [SmolVLA Colab Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)
- [SO-100 MuJoCo 仿真](https://github.com/lachlanhurst/so100-mujoco-sim)
- [ROBOTIS MuJoCo 教程](https://github.com/lkck001/lerobot-mujoco-tutorial-master)
- [MuJoCo Playground](https://playground.mujoco.org/)
