# PIE-OC Embedding 实验 - Llama 3.3 70B INT8 版本

用法：
1. 先在宿主机运行 docker 命令启动容器
2. 在容器内运行 pip install 和 python 命令

## 首次运行（创建新容器）

**宿主机执行：**
```bash
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/lmxxf/work/models:/workspace/models -v /home/lmxxf/work/ai-theorys-study:/workspace/ai-theorys-study --name magical_bhabha nvcr.io/nvidia/pytorch:25.11-py3 bash
```

**容器内执行：**
```bash
# 必须用旧版本，新版本会导致内存爆炸系统崩溃
pip install transformers==4.51.3 compressed-tensors==0.9.0 scikit-learn matplotlib scipy accelerate
python /workspace/ai-theorys-study/arxiv/paper70/pie_oc_embedding_llama.py
```

## 重启电脑后（容器已存在）

```bash
# 查看所有容器（包括停止的）
sudo docker ps -a

# 启动已停止的容器
sudo docker start magical_bhabha

# 进入容器（pip 包还在，不用重装）
sudo docker exec -it magical_bhabha bash
```

## 其他常用命令

```bash
# 停止容器
sudo docker stop magical_bhabha

# 删除容器（删了就没了，pip 包要重装）
sudo docker rm magical_bhabha
```

**注意：** 如果创建时没取名，Docker 会自动分配随机名字（如 `hungry_einstein`）。用 `sudo docker ps -a` 查看，然后用那个名字或 CONTAINER ID 操作。

---

## 踩坑记录（2026-01-22）

### 问题：新版本 transformers 导致内存爆炸

| 包 | 能跑的版本 | 崩溃的版本 |
|---|---|---|
| transformers | **4.51.3** | 4.57.6 |
| compressed-tensors | **0.9.0** | 0.13.0 |

**现象：**
- 加载 Llama-3.3-70B-INT8 模型时，CPU 内存和 GPU 显存同时暴涨
- 128GB 统一内存被撑爆，系统触发 OOM Killer
- 整机卡死，SSH 断开，需要硬重启

**原因：**
- 新版本 transformers (4.57.6) + compressed-tensors (0.13.0) 对 INT8 模型的内存管理有 bug
- 旧版本 (4.51.3 + 0.9.0) 没有这个问题

**解决方案：**
- 降级到旧版本：`pip install transformers==4.51.3 compressed-tensors==0.9.0`

### 问题：模型加载到 CPU 而不是 GPU

**现象：**
- 模型加载完成后显示 `Device: cpu`
- 推理时 CPU 占用 100%，然后被 Killed

**原因：**
- 容器重启后 GPU 连接丢失
- `--gpus all` 参数没有生效

**解决方案：**
- 删除旧容器重新创建：
```bash
sudo docker stop magical_bhabha
sudo docker rm magical_bhabha
sudo docker run --gpus all ...  # 重新创建
```

- 在容器内验证 GPU 可用：
```bash
nvidia-smi  # 应该能看到 GPU
python -c "import torch; print(torch.cuda.is_available())"  # 应该返回 True
```

---

## 实验结果

| 模型 | 同源词对 | 对照组 | Δ |
|------|---------|--------|-----|
| Qwen2.5-72B-AWQ (4-bit) | 0.714 | 0.750 | -0.036 |
| **Llama-3.3-70B-INT8 (8-bit)** | **0.592** | **0.598** | **-0.006** |

**结论：**
- 8-bit 模型的差距（-0.006）比 4-bit（-0.036）小很多
- 但仍然是负数，对照组相似度更高
- **C.C. 的假说"更高精度能捕捉同源性"未被证实**
- 静态 embedding 方法本身可能有局限，需要尝试 SAE probing（动态激活模式）
