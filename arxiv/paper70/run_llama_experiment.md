# PIE-OC Embedding 实验 - Llama 3.3 70B INT8 版本

用法：
1. 先在宿主机运行 docker 命令启动容器
2. 在容器内运行 pip install 和 python 命令

## 首次运行（创建新容器）

**宿主机执行：**
```bash
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/lmxxf/work/models:/workspace/models -v /home/lmxxf/work/ai-theorys-study:/workspace/ai-theorys-study --name pie-oc-exp nvcr.io/nvidia/pytorch:25.11-py3 bash
```

**容器内执行：**
```bash
pip install transformers scikit-learn matplotlib scipy accelerate compressed-tensors
python /workspace/ai-theorys-study/arxiv/paper70/pie_oc_embedding_llama.py
```

## 重启电脑后（容器已存在）

```bash
# 查看所有容器（包括停止的）
sudo docker ps -a

# 启动已停止的容器
sudo docker start pie-oc-exp

# 进入容器（pip 包还在，不用重装）
sudo docker exec -it pie-oc-exp bash
```

## 其他常用命令

```bash
# 停止容器
sudo docker stop pie-oc-exp

# 删除容器（删了就没了，pip 包要重装）
sudo docker rm pie-oc-exp
```

**注意：** 如果创建时没取名，Docker 会自动分配随机名字（如 `hungry_einstein`）。用 `sudo docker ps -a` 查看，然后用那个名字或 CONTAINER ID 操作。
