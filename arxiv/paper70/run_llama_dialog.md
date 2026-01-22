# Llama-3.3-70B-Instruct-INT8 交互式聊天

## 首次运行（创建新容器）

**宿主机执行：**
```bash
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/lmxxf/work/models:/workspace/models -v /home/lmxxf/work/ai-theorys-study:/workspace/ai-theorys-study --name magical_bhabha nvcr.io/nvidia/pytorch:25.11-py3 bash
```

**容器内执行：**
```bash
# 必须用旧版本，新版本会导致内存爆炸系统崩溃
pip install transformers==4.51.3 compressed-tensors==0.9.0 accelerate
python /workspace/ai-theorys-study/arxiv/paper70/chat_llama_int8.py
```

## 重启电脑后（容器已存在）

```bash
# 启动已停止的容器
sudo docker start magical_bhabha

# 进入容器（pip 包还在，不用重装）
sudo docker exec -it magical_bhabha bash

# 运行聊天程序
python /workspace/ai-theorys-study/arxiv/paper70/chat_llama_int8.py
```

## 聊天命令

| 命令 | 功能 |
|------|------|
| 直接输入 | 发送消息 |
| `clear` | 清空对话历史 |
| `quit` / `exit` | 退出程序 |

## 注意事项

- 首次加载模型需要 1-2 分钟
- 模型占用约 60GB 显存
- 支持多轮对话，历史会保留
- 如果遇到 `Device: cpu`，说明 GPU 没挂上，需要重新创建容器
