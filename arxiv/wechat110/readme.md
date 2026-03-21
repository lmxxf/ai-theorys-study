# 110期 Doc-to-LoRA 消融实验

## 环境搭建

### 1. 下载模型和代码

```bash
# Gemma-2-2b-it（需要先去 https://huggingface.co/google/gemma-2-2b-it 点同意协议）
huggingface-cli download google/gemma-2-2b-it --local-dir /home/lmxxf/work/models/gemma-2-2b-it

# Doc-to-LoRA 超网络权重
huggingface-cli download SakanaAI/doc-to-lora --local-dir /home/lmxxf/work/models/doc-to-lora

# 代码仓库
cd /home/lmxxf/work && git clone https://github.com/SakanaAI/doc-to-lora.git
```

### 2. 创建 Docker 容器

DGX Spark 的 GB10 是 Blackwell 架构（sm_121），必须用 NVIDIA 官方镜像自带的 torch，不能 pip 装。

```bash
docker run -d --gpus all --name d2l_exp \
  -v /home/lmxxf/work:/workspace \
  --ipc=host \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  sleep infinity
```

### 3. 安装依赖（容器内）

**关键：`--no-deps` 保护容器原装的 torch/triton/flash-attn 不被覆盖**

```bash
docker exec -it d2l_exp bash
cd /workspace/doc-to-lora && pip install --no-deps -e . && pip install --no-deps peft datasets sentencepiece accelerate bitsandbytes
```

### 4. 验证环境（容器内）

```bash
cd /workspace/doc-to-lora

python -c "
import torch
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

checkpoint_path = '/workspace/models/doc-to-lora/gemma_demo/checkpoint-80000/pytorch_model.bin'
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(state_dict, train=False, use_sequence_packing=False)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

doc = open('data/sakana_wiki.txt', 'r').read()
chat = [{'role': 'user', 'content': 'Tell me about Sakana AI.'}]
chat_ids = tokenizer.apply_chat_template(chat, add_special_tokens=False, return_attention_mask=False, add_generation_prompt=True, return_tensors='pt').to(model.device)

model.internalize(doc)
outputs = model.generate(input_ids=chat_ids, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
"
```

## 容器内路径

| 内容 | 路径 |
|------|------|
| doc-to-lora 代码 | /workspace/doc-to-lora/ |
| Gemma-2-2b-it 模型 | /workspace/models/gemma-2-2b-it/ |
| 超网络权重（正式） | /workspace/models/doc-to-lora/gemma_2b_d2l/checkpoint-20000/ |
| 超网络权重（demo） | /workspace/models/doc-to-lora/gemma_demo/checkpoint-80000/ |
| 实验脚本（待写） | /workspace/ai-theorys-study/arxiv/wechat110/ |

## 超网络关键参数（gemma_2b_d2l/args.yaml）

- `lora_r: 8` — rank 8
- `max_ctx_chunk_len: 512` — 每块最大 512 tokens
- `target_modules: [down_proj]` — 只改 MLP 的 down_proj
- `model_name_or_path: google/gemma-2-2b-it`

## 血泪教训

- **不要在 NVIDIA 官方容器里 `pip install torch`** —— GB10 需要定制编译的 torch，pip 版不支持 sm_121
- **装 doc-to-lora 依赖必须 `--no-deps`** —— 否则会拉下来 torch 2.6 覆盖原装版本，然后 triton/flash-attn 全崩
- poet_exp 容器就是这么被搞坏的，已删，RIP 😂
