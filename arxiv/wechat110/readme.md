# 110期 Doc-to-LoRA 消融实验

## 实验结果

```
Condition     Chunks    Rank   Text%   Accuracy
-----------------------------------------------
Baseline           0       0      0%      25.0%
N=1                1      16    0.6%      16.7%
N=5                5      48    2.8%       8.3%
N=20              20     168   11.4%       4.2%
N=50              50     408   28.4%       0.0%
N=100            100     808   56.9%       0.0%
N=ALL(176)       176    1416  100.0%       0.0%
```

**块越多，准确率越低。** N≥50 时模型输出退化为重复 token（`"îng îng îng..."` / `"первыхпервых..."`）。

### 解读

- **Baseline 25%**: Gemma-2B 预训练里就有红楼梦知识，不需要文档也能答对一些
- **N=1 (16.7%)**: LoRA 注入 512 tokens 的局部信息，反而干扰了模型的全局知识
- **N≥5 持续下降**: rank 线性增长带来的噪声累积超过信息增益
- **N≥50 归零**: LoRA 矩阵太大，基础模型的语言能力被彻底摧毁

### 与 110 期文章的对应

验证了文章的核心论点：**分块拼接的 rank 线性增长是错误的方向。** SHY（突触稳态假说）预测的现象完美复现——没有选择性削减的全面增强 = 信噪比崩溃。Doc-to-LoRA 学会了"睡觉"，但还没学会"做梦"。

### 注意事项

- 超网络训练时 `num_chunk_probs` 最大 key 是 8，本实验最多用了 176 块，远超训练分布
- 退化可能部分是 OOD（out-of-distribution）问题，不完全是架构缺陷
- 但 N=1 < Baseline 这个结果在训练分布内，说明即使单块，LoRA 对已知知识也有干扰

## 实验设计

| 条件 | 块数 | rank | 覆盖文本 |
|------|------|------|----------|
| Baseline | 0 | 0 | 无 |
| N=1 | 1 | 16 | 前 0.6% |
| N=5 | 5 | 48 | 前 2.8% |
| N=20 | 20 | 168 | 前 11.4% |
| N=50 | 50 | 408 | 前 28.4% |
| N=100 | 100 | 808 | 前 56.9% |
| N=ALL | 176 | 1416 | 100% |

- **测试文本**: 红楼梦英译本（H. Bencraft Joly, 1892）前 8 章，~85K tokens
- **测试题目**: 24 道事实性问答（前 8 章，每章 3 题）
- **基础模型**: Gemma-2-2b-it
- **超网络**: SakanaAI/doc-to-lora gemma_2b_d2l checkpoint-20000

## 文件结构

| 文件 | 说明 |
|------|------|
| `run_experiment.py` | 消融实验主脚本 |
| `qa_dataset.json` | 72 道问答题（实验只用前 8 章的 24 道） |
| `story_of_the_stone.txt` | 红楼梦英译本全文 |
| `results.json` | 实验结果（脚本自动生成） |
| `DevHistory.md` | 开发日志（环境搭建踩坑、脚本迭代过程） |

## 复现

### 1. 下载模型和代码

```bash
huggingface-cli download google/gemma-2-2b-it --local-dir /home/lmxxf/work/models/gemma-2-2b-it
huggingface-cli download SakanaAI/doc-to-lora --local-dir /home/lmxxf/work/models/doc-to-lora
cd /home/lmxxf/work && git clone https://github.com/SakanaAI/doc-to-lora.git
```

### 2. 创建 Docker 容器

```bash
docker run -d --gpus all --name d2l_exp \
  -v /home/lmxxf/work:/workspace \
  --ipc=host \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  sleep infinity
```

### 3. 安装依赖（容器内）

```bash
docker exec -it d2l_exp bash

cd /workspace/doc-to-lora && pip install --no-deps -e .
pip install --no-deps peft datasets sentencepiece accelerate bitsandbytes
pip install transformers==4.51.3
pip install jaxtyping wonderwords inflect rouge-score torchmetrics
```

### 4. 代码修改

`src/ctx_to_lora/data/processing.py` 第 989 行，去掉空的 system message（Gemma 2 不支持 system role）：

```python
# 改前: [{"role": "system", "content": ""}, {"role": "user", ...}]
# 改后: [{"role": "user", ...}]
```

### 5. 跑实验

```bash
cd /workspace/doc-to-lora
python /workspace/ai-theorys-study/arxiv/wechat110/run_experiment.py
```

## 技术细节

Doc-to-LoRA 的官方 API 只支持单块场景。多块消融需要拆开黑盒手动控制：

1. 逐块调 `generate_weights()` 得到 rank-8 LoRA，中间结果存 CPU 省显存
2. 沿 dim=0 拼接后调 `combine_lora` 合并
3. 调 `apply_lora_to_layers` 挂载 LoRA hook
4. 直接调 `base_model.generate()` 绕过官方 generate()

详见 `DevHistory.md`。
