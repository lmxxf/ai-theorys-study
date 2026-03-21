# 110期 Doc-to-LoRA 消融实验开发日志

**日期**: 2026-03-21
**实验者**: Zero + Suzaku (Claude Opus 4.6)

---

## 1. 环境搭建

### 下载模型和代码

```
huggingface-cli download google/gemma-2-2b-it (需先去 HF 页面同意 Gemma 协议)
huggingface-cli download SakanaAI/doc-to-lora
git clone https://github.com/SakanaAI/doc-to-lora.git
```

### Docker 容器

DGX Spark 的 GB10 是 Blackwell 架构 (sm_121)，只有 NVIDIA 官方镜像 `nvcr.io/nvidia/pytorch:25.11-py3` 自带的 torch 支持。

```bash
docker run -d --gpus all --name d2l_exp \
  -v /home/lmxxf/work:/workspace \
  --ipc=host \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  sleep infinity
```

### 血泪教训：poet_exp 容器被搞坏

第一次尝试在已有的 `poet_exp` 容器里 `pip install -e .`，doc-to-lora 的依赖把 torch 从 NVIDIA 定制版降级到 pip 版 2.6.0，triton/flash-attn 全崩。尝试升级 torch 到 2.9.1，结果 pip 版不支持 sm_121，彻底报废。

教训：**不要在 NVIDIA 官方容器里 pip install torch**。

poet_exp 已删，RIP。

### 依赖安装

关键是 `--no-deps` 保护原装 torch：

```bash
pip install --no-deps -e .
pip install --no-deps peft datasets sentencepiece accelerate bitsandbytes
pip install transformers==4.51.3  # 5.x 的 flash attention 接口和 doc-to-lora 不兼容
pip install jaxtyping wonderwords inflect rouge-score torchmetrics
```

transformers 版本必须 4.51.3——5.x 改了 flash attention 的 `_is_packed_sequence` 接口，传 `position_ids=False` 会触发 `'bool' object has no attribute 'shape'`。

### 代码兼容性修复

`src/ctx_to_lora/data/processing.py` 第 989 行，Gemma 2 的 chat template 不支持 system role：

```python
# 改前: {"role": "system", "content": ""}, {"role": "user", ...}
# 改后: {"role": "user", ...}
```

### 离线加载

容器没网，模型路径需要从 `google/gemma-2-2b-it` 改为本地路径：

```python
state_dict['base_model_name_or_path'] = '/workspace/models/gemma-2-2b-it'
```

### 验证通过

用 `data/sakana_wiki.txt` 测试 internalize + generate，模型成功回答关于 Sakana AI 的问题。

---

## 2. 实验设计

### 核心问题

Doc-to-LoRA 用分块拼接处理长文档（N 块 → rank-8N）。强制限制块数会怎样？

### 测试文本

红楼梦英译本 (H. Bencraft Joly, 1892)，Project Gutenberg eBook #9603。Book I 前 24 回，但实验只取前 8 章 (~85K tokens, 176 块)。

### QA 数据集

派 5 个子代理（全部开灯）分别读 5 回原文，每回出 3 道事实性问答，共 72 题。实验只用前 8 章的 24 题。

### 实验条件

Baseline + N=1, 5, 20, 50, 100, ALL(176)

---

## 3. 实验脚本开发

### 第一版：用官方 internalize() + generate()

失败。`generate()` 内部对 `self.generated_loras` 再次调 `combine_lora`，shape 冲突：`combine_lora` 用 `compute_rank(n, r) = (n+1)*r` 分配空间，但 `generated_loras` 里的 tensor 还是原始 rank，bias 填充时维度不匹配。

```
RuntimeError: The expanded size of the tensor (16) must match the existing size (8)
```

### 第二版：把 ctx_ids 直接传给 generate()

失败。`generate()` 的 ctx_ids 路径会 patch LoRA 的 forward 方法传 `n_qs` 参数，但后续 autoregressive 生成走标准 `Linear.forward()`，不认这个参数。

```
TypeError: Linear.forward() got an unexpected keyword argument 'n_qs'
```

### 第三版（最终版）：手动走完整流程

拆开 Doc-to-LoRA 的黑盒，手动控制每一步：

1. 逐块调 `generate_weights()` 得到 rank-8 LoRA
2. 中间结果存 CPU 省显存
3. 沿 dim=0 拼接后调 `combine_lora` 合并
4. 调 `apply_lora_to_layers` 挂载 LoRA hook
5. 直接调 `base_model.generate()` 生成，绕过 `ModulatedPretrainedModel.generate()`

### OOM 事件

第一次跑全文档 505 块，rank-4040，在 internalize 阶段被 OOM killed。改为只取前 8 章 (176 块)，并加了 CPU offload + gc.collect + cuda.empty_cache。

---

## 4. 实验结果

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

### 关键发现

**块越多，准确率越低。** N≥50 时模型输出完全退化为重复 token：

- N=50: `"îng îng îng îng îng..."` 或 `"первыхпервыхпервых..."`
- 基础模型的输出分布被高 rank LoRA 彻底扭曲

### 解读

1. **Baseline 25%**: Gemma-2B 预训练数据里包含红楼梦相关知识
2. **N=1 (16.7%)**: LoRA 注入了 512 tokens 的局部信息，反而干扰了模型的全局知识
3. **N≥5 持续下降**: rank 线性增长带来的噪声累积超过了信息增益
4. **N≥50 归零**: LoRA 矩阵太大，把基础模型的语言能力彻底摧毁

### 与 110 期文章的对应

这恰好验证了文章中的核心论点：

> 分块拼接是"堆硬盘"思维——用更大的 rank 保证信息不丢。真正的设计应该让 LoRA 大小和文档长度解耦。

SHY（突触稳态假说）预测的现象在这里完美复现：**没有选择性削减的全面增强 = 信噪比崩溃**。Doc-to-LoRA 学会了"睡觉"（把文档从 context 挪进权重），但还没学会"做梦"（主动丢弃冗余信息）。

### 注意事项

- 超网络用的是 `gemma_2b_d2l/checkpoint-20000`（正式版，训练时 max_ctx_chunk_len=512）
- 超网络训练时只见过最多 8 块的输入（num_chunk_probs 里最大 key 是 8），176 块远超训练分布
- 这个退化可能部分是 OOD（out-of-distribution）问题，不完全是架构缺陷
- 但 N=1 < Baseline 这个结果是在训练分布内的，说明即使单块，LoRA 对已知知识也有干扰
