## 136 草稿：小模型预训练备忘

### 为什么要从零预训练
- 改了模型架构（比如改 attention 机制、改 head 结构）就没法复用现有权重，必须从零来
- 如果不改架构，直接在开源基座上 QLoRA/SFT 就行，不需要预训练

### 显存需求（7B 模型，FP16 + Adam）

| 项目 | 显存占用 |
|------|---------|
| 模型权重（FP16） | ~14 GB |
| 梯度（FP16） | ~14 GB |
| 优化器状态（Adam FP32） | ~28-56 GB |
| 激活值 | 看 batch size 和 context length |
| **总计** | **~70-120 GB** |

省显存手段：FP8 训练、激活检查点（activation checkpointing）、gradient accumulation、DeepSpeed ZeRO / FSDP

### 开源预训练语料

**英文：**
- FineWeb（HuggingFace，15T token）—— 目前开源最大高质量预训练语料
- FineWeb-Edu（1.3T token）—— FineWeb 的教育类高质量子集
- RedPajama-V2（30T token）—— 多语言，有质量评分标签
- SlimPajama（627B token）—— RedPajama 清洗去重版

**中文：**
- WuDaoCorpora（悟道，200GB）
- MNBVC（超大规模中文语料，社区维护）
- SkyPile（天工，中文网页）

**多语言：**
- CulturaX（6.3T token，167 种语言含中文）

**代码：**
- The Stack V2（StarCoder 用的，多语言代码）

### 现实路线：先小后大

**第一步：1B 规模验证（5090 上跑）**
- 参数量：~1B
- 训练数据：100-200B token
- 显存：~10-15 GB，RTX 5090（32GB）绰绰有余，还能开大 batch size
- 时间：几天
- 对比基线：Qwen3-0.6B、Llama 3.2 1B
- 目的：验证架构改进有没有效果
- 注意：5090 是纯显存，训练速度远快于 Spark 的共享内存

**第二步：7B 规模确认（租云 GPU）**
- 参数量：~7B
- 训练数据：0.5-1T token（够做对比，不需要 2T）
- 硬件：8×A100（租 Lambda/Vast.ai/RunPod，~$15-20/小时）
- 时间：2-3 周
- 成本：$5000-8000
- 对比基线：Qwen3-8B、Llama 3.1 8B、Mistral 7B
- 目的：证明 1B 上的趋势在 7B 上一致

**论文只需要证明趋势一致：1B 有效 + 7B 有效 = 够了**

### DGX Spark 能做什么

| 方案 | 显存需求 | Spark 能跑 | 时间 |
|------|---------|-----------|------|
| QLoRA 微调 7B | 16-24 GB | 轻松 | 几小时 |
| 全参数 SFT 7B | 70-120 GB | 勉强，慢 | 几天 |
| 从零预训练 1B | 10-15 GB | 轻松 | 几天 |
| 从零预训练 7B | 70-120 GB | 理论能跑，实际等到死 | 别想了 |

### 参考成本

| 规模 | 硬件 | 数据量 | 时间 | 成本 |
|------|------|--------|------|------|
| 1B | DGX Spark | 100-200B token | 几天 | 免费（自有） |
| 7B | 8×A100 租用 | 0.5-1T token | 2-3 周 | $5000-8000 |
| 7B（Llama 2 级别） | 1000×A100 | 2T token | 数周 | $100万+ |
