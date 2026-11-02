# Qwen3.5-35B-A3B Base / Post 对照

GPU只允许DGX Spark GB10。执行目录：`/home/lmxxf/work/interp-curvature-exp/basepost/`。

模型：官方Qwen3.5-35B-A3B-Base与Qwen3.5-35B-A3B，均原生BF16。首选Hugging Face配置/版本元数据，权重从ModelScope官方同名镜像下载；28个分片的大小和SHA-256与锁定的HF版本全部一致。未改用9B或FP8。

## 运行

```bash
cd /home/lmxxf/work/interp-curvature-exp/basepost
../venv/bin/python prepare_protocol.py
../venv/bin/python run_basepost.py --model Qwen3.5-35B-A3B-Base
../venv/bin/python run_basepost.py --model Qwen3.5-35B-A3B
../venv/bin/python summarize_basepost.py
```

`run_all.sh` 已按以上顺序在后台运行。权重未验证完毕时等待，不提前加载不完整文件。每案原子落盘，重跑跳过已完成记录；修改关键测量/装载代码后应检查源码指纹并明确版本边界，不能混同不同实验定义。

- `status.py`：权重下载/校验与案例进度。range-part文件预分配大小不是实际进度，只按ranges状态里的已完成块计算。
- `download_ranges.py`：ModelScope国内源分段续传，最终以HF SHA验证再改正式文件名；每模型写 `BASEPOST_READY.json`。
- `load_pair.py`：meta骨架，逐分片safe_open CUDA、统一BF16、每片清页缓存。文本常驻约64.56GiB，最大单参数1GiB。每次只驻留一版。
- `protocol.json`：冻结294条（每版147），其中96条仅前向、198条生成；所有生成上限160。
- `raw/<model>/bp_*.json`：逐案完整数据，留在Spark。
- `basepost_results.json`：全量合并，含输入/输出/token/全层几何/选定层logit lens/路由观测/源码和模型来源。
- `实验报告.md`：本节独立报告；汇总器还会在父目录实验报告末尾追加本节，保留早期数据。
- `weight_sample_audit.json`：少量CPU切片核实有效BF16权重并非同一检查点副本，不把文件头/分片顺序变化误当权重变化。

## 对照含义

两版实际都提供同一7756字节内嵌chat template，raw、QA、chat都跑；所有预设非自然历史的93对输入字节和token完全一致。EOS统一[248044,248046]，pad248044。

SVG三条件（普通图表、双星、clean鞍点）使用同一句感受引导。raw SVG首轮给定末尾“我”；报告区分给定前缀和新生成词元。自然首轮历史是各自真实回答，包含文字中介差异；raw格式另有三SVG×六题的固定中间句“我已读。”控制。

H_next在生成前第一次lm_head调用捕获，H_input/conflict/gen来自无缓存重放且明确对齐位置。路由全概率熵与Top8权重熵另测；每词元仍选8个路由专家，另有共享专家。词表预测熵不是流形维数的直接测量。本实验比较官方Base/后训练检查点，不是只开关RLHF的一次训练消融。

## 下载路径记录

HF跨境大文件最初断流；1MiB分段续传与临时本机CPU/SSH回环转发先恢复进度。之后发现并核验ModelScope同文件镜像，切换国内源，临时转发已关闭。独立代理测试也已停止并删除临时私密配置，没有修改Spark现有全局代理选择。保留的公共辅助代码只是复现排障记录，不需要运行代理辅助程序即可用当前下载器。

`append_quick_readout.py` 在汇总后追加本轮结论速读。其文字解释针对本次固定数据快照；若重新生成不同输出或修改病例，应复核解释与数字。
