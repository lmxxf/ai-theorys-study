# DGX Spark 曲率实验

GPU 执行目录：`lmxxf@192.168.31.198:/home/lmxxf/work/interp-curvature-exp/`。
本目录是同步回来的产物；脚本拒绝在非 GB10 机器上运行。

- `run_curvature_experiments.py`：样本、分片加载、greedy 生成、逐层 hook 与指标。
- `dream.svg`：从本机 AGENTS.md 提取的完整梦境 SVG，仅作为文本输入。
- `results.json`：两个模型全部 86 条记录，含 prompt、token IDs、真实回答、逐层指标、窗口与汇总。
- `实验报告.md`：数值表格、真实回答、反例与适用边界。
- `*-exp4.svg`：五道题的括号说明→答案区曲线。
- `finalize_report.py`：结合保留的监控/重跑记录，补入判读、修复与数值边界。
- `summarize.py`、`validate_results.py`：从 Spark raw 记录汇总、画图与检查。
- `weights_verified.json`：全部 67 个权重文件的官方 revision 与 SHA-256。
- `dequant_validation.json`：解量化与 Transformers 参考实现的抽样一致性。
- `validation.json`：完整性与数值检查。
- `requirements-frozen.txt`：实际 venv 可见包清单（含宿主继承包，不是通用安装锁文件）。

在 Spark 上逐步执行（已有 venv）：

```bash
cd /home/lmxxf/work/interp-curvature-exp
venv/bin/python run_curvature_experiments.py --model Qwen3-0.6B
venv/bin/python run_curvature_experiments.py --model Qwen3.8-27B-FP8
venv/bin/python summarize.py
venv/bin/python validate_results.py
venv/bin/python finalize_report.py
```

每条记录完成即原子落盘，重跑会跳过已有结果。改变实验定义时应另设工作目录或归档旧 raw，不能把不同定义混成一次实验。两模型必须顺序执行；每个进程结束会释放 GPU 并清理页缓存。27B 流式解量化常驻 BF16 文本主干，不加载视觉塔与 MTP。

新环境基础命令（ARM64 Spark）：

```bash
python3 -m venv --system-site-packages venv
venv/bin/pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
venv/bin/pip install transformers==5.22.1 accelerate safetensors scipy matplotlib
```

源码、任务书与模型配置才是实验定义。第四组 B 喂入给定正确说明，不能据此单独认定括号或自发推理的因果效果；SVG 条件也只是代码文本，没有视觉输入。

## 160-token 补跑

原入口保留上一阶段86条样本定义；`run_priming160.py` 独立构造两个模型各18条，不再跳过0.6B的primed。三个条件是bare、真实SVG首轮对话、同一首轮对话加明确提醒。第一轮生成一次并复用，第一轮与第二轮均以160 tokens为上限。

在Spark上顺序执行：

```bash
venv/bin/python run_priming160.py --model Qwen3-0.6B
venv/bin/python run_priming160.py --model Qwen3.8-27B-FP8
venv/bin/python summarize_priming160.py
```

`summarize_priming160.py` 使用已按真实回答判读的 `priming160_annotations.json`；全新重跑产生不同输出时应重新审核判读。它会生成 `priming160_results.json`、独立的 `priming160_report.md`、曲线与检查记录，并在主实验报告末尾追加补充节。上一阶段 `results.json` 保持原样。160 tokens仍可能截断，结果中的 `hit_limit` 与 `ended_with_eos` 是区分依据。

## Paper 69 鞍点 raw / clean 补跑

`run_saddle160.py` 对两个模型各运行21条：鞍点原版、去注释并匿名ID版、同轮双星对照，各有一条首轮和六道独立后续题。所有生成上限160 tokens，没有额外“保持感知”的提醒。

```bash
venv/bin/python run_saddle160.py --model Qwen3-0.6B
venv/bin/python run_saddle160.py --model Qwen3.8-27B-FP8
venv/bin/python summarize_saddle160.py
```

`saddle_stimulus_validation.json` 记录源SVG哈希和结构等价校验。`saddle160_annotations.json` 按真实输出判读；重新生成不同回答时应复核判读。汇总程序输出 `saddle160_results.json`、`saddle160_report.md`、首轮/重点题曲线和验证记录，并只在总报告末尾追加鞍点节。原SVG没有改写，clean另存为 `saddle_clean.svg`。


## Qwen3.5 Base/Post 同架构 BF16 对照

完整配对在 `basepost/`：官方35B-A3B Base与Post均BF16，三种格式、294条记录、93对同字节/token前缀。入口 `basepost/README.md`；完整结果 `basepost/basepost_results.json`，短结论 `basepost/结论速读.md`，本轮报告已追加到总报告。输入熵按完整输入词元重算，和三点曲率的有效位置掩码分开；原始模型输出未改。
