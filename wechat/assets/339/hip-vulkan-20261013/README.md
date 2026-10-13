# 339 本机 HIP / Vulkan 实测证据

结论见 [本机实测结果](../本机实测结果.md)。本目录保存小型原始记录和实际使用脚本，不包含模型、可执行程序、运行库、下载缓存或私人清理清单。

- `formal-00-hip` 至 `formal-05-vulkan`：六个正式进程的原始 JSON、标准错误日志与启动参数；`formal-receipts.json` 记录真实退出码。
- `bench-summary.json`：每项 3 进程 × 3 内部重复的汇总，含全部样本范围和进程均值。
- `server-hip-*` / `server-vulkan-*`：每种长度先暖一次、再测三次的原始响应和日志；`server-summary.json` 不含暖机。服务由外层脚本在请求结束后主动停止，其 receipt 中 nullable `exit` 不当成自然退出码 0；对应外层脚本均成功完成。
- `official-qwen-q4.json`、`model-*-receipt.json`：模型来源、大小和两端校验。
- `llama-tag.json`、`binary-assets.json`、`common-binary-hashes.json`、`rocm-receipts.json`：源码、官方二进制包和便携依赖身份。
- `preflight-*`：实际 4K 载入与内存记录。`hip-offload-evidence.txt` 是首次诊断日志的相关行摘录，附原日志 SHA；该诊断 pilot 没有进入正式性能统计。
- `COPY-MANIFEST.json`：从实验区原样复制的 44 个文件及 SHA-256，复制后已逐件复核；本索引、结果文章和 HIP 摘录是新生成说明，不冒充原始文件。

## 复现入口

这些脚本保留实际路径 `C:\LLama-Compare-20261013`，先准备相应目录、官方模型和 b11583 两套后端。测试机器上同名目录已存在。

1. `prepare-runtime.ps1`：按 `rocm-receipts.json` 核对已下载的三个官方 wheel，解压便携依赖；不负责联网下载，不改系统 PATH。
2. `verify-model.ps1`：核对固定模型大小和 SHA。
3. `preflight.ps1 -Backend hip` / `-Backend vulkan`：单设备、4K 上下文、短生成，记录 GPU 内存。
4. `run-bench.ps1`：唯一六进程序列，内置暖机，每进程三次重复。
5. `server-timing.ps1 -Backend vulkan`，随后 `-Backend hip`：原始 completion 的首内容等待与总时间。

模型固定 revision 的下载地址由官方元数据与下列路径组成：

`https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/530227a7d994db8eca5ab5ced2fb692b614357fd/Qwen3-14B-Q4_K_M.gguf`

`preflight.ps1` 的设备/驱动和游戏进程检查是本机约束；换机器时应核实设备编号和环境，再调整这些约束。脚本不应与游戏或另一份模型同时运行。原始计时保留首次实际执行所得，不需要为这份资料重新跑一遍。
