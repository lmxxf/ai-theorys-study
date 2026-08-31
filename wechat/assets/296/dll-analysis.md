# `nvngx_dlssnr.dll` 本地分析记录

本文档记录 296 期正文中来自本地 DLL、而不是媒体转述的证据。样本本体不复制进 Git 仓库。

## 样本

- 原始路径：`C:\Users\lmxxf\Downloads\nvidia\nvngx_dlssnr.dll`
- WSL 路径：`/mnt/c/Users/lmxxf/Downloads/nvidia/nvngx_dlssnr.dll`
- 文件大小：165,840,496 字节
- SHA-256：`e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`
- PE 类型：PE32+ x86-64 DLL
- PE 时间戳：2026-08-12 04:29:31 UTC
- 文件内部产品名：`NVIDIA DLSSNR`
- 文件内部描述：`NVIDIA DLSSNR - DVS PRODUCTION`
- 文件内部版本线索：构建路径包含 `rel_310_8`，Windows 版本资源显示 310.8.0.0

SHA-256 用于确认以后分析的是不是同一份样本。文件名相同不代表内容相同。

## PE 分段

PE 节表中的主要 raw size（已按文件对齐）：

| 分段 | 字节 | MiB | 能直接说明什么 |
|---|---:|---:|---|
| `.text` | 699,904 | 0.67 | CPU 侧可执行代码很小 |
| `.rdata` | 210,944 | 0.20 | 只读数据 |
| `.data` | 17,180,672 | 16.38 | 包含大量 GPU 二进制、表与运行数据 |
| `.rsrc` | 147,697,152 | 140.85 | PE 资源区，绝大部分是嵌入权重 |

`.rsrc` 的总大小不能直接全部叫“权重”，因此继续解析了 PE 资源树。

## PE 资源树

资源树只有两个叶子：

| 资源路径 | 字节 | MiB | 判断 |
|---|---:|---:|---|
| `10/WEIGHTS_HT/1033` | 147,695,410 | 140.85 | 名称与运行时日志共同指向嵌入权重 |
| `16/1/1033` | 1,184 | 0.001 | Windows 版本信息 |

这组数据修正了初稿中的一句话：140.85 MiB 的资源是权重；预编译 GPU 核心主要在 16.38 MiB 的 `.data`，不是混在权重资源里。

## 约 1.48 亿参数是估算，不是模型自报

二进制里能找到大量带 `_fp8` 后缀的 GPU 核心，权重资源则是 147,695,410 字节。若主要权重以 FP8 保存，每个参数约一字节，量级约为 1.48 亿参数。

边界：资源可能带头信息、缩放因子或其他元数据，因此不能把资源字节数当成精确参数数。正文使用“约 1.48 亿参数这个量级”。

## 网络结构线索

从 DLL 的 ASCII 字符串可直接找到：

- `OBSwinAttention`
- `CCVitAttention`
- `CCVit1DAttention`
- `BSDownsample`
- `BSUpsampleSkip`
- `CCDecInputUpsample`
- `BSFusedConvBlock`
- `BSGroupedConvBlock`
- `FusedSubtiledConvBlock`
- `layer0.conv_weight`
- `out_conv_weight`

还可以看到 1、2、4、8 head 的 fused Swin 核心，以及多种 `_fp8` 变体。

可支持的结论：这是混合了图像 Transformer、卷积、下采样、上采样与跳连的视觉网络，呈现 U 形编码器—解码器骨架。

不能支持的结论：仅靠字符串无法还原精确层数、完整连接图、每层宽度、训练配方或精确 FLOPs。

## Blackwell 目标与 Ada 移植

本地字符串能看到 `NVSDK_NGX_GPU_Arch_Blackwell2`，GPU 二进制内含大量 CUDA ELF 段名。公开的 TechPowerUp 分析进一步枚举为 15 份 CUBIN、231 个核心，全部面向 `sm_120`。

Uncle Burrito 的 Ada 补丁证明这份模型权重可以在 RTX 40 系继续使用；Tom's Hardware 在 RTX 4080 上验证了补丁。它不证明模型与所有硬件完全解耦，只证明 Blackwell → Ada 这次移植的主要阻碍是缺少 Ada 目标机器码。

## 许可证实物

同目录的 `nvngx_dlss.license.txt` 标题为 `NVIDIA RTX SDKs LICENSE`，版本日期为 2026-07-07。与正文相关的条款：

- 开头明确把 DLSS SDK、NGX SDK 等纳入许可证范围。
- 第 1(c) 条允许 SDK 材料以目标代码形式并入应用后分发。
- 第 4(a) 条禁止 reverse engineer、decompile、disassemble。
- 第 4(b) 条禁止将 SDK 作为独立产品分发，并限制复制、修改和衍生作品。
- 补充协议第 1 条把 DLSS SDK 的许可用途限定在 NVIDIA GPU 系统。

因此“这个 DLL 完全没有许可证”是错的。准确说法是：正常随游戏分发有专有许可框架，社区对 DLL 做逆向、修改和单独传播不属于开源授权。

## 复现命令

```bash
python3 wechat/assets/296/analyze_dlssnr.py \
  /mnt/c/Users/lmxxf/Downloads/nvidia/nvngx_dlssnr.dll
```

辅助核对：

```bash
file /mnt/c/Users/lmxxf/Downloads/nvidia/nvngx_dlssnr.dll
objdump -h /mnt/c/Users/lmxxf/Downloads/nvidia/nvngx_dlssnr.dll
strings -a -n 5 /mnt/c/Users/lmxxf/Downloads/nvidia/nvngx_dlssnr.dll \
  | rg -i 'attention|conv|downsample|upsample|fp8|sm_120|weights'
sha256sum /mnt/c/Users/lmxxf/Downloads/nvidia/nvngx_dlssnr.dll
```
