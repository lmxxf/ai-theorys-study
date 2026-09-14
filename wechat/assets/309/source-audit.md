# 309 审稿记录

日期：2026-09-14。闇之朱雀。

范围：直接修订 `wechat/309.md` 和 `structure.svg`。保留七章结构、“成品是过程的零头”和学习者复盘语气。没有修改、构建或部署移植仓库的游戏插件，也没有改写历史开发记录。

## 固定对象与方法

- 移植仓库 main：`85feab01987b59afe8311d30f2c8bbaf06c94d6d`，工作树干净。外层仓库原来已有子模块指针变化与 `arxiv/wechat168/` 未跟踪文件，本次不处理。
- 全读用户提供的跨会话上下文（111 行）和 DevHistory（626 行）。DevHistory 中部的“当前状态”“待办”仍是旧快照，后续时间线已推翻 C32 溢出、单靠驻留优先级等结论。720p／900p 不在当前 main 中，文章因此固定到 1080p main。
- 逐段检查主钩子、C64 驱动、PSO/驻留包装器、QKV 着色器、H/fast_exp/SAT8、构建脚本及原 SVG。301、304 的回链内容用本地原文核对，用户给出的四条微信链接保持原样。
- `verify_sources.py` 统计 Git 跟踪文件；`source-checks.json` 保存结果及源码 SHA256。`src + shaders = 9969` 物理行，四目录合计 11329 行；脚本目录是 1271 行而非 925。物理行含注释和空行，不能当工作量指标。
- AMD PDF 第 15 页既读文字又渲染查看。下载 SDK v1.1.4 的三个原始头文件，用 `x86_64-w64-mingw32-g++ -std=c++17` 对 Windows x64 的 17 个字段位置做 `static_assert(offsetof(...))`，全部编译通过。头文件 SHA256 保存在 JSON。

## 主张对照

| 原稿主张 | 核查结果 | 修订 |
|---|---|---|
| CPU 一次浮点乘法都不做 | `native_c64.h` 的权重量化/缩放等明确做浮点运算 | CPU 管资源、权重准备、参数与提交；每帧网络主运算在 GPU |
| 21.5ms GPU + 2～3ms CPU 录制 | 两数字出自不同场景；后者是整帧差额 | 21.5ms 限定为 09-11 无游戏争抢的动画源；差额不当 CPU 录制计时 |
| ffxDispatch 返回就是 FSR 算完 | 该调用录制上采样命令；钩子还等待游戏提交对应列表 | 明确录制、提交、执行，网络用自有列表追加同队列 |
| 网络直接吃原低分辨率色/深度 | 当前主钩子读取 output 与 motion；网络色源来自 FSR 输出 | 正文与图按实际输入路径表述 |
| +360 就是运动缩放，全部字段与官方一致 | +360 jitter、+368 motionVectorScale；当前主钩子赋值与日志自相矛盾 | 拆表并把这次源码复查发现写入第二章，详见下节 |
| 归一化运动向量范围是 0～1 | 位移有正负，归一化表示单位，不是取值范围保证 | 改为按尺寸归一化的位移 |
| 71 块 = 71 次 Record / 一块一个头文件 | 通道家族复用，ViT 分拆，有专门升降采样块及默认跳块 | 按组件/家族组织，不混 block、kernel、source、CSO、PSO |
| 设优先级保证先驱逐游戏贴图 | 包装器旧注释过度断言；后续 DevHistory 说单靠优先级不够 | 以实测和 API 语义为准，补游戏贴图池/降画质与网络省显存 |
| Load 最后 16 表示元素位数 | v1.10.2605.24 `linalg.h:300-304` 参数为 Align；前面的 Stride 是字节 | 改成 16 字节对齐承诺；E4M3 每元素 1 字节 |
| 一条 Multiply 等于一条机器指令 | 接口矩阵操作不承诺一条机器指令实现 | 说明 16×32×16 的 8192 项乘加，避免混层 |
| GetCoordinate 表示不能控制性能 | 线程元素分配由实现决定，但可改块形状/布局/并行度 | 保留“合伙攥着矩阵”比喻，删除“调不动/各自最优” |
| 短 QKV 核末尾 H 就是原版精确行为 | 文件开头注明 fast，完整 K 结束才 H；原精确链有分段 H | 第五、六章明确精确链和快速链边界 |
| H、F 是逐位一致全部内容 | 布局、归约/残差顺序、乘加内部精度、采样等都重要 | 保留构造性优化思想，同时解释所需规则与验证 |
| 浮点位左移约等于指数增加 | 移位作用于整个 half 编码；特定输入区间的尾数重排产生近似 | 按 a∈[1,2) 的固定指数、变化尾数解释特定位映射 |
| NVIDIA 转换天然饱和 | CUDA 有 `__NV_NOSAT`、`__NV_SATFINITE` 两种模式 | 饱和限定到原样本那条路径；AMD 非饱和溢出限定本项目实测 |
| 65 文件最终只有七十多个核 | DevHistory 09-10 曾从零编出 179 CSO，后续有新增 | 使用一百多份变体；179 明确标日期，不报未重编的新数量 |
| FAST PATH 全有效且开着 | 脚本明确保留无收益的多头 FFN+proj0 开关 | 保留 65 次字样计数，去掉等同成功/默认启用的解释 |
| Development 编译用不到所以复制它就有完整上下文 | 编译宿主/shader不依赖；制作权重资产仍需其中脚本，历史含翻案 | 区分编译/资产生成，给 DevHistory→原件→现源码的阅读顺序 |
| 权重约 16GB（图） | README 数字含参考转储；原始 WEIGHTS_HT 147695410 字节 | 图改原始权重资源约 147.7MB，不与预处理资产/显存混淆 |
| IEEE 754-2019 定义 E4M3 | 引用错位，IEEE binary16 不等于该 FP8 编码 | E4M3 改引 NVIDIA 类型与转换文档 |

## 本次发现的运动缩放下标矛盾

固定源码 `src/native_submission_order_probe.cpp`：

- 196 行从 `h + 360` 读 16 字节进 `float mvscale[4]`。
- 199 行用 `mvscale[0]`、`mvscale[1]` 更新 `NativeMotionVectorScale()`。
- 205 行日志将 `[0],[1]` 打印为 jitter，将 `[2],[3]` 打印为 mvscale。
- SDK 与 PDF 对应布局是 jitterOffset 在 360、motionVectorScale 在 368。

因此，在官方这个 ABI 下，赋值应选 `[2],[3]` 或从 368 单独读两个 float。静态布局断言已经通过。这足以否定原稿“所有字段都对上了”，但不等于已经查明运行中二进制、其他分支或每个游戏的实际参数合同。

后续若另做移植修复，先确定当前运行构建/分支，保存一帧完整 descriptor 原始字节，对照对应版本头文件及 motion/render geometry；核对正确系数后的时序 warp 和画面。不在本次审稿里悄悄修改发布代码、切分支或换游戏 DLL。

## 一手来源

1. [AMD FSR 3.1 Integration PDF](https://gpuopen.com/presentations/2024/FidelityFX_Super_Resolution_3-1_Release-Overview_and_Integration.pdf)，第 15 页。
2. [SDK v1.1.4 ffx_upscale.h](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/ffx-api/include/ffx_api/ffx_upscale.h)、同目录 `ffx_api.h`、`ffx_api_types.h`。复核脚本使用未经改写的下载头文件。
3. [FidelityFX 3.1 迁移说明](https://gpuopen.com/manuals/fidelityfx_sdk/getting-started/migrating-to-fsr-3-1/)，ABI-stable structs 与扩展机制。
4. [HLSL Linear Algebra Matrix](https://microsoft.github.io/hlsl-specs/proposals/0035-linalg-matrix/)，Matrix scope、Load 的字节偏移与步长、GetCoordinate。
5. [DXC v1.10.2605.24 linalg.h](https://github.com/microsoft/DirectXShaderCompiler/blob/v1.10.2605.24/tools/clang/lib/Headers/hlsl/dx/linalg.h)，旧接口第五参数 Align。当前在线规范已将部分 Align 改成模板参数，不能直接拿最新写法替换旧源码。
6. [Microsoft Residency](https://learn.microsoft.com/en-us/windows/win32/direct3d12/residency)、[SetResidencyPriority](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device1-setresidencypriority)。
7. [NVIDIA E4M3](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html)、[FP8 转换](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)。
8. [AMD 26.10.07.02](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-10-07-02.html)、[Agility 下载页](https://devblogs.microsoft.com/directx/directx12agility/)、[DXC v1.10.2605.24](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.24)、[实验特性开关](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-d3d12enableexperimentalfeatures)。721 单篇 DevBlogs 正文返回 403，改用官方 SDK 版本表与 DXC 发布说明交叉核对，文章不再依赖无法读取的正文。

## 验证范围

- 成功：源码计数与 SHA256；Windows x64 官方 ABI 的 17 个字段静态断言；SVG XML 解析、渲染与人工检查；文章资源路径与原微信链接检查；`git diff --check`。
- 未做：重新跑原版/AMD GPU 数值对照、重测游戏速度、部署修复。
- HLSL 编译限制：尝试官方 Linux DXC v1.10.2605.24，宿主缺 GLIBC_2.38（libdxcompiler 也要求 GLIBC_2.36）；已有 Debian 12 容器为 GLIBC 2.36，同样不足。未为审稿升级系统。示例按相同版本仓库核和头文件核对，补齐命名空间与 H 声明，但本次不声称着色器编译/设备创建通过。
