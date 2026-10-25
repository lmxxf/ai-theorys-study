# 为什么离线跑分赢了 1.7ms 的 `hip-port`，进游戏帧率反而不如原厂 PTX 魔改稳？

**日期**：2026-10-25  
**标签**：`DLSS5` / `RTX 3080 Laptop` / `D3D12-CUDA Interop` / `Frame Pacing` / `WDDM调度`

---

## 一、现象：离线跑分反超，实机帧率反而抖动

在 RTX 3080 Laptop（`sm_86`，132W）上，我们把从 AMD 9070 XT 迁过来的独立 CUDA 引擎（`hip-port` Phase 10，`LmxxfNrRuntime.dll`）经过十轮重构，在 720p / 900p / 1080p 三档离线基准测试里全面反超了老黄原版 `310.9.16.0` PTX 满血 `O4` 魔改版（`ptx-patcher` `v0.1.0`）：

- **离线 720p（`1280×768`）算子总耗时**：
  - 老黄原厂 PTX 魔改版（`ptx-patcher`）：**`15.83 ms`**
  - 独立 CUDA 引擎（`hip-port` Phase 10）：**`14.10 ms`**（最快 `13.84 ms`，净胜 **`1.73 ms`**）

然而装进《鬼武者：剑之道》（2K 高画质 + 超分“高性能”，即 `1280×720 -> 2560×1440`）实机对比时：
- **原厂 PTX 魔改版（`ptx-patcher`）**：稳在 **31 fps**，帧生成时间（Frame Pacing）非常平顺；
- **独立 CUDA 引擎（`hip-port` Phase 10）**：帧率在 **30～33 fps** 之间来回晃，不仅没有拉开帧率优势，体感稳定性反而不如原厂 PTX 魔改版。

为什么**内核更快**的独立引擎，放进真实游戏渲染管线里反而**更抖**？

---

## 二、排查实机日志与桥接层源码：三个外围吃性能/吃稳定性的凶手

顺着实机打出的 `OptiScaler.log`、`native-re9-runtime.txt` 和 [`hip_d3d12_bridge.h`](file:///home/lmxxf/work/DLSS5-RTX-3080-laptop/Development/hip-port/Development/HIP/hip_d3d12_bridge.h) 一路查下去，原因根本不在那 32 个 CUDA 算子本身，而在**算子外围的几何扩边与 D3D12 $\leftrightarrow$ CUDA 跨 API 桥接架构**。

### 1. 凶手一：`1280×720` 在实机里被扩成了 `1344×768`，平白多算了 5% 的像素

实机运行日志 `native-re9-runtime.txt` 打出了这样一行：
```text
lmxxf: geometry net=1280x720 color_job=1280x720 proc=1344x768
```
离线跑分测 720p 档位时，网络跑的是 `1280×768`（`20 × 12 = 240` 个 `64×64` tile）。  
但在游戏实机启用 `DLSS5_NETWORK_FREE_RES=1` + `SmallProcessing`（[`native_network_geometry.h:69-73`](file:///home/lmxxf/work/DLSS5-RTX-3080-laptop/Development/hip-port/src/native_network_geometry.h#L69-L73)）时，因为 `1280` 和 `768` 同时是 `256`（`4 × 64`）的整数倍，为了保证 ViT 边缘留出零填充 token 防止画面偏移，几何推导规则强制给宽度额外加了一个 `64`：
$$1280 \times 768 \longrightarrow 1344 \times 768 \quad (21 \times 12 = 252 \text{ tiles})$$
像素量和 tile 数量直接涨了 **5%**，仅这一项就把离线省下来的 `1.73 ms` 吃掉了约 `0.7 ms`。

### 2. 凶手二：D3D12 命令列表“腰斩”与跨 API 双重 Fence 握手（帧率抖动的真正元凶）

这是两条路线在架构层级上最本质的区别：

- **老黄原厂 PTX 魔改路线（`ptx-patcher` / `nvngx_dlssnr.dll`）**：
  走的是 NVIDIA 驱动原生的 NGX / Streamline D3D12 内部通道。对于游戏引擎来说，它就是在同一个 D3D12 上下文和命令流里原地完成提交，**不需要劈开游戏的 D3D12 命令列表，也不需要跨 API 队列同步**。GPU 硬件调度器看到的是一条连续、完整的帧管线，所以帧时间极其稳定。

- **独立引擎路线（`hip-port` / `LmxxfNrRuntime.dll`）**：
  这套 `AmdBridge` 桥接层最初是为了在 AMD 9070 XT 上把 D3D12 游戏纹理喂给外部 HIP Runtime 而设计的（`HookToDevice lmxxf same-frame: open-list proxy`）。在每一帧的渲染中途，它必须强行做完以下 **5 步跨 API 接力**：
  1. **腰斩游戏命令列表（Split CommandList）**：拦截游戏正在录制的 `ID3D12GraphicsCommandList`，在 DLSS 超分前强行把它从中间劈成两半，提前 `Close()` 并调用 `ExecuteCommandLists` 提交前半截（`producerSubmitted`）；
  2. **额外显存拷贝（Staging Copy）**：录制 D3D12 `CopyBufferRegion`，把 `1280×720×16B` 的输入缓冲和 `1280×720×16B` 的历史帧拷进 D3D12 与 CUDA 共享的 External Memory 缓冲区；
  3. **第一次跨 API 握手（D3D12 $\rightarrow$ CUDA）**：D3D12 队列执行 `queue->Signal(fence, ++value)`，独立的 CUDA Stream 调用 `cuWaitExternalSemaphoresAsync` 挂起等待；
  4. **散装发射 75 个 CUDA Kernel（`HIP_GRAPH=0`）**：因为在 Staged Bridge 模式下 CUDA Graph 默认关闭，CPU 主线程要在独立 CUDA Stream 上逐个发射 75 次 Kernel，并调用一次 `cuStreamQuery` 冲刷 WDDM 批处理队列；
  5. **第二次跨 API 握手（CUDA $\rightarrow$ D3D12）**：CUDA Stream 调用 `cuSignalExternalSemaphoresAsync`，D3D12 队列调用 `queue->Wait(fence, value)`，最后再录制并提交游戏后半截的 D3D12 命令列表（`continuationSubmitted`）。

在 Windows 的 WDDM 硬件调度器下，**D3D12 Direct Queue** 和 **CUDA Compute Stream** 属于两个独立的硬件上下文队列。每帧劈开一次 D3D12 命令列表 + 两次跨队列外部信号量（External Semaphore）握手，不仅固定多出 `1.0～1.5 ms` 的拷贝与同步开销，更致命的是**只要游戏后半截 D3D12 渲染队列与下一帧状态在 WDDM 上下文切换边界发生争用，就会产生 `1～3 ms` 的调度抖动（Jitter）**——反映在屏幕上，就是帧率在 `30～33 fps` 之间来回跳。

---

## 三、顺手记一笔：首次进游戏时抓到的两个“跨阵营”乌龙

今天第一次把 `hip-port` 装进《鬼武者》时，为什么测出来也是稳稳的 31 帧？翻了 `OptiScaler.log` 才发现两个令人喷饭的乌龙：
1. **`dxgi.dll` 里的 `IsAmd` 显卡厂商锁**：9070 XT 底包的 `OptiScaler`（`dxgi.dll`）在 `DlssNr::AmdBridge::IsAmd`（文件偏移 `0x2b53f`）里硬编码了 `cmpl $0x1002, VendorId`。看到 3080 Laptop 的 `0x10DE`（NVIDIA），它当场打印 `vendor: 10DE, AMD: false` 并退出了 `AmdBridge`，压根没调 `LmxxfNrRuntime.dll`。
2. **微软商店 GDK `_storage_/` 目录的“幽灵备份”**：`AmdBridge` 退出后，`OptiScaler` 自动 fallback 回 NVIDIA NGX 分支并递归搜索游戏目录，结果在 `Content/_storage_/` 镜像子目录里把昨天残留的 `nvngx_dlssnr.dll`（`ptx-patcher v0.1.0`）捡起来跑了😂！
3. 把 `dxgi.dll` 偏移 `0x2b549` 处的 `75 03`（`jne`）补丁成 `90 90`（`nop; nop`）并清空 `_storage_/` 下的残留 DLL 后，第二次进游戏才真正跑上了 `LmxxfNrRuntime.dll`。

---

## 四、一句话总结

**离线跑分比的是“纯算子吞吐”，实机游戏比的是“管线集成度”。**  
在 NVIDIA 显卡上，我们重写的 CUDA 算子虽然靠寄存器瘦身和 `2-shfl` 直通在纯计算上赢了老黄原厂 PTX `1.7 ms`，但 `AmdBridge` 那套为 A 卡跨 API 绕路设计的“劈开 D3D12 命令列表 + 双重共享 Fence 握手 + 额外共享缓冲拷贝”机制，在 WDDM 调度下天生带着抖动税。这也是为什么在 N 卡上，**直接寄居在老黄原生 NGX D3D12 管线里的 `ptx-patcher`，在帧生成时间（Frame Pacing）稳定性上有着天然的主场优势**。
