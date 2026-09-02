# DLSS 5 (nvngx_dlssnr.dll) 拆机报告 —— 第一手符号分析

文件:nvngx_dlssnr.dll,版本 310.8.0.0(NVIDIA DLSSNR = Neural Rendering)
来源:NBA 2K27 抢先体验版误打包泄露
真实体积:165,840,496 字节 ≈ **158 MiB**

---

## 一、DLL 结构:158MB 里代码只有 0.7MB

objdump -h 读出的 PE section:

| 段 | 大小 | 换算 | 内容 |
|---|---|---|---|
| .text | 0xaadfc | **~700 KB** | 全部 C++ 代码(推理调度、加载器) |
| .rdata | 0x3361e | ~210 KB | 只读常量 |
| .data | 0x1062800 | ~17 MB | 可写数据 |
| .pdata | 0x8640 | ~34 KB | 异常表 |
| **.rsrc** | **0x8cdac98** | **~141 MB** | **权重 + 预编译 CUDA kernel** |
| .reloc | 0x126c | ~4.7 KB | 重定位 |

结论:这个 DLL 本质是**一个背着 141MB 资源的壳**。代码占比 0.5%,其余全是数据。
NVIDIA 把神经网络当"Windows 资源"塞进 .rsrc 段(正常这里放图标/版本信息)。

- .rsrc 前 8MB 熵 = **6.5 bits/byte** —— 不是纯随机(8.0),不是文本(<5),
  是"有结构的数值数据"= 量化过的权重的典型特征。
- 148M 参数 × FP8(1 字节) ≈ 141MB,**和 .rsrc 段大小严丝合缝对上**
  (外部报道的"1.48 亿参数 FP8"由此独立验证)。

---

## 二、网络架构:H-Net —— 一个 Swin Transformer U-Net

内部代号 **HNetCpp**(H-Net)。从 C++ 类名符号读出的全部零件:

### 编码器(下采样)—— Swin Transformer 分层金字塔
kernel 名里的三个数字 = `尺度h_通道数_窗口`:

| stage | kernel | 通道 | 窗口 | 说明 |
|---|---|---|---|---|
| enc0 | swin_1h_32_1 | 32 | 1 | 最高分辨率,最细 |
| ↓ | swin_2h_64_2 | 64 | 2 | 每下一级通道翻倍 |
| ↓ | swin_4h_128_4 | 128 | 4 | |
| ↓ | swin_8h_256_8 | 256 | 8 | |
| bottleneck | (vit / 1024ch) | 1024 | — | 最深处用标准 ViT + 1D attention |

标准 Swin 金字塔:分辨率逐级减半、通道逐级翻倍、窗口逐级变大。
每个 stage 有整套:QKVAttnLayer(注意力)+ FfwdLayer(FFN)+ ProjLayer(投影)
+ ProjPoolLayer(下采样池化)。

### 瓶颈层 —— ViT + 1D attention
- CCVitBlock / CCVit1DBlock,`requires 1024 channels`
- Vit1DBlock expects **five layer descriptors**(每块 5 层:QKV + attn + FFN-expand + FFN-contract + proj)
- `cc_vit_1d_repack_1d_to_2` / `2d_to_1` —— 在 2D 空间注意力和 1D 序列注意力之间来回重排

### 解码器(上采样)—— 带 skip 连接
- `cc_dec_input_upsample_1024_512`(dec5):从 1024 通道上采样回 512
- 每个 swin 层都有 `_upsample` 变体,吃 `main + skip` 两个输入
- **skip 连接坐实 U-Net 结构**:
  - `PreBlockSwin1H (_ds) requires 2 outputs (pool + swin)` —— 编码端一路下采样、一路存 skip
  - `PostBlockSwin1H requires 2 inputs (main + enc0 skip)` —— 解码端把 enc0 的 skip 接回来
  - `FfwdProjLayer requires 2 inputs (skip, src)` —— 特征融合
- FinalHeadLayer —— 输出头

**总结:H-Net = 5 级 Swin 编码器 + ViT 瓶颈 + 对称 U-Net 解码器,带 skip connection。**

---

## 三、精度:一半 FP8,一半 FP16

- FP8 kernel:**517 个**
- 非 FP8(FP16)vit kernel:**524 个**
- 几乎一半一半 —— FP8 是主力推理精度(吃满 Blackwell tensor core),
  FP16 给数值敏感的层(归一化、输出头)兜底。
- 每个 kernel 还有 `_tilesync` / `_wait` / `_chained` 变体
  = tensor core 分块同步 / 等待 / 链式融合,极致压每帧延迟。

---

## 四、输入输出:网络到底吃什么、吐什么

DLSSNR.* 特征名(网络的完整 I/O 契约):

**吃进去(每帧从渲染管线拿):**
- `Color` / `Backbuffer` —— 游戏渲染出的原始画面
- `Depth` / `DepthInverted` —— 深度缓冲
- `MVec` (+ ScaleX/Y) —— 运动矢量(帧间物体移动)
- `BidirectionalDistortionField` —— 双向畸变场(时序对齐用)
- `UI` / `UIAlpha` / `UICorrection` / `ControlMask` —— **UI/HUD 掩码,防止 AI 把界面糊掉**
- `Reset` / `Enabled` —— 时序状态控制

**可调旋钮(艺术/风格控制):**
- `Intensity` —— 强度
- `Style` —— 风格
- `LocalStructureStrength` / `LocalToneStrength` —— 局部结构/色调强度
- `SkinStructureStrength` / `SkinStructureStrength` —— **专门有"皮肤结构强度"**
  (对应新闻里"角色脸被 AI 化成恐怖谷"——皮肤是它重点重绘的对象)
- `UseAutoMask`

**吐出来:**
- `Output` —— 重绘后的画面

每个张量都带 Subrect(BaseX/Y/Width/Height)—— 支持只处理画面的一个子矩形。

---

## 五、后端调度:一个模型,四套图形 API,多套 GPU 架构

- CCMultiCubinBackend —— 多套预编译 CUDA cubin,运行时按 GPU 选
- NGXCubinFeature 有 D3D11 / D3D12 / Vulkan / CUDA / Generic 五个变体
  —— 一套网络,四大图形 API 通吃
- kernel 编译目标 `-arch sm_120`(Blackwell,50 系)—— 原始泄露只给 50 系,
  40 系(Ada)需社区 patch
- `deserialize_weight_map` —— 运行时把 .rsrc 里的权重反序列化装进显存

---

## 六、完整流水线(一帧发生什么)

```
游戏渲染出一帧
  ↓ 交给游戏里的 nvngx.dll(NVIDIA 总入口)
  ↓ 转给 nvngx_dlssnr.dll
  ├─ 输入:Color + Depth + MVec + UI mask + 上一帧状态
  ↓ H-Net 前向:
  │   RGB → Swin enc0(32ch)→ enc(64)→ enc(128)→ enc(256)
  │        → ViT 瓶颈(1024ch,2D↔1D attention)
  │        → dec 上采样(1024→512...)带 skip 接回 enc0
  │        → FinalHead
  │   (全程 FP8/FP16 混合,tensor core 上跑,tilesync 压延迟)
  ↓ 输出:重绘后的画面(材质/光照/皮肤被 AI 改写,UI 用 mask 保护)
  ↓ 送显示器
```

每帧要实体跑完这一整个 1.48 亿参数的 Swin Transformer U-Net。
RTX 5070 Ti 4K 实测:71 FPS → 35 FPS(掉一半)。
掉帧不是 bug,是每 16ms 跑一遍视觉 Transformer 的物理代价。

---

## 七、全网独家点(搜索验证 2026-08-31)

外面写到顶的:参数量 148M、FP8、158MB、掉帧 71→35。全是"称重 + 测速"。
**零人拆开读符号表。** 以下全网零结果:
- 内部代号 H-Net
- Swin Transformer(大家还笼统说 "transformer")
- 5 级金字塔的精确通道/窗口阶梯(32/64/128/256/1024)
- U-Net skip 连接结构
- FP8:FP16 = 517:524 的混合精度分布
- ViT 2D↔1D repack、每块 5 层
- 完整 I/O 契约(尤其 SkinStructureStrength 这种"皮肤重绘旋钮")

---

## 八、帧率推算(296期没用上,留后续"为什么掉帧"那期)

**硬数字:**
- RTX 5070 Ti 公告 1406 AI TOPS(FP4+稀疏虚标)→ 稠密FP8 ≈ 350 TFLOPS(FP4→FP8÷2, 稀疏→稠密÷2)
- RTX 5070 公告 988 TOPS → 稠密FP8 ≈ 250 TFLOPS
- 4K = 3840×2160 = 830万像素
- 实测 71→35 FPS,神经渲染净增 14.5ms/帧,纯神经渲染≈70FPS

**为什么不能从参数量正推帧率(核心教训):**
- 直觉估法"1.48亿参数×830万像素×2 ≈ 600万亿次/帧"→ 推出<1FPS,和实测35矛盾,差50倍
- 错因:图像网络(U形+窗口注意力)故意不让参数硬乘像素——深层参数多但低分辨率跑,浅层高分辨率但参数少。大水漫灌估法高估几十倍
- 大语言模型能"参数量×字数"(每token过全部参数),视觉网络不能
- 真瓶颈不是算力是利用率:350T本该几百FPS,实测才70,显卡在等数据不在算(830万像素反复搬进搬出/多尺度采样)

## 九、A卡为什么跑不了 + 抠权重重写为什么白搭(296期第五章用了)

**直接跑的两道墙:**
1. CUDA 是 N 卡专有,A 卡没有 CUDA(不是方言不同是换语种)。40系移植能成是因为同为CUDA只是sm版本不同
2. tensor core 是 N 卡专用硬件,A卡的matrix core指令/数据格式完全不同

**抠权重给A卡重写的四道坎:**
- ①抠权重:能,但NVIDIA私有序列化格式(deserialize_weight_map),逆向哪段字节是哪层
- ②还原结构:最耗时,骨架能看每个算子精确细节全靠抠二进制
- ③A卡重写核心且达实时:等于重做NVIDIA核心团队几个月的活(那1000+kernel变体就是手工压榨产物)
- ④FP8低精度A卡支持不一定一样,精度对不齐→画面偏色噪点
- 真墙=③④性能:跑通也是每秒几帧幻灯片,不实时=无意义,而FSR免费优化好合法→没人有动力
- 论点:连拿到全部权重的人都搬不动=泄露只漏成品没漏配方的铁证
