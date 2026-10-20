# DeepGEMM 和 TileLang：DeepSeek V4 官方推理里的两套算子到底是什么关系？

**日期：2026-10-20**

---

## 起因：一个一直没串起来的疑问

DeepSeek V4 发布前，我去翻了 DeepSeek 官方开源的 `DeepGEMM` 仓库，发现他们的代码目录里突然多了一堆 `sm_100`（数据中心级 Blackwell，即 B200/GB200）的目录和宏定义——当时心里就犯嘀咕：“DeepSeek 这工业级 Blackwell 卡没少备货啊，而且支持新架构的新模型肯定快发了。”

结果等 DeepSeek V4 真正发布、我们去读官方随权重附带的推理参考脚本（`inference/model.py` + `inference/kernel.py`）时，里面满眼却全是用 **`TileLang`** 写的算子，压根没见着 `DeepGEMM` 的影子。

这就产生了一个直觉上的困惑：**既然官方推理脚本用的是 `TileLang`，那他们之前在 `DeepGEMM` 里猛加 `sm_100` 代码是干嘛用的？难道 `TileLang` 底层依赖 `DeepGEMM` 吗？**

今天把两边的源码和生产推理栈对了一遍，彻底破案了。

---

## 结论：两者毫无依赖关系，一个是“门口的塑料样板间”，一个是“机房里的重型机床”

**`TileLang` 底层完全不依赖 `DeepGEMM`，`DeepGEMM` 也完全不用 `TileLang`。** 它们是给两个完全不同的场景准备的平行实现：

### 1. 官方仓库里的 `inference/kernel.py`（100% 用 `TileLang`，教学/参考样板间）

在 DeepSeek V4 官方 Hugging Face 仓库自带的极简参考代码 `inference/model.py` 里，开头只有这一句算子导入：

```python
from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
```

在原版 `inference/kernel.py` 中，**这全部 6 个核心算子（包含 `fp8_gemm`、`fp4_gemm` 矩阵乘法，以及稀疏注意力 `sparse_attn`、超连接 `hc_split_sinkhorn` 和量化函数），统统是用 `TileLang`（`@tilelang.jit`）写的**（外加一个外部小库 `fast_hadamard_transform`），从头到尾没有 `import deep_gemm`。

**为什么官方参考脚本要用 `TileLang`？**
- 因为随模型权重放出的 `inference/` 脚本，定位是**给研究者和同行看清模型数学逻辑的极简参考实现**（没有 Continuous Batching，也没有复杂的分页缓存管理）。
- 如果往这个几百行的 Python 参考目录里塞几万行底层 C++/CuTe 模板，没人读得懂；而当时 OpenAI 的 Triton 对 `sm_100` 的新指令支持还没跟上，于是他们用 `TileLang`（TVM 系团队做的 Python DSL 编译器，定位类似 Triton）写了 6 个短平快的算子放进 `inference/` 目录交差。

### 2. 生产级推理框架 `vLLM / SGLang`（真刀真枪调 `DeepGEMM`）

那 DeepSeek 自己机房里跑线上服务、以及喂给 `vLLM` 和 `SGLang` 主线的算子是谁？**这才是 `DeepGEMM` 的主场。**

- **`DeepGEMM` 是用什么写的？**
  它是直接用 **原生 CUDA C++ + NVIDIA 的 `CuTe`（CUTLASS 的底层张量抽象层）+ 少量内联 `PTX` 汇编** 手写的工业级算子库，运行时通过轻量 JIT 编译成 `.so` 调用。
- **它在生产环境里管什么？**
  在 `vLLM` / `SGLang` 跑 DeepSeek V4 的生产路径上，框架根本不会去调那个教学用的 `TileLang` 版 `kernel.py`，而是直接调用 `DeepGEMM` 来算最吃性能的 **FP8 / FP4 矩阵乘法（含稠密层与 MoE 路由专家的 `fp8_fp4_gemm`）** 以及 **mHC 超连接算子（`hyperconnection.hpp`）**。
- 所以我在 V4 发布前从 `DeepGEMM` 仓库里看到的那些 `sm_100` 目录和宏，正是 DeepSeek 的核心算子团队**提前在自家 B200 集群上把生产环境算子写好、测透，为了在 V4 发布当天直接对接到 `vLLM` / `SGLang` 生产栈里去的**。

---

## 附：一张表看清四个场景下到底是谁在干活

为什么我们在折腾消费级 Blackwell（RTX 5090 / RTX PRO 6000 的 `sm_120`，以及 DGX Spark 的 `sm_121`）时，老把 `TileLang` 和 `DeepGEMM` 放在一起骂？因为这哥俩当时都只写了数据中心卡（`sm_90` Hopper / `sm_100` B200）的指令路径（如 `tcgen05` 和 `TMA`），在消费级 `sm_120/121` 上双双罢工。

把四个不同运行场景摆在一张表里，谁在哪个位置干活就一清二楚了：

| 运行场景 / 代码入口 | 矩阵乘法（`fp8_gemm` / `fp4_gemm`） | 稀疏注意力（`sparse_attn` / `Indexer`） | 超连接（`mHC`）与激活量化（`act_quant`） |
|---|---|---|---|
| **1. 官方仓库自带 `inference/model.py`（参考脚本）** | **`TileLang`**（`kernel.py`） | **`TileLang`**（`kernel.py`） | **`TileLang`**（`kernel.py`） |
| **2. 数据中心卡（`sm_90/100`）跑 `vLLM / SGLang` 生产栈** | **`DeepGEMM`**（CUDA C++ / CuTe） | **FlashMLA / FlashInfer** | **`DeepGEMM`**（`hyperconnection.hpp`） |
| **3. 我们的 `sm_121` 纯 PyTorch 实验平台（`kernel_sm121.py`）** | **Triton**（`tl.dot_scaled`） | **分块纯 PyTorch** | **纯 PyTorch** |
| **4. 消费级 `sm_120/121` 跑 `vLLM` 社区生产栈** | **`b12x`**（早期用 `jasl` Marlin + CDG） | **定制 FlashInfer** | **Triton 重写版** |

一句话记死：**`TileLang` 是挂在官方权重仓库门口给人看公式的参考脚本玩具；`DeepGEMM` 才是 DeepSeek 一开始就为 `vLLM` 和自家 `sm_100` 生产集群准备好的真家伙。**
