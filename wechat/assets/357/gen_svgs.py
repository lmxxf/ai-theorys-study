#!/usr/bin/env python3
"""
Generate 3 vertical SVG diagrams (width=520) for WeChat Issue 357:
1. dual-grid-tl-mechanism.svg (Dual-Grid E2M1*2={0,1,2,3,4,6,8,12} vs INT4 {-7..7} + Two-Level m=1..8 shared E4M3 scale)
2. sass-s4-vs-s8-census.svg (How ptxas compiles mma.s4 vs mma.s8 on sm_80/86/89 vs sm_90/100/120: 1 IMMA vs 2 IMMA + 45 LOP3 + 24 SHF)
3. m8-speed-paradox-breakdown.svg (Why M=1 is 1.81x faster than FP16 while M=8..16 is slower: bandwidth vs dynamic INT8 act-quant & padding)
"""
import subprocess
import tempfile
from pathlib import Path
import cairosvg

OUT_DIR = Path(__file__).resolve().parent

SVG1 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 680" width="520" height="680" font-family="'Noto Sans CJK SC', 'WenQuanYi Micro Hei', sans-serif">
  <rect width="520" height="680" fill="#f8fafc" rx="12"/>
  <text x="260" y="34" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">HINT4 的两项核心巧思：双网格（Dual Grid）与两级缩放（TL）</text>
  <text x="260" y="54" font-size="11" fill="#64748b" text-anchor="middle">把 4-bit 浮点乘以 2 变成精确整数，塞进全系通用的 INT8 Tensor Core</text>

  <!-- Box 1: Dual Grid -->
  <rect x="20" y="72" width="480" height="275" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="72" width="480" height="32" fill="#e0f2fe" rx="8"/>
  <text x="34" y="93" font-size="13" font-weight="bold" fill="#0369a1">第一招：双网格（Dual Grid）—— 借 E4M3 闲置符号位切换两套刻度</text>

  <!-- Grid A: NVFP4 E2M1 -->
  <text x="36" y="124" font-size="12" font-weight="bold" fill="#334155">① 原版 NVFP4（E2M1 浮点正半轴刻度，带小数 .5）：</text>
  <rect x="36" y="132" width="448" height="28" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
  <text x="260" y="151" font-size="12" font-family="monospace" font-weight="bold" fill="#b45309" text-anchor="middle">{ 0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0 }  （必须浮点硬件）</text>

  <!-- Arrow multiply by 2 -->
  <text x="260" y="177" font-size="12" font-weight="bold" fill="#dc2626" text-anchor="middle">↓ 全部乘以 2（缩放因子 S 相应除以 2，乘积不变）↓</text>

  <!-- Grid A*2: Exact Integers -->
  <rect x="36" y="186" width="448" height="30" fill="#dcfce7" stroke="#16a34a" rx="4"/>
  <text x="260" y="206" font-size="12" font-family="monospace" font-weight="bold" fill="#15803d" text-anchor="middle">网格 A（FP4×2）：{ 0, ±1, ±2, ±3, ±4, ±6, ±8, ±12 } （全是整数！）</text>

  <!-- Grid B: Symmetric INT4 -->
  <rect x="36" y="224" width="448" height="30" fill="#e0e7ff" stroke="#4f46e5" rx="4"/>
  <text x="260" y="244" font-size="12" font-family="monospace" font-weight="bold" fill="#3730a3" text-anchor="middle">网格 B（对称 INT4）：{ 0, ±1, ±2, ±3, ±4, ±5, ±6, ±7 } （均匀步长）</text>

  <!-- E4M3 Sign Bit Trick -->
  <rect x="36" y="266" width="448" height="68" fill="#f1f5f9" stroke="#94a3b8" rx="6"/>
  <text x="48" y="286" font-size="11.5" font-weight="bold" fill="#0f172a">标志位藏哪？—— E4M3 缩放因子永远 &gt; 0，最高位（符号位）天生闲置：</text>
  <rect x="48" y="295" width="95" height="28" fill="#fecaca" stroke="#dc2626" rx="4"/>
  <text x="95" y="313" font-size="11" font-weight="bold" fill="#991b1b" text-anchor="middle">1 bit: 0=INT4 / 1=FP4</text>
  <rect x="148" y="295" width="145" height="28" fill="#bfdbfe" stroke="#2563eb" rx="4"/>
  <text x="220" y="313" font-size="11" font-weight="bold" fill="#1e40af" text-anchor="middle">4 bits: 指数 (Exponent)</text>
  <rect x="298" y="295" width="172" height="28" fill="#bbf7d0" stroke="#16a34a" rx="4"/>
  <text x="384" y="313" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">3 bits: 尾数 (Mantissa)</text>

  <!-- Box 2: Two-Level (TL) -->
  <rect x="20" y="364" width="480" height="296" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="364" width="480" height="32" fill="#fae8ff" rx="8"/>
  <text x="34" y="385" font-size="13" font-weight="bold" fill="#86198f">第二招：两级缩放（TL）—— 解决 16 权重分块与 K=32 指令的冲突</text>

  <text x="36" y="416" font-size="11.5" fill="#334155">• 矛盾：NVFP4 每 <tspan font-weight="bold" fill="#0f172a">16 个权重</tspan> 换一次浮点缩放，但 INT8 指令 mma.m16n8k32 一口吃 <tspan font-weight="bold" fill="#dc2626">K=32</tspan>。</text>
  <text x="36" y="436" font-size="11.5" fill="#334155">• 解法：相邻两组（共 32 个权重）共用 1 个 E4M3 缩放 S，各配 1 个小整数倍率 m ∈ {1..8}：</text>

  <!-- Sub-block 0 and Sub-block 1 diagram -->
  <rect x="36" y="450" width="216" height="76" fill="#f0fdf4" stroke="#16a34a" rx="6"/>
  <text x="144" y="470" font-size="12" font-weight="bold" fill="#166534" text-anchor="middle">左半块（16 个 4-bit 权重）</text>
  <text x="144" y="490" font-size="11" fill="#15803d" text-anchor="middle">子块整数倍率 m₀ ∈ {1..8}</text>
  <text x="144" y="510" font-size="11" font-family="monospace" font-weight="bold" fill="#0f172a" text-anchor="middle">寄存器展开：w_i8 = q × m₀</text>

  <rect x="268" y="450" width="216" height="76" fill="#eff6ff" stroke="#2563eb" rx="6"/>
  <text x="376" y="470" font-size="12" font-weight="bold" fill="#1e40af" text-anchor="middle">右半块（16 个 4-bit 权重）</text>
  <text x="376" y="490" font-size="11" fill="#1d4ed8" text-anchor="middle">子块整数倍率 m₁ ∈ {1..8}</text>
  <text x="376" y="510" font-size="11" font-family="monospace" font-weight="bold" fill="#0f172a" text-anchor="middle">寄存器展开：w_i8 = q × m₁</text>

  <!-- Bounds check -->
  <rect x="36" y="538" width="448" height="48" fill="#fffbeb" stroke="#d97706" rx="6"/>
  <text x="260" y="558" font-size="12" font-weight="bold" fill="#b45309" text-anchor="middle">为什么不会溢出 INT8（-128 ~ +127）？</text>
  <text x="260" y="577" font-size="11.5" font-family="monospace" fill="#78350f" text-anchor="middle">最大网格值 (±12) × 最大子倍率 (8) = ±96  ⊆  [-128, +127] （零精度损失！）</text>

  <!-- Final MMA instruction -->
  <rect x="36" y="598" width="448" height="46" fill="#0f172a" rx="6"/>
  <text x="260" y="618" font-size="12" font-family="monospace" font-weight="bold" fill="#38bdf8" text-anchor="middle">1 条 mma.sync.aligned.m16n8k32.s32.s8.s8.s32 吞下全部 32 个权重</text>
  <text x="260" y="635" font-size="11" fill="#cbd5e1" text-anchor="middle">算完 INT32 累加和后，在循环末尾只乘 1 次共用浮点缩放 S_shared</text>
</svg>
"""

SVG2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 560" width="520" height="560" font-family="'Noto Sans CJK SC', 'WenQuanYi Micro Hei', sans-serif">
  <rect width="520" height="560" fill="#f8fafc" rx="12"/>
  <text x="260" y="34" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">PTX 假象 vs SASS 真相：老黄在 Hopper/Blackwell 砍掉了 INT4</text>
  <text x="260" y="54" font-size="11" fill="#64748b" text-anchor="middle">本地 ptxas + cuobjdump 跨六代架构（sm_80 ~ sm_120）实测反汇编结果</text>

  <!-- Left/Top: What PTX promises -->
  <rect x="20" y="72" width="480" height="66" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5" rx="8"/>
  <text x="36" y="95" font-size="12.5" font-weight="bold" fill="#0f172a">源码写同一条 4-bit PTX 指令（PTX 9.3 手册宣称 sm_80 以上全部合法）：</text>
  <text x="260" y="122" font-size="12" font-family="monospace" font-weight="bold" fill="#2563eb" text-anchor="middle">mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32</text>

  <!-- Branch 1: sm_80 / sm_86 / sm_89 -->
  <rect x="20" y="154" width="480" height="118" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5" rx="8"/>
  <text x="36" y="178" font-size="13" font-weight="bold" fill="#166534">① Ampere / Ada（sm_80 A100、sm_86 RTX 3080/3090、sm_89 RTX 4090）：</text>
  <rect x="36" y="190" width="448" height="36" fill="#dcfce7" stroke="#22c55e" rx="4"/>
  <text x="260" y="213" font-size="12.5" font-family="monospace" font-weight="bold" fill="#14532d" text-anchor="middle">SASS: IMMA.16832.S4.S4 R4, R4.ROW, R6.COL, R8  （仅 1 条硬件指令！）</text>
  <text x="36" y="250" font-size="11.5" fill="#15803d">• 额外位运算：0 条 LOP3，0 条 SHF —— 30 系和 40 系确实有原生 INT4 Tensor Core。</text>

  <!-- Branch 2: sm_90 / sm_100 / sm_120 -->
  <rect x="20" y="288" width="480" height="148" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5" rx="8"/>
  <text x="36" y="312" font-size="13" font-weight="bold" fill="#991b1b">② Hopper / Blackwell（sm_90 H100、sm_100 B200、sm_120 RTX 5090）：</text>
  <rect x="36" y="324" width="448" height="58" fill="#fee2e2" stroke="#ef4444" rx="4"/>
  <text x="260" y="345" font-size="11.5" font-family="monospace" font-weight="bold" fill="#7f1d1d" text-anchor="middle">45 条 LOP3.LUT + 24 条 SHF.R.S32.HI（软件移位拆半字节做符号扩展）</text>
  <text x="260" y="368" font-size="12" font-family="monospace" font-weight="bold" fill="#b91c1c" text-anchor="middle">+ 2 条 IMMA.16816.S8.S8（偷偷退回 8-bit 指令分两次算！）</text>
  <text x="36" y="405" font-size="11.5" fill="#991b1b">• 真相：从 H100 开始硬件 INT4 被彻底移除！1 条指令在后台膨胀成 71 条 SASS 指令。</text>
  <text x="36" y="423" font-size="11.5" fill="#991b1b">• 而 50 系新增的原版 FP4 指令（block_scale）又绑死 sm_120a，旧卡根本不能跑。</text>

  <!-- Bottom: The only universal denominator -->
  <rect x="20" y="452" width="480" height="90" fill="#0f172a" rx="8"/>
  <text x="260" y="476" font-size="13" font-weight="bold" fill="#38bdf8" text-anchor="middle">③ 横跨四代显卡（sm_80 → sm_120）唯一全系原生的最大公约数：</text>
  <text x="260" y="502" font-size="12" font-family="monospace" font-weight="bold" fill="#4ade80" text-anchor="middle">mma.sync.aligned.m16n8k32.s32.s8.s8.s32  →  1 条 IMMA.16832.S8.S8</text>
  <text x="260" y="524" font-size="11.5" fill="#cbd5e1" text-anchor="middle">在全部 6 个目标架构上均为 1 条原生指令 + 0 条 LOP3/SHF —— 这就是 HINT4 选 INT8 的根因</text>
</svg>
"""

SVG3 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 580" width="520" height="580" font-family="'Noto Sans CJK SC', 'WenQuanYi Micro Hei', sans-serif">
  <rect width="520" height="580" fill="#f8fafc" rx="12"/>
  <text x="260" y="34" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">破案：为什么 HINT4 在 M=1 快近一倍，到 M=8 却反输给 FP16？</text>
  <text x="260" y="54" font-size="11" fill="#64748b" text-anchor="middle">RTX 3080 Laptop（sm_86）实测 N=11008, K=4096 矩阵乘法的耗时拆解</text>

  <!-- Case 1: M=1 Decode -->
  <rect x="20" y="72" width="480" height="196" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="72" width="480" height="30" fill="#dcfce7" rx="8"/>
  <text x="34" y="92" font-size="13" font-weight="bold" fill="#166534">场景 A：M=1（单词元自回归解码）—— 百分之百卡在显存带宽</text>

  <text x="36" y="122" font-size="11.5" fill="#334155">• FP16 cuBLAS 基线（需搬运 90.2 MB 权重）：</text>
  <rect x="36" y="130" width="380" height="24" fill="#94a3b8" rx="4"/>
  <text x="424" y="147" font-size="12" font-weight="bold" fill="#0f172a">238.6 μs (1.00×)</text>

  <text x="36" y="176" font-size="11.5" fill="#334155">• 4-bit 权重解码核（仅搬运 25.4 MB 权重，省 72% 显存流量）：</text>
  <rect x="36" y="184" width="210" height="24" fill="#22c55e" rx="4"/>
  <text x="254" y="201" font-size="12" font-weight="bold" fill="#15803d">132.1 μs (1.81× 提速！)</text>

  <text x="36" y="232" font-size="11.5" fill="#475569">💡 结论：M=1 时算力严重过剩、全在等显存；只要把权重压到 4-bit，走 DP4A 还是</text>
  <text x="36" y="250" font-size="11.5" fill="#475569">   解包成 FP16 都能快接近一倍。这笔收益来自“少读显存”，不是 INT8 算子更快。</text>

  <!-- Case 2: M=8 Small Batch -->
  <rect x="20" y="284" width="480" height="276" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="284" width="480" height="30" fill="#fee2e2" rx="8"/>
  <text x="34" y="304" font-size="13" font-weight="bold" fill="#991b1b">场景 B：M=8（小批量并行）—— W4A8 多背了三笔“过路费”</text>

  <text x="36" y="334" font-size="11.5" fill="#334155">① FP16 cuBLAS 基线（8 行复用同一份权重，耗时比 M=1 还短）：</text>
  <rect x="36" y="342" width="160" height="22" fill="#3b82f6" rx="4"/>
  <text x="204" y="358" font-size="12" font-weight="bold" fill="#1d4ed8">228.4 μs（N=4096 时仅 96.3 μs）</text>

  <text x="36" y="384" font-size="11.5" fill="#334155">② HINT4（W4A8）多出来的三笔刚性开销（实测拆解）：</text>
  <!-- Stacked bar for W4A8 overhead -->
  <rect x="36" y="394" width="130" height="26" fill="#f59e0b" rx="3"/>
  <text x="101" y="411" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">动态量化激活 119.8μs</text>
  <rect x="168" y="394" width="110" height="26" fill="#ef4444" rx="3"/>
  <text x="223" y="411" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">4b→8b解包+缩放</text>
  <rect x="280" y="394" width="95" height="26" fill="#6366f1" rx="3"/>
  <text x="327" y="411" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">M=16 半空转</text>

  <text x="36" y="444" font-size="11.5" fill="#0f172a">• <tspan font-weight="bold" fill="#b45309">第一笔（激活实时转 INT8）</tspan>：FP16 路线激活直接进核；INT8 路线每次调用前必须先对</text>
  <text x="46" y="462" font-size="11.5" fill="#475569">输入矩阵逐行求最大绝对值、做除法舍入成 INT8（在 N=4096 上单这一步就要 119.8 μs！）。</text>
  <text x="36" y="484" font-size="11.5" fill="#0f172a">• <tspan font-weight="bold" fill="#b91c1c">第二笔（M=8 填不满 m16n8k32）</tspan>：硬件指令强制要求 M=16 行，M=8 时整整一半算力</text>
  <text x="46" y="502" font-size="11.5" fill="#475569">在算补零（Padding），却还要在每段 K 循环后把 INT32 转回 FP32 乘缩放因子。</text>
  <text x="36" y="524" font-size="11.5" fill="#0f172a">• <tspan font-weight="bold" fill="#1e40af">第三笔（FP16 基准本身极强）</tspan>：30 系从 M=1 到 M=8，FP16 耗时几乎不涨（238→228 μs），</text>
  <text x="46" y="542" font-size="11.5" fill="#475569">因为瓶颈刚从显存切向计算，W4A8 省下的显存时间全被前处理和解包指令抵消了。</text>
</svg>
"""

for name, content in [
    ("dual-grid-tl-mechanism.svg", SVG1),
    ("sass-s4-vs-s8-census.svg", SVG2),
    ("m8-speed-paradox-breakdown.svg", SVG3),
]:
    svg_path = OUT_DIR / name
    svg_path.write_text(content.strip() + "\n")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_png:
        # Replace font with WenQuanYi Micro Hei for local rendering check
        temp_svg = content.replace("'Noto Sans CJK SC', ", "")
        cairosvg.svg2png(bytestring=temp_svg.encode("utf-8"), write_to=tmp_png.name, dpi=150)
        print(f"Verified {name} -> PNG size {Path(tmp_png.name).stat().st_size} bytes")
