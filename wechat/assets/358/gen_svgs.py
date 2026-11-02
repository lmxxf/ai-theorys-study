#!/usr/bin/env python3
"""
Generate 3 vertical SVG diagrams (width=520) for WeChat Issue 358:
1. calibration-xray-salient-channels.svg
2. awq-vs-gptq-mechanism.svg
3. qwen3-rtn-awq-gptq-benchmark.svg
"""
import tempfile
from pathlib import Path
import cairosvg

OUT_DIR = Path(__file__).resolve().parent

SVG1 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 630" width="520" height="630" font-family="'WenQuanYi Micro Hei', 'Noto Sans CJK SC', sans-serif">
  <rect width="520" height="630" fill="#f8fafc" rx="12"/>
  <text x="260" y="34" font-size="15.5" font-weight="bold" fill="#0f172a" text-anchor="middle">128 句话照出的 X 光片：权重很平，但激活值有 1% 大激活值通道</text>
  <text x="260" y="54" font-size="11" fill="#64748b" text-anchor="middle">Qwen3-0.6B（28 层）实测：为什么不看激活值直接四舍五入（RTN）会炸歪输出？</text>

  <!-- Box 1: Weight Channels are Flat -->
  <rect x="20" y="72" width="480" height="190" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="72" width="480" height="32" fill="#dcfce7" rx="8"/>
  <text x="34" y="93" font-size="13" font-weight="bold" fill="#166534">① 只看权重矩阵 W：被优化器揉得像镜面一样平（看不出谁重要）</text>

  <text x="36" y="122" font-size="11" fill="#334155">• 实测 Qwen3-0.6B 全网 28 层静态权重矩阵，各通道平均幅度非常平缓：</text>
  <rect x="36" y="134" width="448" height="68" fill="#f0fdf4" stroke="#86efac" rx="6"/>
  <text x="48" y="154" font-size="11" font-weight="bold" fill="#15803d">普通通道权重均值（中位数）：</text>
  <rect x="220" y="142" width="110" height="16" fill="#22c55e" rx="3"/>
  <text x="338" y="154" font-size="11" font-weight="bold" fill="#166534">1.00×（基准 0.0198）</text>

  <text x="48" y="176" font-size="11" font-weight="bold" fill="#15803d">全层最大权重通道 / 中位数：</text>
  <rect x="220" y="164" width="136" height="16" fill="#16a34a" rx="3"/>
  <text x="364" y="176" font-size="11" font-weight="bold" fill="#166534">仅 1.15× 至 1.61×</text>

  <text x="48" y="196" font-size="11" font-weight="bold" fill="#b45309">第 14 层 277 号通道的权重：</text>
  <rect x="220" y="184" width="38" height="14" fill="#f59e0b" rx="3"/>
  <text x="266" y="196" font-size="11" font-weight="bold" fill="#b45309">仅中位数的 0.33×（看似是个小透明）</text>

  <text x="36" y="224" font-size="10.8" fill="#475569">• 结论：如果不跑校准集、只盯着静态权重文件 W 看，所有通道高矮差不多，甚至真正</text>
  <text x="36" y="242" font-size="10.8" fill="#475569">  的命门通道（如 277 号）权重反而更小，在直接四舍五入（RTN）时最容易被抹平。</text>

  <!-- Box 2: Activation Channels have Extreme Salient Outliers -->
  <rect x="20" y="276" width="480" height="222" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="276" width="480" height="32" fill="#fee2e2" rx="8"/>
  <text x="34" y="297" font-size="13" font-weight="bold" fill="#991b1b">② 喂校准集照 X 光（看输入激活值 X）：1% 的大激活值通道大几十倍</text>

  <text x="36" y="326" font-size="11" fill="#334155">• 实测第 14 层 mlp_in（1024 个通道）输入激活值 |X| 的真实幅度对比：</text>

  <rect x="36" y="338" width="448" height="104" fill="#fef2f2" stroke="#fca5a5" rx="6"/>
  <text x="48" y="358" font-size="11" font-weight="bold" fill="#475569">普通通道中位数：</text>
  <rect x="162" y="346" width="12" height="16" fill="#64748b" rx="2"/>
  <text x="182" y="358" font-size="11" fill="#334155">0.544（1.00×）</text>

  <text x="48" y="382" font-size="11" font-weight="bold" fill="#b91c1c">第 62 号大激活通道：</text>
  <rect x="162" y="370" width="84" height="16" fill="#ef4444" rx="2"/>
  <text x="254" y="382" font-size="11" font-weight="bold" fill="#991b1b">6.05（中位数的 11.13×，二阶能量 105.3×）</text>

  <text x="48" y="406" font-size="11" font-weight="bold" fill="#7f1d1d">第 277 号大激活通道：</text>
  <rect x="162" y="394" width="210" height="16" fill="#dc2626" rx="2"/>
  <text x="380" y="406" font-size="11" font-weight="bold" fill="#7f1d1d">18.17（33.42×）</text>

  <text x="48" y="430" font-size="11" font-weight="bold" fill="#991b1b">→ 277 号通道的二阶能量 mean(X²)：高达普通通道中位数的 744.7 倍</text>

  <text x="36" y="464" font-size="10.5" fill="#334155">• 全网 28 层平均：attn_in 最大通道为中位数的 34.0×，mlp_in 为 35.9×，down_in 峰值达 334.9×。</text>
  <text x="36" y="483" font-size="10.5" fill="#334155">• 为什么 277 号通道权重反而小（0.33×）？正因为输入常年达 18.2，训练时权重被压小防输出过大。</text>

  <!-- Box 3: Bottom Verdict -->
  <rect x="20" y="512" width="480" height="98" fill="#0f172a" rx="8"/>
  <text x="260" y="537" font-size="13" font-weight="bold" fill="#f87171" text-anchor="middle">③ 免校准直接四舍五入（RTN）的灾难账单：</text>
  <text x="260" y="560" font-size="11.5" font-weight="bold" fill="#ffffff" text-anchor="middle">输出误差 = 激活值 X × 权重舍入误差 ΔW</text>
  <text x="260" y="582" font-size="11" font-weight="bold" fill="#fde047" text-anchor="middle">实测在 q/k/v 与 gate/up 层中，仅占 1%（10 个）的大激活值通道，</text>
  <text x="260" y="600" font-size="11" font-weight="bold" fill="#fde047" text-anchor="middle">乘上巨大激活值后，竟占了整层单通道误差平方和的一半以上（50.6% 至 51.0%）</text>
</svg>
"""

SVG2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 620" width="520" height="620" font-family="'WenQuanYi Micro Hei', 'Noto Sans CJK SC', sans-serif">
  <defs>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a"/>
    </marker>
    <pattern id="hatch-red" width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <line x1="0" y1="0" x2="0" y2="8" stroke="#f87171" stroke-width="2.5"/>
    </pattern>
  </defs>

  <rect width="520" height="620" fill="#f8fafc" rx="12"/>
  <text x="260" y="32" font-size="15.5" font-weight="bold" fill="#0f172a" text-anchor="middle">GPTQ“木工找平”几何图解：左边锯掉的面积，搬到右边柱子顶上</text>
  <text x="260" y="52" font-size="11" fill="#64748b" text-anchor="middle">底宽 = 激活值 x │ 高度 = 权重 w │ 矩形面积 = 通道得分 (x × w)</text>

  <!-- Panel 1: Original Floating-Point State -->
  <rect x="20" y="68" width="480" height="146" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="68" width="480" height="28" fill="#e2e8f0" rx="8"/>
  <text x="34" y="87" font-size="12.5" font-weight="bold" fill="#1e293b">① 量化前（原始浮点）：两块矩形总面积 = 4.0 + 2.0 = 6.0 分</text>

  <line x1="56" y1="184" x2="360" y2="184" stroke="#475569" stroke-width="1.5"/>
  <line x1="56" y1="142" x2="360" y2="142" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="50" y="188" font-size="10" fill="#64748b" text-anchor="end">w=0</text>
  <text x="50" y="146" font-size="10" fill="#64748b" text-anchor="end">w=1</text>

  <rect x="70" y="166" width="180" height="18" fill="#38bdf8" stroke="#0284c7" stroke-width="1.5"/>
  <text x="160" y="159" font-size="11" font-weight="bold" fill="#0369a1" text-anchor="middle">w1 = 0.4（面积 = 10×0.4 = 4.0）</text>
  <text x="160" y="202" font-size="11" font-weight="bold" fill="#0f172a" text-anchor="middle">第 1 列：底宽 x1 = 10（大激活）</text>

  <rect x="284" y="142" width="44" height="42" fill="#a78bfa" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="306" y="134" font-size="11" font-weight="bold" fill="#6d28d9" text-anchor="middle">w2 = 1.0</text>
  <text x="306" y="167" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">2.0</text>
  <text x="306" y="202" font-size="11" font-weight="bold" fill="#0f172a" text-anchor="middle">第 2 列：底宽 x2 = 2</text>

  <rect x="374" y="114" width="110" height="66" fill="#f8fafc" stroke="#94a3b8" rx="6"/>
  <text x="429" y="138" font-size="11" fill="#475569" text-anchor="middle">理想总面积</text>
  <text x="429" y="164" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">6.0 分</text>

  <!-- Panel 2: Naive RTN -->
  <rect x="20" y="226" width="480" height="146" fill="#ffffff" stroke="#fca5a5" stroke-width="1.5" rx="8"/>
  <rect x="20" y="226" width="480" height="28" fill="#fee2e2" rx="8"/>
  <text x="34" y="245" font-size="12.5" font-weight="bold" fill="#991b1b">② 普通四舍五入（RTN，各管各的）：第 1 列被抹成 0，白白丢掉 4.0 分</text>

  <line x1="56" y1="342" x2="360" y2="342" stroke="#475569" stroke-width="1.5"/>
  <line x1="56" y1="300" x2="360" y2="300" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="50" y="346" font-size="10" fill="#64748b" text-anchor="end">w=0</text>
  <text x="50" y="304" font-size="10" fill="#64748b" text-anchor="end">w=1</text>

  <rect x="70" y="324" width="180" height="18" fill="url(#hatch-red)" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 2"/>
  <text x="160" y="317" font-size="11" font-weight="bold" fill="#dc2626" text-anchor="middle">w1: 0.4 → 四舍五入变 0（锯掉 4.0 分）</text>
  <text x="160" y="360" font-size="11" fill="#64748b" text-anchor="middle">第 1 列得分 = 10 × 0 = 0</text>

  <rect x="284" y="300" width="44" height="42" fill="#a78bfa" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="306" y="292" font-size="11" font-weight="bold" fill="#6d28d9" text-anchor="middle">w2 → 1</text>
  <text x="306" y="325" font-size="10.5" font-weight="bold" fill="#ffffff" text-anchor="middle">2.0</text>
  <text x="306" y="360" font-size="11" fill="#64748b" text-anchor="middle">第 2 列得分 = 2.0</text>

  <rect x="374" y="272" width="110" height="68" fill="#fef2f2" stroke="#f87171" rx="6"/>
  <text x="429" y="294" font-size="11" font-weight="bold" fill="#991b1b" text-anchor="middle">RTN 总面积</text>
  <text x="429" y="318" font-size="15" font-weight="bold" fill="#dc2626" text-anchor="middle">仅剩 2.0 分</text>
  <text x="429" y="333" font-size="10" fill="#b91c1c" text-anchor="middle">（亏空 4.0 分）</text>

  <!-- Panel 3: GPTQ Compensation -->
  <rect x="20" y="384" width="480" height="218" fill="#ffffff" stroke="#86efac" stroke-width="1.5" rx="8"/>
  <rect x="20" y="384" width="480" height="28" fill="#dcfce7" rx="8"/>
  <text x="34" y="403" font-size="12.5" font-weight="bold" fill="#166534">③ GPTQ 逐列找平：把左边锯掉的 4.0 面积，搬到右边窄柱子顶上垫高 +2.0</text>

  <!-- Grid lines for Panel 3: w=0 at 568, w=1 at 538, w=2 at 508, w=3 at 478 (30px per unit) -->
  <line x1="56" y1="568" x2="360" y2="568" stroke="#475569" stroke-width="1.5"/>
  <line x1="56" y1="538" x2="360" y2="538" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="56" y1="508" x2="360" y2="508" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="56" y1="478" x2="360" y2="478" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="50" y="572" font-size="10" fill="#64748b" text-anchor="end">w=0</text>
  <text x="50" y="542" font-size="10" fill="#64748b" text-anchor="end">w=1</text>
  <text x="50" y="512" font-size="10" fill="#64748b" text-anchor="end">w=2</text>
  <text x="50" y="482" font-size="10" fill="#15803d" font-weight="bold" text-anchor="end">w=3</text>

  <!-- Step 1: Chopped Rect 1 -->
  <rect x="70" y="553" width="180" height="15" fill="url(#hatch-red)" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 2"/>
  <text x="160" y="546" font-size="10.5" font-weight="bold" fill="#dc2626" text-anchor="middle">第一步：w1 变 0，亏空面积 4.0</text>
  <text x="160" y="586" font-size="11" fill="#64748b" text-anchor="middle">底宽 x1 = 10</text>

  <!-- Step 2: Callout Box & Curved Arrow moving area 4.0 from col 1 to top of col 2 -->
  <rect x="64" y="422" width="232" height="40" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.2" rx="6"/>
  <text x="180" y="438" font-size="10.5" font-weight="bold" fill="#166534" text-anchor="middle">第二步：面积 4.0 搬到宽 x2=2 的柱子上</text>
  <text x="180" y="454" font-size="10.5" font-weight="bold" fill="#15803d" text-anchor="middle">柱子窄 5 倍 → 高度必须垫高 4.0 ÷ 2 = +2.0</text>
  <path d="M 160 532 C 160 474, 245 474, 279 498" fill="none" stroke="#16a34a" stroke-width="2.5" marker-end="url(#arrow-green)"/>

  <!-- Step 3: Stacked Rect 2 (original w2=1.0 plus compensated +2.0 = 3.0) -->
  <rect x="284" y="538" width="44" height="30" fill="#a78bfa" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="306" y="557" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">原 2.0</text>

  <rect x="284" y="478" width="44" height="60" fill="#22c55e" stroke="#15803d" stroke-width="1.5"/>
  <text x="306" y="504" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">补 +4.0</text>
  <text x="306" y="520" font-size="9.5" fill="#dcfce7" text-anchor="middle">(Δw=+2)</text>

  <text x="306" y="468" font-size="11" font-weight="bold" fill="#15803d" text-anchor="middle">w2_新 = 3</text>
  <text x="306" y="586" font-size="11" font-weight="bold" fill="#0f172a" text-anchor="middle">底宽 x2 = 2</text>

  <!-- GPTQ Total badge -->
  <rect x="372" y="478" width="114" height="90" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5" rx="6"/>
  <text x="429" y="500" font-size="11" font-weight="bold" fill="#166534" text-anchor="middle">第三步：量化 w2=3</text>
  <text x="429" y="524" font-size="14.5" font-weight="bold" fill="#15803d" text-anchor="middle">总面积 = 6.0 分</text>
  <text x="429" y="542" font-size="10" font-weight="bold" fill="#166534" text-anchor="middle">（2 × 3 = 6.0）</text>
  <text x="429" y="558" font-size="10" fill="#15803d" text-anchor="middle">零误差完美找平</text>
</svg>
"""

SVG3 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 600" width="520" height="600" font-family="'WenQuanYi Micro Hei', 'Noto Sans CJK SC', sans-serif">
  <rect width="520" height="600" fill="#f8fafc" rx="12"/>
  <text x="260" y="32" font-size="15.5" font-weight="bold" fill="#0f172a" text-anchor="middle">Qwen3-0.6B 真机实测：RTN vs AWQ vs GPTQ（4-bit 与 3-bit）</text>
  <text x="260" y="52" font-size="11" fill="#64748b" text-anchor="middle">柱条越短代表困惑度 PPL 越低（原版 BF16 基准 = 56.55）│ 右侧对比权重与输出信噪比</text>

  <!-- Box 1: 4-bit (g=128) -->
  <rect x="20" y="68" width="480" height="236" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="68" width="480" height="30" fill="#e0e7ff" rx="8"/>
  <text x="34" y="88" font-size="12" font-weight="bold" fill="#3730a3">① 4-bit（g=128）实测：GPTQ 权重 SNR 最低，但输出最准、PPL 紧贴原版</text>

  <!-- Column headers -->
  <text x="34" y="116" font-size="10.5" font-weight="bold" fill="#64748b">量化方案</text>
  <text x="142" y="116" font-size="10.5" font-weight="bold" fill="#64748b">端到端困惑度 PPL（↓ 越短越好）</text>
  <text x="356" y="116" font-size="10.5" font-weight="bold" fill="#64748b">权重 SNR → 输出 SNR</text>
  <line x1="34" y1="123" x2="486" y2="123" stroke="#e2e8f0" stroke-width="1"/>

  <!-- BF16 reference dashed line at x = 142 + 95 = 237 -->
  <line x1="237" y1="126" x2="237" y2="256" stroke="#64748b" stroke-width="1.2" stroke-dasharray="3 3"/>
  <text x="237" y="270" font-size="9.5" fill="#475569" text-anchor="middle">▲ BF16 基准 (56.55)</text>

  <!-- Row 1: 4-bit RTN -->
  <text x="34" y="146" font-size="11" font-weight="bold" fill="#991b1b">① RTN 免校准</text>
  <rect x="142" y="132" width="144" height="18" fill="#ef4444" rx="3"/>
  <text x="292" y="146" font-size="11" font-weight="bold" fill="#991b1b">85.10</text>
  <text x="356" y="146" font-size="10.5" fill="#475569">18.28 → 19.08 dB</text>

  <!-- Row 2: 4-bit AWQ top1% x2 -->
  <text x="34" y="178" font-size="11" font-weight="bold" fill="#b45309">② AWQ 前1%×2</text>
  <rect x="142" y="164" width="136" height="18" fill="#f59e0b" rx="3"/>
  <text x="284" y="178" font-size="11" font-weight="bold" fill="#b45309">80.53</text>
  <text x="356" y="178" font-size="10.5" fill="#475569">18.11 → 20.09 dB</text>

  <!-- Row 3: 4-bit AWQ full -->
  <text x="34" y="210" font-size="11" font-weight="bold" fill="#0369a1">③ AWQ 搜索α</text>
  <rect x="142" y="196" width="120" height="18" fill="#0ea5e9" rx="3"/>
  <text x="268" y="210" font-size="11" font-weight="bold" fill="#0369a1">70.76</text>
  <text x="356" y="210" font-size="10.5" fill="#475569">17.82 → 20.95 dB</text>

  <!-- Row 4: 4-bit GPTQ -->
  <text x="34" y="242" font-size="11" font-weight="bold" fill="#15803d">④ GPTQ 找平</text>
  <rect x="142" y="228" width="112" height="18" fill="#22c55e" rx="3"/>
  <text x="260" y="242" font-size="11" font-weight="bold" fill="#15803d">66.30</text>
  <text x="356" y="242" font-size="10.5" font-weight="bold" fill="#15803d">15.03 → 35.47 dB</text>

  <text x="260" y="292" font-size="10" fill="#334155" text-anchor="middle">• 倒挂现象：GPTQ 权重本身 SNR 最低（15.03 dB），但乘上激活 X 后输出 SNR 飙至 35.47 dB</text>

  <!-- Box 2: 3-bit (g=128) — exact same layout! -->
  <rect x="20" y="316" width="480" height="266" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="316" width="480" height="30" fill="#fef3c7" rx="8"/>
  <text x="34" y="336" font-size="12" font-weight="bold" fill="#b45309">② 3-bit（g=128）实测：免校准 RTN 彻底崩盘，GPTQ 将困惑度压低 90%</text>

  <!-- Column headers -->
  <text x="34" y="364" font-size="10.5" font-weight="bold" fill="#64748b">量化方案</text>
  <text x="142" y="364" font-size="10.5" font-weight="bold" fill="#64748b">端到端困惑度 PPL（↓ 越短越好）</text>
  <text x="356" y="364" font-size="10.5" font-weight="bold" fill="#64748b">权重 SNR → 输出 SNR</text>
  <line x1="34" y1="371" x2="486" y2="371" stroke="#e2e8f0" stroke-width="1"/>

  <!-- Row 1: 3-bit RTN -->
  <text x="34" y="396" font-size="11" font-weight="bold" fill="#991b1b">① RTN 免校准</text>
  <rect x="142" y="382" width="136" height="18" fill="#dc2626" rx="3"/>
  <text x="284" y="396" font-size="10.5" font-weight="bold" fill="#991b1b">1876.3 (崩)</text>
  <text x="356" y="396" font-size="10.5" fill="#475569">10.96 → 12.00 dB</text>

  <!-- Row 2: 3-bit AWQ top1% x2 -->
  <text x="34" y="428" font-size="11" font-weight="bold" fill="#b45309">② AWQ 前1%×2</text>
  <rect x="142" y="414" width="80" height="18" fill="#f59e0b" rx="3"/>
  <text x="228" y="428" font-size="11" font-weight="bold" fill="#b45309">616.35</text>
  <text x="356" y="428" font-size="10.5" fill="#475569">10.78 → 12.83 dB</text>

  <!-- Row 3: 3-bit AWQ full -->
  <text x="34" y="460" font-size="11" font-weight="bold" fill="#0369a1">③ AWQ 搜索α</text>
  <rect x="142" y="446" width="58" height="18" fill="#0ea5e9" rx="3"/>
  <text x="206" y="460" font-size="11" font-weight="bold" fill="#0369a1">426.69</text>
  <text x="356" y="460" font-size="10.5" fill="#475569">10.54 → 13.64 dB</text>

  <!-- Row 4: 3-bit GPTQ -->
  <text x="34" y="492" font-size="11" font-weight="bold" fill="#15803d">④ GPTQ 找平</text>
  <rect x="142" y="478" width="36" height="18" fill="#22c55e" rx="3"/>
  <text x="184" y="492" font-size="11" font-weight="bold" fill="#15803d">235.78</text>
  <text x="356" y="492" font-size="10.5" font-weight="bold" fill="#15803d"> 7.75 → 28.15 dB</text>

  <!-- Summary footer bar -->
  <rect x="34" y="514" width="452" height="54" fill="#0f172a" rx="6"/>
  <text x="260" y="535" font-size="10.2" font-weight="bold" fill="#fde047" text-anchor="middle">总结：AWQ 通道缩放原型速度快（3.6s vs 40.6s），但缩放倍率受同组邻居尺子制约（70.76）</text>
  <text x="260" y="554" font-size="10.2" fill="#cbd5e1" text-anchor="middle">GPTQ 靠海森矩阵让后排权重替前排背锅，4-bit 达 66.30（追平 g=16），3-bit 较 RTN 降 87%</text>
</svg>
"""

for name, content in [
    ("calibration-xray-salient-channels.svg", SVG1),
    ("awq-vs-gptq-mechanism.svg", SVG2),
    ("qwen3-rtn-awq-gptq-benchmark.svg", SVG3),
]:
    svg_path = OUT_DIR / name
    svg_path.write_text(content.strip() + "\n")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_png:
        cairosvg.svg2png(bytestring=content.encode("utf-8"), write_to=tmp_png.name, dpi=150)
        print(f"Verified {name} -> PNG size {Path(tmp_png.name).stat().st_size} bytes")
