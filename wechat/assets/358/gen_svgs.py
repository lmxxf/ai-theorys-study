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

  <text x="36" y="122" font-size="11" fill="#334155">• 预训练全程开着 AdamW 权重衰减，全网 28 层各通道权重均值高度均匀：</text>
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
  <text x="254" y="382" font-size="11" font-weight="bold" fill="#991b1b">6.10（中位数的 11.20×，平方重要性 125.6×）</text>

  <text x="48" y="406" font-size="11" font-weight="bold" fill="#7f1d1d">第 277 号大激活通道：</text>
  <rect x="162" y="394" width="210" height="16" fill="#dc2626" rx="2"/>
  <text x="380" y="406" font-size="11" font-weight="bold" fill="#7f1d1d">18.30（33.63×）</text>

  <text x="48" y="430" font-size="11" font-weight="bold" fill="#991b1b">→ 277 号通道的海森对角线重要性（X²）：高达普通通道中位数的 1,131.2 倍</text>

  <text x="36" y="464" font-size="10.5" fill="#334155">• 全网 28 层平均：attn_in 最大通道为中位数的 34.0×，mlp_in 为 36.2×，down_in 峰值达 337.5×。</text>
  <text x="36" y="483" font-size="10.5" fill="#334155">• 为什么 277 号通道权重反而小（0.33×）？正因为输入常年达 18.3，训练时权重被迫压小防爆炸。</text>

  <!-- Box 3: Bottom Verdict -->
  <rect x="20" y="512" width="480" height="98" fill="#0f172a" rx="8"/>
  <text x="260" y="537" font-size="13" font-weight="bold" fill="#f87171" text-anchor="middle">③ 免校准直接四舍五入（RTN）的灾难账单：</text>
  <text x="260" y="560" font-size="11.5" font-weight="bold" fill="#ffffff" text-anchor="middle">输出误差 = 激活值 X × 权重舍入误差 ΔW</text>
  <text x="260" y="582" font-size="11" font-weight="bold" fill="#fde047" text-anchor="middle">实测在 q/k/v 与 gate/up 层中，仅占 1%（10 个）的大激活值通道，</text>
  <text x="260" y="600" font-size="11" font-weight="bold" fill="#fde047" text-anchor="middle">乘上巨大激活值后，竟贡献了整层超过一半（50.7% 至 51.4%）的输出误差能量</text>
</svg>
"""

SVG2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 670" width="520" height="670" font-family="'WenQuanYi Micro Hei', 'Noto Sans CJK SC', sans-serif">
  <rect width="520" height="670" fill="#f8fafc" rx="12"/>
  <text x="260" y="34" font-size="15.5" font-weight="bold" fill="#0f172a" text-anchor="middle">同喂 128 句话，两派绝活对比：AWQ 穿防弹衣 vs GPTQ 木工找平</text>
  <text x="260" y="54" font-size="11" fill="#64748b" text-anchor="middle">都不跑反向传播训练，一个靠“乘 2 放大保精度”，一个靠“改右边权重替左边背锅”</text>

  <!-- Box 1: AWQ -->
  <rect x="20" y="72" width="480" height="276" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="72" width="480" height="32" fill="#e0f2fe" rx="8"/>
  <text x="34" y="93" font-size="13" font-weight="bold" fill="#0369a1">第一派：AWQ（MIT MLSys 2024）—— 给 1% 大激活值通道穿防弹衣</text>

  <text x="36" y="120" font-size="11" font-weight="bold" fill="#0f172a">• 题目：重要通道激活 x = 100，权重 w = 0.35（理想乘积 35.0），格点步长 = 0.25</text>

  <!-- Naive RTN -->
  <rect x="36" y="132" width="448" height="48" fill="#fef2f2" stroke="#f87171" rx="6"/>
  <text x="48" y="151" font-size="11" font-weight="bold" fill="#991b1b">【反例】普通直接四舍五入（RTN）：把 0.35 舍入到最近格点 0.25</text>
  <text x="48" y="170" font-size="10.8" fill="#7f1d1d">算出来输出 = 100 × 0.25 = 25.0  →  当场偏了 10.0（误差 0.10 被放大了 100 倍）</text>

  <!-- AWQ Scaling -->
  <rect x="36" y="190" width="448" height="74" fill="#f0fdf4" stroke="#22c55e" rx="6"/>
  <text x="48" y="210" font-size="11" font-weight="bold" fill="#166534">【正例】AWQ 绝活：量化前先把这列权重乘以 s = 2，同时把对面激活除以 2</text>
  <text x="48" y="230" font-size="10.8" fill="#14532d">① 权重放大后量化：0.35 × 2 = 0.70  →  四舍五入到最近格点 0.75（落点更准）</text>
  <text x="48" y="249" font-size="10.8" font-weight="bold" fill="#15803d">② 激活缩小后相乘：(100 ÷ 2) × 0.75 = 50 × 0.75 = 37.5  →  误差从 10.0 降到 2.5</text>

  <rect x="36" y="274" width="448" height="60" fill="#f1f5f9" stroke="#94a3b8" rx="6"/>
  <text x="48" y="294" font-size="11" font-weight="bold" fill="#0f172a">【问】为什么不干脆把这 1% 权重单独留成 FP16？</text>
  <text x="48" y="313" font-size="10.5" fill="#334155">因为 99% 的 4-bit 里夹杂 1% 的 16-bit 会打乱内存对齐、卡死 GPU 张量核心；乘 s 后全员</text>
  <text x="48" y="328" font-size="10.5" fill="#334155">仍是整整齐齐的 4-bit，缩放因子 s 还能顺手融进上一层归一化层里，推理零额外开销。</text>

  <!-- Box 2: GPTQ -->
  <rect x="20" y="364" width="480" height="286" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="364" width="480" height="32" fill="#fae8ff" rx="8"/>
  <text x="34" y="385" font-size="13" font-weight="bold" fill="#86198f">第二派：GPTQ（ICLR 2023）—— 左边锯短一毫米，右边垫厚一毫米</text>

  <text x="36" y="412" font-size="11" font-weight="bold" fill="#0f172a">• 题目：输入 x1 = 10, x2 = 2；权重 w1 = 0.4, w2 = 1.0（理想总和 10×0.4 + 2×1.0 = 6.0）</text>
  <text x="36" y="430" font-size="10.8" fill="#475569">  假设量化尺子只许取整数（0, 1, 2, 3...），对比 RTN 与 GPTQ 的算法：</text>

  <!-- Naive RTN -->
  <rect x="36" y="442" width="448" height="46" fill="#fef2f2" stroke="#f87171" rx="6"/>
  <text x="48" y="461" font-size="11" font-weight="bold" fill="#991b1b">【反例】普通直接四舍五入（RTN）：各管各的 → w1 变成 0，w2 变成 1</text>
  <text x="48" y="479" font-size="10.8" fill="#7f1d1d">算出来总输出 = 10 × 0 + 2 × 1 = 2.0  →  比理想值 6.0 整整丢了 4.0 分</text>

  <!-- GPTQ Compensation -->
  <rect x="36" y="498" width="448" height="84" fill="#f0fdf4" stroke="#22c55e" rx="6"/>
  <text x="48" y="518" font-size="11" font-weight="bold" fill="#166534">【正例】GPTQ 绝活：用相关性矩阵 X^T × X（海森矩阵）把第 1 列的误差补给第 2 列</text>
  <text x="48" y="538" font-size="10.8" fill="#14532d">① 先量化第 1 列：w1 从 0.4 舍入变成 0（产生了 +0.4 的缺口，总和少了 10×0.4 = 4.0）</text>
  <text x="48" y="556" font-size="10.5" fill="#14532d">② 立刻改第 2 列：因 x1 是 x2 的 5 倍（10÷2=5），把还没量化的 w2 垫高 5×0.4 = +2.0，</text>
  <text x="48" y="573" font-size="10.8" font-weight="bold" fill="#15803d">   于是 w2 从 1.0 改成 3.0 再量化，最终输出 = 10×0 + 2×3 = 6.0（零误差完美找平）</text>

  <rect x="36" y="592" width="448" height="44" fill="#0f172a" rx="6"/>
  <text x="260" y="611" font-size="11" font-weight="bold" fill="#38bdf8" text-anchor="middle">核心洞察：GPTQ 故意把次要权重 w2 改歪（1.0→3.0，单看权重误差变大），</text>
  <text x="260" y="628" font-size="11" fill="#cbd5e1" text-anchor="middle">却换来了乘上激活值 X 之后整层输出的严丝合缝</text>
</svg>
"""

SVG3 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 620" width="520" height="620" font-family="'WenQuanYi Micro Hei', 'Noto Sans CJK SC', sans-serif">
  <rect width="520" height="620" fill="#f8fafc" rx="12"/>
  <text x="260" y="34" font-size="15.5" font-weight="bold" fill="#0f172a" text-anchor="middle">Qwen3-0.6B 真机实测：RTN vs AWQ vs GPTQ（4-bit 与 3-bit）</text>
  <text x="260" y="54" font-size="11" fill="#64748b" text-anchor="middle">RTX 3080 Laptop 实测数据：权重信噪比越低，层输出信噪比反而越高</text>

  <!-- Box 1: 4-bit SNR Paradox -->
  <rect x="20" y="72" width="480" height="252" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="72" width="480" height="32" fill="#e0e7ff" rx="8"/>
  <text x="34" y="93" font-size="13" font-weight="bold" fill="#3730a3">① 4-bit（g=128）反直觉现象：权重本身 SNR vs 乘上 X 后的输出 SNR</text>

  <text x="36" y="120" font-size="11" fill="#475569">（注：原始 BF16 基准困惑度 PPL = 56.55；SNR 越高代表越接近原版 BF16）</text>

  <!-- Table/Bars inside Box 1 -->
  <rect x="36" y="130" width="448" height="142" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>
  <text x="46" y="150" font-size="10.8" font-weight="bold" fill="#0f172a">方案名称</text>
  <text x="152" y="150" font-size="10.8" font-weight="bold" fill="#0f172a">权重本身 SNR</text>
  <text x="255" y="150" font-size="10.8" font-weight="bold" fill="#0f172a">校准/留出输出 SNR</text>
  <text x="382" y="150" font-size="10.8" font-weight="bold" fill="#0f172a">端到端 PPL (↓)</text>
  <line x1="36" y1="158" x2="484" y2="158" stroke="#cbd5e1" stroke-width="1"/>

  <text x="46" y="180" font-size="10.8" font-weight="bold" fill="#dc2626">RTN（免校准）</text>
  <text x="152" y="180" font-size="10.8" font-weight="bold" fill="#16a34a">18.28 dB (最高)</text>
  <text x="258" y="180" font-size="10.8" fill="#dc2626">19.08 / 19.17 dB</text>
  <text x="382" y="180" font-size="10.8" font-weight="bold" fill="#dc2626">85.10 (最差)</text>

  <text x="46" y="206" font-size="10.8" font-weight="bold" fill="#d97706">AWQ (前1%×2)</text>
  <text x="152" y="206" font-size="10.8" fill="#334155">18.10 dB</text>
  <text x="258" y="206" font-size="10.8" fill="#d97706">20.08 / 20.22 dB</text>
  <text x="382" y="206" font-size="10.8" font-weight="bold" fill="#d97706">80.13</text>

  <text x="46" y="232" font-size="10.8" font-weight="bold" fill="#0284c7">AWQ (网格搜索)</text>
  <text x="152" y="232" font-size="10.8" fill="#334155">17.85 dB</text>
  <text x="258" y="232" font-size="10.8" fill="#0284c7">20.97 / 21.12 dB</text>
  <text x="382" y="232" font-size="10.8" font-weight="bold" fill="#0284c7">75.82 (耗时 3.5s)</text>

  <text x="46" y="258" font-size="10.8" font-weight="bold" fill="#16a34a">GPTQ (误差找平)</text>
  <text x="152" y="258" font-size="10.8" font-weight="bold" fill="#dc2626">15.03 dB (最低)</text>
  <text x="258" y="258" font-size="10.8" font-weight="bold" fill="#16a34a">35.46 / 23.85 dB</text>
  <text x="382" y="258" font-size="10.8" font-weight="bold" fill="#16a34a">60.50 (逼近 BF16)</text>

  <text x="36" y="293" font-size="10.8" fill="#334155">• 规律：GPTQ 权重本身 SNR 比 RTN 低了 3.25 dB（因为故意改歪了次要权重替前排背锅），</text>
  <text x="36" y="311" font-size="10.8" fill="#334155">  但乘上 X 后输出 SNR 跃升至 35.46 dB（留出集 23.85 dB），KL 散度从 67.94 压至 22.45。</text>

  <!-- Box 2: 3-bit Extreme Compression -->
  <rect x="20" y="338" width="480" height="262" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="8"/>
  <rect x="20" y="338" width="480" height="32" fill="#fef3c7" rx="8"/>
  <text x="34" y="359" font-size="13" font-weight="bold" fill="#b45309">② 3-bit（g=128）极限压缩：不跑校准集的 RTN 当场崩盘</text>

  <text x="36" y="388" font-size="11" fill="#334155">• 当刻度只剩 {-3..+3} 共 7 个格点时，四套方案的全模型困惑度 PPL 对比（越低越好）：</text>

  <!-- Horizontal bars for 3-bit PPL -->
  <text x="36" y="416" font-size="11" font-weight="bold" fill="#991b1b">RTN（免校准直转）：</text>
  <rect x="160" y="402" width="188" height="20" fill="#dc2626" rx="3"/>
  <text x="356" y="417" font-size="11" font-weight="bold" fill="#991b1b">PPL = 1876.29（崩盘）</text>

  <text x="36" y="448" font-size="11" font-weight="bold" fill="#d97706">AWQ（仅前 1%×2）：</text>
  <rect x="160" y="434" width="95" height="20" fill="#f59e0b" rx="3"/>
  <text x="263" y="449" font-size="11" font-weight="bold" fill="#b45309">PPL = 558.95（救回 70%）</text>

  <text x="36" y="480" font-size="11" font-weight="bold" fill="#0284c7">AWQ（完整搜索）：</text>
  <rect x="160" y="466" width="66" height="20" fill="#0ea5e9" rx="3"/>
  <text x="234" y="481" font-size="11" font-weight="bold" fill="#0369a1">PPL = 346.62（耗时 3.6s）</text>

  <text x="36" y="512" font-size="11" font-weight="bold" fill="#166534">GPTQ（逐列找平）：</text>
  <rect x="160" y="498" width="38" height="20" fill="#22c55e" rx="3"/>
  <text x="206" y="513" font-size="11" font-weight="bold" fill="#15803d">PPL = 188.93（比 RTN 低 10 倍）</text>

  <rect x="36" y="534" width="448" height="52" fill="#0f172a" rx="6"/>
  <text x="260" y="554" font-size="11" font-weight="bold" fill="#fde047" text-anchor="middle">工程选型一句话：4-bit 下 AWQ 快 12 倍（3.5s vs 44.5s）且绝不过拟合校准集；</text>
  <text x="260" y="572" font-size="11" fill="#cbd5e1" text-anchor="middle">压到 3-bit 及以下极限位宽时，带海森矩阵逐列找补的 GPTQ 是保命底线。</text>
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
