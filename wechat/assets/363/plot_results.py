#!/usr/bin/env python3
"""Generate SVG comparison figure for Manifold Surface Curvature vs MIT Flat Chord."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / "summary_all_models.json").read_text())

W, H = 1260, 820
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'style="background:#ffffff;font-family:\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif;">'
]

# Title
svg.append(
    '<text x="630" y="34" text-anchor="middle" font-size="20" font-weight="bold" fill="#111827">'
    '不能量平直空间的弦，要顺着流形表面走：MIT 弦夹角 vs 流形表皮转角·曲率·快慢·方向多寡 (LAMBADA 1024篇)'
    '</text>'
)
svg.append(
    '<text x="630" y="56" text-anchor="middle" font-size="13" fill="#4b5563">'
    'GPT-2 XL (79,398 tokens) | Pythia-2.8B (77,130 tokens) | Qwen3-0.6B (71,691 tokens) — 严格因果对齐 (状态 ≤ t 预测 H(t))'
    '</text>'
)

models = [
    ('gpt2-xl', 'GPT-2 XL (48L, d=1600)'),
    ('pythia-2.8b', 'Pythia-2.8B (32L, d=2560)'),
    ('Qwen3-0.6B', 'Qwen3-0.6B (28L, d=1024)'),
]

# Top row: Layer-wise |r| or signed r comparison for key single metrics
for idx, (m_key, m_title) in enumerate(models):
    x0 = 60 + idx * 400
    y0 = 90
    pw, ph = 340, 260
    c_data = d['models'][m_key]['long']['single_metrics']
    L = len(c_data['C_flat_3']['layers'])

    svg.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db" stroke-width="1"/>')
    svg.append(f'<text x="{x0 + pw/2}" y="{y0 - 8}" text-anchor="middle" font-size="14" font-weight="bold" fill="#1f2937">{m_title}: 逐层与预测熵 H(t) 的 Pearson r</text>')

    # y-axis from -0.45 to +0.50
    ymin, ymax = -0.45, 0.50
    def py(val):
        return y0 + ph - ((val - ymin) / (ymax - ymin)) * ph
    def px(l):
        return x0 + (l / max(1, L - 1)) * pw

    # Zero line
    y_zero = py(0.0)
    svg.append(f'<line x1="{x0}" y1="{y_zero:.1f}" x2="{x0+pw}" y2="{y_zero:.1f}" stroke="#9ca3af" stroke-dasharray="4 3" stroke-width="1"/>')
    for tick in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        yt = py(tick)
        svg.append(f'<line x1="{x0-4}" y1="{yt:.1f}" x2="{x0}" y2="{yt:.1f}" stroke="#6b7280"/>')
        svg.append(f'<text x="{x0-7}" y="{yt+4:.1f}" text-anchor="end" font-size="10" fill="#4b5563">{tick:+.1f}</text>')

    series = [
        ('C_flat_3', '#9ca3af', '2.0', '4 2', 'MIT 平直弦 C_flat_3'),
        ('theta_surf', '#f59e0b', '2.0', '', '表皮转角 θ_surf'),
        ('kappa_geo', '#dc2626', '2.5', '', '表皮测地曲率 κ_geo'),
        ('s_geo', '#2563eb', '2.5', '', '表皮步长(快慢) s_geo'),
        ('v_layer', '#7c3aed', '2.0', '', '跨层表皮角速度 v_layer'),
        ('norm', '#059669', '2.0', '2 2', '残差范数 ||x_t||'),
    ]
    for s_key, color, sw, dash, _ in series:
        pts = [f"{px(r['layer']):.1f},{py(r['pearson_r']):.1f}" for r in c_data[s_key]['layers']]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="{sw}"{dash_attr} points="{" ".join(pts)}"/>')

    svg.append(f'<text x="{x0 + pw/2}" y="{y0 + ph + 16}" text-anchor="middle" font-size="11" fill="#4b5563">层号 Layer l (0 .. {L-1})</text>')

# Legend for top row
leg_x = 110
leg_y = 382
legend_items = [
    ('#9ca3af', 'MIT 平直弦 C_flat_3 (被120°球皮底色锁死)'),
    ('#f59e0b', '流形表皮转角 θ_surf'),
    ('#dc2626', '流形表皮测地曲率 κ_geo (转角/弧长)'),
    ('#2563eb', '流形表皮步长 s_geo (水平快慢)'),
    ('#7c3aed', '跨层表皮角速度 v_layer (垂直快慢)'),
    ('#059669', '残差范数 ||x_t||'),
]
for i, (col, label) in enumerate(legend_items):
    lx = leg_x + (i % 3) * 380
    ly = leg_y + (i // 3) * 22
    svg.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+24}" y2="{ly-4}" stroke="{col}" stroke-width="3"/>')
    svg.append(f'<text x="{lx+30}" y="{ly}" font-size="12" fill="#1f2937">{label}</text>')

# Bottom row: Variance Explained R^2 (%) Progression from MIT 1% to Full Manifold Surface (10-fold Passage CV R^2)
y_bot = 455
bh = 300
for idx, (m_key, m_title) in enumerate(models):
    x0 = 60 + idx * 400
    pw = 340
    m_long = d['models'][m_key]['long']
    bars = [
        ('MIT 单层弦 C_flat_3', m_long['single_metrics']['C_flat_3']['best_abs_r2'] * 100, '#9ca3af'),
        ('表皮测地曲率 κ_geo', m_long['single_metrics']['kappa_geo']['best_abs_r2'] * 100, '#f59e0b'),
        ('单层表皮快慢+范数', m_long['single_layer_combos']['3_Surface_Speed_And_Norm']['best_cv_r2'] * 100, '#3b82f6'),
        ('单层全表皮特征联合', m_long['single_layer_combos']['7_All_SingleLayer_Surface']['best_cv_r2'] * 100, '#6366f1'),
        ('跨层方向有效秩 erank', m_long['multilayer_joint']['MultiLayer_Directions_Erank']['cv_r2'] * 100, '#10b981'),
        ('跨层全表皮几何联合', m_long['multilayer_joint']['MultiLayer_Full_Manifold_Surface']['cv_r2'] * 100, '#dc2626'),
    ]

    svg.append(f'<rect x="{x0}" y="{y_bot}" width="{pw}" height="{bh}" fill="#f9fafb" stroke="#d1d5db" stroke-width="1"/>')
    svg.append(f'<text x="{x0 + pw/2}" y="{y_bot - 8}" text-anchor="middle" font-size="14" font-weight="bold" fill="#1f2937">{m_title}: 预测熵解释方差 R² 跃升 (10折CV)</text>')

    max_r2 = 75.0
    for b_idx, (b_label, r2_val, b_col) in enumerate(bars):
        by = y_bot + 18 + b_idx * 46
        bw = (r2_val / max_r2) * (pw - 150)
        svg.append(f'<text x="{x0 + 118}" y="{by + 18}" text-anchor="end" font-size="11" fill="#1f2937">{b_label}</text>')
        svg.append(f'<rect x="{x0 + 124}" y="{by + 4}" width="{max(2.0, bw):.1f}" height="22" fill="{b_col}" rx="3"/>')
        svg.append(f'<text x="{x0 + 128 + max(2.0, bw):.1f}" y="{by + 19}" font-size="11" font-weight="bold" fill="#111827">{r2_val:.2f}%</text>')

svg.append('</svg>')
out_svg = ROOT / "manifold_vs_flat_summary.svg"
out_svg.write_text("\n".join(svg))
print("Saved", out_svg)
