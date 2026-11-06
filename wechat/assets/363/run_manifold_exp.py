#!/usr/bin/env python3
"""
Manifold Surface Curvature, Geodesic Speed & Directional Effective Rank Experiment
Strictly causal alignment: all geometric features at index t use only states <= t
to predict next-token entropy H(t) = H(logits[t]).
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
import time
import math
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = Path('/home/lmxxf/work/curvature-icml2026-361/corpus.json')
EPS = 1e-8

ANGLE_METRICS = {
    'c_flat', 'C_flat_3',
    'theta_surf', 'Theta_surf_3',
    'theta_surf_c', 'Theta_surf_c_3',
    'extrinsic_bias', 'Extrinsic_bias_3',
    's_geo', 'S_geo_3',
    's_geo_c', 'S_geo_c_3',
    'accel_geo',
    'v_layer', 'theta_layer',
}

METRIC_ORDER = [
    # Baseline 0: MIT Flat Euclidean Chord
    'c_flat', 'C_flat_3',
    # Direction 1: Manifold Surface Turning Angle & Geodesic Curvature
    'theta_surf', 'Theta_surf_3',
    'theta_surf_c', 'Theta_surf_c_3',
    'extrinsic_bias', 'Extrinsic_bias_3',
    'kappa_geo', 'Kappa_geo_3',
    'kappa_geo_c', 'Kappa_geo_c_3',
    # Direction 2: Manifold Surface Speed, Step Length, Acceleration & Norms
    's_geo', 'S_geo_3',
    's_geo_c', 'S_geo_c_3',
    's_chord', 'S_chord_3',
    's_rel', 'norm', 'dlog_norm',
    'accel_geo', 'ratio_geo',
    # Direction 3: Directional Multiplicity (Tangent Space Effective Rank) & Cross-Layer Motion
    'erank_dir', 'pr_dir', 'top1_dir',
    'erank_weighted', 'pr_weighted',
    'erank_dir_c',
    'v_layer', 'theta_layer', 'erank_layer',
]


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


@torch.inference_mode()
def entropy_bits(logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(dim=-1) / math.log(2.0)


@torch.inference_mode()
def spectral_stats_from_gram(G: torch.Tensor):
    """
    G: [..., W, W] symmetric positive semi-definite Gram matrix.
    Returns (erank, pr, top1_ratio) each of shape [...]
    """
    eigvals = torch.linalg.eigvalsh(G.float()).clamp_min(0.0)  # ascending [..., W]
    total = eigvals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = eigvals / total
    ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)
    erank = ent.exp()
    pr = (total.squeeze(-1) ** 2) / (eigvals ** 2).sum(dim=-1).clamp_min(1e-20)
    top1 = eigvals[..., -1] / total.squeeze(-1)
    return erank, pr, top1


@torch.inference_mode()
def compute_temporal_metrics_batched_layers(h_all_blocks: torch.Tensor, mu_blocks: torch.Tensor = None):
    """
    h_all_blocks: [L, B, T, D] float32 residual states across L layers.
    mu_blocks: [L, D] optional layer mean vectors for anisotropy-centered sphere projection.
    Returns dict of [L, B, T] tensors (valid for t >= 5).
    All metrics at index t strictly depend ONLY on h[:, :, :t+1, :].
    """
    L, B, T, D = h_all_blocks.shape
    LB = L * B
    h = h_all_blocks.reshape(LB, T, D)
    device = h.device

    nan1 = torch.full((LB, 1), float('nan'), device=device)
    nan2 = torch.full((LB, 2), float('nan'), device=device)
    nan3 = torch.full((LB, 3), float('nan'), device=device)
    nan4 = torch.full((LB, 4), float('nan'), device=device)
    nan5 = torch.full((LB, 5), float('nan'), device=device)

    # 0. Norms and Euclidean chords
    norms = h.norm(dim=-1).clamp_min(EPS)  # [LB, T]
    log_norms = norms.log()
    dlog_norm = torch.cat([nan1, log_norms[:, 1:] - log_norms[:, :-1]], dim=1)

    v = h[:, 1:, :] - h[:, :-1, :]  # [LB, T-1, D]
    s_chord_raw = v.norm(dim=-1)  # [LB, T-1]
    s_chord = torch.cat([nan1, s_chord_raw], dim=1)
    s_rel = s_chord / norms

    u_chord = F.normalize(v, dim=-1, eps=1e-12)
    c_flat_raw = torch.acos((u_chord[:, 1:, :] * u_chord[:, :-1, :]).sum(dim=-1).clamp(-1.0, 1.0))  # [LB, T-2]
    c_flat = torch.cat([nan2, c_flat_raw], dim=1)
    C_flat_3 = torch.cat([nan4, (c_flat_raw[:, :-2] + c_flat_raw[:, 1:-1] + c_flat_raw[:, 2:]) / 3.0], dim=1)

    # 1. Unit sphere projection \hat{x}_t = x_t / ||x_t||
    xhat = F.normalize(h, dim=-1, eps=1e-12)  # [LB, T, D]
    cos_step = (xhat[:, 1:, :] * xhat[:, :-1, :]).sum(dim=-1).clamp(-1.0, 1.0)
    s_geo_raw = torch.acos(cos_step)  # [LB, T-1]
    s_geo = torch.cat([nan1, s_geo_raw], dim=1)
    S_geo_3 = torch.cat([nan3, (s_geo_raw[:, :-2] + s_geo_raw[:, 1:-1] + s_geo_raw[:, 2:]) / 3.0], dim=1)
    S_chord_3 = torch.cat([nan3, (s_chord_raw[:, :-2] + s_chord_raw[:, 1:-1] + s_chord_raw[:, 2:]) / 3.0], dim=1)

    # Acceleration / speed ratio on sphere
    accel_geo = torch.cat([nan2, s_geo_raw[:, 1:] - s_geo_raw[:, :-1]], dim=1)
    ratio_geo = torch.cat([nan2, ((s_geo_raw[:, 1:] + 1e-6) / (s_geo_raw[:, :-1] + 1e-6)).log()], dim=1)

    # Parallel transport turning angle at pivot \hat{x}_{t-1} for path \hat{x}_{t-2} -> \hat{x}_{t-1} -> \hat{x}_t
    x_prev2 = xhat[:, :-2, :]
    x_prev1 = xhat[:, 1:-1, :]
    x_curr = xhat[:, 2:, :]

    dot_in = (x_prev1 * x_prev2).sum(dim=-1, keepdim=True)
    u_in = F.normalize(dot_in * x_prev1 - x_prev2, dim=-1, eps=1e-12)

    dot_out = (x_curr * x_prev1).sum(dim=-1, keepdim=True)
    u_out = F.normalize(x_curr - dot_out * x_prev1, dim=-1, eps=1e-12)

    theta_surf_raw = torch.acos((u_in * u_out).sum(dim=-1).clamp(-1.0, 1.0))  # [LB, T-2]
    theta_surf = torch.cat([nan2, theta_surf_raw], dim=1)
    Theta_surf_3 = torch.cat([
        nan4,
        (theta_surf_raw[:, :-2] + theta_surf_raw[:, 1:-1] + theta_surf_raw[:, 2:]) / 3.0
    ], dim=1)

    # Extrinsic normal bending bias = c_flat - theta_surf
    extrinsic_bias = c_flat - theta_surf
    Extrinsic_bias_3 = C_flat_3 - Theta_surf_3

    # True geodesic curvature kappa_geo = theta_surf / mean_arc_length
    mean_arc = 0.5 * (s_geo_raw[:, 1:] + s_geo_raw[:, :-1]).clamp_min(1e-6)
    kappa_geo_raw = theta_surf_raw / mean_arc
    kappa_geo = torch.cat([nan2, kappa_geo_raw], dim=1)
    Kappa_geo_3 = torch.cat([
        nan4,
        (kappa_geo_raw[:, :-2] + kappa_geo_raw[:, 1:-1] + kappa_geo_raw[:, 2:]) / 3.0
    ], dim=1)

    # Centered sphere projection (removing anisotropy mean cone)
    if mu_blocks is not None:
        hc = (h_all_blocks - mu_blocks.view(L, 1, 1, D)).reshape(LB, T, D)
    else:
        hc = h - h.mean(dim=1, keepdim=True)
    xhat_c = F.normalize(hc, dim=-1, eps=1e-12)
    s_geo_c_raw = torch.acos((xhat_c[:, 1:, :] * xhat_c[:, :-1, :]).sum(dim=-1).clamp(-1.0, 1.0))
    s_geo_c = torch.cat([nan1, s_geo_c_raw], dim=1)
    S_geo_c_3 = torch.cat([nan3, (s_geo_c_raw[:, :-2] + s_geo_c_raw[:, 1:-1] + s_geo_c_raw[:, 2:]) / 3.0], dim=1)

    xc_prev2, xc_prev1, xc_curr = xhat_c[:, :-2, :], xhat_c[:, 1:-1, :], xhat_c[:, 2:, :]
    u_in_c = F.normalize((xc_prev1 * xc_prev2).sum(dim=-1, keepdim=True) * xc_prev1 - xc_prev2, dim=-1, eps=1e-12)
    u_out_c = F.normalize(xc_curr - (xc_curr * xc_prev1).sum(dim=-1, keepdim=True) * xc_prev1, dim=-1, eps=1e-12)
    theta_surf_c_raw = torch.acos((u_in_c * u_out_c).sum(dim=-1).clamp(-1.0, 1.0))
    theta_surf_c = torch.cat([nan2, theta_surf_c_raw], dim=1)
    Theta_surf_c_3 = torch.cat([
        nan4,
        (theta_surf_c_raw[:, :-2] + theta_surf_c_raw[:, 1:-1] + theta_surf_c_raw[:, 2:]) / 3.0
    ], dim=1)

    mean_arc_c = 0.5 * (s_geo_c_raw[:, 1:] + s_geo_c_raw[:, :-1]).clamp_min(1e-6)
    kappa_geo_c_raw = theta_surf_c_raw / mean_arc_c
    kappa_geo_c = torch.cat([nan2, kappa_geo_c_raw], dim=1)
    Kappa_geo_c_3 = torch.cat([
        nan4,
        (kappa_geo_c_raw[:, :-2] + kappa_geo_c_raw[:, 1:-1] + kappa_geo_c_raw[:, 2:]) / 3.0
    ], dim=1)

    # 3. Directional Multiplicity / Local Tangent Space Effective Rank (window W=5 steps ending at t: x_{t-5}..x_t)
    dx = xhat[:, 1:, :] - xhat[:, :-1, :]  # [LB, T-1, D]
    dx_c = xhat_c[:, 1:, :] - xhat_c[:, :-1, :]
    if T > 5:
        dx_win = dx.unfold(dimension=1, size=5, step=1).permute(0, 1, 3, 2).contiguous()  # [LB, T-5, 5, D]
        n_curr = xhat[:, 5:, :].unsqueeze(2)  # [LB, T-5, 1, D]
        w_tan = dx_win - (dx_win * n_curr).sum(dim=-1, keepdim=True) * n_curr
        w_tan_unit = F.normalize(w_tan, dim=-1, eps=1e-12)

        G_weighted = torch.matmul(w_tan, w_tan.transpose(-1, -2))  # [LB, T-5, 5, 5]
        G_dir = torch.matmul(w_tan_unit, w_tan_unit.transpose(-1, -2))

        erank_w, pr_w, _ = spectral_stats_from_gram(G_weighted)
        erank_d, pr_d, top1_d = spectral_stats_from_gram(G_dir)

        dx_c_win = dx_c.unfold(dimension=1, size=5, step=1).permute(0, 1, 3, 2).contiguous()
        n_curr_c = xhat_c[:, 5:, :].unsqueeze(2)
        w_tan_c = dx_c_win - (dx_c_win * n_curr_c).sum(dim=-1, keepdim=True) * n_curr_c
        w_tan_c_unit = F.normalize(w_tan_c, dim=-1, eps=1e-12)
        G_dir_c = torch.matmul(w_tan_c_unit, w_tan_c_unit.transpose(-1, -2))
        erank_dc, _, _ = spectral_stats_from_gram(G_dir_c)

        erank_weighted = torch.cat([nan5, erank_w], dim=1)
        pr_weighted = torch.cat([nan5, pr_w], dim=1)
        erank_dir = torch.cat([nan5, erank_d], dim=1)
        pr_dir = torch.cat([nan5, pr_d], dim=1)
        top1_dir = torch.cat([nan5, top1_d], dim=1)
        erank_dir_c = torch.cat([nan5, erank_dc], dim=1)
    else:
        erank_weighted = pr_weighted = erank_dir = pr_dir = top1_dir = erank_dir_c = nan1.expand(LB, T).clone()

    out = dict(
        c_flat=c_flat,
        C_flat_3=C_flat_3,
        theta_surf=theta_surf,
        Theta_surf_3=Theta_surf_3,
        theta_surf_c=theta_surf_c,
        Theta_surf_c_3=Theta_surf_c_3,
        extrinsic_bias=extrinsic_bias,
        Extrinsic_bias_3=Extrinsic_bias_3,
        kappa_geo=kappa_geo,
        Kappa_geo_3=Kappa_geo_3,
        kappa_geo_c=kappa_geo_c,
        Kappa_geo_c_3=Kappa_geo_c_3,
        s_geo=s_geo,
        S_geo_3=S_geo_3,
        s_geo_c=s_geo_c,
        S_geo_c_3=S_geo_c_3,
        s_chord=s_chord,
        S_chord_3=S_chord_3,
        s_rel=s_rel,
        norm=norms,
        dlog_norm=dlog_norm,
        accel_geo=accel_geo,
        ratio_geo=ratio_geo,
        erank_weighted=erank_weighted,
        pr_weighted=pr_weighted,
        erank_dir=erank_dir,
        pr_dir=pr_dir,
        top1_dir=top1_dir,
        erank_dir_c=erank_dir_c,
    )
    return {k: v.reshape(L, B, T) for k, v in out.items()}


@torch.inference_mode()
def compute_vertical_layer_metrics(all_h: torch.Tensor):
    """
    all_h: [L+1, B, T, D] where index 0 is embedding output, 1..L are block outputs 0..L-1.
    Computes vertical (cross-layer) manifold surface speed & turning angle at each token t.
    Returns dict of [L, B, T] tensors for layers l = 0..L-1.
    """
    Lp1, B, T, D = all_h.shape
    L = Lp1 - 1
    device = all_h.device
    xhat = F.normalize(all_h, dim=-1, eps=1e-12)  # [L+1, B, T, D]

    # 1. Cross-layer geodesic step length v_layer(l, t) = arccos(xhat[l+1] · xhat[l]) for l=0..L-1
    cos_v = (xhat[1:] * xhat[:-1]).sum(dim=-1).clamp(-1.0, 1.0)  # [L, B, T]
    v_layer = torch.acos(cos_v)  # [L, B, T]

    # 2. Cross-layer surface turning angle theta_layer(l, t) for path xhat[l-1] -> xhat[l] -> xhat[l+1]
    x_prev2 = xhat[:-2]   # [L-1, B, T, D]
    x_prev1 = xhat[1:-1]  # [L-1, B, T, D]
    x_curr = xhat[2:]     # [L-1, B, T, D]

    u_in = F.normalize((x_prev1 * x_prev2).sum(dim=-1, keepdim=True) * x_prev1 - x_prev2, dim=-1, eps=1e-12)
    u_out = F.normalize(x_curr - (x_curr * x_prev1).sum(dim=-1, keepdim=True) * x_prev1, dim=-1, eps=1e-12)
    theta_l = torch.acos((u_in * u_out).sum(dim=-1).clamp(-1.0, 1.0))  # [L-1, B, T]
    # For l=0, duplicate l=1 so matrix is full [L, B, T] without losing layer 0 in regressions
    theta_layer = torch.cat([theta_l[:1], theta_l], dim=0)  # [L, B, T]

    # 3. Cross-layer directional effective rank over last up to 5 layer steps ending at layer l
    dx_l = xhat[1:] - xhat[:-1]  # [L, B, T, D]
    erank_layer = torch.ones((L, B, T), dtype=torch.float32, device=device)
    for l in range(1, L):
        w_start = max(0, l - 4)
        win = dx_l[w_start:l+1].permute(1, 2, 0, 3).contiguous()  # [B, T, W_l, D]
        n_c = xhat[l+1].unsqueeze(2)  # [B, T, 1, D]
        w_tan = win - (win * n_c).sum(dim=-1, keepdim=True) * n_c
        w_unit = F.normalize(w_tan, dim=-1, eps=1e-12)
        G_l = torch.matmul(w_unit, w_unit.transpose(-1, -2))  # [B, T, W_l, W_l]
        er, _, _ = spectral_stats_from_gram(G_l)
        erank_layer[l] = er

    return dict(
        v_layer=v_layer,
        theta_layer=theta_layer,
        erank_layer=erank_layer,
    )


def get_model_parts(model, model_name: str):
    if model_name == 'gpt2-xl':
        embed = lambda inp: model.transformer.wte(inp) + model.transformer.wpe(
            torch.arange(inp.shape[1], device=inp.device).unsqueeze(0)
        )
        blocks = model.transformer.h
    elif model_name == 'pythia-2.8b':
        embed = model.gpt_neox.embed_in
        blocks = model.gpt_neox.layers
    else:
        embed = model.model.embed_tokens
        blocks = model.model.layers
    return embed, blocks


def self_test_geometry():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h_ortho = torch.zeros((1, 1, 6, 8), dtype=torch.float32, device=device)
    for i in range(6):
        h_ortho[0, 0, i, i] = 1.0
    m1 = compute_temporal_metrics_batched_layers(h_ortho)
    c_flat_deg = float(m1['c_flat'][0, 0, 2].cpu()) * 180.0 / math.pi
    theta_surf_deg = float(m1['theta_surf'][0, 0, 2].cpu()) * 180.0 / math.pi
    assert abs(c_flat_deg - 120.0) < 1e-4, f"Expected 120 deg, got {c_flat_deg}"
    assert abs(theta_surf_deg - 90.0) < 1e-4, f"Expected 90 deg, got {theta_surf_deg}"

    h_eq = torch.zeros((1, 1, 6, 8), dtype=torch.float32, device=device)
    for i in range(6):
        ang = i * (math.pi / 6.0)
        h_eq[0, 0, i, 0] = math.cos(ang)
        h_eq[0, 0, i, 1] = math.sin(ang)
    m2 = compute_temporal_metrics_batched_layers(h_eq)
    c_flat_eq = float(m2['c_flat'][0, 0, 2].cpu()) * 180.0 / math.pi
    theta_surf_eq = float(m2['theta_surf'][0, 0, 2].cpu()) * 180.0 / math.pi
    erank_eq = float(m2['erank_dir'][0, 0, 5].cpu())
    assert abs(c_flat_eq - 30.0) < 1e-3, f"Expected 30 deg chord angle on equator, got {c_flat_eq}"
    assert abs(theta_surf_eq - 0.0) < 1e-2, f"Expected 0 deg geodesic angle on equator, got {theta_surf_eq}"
    assert abs(erank_eq - 1.0) < 1e-3, f"Expected erank=1.0 on great circle, got {erank_eq}"

    log(f"SELF-TEST PASSED: Ortho -> c_flat={c_flat_deg:.2f}°, theta_surf={theta_surf_deg:.2f}°; "
        f"GreatCircle(step=30°) -> c_flat={c_flat_eq:.2f}°, theta_surf={theta_surf_eq:.4f}°, erank_dir={erank_eq:.4f}")
    return dict(
        ortho_c_flat_deg=c_flat_deg,
        ortho_theta_surf_deg=theta_surf_deg,
        equator_step30_c_flat_deg=c_flat_eq,
        equator_step30_theta_surf_deg=theta_surf_eq,
        equator_erank_dir=erank_eq,
    )


@torch.inference_mode()
def estimate_layer_means(model, tok, model_name: str, strings, n_Calibration=64):
    """Estimate mean vector mu_l per layer on first n_Calibration passages."""
    _, blocks = get_model_parts(model, model_name)
    L = len(blocks)
    D = model.config.hidden_size
    sums = torch.zeros((L, D), dtype=torch.float64, device='cuda')
    counts = 0
    sub = strings[:n_Calibration]
    for start in range(0, len(sub), 4):
        ss = sub[start:start+4]
        seqs = [tok.encode(s, add_special_tokens=False) for s in ss]
        valid = [(i, v) for i, v in enumerate(seqs) if 8 <= len(v) <= 1024]
        if not valid:
            continue
        inp = tok([ss[i] for i, _ in valid], padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
        captured = {}
        def make_hook(l):
            def fn(mod, a, o):
                h = o[0] if isinstance(o, tuple) else o
                captured[l] = h.detach()
            return fn
        handles = [b.register_forward_hook(make_hook(l)) for l, b in enumerate(blocks)]
        try:
            model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for b, (_, ids) in enumerate(valid):
            T_val = len(ids)
            for l in range(L):
                sums[l] += captured[l][b, :T_val, :].double().sum(dim=0)
            counts += T_val
        del captured, inp
    mu = (sums / max(1, counts)).float()
    return mu


def ols_cv_r_and_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray, ridge: float = 1e-4):
    """
    10-fold passage-grouped cross-validation for single or multi-feature OLS/Ridge.
    X: [N, F], y: [N], groups: [N]
    Returns dict with full_r, full_r2, cv_r, cv_r2, standardized_betas
    """
    if X.ndim == 1:
        X = X[:, None]
    N, F_dim = X.shape
    # Replace any tiny NaN/Inf with column mean
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    idx_bad = ~np.isfinite(X)
    if idx_bad.any():
        X = X.copy()
        for j in range(F_dim):
            X[idx_bad[:, j], j] = col_means[j]

    # Standardize X and y for clean beta inspection
    x_mu = X.mean(axis=0, keepdims=True)
    x_sd = X.std(axis=0, keepdims=True)
    x_sd = np.where(x_sd < 1e-12, 1.0, x_sd)
    Xz = (X - x_mu) / x_sd

    y_mu = y.mean()
    y_sd = max(float(y.std()), 1e-12)
    yz = (y - y_mu) / y_sd

    # Full-sample fit
    gram = Xz.T @ Xz + ridge * N * np.eye(F_dim)
    rhs = Xz.T @ yz
    beta_full = np.linalg.solve(gram, rhs)
    pred_full = Xz @ beta_full
    full_r = float(np.corrcoef(pred_full, yz)[0, 1]) if pred_full.std() > 1e-12 else 0.0
    if F_dim == 1:
        # Preserve sign for single feature!
        full_r = float(np.corrcoef(X[:, 0], y)[0, 1])
    full_r2 = float(full_r ** 2)

    # 10-fold grouped CV
    rng = np.random.default_rng(361)
    uniq_g = np.unique(groups)
    rng.shuffle(uniq_g)
    chunks = np.array_split(uniq_g, 10)
    fold_rs = []
    preds_all = np.zeros_like(yz)
    for ch in chunks:
        test = np.isin(groups, ch)
        train = ~test
        X_tr, y_tr = X[train], y[train]
        X_te, y_te = X[test], y[test]
        xm = X_tr.mean(axis=0, keepdims=True)
        xs = X_tr.std(axis=0, keepdims=True)
        xs = np.where(xs < 1e-12, 1.0, xs)
        ym = y_tr.mean()
        X_trz = (X_tr - xm) / xs
        X_tez = (X_te - xm) / xs
        y_trz = y_tr - ym
        g_tr = X_trz.T @ X_trz + ridge * len(y_tr) * np.eye(F_dim)
        b_tr = np.linalg.solve(g_tr, X_trz.T @ y_trz)
        pred_te = ym + X_tez @ b_tr
        preds_all[test] = pred_te
        r_f = float(np.corrcoef(pred_te, y_te)[0, 1]) if pred_te.std() > 1e-12 else 0.0
        fold_rs.append(r_f)

    cv_r = float(np.mean(fold_rs))
    cv_r_pooled = float(np.corrcoef(preds_all, y)[0, 1])
    ss_res = float(np.sum((y - preds_all) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    cv_r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))

    return dict(
        full_r=full_r,
        full_r2=full_r2,
        cv_r=cv_r,
        cv_r_pooled=cv_r_pooled,
        cv_r2=cv_r2,
        betas=[float(b) for b in beta_full],
    )


@torch.inference_mode()
def evaluate_corpus(model, tok, model_name: str, which: str, strings, mu_blocks: torch.Tensor, max_passages: int = None):
    if max_passages is not None:
        strings = strings[:max_passages]
    _, blocks = get_model_parts(model, model_name)
    L = len(blocks)

    storage = {m: [] for m in METRIC_ORDER}
    Y, Ym, Yp, groups, positions = [], [], [], [], []

    t0 = time.time()
    for start in range(0, len(strings), 4):
        ss = strings[start:start+4]
        seqs = [tok.encode(s, add_special_tokens=False) for s in ss]
        valid = [(i, v) for i, v in enumerate(seqs) if 8 <= len(v) <= 1024]
        if not valid:
            continue
        inp = tok([ss[i] for i, _ in valid], padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')

        captured_blocks = [None] * L
        captured_embed = [None]

        def hook_embed(mod, a, o):
            captured_embed[0] = (o[0] if isinstance(o, tuple) else o).detach().float()

        def make_hook(l):
            def fn(mod, a, o):
                captured_blocks[l] = (o[0] if isinstance(o, tuple) else o).detach().float()
            return fn

        if model_name == 'gpt2-xl':
            h_emb = model.transformer.drop.register_forward_hook(hook_embed)
        elif model_name == 'pythia-2.8b':
            h_emb = model.gpt_neox.embed_in.register_forward_hook(hook_embed)
        else:
            h_emb = model.model.embed_tokens.register_forward_hook(hook_embed)

        handles = [h_emb] + [b.register_forward_hook(make_hook(l)) for l, b in enumerate(blocks)]
        try:
            out = model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        H = entropy_bits(out.logits).cpu().numpy()  # [B, T]
        del out

        h_blocks_tensor = torch.stack(captured_blocks, dim=0)  # [L, B, T, D]
        all_h_tensor = torch.cat([captured_embed[0].unsqueeze(0), h_blocks_tensor], dim=0)  # [L+1, B, T, D]

        t_metrics = compute_temporal_metrics_batched_layers(h_blocks_tensor, mu_blocks)
        v_metrics = compute_vertical_layer_metrics(all_h_tensor)
        t_metrics.update(v_metrics)
        t_metrics_cpu = {k: v.cpu().numpy() for k, v in t_metrics.items()}

        del captured_blocks, captured_embed, h_blocks_tensor, all_h_tensor, t_metrics, v_metrics

        for b, (i, ids) in enumerate(valid):
            ix = np.arange(5, len(ids) - 1)
            for m_name in METRIC_ORDER:
                storage[m_name].append(t_metrics_cpu[m_name][:, b, ix])  # [L, len(ix)]
            Y.append(H[b, ix])
            Ym.append(H[b, ix - 1])
            Yp.append(H[b, ix + 1])
            groups.extend([start + i] * len(ix))
            positions.extend(ix.tolist())

        if start % 256 == 0:
            log(f"[{model_name}:{which}] {start}/{len(strings)} ({time.time()-t0:.1f}s)")

    X_dict = {m: np.concatenate(storage[m], axis=1) for m in METRIC_ORDER}  # each [L, N_tokens]
    y = np.concatenate(Y)
    g = np.array(groups)
    pos = np.array(positions)
    N_tokens = len(y)
    log(f"[{model_name}:{which}] Forward complete: {N_tokens} tokens across {len(np.unique(g))} passages in {time.time()-t0:.1f}s")

    # 1. Per-layer single-metric statistics & correlations with H(t)
    metric_summaries = {}
    for m_name in METRIC_ORDER:
        X_m = X_dict[m_name]  # [L, N]
        layer_rows = []
        for l in range(L):
            v = X_m[l]
            valid_mask = np.isfinite(v)
            v_clean = np.where(valid_mask, v, np.nanmean(v))
            r_p = float(np.corrcoef(v_clean, y)[0, 1])
            rho_s = float(spearmanr(v_clean, y).statistic)
            mean_val = float(np.nanmean(v))
            sd_val = float(np.nanstd(v))
            row = dict(
                layer=l,
                mean=mean_val,
                sd=sd_val,
                mean_deg=(mean_val * 180.0 / math.pi if m_name in ANGLE_METRICS else None),
                sd_deg=(sd_val * 180.0 / math.pi if m_name in ANGLE_METRICS else None),
                pearson_r=r_p,
                spearman_rho=rho_s,
                r2=float(r_p ** 2),
            )
            layer_rows.append(row)

        rs = [r['pearson_r'] for r in layer_rows]
        abs_rs = [abs(x) for x in rs]
        best_abs_l = int(np.argmax(abs_rs))
        best_pos_l = int(np.argmax(rs))
        best_neg_l = int(np.argmin(rs))
        # Run 10-fold passage CV at best_abs_l and best_pos_l
        cv_best_abs = ols_cv_r_and_r2(X_m[best_abs_l], y, g)
        cv_best_pos = ols_cv_r_and_r2(X_m[best_pos_l], y, g)

        metric_summaries[m_name] = dict(
            best_abs_layer=best_abs_l,
            best_abs_r=rs[best_abs_l],
            best_abs_rho=layer_rows[best_abs_l]['spearman_rho'],
            best_abs_r2=layer_rows[best_abs_l]['r2'],
            best_abs_cv_r=cv_best_abs['cv_r'],
            best_abs_cv_r2=cv_best_abs['cv_r2'],
            best_pos_layer=best_pos_l,
            best_pos_r=rs[best_pos_l],
            best_pos_cv_r=cv_best_pos['cv_r'],
            best_neg_layer=best_neg_l,
            best_neg_r=rs[best_neg_l],
            mean_across_layers=float(np.mean([r['mean'] for r in layer_rows])),
            mean_deg_across_layers=(float(np.mean([r['mean_deg'] for r in layer_rows])) if m_name in ANGLE_METRICS else None),
            min_mean_layer=int(np.argmin([r['mean'] for r in layer_rows])),
            max_mean_layer=int(np.argmax([r['mean'] for r in layer_rows])),
            layers=layer_rows,
        )

    # 2. Multi-variable regressions per layer
    COMBOS = {
        '0_MIT_Flat_Chord': ['c_flat', 'C_flat_3'],
        '1_Surface_Turn_Angles': ['theta_surf', 'Theta_surf_3', 'theta_surf_c', 'Theta_surf_c_3'],
        '2_Surface_Angle_Plus_Curvature': [
            'theta_surf', 'Theta_surf_3', 'theta_surf_c', 'Theta_surf_c_3',
            'kappa_geo', 'Kappa_geo_3', 'kappa_geo_c', 'Kappa_geo_c_3'
        ],
        '3_Surface_Speed_And_Norm': [
            's_geo', 'S_geo_3', 's_geo_c', 'S_geo_c_3',
            's_rel', 'norm', 'dlog_norm', 'accel_geo', 'ratio_geo'
        ],
        '4_Horizontal_Directions_Erank': [
            'erank_dir', 'pr_dir', 'top1_dir', 'erank_weighted', 'pr_weighted', 'erank_dir_c'
        ],
        '5_Vertical_CrossLayer_Surface': [
            'v_layer', 'theta_layer', 'erank_layer'
        ],
        '6_All_Horizontal_Surface': [
            'theta_surf', 'Theta_surf_3', 'theta_surf_c', 'Theta_surf_c_3',
            'kappa_geo', 'Kappa_geo_3', 'kappa_geo_c', 'Kappa_geo_c_3',
            's_geo', 'S_geo_3', 's_geo_c', 'S_geo_c_3',
            's_rel', 'norm', 'dlog_norm', 'accel_geo', 'ratio_geo',
            'erank_dir', 'pr_dir', 'top1_dir', 'erank_weighted', 'pr_weighted', 'erank_dir_c'
        ],
        '7_All_SingleLayer_Surface': [
            'theta_surf', 'Theta_surf_3', 'theta_surf_c', 'Theta_surf_c_3',
            'kappa_geo', 'Kappa_geo_3', 'kappa_geo_c', 'Kappa_geo_c_3',
            's_geo', 'S_geo_3', 's_geo_c', 'S_geo_c_3',
            's_rel', 'norm', 'dlog_norm', 'accel_geo', 'ratio_geo',
            'erank_dir', 'pr_dir', 'top1_dir', 'erank_weighted', 'pr_weighted', 'erank_dir_c',
            'v_layer', 'theta_layer', 'erank_layer'
        ],
    }

    combo_summaries = {}
    for c_name, feat_names in COMBOS.items():
        c_layers = []
        for l in range(L):
            X_l = np.stack([X_dict[f][l] for f in feat_names], axis=1)  # [N, F]
            res = ols_cv_r_and_r2(X_l, y, g)
            res['layer'] = l
            c_layers.append(res)
        best_l = int(np.argmax([r['cv_r'] for r in c_layers]))
        combo_summaries[c_name] = dict(
            features=feat_names,
            best_layer=best_l,
            best_full_r=c_layers[best_l]['full_r'],
            best_full_r2=c_layers[best_l]['full_r2'],
            best_cv_r=c_layers[best_l]['cv_r'],
            best_cv_r_pooled=c_layers[best_l]['cv_r_pooled'],
            best_cv_r2=c_layers[best_l]['cv_r2'],
            best_layer_betas=dict(zip(feat_names, c_layers[best_l]['betas'])),
            layers=c_layers,
        )

    # 3. Multi-Layer Joint Regression (combining all L layers at token t!)
    multilayer_combos = {
        'MultiLayer_MIT_C_flat_3': ['C_flat_3'],
        'MultiLayer_Surface_Angle_Only': ['theta_surf', 'Theta_surf_3', 'theta_surf_c', 'Theta_surf_c_3'],
        'MultiLayer_Surface_Curvature': ['theta_surf', 'Theta_surf_3', 'kappa_geo', 'Kappa_geo_3', 'kappa_geo_c'],
        'MultiLayer_Surface_Speed': ['s_geo', 'S_geo_3', 's_geo_c', 'S_geo_c_3', 'accel_geo', 'norm', 'dlog_norm'],
        'MultiLayer_Directions_Erank': ['erank_dir', 'pr_dir', 'erank_weighted', 'erank_dir_c', 'erank_layer'],
        'MultiLayer_Vertical_Motion': ['v_layer', 'theta_layer', 'erank_layer'],
        'MultiLayer_Full_Manifold_Surface': [
            'theta_surf', 'Theta_surf_3', 'theta_surf_c', 'kappa_geo', 'Kappa_geo_3',
            's_geo', 'S_geo_3', 's_geo_c', 'accel_geo', 's_rel', 'norm', 'dlog_norm',
            'erank_dir', 'pr_dir', 'erank_weighted', 'v_layer', 'theta_layer', 'erank_layer'
        ],
    }
    multilayer_results = {}
    for ml_name, feat_names in multilayer_combos.items():
        X_ml = np.concatenate([X_dict[f].T for f in feat_names], axis=1)  # [N, L * len(feat_names)]
        res_ml = ols_cv_r_and_r2(X_ml, y, g, ridge=1e-2)
        # Don't dump 500 betas to keep JSON compact
        res_ml.pop('betas', None)
        res_ml['n_features'] = int(X_ml.shape[1])
        multilayer_results[ml_name] = res_ml

    return dict(
        model=model_name,
        corpus=which,
        n_tokens=N_tokens,
        n_passages=int(len(np.unique(g))),
        mean_entropy_bits=float(y.mean()),
        sd_entropy_bits=float(y.std()),
        single_metrics=metric_summaries,
        single_layer_combos=combo_summaries,
        multilayer_joint=multilayer_results,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['gpt2-xl', 'pythia-2.8b', 'Qwen3-0.6B'])
    ap.add_argument('--max-long', type=int, default=1024)
    ap.add_argument('--max-short', type=int, default=512)
    args = ap.parse_args()

    self_test_info = self_test_geometry()
    corpus = json.loads(CORPUS_PATH.read_text())
    texts = {
        k: [x['text'] if isinstance(x, dict) else x for x in corpus[k]]
        for k in ['long', 'short']
    }

    all_results = dict(self_test=self_test_info, models={})
    for m_name in args.models:
        log(f"=== Loading model: {m_name} ===")
        m_path = f"/home/lmxxf/work/models/{m_name}"
        tok = AutoTokenizer.from_pretrained(m_path, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'right'
        model = AutoModelForCausalLM.from_pretrained(
            m_path, local_files_only=True, dtype=torch.float32, device_map='cuda', attn_implementation='sdpa'
        ).eval()
        model.requires_grad_(False)

        log(f"Estimating layer mean vectors for {m_name}...")
        mu_blocks = estimate_layer_means(model, tok, m_name, texts['long'], n_Calibration=64)

        res_long = evaluate_corpus(model, tok, m_name, 'long', texts['long'], mu_blocks, max_passages=args.max_long)
        res_short = evaluate_corpus(model, tok, m_name, 'short', texts['short'], mu_blocks, max_passages=args.max_short)

        model_out = dict(long=res_long, short=res_short)
        all_results['models'][m_name] = model_out
        (ROOT / f"results_{m_name}.json").write_text(json.dumps(model_out, ensure_ascii=False, indent=2))
        log(f"Saved results_{m_name}.json")

        del model, tok, mu_blocks
        gc.collect()
        torch.cuda.empty_cache()

    (ROOT / "summary_all_models.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    log("=== ALL MODELS COMPLETED ===")
