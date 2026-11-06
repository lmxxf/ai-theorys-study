#!/usr/bin/env python3
"""
Micro-level case studies on ROUTINE vs HARD and Trap PAIRS (conflict span vs control span)
across gpt2-xl, pythia-2.8b, and Qwen3-0.6B.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import math
import gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_manifold_exp import (
    ROOT, CORPUS_PATH, METRIC_ORDER, ANGLE_METRICS,
    entropy_bits, compute_temporal_metrics_batched_layers,
    compute_vertical_layer_metrics, estimate_layer_means, get_model_parts, log
)

ROUTINE_EN = [
    "One, two, buckle my shoe. Three, four, knock at the door. Five, six, pick up sticks.",
    "The sun rises in the east and sets in the west. After day comes night, and after winter comes spring.",
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday are the seven days of the week.",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 = 6, 4 + 4 = 8, 5 + 5 = 10, 6 + 6 = 12.",
    "The capital of France is Paris, the capital of Japan is Tokyo, and the capital of England is London.",
    "The cat sat on the mat. The dog sat by the door. The bird flew in the blue sky.",
]

HARD_EN = [
    "If yesterday were tomorrow, then today would be Friday. Question: What day of the week is today actually?",
    "There are 5 candles on a table. The wind blows out 2 of them, and the rest burn down completely. How many candles remain on the table in the morning?",
    "Not everyone who failed the exam did not study hard. Does this imply that everyone who studied hard passed the exam?",
    "A bat and a ball cost 110 dollars in total. The bat costs 100 dollars more than the ball. How much does the ball cost?",
    "Alice says Bob is lying. Bob says Charlie is lying. Charlie says both Alice and Bob are lying. If only one person is telling the truth, who is it?",
    "The horse raced past the barn fell down suddenly. In this sentence, which word is the main predicate verb?",
]

ROUTINE_ZH = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "1+1=2, 2+2=4, 3+3=6, 4+4=8, 5+5=10。",
    "中国的首都是北京，日本的首都是东京，法国的首都是巴黎。",
    "春天之后是夏天，夏天之后是秋天，秋天之后是冬天。",
    "一周有七天，一天有二十四小时，一小时有六十分钟。",
    "太阳从东方升起，从西方落下。白天过去，夜晚来临。",
]

HARD_ZH = [
    "如果昨天是明天的话就好了，这样今天就是周五了。请问句子里的今天实际上是周几？",
    "桌上有5根蜡烛，风吹灭了2根，其余持续燃烧直到耗尽，天亮后桌子上还剩几根蜡烛？",
    "不是所有没有通过考试的人都没有努力，这是否意味着所有努力的人都通过了考试？",
    "一支球拍和一个球共110元，球拍比球贵100元，请问球多少钱？",
    "甲说乙说谎，乙说丙说谎，丙说甲和乙都说谎。如果只有一人说真话，谁说真话？",
    "一辆车以每小时30公里去某地，回来时以每小时60公里返回同一路线，往返平均速度是多少？",
]

PAIRS_EN = [
    (
        "age_en",
        "A mother is 30 years old this year, and her biological son is 35 years old. How old was the mother when she gave birth to her son?",
        "A mother is 60 years old this year, and her biological son is 35 years old. How old was the mother when she gave birth to her son?",
        "35 years old",
    ),
    (
        "table_en",
        "Two people sit at opposite ends of a ten-meter-wide conference table. Without leaving his seat, he casually reaches out and rubs her hair. Is this physically possible?",
        "Two people sit across a half-meter-wide coffee table. Without leaving his seat, he casually reaches out and rubs her hair. Is this physically possible?",
        "rubs her hair",
    ),
    (
        "glass_en",
        "She stands outside a completely sealed glass room with no doors or windows. Without her arm passing through the glass, she directly picks up the apple inside. Is this possible?",
        "She stands outside a glass room with an open window. Reaching her arm through the open window, she directly picks up the apple inside. Is this possible?",
        "picks up the apple",
    ),
    (
        "jet_en",
        "During the ancient Battle of Red Cliffs in 208 AD, Zhuge Liang piloted a stealth fighter jet to sink Cao Cao's wooden warships. How many ships did he sink?",
        "During the ancient Battle of Red Cliffs in 208 AD, Zhuge Liang used fire attacks and the southeastern wind to burn Cao Cao's wooden warships. How many ships did he burn?",
        "stealth fighter jet",
    ),
]

PAIRS_ZH = [
    (
        "fact_book",
        "在《红楼梦》里，林黛玉倒拔垂杨柳发生在第几回？请直接给出回数：",
        "在《水浒传》里，鲁智深倒拔垂杨柳发生在第几回？请直接给出回数：",
        "倒拔垂杨柳",
    ),
    (
        "fact_age",
        "妈妈今年30岁，她亲生儿子今年35岁，请问妈妈生儿子的时候几岁？",
        "妈妈今年60岁，她亲生儿子今年35岁，请问妈妈生儿子的时候几岁？",
        "35岁",
    ),
    (
        "fact_plane",
        "三国演义里，诸葛亮在赤壁之战中驾驶歼-20战机击落了曹操的几艘战船？",
        "三国演义里，诸葛亮在赤壁之战中借东风帮助火攻，曹操的战船遭遇了什么？",
        "歼-20战机",
    ),
    (
        "space_table",
        "两人隔着十米宽的巨大会议桌面对面坐着，他始终没有起身离开座位，只是顺手揉了揉她的头发。这句话在物理空间上合理吗？",
        "两人隔着半米宽的小茶几面对面坐着，他始终没有起身离开座位，只是顺手揉了揉她的头发。这句话在物理空间上合理吗？",
        "揉了揉她的头发",
    ),
    (
        "space_shirt",
        "他身上紧紧穿着一件拉链拉到顶的厚羽绒服，没有拉开羽绒服拉链，就直接把贴身穿的纯棉T恤从头顶脱了下来。这句话在穿衣顺序上合理吗？",
        "他先拉开羽绒服拉链，脱掉厚羽绒服，再把贴身穿的纯棉T恤从头顶脱了下来。这句话在穿衣顺序上合理吗？",
        "脱了下来",
    ),
    (
        "space_wall",
        "她站在完全密封、没有门窗的玻璃房外，手臂没有穿过玻璃，却直接伸手拿起房内桌上的苹果。这句话在物理空间上合理吗？",
        "她站在有敞开窗户的玻璃房外，手臂穿过敞开的窗户，伸手拿起房内桌上的苹果。这句话在物理空间上合理吗？",
        "拿起房内桌上的苹果",
    ),
]


@torch.inference_mode()
def run_single_text(model, tok, model_name: str, text: str, mu_blocks: torch.Tensor):
    _, blocks = get_model_parts(model, model_name)
    L = len(blocks)
    inp = tok(text, add_special_tokens=False, return_tensors='pt').to('cuda')
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

    H = entropy_bits(out.logits)[0].cpu().numpy()  # [T]
    h_blocks_tensor = torch.stack(captured_blocks, dim=0)  # [L, 1, T, D]
    all_h_tensor = torch.cat([captured_embed[0].unsqueeze(0), h_blocks_tensor], dim=0)
    t_metrics = compute_temporal_metrics_batched_layers(h_blocks_tensor, mu_blocks)
    v_metrics = compute_vertical_layer_metrics(all_h_tensor)
    t_metrics.update(v_metrics)
    metrics_np = {k: v[:, 0, :].cpu().numpy() for k, v in t_metrics.items()}  # each [L, T]
    return H, metrics_np, inp.input_ids[0].shape[0]


def summarize_group_texts(model, tok, model_name: str, texts_list, mu_blocks: torch.Tensor):
    _, blocks = get_model_parts(model, model_name)
    L = len(blocks)
    mid_start = int(round(0.45 * L))
    mid_end = int(round(0.85 * L))
    mid_layers = list(range(mid_start, mid_end))

    key_metrics = [
        'C_flat_3', 'Theta_surf_3', 'Theta_surf_c_3', 'Extrinsic_bias_3',
        'Kappa_geo_3', 'S_geo_3', 'S_geo_c_3', 's_rel', 'norm', 'dlog_norm',
        'accel_geo', 'erank_dir', 'pr_dir', 'erank_weighted', 'v_layer', 'theta_layer', 'erank_layer'
    ]
    accum = {m: [] for m in key_metrics}
    H_all = []

    for text in texts_list:
        H, m_np, T = run_single_text(model, tok, model_name, text, mu_blocks)
        if T <= 6:
            continue
        ix = np.arange(5, T - 1)
        H_all.append(H[ix])
        for m in key_metrics:
            # average over mid-to-late layers [mid_start..mid_end]
            val_mid = np.nanmean(m_np[m][mid_layers][:, ix], axis=0)
            accum[m].append(val_mid)

    H_cat = np.concatenate(H_all)
    res = dict(
        n_tokens=int(len(H_cat)),
        mid_layer_range=[mid_start, mid_end - 1],
        mean_H_bits=float(np.mean(H_cat)),
        sd_H_bits=float(np.std(H_cat)),
        metrics={}
    )
    for m in key_metrics:
        v_cat = np.concatenate(accum[m])
        mean_v = float(np.nanmean(v_cat))
        sd_v = float(np.nanstd(v_cat))
        res['metrics'][m] = dict(
            mean=mean_v,
            sd=sd_v,
            mean_deg=(mean_v * 180.0 / math.pi if m in ANGLE_METRICS else None),
            sd_deg=(sd_v * 180.0 / math.pi if m in ANGLE_METRICS else None),
        )
    return res


def evaluate_pairs(model, tok, model_name: str, pairs_list, mu_blocks: torch.Tensor):
    _, blocks = get_model_parts(model, model_name)
    L = len(blocks)
    mid_start = int(round(0.45 * L))
    mid_end = int(round(0.85 * L))
    mid_layers = list(range(mid_start, mid_end))

    key_metrics = [
        'c_flat', 'C_flat_3', 'theta_surf', 'Theta_surf_3', 'theta_surf_c',
        'kappa_geo', 'Kappa_geo_3', 's_geo', 'S_geo_3', 's_rel', 'norm',
        'accel_geo', 'erank_dir', 'pr_dir', 'erank_weighted', 'v_layer', 'theta_layer', 'erank_layer'
    ]
    pair_records = []
    trap_vals = {m: [] for m in key_metrics}
    ctrl_vals = {m: [] for m in key_metrics}
    trap_H, ctrl_H = [], []

    for pid, trap_text, ctrl_text, trigger in pairs_list:
        H_t, m_t, T_t = run_single_text(model, tok, model_name, trap_text, mu_blocks)
        H_c, m_c, T_c = run_single_text(model, tok, model_name, ctrl_text, mu_blocks)
        ix_t = np.arange(5, T_t - 1)
        ix_c = np.arange(5, T_c - 1)
        trap_H.append(float(np.mean(H_t[ix_t])))
        ctrl_H.append(float(np.mean(H_c[ix_c])))

        rec = dict(
            pair_id=pid,
            trap_mean_H_bits=float(np.mean(H_t[ix_t])),
            ctrl_mean_H_bits=float(np.mean(H_c[ix_c])),
            delta_H_bits=float(np.mean(H_t[ix_t]) - np.mean(H_c[ix_c])),
            metrics={}
        )
        for m in key_metrics:
            vt = float(np.nanmean(m_t[m][mid_layers][:, ix_t]))
            vc = float(np.nanmean(m_c[m][mid_layers][:, ix_c]))
            trap_vals[m].append(vt)
            ctrl_vals[m].append(vc)
            rec['metrics'][m] = dict(
                trap=vt,
                ctrl=vc,
                delta=vt - vc,
                trap_deg=(vt * 180.0 / math.pi if m in ANGLE_METRICS else None),
                ctrl_deg=(vc * 180.0 / math.pi if m in ANGLE_METRICS else None),
                delta_deg=((vt - vc) * 180.0 / math.pi if m in ANGLE_METRICS else None),
            )
        pair_records.append(rec)

    summary_diff = {}
    for m in key_metrics:
        tv = np.array(trap_vals[m])
        cv = np.array(ctrl_vals[m])
        dv = tv - cv
        summary_diff[m] = dict(
            trap_mean=float(tv.mean()),
            ctrl_mean=float(cv.mean()),
            mean_delta=float(dv.mean()),
            trap_mean_deg=(float(tv.mean()) * 180.0 / math.pi if m in ANGLE_METRICS else None),
            ctrl_mean_deg=(float(cv.mean()) * 180.0 / math.pi if m in ANGLE_METRICS else None),
            mean_delta_deg=(float(dv.mean()) * 180.0 / math.pi if m in ANGLE_METRICS else None),
            n_trap_greater=int(np.sum(dv > 0)),
            n_pairs=len(dv),
        )

    return dict(
        trap_mean_H_bits=float(np.mean(trap_H)),
        ctrl_mean_H_bits=float(np.mean(ctrl_H)),
        mean_delta_H_bits=float(np.mean(np.array(trap_H) - np.array(ctrl_H))),
        summary_by_metric=summary_diff,
        pairs=pair_records,
    )


if __name__ == '__main__':
    corpus = json.loads(CORPUS_PATH.read_text())
    long_texts = [x['text'] if isinstance(x, dict) else x for x in corpus['long']]

    case_results = {}
    for m_name in ['gpt2-xl', 'pythia-2.8b', 'Qwen3-0.6B']:
        log(f"Running case studies on {m_name}...")
        m_path = f"/home/lmxxf/work/models/{m_name}"
        tok = AutoTokenizer.from_pretrained(m_path, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'right'
        model = AutoModelForCausalLM.from_pretrained(
            m_path, local_files_only=True, dtype=torch.float32, device_map='cuda', attn_implementation='sdpa'
        ).eval()
        model.requires_grad_(False)

        mu_blocks = estimate_layer_means(model, tok, m_name, long_texts, n_Calibration=64)

        is_zh_capable = (m_name == 'Qwen3-0.6B')
        routine_en_res = summarize_group_texts(model, tok, m_name, ROUTINE_EN, mu_blocks)
        hard_en_res = summarize_group_texts(model, tok, m_name, HARD_EN, mu_blocks)
        pairs_en_res = evaluate_pairs(model, tok, m_name, PAIRS_EN, mu_blocks)

        routine_zh_res = summarize_group_texts(model, tok, m_name, ROUTINE_ZH, mu_blocks)
        hard_zh_res = summarize_group_texts(model, tok, m_name, HARD_ZH, mu_blocks)
        pairs_zh_res = evaluate_pairs(model, tok, m_name, PAIRS_ZH, mu_blocks)

        case_results[m_name] = dict(
            en=dict(routine=routine_en_res, hard=hard_en_res, pairs=pairs_en_res),
            zh=dict(routine=routine_zh_res, hard=hard_zh_res, pairs=pairs_zh_res),
        )
        del model, tok, mu_blocks
        gc.collect()
        torch.cuda.empty_cache()

    (ROOT / "case_studies_results.json").write_text(json.dumps(case_results, ensure_ascii=False, indent=2))
    log("Saved case_studies_results.json")
