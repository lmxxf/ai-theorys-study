"""Exclude cached candidates and cached labels; average over actual missing experts."""
import torch

def cold_loss(logits, truth, resident):
    missing = ~resident.gather(1, truth)
    valid = missing.any(1)
    if not valid.any():
        return logits.sum() * 0.0
    logp = logits[valid].masked_fill(resident[valid], float('-inf')).log_softmax(1)
    selected = logp.gather(1, truth[valid]).masked_fill(~missing[valid], 0.0)
    return -selected.sum() / missing[valid].sum()

def cold_recall(logits, truth, resident, k=10):
    missing = ~resident.gather(1, truth)
    pred = logits.masked_fill(resident, float('-inf')).topk(k, dim=1).indices
    hits = ((pred[:, :, None] == truth[:, None, :]).any(1) & missing).sum()
    return float(hits) / int(missing.sum()) if missing.any() else 0.0
