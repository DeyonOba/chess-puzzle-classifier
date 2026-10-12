"""Per-class decision thresholds that maximise F1 on a validation set.

For each label the probabilities are sorted descending; predicting the top-k
rows positive gives tp(k) = cumsum(targets), so F1(k) = 2 tp / (k + P) with P
the number of positives. The best k is found exactly, and the threshold is the
midpoint between the k-th and (k+1)-th probability. Labels with no positives
in the tuning set keep `fallback`.
"""

from __future__ import annotations

import numpy as np
import torch


def tune_thresholds(probs: np.ndarray, targets: np.ndarray, fallback: float = 0.5) -> np.ndarray:
    """probs, targets: (n, L). Returns (L,) float32 thresholds maximising per-class F1."""
    probs = np.asarray(probs, dtype=np.float32)
    targets = np.asarray(targets).astype(bool)
    n, L = probs.shape
    out = np.full(L, fallback, dtype=np.float32)
    k = np.arange(1, n + 1, dtype=np.float64)
    for j in range(L):
        P = int(targets[:, j].sum())
        if P == 0:
            continue
        order = np.argsort(-probs[:, j], kind="stable")
        p_sorted = probs[order, j]
        tp = np.cumsum(targets[order, j]).astype(np.float64)
        f1 = 2 * tp / (k + P)
        best = int(np.argmax(f1))
        upper = p_sorted[best]
        lower = p_sorted[best + 1] if best + 1 < n else 0.0
        out[j] = (upper + lower) / 2
    return out


def f1_at_thresholds(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> dict:
    """micro / macro F1, macro P/R and exact match using per-class thresholds."""
    preds = np.asarray(probs) >= np.asarray(thresholds)[None, :]
    t = np.asarray(targets).astype(bool)
    tp = (preds & t).sum(0).astype(np.float64)
    fp = (preds & ~t).sum(0).astype(np.float64)
    fn = (~preds & t).sum(0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return {
        "micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "exact_match": float((preds == t).all(1).mean()),
        "per_class_f1": f1.astype(np.float32),
    }


def apply_thresholds(logits: torch.Tensor, thresholds: np.ndarray | torch.Tensor) -> torch.Tensor:
    thr = torch.as_tensor(thresholds, dtype=logits.dtype, device=logits.device)
    return torch.sigmoid(logits) >= thr


def decode_grouped(probs: np.ndarray, thresholds: np.ndarray, layout) -> np.ndarray:
    """Consistent decoding for grouped heads: argmax within each exclusive group (a member is
    predicted only if its probability beats the group's 'none' mass), per-class thresholds on
    the free labels. Guarantees at most one label per group."""
    probs = np.asarray(probs)
    preds = probs >= np.asarray(thresholds)[None, :]
    for _, idx in layout.groups:
        p = probs[:, idx]
        none_mass = np.clip(1 - p.sum(1), 0, 1)
        winner = p.argmax(1)
        fire = p.max(1) > none_mass
        block = np.zeros_like(p, dtype=bool)
        block[np.arange(len(p)), winner] = fire
        preds[:, idx] = block
    return preds
