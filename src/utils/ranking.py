"""Per-puzzle ranking metrics for multi-label prediction (threshold-free).

probs, targets: (n, L). All functions return floats; rows with no positive
label are skipped where the metric is undefined.
"""

from __future__ import annotations

import numpy as np


def _valid(targets: np.ndarray) -> np.ndarray:
    return targets.sum(1) > 0


def precision_at_k(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    hits = np.take_along_axis(targets, top, axis=1)
    return float(hits.mean())


def hit_at_k(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Fraction of puzzles with at least one true label in the top k."""
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((np.take_along_axis(targets, top, axis=1).sum(1) > 0).mean())


def one_error(probs: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of puzzles whose top-ranked label is wrong."""
    return float((np.take_along_axis(targets, probs.argmax(1)[:, None], axis=1)[:, 0] == 0).mean())


def coverage_error(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean depth (1-indexed) needed to cover every true label."""
    t = targets.astype(bool); v = _valid(targets)
    ranks = (-probs).argsort(1).argsort(1) + 1  # rank of each label, 1 = highest
    worst = np.where(t, ranks, 0).max(1)
    return float(worst[v].mean())


def label_ranking_average_precision(probs: np.ndarray, targets: np.ndarray) -> float:
    """LRAP: for each true label, fraction of labels ranked at or above it that are true; averaged."""
    t = targets.astype(bool); v = _valid(targets)
    out = []
    for i in np.flatnonzero(v):
        order = np.argsort(-probs[i])
        tr = t[i][order]
        pos = np.flatnonzero(tr) + 1
        out.append((np.cumsum(tr)[pos - 1] / pos).mean())
    return float(np.mean(out))


def ranking_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean fraction of (true, false) label pairs ordered wrongly."""
    t = targets.astype(bool)
    out = []
    for i in range(len(t)):
        p, n = probs[i][t[i]], probs[i][~t[i]]
        if len(p) and len(n):
            out.append((p[:, None] <= n[None, :]).mean())
    return float(np.mean(out))


def per_label_average_precision(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """AP per label (area under the precision-recall step curve); NaN if no positives."""
    L = probs.shape[1]
    ap = np.full(L, np.nan)
    for j in range(L):
        t = targets[:, j].astype(bool)
        P = t.sum()
        if P == 0:
            continue
        order = np.argsort(-probs[:, j], kind="stable")
        tp = np.cumsum(t[order])
        prec = tp / np.arange(1, len(t) + 1)
        ap[j] = (prec * t[order]).sum() / P
    return ap


def ranking_report(probs: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "precision@1": precision_at_k(probs, targets, 1),
        "precision@3": precision_at_k(probs, targets, 3),
        "precision@5": precision_at_k(probs, targets, 5),
        "hit@1": hit_at_k(probs, targets, 1),
        "hit@3": hit_at_k(probs, targets, 3),
        "one_error": one_error(probs, targets),
        "coverage_error": coverage_error(probs, targets),
        "lrap": label_ranking_average_precision(probs, targets),
        "ranking_loss": ranking_loss(probs, targets),
        "macro_ap": float(np.nanmean(per_label_average_precision(probs, targets))),
    }
