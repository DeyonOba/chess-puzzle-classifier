import numpy as np

from src.utils.thresholds import f1_at_thresholds, tune_thresholds


def _brute_best_f1(p, t):
    best = 0.0
    for thr in np.unique(np.r_[p, 0.0, 1.0]):
        pred = p >= thr
        tp = (pred & t).sum(); fp = (pred & ~t).sum(); fn = (~pred & t).sum()
        if 2 * tp + fp + fn:
            best = max(best, 2 * tp / (2 * tp + fp + fn))
    return best


def test_tuned_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    n, L = 2000, 5
    t = rng.random((n, L)) < np.array([0.5, 0.2, 0.05, 0.01, 0.3])
    p = np.clip(t * 0.4 + rng.random((n, L)) * 0.7, 0, 1).astype(np.float32)  # informative but noisy
    thr = tune_thresholds(p, t)
    got = f1_at_thresholds(p, t, thr)["per_class_f1"]
    for j in range(L):
        assert abs(got[j] - _brute_best_f1(p[:, j], t[:, j])) < 1e-6, j


def test_tuned_beats_fixed_half_on_underconfident_rare_label():
    rng = np.random.default_rng(1)
    n = 5000
    t = rng.random(n) < 0.02
    p = np.where(t, rng.uniform(0.2, 0.45, n), rng.uniform(0.0, 0.2, n)).astype(np.float32)  # positives all under 0.5
    P, T = p[:, None], t[:, None]
    assert f1_at_thresholds(P, T, np.array([0.5]))["macro_f1"] == 0.0
    thr = tune_thresholds(P, T)
    assert 0.2 <= thr[0] <= 0.45 and f1_at_thresholds(P, T, thr)["macro_f1"] > 0.95


def test_no_positives_uses_fallback():
    p = np.random.default_rng(0).random((100, 2)).astype(np.float32)
    t = np.zeros((100, 2), dtype=np.uint8); t[:10, 0] = 1
    thr = tune_thresholds(p, t, fallback=0.9)
    assert thr[1] == 0.9 and thr[0] != 0.9


def test_exact_match_and_micro():
    p = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32); t = np.array([[1, 0], [0, 1]])
    m = f1_at_thresholds(p, t, np.array([0.5, 0.5]))
    assert m["micro_f1"] == 1.0 and m["macro_f1"] == 1.0 and m["exact_match"] == 1.0
