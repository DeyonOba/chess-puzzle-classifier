import numpy as np

from src.utils.ranking import (coverage_error, label_ranking_average_precision, one_error, per_label_average_precision,
                               precision_at_k, ranking_loss, ranking_report)


def test_perfect_ranking():
    p = np.array([[0.9, 0.8, 0.1, 0.0], [0.2, 0.9, 0.1, 0.3]]); t = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])
    assert precision_at_k(p, t, 1) == 1.0 and precision_at_k(p, t, 2) == 1.0
    assert one_error(p, t) == 0.0 and ranking_loss(p, t) == 0.0
    assert label_ranking_average_precision(p, t) == 1.0 and coverage_error(p, t) == 2.0
    assert np.allclose(per_label_average_precision(p, t), [1.0, 1.0, np.nan, 1.0], equal_nan=True)


def test_worst_ranking():
    p = np.array([[0.1, 0.2, 0.9, 0.8]]); t = np.array([[1, 1, 0, 0]])
    assert one_error(p, t) == 1.0 and ranking_loss(p, t) == 1.0 and coverage_error(p, t) == 4.0
    assert abs(label_ranking_average_precision(p, t) - (1 / 3 + 2 / 4) / 2) < 1e-9


def test_report_keys_and_ranges():
    rng = np.random.default_rng(0)
    p = rng.random((50, 6)); t = (rng.random((50, 6)) < 0.3).astype(np.uint8); t[:, 0] = 1
    r = ranking_report(p, t)
    assert set(r) == {"precision@1", "precision@3", "precision@5", "hit@1", "hit@3", "one_error", "coverage_error", "lrap", "ranking_loss", "macro_ap"}
    assert all(0 <= v <= 6 for v in r.values())
