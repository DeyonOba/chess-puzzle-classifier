import numpy as np
import pytest

from src.data.components.splits import ContiguousBatchSampler, iterative_stratified_split, split_report


def test_stratified_split_keeps_rare_label_ratio():
    rng = np.random.default_rng(1)
    N, L = 20_000, 8
    freq = np.array([0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    Y = (rng.random((N, L)) < freq).astype(np.uint8)
    Y[:, 0] = 1  # make sure nobody is unlabeled
    fr = (0.8, 0.1, 0.1)
    a = iterative_stratified_split(Y, fr, seed=0)
    assert a.min() == 0 and a.max() == 2 and len(a) == N
    sizes = np.bincount(a) / N
    assert np.allclose(sizes, fr, atol=0.01)
    worst = split_report(Y, a, fr)
    # every label within max(1 sample, 5%) of its target count in every split
    assert abs(worst["actual"] - worst["expected"]) <= max(1.0, 0.05 * worst["expected"]), worst


def test_stratified_split_is_deterministic():
    Y = (np.random.default_rng(0).random((5_000, 4)) < 0.3).astype(np.uint8)
    assert np.array_equal(iterative_stratified_split(Y, seed=3), iterative_stratified_split(Y, seed=3))
    assert not np.array_equal(iterative_stratified_split(Y, seed=3), iterative_stratified_split(Y, seed=4))


def test_contiguous_batch_sampler():
    idx = np.array([3, 1, 7, 8, 9, 20, 21])
    s = ContiguousBatchSampler(idx, batch_size=3, shuffle=False)
    batches = list(s)
    assert batches == [[1, 3, 7], [8, 9, 20], [21]] and len(s) == 3
    s2 = ContiguousBatchSampler(idx, batch_size=3, shuffle=True, seed=0, drop_last=True)
    assert len(s2) == 2 and sorted(sum(list(s2), [])) == [1, 3, 7, 8, 9, 20]
    s2.set_epoch(0); e0 = list(s2); s2.set_epoch(1); e1 = list(s2)
    assert sorted(map(tuple, e0)) == sorted(map(tuple, e1))
