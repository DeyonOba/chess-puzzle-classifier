import pathlib

import numpy as np
import pytest
import torch

from src.models.puzzle_classifier import PuzzleSequenceClassifier

DEV_H5 = pathlib.Path("data/dev/puzzle_boards_100k.h5")


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_forward_shape_and_padding_invariance(pooling):
    torch.manual_seed(0)
    model = PuzzleSequenceClassifier(num_labels=70, hidden_dim=32, num_res_blocks=1, max_positions=12, pooling=pooling).eval()
    boards = torch.rand(4, 12, 18, 8, 8).round()
    lengths = torch.tensor([1, 4, 6, 12])
    logits = model(boards, lengths)
    assert logits.shape == (4, 70)
    # garbage in padded positions must not change the output
    noisy = boards.clone()
    for b, n in enumerate(lengths.tolist()):
        noisy[b, n:] = torch.rand(12 - n, 18, 8, 8).round()
    torch.testing.assert_close(model(noisy, lengths), logits)


def test_position_only_matches_t1():
    torch.manual_seed(0)
    model = PuzzleSequenceClassifier(num_labels=5, hidden_dim=16, num_res_blocks=1, max_positions=12).eval()
    boards = torch.rand(3, 12, 18, 8, 8).round()
    out_full_len1 = model(boards, torch.tensor([1, 1, 1]))
    out_t1 = model(boards[:, :1], torch.tensor([1, 1, 1]))
    torch.testing.assert_close(out_full_len1, out_t1)


@pytest.mark.skipif(not DEV_H5.exists(), reason="dev HDF5 not built")
def test_dataset_batches():
    from src.data.dataset import PuzzleSequenceDataset

    ds = PuzzleSequenceDataset(DEV_H5)
    assert len(ds) == 100_000 and ds.num_labels == 70 and len(ds.labels) == 70
    boards, lengths, targets = ds.__getitems__([5, 2, 9, 3])  # unsorted on purpose
    assert boards.shape == (4, 12, 18, 8, 8) and targets.shape == (4, 70) and lengths.dtype == torch.int64
    b2, l2, t2 = ds[2]
    torch.testing.assert_close(boards[1], b2)
    assert lengths[1] == l2 and torch.equal(targets[1], t2)
    # scattered path
    b3, _, _ = ds.__getitems__([0, 50_000, 99_999])
    torch.testing.assert_close(b3[1], ds[50_000][0])
    # position-only view
    ds1 = PuzzleSequenceDataset(DEV_H5, max_positions=1)
    b, l, _ = ds1.__getitems__([0, 1])
    assert b.shape == (2, 1, 18, 8, 8) and l.tolist() == [1, 1]
    torch.testing.assert_close(b[0, 0], boards[0, 0] if False else ds[0][0][0])
    # padded positions are all-zero planes
    full = ds.__getitems__([0])[0][0]
    assert full[int(ds[0][1]):].sum() == 0 and full[: int(ds[0][1])].sum() > 0
