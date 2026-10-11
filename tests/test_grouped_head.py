import json
import numpy as np
import torch

from src.models.puzzle_classifier import PuzzleSequenceClassifier
from src.utils.label_groups import LabelLayout
from src.utils.losses import GroupedLoss, build_loss

LABELS = ["mate", "crushing", "advantage", "equality", "fork", "pin", "mateIn1", "mateIn2", "oneMove", "short"]


def test_layout_and_group_targets():
    lay = LabelLayout(LABELS)
    assert [n for n, _ in lay.groups] == ["outcome", "length", "mateIn"]
    assert [LABELS[i] for i in lay.free] == ["fork", "pin"]
    t = torch.zeros(3, len(LABELS))
    t[0, [0, 6, 8]] = 1  # mate, mateIn1, oneMove
    t[1, [1, 4, 9]] = 1  # crushing, fork, short
    # row 2: nothing -> 'none' everywhere
    g = lay.group_targets(t)
    assert g[0].tolist() == [0, 1, 4] and g[1].tolist() == [0, 1, 2] and g[2].tolist() == [0, 2, 2]


def test_grouped_head_probs_sum_to_one_within_groups():
    torch.manual_seed(0)
    m = PuzzleSequenceClassifier(len(LABELS), hidden_dim=16, num_res_blocks=1, max_positions=4, head="grouped", labels=LABELS).eval()
    raw = m(torch.rand(5, 4, 18, 8, 8).round(), torch.tensor([1, 2, 3, 4, 4]))
    assert raw.shape == (5, len(LABELS) + 3)
    p = m.probs(raw)
    assert p.shape == (5, len(LABELS)) and (p >= 0).all() and (p <= 1).all()
    lay = m.head.layout
    for _, idx in lay.groups:
        assert (p[:, idx].sum(1) <= 1 + 1e-5).all()  # remainder is 'none'
    none_prob = 1 - p[:, lay.groups[0][1]].sum(1)
    assert (none_prob >= -1e-5).all()


def test_grouped_loss_runs_and_backprops():
    torch.manual_seed(0)
    m = PuzzleSequenceClassifier(len(LABELS), hidden_dim=16, num_res_blocks=1, max_positions=4, head="grouped", labels=LABELS)
    Y = (np.random.default_rng(0).random((200, len(LABELS))) < 0.3).astype(np.uint8)
    free = build_loss("weighted_bce", Y[:, m.head.layout.free])
    loss_fn = GroupedLoss(m.head, free, Y)
    assert len(loss_fn.class_weights) == 3 and loss_fn.class_weights[0].shape == (5,)
    raw = m(torch.rand(8, 4, 18, 8, 8).round(), torch.tensor([1, 2, 3, 4, 4, 1, 2, 3]))
    t = torch.from_numpy(Y[:8]).float()
    loss = loss_fn(raw, t)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0


def test_flat_head_unchanged():
    m = PuzzleSequenceClassifier(len(LABELS), hidden_dim=16, num_res_blocks=1, max_positions=4)
    raw = m(torch.rand(2, 4, 18, 8, 8).round(), torch.tensor([1, 4]))
    assert raw.shape == (2, len(LABELS)) and torch.allclose(m.probs(raw), torch.sigmoid(raw))
