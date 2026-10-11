import numpy as np
import torch

from src.models.heads import MLGCNHead, Query2LabelHead, build_adjacency
from src.models.puzzle_classifier import PuzzleSequenceClassifier


def test_build_adjacency_properties():
    rng = np.random.default_rng(0)
    Y = (rng.random((5000, 6)) < 0.3).astype(np.uint8)
    Y[:, 1] = Y[:, 0]  # label 1 == label 0 -> P(1|0) = 1
    A, cond = build_adjacency(Y, tau=0.4, p=0.2)
    assert A.shape == (6, 6) and cond.shape == (6, 6)
    assert np.allclose(np.diag(cond), 1.0) and cond[0, 1] == 1.0
    assert np.allclose(A, A.T) and np.isfinite(A).all()
    assert A[0, 1] > 0 and A[0, 1] == A[1, 0]


def test_q2l_forward_shape_and_padding_invariance():
    torch.manual_seed(0)
    m = PuzzleSequenceClassifier(7, hidden_dim=32, num_res_blocks=1, max_positions=6, head="q2l",
                                 head_kwargs={"num_layers": 1, "num_heads": 4}).eval()
    boards = torch.rand(3, 6, 18, 8, 8).round(); lengths = torch.tensor([1, 3, 6])
    out = m(boards, lengths)
    assert out.shape == (3, 7)
    noisy = boards.clone()
    for b, n in enumerate(lengths.tolist()):
        noisy[b, n:] = torch.rand(6 - n, 18, 8, 8).round()
    torch.testing.assert_close(m(noisy, lengths), out, atol=1e-5, rtol=1e-4)


def test_mlgcn_forward_and_state_roundtrip():
    torch.manual_seed(0)
    Y = (np.random.default_rng(1).random((2000, 7)) < 0.3).astype(np.uint8)
    A, cond = build_adjacency(Y)
    m = PuzzleSequenceClassifier(7, hidden_dim=32, num_res_blocks=1, max_positions=4, head="mlgcn",
                                 head_kwargs={"gcn_dim": 16, "adjacency": A, "label_init": cond}).eval()
    boards = torch.rand(2, 4, 18, 8, 8).round(); lengths = torch.tensor([2, 4])
    out = m(boards, lengths)
    assert out.shape == (2, 7) and torch.isfinite(out).all()
    # a fresh model with no adjacency must reproduce the output after load_state_dict
    m2 = PuzzleSequenceClassifier(7, hidden_dim=32, num_res_blocks=1, max_positions=4, head="mlgcn",
                                  head_kwargs={"gcn_dim": 16}).eval()
    m2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m2(boards, lengths), out)


def test_heads_backprop():
    for head, kw in (("q2l", {"num_layers": 1, "num_heads": 2}), ("mlgcn", {"gcn_dim": 8})):
        m = PuzzleSequenceClassifier(5, hidden_dim=16, num_res_blocks=1, max_positions=3, head=head, head_kwargs=kw)
        out = m(torch.rand(2, 3, 18, 8, 8).round(), torch.tensor([1, 3]))
        out.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
