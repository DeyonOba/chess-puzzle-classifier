import numpy as np
import torch
import torch.nn.functional as F

from src.utils.losses import AsymmetricLoss, SigmoidFocalLoss, build_loss, compute_pos_weight


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, 7, generator=g), (torch.rand(16, 7, generator=g) < 0.3).float()


def test_focal_gamma0_equals_bce():
    x, y = _batch()
    torch.testing.assert_close(SigmoidFocalLoss(gamma=0.0)(x, y), F.binary_cross_entropy_with_logits(x, y))


def test_asl_degenerate_equals_bce():
    x, y = _batch()
    torch.testing.assert_close(AsymmetricLoss(0.0, 0.0, 0.0)(x, y), F.binary_cross_entropy_with_logits(x, y), atol=1e-6, rtol=1e-5)


def test_focal_and_asl_downweight_easy_examples():
    x, y = _batch()
    assert SigmoidFocalLoss(gamma=2.0)(x, y) < F.binary_cross_entropy_with_logits(x, y)
    assert AsymmetricLoss()(x, y) < F.binary_cross_entropy_with_logits(x, y)


def test_losses_are_finite_and_differentiable():
    x0, y = _batch()
    for loss in (SigmoidFocalLoss(), SigmoidFocalLoss(alpha=0.25), AsymmetricLoss()):
        x = x0.detach().clone().requires_grad_(True)
        out = loss(x, y)
        out.backward()
        assert torch.isfinite(out) and torch.isfinite(x.grad).all()
    big = torch.tensor([[50.0, -50.0]]); t = torch.tensor([[1.0, 0.0]])
    for loss in (SigmoidFocalLoss(), AsymmetricLoss()):
        assert torch.isfinite(loss(big, t))


def test_compute_pos_weight_cap_and_power():
    Y = np.zeros((1000, 3), dtype=np.uint8)
    Y[:500, 0] = 1  # 1:1 -> weight 1
    Y[:10, 1] = 1   # 99:1 -> capped at 20
    # label 2 never positive -> cap
    w = compute_pos_weight(Y, cap=20.0, power=1.0)
    assert w.tolist() == [1.0, 20.0, 20.0]
    w_sqrt = compute_pos_weight(Y, cap=20.0, power=0.5)  # the default power
    assert abs(w_sqrt[1].item() - 99 ** 0.5) < 1e-4 and w_sqrt[2].item() == 20.0
    assert compute_pos_weight(Y, cap=20.0).tolist() == w_sqrt.tolist()


def test_build_loss_names():
    x, y = _batch()
    Y = (np.random.default_rng(0).random((100, 7)) < 0.3).astype(np.uint8)
    for name in ("bce", "weighted_bce", "focal", "asl"):
        assert torch.isfinite(build_loss(name, Y)(x, y))
