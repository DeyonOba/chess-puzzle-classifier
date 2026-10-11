"""Multi-label loss functions for the theme classifier.

All losses take raw logits (B, L) and float targets (B, L) and return a scalar
mean over all (sample, label) pairs, so they are drop-in replacements for
nn.BCEWithLogitsLoss.

- bce:           plain BCE-with-logits (baseline)
- weighted_bce:  BCE with per-label pos_weight = (neg / pos) ** power, clipped to
                 [1, cap]; computed from the *training* split only
- focal:         sigmoid focal loss (Lin et al. 2017), optional alpha
- asl:           Asymmetric Loss (Ridnik et al. 2021): separate focusing for
                 positives / negatives plus probability shifting on negatives,
                 designed for multi-label imbalance
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_NAMES = ("bce", "weighted_bce", "focal", "asl")


def compute_pos_weight(targets: np.ndarray, cap: float = 20.0, power: float = 0.5) -> torch.Tensor:
    """Per-label positive weight from a (n, L) 0/1 matrix of training targets.

    pos_weight_j = clip((neg_j / pos_j) ** power, 1, cap). Labels with no
    positives get `cap`. Raw inverse frequency reaches ~5000 for the rarest
    themes, which destroys precision; the cap keeps the gradient scale sane.
    """
    pos = targets.sum(0).astype(np.float64)
    neg = len(targets) - pos
    ratio = np.where(pos > 0, (neg / np.maximum(pos, 1)) ** power, cap)  # no positives -> cap, regardless of power
    return torch.tensor(np.clip(ratio, 1.0, cap), dtype=torch.float32)


class SigmoidFocalLoss(nn.Module):
    """Focal loss on independent sigmoids. gamma=0, alpha=None reduces to BCE."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al. 2021). gamma_neg=gamma_pos=0 and clip=0 reduces to BCE.

    Negatives are down-weighted harder than positives (gamma_neg > gamma_pos)
    and easy negatives with p < clip are discarded outright, which is the
    behaviour that matters when 69 of 70 labels are negative for most rows.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = (1 - p + self.clip).clamp(max=1) if self.clip > 0 else 1 - p
        loss_pos = targets * torch.log(p.clamp(min=self.eps)) * (1 - p) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=self.eps)) * (1 - p_neg) ** self.gamma_neg
        return -(loss_pos + loss_neg).mean()


def build_loss(
    name: str,
    train_targets: np.ndarray | None = None,
    pos_weight_cap: float = 20.0,
    pos_weight_power: float = 0.5,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 1.0,
    asl_clip: float = 0.05,
) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        if train_targets is None:
            raise ValueError("weighted_bce needs train_targets to compute pos_weight")
        return nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_targets, pos_weight_cap, pos_weight_power))
    if name == "focal":
        return SigmoidFocalLoss(focal_gamma, focal_alpha)
    if name == "asl":
        return AsymmetricLoss(asl_gamma_neg, asl_gamma_pos, asl_clip)
    raise ValueError(f"unknown loss {name!r}; choose from {LOSS_NAMES}")


class GroupedLoss(nn.Module):
    """Cross-entropy per exclusive group (incl. 'none') + a sigmoid loss on the free labels.

    Group class weights follow the same clip((n_total/n_class)**power, 1, cap)
    recipe as pos_weight so rare mate patterns are not drowned by 'none'.
    """

    def __init__(self, head, free_loss: nn.Module, train_targets: np.ndarray | None = None,
                 cap: float = 20.0, power: float = 0.5, group_weight: float = 1.0):
        super().__init__()
        self.head = head
        self.layout = head.layout
        self.free_loss = free_loss
        self.group_weight = group_weight
        self.class_weights = nn.ParameterList()
        for _, idx in self.layout.groups:
            k = len(idx) + 1
            if train_targets is None:
                w = torch.ones(k)
            else:
                sub = train_targets[:, idx]
                counts = np.append(sub.sum(0), (sub.sum(1) == 0).sum()).astype(np.float64)
                w = torch.tensor(np.clip((len(sub) / np.maximum(counts, 1)) ** power, 1.0, cap), dtype=torch.float32)
            self.class_weights.append(nn.Parameter(w, requires_grad=False))

    def forward(self, raw: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        free = self.layout.free
        loss = self.free_loss(raw[:, free], targets[:, free])
        for gl, cls, w in zip(self.head.group_logits(raw), self.layout.group_targets(targets), self.class_weights):
            loss = loss + self.group_weight * F.cross_entropy(gl, cls, weight=w) / self.layout.num_groups
        return loss
