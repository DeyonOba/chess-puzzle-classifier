import torch
import torch.nn as nn

from src.models.backbones.resnet_board import ResNetBoardEncoder
from src.utils.label_groups import LabelLayout


class GroupedHead(nn.Module):
    """Softmax per mutually exclusive group (with a 'none' class) + sigmoid for free labels.

    Output layout is (B, L + G): the first L columns line up with the flat label
    list (each grouped label's column is its logit inside its group's softmax),
    the last G columns are each group's 'none' logit. `probs()` maps this to a
    (B, L) per-label probability vector, so thresholding / evaluation code that
    expects one score per label works unchanged.
    """

    def __init__(self, hidden_dim: int, layout: LabelLayout, dropout: float = 0.3):
        super().__init__()
        self.layout = layout
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=dropout))
        self.out = nn.Linear(hidden_dim, layout.num_labels + layout.num_groups)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(pooled))

    def group_logits(self, raw: torch.Tensor) -> list[torch.Tensor]:
        L = self.layout.num_labels
        return [torch.cat([raw[:, idx], raw[:, L + g : L + g + 1]], dim=1) for g, (_, idx) in enumerate(self.layout.groups)]

    def probs(self, raw: torch.Tensor) -> torch.Tensor:
        L = self.layout.num_labels
        out = torch.sigmoid(raw[:, :L])
        for (_, idx), gl in zip(self.layout.groups, self.group_logits(raw)):
            out[:, idx] = torch.softmax(gl, dim=1)[:, : len(idx)]
        return out


class PuzzleSequenceClassifier(nn.Module):
    """Shared ResNet over every position of the solution line, masked pooling, sigmoid head.

    forward(boards (B, T, 18, 8, 8), lengths (B,)) -> raw head output: (B, num_labels) for the
    flat head, (B, num_labels + num_groups) for the grouped head. Use probs() for per-label scores.

    Only the `lengths[b]` valid positions of each sample are encoded (padded
    positions are skipped, not just masked), so compute scales with the real
    solution length. A learned ply embedding tells the pooler whether a
    position is the start, after our move, or after the opponent's reply.
    With T = 1 this is exactly the position-only baseline.
    """

    def __init__(
        self,
        num_labels: int,
        in_planes: int = 18,
        hidden_dim: int = 256,
        num_res_blocks: int = 6,
        max_positions: int = 12,
        pooling: str = "mean",
        dropout: float = 0.3,
        head: str = "flat",
        labels: list[str] | None = None,
    ):
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError(f"pooling must be 'mean' or 'attention', got {pooling!r}")
        self.encoder = ResNetBoardEncoder(in_planes, hidden_dim, num_res_blocks)
        self.ply_embedding = nn.Embedding(max_positions, hidden_dim)
        self.pooling = pooling
        if pooling == "attention":
            self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head_type = head
        if head == "flat":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        elif head == "grouped":
            if labels is None or len(labels) != num_labels:
                raise ValueError("head='grouped' needs the label list")
            self.head = GroupedHead(hidden_dim, LabelLayout(labels), dropout)
        else:
            raise ValueError(f"head must be 'flat' or 'grouped', got {head!r}")

    def probs(self, raw: torch.Tensor) -> torch.Tensor:
        """Per-label probabilities (B, L) from the raw head output."""
        return self.head.probs(raw) if self.head_type == "grouped" else torch.sigmoid(raw)

    def forward(self, boards: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T = boards.shape[:2]
        mask = torch.arange(T, device=boards.device)[None, :] < lengths[:, None]  # (B, T)
        flat_mask = mask.reshape(-1)

        valid_feats = self.encoder(boards.flatten(0, 1)[flat_mask])  # (sum(lengths), D)
        feats = valid_feats.new_zeros(B * T, valid_feats.shape[1])
        feats[flat_mask] = valid_feats
        feats = feats.view(B, T, -1) + self.ply_embedding.weight[:T][None]

        if self.pooling == "mean":
            pooled = (feats * mask[..., None]).sum(1) / lengths.clamp(min=1)[:, None].to(feats.dtype)
        else:
            scores = self.attn(feats).squeeze(-1).masked_fill(~mask, float("-inf"))
            pooled = (torch.softmax(scores, dim=1)[..., None] * feats).sum(1)
        return self.head(pooled)
