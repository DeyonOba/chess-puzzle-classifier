import torch
import torch.nn as nn

from src.models.backbones.resnet_board import ResNetBoardEncoder
from src.models.heads import GroupedReadoutMixin, MLGCNHead, Query2LabelHead
from src.utils.label_groups import LabelLayout


class GroupedHead(GroupedReadoutMixin, nn.Module):
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
        head_kwargs: dict | None = None,
    ):
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError(f"pooling must be 'mean' or 'attention', got {pooling!r}")
        self.encoder = ResNetBoardEncoder(in_planes, hidden_dim, num_res_blocks)
        self.ply_embedding = nn.Embedding(max_positions, hidden_dim) if not head.startswith("q2l") else None  # q2l adds its own
        self.pooling = pooling
        if pooling == "attention":
            self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head_type = head
        # control for the pooling hypothesis: keep square identity (flatten the 8x8 map, project
        # to hidden_dim per position) but use the ordinary pooled-vector MLP head
        self.spatial_proj = nn.Linear(hidden_dim * 64, hidden_dim) if head == "flat_spatial" else None
        if head in ("flat", "flat_spatial"):
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
        elif head == "q2l":
            self.head = Query2LabelHead(num_labels, hidden_dim, max_positions, **(head_kwargs or {}))
        elif head == "q2l_grouped":
            if labels is None or len(labels) != num_labels:
                raise ValueError("head='q2l_grouped' needs the label list")
            self.head = Query2LabelHead(num_labels, hidden_dim, max_positions, layout=LabelLayout(labels), **(head_kwargs or {}))
        elif head == "mlgcn":
            self.head = MLGCNHead(num_labels, hidden_dim, **(head_kwargs or {}))
        else:
            raise ValueError(f"head must be one of flat/flat_spatial/grouped/q2l/q2l_grouped/mlgcn, got {head!r}")

    def probs(self, raw: torch.Tensor) -> torch.Tensor:
        """Per-label probabilities (B, L) from the raw head output."""
        return self.head.probs(raw) if self.grouped else torch.sigmoid(raw)

    @property
    def grouped(self) -> bool:
        return getattr(self.head, "layout", None) is not None

    def forward(self, boards: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T = boards.shape[:2]
        mask = torch.arange(T, device=boards.device)[None, :] < lengths[:, None]  # (B, T)
        flat_mask = mask.reshape(-1)

        if self.head_type.startswith("q2l"):  # needs the spatial map of every valid position
            valid_spatial = self.encoder.spatial(boards.flatten(0, 1)[flat_mask])  # (sum(lengths), D, 8, 8)
            spatial = valid_spatial.new_zeros(B * T, *valid_spatial.shape[1:])
            spatial[flat_mask] = valid_spatial
            return self.head(spatial.view(B, T, *valid_spatial.shape[1:]), mask)

        if self.head_type == "flat_spatial":
            valid_feats = self.spatial_proj(self.encoder.spatial(boards.flatten(0, 1)[flat_mask]).flatten(1))
        else:
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
