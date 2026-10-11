import torch
import torch.nn as nn

from src.models.backbones.resnet_board import ResNetBoardEncoder


class PuzzleSequenceClassifier(nn.Module):
    """Shared ResNet over every position of the solution line, masked pooling, sigmoid head.

    forward(boards (B, T, 18, 8, 8), lengths (B,)) -> logits (B, num_labels)

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
    ):
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError(f"pooling must be 'mean' or 'attention', got {pooling!r}")
        self.encoder = ResNetBoardEncoder(in_planes, hidden_dim, num_res_blocks)
        self.ply_embedding = nn.Embedding(max_positions, hidden_dim)
        self.pooling = pooling
        if pooling == "attention":
            self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_labels),
        )

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
