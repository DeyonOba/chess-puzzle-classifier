import torch
import torch.nn as nn


class BoardResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class ResNetBoardEncoder(nn.Module):
    """Encodes a batch of (in_planes, 8, 8) boards into (hidden_dim,) feature vectors.

    Unchanged from the position-only baseline: 3x3 stem, `num_res_blocks`
    residual blocks, global average pool.
    """

    def __init__(self, in_planes: int = 18, hidden_dim: int = 256, num_res_blocks: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[BoardResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.res_blocks(self.stem(x))
        return self.pool(feat).flatten(1)
