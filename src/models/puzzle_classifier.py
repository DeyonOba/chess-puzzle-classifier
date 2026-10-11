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


class ChessPuzzleClassifier(nn.Module):
    def __init__(self, in_channels: int = 18, num_classes: int = 70, hidden_dim: int = 256, num_res_blocks: int = 6):
        super().__init__()
        
        # 1. Feature Extractor Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[BoardResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        
        # 2. Pooled Board Feature Vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Independent Sigmoid Head for Multi-Label Classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.res_blocks(feat)
        feat = self.pool(feat).flatten(1)
        logits = self.classifier(feat)
        return logits
