"""Label-aware classification heads: Query2Label and ML-GCN.

Both consume the shared ResNet encoder; they differ in what they attend to.

Query2Label (Liu et al. 2021): one learnable query per label cross-attends over
spatial tokens (here: every square of every valid position in the solution
line), then a per-label linear readout gives the logit.

ML-GCN (Chen et al. 2019): a 2-layer GCN over label embeddings, propagated
along a label co-occurrence graph, produces one classifier vector per label;
logits are dot products with the pooled board feature. The paper feeds GloVe
word vectors as label inputs; puzzle tags have no word vectors, so label
embeddings are initialised from each label's conditional co-occurrence row.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Query2LabelHead(nn.Module):
    def __init__(self, num_labels: int, hidden_dim: int, max_positions: int, num_layers: int = 1,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.queries = nn.Parameter(torch.randn(num_labels, hidden_dim) * 0.02)
        self.square_embedding = nn.Parameter(torch.randn(64, hidden_dim) * 0.02)
        self.ply_embedding = nn.Embedding(max_positions, hidden_dim)
        layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, dim_feedforward=2 * hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.readout_w = nn.Parameter(torch.randn(num_labels, hidden_dim) * 0.02)
        self.readout_b = nn.Parameter(torch.zeros(num_labels))

    def forward(self, spatial: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """spatial: (B, T, D, 8, 8) features (zeros where padded); mask: (B, T) bool valid."""
        B, T, D = spatial.shape[:3]
        tokens = spatial.flatten(3).transpose(2, 3)  # (B, T, 64, D)
        tokens = tokens + self.square_embedding[None, None] + self.ply_embedding.weight[:T][None, :, None]
        tokens = tokens.reshape(B, T * 64, D)
        key_padding = ~mask[:, :, None].expand(B, T, 64).reshape(B, T * 64)
        q = self.queries[None].expand(B, -1, -1)
        h = self.decoder(q, tokens, memory_key_padding_mask=key_padding)  # (B, L, D)
        return (h * self.readout_w[None]).sum(-1) + self.readout_b


def build_adjacency(train_targets: np.ndarray, tau: float = 0.4, p: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """ML-GCN adjacency from training co-occurrence.

    Returns (A_hat, cond) where cond[i, j] = P(label j | label i) and A_hat is
    the binarised (>= tau), re-weighted (off-diagonal mass p, self-loop 1-p),
    symmetrically normalised propagation matrix.
    """
    Y = train_targets.astype(np.float32)
    co = np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
    for s in range(0, len(Y), 500_000):
        c = Y[s : s + 500_000]
        co += c.T @ c
    counts = np.diag(co).copy()
    cond = co / np.maximum(counts[:, None], 1)
    A = (cond >= tau).astype(np.float64)
    np.fill_diagonal(A, 0)
    row = A.sum(1, keepdims=True)
    A = np.where(row > 0, A * p / np.maximum(row, 1), 0.0)
    A = A + (1 - p) * np.eye(len(A))
    d = A.sum(1)
    D_inv_sqrt = np.diag(1 / np.sqrt(d))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32), cond.astype(np.float32)


class MLGCNHead(nn.Module):
    def __init__(self, num_labels: int, hidden_dim: int, gcn_dim: int = 256,
                 adjacency: np.ndarray | None = None, label_init: np.ndarray | None = None):
        super().__init__()
        self.register_buffer("adjacency", torch.zeros(num_labels, num_labels) if adjacency is None else torch.as_tensor(adjacency))
        init = torch.eye(num_labels) if label_init is None else torch.as_tensor(label_init)
        self.label_embedding = nn.Parameter(init.clone().float())  # (L, L): co-occurrence rows
        self.gc1 = nn.Linear(num_labels, gcn_dim, bias=False)
        self.gc2 = nn.Linear(gcn_dim, hidden_dim, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def classifier_weights(self) -> torch.Tensor:
        h = self.act(self.adjacency @ self.gc1(self.label_embedding))
        return self.adjacency @ self.gc2(h)  # (L, D)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled @ self.classifier_weights().T + self.bias
