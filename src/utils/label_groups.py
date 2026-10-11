"""Mutually exclusive theme groups, as produced by cook.py's if/elif chains.

Verified on the full CSV (2026-10-11): fewer than 25 puzzles in 6.16M carry two
members of any group. Groups get a softmax with an extra "none" class; every
other label is an independent sigmoid.
"""

from __future__ import annotations

import torch

EXCLUSIVE_GROUPS: dict[str, list[str]] = {
    "outcome": ["mate", "crushing", "advantage", "equality"],
    "length": ["oneMove", "short", "long", "veryLong"],
    "phase": ["opening", "middlegame", "endgame"],
    "mateIn": ["mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5"],
    "matePattern": [
        "backRankMate", "smotheredMate", "anastasiaMate", "hookMate", "arabianMate", "bodenMate",
        "doubleBishopMate", "dovetailMate", "pillsburysMate", "operaMate", "epauletteMate", "cornerMate",
        "swallowstailMate", "triangleMate", "morphysMate", "blindSwineMate", "killBoxMate", "vukovicMate",
        "balestraMate",
    ],
    "endgameType": ["pawnEndgame", "queenEndgame", "rookEndgame", "bishopEndgame", "knightEndgame", "queenRookEndgame"],
    "attackSide": ["kingsideAttack", "queensideAttack"],
}


class LabelLayout:
    """Index bookkeeping between the flat label list and the grouped head."""

    def __init__(self, labels: list[str], groups: dict[str, list[str]] = EXCLUSIVE_GROUPS):
        index = {l: i for i, l in enumerate(labels)}
        self.labels = labels
        self.groups: list[tuple[str, list[int]]] = []
        grouped = set()
        for name, members in groups.items():
            idx = [index[m] for m in members if m in index]
            if idx:
                self.groups.append((name, idx))
                grouped.update(idx)
        self.free = [i for i in range(len(labels)) if i not in grouped]
        self.num_labels = len(labels)
        self.num_groups = len(self.groups)

    def group_targets(self, targets: torch.Tensor) -> list[torch.Tensor]:
        """Per group, class index in [0, k] where k (the last) is 'none'. targets: (B, L) 0/1."""
        out = []
        for _, idx in self.groups:
            sub = targets[:, idx]
            has = sub.sum(1) > 0
            cls = torch.where(has, sub.argmax(1), torch.full_like(has, len(idx), dtype=torch.long))
            out.append(cls)
        return out
