"""Splits and loaders for the packed-bitboard HDF5 dataset.

Split construction (run once per HDF5 file):
    python -m src.data.components.splits --h5 data/puzzle_boards.h5 --out data/splits.npz \
        [--fractions 0.9 0.05 0.05] [--seed 0]

Produces an .npz with sorted int64 index arrays `train`, `val`, `test`, plus
`fractions`, `seed` and `num_samples`. Splits live outside the HDF5 so you can
re-split without re-encoding.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from src.data.components.dataset import PuzzleSequenceDataset, identity_collate

SPLIT_NAMES = ("train", "val", "test")


def iterative_stratified_split(
    targets: np.ndarray, fractions: tuple[float, ...] = (0.9, 0.05, 0.05), seed: int = 0
) -> np.ndarray:
    """Multi-label stratified assignment of each row to a split (Sechidis et al., 2011).

    Labels are processed from the one with the fewest *unassigned* positives
    upward. For that label, its unassigned rows are shuffled and distributed
    across splits in proportion to each split's remaining desired count for
    the label. This keeps rare labels at their target ratio in every split;
    common labels (which co-occur with everything) follow automatically.

    targets:   (N, L) 0/1 array
    returns:   (N,) int8 split id in range(len(fractions))
    """
    rng = np.random.default_rng(seed)
    fractions = np.asarray(fractions, dtype=np.float64)
    assert np.isclose(fractions.sum(), 1.0), "fractions must sum to 1"
    N, L = targets.shape
    S = len(fractions)
    Y = targets.astype(bool)

    assigned = np.full(N, -1, dtype=np.int8)
    remaining_positives = Y.sum(0).astype(np.int64)  # per label, among unassigned rows
    desired = remaining_positives[:, None] * fractions[None, :]  # (L, S) remaining target per split

    def distribute(rows: np.ndarray, weights: np.ndarray):
        """Shuffle `rows` and assign them to splits proportionally to `weights`."""
        rows = rng.permutation(rows)
        w = np.clip(weights, 0, None)
        w = w / w.sum() if w.sum() > 0 else fractions
        quota = w * len(rows)
        counts = np.floor(quota).astype(np.int64)
        # largest-remainder rounding, so a label with a handful of positives still reaches val/test
        for s in np.argsort(-(quota - counts))[: len(rows) - counts.sum()]:
            counts[s] += 1
        bounds = np.cumsum(counts)
        for s, (start, stop) in enumerate(zip(np.r_[0, bounds[:-1]], bounds)):
            chosen = rows[start:stop]
            if len(chosen) == 0:
                continue
            assigned[chosen] = s
            label_hits = Y[chosen].sum(0)
            desired[:, s] -= label_hits
            remaining_positives[:] -= label_hits

    while True:
        live = np.flatnonzero(remaining_positives > 0)
        if live.size == 0:
            break
        label = live[np.argmin(remaining_positives[live])]
        rows = np.flatnonzero((assigned < 0) & Y[:, label])
        distribute(rows, desired[label])

    leftovers = np.flatnonzero(assigned < 0)  # rows with no labels at all
    if leftovers.size:
        distribute(leftovers, fractions * leftovers.size)
    return assigned


def split_report(targets: np.ndarray, assigned: np.ndarray, fractions, labels: list[str] | None = None) -> dict:
    """Per-split label fractions vs. target; returns the worst relative deviation."""
    worst = {"label": None, "split": None, "actual": None, "expected": None, "rel_dev": 0.0}
    total = targets.sum(0)
    for s, frac in enumerate(fractions):
        got = targets[assigned == s].sum(0)
        expected = total * frac
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(expected > 0, np.abs(got - expected) / np.maximum(expected, 1), 0.0)
        j = int(np.argmax(rel))
        if rel[j] > worst["rel_dev"]:
            worst = {
                "label": labels[j] if labels else j, "split": SPLIT_NAMES[s] if s < 3 else s,
                "actual": int(got[j]), "expected": float(expected[j]), "rel_dev": float(rel[j]),
            }
    return worst


def build_splits(h5_path: str, out_path: str, fractions=(0.9, 0.05, 0.05), seed: int = 0) -> dict:
    ds = PuzzleSequenceDataset(h5_path)
    targets = ds.read_targets()
    assigned = iterative_stratified_split(targets, fractions, seed)
    arrays = {name: np.flatnonzero(assigned == s).astype(np.int64) for s, name in enumerate(SPLIT_NAMES)}
    np.savez(out_path, fractions=np.asarray(fractions), seed=seed, num_samples=len(assigned), **arrays)
    worst = split_report(targets, assigned, fractions, ds.labels)
    return {"sizes": {k: int(v.size) for k, v in arrays.items()}, "worst_label_deviation": worst}


def load_splits(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {name: z[name] for name in SPLIT_NAMES}


class ContiguousBatchSampler(Sampler[list[int]]):
    """Batches are consecutive runs of a sorted index array; batch order is shuffled per epoch.

    Consecutive HDF5 rows read as one slab, which is what makes the lzf-chunked
    file fast. Row order in the file is by random puzzle id, so a contiguous run
    is content-random. Call set_epoch(e) before each epoch for a fresh order.
    """

    def __init__(self, indices: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        self.indices = np.sort(np.asarray(indices, dtype=np.int64))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        n_full, rem = divmod(len(self.indices), batch_size)
        self.num_batches = n_full + (1 if rem and not drop_last else 0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        order = np.arange(self.num_batches)
        if self.shuffle:
            order = np.random.default_rng(self.seed + self.epoch).permutation(order)
        for b in order:
            yield self.indices[b * self.batch_size : (b + 1) * self.batch_size].tolist()


def make_loader(
    dataset: PuzzleSequenceDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    sampler = ContiguousBatchSampler(indices, batch_size, shuffle, seed, drop_last)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=identity_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", default="data/puzzle_boards.h5")
    p.add_argument("--out", default="data/splits.npz")
    p.add_argument("--fractions", type=float, nargs=3, default=(0.9, 0.05, 0.05), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    report = build_splits(args.h5, args.out, tuple(args.fractions), args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
