"""PyTorch Dataset over the HDF5 produced by src.data.preencode.

Random single-row access into an lzf-compressed, chunked HDF5 is slow (each
read decompresses a whole chunk), so this dataset implements `__getitems__`
and reads a batch of *sorted* indices as one contiguous slab. Pair it with
`ContiguousBatchSampler` (src.data.datamodule) so each batch is a run of
neighbouring rows; the CSV is ordered by random puzzle id, so neighbouring
rows are not correlated in content.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.bitboard_encoder import unpack_bitboards


class PuzzleSequenceDataset(Dataset):
    """Yields (boards, lengths, targets) batches from a packed-bitboard HDF5 file.

    boards:  float32 (B, T, 18, 8, 8), zero-padded beyond each sample's length
    lengths: int64   (B,)  valid positions per sample (clamped to T)
    targets: float32 (B, L) multi-hot themes

    `max_positions=1` gives the position-only baseline (start position only).
    """

    def __init__(self, hdf5_path: str, max_positions: int | None = None):
        self.hdf5_path = str(hdf5_path)
        self._h5 = None
        with h5py.File(self.hdf5_path, "r") as f:
            self.num_samples = f["targets"].shape[0]
            self.num_labels = f["targets"].shape[1]
            self.file_max_positions = int(f.attrs["max_positions"])
            self.labels = json.loads(f.attrs["labels"])
        self.max_positions = self.file_max_positions if max_positions is None else int(max_positions)
        if not 1 <= self.max_positions <= self.file_max_positions:
            raise ValueError(f"max_positions must be in [1, {self.file_max_positions}], got {self.max_positions}")

    # h5py handles cannot be pickled, so each DataLoader worker opens its own.
    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        boards, lengths, targets = self.__getitems__([index])
        return boards[0], lengths[0], targets[0]

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(idx, kind="stable")
        idx_sorted = idx[order]
        f = self._file()
        lo, hi = int(idx_sorted[0]), int(idx_sorted[-1]) + 1
        rel = idx_sorted - lo
        if hi - lo <= 4 * len(idx):  # dense enough: one slab read, then subselect
            boards = f["boards"][lo:hi, : self.max_positions][rel]
            lengths = f["lengths"][lo:hi][rel]
            targets = f["targets"][lo:hi][rel]
        else:  # scattered indices: h5py fancy indexing requires increasing order
            boards = f["boards"][idx_sorted, : self.max_positions]
            lengths = f["lengths"][idx_sorted]
            targets = f["targets"][idx_sorted]
        # restore the caller's order
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        boards, lengths, targets = boards[inv], lengths[inv], targets[inv]

        planes = unpack_bitboards(boards)  # (B, T, 18, 8, 8) uint8
        return (
            torch.from_numpy(planes).to(torch.float32),
            torch.from_numpy(np.minimum(lengths, self.max_positions).astype(np.int64)),
            torch.from_numpy(targets).to(torch.float32),
        )

    def read_targets(self) -> np.ndarray:
        """All targets as (N, L) uint8, for split construction and class weights."""
        with h5py.File(self.hdf5_path, "r") as f:
            return f["targets"][:]

    def read_column(self, name: str) -> np.ndarray:
        with h5py.File(self.hdf5_path, "r") as f:
            return f[name][:]


def identity_collate(batch):
    """Collate for DataLoader when the dataset already returns a collated batch via __getitems__."""
    return batch
