"""LightningDataModule over the packed-bitboard HDF5 and the stratified splits.

Keeps the custom pieces: PuzzleSequenceDataset (batched slab reads via
__getitems__) and ContiguousBatchSampler (consecutive-row batches, order
reshuffled per epoch). Lightning does not call set_epoch on a batch sampler,
so PuzzleLitModule does it in on_train_epoch_start.
"""

from __future__ import annotations

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

from src.data.components.dataset import PuzzleSequenceDataset
from src.data.components.splits import load_splits, make_loader


class PuzzleDataModule(L.LightningDataModule):
    def __init__(
        self,
        h5_path: str,
        splits_path: str,
        max_positions: int | None = None,
        batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 0,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset: PuzzleSequenceDataset | None = None
        self.splits: dict[str, np.ndarray] | None = None

    def setup(self, stage: str | None = None):
        if self.dataset is None:
            self.dataset = PuzzleSequenceDataset(self.hparams.h5_path, self.hparams.max_positions)
            self.splits = load_splits(self.hparams.splits_path)

    # -- metadata the model needs --------------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        self.setup()
        return self.dataset.labels

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def num_planes(self) -> int:
        self.setup()
        return self.dataset.num_planes

    @property
    def max_positions(self) -> int:
        self.setup()
        return self.dataset.max_positions

    def train_targets(self) -> np.ndarray:
        """(n_train, L) uint8 targets, for pos_weight / group class weights / ML-GCN adjacency."""
        self.setup()
        return self.dataset.read_targets()[self.splits["train"]]

    # -- loaders --------------------------------------------------------------------------------
    def _loader(self, split: str, shuffle: bool, drop_last: bool = False) -> DataLoader:
        self.setup()
        return make_loader(self.dataset, self.splits[split], self.hparams.batch_size, shuffle,
                           self.hparams.num_workers, self.hparams.seed, drop_last)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True, drop_last=self.hparams.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
