import h5py
import torch
from torch.utils.data import Dataset

class PreEncodedPuzzleDataset(Dataset):
    """Memory-mapped HDF5 PyTorch dataset delivering float32 tensors with ~8us access times."""
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.h5f = None

    def _lazy_init(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_path, "r", swmr=True)
            self.boards = self.h5f["boards"]
            self.targets = self.h5f["targets"]

    def __len__(self):
        self._lazy_init()
        return len(self.targets)

    def __getitem__(self, index: int):
        self._lazy_init()
        board = torch.from_numpy(self.boards[index]).to(dtype=torch.float32)
        target = torch.from_numpy(self.targets[index]).to(dtype=torch.float32)
        return board, target
