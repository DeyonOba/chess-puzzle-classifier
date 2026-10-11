import csv
import json
import pathlib
import chess
import h5py
import numpy as np
from tqdm import tqdm
from src.utils.bitboard_encoder import board_to_tensor_fast

def build_hdf5_dataset(csv_path: str, hdf5_out_path: str, meta_out_path: str):
    """Parses raw Lichess CSV and generates compressed uint8 HDF5 boards and target tensors."""
    theme_counts = {}
    samples = []

    # First pass: Metadata scan and theme mapping
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) > 7:
                fen, moves, themes_str = row[1], row[2], row[7]
                themes = themes_str.strip().split(" ")
                for t in themes:
                    theme_counts[t] = theme_counts.get(t, 0) + 1
                samples.append((fen, moves, themes_str))

    sorted_themes = sorted(theme_counts.keys())
    map_themes = {t: idx for idx, t in enumerate(sorted_themes)}
    num_samples = len(samples)

    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump({"theme_map": map_themes, "counts": theme_counts}, f, indent=2)

    # Second pass: Build binary HDF5 dataset
    num_classes = len(map_themes)
    with h5py.File(hdf5_out_path, "w") as h5f:
        boards_dset = h5f.create_dataset(
            "boards", shape=(num_samples, 18, 8, 8), dtype="uint8", chunks=(1024, 18, 8, 8), compression="lzf"
        )
        targets_dset = h5f.create_dataset(
            "targets", shape=(num_samples, num_classes), dtype="uint8", chunks=(1024, num_classes)
        )

        batch_b, batch_t = [], []
        for i, (fen, moves, themes_str) in enumerate(tqdm(samples, desc="Encoding HDF5")):
            board = chess.Board(fen)
            for move in moves.strip().split(" ")[:2]:
                board.push_uci(move)
            
            b_tensor = board_to_tensor_fast(board).numpy()
            t_vector = np.zeros(num_classes, dtype=np.uint8)
            for t in themes_str.strip().split(" "):
                if t in map_themes:
                    t_vector[map_themes[t]] = 1

            batch_b.append(b_tensor)
            batch_t.append(t_vector)

            if len(batch_b) == 10000 or i == num_samples - 1:
                idx_start = i - len(batch_b) + 1
                boards_dset[idx_start : i + 1] = np.stack(batch_b)
                targets_dset[idx_start : i + 1] = np.stack(batch_t)
                batch_b.clear()
                batch_t.clear()
