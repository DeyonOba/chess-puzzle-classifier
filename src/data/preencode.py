"""Offline conversion of the Lichess puzzle CSV into an HDF5 file of packed bitboard sequences.

Usage:
    python -m src.data.preencode --csv data/lichess_db_puzzle.csv --out data/puzzle_boards.h5 \
        --meta data/puzzle_metadata.json [--max-positions 12] [--limit 100000] [--workers 8]

HDF5 layout (N = number of puzzles, T = max_positions, L = number of labels):
    boards     (N, T, 18) uint64   packed bitboards, position 0 = board after setup move
    lengths    (N,)       uint8    number of valid positions (<= T)
    targets    (N, L)     uint8    multi-hot theme vector
    plies      (N,)       uint8    full solution length in plies (before truncation)
    rating     (N,)       int16    puzzle rating
    puzzle_id  (N,)       S8       Lichess puzzle id (ascii)

Metadata JSON: ordered label list, per-label counts, excluded themes, T, N, csv path.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.bitboard_encoder import NUM_PLANES, encode_solution_line

# Themes derived from the source game's player ratings, not from the board.
# They are unlearnable from the position/solution and are excluded by default.
DEFAULT_EXCLUDED_THEMES = ("master", "masterVsMaster", "superGM")

CSV_COLUMNS = ["PuzzleId", "FEN", "Moves", "Rating", "Themes"]
CHUNK_ROWS = 50_000


def build_vocabulary(csv_path: str, excluded: tuple[str, ...], limit: int | None) -> tuple[list[str], Counter, int]:
    """First pass: count themes and rows. Returns (sorted labels, counts, num_rows)."""
    counts: Counter = Counter()
    num_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=["Themes"], chunksize=500_000, nrows=limit):
        for themes in chunk["Themes"].fillna(""):
            counts.update(themes.split())
        num_rows += len(chunk)
    labels = sorted(t for t in counts if t not in excluded)
    return labels, counts, num_rows


_worker_state: dict = {}


def _init_worker(label_index: dict[str, int], max_positions: int):
    _worker_state["label_index"] = label_index
    _worker_state["max_positions"] = max_positions


def _encode_chunk(rows: list[tuple[str, str, str, int, str]]):
    """Encode a list of (puzzle_id, fen, moves, rating, themes) rows."""
    label_index = _worker_state["label_index"]
    T = _worker_state["max_positions"]
    n = len(rows)
    boards = np.zeros((n, T, NUM_PLANES), dtype=np.uint64)
    lengths = np.zeros(n, dtype=np.uint8)
    targets = np.zeros((n, len(label_index)), dtype=np.uint8)
    plies = np.zeros(n, dtype=np.uint8)
    ratings = np.zeros(n, dtype=np.int16)
    ids = np.zeros(n, dtype="S8")
    for i, (pid, fen, moves, rating, themes) in enumerate(rows):
        packed, length = encode_solution_line(fen, moves, T)
        boards[i] = packed
        lengths[i] = length
        plies[i] = min(len(moves.split()) - 1, 255)
        ratings[i] = rating
        ids[i] = pid.encode("ascii")
        for t in themes.split():
            j = label_index.get(t)
            if j is not None:
                targets[i, j] = 1
    return boards, lengths, targets, plies, ratings, ids


def build_hdf5_dataset(
    csv_path: str,
    hdf5_out_path: str,
    meta_out_path: str,
    max_positions: int = 12,
    excluded_themes: tuple[str, ...] = DEFAULT_EXCLUDED_THEMES,
    limit: int | None = None,
    workers: int | None = None,
) -> dict:
    labels, counts, num_rows = build_vocabulary(csv_path, excluded_themes, limit)
    label_index = {t: i for i, t in enumerate(labels)}
    L = len(labels)

    meta = {
        "labels": labels,
        "counts": {t: counts[t] for t in labels},
        "excluded_themes": {t: counts.get(t, 0) for t in excluded_themes},
        "max_positions": max_positions,
        "num_samples": num_rows,
        "num_labels": L,
        "source_csv": str(pathlib.Path(csv_path).resolve()),
    }

    with h5py.File(hdf5_out_path, "w") as h5f, Pool(workers, _init_worker, (label_index, max_positions)) as pool:
        d_boards = h5f.create_dataset(
            "boards", shape=(num_rows, max_positions, NUM_PLANES), dtype=np.uint64,
            chunks=(4096, max_positions, NUM_PLANES), compression="lzf",
        )
        d_lengths = h5f.create_dataset("lengths", shape=(num_rows,), dtype=np.uint8)
        d_targets = h5f.create_dataset("targets", shape=(num_rows, L), dtype=np.uint8, chunks=(65536, L))
        d_plies = h5f.create_dataset("plies", shape=(num_rows,), dtype=np.uint8)
        d_rating = h5f.create_dataset("rating", shape=(num_rows,), dtype=np.int16)
        d_ids = h5f.create_dataset("puzzle_id", shape=(num_rows,), dtype="S8")
        h5f.attrs["labels"] = json.dumps(labels)
        h5f.attrs["max_positions"] = max_positions

        def row_batches():
            reader = pd.read_csv(csv_path, usecols=CSV_COLUMNS, chunksize=CHUNK_ROWS, nrows=limit, dtype={"Themes": str})
            for chunk in reader:
                chunk["Themes"] = chunk["Themes"].fillna("")
                yield list(chunk[CSV_COLUMNS].itertuples(index=False, name=None))

        offset = 0
        with tqdm(total=num_rows, desc="Encoding", unit="puzzle") as bar:
            for boards, lengths, targets, plies, ratings, ids in pool.imap(_encode_chunk, row_batches()):
                n = len(lengths)
                sl = slice(offset, offset + n)
                d_boards[sl] = boards
                d_lengths[sl] = lengths
                d_targets[sl] = targets
                d_plies[sl] = plies
                d_rating[sl] = ratings
                d_ids[sl] = ids
                offset += n
                bar.update(n)
        assert offset == num_rows, f"wrote {offset} rows, expected {num_rows}"

    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/lichess_db_puzzle.csv")
    p.add_argument("--out", default="data/puzzle_boards.h5")
    p.add_argument("--meta", default="data/puzzle_metadata.json")
    p.add_argument("--max-positions", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="encode only the first N rows (dev subsets)")
    p.add_argument("--workers", type=int, default=None, help="process pool size (default: all cores)")
    p.add_argument("--keep-rating-themes", action="store_true",
                   help="keep master/masterVsMaster/superGM in the vocabulary")
    args = p.parse_args()
    excluded = () if args.keep_rating_themes else DEFAULT_EXCLUDED_THEMES
    meta = build_hdf5_dataset(args.csv, args.out, args.meta, args.max_positions, excluded, args.limit, args.workers)
    print(f"wrote {meta['num_samples']:,} puzzles, {meta['num_labels']} labels, T={meta['max_positions']} -> {args.out}")


if __name__ == "__main__":
    main()
