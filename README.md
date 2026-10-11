# chess-puzzle-classifier

Multi-label classification of Lichess puzzle themes from the board and the
solution line, using python-chess for encoding and PyTorch for training.

## Pipeline

```bash
uv sync                                                        # deps (python-chess, torch, h5py, ...)
scripts/fetch_puzzles.sh && scripts/decompress_zstd.sh         # data/lichess_db_puzzle.csv
uv run python -m src.data.preencode --csv data/lichess_db_puzzle.csv \
    --out data/puzzle_boards.h5 --meta data/puzzle_metadata.json   # ~6 min, ~1.4 GB
uv run python -m src.data.datamodule --h5 data/puzzle_boards.h5 --out data/splits.npz
uv run python -m src.train --h5 data/puzzle_boards.h5 --splits data/splits.npz --run-name line_t12
uv run python -m src.train --h5 data/puzzle_boards.h5 --splits data/splits.npz --run-name pos_only --max-positions 1
uv run pytest
```

Add `--limit 100000` to preencode for a dev subset (see `data/dev/`).

## Design decisions (2026-10-11)

- **Input**: the start position plus the board after every solution ply,
  `(T, 18, 8, 8)`, T_max = 12 (covers ~99.5% of puzzles; longer lines are
  truncated). `--max-positions 1` gives the position-only baseline.
- **Storage**: 18 packed uint64 bitboards per position in HDF5 (8x smaller
  than uint8 planes), unpacked in the DataLoader.
- **Labels**: 70 themes. `master`, `masterVsMaster`, `superGM` are excluded
  because they derive from player ratings, not the board.
- **Split**: iterative multi-label stratification, 90/5/5, stored in
  `data/splits.npz` separately from the encoded boards.
- **Model**: one ResNet encodes every position with shared weights, a learned
  ply embedding is added, masked mean (or attention) pooling, MLP sigmoid head.
- **Loss**: BCE-with-logits with per-label `pos_weight = clip(sqrt(neg/pos), 1, 20)`
  (`--loss weighted_bce --pos-weight-power 0.5`), chosen over plain BCE, focal
  and ASL in `experiments/2026-10-11_loss_comparison.md`. Per-class decision
  thresholds are tuned on validation (`src/evaluate.py`). A grouped
  (softmax-per-exclusive-group) head is a planned experiment.

## Project structure

```
chess-puzzle-classifier/
├── configs/                      # (reserved) experiment configs
├── data/                         # gitignored
│   ├── lichess_db_puzzle.csv     # raw Lichess dump
│   ├── puzzle_boards.h5          # packed boards, lengths, targets, rating, puzzle_id
│   ├── puzzle_metadata.json      # label list, counts, exclusions, T, N
│   ├── splits.npz                # train/val/test index arrays
│   └── dev/                      # 100k-row subset of the above for smoke tests
├── runs/                         # gitignored training outputs (config, history, checkpoints)
├── scripts/
│   ├── fetch_puzzles.sh          # download the Lichess puzzle database
│   └── decompress_zstd.sh        # .zst -> .csv
├── src/
│   ├── data/
│   │   ├── preencode.py          # CSV -> HDF5 (multiprocess, streaming)
│   │   ├── dataset.py            # PuzzleSequenceDataset (batched slab reads, bit unpacking)
│   │   └── datamodule.py         # stratified splits, ContiguousBatchSampler, make_loader
│   ├── models/
│   │   ├── backbones/resnet_board.py   # ResNetBoardEncoder (shared per-position encoder)
│   │   └── puzzle_classifier.py        # PuzzleSequenceClassifier (masked pooling + head)
│   ├── utils/
│   │   ├── bitboard_encoder.py   # packed bitboards, unpacking, solution-line encoding
│   │   ├── metrics.py            # micro/macro F1, P/R, exact match, PR-AUC, per-class F1
│   │   └── puzzle_theme_distribution.py
│   ├── train.py                  # training entrypoint
│   ├── inference.py              # (placeholder)
│   └── app.py                    # (placeholder)
├── tests/                        # pytest: encoder, dataset, model, splits
├── pyproject.toml / uv.lock
└── README.md
```

Theme semantics follow `lichess-puzzler/tagger/cook.py`; most tags are
functions of the solution line, which is why the line is part of the input.
