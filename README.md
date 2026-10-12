# chess-puzzle-classifier

Multi-label classification of Lichess puzzle themes from the board and the
solution line, using python-chess for encoding and PyTorch for training.

## Pipeline

```bash
uv sync                                                        # deps (python-chess, torch, lightning, hydra, ...)
scripts/fetch_puzzles.sh && scripts/decompress_zstd.sh         # data/lichess_db_puzzle.csv
uv run python -m src.data.preencode --csv data/lichess_db_puzzle.csv \
    --out data/puzzle_boards.h5 --meta data/puzzle_metadata.json   # ~6 min, ~1.3 GB
uv run python -m src.data.components.splits --h5 data/puzzle_boards.h5 --out data/splits.npz
uv run python -m src.train run_name=q2lg_s0                    # defaults: q2l_grouped, gw 0.5, weighted BCE
uv run python -m src.eval ckpt_path=logs/train/runs/q2lg_s0/checkpoints/best.ckpt
uv run python -m src.analyze logs/train/runs/q2lg_s0 --eval-results logs/eval/runs/<ts>/eval_results.json
uv run pytest
```

Training and evaluation are Hydra apps (layout after
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)).
Every run writes its resolved config, CSV metrics, `checkpoints/{best,last}.ckpt`
and `test_results.json` to `logs/<task>/runs/<run_name>/`.

```bash
uv run python -m src.train debug=default                       # 5-batch smoke run on data/dev
uv run python -m src.train experiment=bounded_laptop run_name=x   # the 384k-puzzle protocol used in experiments/
uv run python -m src.train experiment=full_gpu seed=1 run_name=q2lg_full_s1
uv run python -m src.train model/net=q2l model.loss=asl data.max_positions=1   # any override
uv run python -m src.train -m model.group_weight=0.25,0.5,1 run_name=gw_sweep  # sweep
uv run python -m src.utils.compare_runs logs/train/runs/A logs/train/runs/B --decoding argmax_groups
```

Config groups: `data/{puzzle,dev}`, `model/net/{q2l_grouped,q2l,flat,flat_spatial,grouped,mlgcn}`,
`trainer/{default,gpu,mps,cpu}`, `experiment/*`, `debug/*`, `callbacks`, `logger/{csv,wandb}`.
Add `--limit 100000` to preencode for the dev subset in `data/dev/`. Full-scale
two-seed runs on a CUDA host: `experiment=full_gpu seed=0|1`.

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
- **Model**: one ResNet encodes every position with shared weights. Default
  head is `q2l_grouped` with `--group-weight 0.5`: a Query2Label decoder (70
  label queries + 7 "none" queries cross-attend over the per-square features
  of every valid position) with a softmax per mutually exclusive tag group
  from cook.py and sigmoids on the 27 free tactical labels. Chosen in
  `experiments/2026-10-12_head_comparison.md` (tuned test macro-F1 0.580 vs
  0.387 for the pooled MLP head). Other heads: `flat`, `flat_spatial`,
  `grouped`, `q2l`, `mlgcn`. On an 8 GB machine run q2l heads with
  `--batch-size 128`.
- **Decoding**: `src/evaluate.py` reports per-class thresholds tuned on val;
  `src/analyze.py` also reports argmax-within-group decoding, which never
  emits two labels from one exclusive group (use that for a deployed tagger;
  see `experiments/2026-10-12_cooccurrence_analysis.md`).
- **Known ceiling**: crushing vs advantage is an engine centipawn boundary
  (cook.py: cp > 600 vs > 200) that the board does not expose; end-of-line
  material does not separate them (best material threshold = majority
  baseline on 200k puzzles). This caps outcome-group accuracy near 75%.
- **Loss**: BCE-with-logits with per-label `pos_weight = clip(sqrt(neg/pos), 1, 20)`
  (`--loss weighted_bce --pos-weight-power 0.5`), chosen over plain BCE, focal
  and ASL in `experiments/2026-10-11_loss_comparison.md`. Per-class decision
  thresholds are tuned on validation (`src/evaluate.py`). A grouped
  (softmax-per-exclusive-group) head is a planned experiment.

## Project structure

```
chess-puzzle-classifier/
├── configs/                      # Hydra config groups (see Pipeline)
│   ├── train.yaml, eval.yaml
│   ├── data/  model/  model/net/  trainer/  callbacks/  logger/  paths/  hydra/
│   ├── experiment/               # versioned experiment definitions (bounded_laptop, full_gpu)
│   └── debug/                    # tiny fast runs
├── data/                         # gitignored: raw CSV, puzzle_boards.h5, puzzle_metadata.json, splits.npz, dev/
├── experiments/                  # results notes (loss comparison, head comparison, co-occurrence analysis)
├── logs/                         # gitignored run outputs: <task>/runs/<run_name>/
├── scripts/                      # fetch / decompress the Lichess dump
├── src/
│   ├── data/
│   │   ├── preencode.py          # CSV -> HDF5 (multiprocess, streaming)
│   │   ├── puzzle_datamodule.py  # LightningDataModule
│   │   └── components/           # dataset.py (slab reads, bit unpacking), splits.py (stratified splits, sampler)
│   ├── models/
│   │   ├── puzzle_module.py      # LightningModule: loss, metrics, threshold tuning, decodings
│   │   └── components/           # resnet_board.py, heads.py (Query2Label, ML-GCN), puzzle_classifier.py
│   ├── utils/                    # bitboard_encoder, label_groups, losses, thresholds, ranking, metrics, compare_runs
│   ├── train.py                  # hydra app: fit + test
│   ├── eval.py                   # hydra app: re-tune thresholds on val, score test
│   └── analyze.py                # co-occurrence confusion + ranking analysis of a checkpoint
├── tests/                        # pytest: encoder, dataset, splits, heads, losses, thresholds, ranking, lightning
├── .project-root                 # rootutils marker (sets PROJECT_ROOT for configs/paths)
├── pyproject.toml / uv.lock
└── README.md
```

Theme semantics follow `lichess-puzzler/tagger/cook.py`; most tags are
functions of the solution line, which is why the line is part of the input.
