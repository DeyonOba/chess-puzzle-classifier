# chess-puzzle-classifier

Multi-label classification of Lichess puzzle themes (the tags produced by
[`lichess-puzzler/tagger/cook.py`](https://github.com/ornicar/lichess-puzzler/blob/master/tagger/cook.py))
from the board and the solution line, without an engine. python-chess for
encoding, PyTorch Lightning + Hydra for experiments, FastAPI for serving.

## Methodology

### Task framing

cook.py assigns most tags by walking the solution line: length tags are the
ply count, `mateInN` is where the line ends, `sacrifice`, `fork`, `quietMove`
and the mate patterns are tests on positions along the line. A model that sees
only the starting position has to solve the puzzle implicitly. We therefore
train on **the line**: the position the solver sees plus the board after every
solution ply, and keep a position-only mode (`data.max_positions=1`) for
puzzles where the solution is unknown at inference time.

The label set has structure that cook.py's `if/elif` chains make exact: seven
groups of mutually exclusive tags (outcome, length, phase, mate depth, mate
pattern, endgame type, attack side) and hard implications such as
`mateInN -> mate`. This is encoded in the head rather than learned.

### Data and encoding

| step | what | where |
|---|---|---|
| source | Lichess puzzle dump, 6.16M puzzles, 4.5 tags per puzzle on average | `scripts/` |
| positions | FEN after the setup move (what the solver sees), then after each solution ply; T_max = 12 covers 99.7%, longer lines truncated | `src/utils/bitboard_encoder.py` |
| encoding | 18 planes per position (12 piece, side to move, 4 castling, en passant) stored as packed `uint64` bitboards, 144 B per position, unpacked in the DataLoader | same |
| storage | HDF5 `(N, 12, 18) uint64` + lengths, targets, plies, rating, puzzle id; 1.3 GB; multiprocess streaming encode in ~5 min | `src/data/preencode.py` |
| vocabulary | **v2: 69 labels.** `crushing` + `advantage` merged into `winning` (an engine centipawn threshold the board does not expose); `master`, `masterVsMaster`, `superGM` dropped (player-rating tags) | `preencode --merge` |
| splits | iterative multi-label stratification (Sechidis 2011), 90 / 5 / 5, index arrays in `splits_v2.npz`; the rarest label lands within one sample of its target share in every split | `src/data/components/splits.py` |
| loading | batches are consecutive HDF5 rows (the file is in random puzzle-id order) read as one slab, so the lzf-chunked file streams at ~50k puzzles/s | `components/dataset.py`, `PuzzleDataModule` |

### Model

```
(B, T, 18, 8, 8) + lengths
   └─ shared ResNet per valid position (stem + residual blocks)  -> (B, T, D, 8, 8)
        └─ Query2Label decoder: 69 label queries + 7 "none" queries cross-attend
           over the T x 64 square tokens (square + ply embeddings, padding masked)
             └─ per-query readout -> 69 label logits + 7 group "none" logits
                  ├─ softmax within each exclusive group (incl. "none")
                  └─ sigmoid on the 27 free tactical labels
```

Default: `model/net=q2l_grouped`, `group_weight=0.5`. Alternatives kept as
configs: `flat` (pooled MLP, the original baseline), `flat_spatial` (pooling
control), `grouped` (pooled + group softmax), `q2l`, `mlgcn`.

### Loss, thresholds, decoding

- **Loss**: per-group cross-entropy (class weights `clip((n/n_class)^0.5, 1, 20)`,
  weight 0.5) plus BCE-with-logits on the free labels with
  `pos_weight = clip(sqrt(neg/pos), 1, 20)`. Focal and ASL are available
  (`model.loss=`) and were beaten in the comparison below.
- **Thresholds**: per-class decision thresholds that maximise F1 are tuned on
  the validation split (exact, one sort per label) and stored in the
  checkpoint. Model selection uses val macro-F1 with tuned thresholds.
- **Decoding**: *independent* applies the per-class thresholds; *argmax
  within groups* predicts at most one label per exclusive group (a member
  only if it beats the group's "none" mass). Independent gives the higher
  macro-F1 by hedging on uncertain labels; argmax never emits contradictory
  tags and gives the best exact match. Both are reported; argmax is what the
  API serves by default.
- **Rules at inference**: tags that are pure functions of the inputs are set
  by cook.py's rule rather than predicted when the inputs are available:
  length from the ply count, `mate`/`mateInN` from a line that ends in
  checkmate, and the fine-grained outcome from an optional engine score.

### Evaluation protocol

Every comparison in `experiments/` uses the same bounded protocol
(`experiment=bounded_laptop`): 384k training puzzles (7% of one epoch),
seed 0, a 128-wide 3-block encoder, thresholds tuned on 100k validation
puzzles, scored on 100k **test** puzzles. Metrics: micro / macro F1,
macro precision / recall, exact match, PR-AUC; per-class F1 by frequency band
(common > 5%, mid 0.5 to 5%, rare < 0.5%) and by group; per-puzzle ranking
metrics (P@k, LRAP, coverage); within-group confusion matrices; substitution
tables; co-occurrence drift. Single-seed differences under ~0.01 macro-F1 are
noise (measured seed spread 0.002 to 0.008).

## Results so far (v1 vocabulary, 70 labels, bounded protocol, test split)

| experiment | configuration | tuned macro-F1 | tuned micro-F1 | note |
|---|---|---|---|---|
| loss | bce | 0.331 | 0.671 | rare labels never fire at 0.5 |
| loss | focal / asl | 0.359 / 0.356 | 0.695 / 0.708 | two seeds for asl |
| loss | **weighted_bce, sqrt** | **0.386** | 0.713 | two seeds; best untuned calibration |
| head | flat (pooled MLP) | 0.387 | 0.713 | baseline |
| head | grouped | 0.421 | 0.704 | gains only on the exclusive groups |
| head | mlgcn | 0.410 | 0.722 | dependencies already deterministic |
| head | q2l | 0.561 | 0.801 | +0.17: label-query attention over squares |
| head | **q2l_grouped, gw 0.5** | **0.580** | **0.815** | 2 dead labels of 70 |

Decomposition of the flat -> q2l gap: optimizer steps +0.02, keeping square
identity +0.03, label-query cross-attention +0.12. Threshold tuning alone is
worth +0.05 to +0.13 depending on the loss. The remaining error mass sits on
the crushing/advantage boundary (hence the v2 merge), on "said nothing"
misses of the big tactical labels (fork, pin, discoveredAttack), and on eight
labels that are unlearned at this budget (equality, underPromotion, castling,
enPassant, interference, intermezzo, collinearMove, capturingDefender).
Details: `experiments/2026-10-11_loss_comparison.md`,
`experiments/2026-10-12_head_comparison.md`,
`experiments/2026-10-12_cooccurrence_analysis.md`.

v2-vocabulary runs (line and position-only) and full-scale two-seed runs on a
CUDA host (`experiment=full_gpu`) are the next results to add here.

## Pipeline

```bash
make setup                      # uv sync
make data                       # download (300 MB) -> decompress (1.1 GB) -> encode HDF5 (~5 min, 1.3 GB) -> splits
make dev                        # 100k-row subset in data/dev/ for smoke runs and tests
make test                       # pytest
make train ARGS="run_name=q2lg_s0"
make eval  CKPT=logs/train/runs/q2lg_s0/checkpoints/best.ckpt
make serve CKPT=logs/train/runs/q2lg_s0/checkpoints/best.ckpt POS_CKPT=logs/train/runs/pos/checkpoints/best.ckpt
```

`make data` is file-driven: each step runs only if its output is missing, so
re-running after a partial download or a changed `--merge` (`make clean-data`)
does the minimum. Variables: `DATA_DIR`, `MERGE` (default
`winning=crushing,advantage`), `T_MAX` (12), `DEV_N` (100000). The underlying
commands:

```bash
scripts/fetch_puzzles.sh && scripts/decompress_zstd.sh         # data/lichess_db_puzzle.csv
uv run python -m src.data.preencode --csv data/lichess_db_puzzle.csv \
    --out data/puzzle_boards_v2.h5 --meta data/puzzle_metadata_v2.json \
    --merge winning=crushing,advantage
uv run python -m src.data.components.splits --h5 data/puzzle_boards_v2.h5 --out data/splits_v2.npz
uv run python -m src.train run_name=q2lg_s0                    # defaults: q2l_grouped, gw 0.5, weighted BCE
uv run python -m src.eval ckpt_path=logs/train/runs/q2lg_s0/checkpoints/best.ckpt
uv run python -m src.analyze logs/train/runs/q2lg_s0 --eval-results logs/eval/runs/<ts>/eval_results.json
```

Training and evaluation are Hydra apps (layout after
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)).
Every run writes its resolved config, CSV metrics, `checkpoints/{best,last}.ckpt`
and `test_results.json` (all three decodings, per-class thresholds and F1) to
`logs/<task>/runs/<run_name>/`.

```bash
uv run python -m src.train debug=default                           # 5-batch smoke run on data/dev
uv run python -m src.train experiment=bounded_laptop run_name=x    # the protocol used in experiments/
uv run python -m src.train experiment=bounded_laptop data.max_positions=1 run_name=pos   # position-only
uv run python -m src.train experiment=full_gpu seed=1 run_name=q2lg_full_s1              # CUDA host
uv run python -m src.train model/net=q2l model.loss=asl            # any override
uv run python -m src.train -m model.group_weight=0.25,0.5,1        # sweep
uv run python -m src.utils.compare_runs logs/train/runs/A logs/train/runs/B --decoding argmax_groups
```

Config groups: `data/{puzzle,dev}`, `model/net/{q2l_grouped,q2l,flat,flat_spatial,grouped,mlgcn}`,
`trainer/{default,gpu,mps,cpu}`, `experiment/*`, `debug/*`, `callbacks`, `logger/{csv,wandb}`.
`preencode --limit 100000` builds the dev subset in `data/dev/`. On an 8 GB
laptop the q2l heads need `data.batch_size=128` (the bounded experiment sets it).

## Serving

```bash
CPC_CKPT=logs/train/runs/NAME/checkpoints/best.ckpt \
CPC_POSITION_CKPT=logs/train/runs/NAME_pos/checkpoints/best.ckpt \
uv run uvicorn src.app:app --port 8000

curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24", "setup_move": "f2g3",
  "moves": ["e6e7", "b2b1", "b3c1", "b1c1", "h6c1"], "cp": 850}'
```

| field | required | meaning |
|---|---|---|
| `fen` | yes | the position the solver sees (side to move = solver) |
| `setup_move` | no | if `fen` is a Lichess CSV FEN, the first move of its Moves column |
| `moves` | no | the solution line in UCI; without it the position-only model answers (or the line model with one position) |
| `cp` | no | engine score after the line, solver POV; sets `winning` / `equality` and the fine-grained `crushing` / `advantage` by rule |
| `decoding` | no | `argmax_groups` (default) or `independent` |
| `apply_rules` | no | default true: length, mate depth and cp-derived tags by rule |

The response carries the tags, every label's probability and threshold, which
tags were set by rule and why, which model answered, and whether a line was
provided. `GET /health`, `GET /labels`. `src/inference.py` exposes the same
logic as `Predictor`.

## Project structure

```
chess-puzzle-classifier/
├── configs/                      # Hydra config groups
│   ├── train.yaml, eval.yaml
│   ├── data/  model/  model/net/  trainer/  callbacks/  logger/  paths/  hydra/
│   ├── experiment/               # versioned experiment definitions (bounded_laptop, full_gpu)
│   └── debug/                    # tiny fast runs
├── data/                         # gitignored: raw CSV, puzzle_boards_v2.h5, puzzle_metadata_v2.json, splits_v2.npz, dev/
├── experiments/                  # results notes: loss comparison, head comparison, co-occurrence analysis
├── logs/                         # gitignored run outputs: <task>/runs/<run_name>/
├── scripts/                      # fetch / decompress the Lichess dump
├── src/
│   ├── data/
│   │   ├── preencode.py          # CSV -> HDF5 (multiprocess, streaming, --merge, --limit)
│   │   ├── puzzle_datamodule.py  # LightningDataModule
│   │   └── components/           # dataset.py (slab reads, bit unpacking), splits.py (stratified splits, sampler)
│   ├── models/
│   │   ├── puzzle_module.py      # LightningModule: loss, metrics, threshold tuning, three decodings
│   │   └── components/           # resnet_board.py, heads.py (Query2Label, ML-GCN), puzzle_classifier.py
│   ├── utils/                    # bitboard_encoder, label_groups, losses, thresholds, ranking, metrics, compare_runs
│   ├── train.py                  # hydra app: fit + test best checkpoint
│   ├── eval.py                   # hydra app: re-tune thresholds on val, score test
│   ├── analyze.py                # within-group confusion, substitutions, co-occurrence drift, ranking metrics
│   ├── inference.py              # Predictor: encode, predict, decode, rules
│   └── app.py                    # FastAPI service
├── tests/                        # pytest: encoder, dataset, splits, heads, losses, thresholds, ranking, lightning, inference
├── Makefile                      # setup / data / dev / test / train / eval / serve
├── .project-root                 # rootutils marker (sets PROJECT_ROOT for configs/paths)
├── pyproject.toml / uv.lock
└── README.md
```

## Known limits

- **Outcome boundary.** crushing vs advantage is `cp > 600` vs `cp > 200` in
  cook.py. End-of-line material does not separate them (the best material
  threshold equals the majority baseline on 200k puzzles), and the model
  reached 75% on that group. v2 merges them; pass `cp` to recover the split.
- **Local hardware.** All numbers above come from 7% of an epoch on an 8 GB
  Apple laptop. Full epochs and multi-seed runs need a CUDA host.
- **Position-only mode** cannot know how a line ends; it is for classifying
  a position as a puzzle type without a solution, and the API labels such
  answers as coming from the position model.
