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
| encoding | 18 planes per position (12 piece, side to move, 4 castling, en passant) stored as packed `uint64` bitboards, 144 B per position, unpacked in the DataLoader. `--planes extended` adds 8 python-chess attack-map planes (attacked squares, pins, hanging pieces, checks): +0.05 macro-F1 on short budgets, neutral at full scale | same |
| storage | HDF5 `(N, 12, 18) uint64` + lengths, targets, plies, rating, puzzle id; 1.26 GB; multiprocess streaming encode in ~4 min (26-plane file: 1.9 GB, ~17 min) | `src/data/preencode.py` |
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

### v2 vocabulary (69 labels, crushing + advantage merged), bounded protocol, test split

Same protocol, same default model (`q2l_grouped`, group weight 0.5), after the
refactor. The position-only model is trained and evaluated on the first
position only (`data.max_positions=1`), for requests without a solution line.

| model | decoding | micro-F1 | macro-F1 | macro-P | macro-R | exact match | dead labels |
|---|---|---|---|---|---|---|---|
| line (T=12) | independent thresholds | 0.859 | **0.586** | 0.572 | 0.622 | 0.342 | 2 |
| line (T=12) | argmax within groups | 0.854 | 0.563 | 0.513 | 0.665 | 0.322 | 3 |
| line (T=12) | fixed 0.5 | **0.862** | 0.553 | 0.515 | 0.644 | **0.352** | 7 |
| position-only (T=1) | independent thresholds | 0.599 | 0.350 | 0.315 | 0.430 | 0.013 | 12 |
| position-only (T=1) | argmax within groups | 0.592 | 0.331 | 0.298 | 0.423 | 0.056 | 14 |
| position-only (T=1) | fixed 0.5 | 0.607 | 0.295 | 0.329 | 0.340 | 0.047 | 24 |

Tuned per-subset F1, line / position-only: outcome 0.65 / 0.51 (winning 0.97 /
0.84, mate 0.95 / 0.68, equality dead in both), length 1.00 / 0.46, mateIn
0.85 / 0.24, matePattern 0.55 / 0.24, endgameType 0.96 / 0.95, phase 0.93 /
0.93, free tactical labels 0.37 / 0.20.

What the merge changed against v1: micro-F1 0.815 -> 0.859 and exact match
0.252 -> 0.342, because the crushing / advantage swap was the largest error
source; macro-F1 0.580 -> 0.586. What the line is worth: the position-only
model keeps the tags that are properties of the position (phase, endgame type,
winning) and loses the ones that are properties of the solution (length, mate
depth, mate patterns, most tactics), which is the expected split and the reason
the API reports which model answered.

### Attack-map input planes (v2 vocabulary, bounded protocol, test split)

`preencode --planes extended` adds 8 exact python-chess planes to the 18 base
planes: squares attacked by each side, pinned pieces per side, hanging pieces
per side, the checking pieces, and an in-check indicator. Same rows, splits,
model and protocol as the v2 line model above (`experiment=attack_planes_laptop`).
Base planes remain the default input: see the full-scale comparison below.

| input | decoding | micro-F1 | macro-F1 | macro-P | macro-R | exact match | dead labels |
|---|---|---|---|---|---|---|---|
| base planes (18) | independent thresholds | 0.859 | 0.586 | 0.572 | 0.622 | 0.342 | 2 |
| base planes (18) | argmax within groups | 0.854 | 0.563 | 0.513 | 0.665 | 0.322 | 3 |
| attack-map planes (26) | independent thresholds | **0.894** | **0.640** | **0.628** | 0.675 | **0.444** | 3 |
| attack-map planes (26) | argmax within groups | 0.888 | 0.614 | 0.572 | **0.716** | 0.419 | 3 |

Tuned per-subset F1, base / attack-map: free tactical labels 0.37 / **0.48**,
mateIn 0.85 / **0.91**, common band 0.81 / **0.88**, mid band 0.60 / **0.67**,
rare band 0.42 / 0.44; length, endgame type and phase unchanged. Largest
per-label gains are exactly the facts the planes encode: doubleCheck +0.55,
pin +0.54, discoveredCheck +0.29, hangingPiece +0.20, fork +0.18, skewer +0.15.
Small losses on four rare mate patterns (swallowstail, killBox, smothered,
castling, 0.08 to 0.15) are within single-seed noise for labels with under
100 test positives.

Reading: +0.054 macro-F1 and +0.10 exact match from input features alone,
larger than the Query2Label-over-flat gain after the pooling controls. The
encoder no longer has to rediscover attacks, pins and hanging pieces from
piece placement, so its capacity goes to the tags. Encoding costs 4.5x
(17 min for the full file instead of 4) and the HDF5 grows from 1.26 GB to
1.8 GB.

### Full-data run on Kaggle (base planes, 18), full test split

`experiment=kaggle` on a Kaggle T4: 192-wide 4-block encoder (3.2M params),
`q2l_grouped` gw 0.5, weighted BCE sqrt, plateau scheduler, fp16, batch 256.
Stopped by the 11-hour session limit after 9.5 epochs (66 minutes per epoch,
three times faster than the README's earlier estimate); early stopping never
triggered and the learning rate was never reduced, so the model was still
improving when it stopped. Thresholds tuned on the full 308k validation
split, scored on the full 308k test split.

| decoding | micro-F1 | macro-F1 | macro-P | macro-R | exact match | PR-AUC | dead labels |
|---|---|---|---|---|---|---|---|
| fixed 0.5 | 0.968 | 0.886 | 0.841 | 0.949 | 0.794 | 0.916 | 0 |
| independent thresholds | **0.976** | **0.905** | **0.894** | 0.918 | **0.845** | 0.916 | 0 |
| argmax within groups | 0.975 | 0.900 | 0.880 | **0.926** | 0.841 | 0.916 | 0 |

Tuned per-subset F1: length 0.995, endgame type 0.992, mate depth 0.982,
phase 0.977, attack side 0.945, mate patterns 0.908, free tactical labels
0.865; common band 0.972, mid 0.896, rare 0.865. Every label is alive; the
weakest are equality 0.16 (227 test positives, an engine-eval boundary),
clearance 0.49, intermezzo 0.62, collinearMove and capturingDefender 0.70.
Validation tuned macro-F1 by half-epoch: 0.79, 0.84, 0.86, 0.87, 0.88, 0.88,
0.89, 0.89, 0.89, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.91, 0.91, 0.91, 0.91.

Scale is what the bounded runs were missing: the same architecture went from
0.586 to 0.905 macro-F1 with 70x the data, and the "said nothing" misses on
fork, pin and discoveredAttack are gone.

### Attack-map planes at full scale: v2 vs v2x (two 11-hour Kaggle sessions)

Identical setup (`experiment=kaggle`, same rows, splits, encoder, loss,
schedule, 10 epochs each in 11 h), the only difference being the input planes.

| run | decoding | micro-F1 | macro-F1 | macro-P | macro-R | exact match | PR-AUC |
|---|---|---|---|---|---|---|---|
| v2, 18 planes | independent thresholds | 0.976 | **0.905** | 0.894 | 0.918 | 0.845 | 0.916 |
| v2, 18 planes | argmax within groups | 0.975 | 0.900 | 0.880 | 0.926 | 0.841 | 0.916 |
| v2x, 26 planes | independent thresholds | 0.976 | 0.902 | 0.894 | 0.914 | **0.846** | 0.914 |
| v2x, 26 planes | argmax within groups | 0.974 | 0.896 | 0.878 | 0.923 | 0.839 | 0.914 |

No subset moved by more than 0.005 (free tactical labels 0.865 vs 0.861,
mateIn 0.982 vs 0.985, attackSide 0.945 vs 0.950); 13 of 69 labels moved by
more than 0.02 in either direction, the largest being clearance +0.04 and
discoveredCheck -0.07, which is single-seed noise on labels with few test
positives. The validation curves tell the story: v2x starts ahead (0.810 vs
0.794 after half an epoch) and the two converge by epoch 3 (0.880 vs 0.879)
to the same ceiling (0.905 vs 0.909 best val).

Reading: the +0.054 the planes gave on 7% of an epoch was a data-efficiency
effect. With 10 epochs over 5.5M puzzles the encoder learns attacks, pins
and hanging pieces from piece placement on its own, and the explicit planes
add nothing at the end while costing 4.5x the encoding time and 50% more
storage. They remain useful for small-data or short-budget training.

### Refactor parity check (v1 vocabulary, same bounded protocol)

The pre-refactor trainer (plain PyTorch loop) and the Lightning/Hydra
`experiment=bounded_laptop` run of the same configuration, seed 0:

| | tuned macro-F1 | tuned micro-F1 | tuned exact | argmax macro-F1 | argmax exact |
|---|---|---|---|---|---|
| before refactor (commit d268320) | 0.580 | 0.815 | 0.252 | 0.555 | 0.272 |
| after refactor (Lightning) | 0.580 | 0.815 | 0.252 | 0.555 | 0.272 |

Model initialisation differs (Lightning's `seed_everything` vs a bare
`torch.manual_seed`), so the runs are not bit-identical; agreement within the
measured seed spread (under 0.01 macro-F1) is the pass criterion.


## Pipeline

```bash
make setup                      # uv sync
make data                       # download (300 MB) -> decompress (1.1 GB) -> encode HDF5 (~4 min, 1.26 GB) -> splits
make dev                        # 100k-row subset in data/dev/ for smoke runs and tests
make test                       # pytest
make train ARGS="run_name=q2lg_s0"
make eval  CKPT=logs/train/runs/q2lg_s0/checkpoints/best.ckpt
make serve CKPT=logs/train/runs/q2lg_s0/checkpoints/best.ckpt POS_CKPT=logs/train/runs/pos/checkpoints/best.ckpt
```

`make data` is file-driven: each step runs only if its output is missing, so
re-running after a partial download or a changed `--merge` (`make clean-data`)
does the minimum. Variables: `DATA_DIR`, `PLANES` (`base`, the default, or
`extended`), `MERGE` (default `winning=crushing,advantage`), `T_MAX` (12),
`DEV_N` (100000). The underlying
commands:

```bash
scripts/fetch_puzzles.sh && scripts/decompress_zstd.sh         # data/lichess_db_puzzle.csv
uv run python -m src.data.preencode --csv data/lichess_db_puzzle.csv \
    --out data/puzzle_boards_v2.h5 --meta data/puzzle_metadata_v2.json \
    --merge winning=crushing,advantage          # add --planes extended for the 26-plane v2x file
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

## Full-data training on Kaggle (free GPU, 30 h / week, 12 h / session)

1. Build the data locally (`make data`) and upload `puzzle_boards_v2.h5`,
   `puzzle_metadata_v2.json` and `splits_v2.npz` as a Kaggle dataset (~1.4 GB).
2. In a GPU notebook: clone the repo, `pip install -e .` (or `uv sync`), then

```bash
python -m src.train experiment=kaggle run_name=q2lg_full \
    paths.data_dir=/kaggle/input/<dataset> paths.log_dir=/kaggle/working/logs
```

3. Save `/kaggle/working/logs` as the notebook output. Next session, resume:

```bash
python -m src.train experiment=kaggle run_name=q2lg_full \
    ckpt_path=/kaggle/working/logs/train/runs/q2lg_full/checkpoints/last.ckpt \
    paths.data_dir=... paths.log_dir=...
```

What `experiment=kaggle` sets and why:

| setting | value | reason |
|---|---|---|
| encoder | 192 wide, 4 blocks (3.2M params) | measured 66 min per epoch on a T4 (base planes); one 11 h session covers ~10 epochs |
| input | base planes by default (`PLANES=extended` for v2x) | the kernel script sets `in_planes` to match the attached file |
| precision | `16-mixed` | T4 has no bf16 |
| `max_time` | 11 h | stop before the session limit with a resumable `last.ckpt` |
| `val_check_interval` | 0.5 | validate and checkpoint twice per epoch |
| early stopping | patience 4 validations, min_delta 0.001 on val tuned macro-F1 | two epochs of no progress before stopping |
| scheduler | `plateau` (halve LR after 2 flat validations) | OneCycle bakes the total step count into its state and is not resumable across sessions or early stops |

Resume restores optimizer, scheduler, epoch, early-stopping state and the
tuned thresholds. Then `python -m src.eval ckpt_path=.../best.ckpt` and
`python -m src.analyze` as usual.

## Serving

```bash
CPC_CKPT=logs/train/runs/q2lg_full_base_kaggle/checkpoints/best.ckpt \
CPC_POSITION_CKPT=logs/train/runs/v2_q2lg_gw05_pos_s0/checkpoints/best.ckpt \
uv run uvicorn src.app:app --port 8000
# or: make serve CKPT=... POS_CKPT=... [PORT=8000]
```

| variable | value | why |
|---|---|---|
| `CPC_CKPT` | a line-model checkpoint (the Kaggle full-data run above) | required; answers requests that include the solution line |
| `CPC_POSITION_CKPT` | a position-only checkpoint (`data.max_positions=1`) | optional; answers FEN-only requests. Without it the line model runs on a single position, which is weaker |
| `CPC_DEVICE` | `auto` (default), `cpu`, `mps`, `cuda` | where the models run |
| `CPC_EVAL_RESULTS` | an `eval_results.json` from `src.eval` | optional; replaces the checkpoint's thresholds with ones re-tuned on the full validation split |

A checkpoint is self-describing: the number of input planes (18 or 26), the
label list and the tuned thresholds are read from it, so base-plane and
attack-map checkpoints are served the same way. `GET /health` reports the
label count, device and whether a position model is loaded; `GET /labels`
lists the vocabulary.

### Requests

`fen` is the only required field, and it must be **the position the solver
sees** (side to move = solver). A Lichess CSV row stores the position one move
earlier, so either apply its first move yourself or pass it as `setup_move`.

```bash
# FEN only (position model): the solver's position
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"fen": "6rk/1p3Rr1/2p2Q2/8/5P2/P5P1/1Pq4P/4R1K1 b - - 0 32"}'
# -> ["endgame", "long", "winning"], "model": "position"

# Lichess CSV row: CSV FEN + setup move + solution line (line model)
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "fen": "6rk/1p1R1pr1/2p2Q2/8/5P2/P5P1/1Pq4P/4R1K1 w - - 3 32", "setup_move": "d7f7",
  "moves": ["c2c5", "g1g2", "c5d5", "g2f2", "d5f7", "f6f7", "g7f7"]}'
# -> ["endgame", "veryLong", "winning"], "model": "line"   (Lichess: advantage endgame veryLong)
```

| field | required | meaning |
|---|---|---|
| `fen` | yes | the position the solver sees, or the CSV FEN together with `setup_move` |
| `setup_move` | no | the opponent's move that leads to the puzzle position (first move of the CSV Moves column) |
| `moves` | no | the solution line in UCI from the solver's position; without it the position model answers |
| `cp` | no | engine score after the line, solver POV; sets `winning` / `equality` and the fine-grained `crushing` / `advantage` by rule |
| `decoding` | no | `argmax_groups` (default: at most one tag per exclusive group) or `independent` (per-class thresholds) |
| `apply_rules` | no | default true: length, mate depth and cp-derived tags are set by cook.py's rule instead of predicted |

The response carries `tags`, every label's `probabilities` and `thresholds`,
`rule_tags` (which tags were set by rule and why), `model` (`line` or
`position`), `line_provided` and `positions` (how many positions the model
saw). Sending the CSV FEN without `setup_move` classifies the wrong side's
puzzle: for the example above it answers "mate in 2" for white instead of
black's winning endgame. `src/inference.py` exposes the same logic as
`Predictor` for use without HTTP.

## Project structure

```
chess-puzzle-classifier/
├── configs/                      # Hydra config groups
│   ├── train.yaml, eval.yaml
│   ├── data/  model/  model/net/  trainer/  callbacks/  logger/  paths/  hydra/
│   ├── experiment/               # versioned experiment definitions (bounded_laptop, full_gpu)
│   └── debug/                    # tiny fast runs
├── data/                         # gitignored: raw CSV, puzzle_boards_v2.h5 (+ v2x attack planes), metadata, splits_v2.npz, dev/
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
