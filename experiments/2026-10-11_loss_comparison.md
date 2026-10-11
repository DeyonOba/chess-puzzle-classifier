# Loss comparison: BCE vs weighted BCE vs focal vs ASL

Date: 2026-10-11. Commit: f181a9f (losses at f7fa5d7).

## Setup

- Line-conditioned input, T=12. Model: hidden 128, 3 residual blocks, mean pooling (934k params).
- 1 epoch of 1500 batches x 256 = 384k training puzzles (7% of train), seed 0, identical batch order.
- Validation on the first 390 batches (99.8k puzzles) of the stratified val split.
- All F1 numbers use a fixed 0.5 threshold. Val loss is plain BCE for every run.
- weighted_bce: pos_weight = clip(neg/pos, 1, 20). focal: gamma 2, no alpha. asl: gamma_neg 4, gamma_pos 1, clip 0.05.
- Hardware: Apple A18 Pro, MPS, about 13 min per run.

## Results

| loss | micro-F1 | macro-F1 | macro-P | macro-R | macro PR-AUC | exact match |
|---|---|---|---|---|---|---|
| bce | 0.788 | 0.212 | 0.276 | 0.202 | 0.309 | 0.237 |
| weighted_bce | 0.609 | **0.325** | 0.274 | **0.532** | **0.353** | 0.047 |
| focal | **0.793** | 0.234 | **0.303** | 0.221 | 0.338 | **0.250** |
| asl | 0.751 | 0.303 | 0.291 | 0.359 | 0.335 | 0.103 |

Mean per-class F1 by label frequency band:

| band | n labels | bce | weighted_bce | focal | asl |
|---|---|---|---|---|---|
| common (>5%) | 19 | 0.595 | 0.677 | 0.620 | **0.683** |
| mid (0.5-5%) | 26 | 0.135 | **0.310** | 0.175 | 0.293 |
| rare (<0.5%) | 25 | 0.000 | **0.072** | 0.001 | 0.024 |

## Reading

- Plain BCE at 0.5 never predicts a rare label (rare-band F1 = 0.000) and misses
  mid-band labels like advancedPawn, promotion, mateIn4 entirely.
- weighted_bce wins macro-F1 by trading precision for recall: micro-F1 drops 18
  points and exact match collapses from 0.24 to 0.05. Common labels like
  mateIn3 and opening lose 15-18 F1 points.
- ASL gets most of the rare/mid gains with far less damage: best common-band
  F1, micro-F1 only 4 points below BCE, and the largest single-label wins
  (mateIn4 0.00 -> 0.60, queenEndgame 0.35 -> 0.83, zugzwang 0.00 -> 0.41).
- Focal is a mild, safe improvement over BCE on every metric.
- Threshold-free PR-AUC is close for weighted_bce, focal and ASL (0.335-0.353),
  so a large share of the macro-F1 gap is calibration at the 0.5 threshold,
  not ranking quality. Per-class threshold tuning should be applied before
  drawing a final conclusion.

## Tuned thresholds (added later the same day, commit a934f77)

Same four checkpoints, no retraining. Per-class thresholds were fitted to
maximise F1 on 99,840 val puzzles and then applied to 99,840 *test* puzzles.
The 0.5 rows are the same checkpoints scored on test at the fixed threshold.

| loss | threshold | micro-F1 | macro-F1 | macro-P | macro-R | exact match |
|---|---|---|---|---|---|---|
| bce | fixed 0.5 | 0.788 | 0.211 | 0.274 | 0.201 | 0.238 |
| bce | val-tuned | 0.671 | 0.331 | 0.302 | 0.421 | 0.078 |
| weighted_bce | fixed 0.5 | 0.609 | 0.324 | 0.271 | 0.531 | 0.046 |
| weighted_bce | val-tuned | 0.714 | **0.376** | **0.340** | 0.461 | **0.116** |
| focal | fixed 0.5 | **0.793** | 0.233 | 0.306 | 0.220 | **0.250** |
| focal | val-tuned | 0.695 | 0.359 | 0.333 | 0.441 | 0.107 |
| asl | fixed 0.5 | 0.751 | 0.302 | 0.288 | 0.358 | 0.102 |
| asl | val-tuned | 0.708 | 0.358 | 0.328 | 0.445 | 0.103 |

Mean per-class F1 by band, fixed 0.5 / val-tuned:

| band | n | bce | weighted_bce | focal | asl |
|---|---|---|---|---|---|
| common (>5%) | 19 | 0.594 / 0.701 | 0.676 / 0.707 | 0.620 / 0.711 | 0.682 / 0.712 |
| mid (0.5-5%) | 26 | 0.133 / 0.325 | 0.309 / **0.393** | 0.173 / 0.368 | 0.291 / 0.361 |
| rare (<0.5%) | 25 | 0.000 / 0.057 | 0.072 / **0.108** | 0.003 / 0.081 | 0.024 / 0.086 |

Labels still at tuned F1 < 0.05: bce 19, focal 14, asl 15, weighted_bce 9.
interference, equality, enPassant, collinearMove and castling are dead for
every loss at this budget.

### What tuning changes

- Thresholds are worth more than the loss choice: +0.12 macro-F1 for bce,
  +0.13 for focal, +0.06 for asl, +0.05 for weighted_bce. On common labels
  all four losses converge to 0.70-0.71 once tuned.
- The untuned ranking flips. weighted_bce (cap 20) is best after tuning at
  0.376 macro-F1, with the best precision and exact match of the tuned rows
  and the most live rare labels. asl and focal tie at 0.358-0.359; bce 0.331.
- The gap between weighted_bce and asl/focal (0.018) is about the noise
  floor for a single seed at 7% of an epoch.
- F1-optimal thresholds trade precision for recall per label, so micro-F1
  and exact match fall for bce/focal/asl relative to their 0.5 rows. If the
  headline metric were micro-F1 or exact match, focal at 0.5 would win.
  Macro-F1 is the headline here because rare themes are the point.

## Decision

- Per-class threshold tuning is on by default (train.py selects on tuned
  macro-F1, src/evaluate.py reports val-tuned test numbers).
- Default loss set to asl (commit a934f77) on the untuned results. With
  tuned thresholds weighted_bce is marginally ahead; whether to switch the
  default, or confirm with a second seed / larger budget, is pending.

## Caveats

Single seed, 7% of one epoch, small model. Differences under about 0.02 are
within noise at this budget.
