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

## Decision (pending approval)

Recommend ASL as the default training loss, and re-run this comparison with
per-class thresholds tuned on validation.

## Caveats

Single seed, 7% of one epoch, small model. Differences under about 0.02 are
within noise at this budget.
