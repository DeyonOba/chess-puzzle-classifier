# Full-data run on Kaggle (base planes)

Date: 2026-10-13. Kernel `gideonoba/chess-puzzle-classifier-full-data-training` v2, dataset
`chess-puzzle-data-v2` (base 18 planes). Run dir archived locally as `logs/train/runs/q2lg_full_base_kaggle`
(best.ckpt = validation step 194,813, epoch 8.5).

## Setup

`experiment=kaggle` (commit 605e355 config, code at 3aa0117): 192-wide 4-block encoder, q2l_grouped
group weight 0.5, weighted BCE sqrt, plateau scheduler (never triggered), early stopping patience 4
(never triggered), fp16-mixed, batch 256, 2 loader workers, val_check_interval 0.5, max_time 11 h.
Thresholds tuned on the full validation split (307,867), scored on the full test split (307,867).

## Result

| decoding | micro-F1 | macro-F1 | macro-P | macro-R | exact match | PR-AUC | dead labels |
|---|---|---|---|---|---|---|---|
| fixed 0.5 | 0.968 | 0.886 | 0.841 | 0.949 | 0.794 | 0.916 | 0 |
| independent thresholds | 0.976 | 0.905 | 0.894 | 0.918 | 0.845 | 0.916 | 0 |
| argmax within groups | 0.975 | 0.900 | 0.880 | 0.926 | 0.841 | 0.916 | 0 |

Per subset (tuned): outcome 0.717 (mate 0.996, winning 0.992, equality 0.163), length 0.995, phase
0.977, mateIn 0.982, matePattern 0.908, endgameType 0.992, attackSide 0.945, free 0.865; common 0.972,
mid 0.896, rare 0.865. Lowest labels: equality 0.16, clearance 0.49, intermezzo 0.62, collinearMove
0.70, capturingDefender 0.70, vukovicMate 0.72, interference 0.72, zugzwang 0.78, quietMove 0.78.

## Training curve

9.5 epochs in 11 h (66 min/epoch; the config comment had estimated 3 h). Train loss per epoch:
0.206, 0.123, 0.106, 0.097, 0.091, 0.086, 0.083, 0.080, 0.078, 0.076. Validation tuned macro-F1 at each
half epoch: 0.794, 0.844, 0.860, 0.870, 0.880, 0.885, 0.886, 0.891, 0.895, 0.895, 0.900, 0.899, 0.901,
0.902, 0.902, 0.902, 0.906, 0.909, 0.907, 0.908. Still improving by ~0.003 per epoch at the cut;
stopped by max_time, not by early stopping.

## Reading

- The bounded laptop protocol (7% of an epoch) under-reported the architecture by 0.32 macro-F1; the
  ranking of design choices it produced still held (every family that was weak there is weak here).
- No label is dead. The remaining weak set is the one predicted in the co-occurrence analysis: the
  engine-defined `equality`, and the move-semantics tactics (clearance, intermezzo, collinearMove,
  capturingDefender, interference).
- Threshold tuning is worth +0.019 macro-F1 and +0.05 exact match at this scale; argmax decoding costs
  0.005 macro-F1 for guaranteed consistency.

## Next

1. Attack-map planes at full scale (expected +0.03 to +0.05 from the bounded result): upload v2x
   (`make kaggle-dataset`), push the kernel; the script selects the 26-plane file automatically.
2. Or resume this run for more epochs (it had not converged): attach the kernel output as a source
   and push; the script resumes from last.ckpt.

## v2x (attack-map planes) at full scale, 2026-10-15

Kernel version 4, dataset version 2 (puzzle_boards_v2x.h5, 26 planes), run `q2lg_full_v2x`
(archived at `logs/train/runs/q2lg_full_v2x_kaggle`). Same config, 10 epochs in 11 h.

| run | tuned micro-F1 | tuned macro-F1 | tuned exact | argmax macro-F1 | argmax exact | PR-AUC |
|---|---|---|---|---|---|---|
| v2 (18 planes) | 0.9764 | 0.9047 | 0.8454 | 0.8999 | 0.8410 | 0.9156 |
| v2x (26 planes) | 0.9762 | 0.9020 | 0.8464 | 0.8963 | 0.8386 | 0.9143 |

Per subset delta (v2x - v2, tuned F1): outcome -0.003, length 0.000, phase -0.004, mateIn +0.003,
matePattern -0.005, endgameType 0.000, attackSide +0.005, free -0.004, common -0.001, mid -0.001,
rare -0.005. Largest label moves: clearance +0.040, vukovicMate +0.027, dovetailMate +0.025, pin +0.020;
discoveredCheck -0.065, morphysMate -0.039, discoveredAttack -0.034, cornerMate -0.028.

Validation tuned macro-F1 per half epoch, v2 / v2x: 0.794/0.810, 0.844/0.845, 0.860/0.860, 0.870/0.868,
0.880/0.879, 0.885/0.880, 0.886/0.886, 0.891/0.889, 0.895/0.891, 0.895/0.894, 0.900/0.895, 0.899/0.896,
0.901/0.898, 0.902/0.900, 0.902/0.902, 0.901/0.902, 0.906/0.901, 0.909/0.904, 0.907/0.906, 0.908/0.905.
Train loss per epoch is equal to the third decimal from epoch 2 on.

Conclusion: the bounded-protocol gain (+0.054) was a data-efficiency effect; at full scale the planes
are neutral (-0.003 macro-F1, +0.001 exact match, within seed noise). The encoder learns the
interactions itself given enough epochs. Default input: pending decision (revert to base planes for
the cheaper pipeline, or keep extended for short-budget runs).
