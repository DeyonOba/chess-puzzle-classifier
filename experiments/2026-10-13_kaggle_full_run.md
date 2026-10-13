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
