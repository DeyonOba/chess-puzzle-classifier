# Head comparison: flat vs grouped softmax vs Query2Label vs ML-GCN

Date: 2026-10-12. Commits: grouped 61bfa03, q2l/mlgcn b560829.

## Setup

- Line-conditioned input, T=12. Shared encoder: hidden 128, 3 residual blocks.
- Loss: weighted BCE, pos_weight = clip(sqrt(neg/pos), 1, 20) (project default).
  Grouped head adds per-group cross-entropy with the same weighting recipe.
- 384k training puzzles, seed 0, identical batch order where the batch size
  matches. Thresholds tuned per class on 99,840 val puzzles; scored on
  99,840 test puzzles.
- Query2Label at batch 256 thrashed into swap on this 8.6 GB machine (the
  per-square tokens and attention for 256 x 12 x 64 keys do not fit). It was
  rerun at **batch 128 x 3000 steps** with 2 loader workers: same puzzles,
  twice the optimizer steps. All other heads: batch 256 x 1500, 4 workers.
- Head details:
  - flat: MLP on the masked-mean pooled position vector (baseline).
  - grouped: softmax (+ "none") per exclusive group from cook.py: outcome,
    length, phase, mateIn, matePattern, endgameType, attackSide (43 labels);
    sigmoid on the remaining 27.
  - q2l (Liu et al. 2021): 70 learnable label queries, 1 transformer decoder
    layer, cross-attention over the **per-square** features of every valid
    position (up to 12 x 64 tokens, square + ply embeddings), per-label
    linear readout.
  - mlgcn (Chen et al. 2019): 2-layer GCN over label embeddings along the
    training co-occurrence graph (tau 0.4, p 0.2) produces per-label
    classifier vectors applied to the pooled vector. Label embeddings
    initialised from conditional co-occurrence rows (no word vectors exist
    for puzzle tags).

## Results (test split)

| head | params | train time | tuned micro-F1 | tuned macro-F1 | tuned macro-P | tuned macro-R | tuned exact | untuned macro-F1 | untuned micro-F1 | untuned exact |
|---|---|---|---|---|---|---|---|---|---|---|
| flat | 0.93M | 13 min | 0.713 | 0.387 | 0.361 | 0.456 | 0.114 | 0.343 | 0.770 | 0.182 |
| grouped | 0.94M | 13 min | 0.704 | 0.421 | 0.394 | 0.488 | 0.131 | 0.364 | 0.778 | 0.223 |
| q2l | 1.13M | 26 min | **0.801** | **0.561** | **0.533** | **0.615** | **0.219** | **0.515** | **0.814** | **0.241** |
| mlgcn | 0.96M | 14 min | 0.722 | 0.410 | 0.379 | 0.484 | 0.116 | 0.373 | 0.768 | 0.168 |

Mean tuned per-class F1 by subset:

| subset | flat | grouped | q2l | mlgcn |
|---|---|---|---|---|
| group outcome (4) | 0.566 | 0.570 | 0.582 | 0.564 |
| group length (4) | 0.996 | 0.996 | 0.996 | 0.996 |
| group phase (3) | 0.879 | 0.887 | 0.930 | 0.885 |
| group mateIn (5) | 0.656 | 0.684 | 0.764 | 0.690 |
| group matePattern (19) | 0.218 | 0.293 | 0.499 | 0.255 |
| group endgameType (6) | 0.752 | 0.892 | 0.956 | 0.776 |
| group attackSide (2) | 0.444 | 0.560 | 0.648 | 0.513 |
| free labels (27) | 0.199 | 0.189 | 0.366 | 0.217 |
| band common (>5%) | 0.702 | 0.704 | 0.781 | 0.710 |
| band mid (0.5-5%) | 0.391 | 0.438 | 0.589 | 0.417 |
| band rare (<0.5%) | 0.144 | 0.189 | 0.366 | 0.176 |
| dead labels (tuned F1 < 0.05) | 7 | 8 | 5 | 7 |

Largest per-label changes vs flat:

- grouped: smotheredMate +0.28, knightEndgame +0.27, arabianMate +0.26,
  queenRookEndgame +0.24, queensideAttack +0.19; promotion -0.11,
  advancedPawn -0.05.
- q2l: anastasiaMate +0.64, arabianMate +0.51, doubleBishopMate +0.44,
  promotion +0.44, advancedPawn +0.40; no label down more than 0.02.
- mlgcn: knightEndgame +0.18, anastasiaMate +0.14, queensideAttack +0.14;
  queenRookEndgame -0.12, dovetailMate -0.07.

## Reading

- **Query2Label is in a different league: +0.17 tuned macro-F1 and +0.09
  micro-F1 over flat**, better on every group, every band, and 65 of 70
  labels, with only 5 dead labels. Its untuned 0.5-threshold macro-F1
  (0.515) beats every other head's *tuned* number.
- **The likely reason is spatial access, not label modelling.** q2l is the
  only head that sees per-square features; flat, grouped and mlgcn all read
  a globally average-pooled 128-d vector per position. The biggest q2l
  gains are location-dependent themes: promotion and advancedPawn (+0.4,
  "is there a pawn on the 7th rank"), the mate patterns (king position and
  the mating net's geometry). Global average pooling erases exactly that.
  mlgcn, which models label structure but still reads the pooled vector,
  lands close to flat/grouped, which supports this reading.
- **Grouped softmax does what the structure predicts and no more**: large
  gains on endgameType, matePattern and attackSide, nothing on the already
  solved groups, a small loss on free labels (the group CE terms compete
  for the shared encoder). +0.034 macro-F1 overall.
- **ML-GCN is a modest gain** (+0.023) spread thinly; with the dominant
  dependencies already deterministic (mateInN -> mate, X -> endgame) the
  co-occurrence graph adds little the encoder cannot learn.
- Confounds on q2l: twice the optimizer steps (batch 128 x 3000) and 20%
  more parameters. Neither plausibly explains a 0.17 gap; the flat head at
  2x steps is a cheap control if wanted.

## Controls (added 2026-10-12, commit afe0643)

Two runs at Query2Label's exact protocol (batch 128 x 3000 steps, 2 loader
workers, seed 0) to separate the two confounds: `flat` again, so the extra
optimizer steps are measured alone, and `flat_spatial`, the pooled MLP head
fed each position's flattened (D, 8, 8) map through a linear projection,
which keeps square identity but has no attention or label queries.

| run | params | tuned micro-F1 | tuned macro-F1 | tuned macro-P | tuned macro-R | untuned macro-F1 | untuned micro-F1 |
|---|---|---|---|---|---|---|---|
| flat, batch 256 x 1500 | 0.93M | 0.713 | 0.387 | 0.361 | 0.456 | 0.343 | 0.770 |
| flat, batch 128 x 3000 | 0.93M | 0.701 | 0.410 | 0.379 | 0.496 | 0.365 | 0.774 |
| flat_spatial, batch 128 x 3000 | 1.98M | 0.734 | 0.442 | 0.415 | 0.507 | 0.402 | 0.777 |
| q2l, batch 128 x 3000 | 1.13M | **0.801** | **0.561** | **0.533** | **0.615** | **0.515** | **0.814** |

| subset | flat | flat b128 | flat_spatial | q2l |
|---|---|---|---|---|
| matePattern (19) | 0.218 | 0.257 | 0.334 | 0.499 |
| endgameType (6) | 0.752 | 0.807 | 0.865 | 0.956 |
| free labels (27) | 0.199 | 0.208 | 0.225 | 0.366 |
| band common | 0.702 | 0.707 | 0.714 | 0.781 |
| band mid | 0.391 | 0.417 | 0.450 | 0.589 |
| band rare | 0.144 | 0.178 | 0.226 | 0.366 |
| promotion + advancedPawn | 0.440 | 0.455 | 0.572 | 0.863 |
| dead labels | 7 | 7 | 7 | 5 |

### Decomposition of the 0.174 gap (flat -> q2l, tuned macro-F1)

- Extra optimizer steps (batch 128 x 3000 vs 256 x 1500): **+0.023**
- Keeping square identity (flat_spatial vs flat b128): **+0.032**
- Attention with label queries (q2l vs flat_spatial): **+0.119**

So the pooling hypothesis is only partly right. Removing global average
pooling helps, most visibly on the location-dependent themes (promotion +
advancedPawn 0.46 -> 0.57), but a fixed linear read of the square map
captures a small fraction of what label-query cross-attention does
(0.57 -> 0.86 on the same pair). The decoder is doing the work: each
label query can look at a different square of a different position, which
a single projection per position cannot. Note flat_spatial has 1.75x the
parameters of q2l and still trails it by 0.12.

Implication: a cheaper head is not an option; the attention over
per-square, per-position tokens is the mechanism. Attention over the
*positions* of the pooled vectors (the existing `--pooling attention`) is
not the same thing and was never a candidate explanation.

## Query2Label + grouped softmax readout (added 2026-10-12, commit 0ab6f4a)

`--head q2l_grouped`: the same decoder with 7 extra "none" queries, one per
exclusive group; group logits go through the grouped cross-entropy, the 27
free labels through the sigmoid loss. Query2Label's protocol (batch 128 x
3000, seed 0).

| head | params | tuned micro-F1 | tuned macro-F1 | tuned macro-P | tuned macro-R | tuned exact | untuned macro-F1 | untuned micro-F1 | untuned exact |
|---|---|---|---|---|---|---|---|---|---|
| grouped (pooled) | 0.94M | 0.704 | 0.421 | 0.394 | 0.488 | 0.131 | 0.364 | 0.778 | 0.223 |
| q2l | 1.13M | **0.801** | 0.561 | 0.533 | 0.615 | 0.219 | 0.515 | **0.814** | 0.241 |
| q2l_grouped | 1.14M | 0.799 | **0.573** | **0.544** | **0.628** | **0.235** | **0.540** | 0.813 | **0.274** |

| subset | grouped (pooled) | q2l | q2l_grouped |
|---|---|---|---|
| group outcome (4) | 0.570 | 0.582 | 0.596 |
| group length (4) | 0.996 | 0.996 | 0.996 |
| group phase (3) | 0.887 | 0.930 | 0.929 |
| group mateIn (5) | 0.684 | 0.764 | **0.850** |
| group matePattern (19) | 0.293 | 0.499 | **0.583** |
| group endgameType (6) | 0.892 | 0.956 | **0.971** |
| group attackSide (2) | 0.560 | 0.648 | **0.699** |
| free labels (27) | 0.189 | **0.366** | 0.312 |
| band common | 0.704 | **0.781** | 0.774 |
| band mid | 0.438 | **0.589** | 0.569 |
| band rare | 0.189 | 0.366 | **0.426** |
| dead labels | 8 | **5** | 6 |

Largest changes, q2l_grouped vs q2l: mateIn5 +0.35, bodenMate +0.26,
arabianMate +0.18, vukovicMate +0.14, morphysMate +0.14; enPassant -0.29,
hangingPiece -0.17, exposedKing -0.16, swallowstailMate -0.11, xRayAttack -0.10.

### Reading

- Net +0.012 tuned macro-F1 and +0.025 untuned, with the best untuned exact
  match of any run (0.274). The grouped readout is the better calibrated
  one at 0.5, as it was on the pooled head.
- The gain is where the structure lives: mateIn +0.09, matePattern +0.08,
  attackSide +0.05, rare band +0.06. The mateIn5 jump (+0.35) is the
  softmax forcing exactly one depth per mate.
- The cost is the same as before but larger: free labels drop 0.05, led by
  enPassant, hangingPiece and exposedKing. The 7 group cross-entropy terms
  at group_weight 1 pull the shared decoder and encoder toward the groups.
  `--group-weight 0.5` is the obvious next knob and has not been tried.
- At one seed, +0.012 is inside the noise seen in the loss comparison
  (spread up to 0.008), so the headline is "no worse, better calibrated,
  gains and losses on different label sets", not a clear win.

## Decision

- Query2Label is the default head (commit afe0643, approved 2026-10-12).
- The pooling control is done (above): the decoder's cross-attention, not
  merely un-pooling, is what delivers the gain.
- q2l + grouped readout tried (above): +0.012 macro-F1, better calibration,
  free labels regress. Default stays q2l pending a group-weight sweep or a
  second seed.
- Remaining follow-ups:
  1. q2l_grouped with --group-weight 0.5 (and seed 1 for both heads).
  2. q2l at batch 256 / seed 1 on a GPU host; q2l decoder depth 2.
  3. Co-occurrence confusion matrix and rank evaluation on the q2l model.

## Caveats

Single seed, 7% of one epoch, 128-wide encoder. q2l's 26 min on this
machine (vs 13) is partly memory pressure; the attention itself is cheap.
