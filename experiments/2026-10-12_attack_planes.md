# Attack-map input planes

Date: 2026-10-12. Commit 3aa0117. Run `v2x_q2lg_gw05_s0` vs `v2_q2lg_gw05_s0`.

Hypothesis: the encoder spends capacity rediscovering piece interactions
(attacks, pins, hanging pieces) from piece placement; giving them as input
planes frees it for the tags. Eight exact python-chess planes were appended to
the 18 base planes (`bitboard_encoder.extra_bitboards`): attacked-by-white,
attacked-by-black, pinned white / black pieces, hanging white / black pieces
(attacked and undefended, outnumbered, or attacked by a cheaper piece), the
checking pieces, in-check.

Protocol: `experiment=attack_planes_laptop` = the bounded laptop protocol
(384k training puzzles, 128x3 encoder, q2l_grouped gw 0.5, weighted BCE sqrt,
seed 0, thresholds tuned on 100k val, scored on 100k test) on the v2x file;
same rows and splits as the base v2 run.

| input | tuned micro-F1 | tuned macro-F1 | tuned exact | argmax macro-F1 | argmax exact | dead labels |
|---|---|---|---|---|---|---|
| base planes (18) | 0.859 | 0.586 | 0.342 | 0.563 | 0.322 | 2 |
| attack-map planes (26) | 0.894 | 0.640 | 0.444 | 0.614 | 0.419 | 3 |

Per subset (tuned F1, base -> attack-map): free tactical labels 0.366 -> 0.479,
mateIn 0.848 -> 0.910, matePattern 0.549 -> 0.562, attackSide 0.698 -> 0.721,
outcome 0.651 -> 0.667, common 0.808 -> 0.875, mid 0.596 -> 0.674, rare
0.416 -> 0.435; length / endgameType / phase unchanged.

Largest gains: doubleCheck +0.55, pin +0.54, discoveredCheck +0.29,
hangingPiece +0.20, fork +0.18, defensiveMove +0.18, vukovicMate +0.16,
skewer +0.15. Losses: swallowstailMate -0.15, castling -0.12, killBoxMate
-0.12, smotheredMate -0.08, collinearMove -0.06 (all under 100 test
positives; single-seed noise).

Reading: the gains land exactly on the tags whose definition is the plane
(pin, hanging piece, double / discovered check, fork via attack sets). The
+0.054 macro-F1 is the largest single improvement since Query2Label and costs
no parameters; encoding is 4.5x slower (17 min for 6.16M puzzles) and the
file is 1.8 GB instead of 1.26 GB.

Decision pending: make `--planes extended` the default data path (configs,
Makefile, Kaggle dataset) and restart the Kaggle full-data run on it.
