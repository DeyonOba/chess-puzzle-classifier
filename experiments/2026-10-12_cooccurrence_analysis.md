# Co-occurrence confusion and ranking analysis

Date: 2026-10-12. Tool: `python -m src.analyze runs/NAME` (commit a5ca805,
decoding comparison in the following commit). Full reports with every
confusion matrix and substitution list: `runs/<run>/analysis.md`.

Model analysed: **q2l_grouped, group_weight 0.5** (the recommended default,
tuned test macro-F1 0.580), with the plain Query2Label head for comparison.
Test split, 99,840 puzzles, per-class thresholds tuned on val.

## 1. Ranking quality (threshold-free)

| metric | q2l | q2l_grouped gw=0.5 |
|---|---|---|
| precision@1 | 0.997 | 0.996 |
| precision@3 | 0.912 | 0.906 |
| hit@3 | 0.999 | 1.000 |
| one-error | 0.003 | 0.004 |
| coverage error (mean 4.5 true labels) | 6.55 | 6.75 |
| LRAP | 0.924 | 0.917 |
| ranking loss | 0.010 | 0.011 |
| macro AP | 0.560 | 0.576 |

The top of the ranking is essentially perfect: the highest-scored label is
right 99.6% of the time and all true labels sit within the top ~7 on
average. Everything the F1 numbers lose, they lose at the decision
threshold and on rare labels, not in the ordering. The grouped head ranks
very slightly worse per puzzle and better per label (macro AP +0.016).

Lowest-AP labels (gw=0.5): underPromotion 0.005 (17 test positives),
equality 0.006, interference 0.036, balestraMate 0.037, castling 0.048,
intermezzo 0.049, collinearMove 0.058, capturingDefender 0.064. These are
the genuinely unlearned themes at this budget: an engine-eval boundary
(equality), three rare move-type tags (underPromotion, castling, enPassant
at 0.008 for q2l), and four tags about subtle line semantics.

## 2. Confusion inside the exclusive groups (gw=0.5, argmax vs 'none')

| group | accuracy | what the errors are |
|---|---|---|
| length | 0.991 | only 'none' rows (puzzles with no length tag) |
| endgameType | 0.993 | a few hundred false endgame types on 'none' rows; no type-vs-type confusion |
| phase | 0.97 | middlegame -> opening 1.4k; opening -> middlegame 0.4k |
| mateIn | 0.955 | **zero** depth-vs-depth confusion; all errors are mate-vs-none: 614 mateIn2 and 174 mateIn3 missed, 1.9k / 1.1k false mateIn2 / mateIn3 |
| matePattern | 0.936 | pattern-vs-none dominates; the pattern-vs-pattern confusions are geometric neighbours (below) |
| attackSide | 0.916 | 5.6k false kingsideAttack, 1.0k missed; kingside vs queenside never confused |
| outcome | 0.753 | **crushing <-> advantage**: 12.8k crushing predicted as advantage, 7.2k the reverse; mate vs non-mate is 97% clean; equality (227) never predicted |

Mate pattern confusions worth knowing (true -> predicted, gw=0.5):
pillsburysMate -> operaMate 89 and back 20; bodenMate -> doubleBishopMate
11, doubleBishopMate -> morphysMate 19, bodenMate -> morphysMate 8 (the
bishop-pair nets); dovetailMate -> swallowstailMate 17, epauletteMate ->
swallowstailMate 14 and hookMate 13 (the king-side nets). balestraMate: 21
of 22 go to 'none'.

Reading: the groups split into three kinds. Length, endgame type, phase and
mate depth are read off the input almost mechanically (depth is the
sequence length once the line ends in mate, which is why depth is never
confused with another depth). The outcome group's crushing/advantage
boundary is an engine centipawn threshold (600) that the board does not
expose; 20k of the 25k outcome errors are that one boundary, and it caps
outcome accuracy near 75% regardless of head. The mate patterns confuse
with their geometric neighbours, which is the expected failure of a model
that has learned the shape but not cook.py's exact square tests.

## 3. Substitutions: what is predicted instead of a missed label

For every label's false negatives, the labels most often falsely predicted
in the same puzzle (gw=0.5, top rows):

- crushing (6.2k FN): advantage 4.9k, mate 1.1k, mateIn2 0.8k
- fork (5.4k FN): advantage 1.6k, crushing 1.5k, pin 1.0k, deflection 0.7k
- advantage (3.9k FN): crushing 3.1k, mate 0.6k, deflection 0.6k
- pin (3.6k FN): advantage 1.1k, crushing 0.9k, fork 0.5k
- discoveredAttack (3.2k FN): advantage 1.0k, crushing 0.8k, pin 0.5k, fork 0.5k
- zugzwang (q2l, 382 FN): defensiveMove 214, quietMove 162

Two things are visible. First, the outcome swap again (crushing <->
advantage is the top substitution in both directions). Second, the
tactical misses are mostly not substitutions at all: the "instead" labels
for fork, pin and discoveredAttack are the ubiquitous outcome/phase tags,
i.e. the model said nothing tactical. The one true tactical substitution
is zugzwang -> defensiveMove/quietMove, which is semantically close
(cook.py's zugzwang is a quiet defensive move in a pawn endgame).

## 4. Co-occurrence drift and the decoding question

Pairs whose predicted joint rate is furthest from the true joint rate
(gw=0.5, independent per-class thresholds):

| pair | true | predicted |
|---|---|---|
| advantage + crushing | 0.000 | **0.286** |
| advantage + middlegame | 0.174 | 0.281 |
| advantage + short | 0.165 | 0.256 |
| crushing + endgame | 0.251 | 0.337 |
| advantage + pin | 0.024 | 0.097 |
| crushing + defensiveMove | 0.043 | 0.115 |

The model predicts both advantage and crushing on 29% of puzzles although
they never co-occur, and this is true of the *grouped* head too. Cause:
per-class F1-optimal thresholds. When the outcome softmax splits mass
~0.45/0.45 between the two, both exceed their (low) tuned thresholds. The
grouped head's exclusivity exists in the probabilities and is thrown away
at decode time. The same mechanism inflates every pair involving
advantage, crushing or pin, which have the lowest tuned thresholds.

Argmax-within-group decoding (one label per group, thresholds only on the
27 free labels) fixes the consistency and changes the headline numbers:

| decoding (gw=0.5, test) | micro-F1 | macro-F1 | macro-P | macro-R | exact match |
|---|---|---|---|---|---|
| independent per-class thresholds | **0.815** | **0.580** | **0.559** | 0.624 | 0.252 |
| argmax within groups + thresholds on free labels | 0.809 | 0.555 | 0.504 | **0.658** | **0.272** |

Under argmax the impossible pairs disappear from the drift table entirely
and exact match rises 2 points, but macro-F1 drops 2.5 points. The
per-label thresholds were buying F1 on crushing and advantage by hedging:
predicting both when unsure is rewarded by per-label F1 and punished only
by exact match. So the two decodings optimise different things.

## Decisions and follow-ups

- **Report both decodings** for grouped heads from now on. For a deployed
  tagger that emits tag sets, argmax decoding is the one to ship: it never
  outputs contradictory tags and has the best exact match. The macro-F1
  leaderboard stays on independent thresholds for comparability with
  earlier runs. (evaluate.py currently reports only the independent one;
  adding the argmax row is a small follow-up.)
- The crushing/advantage boundary is the largest single error source and
  is not a modelling problem. Options are to accept it, to merge the two
  into one "winning" tag for the headline metric, or to add the engine
  evaluation as an input feature if the use case allows.
- The free-label misses (fork 5.4k, pin 3.6k, discoveredAttack 3.2k) are
  "said nothing" misses, which is a recall/threshold problem on the
  largest tactical labels and the natural next target: a second seed and
  more than 7% of an epoch will move these more than any head change.
- Unlearned set to watch: equality, underPromotion, castling, enPassant,
  interference, intermezzo, collinearMove, capturingDefender.
