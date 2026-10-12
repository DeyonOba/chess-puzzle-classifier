"""Co-occurrence confusion and ranking analysis of a checkpoint on the test split.

    python -m src.analyze runs/NAME [--test-batches 780] [--batch-size 128]

Uses the per-class thresholds in runs/NAME/eval.json (tuned on val by
src.evaluate). Writes runs/NAME/analysis.json and runs/NAME/analysis.md with:

1. Confusion matrix inside each mutually exclusive tag group (argmax over the
   group's probabilities vs 'none'), where confusion is well defined.
2. Substitution table for every label: for its false negatives, which labels
   were false positives in the same puzzle (what the model said instead).
3. Co-occurrence drift: label pairs whose predicted joint rate differs most
   from the true joint rate on test.
4. Per-puzzle ranking metrics (P@k, hit@k, one-error, coverage, LRAP,
   ranking loss, macro AP) and the labels with the lowest AP.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from src.data.components.splits import load_splits, make_loader
from src.data.components.dataset import PuzzleSequenceDataset
from src.evaluate import load_model
from src.train import pick_device, predict
from src.utils.label_groups import EXCLUSIVE_GROUPS
from src.utils.label_groups import LabelLayout
from src.utils.ranking import per_label_average_precision, ranking_report
from src.utils.thresholds import decode_grouped, f1_at_thresholds


def group_confusions(probs, targets, labels):
    """Within-group confusion: rows = true class, cols = predicted (argmax incl. 'none')."""
    index = {l: i for i, l in enumerate(labels)}
    out = {}
    for name, members in EXCLUSIVE_GROUPS.items():
        idx = [index[m] for m in members if m in index]
        if not idx:
            continue
        names = [labels[i] for i in idx] + ["none"]
        p = probs[:, idx]
        # predicted class: argmax member if its prob exceeds the 'none' mass (1 - sum), else 'none'
        none_mass = np.clip(1 - p.sum(1), 0, 1)
        pred = np.where(p.max(1) > none_mass, p.argmax(1), len(idx))
        t = targets[:, idx]
        true = np.where(t.sum(1) > 0, t.argmax(1), len(idx))
        k = len(idx) + 1
        cm = np.zeros((k, k), dtype=np.int64)
        np.add.at(cm, (true, pred), 1)
        out[name] = {"classes": names, "matrix": cm.tolist(), "accuracy": float(np.trace(cm) / cm.sum())}
    return out


def substitutions(preds, targets, labels, top=5, min_fn=20):
    """For each label: false-negative count and the labels most often falsely predicted in those puzzles."""
    fn = targets.astype(bool) & ~preds
    fp = preds & ~targets.astype(bool)
    out = {}
    for j, l in enumerate(labels):
        rows = fn[:, j]
        n = int(rows.sum())
        if n < min_fn:
            continue
        co = fp[rows].sum(0)
        order = np.argsort(-co)[:top]
        out[l] = {"false_negatives": n, "instead": [(labels[i], int(co[i])) for i in order if co[i] > 0]}
    return out


def cooccurrence_drift(preds, targets, labels, top=15):
    """Pairs whose predicted joint rate differs most from the true joint rate."""
    n = len(targets)
    T = targets.astype(np.float64); P = preds.astype(np.float64)
    true_joint = (T.T @ T) / n
    pred_joint = (P.T @ P) / n
    diff = pred_joint - true_joint
    iu = np.triu_indices(len(labels), k=1)
    order = np.argsort(-np.abs(diff[iu]))[:top]
    return [{"pair": [labels[iu[0][k]], labels[iu[1][k]]], "true_rate": float(true_joint[iu][k]),
             "pred_rate": float(pred_joint[iu][k]), "diff": float(diff[iu][k])} for k in order]


def to_markdown(result: dict) -> str:
    lines = [f"# Analysis: {result['run']} (test n={result['n_test']:,})", ""]
    lines += ["## Ranking metrics", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in result["ranking"].items()]
    lines += ["", "Lowest-AP labels:", ""] + [f"- {l}: AP {ap:.3f} (n_test={n})" for l, ap, n in result["lowest_ap"]]
    lines += ["", "## Within-group confusion (rows = true, cols = predicted)", ""]
    for g, c in result["group_confusion"].items():
        lines += [f"### {g} (accuracy {c['accuracy']:.3f})", "", "| true \\ pred | " + " | ".join(c["classes"]) + " |", "|---|" + "---|" * len(c["classes"])]
        lines += [f"| {cls} | " + " | ".join(f"{v:,}" for v in row) + " |" for cls, row in zip(c["classes"], c["matrix"])]
        lines.append("")
    lines += ["## Substitutions (what was predicted instead, on each label's false negatives)", ""]
    for l, s in sorted(result["substitutions"].items(), key=lambda kv: -kv[1]["false_negatives"]):
        inst = ", ".join(f"{a} ({b})" for a, b in s["instead"]) or "nothing (no false positives)"
        lines.append(f"- **{l}** ({s['false_negatives']:,} FN): {inst}")
    lines += ["", "## Co-occurrence drift (predicted joint rate vs true joint rate)", "", "| pair | true | predicted | diff |", "|---|---|---|---|"]
    lines += [f"| {a} + {b} | {d['true_rate']:.4f} | {d['pred_rate']:.4f} | {d['diff']:+.4f} |" for d in result["drift"] for a, b in [d["pair"]]]
    c = result.get("consistent_decoding")
    if c:
        lines += ["", "## Grouped head: independent thresholds vs argmax-within-group decoding (test)", "",
                  "| decoding | micro-F1 | macro-F1 | macro-P | macro-R | exact match |", "|---|---|---|---|---|---|"]
        for name, m in (("independent per-class thresholds", c["independent_thresholds"]), ("argmax within groups + thresholds on free labels", c["argmax_groups"])):
            lines.append(f"| {name} | {m['micro_f1']:.4f} | {m['macro_f1']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['exact_match']:.4f} |")
        lines += ["", "Drift under argmax decoding:", "", "| pair | true | predicted | diff |", "|---|---|---|---|"]
        lines += [f"| {a} + {b} | {d['true_rate']:.4f} | {d['pred_rate']:.4f} | {d['diff']:+.4f} |" for d in c["drift_argmax_groups"] for a, b in [d["pair"]]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--h5", default="data/puzzle_boards.h5")
    p.add_argument("--splits", default="data/splits.npz")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--reuse-probs", action="store_true", help="reuse runs/NAME/test_probs.npz from a previous analysis")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    device = pick_device(args.device)
    model, cfg, _ = load_model(run_dir, device, args.checkpoint)
    labels = cfg["labels"]
    ev = json.loads((run_dir / "eval.json").read_text())
    thresholds = np.array([ev["per_class"][l]["threshold"] for l in labels], dtype=np.float32)

    dataset = PuzzleSequenceDataset(args.h5, cfg["T"])
    splits = load_splits(args.splits)
    cache = run_dir / "test_probs.npz"
    if args.reuse_probs and cache.exists():
        z = np.load(cache); probs, targets = z["probs"], z["targets"]
    else:
        probs, targets = predict(model, make_loader(dataset, splits["test"], args.batch_size, False, args.num_workers), device, args.test_batches)
        np.savez_compressed(cache, probs=probs, targets=targets)
    preds = probs >= thresholds[None]
    consistent = None
    if getattr(model, "grouped", False):
        # grouped heads: compare independent per-class thresholds with argmax-within-group decoding
        preds_c = decode_grouped(probs, thresholds, LabelLayout(labels))
        m_ind = f1_at_thresholds(probs, targets, thresholds)
        m_con = {k: v for k, v in f1_at_thresholds(probs, targets, thresholds).items() if k != "per_class_f1"}
        tp = (preds_c & targets.astype(bool)).sum(0); fp = (preds_c & ~targets.astype(bool)).sum(0); fn = (~preds_c & targets.astype(bool)).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
            prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0); rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        m_con = {"micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)), "macro_f1": float(f1.mean()),
                 "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
                 "exact_match": float((preds_c == targets.astype(bool)).all(1).mean())}
        consistent = {"independent_thresholds": {k: v for k, v in m_ind.items() if k != "per_class_f1"}, "argmax_groups": m_con,
                      "drift_argmax_groups": cooccurrence_drift(preds_c, targets, labels, top=8)}

    ap = per_label_average_precision(probs, targets)
    n_pos = targets.sum(0)
    lowest = sorted(((ap[j], labels[j], int(n_pos[j])) for j in range(len(labels)) if not np.isnan(ap[j])))[:10]
    result = {
        "run": run_dir.name, "n_test": int(len(targets)),
        "ranking": ranking_report(probs, targets),
        "lowest_ap": [(l, float(a), n) for a, l, n in lowest],
        "group_confusion": group_confusions(probs, targets, labels),
        "substitutions": substitutions(preds, targets, labels),
        "drift": cooccurrence_drift(preds, targets, labels),
        "per_label_ap": {l: (None if np.isnan(ap[j]) else float(ap[j])) for j, l in enumerate(labels)},
        "consistent_decoding": consistent,
    }
    (run_dir / "analysis.json").write_text(json.dumps(result, indent=2))
    (run_dir / "analysis.md").write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == "__main__":
    main()
