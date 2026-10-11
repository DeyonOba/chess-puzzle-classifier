"""Compare training runs side by side.

    python -m src.utils.compare_runs runs/losscmp_line_* [--top 8]

Prints the final-epoch validation metrics of each run, then the labels whose
per-class F1 moved most against the first run given (the reference), grouped
by frequency band so rare-label effects are visible.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def load_run(run_dir: pathlib.Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    hist = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines() if l.strip()]
    per_class = json.loads((run_dir / "per_class_f1.json").read_text()) if (run_dir / "per_class_f1.json").exists() else {}
    return {"name": run_dir.name, "config": cfg, "last": hist[-1] if hist else {}, "per_class": per_class}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--meta", default="data/puzzle_metadata.json", help="for label counts (frequency bands)")
    args = p.parse_args()
    runs = [load_run(pathlib.Path(r)) for r in args.runs]

    cols = ["val_micro_f1", "val_macro_f1", "val_macro_precision", "val_macro_recall", "val_macro_auprc", "val_exact_match", "val_loss", "train_loss"]
    print(f"{'run':28s}" + "".join(f"{c.replace('val_', ''):>12s}" for c in cols))
    for r in runs:
        print(f"{r['name']:28s}" + "".join(f"{r['last'].get(c, float('nan')):12.4f}" for c in cols))

    counts = {}
    if pathlib.Path(args.meta).exists():
        counts = json.loads(pathlib.Path(args.meta).read_text())["counts"]
    total = sum(counts.values()) and max(counts.values())
    bands = [("common >5%", 0.05, 1.0), ("mid 0.5-5%", 0.005, 0.05), ("rare <0.5%", 0.0, 0.005)]
    ref = runs[0]
    if not ref["per_class"]:
        return
    N = json.loads(pathlib.Path(args.meta).read_text())["num_samples"] if counts else None
    print(f"\nPer-class F1 by frequency band (mean over labels in band); reference = {ref['name']}")
    print(f"{'band':14s}{'n':>4s}" + "".join(f"{r['name'][-14:]:>16s}" for r in runs))
    for band, lo, hi in bands:
        labels = [l for l in ref["per_class"] if counts and lo <= counts[l] / N < hi] if counts else list(ref["per_class"])
        if not labels:
            continue
        print(f"{band:14s}{len(labels):4d}" + "".join(f"{sum(r['per_class'].get(l, 0) for l in labels) / len(labels):16.4f}" for r in runs))

    for r in runs[1:]:
        deltas = sorted(((r["per_class"].get(l, 0) - ref["per_class"][l], l) for l in ref["per_class"]), reverse=True)
        print(f"\n{r['name']} vs {ref['name']}: largest per-class F1 changes")
        for d, l in deltas[: args.top]:
            print(f"  +{d:.3f}  {l:20s} ({ref['per_class'][l]:.3f} -> {r['per_class'][l]:.3f}, n={counts.get(l, '?')})")
        for d, l in deltas[-args.top:]:
            print(f"  {d:+.3f}  {l:20s} ({ref['per_class'][l]:.3f} -> {r['per_class'][l]:.3f}, n={counts.get(l, '?')})")


if __name__ == "__main__":
    main()
