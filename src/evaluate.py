"""Evaluate a checkpoint: tune per-class thresholds on val, score on test.

    python -m src.evaluate runs/NAME [--h5 ...] [--splits ...] [--val-batches 390] [--test-batches 390]

Writes runs/NAME/eval.json with metrics at the fixed 0.5 threshold and at the
val-tuned thresholds, both measured on the test split, plus the thresholds.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from src.data.datamodule import load_splits, make_loader
from src.data.dataset import PuzzleSequenceDataset
from src.models.puzzle_classifier import PuzzleSequenceClassifier
from src.train import pick_device, predict
from src.utils.thresholds import f1_at_thresholds, tune_thresholds


def load_model(run_dir: pathlib.Path, device, checkpoint="best.pt"):
    state = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = PuzzleSequenceClassifier(
        num_labels=cfg["num_labels"], hidden_dim=cfg["hidden_dim"], num_res_blocks=cfg["res_blocks"],
        max_positions=cfg["T"], pooling=cfg["pooling"], dropout=cfg["dropout"],
        head=cfg.get("head", "flat"), labels=cfg["labels"],
    )
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg, state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--h5", default="data/puzzle_boards.h5")
    p.add_argument("--splits", default="data/splits.npz")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-batches", type=int, default=None, help="cap val batches used for tuning")
    p.add_argument("--test-batches", type=int, default=None, help="cap test batches used for scoring")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    run_dir = pathlib.Path(args.run_dir)
    model, cfg, state = load_model(run_dir, device, args.checkpoint)
    dataset = PuzzleSequenceDataset(args.h5, cfg["T"])
    splits = load_splits(args.splits)

    val_probs, val_t = predict(model, make_loader(dataset, splits["val"], args.batch_size, False, args.num_workers), device, args.val_batches)
    thresholds = tune_thresholds(val_probs, val_t)
    test_probs, test_t = predict(model, make_loader(dataset, splits["test"], args.batch_size, False, args.num_workers), device, args.test_batches)

    fixed = f1_at_thresholds(test_probs, test_t, np.full(len(thresholds), 0.5, dtype=np.float32))
    tuned = f1_at_thresholds(test_probs, test_t, thresholds)
    labels = dataset.labels
    result = {
        "run": run_dir.name, "checkpoint": args.checkpoint, "epoch": state.get("epoch"),
        "n_val_tune": int(len(val_t)), "n_test": int(len(test_t)),
        "test_fixed_0.5": {k: v for k, v in fixed.items() if k != "per_class_f1"},
        "test_val_tuned": {k: v for k, v in tuned.items() if k != "per_class_f1"},
        "per_class": {l: {"threshold": float(th), "f1_fixed": float(a), "f1_tuned": float(b)}
                      for l, th, a, b in zip(labels, thresholds, fixed["per_class_f1"], tuned["per_class_f1"])},
    }
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    f, t = result["test_fixed_0.5"], result["test_val_tuned"]
    print(f"{run_dir.name}: test n={len(test_t):,} (thresholds tuned on {len(val_t):,} val)")
    print(f"  fixed 0.5 : micro_f1={f['micro_f1']:.4f} macro_f1={f['macro_f1']:.4f} P={f['macro_precision']:.4f} R={f['macro_recall']:.4f} exact={f['exact_match']:.4f}")
    print(f"  val-tuned : micro_f1={t['micro_f1']:.4f} macro_f1={t['macro_f1']:.4f} P={t['macro_precision']:.4f} R={t['macro_recall']:.4f} exact={t['exact_match']:.4f}")


if __name__ == "__main__":
    main()
