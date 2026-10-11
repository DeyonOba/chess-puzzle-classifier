"""Train the puzzle theme classifier.

Examples:
    # line-conditioned (all solution positions)
    python -m src.train --h5 data/puzzle_boards.h5 --splits data/splits.npz --run-name line_t12

    # position-only baseline from the same file
    python -m src.train --h5 data/puzzle_boards.h5 --splits data/splits.npz --max-positions 1 --run-name pos_only

Each run writes runs/<run-name>/{config.json, history.jsonl, best.pt, last.pt, per_class_f1.json,
thresholds.json}. Model selection uses val macro-F1 with per-class thresholds tuned in-sample on val.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.datamodule import load_splits, make_loader
from src.data.dataset import PuzzleSequenceDataset
from src.models.puzzle_classifier import PuzzleSequenceClassifier
from src.utils.losses import LOSS_NAMES, build_loss
from src.utils.metrics import MultiLabelPuzzleMetrics
from src.utils.thresholds import f1_at_thresholds, tune_thresholds


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device, limit_batches=None):
    """Returns (probs, targets) as float32 / uint8 numpy arrays over the loader."""
    model.eval()
    probs, targets_all = [], []
    for i, (boards, lengths, targets) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        logits = model(boards.to(device, non_blocking=True), lengths.to(device))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets_all.append(targets.numpy().astype(np.uint8))
    return np.concatenate(probs), np.concatenate(targets_all)


@torch.no_grad()
def evaluate(model, loader, metrics: MultiLabelPuzzleMetrics, device, loss_fn, limit_batches=None):
    """Metrics at the fixed 0.5 threshold plus in-sample tuned per-class thresholds.

    Keys prefixed `tuned_` are optimistic (thresholds fitted on the same data);
    use src.evaluate for val-tuned / test-scored numbers.
    """
    probs, targets = predict(model, loader, device, limit_batches)
    metrics.reset()
    logits = torch.logit(torch.from_numpy(probs).clamp(1e-7, 1 - 1e-7))
    t = torch.from_numpy(targets)
    metrics.update(logits, t)
    out = metrics.compute()
    out["loss"] = loss_fn(logits, t.float()).item()
    thresholds = tune_thresholds(probs, targets)
    tuned = f1_at_thresholds(probs, targets, thresholds)
    out.update({f"tuned_{k}": v for k, v in tuned.items() if k != "per_class_f1"})
    out["_thresholds"] = thresholds
    out["_tuned_per_class_f1"] = tuned["per_class_f1"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", default="data/puzzle_boards.h5")
    p.add_argument("--splits", default="data/splits.npz")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--max-positions", type=int, default=None, help="1 = position-only; default = all in file")
    p.add_argument("--pooling", choices=["mean", "attention"], default="mean")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--res-blocks", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-batches", type=int, default=None, help="cap train batches per epoch (bounded runs / smoke tests)")
    p.add_argument("--val-batches", type=int, default=None, help="cap val batches (default: same as --limit-batches)")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--loss", choices=LOSS_NAMES, default="asl")
    p.add_argument("--pos-weight-cap", type=float, default=20.0, help="weighted_bce: clip pos_weight to [1, cap]")
    p.add_argument("--pos-weight-power", type=float, default=1.0, help="weighted_bce: (neg/pos)**power; 0.5 = sqrt")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=None)
    p.add_argument("--asl-gamma-neg", type=float, default=4.0)
    p.add_argument("--asl-gamma-pos", type=float, default=1.0)
    p.add_argument("--asl-clip", type=float, default=0.05)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    run_dir = pathlib.Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = PuzzleSequenceDataset(args.h5, args.max_positions)
    splits = load_splits(args.splits)
    train_loader = make_loader(dataset, splits["train"], args.batch_size, True, args.num_workers, args.seed, drop_last=True)
    val_loader = make_loader(dataset, splits["val"], args.batch_size, False, args.num_workers)

    model = PuzzleSequenceClassifier(
        num_labels=dataset.num_labels, hidden_dim=args.hidden_dim, num_res_blocks=args.res_blocks,
        max_positions=dataset.max_positions, pooling=args.pooling, dropout=args.dropout,
    ).to(device)
    train_targets = dataset.read_targets()[splits["train"]] if args.loss == "weighted_bce" else None
    loss_fn = build_loss(
        args.loss, train_targets, args.pos_weight_cap, args.pos_weight_power, args.focal_gamma, args.focal_alpha,
        args.asl_gamma_neg, args.asl_gamma_pos, args.asl_clip,
    ).to(device)
    del train_targets
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) if args.limit_batches is None else min(len(train_loader), args.limit_batches)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs)
    metrics = MultiLabelPuzzleMetrics(dataset.num_labels)
    val_loss_fn = nn.BCEWithLogitsLoss()  # val loss is always plain BCE so runs with different train losses are comparable
    use_amp = args.amp and device.type == "cuda"

    config = {**vars(args), "device": str(device), "num_labels": dataset.num_labels, "labels": dataset.labels,
              "T": dataset.max_positions, "params": sum(p.numel() for p in model.parameters()),
              "train_size": int(splits["train"].size), "val_size": int(splits["val"].size)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"device={device} T={dataset.max_positions} labels={dataset.num_labels} params={config['params']:,} "
          f"train={config['train_size']:,} val={config['val_size']:,}")

    best_macro_f1 = -1.0
    history = open(run_dir / "history.jsonl", "a")
    for epoch in range(args.epochs):
        model.train()
        train_loader.batch_sampler.set_epoch(epoch)
        t0, seen, running = time.time(), 0, 0.0
        bar = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch}", leave=False)
        for step, (boards, lengths, targets) in enumerate(bar):
            if step >= steps_per_epoch:
                break
            boards, lengths, targets = boards.to(device, non_blocking=True), lengths.to(device), targets.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(boards, lengths).float(), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            seen += len(targets)
            running += loss.item() * len(targets)
            if step % 20 == 0:
                bar.set_postfix(loss=f"{running / seen:.4f}", ips=f"{seen / (time.time() - t0):.0f}/s")

        val = evaluate(model, val_loader, metrics, device, val_loss_fn,
                       args.val_batches if args.val_batches is not None else args.limit_batches)
        thresholds = val.pop("_thresholds")
        tuned_per_class = val.pop("_tuned_per_class_f1")
        record = {"epoch": epoch, "train_loss": running / max(seen, 1), "epoch_sec": time.time() - t0,
                  **{f"val_{k}": v for k, v in val.items()}}
        history.write(json.dumps(record) + "\n"); history.flush()
        print(f"epoch {epoch}: train_loss={record['train_loss']:.4f} val_loss={val['loss']:.4f} "
              f"micro_f1={val['micro_f1']:.4f} macro_f1={val['macro_f1']:.4f} auprc={val['macro_auprc']:.4f} "
              f"exact={val['exact_match']:.4f} | tuned macro_f1={val['tuned_macro_f1']:.4f} "
              f"micro_f1={val['tuned_micro_f1']:.4f} ({record['epoch_sec']:.0f}s)")

        state = {"model": model.state_dict(), "config": config, "epoch": epoch, "val": val,
                 "thresholds": thresholds.tolist()}
        torch.save(state, run_dir / "last.pt")
        # model selection on the tuned macro-F1 (threshold-robust); 0.5-threshold numbers are still logged
        if val["tuned_macro_f1"] > best_macro_f1:
            best_macro_f1 = val["tuned_macro_f1"]
            torch.save(state, run_dir / "best.pt")
            per_class = metrics.compute_per_class_f1().tolist()
            (run_dir / "per_class_f1.json").write_text(json.dumps(dict(zip(dataset.labels, per_class)), indent=2))
            (run_dir / "thresholds.json").write_text(json.dumps(
                {l: {"threshold": float(th), "tuned_f1": float(f)} for l, th, f in zip(dataset.labels, thresholds, tuned_per_class)},
                indent=2))
    history.close()
    print(f"best val tuned macro_f1={best_macro_f1:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
