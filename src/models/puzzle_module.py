"""LightningModule for multi-label puzzle theme classification.

Owns: the network (built from a partial so the label list comes from data),
the loss (weighted BCE / focal / ASL on free labels, grouped cross-entropy
for exclusive groups), validation metrics at the fixed 0.5 threshold and
with per-class thresholds tuned in-sample on validation, test metrics with
the stored thresholds under both decodings, and the optimiser schedule.

Data-derived weights (pos_weight, group class weights, ML-GCN adjacency) are
filled in setup("fit") from the datamodule's training targets and live in
the state dict, so a checkpoint is self-contained.
"""

from __future__ import annotations

import functools
import importlib
import json
from collections.abc import Mapping
from typing import Callable

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.heads import build_adjacency
from src.utils.label_groups import LabelLayout
from src.utils.losses import GroupedLoss, build_loss, compute_pos_weight
from src.utils.metrics import MultiLabelPuzzleMetrics
from src.utils.thresholds import decode_grouped, f1_at_thresholds, tune_thresholds


def _spec_from_partial(net) -> dict:
    """Serialisable recipe of a functools.partial (what hydra's _partial_ returns) so a checkpoint can rebuild the net."""
    if not isinstance(net, functools.partial):
        raise TypeError("net must be a functools.partial (hydra _partial_: true) so it can be stored in the checkpoint")
    kw = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in net.keywords.items()}
    return {"target": f"{net.func.__module__}.{net.func.__qualname__}", "kwargs": kw}


def _partial_from_spec(spec: dict):
    module, _, name = spec["target"].rpartition(".")
    return functools.partial(getattr(importlib.import_module(module), name), **spec["kwargs"])


def _f1_from_preds(preds: np.ndarray, targets: np.ndarray) -> dict:
    t = targets.astype(bool)
    tp = (preds & t).sum(0).astype(np.float64); fp = (preds & ~t).sum(0).astype(np.float64); fn = (~preds & t).sum(0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return {"micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)), "macro_f1": float(f1.mean()),
            "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
            "exact_match": float((preds == t).all(1).mean()), "per_class_f1": f1.astype(np.float32)}


class PuzzleLitModule(L.LightningModule):
    def __init__(
        self,
        net: Callable[..., nn.Module] | None = None,
        net_spec: dict | None = None,
        labels: list[str] | None = None,
        labels_path: str | None = None,
        loss: str = "weighted_bce",
        pos_weight_cap: float = 20.0,
        pos_weight_power: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 1.0,
        asl_clip: float = 0.05,
        group_weight: float = 0.5,
        mlgcn_tau: float = 0.4,
        mlgcn_p: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "onecycle",      # onecycle (needs the total step count up front) | plateau (resume-safe) | none
        plateau_factor: float = 0.5,
        plateau_patience: int = 2,         # validations without improvement before the LR is halved
    ):
        super().__init__()
        if labels is None:
            if labels_path is None:
                raise ValueError("PuzzleLitModule needs labels or labels_path")
            labels = json.loads(open(labels_path).read())["labels"]
        if net is None:
            if net_spec is None:
                raise ValueError("PuzzleLitModule needs net (a partial) or net_spec (from a checkpoint)")
            net = _partial_from_spec(net_spec)
        elif net_spec is None:
            net_spec = _spec_from_partial(net)
        self.save_hyperparameters(ignore=["net"])
        # the checkpoint stores the *initial* hparams snapshot, so write the resolved values into it too
        for k, v in {"labels": list(labels), "net_spec": net_spec}.items():
            self._hparams[k] = v
            self._hparams_initial[k] = v
        self.labels = list(labels)
        L_ = len(self.labels)

        self.net = net(num_labels=L_, labels=self.labels)
        self.layout: LabelLayout | None = getattr(self.net.head, "layout", None)
        n_free = len(self.layout.free) if self.layout is not None else L_
        base = build_loss(loss, None, pos_weight_cap, pos_weight_power, focal_gamma, focal_alpha,
                          asl_gamma_neg, asl_gamma_pos, asl_clip, num_labels=n_free)
        self.criterion = GroupedLoss(self.net.head, base, None, pos_weight_cap, pos_weight_power, group_weight) if self.layout else base

        self.register_buffer("thresholds", torch.full((L_,), 0.5))
        # kept out of the module tree on purpose: metrics are computed on CPU over the whole split.
        # On MPS the torchmetrics kernels produced out-of-bounds indices at ~100k rows (2026-10-12).
        self._metrics_holder = [MultiLabelPuzzleMetrics(L_)]
        self._val_buf: list[tuple[np.ndarray, np.ndarray]] = []
        self._test_buf: list[tuple[np.ndarray, np.ndarray]] = []
        self.test_results: dict | None = None

    # -- data-derived weights -----------------------------------------------------------------
    def setup(self, stage: str | None = None):
        if stage != "fit":
            return
        dm = self.trainer.datamodule
        needs = self.hparams.loss == "weighted_bce" or self.layout is not None or self.net.head_type == "mlgcn"
        if not needs:
            return
        train_targets = dm.train_targets()
        free_loss = self.criterion.free_loss if self.layout is not None else self.criterion
        if self.hparams.loss == "weighted_bce":
            sub = train_targets[:, self.layout.free] if self.layout is not None else train_targets
            free_loss.pos_weight.copy_(compute_pos_weight(sub, self.hparams.pos_weight_cap, self.hparams.pos_weight_power).to(free_loss.pos_weight.device))
        if self.layout is not None:
            self.criterion.set_class_weights(train_targets)
        if self.net.head_type == "mlgcn":
            adjacency, cond = build_adjacency(train_targets, self.hparams.mlgcn_tau, self.hparams.mlgcn_p)
            self.net.head.adjacency.copy_(torch.as_tensor(adjacency))
            self.net.head.label_embedding.data.copy_(torch.as_tensor(cond))

    # -- forward / steps -----------------------------------------------------------------------
    def forward(self, boards: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.net(boards, lengths)

    def probs(self, raw: torch.Tensor) -> torch.Tensor:
        return self.net.probs(raw)

    def on_train_epoch_start(self):
        bs = getattr(self.trainer.train_dataloader, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        boards, lengths, targets = batch
        loss = self.criterion(self(boards, lengths).float(), targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(targets))
        return loss

    def _collect(self, batch, buf):
        boards, lengths, targets = batch
        p = self.probs(self(boards, lengths).float())
        buf.append((p.detach().cpu().numpy(), targets.detach().cpu().numpy().astype(np.uint8)))

    def validation_step(self, batch, batch_idx):
        self._collect(batch, self._val_buf)

    def test_step(self, batch, batch_idx):
        self._collect(batch, self._test_buf)

    def predict_step(self, batch, batch_idx):
        boards, lengths = batch[0], batch[1]
        return self.probs(self(boards, lengths).float())

    # -- epoch ends ----------------------------------------------------------------------------
    def _gather(self, buf):
        probs = np.concatenate([p for p, _ in buf]); targets = np.concatenate([t for _, t in buf]); buf.clear()
        return probs, targets

    def _fixed_metrics(self, probs: np.ndarray, targets: np.ndarray) -> dict:
        metrics = self._metrics_holder[0]  # CPU, see __init__
        logits = torch.logit(torch.from_numpy(probs).clamp(1e-7, 1 - 1e-7))
        t = torch.from_numpy(targets)
        metrics.reset(); metrics.update(logits, t)
        out = metrics.compute()
        out["loss"] = F.binary_cross_entropy_with_logits(logits, t.float()).item()
        return out

    def on_validation_epoch_end(self):
        if not self._val_buf:
            return
        probs, targets = self._gather(self._val_buf)
        if self.trainer.sanity_checking:
            return
        fixed = self._fixed_metrics(probs, targets)
        thresholds = tune_thresholds(probs, targets)
        self.thresholds.copy_(torch.from_numpy(thresholds).to(self.thresholds.device))
        tuned = f1_at_thresholds(probs, targets, thresholds)
        for k, v in fixed.items():
            self.log(f"val/{k}", v, prog_bar=k in ("macro_f1",))
        for k in ("micro_f1", "macro_f1", "macro_precision", "macro_recall", "exact_match"):
            self.log(f"val/tuned_{k}", tuned[k], prog_bar=k == "macro_f1")

    def on_test_epoch_end(self):
        probs, targets = self._gather(self._test_buf)
        thr = self.thresholds.detach().cpu().numpy()
        fixed = _f1_from_preds(probs >= 0.5, targets)
        tuned = _f1_from_preds(probs >= thr[None], targets)
        results = {"n_test": int(len(targets)),
                   "fixed_0.5": {k: v for k, v in fixed.items() if k != "per_class_f1"},
                   "val_tuned": {k: v for k, v in tuned.items() if k != "per_class_f1"},
                   "ranking": self._fixed_metrics(probs, targets)}
        per_class = {l: {"threshold": float(thr[j]), "f1_fixed": float(fixed["per_class_f1"][j]), "f1_tuned": float(tuned["per_class_f1"][j])}
                     for j, l in enumerate(self.labels)}
        if self.layout is not None:
            argmax = _f1_from_preds(decode_grouped(probs, thr, self.layout), targets)
            results["argmax_groups"] = {k: v for k, v in argmax.items() if k != "per_class_f1"}
            for j, l in enumerate(self.labels):
                per_class[l]["f1_argmax"] = float(argmax["per_class_f1"][j])
        results["per_class"] = per_class
        self.test_results = results
        for k, v in results["val_tuned"].items():
            self.log(f"test/tuned_{k}", v)
        for k, v in results["fixed_0.5"].items():
            self.log(f"test/{k}", v)
        if "argmax_groups" in results:
            for k, v in results["argmax_groups"].items():
                self.log(f"test/argmax_{k}", v)
        self.log("test/macro_auprc", results["ranking"]["macro_auprc"])

    # -- optimisation --------------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        if self.hparams.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=self.hparams.plateau_factor,
                                                               patience=self.hparams.plateau_patience)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/tuned_macro_f1", "interval": "epoch"}}
        return opt
