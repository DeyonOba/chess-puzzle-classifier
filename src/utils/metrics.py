import torch
import torch.nn as nn
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelExactMatch,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


class MultiLabelPuzzleMetrics(nn.Module):
    """Aggregates threshold-dependent and ranking metrics for multi-label puzzle themes.

    `update` takes raw logits; thresholds are applied to sigmoid probabilities.
    """

    def __init__(self, num_labels: int, default_threshold: float = 0.5):
        super().__init__()
        self.num_labels = num_labels

        # Overall balance across all (sample, class) pairs
        self.micro_f1 = MultilabelF1Score(num_labels=num_labels, average="micro", threshold=default_threshold)

        # Unweighted average across classes (highlights rare theme failures)
        self.macro_f1 = MultilabelF1Score(num_labels=num_labels, average="macro", threshold=default_threshold)

        # Macro Precision & Recall for diagnostics
        self.macro_precision = MultilabelPrecision(num_labels=num_labels, average="macro", threshold=default_threshold)
        self.macro_recall = MultilabelRecall(num_labels=num_labels, average="macro", threshold=default_threshold)

        # Strict set-equality accuracy (predicted set == target set)
        self.exact_match = MultilabelExactMatch(num_labels=num_labels, threshold=default_threshold)

        # Threshold-independent ranking quality (macro PR-AUC)
        self.macro_auprc = MultilabelAveragePrecision(num_labels=num_labels, average="macro")

        # Per-class F1 for diagnostics and per-class threshold tuning
        self.per_class_f1 = MultilabelF1Score(num_labels=num_labels, average=None, threshold=default_threshold)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        targets = targets.to(torch.long)
        self.micro_f1.update(probs, targets)
        self.macro_f1.update(probs, targets)
        self.macro_precision.update(probs, targets)
        self.macro_recall.update(probs, targets)
        self.exact_match.update(probs, targets)
        self.macro_auprc.update(probs, targets)
        self.per_class_f1.update(probs, targets)

    @torch.no_grad()
    def compute(self) -> dict:
        return {
            "micro_f1": self.micro_f1.compute().item(),
            "macro_f1": self.macro_f1.compute().item(),
            "macro_precision": self.macro_precision.compute().item(),
            "macro_recall": self.macro_recall.compute().item(),
            "exact_match": self.exact_match.compute().item(),
            "macro_auprc": self.macro_auprc.compute().item(),
        }

    @torch.no_grad()
    def compute_per_class_f1(self) -> torch.Tensor:
        return self.per_class_f1.compute()

    def reset(self):
        for m in (
            self.micro_f1,
            self.macro_f1,
            self.macro_precision,
            self.macro_recall,
            self.exact_match,
            self.macro_auprc,
            self.per_class_f1,
        ):
            m.reset()
