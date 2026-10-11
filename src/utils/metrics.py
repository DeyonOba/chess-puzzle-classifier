import torch
import torch.nn as nn
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelExactMatch,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelAUPRC
)

class MultiLabelPuzzleMetrics(nn.Module):
    def __init__(self, num_classes: int, default_threshold: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # Overall balance across all (sample, class) pairs
        self.micro_f1 = MultilabelF1Score(num_classes=num_classes, average="micro", threshold=default_threshold)
        
        # Unweighted average across classes (highlights rare theme failures)
        self.macro_f1 = MultilabelF1Score(num_classes=num_classes, average="macro", threshold=default_threshold)
        
        # Macro Precision & Recall for diagnostics
        self.macro_precision = MultilabelPrecision(num_classes=num_classes, average="macro", threshold=default_threshold)
        self.macro_recall = MultilabelRecall(num_classes=num_classes, average="macro", threshold=default_threshold)

        # Strict set-equality accuracy (Obj == {A, B, E} exactly)
        self.exact_match = MultilabelExactMatch(num_classes=num_classes, threshold=default_threshold)

        # Threshold-independent ranking quality (PR-AUC)
        self.macro_auprc = MultilabelAUPRC(num_classes=num_classes, average="macro")

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        self.micro_f1.update(probs, targets)
        self.macro_f1.update(probs, targets)
        self.macro_precision.update(probs, targets)
        self.macro_recall.update(probs, targets)
        self.exact_match.update(probs, targets)
        self.macro_auprc.update(probs, targets)

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

    def reset(self):
        self.micro_f1.reset()
        self.macro_f1.reset()
        self.macro_precision.reset()
        self.macro_recall.reset()
        self.exact_match.reset()
        self.macro_auprc.reset()