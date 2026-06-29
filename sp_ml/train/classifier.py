"""LitClassifier — supervised classification wrapper with patient-level metrics.

Train per **sample** (more gradient signal); score per **patient** (the clinical claim).
`aggregate_by_patient` mean-pools a patient's region softmaxes into one prediction before
metrics — the one custom step on top of torchmetrics. `class_weights` (inverse train-fold
frequency) optionally rebalances the loss; computed from the train split only (no leakage).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, MetricCollection

from sp_ml.train.litbase import LitBase


def class_weights(labels, num_classes):
    """Inverse-frequency class weights: ``w_c = N / (C * n_c)`` (≈1 when balanced)."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1.0))


def aggregate_by_patient(buf):
    """Mean softmax over each patient's samples → one prediction/patient.

    ``buf`` = list of ``(probs[n,C], y[n], patients:list[str])``. Returns
    ``(patient_probs[P,C], patient_y[P])`` with patients in first-seen order.
    """
    probs = torch.cat([b[0] for b in buf])
    y = torch.cat([b[1] for b in buf])
    patients = [p for b in buf for p in b[2]]
    groups = OrderedDict()
    for i, p in enumerate(patients):
        groups.setdefault(p, []).append(i)
    idx = list(groups.values())
    pat_probs = torch.stack([probs[ix].mean(0) for ix in idx])
    pat_y = torch.stack([y[ix[0]] for ix in idx])      # label constant within a patient
    return pat_probs, pat_y


class LitClassifier(LitBase):
    def __init__(self, model, optimizer, scheduler=None, loss=None, num_classes=2):
        super().__init__(model, optimizer, scheduler)
        self.loss = loss or nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.metrics = nn.ModuleDict({
            s: MetricCollection({
                "auroc": AUROC(task="multiclass", num_classes=num_classes),
                "auprc": AveragePrecision(task="multiclass", num_classes=num_classes),
                "f1":    F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                "bacc":  Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
            }, prefix=f"{s}/")
            for s in ("val", "test")
        })
        self._buf = {"val": [], "test": []}

    def _step(self, batch):
        logits = self.model(batch)
        return logits, self.loss(logits, batch.y)

    def training_step(self, batch, _):
        logits, loss = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def _eval_step(self, batch, split):
        logits, loss = self._step(batch)
        self.log(f"{split}/loss", loss, prog_bar=(split == "val"), batch_size=batch.num_graphs)
        self._buf[split].append((logits.softmax(-1).detach(), batch.y.detach(), list(batch.patient)))

    def _epoch_end(self, split):
        if not self._buf[split]:
            return
        probs, y = aggregate_by_patient(self._buf[split])
        m = self.metrics[split]
        m.update(probs, y)
        self.log_dict(m.compute(), prog_bar=(split == "val"))
        m.reset()
        self._buf[split].clear()

    def validation_step(self, b, _):   self._eval_step(b, "val")
    def test_step(self, b, _):         self._eval_step(b, "test")
    def on_validation_epoch_end(self): self._epoch_end("val")
    def on_test_epoch_end(self):       self._epoch_end("test")
