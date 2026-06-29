"""Cross-validation splits — patient-grouped, label-stratified.

Checkpoint 0 uses a **single** outer fold as a 3:1:1 train/val/test split
(`HoldoutSplit`). The full Repeated Nested Stratified Group K-Fold protocol and
the deterministic `splits.json` artifact (keyed by dataset/task/n_folds/n_repeats/
seed, for the 50-model fan-out) are layered on at the CV scale-out stage.

Geometry (n_folds=5): outer `StratifiedGroupKFold(5)` → fold `fold` is **test**
(20%); on the remainder, inner `StratifiedGroupKFold(4)` → inner fold 0 is **val**
(20% of total), the rest **train** (60%) → 3:1:1. Grouping is on the patient, so a
patient's samples never split across folds; stratification balances the task label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_holdout_split(y, groups, *, n_folds, fold, repeat, seed):
    """One patient-grouped, label-stratified train/val/test split over samples.

    Returns ``(train_idx, val_idx, test_idx)`` as int arrays of sample positions.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold {fold} out of range for n_folds={n_folds}")

    rs = seed + repeat * 1000
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    tr_va, test_idx = list(outer.split(np.arange(n), y, groups))[fold]

    inner = StratifiedGroupKFold(n_splits=n_folds - 1, shuffle=True, random_state=rs + 1)
    inner_tr, inner_va = next(inner.split(tr_va, y[tr_va], groups[tr_va]))
    return tr_va[inner_tr], tr_va[inner_va], test_idx


@dataclass
class HoldoutSplit:
    """A single train/val/test split applied to a list of PyG sample-graphs."""

    n_folds: int = 5
    fold: int = 0
    repeat: int = 0
    seed: int = 0

    def apply(self, graphs):
        y = [int(g.y) for g in graphs]
        groups = [g.patient for g in graphs]
        return make_holdout_split(y, groups, n_folds=self.n_folds,
                                  fold=self.fold, repeat=self.repeat, seed=self.seed)
