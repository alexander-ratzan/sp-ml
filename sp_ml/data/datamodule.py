"""SpatialGraphDataModule — bag-of-cells (no edges) path for Checkpoint 0.

One PyG ``Data`` per tissue **sample** (`sample_col`, e.g. Schürch `Region`):
nodes = cells, node features = `exprs_norm`, graph label `y` = the patient-level
task label mapped to a class index. **No edges / no kNN cache yet** — bag-of-cells
ignores `edge_index`, so the spatial graph + `InMemoryDataset` cache are deferred
to the first graph layer (post-Checkpoint-0). Splits (S4) are also deferred; until
then `setup()` exposes all sample-graphs and `all_dataloader()` yields one batch.
"""

from __future__ import annotations

import anndata as ad
import lightning as L
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _to_dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def build_sample_graphs(adata, *, name, sample_col, patient_col, feature_layer, task):
    """Build one bag-of-cells ``Data`` per sample.

    ``y`` = ``obs[task.source[name]]`` → optional ``task.remap[name]`` → class index
    from ``task.classes``. Samples with a missing/unmapped target are dropped.
    Returns ``(graphs, dropped, num_markers)``.
    """
    if name not in task.source:
        raise KeyError(f"task '{task.name}' has no source column for dataset '{name}'")
    label_col = task.source[name]
    class_to_idx = {c: i for i, c in enumerate(task.classes)}
    remap = dict((task.remap or {}).get(name, {}))

    X = _to_dense(adata.layers[feature_layer]).astype(np.float32)
    if np.isnan(X).any():
        raise ValueError(f"NaNs in layer '{feature_layer}' for dataset '{name}'")
    obs = adata.obs

    graphs, dropped = [], 0
    for sample_id, pos in obs.groupby(sample_col, observed=True).indices.items():
        raw = obs[label_col].iloc[pos[0]]          # patient-level → constant within a sample
        raw = remap.get(raw, raw)
        if raw not in class_to_idx:                 # missing / NaN / unmapped
            dropped += 1
            continue
        g = Data(x=torch.from_numpy(X[pos]),
                 y=torch.tensor([class_to_idx[raw]], dtype=torch.long))
        g.sample_id = str(sample_id)
        g.patient = str(obs[patient_col].iloc[pos[0]])
        graphs.append(g)
    return graphs, dropped, X.shape[1]


class SpatialGraphDataModule(L.LightningDataModule):
    def __init__(self, name, h5ad_path, task, feature_layer="exprs_norm",
                 sample_col="sample_id", patient_col="patient_id",
                 graph_kind="knn", k=20, batch_size=8, num_workers=8,
                 role="cv", split=None, cache_dir="cache/graphs"):
        super().__init__()
        self.save_hyperparameters(ignore=["task", "split"])
        self.task = task
        self.split = split          # consumed at S4; ignored here
        self._graphs = None
        self.num_markers = None
        self.num_classes = None
        self.n_dropped = None
        self.tr = self.va = self.te = None       # sample indices per fold (set if split given)

    def setup(self, stage=None):
        if self._graphs is None:
            adata = ad.read_h5ad(self.hparams.h5ad_path)
            self._graphs, self.n_dropped, self.num_markers = build_sample_graphs(
                adata, name=self.hparams.name,
                sample_col=self.hparams.sample_col,
                patient_col=self.hparams.patient_col,
                feature_layer=self.hparams.feature_layer,
                task=self.task)
            self.num_classes = len(self.task.classes)
        if self.split is not None and self.tr is None:   # no split logic here — the split owns it
            self.tr, self.va, self.te = self.split.apply(self._graphs)

    @property
    def graphs(self):
        return self._graphs

    def _loader(self, graphs, shuffle=False):
        return DataLoader(graphs, batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=self.hparams.num_workers)

    def _subset(self, idx):
        return [self._graphs[i] for i in idx]

    def train_labels(self):
        """Train-fold sample labels (for class weighting; train split only → no leakage)."""
        return [int(self._graphs[i].y) for i in self.tr]

    def train_dataloader(self):  return self._loader(self._subset(self.tr), shuffle=True)
    def val_dataloader(self):    return self._loader(self._subset(self.va))
    def test_dataloader(self):   return self._loader(self._subset(self.te))

    # convenience for the S3 cell (no split needed): one loader over all graphs
    def all_dataloader(self):
        return self._loader(self._graphs)
