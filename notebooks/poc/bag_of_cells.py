# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # POC — Bag-of-Cells Logreg (Checkpoint 0)
#
# Manual validation of the modeling spine on **Schürch CRC / `clr_dii`**: instantiate a
# **dataset + task + mean-pool bag-of-cells logreg** and train it for a single train/val/test
# split. This notebook grows stage by stage and stays runnable at every step:
#
# - **S3** — DataModule (no edges): one batch — shapes, label mapping, patient ids. *(this stage)*
# - **S4** — single `StratifiedGroupKFold`: patient-disjoint train/val/test, zero leakage.
# - **S5** — `SpModel` (identity · none · mean · logreg): batch → logits `[n_samples, n_classes]`.
# - **S6** — `LitClassifier` + patient-level metrics: `trainer.fit` → loss ↓, val AUROC/AUPRC/F1/bacc.
#
# Everything is driven by the Hydra config in `conf/` — the same config `python -m sp_ml.run` uses.

# %%
import collections
import pathlib

import sp_ml
import sp_ml.configs as C
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# Locate repo paths off the installed package → cwd-independent (works from any notebook dir).
REPO = pathlib.Path(sp_ml.__file__).resolve().parent.parent     # .../sp-ml
CONF = str(REPO / "conf")
DATA = str(REPO.parent / "data")                                # .../spatialproteomics/data

C.register_configs()
with initialize_config_dir(version_base=None, config_dir=CONF):
    cfg = compose(config_name="config", overrides=[f"paths.data={DATA}"])

print("dataset:", cfg.data.name, "| task:", cfg.task.name, cfg.task.kind, list(cfg.task.classes))
print("h5ad:", cfg.data.h5ad_path)

# %% [markdown]
# ## S3 — DataModule (bag-of-cells, no edges)
#
# One PyG `Data` per `Region` (the imaged TMA sample). `y` is the patient-level CLR/DII label
# mapped to a class index; `patient` is carried on each graph for patient-grouped CV (S4) and
# patient-level scoring (S6). `num_workers=0` keeps it notebook-safe.

# %%
dm = instantiate(cfg.data, task=cfg.task, num_workers=0)
dm.setup()

print("n sample-graphs:", len(dm.graphs), "| dropped:", dm.n_dropped)
print("num_markers:", dm.num_markers, "| num_classes:", dm.num_classes)
print("n unique patients:", len({g.patient for g in dm.graphs}))
yc = collections.Counter(int(g.y) for g in dm.graphs)
print("graph-label counts (class_idx -> n_regions):", dict(yc), "| classes:", list(cfg.task.classes))

# %% [markdown]
# ### Inspect one batch
# Expect: `x = [total_cells, num_markers]` float32, `y = [batch_size]`, a `batch` vector mapping
# each cell to its graph, and `patient`/`sample_id` as length-`batch_size` string lists.

# %%
batch = next(iter(dm.all_dataloader()))
print("batch:", batch)
print("x:", tuple(batch.x.shape), batch.x.dtype, "| y:", batch.y.tolist())
print("n_graphs:", batch.num_graphs, "| per-graph cells:", batch.batch.bincount().tolist())
print("patient:", batch.patient)
print("sample_id:", batch.sample_id)
print("edge_index present:", getattr(batch, "edge_index", None) is not None, "(bag-of-cells → none)")
assert batch.x.shape[0] == int(batch.batch.bincount().sum())
assert batch.x.shape[1] == dm.num_markers
assert len(batch.patient) == batch.num_graphs == batch.y.numel()
print("\nS3 OK — shapes, label mapping, and patient ids all check out.")

# %% [markdown]
# ## S4 — single train/val/test split (patient-grouped, label-stratified)
#
# One outer fold of `StratifiedGroupKFold` as a **3:1:1** split: grouped on `patient` (a
# patient's 2 regions never split across folds → no leakage) and stratified on the CLR/DII
# label (each fold mirrors the cohort class ratio). The split is deterministic in
# `(seed, repeat, fold)`; Checkpoint 0 uses `fold=0, repeat=0`.

# %%
import collections

from sp_ml.data.crossval import HoldoutSplit

dm.split = HoldoutSplit(n_folds=cfg.cv.n_folds, fold=cfg.cv.fold,
                        repeat=cfg.cv.repeat, seed=cfg.cv.seed)
dm.setup()   # graphs already built in S3; this just applies the split

g = dm.graphs
patients = lambda idx: {g[i].patient for i in idx}
labels = lambda idx: collections.Counter(int(g[i].y) for i in idx)
classes = list(cfg.task.classes)

print("regions : train=%d val=%d test=%d (total %d)" % (len(dm.tr), len(dm.va), len(dm.te), len(g)))
ptr, pva, pte = patients(dm.tr), patients(dm.va), patients(dm.te)
print("patients: train=%d val=%d test=%d" % (len(ptr), len(pva), len(pte)))

for name, idx in [("train", dm.tr), ("val", dm.va), ("test", dm.te)]:
    c = labels(idx); tot = sum(c.values())
    print("  %-5s:" % name, {classes[k]: c[k] for k in sorted(c)}, "| DII frac=%.2f" % (c[1] / tot))

# %% [markdown]
# ### Zero-leakage assertions (the gate)

# %%
assert ptr & pva == set() and ptr & pte == set() and pva & pte == set(), "PATIENT LEAKAGE"
assert ptr | pva | pte == {gg.patient for gg in g}, "a patient is missing from the split"
assert len(dm.tr) + len(dm.va) + len(dm.te) == len(g), "region count mismatch"
for name, idx in [("train", dm.tr), ("val", dm.va), ("test", dm.te)]:
    assert all(c == 2 for c in collections.Counter(g[i].patient for i in idx).values()), \
        f"{name}: a patient's regions were split"
print("S4 OK — train/val/test patient-disjoint (zero leakage), grouping + stratification intact.")

# %% [markdown]
# ## S5 — the model (`encoder → graph → pool → readout`)
#
# The POC `SpModel`: `Identity` encoder · `NoGraph` · `MeanPool` · `LogReg`. Each component
# follows the uniform `in_dim → out_dim` contract, so shapes thread eagerly at build time and
# the readout sees `pool.out_dim` (here 58, since identity/none/mean all preserve width). One
# batch → logits `[n_samples, n_classes]`.

# %%
import torch

from sp_ml.models import SpModel

encoder = instantiate(cfg.model.encoder, in_dim=dm.num_markers)
graph = instantiate(cfg.model.graph, in_dim=encoder.out_dim)
pool = instantiate(cfg.model.pool, in_dim=graph.out_dim)
readout = instantiate(cfg.model.readout, in_dim=pool.out_dim, num_classes=dm.num_classes)
model = SpModel(encoder, graph, pool, readout)

print("shape chain: %d -> %d -> %d -> %d -> classes=%d" %
      (dm.num_markers, encoder.out_dim, graph.out_dim, pool.out_dim, readout.out_dim))
print("trainable params:", sum(p.numel() for p in model.parameters()))
print(model)

b = next(iter(dm.train_dataloader()))
model.eval()
with torch.no_grad():
    logits = model(b)
print("\nbatch graphs:", b.num_graphs, "| logits:", tuple(logits.shape))
assert logits.shape == (b.num_graphs, dm.num_classes)
assert torch.isfinite(logits).all()
print("S5 OK — forward → logits [n_samples, n_classes].")

# %% [markdown]
# ## Benchmark context — simple/standard baselines
#
# This is the **Schürch CRC, CLR vs DII** task (binary, patient-level). Reported baselines on this
# exact dataset frame what "good" looks like (from the NotebookLM notes: SORBET benchmarking, Ali
# et al.):
#
# | Model | Type | AUROC |
# |---|---|---|
# | SORBET | spatial GNN | **0.98** (SOTA) |
# | Augur-SC | non-spatial single-cell | 0.91 |
# | CITRUS-SC | non-spatial single-cell | 0.86 |
# | SPACE-GM | spatial GNN | 0.81 |
# | CytoCommunity | spatial graph | 0.78 |
#
# **Our model is a non-spatial bag-of-cells logreg** (mean-pooled `exprs_norm`, no edges) — its
# closest analogues are Augur-SC / CITRUS-SC (**~0.86–0.91 AUROC**). Ali et al. find spatial edges
# add only ~ΔAUPR 0.052 here, so cell expression alone is highly predictive: a correctly-fit floor
# should clear **AUROC 0.80** and approach the non-spatial 0.86–0.91 regime. Watch **AUROC + AUPR**
# (threshold-free) and **balanced accuracy** (honest under imbalance — currently the metric to move).

# %% [markdown]
# ## Train the floor model + report
#
# Train the bag-of-cells logreg on the single fold **to convergence** (more epochs than the original
# Checkpoint-0 smoke), tracking the **loss curve + metrics to W&B**. Metrics are scored at the
# **patient level** (`aggregate_by_patient`). Then visualize loss curves and a patient-level
# evaluation panel (confusion matrix, ROC, PR, per-class precision/recall).

# %%
import os
import warnings

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from sp_ml.train import LitClassifier, class_weights

warnings.filterwarnings("ignore")
L.seed_everything(cfg.seed, verbose=False)

EPOCHS = 200            # ↑ from the 40-epoch Checkpoint-0 smoke — let the logreg actually converge

# W&B: loss curve + metrics track live. Online by default (account default entity); set env
# WANDB_MODE=offline to log locally (sync later), or WANDB_MODE=disabled to skip W&B entirely.
wb_mode = os.environ.get("WANDB_MODE", "online")
wandb_logger = (
    False if wb_mode == "disabled"
    else WandbLogger(project=cfg.wandb.project, offline=(wb_mode != "online"),
                     group="schurch-clr_dii-bagcells", name=f"poc-logreg-{EPOCHS}ep",
                     save_dir="outputs")
)

w = class_weights(dm.train_labels(), dm.num_classes) if cfg.train.class_weighted else None
print("class weights (train-fold inverse freq):", None if w is None else [round(float(x), 3) for x in w])

loss = instantiate(cfg.train.get("loss"), weight=w)
optimizer = instantiate(cfg.train.optimizer)   # functools.partial (_partial_: true)
lit = LitClassifier(model=model, optimizer=optimizer, loss=loss, num_classes=dm.num_classes)


class LossHistory(L.Callback):
    """Capture epoch-level train & val loss during fit for a local loss curve."""

    def __init__(self):
        self.train, self.val = [], []

    def on_train_epoch_end(self, trainer, _):
        v = trainer.callback_metrics.get("train/loss")
        if v is not None:
            self.train.append(float(v))

    def on_validation_epoch_end(self, trainer, _):
        if trainer.state.fn != "fit":     # ignore the post-fit validate/test passes
            return
        v = trainer.callback_metrics.get("val/loss")
        if v is not None:
            self.val.append(float(v))


hist = LossHistory()
trainer = instantiate(cfg.train.trainer, max_epochs=EPOCHS, logger=wandb_logger,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0, callbacks=[hist])
trainer.fit(lit, dm)
print(f"trained {EPOCHS} epochs | W&B: {wb_mode}")

# %% [markdown]
# ### Loss curves

# %%
import matplotlib.pyplot as plt

print("train loss: first=%.4f  last=%.4f  min=%.4f" % (hist.train[0], hist.train[-1], min(hist.train)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, len(hist.train) + 1), hist.train, label="train")
if hist.val:
    ax.plot(range(1, len(hist.val) + 1), hist.val, label="val")
ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
ax.set_title("Bag-of-cells logreg — loss")
plt.show()

# %% [markdown]
# ### Patient-level metrics + benchmark comparison

# %%
valm = trainer.validate(lit, dm, verbose=False)[0]
testm = trainer.test(lit, dm, verbose=False)[0]
keys = ("auroc", "auprc", "f1", "bacc")
for d, split in [(valm, "val"), (testm, "test")]:
    print("%-5s (patient-level):" % split,
          "  ".join("%s=%.3f" % (k, d[f"{split}/{k}"]) for k in keys))
print("\nbaseline context — non-spatial ≈ 0.86–0.91 AUROC (Augur-SC/CITRUS-SC); SOTA spatial ≈ 0.98 (SORBET)")

# %% [markdown]
# ### Evaluation panel (patient-level, test fold)

# %%
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, classification_report,
)

from sp_ml.train import aggregate_by_patient

dev = next(lit.model.parameters()).device
lit.eval()
buf = []
with torch.no_grad():
    for bb in dm.test_dataloader():
        bb = bb.to(dev)
        buf.append((lit.model(bb).softmax(-1).cpu(), bb.y.cpu(), list(bb.patient)))
probs, yy = aggregate_by_patient(buf)
probs, yy = probs.numpy(), yy.numpy().astype(int)
pred, pos = probs.argmax(1), probs[:, 1]      # pos = P(DII)
classes = list(cfg.task.classes)

print(classification_report(yy, pred, target_names=classes, zero_division=0))
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ConfusionMatrixDisplay.from_predictions(yy, pred, display_labels=classes, ax=ax[0], colorbar=False)
ax[0].set_title("Confusion (patient-level)")
RocCurveDisplay.from_predictions(yy, pos, ax=ax[1], name=classes[1])
ax[1].plot([0, 1], [0, 1], "k--", lw=0.7); ax[1].set_title("ROC")
PrecisionRecallDisplay.from_predictions(yy, pos, ax=ax[2], name=classes[1])
ax[2].set_title("PR")
fig.suptitle(f"{cfg.data.name} · {cfg.task.name} · fold {cfg.cv.fold} — bag-of-cells logreg ({EPOCHS} ep)")
fig.tight_layout()
plt.show()

# %%
import wandb

if wandb_logger:
    wandb.finish()
