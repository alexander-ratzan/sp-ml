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
# # Jackson-Fischer 2020 — Breast Cancer IMC EDA (Full)
#
# Exploratory data analysis of the Jackson & Fischer et al. 2020 breast cancer dataset,
# acquired by Imaging Mass Cytometry (IMC).
#
# Full cohort: 285 patients, 1,240,267 cells, 45 IMC markers.
Expression matrix: raw ion counts in X, arcsinh-transformed in layers["exprs"].

# %%
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
_r = next(p for p in [Path().resolve(), *Path().resolve().parents] if (p / "data").is_dir() and (p / "notebooks").is_dir())
if str(_r) not in sys.path: sys.path.insert(0, str(_r))

import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

from data.EDA import (
    JACKSON_CFG, TECHNICAL_MARKERS, FS,
    summarize_metadata, spatial_info, cat_breakdown,
    plot_marker_distributions,
)

CFG = JACKSON_CFG

DATASET_LABEL = "Jackson 2020 (Full)"

# %% [markdown]
# ## Load Data

# %%
DATA_PATH = "../../../data/jacksonfischer2020/full/full.h5ad"

adata_raw = ad.read_h5ad(DATA_PATH)
adata_raw

# %%
# Re-run this cell to restore clean state without restarting kernel
adata = adata_raw.copy()
adata.uns["dataset"] = DATASET_LABEL
# cell_metacluster is float64; cast to categorical for groupby + discrete coloring
adata.obs["cell_metacluster"] = adata.obs["cell_metacluster"].astype("Int64").astype("string").astype("category")
sample = adata.obs["image_name"].unique()[0]

# %% [markdown]
# ## Dataset Overview

# %%
summarize_metadata(adata, cfg=CFG)

# %%
spatial_info(adata)

# %%
cat_breakdown(adata, cfg=CFG)

# %% [markdown]
# ## Single Sample Viewer

# %%
sq.pl.spatial_scatter(
    adata[adata.obs["image_name"] == sample],
    shape=None,
    color="cell_metacluster",
    size=4,
    figsize=(8, 8),
    dpi=150,
)

# %% [markdown]
# ## Subsample Viewer
#
# 8 of 285 images, colored by cell_metacluster.

# %%
_sample_ids = sorted(adata.obs["image_name"].unique())[:8]
_n_cols, _n_rows = 4, 2
_fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 5), dpi=150, facecolor="white")
for _ax, _sid in zip(_axes.flatten(), _sample_ids):
    sq.pl.spatial_scatter(adata[adata.obs["image_name"] == _sid], shape=None, color="cell_metacluster",
                          size=4, ax=_ax, title=str(_sid))
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Analysis Setup — squidpy & scanpy

# %% [markdown]
# ### `.X` Evaluation
#
# Jackson `.X` contains raw IMC ion counts with 4.5% NaN — do not use for analysis.
# `layers["exprs"]` holds arcsinh-transformed values (0 – 8.96, same NaN pattern) and
# is the most processed source available. Ruthenium bead channels (Ru96–Ru104) are
# purely technical and excluded below.

# %%
# Build a concrete copy whose .X is the densified exprs layer, restricted to
# biological markers. All scanpy calls below then operate on the same dense
# arcsinh matrix (no layer= needed) — avoids sparse/raw-.X mismatch in dendrograms.
_bio_vars = [v for v in adata.var_names if v not in TECHNICAL_MARKERS]
adata_bio = adata[:, _bio_vars].copy()
_L = adata_bio.layers["exprs"]
adata_bio.X = (_L.toarray() if hasattr(_L, "toarray") else np.asarray(_L)).astype(np.float32)
# Drop markers that are entirely NaN (not measured in this cohort, e.g. Basel: EpCAM/CTNNB/SOX9)
_allnan = np.isnan(adata_bio.X).all(axis=0)
if _allnan.any():
    print(f"Dropping all-NaN markers: {list(adata_bio.var_names[_allnan])}")
    adata_bio = adata_bio[:, ~_allnan].copy()
adata_bio.uns["dataset"] = DATASET_LABEL
print(f"Biological markers: {adata_bio.n_vars} / {adata.n_vars} total  (excluded {adata.n_vars - adata_bio.n_vars}: Ru channels + all-NaN)")

# %%
plot_marker_distributions(adata_bio)

# %%
E = adata_bio.X
print("=== exprs stats (biological markers only) ===")
print(f"  range:         [{np.nanmin(E):.4f}, {np.nanmax(E):.4f}]")
print(f"  mean / median: {np.nanmean(E):.4f} / {np.nanmedian(E):.4f}")
print(f"  pct NaN:       {np.isnan(E).mean() * 100:.1f}%")

# %% [markdown]
# ### Cell type x marker profiles (scanpy)
#
# `cell_metacluster` holds phenograph cluster IDs (not named cell types).
# adata_bio.X is the arcsinh exprs matrix.

# %%
sc.pl.dotplot(
    adata_bio,
    var_names=list(adata_bio.var_names),
    groupby="cell_metacluster",
    standard_scale="var",
    figsize=(18, 8),
    dendrogram=True,
)

# %%
sc.pl.matrixplot(
    adata_bio,
    var_names=list(adata_bio.var_names),
    groupby="cell_metacluster",
    standard_scale="var",
    figsize=(18, 8),
    dendrogram=True,
)

# %%
sc.pl.correlation_matrix(adata_bio, groupby="cell_metacluster", figsize=(10, 8))

# %%
_corr = pd.DataFrame(adata_bio.X, columns=adata_bio.var_names).corr()
_fig, _ax = plt.subplots(figsize=(14, 12), dpi=120, facecolor="white")
_im = _ax.imshow(_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
_ax.set_xticks(range(len(_corr.columns)))
_ax.set_xticklabels(_corr.columns, rotation=90, fontsize=FS["sm"])
_ax.set_yticks(range(len(_corr.index)))
_ax.set_yticklabels(_corr.index, fontsize=FS["sm"])
_fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
_ax.set_title(f"{DATASET_LABEL} — marker x marker correlation (exprs layer, no aggregation)", fontsize=FS["md"])
plt.tight_layout()
plt.show()
