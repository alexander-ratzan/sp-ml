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
# # Patwa 2021 — TNBC MIBI EDA
#
# Exploratory data analysis of the Patwa et al. 2021 triple-negative breast cancer (TNBC) dataset,
# acquired by multiplexed ion beam imaging (MIBI).
#
# 38 patients, 190,240 cells, 44 protein markers (lineage + functional + elements).

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
    PATWA_CFG, PATWA_CELLTYPE_CMAP, TECHNICAL_MARKERS, FS,
    summarize_metadata, spatial_info, cat_breakdown,
    plot_all_samples, plot_marker_distributions, register_cmap,
)

CFG = PATWA_CFG

# %% [markdown]
# ## Load Data

# %%
DATA_PATH = "../../../data/rasp-mibi/tnbc_mibi.h5ad"

adata_raw = ad.read_h5ad(DATA_PATH)
adata_raw

# %%
# Re-run this cell to restore clean state without restarting kernel
adata = adata_raw.copy()
adata.uns["dataset"] = "Patwa 2021"
register_cmap(adata, "cell_type", PATWA_CELLTYPE_CMAP)

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
    adata[adata.obs["patient_id"] == 1],
    shape=None,
    color="cell_type",
    size=4,
    figsize=(8, 8),
    dpi=150,
)

# %% [markdown]
# ## Subsample Viewer
#
# 8 of 38 patients, colored by cell_type, shared palette.

# %%
_sample_ids = sorted(adata.obs["patient_id"].unique())[:8]
_n_cols, _n_rows = 4, 2
_fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 5), dpi=150, facecolor="white")
for _ax, _sid in zip(_axes.flatten(), _sample_ids):
    sq.pl.spatial_scatter(adata[adata.obs["patient_id"] == _sid], shape=None, color="cell_type",
                          size=4, ax=_ax, title=str(_sid))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Full Dataset Viewer
#
# All 38 patients.

# %%
plot_all_samples(adata, color_by="cell_type", n_cols=6, s=1, cfg=CFG)

# %% [markdown]
# ## Analysis Setup — squidpy & scanpy

# %% [markdown]
# ### `.X` Evaluation
#
# Patwa `.X` contains raw MIBI intensities (0 – 248, 52% zeros). No pre-transformed
# layer exists — this is the most processed form available. Technical/elemental channels
# (Au, Background, Ca, Fe, Na, P, Si) are present in `var` and excluded below for
# all expression analyses. The `layers["positivity"]` binary matrix is a derived feature,
# not a continuous expression source.

# %%
_bio_vars = [v for v in adata.var_names if v not in TECHNICAL_MARKERS]
adata_bio = adata[:, _bio_vars]
print(f"Biological markers: {len(_bio_vars)} / {adata.n_vars} total  (excluded: {adata.n_vars - len(_bio_vars)} technical)")

# %%
adata_bio.uns["dataset"] = "Patwa 2021"
plot_marker_distributions(adata_bio)

# %%
X = adata_bio.X
print("=== .X global stats (biological markers only) ===")
print(f"  dtype:         {X.dtype}")
print(f"  range:         [{X.min():.4f}, {X.max():.4f}]")
print(f"  mean / median: {X.mean():.4f} / {np.median(X):.4f}")
print(f"  pct zero:      {(X == 0).mean() * 100:.1f}%")

# %% [markdown]
# ### Cell type x marker profiles (scanpy)

# %%
sc.pl.dotplot(
    adata_bio,
    var_names=list(adata_bio.var_names),
    groupby="cell_type",
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.matrixplot(
    adata_bio,
    var_names=list(adata_bio.var_names),
    groupby="cell_type",
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.correlation_matrix(adata_bio, groupby="cell_type", figsize=(10, 8))

# %%
_corr = pd.DataFrame(adata_bio.X, columns=adata_bio.var_names).corr()
_fig, _ax = plt.subplots(figsize=(14, 12), dpi=120, facecolor="white")
_im = _ax.imshow(_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
_ax.set_xticks(range(len(_corr.columns)))
_ax.set_xticklabels(_corr.columns, rotation=90, fontsize=FS["sm"])
_ax.set_yticks(range(len(_corr.index)))
_ax.set_yticklabels(_corr.index, fontsize=FS["sm"])
_fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
_ax.set_title("Patwa 2021 — marker x marker correlation (all cells, biological markers only)", fontsize=FS["md"])
plt.tight_layout()
plt.show()
