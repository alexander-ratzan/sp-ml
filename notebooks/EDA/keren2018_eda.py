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
# # Keren 2018 — TNBC MIBI-TOF EDA
#
# Exploratory data analysis of the Keren et al. 2018 triple-negative breast cancer (TNBC) dataset,
# acquired by multiplexed ion beam imaging (MIBI-TOF).
#
# 34 patients, 173,205 cells, 36 protein markers.

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
    summarize_metadata, spatial_info, cat_breakdown,
    plot_all_samples,
    plot_marker_distributions,
    register_cmap,
    KEREN_CELLTYPE_CMAP, FS,
)

# %% [markdown]
# ## Load Data

# %%
DATA_PATH       = "../../../data/keren2018/tnbc.h5ad"
CHECKPOINT_PATH = "../../../data/keren2018/tnbc_processed.h5ad"

adata_raw = ad.read_h5ad(DATA_PATH)
adata_raw

# %%
# Re-run this cell to restore clean state without restarting kernel
adata = adata_raw.copy()
register_cmap(adata, "all_group_name", KEREN_CELLTYPE_CMAP)

# %% [markdown]
# ## Dataset Overview

# %%
summarize_metadata(adata)

# %%
spatial_info(adata)

# %%
cat_breakdown(adata)

# %% [markdown]
# ## Single Sample Viewer

# %%
sq.pl.spatial_scatter(
    adata[adata.obs["SampleID"] == 1],
    shape=None,
    color="all_group_name",
    size=4,
    figsize=(8, 8),
    dpi=150,
)

# %% [markdown]
# ## Subsample Viewer
#
# 8 of 34 slices, colored by cell type, shared palette.

# %%
_sample_ids = sorted(adata.obs["SampleID"].unique())[:8]
_n_cols, _n_rows = 4, 2
_fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 5), dpi=150, facecolor="white")
for _ax, _sid in zip(_axes.flatten(), _sample_ids):
    sq.pl.spatial_scatter(adata[adata.obs["SampleID"] == _sid], shape=None, color="all_group_name",
                          size=4, ax=_ax, title=str(_sid))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Full Dataset Viewer
#
# All 34 slices.

# %%
plot_all_samples(adata, color_by="all_group_name", n_cols=6)

# %% [markdown]
# ### `.X` Evaluation
#
# Determine the normalization state of `.X` before running any downstream analysis.

# %%
adata.uns["dataset"] = "Keren 2018"
plot_marker_distributions(adata)

# %%
X = adata.X
print("=== .X identity check ===")
print(f"X == obsm[X_data]:  {np.allclose(X, adata.obsm['X_data'])}")
print()
print("=== .X global stats ===")
print(f"  dtype:         {X.dtype}")
print(f"  range:         [{X.min():.4f}, {X.max():.4f}]")
print(f"  mean / median: {X.mean():.4f} / {np.median(X):.4f}")
print(f"  pct negative:  {(X < 0).mean() * 100:.1f}%")

# %% [markdown]
# ### Cell type × marker profiles (scanpy)
#
# Dotplot shows mean expression (dot color) and fraction of cells expressing (dot size) per cell type.
# `standard_scale="var"` rescales each marker 0–1 across cell types so the pattern across types is visible despite the differing absolute scales of arcsinh-transformed markers.

# %%
sc.pl.dotplot(
    adata,
    var_names=list(adata.var_names),
    groupby="all_group_name",
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.matrixplot(
    adata,
    var_names=list(adata.var_names),
    groupby="all_group_name",
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.correlation_matrix(adata, groupby="all_group_name", figsize=(10, 8))

# %%
_corr = pd.DataFrame(adata.X, columns=adata.var_names).corr()
_fig, _ax = plt.subplots(figsize=(14, 12), dpi=120, facecolor="white")
_im = _ax.imshow(_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
_ax.set_xticks(range(len(_corr.columns)))
_ax.set_xticklabels(_corr.columns, rotation=90, fontsize=FS["sm"])
_ax.set_yticks(range(len(_corr.index)))
_ax.set_yticklabels(_corr.index, fontsize=FS["sm"])
_fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
_ax.set_title("Keren 2018 — marker × marker correlation (all cells, no aggregation)", fontsize=FS["md"])
plt.tight_layout()
plt.show()
