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
# # Schurch 2020 — CRC CODEX EDA
#
# Exploratory data analysis of the Schurch et al. 2020 colorectal cancer (CRC) dataset,
# acquired by CODEX multiplexed imaging.
#
# 35 patients, 70 TMA cores (2 per patient), 258,385 cells, 58 protein markers.

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
    SCHURCH_CFG, SCHURCH_NEIGHBORHOOD_CMAP, FS,
    summarize_metadata, spatial_info, cat_breakdown,
    plot_all_samples, plot_marker_distributions, register_cmap,
)

CFG = SCHURCH_CFG

# %% [markdown]
# ## Load Data

# %%
DATA_PATH = "../../../data/schurch2020/crc.h5ad"

adata_raw = ad.read_h5ad(DATA_PATH)
adata_raw

# %%
# Re-run this cell to restore clean state without restarting kernel
adata = adata_raw.copy()
adata.uns["dataset"] = "Schurch 2020"
register_cmap(adata, "neighborhood_name", SCHURCH_NEIGHBORHOOD_CMAP)

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
    adata[adata.obs["Region"] == "reg001"],
    shape=None,
    color="neighborhood_name",
    size=4,
    figsize=(8, 8),
    dpi=150,
)

# %% [markdown]
# ## Subsample Viewer
#
# 8 of 70 cores, colored by neighborhood_name, shared palette.

# %%
_sample_ids = sorted(adata.obs["Region"].unique())[:8]
_n_cols, _n_rows = 4, 2
_fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 5), dpi=150, facecolor="white")
for _ax, _sid in zip(_axes.flatten(), _sample_ids):
    sq.pl.spatial_scatter(adata[adata.obs["Region"] == _sid], shape=None, color="neighborhood_name",
                          size=4, ax=_ax, title=str(_sid))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Full Dataset Viewer
#
# All 70 TMA cores.

# %%
plot_all_samples(adata, color_by="neighborhood_name", n_cols=7, s=1, cfg=CFG)

# %% [markdown]
# ## Analysis Setup — squidpy & scanpy

# %% [markdown]
# ### `.X` Evaluation
#
# Schurch `.X` contains raw CODEX fluorescence counts (0 – 54k). No pre-transformed
# layer exists — this is the most processed form available. Distributions will be
# right-skewed; arcsinh normalization is needed before downstream modeling.

# %%
plot_marker_distributions(adata)

# %%
X = adata.X
print("=== .X global stats ===")
print(f"  dtype:         {X.dtype}")
print(f"  range:         [{X.min():.4f}, {X.max():.4f}]")
print(f"  mean / median: {X.mean():.4f} / {np.median(X):.4f}")
print(f"  pct zero:      {(X == 0).mean() * 100:.1f}%")

# %% [markdown]
# ### Cell type x marker profiles (scanpy)

# %%
sc.pl.dotplot(
    adata,
    var_names=list(adata.var_names),
    groupby="cell_type",
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.matrixplot(
    adata,
    var_names=list(adata.var_names),
    groupby="cell_type",
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.correlation_matrix(adata, groupby="cell_type", figsize=(10, 8))

# %%
_corr = pd.DataFrame(adata.X, columns=adata.var_names).corr()
_fig, _ax = plt.subplots(figsize=(14, 12), dpi=120, facecolor="white")
_im = _ax.imshow(_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
_ax.set_xticks(range(len(_corr.columns)))
_ax.set_xticklabels(_corr.columns, rotation=90, fontsize=FS["sm"])
_ax.set_yticks(range(len(_corr.index)))
_ax.set_yticklabels(_corr.index, fontsize=FS["sm"])
_fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
_ax.set_title("Schurch 2020 — marker x marker correlation (all cells, no aggregation)", fontsize=FS["md"])
plt.tight_layout()
plt.show()
