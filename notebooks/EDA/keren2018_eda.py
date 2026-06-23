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

# %% [markdown]
# ## Spatial Graph Construction — Delaunay vs KNN
#
# Build two graphs using the biologically-motivated defaults
# (see `context_packages/datasets_overview.md`):
# - **Delaunay** — parameter-free physical adjacency (~6 neighbors, adaptive)
# - **KNN k=10** — fixed-size window (Schürch cellular-neighborhood method)
#
# `library_key="SampleID"` is essential: it builds the graph *within* each sample.
# Without it, cells from different samples (which share overlapping coordinate ranges)
# would be wrongly connected across samples.

# %%
adata.obs["SampleID"] = adata.obs["SampleID"].astype("category")
sq.gr.spatial_neighbors(adata, library_key="SampleID", coord_type="generic",
                        delaunay=True, key_added="delaunay")
sq.gr.spatial_neighbors(adata, library_key="SampleID", coord_type="generic",
                        n_neighs=10, key_added="knn")
print("obsp keys:", [k for k in adata.obsp])

# %% [markdown]
# ### Degree distribution — adaptive vs fixed
#
# Delaunay degree varies with local geometry (~6 mean, fewer at tissue edges, more in
# dense regions) — it reflects actual physical adjacency. KNN is fixed: every cell has
# exactly 10 outgoing edges by construction, so its degree distribution is a single spike.

# %%
deg_del = np.asarray((adata.obsp["delaunay_connectivities"] > 0).sum(axis=1)).ravel()
deg_knn = np.asarray((adata.obsp["knn_connectivities"] > 0).sum(axis=1)).ravel()

fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=120, facecolor="white")
for ax, deg, name, color in [(axes[0], deg_del, "Delaunay", "steelblue"),
                             (axes[1], deg_knn, "KNN (k=10)", "indianred")]:
    ax.hist(deg, bins=range(0, int(deg.max()) + 2), color=color, alpha=0.85, align="left")
    ax.axvline(deg.mean(), color="black", ls="--", lw=1.2)
    ax.set_title(f"{name} — degree  (mean {deg.mean():.1f})", fontsize=FS["md"])
    ax.set_xlabel("neighbors per cell", fontsize=FS["sm"])
    ax.set_ylabel("cells", fontsize=FS["sm"])
    ax.tick_params(labelsize=FS["xs"])
plt.suptitle("Degree distribution: adaptive (Delaunay) vs fixed (KNN)", fontsize=FS["lg"], y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Edge-length distribution (µm)
#
# Edge lengths converted to microns (Keren: 0.39 µm/pixel). Delaunay shows a long right
# tail — the spurious cross-gap / convex-hull edges that pruning targets. KNN's lengths
# are bounded by the local 10th-neighbor distance.

# %%
UM_PER_PX = 0.39
dist_del = adata.obsp["delaunay_distances"].data * UM_PER_PX
dist_knn = adata.obsp["knn_distances"].data * UM_PER_PX

fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=120, facecolor="white")
for ax, d, name, color in [(axes[0], dist_del, "Delaunay", "steelblue"),
                           (axes[1], dist_knn, "KNN (k=10)", "indianred")]:
    ax.hist(d, bins=80, range=(0, np.percentile(d, 99)), color=color, alpha=0.85, linewidth=0)
    ax.axvline(np.median(d), color="black", ls="--", lw=1.2, label=f"median {np.median(d):.1f} µm")
    ax.set_title(f"{name} — edge length", fontsize=FS["md"])
    ax.set_xlabel("edge length (µm)", fontsize=FS["sm"])
    ax.set_ylabel("edges", fontsize=FS["sm"])
    ax.legend(fontsize=FS["sm"])
    ax.tick_params(labelsize=FS["xs"])
plt.suptitle("Edge-length distribution (µm)", fontsize=FS["lg"], y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Pruning Delaunay long edges
#
# Drop edges beyond ~2.5× the median nearest-neighbor spacing (estimated per-cell as the
# shortest Delaunay edge). This removes cross-gap/hull artifacts while keeping true local
# adjacency. Pruned graph stored in `obsp["delaunay_pruned_connectivities"]`.

# %%
import scipy.sparse as sp

_D = adata.obsp["delaunay_distances"].tocsr()
_min_edge = np.array([
    _D.data[_D.indptr[i]:_D.indptr[i + 1]].min() if _D.indptr[i + 1] > _D.indptr[i] else np.nan
    for i in range(_D.shape[0])
])
nn_px = np.nanmedian(_min_edge)
thresh_px = 2.5 * nn_px
frac_pruned = (adata.obsp["delaunay_distances"].data > thresh_px).mean()
print(f"median NN spacing : {nn_px:.1f} px  ({nn_px * UM_PER_PX:.1f} µm)")
print(f"prune threshold   : {thresh_px:.1f} px  ({thresh_px * UM_PER_PX:.1f} µm)")
print(f"edges pruned      : {frac_pruned * 100:.2f}%")

_keep = adata.obsp["delaunay_distances"].copy()
_keep.data = (_keep.data <= thresh_px).astype(np.float64)
adata.obsp["delaunay_pruned_connectivities"] = sp.csr_matrix(
    adata.obsp["delaunay_connectivities"].multiply(_keep)
)

# %% [markdown]
# ### Graph overlay on tissue (squidpy)
#
# Edges drawn over one sample, cells colored by cell type. Visual contrast between
# Delaunay's adaptive mesh and KNN's denser, fixed-degree connectivity.

# %%
_one = adata[adata.obs["SampleID"] == 1].copy()
fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=150, facecolor="white")
sq.pl.spatial_scatter(_one, shape=None, color="all_group_name",
                      connectivity_key="delaunay_pruned_connectivities",
                      edges_width=0.3, edges_color="#888888",
                      size=40, ax=axes[0], title="Delaunay (pruned)")
sq.pl.spatial_scatter(_one, shape=None, color="all_group_name",
                      connectivity_key="knn_connectivities",
                      edges_width=0.3, edges_color="#888888",
                      size=40, ax=axes[1], title="KNN (k=10)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Neighborhood enrichment (squidpy)
#
# Which cell types co-localize beyond chance, using the pruned Delaunay graph. Positive
# (red) = enriched adjacency; negative (blue) = avoidance. The canonical squidpy graph-stat
# readout — the payoff of the graph construction above.

# %%
sq.gr.nhood_enrichment(adata, cluster_key="all_group_name",
                       connectivity_key="delaunay_pruned", seed=0)
sq.pl.nhood_enrichment(adata, cluster_key="all_group_name", figsize=(9, 9))
