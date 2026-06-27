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
    plot_all_samples, plot_marker_distributions, register_cmap, order_markers,
    prune_and_eval_graph, representative_samples, marker_cycle_gif, graph_celltype_panels,
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
plot_all_samples(adata, color_by="neighborhood_name", n_cols=7, s=4, cfg=CFG)

# %% [markdown]
# ## Analysis Setup — squidpy & scanpy

# %%
def _layer_stats(M, name):
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    v = M[~np.isnan(M)]
    print(f"=== {name} global stats ===")
    print(f"  shape:         {M.shape}")
    print(f"  range:         [{v.min():.4f}, {v.max():.4f}]")
    print(f"  mean / median: {v.mean():.4f} / {np.median(v):.4f}")
    print(f"  pct zero:      {(v == 0).mean() * 100:.1f}%")
    print(f"  pct NaN:       {np.isnan(M).mean() * 100:.1f}%")


# %% [markdown]
# ### Raw `.X` Evaluation
#
# Schurch `.X` contains raw CODEX fluorescence counts (0 – 54k). Distributions are
# heavily right-skewed across orders of magnitude — motivating the normalization below.

# %%
plot_marker_distributions(adata)

# %%
_layer_stats(adata.X, ".X (raw)")

# %% [markdown]
# ### Normalized `exprs_norm` Evaluation
#
# `exprs_norm` is the final Risom-pipeline layer persisted in the `.h5ad`
# (size-norm by cell area → `arcsinh(./0.5)` → winsorize 99.9 → per-marker 0–1). Every
# marker now lives on a common `[0, 1]` scale with the right tail tamed — the form used
# for all downstream analysis.
#
# > **Note (PCA):** `exprs_norm` is min-max `[0, 1]`, **not** z-scored. Markers with
# > broader spread therefore carry more weight in PCA. If equal per-marker weighting is
# > wanted for the embedding specifically, apply `sc.pp.scale` on top of `exprs_norm` for
# > the PCA step only (do not re-normalize for the distribution/marker-profile views).

# %%
plot_marker_distributions(adata, layer="exprs_norm")

# %%
_layer_stats(adata.layers["exprs_norm"], "exprs_norm")

# %% [markdown]
# ### Switch analysis layer → `exprs_norm`
#
# Set `.X` to the normalized layer so all scanpy/squidpy calls below operate on it (raw
# `X` remains recoverable from disk). All-NaN markers (cohort-unmeasured; relevant for
# Jackson, none here) are dropped by default so sc/sq calls don't choke on NaN.

# %%
_measured = ~np.all(np.isnan(adata.layers["exprs_norm"]), axis=0)
if (~_measured).any():
    print(f"Dropping {(~_measured).sum()} all-NaN marker(s): {list(adata.var_names[~_measured])}")
adata = adata[:, _measured].copy()
adata.X = adata.layers["exprs_norm"].copy()
# canonical functional ordering → all downstream plots inherit it
adata = adata[:, order_markers(list(adata.var_names))].copy()
print(f"Analysis layer set to exprs_norm — {adata.n_vars} markers, {adata.n_obs} cells.")

# %% [markdown]
# ### Cell type x marker profiles (scanpy)
#
# All views below now run on `exprs_norm` (via `.X`).

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
_ax.set_title("Schurch 2020 — marker x marker correlation (exprs_norm, all cells, no aggregation)", fontsize=FS["md"])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spatial Graph Construction — Delaunay vs KNN
#
# Build two graphs using the biologically-motivated defaults
# (see `context_packages/datasets_overview.md`):
# - **Delaunay** — parameter-free physical adjacency (~6 neighbors, adaptive)
# - **KNN k=10** — fixed-size window (Schürch's own cellular-neighborhood method)
#
# `library_key=CFG["sample_col"]` is essential: it builds the graph *within* each TMA core,
# so cells from different cores (which share overlapping coordinate ranges) are never
# wrongly connected. Built on `adata` (the `exprs_norm` analysis object).

# %%
# Drop cells with no cluster label so squidpy's nhood_enrichment gets a clean categorical.
# No-op for Schurch (cell_type is fully labeled) — kept for cross-dataset uniformity.
adata = adata[adata.obs["cell_type"].notna()].copy()
adata.obs["cell_type"] = adata.obs["cell_type"].cat.remove_unused_categories()

# KNN needs > KNN_K cells per library; drop samples too small to graph (no-op for Schurch —
# all cores are large; kept for cross-dataset uniformity). Also protects Delaunay.
KNN_K = 10
_sizes = adata.obs[CFG["sample_col"]].value_counts()
_small = _sizes.index[_sizes <= KNN_K]
if len(_small):
    print(f"Dropping {len(_small)} sample(s) with <= {KNN_K} cells (too small to graph)")
    adata = adata[~adata.obs[CFG["sample_col"]].isin(_small)].copy()
adata.obs[CFG["sample_col"]] = adata.obs[CFG["sample_col"]].astype("category").cat.remove_unused_categories()

sq.gr.spatial_neighbors(adata, library_key=CFG["sample_col"], coord_type="generic",
                        delaunay=True, key_added="delaunay")
sq.gr.spatial_neighbors(adata, library_key=CFG["sample_col"], coord_type="generic",
                        n_neighs=KNN_K, key_added="knn")
print("obsp keys:", [k for k in adata.obsp])

# %% [markdown]
# ### Prune + evaluate (single reproducible call)
#
# `prune_and_eval_graph` estimates the cell pitch (median per-cell shortest edge), prunes
# Delaunay edges beyond `factor`× pitch, writes `obsp["delaunay_pruned_connectivities"]`,
# and auto-plots degree + edge-length distributions (Delaunay pre/post-prune, KNN). µm scale
# comes from `cfg["um_per_px"]`.

# %%
prune_and_eval_graph(adata, cfg=CFG, factor=2.5)

# %% [markdown]
# ### Graph overlay on tissue (squidpy)
#
# Edges over one core, cells colored by cell type — Delaunay's adaptive mesh vs KNN's
# fixed-degree connectivity.

# %%
_one = adata[adata.obs["Region"] == "reg001"].copy()
fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=150, facecolor="white")
sq.pl.spatial_scatter(_one, shape=None, color="cell_type",
                      connectivity_key="delaunay_pruned_connectivities",
                      edges_width=0.3, edges_color="#888888",
                      size=40, ax=axes[0], title="Delaunay (pruned)")
sq.pl.spatial_scatter(_one, shape=None, color="cell_type",
                      connectivity_key="knn_connectivities",
                      edges_width=0.3, edges_color="#888888",
                      size=40, ax=axes[1], title="KNN (k=10)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Neighborhood enrichment (squidpy)
#
# Which cell types co-localize beyond chance, using the pruned Delaunay graph. Red =
# enriched adjacency; blue = avoidance.

# %%
sq.gr.nhood_enrichment(adata, cluster_key="cell_type",
                       connectivity_key="delaunay_pruned", seed=0)
sq.pl.nhood_enrichment(adata, cluster_key="cell_type", figsize=(9, 9))

# %% [markdown]
# ## Cell-Type Graph Panels — representative samples
#
# Static companion to the marker-cycle GIF below: the pruned Delaunay graph for one
# representative sample per group_name, cells colored by cell type (same coloring
# convention as the spatial plots above). The cell-type legend renders as its own figure.

# %%
import importlib, data.EDA as eda
importlib.reload(eda)
from data.EDA import representative_samples, graph_celltype_panels   # rebind the names

# %%
_reps, _titles = representative_samples(adata, cfg=CFG, by="group_name", method="median")
graph_celltype_panels(adata, _reps, cfg=CFG, titles=_titles)

# %% [markdown]
# ## Marker Cycle GIF — expression on the graph, by category
#
# One representative sample per `group_name` (largest), side by side as graphs (pruned Delaunay
# edges as a static backdrop), cycling markers in canonical functional order. Each frame's
# colormap is set by the marker's functional group; the title shows the marker and its
# category. Headless-friendly: frames render via Agg and stitch to a GIF on disk.

# %%
from pathlib import Path as _Path
from IPython.display import Image as _Image

_reps, _titles = representative_samples(adata, cfg=CFG, by="group_name")
_Path("figures").mkdir(exist_ok=True)
_gif = marker_cycle_gif(
    adata, _reps, cfg=CFG, layer="exprs_norm",
    titles=_titles, fps=2, out_path="figures/schurch2020_marker_cycle.gif",
)
_Image(filename=_gif)

# %%
