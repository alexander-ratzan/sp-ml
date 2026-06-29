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
# # Keren 2018 — TNBC MIBI EDA
#
# Exploratory data analysis of the Keren et al. 2018 triple-negative breast cancer (TNBC) dataset,
# acquired by multiplexed ion beam imaging (MIBI).
#
# 34 patients, 173,205 cells, 36 protein markers.

# %%
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
_r = next(p for p in [Path().resolve(), *Path().resolve().parents] if (p / "sp_ml").is_dir() and (p / "notebooks").is_dir())
if str(_r) not in sys.path: sys.path.insert(0, str(_r))

import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

from sp_ml.data.EDA import (
    KEREN_CFG,
    summarize_metadata, spatial_info, cat_breakdown,
    plot_all_samples,
    plot_marker_distributions,
    prune_and_eval_graph,
    representative_samples, marker_cycle_gif, graph_celltype_panels,
    register_cmap, order_markers,
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
# Keren `.X` is already arcsinh-transformed by the Nolan-lab MIBI pipeline
# (background-subtracted, so values can be negative). This is the publisher's processed
# form — shown here as the baseline before our normalization.

# %%
adata.uns["dataset"] = "Keren 2018"
plot_marker_distributions(adata)

# %%
print("=== .X identity check ===")
print(f"X == obsm[X_data]:  {np.allclose(adata.X, adata.obsm['X_data'])}\n")
_layer_stats(adata.X, ".X (Nolan arcsinh)")

# %% [markdown]
# ### Normalized `exprs_norm` Evaluation
#
# `exprs_norm` is the final Risom-pipeline layer persisted in the `.h5ad` (Keren's `X` is
# already arcsinh, so the pipeline just winsorizes 99.9 → per-marker 0–1). Every marker now
# lives on a common `[0, 1]` scale — the form used for all downstream analysis.
#
# > **Note (PCA):** `exprs_norm` is min-max `[0, 1]`, **not** z-scored. Markers with broader
# > spread carry more weight in PCA; apply `sc.pp.scale` on top for the embedding step only
# > if equal per-marker weighting is wanted.

# %%
plot_marker_distributions(adata, layer="exprs_norm")

# %%
_layer_stats(adata.layers["exprs_norm"], "exprs_norm")

# %% [markdown]
# ### Switch analysis layer → `exprs_norm`
#
# Set `.X` to the normalized layer so all scanpy/squidpy calls below operate on it (raw
# `X` remains recoverable from disk). All-NaN markers (cohort-unmeasured; none here) are
# dropped by default.

# %%
_measured = ~np.all(np.isnan(adata.layers["exprs_norm"]), axis=0)
if (~_measured).any():
    print(f"Dropping {(~_measured).sum()} all-NaN marker(s): {list(adata.var_names[~_measured])}")
adata = adata[:, _measured].copy()
adata.X = adata.layers["exprs_norm"].copy()
# canonical functional ordering → all downstream plots (dotplot/matrixplot/corr) inherit it
adata = adata[:, order_markers(list(adata.var_names))].copy()
print(f"Analysis layer set to exprs_norm — {adata.n_vars} markers, {adata.n_obs} cells.")

# %% [markdown]
# ### Cell type × marker profiles (scanpy)
#
# All views below now run on `exprs_norm` (via `.X`). Dotplot shows mean expression (dot
# color) and fraction of cells expressing (dot size) per cell type. `standard_scale="var"`
# rescales each marker 0–1 across cell types so the pattern across types stays visible.

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
_ax.set_title("Keren 2018 — marker × marker correlation (exprs_norm, all cells, no aggregation)", fontsize=FS["md"])
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
# ### Prune + evaluate (single reproducible call)
#
# `prune_and_eval_graph` estimates the cell pitch (median per-cell shortest edge), prunes
# Delaunay edges beyond `factor`× pitch, writes `obsp["delaunay_pruned_connectivities"]`,
# and auto-plots degree + edge-length distributions across three columns: Delaunay
# pre-prune, Delaunay post-prune, and KNN (for the adaptive-vs-fixed comparison). µm scale
# comes from `cfg["um_per_px"]`. Reproduce on any dataset by passing its CFG.

# %%
prune_and_eval_graph(adata, cfg=KEREN_CFG, factor=2.5)

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

# %% [markdown]
# ## Cell-Type Graph Panels — representative samples
#
# Static companion to the marker-cycle GIF below: the pruned Delaunay graph for one
# representative sample per subtype, cells colored by cell type (same coloring convention
# as the spatial plots above). The cell-type legend renders as its own figure.

# %%
_reps, _titles = representative_samples(adata, cfg=KEREN_CFG, by="subtype")
graph_celltype_panels(adata, _reps, cfg=KEREN_CFG, titles=_titles)

# %% [markdown]
# ## Marker Cycle GIF — expression on the graph, by category
#
# One representative sample per `subtype` (the largest), shown side by side as graphs (pruned
# Delaunay edges as a static backdrop), cycling identically through every marker on the shared
# `exprs_norm` 0–1 scale. Lets you scan how each marker's spatial pattern differs across
# categories with structure held constant. Headless-friendly: frames render via Agg and stitch
# to a GIF written to disk, displayed inline below.

# %%
from pathlib import Path as _Path
from IPython.display import Image as _Image

_reps, _titles = representative_samples(adata, cfg=KEREN_CFG, by="subtype")
_Path("figures").mkdir(exist_ok=True)
_gif = marker_cycle_gif(
    adata, _reps, cfg=KEREN_CFG, layer="exprs_norm",
    titles=_titles, fps=2, out_path="figures/keren_marker_cycle.gif",
)
_Image(filename=_gif)
