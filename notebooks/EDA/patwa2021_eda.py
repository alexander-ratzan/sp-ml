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
_r = next(p for p in [Path().resolve(), *Path().resolve().parents] if (p / "sp_ml").is_dir() and (p / "notebooks").is_dir())
if str(_r) not in sys.path: sys.path.insert(0, str(_r))

import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

from sp_ml.data.EDA import (
    PATWA_CFG, PATWA_CELLTYPE_CMAP, TECHNICAL_MARKERS, FS,
    summarize_metadata, spatial_info, cat_breakdown,
    plot_all_samples, plot_marker_distributions, register_cmap, order_markers,
    prune_and_eval_graph, representative_samples, marker_cycle_gif, graph_celltype_panels,
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
# Patwa `.X` contains MIBI intensities already area-normalized upstream by RASP-MIBI
# (0 – 248, ~52% zeros). Technical/elemental channels (Au, Background, Ca, Fe, Na, P, Si)
# are excluded for all expression analyses. The `layers["positivity"]` binary matrix is a
# derived feature, not a continuous expression source.

# %%
_bio_vars = [v for v in adata.var_names if v not in TECHNICAL_MARKERS]
adata_bio = adata[:, _bio_vars].copy()
adata_bio.uns["dataset"] = "Patwa 2021"
print(f"Biological markers: {len(_bio_vars)} / {adata.n_vars} total  (excluded: {adata.n_vars - len(_bio_vars)} technical)")

# %%
plot_marker_distributions(adata_bio)

# %%
_layer_stats(adata_bio.X, ".X (raw, biological markers)")

# %% [markdown]
# ### Normalized `exprs_norm` Evaluation
#
# `exprs_norm` is the final Risom-pipeline layer persisted in the `.h5ad`
# (`arcsinh(./5)` → winsorize 99.9 → per-marker 0–1). Every marker now lives on a common
# `[0, 1]` scale — the form used for all downstream analysis.
#
# > **Note (PCA):** `exprs_norm` is min-max `[0, 1]`, **not** z-scored. Markers with broader
# > spread carry more weight in PCA; apply `sc.pp.scale` on top for the embedding step only
# > if equal per-marker weighting is wanted.

# %%
plot_marker_distributions(adata_bio, layer="exprs_norm")

# %%
_layer_stats(adata_bio.layers["exprs_norm"], "exprs_norm (biological markers)")

# %% [markdown]
# ### Switch analysis layer → `exprs_norm`
#
# Set `.X` to the normalized layer so all scanpy/squidpy calls below operate on it (raw
# `X` remains recoverable from disk). All-NaN markers (cohort-unmeasured; none here) are
# dropped by default.

# %%
_measured = ~np.all(np.isnan(adata_bio.layers["exprs_norm"]), axis=0)
if (~_measured).any():
    print(f"Dropping {(~_measured).sum()} all-NaN marker(s): {list(adata_bio.var_names[~_measured])}")
adata_bio = adata_bio[:, _measured].copy()
adata_bio.X = adata_bio.layers["exprs_norm"].copy()
# canonical functional ordering → all downstream plots inherit it
adata_bio = adata_bio[:, order_markers(list(adata_bio.var_names))].copy()
print(f"Analysis layer set to exprs_norm — {adata_bio.n_vars} markers, {adata_bio.n_obs} cells.")

# %% [markdown]
# ### Cell type x marker profiles (scanpy)
#
# All views below now run on `exprs_norm` (via `.X`).

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
_ax.set_title("Patwa 2021 — marker x marker correlation (exprs_norm, biological markers)", fontsize=FS["md"])
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
# `library_key=CFG["sample_col"]` is essential: it builds the graph *within* each sample,
# so cells from different samples (which share overlapping coordinate ranges) are never
# wrongly connected. Built on `adata_bio` (the `exprs_norm` analysis object).

# %%
# Drop cells with no cluster label so squidpy's nhood_enrichment gets a clean categorical.
# No-op for Patwa (cell_type is fully labeled) — kept for cross-dataset uniformity.
adata_bio = adata_bio[adata_bio.obs["cell_type"].notna()].copy()
adata_bio.obs["cell_type"] = adata_bio.obs["cell_type"].cat.remove_unused_categories()

# KNN needs > KNN_K cells per library; drop samples too small to graph (no-op for Patwa —
# all samples are large; kept for cross-dataset uniformity). Also protects Delaunay.
KNN_K = 10
_sizes = adata_bio.obs[CFG["sample_col"]].value_counts()
_small = _sizes.index[_sizes <= KNN_K]
if len(_small):
    print(f"Dropping {len(_small)} sample(s) with <= {KNN_K} cells (too small to graph)")
    adata_bio = adata_bio[~adata_bio.obs[CFG["sample_col"]].isin(_small)].copy()
adata_bio.obs[CFG["sample_col"]] = adata_bio.obs[CFG["sample_col"]].astype("category").cat.remove_unused_categories()

sq.gr.spatial_neighbors(adata_bio, library_key=CFG["sample_col"], coord_type="generic",
                        delaunay=True, key_added="delaunay")
sq.gr.spatial_neighbors(adata_bio, library_key=CFG["sample_col"], coord_type="generic",
                        n_neighs=KNN_K, key_added="knn")
print("obsp keys:", [k for k in adata_bio.obsp])

# %% [markdown]
# ### Prune + evaluate (single reproducible call)
#
# `prune_and_eval_graph` estimates the cell pitch (median per-cell shortest edge), prunes
# Delaunay edges beyond `factor`× pitch, writes `obsp["delaunay_pruned_connectivities"]`,
# and auto-plots degree + edge-length distributions (Delaunay pre/post-prune, KNN). µm scale
# comes from `cfg["um_per_px"]`.

# %%
prune_and_eval_graph(adata_bio, cfg=CFG, factor=2.5)

# %% [markdown]
# ### Graph overlay on tissue (squidpy)
#
# Edges over one sample, cells colored by cell type — Delaunay's adaptive mesh vs KNN's
# fixed-degree connectivity.

# %%
_one = adata_bio[adata_bio.obs["patient_id"] == 1].copy()
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
sq.gr.nhood_enrichment(adata_bio, cluster_key="cell_type",
                       connectivity_key="delaunay_pruned", seed=0)
sq.pl.nhood_enrichment(adata_bio, cluster_key="cell_type", figsize=(9, 9))

# %% [markdown]
# ## Cell-Type Graph Panels — representative samples
#
# Static companion to the marker-cycle GIF below: the pruned Delaunay graph for one
# representative sample per Architecture, cells colored by cell type (same coloring
# convention as the spatial plots above). The cell-type legend renders as its own figure.

# %%
_reps, _titles = representative_samples(adata_bio, cfg=CFG, by="Architecture")
graph_celltype_panels(adata_bio, _reps, cfg=CFG, titles=_titles)

# %% [markdown]
# ## Marker Cycle GIF — expression on the graph, by category
#
# One representative sample per `Architecture` (largest), side by side as graphs (pruned Delaunay
# edges as a static backdrop), cycling markers in canonical functional order. Each frame's
# colormap is set by the marker's functional group; the title shows the marker and its
# category. Headless-friendly: frames render via Agg and stitch to a GIF on disk.

# %%
from pathlib import Path as _Path
from IPython.display import Image as _Image

_reps, _titles = representative_samples(adata_bio, cfg=CFG, by="Architecture")
_Path("figures").mkdir(exist_ok=True)
_gif = marker_cycle_gif(
    adata_bio, _reps, cfg=CFG, layer="exprs_norm",
    titles=_titles, fps=2, out_path="figures/patwa2021_marker_cycle.gif",
)
_Image(filename=_gif)
