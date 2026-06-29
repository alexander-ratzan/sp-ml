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
# # Jackson-Fischer 2020 — Breast Cancer IMC EDA (Basel)
#
# Exploratory data analysis of the Jackson & Fischer et al. 2020 breast cancer dataset,
# acquired by Imaging Mass Cytometry (IMC).
#
# Basel cohort: 100 patients, 285,851 cells, 45 IMC markers.
# Expression matrix: raw ion counts in X, arcsinh-transformed in layers["exprs"].

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
    JACKSON_CFG, TECHNICAL_MARKERS, FS,
    summarize_metadata, spatial_info, cat_breakdown,
    plot_marker_distributions, prune_and_eval_graph, order_markers,
    representative_samples, marker_cycle_gif, graph_celltype_panels,
)

CFG = JACKSON_CFG

DATASET_LABEL = "Jackson 2020 (Basel)"

# %% [markdown]
# ## Load Data

# %%
DATA_PATH = "../../../data/jacksonfischer2020/basel/basel.h5ad"

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
# 8 of 100 images, colored by cell_metacluster.

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
# ## Full Dataset Viewer
#
# All 100 images.

# %%
from sp_ml.data.EDA import plot_all_samples
plot_all_samples(adata, color_by="tumor_clinical_type", n_cols=10, s=1, cfg=CFG)


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
    print(f"  pct NaN:       {np.isnan(M).mean() * 100:.1f}%")


# %% [markdown]
# ### Raw `.X` Evaluation
#
# Jackson `.X` contains raw IMC ion counts with NaN where markers were unmeasured in a
# cohort. Ruthenium bead channels (Ru96–Ru104) are purely technical and excluded below.
# Shown as the baseline before our normalization.

# %%
_bio_vars = [v for v in adata.var_names if v not in TECHNICAL_MARKERS]
adata_bio = adata[:, _bio_vars].copy()
adata_bio.uns["dataset"] = DATASET_LABEL
print(f"Biological markers: {adata_bio.n_vars} / {adata.n_vars} total  (excluded {adata.n_vars - adata_bio.n_vars} Ru channels)")

# %%
plot_marker_distributions(adata_bio)

# %%
_layer_stats(adata_bio.X, ".X (raw IMC counts, biological markers)")

# %% [markdown]
# ### Normalized `exprs_norm` Evaluation
#
# `exprs_norm` is the final Risom-pipeline layer persisted in the `.h5ad` (Jackson's
# publisher `exprs` arcsinh c=1 → winsorize 99.9 → per-marker 0–1). Every marker now lives
# on a common `[0, 1]` scale — the form used for all downstream analysis.
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
# Densify `exprs_norm` into `.X` so all scanpy/squidpy calls below operate on it (raw `X`
# remains recoverable from disk). All-NaN markers (cohort-unmeasured — Basel:
# EpCAM/CTNNB/SOX9) are **dropped by default** so sc/sq calls don't choke on NaN.

# %%
_E = adata_bio.layers["exprs_norm"]
_E = (_E.toarray() if hasattr(_E, "toarray") else np.asarray(_E)).astype(np.float32)
_measured = ~np.isnan(_E).all(axis=0)
if (~_measured).any():
    print(f"Dropping {(~_measured).sum()} all-NaN marker(s): {list(adata_bio.var_names[~_measured])}")
adata_bio = adata_bio[:, _measured].copy()
adata_bio.X = _E[:, _measured]
# canonical functional ordering → all downstream plots inherit it
adata_bio = adata_bio[:, order_markers(list(adata_bio.var_names))].copy()
print(f"Analysis layer set to exprs_norm — {adata_bio.n_vars} markers, {adata_bio.n_obs} cells.")

# %% [markdown]
# ### Cell type x marker profiles (scanpy)
#
# `cell_metacluster` holds phenograph cluster IDs (not named cell types).
# All views below run on `exprs_norm` (via `.X`).

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
_ax.set_title(f"{DATASET_LABEL} — marker x marker correlation (exprs_norm, no aggregation)", fontsize=FS["md"])
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
# `library_key=CFG["sample_col"]` (image_name) is essential: it builds the graph *within*
# each image, never connecting cells across images. Built on `adata_bio` (the `exprs_norm`
# analysis object).

# %%
# Drop cells with no cluster label — cell_metacluster has NA in IMC data, and squidpy's
# nhood_enrichment requires a clean categorical. No-op for fully-labeled datasets.
adata_bio = adata_bio[adata_bio.obs["cell_metacluster"].notna()].copy()
adata_bio.obs["cell_metacluster"] = adata_bio.obs["cell_metacluster"].cat.remove_unused_categories()

# KNN needs > KNN_K cells per library; drop samples too small to graph (Jackson has tiny
# images — no-op for datasets with large samples). Also protects Delaunay from degenerate FOVs.
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
# comes from `cfg["um_per_px"]` (Jackson = 1.0).

# %%
prune_and_eval_graph(adata_bio, cfg=CFG, factor=2.5)

# %% [markdown]
# ### Graph overlay on tissue (squidpy)
#
# Edges over one image, cells colored by phenograph metacluster — Delaunay's adaptive mesh
# vs KNN's fixed-degree connectivity.

# %%
_one = adata_bio[adata_bio.obs["image_name"] == sample].copy()
fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=150, facecolor="white")
sq.pl.spatial_scatter(_one, shape=None, color="cell_metacluster",
                      connectivity_key="delaunay_pruned_connectivities",
                      edges_width=0.3, edges_color="#888888",
                      size=40, ax=axes[0], title="Delaunay (pruned)")
sq.pl.spatial_scatter(_one, shape=None, color="cell_metacluster",
                      connectivity_key="knn_connectivities",
                      edges_width=0.3, edges_color="#888888",
                      size=40, ax=axes[1], title="KNN (k=10)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Neighborhood enrichment (squidpy)
#
# Which metaclusters co-localize beyond chance, using the pruned Delaunay graph. Red =
# enriched adjacency; blue = avoidance.

# %%
sq.gr.nhood_enrichment(adata_bio, cluster_key="cell_metacluster",
                       connectivity_key="delaunay_pruned", seed=0)
sq.pl.nhood_enrichment(adata_bio, cluster_key="cell_metacluster", figsize=(9, 9))

# %% [markdown]
# ## Cell-Type Graph Panels — representative samples
#
# Static companion to the marker-cycle GIF below: the pruned Delaunay graph for one
# representative sample per tumor_clinical_type, cells colored by cell metacluster (same
# coloring convention as the spatial plots above). The legend renders as its own figure.

# %%
_reps, _titles = representative_samples(adata_bio, cfg=CFG, by="tumor_clinical_type")
graph_celltype_panels(adata_bio, _reps, cfg=CFG, color_by="cell_metacluster", titles=_titles)

# %% [markdown]
# ## Marker Cycle GIF — expression on the graph, by category
#
# One representative sample per `tumor_clinical_type` (largest), side by side as graphs (pruned Delaunay
# edges as a static backdrop), cycling markers in canonical functional order. Each frame's
# colormap is set by the marker's functional group; the title shows the marker and its
# category. Headless-friendly: frames render via Agg and stitch to a GIF on disk.

# %%
from pathlib import Path as _Path
from IPython.display import Image as _Image

_reps, _titles = representative_samples(adata_bio, cfg=CFG, by="tumor_clinical_type")
_Path("figures").mkdir(exist_ok=True)
_gif = marker_cycle_gif(
    adata_bio, _reps, cfg=CFG, layer="exprs_norm",
    titles=_titles, fps=2, out_path="figures/jacksonfischer2020_basel_marker_cycle.gif",
)
_Image(filename=_gif)
