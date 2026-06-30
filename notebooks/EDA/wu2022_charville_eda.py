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
# # Wu 2022 (Charville / Stanford-CRC) — CRC CODEX EDA
#
# Exploratory data analysis of the Charville cohort from Wu et al. 2022 (SPACE-GM),
# acquired by CODEX multiplexed imaging.
#
# 162 patients, 292 regions, 632,280 cells, 40 protein markers (native Charville panel).
#
# This notebook mirrors `schurch2020_eda.py` section-for-section (both are CRC/CODEX).
# Key schema differences from Schurch, per
# `context_packages/wu2022_integration_contract.md`:
# - **`.X` is publisher z-scored** (per-marker centered near 0, ~67% negative), **not raw
#   counts** — the raw-`.X` markdown below is adapted accordingly.
# - `layers["exprs_norm"]` = per-marker min-max 0–1 within the cohort (the modeling input).
# - No neighborhood / cellular-niche annotation exists for Wu, so the Schurch
#   `neighborhood_name` views are replaced with `cell_type` colorings throughout.
# - Charville is **RESOLVED** for patients (162) — patient-level grouping is valid here
#   (unlike the UPMC/DFCI Wu cohorts where `patient_id == region_id`).

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
    FS,
    summarize_metadata, spatial_info, cat_breakdown,
    plot_marker_distributions, register_cmap, order_markers,
    prune_and_eval_graph, representative_samples, graph_celltype_panels,
)

# %% [markdown]
# ## Inline CFG
#
# The Wu cohorts are not wired into `EDA.py`'s CFG system (see the integration contract),
# so we define a small inline `cfg` mirroring `SCHURCH_CFG`'s keys. `sample_col` is the
# CODEX region, `patient_col` is the resolved patient, `celltype_col` is the coarse shared
# lineage, and `cat_cols` lists the Charville clinical label columns broadcast per region.
# `label_col` is set to `alive_or_deceased` (a clean 2-class, no-NaN clinical label) so the
# representative-sample / GIF helpers — which group by `cfg["label_col"]` — have something
# meaningful to stratify on (Wu has no `group_name`/neighborhood annotation).

# %%
# CODEX lateral resolution (same instrument family as Schurch: ~377 nm/px).
CFG = {
    "publication": "Wu et al. 2022 — Charville",
    "technology": "CODEX",
    "disease": "CRC",
    "sample_col": "region_id",
    "patient_col": "patient_id",
    "celltype_col": "cell_type",
    "label_col": "alive_or_deceased",
    "um_per_px": 0.377,
    "size_col": "size",
    "arcsinh_cofactor": None,   # publisher z-scored; no arcsinh applied
    "cat_cols": [
        "region_id", "patient_id", "cell_type", "cell_type_raw",
        "primary_outcome", "recurrence", "alive_or_deceased",
        "type_of_first_recurrence", "grade_differentiation",
    ],
}

# %% [markdown]
# ## Load Data

# %%
DATA_PATH = "../../../data/wu2022/charville.h5ad"

adata_raw = ad.read_h5ad(DATA_PATH)
adata_raw

# %%
# Re-run this cell to restore clean state without restarting kernel
adata = adata_raw.copy()
print("dataset:", adata.uns.get("dataset"))
print("region_patient_resolution:", adata.uns.get("region_patient_resolution"))

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
#
# Wu has no neighborhood annotation, so cells are colored by `cell_type` (coarse lineage).

# %%
_one_region = sorted(adata.obs[CFG["sample_col"]].unique())[0]
sq.pl.spatial_scatter(
    adata[adata.obs[CFG["sample_col"]] == _one_region],
    shape=None,
    color=CFG["celltype_col"],
    size=4,
    figsize=(8, 8),
    dpi=150,
    title=str(_one_region),
)

# %% [markdown]
# ## Subsample Viewer
#
# 8 of 292 regions, colored by `cell_type`, shared palette. (We never plot all 292 regions
# — too many to render tractably; see the contract's tractability note.)

# %%
_sample_ids = sorted(adata.obs[CFG["sample_col"]].unique())[:8]
_n_cols, _n_rows = 4, 2
_fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 5), dpi=150, facecolor="white")
for _ax, _sid in zip(_axes.flatten(), _sample_ids):
    sq.pl.spatial_scatter(adata[adata.obs[CFG["sample_col"]] == _sid], shape=None,
                          color=CFG["celltype_col"], size=4, ax=_ax, title=str(_sid))
plt.tight_layout()
plt.show()

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
    print(f"  pct negative:  {(v < 0).mean() * 100:.1f}%")
    print(f"  pct NaN:       {np.isnan(M).mean() * 100:.1f}%")


# %% [markdown]
# ### Raw `.X` Evaluation
#
# Unlike Schurch (raw CODEX fluorescence counts, 0–54k), Charville `.X` is **publisher
# z-scored** on the native 40-marker panel — per-marker centered near 0 with a substantial
# negative fraction (~67% of values < 0, verified in the contract). Distributions are
# therefore roughly symmetric around zero rather than right-skewed, and `pct zero` is ~0
# (z-scores rarely land exactly on 0). This is already variance-stabilized by the
# publisher, which is why the `exprs_norm` recipe below is a plain min-max with no
# arcsinh / size-norm / winsorize.

# %%
plot_marker_distributions(adata)

# %%
_layer_stats(adata.X, ".X (publisher z-scored)")

# %% [markdown]
# ### Normalized `exprs_norm` Evaluation
#
# `exprs_norm` is the modeling layer: per-marker **min-max 0–1 within the cohort** applied
# on top of the publisher z-scores (no arcsinh, no size norm, no winsorize — the publisher
# already variance-stabilized via z-score). Every marker now lives on a common `[0, 1]`
# scale — the form used for all downstream analysis.
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
# z-scored `X` remains recoverable from disk). All-NaN markers (cohort-unmeasured) are
# dropped by default so sc/sq calls don't choke on NaN — Charville has none (all 40
# markers measured), but the guard is kept for cross-dataset uniformity.

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
    groupby=CFG["celltype_col"],
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.matrixplot(
    adata,
    var_names=list(adata.var_names),
    groupby=CFG["celltype_col"],
    standard_scale="var",
    figsize=(18, 6),
    dendrogram=True,
)

# %%
sc.pl.correlation_matrix(adata, groupby=CFG["celltype_col"], figsize=(10, 8))

# %%
_corr = pd.DataFrame(adata.X, columns=adata.var_names).corr()
_fig, _ax = plt.subplots(figsize=(14, 12), dpi=120, facecolor="white")
_im = _ax.imshow(_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
_ax.set_xticks(range(len(_corr.columns)))
_ax.set_xticklabels(_corr.columns, rotation=90, fontsize=FS["sm"])
_ax.set_yticks(range(len(_corr.index)))
_ax.set_yticklabels(_corr.index, fontsize=FS["sm"])
_fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
_ax.set_title("Wu Charville — marker x marker correlation (exprs_norm, all cells, no aggregation)", fontsize=FS["md"])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spatial Graph Construction — Delaunay vs KNN
#
# Build two graphs using the biologically-motivated defaults:
# - **Delaunay** — parameter-free physical adjacency (~6 neighbors, adaptive)
# - **KNN k=10** — fixed-size window
#
# `library_key=CFG["sample_col"]` (`region_id`) is essential: it builds the graph *within*
# each CODEX region, so cells from different regions (which share overlapping coordinate
# ranges) are never wrongly connected.

# %%
# Drop cells with no cluster label so squidpy's nhood_enrichment gets a clean categorical.
adata = adata[adata.obs[CFG["celltype_col"]].notna()].copy()
adata.obs[CFG["celltype_col"]] = adata.obs[CFG["celltype_col"]].astype("category").cat.remove_unused_categories()

# KNN needs > KNN_K cells per library; drop regions too small to graph (protects Delaunay too).
KNN_K = 10
_sizes = adata.obs[CFG["sample_col"]].value_counts()
_small = _sizes.index[_sizes <= KNN_K]
if len(_small):
    print(f"Dropping {len(_small)} region(s) with <= {KNN_K} cells (too small to graph)")
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
# Edges over ONE region, cells colored by cell type — Delaunay's adaptive mesh vs KNN's
# fixed-degree connectivity.

# %%
_one = adata[adata.obs[CFG["sample_col"]] == _one_region].copy()
fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=150, facecolor="white")
sq.pl.spatial_scatter(_one, shape=None, color=CFG["celltype_col"],
                      connectivity_key="delaunay_pruned_connectivities",
                      edges_width=0.3, edges_color="#888888",
                      size=40, ax=axes[0], title="Delaunay (pruned)")
sq.pl.spatial_scatter(_one, shape=None, color=CFG["celltype_col"],
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
sq.gr.nhood_enrichment(adata, cluster_key=CFG["celltype_col"],
                       connectivity_key="delaunay_pruned", seed=0)
sq.pl.nhood_enrichment(adata, cluster_key=CFG["celltype_col"], figsize=(9, 9))

# %% [markdown]
# ## Cell-Type Graph Panels — representative regions
#
# Static companion view: the pruned Delaunay graph for one representative region per
# `alive_or_deceased` group (median-sized), cells colored by cell type. The cell-type
# legend renders as its own figure. We deliberately use a handful of representative regions
# (one per clinical group) rather than all 292 to stay tractable.

# %%
_reps, _titles = representative_samples(adata, cfg=CFG, by=CFG["label_col"], method="median")
graph_celltype_panels(adata, _reps, cfg=CFG, titles=_titles)

# %%
