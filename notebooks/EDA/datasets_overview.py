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
# # Datasets Overview
#
# Attribute summary for all spatial proteomics datasets used in this project.
# All numbers are derived live from the dataset objects.

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
import numpy as np

from data.EDA import (
    KEREN_CFG, SCHURCH_CFG, PATWA_CFG, JACKSON_CFG,
    dataset_stats, overview_table,
    normalize_panel, normalize_celltypes, panel_heatmap,
)

# %%
DATASETS = [
    ("Keren 2018",   "../../../data/keren2018/tnbc.h5ad",               KEREN_CFG),
    ("Schurch 2020", "../../../data/schurch2020/crc.h5ad",              SCHURCH_CFG),
    ("Patwa 2021",   "../../../data/rasp-mibi/tnbc_mibi.h5ad",          PATWA_CFG),
    ("Jackson 2020", "../../../data/jacksonfischer2020/full/full.h5ad", JACKSON_CFG),
]

stats, panels, celltypes = [], {}, {}
for name, path, cfg in DATASETS:
    adata = ad.read_h5ad(path, backed="r")
    stats.append(dataset_stats(adata, cfg))
    panels[name] = normalize_panel(adata.var_names)
    col = cfg.get("celltype_col")
    if col and col in adata.obs.columns:
        celltypes[name] = normalize_celltypes(adata.obs[col])
    adata.file.close()

# %%
overview_table(stats)

# %% [markdown]
# ## Protein Panel Coverage
#
# Binary presence matrix across all four datasets after canonical name normalization
# (`PANEL_ALIASES`). Technical/elemental channels excluded. Count on right = number
# of datasets sharing each marker. Rows sorted top → bottom by prevalence.

# %%
panel_heatmap(panels)

# %% [markdown]
# ## Cell Type Coverage
#
# Canonical cell type harmonization across the three datasets with curated annotations
# (`CELLTYPE_ALIASES`). Ambiguous/artifact labels excluded. Jackson-Fischer omitted
# (phenograph clusters, not named cell types).

# %%
panel_heatmap(celltypes, title="Cell Type Coverage")

# %% [markdown]
# ## Jackson-Fischer Effective Panel by Cohort
#
# The Jackson-Fischer h5ad objects share a common 45-marker schema, but the Basel and
# Zurich cohorts were acquired with different panels — markers not measured in a cohort
# are NaN-filled. The top "Protein Panel Coverage" heatmap reports the full Jackson
# schema (the **union** of both cohorts) and so hides this. Below, the *effective* panel
# is the set of markers actually measured (not all-NaN in `layers["exprs"]`) per cohort.
# Basel lacks EpCAM, CTNNB (β-catenin), and SOX9. Union = measured in either cohort
# (matches the full object); intersection = measured in both (the safe panel for any
# combined cross-cohort analysis).

# %%
def _effective_panel(path):
    a = ad.read_h5ad(path)
    L = a.layers["exprs"]
    L = (L.toarray() if hasattr(L, "toarray") else np.asarray(L)).astype("float32")
    measured = [v for v, allnan in zip(a.var_names, np.isnan(L).all(axis=0)) if not allnan]
    return normalize_panel(measured)

_basel  = _effective_panel("../../../data/jacksonfischer2020/basel/basel.h5ad")
_zurich = _effective_panel("../../../data/jacksonfischer2020/zurich/zurich.h5ad")
jackson_panels = {
    "Jackson Basel":              _basel,
    "Jackson Zurich":             _zurich,
    "Jackson Union (full)":       _basel | _zurich,
    "Jackson Intersection":       _basel & _zurich,
}
print(f"Union (full): {len(_basel | _zurich)} markers   "
      f"Intersection: {len(_basel & _zurich)} markers   "
      f"Zurich-only: {sorted(_zurich - _basel)}")
panel_heatmap(jackson_panels, title="Jackson-Fischer Effective Panel by Cohort")
