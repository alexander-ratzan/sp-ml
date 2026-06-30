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
# # Wu 2022 Datasets Overview
#
# Cross-cohort attribute summary for the **three Wu et al. 2022 / SPACE-GM CODEX cohorts**
# (UPMC, Charville, DFCI) — the head-and-neck / cutaneous SCC collection used as this
# project's **pretraining & discovery tier**. The other four datasets (Keren, Schürch,
# Patwa, Jackson-Fischer) are covered in `datasets_overview.py` and are **not** included
# here. All numbers are derived live from the dataset objects.
#
# **Patient-resolution caveat (read first).** Each region's patient identity comes from
# the cohort's labels CSV, not from the `region_id`. Only **Charville** is resolved
# (`region → patient` derived from `sample_label_visualizer`). For **UPMC** (120/308
# regions carry a NaN label) and **DFCI** (no patient column exists at all) the mapping
# is **UNRESOLVED**, so `patient_id == region_id` as a fallback — the "Patients" column
# for those two cohorts therefore equals their region count and is *not* a true patient
# tally. `uns["region_patient_resolution"]` records this per cohort. Patient-level CV is
# blocked for UPMC/DFCI until external SPACE-GM region→patient metadata arrives.
#
# **Panels differ and are NOT reconciled.** Each cohort keeps its own native panel
# (UPMC 22 / Charville 40 / DFCI 41 markers). The panel heatmap below is **descriptive
# only** — it visualizes overlap of the native panels after canonical name normalization;
# it does not subset to or impose a shared panel.

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

from sp_ml.data.EDA import (
    dataset_stats, overview_table,
    normalize_panel, normalize_celltypes, panel_heatmap,
)

# %% [markdown]
# ## Inline cohort configs
#
# The Wu cohorts are not in `EDA.py`'s CFG registry, so we define small inline `cfg`
# dicts here mirroring `SCHURCH_CFG`'s keys. Shared core schema: `sample_col="region_id"`,
# `patient_col="patient_id"`, `celltype_col="cell_type"` (the coarse shared lineage).
# `label_col` and `cat_cols` use each cohort's own broadcast clinical columns.

# %%
_WU_COMMON = dict(
    technology="CODEX",
    sample_col="region_id",
    patient_col="patient_id",
    celltype_col="cell_type",
    um_per_px=0.377,         # CODEX lateral resolution (shared with Schürch CODEX)
    arcsinh_cofactor=None,   # X already publisher z-scored
    expr_layer="exprs_norm",
    pinned_cmaps={},
)

UPMC_CFG = {
    **_WU_COMMON,
    "publication": "Wu 2022 — UPMC",
    "disease": "HNSCC",
    "label_col": "primary_outcome",
    "cat_cols": [
        "region_id", "patient_id", "cell_type",
        "status", "primary_outcome", "recurred", "hpvstatus",
        "survival_status",
    ],
}

CHARVILLE_CFG = {
    **_WU_COMMON,
    "publication": "Wu 2022 — Charville",
    "disease": "cSCC",
    "label_col": "primary_outcome",
    "cat_cols": [
        "region_id", "patient_id", "cell_type",
        "primary_outcome", "recurrence", "alive_or_deceased",
        "type_of_first_recurrence", "grade_differentiation",
    ],
}

DFCI_CFG = {
    **_WU_COMMON,
    "publication": "Wu 2022 — DFCI",
    "disease": "HNSCC",
    "label_col": "pTR_category",
    "cat_cols": [
        "region_id", "patient_id", "cell_type",
        "pTR_label", "pTR_category", "pTR_PRIMARY", "CANCER_SITE", "cAJCC_Stage",
    ],
}

# %%
COHORTS = [
    ("Wu — UPMC",      "../../../data/wu2022/upmc.h5ad",      UPMC_CFG),
    ("Wu — Charville", "../../../data/wu2022/charville.h5ad", CHARVILLE_CFG),
    ("Wu — DFCI",      "../../../data/wu2022/dfci.h5ad",      DFCI_CFG),
]

stats, panels, celltypes, resolution = [], {}, {}, {}
for name, path, cfg in COHORTS:
    adata = ad.read_h5ad(path, backed="r")
    s = dataset_stats(adata, cfg)
    s["publication"] = name
    # surface the patient-resolution status as the overview note
    res = str(adata.uns.get("region_patient_resolution", "UNRESOLVED"))
    resolution[name] = res
    s["note"] = "patients resolved" if res == "RESOLVED" else "patient_id == region_id (UNRESOLVED)"
    stats.append(s)
    panels[name] = normalize_panel(adata.var_names)
    celltypes[name] = normalize_celltypes(adata.obs[cfg["celltype_col"]])
    adata.file.close()

# %% [markdown]
# ## Overview table
#
# Cells, native markers, regions (= `region_id`), patients, and the patient-resolution
# status per cohort. **Note the caveat:** UPMC/DFCI "Patients" equals their region count
# (`patient_id == region_id` fallback) and is not a true patient tally; only Charville's
# patient count is resolved.

# %%
overview_table(stats)

# %% [markdown]
# ## Protein panel coverage (native panels, descriptive)
#
# Binary cohort × marker presence matrix across the three native panels after canonical
# name normalization (`WU_PANEL_ALIASES`-equivalent via `normalize_panel`; e.g.
# `CD3e→CD3`, `PanCK→Pan-Keratin`, `aSMA→SMA`, `DAPI→DNA`). Technical channels excluded.
# Count on the right = number of cohorts sharing each marker. **The panels are not
# reconciled** — this only visualizes the overlap of what each cohort natively measured.

# %%
panel_heatmap(panels, title="Wu 2022 — Native Panel Coverage")

# %% [markdown]
# ## Cell-type coverage (coarse shared lineage)
#
# Coarse `obs["cell_type"]` lineage classes present per cohort. This is the Wu-internal
# shared level harmonized from the incompatible publisher schemes (UPMC tumor
# sub-phenotypes, Charville anonymous numbered tumor clusters, DFCI functional T-cell
# states); the verbatim labels live in `obs["cell_type_raw"]`. Count on the right =
# number of cohorts in which each coarse class appears.

# %%
panel_heatmap(celltypes, title="Wu 2022 — Coarse Cell-Type Coverage")
