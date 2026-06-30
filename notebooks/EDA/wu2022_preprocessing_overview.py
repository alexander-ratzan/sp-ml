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
# # Wu 2022 — Cross-Cohort Preprocessing Overview
#
# Preprocessing overview for the **three Wu et al. 2022 CODEX cohorts** (UPMC, Charville,
# DFCI), mirroring the cross-platform `preprocessing_overview.py` but scoped to the Wu
# datasets only. These three cohorts share a recipe that starts from publisher-z-scored
# markers and applies the **Risom final two steps** — an upper-99.9th-percentile
# winsorize (per-marker) followed by a per-marker 0–1 min-max — to build the modeling
# layer. Only the upstream steps (size-norm, arcsinh) are skipped, since the publisher
# already variance-stabilized each marker.
#
# **What we show, per cohort:**
#
# 1. **`X` is publisher z-scored** — centered near 0, with a large negative fraction
#    (~62–69%). NOT raw counts. Strongly **right-skewed** (per-marker skew ~+3) with
#    extreme positive artifacts (max ~41 sd vs the 99.9th pct ~7.5).
# 2. **`layers["exprs_norm"]` is the modeling layer** — built as **upper-99.9 winsorize
#    (per-marker) → min-max 0–1**. The single downstream-facing layer.
# 3. **`X` vs `exprs_norm`** side-by-side via `plot_marker_distributions` + a `_layer_stats`
#    helper (mirroring `schurch2020_eda.py`), for distribution shape and global range.
# 4. **Winsorization rescue** (§7a) — a NAIVE min-max on raw `X` (no cap) collapses
#    `exprs_norm` to ~[0, 0.02]; the stored upper-99.9-winsorized layer rescues it so
#    ~99% of values span ~0.1–0.9.
#
# **Layer scheme** (same two-slot shape as `crc.h5ad`, but `X` here is z-scores not raw):
#
# | Layer | Contents |
# |---|---|
# | `X` | publisher **z-scores** on the cohort's native panel (kept verbatim) |
# | `layers["exprs_norm"]` | **upper-99.9 winsorize (per-marker) → min-max 0–1 within the cohort** — **final modeling layer** |
#
# > **Panels are NOT reconciled.** Each cohort keeps its own native panel (UPMC 22,
# > Charville 40, DFCI 41 markers). There is no shared-marker grid here — that is a
# > deferred reconciliation step (per the integration contract).
#
# > **Tractability.** UPMC is ~2.1M cells; it is read `backed="r"` and subsampled for the
# > numeric/distribution views. Charville and DFCI are loaded fully. No per-sample spatial
# > viz in this notebook — it is summary-level only.

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sp_ml.data.EDA import summarize_metadata, plot_marker_distributions

# %% [markdown]
# ## 0. Cohort configs (inline)
#
# The Wu cohorts are not in `EDA.py`'s CFG registry, so we define a small inline `cfg`
# per cohort mirroring `SCHURCH_CFG`'s keys. Per the contract: `sample_col="region_id"`,
# `patient_col="patient_id"`, `celltype_col="cell_type"`, and each cohort's clinical
# label columns go in `cat_cols`. `arcsinh_cofactor=None` and `size_col=None` encode that
# the upstream steps are skipped (publisher already z-scored); the winsorize + 0–1 min-max
# that build `exprs_norm` are baked into the `.h5ad` (see `uns["preprocessing"]`).

# %%
_BASE = dict(
    publication="Wu et al. 2022",
    technology="CODEX",
    sample_col="region_id",
    patient_col="patient_id",
    celltype_col="cell_type",
    label_col="cell_type",
    um_per_px=0.377,          # CODEX lateral resolution (same as Schurch CODEX)
    size_col=None,            # publisher z-scored — no size normalization
    arcsinh_cofactor=None,    # publisher z-scored — no arcsinh
    pinned_cmaps={},
)

WU_CFGS = {
    "upmc": {**_BASE, "disease": "HNSCC (UPMC)", "cat_cols": [
        "region_id", "patient_id", "cell_type", "cell_type_raw",
        "status", "primary_outcome", "recurred", "hpvstatus",
        "survival_status",
    ]},
    "charville": {**_BASE, "disease": "HNSCC (Charville)", "cat_cols": [
        "region_id", "patient_id", "cell_type", "cell_type_raw",
        "primary_outcome", "recurrence", "alive_or_deceased",
        "type_of_first_recurrence", "grade_differentiation",
    ]},
    "dfci": {**_BASE, "disease": "HNSCC (DFCI)", "cat_cols": [
        "region_id", "patient_id", "cell_type", "cell_type_raw",
        "pTR_label", "pTR_category", "pTR_PRIMARY", "pTR_max",
        "CANCER_SITE", "cAJCC_Stage",
    ]},
}

WU_PATHS = {
    "upmc":      "../../../data/wu2022/upmc.h5ad",
    "charville": "../../../data/wu2022/charville.h5ad",
    "dfci":      "../../../data/wu2022/dfci.h5ad",
}

# UPMC is huge (~2.1M cells) → backed read + subsample for the summary views.
BACKED = {"upmc"}
SUBSAMPLE_N = 200_000   # cells used for distribution/stats views (full-cohort = exact)
SEED = 0

# %% [markdown]
# ## 1. Load cohorts
#
# Charville and DFCI fully in-memory; UPMC `backed="r"`. Each `.h5ad` already carries
# `uns["dataset"]` (`"Wu2022 — <cohort>"`) so the diagnostic plots self-label.

# %%
adatas = {}
for cohort, path in WU_PATHS.items():
    backed = "r" if cohort in BACKED else None
    print(f"Loading {cohort:10s} (backed={backed}) ...")
    adatas[cohort] = ad.read_h5ad(path, backed=backed)

for cohort, a in adatas.items():
    print(f"  {cohort:10s} {a.n_obs:>9,} cells × {a.n_vars:>2} markers  "
          f"layers={list(a.layers.keys())}  backed={a.isbacked}")

# %% [markdown]
# ## 2. Per-cohort metadata snapshot

# %%
for cohort, a in adatas.items():
    print(f"\n{'='*70}\n{a.uns['dataset']}  —  {WU_CFGS[cohort]['disease']}\n{'='*70}")
    summarize_metadata(a, cfg=WU_CFGS[cohort])

# %% [markdown]
# ## 3. Recorded preprocessing recipe (`uns["preprocessing"]`)
#
# The recipe baked into each `.h5ad`. Note the same dict across all three cohorts:
# `size_norm=False`, `arcsinh_cofactor=None`, `winsorize_pct=99.9`, `norm="minmax_01"`,
# `final_layer="exprs_norm"`. The `exprs_source` is `X (publisher z-scored)` — the recipe
# starts from already-variance-stabilized values, caps the upper 99.9th pct per marker,
# then min-max'es to 0–1.

# %%
_recipe = pd.DataFrame(
    {cohort: dict(adatas[cohort].uns["preprocessing"]) for cohort in WU_PATHS}
).T
_recipe.index.name = "cohort"
display(_recipe)

# %% [markdown]
# ### 3a. Region → patient resolution caveat
#
# `uns["region_patient_resolution"]` is **RESOLVED only for Charville**. For UPMC and DFCI
# the publisher metadata was insufficient to map regions to patients, so the build sets
# `patient_id == region_id` as a fallback — **patient-level CV is blocked** for those two
# until external SPACE-GM metadata arrives. Surfaced here so any patient-grouped analysis
# downstream treats UPMC/DFCI `patient_id` as a per-region placeholder.

# %%
for cohort, a in adatas.items():
    res = a.uns.get("region_patient_resolution", "?")
    n_reg = a.obs["region_id"].nunique()
    n_pat = a.obs["patient_id"].nunique()
    flag = "" if res == "RESOLVED" else "  ← patient_id == region_id fallback"
    print(f"{cohort:10s} {res:11s}  regions={n_reg:>4}  patients={n_pat:>4}{flag}")

# %% [markdown]
# ## 4. Layer-stats helper
#
# Mirrors the `_layer_stats` pattern from `schurch2020_eda.py`, with two additions tuned
# to this recipe: the **negative fraction** (the z-score signature) and the **per-marker
# centering spread** (mean of per-marker means; should sit ~0 for `X`). Materializes a
# (subsampled, for UPMC) dense view so backed reads stay cheap.

# %%
def _materialize(adata, layer=None, n_sample=None, seed=SEED):
    """Return a dense (cells × markers) array for `X` or a layer.

    For backed/large cohorts, take a random `n_sample` cell subset first so we never
    pull the whole 2.1M-cell matrix into memory.
    """
    n = adata.n_obs
    if n_sample is not None and n_sample < n:
        idx = np.sort(np.random.default_rng(seed).choice(n, n_sample, replace=False))
        sub = adata[idx]
    else:
        sub = adata
    M = sub.layers[layer] if layer else sub.X
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    return np.asarray(M, dtype=float)


def _layer_stats(M, name):
    v = M[~np.isnan(M)]
    col_means = np.nanmean(M, axis=0)
    print(f"=== {name} ===")
    print(f"  shape:           {M.shape}")
    print(f"  range:           [{v.min():.4f}, {v.max():.4f}]")
    print(f"  mean / median:   {v.mean():.4f} / {np.median(v):.4f}")
    print(f"  std:             {v.std():.4f}")
    print(f"  pct negative:    {(v < 0).mean() * 100:.1f}%")
    print(f"  pct zero:        {(v == 0).mean() * 100:.1f}%")
    print(f"  per-marker mean: avg {col_means.mean():.4f}  "
          f"[{col_means.min():.4f}, {col_means.max():.4f}]")


# %% [markdown]
# ## 5. `X` vs `exprs_norm` — global stats, per cohort
#
# For each cohort: `X` (publisher z-scores) then `exprs_norm` (modeling layer). Expect for
# `X` — mean/median near 0, std near 1, large negative fraction; for `exprs_norm` — exact
# `[0, 1]` range, no negatives. UPMC stats are on a 200,000-cell subsample.

# %%
for cohort, a in adatas.items():
    n_sample = SUBSAMPLE_N if cohort in BACKED else None
    tag = f" (subsample {n_sample:,})" if n_sample else " (all cells)"
    print(f"\n{'#'*70}\n# {a.uns['dataset']}{tag}\n{'#'*70}")
    Xd = _materialize(a, layer=None,          n_sample=n_sample)
    Nd = _materialize(a, layer="exprs_norm",  n_sample=n_sample)
    _layer_stats(Xd, "X (publisher z-scored)")
    _layer_stats(Nd, "exprs_norm (per-marker min-max 0–1)")

# %% [markdown]
# ### 5a. Negative-fraction summary table
#
# The compact z-score evidence: the share of `X` values below 0 per cohort (the publisher
# z-score centering) next to `exprs_norm` (0% by construction).

# %%
_neg_rows = []
for cohort, a in adatas.items():
    n_sample = SUBSAMPLE_N if cohort in BACKED else None
    Xd = _materialize(a, layer=None,         n_sample=n_sample)
    Nd = _materialize(a, layer="exprs_norm", n_sample=n_sample)
    xv, nv = Xd[~np.isnan(Xd)], Nd[~np.isnan(Nd)]
    _neg_rows.append(dict(
        cohort=cohort, markers=a.n_vars,
        X_neg_pct=round((xv < 0).mean() * 100, 1),
        X_mean=round(float(xv.mean()), 4), X_std=round(float(xv.std()), 3),
        exprs_norm_neg_pct=round((nv < 0).mean() * 100, 1),
        exprs_norm_min=round(float(nv.min()), 3), exprs_norm_max=round(float(nv.max()), 3),
    ))
display(pd.DataFrame(_neg_rows).set_index("cohort"))

# %% [markdown]
# ## 6. `X` distributions (publisher z-scores) — per cohort
#
# Per-marker grids straight from `X`. Red dashed line at 0: histograms should straddle it
# with mass on both sides (centered, large left tail) — the z-score signature. For UPMC we
# pass a subsampled in-memory copy so the backed matrix isn't fully realized.

# %%
def _view_for_plot(adata, cohort):
    """In-memory AnnData (subsampled for backed cohorts) for the distribution plots,
    carrying X, exprs_norm, var_names and uns['dataset']."""
    if cohort in BACKED:
        idx = np.sort(np.random.default_rng(SEED).choice(adata.n_obs, SUBSAMPLE_N, replace=False))
        v = adata[idx].to_memory()
    else:
        v = adata
    v.uns["dataset"] = adata.uns["dataset"]
    return v

views = {cohort: _view_for_plot(a, cohort) for cohort, a in adatas.items()}

# %%
for cohort in WU_PATHS:
    plot_marker_distributions(views[cohort], layer=None)

# %% [markdown]
# ## 7. `exprs_norm` distributions (0–1 modeling layer) — per cohort
#
# Same grids on `exprs_norm`. Every marker now lives in `[0, 1]` (red line at 0 sits at the
# left edge). Because the upper 99.9th pct is winsorized per marker *before* the min-max,
# the bulk of each distribution is spread across the unit interval rather than crushed
# against 0 — the rescue quantified in §7a below.

# %%
for cohort in WU_PATHS:
    plot_marker_distributions(views[cohort], layer="exprs_norm", n_cols=6)

# %% [markdown]
# ## 7a. Winsorization rescue — naive min-max vs stored `exprs_norm`
#
# The publisher z-scores in `X` are **strongly right-skewed** (per-marker skew ~+3) with
# **one-sided extreme positive artifacts** — CODEX bright-pixel spikes push a marker's max
# to ~41 sd while its 99.9th percentile sits near ~7.5 sd. A plain per-marker min-max maps
# `[min, max] → [0, 1]`, so that single spike sets `max = 1` and **crushes ~99% of the
# values into `[0, 0.02]`** — the modeling layer collapses to a near-degenerate band.
#
# The fix (baked into the `.h5ad`) is the **Risom upper-99.9 winsorize**: cap each marker
# at its 99.9th percentile *before* the min-max. The spike is clipped to the cap, the cap
# becomes `1`, and the dense bulk of the distribution re-expands across the unit interval.
#
# **Why one-sided / upper (not a σ-symmetric ±kσ rule):**
#
# - The **outlier process is one-sided** — bright-pixel/antibody-aggregate spikes only
#   inflate the *upper* tail; there is no matching mechanism producing extreme negatives.
# - The **lower tail is a bounded background floor** (z-scored background, not an artifact),
#   so it carries real signal and should not be clipped.
# - The distribution is **skewed**, so a symmetric ±kσ window (or two-sided percentile)
#   would over-clip the informative low end and/or under-clip the artifact-laden high end.
#   A single upper-percentile cap targets exactly the artifact tail — matching the Risom
#   upper-99.9 convention used for the core-4 datasets.
#
# Below, per cohort: a **NAIVE min-max computed directly on raw `X`** (no winsorize) vs the
# stored, winsorized `layers["exprs_norm"]`. We quantify the collapse → rescue (% of values
# `< 0.02`, % within `(0.1, 0.9)`, median) and show before/after histograms.

# %%
def _naive_minmax(M):
    """Per-marker min-max of a raw matrix (NO winsorize) — the un-rescued baseline."""
    lo = np.nanmin(M, axis=0)
    hi = np.nanmax(M, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (M - lo) / rng


def _dist_stats(v):
    """Collapse/rescue summary for a flattened 0–1 vector."""
    return dict(
        pct_lt_002=round(float((v < 0.02).mean() * 100), 2),
        pct_in_0109=round(float(((v > 0.1) & (v < 0.9)).mean() * 100), 2),
        median=round(float(np.median(v)), 4),
        mean=round(float(v.mean()), 4),
    )


_rescue_rows = []
for cohort, a in adatas.items():
    n_sample = SUBSAMPLE_N if cohort in BACKED else None
    Xd = _materialize(a, layer=None,         n_sample=n_sample)   # raw z-scores
    Nd = _materialize(a, layer="exprs_norm", n_sample=n_sample)   # stored (winsorized) layer
    naive = _naive_minmax(Xd)

    nv = naive[~np.isnan(naive)]
    wv = Nd[~np.isnan(Nd)]
    naive_s = _dist_stats(nv)
    wins_s = _dist_stats(wv)
    # per-marker skew of X (mean across markers), for the right-skew evidence
    col_mean = np.nanmean(Xd, axis=0)
    col_std = np.nanstd(Xd, axis=0)
    col_skew = np.nanmean(((Xd - col_mean) / np.where(col_std > 0, col_std, 1.0)) ** 3, axis=0)

    _rescue_rows.append(dict(
        cohort=cohort,
        X_skew_mean=round(float(np.nanmean(col_skew)), 2),
        X_max=round(float(np.nanmax(Xd)), 1),
        naive_pct_lt_002=naive_s["pct_lt_002"],
        naive_pct_in_0109=naive_s["pct_in_0109"],
        naive_median=naive_s["median"],
        wins_pct_lt_002=wins_s["pct_lt_002"],
        wins_pct_in_0109=wins_s["pct_in_0109"],
        wins_median=wins_s["median"],
    ))
    print(f"\n{'#'*70}\n# {a.uns['dataset']} — winsorization rescue\n{'#'*70}")
    print(f"  X: per-marker skew ~{np.nanmean(col_skew):+.2f}, max {np.nanmax(Xd):.1f} sd")
    print(f"  NAIVE min-max (no winsorize): {naive_s}")
    print(f"  stored exprs_norm (upper-99.9): {wins_s}")

display(pd.DataFrame(_rescue_rows).set_index("cohort"))

# %% [markdown]
# ### 7a (cont). Before / after histograms
#
# Per cohort, the pooled all-marker distribution: **naive min-max on raw `X`** (collapsed
# against 0) vs the stored **upper-99.9-winsorized `exprs_norm`** (spread across 0–1).

# %%
fig, axes = plt.subplots(len(adatas), 2, figsize=(11, 3.2 * len(adatas)))
if len(adatas) == 1:
    axes = axes[None, :]
bins = np.linspace(0, 1, 60)
for row, (cohort, a) in enumerate(adatas.items()):
    n_sample = SUBSAMPLE_N if cohort in BACKED else None
    Xd = _materialize(a, layer=None,         n_sample=n_sample)
    Nd = _materialize(a, layer="exprs_norm", n_sample=n_sample)
    nv = _naive_minmax(Xd)
    nv = nv[~np.isnan(nv)]
    wv = Nd[~np.isnan(Nd)]

    ax_l, ax_r = axes[row]
    ax_l.hist(nv, bins=bins, color="#c0392b", alpha=0.85)
    ax_l.set_title(f"{a.uns['dataset']} — NAIVE min-max (no winsorize)")
    ax_l.set_xlabel("value"); ax_l.set_ylabel("count"); ax_l.set_yscale("log")
    ax_r.hist(wv, bins=bins, color="#2471a3", alpha=0.85)
    ax_r.set_title(f"{a.uns['dataset']} — stored exprs_norm (upper-99.9)")
    ax_r.set_xlabel("value"); ax_r.set_ylabel("count"); ax_r.set_yscale("log")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Recipe contrast — Wu cohorts vs core-4 Risom pipeline
#
# How the three Wu cohorts diverge from the core-4 cross-platform recipe (the one the
# Risom-style `preprocessing_overview.py` applies, with **Risom/Patwa MIBI** as the
# reference raw pipeline):
#
# | Step | Core-4 Risom recipe (e.g. Patwa MIBI) | Wu 2022 cohorts |
# |---|---|---|
# | **Starting `X`** | raw / per-pixel counts | **publisher z-scores** (already variance-stabilized) |
# | **1. Size normalization** | `X / size` (datasets with an area column) | **skipped** — `size_col=None` (z-scored upstream) |
# | **2. arcsinh variance-stabilization** | `arcsinh(X / cofactor)` (e.g. Patwa cofactor 5) | **skipped** — `arcsinh_cofactor=None` |
# | **3. Winsorization (upper 99.9th pct)** | cap antibody-aggregate outliers per marker | **same** — `winsorize_pct=99.9`, per-marker upper cap |
# | **4. 0–1 normalization** | per-marker min-max → `exprs_norm` | **same** — per-marker min-max → `exprs_norm` |
# | **Layers** | `X`(raw) → `size_norm` → `exprs`(arcsinh+winsor) → `exprs_norm` | `X`(z-score) → `exprs_norm` (winsor+minmax) |
#
# **What the Wu recipe shares vs skips.** The publisher already z-scored each marker, which
# does the job of steps 1–2 (cell-size effects and variance stabilization are absorbed into
# a centered, unit-variance distribution). Re-applying arcsinh or size-norm on z-scores
# would be meaningless (and arcsinh on negatives is ill-defined), so those two are skipped.
# But the z-scores remain right-skewed with one-sided positive artifacts, so the recipe
# **keeps the Risom final two steps** — upper-99.9 winsorize then per-marker 0–1 min-max —
# to build `exprs_norm` for cross-marker comparability in modeling. The final layer is 0–1
# (not z-score), exactly mirroring Risom's choice.
#
# **Consequences to keep in mind downstream:**
#
# - `exprs_norm` is min-max `[0, 1]`, **not** z-scored. Because the upper 99.9th pct is
#   winsorized per marker before the min-max, the bulk of each marker is spread across the
#   unit interval (not crushed toward 0 — see §7a). Markers with broader spread still carry
#   more weight in PCA; if equal per-marker weighting is wanted for an embedding, apply
#   `sc.pp.scale` on top of `exprs_norm` for that step only.
# - Panels are **per-cohort native and unreconciled** — no cross-Wu shared-marker layer
#   exists yet; treat each cohort independently until reconciliation lands.
# - `exprs_norm` min-max is **global per cohort**, not per region.

# %% [markdown]
# ## 9. Provenance summary

# %%
for cohort, a in adatas.items():
    print(f"{a.uns['dataset']:18s} "
          f"{a.n_obs:>9,} cells × {a.n_vars:>2} markers  "
          f"layers={list(a.layers.keys())}  "
          f"resolution={a.uns.get('region_patient_resolution')}")
    print(f"{'':18s} preprocessing={dict(a.uns['preprocessing'])}\n")
