"""Parser for the Wu et al. 2022 / SPACE-GM head-&-neck + colon CODEX cohorts.

Three cohorts (UPMC, Charville, DFCI) ship as one zip each, holding per-region
quads ``raw_data/<region_id>.{expression,cell_data,cell_types,cell_features}.csv``
(plus a ``.json`` segmentation file we ignore). Each cohort keeps its OWN native
marker panel — UPMC 22, Charville 40, DFCI 41 — and is parsed into a separate
``.h5ad`` that mirrors the unified schema of the existing repo datasets.

Expression values are **publisher z-scored** (per-marker, centered near 0, ~65%
negative), NOT raw counts — so there is no size normalization and no arcsinh.
``X`` keeps the z-scores verbatim; ``layers["exprs_norm"]`` is the modeling input —
**upper 99.9th-pct winsorize (per-marker) → min-max 0–1** within the cohort. The
winsorize is essential: the z-scores are right-skewed (skew ~+3) with extreme
positive artifacts (max ~41 sd vs p99.9 ~7.5), so capping the upper tail stops a
single spike from crushing the 0–1 scale. Mirrors ``preprocessing.py``'s Risom
final steps for pre-transformed datasets (one-sided upper cap, per-marker).

Region→patient mapping comes from ``<cohort>_labels.csv`` (``sample_label_visualizer``),
not from the ``region_id`` itself. It resolves for Charville; it is UNRESOLVED for
UPMC and DFCI (see ``REGION_PATIENT_RESOLUTION``), where ``patient_id = region_id``.
"""

import io
import re
import zipfile

import numpy as np
import pandas as pd
import anndata as ad


# ── Cohorts ──────────────────────────────────────────────────────────────────
COHORTS = ("upmc", "charville", "dfci")

_DATA_DIR = "data/wu2022"   # relative to repo root; holds the zips + label CSVs

WINSORIZE_PCT = 99.9        # upper-tail cap (per-marker) before min-max; see build_wu_anndata


# ── Marker aliases → repo canonical names (PANEL_ALIASES convention) ──────────
# NOT applied to var_names (each cohort keeps its native panel; subsetting is a
# modeling-time concern). Stored in var["canonical"] for cross-dataset matching,
# exactly mirroring how EDA.normalize_panel() would resolve them downstream.
WU_PANEL_ALIASES = {
    "CD3e":      "CD3",
    "PanCK":     "Pan-Keratin",
    "aSMA":      "SMA",
    "DAPI":      "DNA",
    "FoxP3":     "FOXP3",
    "PD1":       "PD-1",
    "PDL1":      "PD-L1",
    "LAG3":      "LAG-3",
    "HLA-ABC":   "HLA-I",
    "CollagenIV":"Collagen IV",
    "GranzymeB": "Granzyme B",
}


# ── Cell-type harmonization ──────────────────────────────────────────────────
# obs["cell_type_raw"] keeps the verbatim publisher label. obs["cell_type"] is a
# coarse lineage shared across all three cohorts, since the native schemes are
# incompatible (UPMC has tumor sub-phenotypes, Charville has anonymous numbered
# tumor clusters, DFCI has functional T-cell states). Any raw label not listed
# falls back to "Other".
_CELLTYPE_COARSE = {
    # Tumor / epithelial
    "Tumor": "Tumor", "Tumor (Ki67+)": "Tumor", "Tumor (Podo+)": "Tumor",
    "Tumor (CD15+)": "Tumor", "Tumor (CD21+)": "Tumor", "Tumor (CD20+)": "Tumor",
    "Tumor (PanCK hi)": "Tumor", "Tumor (PanCK low)": "Tumor",
    "Tumor 1": "Tumor", "Tumor 2 (Ki67 Proliferating)": "Tumor", "Tumor 3": "Tumor",
    "Tumor 4": "Tumor", "Tumor 5": "Tumor", "Tumor 6 / DC": "Tumor", "Tumor 7": "Tumor",
    # T cells
    "CD4 T cell": "T cell", "CD8 T cell": "T cell",
    "CD4 T cell (ICOS+/FoxP3+)": "T cell",
    "T cell (CD45RO+/FoxP3+/ICOS+)": "T cell",
    "T cell (GranzymeB+/LAG3+)": "T cell",
    "Naive lymphocyte (CD45RA+/CD38+)": "T cell",
    "Unknown (TCF1+)": "T cell",
    # B cells
    "B cell": "B cell", "Naive B cell": "B cell",
    # NK
    "NK cell": "NK cell",
    # Myeloid / APC
    "Macrophage": "Myeloid", "APC": "Myeloid", "APC/macrophage": "Myeloid",
    "Dendritic cell": "Myeloid", "Granulocyte": "Myeloid", "Mast cell": "Myeloid",
    # Other immune (lineage unresolved in the publisher annotation)
    "Naive immune cell": "Other immune",
    # Stroma
    "Stromal / Fibroblast": "Stroma", "Stroma": "Stroma",
    # Vessel / endothelium
    "Vessel": "Vessel", "Vessel endothelium": "Vessel", "Blood vessel": "Vessel",
    "Lymph vessel": "Vessel",
    # Unassigned / unknown
    "Unassigned": "Unassigned", "Unclassified": "Unassigned",
    "Unknown": "Unassigned", "Other": "Other",
}


def harmonize_celltype(raw: str) -> str:
    """Map a verbatim publisher cell-type label to the coarse shared lineage."""
    return _CELLTYPE_COARSE.get(raw, "Other")


# ── Region → patient resolution ──────────────────────────────────────────────
# Patient identity is NOT encoded in region_id (the c-field is the CODEX batch).
# It comes from the labels CSV column `sample_label_visualizer`:
#   UPMC      : "<clinical_id> <block>_<spot>"  → patient = token before 1st space
#               but 120/308 regions have a NaN label → UNRESOLVED.
#   Charville : "<patient>_<date>"              → patient = token before 1st "_"
#               resolves to 162 (target 161; off-by-one, see contract note).
#   DFCI      : labels CSV has NO patient column at all → UNRESOLVED.
REGION_PATIENT_RESOLUTION = {
    "upmc":      "UNRESOLVED",   # sample_label_visualizer present for only 188/308
    "charville": "RESOLVED",     # patient = sample_label_visualizer.split('_')[0]
    "dfci":      "UNRESOLVED",   # no patient column in labels CSV
}


def _derive_patient_id(cohort: str, labels: pd.DataFrame) -> pd.Series:
    """Return a region_id → patient_id Series. Falls back to region_id when the
    mapping is UNRESOLVED so the .h5ad still builds (flag downstream for CV)."""
    rid = labels["region_id"].astype(str)
    if cohort == "charville":
        pat = labels["sample_label_visualizer"].astype(str).str.split("_").str[0]
        return pd.Series(pat.to_numpy(), index=rid.to_numpy())
    # UPMC / DFCI — unresolved: patient == region
    return pd.Series(rid.to_numpy(), index=rid.to_numpy())


# ── Region listing / reading from the cohort zip ─────────────────────────────
def _zip_path(cohort: str, data_dir: str) -> str:
    return f"{data_dir}/{cohort}_raw_data.zip"


def _list_regions(zf: zipfile.ZipFile) -> list[str]:
    """All region_ids in a cohort zip (one per expression.csv), sorted."""
    pat = re.compile(r"raw_data/(.+)\.expression\.csv$")
    regions = [m.group(1) for n in zf.namelist() if (m := pat.match(n))]
    return sorted(regions)


def _read_csv(zf: zipfile.ZipFile, region: str, kind: str) -> pd.DataFrame:
    with zf.open(f"raw_data/{region}.{kind}.csv") as fh:
        return pd.read_csv(io.BytesIO(fh.read()))


def _read_region(zf: zipfile.ZipFile, region: str):
    """Read one region's quad, inner-joined on CELL_ID. Returns (X, var_names, obs)."""
    expr = _read_csv(zf, region, "expression").set_index("CELL_ID")
    cd   = _read_csv(zf, region, "cell_data").set_index("CELL_ID")          # X, Y
    ct   = _read_csv(zf, region, "cell_types").set_index("CELL_ID")         # CELL_TYPE
    cf   = _read_csv(zf, region, "cell_features").set_index("CELL_ID")      # SIZE

    # Align all four on the expression CELL_ID order (cells present in expression).
    cells = expr.index
    cd, ct, cf = cd.reindex(cells), ct.reindex(cells), cf.reindex(cells)

    var_names = list(expr.columns)
    X = expr.to_numpy(dtype=np.float32)
    obs = pd.DataFrame({
        "region_id":     region,
        "cell_id":       cells.astype(str),
        "cell_type_raw": ct["CELL_TYPE"].astype(str).to_numpy(),
        "size":          cf["SIZE"].to_numpy(dtype=np.float32),
        "x":             cd["X"].to_numpy(dtype=np.float32),
        "y":             cd["Y"].to_numpy(dtype=np.float32),
    })
    return X, var_names, obs


# ── Builder ──────────────────────────────────────────────────────────────────
def build_wu_anndata(cohort: str, data_dir: str = _DATA_DIR,
                     regions: list[str] | None = None) -> ad.AnnData:
    """Build the unified-schema AnnData for one Wu 2022 cohort.

    Parameters
    ----------
    cohort : {"upmc", "charville", "dfci"}
    data_dir : directory holding ``<cohort>_raw_data.zip`` and ``<cohort>_labels.csv``.
    regions : optional explicit region_id subset (used for small-sample validation);
              None → all regions in the zip.

    Schema (mirrors data/schurch2020/crc.h5ad)
    ------------------------------------------
    X                    : float32, publisher z-scores on the cohort's native panel.
    layers["exprs_norm"] : float32, per-marker min-max 0–1 computed WITHIN the cohort.
    obsm["spatial"]      : (x, y) centroids from cell_data.
    obs                  : region_id (= sample_col), patient_id, cell_type,
                           cell_type_raw, size, x, y, + broadcast clinical labels.
    var                  : var_name (native) + canonical (PANEL_ALIASES-resolved).
    uns["preprocessing"] : provenance dict (Risom2026-style, arcsinh=None).
    uns["dataset"]       : "Wu2022 — <cohort>".
    """
    cohort = cohort.lower()
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}, got {cohort!r}")

    labels = pd.read_csv(f"{data_dir}/{cohort}_labels.csv")
    label_cols = [c for c in labels.columns if c != "region_id"]
    region_to_patient = _derive_patient_id(cohort, labels)
    labels_by_region = labels.set_index(labels["region_id"].astype(str))

    zf = zipfile.ZipFile(_zip_path(cohort, data_dir))
    region_ids = regions if regions is not None else _list_regions(zf)

    Xs, obs_parts, var_names = [], [], None
    for region in region_ids:
        X, vn, obs = _read_region(zf, region)
        if var_names is None:
            var_names = vn
        elif vn != var_names:
            raise ValueError(f"{cohort}: panel mismatch in region {region}")
        Xs.append(X)
        obs_parts.append(obs)
    zf.close()

    X = np.vstack(Xs)
    obs = pd.concat(obs_parts, ignore_index=True)

    # cell_type (coarse) + patient_id
    obs["cell_type"] = obs["cell_type_raw"].map(harmonize_celltype).astype("category")
    obs["patient_id"] = obs["region_id"].map(region_to_patient).astype(str)

    # Broadcast per-region clinical labels (one row per region in labels CSV).
    region_key = obs["region_id"].astype(str)
    for col in label_cols:
        obs[col] = labels_by_region[col].reindex(region_key).to_numpy()

    # Categoricals (mirror repo: sample/patient/celltype + clinical labels)
    for col in ["region_id", "patient_id", "cell_type", "cell_type_raw", *label_cols]:
        if col in obs.columns and obs[col].dtype == object:
            obs[col] = obs[col].astype("category")
    obs["region_id"] = obs["region_id"].astype("category")
    obs["patient_id"] = obs["patient_id"].astype("category")
    obs.index = obs.index.astype(str)

    # var: native name + canonical alias for cross-dataset matching
    var = pd.DataFrame(index=pd.Index(var_names, name=None))
    var["var_name"] = var_names
    var["canonical"] = [WU_PANEL_ALIASES.get(v, v) for v in var_names]

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)

    # exprs_norm = upper-99.9 winsorize (per-marker) → min-max 0–1 within this cohort
    # (modeling input). The publisher z-scores are right-skewed with extreme positive
    # artifacts (max ~41 sd); capping the upper tail per marker before min-max stops a
    # single spike from crushing the scale. Mirrors finalize_preprocessing (Risom final
    # steps): one-sided upper cap, then per-marker min-max. X (z-scores) is untouched.
    M = np.asarray(adata.X, dtype=float)
    caps = np.nanpercentile(M, WINSORIZE_PCT, axis=0)
    M = np.minimum(M, caps)
    lo = np.nanmin(M, axis=0)
    hi = np.nanmax(M, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    adata.layers["exprs_norm"] = ((M - lo) / rng).astype("float32")

    adata.uns["dataset"] = f"Wu2022 — {cohort}"
    adata.uns["preprocessing"] = {
        "exprs_source": "X (publisher z-scored)",
        "size_norm": False,
        "size_col": None,
        "arcsinh_cofactor": None,
        "winsorize_pct": WINSORIZE_PCT,
        "norm": "minmax_01",
        "final_layer": "exprs_norm",
        "pipeline": "Risom2026",
        "note": "publisher z-scored, arcsinh=None; exprs_norm = upper-99.9 winsorize (per-marker) → min-max 0–1 within dataset",
    }
    adata.uns["region_patient_resolution"] = REGION_PATIENT_RESOLUTION[cohort]
    return adata


def write_wu_h5ad(cohort: str, data_dir: str = _DATA_DIR,
                  out_path: str | None = None) -> str:
    """Build a full cohort AnnData and write it to ``data/wu2022/<cohort>.h5ad``."""
    adata = build_wu_anndata(cohort, data_dir=data_dir)
    out_path = out_path or f"{data_dir}/{cohort}.h5ad"
    adata.write_h5ad(out_path)
    return out_path
