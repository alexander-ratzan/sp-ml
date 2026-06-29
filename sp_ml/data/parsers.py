import re
import os
import numpy as np
import pandas as pd
import anndata as ad


# ── Schurch 2020 (CRC CODEX) ─────────────────────────────────────────────────

# Columns that hold spatial coordinates
_COORD_COLS = {
    "X:X": "x",
    "Y:Y": "y",
    "X_withinTile:X_withinTile": "x_within_tile",
    "Y_withinTile:Y_withinTile": "y_within_tile",
    "Z:Z": "z",
}

# Columns to clean up (col:col duplicated-name pattern)
_RENAME_OBS = {
    "size:size": "size",
    "tile_nr:tile_nr": "tile_nr",
    "cell_id:cell_id": "cell_id",
    "Profile_Homogeneity:Fiter1": "profile_homogeneity",
    "neighborhood number final": "neighborhood_id",
    "neighborhood name": "neighborhood_name",
    "File Name": "file_name",
    "TMA_AB": "tma_ab",
    "TMA_12": "tma_12",
    "Index in File": "index_in_file",
    "ClusterName": "cell_type",
    "ClusterSize": "cluster_size",
    "ClusterID": "cluster_id",
    "neighborhood10": "neighborhood10",
}


# Patient-level clinical/survival fields from crc_metadata.xlsx Sheet A → obs names.
# Survival drives the overall-survival secondary task; the rest are covariates.
# OS/DFS in months; *_Censor: 1 = event (OS death / DFS recurrence), 0 = censored
# (validated against group prognosis — CLR 4/17 deaths vs DII 13/18).
_SCHURCH_META_SHEET = "A. Patient_data_TMA_annotations"
_SCHURCH_META_COLS = {
    "OS": "OS", "OS_Censor": "OS_Censor",
    "DFS": "DFS", "DFS_Censor": "DFS_Censor",
    "Sex": "sex", "Age": "age",
    "pT": "pT", "pN": "pN", "p_TNM": "p_TNM", "G": "tumor_grade",
    "MSI_IHC": "MSI_IHC", "MSI_PCR": "MSI_PCR",
    "MLH1": "MLH1", "PMS2": "PMS2", "MSH6": "MSH6", "MSH2": "MSH2",
}
# Categorical fields; coerced via nullable string so Excel's mixed int/str levels
# (e.g. pT "3"/"1c") serialize cleanly. tumor_grade + MSI_PCR stay numeric.
_SCHURCH_META_CAT = ["sex", "pT", "pN", "p_TNM",
                     "MSI_IHC", "MLH1", "PMS2", "MSH6", "MSH2"]


def _is_protein_col(col: str) -> bool:
    """True if column is a protein/marker expression column (Name:Cyc_X_ch_Y pattern)."""
    return bool(re.search(r":Cyc_\d+_ch_\d+$", col)) or col == "HOECHST1:Cyc_1_ch_1"


def _parse_var_name(col: str) -> dict:
    """Extract short name, cycle, channel from a raw protein column name."""
    m = re.match(r"^(.+):Cyc_(\d+)_ch_(\d+)$", col)
    if m:
        full_name, cycle, channel = m.group(1), int(m.group(2)), int(m.group(3))
        # Short name = part before ' - ' if present, else full name
        short = full_name.split(" - ")[0].strip()
        return {"var_name": short, "full_name": full_name, "original_col": col,
                "cycle": cycle, "channel": channel}
    return {"var_name": col, "full_name": col, "original_col": col,
            "cycle": None, "channel": None}


def parse_schurch2020(csv_path: str, metadata_path: str = None) -> ad.AnnData:
    """
    Parse Schurch et al. 2020 CRC CODEX CSV into an AnnData object.

    Parameters
    ----------
    csv_path : str
        Path to CRC_clusters_neighborhoods_markers.csv
    metadata_path : str, optional
        Path to crc_metadata.xlsx (Supplementary Table S1). If omitted, looks for
        ``crc_metadata.xlsx`` alongside ``csv_path``. When found, per-patient
        survival + clinical fields (Sheet A) are broadcast cell→patient onto obs
        (join on ``patients``, verified 1:1). Skipped silently if absent.

    Returns
    -------
    AnnData
        X         : float32 expression matrix (cells × proteins)
        obs       : cell metadata (Region is the per-slice identifier; plus
                    patient-level OS/DFS survival + clinical covariates when
                    metadata is available — see ``_SCHURCH_META_COLS``)
        var       : protein metadata (var_name, full_name, cycle, channel)
        obsm      : 'spatial' = (x, y) coordinates
    """
    df = pd.read_csv(csv_path, index_col=0)

    # ── Separate protein vs metadata columns ─────────────────────────────────
    protein_cols = [c for c in df.columns if _is_protein_col(c)]
    coord_cols   = list(_COORD_COLS.keys())
    obs_exclude  = set(protein_cols + coord_cols)
    meta_cols    = [c for c in df.columns if c not in obs_exclude]

    # ── Build X ──────────────────────────────────────────────────────────────
    X = df[protein_cols].values.astype(np.float32)

    # ── Build var ────────────────────────────────────────────────────────────
    var_records = [_parse_var_name(c) for c in protein_cols]
    var = pd.DataFrame(var_records)
    var.index = var["var_name"]
    var.index.name = None

    # ── Build obs ────────────────────────────────────────────────────────────
    obs = df[meta_cols].copy()
    obs = obs.rename(columns=_RENAME_OBS)

    # Clean any remaining col:col duplicates not explicitly listed
    obs.columns = [c.split(":")[0] if c.count(":") == 1 and c.split(":")[0] == c.split(":")[1]
                   else c for c in obs.columns]

    # Categorical columns
    for col in ["Region", "tma_ab", "cell_type", "neighborhood_name", "patients", "groups"]:
        if col in obs.columns:
            obs[col] = obs[col].astype("category")

    # Group labels (Schurch 2020: 1 = CLR, 2 = DII)
    _GROUP_NAMES = {1: "CLR", 2: "DII"}
    obs["group_name"] = obs["groups"].astype(int).map(_GROUP_NAMES).astype("category")

    # ── Patient-level clinical + survival metadata (Supp. Table S1, Sheet A) ──
    if metadata_path is None:
        cand = os.path.join(os.path.dirname(csv_path), "crc_metadata.xlsx")
        metadata_path = cand if os.path.exists(cand) else None
    if metadata_path and os.path.exists(metadata_path):
        meta = pd.read_excel(metadata_path, sheet_name=_SCHURCH_META_SHEET)
        meta = meta[meta["Patient"].notna()].copy()
        meta["Patient"] = meta["Patient"].astype(int)
        meta = meta.set_index("Patient")
        pid = obs["patients"].astype(int).to_numpy()         # 1:1 join, verified
        for src, dst in _SCHURCH_META_COLS.items():
            obs[dst] = meta[src].reindex(pid).to_numpy()
        for col in _SCHURCH_META_CAT:
            obs[col] = obs[col].astype("string").astype("category")

    # Rename x/y into obs for EDA compat
    obs["x"] = df["X:X"].values.astype(np.float32)
    obs["y"] = df["Y:Y"].values.astype(np.float32)
    obs.index = obs.index.astype(str)

    # ── Build obsm ───────────────────────────────────────────────────────────
    spatial = df[["X:X", "Y:Y"]].values.astype(np.float32)

    # ── Assemble AnnData ─────────────────────────────────────────────────────
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial

    return adata


# ── Patwa 2021 (TNBC MIBI — RASP-MIBI) ──────────────────────────────────────

# Integer celltype code → name (from calculate_cell_prevalence.py, codes 2–16)
_PATWA_CELLTYPE_MAP = {
    2:  "Endothelial",
    3:  "Mesenchyme",
    4:  "Tumor",
    5:  "Treg",
    6:  "CD4_T",
    7:  "CD8_T",
    8:  "CD3_T",
    9:  "NK",
    10: "B",
    11: "Neutrophil",
    12: "Macrophage",
    13: "DC",
    14: "DC_Mono",
    15: "Mono_neutrophil",
    16: "Other",
}

_EXCLUDED_PATIENTS = {22, 30, 38}


def _read_patient_file(directory, patient_id):
    """Read a per-patient CSV, handling the t1.csv naming quirk for patient 1."""
    path = os.path.join(directory, f"{patient_id}.csv")
    if not os.path.exists(path) and patient_id == 1:
        path = os.path.join(directory, "t1.csv")
    return pd.read_csv(path)


def parse_patwa2021(data_dir: str) -> ad.AnnData:
    """
    Parse Patwa et al. 2021 TNBC MIBI dataset into an AnnData object.

    Parameters
    ----------
    data_dir : str
        Path to the rasp-mibi root directory (containing rawdata/ and intermediate_data/).

    Returns
    -------
    AnnData
        X         : float32 expression matrix (cells × 44 proteins)
        obs       : cell metadata — patient_id, cell_type, x, y, plus patient-level
                    clinical and covariate data broadcast to cell level
        var       : protein metadata — name, frame index, category (lineage/functional/etc.)
        obsm      : 'spatial' = (x, y) coordinates
        layers    : 'positivity' = binary protein positivity matrix
    """
    raw_dir          = os.path.join(data_dir, "rawdata")
    intermediate_dir = os.path.join(data_dir, "intermediate_data")
    expr_dir         = os.path.join(intermediate_dir, "protein_expression")
    pos_dir          = os.path.join(intermediate_dir, "protein_positivity")

    # ── Protein metadata (var) ────────────────────────────────────────────────
    proteins = pd.read_csv(os.path.join(raw_dir, "proteins_by_frame.csv"))
    proteins.columns = proteins.columns.str.strip("﻿")  # strip BOM
    proteins = proteins.rename(columns={"Frame": "frame", "Biomarker": "protein",
                                        "Purpose": "category"})
    proteins["category"] = proteins["category"].str.strip()
    var = proteins.set_index("protein")
    var.index.name = None

    frame_to_protein = {str(row["frame"]): name for name, row in var.iterrows()}
    n_proteins = len(var)

    # ── Patient-level metadata ────────────────────────────────────────────────
    clinical  = pd.read_csv(os.path.join(raw_dir, "clinical_data.csv"))
    clinical.columns = clinical.columns.str.strip("﻿")
    clinical  = clinical.set_index("ID")

    covariates = pd.read_csv(os.path.join(intermediate_dir, "covariate_rsf_data.csv"))
    covariates = covariates.set_index("ID")
    # Drop columns already in clinical to avoid duplication
    cov_extra  = covariates.drop(columns=["Recurrence", "Recurrence_time",
                                           "Survival", "Survival_time"], errors="ignore")

    patient_meta = clinical.join(cov_extra, how="left")

    # ── Per-patient cell data ─────────────────────────────────────────────────
    valid_ids = sorted(
        int(f.replace("t", "").replace(".csv", ""))
        for f in os.listdir(expr_dir)
        if f.endswith(".csv")
        and int(f.replace("t", "").replace(".csv", "")) not in _EXCLUDED_PATIENTS
    )

    all_expr = []
    all_pos  = []
    all_obs  = []

    for pid in valid_ids:
        expr_df = _read_patient_file(expr_dir, pid)
        pos_df  = _read_patient_file(pos_dir,  pid)

        n_cells = len(expr_df)

        # Expression matrix — columns 3 onward are frame-indexed proteins
        expr_vals = expr_df.iloc[:, 3:].values.astype(np.float32)
        pos_vals  = pos_df.iloc[:, 3:].values.astype(np.float32)

        # obs metadata for this patient
        obs_pat = pd.DataFrame({
            "patient_id": pid,
            "cell_type":  expr_df["Celltype"].map(_PATWA_CELLTYPE_MAP).fillna("Unknown"),
            "x":          expr_df["Column"].values.astype(np.float32),
            "y":          expr_df["Row"].values.astype(np.float32),
        })

        # Broadcast patient-level metadata to cell level
        if pid in patient_meta.index:
            for col, val in patient_meta.loc[pid].items():
                obs_pat[col] = val

        all_expr.append(expr_vals)
        all_pos.append(pos_vals)
        all_obs.append(obs_pat)

    # ── Concatenate ───────────────────────────────────────────────────────────
    X          = np.vstack(all_expr)
    positivity = np.vstack(all_pos)
    obs        = pd.concat(all_obs, ignore_index=True)

    # Categorical columns
    for col in ["patient_id", "cell_type", "Architecture"]:
        if col in obs.columns:
            obs[col] = obs[col].astype("category")

    obs.index = obs.index.astype(str)

    # ── Assemble AnnData ──────────────────────────────────────────────────────
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"]    = obs[["x", "y"]].values.astype(np.float32)
    adata.layers["positivity"] = positivity

    return adata
