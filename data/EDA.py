import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
import ipywidgets as widgets
from IPython.display import display

# ── Font size scale ───────────────────────────────────────────────────────────
FS = dict(xs=10, sm=12, md=14, lg=16, xl=20)

# ── Reference color maps (paper figures) ─────────────────────────────────────

# Keren 2018 Fig 2 — all_group_name cell types
KEREN_CELLTYPE_CMAP = {
    "Tregs":            "#c232c8",
    "CD4 T":            "#e87cdc",
    "CD8 T":            "#d46496",
    "CD3 T":            "#cc2800",
    "NK":               "#1a1a1a",
    "B":                "#d4dc3c",
    "Neutrophils":      "#78dc00",
    "Macrophages":      "#3cb450",
    "DC":               "#141e14",
    "DC/Mono":          "#00e6e6",
    "Mono/Neu":         "#5a96c8",
    "Other immune":     "#0a0a6e",
    "Tumor":            "#8c6914",
    "Keratin+ tumor":   "#c8a030",
    "Endothelial":      "#fa8072",
    "Mesenchymal like": "#d2b48c",
}

# Patwa 2021 — cell_type, mapped from Keren reference colors where equivalent
# Naming convention differs (e.g. "Treg" vs "Tregs", "CD4_T" vs "CD4 T")
PATWA_CELLTYPE_CMAP = {
    "Treg":            "#c232c8",
    "CD4_T":           "#e87cdc",
    "CD8_T":           "#d46496",
    "CD3_T":           "#cc2800",
    "NK":              "#1a1a1a",
    "B":               "#d4dc3c",
    "Neutrophil":      "#78dc00",
    "Macrophage":      "#3cb450",
    "DC":              "#141e14",
    "DC_Mono":         "#00e6e6",
    "Mono_neutrophil": "#5a96c8",
    "Other":           "#0a0a6e",
    "Tumor":           "#8c6914",
    "Endothelial":     "#fa8072",
    "Mesenchyme":      "#d2b48c",
    "Unknown":         "#aaaaaa",
}

# Schurch 2020 Fig 4 — neighborhood_name (numbered 1–9 in paper)
SCHURCH_NEIGHBORHOOD_CMAP = {
    "T cell enriched":            "#5b9bd5",
    "Bulk tumor":                 "#70ad47",
    "Immune-infiltrated stroma":  "#ed7d31",
    "Macrophage enriched":        "#ffc000",
    "Follicle":                   "#7030a0",
    "Tumor boundary":             "#c00000",
    "Vascularized smooth muscle": "#808080",
    "Smooth muscle":              "#c9a227",
    "Granulocyte enriched":       "#b4d7f0",
}

# ── Protein panel normalization ───────────────────────────────────────────────

PANEL_ALIASES = {
    # PD markers
    "PD1":          "PD-1",
    "PD.L1":        "PD-L1",
    # FOXP3
    "FoxP3":        "FOXP3",
    # HLA
    "HLA.DR":       "HLA-DR",
    "HLA_Class_1":  "HLA-I",
    # IDO
    "IDO-1":        "IDO",
    # LAG-3
    "Lag3":         "LAG-3",
    # SMA
    "aSMA":         "SMA",
    # Vimentin
    "VIM":          "Vimentin",
    # Beta-catenin
    "Beta.catenin": "Beta-catenin",
    "beta-catenin": "Beta-catenin",
    "Beta catenin": "Beta-catenin",
    "CTNNB":        "Beta-catenin",
    # Pan-Keratin
    "Pan.Keratin":  "Pan-Keratin",
    "PanCK":        "Pan-Keratin",
    "Cytokeratin":  "Pan-Keratin",
    # Phospho-S6
    "phospho.S6":   "phospho-S6",
    "p_S6":         "phospho-S6",
    # CD3
    "CD3e":         "CD3",
    # DNA / nuclear stains
    "dsDNA":        "DNA",
    "DRAQ5":        "DNA",
    "HOECHST1":     "DNA",
    "DNA1":         "DNA",
    "DNA2":         "DNA",
    # Kinase/signaling
    "p_H3":         "pH3",
    "p_mTOR":       "phospho-mTOR",
}

# Pure technical/elemental channels with no biological marker interpretation
TECHNICAL_MARKERS = {
    # Patwa MIBI elemental channels
    "Au", "Ta", "Ca", "Fe", "Na", "P", "Si", "Background",
    # Jackson IMC bead normalization channels
    "Ru96", "Ru98", "Ru99", "Ru100", "Ru101", "Ru102", "Ru104",
}


def normalize_panel(var_names, aliases=None, exclude=None):
    """Map raw marker names to canonical names, dropping technical channels."""
    if aliases is None:
        aliases = PANEL_ALIASES
    if exclude is None:
        exclude = TECHNICAL_MARKERS
    return {aliases.get(n, n) for n in var_names if n not in exclude}


def shared_markers(adatas):
    """Canonical markers present (after normalize_panel) in every dataset."""
    sets = [normalize_panel(a.var_names) for a in adatas.values()]
    return sorted(set.intersection(*sets))


# ── Marker functional categories + canonical plotting order ────────────────────
# Fine ~8-group functional taxonomy over the canonical marker union (post-PANEL_ALIASES).
# Used ONLY for a consistent plotting *order* (dict insertion order = global order) so the
# same markers land in roughly the same position across datasets with differing panels.
# A few markers are genuinely multi-role; each is assigned to its dominant use here:
#   GATA3 → tumor/epithelial (luminal breast) though also a Th2 T-cell TF
#   Beta-catenin → functional (Wnt signaling) though also junctional
#   CD44 → functional (activation/stemness) though broadly expressed
MARKER_CATEGORY_ORDER = {
    "Immune lineage": [
        "CD45", "CD45RA", "CD45RO", "CD3", "CD2", "CD5", "CD7", "CD4", "CD8", "CD25",
        "FOXP3", "T-bet", "CD194", "CD30", "CD20", "CD21", "CD38", "CD138",
        "CD68", "CD163", "CD11b", "CD11c", "CD16", "CD209", "CD15", "MPO",
        "CD56", "CD57", "HLA-DR", "HLA-I",
    ],
    "Tumor / epithelial": [
        "Pan-Keratin", "KRT5", "KRT7", "KRT14", "KRT19", "KRT8_18", "Keratin6", "Keratin17",
        "EpCAM", "CDH1", "MUC-1", "EGFR", "c_Myc", "CA9", "SOX9", "CDX2", "Na-K-ATPase",
        "GATA3", "Chromogranin A", "Synaptophysin",
    ],
    "Hormone receptor": ["ERa", "PGR", "HER2"],
    "Stromal / mesenchymal": [
        "Vimentin", "SMA", "FN1", "Collagen IV", "TWIST1", "SNAI2", "MMP9", "MMP12",
        "Podoplanin", "GFAP",
    ],
    "Vascular / endothelial": ["CD31", "CD34", "vWF"],
    "Immune checkpoint": ["PD-1", "PD-L1", "LAG-3", "IDO", "VISTA", "ICOS"],
    "Functional state": [
        "Ki67", "cPARP_cCASP3", "p53", "BCL-2", "Beta-catenin",
        "phospho-S6", "phospho-mTOR", "pH3", "Granzyme B", "CD44", "CD71", "CD63",
    ],
    "Chromatin / nuclear": ["DNA", "H3", "H3K27me3", "H3K9ac"],
}

# canonical marker -> category, and canonical marker -> global order index
MARKER_CATEGORIES = {m: cat for cat, ms in MARKER_CATEGORY_ORDER.items() for m in ms}
_MARKER_ORDER = {m: i for i, m in enumerate(MARKER_CATEGORIES)}


def _loose(s):
    """Punctuation/case-insensitive key for tolerant marker matching."""
    return s.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")


_MARKER_ORDER_LOOSE = {_loose(m): i for m, i in _MARKER_ORDER.items()}
_MARKER_CAT_LOOSE   = {_loose(m): cat for m, cat in MARKER_CATEGORIES.items()}

# one sequential colormap per functional category (light = low, dark = high expression)
MARKER_CATEGORY_CMAP = {
    "Immune lineage":         "Blues",
    "Tumor / epithelial":     "Oranges",
    "Hormone receptor":       "RdPu",
    "Stromal / mesenchymal":  "Greens",
    "Vascular / endothelial": "Reds",
    "Immune checkpoint":      "Purples",
    "Functional state":       "YlOrBr",
    "Chromatin / nuclear":    "Greys",
}


def marker_category(name, aliases=None):
    """Return the functional category for a marker name (alias/punctuation tolerant), or None."""
    aliases = aliases or PANEL_ALIASES
    canon = aliases.get(name, name)
    if canon in MARKER_CATEGORIES:
        return MARKER_CATEGORIES[canon]
    return _MARKER_CAT_LOOSE.get(_loose(canon))


def order_markers(var_names, aliases=None):
    """Sort marker names by the canonical functional order (`MARKER_CATEGORY_ORDER`).

    Each name is mapped through `PANEL_ALIASES` and a punctuation/case-insensitive fallback,
    so naming variants across datasets (dsDNA→DNA, Pan.Keratin/PanCK→Pan-Keratin, p_S6→
    phospho-S6, CTNNB→Beta-catenin, …) align to the same position. Unknown markers sort to
    the end alphabetically. Returns a new list of the *input* names, reordered.
    """
    aliases = aliases or PANEL_ALIASES
    big = len(_MARKER_ORDER)

    def idx(n):
        canon = aliases.get(n, n)
        if canon in _MARKER_ORDER:
            return _MARKER_ORDER[canon]
        return _MARKER_ORDER_LOOSE.get(_loose(canon), big)

    return sorted(var_names, key=lambda n: (idx(n), n.lower()))


# ── Cell type normalization ───────────────────────────────────────────────────

CELLTYPE_ALIASES = {
    # CD4 T
    "CD4 T":                   "CD4+ T",
    "CD4+ T cells":            "CD4+ T",
    "CD4+ T cells CD45RO+":    "CD4+ T",
    "CD4+ T cells GATA3+":     "CD4+ T",
    "CD4_T":                   "CD4+ T",
    # CD8 T
    "CD8 T":                   "CD8+ T",
    "CD8+ T cells":            "CD8+ T",
    "CD8_T":                   "CD8+ T",
    # CD3 T (pan-T, unresolved)
    "CD3 T":                   "CD3+ T",
    "CD3+ T cells":            "CD3+ T",
    "CD3_T":                   "CD3+ T",
    # Treg
    "Tregs":                   "Treg",
    # NK
    "NK cells":                "NK",
    # B
    "B cells":                 "B",
    # Plasma
    "plasma cells":            "Plasma cell",
    # Macrophage (all subtypes collapsed)
    "Macrophages":             "Macrophage",
    "CD68+ macrophages":       "Macrophage",
    "CD163+ macrophages":      "Macrophage",
    "CD68+CD163+ macrophages": "Macrophage",
    "CD68+ macrophages GzmB+": "Macrophage",
    "CD11b+CD68+ macrophages": "Macrophage",
    # DC
    "CD11c+ DCs":              "DC",
    # DC/Mono
    "DC/Mono":                 "DC/Mono",
    "DC_Mono":                 "DC/Mono",
    # Monocyte / Neutrophil
    "CD11b+ monocytes":        "Monocyte",
    "Mono/Neu":                "Mono/Neu",
    "Mono_neutrophil":         "Mono/Neu",
    "Neutrophils":             "Neutrophil",
    "granulocytes":            "Neutrophil",
    # Tumor
    "Keratin+ tumor":          "Tumor",
    "tumor cells":             "Tumor",
    # Endothelial
    "vasculature":             "Endothelial",
    "lymphatics":              "Endothelial",
    # Mesenchymal
    "Mesenchymal like":        "Mesenchymal",
    "stroma":                  "Mesenchymal",
    "Mesenchyme":              "Mesenchymal",
    # Smooth muscle
    "smooth muscle":           "Smooth muscle",
    # Other
    "adipocytes":              "Adipocyte",
    "nerves":                  "Nerve",
    "Other immune":            "Other immune",
    "immune cells":            "Other immune",
    "Other":                   "Other",
}

EXCLUDE_CELLTYPES = {
    "dirt", "undefined", "Unknown",
    "tumor cells / immune cells",
    "immune cells / vasculature",
}


def normalize_celltypes(series, aliases=None, exclude=None):
    """Return set of canonical cell type names present in an obs column."""
    if aliases is None:
        aliases = CELLTYPE_ALIASES
    if exclude is None:
        exclude = EXCLUDE_CELLTYPES
    return {aliases.get(v, v) for v in series.astype(str).unique() if v not in exclude}


def panel_heatmap(panels, figsize=None, title="Protein Panel Coverage"):
    """Binary dataset × feature presence heatmap, columns sorted by prevalence."""
    ds_names    = list(panels.keys())
    all_markers = sorted(set().union(*panels.values()))
    counts      = {m: sum(1 for p in panels.values() if m in p) for m in all_markers}
    all_markers = sorted(all_markers, key=lambda m: (-counts[m], m))

    # Transposed: rows = datasets, cols = markers
    matrix = np.array(
        [[1 if m in panels[ds] else 0 for m in all_markers] for ds in ds_names],
        dtype=float,
    )
    n_m, n_ds = len(all_markers), len(ds_names)
    fig, ax = plt.subplots(
        figsize=figsize or (n_m * 0.3 + 1.5, n_ds * 1.8 + 2.5),
        facecolor="white",
    )
    ax.set_facecolor("white")

    cmap = mcolors.ListedColormap(["#ebebeb", "#1f4e79"])
    ax.pcolormesh(matrix, cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=0.8)

    # Dashed tier separators (now vertical)
    seen_counts = set()
    for j, m in enumerate(all_markers):
        c = counts[m]
        if c in seen_counts:
            continue
        seen_counts.add(c)
        if j > 0:
            ax.axvline(j, color="#aaaaaa", linewidth=0.8, linestyle="--")

    ax.set_yticks(np.arange(n_ds) + 0.5)
    ax.set_yticklabels(ds_names, fontsize=FS["xl"], fontweight="bold")

    ax.set_xticks(np.arange(n_m) + 0.5)
    ax.set_xticklabels(all_markers, fontsize=FS["lg"], rotation=90, fontweight="bold")
    ax.xaxis.set_tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Secondary y-axis for per-dataset marker counts
    ax2 = ax.twinx()
    ax2.set_ylim(n_ds, 0)
    ax2.set_yticks(np.arange(n_ds) + 0.5)
    ax2.set_yticklabels(
        [f"(n={len(panels[ds])})" for ds in ds_names],
        fontsize=FS["lg"], color="#555555",
    )
    ax2.tick_params(left=False, right=False)

    ax.set_xlim(0, n_m)
    ax.set_ylim(n_ds, 0)
    ax.tick_params(left=False, bottom=False)
    ax.set_title(title, fontsize=FS["xl"], pad=20)

    plt.tight_layout()
    plt.show()


# ── Per-dataset configs ───────────────────────────────────────────────────────

KEREN_CFG = dict(
    publication="Keren et al. 2018",
    technology="MIBI",
    disease="TNBC",
    sample_col="SampleID",
    label_col="subtype",
    patient_col=None,
    celltype_col="all_group_name",
    um_per_px=0.39,          # MIBI resolution (paper: 100px = 39µm)
    arcsinh_cofactor=None,   # X already arcsinh-transformed (Nolan lab pipeline)
    cat_cols=[
        "SampleID", "tumorYN", "tumorCluster", "Group", "immuneCluster",
        "immuneGroup", "group_name", "immuneGroup_name", "all_group_name",
        "all_group_name2", "leiden", "scNiche", "subtype",
    ],
    pinned_cmaps={
        "all_group_name": KEREN_CELLTYPE_CMAP,
    },
)

SCHURCH_CFG = dict(
    publication="Schürch et al. 2020",
    technology="CODEX",
    disease="CRC",
    sample_col="Region",
    label_col="group_name",
    patient_col="patients",
    celltype_col="cell_type",
    um_per_px=0.377,         # CODEX lateral resolution (377.44 nm/px)
    size_col="size",         # cell area (px) — enables step-1 size normalization
    arcsinh_cofactor=0.5,    # applied to size-normalized (per-pixel) values, not raw counts
    cat_cols=[
        "Region", "patients", "tma_ab", "tma_12", "groups", "group_name",
        "cell_type", "neighborhood_name", "neighborhood_id", "neighborhood10",
    ],
    pinned_cmaps={
        "neighborhood_name": SCHURCH_NEIGHBORHOOD_CMAP,
    },
)

JACKSON_CFG = dict(
    publication="Jackson & Fischer et al. 2020",
    technology="IMC",
    disease="Breast Cancer",
    sample_col="image_name",
    label_col="tumor_clinical_type",
    patient_col="patient_id",
    celltype_col=None,
    um_per_px=1.0,           # IMC resolution (1µm/px, pixel:micron 1:1)
    arcsinh_cofactor=None,   # exprs layer already arcsinh-transformed
    expr_layer="exprs",
    cat_cols=[
        "image_name", "patient_id", "patient_cohort",
        "tumor_clinical_type", "tumor_ER_status", "tumor_PR_status", "tumor_HER2_status",
        "tumor_grade", "cell_metacluster", "tumor_response",
    ],
    pinned_cmaps={},
)

JACKSON_ZURICH_CFG = {**JACKSON_CFG, "publication": "Jackson & Fischer et al. 2020 — Zurich"}
JACKSON_BASEL_CFG  = {**JACKSON_CFG, "publication": "Jackson & Fischer et al. 2020 — Basel"}

PATWA_CFG = dict(
    publication="Patwa et al. 2021",
    technology="MIBI",
    disease="TNBC",
    sample_col="patient_id",
    label_col="Architecture",
    patient_col=None,
    celltype_col="cell_type",
    um_per_px=0.39,          # MIBI (Keren subset) — same acquisition resolution
    arcsinh_cofactor=5,      # raw counts
    cat_cols=[
        "patient_id", "cell_type", "Architecture",
        "Recurrence", "Survival",
        "functional_proteins_cluster", "immunoregulatory_protein_cluster",
        "coexpression_cluster",
    ],
    pinned_cmaps={
        "cell_type": PATWA_CELLTYPE_CMAP,
        "Architecture": {
            "Mixed":            "#e07b39",
            "Compartmentalized":"#4a90d9",
            "Cold":             "#a0a0a0",
        },
    },
)


# ── Metadata ──────────────────────────────────────────────────────────────────

def summarize_metadata(adata, cfg=KEREN_CFG):
    sample_col = cfg["sample_col"]
    cat_cols   = cfg["cat_cols"]

    print(f"Cells:   {adata.n_obs:,}")
    print(f"Markers: {adata.n_vars}")
    print(f"Samples: {adata.obs[sample_col].nunique()}")
    print(f"\nObs columns ({len(adata.obs.columns)}):")
    print("  " + ", ".join(adata.obs.columns.tolist()))
    print(f"\nObsm keys: {list(adata.obsm.keys())}")

    print(f"\nCategorical summaries:")
    for col in cat_cols:
        if col not in adata.obs.columns:
            continue
        n = adata.obs[col].nunique()
        vals = sorted(_to_str_series(adata.obs[col]).unique())
        preview = vals[:6]
        suffix = "..." if n > 6 else ""
        print(f"  {col:<24} {n:>4} unique  {preview}{suffix}")

    print(f"\nVar names: {list(adata.var_names)}")


def spatial_info(adata):
    x, y = adata.obs["x"], adata.obs["y"]
    print(f"x: [{x.min():.1f}, {x.max():.1f}]  range = {x.max() - x.min():.1f}")
    print(f"y: [{y.min():.1f}, {y.max():.1f}]  range = {y.max() - y.min():.1f}")
    coords = adata.obs[["x", "y"]].sample(min(3000, len(adata.obs)), random_state=0).values
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    print(f"Estimated resolution (median NN spacing): {np.median(dists[:, 1]):.2f} units")


def cat_breakdown(adata, cfg=KEREN_CFG, cols=None):
    cols = cols or [c for c in cfg["cat_cols"] if c in adata.obs.columns]
    for col in cols:
        print(f"\n{'─'*44}")
        print(f"{col}  ({adata.obs[col].nunique()} unique)")
        print(adata.obs[col].value_counts().to_string())


def dataset_stats(adata, cfg, note=""):
    """Extract a flat dict of summary statistics for the dataset overview table.

    n_markers is the biological panel size after dropping technical channels
    and harmonizing aliases (same count as shown in the panel heatmap).
    """
    sample_col  = cfg["sample_col"]
    patient_col = cfg.get("patient_col")
    label_col   = cfg["label_col"]

    n_patients = (
        adata.obs[patient_col].nunique() if patient_col
        else adata.obs[sample_col].nunique()
    )
    vals = sorted(
        v for v in _to_str_series(adata.obs[label_col]).unique()
        if v not in ("NA", "nan")
    )
    return dict(
        publication=cfg.get("publication", ""),
        technology=cfg.get("technology", ""),
        disease=cfg.get("disease", ""),
        condition=" / ".join(vals),
        n_patients=n_patients,
        n_samples=adata.obs[sample_col].nunique(),
        n_cells=adata.n_obs,
        n_markers=len(normalize_panel(adata.var_names)),
        note=note,
    )


def overview_table(stats_list, unique_exclude=None):
    """Render a styled HTML comparison table from a list of dataset_stats dicts.

    unique_exclude : list of Publication names to drop from a second "Unique Total" row.
    Use when datasets overlap (e.g. Keren ⊂ Patwa — same patients/images) so the naive
    "Total" double-counts; the "Unique Total" sums the distinct cohort only.
    """
    df = pd.DataFrame(stats_list)
    df = df.rename(columns={
        "publication": "Publication",
        "technology":  "Technology",
        "disease":     "Disease",
        "condition":   "Condition",
        "n_patients":  "Patients",
        "n_samples":   "Samples",
        "n_cells":     "Cells",
        "n_markers":   "Markers",
        "note":        "Note",
    })

    def _totals_row(label, sub):
        return {
            "Publication": label, "Technology": "", "Disease": "", "Condition": "",
            "Patients": sub["Patients"].sum(),
            "Samples":  sub["Samples"].sum(),
            "Cells":    sub["Cells"].sum(),
            "Markers": "", "Note": "",
        }

    total_rows = [_totals_row("Total", df)]
    if unique_exclude:
        kept = df[~df["Publication"].isin(unique_exclude)]
        total_rows.append(_totals_row("Unique Total", kept))
    totals = pd.DataFrame(total_rows)

    df["Cells"] = df["Cells"].map("{:,}".format)
    totals["Cells"] = totals["Cells"].map("{:,}".format)
    df = pd.concat([df, totals], ignore_index=True)

    cols = [c for c in df.columns if c != "Note"]
    if "Note" in df.columns and not df["Note"].eq("").all():
        cols.append("Note")
    display(df[cols].style.hide(axis="index").set_properties(**{"text-align": "left"}))


# ── Color helpers ─────────────────────────────────────────────────────────────

def register_cmap(adata, col, cmap):
    """Write pinned colors into adata.uns so scanpy/squidpy pick them up automatically.
    Must be called once after loading; no palette= arg needed in plot calls after that."""
    cats = adata.obs[col].astype("category").cat.categories
    adata.uns[f"{col}_colors"] = [cmap.get(c, "#aaaaaa") for c in cats]

_FALLBACK_COLOR = (0.7, 0.7, 0.7, 1.0)  # grey for unknown/NA values


def _to_str_series(series):
    """Convert any series (incl. category dtype) to strings with NaN → 'NA'.
    Must cast to object first to allow fillna on Categorical series."""
    return series.astype(object).fillna("NA").astype(str)


def _to_rgba(color):
    """Normalize any color spec (hex str, rgb tuple, rgba tuple) to (r,g,b,a) floats."""
    return mcolors.to_rgba(color)


def _build_color_map(series):
    """Build a str→RGBA dict from all unique values in series.
    NA is always grey. Uses tab20 for ≤20 categories, hsv for more."""
    s = _to_str_series(series)
    vals = sorted(v for v in s.unique() if v != "NA")
    n = len(vals)
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i / 20) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / n) for i in range(n)]
    color_map = {v: colors[i] for i, v in enumerate(vals)}
    color_map["NA"] = _FALLBACK_COLOR
    return color_map


def _resolve_color_map(color_by, adata, cfg):
    """Return the color map for color_by: pinned if available, else auto-built."""
    pinned = cfg.get("pinned_cmaps", {})
    if color_by in pinned:
        # Pinned cmap from paper — augment with fallback for any unknown categories
        # so new values render as grey rather than crashing
        return pinned[color_by]
    cat_cols = cfg["cat_cols"]
    if color_by in cat_cols and color_by in adata.obs.columns:
        return _build_color_map(adata.obs[color_by])
    return None


def _cat_scatter(ax, x, y, series, color_map, s=1):
    """Scatter cells colored by a categorical series using the provided color_map.
    Unknown values (new categories, NA) fall back to grey."""
    s_str = _to_str_series(series)
    rgba = np.array([_to_rgba(color_map.get(v, _FALLBACK_COLOR)) for v in s_str],
                    dtype=np.float32)
    ax.scatter(x, y, c=rgba, s=s, linewidths=0, rasterized=True)


def _cont_scatter(ax, x, y, values, cmap="Blues", s=1):
    sc = ax.scatter(x, y, c=values, s=s, cmap=cmap, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)


def _make_legend_handles(color_map):
    """Build legend handles from a color_map dict. Handles any color spec."""
    return [
        mlines.Line2D([], [], marker="o", color="black",
                      markerfacecolor=_to_rgba(v),
                      markeredgewidth=0, label=str(k), markersize=10)
        for k, v in sorted(color_map.items())
    ]


def _style_ax(ax, title):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=FS["md"], pad=8)


# ── Marker distributions ──────────────────────────────────────────────────────

def plot_marker_distributions(adata, layer=None, n_cols=6, figsize=None, dpi=100):
    X = adata.layers[layer] if layer else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    markers = order_markers(list(adata.var_names))  # canonical functional order
    n = adata.n_vars
    n_rows = int(np.ceil(n / n_cols))
    figsize = figsize or (n_cols * 3.5, n_rows * 3.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, facecolor="white")
    axes = np.array(axes).flatten()
    for i, marker in enumerate(markers):
        vals = X[:, adata.var_names.get_loc(marker)]
        vals = vals[~np.isnan(vals)]  # drop NaN (some markers unmeasured in a cohort)
        axes[i].set_title(marker, fontsize=FS["lg"], pad=6)
        axes[i].tick_params(labelsize=FS["sm"])
        if vals.size == 0:
            axes[i].text(0.5, 0.5, "all NaN", transform=axes[i].transAxes,
                         fontsize=FS["md"], va="center", ha="center", color="grey")
            continue
        axes[i].hist(vals, bins=60, color="steelblue", alpha=0.8, linewidth=0)
        axes[i].axvline(0, color="crimson", lw=0.9, ls="--")
        stats_txt = (f"min  {vals.min():7.3f}\n"
                     f"max  {vals.max():7.3f}\n"
                     f"med  {np.median(vals):7.3f}\n"
                     f"std  {vals.std():7.3f}")
        axes[i].text(0.97, 0.97, stats_txt, transform=axes[i].transAxes,
                     fontsize=FS["sm"], va="top", ha="right", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.85, lw=0))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    src = f"layer: {layer}" if layer else ".X"
    plt.suptitle(f"{adata.uns.get('dataset', 'Dataset')} — marker distributions  ({src},  red = 0)",
                 fontsize=FS["xl"], y=1.01)
    plt.tight_layout()
    plt.show()


# ── Spatial graph pruning + evaluation ────────────────────────────────────────

def prune_and_eval_graph(adata, cfg=KEREN_CFG, factor=2.5,
                         conn_key="delaunay_connectivities",
                         dist_key="delaunay_distances",
                         pruned_key="delaunay_pruned_connectivities",
                         knn_conn_key="knn_connectivities",
                         knn_dist_key="knn_distances",
                         plot=True):
    """Prune long Delaunay edges and evaluate the graph in one call.

    Estimates the cell-to-cell pitch as the median per-cell shortest edge, prunes
    edges longer than `factor` × pitch (default 2.5×), and writes the pruned graph to
    `obsp[pruned_key]`. With plot=True, auto-creates a degree / edge-length grid: columns
    are Delaunay pre-prune, Delaunay post-prune, and (if the KNN graph is present) KNN —
    for continuity with the adaptive-vs-fixed comparison. Build graphs first via
    `sq.gr.spatial_neighbors(..., delaunay=True, key_added="delaunay")` and
    `sq.gr.spatial_neighbors(..., n_neighs=k, key_added="knn")`.

    Returns the prune threshold in pixels.
    """
    import scipy.sparse as sp

    um    = cfg.get("um_per_px", 1.0)
    label = adata.uns.get("dataset", cfg.get("publication", "Dataset"))

    conn = adata.obsp[conn_key]
    dist = adata.obsp[dist_key].tocsr()

    # cell pitch = median of each cell's shortest edge
    min_edge = np.array([
        dist.data[dist.indptr[i]:dist.indptr[i + 1]].min()
        if dist.indptr[i + 1] > dist.indptr[i] else np.nan
        for i in range(dist.shape[0])
    ])
    nn_px     = np.nanmedian(min_edge)
    thresh_px = factor * nn_px

    # pruned connectivity (keep edges within threshold)
    keep = adata.obsp[dist_key].copy()
    keep.data = (keep.data <= thresh_px).astype(np.float64)
    adata.obsp[pruned_key] = sp.csr_matrix(conn.multiply(keep))

    # distributions
    deg_pre  = np.asarray((conn > 0).sum(axis=1)).ravel()
    deg_post = np.asarray((adata.obsp[pruned_key] > 0).sum(axis=1)).ravel()
    ed_all   = adata.obsp[dist_key].data
    ed_pre   = ed_all * um
    ed_post  = ed_all[ed_all <= thresh_px] * um
    frac     = (ed_all > thresh_px).mean()

    # optional KNN graph for continuity
    has_knn = knn_conn_key in adata.obsp and knn_dist_key in adata.obsp
    if has_knn:
        deg_knn = np.asarray((adata.obsp[knn_conn_key] > 0).sum(axis=1)).ravel()
        ed_knn  = adata.obsp[knn_dist_key].data * um

    print(f"{label}: pitch {nn_px:.1f}px ({nn_px * um:.1f}µm)  "
          f"prune >{thresh_px:.1f}px ({thresh_px * um:.1f}µm)  "
          f"removed {frac * 100:.2f}% edges  "
          f"Delaunay degree {deg_pre.mean():.2f} → {deg_post.mean():.2f}"
          + (f"  |  KNN degree {deg_knn.mean():.2f}" if has_knn else ""))

    if plot:
        # columns: Delaunay pre, Delaunay post, [KNN]
        deg_cols = [(deg_pre,  "Delaunay — pre-prune",  "steelblue"),
                    (deg_post, "Delaunay — post-prune", "seagreen")]
        ed_cols  = [(ed_pre,  "Delaunay — pre-prune",  "steelblue"),
                    (ed_post, "Delaunay — post-prune", "seagreen")]
        if has_knn:
            deg_cols.append((deg_knn, "KNN (fixed k)", "indianred"))
            ed_cols.append((ed_knn,  "KNN (fixed k)", "indianred"))

        ncols = len(deg_cols)
        fig, axes = plt.subplots(2, ncols, figsize=(8 * ncols, 10), dpi=120, facecolor="white")
        xmax = np.percentile(np.concatenate([ed_pre] + ([ed_knn] if has_knn else [])), 99.5)

        for j, (deg, ttl, color) in enumerate(deg_cols):
            ax = axes[0, j]
            ax.hist(deg, bins=range(0, int(max(deg.max(), 1)) + 2), color=color, alpha=0.85, align="left")
            ax.axvline(deg.mean(), color="black", ls="--", lw=1.2)
            ax.set_title(f"Degree — {ttl}  (mean {deg.mean():.1f})", fontsize=FS["md"])
            ax.set_xlabel("neighbors per cell", fontsize=FS["sm"])
            ax.set_ylabel("cells", fontsize=FS["sm"])
            ax.tick_params(labelsize=FS["xs"])

        for j, (ed, ttl, color) in enumerate(ed_cols):
            ax = axes[1, j]
            ax.hist(ed, bins=80, range=(0, xmax), color=color, alpha=0.85, linewidth=0)
            ax.axvline(thresh_px * um, color="crimson", ls="--", lw=1.2,
                       label=f"threshold {thresh_px * um:.1f}µm")
            ax.set_title(f"Edge length — {ttl}", fontsize=FS["md"])
            ax.set_xlabel("edge length (µm)", fontsize=FS["sm"])
            ax.set_ylabel("edges", fontsize=FS["sm"])
            ax.legend(fontsize=FS["sm"])
            ax.tick_params(labelsize=FS["xs"])

        plt.suptitle(f"{label} — graph evaluation (Delaunay pruning {factor}× median pitch"
                     + (", vs KNN" if has_knn else "") + ")", fontsize=FS["xl"], y=1.01)
        plt.tight_layout()
        plt.show()

    return thresh_px


def representative_samples(adata, cfg=KEREN_CFG, by=None):
    """Pick one representative sample per category: the sample with the most cells in it.
    Returns (sample_ids, titles) where titles maps sample_id -> 'category: sample_id'."""
    sample_col = cfg["sample_col"]
    by = by or cfg["label_col"]
    reps, titles = [], {}
    for cat, grp in adata.obs.groupby(by, observed=True):
        if str(cat).lower() in {"nan", "na", "none", ""}:
            continue
        sid = grp[sample_col].value_counts().idxmax()
        reps.append(sid)
        titles[sid] = f"{cat}: {sid}"
    return reps, titles


def marker_cycle_gif(adata, sample_ids, cfg=KEREN_CFG, layer="exprs_norm", markers=None,
                     conn_key="delaunay_pruned_connectivities", titles=None,
                     cmap="magma", s=14, fps=2, dpi=110, out_path="marker_cycle.gif"):
    """Animated GIF cycling identically through markers across side-by-side sample panels.

    Markers cycle in canonical functional order; each frame's colormap is set by the marker's
    functional group (`MARKER_CATEGORY_CMAP`) so related markers share a palette, and the title
    reads ``{dataset} — {marker} ({category})``. Each panel = one sample's cells at spatial
    coords with the (pruned) graph drawn once as a static grey backdrop. With
    `layer="exprs_norm"` (min-max 0-1) the scale is fixed 0-1 across all markers/panels;
    otherwise a shared per-marker clim across panels is used each frame. `cmap` is the fallback
    for markers with no category. Returns out_path.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.animation import FuncAnimation, PillowWriter

    sample_col = cfg["sample_col"]
    markers    = markers or order_markers(list(adata.var_names))  # canonical functional order
    label      = adata.uns.get("dataset", cfg.get("publication", "Dataset"))
    fixed01    = layer == "exprs_norm"
    n          = len(sample_ids)

    def _vals(sub, marker):
        M = sub.layers[layer] if layer else sub.X
        col = M[:, sub.var_names.get_loc(marker)]
        return col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()

    def _cmap_cat(marker):
        cat = marker_category(marker)
        return MARKER_CATEGORY_CMAP.get(cat, cmap), cat

    def _frame_title(marker):
        cat = marker_category(marker)
        return f"{label} — {marker}" + (f" ({cat})" if cat else "")

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.0), dpi=dpi, facecolor="white",
                             squeeze=False)
    axes = axes.ravel()
    subs, scatters = [], []
    for ax, sid in zip(axes, sample_ids):
        sub = adata[adata.obs[sample_col] == sid]
        xy  = sub.obsm["spatial"] if "spatial" in sub.obsm else sub.obs[["x", "y"]].values
        subs.append(sub)
        ax.set_aspect("equal"); ax.axis("off"); ax.margins(0)
        if conn_key and conn_key in sub.obsp:
            A = sub.obsp[conn_key].tocoo()
            m = A.row < A.col  # undirected: one segment per edge
            segs = np.stack([xy[A.row[m]], xy[A.col[m]]], axis=1)
            ax.add_collection(LineCollection(segs, colors="#cccccc", linewidths=0.2, zorder=1))
        sc0 = ax.scatter(xy[:, 0], xy[:, 1], c=_vals(sub, markers[0]), cmap=_cmap_cat(markers[0])[0],
                         s=s, vmin=0 if fixed01 else None, vmax=1 if fixed01 else None,
                         linewidths=0, zorder=2)
        scatters.append(sc0)
        ax.set_title(titles.get(sid, str(sid)) if titles else str(sid), fontsize=FS["md"], pad=3)

    # tight margins; leave a sliver on the right for the colorbar and a strip at the
    # bottom for the title placed directly under the images
    fig.subplots_adjust(left=0.01, right=0.93, top=0.95, bottom=0.10, wspace=0.04)

    # colorbar sized to the last panel's height (matches the slice view)
    _pos = axes[-1].get_position()
    cax  = fig.add_axes([_pos.x1 + 0.012, _pos.y0, 0.012, _pos.height])
    cbar = fig.colorbar(scatters[0], cax=cax)
    cax.tick_params(labelsize=FS["xs"])

    # title directly beneath the paired images (centered over the panels, not the colorbar)
    _cx = 0.5 * (axes[0].get_position().x0 + _pos.x1)
    title = fig.text(_cx, _pos.y0 - 0.04, _frame_title(markers[0]),
                     ha="center", va="top", fontsize=FS["xl"])

    def update(i):
        marker = markers[i]
        cm = _cmap_cat(marker)[0]
        per = [_vals(sub, marker) for sub in subs]
        if not fixed01:
            lo = min(v.min() for v in per); hi = max(v.max() for v in per)
        for sc0, vals in zip(scatters, per):
            sc0.set_array(vals)
            sc0.set_cmap(cm)
            if not fixed01:
                sc0.set_clim(lo, hi)
        cbar.update_normal(scatters[0])  # refresh colorbar to the current category cmap
        title.set_text(_frame_title(marker))
        return scatters

    anim = FuncAnimation(fig, update, frames=len(markers), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out_path}  ({len(markers)} frames, {n} panels)")
    return out_path


# ── Single sample ─────────────────────────────────────────────────────────────

def plot_sample(adata, sample_id, color_by=None, ax=None, s=6, dpi=150,
                color_map=None, show_legend=True, cfg=KEREN_CFG):
    sample_col  = cfg["sample_col"]
    label_col   = cfg["label_col"]
    patient_col = cfg.get("patient_col")
    cat_cols    = cfg["cat_cols"]
    color_by    = color_by or label_col

    all_ids = sorted(adata.obs[sample_col].unique())
    idx   = all_ids.index(sample_id) + 1
    total = len(all_ids)

    mask = adata.obs[sample_col] == sample_id
    sub  = adata[mask]
    x, y = sub.obs["x"].values, sub.obs["y"].values

    patient = ""
    if patient_col and patient_col in sub.obs.columns:
        patient = f"  subj {sub.obs[patient_col].iloc[0]}"

    label = ""
    if label_col in sub.obs.columns:
        mode_vals = _to_str_series(sub.obs[label_col])
        if len(mode_vals):
            label = f"  grp {mode_vals.mode()[0]}"

    title = f"{sample_id}  ({idx}/{total}){patient}{label}"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi, facecolor="white")

    _style_ax(ax, title)

    if color_by in cat_cols and color_by in sub.obs.columns:
        if color_map is None:
            color_map = _resolve_color_map(color_by, adata, cfg)
        _cat_scatter(ax, x, y, sub.obs[color_by], color_map=color_map, s=s)
        if show_legend and standalone:
            handles = _make_legend_handles(color_map)
            ax.legend(handles=handles, fontsize=FS["sm"], frameon=True,
                      loc="upper right", ncol=max(1, len(handles) // 12))
    elif color_by in sub.var_names:
        expr_layer = cfg.get("expr_layer")
        if expr_layer and expr_layer in sub.layers:
            col = sub.layers[expr_layer][:, sub.var_names.get_loc(color_by)]
        else:
            col = sub[:, color_by].X
        expr = col.toarray().flatten() if hasattr(col, 'toarray') else np.asarray(col).flatten()
        _cont_scatter(ax, x, y, expr, s=s)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Subsample viewer ──────────────────────────────────────────────────────────

def plot_samples(adata, sample_ids=None, color_by=None, n_cols=4, s=4, dpi=150,
                 cfg=KEREN_CFG):
    sample_col = cfg["sample_col"]
    color_by   = color_by or cfg["label_col"]

    all_ids = sorted(adata.obs[sample_col].unique())
    if sample_ids is None:
        sample_ids = all_ids[:8]
    n      = len(sample_ids)
    n_rows = int(np.ceil(n / n_cols))

    color_map = _resolve_color_map(color_by, adata, cfg)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5),
                             dpi=dpi, facecolor="white")
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()

    for i, sid in enumerate(sample_ids):
        plot_sample(adata, sid, color_by=color_by, ax=axes[i], s=s,
                    color_map=color_map, show_legend=False, cfg=cfg)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Strata: {color_by}  —  {n} of {len(all_ids)} slices", fontsize=FS["lg"], y=1.01)

    if color_map is not None:
        handles = _make_legend_handles(color_map)
        fig.legend(handles=handles, fontsize=FS["sm"], loc="lower center",
                   ncol=min(len(handles), 6),
                   bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout()
    plt.show()


# ── Full dataset viewer ───────────────────────────────────────────────────────

def plot_all_samples(adata, color_by=None, n_cols=6, s=2, dpi=120, cfg=KEREN_CFG):
    sample_col = cfg["sample_col"]
    color_by   = color_by or cfg["label_col"]

    all_ids = sorted(adata.obs[sample_col].unique())
    n_rows  = int(np.ceil(len(all_ids) / n_cols))

    color_map = _resolve_color_map(color_by, adata, cfg)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4),
                             dpi=dpi, facecolor="white")
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()

    for i, sid in enumerate(all_ids):
        plot_sample(adata, sid, color_by=color_by, ax=axes[i], s=s,
                    color_map=color_map, show_legend=False, cfg=cfg)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Strata: {color_by}", fontsize=FS["lg"], y=1.01)

    if color_map is not None:
        handles = _make_legend_handles(color_map)
        fig.legend(handles=handles, fontsize=FS["sm"], loc="lower center",
                   ncol=min(len(handles), 8),
                   bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout()
    plt.show()
