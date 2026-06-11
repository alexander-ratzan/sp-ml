import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
import ipywidgets as widgets
from IPython.display import display

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

# ── Per-dataset configs ───────────────────────────────────────────────────────

KEREN_CFG = dict(
    sample_col="SampleID",
    label_col="subtype",
    patient_col=None,
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
    sample_col="Region",
    label_col="group_name",
    patient_col="patients",
    cat_cols=[
        "Region", "patients", "tma_ab", "tma_12", "groups", "group_name",
        "cell_type", "neighborhood_name", "neighborhood_id", "neighborhood10",
    ],
    pinned_cmaps={
        "neighborhood_name": SCHURCH_NEIGHBORHOOD_CMAP,
    },
)

PATWA_CFG = dict(
    sample_col="patient_id",
    label_col="Architecture",
    patient_col=None,
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

    print(f"\nProtein expression summary (across all cells):")
    print(f"  {'Protein':<24} {'Min':>10} {'Median':>10} {'Max':>10}")
    print(f"  {'─'*56}")
    for protein in adata.var_names:
        vals = np.asarray(adata[:, protein].X).flatten()
        print(f"  {protein:<24} {vals.min():>10.3f} {np.median(vals):>10.3f} {vals.max():>10.3f}")


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


# ── Color helpers ─────────────────────────────────────────────────────────────

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
    Uses tab20 for ≤20 categories, hsv for more to avoid color repetition."""
    s = _to_str_series(series)
    vals = sorted(s.unique())
    n = len(vals)
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i / 20) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / n) for i in range(n)]
    return {v: colors[i] for i, v in enumerate(vals)}


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
    ax.set_title(title, fontsize=13, pad=8)


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
            ax.legend(handles=handles, fontsize=11, frameon=True,
                      loc="upper right", ncol=max(1, len(handles) // 12))
    elif color_by in sub.var_names:
        expr = np.asarray(sub[:, color_by].X).flatten()
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

    fig.suptitle(f"{color_by}  —  {n} of {len(all_ids)} slices", fontsize=16, y=1.01)

    if color_map is not None:
        handles = _make_legend_handles(color_map)
        fig.legend(handles=handles, fontsize=12, loc="lower center",
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

    fig.suptitle(color_by, fontsize=16, y=1.01)

    if color_map is not None:
        handles = _make_legend_handles(color_map)
        fig.legend(handles=handles, fontsize=12, loc="lower center",
                   ncol=min(len(handles), 8),
                   bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout()
    plt.show()
