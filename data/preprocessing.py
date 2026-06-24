import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from IPython.display import display


# ── Diagnostics ───────────────────────────────────────────────────────────────

def expression_stats(adata, layer=None, n_sample=50_000, seed=0):
    """Sample cells and compute per-value expression statistics."""
    X = adata.layers[layer] if layer else adata.X
    idx = np.sort(
        np.random.default_rng(seed).choice(adata.n_obs, min(n_sample, adata.n_obs), replace=False)
    )
    X_sub = X[idx]
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()
    v = X_sub.astype(float).flatten()
    nan_mask = np.isnan(v)
    v_valid  = v[~nan_mask]
    return dict(
        nan_pct  = float(nan_mask.mean() * 100),
        min      = float(v_valid.min()),
        p1       = float(np.percentile(v_valid, 1)),
        median   = float(np.median(v_valid)),
        p99      = float(np.percentile(v_valid, 99)),
        max      = float(v_valid.max()),
        neg_pct  = float((v_valid < 0).mean() * 100),
        zero_pct = float((v_valid == 0).mean() * 100),
    )


def expression_stats_table(records):
    """Render a styled table from a list of expression_stats dicts."""
    df = pd.DataFrame(records)
    # move dataset/layer columns to front if present
    front = [c for c in ("dataset", "layer", "prior_preproc") if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    for c in ("min", "p1", "median", "p99", "max"):
        if c in df.columns:
            df[c] = df[c].map("{:.3f}".format)
    for c in ("nan_pct", "neg_pct", "zero_pct"):
        if c in df.columns:
            df[c] = df[c].map("{:.1f}%".format)
    df = df.rename(columns={"nan_pct": "NaN%", "neg_pct": "Neg%", "zero_pct": "Zero%"})
    display(df.style.hide(axis="index").set_properties(**{"text-align": "left"}))


def marker_distributions(adatas, marker, layer=None, bins=80, figsize=None):
    """Histogram of a single marker across datasets.

    adatas : dict[name -> AnnData]
    layer  : layer name to use (None = X)
    """
    n = len(adatas)
    fig, axes = plt.subplots(1, n, figsize=figsize or (n * 4, 3.5), facecolor="white")
    if n == 1:
        axes = [axes]
    for ax, (name, adata) in zip(axes, adatas.items()):
        if marker not in adata.var_names:
            ax.set_visible(False)
            continue
        X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
        col_idx = adata.var_names.get_loc(marker)
        col = X[:, col_idx]
        if sp.issparse(col):
            col = col.toarray().flatten()
        else:
            col = np.asarray(col).flatten()
        col = col[~np.isnan(col.astype(float))]
        ax.hist(col.astype(float), bins=bins, color="#1f4e79", alpha=0.85, edgecolor="none")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel(marker, fontsize=10)
        ax.set_ylabel("Cells", fontsize=10)
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
    title_layer = layer or "X"
    plt.suptitle(f"{marker}  [{title_layer}]", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_marker_distributions(adata, layer=None, markers=None, bins=60, ncols=6,
                              figsize=None, color="#1f4e79"):
    """Grid of per-marker histograms for ONE dataset (title from uns['dataset']).

    markers : list of canonical marker names (resolved to this adata's raw var_names
              via PANEL_ALIASES). None → every biological marker in the panel.
    """
    from data.EDA import PANEL_ALIASES, TECHNICAL_MARKERS

    name = adata.uns.get("dataset", "")
    canon_to_raw = {}
    for raw in adata.var_names:
        if raw in TECHNICAL_MARKERS:
            continue
        canon_to_raw.setdefault(PANEL_ALIASES.get(raw, raw), raw)

    if markers is None:
        sel = [(c, canon_to_raw[c]) for c in sorted(canon_to_raw)]
    else:
        sel = [(m, canon_to_raw[m]) for m in markers if m in canon_to_raw]

    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    n = len(sel)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (ncols * 2.4, nrows * 1.8),
                             facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for ax, (canon, raw) in zip(axes, sel):
        col = X[:, adata.var_names.get_loc(raw)]
        col = col[~np.isnan(col)]
        ax.hist(col, bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(canon, fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[n:]:
        ax.set_visible(False)
    layer_lbl = layer or "X"
    fig.suptitle(f"{name}  —  [{layer_lbl}]", fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.show()


# ── Transformation steps (Risom 2026 pipeline) ──────────────────────────────────

def apply_size_norm(adata, size_col, source_layer=None, target_layer="size_norm"):
    """Step 1 — divide each cell's intensities by its area → mean per-pixel intensity.

    Divergence from Risom: we do NOT rescale by the dataset's mean cell area. Patwa's
    published matrix is already per-pixel with no area column to recover that constant,
    so we drop the rescale for consistency. It is only a global multiplicative factor
    (absorbed by the arcsinh cofactor); can be reinstated per-dataset later if needed.
    """
    X = adata.layers[source_layer] if source_layer else adata.X
    if sp.issparse(X):
        X = X.toarray()
    sizes = adata.obs[size_col].to_numpy(dtype=float)[:, None]
    adata.layers[target_layer] = np.asarray(X, dtype=float) / sizes


def apply_arcsinh(adata, cofactor=5, source_layer=None, target_layer="exprs"):
    """Step 2 — arcsinh(x / cofactor) variance stabilization."""
    X = adata.layers[source_layer] if source_layer else adata.X
    if sp.issparse(X):
        X = X.toarray()
    adata.layers[target_layer] = np.arcsinh(np.asarray(X, dtype=float) / cofactor)


def apply_winsorize(adata, layer="exprs", pct=99.9):
    """Step 3 — cap each marker at its pct-th percentile (upper tail), in place."""
    X = adata.layers[layer]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    caps = np.nanpercentile(X, pct, axis=0)
    adata.layers[layer] = np.minimum(X, caps)


def apply_minmax(adata, layer="exprs", target_layer="exprs_norm"):
    """Step 4 — per-marker min-max scale to [0, 1] (Risom 0-1 normalization)."""
    X = np.asarray(adata.layers[layer], dtype=float)
    lo = np.nanmin(X, axis=0)
    hi = np.nanmax(X, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    adata.layers[target_layer] = (X - lo) / rng


def finalize_preprocessing(adata, cfg, winsorize_pct=99.9):
    """Compute ONLY the final Risom layer `exprs_norm` + complete provenance.

    Runs the full pipeline (size norm → arcsinh → winsorize → 0-1) on a working copy and
    writes just `layers["exprs_norm"]` (float32) and `uns["preprocessing"]`. X and all
    existing layers are left untouched — so Jackson's publisher `exprs`/`quant_norm` and
    Patwa's `positivity` survive, and the intermediate steps (size_norm, clean exprs)
    are NOT persisted (reproducible by rerunning preprocessing_EDA.ipynb).
    """
    cofactor = cfg.get("arcsinh_cofactor")
    size_col = cfg.get("size_col")

    if "exprs" in adata.layers:                       # Jackson — publisher arcsinh
        L = adata.layers["exprs"]
        M = (L.toarray() if sp.issparse(L) else np.asarray(L, dtype=float)).astype(float)
        prov = {"exprs_source": "pre-existing arcsinh (publisher)",
                "size_norm": False, "arcsinh_cofactor": None}
    elif cofactor is not None:                        # Schurch / Patwa — raw or per-pixel
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=float)
        X = X.astype(float)
        if size_col and size_col in adata.obs:
            X = X / adata.obs[size_col].to_numpy(dtype=float)[:, None]
            src, sn = f"X / {size_col} → arcsinh", True
        else:
            src, sn = "X → arcsinh", False
        M = np.arcsinh(X / cofactor)
        prov = {"exprs_source": src, "size_norm": sn,
                "size_col": size_col if sn else None, "arcsinh_cofactor": cofactor}
    else:                                             # Keren — X already arcsinh
        M = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=float).copy()
        prov = {"exprs_source": "X (pre-transformed)", "size_norm": False, "arcsinh_cofactor": None}

    caps = np.nanpercentile(M, winsorize_pct, axis=0)
    M = np.minimum(M, caps)
    lo = np.nanmin(M, axis=0)
    hi = np.nanmax(M, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    adata.layers["exprs_norm"] = ((M - lo) / rng).astype("float32")

    prov.update(winsorize_pct=winsorize_pct, norm="minmax_01",
                final_layer="exprs_norm", pipeline="Risom2026")
    adata.uns["preprocessing"] = prov


def prepare_exprs(adata, cfg):
    """Bring adata to the common variance-stabilized checkpoint: layers['exprs'].

    Combines step 1 (size norm, if cfg['size_col'] set) and step 2 (arcsinh).
    Dispatch (checked in order):
      1. 'exprs' already in layers → pre-existing arcsinh (Jackson)
      2. cfg['arcsinh_cofactor'] set → (optional size norm) → arcsinh(./cofactor)
      3. fallback → X already transformed (Keren) → copy to layers['exprs']
    """
    if "exprs" in adata.layers:
        adata.uns.setdefault("preprocessing",
                             {"exprs_source": "pre-existing arcsinh", "arcsinh_cofactor": None})
        return

    cofactor = cfg.get("arcsinh_cofactor")
    size_col = cfg.get("size_col")

    if cofactor is not None:
        source = None
        if size_col and size_col in adata.obs:
            if "size_norm" not in adata.layers:
                apply_size_norm(adata, size_col)
            source = "size_norm"
        apply_arcsinh(adata, cofactor=cofactor, source_layer=source)
        adata.uns["preprocessing"] = {
            "exprs_source": f"X / {size_col} → arcsinh" if source else "X → arcsinh",
            "arcsinh_cofactor": cofactor,
            "size_norm": bool(source),
        }
    else:
        X = adata.X
        adata.layers["exprs"] = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float).copy()
        adata.uns["preprocessing"] = {"exprs_source": "X (pre-transformed)", "arcsinh_cofactor": None}
