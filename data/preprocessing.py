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


# ── Transformation ────────────────────────────────────────────────────────────

def apply_arcsinh(adata, cofactor=5, source_layer=None, target_layer="exprs"):
    """Apply arcsinh(x / cofactor) and store in adata.layers[target_layer]."""
    X = adata.layers[source_layer] if source_layer else adata.X
    if sp.issparse(X):
        X = X.toarray()
    adata.layers[target_layer] = np.arcsinh(np.asarray(X, dtype=float) / cofactor)


def prepare_exprs(adata, cfg):
    """Ensure adata.layers['exprs'] contains arcsinh-transformed expression.

    Dispatch logic (checked in order):
      1. 'exprs' already in layers → nothing to do (Jackson)
      2. cfg['expr_layer'] set → that layer already arcsinh → rename/copy
      3. cfg['arcsinh_cofactor'] set → apply arcsinh(X / cofactor)
      4. fallback → X already transformed (Keren) → copy to layers['exprs']
    """
    if "exprs" in adata.layers:
        if "preprocessing" not in adata.uns:
            adata.uns["preprocessing"] = {"exprs_source": "pre-existing", "arcsinh_cofactor": None}
        return

    src      = cfg.get("expr_layer")
    cofactor = cfg.get("arcsinh_cofactor")

    if src and src in adata.layers:
        X = adata.layers[src]
        adata.layers["exprs"] = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float).copy()
        adata.uns["preprocessing"] = {"exprs_source": src, "arcsinh_cofactor": None}
    elif cofactor is not None:
        apply_arcsinh(adata, cofactor=cofactor)
        adata.uns["preprocessing"] = {"exprs_source": "X", "arcsinh_cofactor": cofactor}
    else:
        # X is already transformed (e.g. Keren — Nolan lab pipeline applies arcsinh + background sub)
        X = adata.X
        adata.layers["exprs"] = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float).copy()
        adata.uns["preprocessing"] = {"exprs_source": "X", "arcsinh_cofactor": None}
