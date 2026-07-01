"""Minimal evaluation panel — mean±std over folds from the read-back DataFrame.

No literature baselines (comparison to published methods is a separate manual exercise).
Every roll-up is a `groupby`; `panel(df, by=…)` covers CV folds, HPO configs, and pooling
ablations with the same call.
"""

from __future__ import annotations

import pandas as pd

HEADLINE = ["test/auroc", "test/auprc", "test/bacc", "test/f1"]


def panel(df, metrics=HEADLINE, by=None):
    """Tidy mean±std table. `by=None` → overall; `by="cfg.model.pool._target_"` (or a list)
    → one block per group. Returns a DataFrame with `mean`, `std`, `n`, and a `summary` string."""
    metrics = [m for m in metrics if m in df.columns]
    keys = [by] if isinstance(by, str) else list(by) if by else []

    def _agg(sub):
        return [{"metric": m, "mean": sub[m].mean(), "std": sub[m].std(), "n": int(sub[m].count())}
                for m in metrics]

    if keys:
        rows = []
        for key, sub in df.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            for rec in _agg(sub):
                rows.append({**dict(zip(keys, key)), **rec})
        out = pd.DataFrame(rows)
    else:
        out = pd.DataFrame(_agg(df))

    out["summary"] = out.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    return out


def repeated_cv_panel(df, repeat_col="cfg.cv.repeat", metrics=HEADLINE):
    """Two clearly-labeled estimators of the repeated-CV score, side by side per metric:

    - **pooled**       — mean ± std over ALL fold-runs (n = runs). The std here is *per-fold
      dispersion* (each fold scored on few patients), NOT the uncertainty of the estimate.
    - **repeat_mean**  — mean ± std over the per-repeat means (n = repeats). Each repeat is a full
      K-fold partition, so this is the **stable headline** — the number to report.

    Columns are named with the estimator + n so it's unambiguous which is which.
    """
    metrics = [m for m in metrics if m in df.columns]
    rep_means = df.groupby(repeat_col)[metrics].mean()      # one row per repeat (mean of its folds)
    pooled_col = f"pooled  (mean±std, n={len(df)} runs)"
    stable_col = f"repeat_mean  (mean±std, n={len(rep_means)} repeats) ← report"
    rows = [{
        "metric": m,
        stable_col: f"{rep_means[m].mean():.3f} ± {rep_means[m].std():.3f}",
        pooled_col: f"{df[m].mean():.3f} ± {df[m].std():.3f}",
    } for m in metrics]
    return pd.DataFrame(rows)
