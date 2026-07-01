"""Pooled prediction visualizations from `fetch_predictions` output.

Operates on the patient-level prediction frame (`patient, y_true, prob_0..prob_{C-1}`),
pooled across all folds/repeats. Binary-focused (confusion + report work for any C; ROC/PR
use the positive class). Plotting code only — figure artifacts belong in `notebooks/figures/`.
"""

from __future__ import annotations

import numpy as np


def prob_matrix(df):
    cols = sorted([c for c in df.columns if c.startswith("prob_")],
                  key=lambda c: int(c.split("_")[1]))
    return df[cols].to_numpy(), cols


def _reduce(df, mode):
    """`patient` → one row/patient (mean prob over repeats, honest N); `pooled` → as-is."""
    if mode == "pooled":
        return df
    probcols = [c for c in df.columns if c.startswith("prob_")]
    agg = {c: "mean" for c in probcols}
    agg["y_true"] = "first"
    return df.groupby("patient", as_index=False).agg(agg)


def report(df, class_names=None, mode="patient"):
    """sklearn classification_report on argmax predictions (per-patient averaged by default)."""
    from sklearn.metrics import classification_report
    df = _reduce(df, mode)
    P, _ = prob_matrix(df)
    y = df["y_true"].to_numpy()
    return classification_report(y, P.argmax(1), target_names=class_names, zero_division=0)


def prediction_panels(df, class_names=None, mode="patient"):
    """2×2 panel: confusion (row-normalized) · ROC · PR · calibration. Returns the Figure.

    `mode="patient"` (default) averages each patient's repeat-probabilities → one row/patient
    (honest N, avoids repeat double-counting); `mode="pooled"` uses every fold×repeat row."""
    df = _reduce(df, mode)
    import matplotlib.pyplot as plt
    from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                                 roc_curve)

    P, _ = prob_matrix(df)
    y = df["y_true"].to_numpy()
    C = P.shape[1]
    names = class_names or [str(i) for i in range(C)]
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))

    # confusion (row-normalized = per-true-class recall)
    cm = confusion_matrix(y, P.argmax(1), labels=range(C))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    a = ax[0, 0]
    im = a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    a.set(xticks=range(C), yticks=range(C), xticklabels=names, yticklabels=names,
          xlabel="predicted", ylabel="true", title=f"Confusion (row-norm)  n={len(y)}")
    for i in range(C):
        for j in range(C):
            a.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                   color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046)

    if C == 2:  # ROC + PR on the positive class
        s = P[:, 1]
        fpr, tpr, _ = roc_curve(y, s)
        ax[0, 1].plot(fpr, tpr, lw=2, label=f"AUROC={auc(fpr, tpr):.3f}")
        ax[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
        ax[0, 1].set(xlabel="FPR", ylabel="TPR", title="ROC (pooled)"); ax[0, 1].legend(loc="lower right")

        prec, rec, _ = precision_recall_curve(y, s)
        ax[1, 0].plot(rec, prec, lw=2, label=f"AUPRC={auc(rec, prec):.3f}")
        ax[1, 0].axhline(y.mean(), ls="--", c="grey", lw=1, label=f"base={y.mean():.2f}")
        ax[1, 0].set(xlabel="recall", ylabel="precision", title="PR (pooled)"); ax[1, 0].legend(loc="lower left")

        # calibration (reliability) on positive-class prob
        bins = np.linspace(0, 1, 11)
        ids = np.digitize(s, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = ids == b
            if m.any():
                xs.append(s[m].mean()); ys.append(y[m].mean())
        ax[1, 1].plot([0, 1], [0, 1], "k--", lw=1)
        ax[1, 1].plot(xs, ys, "o-", lw=2)
        ax[1, 1].set(xlabel="mean predicted prob", ylabel="observed frequency",
                     title="Calibration", xlim=(0, 1), ylim=(0, 1))
    else:
        for a in (ax[0, 1], ax[1, 0], ax[1, 1]):
            a.axis("off"); a.text(0.5, 0.5, "ROC/PR/cal: binary only", ha="center")

    fig.tight_layout()
    return fig
