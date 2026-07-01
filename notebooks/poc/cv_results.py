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
# # CV Floor Results — read back from W&B
#
# Bidirectional results exploration: pull a finished CV group from W&B and report it. Swap `GROUP`
# to inspect any experiment. Independent of `bag_of_cells` (which trains/submits); this only *reads*.

# %%
import pandas as pd

from sp_ml.eval import fetch_runs, panel

pd.set_option("display.width", 160)

GROUP = "floor-schurch-clr_dii-5fold"   # single 5-fold (1 repeat) floor: identity·none·mean·logreg
df = fetch_runs(group=GROUP).sort_values("cfg.cv.fold").reset_index(drop=True)
print(f"{len(df)} runs in group: {GROUP}")

# %% [markdown]
# ## Experiment provenance — what produced these numbers

# %%
r0 = df.iloc[0]
short = lambda c: str(r0[c]).split(".")[-1]
print("dataset :", r0["cfg.data.name"], "| task:", r0["cfg.task.name"])
print("model   :", " → ".join(short(f"cfg.model.{s}._target_")
                              for s in ["encoder", "graph", "pool", "readout"]))
print("CV      : n_folds=%s n_repeats=%s  (runs=%d)" %
      (r0["cfg.cv.n_folds"], r0["cfg.cv.n_repeats"], len(df)))
print("train   : epochs=%s lr=%s class_weighted=%s seed=%s" %
      (r0["cfg.train.trainer.max_epochs"], r0["cfg.train.optimizer.lr"],
       r0["cfg.train.class_weighted"], r0["cfg.seed"]))

# %% [markdown]
# ## Headline — patient-level test metrics, mean ± std over folds

# %%
print(panel(df)[["metric", "summary", "n"]].to_string(index=False))

# %% [markdown]
# ## Per-fold breakdown (spread, best/worst)

# %%
TEST = ["test/auroc", "test/auprc", "test/bacc", "test/f1"]
per = (df[["cfg.cv.fold"] + TEST]
       .rename(columns=lambda c: c.replace("cfg.cv.", "").replace("test/", "")))
print(per.to_string(index=False))
print("\nrange (max−min):", {m.replace("test/", ""): round(float(df[m].max() - df[m].min()), 3) for m in TEST})

# %% [markdown]
# ## Val vs test (generalization sanity — selection/checkpoint was on `val/auroc`)

# %%
for split in ["val", "test"]:
    ms = [f"{split}/{m}" for m in ["auroc", "auprc", "bacc", "f1"]]
    print("%-4s" % split, {m.split("/")[1]: f"{df[m].mean():.3f}±{df[m].std():.3f}" for m in ms})

# %% [markdown]
# ## Drill-down links

# %%
ENTITY = "alexander-ratzan-new-york-university"
for _, r in df.iterrows():
    print(f"  fold {int(r['cfg.cv.fold'])}: https://wandb.ai/{ENTITY}/sp-ml/runs/{r['run_id']}")

# %% [markdown]
# ## Pooled out-of-fold predictions — confusion / ROC / PR / calibration
# One repeat → each patient is tested exactly once, so pooling the 5 folds = 35 out-of-fold
# predictions (one per patient). Descriptive companion to the per-fold mean±std above: read the
# confusion as row-normalized recall; AUROC/AUPRC (rank-based) are the discrimination evidence.

# %%
from sp_ml.eval import fetch_predictions, prediction_panels, report

CLASSES = list(df["cfg.task.classes"].iloc[0])
sf_preds = fetch_predictions(group=GROUP)
print("out-of-fold predictions:", sf_preds.shape, "| patients:", sf_preds["patient"].nunique())
print(report(sf_preds, class_names=CLASSES))
prediction_panels(sf_preds, class_names=CLASSES)

# %% [markdown]
# # Full nested CV (5 folds × 10 repeats) — stable floor
#
# The single 5-fold run above is high-variance at 35 patients (AUROC range ~0.25). The full
# Repeated Nested Stratified Group K-Fold (50 runs) averages that into a stable mean±std and gives
# enough pooled patient-level predictions for honest confusion / ROC / PR / calibration.

# %%
NESTED_GROUP = "floor-schurch-clr_dii-5x10"   # explicit group for the nested experiment
ndf = fetch_runs(group=NESTED_GROUP)
print(f"{len(ndf)} runs (expect 50) | repeats: {sorted(ndf['cfg.cv.repeat'].unique())}")
CLASSES = list(ndf["cfg.task.classes"].iloc[0]) if len(ndf) else ["CLR", "DII"]

# %% [markdown]
# ### Headline — two estimators, clearly labeled
# **`repeat_mean` is the number to report**: mean ± std over the 10 per-repeat means (each a full
# 5-fold partition) → the stable estimate. `pooled` is mean ± std over all 50 individual fold-runs;
# its std is *per-fold dispersion* (few patients/fold), not the uncertainty of the estimate.

# %%
from sp_ml.eval import repeated_cv_panel

print(repeated_cv_panel(ndf).to_string(index=False))

# %% [markdown]
# ### Per-repeat stability (mean over each repeat's 5 folds)

# %%
print(panel(ndf, by="cfg.cv.repeat")
      .pivot(index="cfg.cv.repeat", columns="metric", values="mean").round(3).to_string())

# %% [markdown]
# ### Patient-level predictions — confusion / ROC / PR / calibration
# 350 test predictions (35 patients × 10 repeats) averaged per patient → one row/patient (honest
# N=35). `prediction_panels(..., mode="pooled")` would instead use all 350.

# %%
preds = fetch_predictions(group=NESTED_GROUP)
print("raw predictions:", preds.shape, "| unique patients:", preds["patient"].nunique())
print(report(preds, class_names=CLASSES))
prediction_panels(preds, class_names=CLASSES)
