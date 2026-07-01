"""Bidirectional W&B — read runs back into a tidy DataFrame.

`fetch_runs(...)` returns one row per run: flattened config (`cfg.*`) + summary metrics
(`val/*`, `test/*`, …). Every roll-up downstream (CV mean±std, HPO selection, pooling
ablation) is a `groupby` on this frame — the same primitive in the batch panel and live in
a notebook. The write side is `run.py` (logs the resolved config + metrics per run).
"""

from __future__ import annotations

import pandas as pd
import wandb

DEFAULT_PROJECT = "sp-ml"


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def fetch_runs(group=None, filters=None, project=DEFAULT_PROJECT, entity=None,
               state="finished"):
    """Pull runs into a DataFrame (one row/run): config under ``cfg.*`` + summary metrics.

    `group` filters to one experiment group; `filters` is a raw W&B Mongo-style filter
    (merged with `group`/`state`). `entity=None` → the account default entity.
    """
    api = wandb.Api()
    entity = entity or api.default_entity
    f = dict(filters or {})
    if group is not None:
        f["group"] = group
    if state is not None:
        f["state"] = state
    runs = api.runs(f"{entity}/{project}", filters=f or None)

    rows = []
    for r in runs:
        row = {"run_id": r.id, "name": r.name, "group": r.group, "state": r.state}
        row.update(_flatten({k: v for k, v in r.config.items() if not k.startswith("_")},
                            prefix="cfg."))
        row.update({k: v for k, v in r.summary.items() if not k.startswith("_")})
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_predictions(group=None, filters=None, project=DEFAULT_PROJECT, entity=None,
                      state="finished"):
    """Download per-run test-prediction artifacts and concatenate → pooled patient-level frame.

    Columns: ``patient, y_true, fold, repeat, prob_0..prob_{C-1}, run_id``. One row per
    patient-level test prediction; a patient recurs once per repeat (→ pooled CV predictions).
    """
    import glob
    import os
    import tempfile

    api = wandb.Api()
    entity = entity or api.default_entity
    f = dict(filters or {})
    if group is not None:
        f["group"] = group
    if state is not None:
        f["state"] = state
    runs = api.runs(f"{entity}/{project}", filters=f or None)

    frames = []
    with tempfile.TemporaryDirectory() as td:
        for r in runs:
            for art in r.logged_artifacts():
                if art.type != "predictions":
                    continue
                d = art.download(root=os.path.join(td, r.id))
                for csv in glob.glob(os.path.join(d, "*.csv")):
                    sub = pd.read_csv(csv)
                    sub["run_id"] = r.id
                    frames.append(sub)
                break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
