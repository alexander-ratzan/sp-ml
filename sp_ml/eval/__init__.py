"""sp_ml.eval — post-hoc evaluation: bidirectional W&B read-back + the metric panel."""

from sp_ml.eval.panel import panel, repeated_cv_panel
from sp_ml.eval.plots import prediction_panels, report
from sp_ml.eval.wandb_io import fetch_predictions, fetch_runs

__all__ = ["fetch_runs", "fetch_predictions", "panel", "repeated_cv_panel",
           "prediction_panels", "report"]
