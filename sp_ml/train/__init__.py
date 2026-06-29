"""sp_ml.train — Lightning wrappers (the training contract).

Holds an `nn.Module` from `sp_ml.models`; owns loss, optimizer, metrics. Knows nothing
about paths or sweeps. In-loop metrics live here (one code path, no train/eval skew);
patient-level aggregation is the one custom step on top of torchmetrics.
"""

from sp_ml.train.classifier import LitClassifier, aggregate_by_patient, class_weights
from sp_ml.train.litbase import LitBase

__all__ = ["LitBase", "LitClassifier", "aggregate_by_patient", "class_weights"]
