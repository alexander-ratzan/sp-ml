"""Structured (dataclass) Hydra config schemas.

Only the two value objects that the spec types are registered here — ``DataCfg``
and ``TaskCfg`` — so their ``conf/`` leaves get struct validation. The model /
train / cv / eval groups stay plain DictConfigs (free-form ``_target_`` leaves).

``register_configs()`` must run before any ``hydra.compose`` / ``@hydra.main``
so the ``base_data`` / ``base_task`` schema nodes exist in the ConfigStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TaskCfg:
    """A dataset-spanning task: canonical vocab + per-dataset source column."""

    name: str = MISSING
    kind: str = MISSING                                # "binary" | "multiclass"
    classes: list[str] = field(default_factory=list)   # ordered vocab → class index
    source: dict[str, str] = field(default_factory=dict)  # dataset -> obs column
    remap: Optional[dict] = None                       # dataset -> {raw: canonical}


@dataclass
class DataCfg:
    """Shared DataModule defaults; per-dataset leaves carry the deltas."""

    _target_: str = "sp_ml.data.datamodule.SpatialGraphDataModule"
    name: str = MISSING
    h5ad_path: str = MISSING
    feature_layer: str = "exprs_norm"
    sample_col: str = "sample_id"
    patient_col: str = "patient_id"
    graph_kind: str = "knn"
    k: int = 20
    batch_size: int = 8
    num_workers: int = 8
    role: str = "cv"                                   # "cv" | "holdout"
    cache_dir: str = "cache/graphs"


def register_configs() -> None:
    """Register the structured schemas as group base nodes."""
    cs = ConfigStore.instance()
    cs.store(group="data", name="base_data", node=DataCfg)
    cs.store(group="task", name="base_task", node=TaskCfg)
