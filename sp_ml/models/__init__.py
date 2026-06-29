"""sp_ml.models — pure nn.Module architecture (no losses/optimizers/Lightning).

Four swappable components, each with a uniform shape contract — ``__init__(in_dim, …)``
and an ``out_dim`` attribute — composed by ``SpModel`` as ``encoder → graph → pool →
readout``. Shapes are threaded eagerly at build time (see ``sp_ml/run.py``), so the
readout's input width is always ``pool.out_dim``.
"""

from sp_ml.models.encoder import Identity
from sp_ml.models.graph import NoGraph
from sp_ml.models.model import SpModel
from sp_ml.models.pool import MeanPool
from sp_ml.models.readout import LogReg

__all__ = ["Identity", "NoGraph", "MeanPool", "LogReg", "SpModel"]
