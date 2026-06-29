"""Pooling: per-graph node set → one vector. Contract: ``__init__(in_dim, …)`` + ``out_dim``."""

import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class MeanPool(nn.Module):
    """Mean over a graph's cells (POC). ``out_dim == in_dim``."""

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = self.out_dim = in_dim

    def forward(self, x, batch):
        return global_mean_pool(x, batch)
