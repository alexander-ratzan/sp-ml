"""Graph interaction layers. Contract: ``__init__(in_dim, …)`` + ``out_dim``."""

import torch.nn as nn


class NoGraph(nn.Module):
    """No graph interaction (POC bag-of-cells). Ignores ``edge_index``; ``out_dim == in_dim``."""

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = self.out_dim = in_dim

    def forward(self, x, edge_index=None):
        return x
