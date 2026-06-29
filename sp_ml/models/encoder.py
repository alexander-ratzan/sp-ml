"""Cell-wise encoders. Contract: ``__init__(in_dim, …)`` + ``out_dim``."""

import torch.nn as nn


class Identity(nn.Module):
    """Pass-through encoder (POC). ``out_dim == in_dim``."""

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = self.out_dim = in_dim

    def forward(self, x):
        return x
