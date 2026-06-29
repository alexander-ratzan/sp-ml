"""Classification heads. Contract: ``__init__(in_dim, num_classes)`` + ``out_dim``."""

import torch.nn as nn


class LogReg(nn.Module):
    """Linear head — logistic regression on the pooled vector (POC)."""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = num_classes
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, g):
        return self.linear(g)
