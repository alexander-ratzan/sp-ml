"""SpModel — the four-component container, trainable end-to-end from the loss."""

import torch.nn as nn


class SpModel(nn.Module):
    """``encoder → graph → pool → readout`` over a PyG ``Batch``."""

    def __init__(self, encoder, graph, pool, readout):
        super().__init__()
        self.encoder, self.graph, self.pool, self.readout = encoder, graph, pool, readout

    def forward(self, data):
        x = self.encoder(data.x)
        x = self.graph(x, getattr(data, "edge_index", None))   # None for bag-of-cells
        g = self.pool(x, data.batch)
        return self.readout(g)
