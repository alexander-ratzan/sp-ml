"""LitBase — shared optimizer/scheduler plumbing for the supervised wrappers."""

import lightning as L


class LitBase(L.LightningModule):
    def __init__(self, model, optimizer, scheduler=None):
        super().__init__()
        self.model = model
        self._optim = optimizer          # partials from Hydra (_partial_: true)
        self._sched = scheduler

    def configure_optimizers(self):
        params = (p for p in self.parameters() if p.requires_grad)   # respects frozen components
        opt = self._optim(params)
        if self._sched is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": self._sched(opt)}
