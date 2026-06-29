"""Hydra entrypoint: compose → preflight → build → fit → test.

    python -m sp_ml.run                          # POC default (Schürch · clr_dii · bag-of-cells)
    python -m sp_ml.run model/graph=gcn data=patwa wandb.mode=online   # overrides

Assembles the exact spine the POC notebook validates, driven entirely by the composed config.
W&B defaults to offline (safe on unauthed nodes); flip with `wandb.mode=online` or disable
with `wandb.mode=disabled`.
"""

from __future__ import annotations

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from sp_ml.configs import register_configs
from sp_ml.data.crossval import HoldoutSplit
from sp_ml.models import SpModel
from sp_ml.train import class_weights

register_configs()


def _arch_covered(major, minor, arches):
    """True if torch's compiled arch list covers capability sm_{major}{minor} (minor fwd-compat)."""
    for a in arches:
        n = a[3:] if a.startswith("sm_") else ""
        if n.isdigit():
            amaj, amin = divmod(int(n), 10)
            if amaj == major and amin <= minor:
                return True
    return False


def gpu_preflight():
    """Fail fast on a GPU node whose arch torch can't run; CPU is allowed for the POC."""
    if not torch.cuda.is_available():
        print("[preflight] CUDA not available — running on CPU (fine for the POC).")
        return
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    sm = f"sm_{major}{minor}"
    print(f"[preflight] GPU={name} capability={sm} torch_arch_list={arches}")
    if not _arch_covered(major, minor, arches):
        raise RuntimeError(
            f"[preflight] this torch build does not cover {sm} (have {arches}). "
            f"On Converge try `ml cuda-compat/13`, or install a torch wheel that targets {sm}."
        )
    print(f"[preflight] {sm} covered ✓")


def make_logger(cfg):
    wb = cfg.get("wandb")
    if wb is None or wb.mode == "disabled":
        return False
    group = wb.get("group") or "-".join([
        cfg.data.name, cfg.task.name,
        "-".join(c._target_.split(".")[-1]
                 for c in (cfg.model.encoder, cfg.model.graph, cfg.model.pool, cfg.model.readout)),
    ])
    return WandbLogger(project=wb.project, entity=wb.get("entity"),
                       offline=(wb.mode == "offline"), group=group,
                       name=wb.get("name"), save_dir=cfg.output_dir)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    L.seed_everything(cfg.seed, workers=True)
    gpu_preflight()

    dm = instantiate(cfg.data, task=cfg.task)
    dm.split = HoldoutSplit(n_folds=cfg.cv.n_folds, fold=cfg.cv.fold,
                            repeat=cfg.cv.repeat, seed=cfg.cv.seed)
    dm.setup()

    encoder = instantiate(cfg.model.encoder, in_dim=dm.num_markers)
    graph = instantiate(cfg.model.graph, in_dim=encoder.out_dim)
    pool = instantiate(cfg.model.pool, in_dim=graph.out_dim)
    readout = instantiate(cfg.model.readout, in_dim=pool.out_dim, num_classes=dm.num_classes)
    model = SpModel(encoder, graph, pool, readout)
    # init_and_freeze (per-component finetuning) is deferred — no init_from/freeze in POC config.

    weight = class_weights(dm.train_labels(), dm.num_classes) if cfg.train.class_weighted else None
    loss = instantiate(cfg.train.get("loss"), weight=weight)
    lit = instantiate(cfg.train.litmodule, model=model,
                      optimizer=instantiate(cfg.train.optimizer),
                      loss=loss, num_classes=dm.num_classes)

    logger = make_logger(cfg)
    callbacks = [ModelCheckpoint(monitor="val/auroc", mode="max", save_top_k=1)]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = instantiate(cfg.train.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(lit, dm)
    trainer.test(lit, dm)
    if logger:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
