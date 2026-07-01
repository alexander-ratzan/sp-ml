"""Fast smoke tests on a tiny synthetic AnnData — no real data, runs in seconds.

Covers the Checkpoint-0 contract: graph build + label mapping, patient-grouped
zero-leakage split, SpModel forward shape, patient aggregation, class weights.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sp_ml.configs import TaskCfg
from sp_ml.data.crossval import HoldoutSplit, make_holdout_split
from sp_ml.data.datamodule import SpatialGraphDataModule, build_sample_graphs
from sp_ml.models import Identity, LogReg, MeanPool, NoGraph, SpModel
from sp_ml.train import aggregate_by_patient, class_weights

N_PAT, REGIONS, CELLS, MARKERS = 6, 2, 40, 10
TASK = TaskCfg(name="tiny", kind="binary", classes=["A", "B"], source={"tiny": "group_name"})


def _make_anndata():
    import anndata as ad
    rng = np.random.default_rng(0)
    obs_rows, blocks = [], []
    for p in range(N_PAT):
        grp = "A" if p < N_PAT // 2 else "B"
        for r in range(REGIONS):
            region = f"p{p}_r{r}"
            obs_rows += [(region, str(p), grp)] * CELLS
            blocks.append(rng.random((CELLS, MARKERS), dtype=np.float32))
    X = np.concatenate(blocks)
    obs = pd.DataFrame(obs_rows, columns=["Region", "patients", "group_name"])
    a = ad.AnnData(X=X.copy(), obs=obs)
    a.layers["exprs_norm"] = X
    a.var_names = [f"m{i}" for i in range(MARKERS)]
    return a


@pytest.fixture
def tiny_h5ad(tmp_path):
    p = tmp_path / "tiny.h5ad"
    _make_anndata().write_h5ad(p)
    return str(p)


def test_build_sample_graphs():
    graphs, dropped, n_markers = build_sample_graphs(
        _make_anndata(), name="tiny", sample_col="Region",
        patient_col="patients", feature_layer="exprs_norm", task=TASK)
    assert dropped == 0
    assert n_markers == MARKERS
    assert len(graphs) == N_PAT * REGIONS
    g = graphs[0]
    assert g.x.shape == (CELLS, MARKERS) and g.x.dtype == torch.float32
    assert int(g.y) in (0, 1)
    assert isinstance(g.patient, str) and isinstance(g.sample_id, str)


def test_holdout_split_no_leakage():
    y = np.array([0, 0, 1, 1] * 3)          # 12 samples
    groups = np.repeat(np.arange(6), 2)     # 6 patients, 2 each
    tr, va, te = make_holdout_split(y, groups, n_folds=3, fold=0, repeat=0, seed=0)
    parts = [set(groups[idx]) for idx in (tr, va, te)]
    assert parts[0] & parts[1] == set() and parts[0] & parts[2] == set() and parts[1] & parts[2] == set()
    assert set().union(*parts) == set(range(6))
    assert len(tr) + len(va) + len(te) == 12


def test_datamodule_to_model(tiny_h5ad):
    dm = SpatialGraphDataModule(name="tiny", h5ad_path=tiny_h5ad, task=TASK,
                                sample_col="Region", patient_col="patients",
                                batch_size=4, num_workers=0)
    dm.split = HoldoutSplit(n_folds=3, fold=0, repeat=0, seed=0)
    dm.setup()
    assert dm.num_markers == MARKERS and dm.num_classes == 2
    # patient-disjoint folds
    pat = lambda idx: {dm.graphs[i].patient for i in idx}
    assert pat(dm.tr) & pat(dm.te) == set()

    enc = Identity(in_dim=dm.num_markers)
    gph = NoGraph(in_dim=enc.out_dim)
    pool = MeanPool(in_dim=gph.out_dim)
    readout = LogReg(in_dim=pool.out_dim, num_classes=dm.num_classes)
    model = SpModel(enc, gph, pool, readout)
    batch = next(iter(dm.train_dataloader()))
    logits = model(batch)
    assert logits.shape == (batch.num_graphs, dm.num_classes)
    assert torch.isfinite(logits).all()


def test_aggregate_by_patient():
    # two patients, two samples each; mean softmax per patient
    probs = torch.tensor([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])
    y = torch.tensor([0, 0, 1, 1])
    buf = [(probs, y, ["pa", "pa", "pb", "pb"])]
    pp, py, pats = aggregate_by_patient(buf)
    assert pp.shape == (2, 2) and py.tolist() == [0, 1] and pats == ["pa", "pb"]
    assert torch.allclose(pp[0], torch.tensor([0.8, 0.2]))
    assert torch.allclose(pp[1], torch.tensor([0.3, 0.7]))


def test_class_weights():
    assert torch.allclose(class_weights([0, 0, 1, 1], 2), torch.ones(2))   # balanced → 1
    w = class_weights([0, 0, 0, 1], 2)                                     # minority class 1 up-weighted
    assert w[1] > w[0]
