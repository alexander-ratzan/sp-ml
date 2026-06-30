# AGENTS — sp-ml

Internal ledger for AI agents and collaborators. Keep lean.

## Repo purpose

Spatial proteomics ML research. Phase 1 = data ingestion and EDA across four core published datasets (+ a Wu et al. 2022 / SPACE-GM pretraining tier of 3 CODEX cohorts). Phase 2 = feature engineering and model development (in progress — see `context_packages/repo_spec_v3.md`).

## File conventions

- `sp_ml/` — the editable-installed Python package; all modeling code lives here. Built: `data/`, `models/`, `train/`, `configs.py` (schemas) + `conf/` (experiment surface), `run.py` (Hydra entrypoint: compose → GPU preflight → build → fit → test; W&B offline-default via `cfg.wandb`). Pending: `eval/` (post-hoc). Architecture + rationale: `context_packages/repo_spec_v3.md`.
- `pyproject.toml` — package metadata + deps; torch pinned to the **cu128** build (see file note; default PyPI wheel is cu130 and won't see a 12.8-driver GPU).
- `context_packages/` - images, figures, schematic references for task execution. only interface with when referred to.
  - `context_packages/datasets_overview.md` — **authoritative dataset reference.** Per-dataset AnnData structure, preprocessing state, panel harmonization, and **spatial resolution + biologically-motivated graph-construction defaults** (µm/pixel, paper graph params, empirical NN spacing, recommended squidpy `spatial_neighbors` settings). Read before any cross-dataset or graph/neighborhood work.
- `sp_ml/data/` - data parsers, loaders, preprocessing, EDA, viz, and the modeling DataModule
- `sp_ml/data/datamodule.py` — `SpatialGraphDataModule` + `build_sample_graphs`: AnnData → one PyG `Data` per `sample_col` (bag-of-cells, **no edges yet**); `y` from the task config, `patient`/`sample_id` carried for patient-grouped CV + patient-level scoring. Spatial-graph edges + `InMemoryDataset` cache are deferred to the first graph layer (post-Checkpoint-0).
- `sp_ml/data/crossval.py` — `make_holdout_split` + `HoldoutSplit`: patient-grouped, label-stratified single 3:1:1 train/val/test split (nested `StratifiedGroupKFold`), deterministic in `(seed, repeat, fold)`. The DataModule consumes a split via `split.apply(graphs)`; assign it as `dm.split = HoldoutSplit(...)` **after** `instantiate` (Hydra converts a dataclass kwarg into a DictConfig and strips its methods). Full repeated-nested protocol + `splits.json` artifact deferred to CV scale-out.
- `sp_ml/models/` — pure architecture (`nn.Module`, no Lightning). `encoder.py` (`Identity`), `graph.py` (`NoGraph`), `pool.py` (`MeanPool`), `readout.py` (`LogReg`), `model.py` (`SpModel`). Every component: `__init__(in_dim, …)` + `out_dim`; `_target_` leaves in `conf/model/{encoder,graph,pool,readout}/` select them. New architecture = one ~4-line component + one leaf.
- `sp_ml/train/` — Lightning wrappers. `litbase.py` (`LitBase`: optimizer/scheduler from Hydra partials, over `requires_grad` params only). `classifier.py` (`LitClassifier`: CrossEntropy + torchmetrics; `aggregate_by_patient` = eval-only mean-softmax patient rollup; `class_weights` = train-fold inverse freq). `_target_: sp_ml.train.LitClassifier` in `conf/train/supervised.yaml`. Metrics live in the wrapper (one path, no train/eval skew).
- `sp_ml/data/parsers.py` — raw format → AnnData. One `parse_<dataset><year>()` function per dataset. Currently `parse_schurch2020`, `parse_patwa2021` (Keren loads a downloaded `.h5ad`; Jackson is built by `../data/jacksonfischer2020/scripts/`). `parse_schurch2020` also merges per-patient survival + clinical metadata from a sibling `crc_metadata.xlsx` (Supp. Table S1, Sheet A) — survival `OS`/`DFS`(+censors), `sex`/`age`, staging, MMR/MSI — broadcast cell→patient (`_SCHURCH_META_COLS`).
- `sp_ml/data/parse_wu2022.py` — Wu 2022 / SPACE-GM parser (UPMC, Charville, DFCI). `build_wu_anndata`/`write_wu_h5ad(cohort)` read the per-region CSV quads **directly from the cohort zips** → one unified-schema `.h5ad` per cohort. Each cohort keeps its **native panel** (22/40/41 markers — NOT reconciled). `X` = publisher z-scores (verbatim); `exprs_norm` = **upper-99.9 winsorize (per-marker) → min-max 0–1** (z-scores are right-skewed with extreme positive artifacts; winsorize is required before min-max). Region→patient resolves for Charville only; UPMC/DFCI fall back to `region_id` (public release is region-level → pretraining/region-level use). Full contract: `context_packages/wu2022_integration_contract.md`.
- `sp_ml/data/preprocessing.py` — expression diagnostics + the Risom 2026 pipeline. Diagnostics (`expression_stats`, `expression_stats_table`, `marker_distributions`, `plot_marker_distributions`); steps (`apply_size_norm`, `apply_arcsinh`, `apply_winsorize`, `apply_minmax`); `prepare_exprs` (→ variance-stabilized `exprs` checkpoint) and `finalize_preprocessing` (→ persists final `exprs_norm` layer + `adata.uns["preprocessing"]` provenance). `X` stays raw; `exprs_norm` is the modeling input.
- `sp_ml/data/EDA.py` — AnnData analysis and visualization. Holds per-dataset configs (`*_CFG`, incl. `sample_col`, `um_per_px`, `size_col`/`arcsinh_cofactor`), reference color maps (`*_CMAP`), font scale (`FS`), metadata summaries (`summarize_metadata`, `spatial_info`, `cat_breakdown`), spatial plots (`plot_sample/samples/all_samples`, `plot_marker_distributions`), `register_cmap`, panel/cell-type harmonization (`normalize_panel`, `shared_markers`, `normalize_celltypes`, `panel_heatmap` + alias/exclude tables), marker functional categories + canonical plotting order (`MARKER_CATEGORY_ORDER`, `MARKER_CATEGORIES`, `order_markers` — alias-aware ordering applied as the default across all marker plots; reorder `adata[:, order_markers(...)]` once so dotplot/matrixplot/corr inherit it), cross-dataset overview (`dataset_stats`, `overview_table`), spatial graph build+eval (`prune_and_eval_graph`), marker-cycle GIF (`representative_samples`, `marker_cycle_gif`), and its static companion `graph_celltype_panels` (pruned-Delaunay graph per representative sample, cells colored by the dataset cell-type column, cell-type legend as a separate figure).

- `notebooks/` - experimental notebook directory and data viz
- `notebooks/EDA/` — per-dataset cohort notebooks (cell order: imports → load `.h5ad` → metadata summary → spatial viz → raw `.X` vs `exprs_norm` eval → switch `.X`=`exprs_norm` (drops all-NaN markers) → scanpy profiles → graph construction: `spatial_neighbors` Delaunay+KNN with `library_key=CFG["sample_col"]` → `prune_and_eval_graph` → `nhood_enrichment`) plus cross-dataset notebooks (`datasets_overview`, `preprocessing_EDA`) and the Wu2022 set (`wu2022_datasets_overview`, `wu2022_preprocessing_overview` cross-Wu; `wu2022_charville_eda` interactive Stanford-CRC EDA). Wu notebooks define inline cfgs (the Wu cohorts are intentionally not in `EDA.py`'s `*_CFG` system — v3 wiring deferred).

## Import pattern

The package is **editable-installed** (`uv pip install -e . --no-deps`), so notebooks and code
import it directly:

```python
from sp_ml.data.EDA import SCHURCH_CFG, shared_markers      # etc.
from sp_ml.data.preprocessing import prepare_exprs
```

Existing EDA notebooks keep a belt-and-suspenders root-finder (now keyed on `sp_ml/`), redundant
given the install.

## Completed milestones

- [x] Repo scaffold — `.gitignore`, README, directory structure
- [x] Raw parsers — `parse_schurch2020()`, `parse_patwa2021()` in `parsers.py`
- [x] EDA utilities — configs, reference color maps, `summarize_metadata`, `spatial_info`, `cat_breakdown`, `plot_sample/samples/all_samples` in `EDA.py`
- [x] Panel/cell-type harmonization + cross-dataset overview — `normalize_panel`, `normalize_celltypes`, `panel_heatmap`, `dataset_stats`, `overview_table` in `EDA.py`
- [x] Preprocessing — Risom 2026 pipeline (`apply_size_norm/arcsinh/winsorize/minmax`, `prepare_exprs`, `finalize_preprocessing`) in `preprocessing.py`; `exprs_norm` + provenance persisted to all dataset `.h5ad` files
- [x] EDA notebooks — Keren, Schürch, Patwa, Jackson & Fischer (full + Basel + Zurich cohorts), datasets overview, preprocessing EDA
- [x] Dynamic repo-root sys.path in all notebooks
- [x] EDA notebooks switch analysis layer to `exprs_norm` + spatial graph construction (Delaunay/KNN → prune → neighborhood enrichment) via `prune_and_eval_graph`
- [x] Wu et al. 2022 / SPACE-GM integration — `parse_wu2022.py` builds UPMC/Charville/DFCI `.h5ad`s (native panels; `X` z-scores → `exprs_norm` upper-99.9 winsorize → min-max 0–1); cross-Wu `datasets_overview` + `preprocessing_overview` notebooks + interactive Charville/Stanford-CRC EDA. Region→patient unresolved for UPMC/DFCI (public release is region-level) → those two are pretraining/region-level only; Charville resolves (162). v3 config/task wiring deferred. Contract + provenance: `context_packages/wu2022_integration_contract.md`.
- [x] Phase 2 — **Checkpoint 0 complete (S0–S9)**: Hydra + Lightning + PyG spine. `pyproject` (cu128) · `sp_ml/{configs.py, data, models, train, run.py}` · `conf/` · POC notebook · `scripts/poc.sbatch` (CPU `defq` default) · `tests/`. Notebook, CLI (`python -m sp_ml.run`), and sbatch all reproduce identical metrics; W&B offline+online tested. **Per-stage status, gates, and design rationale: `context_packages/repo_spec_v3.md` (S0–S9 build table) — the detailed ledger; don't duplicate here.**
