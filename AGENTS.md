# AGENTS — sp-ml

Internal ledger for AI agents and collaborators. Keep lean.

## Repo purpose

Spatial proteomics ML research. Phase 1 = data ingestion and EDA across four published datasets. Phase 2 = feature engineering and model development (not started).

## File conventions

- `models/` — training and model development
- `context_packages/` - images, figures, schematic references for task execution. only interface with when referred to.
- `data/` - data parsers, loaders, preprocessing, EDA, and viz
- `data/parsers.py` — raw format → AnnData. One `parse_<dataset><year>()` function per dataset. Currently `parse_schurch2020`, `parse_patwa2021` (Keren loads a downloaded `.h5ad`; Jackson is built by `../data/jacksonfischer2020/scripts/`).
- `data/preprocessing.py` — expression diagnostics and transformation. Diagnostics (`expression_stats`, `expression_stats_table`, `marker_distributions`); arcsinh handling (`apply_arcsinh`, `prepare_exprs` → writes `exprs` layer + provenance to `adata.uns["preprocessing"]`).
- `data/EDA.py` — AnnData analysis and visualization. Holds per-dataset configs (`*_CFG`), reference color maps (`*_CMAP`), font scale (`FS`), metadata summaries (`summarize_metadata`, `spatial_info`, `cat_breakdown`), spatial plots (`plot_sample/samples/all_samples`, `plot_marker_distributions`), `register_cmap`, panel/cell-type harmonization (`normalize_panel`, `normalize_celltypes`, `panel_heatmap` + alias/exclude tables), and cross-dataset overview (`dataset_stats`, `overview_table`).

- `notebooks/` - experimental notebook directory and data viz
- `notebooks/EDA/` — per-dataset cohort notebooks (cell order: imports → load `.h5ad` → metadata summary → visualizations) plus cross-dataset notebooks (`datasets_overview`, `preprocessing_EDA`).

## Import pattern

Notebooks resolve the repo root dynamically (no hardcoded `sys.path.insert`):

```python
from pathlib import Path
_r = next(p for p in [Path().resolve(), *Path().resolve().parents]
          if (p / "data").is_dir() and (p / "notebooks").is_dir())
if str(_r) not in sys.path: sys.path.insert(0, str(_r))
```

## Completed milestones

- [x] Repo scaffold — `.gitignore`, README, directory structure
- [x] Raw parsers — `parse_schurch2020()`, `parse_patwa2021()` in `parsers.py`
- [x] EDA utilities — configs, reference color maps, `summarize_metadata`, `spatial_info`, `cat_breakdown`, `plot_sample/samples/all_samples` in `EDA.py`
- [x] Panel/cell-type harmonization + cross-dataset overview — `normalize_panel`, `normalize_celltypes`, `panel_heatmap`, `dataset_stats`, `overview_table` in `EDA.py`
- [x] Preprocessing — expression diagnostics + arcsinh transform (`prepare_exprs`, `expression_stats`) in `preprocessing.py`
- [x] EDA notebooks — Keren, Schürch, Patwa, Jackson & Fischer (full + Basel + Zurich cohorts), datasets overview, preprocessing EDA
- [x] Dynamic repo-root sys.path in all notebooks
