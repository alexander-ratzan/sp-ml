# AGENTS — sp-ml

Internal ledger for AI agents and collaborators. Keep lean.

## Repo purpose

Spatial proteomics ML research. Phase 1 = data ingestion and EDA across four published datasets. Phase 2 = feature engineering and model development (not started).

## File conventions

- `models/` — training and model development
- `context_packages/` - images, figures, schematic references for task execution. only interface with when referred to.
- `data/` - data parsers, loaders, EDA, and viz
- `data/dataset_parsers.py` — raw format → AnnData. One `parse_<dataset><year>()` function per dataset.
- `data/dataset_EDA.py` — AnnData analysis and visualization only. Holds per-dataset configs (`*_CFG`), reference color maps (`*_CMAP`), summary functions, and spatial plot functions.

- `notebooks/` - experimental notebook directory and data viz
- `notebooks/EDA/` — one notebook per dataset cohort. All follow the same cell order: imports → load `.h5ad` → metadata summary → visualizations.

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
- [x] Raw parsers — `parse_schurch2020()`, `parse_patwa2021()` in `dataset_parsers.py`
- [x] EDA utilities — configs, reference color maps, `summarize_metadata`, `spatial_info`, `cat_breakdown`, `plot_sample/samples/all_samples` in `dataset_EDA.py`
- [x] EDA notebooks — Keren, Schürch, Patwa, Jackson & Fischer (full + Basel + Zurich cohorts), datasets overview
- [x] Dynamic repo-root sys.path in all notebooks
