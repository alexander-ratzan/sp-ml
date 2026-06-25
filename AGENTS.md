# AGENTS — sp-ml

Internal ledger for AI agents and collaborators. Keep lean.

## Repo purpose

Spatial proteomics ML research. Phase 1 = data ingestion and EDA across four published datasets. Phase 2 = feature engineering and model development (not started).

## File conventions

- `models/` — training and model development
- `context_packages/` - images, figures, schematic references for task execution. only interface with when referred to.
  - `context_packages/datasets_overview.md` — **authoritative dataset reference.** Per-dataset AnnData structure, preprocessing state, panel harmonization, and **spatial resolution + biologically-motivated graph-construction defaults** (µm/pixel, paper graph params, empirical NN spacing, recommended squidpy `spatial_neighbors` settings). Read before any cross-dataset or graph/neighborhood work.
- `data/` - data parsers, loaders, preprocessing, EDA, and viz
- `data/parsers.py` — raw format → AnnData. One `parse_<dataset><year>()` function per dataset. Currently `parse_schurch2020`, `parse_patwa2021` (Keren loads a downloaded `.h5ad`; Jackson is built by `../data/jacksonfischer2020/scripts/`).
- `data/preprocessing.py` — expression diagnostics + the Risom 2026 pipeline. Diagnostics (`expression_stats`, `expression_stats_table`, `marker_distributions`, `plot_marker_distributions`); steps (`apply_size_norm`, `apply_arcsinh`, `apply_winsorize`, `apply_minmax`); `prepare_exprs` (→ variance-stabilized `exprs` checkpoint) and `finalize_preprocessing` (→ persists final `exprs_norm` layer + `adata.uns["preprocessing"]` provenance). `X` stays raw; `exprs_norm` is the modeling input.
- `data/EDA.py` — AnnData analysis and visualization. Holds per-dataset configs (`*_CFG`, incl. `sample_col`, `um_per_px`, `size_col`/`arcsinh_cofactor`), reference color maps (`*_CMAP`), font scale (`FS`), metadata summaries (`summarize_metadata`, `spatial_info`, `cat_breakdown`), spatial plots (`plot_sample/samples/all_samples`, `plot_marker_distributions`), `register_cmap`, panel/cell-type harmonization (`normalize_panel`, `shared_markers`, `normalize_celltypes`, `panel_heatmap` + alias/exclude tables), marker functional categories + canonical plotting order (`MARKER_CATEGORY_ORDER`, `MARKER_CATEGORIES`, `order_markers` — alias-aware ordering applied as the default across all marker plots; reorder `adata[:, order_markers(...)]` once so dotplot/matrixplot/corr inherit it), cross-dataset overview (`dataset_stats`, `overview_table`), spatial graph build+eval (`prune_and_eval_graph`), marker-cycle GIF (`representative_samples`, `marker_cycle_gif`), and its static companion `graph_celltype_panels` (pruned-Delaunay graph per representative sample, cells colored by the dataset cell-type column, cell-type legend as a separate figure).

- `notebooks/` - experimental notebook directory and data viz
- `notebooks/EDA/` — per-dataset cohort notebooks (cell order: imports → load `.h5ad` → metadata summary → spatial viz → raw `.X` vs `exprs_norm` eval → switch `.X`=`exprs_norm` (drops all-NaN markers) → scanpy profiles → graph construction: `spatial_neighbors` Delaunay+KNN with `library_key=CFG["sample_col"]` → `prune_and_eval_graph` → `nhood_enrichment`) plus cross-dataset notebooks (`datasets_overview`, `preprocessing_EDA`).

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
- [x] Preprocessing — Risom 2026 pipeline (`apply_size_norm/arcsinh/winsorize/minmax`, `prepare_exprs`, `finalize_preprocessing`) in `preprocessing.py`; `exprs_norm` + provenance persisted to all dataset `.h5ad` files
- [x] EDA notebooks — Keren, Schürch, Patwa, Jackson & Fischer (full + Basel + Zurich cohorts), datasets overview, preprocessing EDA
- [x] Dynamic repo-root sys.path in all notebooks
- [x] EDA notebooks switch analysis layer to `exprs_norm` + spatial graph construction (Delaunay/KNN → prune → neighborhood enrichment) via `prune_and_eval_graph`
