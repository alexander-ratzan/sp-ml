# sp-ml — Spatial Proteomics ML

ML workflows for spatial proteomics data. Raw datasets live outside the repo in `../data/`; all large files (`*.h5ad`, `*.csv`, `*.parquet`, `*.zarr`, `*.tiff`) are gitignored.


## Datasets

All datasets are loaded as `.h5ad` from `../data/<dataset>/`.

Marker counts are biological panel size after dropping technical/elemental channels and harmonizing aliases.

| Dataset | Technology | Disease | Cells | Markers | Samples |
|---|---|---|---|---|---|
| Keren et al. 2018 | MIBI-TOF | TNBC | 173,205 | 36 | 34 |
| Schürch et al. 2020 | CODEX | CRC | 258,385 | 57 | 70 |
| Patwa et al. 2021 | MIBI | TNBC | 190,240 | 36 | 38 |
| Jackson & Fischer et al. 2020 | IMC | Breast Cancer | 1,240,267 | 37 | 723 |
| **Total** | | | **1,862,097** | | **865** |


## Structure

```
sp-ml/
├── models/                  # training scripts and model weights (Phase 2, empty)
├── data/
│   ├── parsers.py           # raw → AnnData parsers (parse_schurch2020, parse_patwa2021)
│   ├── preprocessing.py     # expression diagnostics + arcsinh transform (exprs layer)
│   └── EDA.py               # summaries, panel/celltype harmonization, spatial viz
├── notebooks/
│   └── EDA/                 # per-dataset + cross-dataset exploratory notebooks
└── context_packages/        # reference images, figures, schematics (.gitignored)
```

## Notebooks

**Per-dataset** EDA notebooks (Keren, Schürch, Patwa, Jackson — full/Basel/Zurich cohorts) follow the same structure: load `.h5ad` → `summarize_metadata` → `spatial_info` → `cat_breakdown` → spatial visualizations.

**Cross-dataset** notebooks:
- `datasets_overview` — harmonized panel/cell-type coverage and a combined stats table across all four datasets.
- `preprocessing_EDA` — expression diagnostics and arcsinh transform comparison across datasets.

## Setup

The shared `.venv` and `requirements.txt` live in the parent workspace dir (one level above this repo):

```bash
uv venv ../.venv --python 3.12
source ../.venv/bin/activate
uv pip install -r ../requirements.txt
```
