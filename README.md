# sp-ml — Spatial Proteomics ML

ML workflows for spatial proteomics data. Raw datasets live outside the repo in `../data/`; all large files (`*.h5ad`, `*.csv`, `*.parquet`, `*.zarr`, `*.tiff`) are gitignored.

## Structure

```
sp-ml/
├── data/
│   ├── dataset_parsers.py   # raw → AnnData parsers (Schurch, Patwa, ...)
│   └── dataset_EDA.py       # AnnData analysis and visualization utilities
├── models/                  # training scripts and model weights
├── notebooks/
│   └── EDA/                 # per-dataset exploratory notebooks
└── context_packages/        # reference images (paper figure color maps)
```

## Datasets

All datasets are loaded as `.h5ad` from `../data/<dataset>/`.

| Dataset | Technology | Disease | Cells | Markers | Samples |
|---|---|---|---|---|---|
| Keren et al. 2018 | MIBI-TOF | TNBC | 173,205 | 36 | 34 |
| Schürch et al. 2020 | CODEX | CRC | 258,385 | 58 | 70 |
| Patwa et al. 2021 | MIBI | TNBC | 190,240 | 44 | 38 |
| Jackson & Fischer et al. 2020 | IMC | Breast Cancer | 1,240,267 | 45 | 723 |

## Notebooks

Per dataset EDA notebooks follow the same structure: load `.h5ad` → `summarize_metadata` → `spatial_info` → `cat_breakdown` → spatial visualizations.

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```
