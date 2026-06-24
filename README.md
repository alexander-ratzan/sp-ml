# sp-ml — Spatial Proteomics ML

ML workflows for spatial proteomics data. Raw datasets live outside the repo in `../data/`; all large files (`*.h5ad`, `*.csv`, `*.parquet`, `*.zarr`, `*.tiff`) are gitignored.


## Datasets

All datasets are loaded as `.h5ad` from `../data/<dataset>/`.

Marker counts are biological panel size after dropping technical/elemental channels and harmonizing aliases.

**TNBC (Keren 2018) and TNBC (Patwa 2021) profile the same patients and the same MIBI images** — Patwa re-analyzes Keren's cohort with its own segmentation and richer clinical labels. Keren is listed separately but is a **subset of Patwa**, so the **Total is computed over the unique cohort**: Patwa is the TNBC representative and Keren is excluded to avoid double-counting (Keren can be used as alternate preprocessing pipeline validation dataset). Patients = unique patients in the loaded data.

| Dataset | Technology | Disease | Cells | Markers | Samples | Patients |
|---|---|---|---|---|---|---|
| TNBC (Keren 2018) | MIBI | TNBC | 173,205 | 36 | 34 | 34 |
| CRC (Schürch 2020) | CODEX | CRC | 258,385 | 57 | 70 | 35 |
| TNBC (Patwa 2021) | MIBI | TNBC | 190,240 | 36 | 38 | 38 |
| Breast Cancer (Jackson & Fischer 2020) | IMC | Breast Cancer | 1,240,267 | 37 | 723 | 285 |
| **Total (unique)** | | | **1,688,892** | | **831** | **358** |


## Structure

```
sp-ml/
├── models/                  # training scripts and model weights (Phase 2, empty)
├── data/
│   ├── parsers.py           # raw → AnnData parsers (parse_schurch2020, parse_patwa2021)
│   ├── preprocessing.py     # Risom pipeline: size-norm → arcsinh → winsorize → 0–1 (exprs_norm layer)
│   └── EDA.py               # summaries, panel/celltype harmonization, spatial viz
├── notebooks/
│   └── EDA/                 # per-dataset + cross-dataset exploratory notebooks
└── context_packages/        # reference images, figures, schematics (.gitignored)
```

## Notebooks

**Per-dataset** EDA notebooks (Keren, Schürch, Patwa, Jackson — full/Basel/Zurich cohorts) follow the same structure: load `.h5ad` → `summarize_metadata` → `spatial_info` → `cat_breakdown` → spatial viz → raw `.X` vs `exprs_norm` evaluation → switch `.X` to `exprs_norm` → scanpy profiles → spatial graph construction (Delaunay/KNN, prune, neighborhood enrichment).

**Cross-dataset** overview notebooks:
- `datasets_overview` — harmonized panel/cell-type coverage and a combined stats table across all four datasets.
- `preprocessing_overview` — the Risom preprocessing pipeline (size-norm → arcsinh → winsorize → 0–1) with per-step diagnostics across datasets.

## Setup

The shared `.venv` and `requirements.txt` live in the parent workspace dir (one level above this repo):

```bash
uv venv ../.venv --python 3.12
source ../.venv/bin/activate
uv pip install -r ../requirements.txt
```
