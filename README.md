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

### Wu et al. 2022 / SPACE-GM — pretraining & discovery tier

Three CODEX cohorts added as a separate tier (native panels kept, **not** reconciled). `X` = publisher z-scores; `exprs_norm` = upper-99.9 winsorize (per-marker) → min-max 0–1. **Region→patient identity is not in the public release for UPMC/DFCI** (region-level only → spatial pretraining / region-level tasks); it resolves for Charville. Built by `sp_ml/data/parse_wu2022.py`; see `context_packages/wu2022_integration_contract.md`.

| Cohort | Disease | Cells | Markers | Regions | Patients |
|---|---|---|---|---|---|
| UPMC-HNC | Head & Neck | 2,165,215 | 22 | 308 | region-level |
| Charville / Stanford-CRC | Colorectal | 632,280 | 40 | 292 | 162 |
| DFCI-HNC | Head & Neck | 136,680 | 41 | 58 | region-level |


## Structure

```
sp-ml/
├── pyproject.toml           # package + deps (torch pinned cu128); editable-installed
├── conf/                    # Hydra config tree — the experiment surface
├── sp_ml/                   # the Python package (the "meat")
│   ├── configs.py           # structured config schemas (DataCfg / TaskCfg)
│   ├── data/                # parsers, preprocessing, EDA/viz + modeling DataModule + crossval
│   ├── models/              # encoder / graph / pool / readout + SpModel
│   └── train/               # Lightning wrappers (LitClassifier, patient-level metrics)
│   # run.py (Hydra entrypoint) + eval/ (post-hoc) are next — see context_packages/repo_spec_v3.md
├── notebooks/
│   ├── EDA/                 # per-dataset + cross-dataset exploratory notebooks
│   └── poc/                 # Checkpoint-0 bag-of-cells walkthrough (bag_of_cells.ipynb)
└── context_packages/        # reference images, figures, schematics, repo spec (.gitignored)
```

## Notebooks

**Per-dataset** EDA notebooks (Keren, Schürch, Patwa, Jackson — full/Basel/Zurich cohorts) follow the same structure: load `.h5ad` → `summarize_metadata` → `spatial_info` → `cat_breakdown` → spatial viz → raw `.X` vs `exprs_norm` evaluation → switch `.X` to `exprs_norm` → scanpy profiles → spatial graph construction (Delaunay/KNN, prune, neighborhood enrichment).

**Cross-dataset** overview notebooks:
- `datasets_overview` — harmonized panel/cell-type coverage and a combined stats table across all four datasets.
- `preprocessing_overview` — the Risom preprocessing pipeline (size-norm → arcsinh → winsorize → 0–1) with per-step diagnostics across datasets.
- `wu2022_datasets_overview` / `wu2022_preprocessing_overview` — cross-Wu cohort overview + preprocessing (z-score → winsorize → 0–1), and `wu2022_charville_eda` — interactive Stanford-CRC EDA.

## Setup

The shared `.venv` and `requirements.txt` live in the parent workspace dir (one level above this repo):

```bash
uv venv ../.venv --python 3.12
source ../.venv/bin/activate
uv pip install -r ../requirements.txt          # EDA / scverse base stack

# ML stack — torch MUST be the cu128 build (matches the Converge A10G driver, CUDA 12.8;
# the default PyPI wheel is cu130 and will NOT see the GPU on this driver):
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install torch-geometric lightning torchmetrics hydra-core wandb   # torch already satisfied
uv pip install -e . --no-deps                  # install the sp_ml package only (preserves cu128 pin)
```

Verify the GPU is visible: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
