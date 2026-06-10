# Spatial Proteomics Machine Learning

Machine learning workflows for spatial proteomics data analysis.

## Overview

This repository contains model development, training scripts, and analysis notebooks for spatial proteomics experiments. It is intentionally lightweight — raw datasets are large and live **outside this repo**.

## Data

Core datasets reside in the shared scratch directory:

```
../data/
```

Do not commit raw data files. All `*.h5ad`, `*.csv`, `*.parquet`, `*.zarr`, and `*.tiff` files are gitignored.

## Structure

```
sp-ml/
├── data/        # Symlinks or small derived artifacts only
├── models/      # Trained model weights and configs
└── notebooks/   # Exploratory and reporting notebooks
```

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---
*last updated at 2026-06-09*
