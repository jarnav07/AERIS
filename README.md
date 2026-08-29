# ML Aerofoil Predictor

Machine-learning surrogate models for airfoil aerodynamic performance. The canonical research pipeline generates stochastic Kulfan airfoil geometries, analyses them with XFOIL, stores one reproducible training dataset, trains multiple models on the same data, and evaluates their accuracy.

## Repository structure

```text
ml-aerofoil-predictor/
├── src/airfoil_ml/          # reusable research/package code
│   ├── training_data_generation.py  # canonical Kulfan + XFOIL generator
│   ├── training.py          # model training and model bundles
│   ├── models.py            # baseline model configuration
│   ├── torch_mlp.py         # PyTorch MLP implementation
│   ├── geometry.py          # geometry processing
│   ├── features.py          # feature construction
│   ├── evaluation.py        # evaluation plots/utilities
│   ├── error_analysis.py    # error/regime analysis
│   ├── drag_analysis.py     # drag-focused analysis
│   └── cli.py               # `airfoil-ml` command-line interface
├── scripts/                 # thin user-facing entry points
│   ├── generate_dataset.py
│   ├── train.py
│   └── evaluate_models.py
├── data/
│   ├── raw/                 # source data; normally not committed
│   ├── generated/            # canonical generated training dataset
│   └── processed/            # downstream processed datasets
├── models/                  # trained model artifacts
├── results/                 # plots, metrics and experiment outputs
└── tests/                   # automated tests
```

## Canonical data workflow

```text
AeroSandbox airfoil database
        ↓
Convert source airfoils to Kulfan parameters
        ↓
Random parent selection + weighted combination
        ↓
Covariance perturbation + random scaling
        ↓
Generate operating points
        ↓
XFOIL analysis
        ↓
Aerodynamic + boundary-layer outputs
        ↓
Parquet shards
        ↓
training_data.csv
        ↓
Train all ML models on the same dataset
        ↓
Evaluate and compare models
```

## Generate the dataset

From the repository root:

```bash
uv run python scripts/generate_dataset.py --cases 30000 --workers 8 --resume
```

The default output is `data/generated/`. Generation is deterministic for a fixed seed, uses independent XFOIL workers, supports resuming, records failures, and writes provenance alongside the combined CSV.

The package CLI is also available:

```bash
airfoil-ml generate-training-data --cases 30000 --workers 8 --resume
```

## Train models

```bash
uv run python scripts/train.py
```

or:

```bash
airfoil-ml train
```

## Evaluate

```bash
uv run python scripts/evaluate_models.py
```

## Development

Install the project with its development dependencies and run:

```bash
uv sync --extra dev
uv run pytest
```

XFOIL and `xvfb-run` are required for dataset generation on a headless Linux machine.
