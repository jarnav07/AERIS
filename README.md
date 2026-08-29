# ML Aerofoil Predictor

Machine-learning surrogate modelling for airfoil aerodynamics. The project investigates whether machine-learning models can learn the mapping from airfoil geometry and flow conditions to aerodynamic coefficients, using XFOIL-generated data as the reference dataset and comparing multiple models on exactly the same data.

## Research pipeline

```text
AeroSandbox built-in airfoil database
        |
        | convert source shapes to Kulfan/CST parameters
        v
Kulfan statistical distribution
        |
        | random parent selection + convex weighting
        | covariance perturbation + random scale
        v
New synthetic airfoil geometry
        |
        | geometry + stochastic operating conditions
        v
XFOIL viscous aerodynamic analysis
        |
        | CL, CD, CM + transition/boundary-layer data
        v
Parquet shards
        |
        v
One combined training dataset
        |
        +-----------------------------+
        |                             |
        v                             v
Grouped train/validation/test     Same data for every model
        |                             |
        +--------------+--------------+
                       v
              Model training
                       |
                       v
                Evaluation
                       |
                       v
             Compare model errors
```

The canonical dataset is generated once and shared by all ML models. This makes the model comparison a controlled experiment rather than a comparison between different datasets.

## Data source and generation methodology

The canonical generator is **not based on the UIUC coordinate database** and does **not train on the fixed-Re Kanakaero dataset**.

AeroSandbox's bundled airfoil database is used only as the source population for learning a statistical distribution of Kulfan parameters. The source airfoils are converted to Kulfan representations and used to calculate the mean/covariance of the geometry population. New synthetic airfoils are then sampled from that distribution.

The generator combines randomly selected parent airfoils using convex random weights, adds covariance-based perturbations, and applies random scaling. It then samples operating conditions and runs XFOIL. Reynolds number is sampled stochastically rather than selected from a fixed Reynolds-number list.

## Repository structure

```text
ml-aerofoil-predictor/
├── src/airfoil_ml/
│   ├── training_data_generation.py  # canonical Kulfan + XFOIL generator
│   ├── training.py                  # model training orchestration
│   ├── models.py                    # model definitions/configurations
│   ├── torch_mlp.py                 # PyTorch MLP
│   ├── geometry.py                  # geometry processing
│   ├── features.py                  # feature construction/scaling
│   ├── data.py                      # validation and grouped splitting
│   ├── evaluation.py                # metrics and evaluation utilities
│   ├── error_analysis.py             # error/regime analysis
│   ├── drag_analysis.py              # drag-focused analysis
│   └── cli.py                       # command-line interface
├── scripts/
│   ├── generate_dataset.py           # dataset-generation entry point
│   ├── train.py                     # training entry point
│   └── evaluate_models.py            # evaluation entry point
├── data/
│   ├── raw/                          # optional historical/source data
│   ├── generated/                    # canonical generated dataset
│   └── processed/                    # derived/processed data
├── models/                           # trained model artifacts
├── results/                          # metrics, plots and experiment outputs
└── tests/                            # automated tests
```

## Dataset generation

The canonical implementation is `src/airfoil_ml/training_data_generation.py`; `scripts/generate_dataset.py` is the user-facing entry point.

The generator:

1. Loads airfoils bundled with AeroSandbox.
2. Converts the source geometries into Kulfan representations.
3. Computes the Kulfan population mean and covariance.
4. Selects multiple parent airfoils for each synthetic case.
5. Forms a convex random weighted combination of the parents.
6. Applies a covariance-based perturbation and random geometric scaling.
7. Samples Reynolds number, angle of attack, Mach number and transition/turbulence conditions.
8. Runs XFOIL in isolated worker processes.
9. Extracts aerodynamic coefficients and boundary-layer information.
10. Writes resumable Parquet shards and records failed cases.
11. Combines successful shards into one canonical training dataset.

The current default configuration uses three parent airfoils, a seven-point alpha grid from -15 to +15 degrees with shared random jitter, a log-normal Reynolds distribution centred on log10(Re)=5.5 with sigma 1.5, Mach 0, and up to 200 XFOIL iterations per case. These are configurable defaults.

Generate data with:

```bash
uv run python scripts/generate_dataset.py --cases 30000 --workers 8 --resume
```

or:

```bash
airfoil-ml generate-training-data --cases 30000 --workers 8 --resume
```

Generation supports reproducible random seeds, parallel XFOIL workers, resumable shards, failure logging and provenance information.

## Dataset contents

The generated dataset contains Kulfan geometry parameters, operating/transition conditions, aerodynamic outputs and sampled boundary-layer information.

Principal aerodynamic targets are:

- `CL` — lift coefficient
- `CD` — drag coefficient
- `CM` — pitching-moment coefficient

The dataset also contains transition information, solver confidence information and upper/lower boundary-layer profiles.

## Models being trained and compared

Every model is trained using the **same generated dataset**, the same feature representation and the same grouped data split. The purpose is to determine which modelling approach gives the best aerodynamic surrogate performance.

### 1. Ridge Regression

Ridge is the linear baseline. It learns a linear mapping from the geometry and flow-condition features to the aerodynamic targets while applying L2 regularisation to penalise excessively large coefficients.

```text
features → weighted linear combination → aerodynamic outputs
```

It provides a low-complexity benchmark. Strong Ridge performance would indicate that a substantial part of the relationship is approximately linear; a large improvement from non-linear models indicates that additional model capacity is useful.

### 2. Random Forest

Random Forest is an ensemble of decision trees. Each tree recursively partitions the feature space into regions with similar target values, and the forest averages the predictions of many trees.

```text
                 Features
                    |
          +---------+---------+
          |         |         |
        Tree 1    Tree 2    ... Tree N
          |         |         |
          +---------+---------+
                    |
              average prediction
```

The current configuration uses 250 trees. It is a classical non-linear benchmark that can capture feature interactions without gradient-based neural-network training.

### 3. Scikit-learn MLP

The MLP is a fully connected feed-forward neural network with the current hidden architecture:

```text
Input → 128 → 64 → 32 → Output
```

ReLU activations introduce non-linearity. Backpropagation and the Adam optimiser adjust the network weights to minimise prediction error.

```text
geometry + flow conditions
            ↓
         Dense 128
            ↓ ReLU
          Dense 64
            ↓ ReLU
          Dense 32
            ↓ ReLU
          aerodynamic outputs
```

This model tests whether a conventional neural network can learn the non-linear mapping between airfoil shape, operating conditions and aerodynamic behaviour.

### 4. Histogram Gradient Boosting

Histogram Gradient Boosting builds decision trees sequentially. Each new tree is trained to reduce errors remaining after the preceding ensemble.

```text
Initial prediction
       ↓
Tree 1 → correction
       ↓
Tree 2 → correction
       ↓
  ...
       ↓
Final prediction
```

The current configuration uses 200 boosting iterations and wraps separate regressors for the multiple targets. It provides a different tree-based learning mechanism from Random Forest.

### 5. PyTorch MLP

The PyTorch MLP is a second fully connected neural network using the core `(128, 64, 32)` architecture, implemented directly in PyTorch. It uses tensor operations, Adam optimisation and mean-squared-error loss, with training controls such as early stopping/checkpointing available through the training pipeline.

It is included separately from the scikit-learn MLP because PyTorch provides greater control over the neural-network architecture and training process and can be extended more readily for larger datasets or custom architectures.

## Why compare these models?

The five models deliberately cover three major approaches:

```text
                     ML surrogate models
                            |
          +-----------------+------------------+
          |                 |                  |
        Linear          Tree ensembles     Neural networks
          |                 |                  |
        Ridge       Random Forest + HGB    MLP + PyTorch MLP
```

This allows the project to investigate:

- how important non-linearity is;
- whether tree models or neural networks generalise better;
- whether increased model complexity improves aerodynamic accuracy;
- which model predicts `CL`, `CD` and `CM` most accurately;
- accuracy versus computational cost;
- behaviour in difficult aerodynamic regimes.

## Training methodology

The training pipeline constructs the input features and applies scaling fitted only on the training partition. The same preprocessing and target definitions are used for each candidate model as far as the model implementation permits.

Rows are split **by airfoil identity**, not independently by operating point. Therefore, all operating conditions belonging to a generated airfoil remain in the same train, validation or test partition.

This prevents leakage such as training on one angle of attack for an airfoil and testing on another angle of attack for the same geometry.

## Evaluation

The evaluation pipeline compares models using metrics including:

- MAE
- RMSE
- R²

Metrics are calculated for the principal aerodynamic targets and can be analysed against angle of attack, Reynolds number, airfoil identity and other operating regimes. Parity plots and error-analysis outputs identify where each model succeeds or fails.

Run:

```bash
uv run python scripts/evaluate_models.py
```

The scientific reference for the generated training labels is XFOIL. CFD results can be used as an independent higher-fidelity comparison and should not be mixed into the ML training labels when the objective is to measure how accurately the models reproduce XFOIL.

## Legacy and alternative datasets

The repository may contain code from historical experiments involving downloaded coordinate datasets, UIUC, fixed-Re external datasets, NACA sweeps or older generation methods. These are alternative/legacy workflows and are **not the canonical training-data pipeline** described in this README.

Do not combine historical datasets with the canonical dataset unless the experiment explicitly requires it and the resulting provenance and methodology are documented.

## Development

Install dependencies and run tests with:

```bash
uv sync --extra dev
uv run pytest
```

Dataset generation requires XFOIL. On a headless Linux system, the XFOIL execution path may require `xvfb-run`.

## Reproducibility

For every reported dataset/experiment, retain:

- random seed;
- generator configuration;
- requested case count;
- XFOIL version/executable;
- worker count;
- generation manifest;
- failure log;
- repository commit.

Large generated datasets, trained model binaries and experiment outputs should normally remain outside Git unless there is a specific reason to version them.
