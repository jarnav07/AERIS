# ML Aerofoil Predictor

Machine-learning surrogate modelling for airfoil aerodynamics. The project investigates whether machine-learning models can learn the mapping from airfoil geometry and flow conditions to aerodynamic coefficients, using XFOIL-generated data as the reference dataset and comparing multiple models on exactly the same data.

## Research pipeline

The canonical workflow is:

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

**The canonical dataset is generated once and shared by all ML models.** Models must not receive separate or differently sampled training datasets.

## Important data-source distinction

The current canonical generator is **not** based on the UIUC coordinate database and does **not** train on the fixed-Re Kanakaero dataset.

AeroSandbox's bundled airfoil database is used as a source of existing geometries from which the generator constructs a statistical distribution of Kulfan parameters. Those source airfoils are not simply copied into the training set. New airfoils are sampled from their Kulfan distribution.

The generator samples Reynolds number stochastically rather than selecting from a fixed Reynolds-number list. Angle of attack is sampled as a jittered sweep, while transition and turbulence parameters are also sampled according to the generation configuration.

## Repository structure

```text
ml-aerofoil-predictor/
├── src/airfoil_ml/
│   ├── training_data_generation.py  # canonical Kulfan + XFOIL generator
│   ├── training.py                  # training orchestration and model bundles
│   ├── models.py                    # model configurations/baselines
│   ├── torch_mlp.py                 # PyTorch MLP implementation
│   ├── geometry.py                  # geometry processing
│   ├── features.py                  # feature construction and scaling
│   ├── data.py                      # dataset validation and grouped splitting
│   ├── evaluation.py                # regression metrics and plots
│   ├── error_analysis.py             # error/regime analysis
│   ├── drag_analysis.py              # drag-focused analysis
│   └── cli.py                       # airfoil-ml command-line interface
├── scripts/
│   ├── generate_dataset.py           # dataset-generation entry point
│   ├── train.py                     # training entry point
│   └── evaluate_models.py            # evaluation entry point
├── data/
│   ├── raw/                          # optional/source data; not canonical training data
│   ├── generated/                    # canonical generated training data
│   └── processed/                    # downstream processed datasets
├── models/                           # trained model artifacts
├── results/                          # metrics, plots and experiment outputs
└── tests/                            # automated tests
```

## Dataset generation

The main implementation is `src/airfoil_ml/training_data_generation.py`; `scripts/generate_dataset.py` is the user-facing entry point.

The generator:

1. Loads the airfoils bundled with AeroSandbox.
2. Normalises them and converts them to Kulfan representations.
3. Computes the mean and covariance of the Kulfan parameter population.
4. Selects multiple parent airfoils for each synthetic sample.
5. Creates a convex random weighted combination of the parents.
6. Applies a covariance-based perturbation and random geometric scaling.
7. Samples an operating-point sweep, including Reynolds number, angle of attack, transition parameters and turbulence criterion.
8. Runs XFOIL in isolated worker processes.
9. Extracts aerodynamic coefficients and boundary-layer information.
10. Writes resumable Parquet shards and records failed cases.
11. Combines successful shards into the single training dataset.

The default generator configuration currently uses three parent airfoils, a seven-point alpha grid from -15 to +15 degrees with shared random jitter, a log-normal Reynolds distribution centred on log10(Re)=5.5 with sigma 1.5, Mach 0, and up to 200 XFOIL iterations per case. These are configuration defaults, not restrictions on the research design.

### Generate data

From the repository root:

```bash
uv run python scripts/generate_dataset.py --cases 30000 --workers 8 --resume
```

The package CLI provides the equivalent interface:

```bash
airfoil-ml generate-training-data --cases 30000 --workers 8 --resume
```

Generation is reproducible for a fixed random seed. It supports multiple XFOIL workers, resuming from completed shards, failure logging, and provenance information.

## Dataset contents

Each generated row contains the Kulfan geometry parameters, flow/transition conditions, aerodynamic outputs and sampled boundary-layer quantities.

The principal aerodynamic targets are:

- `CL` — lift coefficient
- `CD` — drag coefficient
- `CM` — pitching-moment coefficient

The dataset also contains transition information, an analysis-confidence field and boundary-layer profiles for the upper and lower surfaces.

## Training

All candidate ML models use the same generated dataset. The training pipeline constructs geometry and flow features, applies scaling fitted only to the training partition, and evaluates on held-out airfoil identities.

Run:

```bash
uv run python scripts/train.py
```

or:

```bash
airfoil-ml train
```

The repository includes linear/tree baselines and neural-network models, including a CPU-oriented PyTorch MLP. Model implementations are deliberately kept behind a common training interface so their performance can be compared fairly.

## Train/validation/test methodology

Rows are split **by airfoil identity**, not randomly by individual aerodynamic operating point. Therefore, a generated airfoil and all of its operating conditions remain in the same partition.

This prevents leakage such as training on one angle of attack for an airfoil and testing on another angle of attack for the same geometry.

Input and target scalers are fitted only on the training partition.

## Evaluation

Run:

```bash
uv run python scripts/evaluate_models.py
```

The evaluation pipeline reports MAE, RMSE and R² for `CL`, `CD` and `CM`, and can generate parity plots and error analyses against angle of attack, Reynolds number and airfoil identity.

The research objective is to compare model accuracy using identical training data and held-out geometries, with XFOIL as the reference aerodynamic solver. CFD data can subsequently be used as an independent higher-fidelity comparison rather than being mixed into the ML training labels.

## Legacy/alternative datasets

The repository may contain code or documentation for historical experiments involving downloaded coordinate datasets, fixed-Re external datasets or older parametric generators. Those experiments are not part of the canonical stochastic Kulfan/XFOIL training workflow described above and should not be combined with the canonical dataset without an explicit experimental reason.

## Development

Install dependencies and run the test suite with:

```bash
uv sync --extra dev
uv run pytest
```

Dataset generation requires XFOIL. On a headless Linux machine, `xvfb-run` may also be required by the XFOIL execution path.

## Reproducibility

Record the random seed, generator configuration, number of requested cases, worker count, XFOIL configuration and generation manifest for every dataset used in a reported experiment. Generated datasets, trained models and result plots should normally remain outside Git unless there is a specific reason to version them.
