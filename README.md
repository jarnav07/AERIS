# AERIS

**Aerodynamic Estimation and Response Inference System**

AERIS is a machine-learning framework for surrogate modelling of aerofoil aerodynamics. The project investigates whether machine-learning models can learn the mapping from aerofoil geometry and flow conditions to aerodynamic coefficients, using a single large XFOIL-generated dataset as the reference dataset and comparing multiple models on exactly the same data.

## Research pipeline

```text
AeroSandbox built-in aerofoil database
        |
        | convert source shapes to Kulfan/CST parameters
        v
Kulfan statistical distribution
        |
        | random parent selection + convex weighting
        | covariance perturbation + random scale
        v
New synthetic aerofoil geometry
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

AeroSandbox's bundled aerofoil database is used only as the source population for learning a statistical distribution of Kulfan parameters. The source aerofoils are converted to Kulfan representations and used to calculate the mean/covariance of the geometry population. New synthetic aerofoils are then sampled from that distribution.

The generator combines randomly selected parent aerofoils using convex random weights, adds covariance-based perturbations, and applies random scaling. It then samples operating conditions and runs XFOIL. Reynolds number is sampled stochastically rather than selected from a fixed Reynolds-number list.

## Repository structure

```text
aeris/
├── src/airfoil_ml/
│   ├── training_data_generation.py  # canonical Kulfan + XFOIL generator
│   ├── training.py                  # model training orchestration
│   ├── models.py                    # model definitions/configurations
│   ├── torch_mlp.py                 # PyTorch MLP
│   ├── geometry.py                  # geometry processing
│   ├── features.py                  # feature construction/scaling
│   ├── data.py                      # validation and grouped splitting
│   ├── predict.py                   # stacking-ensemble inference interface
│   ├── airfoil_sources.py           # resolve/sample/save/plot aerofoils
│   ├── evaluation.py                # metrics and evaluation utilities
│   ├── error_analysis.py             # error/regime analysis
│   ├── drag_analysis.py              # drag-focused analysis
│   └── cli.py                       # command-line interface
├── scripts/
│   ├── generate_dataset.py           # dataset-generation entry point
│   ├── train.py                     # training entry point
│   ├── evaluate_models.py            # evaluation entry point
│   ├── fit_stacking_ensemble.py      # fits the stacking ensemble artifact
│   ├── predict.py                    # predict CL/CD/CM for one aerofoil
│   └── generate_airfoils.py          # sample and view custom aerofoils
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

1. Loads aerofoils bundled with AeroSandbox.
2. Converts the source geometries into Kulfan representations.
3. Computes the Kulfan population mean and covariance.
4. Selects multiple parent aerofoils for each synthetic case.
5. Forms a convex random weighted combination of the parents.
6. Applies a covariance-based perturbation and random geometric scaling.
7. Samples Reynolds number, angle of attack, Mach number and transition/turbulence conditions.
8. Runs XFOIL in isolated worker processes.
9. Extracts aerodynamic coefficients and boundary-layer information.
10. Writes resumable Parquet shards and records failed cases.
11. Combines successful shards into one canonical training dataset.

The current default configuration uses three parent aerofoils, a seven-point alpha grid from -15 to +15 degrees with shared random jitter, a log-normal Reynolds distribution centred on log10(Re)=5.5 with sigma 1.5, Mach 0, and up to 200 XFOIL iterations per case. These are configurable defaults.

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

The MLP is a fully connected feed-forward neural network. The hidden architecture (`ModelConfig.hidden_layers` in `models.py`) defaults to:

```text
Input → 256 → 128 → 64 → 32 → Output
```

ReLU activations introduce non-linearity. Backpropagation and the Adam optimiser adjust the network weights to minimise prediction error. This architecture — along with Random Forest's and HistGB's hyperparameters below — was chosen by a search (`scripts/tune_models.py`) rather than by hand; see [Results](#results).

```text
geometry + flow conditions
            ↓
         Dense 256
            ↓ ReLU
         Dense 128
            ↓ ReLU
          Dense 64
            ↓ ReLU
          Dense 32
            ↓ ReLU
          aerodynamic outputs
```

This model tests whether a conventional neural network can learn the non-linear mapping between aerofoil shape, operating conditions and aerodynamic behaviour.

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

The current configuration uses 250 boosting iterations and wraps separate regressors for the multiple targets. It provides a different tree-based learning mechanism from Random Forest.

### 5. PyTorch MLP

The PyTorch MLP is a second fully connected neural network sharing `ModelConfig.hidden_layers` with the scikit-learn MLP above, implemented directly in PyTorch. It uses tensor operations, Adam optimisation and mean-squared-error loss, with training controls such as early stopping/checkpointing available through the training pipeline.

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

Rows are split **by aerofoil identity**, not independently by operating point. Therefore, all operating conditions belonging to a generated aerofoil remain in the same train, validation or test partition.

This prevents leakage such as training on one angle of attack for an aerofoil and testing on another angle of attack for the same geometry.

## Evaluation

The evaluation pipeline compares models using metrics including:

- MAE
- RMSE
- R²

Metrics are calculated for the principal aerodynamic targets and can be analysed against angle of attack, Reynolds number, aerofoil identity and other operating regimes. Parity plots and error-analysis outputs identify where each model succeeds or fails.

Run:

```bash
uv run python scripts/evaluate_models.py
```

The scientific reference for the generated training labels is XFOIL. CFD results can be used as an independent higher-fidelity comparison and should not be mixed into the ML training labels when the objective is to measure how accurately the models reproduce XFOIL.

## Results

Metrics below are computed on the held-out test partition (grouped by aerofoil identity, never seen during training) of the canonical dataset's full production run: 610,652 rows across 100,279 aerofoils (`training_data_generation.py`, 106,000 requested cases, ~94.6% XFOIL success rate). "Relative error" is MAE divided by the dataset-wide mean `|actual value|` for that target (CL≈0.68, CD≈0.064, CM≈0.060). Percentage-based metrics (MAPE-style, computed per row) are noted separately where used — they are distorted for CL/CM by rows where the true value is near zero, since a small absolute error becomes a huge percentage there; CD does not have this problem, since drag is always positive.

### Baseline: all five models, full 197-target training

Each model trained multi-output against all 197 targets (CL, CD, CM, `Top_Xtr`, `Bot_Xtr`, plus 192 boundary-layer columns), default hyperparameters, no log-CD transform, no engineered geometry features:

| Model | CL rel. error | CD rel. error | CM rel. error |
|---|---|---|---|
| mlp | 10.2% | 17.6% | 18.9% |
| mlp_torch | 11.2% | 19.6% | 22.4% |
| hist_gb | 16.6% | 19.2% | 27.0% |
| random_forest | 21.2% | 21.4% | 40.7% |
| ridge | 39.0% | 66.3% | 48.9% |

### Improving accuracy

Starting from that baseline, the following were investigated in turn:

- **Log-CD transform** (`--log-cd`): trains on `log(CD)` instead of raw `CD`, so the model minimises relative rather than absolute drag error. CD's true per-row percentage error dropped from 37.0% (mean) / 20.8% (median) to 16.1% / 11.2%. (A naive MAE-based comparison initially made this look like it *hurt* CD — MAE trades off some absolute accuracy on the high-CD tail for better relative accuracy on the many low-CD rows, so it has to be judged on a relative metric to see the improvement.)
- **Derived geometry features** (`training.py::add_geometry_features`): max thickness/camber, leading-edge radius, trailing-edge angle, area, perimeter, plus thickness and camber sampled at 5 chordwise stations — computed once per aerofoil from its Kulfan coefficients (~0.5 ms/aerofoil). CM in particular depends on the whole chordwise pressure distribution, not a single max-camber scalar, so the per-station samples matter more for CM than for CL/CD.
- **Hyperparameter search** (`scripts/tune_models.py`, searched on a 12k-aerofoil subsample so each candidate fit stays fast): grew the MLP architecture to `(256, 128, 64, 32)`, Random Forest's leaf cap to `max_leaf_nodes=12000` (still memory-bounded — see the comment in `models.py`), and increased HistGB's learning rate, leaf count and iteration count.
- **Training a model dedicated to just CL/CD/CM** (`--target-cols CL CD CM`) instead of splitting capacity/loss budget across all 197 targets including the 194 boundary-layer columns — this was the single largest improvement, roughly halving CM's error.
- **Ensembling**: an equal-weight average of the best individual models (`scripts/evaluate_ensemble.py`) beat every one of them individually; a *learned* per-target stacking weight (`scripts/fit_stacking_ensemble.py`, `Ridge(positive=True)` fit on the validation split) beat the equal-weight average further, since a model that's strong on CL but weak on CM can be weighted accordingly per target instead of with one compromise weight for both.

### Best result: CL/CD/CM-dedicated stacking ensemble

`mlp` + `mlp_torch` + `hist_gb`, each retrained with `--log-cd --target-cols CL CD CM` on the tuned hyperparameters above, combined with per-target learned stacking weights:

| Target | MAE ÷ mean\|actual\| | Median per-row % error |
|---|---|---|
| CL | 6.5% | 4.4% |
| CD | 6.9% | 4.5% |
| CM | 9.9% | 8.1% |

Reproduce with:

```bash
uv run python scripts/train.py --output-dir models_dedicated \
  --only mlp mlp_torch hist_gb --log-cd --target-cols CL CD CM
uv run python scripts/fit_stacking_ensemble.py --model-dir models_dedicated \
  --models mlp mlp_torch hist_gb
```

The learned weights are saved to `<model-dir>/stacking_weights.json`, so the ensemble is a reproducible artifact rather than a one-off result. This model only predicts CL/CD/CM — use the default full-target training (no `--target-cols`) if boundary-layer predictions are also needed.

## Using the trained ensemble

Once `<model-dir>/stacking_weights.json` exists, `airfoil-ml predict` is the inference
interface: give it an aerofoil and a flow condition, get CL/CD/CM back. It never touches
the training CSV, so it is the only step that works from the model artifacts alone.

```bash
# One aerofoil, one alpha
uv run python scripts/predict.py --model-dir models_dedicated --airfoil naca2412 --re 5e5

# A full predicted polar, written to CSV and plotted
uv run python scripts/predict.py --model-dir models_dedicated \
  --airfoil naca2412 --alpha-range -5 15 1 --re 5e5 \
  --csv-out results/predictions/naca2412.csv --plot results/predictions/naca2412.png

# or: airfoil-ml predict --model-dir models_dedicated --airfoil naca2412 --alpha 0 5 10
```

Output format (coefficient values below are illustrative, not a published result):

```text
aerofoil: naca2412
geometry: max_thickness=0.1201  max_camber=0.0192  LE_radius=0.0144  TE_angle_deg=15.9514  area=0.0823
ensemble: mlp + mlp_torch + hist_gb  (from models_dedicated)
flow:     Re=5e+05  mach=0  n_crit=9  xtr_upper=1  xtr_lower=1

    alpha        CL        CD        CM  L_over_D
 -5.00000  -0.38237   0.01637  -0.03819 -23.35543
  0.00000   0.13585   0.01634  -0.03883   8.31443
  5.00000   0.68945   0.01639  -0.03942  42.07540
```

The `--airfoil` argument accepts any of:

| Form | Example |
|---|---|
| AeroSandbox database name | `--airfoil e63` |
| NACA designation | `--airfoil naca2412` |
| Coordinate file (`.dat`/`.txt`) | `--airfoil data/raw/coords/ah79k135.dat` |
| Kulfan `.json` from `generate-airfoils` | `--airfoil data/generated/custom_airfoils/sampled_0000.json` |
| A fresh sample from the dataset generator | `--random --seed 7` |

Other useful flags: `--n-crit`/`--xtr-upper`/`--xtr-lower` to set the transition model
(defaults: `9`, free, free — matching XFOIL's standard criterion and the 80% of generated
cases that used free transition), `--per-model` to see each component model's own
prediction alongside the ensemble, `--json` for machine-readable output on stdout, and
`--show` to open the predicted polar and the aerofoil section in the platform viewer.

Predictions outside the sampled operating envelope (Re outside ~1.8e3–5.6e7, |alpha| > 25
degrees, or mach > 0) are extrapolation and are reported as `warning:` lines.

From Python:

```python
from airfoil_ml.predict import StackingEnsemblePredictor

predictor = StackingEnsemblePredictor.load("models_dedicated")
table = predictor.predict("naca2412", alpha=[0, 5, 10], Re=5e5)   # DataFrame: CL, CD, CM, L_over_D
```

## Generating and viewing custom aerofoils

`airfoil-ml generate-airfoils` draws aerofoils from the same sampler
`training_data_generation.py` uses to build the dataset (convex combination of three
database parents plus a covariance-based perturbation), so generated shapes are
in-distribution for the surrogate. Unlike `scripts/generate_dataset.py` it only draws
*shapes* and never calls XFOIL, so it runs anywhere:

```bash
uv run python scripts/generate_airfoils.py --count 6 --seed 7 \
  --output-dir data/generated/custom_airfoils --show
```

Each aerofoil is written as both a Kulfan `.json` (round-trips the 18 coefficients
exactly — this is the form to feed back to `predict`) and a coordinate `.dat`, alongside a
PNG grid of the sections. `--show` opens that PNG in the platform image viewer.

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
