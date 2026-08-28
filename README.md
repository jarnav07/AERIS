# ML Airfoil Predictor

A reproducible machine-learning surrogate for 2D airfoil aerodynamics.

The experiment is intentionally simple:

```text
random 3-airfoil Kulfan mixture
            |
            v
     XFOIL aerodynamic data
            |
            v
     one shared dataset
            |
     +------+------+------+
     |      |      |      |
   Ridge Extra  HistGB  MLP
     |      |      |      |
     +------+------+------+
            |
            v
 compare against held-out XFOIL
            |
            v
 compare independently against CFD
```

The important design rule is that **there is one training dataset**. Every
candidate model is trained on exactly the same rows, with exactly the same
feature preprocessing and the same geometry-level train/validation/test split.
CFD data is never mixed into training; it is an independent evaluation
reference.

## Research objective

The project asks how accurately several lightweight ML models can reproduce
XFOIL's predictions for previously unseen airfoil geometries, and how well the
same trained models agree with an independent CFD reference.

The target outputs are:

- `Cl` — lift coefficient
- `Cd` — drag coefficient
- `Cm` — pitching-moment coefficient

The input is an 18-parameter Kulfan (CST) representation plus:

- angle of attack, `alpha_deg`
- `log10(Reynolds)`
- Mach number

`Cd` is optionally trained in logarithmic space. This prevents the very small
magnitude of drag from being overwhelmed by the much larger lift target and
also makes the loss more closely related to relative drag accuracy.

## Why the dataset is generated this way

The geometry generator follows the core stochastic idea used by NeuralFoil.
Three airfoils are selected from an airfoil database. Two random cut points are
drawn to create three non-negative weights that sum to one, and the parent
Kulfan parameters are linearly mixed. The result is then multiplied by a
lognormal scale factor and can receive a covariance-based perturbation derived
from the source geometry database.

This produces many geometries that are not simply copies of the source
families, while keeping the representation entirely in a common 18-parameter
Kulfan space. The NeuralFoil source generator uses the same three-parent
mixture, lognormal scale, and covariance-perturbation strategy.

For this project, XFOIL's transition settings are fixed for the canonical
experiment (`Ncrit=9`, forced transition at the trailing edge). Fixing those
settings is deliberate: a later CFD comparison then has an unambiguous model
input of geometry + alpha + Reynolds + Mach rather than hidden transition-model
variables that CFD does not necessarily share.

The generator samples Reynolds number log-normally around `10^5.5` with a
standard deviation of 1.5 decades. Each generated geometry is analysed at a
jittered seven-point alpha pattern based on `[-15, -10, -5, 0, 5, 10, 15]`.
XFOIL failures are skipped and written to a JSONL failure log rather than being
filled with synthetic labels.

## Canonical dataset

The generated CSV contains one row per successful XFOIL operating point:

```text
kulfan_upper_0 ... kulfan_upper_7
kulfan_lower_0 ... kulfan_lower_7
kulfan_LE_weight
kulfan_TE_thickness
alpha_deg
reynolds
mach
cl
cd
cm
airfoil_id
case_id
```

The 18 Kulfan values are the actual model geometry inputs. The generated `.dat`
files are kept alongside the CSV so individual geometries can be inspected or
re-analysed independently.

The repository does **not** commit the generated dataset. Dataset files and
model artefacts are ignored by Git so large experiments can be run locally,
on Deepnote, or on another compute environment without bloating the source
repository.

## Model comparison

The default comparison set is deliberately small but diverse:

| Model | Purpose |
|---|---|
| Ridge | linear baseline |
| Extra Trees | nonlinear randomized tree ensemble |
| HistGradientBoosting | nonlinear boosted-tree baseline |
| PyTorch MLP | learned dense neural-network surrogate |

All models predict the same three targets and are evaluated on exactly the same
held-out airfoils. The PyTorch model is optional; when PyTorch is unavailable,
the other three remain usable.

The goal is not to pick a winner from one metric. The evaluation reports MAE,
RMSE, R², and a **defined MAPE** for each target. MAPE excludes values too close
to zero because percentage error at a zero crossing (especially `Cl` and `Cm`)
is not physically informative. Absolute errors and R² are still reported for
all rows.

## XFOIL versus CFD

XFOIL is the training reference and the first evaluation reference. Because the
test split is performed by `airfoil_id`, the XFOIL test score measures
interpolation in operating conditions combined with generalisation to an
unseen geometry.

CFD evaluation is deliberately separate. Supply a CSV containing the same 18
Kulfan parameters, alpha, Reynolds, Mach, and CFD `cl`, `cd`, `cm`. The models
predict that reference without ever seeing those CFD labels during training.
This gives two useful comparisons:

1. **ML → XFOIL:** how closely the surrogate reproduces the source solver.
2. **ML → CFD:** how the XFOIL-trained surrogate compares with a higher-fidelity
   aerodynamic reference.

The second comparison must be interpreted carefully: any systematic difference
between XFOIL and CFD is inherited by a model trained on XFOIL. A low ML→XFOIL
error therefore does not, by itself, establish CFD-level physical accuracy.

## Project structure

```text
.
├── data/
│   ├── README.md
│   └── raw/                         # generated data; ignored by Git
├── models/                          # saved models; ignored by Git
├── results/                         # metrics/plots; ignored by Git
├── src/airfoil_ml/
│   ├── cli.py                       # four user-facing commands
│   ├── data.py                      # canonical schema + grouped split
│   ├── evaluation.py                # XFOIL/CFD metrics + plots
│   ├── features.py                  # feature construction + scaling
│   ├── geometry.py                  # coordinate -> Kulfan inference helper
│   ├── models.py                    # model definitions
│   ├── torch_mlp.py                 # optional CPU-friendly MLP
│   ├── training.py                  # training + artefact persistence
│   └── training_data_generation.py  # canonical stochastic XFOIL generator
├── tests/
├── pyproject.toml
└── README.md
```

## Installation

Python 3.10+ is required.

Using `uv`:

```bash
uv sync --extra dev
```

Install XFOIL separately and make the executable available as `xfoil`. On a
headless Linux machine, the generator automatically uses `xvfb-run` when it is
available.

For the optional neural-network candidate:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Run tests with:

```bash
uv run pytest
```

## 1. Generate the one large XFOIL dataset

A small smoke test:

```bash
uv run airfoil-ml generate-data \
  --cases 100 \
  --seed 42 \
  --training-output data/raw/xfoil_training.csv \
  --coordinates-output data/raw/generated_airfoils
```

For a serious training run, increase the number of cases substantially. For
example:

```bash
uv run airfoil-ml generate-data \
  --cases 10000 \
  --seed 42 \
  --training-output data/raw/xfoil_training.csv \
  --coordinates-output data/raw/generated_airfoils
```

One generated case normally contributes several successful alpha rows, so the
final dataset will be considerably larger than the requested number of
geometries.

The generator writes:

```text
data/raw/xfoil_training.csv
 data/raw/generated_airfoils/*.dat
 data/raw/xfoil_training.provenance.json
 data/raw/xfoil_training.failures.jsonl
```

The provenance file records the seed and complete sampling/XFOIL configuration.
The failure log records cases where geometry generation or XFOIL did not
produce usable labels.

## 2. Train every model on the same dataset

```bash
uv run airfoil-ml train \
  --dataset data/raw/xfoil_training.csv \
  --output models/main \
  --seed 42 \
  --mlp-epochs 250
```

The training command:

1. validates the dataset;
2. creates one grouped train/validation/test split by `airfoil_id`;
3. fits input and target scalers on **training rows only**;
4. trains every selected model using the same transformed data;
5. saves every model and the common preprocessing bundle;
6. saves the split manifest and machine-readable metrics.

The default split is approximately 60% train, 20% validation, and 20% test by
unique geometry identity.

To run only selected candidates during an experiment:

```bash
uv run airfoil-ml train ... --only ridge extra_trees
```

The default run still trains all available candidates.

## 3. Evaluate against held-out XFOIL and CFD

XFOIL-only evaluation:

```bash
uv run airfoil-ml evaluate \
  --dataset data/raw/xfoil_training.csv \
  --model-dir models/main \
  --output results/main/xfoil
```

Combined XFOIL + CFD evaluation:

```bash
uv run airfoil-ml evaluate \
  --dataset data/raw/xfoil_training.csv \
  --model-dir models/main \
  --output results/main \
  --cfd-csv path/to/cfd_reference.csv
```

The result directory contains metrics and parity plots for each reference.
The model comparison plot makes it easy to compare the candidates on the same
reference dataset.

### CFD CSV format

The CFD file should use the same geometry and operating-point convention as
the training CSV, with CFD values in the target columns:

```text
kulfan_upper_0 ... kulfan_upper_7
kulfan_lower_0 ... kulfan_lower_7
kulfan_LE_weight
kulfan_TE_thickness
alpha_deg
reynolds
mach
cl
cd
cm
```

An `airfoil_id` column is optional for CFD evaluation; it is only used for
labelling/group counts in plots and metrics.

## 4. Predict a single case

```bash
uv run airfoil-ml predict \
  --model-dir models/main \
  --model mlp_torch \
  --coordinates path/to/airfoil.dat \
  --alpha 5 \
  --reynolds 500000
```

The coordinate file is converted to a normalised Kulfan representation before
prediction.

## Experimental rules

For a defensible comparison, keep these rules for every main experiment:

- Generate one canonical XFOIL dataset and do not mix unrelated external
  datasets into it.
- Keep the test geometries fixed when comparing model changes.
- Never fit preprocessing on validation, test, or CFD rows.
- Do not use CFD labels during model selection if CFD is intended to remain a
  genuinely independent benchmark.
- Report both absolute errors and percentage-based errors, especially for drag.
- Treat XFOIL as the source of the training labels, not as experimental ground
  truth.

## Relation to NeuralFoil

The data generator was adapted from the public NeuralFoil training-data
strategy described in its repository. NeuralFoil reports nearly 8 million
XFOIL runs and uses an 18-parameter Kulfan geometry representation; its public
generator likewise mixes three parent airfoils and applies stochastic shape
perturbations. This project uses that idea as a reproducible starting point,
then focuses on comparing simpler surrogate architectures on one shared
XFOIL dataset.

NeuralFoil:
https://github.com/peterdsharpe/NeuralFoil

## Status

This repository is an experimental research codebase. Accuracy numbers should
only be quoted from a concrete dataset/model run with its provenance and split
manifest, rather than from the source tree itself.
