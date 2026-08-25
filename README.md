# ML Airfoil Predictor

An initial research-grade foundation for an aerodynamic surrogate model:

```text
airfoil coordinates + (alpha, Reynolds, Mach)
                -> ML regression model
                -> (Cl, Cd, Cm) and derived L/D
```

The project is deliberately built around traceable reference data. It does not
ship fabricated aerodynamic coefficients and it does not claim model accuracy
until the acquisition and training workflow has been run.

## Why this project exists

A full aerodynamic analysis is valuable but relatively expensive when it must
be repeated inside a design search. A surrogate model can provide rapid
estimates after training, but only if its validation respects the distinction
between interpolation over operating conditions and generalisation to a new
shape. This project therefore treats the problem as both an aerodynamic data
problem and a leakage-sensitive multi-output regression problem.

The first version is intended to be a clean experimental baseline, not a claim
that XFOIL or a small MLP replaces CFD or experiment.

## Methodology decisions

### Reference data

1. **Small multi-condition reference:** geometry is downloaded from the [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html), and aerodynamic labels are generated locally by [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/), an established inviscid-panel plus integral boundary-layer solver.
2. **Large fixed-Re experiment:** the [Kanakaero airfoil dataset](https://github.com/kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils), associated with [10.3390/data9050064](https://doi.org/10.3390/data9050064), supplies 33,705 rows from 2,946 airfoils with eight CST shape coefficients, angle of attack, `Cl`, and `Cd` at documented `Re=100,000`.

The two sources are deliberately kept as separate experiments. The large
source has no `Cm` or Reynolds sweep, so it must not be concatenated with the
multi-Reynolds XFOIL table as if the labels were interchangeable.

This is a practical first dataset source because geometry and polars are
connected by a reproducible command. XFOIL is not high-fidelity CFD: transition
and separation predictions are model-dependent, and convergence can fail near
stall. Treat the resulting data as a consistent reference/surrogate benchmark,
not ground truth.

The repository does not commit downloaded data. This keeps provenance explicit,
avoids silently redistributing a large external dataset, and makes each run
reproducible from a fresh checkout.

### Geometry representation

Each coordinate file is:

- parsed while ignoring header text and malformed rows;
- de-duplicated at consecutive repeated points;
- translated and scaled to a unit chord;
- split into upper and lower surfaces for Selig or Lednicer traversal order;
- interpolated onto the same 100-point cosine-spaced chord grid.

The model input is the concatenation of 100 upper-surface ordinates and 100
lower-surface ordinates. The common x-grid is not learned as a variable. Cosine
spacing places more points near the leading edge, where curvature and pressure
gradients change quickly. This representation is more information-preserving
than a few hand-selected thickness/camber values while remaining interpretable
and easy to reuse for unseen coordinates. PCA can be added later if the
200-dimensional shape vector proves unnecessarily redundant.

### Flow and target variables

The input is:

- `alpha_deg`: angle of attack in degrees;
- `log10(reynolds)`: logarithmically transformed Reynolds number;
- `mach`: freestream Mach number;
- the 200 geometry ordinates described above.

The targets are the XFOIL polar's:

- `cl`: lift coefficient;
- `cd`: drag coefficient;
- `cm`: pitching moment coefficient.

`L/D = Cl/Cd` is calculated for plotting and analysis rather than trained as a
fourth target.

### Leakage-safe split

The split is performed on unique `airfoil_id` values, not individual polar
rows. Every angle/Re case for an airfoil belongs to exactly one of train,
validation, or test. The default is approximately 60/20/20 percent of the
available airfoil identities, with a fixed seed of 42. The selected identities
are written to `split_manifest.json`.

This answers a meaningful first question: can the model predict a completely
unseen geometry at known ranges of flow conditions? A random row split would
usually be easier and would overstate that capability because the same shape
would appear in both fitting and evaluation.

### Candidate models

- **Ridge regression:** a regularised linear reference that tests whether the
  relationship is approximately linear in the chosen features.
- **Random forest:** a nonlinear, nonparametric baseline that can capture
  interactions but may interpolate poorly outside the training geometry
  manifold.
- **MLPRegressor:** the sklearn baseline, with hidden layers `(128, 64, 32)`,
  ReLU activations, Adam optimisation, standardised inputs and targets,
  mini-batches of 64, and internal early stopping.
- **HistGradientBoostingRegressor** (wrapped in `MultiOutputRegressor`): a fast
  histogram-based gradient boosting baseline that trains in seconds even on
  hundreds of geometry features and is competitive with or better than the
  random forest on this tabular problem.
- **TorchMLPRegressor** (`mlp_torch`): the same MLP role implemented in
  PyTorch. The sklearn MLP converges very slowly on CPU for 200-dimensional
  geometry inputs (roughly 5 s/epoch here, so 100+ epochs is impractical in a
  bounded job), whereas the vectorised PyTorch version trains the same network
  to convergence in tens of seconds. It has a scikit-learn-compatible
  `fit`/`predict` API, Adam with early stopping on an internal validation
  split, and safe `joblib` persistence. If PyTorch is not installed, this
  candidate is skipped gracefully.

All five are multi-output regressors. The MLP models are intentionally modest:
the first goal is a trustworthy end-to-end baseline, not architectural novelty.
Models are saved with `joblib`; loss and validation histories are saved as
JSON when available.

## Project structure

```text
.
├── data/
│   ├── README.md              # provenance and data policy
│   └── raw/                   # downloaded coordinates/polars (gitignored)
├── models/                    # saved estimators and preprocessing (gitignored)
├── results/                   # plots and analysis output (gitignored)
├── scripts/
│   ├── acquire_data.py       # CLI wrapper
│   └── train.py              # CLI wrapper
├── src/airfoil_ml/
│   ├── acquisition.py        # UIUC download and XFOIL execution/parsing
│   ├── cli.py                # acquire, train, evaluate, predict, analyze-drag/error
│   ├── data.py               # schema validation and grouped splitting
│   ├── drag_analysis.py      # drag-focused diagnostics (regime/Re/L-D)
│   ├── error_analysis.py     # % error vs XFOIL and wall-clock benchmark
│   ├── evaluation.py         # metrics, parity, polar, and error plots
│   ├── features.py           # feature matrix and train-only scalers
│   ├── geometry.py           # coordinate canonicalisation/resampling
│   ├── geometry_generation.py # parametric camber/thickness airfoil generator
│   ├── large_dataset.py      # fixed-Re CST dataset adapter and training
│   ├── large_evaluation.py   # fixed-Re CST experiment plots
│   ├── models.py             # Ridge, random forest, HistGB, MLP candidates
│   ├── multi_re_batch.py     # sharded, resumable multi-Re XFOIL generation
│   ├── torch_mlp.py          # fast converged PyTorch MLP (optional dep)
│   └── training.py           # fitting, checkpoint artifacts, and persistence
├── tests/                    # unit tests for geometry, data, features, models
├── pyproject.toml            # dependencies and package metadata
└── README.md
```

## Setup

Python 3.10+ is required. The project uses `uv` in the examples, but the
package metadata is standard setuptools configuration.

```bash
uv sync --extra dev
```

Install XFOIL separately and make the executable available as `xfoil` on
`PATH`. In headless Linux environments, the acquisition code automatically
uses `xvfb-run` when available. No XFOIL executable or downloaded dataset is
bundled in this repo.

The PyTorch MLP candidate (`mlp_torch`) is optional: if `torch` is not
importable, `make_models` skips it and the remaining candidates train
normally. To enable it on a CPU machine, install the CPU wheel into the
project environment (the default PyPI wheel bundles CUDA and is much larger):

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Run the tests:

```bash
uv run pytest
```

The tests use small in-memory fixtures only to verify software behavior; those
fixtures are not aerodynamic training data and are never reported as results.

## Reproduce the first dataset

Use a deliberately mixed set of geometries rather than only one NACA family.
For example, the following downloads the named UIUC coordinate files and runs
XFOIL at two Reynolds numbers over a polar sweep:

```bash
uv run airfoil-ml acquire \
  --airfoils naca0015 naca2412 naca4412 e387 s1223 \
  --coordinates data/raw/coordinates \
  --polar-csv data/raw/polars.csv \
  --reynolds 500000 1000000 \
  --alpha-start -2 --alpha-end 15 --alpha-step 1 \
  --xfoil-timeout 60
```

The generated polar table has the schema:

```text
airfoil_id, alpha_deg, reynolds, mach, cl, cd, cm, cdp
```

The extra `cdp` column is retained as XFOIL's pressure-drag diagnostic; the
training pipeline uses `cl`, `cd`, and `cm`.

XFOIL may fail, time out, or omit points near separated flow. The generator
keeps successful cases, records failed airfoil/Re combinations in
`polars.provenance.json`, and never fills labels with invented values. Inspect
the solver output and provenance before training; a later version should add
explicit convergence flags and a policy for censored stall points.

## Assessment of the alternate XFOIL generator

The [alternate MIT/XFOIL dataset repository](https://github.com/MdSyamul/Dataset-for-Airfoil-Aerodynamics-in-Incompressible-Flow-Regime)
contains an MIT-licensed script plus `Cord.rar` (about 18 MB) and `MPT.rar`
(about 13 MB). It is a useful lead for scaling, but the script is not ready to
run unchanged in this project:

- it hardcodes a Windows XFOIL path;
- its default airfoil list contains only `NACA 4412`;
- its default Reynolds sweep is a single value (`Re=30,000`);
- it assumes NACA-style names when deriving `M`, `P`, and `T`, which is not a
  valid geometry representation for arbitrary airfoils;
- it does not provide the same provenance, canonical geometry validation,
  headless XFOIL handling, or failed-case manifest as this project.

Its intended output schema is valuable—Mach, Reynolds number, angle of attack,
`Cl`, `Cd`, `Cm`, and `Cdp`—but the robust path is to reuse this repository's
XFOIL runner after extracting and enumerating its coordinate archive. That
runner already handles canonicalization, Xvfb, timeouts, stale-file cleanup,
and parseable-case validation. A large run should be sharded by airfoil and
flow condition, with each shard producing logs and convergence metadata, rather
than launched as one unbounded process.

## Train the large fixed-Re dataset

The large public CSV can be downloaded reproducibly using its pinned dataset
commit:

```bash
uv run airfoil-ml download-large
uv run airfoil-ml train-large \
  --csv data/raw/external/kanakaero/compiled_airfoil_data.csv \
  --output models/large_fixed_re \
  --seed 42
uv run airfoil-ml evaluate-large \
  --csv data/raw/external/kanakaero/compiled_airfoil_data.csv \
  --model-dir models/large_fixed_re \
  --model mlp \
  --output results/large_fixed_re
```

This experiment uses the eight CST coefficients plus angle of attack as input
and predicts `Cl` and `Cd`. `Cd` is trained in logarithmic space and converted
back to physical units for metrics and plots. The split is grouped by airfoil
filename; the current run holds out 589 complete airfoil identities for testing.
Reynolds number is metadata fixed at 100,000 and is not learned as a varying
feature.

## Large multi-Reynolds XFOIL experiment (100 airfoils)

The fixed-Re public dataset has no `Cm` and no Reynolds sweep, so it cannot
answer questions about Reynolds-number dependence or complete polar prediction.
To fix that, the XFOIL runner was scaled up: `airfoil-ml batch-generate` ran
100 geometries from the Kanakaero processed-coordinate archive (a diverse
sample spanning NACA, Eppler, Wortmann, AG, AH, Boeing, and other families)
through XFOIL at six Reynolds numbers (30k, 50k, 100k, 250k, 500k, 1M) over
`α = -2° … 10°`, producing `data/raw/multi_re_100/combined.csv`: 6,294 rows
from 569 successful (airfoil, Re) cases, each row with `Cl`, `Cd`, `Cm`, `Cdp`,
and a convergence flag. Failures are recorded in `_failed_cases.jsonl` and are
never retried or filled with synthetic values; the run is resumable by shard.

The grouped split holds out 20 complete airfoil identities for testing (1,310
rows). Test results on completely unseen geometries:

```text
model            cl_MAE   cl_R2   cd_MAE   cd_R2   cm_MAE   cm_R2
ridge            0.1404  0.8111  0.01108  0.6189  0.01527  0.5404
random_forest    0.0595  0.9547  0.00385  0.9076  0.00903  0.8110
hist_gb          0.0584  0.9569  0.00298  0.9479  0.01121  0.7547
mlp_torch        0.0545  0.9582  0.00382  0.8964  0.00786  0.8379
```

(Full metrics, training history, and per-model test predictions live in
`models/multi_re_100/`; plots in `results/multi_re_100_*`.)

Reproduce the training with two budgeted commands (the torch MLP converges in
about two minutes; the other three fit in well under one):

```bash
uv run airfoil-ml train \
  --polar-csv data/raw/multi_re_100/combined.csv \
  --coordinates data/raw/external/kanakaero/processed_coordinates \
  --output models/multi_re_100 --seed 42 \
  --only ridge random_forest hist_gb

uv run airfoil-ml train \
  --polar-csv data/raw/multi_re_100/combined.csv \
  --coordinates data/raw/external/kanakaero/processed_coordinates \
  --output models/multi_re_100 --seed 42 \
  --mlp-max-iter 200 --only mlp_torch
```

The `--only` flag trains a subset of models and merges results into the
existing `metrics.json`, so slow candidates can be added in their own
budgeted run without retraining fast ones. `--mlp-hidden` and `--mlp-max-iter`
control the MLP architecture and epoch budget. The sklearn `mlp` candidate is
deliberately excluded from the large multi-Re run: at ~5 s/epoch on 203
features it needs tens of minutes to converge, which is exactly the problem
`mlp_torch` solves. Evaluate and analyze with:

```bash
uv run airfoil-ml evaluate \
  --polar-csv data/raw/multi_re_100/combined.csv \
  --coordinates data/raw/external/kanakaero/processed_coordinates \
  --model-dir models/multi_re_100 --model hist_gb \
  --output results/multi_re_100_hist_gb

uv run airfoil-ml analyze-drag \
  --polar-csv data/raw/multi_re_100/combined.csv \
  --coordinates data/raw/external/kanakaero/processed_coordinates \
  --model-dir models/multi_re_100 --output results/multi_re_100_drag
```

### What the drag analysis shows

Drag is the hardest coefficient, and the error is not uniform across the
flight envelope. From `results/multi_re_100_drag/drag_summary.json`:

- **Error concentrates near stall.** For `hist_gb`, mean `|ΔCd|` is 0.0017 in
  the attached-flow regime (`0° ≤ α < 4°`) but 0.0063 near stall
  (`8° ≤ α < 11°`) — roughly 3.7× worse. Every model shows the same pattern,
  which is physically expected: separation onset is abrupt, XFOIL convergence
  degrades there, and stall angle varies sharply with geometry and Reynolds
  number.
- **Low Reynolds numbers are the hardest drag regime.** Mean `|ΔCd|` for
  `hist_gb` is 0.0055 at Re = 100k but drops to 0.0014–0.0020 at Re ≥ 250k.
  Low-Re laminar separation bubbles make Cd especially sensitive to
  transition location, so this is the regime where a surrogate needs the most
  training coverage (and where XFOIL itself is least certain).
- **L/D is sensitive to Cd.** A small absolute Cd error at low drag produces a
  large L/D error: `hist_gb` L/D MAE is about 5.8 on the held-out polars
  (relative error is inflated by the near-zero-lift crossing at negative α and
  by the stall region, where L/D is small by definition). A surrogate that
  nails Cl but not Cd is therefore not yet a reliable design tool; this is the
  strongest argument for the log-Cd training used in the fixed-Re experiment
  and for denser low-Re/stall sampling.
- **Model ranking is coefficient-dependent.** `mlp_torch` is best for Cl and
  Cm, `hist_gb` is best for Cd, and `ridge` is only competitive for the
  most linear part of the lift curve. No single candidate dominates, so the
  research question "which model for which coefficient" is an active one.

The drag diagnostic is reproducible with `airfoil-ml analyze-drag`, which
writes `drag_summary.json` plus `drag_error_by_condition.png` (Cd and L/D
error versus α and Re across models) and `drag_parity_all_models.png`.

## Accuracy versus XFOIL, and wall-clock time saved

`airfoil-ml analyze-error` reports the surrogate's error as a **percentage of
the quantity being predicted** and measures the wall-clock advantage over
running XFOIL on the same machine. Results below are from
`results/multi_re_100_error/`.

**Reference honesty first:** the ground truth for every number here is XFOIL
polar output on 20 airfoils that were completely held out of training (1,310
rows). XFOIL is a low-order viscous/inviscid panel method, not Navier-Stokes
CFD or wind-tunnel data. A surrogate cannot beat its training reference:
XFOIL's own bias versus experiment/RANS is small for Cl in attached flow but
reaches several to tens of percent for Cd at low Reynolds numbers and near
stall, and that bias is inherited by the model. Treat these as
"surrogate-vs-XFOIL" errors, not absolute physical accuracy.

Percentage error (MAPE) on the held-out airfoils, with Cd also in drag counts
(1 count = 0.0001):

```text
model            cl MAPE   cd MAPE   cm MAPE   L/D MAE
hist_gb          23.5%     11.1%     46.3%     5.8
mlp_torch        24.5%     13.8%     33.4%     6.7
random_forest    24.3%     14.1%     37.8%     6.2
ridge            62.5%     58.1%     67.5%     33*   (*123 negative-Cd rows excluded)
```

Read the table carefully, because MAPE punishes zero crossings: Cl crosses
zero near α = 0° and Cm is a small quantity, so their raw MAPE is inflated by
near-zero denominators. Excluding rows where |reference| is below the
reporting floor, the best models sit at **Cl ≈ 15%**, **Cd ≈ 11%**
(≈ 30 drag counts), and Cm ≈ 33%.

- **Cd error is roughly constant in relative terms across the envelope
  (10-14%) but grows in absolute terms near stall:** ~17 drag counts in
  attached flow vs ~63 near stall for `hist_gb`. That is the same stall-region
  difficulty seen in the drag analysis, expressed in counts.
- **Low Re costs drag counts, high Re costs relative accuracy:** mean Cd error
  is 33-55 counts at Re ≤ 100k but 14-20 counts at Re ≥ 250k; because Cd
  itself is smaller at high Re, the relative error there (10-17%) is not
  correspondingly better.
- **Physics violations are surfaced, not averaged away:** `ridge` predicts
  negative Cd on 123 of 1,310 test rows (9.4%). Those rows are excluded from
  L/D statistics (they have no physical L/D), leaving ridge at L/D MAE 33;
  the nonlinear models never predict negative drag. This is a concrete
  argument for the constrained/log-Cd training already planned.

### Can every parameter reach 10% of XFOIL?

Short answer: **Cd yes, Cl close, Cm and L/D no — not with the current
100-airfoil dataset.** Full numbers in
`results/multi_re_100_error/target_10_percent.json`. One framing caveat
first: Cl, Cm, and L/D all cross zero somewhere in the polar, where a
"10% error" is undefined (10% of zero is zero). The meaningful target is
10% MAPE on the rows where each quantity is well defined
(|reference| ≥ 2× the reporting floor); the numbers below use that definition.

```text
model (defined MAPE)                       cl     cd     cm     ld
hist_gb baseline                           15.2%  10.2%  25.7%  28.1%
mlp_torch baseline                         14.7%  11.9%  17.0%  34.8%
random_forest log-Cd                       16.9%  11.7%  19.7%  30.6%
ensemble hist_gb + mlp_torch                13.5%   9.8%  18.2%  27.9%
ensemble 3-way (hist_gb, mlp_torch, rf)     13.7%   9.9%  16.5%  27.1%
```

- **Cd is already there.** The two-model ensemble sits at **9.8%**; `hist_gb`
  alone is 10.2%. Training Cd in log space (`airfoil-ml train --log-cd`,
  now wired into the multi-Re pipeline) shifts the fitted loss toward
  relative error and buys a further ~1-2 points for the tree models.
- **Cl is within reach.** 13.5% defined MAPE, and the raw 23.5% MAPE is
  mostly zero-crossing inflation at α ≈ 0°. The remaining gap is geometry
  generalisation: the 2,900+ unused Kanakaero coordinates and denser α
  sampling are the obvious levers.
- **Cm is the hard one.** Best is 16.5% (ensemble). Cm is a small quantity
  driven by the full chordwise pressure distribution, so it needs
  geometry-space coverage and features (camber/thickness shape, not just raw
  ordinates) that the other coefficients barely care about. 10% Cm MAPE
  requires a dedicated effort, not a tweak.
- **L/D compounds Cl and Cd.** 27% defined MAPE even though Cd is at 10% —
  L/D errors are dominated by Cl errors at low lift and by the stall region.
  It will improve as Cl does, but 10% L/D MAPE is the least realistic of the
  four targets.

So the honest headline: a **Cd-under-10% surrogate exists today**, a
Cl-under-10% version is a data-scaling exercise away, and Cm/L/D under 10%
are open research problems that will need more data, physics-aware features,
and probably a Cm-specific model.

### Time saved: XFOIL vs the surrogate (measured on this machine)

`analyze-error --xfoil-benchmark` timed real XFOIL polars: **median 0.61 s
per 13-point polar** over 6 runs (range 0.31-1.41 s, machine-load dependent;
≈ 0.06 s per converged operating point). The same inputs through the trained
models:

```text
model            batch throughput        single-row latency
ridge            6.7M rows/s             0.04 ms
mlp_torch        0.98M rows/s            0.37 ms
random_forest    40k rows/s              23 ms
hist_gb          36k rows/s              4.0 ms
```

Projecting a realistic design sweep — 100 airfoils × 6 Reynolds numbers × 13
alpha = 600 polars (7,800 operating points):

- **XFOIL serial:** ≈ 6 minutes (≈ 1 minute with 6 parallel workers).
- **ML surrogate:** 0.22 s for `hist_gb`, 0.008 s for `mlp_torch` — a
  **≈ 1,700-45,000× speedup**, i.e. the sweep goes from minutes to
  milliseconds.

Even the slowest candidate (`random_forest`, 23 ms per single row) is ~2.7×
faster per point than XFOIL, and the ~2,000-45,000× batch figures are the
regime that matters for design-space exploration and optimisation loops. The
measurement excludes geometry preprocessing (identical for both paths); the
point is the *repeated-evaluation* cost that dominates optimisation and Monte
Carlo workflows. The speedup is real but load-dependent: re-run
`analyze-error --xfoil-benchmark` on the target machine before quoting it.

Reproduce with:

```bash
uv run airfoil-ml analyze-error \
  --polar-csv data/raw/multi_re_100/combined.csv \
  --coordinates data/raw/external/kanakaero/processed_coordinates \
  --model-dir models/multi_re_100 --output results/multi_re_100_error \
  --xfoil-benchmark
```

This writes `error_summary.json` (overall + per-regime percentage errors),
`timing_summary.json` (XFOIL and ML wall-clock plus the projected sweep), and
plots of relative error vs α and runtime per operating point.

## Scaling the dataset with parametric camber/thickness generation

Aerodynamic behaviour is dominated by two geometry knobs: the **camber line**
(drives Cl level and pitching-moment Cm0) and the **thickness distribution**
(drives Cd and stall behaviour). Sweeping those knobs generates an unlimited
supply of training airfoils — the same idea behind the Kanakaero CST dataset
(shape coefficients → XFOIL polars) already used here, but with the
multi-Reynolds sweep and Cm that the public fixed-Re table lacks.

```bash
uv run airfoil-ml generate-camber \
  --output data/raw/generated_coordinates \
  --run-xfoil --polar-output data/raw/generated_multi_re \
  --workers 6
```

The generator builds NACA-style camber/thickness sections (piecewise-parabola
camber m/p + closed-TE thickness t), names each airfoil from its parameters
(`CAMBER_m040_p30_t120`), writes a provenance manifest, and runs the existing
sharded, resumable XFOIL batch runner. The default grid — camber 0-6%,
camber position 20-60%, thickness 8-16% — produced **2,592 rows from 36
airfoils × 6 Reynolds numbers** (213/216 solver cases succeeded; the 3 extreme
forward-camber failures are recorded in the manifest, never silently
retried or filled in).

The generated table drops straight into the existing pipeline
(`airfoil-ml train --polar-csv data/raw/generated_multi_re/combined.csv
--coordinates data/raw/generated_coordinates`). Its envelope matches and
slightly extends the real-airfoil sample where it matters most for the
remaining accuracy gaps:

```text
                    real 100-airfoil    generated camber
Cl                  -0.68 .. 1.50      -0.57 .. 1.54
Cd                   0.003 .. 0.152     0.004 .. 0.152
Cm                  -0.215 .. 0.059    -0.216 .. 0.067
```

**Scaling economics:** coordinates are free to generate; the cost is XFOIL
wall-clock (~0.6 s/polar, so 1,000 airfoils × 6 Re ≈ 6,000 polars ≈ 10 min
at 10 workers, and the batch runner resumes from completed shards).

**The honest caveat:** a parameter family is *dense but narrow*. The NACA
four-digit family is a small, smooth subset of airfoil space, so a
parametric-only training set would generalise well to parametric test
airfoils and poorly to exotic real sections. The right use is **hybrid**: keep
the real-airfoil data as the generalization backbone and add parametric
coverage in the regimes it lacks (high camber, extreme thickness, wide Cm0),
which is exactly the Cm gap from the 10%-accuracy analysis. A wider design
space should upgrade the generator from NACA camber/thickness to CST
coefficients, matching the Kanakaero format.

## Large hybrid dataset: 100 real + 336 parametric airfoils

The camber grid was widened to camber 0-8%, camber position 20-70%, thickness
6-18% (336 designs) and the polar sweep extended to `α = -4° … 14°` so the
training envelope reaches past stall onset. Generation used the same sharded,
resumable batch runner:

```bash
uv run airfoil-ml generate-camber \
  --output data/raw/generated_large_coordinates \
  --run-xfoil --polar-output data/raw/generated_large \
  --camber-values 0,0.01,0.02,0.03,0.04,0.05,0.06,0.08 \
  --camber-positions 0.2,0.3,0.4,0.5,0.6,0.7 \
  --thickness-values 0.06,0.08,0.10,0.12,0.14,0.16,0.18 \
  --alpha-start -4 --alpha-end 14 \
  --workers 6
```

The run produced **31,311 rows from 1,838 successful (airfoil, Re) cases**
(166 of 2,016 cases failed, 8.2%, concentrated in 6%-thick and 8%-camber
designs where XFOIL convergence is genuinely poor; failures are recorded in
`_failed_cases.jsonl` and never filled in). Combined with the 100-airfoil real
dataset this is **37,605 rows across 435 airfoils**, a 4.2x scale-up of the
training data, with the parametric family supplying the multi-Reynolds sweep,
Cm, and post-stall alpha coverage that the public fixed-Re table lacks.

```bash
# assemble the hybrid table and coordinate set, then train with the SAME
# 20-airfoil holdout used by the multi_re_100 experiment
uv run airfoil-ml train \
  --polar-csv data/raw/hybrid_large/combined.csv \
  --coordinates data/raw/hybrid_coordinates \
  --output models/hybrid_large --seed 42 --log-cd \
  --test-airfoils-json models/multi_re_100/split_manifest.json \
  --only ridge random_forest hist_gb
uv run airfoil-ml train \
  --polar-csv data/raw/hybrid_large/combined.csv \
  --coordinates data/raw/hybrid_coordinates \
  --output models/hybrid_large --seed 42 --log-cd \
  --test-airfoils-json models/multi_re_100/split_manifest.json \
  --mlp-max-iter 150 --only mlp_torch
```

`--test-airfoils-json` fixes the held-out identities, so the comparison below
is the same 20 unseen real airfoils (1,310 rows) used by the earlier
experiment: the only thing that changed is the training data.

### Does 6x more data help generalise to real airfoils? Yes for most models

Defined MAPE (rows where the quantity is well defined, |reference| >= 2x the
reporting floor), baseline -> hybrid, same test split:

```text
model            cl              cd              cm              ld
hist_gb          15.2 -> 17.3%   10.4 -> 11.4%   25.7 -> 18.9%   28.1 -> 31.6%
mlp_torch        14.7 -> 13.7%   11.9 -> 10.0%   17.0 -> 15.0%   34.8 -> 27.6%
random_forest    16.9 -> 13.8%   11.7 -> 10.3%   19.7 -> 15.3%   30.6 -> 27.4%
ridge             --  -> 37.2%    --  -> 29.4%    --  -> 31.9%    --  -> 92.2%
```

- **The MLP and random forest improved on every coefficient.** The extra
  geometry and envelope coverage reduces variance (random forest) and feeds a
  shared smooth representation (MLP). The MLP is now the best single model
  for every coefficient, including Cd (10.0% defined MAPE, ~31 drag counts
  overall) where hist_gb used to lead.
- **HistGradientBoosting regressed on Cl and Cd.** Its histogram bins and
  capacity are now dominated by 31k parametric rows (the real-airfoil rows
  are only ~19% of training), and its internal early stopping validates on a
  camber-dominated random 10% of the data, so the fitted model tracks the
  parametric family better than the real-airfoil manifold. This is a
  concrete demonstration that "more data" is not uniformly helpful across
  model classes - it depends on how the extra data shapes the fitted
  function and its stopping rule.
- **Ridge degrades.** A linear model fitted on a much wider envelope cannot
  specialise to the real-airfoil region; it is included here as the
  capacity-limited counterexample.

### Ensembles: every target improves

```text
ensemble (defined MAPE)          cl      cd      cm      ld
hist_gb + mlp_torch baseline     13.5%   9.8%    18.2%   27.9%
3-way baseline                   13.7%   9.9%    16.5%   27.1%
hist_gb + mlp_torch hybrid       12.9%   9.2%    14.8%   24.9%
3-way hybrid                     12.2%   9.0%    13.7%   23.5%
mlp_torch + random_forest hybrid 11.3%   8.9%    13.1%   22.6%
```

Averaging the two models that benefited from the data (MLP + random forest)
is the best predictor: **Cl 11.3%, Cd 8.9%, Cm 13.1%, L/D 22.6%** on the
same 20 unseen real airfoils. Every target improved versus the best baseline
ensemble; Cd is comfortably under 10%, Cl and Cm are within reach of 10%,
and the remaining L/D gap is still dominated by the stall region and by Cd
sensitivity at low drag. The ensemble analysis is reproduced by averaging
the saved `test_predictions_*.npz` arrays; the numbers are also written to
`results/hybrid_large_error/ensemble_summary.json`.

The drag-regime picture is unchanged in shape: Cd error still concentrates
near stall (`mlp_torch` mean |dCd| 0.0015 in attached flow vs 0.0065 near
stall) and at low Reynolds numbers (0.0052 at Re=30k/50k vs 0.0013-0.0021 at
Re >= 250k), so the low-Re and stall regions remain the targets for further
data and for a stall-aware loss.

## Train and evaluate

```bash
uv run airfoil-ml train \
  --polar-csv data/raw/polars.csv \
  --coordinates data/raw/coordinates \
  --output models/initial \
  --points 100 \
  --seed 42

uv run airfoil-ml evaluate \
  --polar-csv data/raw/polars.csv \
  --coordinates data/raw/coordinates \
  --model-dir models/initial \
  --model mlp \
  --output results/initial
```

Training writes:

- `ridge.joblib`, `random_forest.joblib`, `mlp.joblib`;
- `preprocessor.joblib`, whose scalers are fitted on training rows only;
- `split_manifest.json`, the immutable-by-convention airfoil split;
- `metrics.json`, with MAE, RMSE, and R² for validation and unseen-airfoil test;
- `history_*.json`, including the MLP loss curve where available;
- `test_predictions_*.npz`, for repeatable plot generation;
- `training_config.json`, including the random seed and model settings.

Make a single inference prediction after training:

```bash
uv run airfoil-ml predict \
  --model-dir models/initial \
  --model mlp \
  --coordinates data/raw/coordinates/naca2412.dat \
  --alpha 5 \
  --reynolds 1000000
```

The returned coefficients are estimates at the requested operating point. Do
not extrapolate beyond the geometry, Reynolds, Mach, or angle-of-attack range
represented in the training data without a separate uncertainty study.

## How to interpret the evaluation

The parity plots show whether predictions track each coefficient globally, but
coefficient scales differ: a small absolute `Cd` error can be important to
`L/D` even when it looks visually small. Use all of:

- **MAE:** average absolute coefficient error in coefficient units;
- **RMSE:** penalises large misses, useful for stall or convergence outliers;
- **R²:** variance explained, but not a substitute for engineering accuracy;
- **polar plots:** whether the model preserves lift-curve slope, drag rise,
  moment trends, and the shape of `L/D` versus alpha;
- **error versus alpha:** reveals nonlinear/stall regions that global metrics
  can hide.

A physically useful model should reproduce the mostly linear pre-stall lift
trend, the increase in drag with lift/loading, and the qualitative change in
behavior near separation. It is acceptable for the first model to struggle
near stall; it is not acceptable to hide that behavior behind one aggregate
score.

The large fixed-Re experiment has now been run. On its grouped unseen-airfoil
test split, the log-`Cd` MLP measured `Cl` MAE 0.0487, `Cl` R² 0.9651,
`Cd` MAE 0.00832, and `Cd` R² 0.1525. These are actual measured results for
the fixed-Re source,
not claims about multi-Reynolds or `Cm` performance. The full metrics are in
`models/large_fixed_re/metrics.json`; the plots are in
`results/large_fixed_re_mlp/`. The low `Cd` R² shows that drag remains much
harder to model than lift even with thousands of shapes.

## Important limitations

1. **XFOIL fidelity:** it is a low-order engineering solver, not Navier-Stokes
   CFD or wind-tunnel data. Transition, turbulence, roughness, and separation
   are simplified.
2. **Fixed-Re dataset scope:** the large public dataset has one documented
   Reynolds number and no `Cm`; it is suitable for geometry generalisation,
   not Reynolds-number dependence or complete polar prediction.
3. **Multi-condition dataset scope:** the 100-airfoil multi-Reynolds XFOIL
   table (6,294 rows, six Reynolds numbers, `α = -2° … 10°`) is large enough
   for a first surrogate comparison, but it samples only a fraction of the
   geometry space, does not cover post-stall angles, and inherits XFOIL's
   transition-model uncertainty at low Reynolds numbers.
4. **Grouped extrapolation:** unseen-airfoil testing measures generalisation
   within the sampled shape family. It does not prove performance on arbitrary
   aircraft sections.
5. **Stall and invalid points:** solver non-convergence and hysteresis are not
   yet represented explicitly. A stall-aware dataset needs convergence flags,
   continuation strategy, and possibly separate attached/separated regimes.
6. **Uncertainty:** the models output point estimates only. Ensembles,
   conformal intervals, or a probabilistic model should be added before using
   predictions for design decisions.
7. **Physics constraints:** the current regressors are not guaranteed to
   preserve monotonicity, positivity of drag, or consistent polar smoothness.
   A physics-aware loss or post-training constraint analysis is future work.

## Prioritised next steps

1. (Done, tested) The parametric half of the scale-up is complete: the
   336-airfoil camber/thickness grid with a `α = -4° … 14°` sweep added
   31k rows (37.6k rows, 435 airfoils in the hybrid table). It improved the
   MLP/random-forest ensembles on every target (Cl 11.3%, Cd 8.9%, Cm 13.1%,
   L/D 22.6% defined MAPE on the same 20-airfoil real holdout). Two gaps
   remain in the data axis: the 2,900+ remaining Kanakaero geometries (real
   shapes, not parametric), and denser low-Re sampling where drag error is
   still ~3x the mid-Re value. Both generators are sharded and resumable.
2. Add a separate external validation set from published experimental polars,
   clearly distinguishing it from the XFOIL training reference.
3. Compare raw ordinate features with PCA and physically motivated camber/
   thickness descriptors using the same grouped splits.
4. Add repeated grouped cross-validation and confidence intervals rather than
   relying on one random split.
5. (Done, tested) Train Cd in log space for the multi-Reynolds model via
   `airfoil-ml train --log-cd`: it shifts the fitted loss toward relative
   error and buys ~1-2 points of Cd MAPE for the tree models, and it makes
   negative-Cd predictions impossible for the MLP (log output is exponentiated
   to positive values). Next: a shared multi-output network trained with a
   log-Cd loss against the current independent-output MLP.
6. (Done, tested) Ensemble the top candidates: the hist_gb + mlp_torch
   ensemble is the first model under 10% Cd MAPE (9.8%) and improves Cl to
   13.5%. Promote ensembling to a first-class pipeline step with
   validation-tuned weights, then target the remaining Cm gap with a
   dedicated Cm model and camber/thickness features.
7. Add polar-level smoothness/physics diagnostics, stall-regime flags, and
   uncertainty estimates (ensembles or conformal intervals) before using
   predictions for design decisions.
8. Extend the runtime study to the full prediction path (feature construction
   + inference + serving overhead) and to XFOIL with a finer panel count, so
   the speedup claim holds for the shipped artifact, not just the model call.
9. Expose the saved predictor through a small API or UI, then add airfoil
   optimisation only after surrogate validity is established.

## Research questions this foundation supports

- How much does geometry representation affect unseen-airfoil generalisation?
- Is the nonlinear MLP materially better than Ridge or a random forest?
- Which coefficient is hardest to predict, and does that change with Reynolds
  number?
- Does error concentrate around drag rise and stall?
- How much of the apparent accuracy comes from interpolation in operating
  conditions versus true geometry generalisation?

Those questions should be answered from generated artifacts and held-out data,
not from assumptions about what an ML model ought to achieve.
