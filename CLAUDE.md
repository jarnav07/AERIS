# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

AERIS (Aerodynamic Estimation and Response Inference System) is a research codebase that generates a
synthetic airfoil aerodynamics dataset with XFOIL and compares several ML surrogate models (Ridge,
Random Forest, scikit-learn MLP, HistGradientBoosting, PyTorch MLP) trained on exactly that dataset.
The point of the project is a controlled comparison — one canonical dataset, one feature pipeline, one
grouped split — feeding every model.

## Commands

Install and test:
```bash
uv sync --extra dev          # add --extra torch for the PyTorch MLP candidate
uv run pytest
uv run pytest tests/test_geometry.py::test_name   # single test
```

Generate the canonical dataset (requires an `xfoil` executable on PATH; on headless Linux it is
auto-run under `xvfb-run` when no `DISPLAY` is set — see `direct_xfoil.py`):
```bash
uv run python scripts/generate_dataset.py --cases 30000 --workers 8 --resume
# or: airfoil-ml generate-training-data --cases 30000 --workers 8 --resume
```

Train models on the generated CSV:
```bash
uv run python scripts/train.py --csv data/generated/training_data.csv --output-dir models
# or: airfoil-ml train --csv ... --output-dir models --only ridge mlp
```

Evaluate a trained model and produce plots:
```bash
uv run python scripts/evaluate_models.py --model mlp --model-dir models --output results/evaluation
# or: airfoil-ml evaluate --model mlp --model-dir models --output results/evaluation
```
Evaluation always scores the model against the exact test-airfoil identities recorded in
`model_dir/split_manifest.json` at training time (never a freshly redrawn split), and auto-detects
`log_cd` from `model_dir/training_config.json` unless explicitly overridden — see
`training.py::evaluate_kulfan_model`.

## Remote data generation (GCP VM)

Large-scale dataset generation runs on the `ml-aerofoil-data-generation` GCP VM (project
`ml-aerofoil-predictor`, zone `us-central1-f`). The repo there lives under a *different* user account
than the SSH login user: `/home/arnavdoodler007/AERIS`, owned by `arnavdoodler007` with a `700` home
directory, so it must be accessed via `sudo -u arnavdoodler007` (or `sudo` for file copies) rather than
a direct `cd`/`scp` as the login user. Its `.venv` is `uv`-managed and must be activated
(`source .venv/bin/activate`) before running any Python/XFOIL command. The VM is preemptible and the
zone occasionally lacks `e2-standard-8` capacity — expect to retry starts/SSH and batch remote work
into as few round-trips as possible.

## Architecture

### Two separate data/training pipelines exist — know which one is canonical

- **Canonical pipeline** (used by the CLI and both entry-point scripts):
  `training_data_generation.py` → `training_data.csv` → `training.py::train_from_kulfan_csv`.
  This path reads Kulfan geometry parameters and flow conditions directly as flat CSV columns (`Re`,
  `alpha`, `kulfan_upper_0..7`, etc.) and does its own inline scaling (it builds a `StandardScaler`
  itself and stores it inside a `FeaturePreprocessor` largely as a container — it does **not** call
  `fit_preprocessor`/`transform_inputs` from `features.py`).
- **Generic/legacy pipeline**: `data.py` (`AeroDataset`, `load_dataset`, coordinate-file-based
  geometry) + `features.py` (`fit_preprocessor`, `raw_input_matrix`, `transform_inputs`) assumes a
  polar CSV (`airfoil_id, alpha_deg, reynolds, mach, cl, cd, cm`) plus a directory of `.dat`/`.txt`
  coordinate files per airfoil. This is a more general/reusable pipeline (with its own leakage-safe
  `grouped_split`) but is not what `train_from_kulfan_csv` actually runs.

When changing feature scaling, target columns, or the train/val/test split logic, check whether the
change needs to be made in `training.py` (canonical path), `data.py`/`features.py` (generic path), or
both — they currently duplicate the grouped-split-by-`airfoil_id` logic independently.

### Dataset generation pipeline

`training_data_generation.py` samples synthetic Kulfan/CST airfoils by convex-combining 3 parent
airfoils from AeroSandbox's bundled database, adding a covariance-based perturbation, and applying a
random log-normal scale. Operating conditions (`alpha` grid with jitter, log-normal `Re`, `Mach=0`,
`n_crit`, forced/free transition) are sampled per case and each case is run through XFOIL directly (not
via AeroSandbox's XFOIL wrapper) using `direct_xfoil.py`. Results are written as resumable per-airfoil
Parquet shards under `data/generated/shards/`, then combined into one `training_data.csv` plus a
`training_data.provenance.json` manifest recording the seed, config, and per-case failures. A synthetic
"CYLINDER" case (all-zero Kulfan vector, low confidence) is appended by default as a
known-drag-coefficient sanity anchor.

`direct_xfoil.py` drives XFOIL as a subprocess with a hand-built batch command script (`_build_commands`)
that sweeps positive then negative alpha (re-`INIT`-ing between branches so a failed high-alpha solution
doesn't poison the rest of the sweep) and dumps boundary-layer profiles (`DUMP`) at each converged alpha.
Notable hardening baked into this file, from real failures seen in production runs — don't undo these
without understanding why they're there:
- `_run_process` launches XFOIL with `start_new_session=True` and kills the whole process group on
  timeout, because `xvfb-run` is a shell script whose child `xfoil` process escapes a plain
  `subprocess.run(timeout=...)` kill and can spin at 100% CPU forever.
- An `LD_PRELOAD` shim (`_fpe_shim_path`, compiled on the fly with `gcc`/`cc`) no-ops
  `_gfortran_set_fpe` because Debian/Ubuntu's packaged `xfoil` binary unmasks hardware FPE traps and
  crashes with `SIGFPE` on ordinary benign NaN/Inf results (e.g. zero boundary-layer momentum thickness
  at a stagnation point) that upstream XFOIL just reports as a warning.
- `GFORTRAN_UNBUFFERED_ALL=1` is set so polar output is flushed even when XFOIL is killed/times out.
- `TrainingDataConfig.log10_re_sigma` is deliberately `0.75`, not the more "natural" `1.5`, to keep the
  3-sigma tail above `Re~1800` — a wider sigma let ~1-2% of samples draw non-physical low-Re cases that
  produced laminar-separation-bubble shape-factor spikes in the boundary-layer targets.

### Model layer

`models.py::make_models` returns all candidate models for a given `ModelConfig` (Ridge, Random Forest,
sklearn MLP, HistGradientBoosting wrapped in `MultiOutputRegressor`); it silently omits `mlp_torch` if
`torch` isn't installed (it's an optional extra). `torch_mlp.py` implements `TorchMLPRegressor` with a
scikit-learn-compatible `fit`/`predict` interface so it drops into the same training loop as the other
models. `training.py` optionally log-transforms `CD` (`--log-cd`) so the model minimizes relative rather
than absolute drag error, then inverse-transforms predictions before computing metrics — always keep
`_transform_labels`/`_inverse_labels` symmetric when touching this.

### Evaluation layer

`evaluation.py` holds the generic metrics/plotting utilities (`regression_metrics`,
`save_evaluation_plots`); `save_evaluation_plots` expects a frame with `airfoil_id`, `alpha_deg`,
`reynolds`, `cl`, `cd` columns (the legacy/generic pipeline's naming) plus `(N, 3)` `actual`/`predicted`
arrays in `cl, cd, cm` order — it does not know about the canonical pipeline's `alpha`/`Re` column
names. `training.py::evaluate_kulfan_model` is the adapter: it rebuilds the canonical test split from
`split_manifest.json`, renames columns to what `save_evaluation_plots` expects, and is what both
`scripts/evaluate_models.py` and `airfoil-ml evaluate` call. (`scripts/evaluate_models.py` previously
called `save_evaluation_plots` directly with the wrong argument shape and crashed unconditionally; if
you see that pattern reappear, route through `evaluate_kulfan_model` instead of re-wiring it by hand.)

### Data provenance rules (see `data/README.md` and top-level `README.md` for full detail)

The canonical generator does **not** use the UIUC coordinate database or the Kanakaero fixed-Re dataset;
those exist only as legacy/alternative experiments (`scripts/acquire_data.py` and the generic pipeline
above). Never silently concatenate legacy/downloaded datasets with the canonical stochastic dataset —
they don't share the same target set or Reynolds distribution, and doing so would invalidate the
controlled-comparison premise of the project. Large generated datasets, trained model binaries, and
result artifacts are intentionally kept out of Git.
