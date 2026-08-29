# ML Airfoil Predictor

A reproducible machine learning surrogate for predicting airfoil aerodynamic performance and boundary-layer state.

This project implements a single, unified canonical data generation pipeline inspired by NeuralFoil. It generates random Kulfan-parameterized airfoils, runs them through XFOIL across various operating conditions, and builds a comprehensive dataset for training surrogate models (Random Forest, Ridge, Gradient Boosting, PyTorch MLP). 

By design, this pipeline **does not rely on downloading pre-existing datasets**. All geometries and flow results are generated locally, synthetically, and scalably using parallel processing.

## Key Features

1. **Standalone Debian-ready script:** The entire pipeline (dependency installation, generation, and model training) is orchestrated by a single `generate_dataset.py` script.
2. **Comprehensive Targets:** The surrogate models learn to predict not just global coefficients ($C_l$, $C_d$, $C_m$, transition locations) but also 192 boundary layer parameters ($\theta$, $H$, $u_e/V_\infty$) across both surfaces, significantly improving accuracy by teaching the network the underlying flow physics.
3. **Resumable Parquet Sharding:** Data generation writes compressed `pyarrow.parquet` shards per airfoil, allowing massive parallel runs (defaulting to 30,000 cases) that can be safely interrupted and resumed.

## Running the Pipeline

The easiest way to generate the dataset and train the models is to run the standalone script on a fresh Debian/Ubuntu machine. 

```bash
sudo apt-get update
sudo apt-get install xfoil xvfb python3-pip

# Run generation and training (~30,000 airfoils, will take time)
python3 generate_dataset.py --cases 30000 --output-dir ./output
```

**What the script does:**
- Automatically installs required Python packages (`aerosandbox`, `pandas`, `pyarrow`, `scikit-learn`, `tqdm`, `torch`).
- Uses `xvfb-run` to run XFOIL headlessly.
- Generates 30,000 random Kulfan shapes, varying $C_l$, Reynolds, Mach, and angle of attack.
- Saves progress in `output/shards/` as `.parquet` files and eventually merges them into `output/training_data.csv`.
- Trains multiple models (including a PyTorch MLP) on the generated dataset and saves them to `output/models/`.

### CLI Arguments

- `--cases N`: Number of random airfoils to generate (default: 30000).
- `--workers W`: Number of parallel XFOIL subprocesses to spawn (defaults to half of your CPU cores).
- `--output-dir PATH`: Where to save data shards, combined CSV, and trained models (default: `./output`).
- `--resume`: If interrupted, pass this flag to skip already-generated shards.
- `--skip-train`: Stop after generating the data, without training the models.

## Repository Structure

- `generate_dataset.py`: The single entry point for end-to-end dataset generation and training.
- `src/airfoil_ml/`: Core package containing modules for data handling, features, models, and evaluation.
  - `training_data_generation.py`: The Kulfan randomizer and parallel XFOIL orchestrator.
  - `training.py`: Model fitting, scaling, and validation loops.
  - `torch_mlp.py`: Fast PyTorch MLP regressor.
  - `features.py`, `models.py`: Feature standardisation and model definitions.
  - `cli.py`: The underlying module command-line interface.
- `tests/`: Pytest suite (run with `pytest tests/`).

## Architecture Details

**Geometry Representation:**
Instead of raw coordinates, geometries are sampled via 18 Kulfan parameters (8 upper, 8 lower, LE weight, TE thickness). The covariance of the sampling is anchored by the built-in `aerosandbox` airfoil database, ensuring physical shapes while exploring a vast parametric space.

**Model Inputs (24 features):**
- 18 Kulfan geometry parameters
- 6 flow conditions: $\alpha$ (Angle of Attack), $Re$ (Reynolds Number, log-scaled internally), $M$ (Mach), $N_{crit}$, forced transition locations.

**Model Outputs (197 targets):**
- 5 aerodynamic metrics ($C_l$, $C_d$, $C_m$, upper/lower transition points).
- 192 boundary layer variables (32 stations $\times$ 3 metrics $\times$ 2 surfaces).

By predicting the boundary layer alongside global coefficients, the network builds internal representations of boundary-layer displacement and momentum thickness, making drag predictions substantially more accurate.
