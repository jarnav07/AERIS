"""Adapter and training workflow for the large fixed-Re CST dataset.

Source:
https://github.com/kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils

The published CSV contains eight CST shape coefficients, angle of attack, Cl,
and Cd for approximately 2,900 airfoils at a documented fixed Reynolds number
of 100,000. It is intentionally kept separate from the coordinate/XFOIL
pipeline because it has no Cm or Reynolds sweep.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .evaluation import regression_metrics_for_targets
from .features import FeaturePreprocessor
from .models import ModelConfig, make_models

LARGE_DATASET_URL = (
    "https://raw.githubusercontent.com/kanakaero/"
    "Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils/"
    "90dae87066913316bc28547b966951294fb885c8/"
    "Final%20Results/COMPILED%20AIRFOIL%20DATA.csv"
)
LARGE_DATASET_REYNOLDS = 100000.0
CST_COLUMNS = tuple(f"CST Coeff {i}" for i in range(1, 9))
FIXED_TARGET_COLUMNS = ("Cl", "Cd")


def transform_fixed_targets(targets: np.ndarray, log_cd: bool = True) -> np.ndarray:
    """Transform training targets while retaining physical-unit evaluation."""
    transformed = np.asarray(targets, dtype=float).copy()
    if log_cd:
        if np.any(transformed[:, 1] <= 0):
            raise ValueError("Cd must be positive for log transformation")
        transformed[:, 1] = np.log(transformed[:, 1])
    return transformed


def inverse_fixed_targets(processor: FeaturePreprocessor, predictions: np.ndarray, log_cd: bool = True) -> np.ndarray:
    physical = processor.inverse_targets(predictions)
    if log_cd:
        physical[:, 1] = np.exp(physical[:, 1])
    return physical


@dataclass
class FixedReDataset:
    frame: pd.DataFrame
    reynolds: float = LARGE_DATASET_REYNOLDS

    @property
    def airfoil_ids(self) -> np.ndarray:
        return np.array(sorted(self.frame["airfoil_id"].unique()))

    def validate(self) -> None:
        required = {"airfoil_id", "alpha_deg", *CST_COLUMNS, *FIXED_TARGET_COLUMNS}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"large dataset is missing columns: {sorted(missing)}")
        if self.frame.empty:
            raise ValueError("large dataset is empty")
        numeric = ["alpha_deg", *CST_COLUMNS, *FIXED_TARGET_COLUMNS]
        if not np.isfinite(self.frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError("large dataset contains non-finite values")
        if (self.frame["Cd"] < 0).any():
            raise ValueError("negative drag coefficients are not physically valid")
        if self.frame["airfoil_id"].nunique() < 3:
            raise ValueError("at least three airfoils are required for grouped splitting")

    def grouped_split(self, test_fraction: float = 0.2, validation_fraction: float = 0.2, seed: int = 42) -> dict[str, np.ndarray]:
        if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
            raise ValueError("fractions must be positive and sum to less than one")
        rng = np.random.default_rng(seed)
        ids = rng.permutation(self.airfoil_ids)
        n_test = max(1, int(round(len(ids) * test_fraction)))
        n_validation = max(1, int(round(len(ids) * validation_fraction)))
        test_ids = ids[:n_test]
        validation_ids = ids[n_test : n_test + n_validation]
        train_ids = ids[n_test + n_validation :]
        return {
            "train": np.flatnonzero(self.frame.airfoil_id.isin(train_ids)),
            "validation": np.flatnonzero(self.frame.airfoil_id.isin(validation_ids)),
            "test": np.flatnonzero(self.frame.airfoil_id.isin(test_ids)),
            "train_airfoils": train_ids,
            "validation_airfoils": validation_ids,
            "test_airfoils": test_ids,
        }


def download_large_dataset(output_path: str | Path, timeout: int = 120) -> Path:
    """Download the pinned public CSV without changing source code or secrets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(LARGE_DATASET_URL, headers={"User-Agent": "ml-airfoil-predictor/0.1"})
    with urlopen(request, timeout=timeout) as response:
        output_path.write_bytes(response.read())
    return output_path


def load_fixed_re_dataset(path: str | Path, reynolds: float = LARGE_DATASET_REYNOLDS) -> FixedReDataset:
    raw = pd.read_csv(path)
    required = {"Filename", "AoA", *CST_COLUMNS, *FIXED_TARGET_COLUMNS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"large dataset is missing source columns: {sorted(missing)}")
    frame = raw.rename(columns={"Filename": "airfoil_id", "AoA": "alpha_deg"}).copy()
    frame["airfoil_id"] = frame["airfoil_id"].astype(str).str.strip()
    dataset = FixedReDataset(frame, reynolds=float(reynolds))
    dataset.validate()
    return dataset


def _matrices(dataset: FixedReDataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.frame.iloc[indices]
    inputs = frame[[*CST_COLUMNS, "alpha_deg"]].to_numpy(float)
    targets = frame[list(FIXED_TARGET_COLUMNS)].to_numpy(float)
    return inputs, targets


def train_fixed_re(
    dataset: FixedReDataset,
    output_dir: str | Path,
    seed: int = 42,
    model_config: ModelConfig | None = None,
    log_cd: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """Train candidate models while holding the source Reynolds number fixed."""
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = dataset.grouped_split(seed=seed)
    manifest = {"seed": seed, "reynolds": dataset.reynolds}
    for key in ("train_airfoils", "validation_airfoils", "test_airfoils"):
        manifest[key] = split[key].tolist()
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    train_x, train_y = _matrices(dataset, split["train"])
    validation_x, validation_y = _matrices(dataset, split["validation"])
    test_x, test_y = _matrices(dataset, split["test"])
    processor = FeaturePreprocessor(
        StandardScaler().fit(train_x),
        StandardScaler().fit(transform_fixed_targets(train_y, log_cd=log_cd)),
        n_geometry_points=0,
    )
    processor.save(output_dir / "preprocessor.joblib")
    train_xs = processor.input_scaler.transform(train_x)
    validation_xs = processor.input_scaler.transform(validation_x)
    test_xs = processor.input_scaler.transform(test_x)
    train_ys = processor.target_scaler.transform(transform_fixed_targets(train_y, log_cd=log_cd))

    config = model_config or ModelConfig(seed=seed)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, model in make_models(config).items():
        model.fit(train_xs, train_ys)
        validation_prediction = inverse_fixed_targets(processor, model.predict(validation_xs), log_cd=log_cd)
        test_prediction = inverse_fixed_targets(processor, model.predict(test_xs), log_cd=log_cd)
        results[name] = {
            "validation": regression_metrics_for_targets(validation_y, validation_prediction, FIXED_TARGET_COLUMNS),
            "test": regression_metrics_for_targets(test_y, test_prediction, FIXED_TARGET_COLUMNS),
        }
        joblib.dump(model, output_dir / f"{name}.joblib")
        history = {"loss_curve": [float(value) for value in getattr(model, "loss_curve_", [])]}
        validation_scores = getattr(model, "validation_scores_", None)
        if validation_scores is not None:
            history["validation_scores"] = [float(value) for value in validation_scores]
        (output_dir / f"history_{name}.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(
            output_dir / f"test_predictions_{name}.npz",
            actual=test_y,
            predicted=test_prediction,
            indices=split["test"],
        )

    metadata = {
        "source_url": LARGE_DATASET_URL,
        "reynolds": dataset.reynolds,
        "rows": len(dataset.frame),
        "airfoils": int(dataset.frame.airfoil_id.nunique()),
        "input_features": [*CST_COLUMNS, "alpha_deg"],
        "target_columns": list(FIXED_TARGET_COLUMNS),
        "log_cd": log_cd,
        "note": "Fixed-Re CST dataset; not interchangeable with multi-Re/Cm XFOIL data.",
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (output_dir / "training_config.json").write_text(
        json.dumps({"seed": seed, "model_config": asdict(config)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return results
