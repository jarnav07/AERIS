"""Training orchestration for grouped, leakage-safe surrogate modelling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from .data import AeroDataset, save_split_manifest
from .evaluation import regression_metrics
from .features import FeaturePreprocessor, fit_preprocessor, raw_input_matrix
from .models import ModelConfig, make_models


def _matrices(dataset: AeroDataset, indices: np.ndarray, n_geometry_points: int) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.frame.iloc[indices]
    geometries = [dataset.geometries[str(airfoil_id)] for airfoil_id in frame.airfoil_id]
    inputs = raw_input_matrix(
        geometries,
        frame.alpha_deg.to_numpy(float),
        frame.reynolds.to_numpy(float),
        frame.mach.to_numpy(float),
    )
    targets = frame[["cl", "cd", "cm"]].to_numpy(float)
    return inputs, targets


def _transform_labels(targets: np.ndarray, log_cd: bool) -> np.ndarray:
    """Transform training labels; log-Cd makes the model minimize relative Cd error."""
    transformed = np.asarray(targets, dtype=float).copy()
    if log_cd:
        if np.any(transformed[:, 1] <= 0):
            raise ValueError("Cd must be positive for log transformation")
        transformed[:, 1] = np.log(transformed[:, 1])
    return transformed


def _inverse_labels(processor: FeaturePreprocessor, scaled_prediction: np.ndarray, log_cd: bool) -> np.ndarray:
    """Undo the target scaler and, for log-Cd models, exponentiate Cd back to counts."""
    physical = processor.inverse_targets(scaled_prediction)
    if log_cd:
        physical[:, 1] = np.exp(physical[:, 1])
    return physical


def train_all(
    dataset: AeroDataset,
    output_dir: str | Path,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    model_config: ModelConfig | None = None,
    only: list[str] | None = None,
    log_cd: bool = False,
    test_airfoils: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit candidate models and save their estimators and train-only preprocessors.

    ``only`` restricts training to the named models so a slow candidate (for
    example the sklearn MLP on a large dataset) can be trained in a separate,
    budgeted command without discarding previously trained models. Metrics are
    merged into an existing ``metrics.json`` when one is present, so partial
    runs accumulate into one authoritative result file.

    ``test_airfoils`` fixes the held-out identities (e.g. reusing a previous
    experiment's split manifest), so adding training data can be evaluated
    against the exact same unseen-airfoil test set.
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = dataset.grouped_split(test_fraction, validation_fraction, seed, test_airfoils=test_airfoils)
    save_split_manifest(output_dir / "split_manifest.json", split, seed)
    train_x, train_y = _matrices(dataset, split["train"], next(iter(dataset.geometries.values())).n_points)
    val_x, val_y = _matrices(dataset, split["validation"], next(iter(dataset.geometries.values())).n_points)
    test_x, test_y = _matrices(dataset, split["test"], next(iter(dataset.geometries.values())).n_points)
    # Fit scalers on the training airfoils only, then apply exactly those
    # transforms to validation, test, and future inference cases. With
    # log_cd=True the scaler and the model see log(Cd), so the fitted loss is
    # a relative (percentage) drag error; metrics are always reported in
    # physical units via _inverse_labels.
    train_labels = _transform_labels(train_y, log_cd)
    processor = fit_preprocessor(train_x, train_labels, next(iter(dataset.geometries.values())).n_points)
    processor.save(output_dir / "preprocessor.joblib")
    train_xs, val_xs, test_xs = processor.input_scaler.transform(train_x), processor.input_scaler.transform(val_x), processor.input_scaler.transform(test_x)
    train_ys = processor.target_scaler.transform(train_labels)
    val_ys = processor.target_scaler.transform(_transform_labels(val_y, log_cd))
    test_ys = processor.target_scaler.transform(_transform_labels(test_y, log_cd))
    results_path = output_dir / "metrics.json"
    results: dict[str, dict[str, dict[str, float]]] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    config = model_config or ModelConfig(seed=seed)
    models = make_models(config)
    if only is not None:
        unknown = set(only) - set(models)
        if unknown:
            raise ValueError(f"unknown model names: {sorted(unknown)}; available: {sorted(models)}")
        models = {name: model for name, model in models.items() if name in only}
    for name, model in models.items():
        model.fit(train_xs, train_ys)
        val_pred = _inverse_labels(processor, model.predict(val_xs), log_cd)
        test_pred = _inverse_labels(processor, model.predict(test_xs), log_cd)
        results[name] = {
            "validation": regression_metrics(val_y, val_pred),
            "test": regression_metrics(test_y, test_pred),
        }
        joblib.dump(model, output_dir / f"{name}.joblib")
        history = {"loss_curve": [float(value) for value in getattr(model, "loss_curve_", [])]}
        validation_scores = getattr(model, "validation_scores_", None)
        if validation_scores is not None:
            history["validation_scores"] = [float(value) for value in validation_scores]
        (output_dir / f"history_{name}.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(output_dir / f"test_predictions_{name}.npz", actual=test_y, predicted=test_pred, indices=split["test"])
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps({"seed": seed, "log_cd": log_cd, "model_config": asdict(config)}, indent=2) + "\n", encoding="utf-8")
    return results


def load_model_bundle(model_dir: str | Path, model_name: str = "mlp") -> tuple[object, FeaturePreprocessor]:
    model_dir = Path(model_dir)
    return joblib.load(model_dir / f"{model_name}.joblib"), FeaturePreprocessor.load(model_dir / "preprocessor.joblib")
