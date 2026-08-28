"""Train all surrogate models from the one canonical XFOIL dataset."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from .data import KULFAN_COLUMNS, TARGET_COLUMNS, AeroDataset, save_split_manifest
from .features import FeaturePreprocessor, build_feature_matrix, fit_preprocessor
from .models import ModelConfig, make_models


def _targets(frame) -> np.ndarray:
    return frame[list(TARGET_COLUMNS)].to_numpy(float)


def _transform_targets(targets: np.ndarray, log_cd: bool) -> np.ndarray:
    transformed = np.asarray(targets, dtype=float).copy()
    if log_cd:
        if np.any(transformed[:, 1] <= 0):
            raise ValueError("Cd must be positive for log-Cd training")
        transformed[:, 1] = np.log(transformed[:, 1])
    return transformed


def inverse_targets(
    processor: FeaturePreprocessor,
    predictions: np.ndarray,
    log_cd: bool,
) -> np.ndarray:
    physical = processor.inverse_targets(predictions)
    if log_cd:
        physical[:, 1] = np.exp(physical[:, 1])
    return physical


def train_all(
    dataset: AeroDataset,
    output_dir: str | Path,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    model_config: ModelConfig | None = None,
    only: list[str] | None = None,
    log_cd: bool = True,
    test_airfoils: list[str] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Train the same candidate set on the same leakage-safe split.

    The dataset is split by generated-airfoil identity. Every model receives
    the identical rows, preprocessing, targets, and held-out test identities.
    This makes the resulting model-to-model comparison meaningful.
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = dataset.grouped_split(
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
        test_airfoils=test_airfoils,
    )
    save_split_manifest(output_dir / "split_manifest.json", split, seed)

    train_frame = dataset.frame.iloc[split["train"]]
    validation_frame = dataset.frame.iloc[split["validation"]]
    test_frame = dataset.frame.iloc[split["test"]]

    train_x = build_feature_matrix(train_frame)
    validation_x = build_feature_matrix(validation_frame)
    test_x = build_feature_matrix(test_frame)
    train_y = _targets(train_frame)
    validation_y = _targets(validation_frame)
    test_y = _targets(test_frame)

    train_y_model = _transform_targets(train_y, log_cd)
    processor = fit_preprocessor(train_x, train_y_model)
    processor.save(output_dir / "preprocessor.joblib")

    train_x_scaled = processor.transform_inputs(train_x)
    validation_x_scaled = processor.transform_inputs(validation_x)
    test_x_scaled = processor.transform_inputs(test_x)
    train_y_scaled = processor.transform_targets(train_y_model)

    config = model_config or ModelConfig(seed=seed)
    available = make_models(config)
    if only is None:
        selected = available
    else:
        unknown = sorted(set(only) - set(available))
        if unknown:
            raise ValueError(f"unknown models: {unknown}; available: {sorted(available)}")
        selected = {name: available[name] for name in only}

    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for name, model in selected.items():
        model.fit(train_x_scaled, train_y_scaled)
        validation_prediction = inverse_targets(
            processor, model.predict(validation_x_scaled), log_cd
        )
        test_prediction = inverse_targets(
            processor, model.predict(test_x_scaled), log_cd
        )

        from .evaluation import regression_metrics

        results[name] = {
            "validation": regression_metrics(validation_y, validation_prediction),
            "test": regression_metrics(test_y, test_prediction),
        }

        joblib.dump(model, output_dir / f"{name}.joblib")
        np.savez_compressed(
            output_dir / f"test_predictions_{name}.npz",
            actual=test_y,
            predicted=test_prediction,
            indices=split["test"],
        )

        history: dict[str, list[float]] = {}
        for attribute in ("loss_curve_", "validation_scores_"):
            values = getattr(model, attribute, None)
            if values is not None:
                history[attribute] = [float(value) for value in values]
        (output_dir / f"history_{name}.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    training_metadata = {
        "dataset_rows": len(dataset.frame),
        "dataset_airfoils": int(dataset.frame["airfoil_id"].nunique()),
        "input_features": [*KULFAN_COLUMNS, "alpha_deg", "log10_reynolds", "mach"],
        "target_columns": list(TARGET_COLUMNS),
        "log_cd": log_cd,
        "seed": seed,
        "model_names": list(selected),
        "model_config": asdict(config),
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(training_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def load_model_bundle(model_dir: str | Path, model_name: str):
    model_dir = Path(model_dir)
    return (
        joblib.load(model_dir / f"{model_name}.joblib"),
        FeaturePreprocessor.load(model_dir / "preprocessor.joblib"),
    )
