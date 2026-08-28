"""Metrics and plots for model comparison against XFOIL and CFD references."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import KULFAN_COLUMNS, TARGET_COLUMNS, AeroDataset
from .features import build_feature_matrix


# Percentage error is undefined or misleading near zero crossings. The floor
# is deliberately explicit and reported with every metrics file.
DEFAULT_PERCENTAGE_FLOORS = {"cl": 0.05, "cd": 1e-4, "cm": 0.01}


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    percentage_floor: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Return MAE, RMSE, R² and defined MAPE for Cl/Cd/Cm."""
    floors = percentage_floor or DEFAULT_PERCENTAGE_FLOORS
    result: dict[str, dict[str, float]] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        reference = np.asarray(actual[:, index], dtype=float)
        prediction = np.asarray(predicted[:, index], dtype=float)
        absolute_error = np.abs(reference - prediction)
        mask = np.abs(reference) >= floors[target]
        mape = float(np.mean(absolute_error[mask] / np.abs(reference[mask])) * 100) if mask.any() else float("nan")
        result[target] = {
            "mae": float(mean_absolute_error(reference, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(reference, prediction))),
            "r2": float(r2_score(reference, prediction)),
            "mape_percent": mape,
            "percentage_floor": floors[target],
            "percentage_rows": int(mask.sum()),
        }
    return result


def discover_models(model_dir: str | Path) -> list[str]:
    return sorted(
        path.stem
        for path in Path(model_dir).glob("*.joblib")
        if path.stem != "preprocessor"
    )


def _load_training_config(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "training_config.json").read_text(encoding="utf-8"))


def _predict_saved_models(
    frame: pd.DataFrame,
    model_dir: str | Path,
    model_names: list[str],
) -> dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    processor = joblib.load(model_dir / "preprocessor.joblib")
    inputs = processor.transform_inputs(build_feature_matrix(frame))
    log_cd = bool(_load_training_config(model_dir).get("log_cd", True))

    predictions: dict[str, np.ndarray] = {}
    for name in model_names:
        model = joblib.load(model_dir / f"{name}.joblib")
        prediction = processor.inverse_targets(model.predict(inputs))
        if log_cd:
            prediction[:, 1] = np.exp(prediction[:, 1])
        predictions[name] = prediction
    return predictions


def load_test_reference(dataset: AeroDataset, model_dir: str | Path) -> pd.DataFrame:
    manifest = json.loads((Path(model_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    test_ids = set(map(str, manifest["test_airfoils"]))
    frame = dataset.frame[dataset.frame["airfoil_id"].astype(str).isin(test_ids)].copy()
    if frame.empty:
        raise ValueError("saved test split contains no rows in the supplied dataset")
    return frame


def _save_parity_plots(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    reference_name: str,
) -> None:
    for index, target in enumerate(TARGET_COLUMNS):
        fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
        combined = np.concatenate([actual[:, index], *[pred[:, index] for pred in predictions.values()]])
        low, high = float(np.min(combined)), float(np.max(combined))
        padding = max((high - low) * 0.03, 1e-8)
        ax.plot([low - padding, high + padding], [low - padding, high + padding], "k--", linewidth=1)
        for name, prediction in predictions.items():
            ax.scatter(actual[:, index], prediction[:, index], s=9, alpha=0.25, label=name)
        ax.set(
            xlabel=f"Reference {target}",
            ylabel="Model prediction",
            title=f"{reference_name}: {target} parity",
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f"parity_{target}_{reference_name.lower()}.png", dpi=180)
        plt.close(fig)


def _save_model_comparison_plot(
    metrics: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    reference_name: str,
) -> None:
    models = list(metrics)
    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for offset, target in enumerate(TARGET_COLUMNS):
        values = [metrics[model][target]["mape_percent"] for model in models]
        ax.bar(x + (offset - 1) * width, values, width, label=target)
    ax.set_xticks(x, models, rotation=20)
    ax.set_ylabel("Defined MAPE [%]")
    ax.set_title(f"{reference_name}: model comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / f"model_comparison_{reference_name.lower()}.png", dpi=180)
    plt.close(fig)


def save_comparison_report(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    reference_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Save metrics and comparable plots for one reference dataset."""
    if not predictions:
        raise ValueError("no models were selected for evaluation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = frame[list(TARGET_COLUMNS)].to_numpy(float)
    metrics = {name: regression_metrics(actual, pred) for name, pred in predictions.items()}
    payload = {
        "reference": reference_name,
        "rows": len(frame),
        "airfoils": int(frame["airfoil_id"].nunique()) if "airfoil_id" in frame else None,
        "models": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    _save_parity_plots(actual, predictions, output_dir, reference_name)
    _save_model_comparison_plot(metrics, output_dir, reference_name)
    return metrics


def evaluate_xfoil_test(
    dataset: AeroDataset,
    model_dir: str | Path,
    output_dir: str | Path,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate all saved models on the unseen-airfoil XFOIL test partition."""
    names = model_names or discover_models(model_dir)
    frame = load_test_reference(dataset, model_dir)
    predictions = _predict_saved_models(frame, model_dir, names)
    return save_comparison_report(frame, predictions, output_dir, reference_name="XFOIL")


def evaluate_cfd_reference(
    reference_csv: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare trained models with an independent CFD CSV.

    CFD data is evaluation-only and is never merged into the XFOIL training
    dataset. It uses the same 18 Kulfan geometry parameters and alpha/Re/Mach
    inputs as training, plus CFD Cl/Cd/Cm targets.
    """
    frame = pd.read_csv(reference_csv)
    required = set((*KULFAN_COLUMNS, "alpha_deg", "reynolds", "mach", *TARGET_COLUMNS))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CFD reference is missing columns: {sorted(missing)}")
    if (frame["reynolds"] <= 0).any() or (frame["cd"] <= 0).any():
        raise ValueError("CFD Reynolds numbers must be positive and CFD Cd must be strictly positive")
    names = model_names or discover_models(model_dir)
    predictions = _predict_saved_models(frame, model_dir, names)
    return save_comparison_report(frame, predictions, output_dir, reference_name="CFD")
