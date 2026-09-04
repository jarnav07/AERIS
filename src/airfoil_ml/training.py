"""Training orchestration for grouped, leakage-safe surrogate modelling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import aerosandbox as asb
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import grouped_split_by_id, save_split_manifest
from .evaluation import regression_metrics, save_evaluation_plots
from .features import FeaturePreprocessor
from .models import ModelConfig, make_models

_CHORD_STATIONS = (0.15, 0.30, 0.45, 0.60, 0.75)
GEOMETRY_FEATURE_COLUMNS = [
    "geom_max_thickness", "geom_max_camber", "geom_le_radius", "geom_te_angle", "geom_area", "geom_perimeter",
    # Pitching moment depends on the pressure distribution over the whole chord, not
    # just a single max-thickness/max-camber scalar, so sample the thickness and
    # camber lines at a handful of chordwise stations too.
    *[f"geom_thickness_{int(x * 100)}" for x in _CHORD_STATIONS],
    *[f"geom_camber_{int(x * 100)}" for x in _CHORD_STATIONS],
]
KULFAN_FEATURE_COLUMNS = [
    *[f"kulfan_upper_{i}" for i in range(8)],
    *[f"kulfan_lower_{i}" for i in range(8)],
    "kulfan_LE_weight", "kulfan_TE_thickness",
    "alpha", "Re", "mach", "n_crit", "xtr_upper", "xtr_lower",
    *GEOMETRY_FEATURE_COLUMNS,
]
RE_COLUMN_INDEX = KULFAN_FEATURE_COLUMNS.index("Re")
_KULFAN_VECTOR_COLUMNS = [
    *[f"kulfan_upper_{i}" for i in range(8)],
    *[f"kulfan_lower_{i}" for i in range(8)],
    "kulfan_LE_weight", "kulfan_TE_thickness",
]


def _geometry_features_for_airfoil(kulfan_row: np.ndarray) -> np.ndarray:
    """Derive engineered geometry descriptors from one airfoil's 18 Kulfan coefficients.

    These give the models a more direct aerodynamic signal (thickness, camber, LE
    radius, TE angle, area, perimeter) than the raw CST basis coefficients, which are
    a comparatively opaque geometric representation. Falls back to zeros for
    degenerate geometry (e.g. the all-zero CYLINDER sanity-check airfoil, whose
    zero-thickness shape makes AeroSandbox's LE_radius softness parameter invalid).
    """
    airfoil = asb.KulfanAirfoil(
        name="_geom_feature_probe",
        upper_weights=kulfan_row[0:8],
        lower_weights=kulfan_row[8:16],
        leading_edge_weight=kulfan_row[16],
        TE_thickness=kulfan_row[17],
    )
    try:
        stations = np.array(_CHORD_STATIONS)
        return np.concatenate([
            [
                float(airfoil.max_thickness()),
                float(airfoil.max_camber()),
                float(airfoil.LE_radius()),
                float(airfoil.TE_angle()),
                float(airfoil.area()),
                float(airfoil.perimeter()),
            ],
            np.asarray(airfoil.local_thickness(stations), dtype=float),
            np.asarray(airfoil.local_camber(stations), dtype=float),
        ])
    except Exception:
        return np.zeros(len(GEOMETRY_FEATURE_COLUMNS))


def add_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach engineered geometry columns, computed once per unique airfoil_id.

    Geometry only depends on the Kulfan coefficients, which are constant across every
    row (alpha/Re case) belonging to the same airfoil_id, so this dedupes before the
    (comparatively expensive, ~0.5ms each) AeroSandbox geometry calls and broadcasts
    the result back rather than recomputing per row.
    """
    if all(col in frame.columns for col in GEOMETRY_FEATURE_COLUMNS):
        return frame

    # Drop any partial overlap (e.g. a CSV saved by an older version of this function
    # with a subset of today's GEOMETRY_FEATURE_COLUMNS) so the merge below can't
    # produce pandas' _x/_y suffixed duplicate columns.
    frame = frame.drop(columns=[c for c in GEOMETRY_FEATURE_COLUMNS if c in frame.columns])

    unique = frame.drop_duplicates(subset="airfoil_id")[["airfoil_id", *_KULFAN_VECTOR_COLUMNS]]
    kulfan_matrix = unique[_KULFAN_VECTOR_COLUMNS].to_numpy(float)
    geometry = np.stack([_geometry_features_for_airfoil(row) for row in kulfan_matrix])
    geometry_by_airfoil = pd.DataFrame(geometry, columns=GEOMETRY_FEATURE_COLUMNS)
    geometry_by_airfoil.insert(0, "airfoil_id", unique["airfoil_id"].to_numpy())

    return frame.merge(geometry_by_airfoil, on="airfoil_id", how="left")


_MIN_CD_FOR_LOG = 1e-6


def _transform_labels(targets: np.ndarray, log_cd: bool) -> np.ndarray:
    """Transform training labels; log-Cd makes the model minimize relative Cd error."""
    transformed = np.asarray(targets, dtype=float).copy()
    if log_cd:
        if np.any(transformed[:, 1] < -_MIN_CD_FOR_LOG):
            raise ValueError("Cd must be positive for log transformation")
        # A handful of converged-but-vanishingly-low-drag XFOIL cases round to
        # exactly 0.0 in the generated dataset (observed: 1 row out of 610k),
        # which log(0) can't represent. Floor rather than error, since these are
        # genuine near-zero-drag solutions, not corrupted data.
        transformed[:, 1] = np.log(np.maximum(transformed[:, 1], _MIN_CD_FOR_LOG))
    return transformed


def _inverse_labels(processor: FeaturePreprocessor, scaled_prediction: np.ndarray, log_cd: bool) -> np.ndarray:
    """Undo the target scaler and, for log-Cd models, exponentiate Cd back to counts."""
    physical = processor.inverse_targets(scaled_prediction)
    if log_cd:
        physical[:, 1] = np.exp(physical[:, 1])
    return physical


def _select_columns(frame: pd.DataFrame, columns: list[str], description: str) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"training CSV is missing required {description} columns: {missing}")
    return frame[columns].to_numpy(float)


def _target_columns(frame: pd.DataFrame) -> list[str]:
    return [
        "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr",
        *[c for c in frame.columns if "bl_" in c],
    ]


def transform_kulfan_inputs(processor: FeaturePreprocessor, x: np.ndarray) -> np.ndarray:
    """Apply the same log10(Re) + StandardScaler transform used at training time."""
    xc = np.asarray(x, dtype=float).copy()
    xc[:, RE_COLUMN_INDEX] = np.log10(xc[:, RE_COLUMN_INDEX])
    return processor.input_scaler.transform(xc)


def train_from_kulfan_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    test_airfoils: list[str] | None = None,
    model_config: ModelConfig | None = None,
    only: list[str] | None = None,
    log_cd: bool = False,
    target_cols: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Train on the generated canonical Kulfan/XFOIL dataset.

    Features and targets are read directly as flat columns of the generated
    CSV (see ``training_data_generation.py``); scaling is fit here rather than
    through ``features.fit_preprocessor``, which targets the separate
    coordinate-file-based pipeline in ``data.py``.

    ``target_cols`` defaults to the full 197-column set (CL/CD/CM/Top_Xtr/Bot_Xtr
    plus every boundary-layer column); pass e.g. ``["CL", "CD", "CM"]`` to train a
    model dedicated to just the headline aero coefficients instead of splitting its
    capacity/loss budget across the 194 auxiliary boundary-layer targets too.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(csv_path)
    frame = add_geometry_features(frame)
    target_cols = target_cols if target_cols is not None else _target_columns(frame)

    split = grouped_split_by_id(
        frame,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
        test_airfoils=test_airfoils,
    )
    save_split_manifest(output_dir / "split_manifest.json", split, seed)

    train_frame, val_frame, test_frame = (
        frame.iloc[split["train"]], frame.iloc[split["validation"]], frame.iloc[split["test"]]
    )
    train_x = _select_columns(train_frame, KULFAN_FEATURE_COLUMNS, "feature")
    val_x = _select_columns(val_frame, KULFAN_FEATURE_COLUMNS, "feature")
    test_x = _select_columns(test_frame, KULFAN_FEATURE_COLUMNS, "feature")

    train_y = _select_columns(train_frame, target_cols, "target")
    val_y = _select_columns(val_frame, target_cols, "target")
    test_y = _select_columns(test_frame, target_cols, "target")

    train_labels = _transform_labels(train_y, log_cd)

    train_x_log_re = train_x.copy()
    train_x_log_re[:, RE_COLUMN_INDEX] = np.log10(train_x_log_re[:, RE_COLUMN_INDEX])

    # n_geometry_points=0: this preprocessor is used only as a container for the
    # two fitted StandardScalers, not through FeaturePreprocessor.transform_inputs.
    processor = FeaturePreprocessor(
        input_scaler=StandardScaler().fit(train_x_log_re),
        target_scaler=StandardScaler().fit(train_labels),
        n_geometry_points=0,
    )
    processor.save(output_dir / "preprocessor.joblib")

    train_xs = transform_kulfan_inputs(processor, train_x)
    val_xs = transform_kulfan_inputs(processor, val_x)
    test_xs = transform_kulfan_inputs(processor, test_x)

    train_ys = processor.target_scaler.transform(train_labels)
    val_ys = processor.target_scaler.transform(_transform_labels(val_y, log_cd))
    test_ys = processor.target_scaler.transform(_transform_labels(test_y, log_cd))

    results_path = output_dir / "metrics.json"
    results: dict[str, dict[str, dict[str, float]]] = {}
    if results_path.exists():
        # Preserve metrics for models not retrained this run (e.g. a `--only`
        # rerun of a single model) instead of clobbering the whole file.
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

        # regression_metrics expects targets in physical units, in target_cols order.
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
    (output_dir / "training_config.json").write_text(
        json.dumps({"seed": seed, "log_cd": log_cd, "target_cols": target_cols, "model_config": asdict(config)}, indent=2) + "\n",
        encoding="utf-8",
    )

    return results


def _read_target_cols(model_dir: Path, frame: pd.DataFrame) -> list[str]:
    """The target columns a model was trained on, from training_config.json if recorded.

    Falls back to the full 197-column set for model directories saved before
    ``target_cols`` was recorded (train_from_kulfan_csv always used the full set then).
    """
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        recorded = json.loads(config_path.read_text(encoding="utf-8")).get("target_cols")
        if recorded is not None:
            return recorded
    return _target_columns(frame)


def load_model_bundle(model_dir: str | Path, model_name: str = "mlp") -> tuple[object, FeaturePreprocessor]:
    model_dir = Path(model_dir)
    return joblib.load(model_dir / f"{model_name}.joblib"), FeaturePreprocessor.load(model_dir / "preprocessor.joblib")


def evaluate_ensemble(
    csv_path: str | Path,
    model_dir: str | Path,
    model_names: list[str],
    output_dir: str | Path = "results/evaluation",
    log_cd: bool | None = None,
    max_polar_plots: int | None = None,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Evaluate an unweighted (or explicitly weighted) average ensemble of already-trained models.

    All ``model_names`` must live in the same ``model_dir`` (so they share one
    ``split_manifest.json`` test partition and ``training_config.json`` log_cd
    setting) but each keeps its own fitted preprocessor: predictions are
    inverse-transformed to physical units per model before averaging, since
    averaging in each model's own scaled space would not be meaningful across
    independently-fit scalers.
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no split manifest found at {manifest_path}; train a model first")
    test_airfoils = set(json.loads(manifest_path.read_text(encoding="utf-8"))["test_airfoils"])

    if log_cd is None:
        config_path = model_dir / "training_config.json"
        log_cd = bool(json.loads(config_path.read_text(encoding="utf-8"))["log_cd"]) if config_path.exists() else False

    frame = pd.read_csv(csv_path)
    frame = add_geometry_features(frame)
    test_frame = frame[frame["airfoil_id"].astype(str).isin(test_airfoils)].reset_index(drop=True)
    if test_frame.empty:
        raise ValueError(f"none of the test airfoils recorded in {manifest_path} are present in {csv_path}")

    test_y = _select_columns(test_frame, _read_target_cols(model_dir, frame), "target")
    weights = weights or [1.0] * len(model_names)
    if len(weights) != len(model_names):
        raise ValueError("weights must have the same length as model_names")

    weighted_sum = None
    for name, weight in zip(model_names, weights):
        model, processor = load_model_bundle(model_dir, name)
        test_x = _select_columns(test_frame, KULFAN_FEATURE_COLUMNS, "feature")
        pred = _inverse_labels(processor, model.predict(transform_kulfan_inputs(processor, test_x)), log_cd)
        weighted_sum = weight * pred if weighted_sum is None else weighted_sum + weight * pred
    test_pred = weighted_sum / sum(weights)

    metrics = regression_metrics(test_y, test_pred)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    plot_frame = test_frame[["airfoil_id", "alpha", "Re"]].rename(columns={"alpha": "alpha_deg", "Re": "reynolds"})
    plot_frame["cl"], plot_frame["cd"], plot_frame["cm"] = test_y[:, 0], test_y[:, 1], test_y[:, 2]
    save_evaluation_plots(plot_frame, test_y[:, :3], test_pred[:, :3], output_dir, title_prefix="ensemble", max_polar_plots=max_polar_plots)

    return metrics


def fit_stacking_ensemble(
    csv_path: str | Path,
    model_dir: str | Path,
    model_names: list[str],
    output_dir: str | Path = "results/evaluation",
    log_cd: bool | None = None,
    max_polar_plots: int | None = None,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Fit and evaluate a per-target learned-weight stacking ensemble over CL/CD/CM.

    Unlike ``evaluate_ensemble``'s single scalar weight per model (applied
    identically to every target), this fits one non-negative-weighted linear
    combination *per target* (via ``Ridge(positive=True)``) on the validation
    split, then scores it on the held-out test split. A model that is strong on
    CL but weak on CM, say, can end up with a high CL weight and a low CM weight
    instead of one compromise weight for both. Weights are saved to
    ``model_dir/stacking_weights.json`` (keyed by target, model name, and
    "intercept") so the ensemble is a reproducible artifact, not just a script run.
    """
    from sklearn.linear_model import Ridge

    model_dir = Path(model_dir)
    manifest_path = model_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no split manifest found at {manifest_path}; train a model first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_airfoils = manifest["test_airfoils"]

    if log_cd is None:
        config_path = model_dir / "training_config.json"
        log_cd = bool(json.loads(config_path.read_text(encoding="utf-8"))["log_cd"]) if config_path.exists() else False

    frame = pd.read_csv(csv_path)
    frame = add_geometry_features(frame)
    # Reproduce the exact train/validation/test split used at training time (same
    # seed default and fixed test_airfoils) so validation rows used to fit the
    # stacking weights never overlap the held-out test partition.
    split = grouped_split_by_id(frame, seed=manifest.get("seed", 42), test_airfoils=test_airfoils)
    val_frame, test_frame = frame.iloc[split["validation"]], frame.iloc[split["test"]]
    target_cols = _read_target_cols(model_dir, frame)[:3]

    def _predict_all(sub_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        x = _select_columns(sub_frame, KULFAN_FEATURE_COLUMNS, "feature")
        preds = {}
        for name in model_names:
            model, processor = load_model_bundle(model_dir, name)
            preds[name] = _inverse_labels(processor, model.predict(transform_kulfan_inputs(processor, x)), log_cd)
        return preds

    val_y = _select_columns(val_frame, target_cols, "target")
    test_y = _select_columns(test_frame, target_cols, "target")
    val_preds = _predict_all(val_frame)
    test_preds = _predict_all(test_frame)

    stacking_weights: dict[str, dict[str, float]] = {}
    test_pred = np.zeros_like(test_y)
    for i, target in enumerate(("cl", "cd", "cm")):
        x_val = np.column_stack([val_preds[name][:, i] for name in model_names])
        reg = Ridge(alpha=alpha, positive=True)
        reg.fit(x_val, val_y[:, i])
        stacking_weights[target] = {name: float(w) for name, w in zip(model_names, reg.coef_)}
        stacking_weights[target]["intercept"] = float(reg.intercept_)
        x_test = np.column_stack([test_preds[name][:, i] for name in model_names])
        test_pred[:, i] = reg.predict(x_test)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "stacking_weights.json").write_text(
        json.dumps({"model_names": model_names, "weights": stacking_weights}, indent=2) + "\n", encoding="utf-8"
    )

    metrics = regression_metrics(test_y, test_pred)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    plot_frame = test_frame[["airfoil_id", "alpha", "Re"]].rename(columns={"alpha": "alpha_deg", "Re": "reynolds"})
    plot_frame["cl"], plot_frame["cd"], plot_frame["cm"] = test_y[:, 0], test_y[:, 1], test_y[:, 2]
    save_evaluation_plots(plot_frame, test_y, test_pred, output_dir, title_prefix="stacking_ensemble", max_polar_plots=max_polar_plots)

    return metrics


def evaluate_kulfan_model(
    csv_path: str | Path,
    model_dir: str | Path,
    model_name: str = "mlp",
    output_dir: str | Path = "results/evaluation",
    log_cd: bool | None = None,
    max_polar_plots: int | None = None,
) -> dict[str, float]:
    """Evaluate a trained canonical-pipeline model and write parity/error plots.

    Scores the model against the exact test-airfoil identities recorded in
    ``model_dir/split_manifest.json`` at training time, rather than drawing a
    fresh split, so evaluation can never accidentally score rows the model was
    trained on. ``log_cd`` defaults to whatever ``model_dir/training_config.json``
    recorded for this model, so callers don't have to remember and re-pass it.
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no split manifest found at {manifest_path}; train a model first")
    test_airfoils = set(json.loads(manifest_path.read_text(encoding="utf-8"))["test_airfoils"])

    if log_cd is None:
        config_path = model_dir / "training_config.json"
        log_cd = bool(json.loads(config_path.read_text(encoding="utf-8"))["log_cd"]) if config_path.exists() else False

    frame = pd.read_csv(csv_path)
    frame = add_geometry_features(frame)
    test_frame = frame[frame["airfoil_id"].astype(str).isin(test_airfoils)].reset_index(drop=True)
    if test_frame.empty:
        raise ValueError(f"none of the test airfoils recorded in {manifest_path} are present in {csv_path}")

    model, processor = load_model_bundle(model_dir, model_name)
    test_x = _select_columns(test_frame, KULFAN_FEATURE_COLUMNS, "feature")
    test_y = _select_columns(test_frame, _read_target_cols(model_dir, frame), "target")
    test_pred = _inverse_labels(processor, model.predict(transform_kulfan_inputs(processor, test_x)), log_cd)

    metrics = regression_metrics(test_y, test_pred)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    plot_frame = test_frame[["airfoil_id", "alpha", "Re"]].rename(columns={"alpha": "alpha_deg", "Re": "reynolds"})
    plot_frame["cl"], plot_frame["cd"], plot_frame["cm"] = test_y[:, 0], test_y[:, 1], test_y[:, 2]
    save_evaluation_plots(plot_frame, test_y[:, :3], test_pred[:, :3], output_dir, title_prefix=model_name, max_polar_plots=max_polar_plots)

    return metrics
