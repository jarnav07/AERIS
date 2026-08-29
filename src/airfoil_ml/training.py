"""Training orchestration for grouped, leakage-safe surrogate modelling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import AeroDataset, save_split_manifest
from .evaluation import regression_metrics
from .features import FeaturePreprocessor, fit_preprocessor, raw_input_matrix
from .models import ModelConfig, make_models


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


def train_from_kulfan_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    model_config: ModelConfig | None = None,
    only: list[str] | None = None,
    log_cd: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Train on the generated canonical Kulfan/XFOIL dataset."""
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame = pd.read_csv(csv_path)
    
    # Exclude analysis_confidence, airfoil_id, and split features vs targets
    feature_cols = [
        *[f"kulfan_upper_{i}" for i in range(8)],
        *[f"kulfan_lower_{i}" for i in range(8)],
        "kulfan_LE_weight", "kulfan_TE_thickness",
        "alpha", "Re", "mach", "n_crit", "xtr_upper", "xtr_lower"
    ]
    target_cols = [
        "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr",
        *[c for c in frame.columns if "bl_" in c]
    ]
    
    # Split by airfoil_id
    all_ids = np.array(sorted(frame["airfoil_id"].unique()))
    rng = np.random.default_rng(seed)
    ids = rng.permutation(all_ids)
    
    n_test = max(1, int(round(len(ids) * test_fraction)))
    n_val = max(1, int(round(len(ids) * validation_fraction)))
    if n_test + n_val >= len(ids):
        n_test, n_val = 1, 1
    
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    
    split = {
        "train": np.flatnonzero(frame.airfoil_id.isin(train_ids)),
        "validation": np.flatnonzero(frame.airfoil_id.isin(val_ids)),
        "test": np.flatnonzero(frame.airfoil_id.isin(test_ids)),
        "train_airfoils": train_ids,
        "validation_airfoils": val_ids,
        "test_airfoils": test_ids,
    }
    
    save_split_manifest(output_dir / "split_manifest.json", split, seed)
    
    train_x = frame.iloc[split["train"]][feature_cols].to_numpy(float)
    val_x = frame.iloc[split["validation"]][feature_cols].to_numpy(float)
    test_x = frame.iloc[split["test"]][feature_cols].to_numpy(float)
    
    train_y = frame.iloc[split["train"]][target_cols].to_numpy(float)
    val_y = frame.iloc[split["validation"]][target_cols].to_numpy(float)
    test_y = frame.iloc[split["test"]][target_cols].to_numpy(float)
    
    # log transform log_cd uses column index 1 which corresponds to CD
    train_labels = _transform_labels(train_y, log_cd)
    
    # We pass n_geometry_points=0 because we don't have separate geometric and flow scaling here,
    # the FeaturePreprocessor might expect raw_input_matrix but we can just use StandardScaler.
    # Actually, FeaturePreprocessor in features.py uses np.column_stack to log10(reynolds).
    # Since we use train_from_kulfan_csv, Re is at index 19 (18 kulfan + alpha). 
    # Let's bypass FeaturePreprocessor's transform_inputs logic and just use it as a container.
    processor = FeaturePreprocessor(
        input_scaler=None,
        target_scaler=None,
        n_geometry_points=0,
    )
    # Re-implement fitting directly
    from sklearn.preprocessing import StandardScaler
    # Log transform Re (index 19)
    train_x_scaled_base = train_x.copy()
    train_x_scaled_base[:, 19] = np.log10(train_x_scaled_base[:, 19])
    
    processor.input_scaler = StandardScaler().fit(train_x_scaled_base)
    processor.target_scaler = StandardScaler().fit(train_labels)
    processor.save(output_dir / "preprocessor.joblib")
    
    def transform_x(x: np.ndarray) -> np.ndarray:
        xc = x.copy()
        xc[:, 19] = np.log10(xc[:, 19])
        return processor.input_scaler.transform(xc)

    train_xs = transform_x(train_x)
    val_xs = transform_x(val_x)
    test_xs = transform_x(test_x)
    
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
        
        # evaluation metrics expect actual and predicted.
        # we can just pass them as they are, evaluation.py expects CL, CD, CM in indices 0,1,2.
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
