#!/usr/bin/env python3
"""Hyperparameter search for mlp/random_forest/hist_gb, run on an airfoil subsample.

Full-dataset fits take 15-30 minutes each (measured on the ~610k-row / ~100k-airfoil
generated dataset), so a grid search directly on the full data is impractical. This
searches on a random, grouped-by-airfoil subsample instead (same leakage-safe split
logic as training, via ``grouped_split_by_id``) so each candidate fits in a couple of
minutes, and scores only CL/CD/CM (not the other 194 targets in the full production
models) since that is what the search is optimizing for -- ``hist_gb`` in particular
fits one model per target, so this is a ~65x speedup for it versus scoring all 197.

The winning config per model is printed and written to ``--output`` as JSON; apply it
by hand to ``ModelConfig``/``make_models`` in ``src/airfoil_ml/models.py`` (or extend
those with the swept parameters) and rerun ``scripts/train.py`` on the full dataset.
This script deliberately does not retrain on the full dataset itself -- the point is
a cheap search, not a production run.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from airfoil_ml.data import grouped_split_by_id
from airfoil_ml.evaluation import regression_metrics
from airfoil_ml.features import FeaturePreprocessor
from airfoil_ml.training import (
    RE_COLUMN_INDEX,
    KULFAN_FEATURE_COLUMNS,
    _inverse_labels,
    _select_columns,
    _transform_labels,
    add_geometry_features,
    transform_kulfan_inputs,
)

_TUNE_TARGET_COLUMNS = ["CL", "CD", "CM"]


def _sample_by_airfoil(frame: pd.DataFrame, n_airfoils: int, seed: int) -> pd.DataFrame:
    ids = frame["airfoil_id"].astype(str).unique()
    if len(ids) <= n_airfoils:
        return frame
    rng = np.random.default_rng(seed)
    sampled = set(rng.choice(ids, size=n_airfoils, replace=False))
    return frame[frame["airfoil_id"].astype(str).isin(sampled)].reset_index(drop=True)


def _fit_and_score(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    log_cd: bool,
) -> tuple[dict[str, dict[str, float]], float, float]:
    train_labels = _transform_labels(train_y, log_cd)
    train_x_log_re = train_x.copy()
    train_x_log_re[:, RE_COLUMN_INDEX] = np.log10(train_x_log_re[:, RE_COLUMN_INDEX])
    processor = FeaturePreprocessor(
        input_scaler=StandardScaler().fit(train_x_log_re),
        target_scaler=StandardScaler().fit(train_labels),
        n_geometry_points=0,
    )
    train_xs = transform_kulfan_inputs(processor, train_x)
    train_ys = processor.target_scaler.transform(train_labels)
    val_xs = transform_kulfan_inputs(processor, val_x)

    start = time.time()
    model.fit(train_xs, train_ys)
    fit_seconds = time.time() - start

    val_pred = _inverse_labels(processor, model.predict(val_xs), log_cd)
    metrics = regression_metrics(val_y, val_pred)
    mean_abs = np.maximum(np.abs(val_y[:, :3]).mean(axis=0), 1e-8)
    relative_mae = np.array([metrics[t]["mae"] for t in ("cl", "cd", "cm")]) / mean_abs
    score = float(relative_mae.mean())
    return metrics, score, fit_seconds


def _mlp_candidates(seed: int) -> list[dict[str, Any]]:
    grid = []
    for hidden, batch_size, lr in product(
        [(128, 64, 32), (256, 128, 64), (256, 128, 64, 32)],
        [64, 512, 2048],
        [1e-3, 3e-3],
    ):
        grid.append({"hidden_layer_sizes": hidden, "batch_size": batch_size, "learning_rate_init": lr})
    return grid


def _random_forest_candidates() -> list[dict[str, Any]]:
    return [
        {"n_estimators": 250, "min_samples_leaf": 25, "max_leaf_nodes": 5000, "max_features": 0.7},
        {"n_estimators": 250, "min_samples_leaf": 10, "max_leaf_nodes": 8000, "max_features": 0.7},
        {"n_estimators": 400, "min_samples_leaf": 15, "max_leaf_nodes": 8000, "max_features": 0.8},
        {"n_estimators": 250, "min_samples_leaf": 5, "max_leaf_nodes": 12000, "max_features": 0.8},
    ]


def _hist_gb_candidates() -> list[dict[str, Any]]:
    return [
        {"learning_rate": 0.08, "max_leaf_nodes": 31, "max_iter": 200},
        {"learning_rate": 0.10, "max_leaf_nodes": 63, "max_iter": 300},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400},
        {"learning_rate": 0.10, "max_leaf_nodes": 127, "max_iter": 250},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/generated/training_data.csv")
    parser.add_argument("--output", default="results/tuning.json")
    parser.add_argument("--n-airfoils", type=int, default=12000, help="subsample size for the search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-cd", action="store_true")
    parser.add_argument("--models", nargs="+", default=["mlp", "random_forest", "hist_gb"])
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    frame = add_geometry_features(frame)
    frame = _sample_by_airfoil(frame, args.n_airfoils, args.seed)
    print(f"Tuning on {frame['airfoil_id'].nunique()} airfoils, {len(frame)} rows")

    split = grouped_split_by_id(frame, seed=args.seed)
    train_frame, val_frame = frame.iloc[split["train"]], frame.iloc[split["validation"]]
    train_x = _select_columns(train_frame, KULFAN_FEATURE_COLUMNS, "feature")
    val_x = _select_columns(val_frame, KULFAN_FEATURE_COLUMNS, "feature")
    train_y = _select_columns(train_frame, _TUNE_TARGET_COLUMNS, "target")
    val_y = _select_columns(val_frame, _TUNE_TARGET_COLUMNS, "target")

    results: dict[str, list[dict[str, Any]]] = {}

    if "mlp" in args.models:
        results["mlp"] = []
        for params in _mlp_candidates(args.seed):
            model = MLPRegressor(
                activation="relu", solver="adam", alpha=1e-4, max_iter=600,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=40,
                random_state=args.seed, **params,
            )
            metrics, score, fit_seconds = _fit_and_score(model, train_x, train_y, val_x, val_y, args.log_cd)
            entry = {"params": params, "score_mean_rel_mae": score, "fit_seconds": fit_seconds, "metrics": metrics}
            results["mlp"].append(entry)
            print(f"[mlp] {params} -> score={score:.4f} ({fit_seconds:.1f}s)")

    if "random_forest" in args.models:
        results["random_forest"] = []
        for params in _random_forest_candidates():
            model = RandomForestRegressor(random_state=args.seed, n_jobs=-1, **params)
            metrics, score, fit_seconds = _fit_and_score(model, train_x, train_y, val_x, val_y, args.log_cd)
            entry = {"params": params, "score_mean_rel_mae": score, "fit_seconds": fit_seconds, "metrics": metrics}
            results["random_forest"].append(entry)
            print(f"[random_forest] {params} -> score={score:.4f} ({fit_seconds:.1f}s)")

    if "hist_gb" in args.models:
        results["hist_gb"] = []
        for params in _hist_gb_candidates():
            model = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    early_stopping=True, n_iter_no_change=25, validation_fraction=0.1,
                    random_state=args.seed, **params,
                )
            )
            metrics, score, fit_seconds = _fit_and_score(model, train_x, train_y, val_x, val_y, args.log_cd)
            entry = {"params": params, "score_mean_rel_mae": score, "fit_seconds": fit_seconds, "metrics": metrics}
            results["hist_gb"].append(entry)
            print(f"[hist_gb] {params} -> score={score:.4f} ({fit_seconds:.1f}s)")

    output_path = args.output
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")

    print("\n=== Best per model (lowest mean relative MAE across CL/CD/CM) ===")
    for name, candidates in results.items():
        best = min(candidates, key=lambda c: c["score_mean_rel_mae"])
        print(f"{name}: {best['params']} -> {best['score_mean_rel_mae']:.4f}")


if __name__ == "__main__":
    main()
