"""Candidate regression models for the aerodynamic surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    # (256, 128, 64, 32) beat (128, 64, 32) and every other depth/width tried in
    # scripts/tune_models.py's search on a 12k-airfoil subsample of the full
    # generated dataset (score 0.1158 vs the next-best 0.1184); batch_size=64
    # (models.py's mlp below) also won that search outright, contrary to the
    # assumption that it was too small for a 600k+-row dataset.
    hidden_layers: tuple[int, ...] = (256, 128, 64, 32)
    max_iter: int = 600
    early_stopping: bool = True


def make_models(config: ModelConfig | None = None) -> dict[str, Any]:
    config = config or ModelConfig()
    models: dict[str, Any] = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            # min_samples_leaf=2 is fine for a handful of targets, but this
            # model is trained multi-output against 197 targets (CL/CD/CM/
            # Top_Xtr/Bot_Xtr + 192 boundary-layer columns): every leaf stores
            # a per-target mean, so a near-unbounded leaf count at
            # dataset-generation scale (hundreds of thousands of training
            # rows) multiplies out to tens of GB across 250 trees and can
            # exhaust host memory. max_leaf_nodes is an explicit hard cap on
            # per-tree memory regardless of how large the training set grows
            # (12000 leaves x 197 targets x 8 bytes x 250 trees ~= 4.7GB,
            # still well within budget); min_samples_leaf=5/max_leaf_nodes=
            # 12000 won scripts/tune_models.py's search over the previous
            # min_samples_leaf=25/max_leaf_nodes=5000 (score 0.2286 vs 0.2613).
            min_samples_leaf=5,
            max_leaf_nodes=12000,
            max_features=0.8,
            random_state=config.seed,
            n_jobs=-1,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=config.hidden_layers,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=config.max_iter,
            early_stopping=config.early_stopping,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=config.seed,
        ),
        "hist_gb": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                # learning_rate=0.1/max_leaf_nodes=127/max_iter=250 won
                # scripts/tune_models.py's search over the previous
                # 0.08/31/200 (score 0.1868 vs 0.2105).
                max_iter=250,
                learning_rate=0.1,
                max_leaf_nodes=127,
                early_stopping=True,
                n_iter_no_change=25,
                validation_fraction=0.1,
                random_state=config.seed,
            )
        ),
    }
    try:
        from .torch_mlp import TorchMLPRegressor
    except ImportError:
        return models
    models["mlp_torch"] = TorchMLPRegressor(
        hidden_layers=config.hidden_layers,
        max_epochs=config.max_iter,
        seed=config.seed,
        early_stopping=config.early_stopping,
    )
    return models
