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
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    max_iter: int = 600
    early_stopping: bool = True


def make_models(config: ModelConfig | None = None) -> dict[str, Any]:
    config = config or ModelConfig()
    models: dict[str, Any] = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.7,
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
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=31,
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
