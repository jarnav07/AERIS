"""Candidate surrogate models for the shared XFOIL training experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


@dataclass(frozen=True)
class ModelConfig:
    """Shared hyperparameters for the model comparison."""

    seed: int = 42
    mlp_hidden_layers: tuple[int, ...] = (128, 64, 32)
    mlp_epochs: int = 250


def make_models(config: ModelConfig | None = None) -> dict[str, Any]:
    """Create the fixed comparison set.

    Every model receives exactly the same scaled inputs, targets, split, and
    training data. The models are intentionally different in inductive bias:
    linear, randomized tree ensemble, gradient boosting, and neural network.
    """
    config = config or ModelConfig()
    models: dict[str, Any] = {
        "ridge": Ridge(alpha=1.0),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=config.seed,
            n_jobs=-1,
        ),
        "hist_gb": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=250,
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
        hidden_layers=config.mlp_hidden_layers,
        max_epochs=config.mlp_epochs,
        seed=config.seed,
        early_stopping=True,
    )
    return models
