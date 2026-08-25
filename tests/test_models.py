import numpy as np

from airfoil_ml.models import ModelConfig, make_models


def test_models_are_multi_output_regressors() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(24, 18))
    y = rng.normal(size=(24, 3))
    models = make_models(ModelConfig(hidden_layers=(8,), max_iter=5, early_stopping=False))
    for model in models.values():
        model.fit(x, y)
        assert model.predict(x[:2]).shape == (2, 3)
