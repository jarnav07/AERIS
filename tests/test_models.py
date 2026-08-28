import numpy as np

from airfoil_ml.models import ModelConfig, make_models


def test_common_model_set_is_present():
    models = make_models(ModelConfig(seed=1, mlp_epochs=2, mlp_hidden_layers=(8,)))
    assert {"ridge", "extra_trees", "hist_gb"}.issubset(models)


def test_models_predict_three_targets():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(24, 21))
    y = rng.normal(size=(24, 3))
    models = make_models(ModelConfig(seed=1, mlp_epochs=2, mlp_hidden_layers=(8,)))
    for model in models.values():
        model.fit(x, y)
        assert model.predict(x[:2]).shape == (2, 3)
