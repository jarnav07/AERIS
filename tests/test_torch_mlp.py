import numpy as np
import joblib
import pytest

from airfoil_ml.torch_mlp import TorchMLPRegressor

torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False


@pytest.mark.skipif(not torch_available, reason="torch is an optional dependency")
def test_torch_mlp_fits_and_predicts() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 12))
    y = rng.normal(size=(64, 3))
    model = TorchMLPRegressor(hidden_layers=(8, 4), max_epochs=5, seed=42)
    model.fit(x, y)
    assert model.predict(x[:2]).shape == (2, 3)
    assert len(model.loss_curve_) > 0


@pytest.mark.skipif(not torch_available, reason="torch is an optional dependency")
def test_torch_mlp_joblib_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 10))
    y = rng.normal(size=(48, 3))
    model = TorchMLPRegressor(hidden_layers=(8,), max_epochs=4, seed=42)
    model.fit(x, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert np.allclose(model.predict(x[:4]), loaded.predict(x[:4]))
