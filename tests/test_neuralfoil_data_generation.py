import numpy as np
import pytest

from airfoil_ml.neuralfoil_data_generation import (
    NeuralFoilSamplingConfig,
    sample_kulfan_airfoil,
    sample_operating_point,
)


def _fake_airfoil():
    class FakeAirfoil:
        upper_weights = np.arange(8, dtype=float)
        lower_weights = -np.arange(8, dtype=float)
        leading_edge_weight = 0.2
        TE_thickness = 0.1

    return FakeAirfoil()


def test_operating_point_matches_neuralfoil_ranges() -> None:
    rng = np.random.default_rng(123)
    config = NeuralFoilSamplingConfig()
    point = sample_operating_point(rng, config)

    assert point["alphas"].shape == (7,)
    assert point["reynolds"] > 0
    assert 0.0 <= point["n_crit"] <= 18.0
    assert 0.0 <= point["xtr_upper"] <= 1.0
    assert 0.0 <= point["xtr_lower"] <= 1.0


def test_kulfan_sampling_uses_convex_parent_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    # The implementation needs AeroSandbox for the final object construction;
    # patch the module-level constructor so this unit test isolates the sampler.
    import airfoil_ml.neuralfoil_data_generation as module

    class FakeKulfan:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def scale(self, chord, scale):
            self.scale_factor = scale
            return self

    class FakeASB:
        KulfanAirfoil = FakeKulfan

    monkeypatch.setattr(module, "asb", FakeASB)
    parents = (_fake_airfoil(), _fake_airfoil(), _fake_airfoil())
    db = module.KulfanDatabase(
        airfoils=parents,
        mean=np.zeros(18),
        covariance=np.eye(18) * 1e-12,
    )
    config = module.NeuralFoilSamplingConfig(n_airfoils_to_combine=3)
    sampled = sample_kulfan_airfoil(db, np.random.default_rng(0), config)

    assert sampled.upper_weights.shape == (8,)
    assert sampled.lower_weights.shape == (8,)
    assert sampled.leading_edge_weight == pytest.approx(0.2, abs=1e-4)
    assert sampled.TE_thickness == pytest.approx(0.1, abs=1e-4)
