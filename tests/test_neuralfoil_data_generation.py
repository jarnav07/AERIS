"""Tests for the NeuralFoil-style stochastic data generation pipeline.

These tests were originally written against a planned ``neuralfoil_data_generation``
module that was never created.  They now target the actual implementation in
``airfoil_ml.training_data_generation``, using the correct public API names:

* ``TrainingDataConfig``   (was ``NeuralFoilSamplingConfig``)
* ``sample_airfoil``       (was ``sample_kulfan_airfoil``)
* ``sample_operating_point`` (unchanged)
"""
import numpy as np
import pytest

from airfoil_ml.training_data_generation import (
    TrainingDataConfig,
    sample_airfoil,
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
    config = TrainingDataConfig()
    point = sample_operating_point(rng, config)

    assert point["alphas"].shape == (7,)
    assert point["reynolds"] > 0
    assert 0.0 <= point["n_crit"] <= 18.0
    assert 0.0 <= point["xtr_upper"] <= 1.0
    assert 0.0 <= point["xtr_lower"] <= 1.0


def test_operating_point_alpha_offset_is_shared() -> None:
    """All alphas in a sweep must share the same random offset.

    Draw two operating points from the same RNG state and verify that the
    *differences* between consecutive alpha values in each sweep match the
    original grid spacing (i.e. the offset shifts the whole grid rigidly).
    """
    config = TrainingDataConfig()
    grid = np.asarray(config.alpha_grid, dtype=float)
    expected_diffs = np.diff(grid)

    rng = np.random.default_rng(42)
    point = sample_operating_point(rng, config)
    actual_diffs = np.diff(point["alphas"])

    np.testing.assert_allclose(actual_diffs, expected_diffs, atol=1e-10)


def test_kulfan_sampling_uses_convex_parent_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    # The implementation needs AeroSandbox for the final object construction;
    # patch the module-level constructor so this unit test isolates the sampler.
    import airfoil_ml.training_data_generation as module

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
    config = module.TrainingDataConfig(n_airfoils_to_combine=3)
    sampled = sample_airfoil(db, np.random.default_rng(0), config)

    assert sampled.upper_weights.shape == (8,)
    assert sampled.lower_weights.shape == (8,)
    assert sampled.leading_edge_weight == pytest.approx(0.2, abs=1e-4)
    assert sampled.TE_thickness == pytest.approx(0.1, abs=1e-4)


def test_te_thickness_is_non_negative() -> None:
    """TE_thickness must never be negative after covariance perturbation."""
    import airfoil_ml.training_data_generation as module

    class FakeKulfan:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def scale(self, chord, scale):
            return self

    class FakeASB:
        KulfanAirfoil = FakeKulfan

    # Force a large negative deviation on TE_thickness slot (index 17).
    huge_cov = np.zeros((18, 18))
    huge_cov[17, 17] = 1e6  # will draw a very large deviation

    parents = (_fake_airfoil(),) * 3
    db = module.KulfanDatabase(
        airfoils=parents,
        mean=np.zeros(18),
        covariance=huge_cov,
    )

    import airfoil_ml.training_data_generation as real_module
    import unittest.mock as mock

    config = module.TrainingDataConfig(n_airfoils_to_combine=3)
    # Run many samples; TE_thickness must never be negative.
    rng = np.random.default_rng(99)

    with mock.patch.object(real_module, "asb", FakeASB):
        for _ in range(50):
            sampled = real_module.sample_airfoil(db, rng, config)
            assert sampled.TE_thickness >= 0.0, "TE_thickness went negative"
