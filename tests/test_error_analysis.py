import numpy as np
import pandas as pd
import pytest

from airfoil_ml.error_analysis import error_by_regime, percentage_metrics


def test_percentage_metrics_perfect_prediction() -> None:
    actual = np.array([[0.5, 0.01, -0.02], [1.0, 0.03, -0.05], [0.8, 0.02, -0.03]])
    metrics = percentage_metrics(actual, actual.copy())
    for name in ("cl", "cd", "cm", "ld"):
        assert metrics[name]["mape_percent"] == 0.0
        assert metrics[name]["mae"] == 0.0


def test_drag_metrics_use_engineering_units() -> None:
    actual = np.array([[0.5, 0.0100, -0.02], [1.0, 0.0300, -0.05]])
    predicted = np.array([[0.5, 0.0103, -0.02], [1.0, 0.0303, -0.05]])
    metrics = percentage_metrics(actual, predicted)
    # 0.0003 mean absolute error == 3 drag counts.
    assert metrics["cd"]["mae_drag_counts"] == pytest.approx(3.0)
    # Relative error on Cd near 0.01 is a few percent, not hundreds.
    assert 2.0 < metrics["cd"]["mape_percent"] < 4.0


def test_negative_drag_predictions_are_counted() -> None:
    actual = np.array([[0.5, 0.0100, -0.02], [1.0, 0.0300, -0.05], [0.6, 0.0200, -0.04]])
    predicted = np.array([[0.5, -0.0020, -0.02], [1.0, 0.0300, -0.05], [0.6, 0.0200, -0.04]])
    metrics = percentage_metrics(actual, predicted)
    assert metrics["cd"]["n_negative_predictions"] == 1
    # The physics-violating row is excluded from L/D, not allowed to explode
    # the mean; its absence is reported explicitly.
    assert metrics["ld"]["n"] == 2
    assert metrics["ld"]["n_excluded"] == 1
    assert metrics["ld"]["max_abs"] < 1000.0


def test_relative_error_respects_reporting_floor() -> None:
    # Cl near zero (α ≈ 0°): floored MAPE is bounded even though the true
    # relative error is undefined.
    actual = np.array([[0.02, 0.0100, -0.02], [0.5, 0.0100, -0.02], [1.0, 0.0300, -0.05]])
    predicted = np.array([[0.04, 0.0100, -0.02], [0.5, 0.0100, -0.02], [1.0, 0.0300, -0.05]])
    metrics = percentage_metrics(actual, predicted)
    assert metrics["cl"]["mape_percent"] < 100.0
    # The "defined" variant excludes the near-zero row entirely.
    assert metrics["cl"]["mape_percent_defined"] == 0.0


def test_error_by_regime_partitions_rows() -> None:
    frame = pd.DataFrame(
        {
            "alpha_deg": [-1.0, 2.0, 6.0, 9.0],
            "reynolds": [1e5, 1e5, 5e5, 5e5],
        }
    )
    actual = np.array([[0.1, 0.01, -0.02], [0.5, 0.01, -0.02], [0.8, 0.02, -0.04], [1.1, 0.04, -0.06]])
    predicted = actual.copy()
    predictions = {"hist_gb": (actual, predicted)}
    by_regime = error_by_regime(frame, predictions)
    regimes = set(by_regime["by_alpha_regime"])
    assert {"negative lift", "attached flow", "drag rise", "near stall"}.issubset(regimes)
    re_keys = set(by_regime["by_reynolds"])
    assert {"re_100000", "re_500000"}.issubset(re_keys)
    for partition in by_regime["by_alpha_regime"].values():
        assert partition["hist_gb"]["cl"]["n"] == 1
