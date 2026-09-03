import numpy as np
import pandas as pd
import pytest

from airfoil_ml.evaluation import error_by_condition, regression_metrics, save_evaluation_plots


def test_regression_metrics_perfect_prediction_is_zero_error() -> None:
    actual = np.array([[0.5, 0.02, -0.1], [0.6, 0.021, -0.11]])
    metrics = regression_metrics(actual, actual.copy())

    assert set(metrics) == {"cl", "cd", "cm"}
    for target in metrics:
        assert metrics[target]["mae"] == 0.0
        assert metrics[target]["rmse"] == 0.0
        assert metrics[target]["r2"] == 1.0


def test_regression_metrics_detects_constant_offset_error() -> None:
    actual = np.array([[0.5, 0.02, -0.1], [0.6, 0.021, -0.11], [0.7, 0.022, -0.12]])
    predicted = actual + np.array([0.1, 0.0, 0.0])

    metrics = regression_metrics(actual, predicted)

    assert metrics["cl"]["mae"] == pytest.approx(0.1)
    assert metrics["cd"]["mae"] == 0.0
    assert metrics["cm"]["mae"] == 0.0


def _sample_frame(n_airfoils: int = 2, alphas: tuple[float, ...] = (-2.0, 0.0, 2.0)) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    actual = []
    for airfoil_index in range(n_airfoils):
        for alpha in alphas:
            rows.append({
                "airfoil_id": f"foil_{airfoil_index}",
                "alpha_deg": alpha,
                "reynolds": 500_000.0 + airfoil_index * 10_000,
                "cl": 0.1 * alpha,
                "cd": 0.02 + 0.001 * alpha * alpha,
            })
            actual.append([0.1 * alpha, 0.02 + 0.001 * alpha * alpha, -0.01 * alpha])
    frame = pd.DataFrame(rows)
    actual_arr = np.asarray(actual)
    predicted_arr = actual_arr + 0.01
    return frame, actual_arr, predicted_arr


def test_error_by_condition_groups_by_alpha() -> None:
    frame, actual, predicted = _sample_frame()
    errors = error_by_condition(frame, actual, predicted)

    assert sorted(errors["alpha_deg"]) == [-2.0, 0.0, 2.0]
    assert np.allclose(errors["cl_abs_error"], 0.01)


def test_save_evaluation_plots_writes_expected_files(tmp_path) -> None:
    frame, actual, predicted = _sample_frame()
    output_dir = tmp_path / "evaluation"

    save_evaluation_plots(frame, actual, predicted, output_dir, title_prefix="Ridge")

    for target in ("cl", "cd", "cm"):
        assert (output_dir / f"parity_{target}.png").exists()
    for airfoil_id in frame["airfoil_id"].unique():
        assert (output_dir / f"polar_{airfoil_id}.png").exists()
    assert (output_dir / "error_vs_alpha.png").exists()
    assert (output_dir / "error_vs_reynolds.png").exists()
    assert (output_dir / "error_by_airfoil.png").exists()
