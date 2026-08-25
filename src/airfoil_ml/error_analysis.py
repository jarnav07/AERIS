"""Percentage-error and runtime analysis for the multi-Re surrogate.

Global MAE/RMSE/R2 scores answer "how big is the error" but not "how big is
the error relative to the value being predicted". A Cd error of 0.003 is 3
drag counts — small in absolute terms but 30% of a 0.01 Cd. This module
reports relative (percentage) error per coefficient, splits it by angle-of-
attack regime and Reynolds number, and quantifies the wall-clock advantage of
the trained surrogate over running XFOIL for the same operating points.

Reference honesty: every percentage below is measured against XFOIL polar
output on airfoils that were completely held out of training. XFOIL is a
low-order viscous/inviscid panel solver, not Navier-Stokes CFD; the surrogate
cannot be more accurate than the reference it was trained on, and XFOIL's own
bias versus wind-tunnel data or RANS CFD (small for Cl in attached flow, a
few percent to tens of percent for Cd depending on Reynolds number and stall
proximity) is inherited by the surrogate.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .acquisition import run_xfoil_polar

TARGETS = ("cl", "cd", "cm")
# Relative error is undefined when the reference crosses zero (Cl and Cm at
# low angle of attack) and misleading when the reference is tiny. Report the
# floored MAPE alongside the engineering unit for drag (1 drag count = 0.0001).
REL_FLOORS = {"cl": 0.05, "cd": 0.005, "cm": 0.01, "ld": 1.0}
DRAG_COUNT = 1e4
ALPHA_REGIMES = [(-2.0, 0.0, "negative lift"), (0.0, 4.0, "attached flow"), (4.0, 8.0, "drag rise"), (8.0, 11.0, "near stall")]
COLORS = {"ridge": "#8c8c8c", "random_forest": "#ca6702", "hist_gb": "#176b87", "mlp_torch": "#5e3c99"}


def percentage_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    """Absolute and floored-relative error per coefficient, plus L/D."""
    result: dict[str, dict[str, float]] = {}
    for i, name in enumerate(TARGETS):
        abs_error = np.abs(actual[:, i] - predicted[:, i])
        relative = abs_error / np.maximum(np.abs(actual[:, i]), REL_FLOORS[name])
        defined = np.abs(actual[:, i]) >= 2.0 * REL_FLOORS[name]
        entry: dict[str, float] = {
            "mae": float(mean_absolute_error(actual[:, i], predicted[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))),
            "mape_percent": float(np.mean(relative) * 100.0),
            "median_ape_percent": float(np.median(relative) * 100.0),
            "mape_percent_defined": float(np.mean(relative[defined]) * 100.0) if defined.any() else 0.0,
            "p95_abs": float(np.percentile(abs_error, 95)),
            "max_abs": float(np.max(abs_error)),
            "n": int(len(abs_error)),
        }
        if name == "cd":
            entry["mae_drag_counts"] = float(np.mean(abs_error) * DRAG_COUNT)
            entry["p95_drag_counts"] = float(np.percentile(abs_error, 95) * DRAG_COUNT)
            entry["n_negative_predictions"] = int(np.sum(predicted[:, i] < 0.0))
        result[name] = entry
    # L/D is only meaningful where both the reference and the prediction have
    # a positive drag. A predicted Cd <= 0 is a physics violation: the row is
    # excluded from L/D statistics and counted under "n_excluded" so it is
    # surfaced rather than silently averaged away.
    cd_floor = 1e-4
    valid = (actual[:, 1] >= cd_floor) & (predicted[:, 1] > 0.0)
    ld_ref = actual[valid, 0] / actual[valid, 1]
    ld_pred = predicted[valid, 0] / predicted[valid, 1]
    ld_error = np.abs(ld_ref - ld_pred)
    ld_relative = ld_error / np.maximum(np.abs(ld_ref), REL_FLOORS["ld"])
    result["ld"] = {
        "mae": float(mean_absolute_error(ld_ref, ld_pred)) if valid.any() else 0.0,
        "mape_percent": float(np.mean(ld_relative) * 100.0) if valid.any() else 0.0,
        "median_ape_percent": float(np.median(ld_relative) * 100.0) if valid.any() else 0.0,
        "p95_abs": float(np.percentile(ld_error, 95)) if valid.any() else 0.0,
        "max_abs": float(np.max(ld_error)) if valid.any() else 0.0,
        "n": int(valid.sum()),
        "n_excluded": int(len(actual) - valid.sum()),
    }
    return result


def _subset_metrics(frame: pd.DataFrame, predictions: dict[str, tuple[np.ndarray, np.ndarray]], mask: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, (actual, predicted) in predictions.items():
        out[name] = percentage_metrics(actual[mask], predicted[mask])
    return out


def error_by_regime(frame: pd.DataFrame, predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Percentage error by angle-of-attack regime and by Reynolds number."""
    alpha = frame.alpha_deg.to_numpy()
    reynolds = frame.reynolds.to_numpy()
    by_alpha: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for low, high, label in ALPHA_REGIMES:
        mask = (alpha >= low) & (alpha < high)
        if mask.sum() == 0:
            continue
        by_alpha[label] = _subset_metrics(frame, predictions, mask)
    by_re: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for re_value in sorted(set(reynolds.tolist())):
        mask = reynolds == re_value
        by_re[f"re_{int(re_value)}"] = _subset_metrics(frame, predictions, mask)
    return {"by_alpha_regime": by_alpha, "by_reynolds": by_re}


def benchmark_ml(model_dir: str | Path, model_names: list[str], x_scaled: np.ndarray, batch_repeats: int = 5, single_repeats: int = 300) -> dict[str, dict[str, float]]:
    """Wall-clock throughput of each saved model on the scaled input matrix."""
    import joblib

    model_dir = Path(model_dir)
    results: dict[str, dict[str, float]] = {}
    for name in model_names:
        model = joblib.load(model_dir / f"{name}.joblib")
        model.predict(x_scaled[:8])  # warm-up caches, thread pools, BLAS paths
        batch_times: list[float] = []
        for _ in range(batch_repeats):
            start = time.perf_counter()
            model.predict(x_scaled)
            batch_times.append(time.perf_counter() - start)
        batch_seconds = float(np.median(batch_times))
        start = time.perf_counter()
        for _ in range(single_repeats):
            model.predict(x_scaled[:1])
        single_seconds = (time.perf_counter() - start) / single_repeats
        results[name] = {
            "batch_rows": int(len(x_scaled)),
            "batch_seconds": batch_seconds,
            "rows_per_second": float(len(x_scaled) / batch_seconds),
            "single_row_ms": float(single_seconds * 1e3),
        }
    return results


def benchmark_xfoil(
    coordinates_dir: str | Path,
    airfoil_ids: list[str],
    reynolds_values: list[float],
    alpha_start: float = -2.0,
    alpha_end: float = 10.0,
    alpha_step: float = 1.0,
    xfoil_executable: str = "xfoil",
) -> dict[str, float]:
    """Wall-clock time per XFOIL polar on this machine for representative foils."""
    coordinates_dir = Path(coordinates_dir)
    polar_times: list[float] = []
    n_points = 0
    with tempfile.TemporaryDirectory(prefix="xfoil_bench_") as scratch:
        for airfoil_id in airfoil_ids:
            coordinate_path = coordinates_dir / f"{airfoil_id}.dat"
            if not coordinate_path.exists():
                raise FileNotFoundError(f"no coordinate file for benchmark airfoil {airfoil_id}")
            for reynolds in reynolds_values:
                output_path = Path(scratch) / f"bench_{int(reynolds)}.csv"
                start = time.perf_counter()
                frame = run_xfoil_polar(
                    coordinate_path,
                    output_path,
                    reynolds=float(reynolds),
                    alpha_start=alpha_start,
                    alpha_end=alpha_end,
                    alpha_step=alpha_step,
                    xfoil_executable=xfoil_executable,
                    timeout_seconds=60,
                )
                polar_times.append(time.perf_counter() - start)
                n_points += len(frame)
    per_polar = float(np.mean(polar_times))
    return {
        "n_polars": len(polar_times),
        "n_points": n_points,
        "seconds_per_polar": per_polar,
        "seconds_per_point": per_polar / max(1, n_points // max(1, len(polar_times))),
        "points_per_polar": float(n_points / max(1, len(polar_times))),
    }


def time_saved_summary(xfoil: dict[str, float], ml: dict[str, dict[str, float]], model_name: str, n_airfoils: int = 100, n_reynolds: int = 6, n_alpha: int = 13, xfoil_workers: int = 1) -> dict[str, float]:
    """Projected wall-clock for a full design sweep: XFOIL vs the surrogate."""
    n_polars = n_airfoils * n_reynolds
    n_points = n_polars * n_alpha
    xfoil_wall_seconds = n_polars * xfoil["seconds_per_polar"] / xfoil_workers
    ml_rate = ml[model_name]["rows_per_second"]
    ml_wall_seconds = n_points / ml_rate
    return {
        "scenario": f"{n_airfoils} airfoils x {n_reynolds} Re x {n_alpha} alpha",
        "n_polars": n_polars,
        "n_points": n_points,
        "xfoil_wall_seconds": xfoil_wall_seconds,
        "ml_wall_seconds": ml_wall_seconds,
        "speedup_x": xfoil_wall_seconds / max(ml_wall_seconds, 1e-12),
        "hours_saved": (xfoil_wall_seconds - ml_wall_seconds) / 3600.0,
    }


def save_error_plots(frame: pd.DataFrame, predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: str | Path, timing: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = frame.alpha_deg.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    for name, (actual, predicted) in predictions.items():
        color = COLORS.get(name, "#333333")
        for i, target in enumerate(TARGETS):
            relative = np.abs(actual[:, i] - predicted[:, i]) / np.maximum(np.abs(actual[:, i]), REL_FLOORS[target])
            by_alpha = pd.DataFrame({"alpha": alpha, "ape": relative * 100.0}).groupby("alpha", as_index=False).mean()
            axes[i].plot(by_alpha.alpha, by_alpha.ape, "o-", color=color, label=name, markersize=4)
    for ax, target in zip(axes, TARGETS):
        ax.set(xlabel="Angle of attack, α [deg]", ylabel="Mean |relative error| [%]", title=f"Relative error vs α: {target}")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output_dir / "relative_error_vs_alpha.png", dpi=180)
    plt.close(fig)

    xfoil_benchmark = timing.get("xfoil_benchmark")
    if xfoil_benchmark is not None:
        fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
        labels: list[str] = []
        seconds: list[float] = []
        labels.append("XFOIL (measured, per operating point)")
        seconds.append(float(xfoil_benchmark["seconds_per_point"]))  # type: ignore[arg-type]
        ml_names = list(timing["ml_benchmark"].keys())  # type: ignore[union-attr]
        for name in ml_names:
            labels.append(f"ML {name} (per point, batched)")
            seconds.append(1.0 / float(timing["ml_benchmark"][name]["rows_per_second"]))  # type: ignore[index]
        order = np.argsort(seconds)
        colors = ["#b33939"] + [COLORS.get(name, "#333333") for name in ml_names]
        ax.barh([labels[i] for i in order], [seconds[i] for i in order], color=[colors[i] for i in order])
        ax.set_xscale("log")
        ax.set_xlabel("Seconds per operating point (log scale)")
        ax.set_title("Wall-clock per operating point: XFOIL vs ML surrogate")
        ax.grid(axis="x", alpha=0.25)
        fig.savefig(output_dir / "runtime_comparison.png", dpi=180)
        plt.close(fig)


def save_error_summary(frame: pd.DataFrame, predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: str | Path, timing: dict[str, object], summary: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_error_plots(frame, predictions, output_dir, timing)
    (output_dir / "error_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "timing_summary.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
