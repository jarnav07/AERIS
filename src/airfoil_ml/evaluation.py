"""Evaluation metrics and engineering visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGETS = ("cl", "cd", "cm")


def regression_metrics_for_targets(actual: np.ndarray, predicted: np.ndarray, targets: tuple[str, ...] | list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for i, target in enumerate(targets):
        result[target] = {
            "mae": float(mean_absolute_error(actual[:, i], predicted[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))),
            "r2": float(r2_score(actual[:, i], predicted[:, i])),
        }
    return result


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    return regression_metrics_for_targets(actual, predicted, TARGETS)


def error_by_condition(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    errors = pd.DataFrame({"alpha_deg": frame.alpha_deg.to_numpy(), "reynolds": frame.reynolds.to_numpy()})
    errors["cl_abs_error"] = np.abs(actual[:, 0] - predicted[:, 0])
    errors["cd_abs_error"] = np.abs(actual[:, 1] - predicted[:, 1])
    errors["cm_abs_error"] = np.abs(actual[:, 2] - predicted[:, 2])
    return errors.groupby("alpha_deg", as_index=False).mean(numeric_only=True)


def save_evaluation_plots(
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_dir: str | Path,
    title_prefix: str = "Test",
    max_polar_plots: int | None = None,
) -> None:
    """Write parity, per-airfoil polar, and error-vs-condition plots for one model.

    ``frame`` must have one row per ``actual``/``predicted`` row and include
    ``airfoil_id``, ``alpha_deg``, ``reynolds``, ``cl`` and ``cd`` (the
    reference values plotted alongside predictions); ``actual``/``predicted``
    are ``(N, 3)`` arrays in ``cl, cd, cm`` order.

    ``max_polar_plots`` caps how many distinct airfoils get individual polar
    PNGs (one ``fig.savefig`` per airfoil) and how many go into the
    ``error_by_airfoil`` bar chart. The default of ``None`` plots every test
    airfoil, which is fine for the handful of airfoils in a small dataset but
    does not scale: at the ~20k test airfoils a full generated-dataset run
    produces, that loop takes on the order of hours per model and the bar
    chart becomes an unreadable/slow one-bar-per-airfoil plot. Pass ``0`` to
    skip both entirely, or a small int (e.g. 30) to sample that many airfoils
    (fixed seed, so the sample is reproducible) for a quick spot-check.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"cl": "#176b87", "cd": "#d95f02", "cm": "#5e3c99"}
    for i, target in enumerate(TARGETS):
        fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
        ax.scatter(actual[:, i], predicted[:, i], s=18, alpha=0.7, color=colors[target], edgecolor="none")
        limits = [min(actual[:, i].min(), predicted[:, i].min()), max(actual[:, i].max(), predicted[:, i].max())]
        ax.plot(limits, limits, "k--", linewidth=1, label="Perfect prediction")
        ax.set(xlabel=f"Reference {target}", ylabel=f"Predicted {target}", title=f"{title_prefix}: {target} parity")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f"parity_{target}.png", dpi=180)
        plt.close(fig)

    all_airfoil_ids = frame.airfoil_id.unique()
    if max_polar_plots is None:
        polar_airfoil_ids = set(all_airfoil_ids)
    elif max_polar_plots <= 0:
        polar_airfoil_ids = set()
    else:
        rng = np.random.default_rng(0)
        sample_size = min(max_polar_plots, len(all_airfoil_ids))
        polar_airfoil_ids = set(rng.choice(all_airfoil_ids, size=sample_size, replace=False))

    plot_frame = frame.copy()
    plot_frame["pred_cl"], plot_frame["pred_cd"], plot_frame["pred_cm"] = predicted.T
    for airfoil_id, group in plot_frame.groupby("airfoil_id"):
        if airfoil_id not in polar_airfoil_ids:
            continue
        group = group.sort_values("alpha_deg")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, target, label in zip(axes, ("cl", "cd", "ld"), ("$C_l$", "$C_d$", "$L/D$")):
            if target == "ld":
                ref = group.cl / group.cd
                pred = group.pred_cl / group.pred_cd.clip(lower=1e-8)
            else:
                ref = group[target]
                pred = group[f"pred_{target}"]
            ax.plot(group.alpha_deg, ref, "o-", label="Reference", color="#1b4965")
            ax.plot(group.alpha_deg, pred, "s--", label="ML prediction", color="#ca6702")
            ax.set(xlabel="Angle of attack, α [deg]", ylabel=label, title=f"{airfoil_id}: {label}")
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False)
        fig.savefig(output_dir / f"polar_{airfoil_id}.png", dpi=180)
        plt.close(fig)

    errors = error_by_condition(frame, actual, predicted)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        ax.plot(errors.alpha_deg, errors[f"{target}_abs_error"], "o-", color=colors[target])
        ax.set(xlabel="Angle of attack, α [deg]", ylabel=f"|error in {target}|", title=f"Error vs α: {target}")
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "error_vs_alpha.png", dpi=180)
    plt.close(fig)

    condition_errors = pd.DataFrame({"reynolds": frame.reynolds.to_numpy()})
    condition_errors["cl_abs_error"] = np.abs(actual[:, 0] - predicted[:, 0])
    condition_errors["cd_abs_error"] = np.abs(actual[:, 1] - predicted[:, 1])
    condition_errors["cm_abs_error"] = np.abs(actual[:, 2] - predicted[:, 2])
    by_re = condition_errors.groupby("reynolds", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        ax.plot(by_re.reynolds, by_re[f"{target}_abs_error"], "o-", color=colors[target])
        ax.set(xlabel="Reynolds number, Re", ylabel=f"|error in {target}|", title=f"Error vs Re: {target}")
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "error_vs_reynolds.png", dpi=180)
    plt.close(fig)

    by_airfoil = condition_errors.copy()
    by_airfoil["airfoil_id"] = frame.airfoil_id.to_numpy()
    by_airfoil = by_airfoil.groupby("airfoil_id", as_index=False).mean(numeric_only=True)
    if max_polar_plots is not None and len(by_airfoil) > max(max_polar_plots, 1):
        by_airfoil = by_airfoil.sample(n=max(max_polar_plots, 1), random_state=0).sort_values("airfoil_id")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        ax.bar(by_airfoil.airfoil_id, by_airfoil[f"{target}_abs_error"], color=colors[target])
        ax.set(xlabel="Airfoil identity", ylabel=f"mean |error in {target}|", title=f"Error by airfoil: {target}")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "error_by_airfoil.png", dpi=180)
    plt.close(fig)
