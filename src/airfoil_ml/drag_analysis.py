"""Drag-focused diagnostics for the held-out airfoil split.

Global metrics hide where drag error lives. Drag is the hardest coefficient to
surrogate because it is a small, second-order quantity that responds sharply to
boundary-layer state: low-Reynolds laminar separation bubbles, drag rise with
loading, and separation near stall all change Cd by large relative amounts. This
module quantifies Cd error by angle-of-attack regime and by Reynolds number, and
reports how those errors propagate into the L/D ratio, which is the quantity
designers actually use.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ALPHA_REGIMES = [(-2.0, 0.0, "negative lift"), (0.0, 4.0, "attached flow"), (4.0, 8.0, "drag rise"), (8.0, 11.0, "near stall")]
ALPHA_BINS = [(-2.5, -0.5), (-0.5, 2.0), (2.0, 4.5), (4.5, 7.0), (7.0, 9.0), (9.0, 10.5)]
COLORS = {"ridge": "#8c8c8c", "random_forest": "#ca6702", "hist_gb": "#176b87", "mlp_torch": "#5e3c99"}


def _coef_metrics(actual: np.ndarray, predicted: np.ndarray, column: int) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual[:, column], predicted[:, column])),
        "rmse": float(np.sqrt(mean_squared_error(actual[:, column], predicted[:, column]))),
        "r2": float(r2_score(actual[:, column], predicted[:, column])),
    }


def compute_drag_summary(frame: pd.DataFrame, predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    """Return per-model Cd and L/D metrics, overall and by flow regime."""
    summary: dict[str, object] = {}
    for name, (actual, predicted) in predictions.items():
        model_summary: dict[str, object] = {"overall": _coef_metrics(actual, predicted, 1)}
        ld_ref = actual[:, 0] / actual[:, 1]
        ld_pred = predicted[:, 0] / predicted[:, 1]
        model_summary["ld_mae"] = float(mean_absolute_error(ld_ref, ld_pred))
        model_summary["ld_relative_mae"] = float(np.mean(np.abs(ld_ref - ld_pred) / np.abs(ld_ref).clip(min=1e-6)))
        alpha = frame.alpha_deg.to_numpy()
        reynolds = frame.reynolds.to_numpy()
        by_alpha: dict[str, dict[str, float]] = {}
        for low, high, label in ALPHA_REGIMES:
            mask = (alpha >= low) & (alpha < high)
            if mask.sum() == 0:
                continue
            by_alpha[label] = _coef_metrics(actual[mask], predicted[mask], 1)
        model_summary["by_alpha_regime"] = by_alpha
        by_re: dict[str, dict[str, float]] = {}
        for re_value in sorted(set(reynolds.tolist())):
            mask = reynolds == re_value
            by_re[f"re_{int(re_value)}"] = _coef_metrics(actual[mask], predicted[mask], 1)
        model_summary["by_reynolds"] = by_re
        summary[name] = model_summary
    return summary


def save_drag_analysis_plots(frame: pd.DataFrame, predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: str | Path, summary: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = frame.alpha_deg.to_numpy()
    reynolds = frame.reynolds.to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    for name, (actual, predicted) in predictions.items():
        color = COLORS.get(name, "#333333")
        cd_error = np.abs(actual[:, 1] - predicted[:, 1])
        ld_ref = actual[:, 0] / actual[:, 1]
        ld_pred = predicted[:, 0] / predicted[:, 1]
        errors = pd.DataFrame({"alpha": alpha, "re": reynolds, "cd": cd_error, "ld": np.abs(ld_ref - ld_pred)})
        by_alpha = errors.groupby("alpha", as_index=False).mean()
        by_re = errors.groupby("re", as_index=False).mean()
        axes[0].plot(by_alpha.alpha, by_alpha.cd, "o-", color=color, label=name, markersize=4)
        axes[1].plot(by_re.re, by_re.cd, "o-", color=color, label=name, markersize=4)
        axes[2].plot(by_alpha.alpha, by_alpha.ld, "o-", color=color, label=name, markersize=4)
    axes[0].set(xlabel="Angle of attack, α [deg]", ylabel="Mean |ΔCd|", title="Drag error vs α")
    axes[1].set(xlabel="Reynolds number, Re", ylabel="Mean |ΔCd|", title="Drag error vs Re")
    axes[2].set(xlabel="Angle of attack, α [deg]", ylabel="Mean |Δ(L/D)|", title="L/D error vs α")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output_dir / "drag_error_by_condition.png", dpi=180)
    plt.close(fig)

    # One parity panel per model for Cd, with the low-drag region zoomed.
    fig, axes = plt.subplots(1, len(predictions), figsize=(4.2 * len(predictions), 4.0), constrained_layout=True)
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, (actual, predicted)) in zip(axes, predictions.items()):
        ax.scatter(actual[:, 1], predicted[:, 1], s=10, alpha=0.6, color=COLORS.get(name, "#333333"))
        limits = [0.0, float(np.percentile(actual[:, 1], 99.5) * 1.05)]
        ax.plot(limits, limits, "k--", linewidth=0.9)
        ax.set(xlabel="Reference Cd", ylabel="Predicted Cd", title=f"{name}")
        ax.grid(alpha=0.25)
    fig.suptitle("Drag parity on unseen airfoils", y=1.02)
    fig.savefig(output_dir / "drag_parity_all_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (output_dir / "drag_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
