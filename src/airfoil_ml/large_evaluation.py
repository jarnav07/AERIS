"""Plots for the fixed-Re CST airfoil experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .large_dataset import FIXED_TARGET_COLUMNS


def save_fixed_re_plots(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Cl": "#176b87", "Cd": "#d95f02"}

    for i, target in enumerate(FIXED_TARGET_COLUMNS):
        fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
        ax.scatter(actual[:, i], predicted[:, i], s=8, alpha=0.35, color=colors[target], edgecolor="none")
        limits = [min(actual[:, i].min(), predicted[:, i].min()), max(actual[:, i].max(), predicted[:, i].max())]
        ax.plot(limits, limits, "k--", linewidth=1, label="Perfect prediction")
        ax.set(xlabel=f"Reference {target}", ylabel=f"Predicted {target}", title=f"Fixed Re: {target} parity")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f"parity_{target.lower()}.png", dpi=180)
        plt.close(fig)

    plot_frame = frame.copy()
    plot_frame["pred_Cl"], plot_frame["pred_Cd"] = predicted.T
    examples = list(plot_frame.airfoil_id.drop_duplicates().head(6))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True, squeeze=False)
    for ax, airfoil_id in zip(axes.flat, examples):
        group = plot_frame[plot_frame.airfoil_id == airfoil_id].sort_values("alpha_deg")
        ax.plot(group.alpha_deg, group.Cl, "o-", label="Reference Cl", color=colors["Cl"])
        ax.plot(group.alpha_deg, group.pred_Cl, "s--", label="Predicted Cl", color="#ca6702")
        ax.set_title(str(airfoil_id))
        ax.set_xlabel("Angle of attack, α [deg]")
        ax.set_ylabel("Cl")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(examples) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.savefig(output_dir / "polar_examples_cl.png", dpi=180)
    plt.close(fig)

    error_frame = pd.DataFrame({"alpha_deg": frame.alpha_deg.to_numpy()})
    error_frame["Cl_abs_error"] = np.abs(actual[:, 0] - predicted[:, 0])
    error_frame["Cd_abs_error"] = np.abs(actual[:, 1] - predicted[:, 1])
    errors = error_frame.groupby("alpha_deg", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, target in zip(axes, FIXED_TARGET_COLUMNS):
        ax.plot(errors.alpha_deg, errors[f"{target}_abs_error"], "o-", color=colors[target])
        ax.set(xlabel="Angle of attack, α [deg]", ylabel=f"|error in {target}|", title=f"Error vs α: {target}")
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "error_vs_alpha.png", dpi=180)
    plt.close(fig)
