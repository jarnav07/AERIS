"""Inference interface for the trained stacking ensemble.

``fit_stacking_ensemble`` (in ``training.py``) produces the project's most accurate
model as a set of artifacts in one model directory: the component models, their
shared preprocessor, and ``stacking_weights.json``. This module is the read side of
that artifact — give it an aerofoil and a flow condition, get CL/CD/CM back — so a
prediction never has to go through the training/evaluation scripts or the
canonical CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd

from .features import FeaturePreprocessor
from .training import (
    KULFAN_FEATURE_COLUMNS,
    KULFAN_VECTOR_COLUMNS,
    _inverse_labels,
    _select_columns,
    add_geometry_features,
    apply_stacking_weights,
    load_model_bundle,
    transform_kulfan_inputs,
)

#: Flow-condition defaults. ``mach=0`` because the canonical dataset is
#: incompressible throughout; ``n_crit=9`` is XFOIL's standard "average wind
#: tunnel" transition criterion (the dataset samples 0-18 uniformly); ``xtr=1``
#: means free transition, which 80% of generated cases use.
DEFAULT_MACH = 0.0
DEFAULT_N_CRIT = 9.0
DEFAULT_XTR = 1.0

#: Reynolds range the dataset actually covers (log10 mean 5.5, sigma 0.75, so
#: +/-3 sigma). Predictions outside it are extrapolation and are warned about.
RE_TRAINED_RANGE = (10 ** (5.5 - 3 * 0.75), 10 ** (5.5 + 3 * 0.75))
#: Alpha grid (-15..15) plus the jitter the generator adds, rounded outwards.
ALPHA_TRAINED_RANGE = (-25.0, 25.0)

_OUTPUT_TARGETS = ("CL", "CD", "CM")


def kulfan_feature_frame(
    airfoil: asb.KulfanAirfoil,
    alpha: np.ndarray | float,
    Re: np.ndarray | float,
    mach: np.ndarray | float = DEFAULT_MACH,
    n_crit: np.ndarray | float = DEFAULT_N_CRIT,
    xtr_upper: np.ndarray | float = DEFAULT_XTR,
    xtr_lower: np.ndarray | float = DEFAULT_XTR,
    airfoil_id: str = "query",
) -> pd.DataFrame:
    """Build a frame with exactly the columns ``train_from_kulfan_csv`` consumed.

    Flow arguments broadcast against each other, so a single aerofoil can be swept
    over an alpha array at one Reynolds number (or over paired alpha/Re arrays).
    Geometry features come from ``training.add_geometry_features``, so the derived
    columns are computed identically to training rather than reimplemented here.
    """
    alpha, Re, mach, n_crit, xtr_upper, xtr_lower = (
        np.atleast_1d(a).astype(float)
        for a in np.broadcast_arrays(alpha, Re, mach, n_crit, xtr_upper, xtr_lower)
    )
    if np.any(Re <= 0):
        raise ValueError("Reynolds number must be positive")

    kulfan_vector = np.concatenate([
        np.asarray(airfoil.upper_weights, dtype=float),
        np.asarray(airfoil.lower_weights, dtype=float),
        [float(airfoil.leading_edge_weight), float(airfoil.TE_thickness)],
    ])
    frame = pd.DataFrame(
        np.tile(kulfan_vector, (len(alpha), 1)),
        columns=KULFAN_VECTOR_COLUMNS,
    )
    frame["airfoil_id"] = airfoil_id
    frame["alpha"], frame["Re"], frame["mach"] = alpha, Re, mach
    frame["n_crit"], frame["xtr_upper"], frame["xtr_lower"] = n_crit, xtr_upper, xtr_lower
    return add_geometry_features(frame)


def out_of_distribution_warnings(frame: pd.DataFrame) -> list[str]:
    """Flag query rows outside the operating envelope the dataset was sampled over."""
    warnings = []
    if frame["Re"].min() < RE_TRAINED_RANGE[0] or frame["Re"].max() > RE_TRAINED_RANGE[1]:
        warnings.append(
            f"Re outside the trained range {RE_TRAINED_RANGE[0]:.0f}-{RE_TRAINED_RANGE[1]:.0f}"
        )
    if frame["alpha"].min() < ALPHA_TRAINED_RANGE[0] or frame["alpha"].max() > ALPHA_TRAINED_RANGE[1]:
        warnings.append(
            f"alpha outside the trained range {ALPHA_TRAINED_RANGE[0]:.0f} to {ALPHA_TRAINED_RANGE[1]:.0f} deg"
        )
    if float(np.max(np.abs(frame["mach"]))) > 0:
        warnings.append("mach > 0; the canonical dataset is incompressible (mach=0) throughout")
    return warnings


@dataclass
class StackingEnsemblePredictor:
    """The per-target learned stacking ensemble, loaded for inference.

    Construct with :meth:`load`. Component models each keep their own fitted
    preprocessor and are inverse-transformed to physical units *before* the
    weighted combination, exactly as in ``training.fit_stacking_ensemble`` — the
    weights were fitted on physical-unit predictions, so combining in any scaled
    space would be wrong.
    """

    model_dir: Path
    model_names: list[str]
    weights: dict[str, dict[str, float]]
    log_cd: bool
    bundles: dict[str, tuple[object, FeaturePreprocessor]] = field(repr=False)

    @classmethod
    def load(cls, model_dir: str | Path) -> "StackingEnsemblePredictor":
        model_dir = Path(model_dir)
        weights_path = model_dir / "stacking_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"no stacking ensemble found at {weights_path}; fit one first with\n"
                f"  uv run python scripts/fit_stacking_ensemble.py --model-dir {model_dir} "
                "--models mlp mlp_torch hist_gb"
            )
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        model_names = list(payload["model_names"])

        config_path = model_dir / "training_config.json"
        log_cd = bool(json.loads(config_path.read_text(encoding="utf-8"))["log_cd"]) if config_path.exists() else False

        return cls(
            model_dir=model_dir,
            model_names=model_names,
            weights=payload["weights"],
            log_cd=log_cd,
            bundles={name: load_model_bundle(model_dir, name) for name in model_names},
        )

    def component_predictions(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Each component model's physical-unit prediction for a raw feature matrix."""
        predictions = {}
        for name, (model, processor) in self.bundles.items():
            scaled = model.predict(transform_kulfan_inputs(processor, features))
            predictions[name] = _inverse_labels(processor, scaled, self.log_cd)
        return predictions

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Ensemble CL/CD/CM for a frame built by :func:`kulfan_feature_frame`."""
        features = _select_columns(frame, KULFAN_FEATURE_COLUMNS, "feature")
        return apply_stacking_weights(self.component_predictions(features), self.model_names, self.weights)

    def predict(
        self,
        airfoil: asb.KulfanAirfoil | str,
        alpha: np.ndarray | float,
        Re: np.ndarray | float,
        mach: np.ndarray | float = DEFAULT_MACH,
        n_crit: np.ndarray | float = DEFAULT_N_CRIT,
        xtr_upper: np.ndarray | float = DEFAULT_XTR,
        xtr_lower: np.ndarray | float = DEFAULT_XTR,
        per_model: bool = False,
    ) -> pd.DataFrame:
        """Predict CL/CD/CM for one aerofoil over one or more flow conditions.

        ``airfoil`` may be a ``KulfanAirfoil`` or any specification
        ``airfoil_sources.resolve_airfoil`` accepts (database name, NACA
        designation, coordinate file, Kulfan JSON). Returns a frame of the flow
        conditions alongside ``CL``/``CD``/``CM`` and the derived ``L_over_D``,
        plus each component model's own prediction when ``per_model`` is set.
        """
        if isinstance(airfoil, str):
            from .airfoil_sources import resolve_airfoil

            airfoil = resolve_airfoil(airfoil)

        frame = kulfan_feature_frame(airfoil, alpha, Re, mach, n_crit, xtr_upper, xtr_lower)
        features = _select_columns(frame, KULFAN_FEATURE_COLUMNS, "feature")
        components = self.component_predictions(features)
        combined = apply_stacking_weights(components, self.model_names, self.weights)

        result = frame[["alpha", "Re", "mach", "n_crit", "xtr_upper", "xtr_lower"]].copy()
        for index, target in enumerate(_OUTPUT_TARGETS):
            result[target] = combined[:, index]
        # Guard the ratio rather than letting a non-physical near-zero CD
        # prediction produce an inf that then poisons plots and CSV output.
        result["L_over_D"] = np.where(result["CD"] > 1e-9, result["CL"] / result["CD"], np.nan)

        if per_model:
            for name in self.model_names:
                for index, target in enumerate(_OUTPUT_TARGETS):
                    result[f"{name}_{target}"] = components[name][:, index]
        return result


def plot_prediction_polars(
    result: pd.DataFrame,
    path: str | Path,
    title: str = "predicted polar",
) -> Path:
    """Plot a predicted alpha sweep (CL, CD, CM, and the drag polar) to a PNG."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    alpha = result["alpha"].to_numpy()
    order = np.argsort(alpha)
    for axis, target, label in zip(axes.ravel(), _OUTPUT_TARGETS, ("CL", "CD", "CM")):
        axis.plot(alpha[order], result[target].to_numpy()[order], marker="o", markersize=3, color="#1f77b4")
        axis.set_xlabel("alpha (deg)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)

    drag_polar = axes.ravel()[3]
    drag_polar.plot(result["CD"].to_numpy()[order], result["CL"].to_numpy()[order], marker="o", markersize=3, color="#d62728")
    drag_polar.set_xlabel("CD")
    drag_polar.set_ylabel("CL")
    drag_polar.grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
