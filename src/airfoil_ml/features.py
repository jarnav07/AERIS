"""Feature construction and train-only scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .data import KULFAN_COLUMNS


@dataclass
class FeaturePreprocessor:
    """Input/target scalers fitted exclusively on the training partition."""

    input_scaler: StandardScaler
    target_scaler: StandardScaler

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        return self.input_scaler.transform(inputs)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(targets)

    def inverse_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(targets)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        return joblib.load(path)


def build_feature_matrix(frame) -> np.ndarray:
    """Build the common 21-feature matrix used by every surrogate model.

    The 18 Kulfan geometry parameters are followed by angle of attack,
    log10(Reynolds number), and Mach number.
    """
    geometry = frame[list(KULFAN_COLUMNS)].to_numpy(float)
    flow = frame[["alpha_deg", "reynolds", "mach"]].to_numpy(float)
    if np.any(flow[:, 1] <= 0):
        raise ValueError("Reynolds number must be positive")
    return np.column_stack((geometry, flow[:, 0], np.log10(flow[:, 1]), flow[:, 2]))


def fit_preprocessor(inputs: np.ndarray, targets: np.ndarray) -> FeaturePreprocessor:
    if inputs.ndim != 2 or targets.ndim != 2 or len(inputs) != len(targets):
        raise ValueError("inputs and targets must be 2-D matrices with equal row counts")
    return FeaturePreprocessor(
        input_scaler=StandardScaler().fit(inputs),
        target_scaler=StandardScaler().fit(targets),
    )
