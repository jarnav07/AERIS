"""Feature construction and leakage-safe scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .geometry import AirfoilGeometry

FLOW_COLUMNS = ("alpha_deg", "reynolds", "mach")
TARGET_COLUMNS = ("cl", "cd", "cm")


@dataclass
class FeaturePreprocessor:
    """Scalers fitted only on the training partition."""

    input_scaler: StandardScaler
    target_scaler: StandardScaler
    n_geometry_points: int

    def transform_inputs(self, geometry_features: np.ndarray, flow: np.ndarray) -> np.ndarray:
        raw = np.column_stack([geometry_features, flow[:, 0], np.log10(flow[:, 1]), flow[:, 2]])
        return self.input_scaler.transform(raw)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(targets)

    def inverse_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(targets)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        return joblib.load(path)


def raw_input_matrix(geometries: list[AirfoilGeometry], alpha_deg: np.ndarray, reynolds: np.ndarray, mach: np.ndarray) -> np.ndarray:
    if not (len(geometries) == len(alpha_deg) == len(reynolds) == len(mach)):
        raise ValueError("geometry and flow arrays must have the same length")
    if np.any(reynolds <= 0):
        raise ValueError("Reynolds number must be positive")
    geometry_features = np.vstack([g.as_feature_vector() for g in geometries])
    flow = np.column_stack([alpha_deg, reynolds, mach])
    return np.column_stack([geometry_features, alpha_deg, np.log10(reynolds), mach])


def fit_preprocessor(inputs: np.ndarray, targets: np.ndarray, n_geometry_points: int) -> FeaturePreprocessor:
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("inputs and targets must be two-dimensional")
    input_scaler = StandardScaler().fit(inputs)
    target_scaler = StandardScaler().fit(targets)
    return FeaturePreprocessor(input_scaler, target_scaler, n_geometry_points)
