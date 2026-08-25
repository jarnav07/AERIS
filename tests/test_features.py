import numpy as np
from sklearn.preprocessing import StandardScaler

from airfoil_ml.features import FeaturePreprocessor


def test_scaling_round_trip() -> None:
    input_scaler = StandardScaler().fit(np.array([[0.0, 1.0], [1.0, 2.0]]))
    target_scaler = StandardScaler().fit(np.array([[1.0, 10.0, 0.0], [3.0, 20.0, 2.0]]))
    processor = FeaturePreprocessor(input_scaler, target_scaler, 1)
    targets = np.array([[2.0, 15.0, 1.0]])
    assert np.allclose(processor.inverse_targets(processor.transform_targets(targets)), targets)
