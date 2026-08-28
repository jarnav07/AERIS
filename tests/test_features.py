import numpy as np

from airfoil_ml.features import build_feature_matrix, fit_preprocessor
from tests.test_data import make_frame


def test_feature_matrix_uses_21_common_features():
    frame = make_frame()
    matrix = build_feature_matrix(frame)
    assert matrix.shape == (len(frame), 21)
    assert np.isclose(matrix[0, -2], np.log10(frame.iloc[0].reynolds))


def test_preprocessor_roundtrip_targets():
    frame = make_frame()
    inputs = build_feature_matrix(frame)
    targets = frame[["cl", "cd", "cm"]].to_numpy(float)
    processor = fit_preprocessor(inputs, targets)
    transformed = processor.transform_targets(targets)
    recovered = processor.inverse_targets(transformed)
    assert np.allclose(recovered, targets)
