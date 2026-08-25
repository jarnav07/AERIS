import numpy as np

from airfoil_ml.geometry import GeometryError, parse_airfoil


def sample_coordinates() -> str:
    return """NACA test\n1.0 0.0\n0.75 0.05\n0.5 0.08\n0.25 0.06\n0.0 0.0\n0.25 -0.03\n0.5 -0.04\n0.75 -0.02\n1.0 0.0\n"""


def test_parse_normalizes_and_resamples_surfaces() -> None:
    geometry = parse_airfoil(sample_coordinates(), n_points=16)
    assert geometry.n_points == 16
    assert np.isclose(geometry.x[0], 0.0)
    assert np.isclose(geometry.x[-1], 1.0)
    assert geometry.upper_y[8] > geometry.lower_y[8]
    assert geometry.as_feature_vector().shape == (32,)


def test_accepts_lednicer_leading_edge_first_order() -> None:
    geometry = parse_airfoil("0 0\n.25 .06\n.5 .08\n.75 .04\n1 0\n.75 -.02\n.5 -.04\n.25 -.03\n0 0\n", n_points=12)
    assert geometry.n_points == 12
    assert np.isclose(geometry.x[0], 0.0)
    assert np.isclose(geometry.x[-1], 1.0)


def test_ignores_lednicer_point_count_metadata() -> None:
    geometry = parse_airfoil("FX test\n49 49\n0 0\n.25 .06\n.5 .1\n.75 .06\n1 0\n.75 -.03\n.5 -.08\n.25 -.04\n0 0\n", n_points=12)
    assert geometry.n_points == 12
    assert np.isclose(geometry.x[-1], 1.0)


def test_rejects_degenerate_coordinates() -> None:
    try:
        parse_airfoil("0 0\n0 0\n0 0\n0 0\n0 0\n0 0\n0 0\n0 0\n")
    except GeometryError:
        pass
    else:
        raise AssertionError("degenerate geometry should fail")
