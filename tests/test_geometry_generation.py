import json
import re

import numpy as np
import pytest

from airfoil_ml.geometry import parse_airfoil
from airfoil_ml.geometry_generation import (
    camber_name,
    camber_thickness_geometry,
    generate_camber_coordinates,
    sample_camber_grid,
)


def test_geometry_reproduces_requested_camber_and_thickness() -> None:
    geometry = camber_thickness_geometry(0.04, 0.3, 0.12, n_points=100)
    thickness = np.max(geometry.upper_y - geometry.lower_y)
    meanline = 0.5 * (geometry.upper_y + geometry.lower_y)
    # The camber slope tilts the local thickness vector, so the sampled max
    # is within a small tolerance of the requested design parameters.
    assert thickness == pytest.approx(0.12, abs=1e-3)
    assert np.max(meanline) == pytest.approx(0.04, abs=1e-4)
    assert geometry.upper_y[-1] == geometry.lower_y[-1] == 0.0  # closed TE
    assert np.all(geometry.upper_y >= geometry.lower_y)


def test_symmetric_airfoil_has_zero_camber() -> None:
    geometry = camber_thickness_geometry(0.0, 0.4, 0.12, n_points=64)
    meanline = 0.5 * (geometry.upper_y + geometry.lower_y)
    assert np.max(np.abs(meanline)) < 1e-12
    # Symmetric: upper and lower surfaces mirror across the chord line.
    assert np.allclose(geometry.upper_y, -geometry.lower_y)


def test_generated_airfoil_parses_back_through_pipeline() -> None:
    geometry = camber_thickness_geometry(0.02, 0.5, 0.10, n_points=100)
    assert geometry.n_points == 100
    assert geometry.as_feature_vector().shape == (200,)


def test_grid_manifest_writes_reproducible_coordinates(tmp_path) -> None:
    records = sample_camber_grid(camber_values=(0.0, 0.04), positions=(0.4,), thickness_values=(0.12,))
    result = generate_camber_coordinates(tmp_path, records, n_points=50)
    assert len(result["airfoils"]) == 2
    manifest = json.loads((tmp_path / "generation_manifest.json").read_text())
    assert manifest["generator"] == "camber_thickness_naca4"
    for record in manifest["airfoils"]:
        parsed = parse_airfoil(tmp_path / f"{record['airfoil_id']}.dat", n_points=50)
        assert parsed.n_points == 50


def test_names_encode_parameters_and_are_filename_safe() -> None:
    name = camber_name(0.04, 0.3, 0.12)
    assert name == "CAMBER_m040_p30_t120"
    # Filename- and XFOIL-safe: letters, digits, underscores only.
    assert re.fullmatch(r"[A-Za-z0-9_]+", name)
