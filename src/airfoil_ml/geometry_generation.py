"""Parametric camber/thickness airfoil generation.

The aerodynamic performance of an airfoil is dominated by its camber line
(drives Cl and Cm0) and its thickness distribution (drives Cd and stall
behaviour). Generating airfoils by sweeping these two knobs produces a
dataset that is deliberately dense where the current real-airfoil sample is
sparse: high camber, unusual camber positions, thin/thick sections, and
symmetric shapes. It is the same idea behind the Kanakaero CST dataset
already used in this project (camber/thickness shape coefficients -> XFOIL
polars), with the difference that this generator keeps the full multi-Reynolds
sweep and Cm that the public fixed-Re table lacks.

Design choices:
- The camber line is the classic NACA four-digit piecewise parabola
  (parameters m = max camber, p = max-camber position), which is guaranteed
  smooth and XFOIL-friendly.
- The thickness distribution is the modified NACA four-digit form with a
  closed trailing edge (coefficient -0.1036 instead of -0.1015), so every
  generated geometry is a closed, physically sensible section.
- Surfaces are built as y_c +/- y_t * cos(theta) with theta from the local
  camber slope, the standard construction, and returned as the project's
  canonical AirfoilGeometry (cosine x-grid), so generated airfoils drop
  directly into the existing feature pipeline.

Provenance: every generated airfoil is named from its parameters and recorded
in a manifest JSON, so a training table built from this generator is fully
reproducible. The XFOIL labels themselves come from the existing sharded,
resumable batch runner; nothing here fabricates aerodynamic coefficients.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import AirfoilGeometry, write_geometry

DEFAULT_CAMBER = (0.0, 0.02, 0.04, 0.06)
DEFAULT_CAMBER_POSITION = (0.2, 0.4, 0.6)
DEFAULT_THICKNESS = (0.08, 0.12, 0.16)


def camber_line(x: np.ndarray, max_camber: float, position: float) -> tuple[np.ndarray, np.ndarray]:
    """NACA four-digit camber line and its slope at each x.

    Returns (y_c, dy_c/dx). The line is the piecewise parabola that peaks at
    (position, max_camber) and is zero at both leading and trailing edges.
    """
    if not 0.0 < position < 1.0:
        raise ValueError("camber position must be strictly between 0 and 1")
    y_c = np.where(
        x <= position,
        max_camber / position**2 * (2.0 * position * x - x**2),
        max_camber / (1.0 - position) ** 2 * ((1.0 - 2.0 * position) + 2.0 * position * x - x**2),
    )
    slope = np.where(
        x <= position,
        2.0 * max_camber / position**2 * (position - x),
        2.0 * max_camber / (1.0 - position) ** 2 * (position - x),
    )
    return y_c, slope


def thickness_distribution(x: np.ndarray, thickness: float) -> np.ndarray:
    """Modified NACA four-digit thickness with a closed trailing edge."""
    if not 0.0 < thickness <= 0.5:
        raise ValueError("thickness must be positive and at most 0.5")
    return (thickness / 0.2) * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )


def camber_thickness_geometry(max_camber: float, position: float, thickness: float, n_points: int = 100) -> AirfoilGeometry:
    """Build a canonical AirfoilGeometry from camber/thickness parameters."""
    if n_points < 4:
        raise ValueError("n_points must be at least 4")
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))
    y_c, slope = camber_line(x, max_camber, position)
    y_t = thickness_distribution(x, thickness)
    cosine = np.cos(np.arctan(slope))
    upper_y = y_c + y_t * cosine
    lower_y = y_c - y_t * cosine
    # Numerically close the trailing edge exactly.
    upper_y[-1] = lower_y[-1] = 0.0
    return AirfoilGeometry(x, upper_y, lower_y)


def camber_name(max_camber: float, position: float, thickness: float) -> str:
    """Deterministic, parameter-encoding airfoil name (safe for filenames)."""
    return f"CAMBER_m{int(round(max_camber * 1000)):03d}_p{int(round(position * 100)):02d}_t{int(round(thickness * 1000)):03d}"


def sample_camber_grid(
    camber_values: tuple[float, ...] = DEFAULT_CAMBER,
    positions: tuple[float, ...] = DEFAULT_CAMBER_POSITION,
    thickness_values: tuple[float, ...] = DEFAULT_THICKNESS,
) -> list[dict[str, float | str]]:
    """Enumerate a deterministic camber/thickness design grid.

    Returns a list of parameter records; each record carries the airfoil name
    so generated coordinates and later training tables can be traced back to
    the exact design parameters.
    """
    records: list[dict[str, float | str]] = []
    for max_camber in camber_values:
        for position in positions:
            for thickness in thickness_values:
                records.append(
                    {
                        "airfoil_id": camber_name(max_camber, position, thickness),
                        "max_camber": float(max_camber),
                        "camber_position": float(position),
                        "thickness": float(thickness),
                    }
                )
    return records


def generate_camber_coordinates(
    output_dir: str | Path,
    records: list[dict[str, float | str]] | None = None,
    camber_values: tuple[float, ...] = DEFAULT_CAMBER,
    positions: tuple[float, ...] = DEFAULT_CAMBER_POSITION,
    thickness_values: tuple[float, ...] = DEFAULT_THICKNESS,
    n_points: int = 100,
) -> dict[str, list[dict[str, float | str]]]:
    """Write one .dat coordinate file per design and a provenance manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = records if records is not None else sample_camber_grid(camber_values, positions, thickness_values)
    written: list[dict[str, float | str]] = []
    for record in records:
        airfoil_id = str(record["airfoil_id"])
        geometry = camber_thickness_geometry(
            float(record["max_camber"]), float(record["camber_position"]), float(record["thickness"]), n_points=n_points
        )
        write_geometry(output_dir / f"{airfoil_id}.dat", geometry)
        written.append(record)
    (output_dir / "generation_manifest.json").write_text(
        json.dumps({"generator": "camber_thickness_naca4", "n_points": n_points, "airfoils": written}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"airfoils": written}
