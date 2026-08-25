"""Airfoil coordinate validation and fixed-length geometry representation.

The model stores ordinates on a common cosine-spaced x-grid.  This retains
surface shape information while making every airfoil a fixed-width feature
vector and avoids treating the x-coordinate as a learned variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from typing import Iterable

import numpy as np


class GeometryError(ValueError):
    """Raised when coordinates cannot be converted to a physical airfoil."""


@dataclass(frozen=True)
class AirfoilGeometry:
    """Canonical airfoil geometry on a shared x-grid."""

    x: np.ndarray
    upper_y: np.ndarray
    lower_y: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.upper_y.shape != self.x.shape or self.lower_y.shape != self.x.shape:
            raise GeometryError("x, upper_y, and lower_y must be one-dimensional arrays of equal length")
        if len(self.x) < 4 or not np.all(np.isfinite(np.concatenate([self.x, self.upper_y, self.lower_y]))):
            raise GeometryError("geometry contains too few or non-finite points")

    @property
    def n_points(self) -> int:
        return len(self.x)

    def as_feature_vector(self) -> np.ndarray:
        """Return [upper ordinates, lower ordinates] for ML input."""
        return np.concatenate([self.upper_y, self.lower_y]).astype(float)

    def as_coordinates(self) -> np.ndarray:
        """Return Selig-style coordinates: upper TE->LE, then lower LE->TE."""
        upper = np.column_stack([self.x[::-1], self.upper_y[::-1]])
        lower = np.column_stack([self.x[1:], self.lower_y[1:]])
        return np.vstack([upper, lower])


def _numeric_rows(source: str | Path | TextIOBase | Iterable[str]) -> np.ndarray:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8", errors="replace")
    elif isinstance(source, str):
        # Coordinate text is also accepted directly; avoid treating multiline
        # text as a filesystem path.
        candidate = Path(source) if "\n" not in source and "\r" not in source else None
        text = candidate.read_text(encoding="utf-8", errors="replace") if candidate is not None and candidate.exists() else source
    elif hasattr(source, "read"):
        text = source.read()
    else:
        text = "\n".join(source)

    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) < 2:
            continue
        try:
            x, y = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y):
            rows.append((x, y))
    if len(rows) < 8:
        raise GeometryError("at least eight numeric coordinate rows are required")
    values = np.asarray(rows, dtype=float)
    # Some Lednicer files place the upper/lower point counts (for example
    # ``49 49``) after the name. That metadata is not a coordinate pair.
    if (
        len(values) > 9
        and np.all(values[0] > 2.0)
        and np.allclose(values[0], np.round(values[0]))
        and np.max(np.abs(values[1:])) <= 2.0
    ):
        values = values[1:]
    return values


def _unique_sorted_branch(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort a surface by x and average repeated x values."""
    order = np.argsort(points[:, 0])
    sorted_points = points[order]
    x_values = sorted_points[:, 0]
    unique_x, inverse = np.unique(x_values, return_inverse=True)
    y_values = np.zeros_like(unique_x)
    for i in range(len(unique_x)):
        y_values[i] = sorted_points[inverse == i, 1].mean()
    return unique_x, y_values


def parse_airfoil(source: str | Path | TextIOBase | Iterable[str], n_points: int = 100) -> AirfoilGeometry:
    """Parse common Selig or Lednicer coordinates and resample both surfaces.

    Coordinates are translated/scaled so the chord spans x=0..1. The parser
    handles common Selig (TE->LE->TE) and Lednicer (LE->TE->LE) traversal
    conventions; the branch with the larger mean ordinate is treated as the
    upper surface. Cosine spacing gives more resolution near the leading edge,
    where curvature changes rapidly.
    """
    if n_points < 4:
        raise ValueError("n_points must be at least 4")
    points = _numeric_rows(source)
    points = points[np.isfinite(points).all(axis=1)]
    # Remove consecutive duplicate rows, which are common at a closed trailing
    # edge and can otherwise create a zero-width interpolation interval.
    if len(points) > 1:
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.any(np.diff(points, axis=0) != 0, axis=1)
        points = points[keep]
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    chord = x_max - x_min
    if chord <= 1e-10:
        raise GeometryError("airfoil chord must have non-zero extent")
    points = points.copy()
    points[:, 0] = (points[:, 0] - x_min) / chord
    le_index = int(np.argmin(points[:, 0]))
    if le_index == 0:
        # Lednicer files commonly list LE->upper TE, then TE->lower LE.
        # Splitting at the interior trailing-edge maximum handles that form.
        te_index = int(np.argmax(points[:, 0]))
        if te_index <= 0 or te_index >= len(points) - 1:
            raise GeometryError("coordinates must contain points on both sides of the trailing edge")
        branch_a = points[: te_index + 1]
        branch_b = points[te_index:]
    elif le_index == len(points) - 1:
        # Accept the same convention in reverse traversal order.
        points = points[::-1]
        te_index = int(np.argmax(points[:, 0]))
        if te_index <= 0 or te_index >= len(points) - 1:
            raise GeometryError("coordinates must contain points on both sides of the trailing edge")
        branch_a = points[: te_index + 1]
        branch_b = points[te_index:]
    else:
        branch_a = points[: le_index + 1]
        branch_b = points[le_index:]
    if branch_a[:, 1].mean() >= branch_b[:, 1].mean():
        upper, lower = branch_a, branch_b
    else:
        upper, lower = branch_b, branch_a
    upper_x, upper_y = _unique_sorted_branch(upper)
    lower_x, lower_y = _unique_sorted_branch(lower)
    if upper_x[0] > 1e-5 or lower_x[0] > 1e-5:
        raise GeometryError("both surfaces must reach the leading edge")
    x_grid = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))
    upper_interp = np.interp(x_grid, upper_x, upper_y)
    lower_interp = np.interp(x_grid, lower_x, lower_y)
    return AirfoilGeometry(x_grid, upper_interp, lower_interp)


def geometry_from_file(path: str | Path, n_points: int = 100) -> AirfoilGeometry:
    return parse_airfoil(Path(path), n_points=n_points)


def write_geometry(path: str | Path, geometry: AirfoilGeometry) -> None:
    """Write canonical coordinates in a solver-friendly Selig format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, geometry.as_coordinates(), fmt="%.10f", header=target.stem, comments="")
