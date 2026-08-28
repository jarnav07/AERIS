"""Small geometry helpers used by the inference CLI."""

from __future__ import annotations

from pathlib import Path

import aerosandbox as asb


def kulfan_from_file(path: str | Path):
    """Load, normalise, and convert a coordinate file to Kulfan form."""
    return asb.Airfoil(coordinates=Path(path)).normalize().to_kulfan_airfoil()
