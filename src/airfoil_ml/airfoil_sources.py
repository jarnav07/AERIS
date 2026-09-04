"""Resolve, save, sample and plot the aerofoils fed to the trained surrogate.

The canonical pipeline works entirely in Kulfan/CST space (see
``training_data_generation.py``), so everything here funnels down to a single
``asb.KulfanAirfoil``: a name from AeroSandbox's bundled database, a NACA
designation, a coordinate ``.dat``/``.txt`` file, a Kulfan ``.json`` file written
by this module, or a fresh draw from the dataset generator's own sampler.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import aerosandbox as asb
import numpy as np

from .training_data_generation import TrainingDataConfig, load_kulfan_database, sample_airfoil

_COORDINATE_SUFFIXES = {".dat", ".txt", ".csv"}


def kulfan_to_dict(airfoil: asb.KulfanAirfoil) -> dict[str, object]:
    """Serialise an aerofoil to the exact 18 numbers the models consume."""
    return {
        "name": str(airfoil.name),
        "upper_weights": [float(w) for w in np.asarray(airfoil.upper_weights)],
        "lower_weights": [float(w) for w in np.asarray(airfoil.lower_weights)],
        "leading_edge_weight": float(airfoil.leading_edge_weight),
        "TE_thickness": float(airfoil.TE_thickness),
    }


def kulfan_from_dict(payload: dict[str, object]) -> asb.KulfanAirfoil:
    return asb.KulfanAirfoil(
        name=str(payload.get("name", "airfoil")),
        upper_weights=np.asarray(payload["upper_weights"], dtype=float),
        lower_weights=np.asarray(payload["lower_weights"], dtype=float),
        leading_edge_weight=float(payload.get("leading_edge_weight", 0.0)),
        TE_thickness=float(payload.get("TE_thickness", 0.0)),
    )


def resolve_airfoil(spec: str) -> asb.KulfanAirfoil:
    """Turn a user-supplied aerofoil specification into a ``KulfanAirfoil``.

    ``spec`` may be a path to a Kulfan ``.json`` (round-trips exactly), a path to a
    coordinate file, or a name AeroSandbox can resolve — either one of its bundled
    database aerofoils (``e63``, ``s1223``, ...) or a NACA designation
    (``naca2412``). Coordinate-derived aerofoils are normalised before the CST fit,
    matching how the training-data generator built its parent aerofoils.
    """
    path = Path(spec).expanduser()
    if path.is_file():
        if path.suffix.lower() == ".json":
            return kulfan_from_dict(json.loads(path.read_text(encoding="utf-8")))
        if path.suffix.lower() not in _COORDINATE_SUFFIXES:
            raise ValueError(
                f"don't know how to read aerofoil file {path} "
                f"(expected a Kulfan .json or a coordinate file: {sorted(_COORDINATE_SUFFIXES)})"
            )
        airfoil = asb.Airfoil(name=path.stem, coordinates=path)
    else:
        airfoil = asb.Airfoil(name=spec)

    if airfoil.coordinates is None:
        raise ValueError(
            f"could not resolve aerofoil {spec!r}: it is not a readable file, an AeroSandbox "
            "database aerofoil, or a NACA designation (e.g. 'naca2412')"
        )
    return airfoil.normalize().to_kulfan_airfoil()


def sample_random_airfoils(
    count: int = 1,
    seed: int | None = None,
    config: TrainingDataConfig | None = None,
) -> list[asb.KulfanAirfoil]:
    """Draw aerofoils from the same sampler the training dataset was generated with.

    Uses ``training_data_generation.sample_airfoil`` unchanged, so generated shapes
    are in-distribution for the surrogate rather than arbitrary CST vectors.
    """
    config = config or TrainingDataConfig()
    database = load_kulfan_database()
    rng = np.random.default_rng(seed)
    airfoils = []
    for index in range(count):
        airfoil = sample_airfoil(database, rng, config)
        airfoil.name = f"sampled_{index:04d}"
        airfoils.append(airfoil)
    return airfoils


def save_airfoil(airfoil: asb.KulfanAirfoil, directory: str | Path, stem: str | None = None) -> dict[str, Path]:
    """Write both a Kulfan ``.json`` and a coordinate ``.dat`` for one aerofoil."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = stem or str(airfoil.name)
    json_path = directory / f"{stem}.json"
    dat_path = directory / f"{stem}.dat"
    json_path.write_text(json.dumps(kulfan_to_dict(airfoil), indent=2) + "\n", encoding="utf-8")
    airfoil.write_dat(str(dat_path))
    return {"json": json_path, "dat": dat_path}


def describe_airfoil(airfoil: asb.KulfanAirfoil) -> dict[str, float]:
    """Headline geometric properties, for printing alongside a prediction."""
    return {
        "max_thickness": float(airfoil.max_thickness()),
        "max_camber": float(airfoil.max_camber()),
        "LE_radius": float(airfoil.LE_radius()),
        "TE_angle_deg": float(airfoil.TE_angle()),
        "area": float(airfoil.area()),
    }


def plot_airfoils(airfoils: list[asb.KulfanAirfoil], path: str | Path, columns: int = 3) -> Path:
    """Render a grid of aerofoil sections to a PNG and return the written path."""
    import matplotlib.pyplot as plt

    if not airfoils:
        raise ValueError("no aerofoils to plot")

    columns = max(1, min(columns, len(airfoils)))
    rows = int(np.ceil(len(airfoils) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 2.2 * rows), squeeze=False)
    for axis, airfoil in zip(axes.ravel(), airfoils):
        coordinates = airfoil.coordinates
        axis.plot(coordinates[:, 0], coordinates[:, 1], color="#1f77b4", linewidth=1.4)
        axis.fill(coordinates[:, 0], coordinates[:, 1], color="#1f77b4", alpha=0.12)
        axis.set_title(
            f"{airfoil.name}\nt/c={airfoil.max_thickness():.3f}  camber={airfoil.max_camber():.3f}",
            fontsize=9,
        )
        # Equal aspect is essential here: an auto-scaled y axis makes a 12%-thick
        # section look like a blimp and hides genuinely degenerate shapes.
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
    for axis in axes.ravel()[len(airfoils):]:
        axis.axis("off")

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def open_in_viewer(path: str | Path) -> bool:
    """Open a written file with the platform image viewer; ``False`` if unavailable.

    ``evaluation.py`` pins matplotlib to the headless Agg backend at import time
    (and this package is imported on headless generation VMs), so ``plt.show()``
    is not a reliable way to *view* anything here — write the PNG, then hand it to
    the OS.
    """
    path = str(Path(path))
    command = {"darwin": ["open", path], "win32": ["cmd", "/c", "start", "", path]}.get(sys.platform, ["xdg-open", path])
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False
