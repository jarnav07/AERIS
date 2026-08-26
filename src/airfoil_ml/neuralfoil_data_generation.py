"""NeuralFoil-style stochastic airfoil/data generation.

This module reproduces the core distribution used by NeuralFoil's training
pipeline while fitting the project's existing geometry/XFOIL interfaces:

1. Load the available UIUC/Selig airfoil database.
2. Normalize each airfoil and convert it to Kulfan parameters.
3. Randomly blend three parent airfoils using convex weights.
4. Add a multivariate-normal perturbation sampled from the database covariance.
5. Randomly scale the chord.
6. Sample alpha, Reynolds number, Ncrit and transition locations using the
   same distributions as NeuralFoil.
7. Run XFOIL and retain only the solver-generated labels.

The generator is deliberately configurable and finite: unlike NeuralFoil's
original worker loop, callers choose the number of operating-point batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .acquisition import run_xfoil_polar, safe_solver_name
from .geometry import AirfoilGeometry, geometry_from_file, write_geometry

try:
    import aerosandbox as asb
except ImportError:  # pragma: no cover - dependency is checked at runtime
    asb = None


@dataclass(frozen=True)
class NeuralFoilSamplingConfig:
    """Sampling parameters matching NeuralFoil's current data generator."""

    n_airfoils_to_combine: int = 3
    alpha_grid: tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    alpha_jitter_uniform: float = 2.5
    alpha_jitter_normal_sigma: float = 2.5
    log10_re_mean: float = 5.5
    log10_re_sigma: float = 1.5
    n_crit_min: float = 0.0
    n_crit_max: float = 18.0
    forced_transition_probability: float = 0.8
    scale_log_sigma: float = 0.25
    mach: float = 0.0
    xfoil_iterations: int = 200
    xfoil_timeout: int = 30


@dataclass(frozen=True)
class KulfanDatabase:
    airfoils: tuple[object, ...]
    mean: np.ndarray
    covariance: np.ndarray


def _require_aerosandbox() -> None:
    if asb is None:
        raise RuntimeError(
            "The NeuralFoil-style generator requires aerosandbox. "
            "Install it with `pip install aerosandbox`."
        )


def load_kulfan_database(coordinates_dir: str | Path | None = None) -> KulfanDatabase:
    """Load and parameterize a Kulfan airfoil database.

    When *coordinates_dir* is omitted, use AeroSandbox's bundled airfoil
    database, as NeuralFoil does. A project-local directory may also be
    supplied to make the sampling distribution entirely project-owned.
    """
    _require_aerosandbox()

    if coordinates_dir is None:
        database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
        airfoils = tuple(
            asb.Airfoil(name=filename.stem).normalize().to_kulfan_airfoil()
            for filename in database_path.glob("*.dat")
        )
    else:
        database_path = Path(coordinates_dir)
        airfoils = tuple(
            asb.Airfoil(name=path.stem, coordinates=geometry_from_file(path).coordinates)
            .normalize()
            .to_kulfan_airfoil()
            for path in sorted(database_path.glob("*.dat"))
        )

    if not airfoils:
        raise FileNotFoundError(f"No .dat airfoils found in {database_path}")

    kulfans = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(airfoil.upper_weights, dtype=float),
                    np.asarray(airfoil.lower_weights, dtype=float),
                    np.atleast_1d(airfoil.leading_edge_weight),
                    np.atleast_1d(airfoil.TE_thickness),
                ]
            )
            for airfoil in airfoils
        ],
        axis=0,
    )
    return KulfanDatabase(
        airfoils=airfoils,
        mean=np.mean(kulfans, axis=0),
        covariance=np.cov(kulfans, rowvar=False),
    )


def sample_kulfan_airfoil(
    database: KulfanDatabase,
    rng: np.random.Generator,
    config: NeuralFoilSamplingConfig = NeuralFoilSamplingConfig(),
):
    """Draw one stochastic Kulfan airfoil using NeuralFoil's distribution."""
    _require_aerosandbox()
    if config.n_airfoils_to_combine < 2:
        raise ValueError("n_airfoils_to_combine must be at least 2")

    cuts = np.sort(rng.random(config.n_airfoils_to_combine - 1))
    cuts = np.concatenate(([0.0], cuts, [1.0]))
    weights = np.diff(cuts)
    parents = rng.choice(database.airfoils, size=config.n_airfoils_to_combine, replace=True)

    def blend(values: Iterable[float]) -> float:
        return float(np.dot(weights, np.asarray(values, dtype=float)))

    upper_weights = np.dot(
        weights, np.stack([parent.upper_weights for parent in parents]),
    )
    lower_weights = np.dot(
        weights, np.stack([parent.lower_weights for parent in parents]),
    )
    leading_edge_weight = blend(parent.leading_edge_weight for parent in parents)
    te_thickness = blend(parent.TE_thickness for parent in parents)

    airfoil = asb.KulfanAirfoil(
        name="NeuralFoil sampled airfoil",
        upper_weights=upper_weights,
        lower_weights=lower_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=te_thickness,
    )

    airfoil = airfoil.scale(1.0, float(rng.lognormal(0.0, config.scale_log_sigma)))
    deviation = rng.multivariate_normal(np.zeros_like(database.mean), database.covariance)
    airfoil.upper_weights += deviation[:8]
    airfoil.lower_weights += deviation[8:16]
    airfoil.leading_edge_weight += deviation[16]
    airfoil.TE_thickness += deviation[17]
    return airfoil


def sample_operating_point(
    rng: np.random.Generator,
    config: NeuralFoilSamplingConfig = NeuralFoilSamplingConfig(),
) -> dict[str, object]:
    """Sample alpha sweep and viscous-operation parameters."""
    alpha = (
        np.asarray(config.alpha_grid, dtype=float)
        + rng.uniform(-config.alpha_jitter_uniform, config.alpha_jitter_uniform)
        + config.alpha_jitter_normal_sigma * rng.standard_normal()
    )
    reynolds = float(10 ** (config.log10_re_mean + config.log10_re_sigma * rng.standard_normal()))
    n_crit = float(rng.uniform(config.n_crit_min, config.n_crit_max))

    def transition() -> float:
        return 1.0 if rng.random() < config.forced_transition_probability else float(rng.uniform(0.0, 1.0))

    return {
        "alphas": alpha,
        "reynolds": reynolds,
        "mach": config.mach,
        "n_crit": n_crit,
        "xtr_upper": transition(),
        "xtr_lower": transition(),
    }


def generate_neuralfoil_style_dataset(
    output_csv: str | Path,
    coordinate_output_dir: str | Path,
    *,
    n_cases: int = 100,
    seed: int = 42,
    database_coordinates_dir: str | Path | None = None,
    xfoil_executable: str = "xfoil",
    config: NeuralFoilSamplingConfig = NeuralFoilSamplingConfig(),
) -> dict[str, object]:
    """Generate a finite NeuralFoil-style XFOIL dataset.

    Each sampled geometry gets one coordinate file and a CSV of all successful
    alpha points. Geometry and solver parameters remain traceable through the
    generated file names and row columns.
    """
    _require_aerosandbox()
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")

    rng = np.random.default_rng(seed)
    database = load_kulfan_database(database_coordinates_dir)
    coordinate_output_dir = Path(coordinate_output_dir)
    coordinate_output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for case_index in range(n_cases):
        airfoil = sample_kulfan_airfoil(database, rng, config)
        operating = sample_operating_point(rng, config)
        airfoil_id = f"NF_{case_index:06d}"

        coordinates = airfoil.coordinates
        geometry = AirfoilGeometry(
            np.asarray(coordinates[:, 0], dtype=float),
            np.asarray(coordinates[:, 1], dtype=float),
            np.asarray(coordinates[:, 1], dtype=float),
        )
        # Use AeroSandbox's own coordinate export rather than fabricating a
        # second geometry representation.
        coordinate_path = coordinate_output_dir / f"{airfoil_id}.dat"
        airfoil.write_dat(coordinate_path)

        scratch = output_csv.parent / "_xfoil_neuralfoil" / safe_solver_name(airfoil_id)
        partial = scratch / "polar.csv"
        try:
            frame = run_xfoil_polar(
                coordinate_path,
                partial,
                reynolds=float(operating["reynolds"]),
                alpha_start=float(np.min(operating["alphas"])),
                alpha_end=float(np.max(operating["alphas"])),
                alpha_step=0.1,
                mach=float(operating["mach"]),
                xfoil_executable=xfoil_executable,
                iterations=config.xfoil_iterations,
                timeout_seconds=config.xfoil_timeout,
            )
            # Restrict the dense XFOIL sweep back to the exact alpha requests
            # sampled by the NeuralFoil distribution.
            frame["requested_alpha_deg"] = np.nan
            for alpha in np.asarray(operating["alphas"], dtype=float):
                idx = (frame["alpha_deg"] - alpha).abs().idxmin()
                if abs(float(frame.loc[idx, "alpha_deg"]) - float(alpha)) <= 0.051:
                    frame.loc[idx, "requested_alpha_deg"] = float(alpha)
            frame = frame.dropna(subset=["requested_alpha_deg"]).copy()
            frame["airfoil_id"] = airfoil_id
            frame["n_crit"] = float(operating["n_crit"])
            frame["xtr_upper"] = float(operating["xtr_upper"])
            frame["xtr_lower"] = float(operating["xtr_lower"])
            frame["mach"] = float(operating["mach"])
            rows.extend(frame.to_dict("records"))
        except Exception as exc:  # preserve successful cases and diagnose failures
            failures.append({"airfoil_id": airfoil_id, "error": str(exc)})

    import pandas as pd

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No successful XFOIL cases were generated")
    result.to_csv(output_csv, index=False)

    manifest = {
        "generator": "neuralfoil_style",
        "seed": seed,
        "requested_cases": n_cases,
        "successful_cases": int(result["airfoil_id"].nunique()),
        "rows": len(result),
        "failed_cases": failures,
        "sampling_config": {
            key: value for key, value in config.__dict__.items()
        },
        "database_size": len(database.airfoils),
    }
    output_csv.with_suffix(".provenance.json").write_text(
        __import__("json").dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
