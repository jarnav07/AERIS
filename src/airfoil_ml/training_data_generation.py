"""Generate the single canonical XFOIL dataset from random Kulfan mixtures.

The geometry sampler mirrors the core strategy used by NeuralFoil: choose three
parent airfoils, draw random simplex weights, mix their 18 Kulfan parameters,
apply a lognormal scale, and optionally perturb the result using the covariance
of the source database.

This project deliberately fixes XFOIL's transition settings for the canonical
experiment. The learned model therefore maps only geometry + alpha + Reynolds
+ Mach to Cl/Cd/Cm, which keeps later CFD comparison well-defined.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd

from .data import KULFAN_COLUMNS


@dataclass(frozen=True)
class TrainingDataConfig:
    """Sampling and XFOIL settings for one reproducible dataset."""

    n_airfoils_to_combine: int = 3
    alpha_grid: tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    alpha_jitter_uniform: float = 2.5
    alpha_jitter_normal_sigma: float = 2.5
    log10_re_mean: float = 5.5
    log10_re_sigma: float = 1.5
    mach: float = 0.0
    n_crit: float = 9.0
    xtr_upper: float = 1.0
    xtr_lower: float = 1.0
    scale_log_sigma: float = 0.25
    covariance_perturbation: bool = True
    xfoil_iterations: int = 200
    xfoil_timeout: int = 30
    max_shape_attempts: int = 25


@dataclass(frozen=True)
class KulfanDatabase:
    airfoils: tuple[asb.KulfanAirfoil, ...]
    mean: np.ndarray
    covariance: np.ndarray


def _kulfan_vector(airfoil: asb.KulfanAirfoil) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(airfoil.upper_weights, dtype=float),
            np.asarray(airfoil.lower_weights, dtype=float),
            np.atleast_1d(airfoil.leading_edge_weight).astype(float),
            np.atleast_1d(airfoil.TE_thickness).astype(float),
        ]
    )


def load_kulfan_database(coordinates_dir: str | Path | None = None) -> KulfanDatabase:
    """Load a Kulfan source database from AeroSandbox or a directory of .dat files."""
    if coordinates_dir is None:
        database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
        paths = sorted(database_path.glob("*.dat"))
        airfoils = tuple(
            asb.Airfoil(name=path.stem).normalize().to_kulfan_airfoil()
            for path in paths
        )
    else:
        database_path = Path(coordinates_dir)
        paths = sorted(database_path.glob("*.dat"))
        airfoils = tuple(
            asb.Airfoil(coordinates=path).normalize().to_kulfan_airfoil()
            for path in paths
        )

    if not airfoils:
        raise FileNotFoundError(f"no .dat airfoils found in {database_path}")

    vectors = np.stack([_kulfan_vector(airfoil) for airfoil in airfoils])
    return KulfanDatabase(
        airfoils=airfoils,
        mean=np.mean(vectors, axis=0),
        covariance=np.cov(vectors, rowvar=False),
    )


def sample_airfoil(
    database: KulfanDatabase,
    rng: np.random.Generator,
    config: TrainingDataConfig,
) -> asb.KulfanAirfoil:
    """Create one stochastic three-parent Kulfan airfoil."""
    if config.n_airfoils_to_combine != 3:
        raise ValueError("the canonical experiment requires exactly three parent airfoils")

    cuts = np.sort(rng.random(2))
    weights = np.diff(np.concatenate(([0.0], cuts, [1.0])))
    parents = rng.choice(database.airfoils, size=3, replace=True)

    airfoil = asb.KulfanAirfoil(
        name="sampled-training-airfoil",
        upper_weights=np.dot(weights, np.stack([p.upper_weights for p in parents])),
        lower_weights=np.dot(weights, np.stack([p.lower_weights for p in parents])),
        leading_edge_weight=float(
            np.dot(weights, [p.leading_edge_weight for p in parents])
        ),
        TE_thickness=float(np.dot(weights, [p.TE_thickness for p in parents])),
    )

    airfoil = airfoil.scale(1.0, float(rng.lognormal(0.0, config.scale_log_sigma)))

    if config.covariance_perturbation:
        deviation = rng.multivariate_normal(
            np.zeros_like(database.mean), database.covariance
        )
        airfoil.upper_weights += deviation[:8]
        airfoil.lower_weights += deviation[8:16]
        airfoil.leading_edge_weight += deviation[16]
        airfoil.TE_thickness += deviation[17]

    return airfoil


def sample_alphas(rng: np.random.Generator, config: TrainingDataConfig) -> np.ndarray:
    """Sample one jittered seven-point alpha pattern."""
    return (
        np.asarray(config.alpha_grid, dtype=float)
        + rng.uniform(-config.alpha_jitter_uniform, config.alpha_jitter_uniform)
        + config.alpha_jitter_normal_sigma * rng.standard_normal(len(config.alpha_grid))
    )


def sample_reynolds(rng: np.random.Generator, config: TrainingDataConfig) -> float:
    return float(10 ** (config.log10_re_mean + config.log10_re_sigma * rng.standard_normal()))


def _is_valid_shape(airfoil: asb.KulfanAirfoil) -> bool:
    vector = _kulfan_vector(airfoil)
    if not np.isfinite(vector).all() or float(airfoil.TE_thickness) < 0:
        return False
    try:
        return bool(airfoil.as_shapely_polygon().is_valid)
    except Exception:
        # Geometry validity is best-effort if Shapely support is unavailable.
        return True


def _xfoil_command(executable: str) -> str:
    """Use Xvfb automatically for headless Linux runs when available."""
    if os.environ.get("DISPLAY") or shutil.which("xvfb-run") is None:
        return executable
    return f"xvfb-run --auto-servernum {executable}"


def run_xfoil(
    airfoil: asb.KulfanAirfoil,
    alphas: np.ndarray,
    reynolds: float,
    config: TrainingDataConfig,
    xfoil_executable: str,
) -> list[dict[str, float]]:
    """Run XFOIL and return only finite, positive-drag operating points."""
    solver = asb.XFoil(
        airfoil=airfoil,
        Re=float(reynolds),
        mach=config.mach,
        n_crit=config.n_crit,
        xtr_upper=config.xtr_upper,
        xtr_lower=config.xtr_lower,
        xfoil_repanel=True,
        max_iter=config.xfoil_iterations,
        timeout=config.xfoil_timeout,
        xfoil_command=_xfoil_command(xfoil_executable),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        outputs = solver.alpha(np.asarray(alphas, dtype=float))

    returned_alphas = np.asarray(outputs.get("alpha", []), dtype=float)
    if returned_alphas.size == 0:
        return []

    rows: list[dict[str, float]] = []
    for requested_alpha in np.asarray(alphas, dtype=float):
        index = int(np.argmin(np.abs(returned_alphas - requested_alpha)))
        if abs(returned_alphas[index] - requested_alpha) > 1e-3:
            continue
        try:
            cl = float(outputs["CL"][index])
            cd = float(outputs["CD"][index])
            cm = float(outputs["CM"][index])
        except (KeyError, IndexError, TypeError):
            continue
        if not np.isfinite([cl, cd, cm]).all() or cd <= 0:
            continue
        rows.append(
            {
                "alpha_deg": float(requested_alpha),
                "reynolds": float(reynolds),
                "mach": float(config.mach),
                "cl": cl,
                "cd": cd,
                "cm": cm,
            }
        )
    return rows


def generate_training_dataset(
    output_csv: str | Path,
    coordinate_output_dir: str | Path,
    *,
    n_cases: int = 1000,
    seed: int = 42,
    database_coordinates_dir: str | Path | None = None,
    xfoil_executable: str = "xfoil",
    config: TrainingDataConfig | None = None,
) -> dict[str, object]:
    """Generate one large CSV shared by every downstream model."""
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    config = config or TrainingDataConfig()
    rng = np.random.default_rng(seed)
    database = load_kulfan_database(database_coordinates_dir)

    output_csv = Path(output_csv)
    coordinate_output_dir = Path(coordinate_output_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    coordinate_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for case_index in range(n_cases):
        case_id = f"TRAIN_{case_index:07d}"

        for _ in range(config.max_shape_attempts):
            airfoil = sample_airfoil(database, rng, config)
            if _is_valid_shape(airfoil):
                break
        else:
            failures.append({"case_id": case_id, "error": "unable to construct valid geometry"})
            continue

        airfoil_id = case_id
        airfoil.write_dat(coordinate_output_dir / f"{airfoil_id}.dat")
        alphas = sample_alphas(rng, config)
        reynolds = sample_reynolds(rng, config)

        try:
            aerodynamic_rows = run_xfoil(
                airfoil, alphas, reynolds, config, xfoil_executable
            )
        except Exception as exc:
            failures.append({"case_id": case_id, "error": str(exc)})
            continue

        geometry = dict(zip(KULFAN_COLUMNS, _kulfan_vector(airfoil).tolist()))
        for result in aerodynamic_rows:
            rows.append({**geometry, **result, "airfoil_id": airfoil_id, "case_id": case_id})

        if not aerodynamic_rows:
            failures.append({"case_id": case_id, "error": "no usable XFOIL points"})

    if not rows:
        raise RuntimeError("XFOIL produced no usable training rows")

    frame = pd.DataFrame(rows)
    frame = frame[
        [
            *KULFAN_COLUMNS,
            "alpha_deg",
            "reynolds",
            "mach",
            "cl",
            "cd",
            "cm",
            "airfoil_id",
            "case_id",
        ]
    ]
    frame.to_csv(output_csv, index=False)

    failures_path = output_csv.with_suffix(".failures.jsonl")
    with failures_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure) + "\n")

    manifest = {
        "generator": "kulfan_mixture_xfoil",
        "seed": seed,
        "requested_cases": n_cases,
        "successful_cases": int(frame.airfoil_id.nunique()),
        "successful_rows": len(frame),
        "failed_cases": len(failures),
        "database_size": len(database.airfoils),
        "training_columns": frame.columns.tolist(),
        "config": asdict(config),
        "failures_file": str(failures_path),
    }
    output_csv.with_suffix(".provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
