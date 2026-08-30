"""Reproducible stochastic training-data generation for the airfoil surrogate.

The generator uses Kulfan geometries derived from AeroSandbox's built-in
airfoil database to form a covariance matrix for random sampling. XFOIL is
run directly through its batch interface; AeroSandbox is not used as the
XFOIL execution wrapper.

Parallel processing and sharding are used to support large-scale generation
across multiple CPUs.
"""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd
from tqdm import tqdm

from .direct_xfoil import run_xfoil_case

N_BL_POINTS = 32
BL_X_POINTS = (np.arange(N_BL_POINTS, dtype=float) + 0.5) / N_BL_POINTS


def _column_names() -> list[str]:
    return [
        *[f"kulfan_upper_{i}" for i in range(8)],
        *[f"kulfan_lower_{i}" for i in range(8)],
        "kulfan_LE_weight", "kulfan_TE_thickness",
        "alpha", "Re", "mach", "n_crit", "xtr_upper", "xtr_lower",
        "analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr",
        *[f"upper_bl_theta_{i}" for i in range(N_BL_POINTS)],
        *[f"upper_bl_H_{i}" for i in range(N_BL_POINTS)],
        *[f"upper_bl_ue_vinf_{i}" for i in range(N_BL_POINTS)],
        *[f"lower_bl_theta_{i}" for i in range(N_BL_POINTS)],
        *[f"lower_bl_H_{i}" for i in range(N_BL_POINTS)],
        *[f"lower_bl_ue_vinf_{i}" for i in range(N_BL_POINTS)],
    ]


TRAINING_VECTOR_COLUMNS = _column_names()
TRAINING_VECTOR_SIZE = len(TRAINING_VECTOR_COLUMNS)


@dataclass(frozen=True)
class TrainingDataConfig:
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
    airfoils: tuple[asb.KulfanAirfoil, ...]
    mean: np.ndarray
    covariance: np.ndarray


def _kulfan_vector(airfoil: asb.KulfanAirfoil) -> np.ndarray:
    return np.concatenate([
        np.asarray(airfoil.upper_weights, dtype=float),
        np.asarray(airfoil.lower_weights, dtype=float),
        np.atleast_1d(airfoil.leading_edge_weight).astype(float),
        np.atleast_1d(airfoil.TE_thickness).astype(float),
    ])


def load_kulfan_database() -> KulfanDatabase:
    database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
    paths = sorted(database_path.glob("*.dat"))
    airfoils = tuple(asb.Airfoil(name=p.stem).normalize().to_kulfan_airfoil() for p in paths)
    if not airfoils:
        raise FileNotFoundError(f"No .dat airfoils found in {database_path}")
    kulfans = np.stack([_kulfan_vector(airfoil) for airfoil in airfoils])
    return KulfanDatabase(airfoils, np.mean(kulfans, axis=0), np.cov(kulfans, rowvar=False))


def sample_airfoil(database: KulfanDatabase, rng: np.random.Generator, config: TrainingDataConfig) -> asb.KulfanAirfoil:
    cuts = np.sort(rng.random(config.n_airfoils_to_combine - 1))
    weights = np.diff(np.concatenate(([0.0], cuts, [1.0])))
    parents = rng.choice(database.airfoils, size=config.n_airfoils_to_combine, replace=True)
    airfoil = asb.KulfanAirfoil(
        name="Sampled training airfoil",
        upper_weights=np.dot(weights, np.stack([p.upper_weights for p in parents])),
        lower_weights=np.dot(weights, np.stack([p.lower_weights for p in parents])),
        leading_edge_weight=float(np.dot(weights, [p.leading_edge_weight for p in parents])),
        TE_thickness=float(np.dot(weights, [p.TE_thickness for p in parents])),
    )
    airfoil = airfoil.scale(1.0, float(rng.lognormal(0.0, config.scale_log_sigma)))
    deviation = rng.multivariate_normal(np.zeros_like(database.mean), database.covariance)
    airfoil.upper_weights += deviation[:8]
    airfoil.lower_weights += deviation[8:16]
    airfoil.leading_edge_weight += deviation[16]
    airfoil.TE_thickness += deviation[17]
    return airfoil


def sample_operating_point(rng: np.random.Generator, config: TrainingDataConfig) -> dict[str, object]:
    alphas = (
        np.asarray(config.alpha_grid, dtype=float)
        + rng.uniform(-config.alpha_jitter_uniform, config.alpha_jitter_uniform)
        + config.alpha_jitter_normal_sigma * rng.standard_normal()
    )
    reynolds = float(10 ** (config.log10_re_mean + config.log10_re_sigma * rng.standard_normal()))
    n_crit = float(rng.uniform(config.n_crit_min, config.n_crit_max))

    def transition() -> float:
        return 1.0 if rng.random() < config.forced_transition_probability else float(rng.uniform(0.0, 1.0))

    return {
        "alphas": alphas,
        "reynolds": reynolds,
        "mach": config.mach,
        "n_crit": n_crit,
        "xtr_upper": transition(),
        "xtr_lower": transition(),
    }


def _split_boundary_layer(bl_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = bl_data["x"].to_numpy(float)
    dx = np.diff(x)
    negative, positive = np.flatnonzero(dx < 0), np.flatnonzero(dx > 0)
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("could not identify upper/lower boundary-layer surfaces")
    upper = bl_data.iloc[:negative[-1] + 2].iloc[::-1].copy()
    lower = bl_data.iloc[positive[0]:].copy()
    if len(upper) <= 4 or len(lower) <= 4:
        raise ValueError("boundary-layer surface data is too short")
    return upper, lower


def _interpolate_surface(data: pd.DataFrame, value_column: str) -> np.ndarray:
    x, y = data["x"].to_numpy(float), data[value_column].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        raise ValueError("boundary-layer data is too short")
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique = np.concatenate(([True], np.diff(x) > 1e-10))
    x, y = x[unique], y[unique]
    if len(x) < 4:
        raise ValueError("boundary-layer data has insufficient unique x points")
    return np.interp(BL_X_POINTS, x, y)


def _analyse_airfoil_worker(
    airfoil: asb.KulfanAirfoil,
    operating: dict[str, object],
    config: TrainingDataConfig,
    xfoil_executable: str,
    airfoil_id: str,
    output_dir: Path,
) -> tuple[str, list[np.ndarray], str | None]:
    """Run one stochastic case using the direct XFOIL batch interface."""
    del output_dir
    with tempfile.TemporaryDirectory(prefix=f"xfoil_{airfoil_id}_") as tempdir:
        cwd = Path(tempdir)
        try:
            polar, dumps, stdout = run_xfoil_case(
                airfoil=airfoil,
                operating=operating,
                executable=xfoil_executable,
                iterations=config.xfoil_iterations,
                timeout=config.xfoil_timeout,
                working_directory=cwd,
            )
        except Exception as exc:
            return airfoil_id, [], str(exc)

    returned_alphas = polar["alpha"].to_numpy(float)
    if len(returned_alphas) == 0:
        return airfoil_id, [], "no converged/usable XFOIL points"

    vectors: list[np.ndarray] = []
    interpolation_failures: list[str] = []
    requested_alphas = np.asarray(operating["alphas"], dtype=float)

    for requested_alpha in requested_alphas:
        differences = np.abs(returned_alphas - requested_alpha)
        if np.min(differences) > 1e-3:
            interpolation_failures.append(f"alpha {requested_alpha:.3f}: not returned by XFOIL")
            continue

        index = int(np.argmin(differences))
        actual_alpha = float(polar.iloc[index]["alpha"])
        try:
            bl_data = dumps.get(actual_alpha)
            if bl_data is None:
                raise ValueError("boundary-layer dump was not produced")
            upper, lower = _split_boundary_layer(bl_data)
            fields = [
                _interpolate_surface(upper, "theta"),
                _interpolate_surface(upper, "H"),
                _interpolate_surface(upper, "ue/vinf"),
                _interpolate_surface(lower, "theta"),
                _interpolate_surface(lower, "H"),
                _interpolate_surface(lower, "ue/vinf"),
            ]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            interpolation_failures.append(f"alpha {requested_alpha:.3f}: {exc}")
            continue

        row = polar.iloc[index]
        vector = np.concatenate([
            _kulfan_vector(airfoil),
            np.array([
                requested_alpha,
                operating["reynolds"], operating["mach"], operating["n_crit"],
                operating["xtr_upper"], operating["xtr_lower"], 1.0,
                row["CL"], row["CD"], row["CM"], row["Top_Xtr"], row["Bot_Xtr"],
            ], dtype=float),
            *fields,
        ]).astype(np.float32)

        if len(vector) != TRAINING_VECTOR_SIZE:
            return airfoil_id, [], f"unexpected training-vector size: {len(vector)}"
        vectors.append(vector)

    if not vectors:
        detail = "; ".join(interpolation_failures[:4])
        if stdout:
            detail = f"{detail}; XFOIL tail: {stdout[-500:].strip()}"
        return airfoil_id, [], f"no boundary layer points could be interpolated ({detail})"

    return airfoil_id, vectors, None


def generate_training_dataset(
    output_dir: str | Path,
    *,
    n_cases: int = 100,
    seed: int = 42,
    workers: int = 1,
    xfoil_executable: str = "xfoil",
    config: TrainingDataConfig = TrainingDataConfig(),
    resume: bool = False,
    include_cylinder_augmentation: bool = True,
) -> dict[str, object]:
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")

    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    database = load_kulfan_database()

    cases = []
    for case_index in range(n_cases):
        airfoil = sample_airfoil(database, rng, config)
        operating = sample_operating_point(rng, config)
        airfoil_id = f"TRAIN_{case_index:06d}"
        cases.append((airfoil_id, airfoil, operating))

    existing_shards = set()
    if resume:
        for shard in shards_dir.glob("*.parquet"):
            existing_shards.add(shard.stem)

    cases_to_run = [case for case in cases if case[0] not in existing_shards]
    failures = []

    if cases_to_run:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _analyse_airfoil_worker,
                    airfoil,
                    operating,
                    config,
                    xfoil_executable,
                    airfoil_id,
                    output_dir,
                )
                for airfoil_id, airfoil, operating in cases_to_run
            ]

            with tqdm(total=len(cases_to_run), desc="Generating data") as pbar:
                for future in as_completed(futures):
                    airfoil_id, vectors, error = future.result()
                    if error:
                        failures.append({"airfoil_id": airfoil_id, "error": error})
                    else:
                        frame = pd.DataFrame(np.vstack(vectors), columns=TRAINING_VECTOR_COLUMNS)
                        frame.insert(0, "airfoil_id", airfoil_id)
                        frame.to_parquet(shards_dir / f"{airfoil_id}.parquet", index=False)
                    pbar.update(1)

    all_frames = [
        pd.read_parquet(shard)
        for shard in tqdm(sorted(shards_dir.glob("*.parquet")), desc="Combining shards")
    ]

    if not all_frames:
        failure_preview = "\n".join(
            f"  {item['airfoil_id']}: {item['error']}" for item in failures[:10]
        )
        raise RuntimeError(
            "No usable XFOIL training vectors were generated.\n"
            f"Failure details:\n{failure_preview or 'No worker failures were recorded.'}"
        )

    combined = pd.concat(all_frames, ignore_index=True)
    if include_cylinder_augmentation:
        combined = pd.concat([combined, generate_cylinder_rows()], ignore_index=True)

    output_csv = output_dir / "training_data.csv"
    combined.to_csv(output_csv, index=False)

    manifest = {
        "generator": "training_data_generation",
        "xfoil_runner": "direct_batch",
        "seed": seed,
        "requested_cases": n_cases,
        "successful_vectors": int(combined[combined["airfoil_id"] != "CYLINDER"].shape[0]),
        "successful_airfoils": int(combined[combined["airfoil_id"] != "CYLINDER"]["airfoil_id"].nunique()),
        "cylinder_rows": int((combined["airfoil_id"] == "CYLINDER").sum()),
        "failed_cases": failures,
        "training_vector_size": TRAINING_VECTOR_SIZE,
        "boundary_layer_points_per_surface": N_BL_POINTS,
        "database_size": len(database.airfoils),
        "sampling_config": asdict(config),
    }
    output_csv.with_suffix(".provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


_CYLINDER_CONFIDENCE = 0.1


def generate_cylinder_rows(
    reynolds_values: tuple[float, ...] = (1e4, 1e5, 1e6),
    alpha_values: tuple[float, ...] = tuple(np.linspace(-180.0, 180.0, 37)),
) -> pd.DataFrame:
    records = []
    kulfan_zeros = np.zeros(18, dtype=np.float32)

    for reynolds in reynolds_values:
        for alpha in alpha_values:
            vector = np.concatenate([
                kulfan_zeros,
                np.array([
                    float(alpha), float(reynolds), 0.0, 9.0,
                    1.0, 1.0,
                    _CYLINDER_CONFIDENCE,
                    0.0, 1.0, 0.0,
                    0.5, 0.5,
                ], dtype=np.float32),
                np.zeros(6 * N_BL_POINTS, dtype=np.float32),
            ])
            records.append(vector)

    frame = pd.DataFrame(np.vstack(records), columns=TRAINING_VECTOR_COLUMNS)
    frame.insert(0, "airfoil_id", "CYLINDER")
    return frame
