"""Reproducible stochastic training-data generation for the airfoil surrogate.

The generator follows the stochastic geometry/operating-point strategy used by
NeuralFoil. It uses Kulfan geometries derived from aerosandbox's built-in
airfoil database to form a covariance matrix for random sampling.

Parallel processing and sharding are used to support large-scale generation
across multiple CPUs.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    alphas = (np.asarray(config.alpha_grid, dtype=float)
              + rng.uniform(-config.alpha_jitter_uniform, config.alpha_jitter_uniform)
              + config.alpha_jitter_normal_sigma * rng.standard_normal())
    reynolds = float(10 ** (config.log10_re_mean + config.log10_re_sigma * rng.standard_normal()))
    n_crit = float(rng.uniform(config.n_crit_min, config.n_crit_max))
    def transition() -> float:
        return 1.0 if rng.random() < config.forced_transition_probability else float(rng.uniform(0.0, 1.0))
    return {"alphas": alphas, "reynolds": reynolds, "mach": config.mach,
            "n_crit": n_crit, "xtr_upper": transition(), "xtr_lower": transition()}


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
    output_dir: Path
) -> tuple[str, list[np.ndarray], str | None]:
    """Worker function to run XFOIL in a separate temporary directory."""
    with tempfile.TemporaryDirectory(prefix=f"xfoil_{airfoil_id}_") as tempdir:
        cwd = Path(tempdir)
        try:
            xf = asb.XFoil(
                airfoil=airfoil.normalize().to_kulfan_airfoil(),
                Re=float(operating["reynolds"]), mach=float(operating["mach"]),
                n_crit=float(operating["n_crit"]), xtr_upper=float(operating["xtr_upper"]),
                xtr_lower=float(operating["xtr_lower"]), xfoil_repanel=True,
                max_iter=config.xfoil_iterations, timeout=config.xfoil_timeout,
                xfoil_command=xfoil_executable, include_bl_data=True,
                working_directory=str(cwd)
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                outputs = xf.alpha(np.asarray(operating["alphas"], dtype=float))
        except Exception as e:
            return airfoil_id, [], str(e)

    returned_alphas = np.asarray(outputs.get("alpha", []), dtype=float)
    if len(returned_alphas) == 0:
        return airfoil_id, [], "no converged/usable XFOIL points"

    vectors = []
    for requested_alpha in np.asarray(operating["alphas"], dtype=float):
        differences = np.abs(returned_alphas - requested_alpha)
        if np.min(differences) > 1e-3:
            continue
        index = int(np.argmin(differences))
        try:
            upper, lower = _split_boundary_layer(outputs["bl_data"][index])
            fields = [
                _interpolate_surface(upper, "theta"), _interpolate_surface(upper, "H"), _interpolate_surface(upper, "ue/vinf"),
                _interpolate_surface(lower, "theta"), _interpolate_surface(lower, "H"), _interpolate_surface(lower, "ue/vinf"),
            ]
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        vector = np.concatenate([
            _kulfan_vector(airfoil),
            np.array([
                requested_alpha, operating["reynolds"], operating["mach"], operating["n_crit"],
                operating["xtr_upper"], operating["xtr_lower"], 1.0,
                outputs["CL"][index], outputs["CD"][index], outputs["CM"][index],
                outputs["Top_Xtr"][index], outputs["Bot_Xtr"][index],
            ]),
            *fields,
        ]).astype(np.float32)
        if len(vector) != TRAINING_VECTOR_SIZE:
            return airfoil_id, [], f"unexpected training-vector size: {len(vector)}"
        vectors.append(vector)
    
    if not vectors:
        return airfoil_id, [], "no boundary layer points could be interpolated"
        
    return airfoil_id, vectors, None


def generate_training_dataset(
    output_dir: str | Path,
    *,
    n_cases: int = 100,
    seed: int = 42,
    workers: int = 1,
    xfoil_executable: str = "xfoil",
    config: TrainingDataConfig = TrainingDataConfig(),
    resume: bool = False
) -> dict[str, object]:
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    database = load_kulfan_database()

    # Pre-generate parameters to ensure reproducibility regardless of workers
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

    cases_to_run = [c for c in cases if c[0] not in existing_shards]
    
    failures = []
    total_vectors = 0
    total_airfoils = 0

    if cases_to_run:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for airfoil_id, airfoil, operating in cases_to_run:
                # XFOIL uses xvfb automatically if DISPLAY is not set
                if not os.environ.get("DISPLAY") and shutil.which("xvfb-run") and not xfoil_executable.startswith("xvfb-run"):
                    cmd = f"xvfb-run -a {xfoil_executable}"
                else:
                    cmd = xfoil_executable
                futures.append(executor.submit(
                    _analyse_airfoil_worker, airfoil, operating, config, cmd, airfoil_id, output_dir
                ))
            
            with tqdm(total=len(cases_to_run), desc="Generating data") as pbar:
                for future in as_completed(futures):
                    airfoil_id, vectors, error = future.result()
                    if error:
                        failures.append({"airfoil_id": airfoil_id, "error": error})
                    else:
                        frame = pd.DataFrame(np.vstack(vectors), columns=TRAINING_VECTOR_COLUMNS)
                        frame.insert(0, "airfoil_id", airfoil_id)
                        shard_path = shards_dir / f"{airfoil_id}.parquet"
                        frame.to_parquet(shard_path, index=False)
                    pbar.update(1)

    # Combine shards
    all_frames = []
    for shard in tqdm(sorted(shards_dir.glob("*.parquet")), desc="Combining shards"):
        all_frames.append(pd.read_parquet(shard))
    
    if not all_frames:
        raise RuntimeError("No usable XFOIL training vectors were generated")
        
    combined = pd.concat(all_frames, ignore_index=True)
    output_csv = output_dir / "training_data.csv"
    combined.to_csv(output_csv, index=False)
    
    manifest = {
        "generator": "training_data_generation", "seed": seed, "requested_cases": n_cases,
        "successful_vectors": len(combined), "successful_airfoils": int(combined.airfoil_id.nunique()),
        "failed_cases": failures, "training_vector_size": TRAINING_VECTOR_SIZE,
        "boundary_layer_points_per_surface": N_BL_POINTS, "database_size": len(database.airfoils),
        "sampling_config": asdict(config),
    }
    output_csv.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
