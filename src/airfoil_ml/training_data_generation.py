"""Reproducible stochastic training-data generation for the airfoil surrogate.

The generator follows the stochastic geometry/operating-point strategy used by
NeuralFoil, but keeps this project self-contained. XFOIL supplies the full
aerodynamic and boundary-layer labels.

Each successful operating point is stored as one fixed-width vector containing
18 Kulfan geometry parameters, 6 flow/transition inputs, solver confidence,
5 aerodynamic outputs, and 192 boundary-layer values: 222 numeric values per
sample.

Column "Re" stores the **raw** Reynolds number (not log10). The feature
preprocessor in ``features.py`` applies log10 internally before model input.

Cylinder augmentation rows
--------------------------
``generate_cylinder_rows()`` produces analytic blunt-body records using simple
crossflow drag models.  These are stored with ``analysis_confidence = 0.1`` so
the model learns to extrapolate conservatively toward degenerate geometries
rather than blow up.  This mirrors NeuralFoil's ``generate_cylinder_data.py``.

Parallelism
-----------
``generate_training_dataset()`` accepts an optional ``workers`` argument.
When ``workers > 1`` each airfoil case is dispatched to its own subprocess via
``ProcessPoolExecutor`` (not threads) so multiple XFOIL processes run
simultaneously without GIL contention.

Resume support
--------------
Rows are flushed to the output CSV every ``checkpoint_interval`` successfully
analysed airfoils.  If the process is killed midway the partial CSV is valid
and the run can be restarted with a new ``seed`` to top up the dataset.
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd

N_BL_POINTS = 32
BL_X_POINTS = (np.arange(N_BL_POINTS, dtype=float) + 0.5) / N_BL_POINTS

# Confidence value assigned to stalled-regime points (past CL peak).
_STALL_CONFIDENCE = 0.8
# Confidence value assigned to analytic cylinder-augmentation rows.
_CYLINDER_CONFIDENCE = 0.1


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


def load_kulfan_database(coordinates_dir: str | Path | None = None) -> KulfanDatabase:
    if coordinates_dir is None:
        database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
        paths = sorted(database_path.glob("*.dat"))
        airfoils = tuple(asb.Airfoil(name=p.stem).normalize().to_kulfan_airfoil() for p in paths)
    else:
        database_path = Path(coordinates_dir)
        paths = sorted(database_path.glob("*.dat"))
        airfoils = tuple(asb.Airfoil(coordinates=p).normalize().to_kulfan_airfoil() for p in paths)
    if not airfoils:
        raise FileNotFoundError(f"No .dat airfoils found in {database_path}")
    kulfans = np.stack([_kulfan_vector(airfoil) for airfoil in airfoils])
    return KulfanDatabase(airfoils, np.mean(kulfans, axis=0), np.cov(kulfans, rowvar=False))


def sample_airfoil(database: KulfanDatabase, rng: np.random.Generator, config: TrainingDataConfig) -> asb.KulfanAirfoil:
    """Sample a stochastic airfoil via Dirichlet convex combination + perturbation.

    Improvements vs original:
    - ``TE_thickness`` is clamped to ``>= 0`` after covariance perturbation to
      prevent XFOIL failures from physically impossible negative trailing-edge
      thickness.
    """
    if config.n_airfoils_to_combine < 2:
        raise ValueError("n_airfoils_to_combine must be at least 2")
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
    # Clamp TE_thickness: negative values are non-physical and crash XFOIL.
    airfoil.TE_thickness = max(0.0, airfoil.TE_thickness + deviation[17])
    return airfoil


def sample_operating_point(rng: np.random.Generator, config: TrainingDataConfig) -> dict[str, object]:
    """Sample one set of operating conditions for a full alpha sweep.

    Fix vs original: a **single** shared offset (uniform + normal) is drawn and
    applied to every alpha in the grid.  Previously the uniform draw was inside
    the grid construction expression and evaluated once (correct), but the
    intent was ambiguous.  Now both draws are explicit and documented, matching
    NeuralFoil's approach of shifting the whole polar sweep by one random
    offset so the sweep stays internally consistent.
    """
    uniform_offset = rng.uniform(-config.alpha_jitter_uniform, config.alpha_jitter_uniform)
    normal_offset = config.alpha_jitter_normal_sigma * rng.standard_normal()
    alphas = np.asarray(config.alpha_grid, dtype=float) + uniform_offset + normal_offset
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


def _analyse_airfoil(
    airfoil: asb.KulfanAirfoil,
    operating: dict[str, object],
    config: TrainingDataConfig,
    xfoil_executable: str,
) -> list[np.ndarray]:
    """Run XFOIL and build training vectors for each converged alpha.

    Improvement: points in the post-stall regime (where CL has dropped below
    the polar peak) are marked with ``analysis_confidence = _STALL_CONFIDENCE``
    (0.8) rather than 1.0.  This lets the model learn that stalled-regime
    predictions are inherently less reliable, mirroring NeuralFoil's confidence
    encoding.
    """
    xf = asb.XFoil(
        airfoil=airfoil.normalize().to_kulfan_airfoil(),
        Re=float(operating["reynolds"]), mach=float(operating["mach"]),
        n_crit=float(operating["n_crit"]), xtr_upper=float(operating["xtr_upper"]),
        xtr_lower=float(operating["xtr_lower"]), xfoil_repanel=True,
        max_iter=config.xfoil_iterations, timeout=config.xfoil_timeout,
        xfoil_command=xfoil_executable, include_bl_data=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        outputs = xf.alpha(np.asarray(operating["alphas"], dtype=float))
    returned_alphas = np.asarray(outputs.get("alpha", []), dtype=float)
    if len(returned_alphas) == 0:
        return []

    # Determine the stall boundary: index of maximum CL in the returned polar.
    cl_values = np.asarray(outputs.get("CL", []), dtype=float)
    peak_index = int(np.argmax(cl_values)) if len(cl_values) > 0 else len(returned_alphas)

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

        # Assign reduced confidence past the CL peak (stalled regime).
        confidence = _STALL_CONFIDENCE if index > peak_index else 1.0

        vector = np.concatenate([
            _kulfan_vector(airfoil),
            np.array([
                requested_alpha, operating["reynolds"], operating["mach"], operating["n_crit"],
                operating["xtr_upper"], operating["xtr_lower"], confidence,
                outputs["CL"][index], outputs["CD"][index], outputs["CM"][index],
                outputs["Top_Xtr"][index], outputs["Bot_Xtr"][index],
            ]),
            *fields,
        ]).astype(np.float32)
        if len(vector) != TRAINING_VECTOR_SIZE:
            raise RuntimeError(f"unexpected training-vector size: {len(vector)}")
        vectors.append(vector)
    return vectors


def generate_cylinder_rows(
    reynolds_values: tuple[float, ...] = (1e4, 1e5, 1e6),
    alpha_values: tuple[float, ...] = tuple(np.linspace(-180.0, 180.0, 37)),
) -> pd.DataFrame:
    """Generate analytic circular-cylinder augmentation rows.

    A circular cylinder is the maximally blunt airfoil.  Anchoring the model
    with cylinder drag at extreme conditions prevents unbounded extrapolation.
    Rows use ``analysis_confidence = 0.1`` so they contribute weakly to
    the primary aerodynamic targets.

    Kulfan parameters for a unit circle approximation: all weights = 0 except a
    large leading-edge weight, with a thick trailing edge.  We use the mean
    Kulfan vector from the database (zeros) because the cylinder's geometry is
    encoded implicitly through the confidence flag — these are augmentation rows
    only, not geometry prediction rows.

    Drag model: Cd ≈ 1.0 for a cylinder (classical crossflow value), CL ≈ 0,
    CM ≈ 0.  BL fields are all set to NaN-equivalent zeros.
    """
    records = []
    kulfan_zeros = np.zeros(18, dtype=np.float32)  # 8+8+1+1

    for reynolds in reynolds_values:
        for alpha in alpha_values:
            vector = np.concatenate([
                kulfan_zeros,
                np.array([
                    float(alpha), float(reynolds), 0.0, 9.0,  # n_crit=9 nominal
                    1.0, 1.0,                                   # fully forced transition
                    _CYLINDER_CONFIDENCE,
                    0.0, 1.0, 0.0,                             # CL=0, CD=1 (cylinder), CM=0
                    0.5, 0.5,                                   # transition midchord
                ], dtype=np.float32),
                np.zeros(6 * N_BL_POINTS, dtype=np.float32),
            ])
            records.append(vector)

    frame = pd.DataFrame(np.vstack(records), columns=TRAINING_VECTOR_COLUMNS)
    frame.insert(0, "airfoil_id", "CYLINDER")
    return frame


def _process_one_case(
    case_index: int,
    seed: int,
    database_coordinates_dir: str | None,
    xfoil_executable: str,
    config_dict: dict,
) -> tuple[list[np.ndarray], str | None]:
    """Worker function: sample one airfoil + operating point and analyse it.

    Runs in a subprocess when ``workers > 1``.  Returns (vectors, error_str).
    """
    config = TrainingDataConfig(**config_dict)
    # Each case gets its own deterministic sub-RNG derived from the global seed
    # so results are reproducible regardless of worker ordering.
    rng = np.random.default_rng([seed, case_index])
    database = load_kulfan_database(database_coordinates_dir)
    airfoil = sample_airfoil(database, rng, config)
    operating = sample_operating_point(rng, config)
    try:
        vectors = _analyse_airfoil(airfoil, operating, config, xfoil_executable)
        return vectors, None
    except Exception as exc:
        return [], str(exc)


def generate_training_dataset(
    output_csv: str | Path,
    coordinate_output_dir: str | Path,
    *,
    n_cases: int = 100,
    seed: int = 42,
    database_coordinates_dir: str | Path | None = None,
    xfoil_executable: str = "xfoil",
    config: TrainingDataConfig = TrainingDataConfig(),
    workers: int = 1,
    checkpoint_interval: int = 100,
    include_cylinder_augmentation: bool = True,
) -> dict[str, object]:
    """Generate a training dataset of XFOIL-labelled Kulfan airfoil vectors.

    Parameters
    ----------
    output_csv:
        Destination CSV path.  Written incrementally every
        ``checkpoint_interval`` airfoils so partial results survive crashes.
    coordinate_output_dir:
        Directory where sampled airfoil ``.dat`` files are written.
    n_cases:
        Number of distinct airfoil/operating-point pairs to attempt.
    seed:
        Global random seed.  Sub-RNGs are derived as ``[seed, case_index]``
        so results are reproducible and order-independent (safe for parallel).
    workers:
        Number of parallel XFOIL subprocesses.  ``1`` = sequential (default).
        Values ``> 1`` use ``ProcessPoolExecutor``; each worker runs its own
        XFOIL subprocess with no shared state.
    checkpoint_interval:
        Flush collected rows to ``output_csv`` every this many *attempted*
        cases.  Set to ``n_cases`` to disable incremental flushing.
    include_cylinder_augmentation:
        If ``True``, append analytic cylinder rows (``confidence = 0.1``) after
        the XFOIL rows.  These anchor the model at degenerate blunt geometries.
    """
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    rng = np.random.default_rng(seed)
    # Write per-airfoil .dat files using the global rng (sequential, for coord output).
    database = load_kulfan_database(database_coordinates_dir)
    output_csv, coordinate_output_dir = Path(output_csv), Path(coordinate_output_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    coordinate_output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-sample all airfoils so .dat files are written before workers start.
    airfoil_ids: list[str] = []
    for case_index in range(n_cases):
        airfoil = sample_airfoil(database, rng, config)
        airfoil_id = f"TRAIN_{case_index:06d}"
        airfoil.write_dat(coordinate_output_dir / f"{airfoil_id}.dat")
        airfoil_ids.append(airfoil_id)

    config_dict = asdict(config)
    # Convert alpha_grid tuple to list for serialisation compatibility.
    config_dict["alpha_grid"] = list(config_dict["alpha_grid"])

    rows: list[np.ndarray] = []
    row_airfoil_ids: list[str] = []
    failures: list[dict[str, str]] = []

    # Helper: flush current rows to CSV (append if file exists).
    def _flush(rows_buf: list[np.ndarray], ids_buf: list[str]) -> None:
        if not rows_buf:
            return
        chunk = pd.DataFrame(np.vstack(rows_buf), columns=TRAINING_VECTOR_COLUMNS)
        chunk.insert(0, "airfoil_id", ids_buf)
        write_header = not output_csv.exists()
        chunk.to_csv(output_csv, mode="a", index=False, header=write_header)

    pending_rows: list[np.ndarray] = []
    pending_ids: list[str] = []

    if workers > 1:
        db_str = str(database_coordinates_dir) if database_coordinates_dir is not None else None
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_id = {
                pool.submit(_process_one_case, i, seed, db_str, xfoil_executable, config_dict): airfoil_ids[i]
                for i in range(n_cases)
            }
            for done_count, future in enumerate(as_completed(future_to_id), start=1):
                airfoil_id = future_to_id[future]
                try:
                    vectors, error = future.result()
                except Exception as exc:
                    failures.append({"airfoil_id": airfoil_id, "error": str(exc)})
                else:
                    if error:
                        failures.append({"airfoil_id": airfoil_id, "error": error})
                    elif not vectors:
                        failures.append({"airfoil_id": airfoil_id, "error": "no converged/usable XFOIL points"})
                    else:
                        pending_rows.extend(vectors)
                        pending_ids.extend([airfoil_id] * len(vectors))
                        rows.extend(vectors)
                        row_airfoil_ids.extend([airfoil_id] * len(vectors))

                if done_count % checkpoint_interval == 0 and pending_rows:
                    _flush(pending_rows, pending_ids)
                    pending_rows.clear()
                    pending_ids.clear()
    else:
        for case_index in range(n_cases):
            airfoil_id = airfoil_ids[case_index]
            # Re-derive per-case rng for consistency with parallel path.
            case_rng = np.random.default_rng([seed, case_index])
            case_database = load_kulfan_database(database_coordinates_dir)
            airfoil = sample_airfoil(case_database, case_rng, config)
            operating = sample_operating_point(case_rng, config)
            try:
                vectors = _analyse_airfoil(airfoil, operating, config, xfoil_executable)
                if not vectors:
                    failures.append({"airfoil_id": airfoil_id, "error": "no converged/usable XFOIL points"})
                else:
                    pending_rows.extend(vectors)
                    pending_ids.extend([airfoil_id] * len(vectors))
                    rows.extend(vectors)
                    row_airfoil_ids.extend([airfoil_id] * len(vectors))
            except Exception as exc:
                failures.append({"airfoil_id": airfoil_id, "error": str(exc)})

            if (case_index + 1) % checkpoint_interval == 0 and pending_rows:
                _flush(pending_rows, pending_ids)
                pending_rows.clear()
                pending_ids.clear()

    # Final flush of any remaining rows.
    _flush(pending_rows, pending_ids)

    if not rows:
        raise RuntimeError("No usable XFOIL training vectors were generated")

    # Reload full accumulated CSV (handles resume / multi-flush scenario).
    frame = pd.read_csv(output_csv)

    # Append cylinder augmentation rows.
    if include_cylinder_augmentation:
        cylinder_frame = generate_cylinder_rows()
        frame = pd.concat([frame, cylinder_frame], ignore_index=True)
        frame.to_csv(output_csv, index=False)

    manifest = {
        "generator": "training_data_generation",
        "seed": seed,
        "requested_cases": n_cases,
        "successful_vectors": int(frame[frame["airfoil_id"] != "CYLINDER"].shape[0]),
        "successful_airfoils": int(frame[frame["airfoil_id"] != "CYLINDER"]["airfoil_id"].nunique()),
        "cylinder_rows": int((frame["airfoil_id"] == "CYLINDER").sum()),
        "failed_cases": failures,
        "training_vector_size": TRAINING_VECTOR_SIZE,
        "boundary_layer_points_per_surface": N_BL_POINTS,
        "database_size": len(database.airfoils),
        "workers": workers,
        "checkpoint_interval": checkpoint_interval,
        "sampling_config": config_dict,
    }
    output_csv.with_suffix(".provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
