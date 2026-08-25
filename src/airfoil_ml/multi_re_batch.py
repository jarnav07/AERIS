"""Large multi-Reynolds XFOIL dataset generation.

Sharded by (airfoil, Reynolds) so a large run can be interrupted and resumed.
Every produced row comes from a parseable XFOIL polar; failed cases are
recorded in a manifest and never replaced with synthetic values.

Parallelism: XFOIL is a subprocess-bound single-threaded solver, so workers
run as threads that each spawn their own XFOIL process under its own scratch
directory. XFOIL writes a fixed-name boundary-layer scratch file (``:00.bl``)
into its working directory, so per-case scratch directories are mandatory
when workers > 1.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .acquisition import run_xfoil_polar, safe_solver_name
from .geometry import GeometryError

REYNOLDS_DEFAULT = (30000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0)
FAILED_CASES_FILE = "_failed_cases.jsonl"


def build_id_geometry_map(coordinates_dir: str | Path, manifest_path: str | Path) -> dict[str, str]:
    """Map airfoil identifiers to canonical coordinate filenames."""
    coordinates_dir = Path(coordinates_dir)
    canonical: dict[str, str] = {}
    for path in sorted(coordinates_dir.glob("*.dat")):
        canonical[path.stem] = path.name
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    return canonical


def _load_failed_cases(path: Path) -> set[tuple[str, float]]:
    """Load recorded (airfoil_id, reynolds) solver failures, if any."""
    if not path.exists():
        return set()
    records: set[tuple[str, float]] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records.add((str(record["airfoil_id"]), float(record["reynolds"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return records


def _append_failed_cases(path: Path, failures: list[dict[str, str | float]]) -> None:
    """Append failure records so interrupted batches never retry them."""
    lines = [json.dumps(failure) + "\n" for failure in failures]
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)


def _process_case(
    coordinate_path: Path,
    shard: Path,
    reynolds: float,
    *,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    mach: float,
    timeout_seconds: int,
    xfoil_executable: str,
    scratch_root: Path,
) -> tuple[pd.DataFrame | None, dict[str, str | float] | None]:
    """Run one (airfoil, Re) case in its own scratch directory.

    Returns (frame, None) on success or (None, failure_record) on solver,
    timeout, or geometry errors. The finished shard is atomically moved into
    the standard output layout so interrupted runs resume from complete files.
    """
    scratch = scratch_root / f"{safe_solver_name(coordinate_path.stem)}_re{int(reynolds)}"
    scratch.mkdir(parents=True, exist_ok=True)
    work_shard = scratch / "out.csv"
    try:
        frame = run_xfoil_polar(
            coordinate_path,
            work_shard,
            reynolds=float(reynolds),
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=alpha_step,
            mach=mach,
            timeout_seconds=timeout_seconds,
            xfoil_executable=xfoil_executable,
        )
    except (RuntimeError, subprocess.TimeoutExpired, GeometryError) as error:
        return None, {"airfoil_id": coordinate_path.stem, "reynolds": reynolds, "error": str(error)}
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_name(shard.name + ".tmp")
    temporary.write_bytes(work_shard.read_bytes())
    temporary.replace(shard)
    return frame, None


def generate_batch(
    coordinates_dir: str | Path,
    output_dir: str | Path,
    airfoil_ids: list[str],
    reynolds_values: list[float] | tuple[float, ...] = REYNOLDS_DEFAULT,
    alpha_start: float = -2.0,
    alpha_end: float = 10.0,
    alpha_step: float = 1.0,
    mach: float = 0.0,
    timeout_seconds: int = 45,
    xfoil_executable: str = "xfoil",
    workers: int = 1,
) -> dict[str, int | list[dict[str, str | float]]]:
    """Run one shard per airfoil/Re case and combine validated rows."""
    coordinates_dir = Path(coordinates_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str | float]] = []
    tasks: list[tuple[Path, Path, float]] = []
    failed_cases_path = output_dir / FAILED_CASES_FILE
    previously_failed = _load_failed_cases(failed_cases_path)
    for airfoil_id in airfoil_ids:
        coordinate_path = coordinates_dir / f"{airfoil_id}.dat"
        if not coordinate_path.exists():
            failures.append({"airfoil_id": airfoil_id, "reynolds": 0.0, "error": "coordinate file missing"})
            continue
        for reynolds in reynolds_values:
            shard = output_dir / f"{airfoil_id}_re{int(reynolds)}.csv"
            key = (airfoil_id, float(reynolds))
            if key in previously_failed:
                # A previously recorded solver failure is never retried, so a
                # handful of pathological geometries cannot consume the whole
                # batch budget on every invocation.
                continue
            if shard.exists() and shard.stat().st_size > 0:
                # Resumable: an existing parseable shard is never regenerated.
                try:
                    existing = pd.read_csv(shard)
                    if len(existing) > 0 and "airfoil_id" in existing.columns:
                        frames.append(existing)
                        continue
                except pd.errors.EmptyDataError:
                    pass
            tasks.append((coordinate_path, shard, float(reynolds)))

    scratch_root = output_dir / "_work"
    common = {
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "alpha_step": alpha_step,
        "mach": mach,
        "timeout_seconds": timeout_seconds,
        "xfoil_executable": xfoil_executable,
        "scratch_root": scratch_root,
    }
    if workers > 1 and tasks:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_case, *task, **common) for task in tasks]
            for future in futures:
                frame, failure = future.result()
                if frame is not None:
                    frames.append(frame)
                else:
                    failures.append(failure)  # type: ignore[arg-type]
                    _append_failed_cases(failed_cases_path, [failure])  # type: ignore[list-item]
    else:
        for coordinate_path, shard, reynolds in tasks:
            frame, failure = _process_case(coordinate_path, shard, reynolds, **common)
            if frame is not None:
                frames.append(frame)
            else:
                failures.append(failure)  # type: ignore[arg-type]
                _append_failed_cases(failed_cases_path, [failure])  # type: ignore[list-item]
    # Merge every finished shard in the output directory so interrupted batch
    # runs still produce one authoritative, cumulative dataset.
    all_shards = sorted(output_dir.glob("*_re*.csv"))
    shard_frames = [pd.read_csv(shard) for shard in all_shards if shard.stat().st_size > 0]
    if not shard_frames:
        raise RuntimeError("no parseable XFOIL shards found")
    combined = pd.concat(shard_frames, ignore_index=True)
    combined.to_csv(output_dir / "combined.csv", index=False)
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                "coordinates_dir": str(coordinates_dir),
                "airfoils_requested": airfoil_ids,
                "reynolds_values": list(reynolds_values),
                "alpha": [alpha_start, alpha_end, alpha_step],
                "mach": mach,
                "workers": workers,
                "rows": len(combined),
                "successful_cases": len(shard_frames),
                "shards": len(shard_frames),
                "failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"rows": len(combined), "successful_cases": len(shard_frames), "failures": failures}
