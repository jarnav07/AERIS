"""Acquire geometry from UIUC and generate reference polars with XFOIL.

The aerodynamic labels are solver outputs, not analytical or synthetic values.
The provenance file records the source, command, and solver settings.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from .geometry import geometry_from_file, write_geometry

UIUC_COORDINATE_URL = "https://m-selig.ae.illinois.edu/ads/coord/{airfoil_id}.dat"
UIUC_COORDINATE_ARCHIVE_URL = "https://m-selig.ae.illinois.edu/ads/coord_seligFmt.zip"


def safe_solver_name(name: str, max_len: int = 24) -> str:
    """Short, whitespace-free filename for XFOIL's space-delimited commands.

    UIUC/Kanakaero coordinate stems routinely contain spaces and punctuation
    (for example ``74130 WP2``), which XFOIL's LOAD/PACC commands would
    misparse. The sanitized slug is truncated and suffixed with a short hash
    of the original name to keep collisions effectively impossible while
    remaining inside XFOIL's short filename buffer.
    """
    slug = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    slug = slug.strip("_") or "airfoil"
    slug = slug[:max_len]
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _request_bytes(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ml-airfoil-predictor/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_uiuc_coordinates(airfoil_ids: Iterable[str], output_dir: str | Path, timeout: int = 30) -> list[Path]:
    """Download selected UIUC coordinate files, falling back to the official archive.

    UIUC periodically changes the individual-file directory layout. The archive
    URL is the stable endpoint advertised by the database page, so a 404 from
    the legacy individual URL is handled without changing the data source.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_ids = tuple(str(airfoil_id) for airfoil_id in airfoil_ids)
    downloaded: list[Path] = []
    archive: zipfile.ZipFile | None = None
    try:
        for airfoil_id in requested_ids:
            target = output_dir / f"{airfoil_id}.dat"
            url = UIUC_COORDINATE_URL.format(airfoil_id=airfoil_id)
            try:
                target.write_bytes(_request_bytes(url, timeout))
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                if archive is None:
                    archive = zipfile.ZipFile(io.BytesIO(_request_bytes(UIUC_COORDINATE_ARCHIVE_URL, timeout)))
                matches = [name for name in archive.namelist() if Path(name).name.lower() == f"{airfoil_id}.dat".lower()]
                if not matches:
                    raise FileNotFoundError(f"{airfoil_id}.dat was not found in the UIUC coordinate archive") from error
                target.write_bytes(archive.read(matches[0]))
            downloaded.append(target)
    finally:
        if archive is not None:
            archive.close()
    return downloaded


def _parse_xfoil_polar(path: Path, airfoil_id: str, reynolds: float, mach: float) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            alpha, cl, cd, cdp, cm = map(float, fields[:5])
        except ValueError:
            continue
        rows.append({"airfoil_id": airfoil_id, "alpha_deg": alpha, "reynolds": reynolds, "mach": mach, "cl": cl, "cd": cd, "cm": cm, "cdp": cdp, "converged": True})
    if not rows:
        raise RuntimeError(f"XFOIL produced no parseable polar rows in {path}")
    return pd.DataFrame(rows)


def run_xfoil_polar(
    coordinate_path: str | Path,
    output_path: str | Path,
    reynolds: float,
    alpha_start: float = -10.0,
    alpha_end: float = 20.0,
    alpha_step: float = 1.0,
    mach: float = 0.0,
    xfoil_executable: str = "xfoil",
    iterations: int = 100,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    """Run XFOIL's viscous panel solver for one airfoil/Re condition."""
    coordinate_path, output_path = Path(coordinate_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # XFOIL 6.99 has a short filename buffer and space-delimited commands, so
    # keep solver inputs and outputs in one working directory under sanitized
    # short relative filenames derived from (not identical to) the airfoil id.
    solver_stem = safe_solver_name(coordinate_path.stem)
    solver_coordinate_path = output_path.parent / f"{solver_stem}.dat"
    write_geometry(solver_coordinate_path, geometry_from_file(coordinate_path, n_points=100))
    polar_path = output_path.parent / f"{solver_stem}.polar"
    # PACC appends if a polar file already exists; remove stale output so
    # rerunning an experiment is idempotent rather than duplicating rows.
    polar_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    script = "\n".join([
        f"LOAD {solver_coordinate_path.name}",
        "PANE",
        "OPER",
        f"VISC {reynolds:g}",
        f"MACH {mach:g}",
        f"ITER {iterations}",
        "PACC",
        polar_path.name,
        "",
        f"ASEQ {alpha_start:g} {alpha_end:g} {alpha_step:g}",
        "PACC",
        "QUIT",
        "",
    ])
    command = shlex.split(xfoil_executable)
    if not command:
        raise ValueError("xfoil_executable must not be empty")
    # Debian's XFOIL build initializes X11 even for batch analysis. Use a
    # virtual display automatically in headless workspaces.
    if not os.environ.get("DISPLAY") and shutil.which("xvfb-run") and command[0] != "xvfb-run":
        command = ["xvfb-run", "-a", *command]
    completed = subprocess.run(command, input=script, text=True, capture_output=True, timeout=timeout_seconds, cwd=output_path.parent)
    if not polar_path.exists():
        details = (completed.stderr + "\n" + completed.stdout)[-2000:]
        raise RuntimeError(f"XFOIL failed for {coordinate_path.name}: {details}")
    airfoil_id = coordinate_path.stem
    # XFOIL 6.99 can return non-zero after writing a valid polar because the
    # batch input reaches EOF while still inside OPER. Parsing is the decisive
    # check: a missing/empty/unparseable file still raises below.
    frame = _parse_xfoil_polar(polar_path, airfoil_id, reynolds, mach)
    frame.to_csv(output_path, index=False)
    return frame


def generate_dataset(
    coordinates_dir: str | Path,
    output_csv: str | Path,
    reynolds_values: Iterable[float],
    airfoil_ids: Iterable[str] | None = None,
    **xfoil_kwargs: object,
) -> pd.DataFrame:
    """Generate and combine XFOIL labels for selected coordinate files and Re."""
    coordinates_dir = Path(coordinates_dir)
    if airfoil_ids is None:
        coordinates = sorted(coordinates_dir.glob("*.dat"))
    else:
        coordinates = []
        for airfoil_id in airfoil_ids:
            candidate = coordinates_dir / f"{airfoil_id}.dat"
            if not candidate.exists():
                raise FileNotFoundError(f"no coordinate file found for requested airfoil {airfoil_id}")
            coordinates.append(candidate)
    reynolds_values = tuple(float(value) for value in reynolds_values)
    if not reynolds_values:
        raise ValueError("at least one Reynolds number is required")
    if not coordinates:
        raise FileNotFoundError(f"no .dat coordinate files found in {coordinates_dir}")
    frames = []
    failures: list[dict[str, str | float]] = []
    for coordinate_path in coordinates:
        for reynolds in reynolds_values:
            partial = Path(output_csv).parent / "_xfoil" / f"{coordinate_path.stem}_re{int(reynolds)}.csv"
            try:
                frames.append(run_xfoil_polar(coordinate_path, partial, reynolds=float(reynolds), **xfoil_kwargs))
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                # Preserve successful solver cases and record failed cases; do
                # not replace missing aerodynamic labels with synthetic data.
                failures.append({"airfoil_id": coordinate_path.stem, "reynolds": reynolds, "error": str(error)})
    if not frames:
        raise RuntimeError("XFOIL produced no usable polar cases")
    result = pd.concat(frames, ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    provenance = {
        "geometry_source": "UIUC Airfoil Coordinates Database",
        "geometry_url": "https://m-selig.ae.illinois.edu/ads/coord_database.html",
        "geometry_archive_url": UIUC_COORDINATE_ARCHIVE_URL,
        "aerodynamic_source": "XFOIL viscous panel/boundary-layer solver",
        "reynolds": list(reynolds_values),
        "xfoil_kwargs": xfoil_kwargs,
        "rows": len(result),
        "successful_cases": int(len(frames)),
        "failed_cases": failures,
    }
    Path(output_csv).with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return result
