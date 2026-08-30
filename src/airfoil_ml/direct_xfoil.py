"""Direct, headless XFOIL runner used by AERIS dataset generation.

This module deliberately bypasses AeroSandbox's XFoil wrapper. AeroSandbox is
still used for geometry/Kulfan representation, but XFOIL is driven directly
through its documented batch-command interface.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


_POLAR_COLUMNS = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]


def _parse_polar(path: Path) -> pd.DataFrame:
    rows: list[list[float]] = []
    if not path.exists():
        return pd.DataFrame(columns=_POLAR_COLUMNS)

    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 7:
            continue
        try:
            rows.append([float(value) for value in parts])
        except ValueError:
            continue

    return pd.DataFrame(rows, columns=_POLAR_COLUMNS)


def _parse_dump(path: Path) -> pd.DataFrame:
    """Parse XFOIL's DUMP output: s, x, y, Ue/Vinf, Dstar, Theta, Cf."""
    rows: list[list[float]] = []
    if not path.exists():
        raise FileNotFoundError(f"XFOIL boundary-layer dump was not created: {path}")

    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            values = [float(value) for value in parts[:7]]
        except ValueError:
            continue
        rows.append(values)

    if len(rows) < 8:
        raise ValueError(f"XFOIL boundary-layer dump contains too few data points: {path}")

    frame = pd.DataFrame(rows, columns=["s", "x", "y", "ue/vinf", "dstar", "theta", "cf"])
    frame["H"] = frame["dstar"] / frame["theta"].replace(0.0, np.nan)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _run_process(
    executable: str,
    commands: str,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [executable],
        input=commands,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        timeout=timeout,
        check=False,
    )


def _build_commands(
    airfoil_file: Path,
    polar_file: Path,
    dump_files: dict[float, Path],
    alphas: np.ndarray,
    reynolds: float,
    mach: float,
    n_crit: float,
    xtr_upper: float,
    xtr_lower: float,
    iterations: int,
) -> str:
    # XFOIL's viscous solution is path-dependent. Start at alpha=0, then
    # march outward on each side, using INIT before switching direction.
    positive = sorted(float(a) for a in alphas if a > 0.0)
    negative = sorted((float(a) for a in alphas if a < 0.0), reverse=True)
    zero_requested = any(np.isclose(alphas, 0.0, atol=1e-10))

    commands = [
        f"LOAD {airfoil_file.name}",
        "PANE",
        "OPER",
        f"VISC {reynolds:.12g}",
        f"MACH {mach:.12g}",
        "VPAR",
        f"N {n_crit:.12g}",
        f"XTR {xtr_upper:.12g} {xtr_lower:.12g}",
        "",
        f"ITER {int(iterations)}",
        "PACC",
        polar_file.name,
        "",
    ]

    def add_alpha(alpha: float) -> None:
        dump_file = dump_files[alpha]
        commands.extend([f"ALFA {alpha:.12g}", f"DUMP {dump_file.name}"])

    # Establish a benign viscous solution at zero degrees even when zero is
    # not one of the requested training points.
    commands.extend(["ALFA 0.0"])
    if zero_requested:
        add_alpha(0.0)

    for alpha in positive:
        add_alpha(alpha)

    if negative:
        commands.extend(["INIT"])
        for alpha in negative:
            add_alpha(alpha)

    commands.extend(["PACC", "", "QUIT", ""])
    return "\n".join(commands)


def run_xfoil_case(
    airfoil,
    operating: dict[str, object],
    executable: str,
    iterations: int,
    timeout: int,
    working_directory: Path,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], str]:
    """Run one airfoil/operating-point case directly through XFOIL."""
    working_directory.mkdir(parents=True, exist_ok=True)
    normalized = airfoil.normalize().to_airfoil(n_coordinates_per_side=160)
    airfoil_file = working_directory / "airfoil.dat"
    airfoil_file.write_text(normalized.write_dat(include_name=True), encoding="utf-8")

    requested = np.asarray(operating["alphas"], dtype=float)
    unique_alphas = sorted({float(a) for a in requested})
    polar_file = working_directory / "polar.txt"
    dump_files = {
        alpha: working_directory / f"dump_{index:04d}.txt"
        for index, alpha in enumerate(unique_alphas)
    }

    commands = _build_commands(
        airfoil_file,
        polar_file,
        dump_files,
        np.asarray(unique_alphas),
        float(operating["reynolds"]),
        float(operating["mach"]),
        float(operating["n_crit"]),
        float(operating["xtr_upper"]),
        float(operating["xtr_lower"]),
        iterations,
    )

    try:
        result = _run_process(executable, commands, working_directory, timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"XFOIL timed out after {timeout}s") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not launch XFOIL executable '{executable}': {exc}") from exc

    polar = _parse_polar(polar_file)
    dumps: dict[float, pd.DataFrame] = {}
    for alpha, path in dump_files.items():
        if path.exists():
            try:
                dumps[alpha] = _parse_dump(path)
            except ValueError:
                continue

    if polar.empty:
        tail = result.stdout[-4000:].strip()
        raise RuntimeError(
            f"XFOIL produced no usable polar points (return code {result.returncode}).\n{tail}"
        )

    return polar, dumps, result.stdout
