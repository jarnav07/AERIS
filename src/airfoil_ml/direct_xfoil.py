"""Direct, headless XFOIL runner used by AERIS dataset generation.

AeroSandbox is still used for the Kulfan geometry representation, but XFOIL
is driven directly through its batch-command interface. This keeps the
192-point boundary-layer targets in the dataset while avoiding the
AeroSandbox XFOIL execution path that was unreliable on the VM.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


_POLAR_COLUMNS = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]

# Debian/Ubuntu's packaged xfoil binary is built with gfortran's
# -ffpe-trap=invalid,zero,overflow, so its runtime calls _gfortran_set_fpe()
# at startup to unmask hardware floating-point exceptions. That makes XFOIL
# crash with SIGFPE on ordinary IEEE div-by-zero/invalid results (e.g. a
# zero boundary-layer momentum thickness at a stagnation point) that
# upstream XFOIL treats as benign NaN/Inf and simply reports as a warning.
# We override _gfortran_set_fpe with a no-op via LD_PRELOAD symbol
# interposition so the trap is never (re-)enabled.
_FPE_SHIM_SOURCE = """\
void _gfortran_set_fpe(int mode) {
    (void)mode;
}
"""


def _fpe_shim_path() -> str | None:
    shim = Path(tempfile.gettempdir()) / "aeris_xfoil_nofpe.so"
    if shim.exists():
        return str(shim)

    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "nofpe.c"
        source.write_text(_FPE_SHIM_SOURCE, encoding="utf-8")
        built = Path(tmp) / "nofpe.so"
        result = subprocess.run(
            [compiler, "-shared", "-fPIC", "-o", str(built), str(source)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not built.exists():
            return None
        os.replace(built, shim)

    return str(shim)


def _parse_polar(path: Path) -> pd.DataFrame:
    rows: list[list[float]] = []
    if not path.exists():
        return pd.DataFrame(columns=_POLAR_COLUMNS)

    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            rows.append([float(value) for value in parts[:7]])
        except ValueError:
            continue

    return pd.DataFrame(rows, columns=_POLAR_COLUMNS)


def _parse_dump(path: Path) -> pd.DataFrame:
    """Parse XFOIL DUMP output: s, x, y, Ue/Vinf, Dstar, Theta, Cf."""
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


def _split_boundary_layer(bl_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an XFOIL dump at the leading edge.

    XFOIL writes the upper surface from trailing edge to leading edge and the
    lower surface from leading edge back to trailing edge. Splitting at the
    minimum-x point is substantially more reliable than looking for arbitrary
    sign changes in dx, particularly when duplicate/near-duplicate points are
    present around the leading edge.
    """
    x = bl_data["x"].to_numpy(float)
    finite = np.isfinite(x)
    if finite.sum() < 8:
        raise ValueError("boundary-layer x data is too short")

    le_index = int(np.nanargmin(np.where(finite, x, np.nan)))
    upper = bl_data.iloc[: le_index + 1].copy()
    lower = bl_data.iloc[le_index:].copy()

    if len(upper) <= 4 or len(lower) <= 4:
        raise ValueError("boundary-layer surface data is too short")

    return upper, lower


def _interpolate_surface(data: pd.DataFrame, value_column: str) -> np.ndarray:
    x = data["x"].to_numpy(float)
    y = data[value_column].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        raise ValueError(f"boundary-layer {value_column} data is too short")

    order = np.argsort(x)
    x, y = x[order], y[order]
    unique = np.concatenate(([True], np.diff(x) > 1e-10))
    x, y = x[unique], y[unique]
    if len(x) < 4:
        raise ValueError(f"boundary-layer {value_column} has insufficient unique x points")

    # BL_X_POINTS in training_data_generation.py lies strictly inside (0, 1),
    # so np.interp is well behaved for the usual XFOIL surface coverage.
    return np.interp((np.arange(32, dtype=float) + 0.5) / 32.0, x, y)


def _run_process(
    executable: str,
    commands: str,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run XFOIL, automatically using Xvfb when no display is available."""
    executable_path = shutil.which(executable) if not Path(executable).is_absolute() else executable
    if executable_path is None:
        raise RuntimeError(f"XFOIL executable not found: {executable}")

    command = [executable_path]
    if not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
        command = ["xvfb-run", "--auto-servernum", executable_path]

    env = os.environ.copy()
    env["GFORTRAN_UNBUFFERED_ALL"] = "1"

    shim = _fpe_shim_path()
    if shim is not None:
        existing_preload = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = f"{shim}:{existing_preload}" if existing_preload else shim

    # xvfb-run is a shell script: it spawns Xvfb and the real xfoil binary as
    # children of its own, so a plain subprocess.run(..., timeout=...) kill
    # only reaches the xvfb-run shell itself. If xfoil hangs (observed: it
    # can sit spinning at 100% CPU indefinitely rather than exiting), the
    # orphaned xfoil process is reparented to init and survives the timeout
    # forever, permanently stealing a CPU core from the worker pool. Running
    # in a new session puts the whole xvfb-run/Xvfb/xfoil tree in one process
    # group so a timeout can kill all of it at once.
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(input=commands, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)

    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)


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
    """Build a deterministic XFOIL batch session."""
    positive = sorted(float(a) for a in alphas if a > 0.0)
    negative = sorted((float(a) for a in alphas if a < 0.0), reverse=True)
    zero_requested = any(np.isclose(alphas, 0.0, atol=1e-10))

    commands = [
        "PLOP",
        "G",
        "",
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
        # DUMP must follow a converged ALFA command. XFOIL writes the current
        # boundary-layer solution to this file. XFOIL prompts for the filename.
        commands.extend([
            f"ALFA {alpha:.12g}",
            "DUMP",
            dump_files[alpha].name
        ])

    # Establish a benign viscous solution before moving away from alpha=0.
    # If 0.0 is requested, do it via add_alpha to capture the DUMP.
    if zero_requested:
        add_alpha(0.0)
    else:
        commands.append("ALFA 0.0")

    for alpha in positive:
        add_alpha(alpha)

    # Re-initialize before sweeping the negative branch so a failed/high-alpha
    # positive solution does not poison the negative branch.
    if negative:
        commands.append("INIT")
        commands.append("ALFA 0.0")
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
        raise RuntimeError("XFOIL execution timed out.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not launch XFOIL executable '{executable}': {exc}") from exc

    polar = _parse_polar(polar_file)

    dumps: dict[float, pd.DataFrame] = {}
    for alpha, path in dump_files.items():
        if not path.exists():
            continue
        try:
            frame = _parse_dump(path)
        except ValueError:
            continue
        dumps[alpha] = frame
        dumps[float(f"{alpha:.3f}")] = frame

    if polar.empty:
        tail = result.stdout[-4000:].strip()
        raise RuntimeError(
            f"XFOIL produced no usable polar points (return code {result.returncode}).\n{tail}"
        )

    return polar, dumps, result.stdout
