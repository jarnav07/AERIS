import json
from pathlib import Path

import pandas as pd
import pytest

import airfoil_ml.multi_re_batch as mrb
from airfoil_ml.multi_re_batch import FAILED_CASES_FILE, generate_batch


def _fake_solver(failures: set[str]):
    def run_xfoil_polar(coordinate_path, output_path, reynolds, **kwargs):
        if coordinate_path.stem in failures:
            raise RuntimeError(f"synthetic solver failure for {coordinate_path.stem}")
        frame = pd.DataFrame(
            {
                "airfoil_id": [coordinate_path.stem],
                "alpha_deg": [0.0],
                "reynolds": [float(reynolds)],
                "mach": [0.0],
                "cl": [0.3],
                "cd": [0.01],
                "cm": [0.0],
                "cdp": [0.005],
                "converged": [True],
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return frame

    return run_xfoil_polar


@pytest.fixture
def batch_dirs(tmp_path, monkeypatch):
    coordinates = tmp_path / "coords"
    output = tmp_path / "out"
    coordinates.mkdir()
    for stem in ("good_foil", "bad_foil"):
        (coordinates / f"{stem}.dat").write_text("0 0\n0.5 0.1\n1 0\n", encoding="utf-8")
    monkeypatch.setattr(mrb, "run_xfoil_polar", _fake_solver(failures={"bad_foil"}))
    return coordinates, output


def test_failed_cases_are_persisted_and_skipped(batch_dirs) -> None:
    coordinates, output = batch_dirs
    first = generate_batch(coordinates, output, ["good_foil", "bad_foil"], reynolds_values=[100000.0], timeout_seconds=10, workers=2)
    assert first["successful_cases"] == 1
    assert len(first["failures"]) == 1
    failed_path = output / FAILED_CASES_FILE
    assert failed_path.exists()
    records = [json.loads(line) for line in failed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["airfoil_id"] == "bad_foil"
    assert records[0]["reynolds"] == 100000.0

    before = failed_path.read_text(encoding="utf-8")
    second = generate_batch(coordinates, output, ["good_foil", "bad_foil"], reynolds_values=[100000.0], timeout_seconds=10, workers=2)
    assert second["successful_cases"] == 1
    assert second["failures"] == []
    assert failed_path.read_text(encoding="utf-8") == before
    assert (output / "good_foil_re100000.csv").exists()
    assert not (output / "bad_foil_re100000.csv").exists()


def test_completed_shard_csv_is_valid(batch_dirs) -> None:
    coordinates, output = batch_dirs
    generate_batch(coordinates, output, ["good_foil"], reynolds_values=[100000.0], timeout_seconds=10, workers=2)
    frame = pd.read_csv(output / "good_foil_re100000.csv")
    assert frame.airfoil_id.tolist() == ["good_foil"] * len(frame)
    assert {"cl", "cd", "cm"}.issubset(frame.columns)


def test_serial_and_parallel_paths_agree(batch_dirs) -> None:
    coordinates, output = batch_dirs
    serial = generate_batch(coordinates, output, ["good_foil"], reynolds_values=[30000.0, 100000.0], timeout_seconds=10, workers=1)
    parallel = generate_batch(coordinates, output, ["good_foil"], reynolds_values=[30000.0, 100000.0], timeout_seconds=10, workers=4)
    assert serial["successful_cases"] == parallel["successful_cases"] == 2
