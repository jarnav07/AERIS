"""Dataset assembly and reproducible grouped splitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .geometry import AirfoilGeometry, geometry_from_file

REQUIRED_POLAR_COLUMNS = ("airfoil_id", "alpha_deg", "reynolds", "mach", "cl", "cd", "cm")


@dataclass
class AeroDataset:
    frame: pd.DataFrame
    geometries: dict[str, AirfoilGeometry]

    def validate(self) -> None:
        missing = set(REQUIRED_POLAR_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"polar data is missing columns: {sorted(missing)}")
        if self.frame.empty:
            raise ValueError("polar data is empty")
        numeric = [c for c in REQUIRED_POLAR_COLUMNS if c != "airfoil_id"]
        if not np.isfinite(self.frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError("polar data contains non-finite values")
        if (self.frame["reynolds"] <= 0).any():
            raise ValueError("Reynolds numbers must be positive")
        if (self.frame["cd"] < 0).any():
            raise ValueError("negative drag coefficients are not physically valid")
        unknown = set(self.frame["airfoil_id"].astype(str)) - set(self.geometries)
        if unknown:
            raise ValueError(f"no coordinate file was found for airfoils: {sorted(unknown)}")

    def grouped_split(
        self,
        test_fraction: float = 0.2,
        validation_fraction: float = 0.2,
        seed: int = 42,
        test_airfoils: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Split by airfoil ID so no geometry appears in multiple partitions.

        When ``test_airfoils`` is given, exactly those identities form the test
        partition (a fixed holdout, e.g. reused from a previous experiment so
        results are directly comparable). The remaining identities are split
        into train/validation with the same grouped, leakage-safe logic.
        """
        if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
            raise ValueError("fractions must be positive and sum to less than one")
        all_ids = np.array(sorted(self.frame["airfoil_id"].astype(str).unique()))
        if len(all_ids) < 3:
            raise ValueError("at least three distinct airfoils are required for grouped splitting")
        rng = np.random.default_rng(seed)
        if test_airfoils is not None:
            test_ids = np.array(sorted(set(str(a) for a in test_airfoils)))
            unknown = set(test_ids) - set(all_ids)
            if unknown:
                raise ValueError(f"test airfoils not present in the dataset: {sorted(unknown)}")
            remaining = np.array(sorted(set(all_ids) - set(test_ids)))
            remaining = rng.permutation(remaining)
            n_val = max(1, int(round(len(remaining) * validation_fraction)))
            if n_val >= len(remaining):
                n_val = 1
            val_ids, train_ids = remaining[:n_val], remaining[n_val:]
        else:
            ids = rng.permutation(all_ids)
            n_test = max(1, int(round(len(ids) * test_fraction)))
            n_val = max(1, int(round(len(ids) * validation_fraction)))
            if n_test + n_val >= len(ids):
                n_test, n_val = 1, 1
            test_ids, val_ids, train_ids = ids[:n_test], ids[n_test : n_test + n_val], ids[n_test + n_val :]
        return {
            "train": np.flatnonzero(self.frame.airfoil_id.astype(str).isin(train_ids)),
            "validation": np.flatnonzero(self.frame.airfoil_id.astype(str).isin(val_ids)),
            "test": np.flatnonzero(self.frame.airfoil_id.astype(str).isin(test_ids)),
            "train_airfoils": train_ids,
            "validation_airfoils": val_ids,
            "test_airfoils": test_ids,
        }


def load_dataset(polar_csv: str | Path, coordinates_dir: str | Path, n_geometry_points: int = 100) -> AeroDataset:
    frame = pd.read_csv(polar_csv)
    frame["airfoil_id"] = frame["airfoil_id"].astype(str)
    geometries: dict[str, AirfoilGeometry] = {}
    coordinates_dir = Path(coordinates_dir)
    for airfoil_id in frame.airfoil_id.unique():
        candidates = [coordinates_dir / f"{airfoil_id}.dat", coordinates_dir / f"{airfoil_id}.txt", coordinates_dir / airfoil_id]
        coordinate_path = next((p for p in candidates if p.exists()), None)
        if coordinate_path is None:
            raise FileNotFoundError(f"no coordinate file found for {airfoil_id} in {coordinates_dir}")
        geometries[airfoil_id] = geometry_from_file(coordinate_path, n_points=n_geometry_points)
    dataset = AeroDataset(frame, geometries)
    dataset.validate()
    return dataset


def save_split_manifest(path: str | Path, split: dict[str, np.ndarray], seed: int) -> None:
    manifest = {"seed": seed}
    for key in ("train_airfoils", "validation_airfoils", "test_airfoils"):
        manifest[key] = split[key].tolist()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
