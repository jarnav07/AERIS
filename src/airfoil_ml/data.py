"""Dataset schema, validation, and leakage-safe grouped splitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KULFAN_COLUMNS = (
    *[f"kulfan_upper_{i}" for i in range(8)],
    *[f"kulfan_lower_{i}" for i in range(8)],
    "kulfan_LE_weight",
    "kulfan_TE_thickness",
)
FLOW_COLUMNS = ("alpha_deg", "reynolds", "mach")
TARGET_COLUMNS = ("cl", "cd", "cm")


@dataclass
class AeroDataset:
    """The single canonical XFOIL dataset used by every model."""

    frame: pd.DataFrame

    def validate(self) -> None:
        required = set((*KULFAN_COLUMNS, *FLOW_COLUMNS, *TARGET_COLUMNS, "airfoil_id"))
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"dataset is missing columns: {sorted(missing)}")
        if self.frame.empty:
            raise ValueError("dataset is empty")

        numeric = [*KULFAN_COLUMNS, *FLOW_COLUMNS, *TARGET_COLUMNS]
        values = self.frame[numeric].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("dataset contains non-finite numeric values")
        if (self.frame["reynolds"] <= 0).any():
            raise ValueError("Reynolds numbers must be positive")
        if (self.frame["cd"] <= 0).any():
            raise ValueError("Cd must be strictly positive")
        if self.frame["airfoil_id"].astype(str).nunique() < 3:
            raise ValueError("at least three distinct airfoil identities are required")

    @property
    def airfoil_ids(self) -> np.ndarray:
        return np.array(sorted(self.frame["airfoil_id"].astype(str).unique()))

    def grouped_split(
        self,
        test_fraction: float = 0.2,
        validation_fraction: float = 0.2,
        seed: int = 42,
        test_airfoils: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Split by airfoil identity so one geometry never crosses partitions."""
        if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
            raise ValueError("fractions must be positive and sum to less than one")

        all_ids = self.airfoil_ids
        rng = np.random.default_rng(seed)

        if test_airfoils is not None:
            test_ids = np.array(sorted(set(map(str, test_airfoils))))
            unknown = set(test_ids) - set(all_ids)
            if unknown:
                raise ValueError(f"test airfoils are not present in the dataset: {sorted(unknown)}")
            remaining = rng.permutation(np.array(sorted(set(all_ids) - set(test_ids))))
            n_val = max(1, int(round(len(remaining) * validation_fraction)))
            if n_val >= len(remaining):
                raise ValueError("not enough remaining airfoils for train/validation")
            val_ids = remaining[:n_val]
            train_ids = remaining[n_val:]
        else:
            shuffled = rng.permutation(all_ids)
            n_test = max(1, int(round(len(shuffled) * test_fraction)))
            n_val = max(1, int(round(len(shuffled) * validation_fraction)))
            if n_test + n_val >= len(shuffled):
                raise ValueError("not enough airfoils for train/validation/test")
            test_ids = shuffled[:n_test]
            val_ids = shuffled[n_test : n_test + n_val]
            train_ids = shuffled[n_test + n_val :]

        ids = self.frame["airfoil_id"].astype(str)
        return {
            "train": np.flatnonzero(ids.isin(train_ids)),
            "validation": np.flatnonzero(ids.isin(val_ids)),
            "test": np.flatnonzero(ids.isin(test_ids)),
            "train_airfoils": train_ids,
            "validation_airfoils": val_ids,
            "test_airfoils": test_ids,
        }


def load_dataset(path: str | Path) -> AeroDataset:
    """Load and validate the canonical generated dataset."""
    frame = pd.read_csv(path)
    frame["airfoil_id"] = frame["airfoil_id"].astype(str)
    dataset = AeroDataset(frame)
    dataset.validate()
    return dataset


def save_split_manifest(path: str | Path, split: dict[str, np.ndarray], seed: int) -> None:
    payload = {"seed": seed}
    for key in ("train_airfoils", "validation_airfoils", "test_airfoils"):
        payload[key] = split[key].tolist()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
