import numpy as np
import pandas as pd
import pytest

from airfoil_ml.data import KULFAN_COLUMNS, AeroDataset, load_dataset


def make_frame() -> pd.DataFrame:
    rows = []
    for airfoil_index in range(6):
        for alpha in (-5.0, 0.0, 5.0):
            geometry = [0.1 + 0.001 * airfoil_index] * 16 + [1.0, 0.01]
            rows.append(
                [
                    *geometry,
                    alpha,
                    100_000 + airfoil_index * 10_000,
                    0.0,
                    0.5 + 0.01 * alpha,
                    0.01,
                    -0.02,
                    f"AF_{airfoil_index}",
                ]
            )
    return pd.DataFrame(
        rows,
        columns=[
            *KULFAN_COLUMNS,
            "alpha_deg",
            "reynolds",
            "mach",
            "cl",
            "cd",
            "cm",
            "airfoil_id",
        ],
    )


def test_grouped_split_has_no_geometry_leakage():
    dataset = AeroDataset(make_frame())
    dataset.validate()
    split = dataset.grouped_split(seed=123)
    groups = {key: set(split[f"{key}_airfoils"]) for key in ("train", "validation", "test")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert set(np.concatenate([split["train"], split["validation"], split["test"]])) == set(range(len(dataset.frame)))


def test_grouped_split_is_reproducible():
    dataset = AeroDataset(make_frame())
    a = dataset.grouped_split(seed=42)
    b = dataset.grouped_split(seed=42)
    assert a["test_airfoils"].tolist() == b["test_airfoils"].tolist()


def test_validation_rejects_non_positive_drag():
    frame = make_frame()
    frame.loc[0, "cd"] = 0.0
    with pytest.raises(ValueError, match="Cd"):
        AeroDataset(frame).validate()


def test_load_dataset(tmp_path):
    path = tmp_path / "dataset.csv"
    make_frame().to_csv(path, index=False)
    loaded = load_dataset(path)
    assert len(loaded.frame) == 18
    assert loaded.frame["airfoil_id"].dtype == object
