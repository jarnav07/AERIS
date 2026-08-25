import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler

from airfoil_ml.features import FeaturePreprocessor
from airfoil_ml.large_dataset import CST_COLUMNS, inverse_fixed_targets, load_fixed_re_dataset, transform_fixed_targets


def test_fixed_re_adapter_normalizes_source_columns(tmp_path) -> None:
    rows = []
    for airfoil_id in ("A", "B", "C"):
        for alpha in (-2, 0, 2):
            row = {"Filename": airfoil_id, "AoA": alpha, "Cl": 0.1 * alpha, "Cd": 0.02}
            row.update({column: 0.01 * index for index, column in enumerate(CST_COLUMNS)})
            rows.append(row)
    path = tmp_path / "large.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    dataset = load_fixed_re_dataset(path)
    split = dataset.grouped_split(seed=42)
    assert dataset.frame.airfoil_id.tolist()[0] == "A"
    assert dataset.reynolds == 100000.0
    assert set.union(*(set(split[f"{part}_airfoils"]) for part in ("train", "validation", "test"))) == {"A", "B", "C"}


def test_log_cd_target_round_trip() -> None:
    targets = np.array([[0.4, 0.01], [0.8, 0.04]])
    transformed = transform_fixed_targets(targets)
    processor = FeaturePreprocessor(StandardScaler().fit(np.zeros((2, 1))), StandardScaler().fit(transformed), 0)
    scaled = processor.target_scaler.transform(transformed)
    assert np.allclose(inverse_fixed_targets(processor, scaled), targets)
