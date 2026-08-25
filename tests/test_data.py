import numpy as np
import pandas as pd

from airfoil_ml.data import AeroDataset
from airfoil_ml.geometry import parse_airfoil


def geometry():
    return parse_airfoil("1 0\n.75 .06\n.5 .1\n.25 .07\n0 0\n.25 -.04\n.5 -.1\n.75 -.05\n1 0\n", n_points=8)


def test_grouped_split_has_disjoint_airfoils() -> None:
    frame = pd.DataFrame({
        "airfoil_id": list("AAABBBCCCDDDEEE"),
        "alpha_deg": np.tile([-2, 0, 2], 5),
        "reynolds": 1e6,
        "mach": 0.0,
        "cl": 0.1,
        "cd": 0.02,
        "cm": 0.0,
    })
    dataset = AeroDataset(frame, {key: geometry() for key in "ABCDE"})
    split = dataset.grouped_split(seed=7)
    sets = [set(split[f"{name}_airfoils"]) for name in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set("ABCDE")
