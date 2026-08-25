import pandas as pd

from airfoil_ml.data import AeroDataset
from airfoil_ml.geometry import parse_airfoil
from airfoil_ml.models import ModelConfig
from airfoil_ml.training import train_all


def test_training_writes_models_and_metrics(tmp_path) -> None:
    geometry = parse_airfoil("1 0\n.75 .06\n.5 .1\n.25 .07\n0 0\n.25 -.04\n.5 -.1\n.75 -.05\n1 0\n", n_points=8)
    rows = []
    for airfoil_index in range(6):
        for alpha in (-2.0, 0.0, 2.0):
            rows.append({
                "airfoil_id": f"foil_{airfoil_index}",
                "alpha_deg": alpha,
                "reynolds": 500000.0 + airfoil_index * 10000,
                "mach": 0.0,
                "cl": 0.1 * alpha + airfoil_index * 0.01,
                "cd": 0.02 + 0.001 * alpha * alpha,
                "cm": -0.01 * alpha,
            })
    dataset = AeroDataset(pd.DataFrame(rows), {f"foil_{i}": geometry for i in range(6)})
    output = tmp_path / "models"
    train_all(dataset, output, model_config=ModelConfig(hidden_layers=(8,), max_iter=3, early_stopping=False))
    assert (output / "metrics.json").exists()
    assert (output / "mlp.joblib").exists()
    assert (output / "split_manifest.json").exists()
