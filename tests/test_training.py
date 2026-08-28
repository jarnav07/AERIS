import json

from airfoil_ml.data import AeroDataset
from airfoil_ml.models import ModelConfig
from airfoil_ml.training import load_model_bundle, train_all
from tests.test_data import make_frame


def test_train_all_uses_one_shared_split(tmp_path):
    dataset = AeroDataset(make_frame())
    dataset.validate()
    output = tmp_path / "models"
    results = train_all(
        dataset,
        output,
        seed=7,
        model_config=ModelConfig(seed=7, mlp_epochs=2, mlp_hidden_layers=(8,)),
        only=["ridge"],
    )
    assert "ridge" in results
    assert (output / "ridge.joblib").exists()
    assert (output / "preprocessor.joblib").exists()
    assert (output / "split_manifest.json").exists()
    assert json.loads((output / "training_config.json").read_text())["model_names"] == ["ridge"]
    model, processor = load_model_bundle(output, "ridge")
    assert processor is not None
    assert model is not None
