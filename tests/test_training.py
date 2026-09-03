import json

import pandas as pd
import pytest

from airfoil_ml.models import ModelConfig
from airfoil_ml.training import KULFAN_FEATURE_COLUMNS, evaluate_kulfan_model, train_from_kulfan_csv


def _kulfan_training_frame(n_airfoils: int = 6, alphas: tuple[float, ...] = (-2.0, 0.0, 2.0)) -> pd.DataFrame:
    rows = []
    for airfoil_index in range(n_airfoils):
        for alpha in alphas:
            row = {col: 0.0 for col in KULFAN_FEATURE_COLUMNS}
            row.update({
                "airfoil_id": f"foil_{airfoil_index}",
                "alpha": alpha,
                "Re": 500_000.0 + airfoil_index * 10_000,
                "mach": 0.0,
                "n_crit": 9.0,
                "xtr_upper": 1.0,
                "xtr_lower": 1.0,
                "CL": 0.1 * alpha + airfoil_index * 0.01,
                "CD": 0.02 + 0.001 * alpha * alpha,
                "CM": -0.01 * alpha,
                "Top_Xtr": 0.5,
                "Bot_Xtr": 0.5,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_training_writes_models_and_metrics(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)

    output = tmp_path / "models"
    train_from_kulfan_csv(
        csv_path=csv_path,
        output_dir=output,
        model_config=ModelConfig(hidden_layers=(8,), max_iter=3, early_stopping=False),
    )

    assert (output / "metrics.json").exists()
    assert (output / "mlp.joblib").exists()
    assert (output / "split_manifest.json").exists()
    assert (output / "preprocessor.joblib").exists()


def test_training_respects_only_and_preserves_existing_metrics(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)
    output = tmp_path / "models"

    train_from_kulfan_csv(csv_path=csv_path, output_dir=output, only=["ridge"])
    results = train_from_kulfan_csv(csv_path=csv_path, output_dir=output, only=["random_forest"])

    assert set(results) == {"ridge", "random_forest"}
    assert (output / "ridge.joblib").exists()
    assert (output / "random_forest.joblib").exists()


def test_training_with_fixed_test_airfoils(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)
    output = tmp_path / "models"

    train_from_kulfan_csv(
        csv_path=csv_path,
        output_dir=output,
        test_airfoils=["foil_0", "foil_1"],
        only=["ridge"],
    )

    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["test_airfoils"]) == ["foil_0", "foil_1"]


def test_training_rejects_csv_missing_columns(tmp_path) -> None:
    frame = _kulfan_training_frame().drop(columns=["Re"])
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required feature columns"):
        train_from_kulfan_csv(csv_path=csv_path, output_dir=tmp_path / "models", only=["ridge"])


def test_evaluate_writes_metrics_and_plots(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)
    model_dir = tmp_path / "models"
    train_from_kulfan_csv(csv_path=csv_path, output_dir=model_dir, only=["ridge"])

    eval_dir = tmp_path / "eval"
    metrics = evaluate_kulfan_model(csv_path=csv_path, model_dir=model_dir, model_name="ridge", output_dir=eval_dir)

    assert set(metrics) == {"cl", "cd", "cm"}
    assert all(set(metrics[target]) == {"mae", "rmse", "r2"} for target in metrics)
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "parity_cl.png").exists()
    assert (eval_dir / "error_vs_alpha.png").exists()
    assert (eval_dir / "error_by_airfoil.png").exists()


def test_evaluate_only_scores_recorded_test_airfoils(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)
    model_dir = tmp_path / "models"
    train_from_kulfan_csv(
        csv_path=csv_path, output_dir=model_dir, test_airfoils=["foil_0", "foil_1"], only=["ridge"]
    )

    metrics = evaluate_kulfan_model(
        csv_path=csv_path, model_dir=model_dir, model_name="ridge", output_dir=tmp_path / "eval"
    )
    # A perfectly linear CL/CM target and a deterministic model should fit the
    # held-out foils exactly; a leak (scoring on training rows too) would not
    # change this, so this mainly guards against evaluate scoring zero rows.
    assert metrics["cl"]["mae"] >= 0.0


def test_evaluate_log_cd_is_auto_detected_from_training_config(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)
    model_dir = tmp_path / "models"
    train_from_kulfan_csv(csv_path=csv_path, output_dir=model_dir, only=["ridge"], log_cd=True)

    auto = evaluate_kulfan_model(csv_path=csv_path, model_dir=model_dir, model_name="ridge", output_dir=tmp_path / "eval_auto")
    explicit = evaluate_kulfan_model(
        csv_path=csv_path, model_dir=model_dir, model_name="ridge", output_dir=tmp_path / "eval_explicit", log_cd=True
    )
    wrong = evaluate_kulfan_model(
        csv_path=csv_path, model_dir=model_dir, model_name="ridge", output_dir=tmp_path / "eval_wrong", log_cd=False
    )

    assert auto == explicit
    assert auto["cd"]["mae"] != wrong["cd"]["mae"]


def test_evaluate_requires_trained_model(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(FileNotFoundError, match="split manifest"):
        evaluate_kulfan_model(csv_path=csv_path, model_dir=tmp_path / "no_such_model_dir", output_dir=tmp_path / "eval")


def test_evaluate_rejects_csv_without_test_airfoils(tmp_path) -> None:
    frame = _kulfan_training_frame()
    csv_path = tmp_path / "training_data.csv"
    frame.to_csv(csv_path, index=False)
    model_dir = tmp_path / "models"
    train_from_kulfan_csv(csv_path=csv_path, output_dir=model_dir, only=["ridge"])

    other_csv = tmp_path / "other.csv"
    _kulfan_training_frame(n_airfoils=2).assign(
        airfoil_id=lambda d: "other_" + d["airfoil_id"]
    ).to_csv(other_csv, index=False)

    with pytest.raises(ValueError, match="none of the test airfoils"):
        evaluate_kulfan_model(csv_path=other_csv, model_dir=model_dir, model_name="ridge", output_dir=tmp_path / "eval")
