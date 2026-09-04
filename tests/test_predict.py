import json

import numpy as np
import pandas as pd
import pytest

from airfoil_ml.airfoil_sources import kulfan_from_dict, kulfan_to_dict, resolve_airfoil, sample_random_airfoils, save_airfoil
from airfoil_ml.predict import (
    StackingEnsemblePredictor,
    kulfan_feature_frame,
    out_of_distribution_warnings,
    plot_prediction_polars,
)
from airfoil_ml.training import (
    KULFAN_FEATURE_COLUMNS,
    KULFAN_VECTOR_COLUMNS,
    apply_stacking_weights,
    fit_stacking_ensemble,
    train_from_kulfan_csv,
)

# NACA sections give the geometry features a non-degenerate signal, unlike the
# all-zero Kulfan vectors used by tests/test_training.py.
_TEST_AIRFOIL_NAMES = ("naca0009", "naca0012", "naca1408", "naca2412", "naca4412", "naca4415")


def _kulfan_frame_from_named_airfoils() -> pd.DataFrame:
    rows = []
    for name in _TEST_AIRFOIL_NAMES:
        airfoil = resolve_airfoil(name)
        vector = np.concatenate([
            airfoil.upper_weights, airfoil.lower_weights,
            [airfoil.leading_edge_weight, airfoil.TE_thickness],
        ])
        camber, thickness = float(airfoil.max_camber()), float(airfoil.max_thickness())
        for alpha in np.arange(-6.0, 11.0, 2.0):
            for reynolds in (2e5, 5e5, 1e6):
                row = dict(zip(KULFAN_VECTOR_COLUMNS, vector))
                row.update({
                    "airfoil_id": name,
                    "alpha": alpha, "Re": reynolds, "mach": 0.0,
                    "n_crit": 9.0, "xtr_upper": 1.0, "xtr_lower": 1.0,
                    "CL": 0.11 * alpha + 10 * camber,
                    "CD": 0.006 + 2e-4 * alpha ** 2 + 0.02 * thickness,
                    "CM": -2.5 * camber - 0.002 * alpha,
                    "Top_Xtr": 0.5, "Bot_Xtr": 0.5,
                })
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def stacked_model_dir(tmp_path_factory) -> "object":
    """A trained two-model stacking ensemble on a small synthetic CSV."""
    directory = tmp_path_factory.mktemp("stacked")
    csv_path = directory / "training_data.csv"
    _kulfan_frame_from_named_airfoils().to_csv(csv_path, index=False)

    model_dir = directory / "models"
    train_from_kulfan_csv(
        csv_path=csv_path,
        output_dir=model_dir,
        only=["ridge", "hist_gb"],
        log_cd=True,
        target_cols=["CL", "CD", "CM"],
    )
    fit_stacking_ensemble(
        csv_path=csv_path,
        model_dir=model_dir,
        model_names=["ridge", "hist_gb"],
        output_dir=directory / "eval",
        max_polar_plots=0,
    )
    return model_dir


def test_fit_stacking_ensemble_saves_reusable_weights(stacked_model_dir) -> None:
    payload = json.loads((stacked_model_dir / "stacking_weights.json").read_text(encoding="utf-8"))

    assert payload["model_names"] == ["ridge", "hist_gb"]
    assert set(payload["weights"]) == {"cl", "cd", "cm"}
    for target_weights in payload["weights"].values():
        assert set(target_weights) == {"ridge", "hist_gb", "intercept"}
        # Ridge(positive=True) must never hand back a negative model weight.
        assert all(target_weights[name] >= 0.0 for name in ("ridge", "hist_gb"))


def test_apply_stacking_weights_matches_manual_combination() -> None:
    predictions = {
        "a": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "b": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
    }
    weights = {
        "cl": {"a": 0.5, "b": 0.5, "intercept": 1.0},
        "cd": {"a": 1.0, "b": 0.0, "intercept": 0.0},
        "cm": {"a": 0.0, "b": 2.0, "intercept": -1.0},
    }

    combined = apply_stacking_weights(predictions, ["a", "b"], weights)

    assert combined == pytest.approx(np.array([[6.5, 2.0, 59.0], [23.0, 5.0, 119.0]]))


def test_predictor_reproduces_the_component_combination(stacked_model_dir) -> None:
    predictor = StackingEnsemblePredictor.load(stacked_model_dir)
    table = predictor.predict("naca2412", alpha=[0.0, 5.0], Re=5e5, per_model=True)

    weights = predictor.weights
    for row_index in range(len(table)):
        expected_cl = weights["cl"]["intercept"] + sum(
            weights["cl"][name] * table.loc[row_index, f"{name}_CL"] for name in predictor.model_names
        )
        assert table.loc[row_index, "CL"] == pytest.approx(expected_cl)


def test_predict_returns_one_row_per_alpha_with_derived_l_over_d(stacked_model_dir) -> None:
    predictor = StackingEnsemblePredictor.load(stacked_model_dir)
    alpha = np.arange(-5.0, 10.0, 2.5)

    table = predictor.predict("naca4412", alpha=alpha, Re=4e5)

    assert list(table["alpha"]) == pytest.approx(list(alpha))
    assert set(("CL", "CD", "CM", "L_over_D")).issubset(table.columns)
    assert table["L_over_D"].to_numpy() == pytest.approx((table["CL"] / table["CD"]).to_numpy())
    assert (table["Re"] == 4e5).all()


def test_predict_accepts_a_kulfan_airfoil_object_and_a_json_file(stacked_model_dir, tmp_path) -> None:
    predictor = StackingEnsemblePredictor.load(stacked_model_dir)
    airfoil = resolve_airfoil("naca2412")
    paths = save_airfoil(airfoil, tmp_path, stem="saved")

    from_object = predictor.predict(airfoil, alpha=3.0, Re=5e5)
    from_json = predictor.predict(str(paths["json"]), alpha=3.0, Re=5e5)

    assert from_object["CL"].to_numpy() == pytest.approx(from_json["CL"].to_numpy())


def test_load_reports_a_missing_ensemble_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no stacking ensemble found"):
        StackingEnsemblePredictor.load(tmp_path)


def test_feature_frame_matches_the_training_feature_columns() -> None:
    frame = kulfan_feature_frame(resolve_airfoil("naca2412"), alpha=[0.0, 4.0], Re=5e5)

    assert len(frame) == 2
    assert not [c for c in KULFAN_FEATURE_COLUMNS if c not in frame.columns]
    # Geometry must be non-degenerate for a real section; add_geometry_features
    # falls back to all-zeros only for shapes AeroSandbox can't measure.
    assert frame["geom_max_thickness"].iloc[0] > 0.0


def test_feature_frame_broadcasts_flow_arguments() -> None:
    frame = kulfan_feature_frame(resolve_airfoil("naca0012"), alpha=[0.0, 2.0, 4.0], Re=[1e5, 2e5, 3e5])

    assert list(frame["Re"]) == [1e5, 2e5, 3e5]
    assert (frame["mach"] == 0.0).all()


def test_feature_frame_rejects_non_positive_reynolds() -> None:
    with pytest.raises(ValueError, match="Reynolds number must be positive"):
        kulfan_feature_frame(resolve_airfoil("naca0012"), alpha=0.0, Re=0.0)


def test_out_of_distribution_warnings_flag_extrapolation() -> None:
    airfoil = resolve_airfoil("naca0012")

    assert out_of_distribution_warnings(kulfan_feature_frame(airfoil, alpha=0.0, Re=5e5)) == []
    assert any("Re" in w for w in out_of_distribution_warnings(kulfan_feature_frame(airfoil, alpha=0.0, Re=1e9)))
    assert any("alpha" in w for w in out_of_distribution_warnings(kulfan_feature_frame(airfoil, alpha=40.0, Re=5e5)))
    assert any("mach" in w for w in out_of_distribution_warnings(kulfan_feature_frame(airfoil, alpha=0.0, Re=5e5, mach=0.5)))


def test_plot_prediction_polars_writes_a_png(stacked_model_dir, tmp_path) -> None:
    predictor = StackingEnsemblePredictor.load(stacked_model_dir)
    table = predictor.predict("naca2412", alpha=np.arange(-4.0, 9.0, 2.0), Re=5e5)

    path = plot_prediction_polars(table, tmp_path / "nested" / "polar.png", title="test")

    assert path.exists() and path.stat().st_size > 0


def test_resolve_airfoil_handles_names_files_and_json(tmp_path) -> None:
    by_name = resolve_airfoil("naca2412")
    paths = save_airfoil(by_name, tmp_path, stem="foil")

    # The Kulfan JSON round-trips the 18 coefficients exactly; the .dat is a
    # coordinate discretisation, so it only has to come back close.
    assert np.allclose(resolve_airfoil(str(paths["json"])).upper_weights, by_name.upper_weights)
    assert np.allclose(resolve_airfoil(str(paths["dat"])).upper_weights, by_name.upper_weights, atol=5e-3)


def test_resolve_airfoil_rejects_unknown_names_and_suffixes(tmp_path) -> None:
    with pytest.raises(ValueError, match="could not resolve aerofoil"):
        resolve_airfoil("definitely_not_an_airfoil_xyz")

    unsupported = tmp_path / "foil.step"
    unsupported.write_text("not coordinates", encoding="utf-8")
    with pytest.raises(ValueError, match="don't know how to read"):
        resolve_airfoil(str(unsupported))


def test_kulfan_dict_round_trip() -> None:
    original = resolve_airfoil("naca4412")
    restored = kulfan_from_dict(json.loads(json.dumps(kulfan_to_dict(original))))

    assert np.allclose(restored.lower_weights, original.lower_weights)
    assert restored.TE_thickness == pytest.approx(float(original.TE_thickness))


def test_sample_random_airfoils_is_seed_reproducible() -> None:
    first = sample_random_airfoils(2, seed=11)
    second = sample_random_airfoils(2, seed=11)

    assert [a.name for a in first] == ["sampled_0000", "sampled_0001"]
    assert np.allclose(first[0].upper_weights, second[0].upper_weights)
    assert not np.allclose(first[0].upper_weights, first[1].upper_weights)


def test_cli_alpha_arguments_build_the_expected_sweeps() -> None:
    from airfoil_ml.cli import _alpha_values, add_predict_arguments
    import argparse

    parser = add_predict_arguments(argparse.ArgumentParser())

    single = _alpha_values(parser.parse_args(["--airfoil", "naca0012"]))
    listed = _alpha_values(parser.parse_args(["--airfoil", "naca0012", "--alpha", "-5", "0", "5"]))
    swept = _alpha_values(parser.parse_args(["--airfoil", "naca0012", "--alpha-range", "-5", "5", "2.5"]))

    assert list(single) == [0.0]
    assert list(listed) == [-5.0, 0.0, 5.0]
    # STOP is inclusive, so a 10-degree span at 2.5-degree steps gives 5 points.
    assert list(swept) == pytest.approx([-5.0, -2.5, 0.0, 2.5, 5.0])


def test_cli_predict_writes_csv_and_plot(stacked_model_dir, tmp_path, capsys) -> None:
    import argparse

    from airfoil_ml.cli import add_predict_arguments, run_predict

    parser = add_predict_arguments(argparse.ArgumentParser())
    run_predict(parser.parse_args([
        "--model-dir", str(stacked_model_dir),
        "--airfoil", "naca2412",
        "--alpha-range", "-4", "8", "2",
        "--re", "5e5",
        "--csv-out", str(tmp_path / "pred.csv"),
        "--plot", str(tmp_path / "polar.png"),
        "--json",
    ]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["airfoil"] == "naca2412"
    assert len(payload["predictions"]) == 7
    assert payload["warnings"] == []
    assert (tmp_path / "pred.csv").exists()
    assert (tmp_path / "polar.png").exists()


def test_cli_generate_airfoils_saves_and_plots(tmp_path) -> None:
    import argparse

    from airfoil_ml.cli import add_generate_airfoils_arguments, run_generate_airfoils

    parser = add_generate_airfoils_arguments(argparse.ArgumentParser())
    run_generate_airfoils(parser.parse_args([
        "--count", "2", "--seed", "5", "--output-dir", str(tmp_path), "--columns", "2",
    ]))

    assert (tmp_path / "sampled_0000.json").exists()
    assert (tmp_path / "sampled_0000.dat").exists()
    assert (tmp_path / "airfoils.png").exists()
