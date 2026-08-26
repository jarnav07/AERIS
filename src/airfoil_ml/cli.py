"""Command-line interface for the reproducible research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .acquisition import download_uiuc_coordinates, generate_dataset
from .data import load_dataset
from .drag_analysis import compute_drag_summary, save_drag_analysis_plots
from .error_analysis import benchmark_ml, benchmark_xfoil, error_by_regime, percentage_metrics, save_error_summary, time_saved_summary
from .evaluation import save_evaluation_plots
from .features import raw_input_matrix
from .geometry import geometry_from_file
from .geometry_generation import DEFAULT_CAMBER, DEFAULT_CAMBER_POSITION, DEFAULT_THICKNESS, generate_camber_coordinates, sample_camber_grid
from .large_dataset import CST_COLUMNS, FIXED_TARGET_COLUMNS, download_large_dataset, inverse_fixed_targets, load_fixed_re_dataset, train_fixed_re
from .large_evaluation import save_fixed_re_plots
from .models import ModelConfig
from .multi_re_batch import generate_batch
from .neuralfoil_data_generation import NeuralFoilSamplingConfig, generate_neuralfoil_style_dataset
from .training import load_model_bundle, train_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airfoil-ml")
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire", help="download UIUC coordinates and generate XFOIL polars")
    acquire.add_argument("--airfoils", nargs="+", required=True)
    acquire.add_argument("--coordinates", default="data/raw/coordinates")
    acquire.add_argument("--polar-csv", default="data/raw/polars.csv")
    acquire.add_argument("--xfoil", default="xfoil")
    acquire.add_argument("--reynolds", nargs="+", type=float, default=[500000.0, 1000000.0])
    acquire.add_argument("--alpha-start", type=float, default=-10.0)
    acquire.add_argument("--alpha-end", type=float, default=20.0)
    acquire.add_argument("--alpha-step", type=float, default=1.0)
    acquire.add_argument("--xfoil-timeout", type=int, default=60)

    train = sub.add_parser("train", help="train grouped baseline and MLP models")
    train.add_argument("--polar-csv", default="data/raw/polars.csv")
    train.add_argument("--coordinates", default="data/raw/coordinates")
    train.add_argument("--output", default="models/initial")
    train.add_argument("--points", type=int, default=100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--mlp-hidden", default="128,64,32")
    train.add_argument("--mlp-max-iter", type=int, default=600)
    train.add_argument("--only", nargs="+", default=None)
    train.add_argument("--log-cd", action="store_true")
    train.add_argument("--test-airfoils-json", default=None)

    evaluate = sub.add_parser("evaluate", help="create plots from a saved test prediction")
    evaluate.add_argument("--polar-csv", default="data/raw/polars.csv")
    evaluate.add_argument("--coordinates", default="data/raw/coordinates")
    evaluate.add_argument("--model-dir", default="models/initial")
    evaluate.add_argument("--model", default="mlp")
    evaluate.add_argument("--output", default="results/initial")
    evaluate.add_argument("--points", type=int, default=100)

    analyze_error = sub.add_parser("analyze-error", help="percentage error vs XFOIL reference and wall-clock time saved")
    analyze_error.add_argument("--polar-csv", default="data/raw/polars.csv")
    analyze_error.add_argument("--coordinates", default="data/raw/coordinates")
    analyze_error.add_argument("--model-dir", default="models/initial")
    analyze_error.add_argument("--output", default="results/error_analysis")
    analyze_error.add_argument("--points", type=int, default=100)
    analyze_error.add_argument("--models", nargs="+", default=None)
    analyze_error.add_argument("--xfoil-benchmark", action="store_true")
    analyze_error.add_argument("--benchmark-airfoils", nargs="+", default=None)
    analyze_error.add_argument("--benchmark-reynolds", nargs="+", type=float, default=[100000.0, 1000000.0])

    analyze = sub.add_parser("analyze-drag", help="drag-focused metrics and plots from saved test predictions")
    analyze.add_argument("--polar-csv", default="data/raw/polars.csv")
    analyze.add_argument("--coordinates", default="data/raw/coordinates")
    analyze.add_argument("--model-dir", default="models/initial")
    analyze.add_argument("--output", default="results/drag_analysis")
    analyze.add_argument("--points", type=int, default=100)
    analyze.add_argument("--models", nargs="+", default=None)

    large_download = sub.add_parser("download-large", help="download the large fixed-Re CST airfoil dataset")
    large_download.add_argument("--output", default="data/raw/external/kanakaero/compiled_airfoil_data.csv")

    train_large = sub.add_parser("train-large", help="train models on the large fixed-Re CST dataset")
    train_large.add_argument("--csv", default="data/raw/external/kanakaero/compiled_airfoil_data.csv")
    train_large.add_argument("--output", default="models/large_fixed_re")
    train_large.add_argument("--seed", type=int, default=42)
    train_large.add_argument("--no-log-cd", action="store_true")

    evaluate_large = sub.add_parser("evaluate-large", help="plot the held-out airfoil test set for the fixed-Re dataset")
    evaluate_large.add_argument("--csv", default="data/raw/external/kanakaero/compiled_airfoil_data.csv")
    evaluate_large.add_argument("--model-dir", default="models/large_fixed_re")
    evaluate_large.add_argument("--model", default="mlp")
    evaluate_large.add_argument("--output", default="results/large_fixed_re")

    camber = sub.add_parser("generate-camber", help="generate parametric camber/thickness airfoils (optionally with XFOIL labels)")
    camber.add_argument("--output", default="data/raw/generated_coordinates")
    camber.add_argument("--n", type=int, default=None)
    camber.add_argument("--seed", type=int, default=42)
    camber.add_argument("--points", type=int, default=100)
    camber.add_argument("--camber-values", default=None)
    camber.add_argument("--camber-positions", default=None)
    camber.add_argument("--thickness-values", default=None)
    camber.add_argument("--run-xfoil", action="store_true")
    camber.add_argument("--polar-output", default="data/raw/generated_multi_re")
    camber.add_argument("--reynolds", nargs="+", type=float, default=[30000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0])
    camber.add_argument("--alpha-start", type=float, default=-2.0)
    camber.add_argument("--alpha-end", type=float, default=10.0)
    camber.add_argument("--alpha-step", type=float, default=1.0)
    camber.add_argument("--workers", type=int, default=6)

    batch = sub.add_parser("batch-generate", help="generate a sharded multi-Reynolds XFOIL dataset")
    batch.add_argument("--coordinates", default="data/raw/external/kanakaero/processed_coordinates")
    batch.add_argument("--output", default="data/raw/multi_re")
    batch.add_argument("--airfoils", nargs="+", default=None)
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--reynolds", nargs="+", type=float, default=[30000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0])
    batch.add_argument("--alpha-start", type=float, default=-2.0)
    batch.add_argument("--alpha-end", type=float, default=10.0)
    batch.add_argument("--alpha-step", type=float, default=1.0)
    batch.add_argument("--mach", type=float, default=0.0)
    batch.add_argument("--xfoil-timeout", type=int, default=45)
    batch.add_argument("--workers", type=int, default=1)

    neuralfoil = sub.add_parser("generate-neuralfoil", help="generate stochastic airfoil/XFOIL data using NeuralFoil's training distribution")
    neuralfoil.add_argument("--cases", type=int, default=100)
    neuralfoil.add_argument("--seed", type=int, default=42)
    neuralfoil.add_argument("--coordinates-output", default="data/raw/neuralfoil_coordinates")
    neuralfoil.add_argument("--polar-output", default="data/raw/neuralfoil_style.csv")
    neuralfoil.add_argument("--database-coordinates", default=None)
    neuralfoil.add_argument("--xfoil", default="xfoil")
    neuralfoil.add_argument("--xfoil-timeout", type=int, default=30)
    neuralfoil.add_argument("--xfoil-iterations", type=int, default=200)

    predict = sub.add_parser("predict", help="predict one operating point")
    predict.add_argument("--model-dir", default="models/initial")
    predict.add_argument("--model", default="mlp")
    predict.add_argument("--coordinates", required=True)
    predict.add_argument("--alpha", type=float, required=True)
    predict.add_argument("--reynolds", type=float, required=True)
    predict.add_argument("--mach", type=float, default=0.0)
    predict.add_argument("--points", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "acquire":
        download_uiuc_coordinates(args.airfoils, args.coordinates)
        generate_dataset(args.coordinates, args.polar_csv, args.reynolds, airfoil_ids=args.airfoils, xfoil_executable=args.xfoil, alpha_start=args.alpha_start, alpha_end=args.alpha_end, alpha_step=args.alpha_step, timeout_seconds=args.xfoil_timeout)
    elif args.command == "generate-neuralfoil":
        config = NeuralFoilSamplingConfig(xfoil_timeout=args.xfoil_timeout, xfoil_iterations=args.xfoil_iterations)
        print(json.dumps(generate_neuralfoil_style_dataset(output_csv=args.polar_output, coordinate_output_dir=args.coordinates_output, n_cases=args.cases, seed=args.seed, database_coordinates_dir=args.database_coordinates, xfoil_executable=args.xfoil, config=config), indent=2))
    elif args.command == "download-large":
        path = download_large_dataset(args.output)
        print(json.dumps({"path": str(path)}))
    elif args.command == "generate-camber":
        camber_values = tuple(float(v) for v in args.camber_values.split(",")) if args.camber_values else None
        camber_positions = tuple(float(v) for v in args.camber_positions.split(",")) if args.camber_positions else None
        thickness_values = tuple(float(v) for v in args.thickness_values.split(",")) if args.thickness_values else None
        records = sample_camber_grid(camber_values=camber_values or DEFAULT_CAMBER, positions=camber_positions or DEFAULT_CAMBER_POSITION, thickness_values=thickness_values or DEFAULT_THICKNESS)
        if args.n is not None:
            rng = np.random.default_rng(args.seed)
            records = [records[i] for i in rng.permutation(len(records))[: args.n]]
        generated = generate_camber_coordinates(args.output, records, n_points=args.points)
        result: dict[str, object] = {"coordinates_dir": args.output, "airfoils": len(generated["airfoils"])}
        if args.run_xfoil:
            airfoil_ids = [str(record["airfoil_id"]) for record in generated["airfoils"]]
            batch_result = generate_batch(args.output, args.polar_output, airfoil_ids, reynolds_values=args.reynolds, alpha_start=args.alpha_start, alpha_end=args.alpha_end, alpha_step=args.alpha_step, workers=args.workers)
            result["xfoil"] = batch_result
            result["combined_csv"] = str(Path(args.polar_output) / "combined.csv")
        print(json.dumps(result, indent=2))
    elif args.command == "batch-generate":
        if args.airfoils is None:
            coordinates = sorted(Path(args.coordinates).glob("*.dat"))
            selected = [path.stem for path in coordinates[: args.limit]] if args.limit else [path.stem for path in coordinates]
        else:
            selected = args.airfoils
        print(json.dumps(generate_batch(args.coordinates, args.output, selected, reynolds_values=args.reynolds, alpha_start=args.alpha_start, alpha_end=args.alpha_end, alpha_step=args.alpha_step, mach=args.mach, timeout_seconds=args.xfoil_timeout, workers=args.workers), indent=2))
    elif args.command == "train-large":
        dataset = load_fixed_re_dataset(args.csv)
        metrics = train_fixed_re(dataset, args.output, seed=args.seed, log_cd=not args.no_log_cd)
        print(json.dumps(metrics, indent=2))
    elif args.command == "evaluate-large":
        dataset = load_fixed_re_dataset(args.csv)
        model, processor = load_model_bundle(args.model_dir, args.model)
        manifest = json.loads((Path(args.model_dir) / "split_manifest.json").read_text())
        test_ids = set(manifest["test_airfoils"])
        frame = dataset.frame[dataset.frame.airfoil_id.isin(test_ids)].copy()
        x = frame[[*CST_COLUMNS, "alpha_deg"]].to_numpy(float)
        y = frame[list(FIXED_TARGET_COLUMNS)].to_numpy(float)
        metadata = json.loads((Path(args.model_dir) / "dataset_metadata.json").read_text())
        predictions = inverse_fixed_targets(processor, model.predict(processor.input_scaler.transform(x)), log_cd=metadata.get("log_cd", False))
        save_fixed_re_plots(frame, y, predictions, args.output)
        print(json.dumps({"model": args.model, "rows": len(frame), "airfoils": int(frame.airfoil_id.nunique())}))
    elif args.command == "train":
        dataset = load_dataset(args.polar_csv, args.coordinates, args.points)
        hidden = tuple(int(size) for size in args.mlp_hidden.split(","))
        config = ModelConfig(seed=args.seed, hidden_layers=hidden, max_iter=args.mlp_max_iter)
        test_airfoils = None
        if args.test_airfoils_json:
            test_airfoils = json.loads(Path(args.test_airfoils_json).read_text())["test_airfoils"]
        metrics = train_all(dataset, args.output, seed=args.seed, model_config=config, only=args.only, log_cd=args.log_cd, test_airfoils=test_airfoils)
        print(json.dumps(metrics, indent=2))
    elif args.command == "evaluate":
        dataset = load_dataset(args.polar_csv, args.coordinates, args.points)
        model, processor = load_model_bundle(args.model_dir, args.model)
        manifest = json.loads((Path(args.model_dir) / "split_manifest.json").read_text())
        test_ids = set(manifest["test_airfoils"])
        frame = dataset.frame[dataset.frame.airfoil_id.isin(test_ids)].copy()
        indices = frame.index.to_numpy()
        x, y = [], frame[["cl", "cd", "cm"]].to_numpy(float)
        for row in frame.itertuples():
            x.append(raw_input_matrix([dataset.geometries[str(row.airfoil_id)]], np.array([row.alpha_deg]), np.array([row.reynolds]), np.array([row.mach]))[0])
        predictions = processor.inverse_targets(model.predict(processor.input_scaler.transform(np.asarray(x))))
        save_evaluation_plots(frame, y, predictions, args.output)
        print(json.dumps({"model": args.model, "rows": len(frame), "indices": indices.tolist()}))
    elif args.command == "analyze-error":
        dataset = load_dataset(args.polar_csv, args.coordinates, args.points)
        manifest = json.loads((Path(args.model_dir) / "split_manifest.json").read_text())
        test_ids = set(manifest["test_airfoils"])
        test_frame = dataset.frame[dataset.frame.airfoil_id.isin(test_ids)].copy()
        model_names = args.models or [name for name in ["ridge", "random_forest", "hist_gb", "mlp", "mlp_torch"] if (Path(args.model_dir) / (f"{name}.joblib" if name != "mlp_torch" else "mlp_torch.pt")).exists()]
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in model_names:
            model, processor = load_model_bundle(args.model_dir, name)
            x = np.vstack([raw_input_matrix([dataset.geometries[str(row.airfoil_id)]], np.array([row.alpha_deg]), np.array([row.reynolds]), np.array([row.mach]))[0] for row in test_frame.itertuples()])
            actual = test_frame[["cl", "cd", "cm"]].to_numpy(float)
            predicted = processor.inverse_targets(model.predict(processor.input_scaler.transform(x)))
            predictions[name] = (actual, predicted)
        summary = {"overall": {name: percentage_metrics(actual, predicted) for name, (actual, predicted) in predictions.items()}, "regimes": error_by_regime(test_frame, predictions)}
        timing: dict[str, object] = {}
        if args.xfoil_benchmark:
            benchmark_airfoils = args.benchmark_airfoils or sorted(test_ids)[:2]
            timing["xfoil_benchmark"] = benchmark_xfoil(args.coordinates, benchmark_airfoils, args.benchmark_reynolds)
        timing["ml_benchmark"] = benchmark_ml(args.model_dir, model_names, np.vstack([raw_input_matrix([dataset.geometries[str(row.airfoil_id)]], np.array([row.alpha_deg]), np.array([row.reynolds]), np.array([row.mach]))[0] for row in test_frame.itertuples()]),)
        if "xfoil_benchmark" in timing and model_names:
            timing["time_saved"] = time_saved_summary(timing["xfoil_benchmark"], timing["ml_benchmark"], model_names[0])
        save_error_summary(test_frame, predictions, args.output, timing, summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "analyze-drag":
        dataset = load_dataset(args.polar_csv, args.coordinates, args.points)
        manifest = json.loads((Path(args.model_dir) / "split_manifest.json").read_text())
        test_ids = set(manifest["test_airfoils"])
        test_frame = dataset.frame[dataset.frame.airfoil_id.isin(test_ids)].copy()
        model_names = args.models or [name for name in ["ridge", "random_forest", "hist_gb", "mlp", "mlp_torch"] if (Path(args.model_dir) / (f"{name}.joblib" if name != "mlp_torch" else "mlp_torch.pt")).exists()]
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in model_names:
            model, processor = load_model_bundle(args.model_dir, name)
            x = np.vstack([raw_input_matrix([dataset.geometries[str(row.airfoil_id)]], np.array([row.alpha_deg]), np.array([row.reynolds]), np.array([row.mach]))[0] for row in test_frame.itertuples()])
            actual = test_frame[["cl", "cd", "cm"]].to_numpy(float)
            predicted = processor.inverse_targets(model.predict(processor.input_scaler.transform(x)))
            predictions[name] = (actual, predicted)
        summary = {name: compute_drag_summary(test_frame, actual, predicted) for name, (actual, predicted) in predictions.items()}
        save_drag_analysis_plots(test_frame, predictions, args.output, summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "predict":
        model, processor = load_model_bundle(args.model_dir, args.model)
        geometry = geometry_from_file(args.coordinates, n_points=args.points)
        x = raw_input_matrix([geometry], np.array([args.alpha]), np.array([args.reynolds]), np.array([args.mach]))
        prediction = processor.inverse_targets(model.predict(processor.input_scaler.transform(x)))[0]
        print(json.dumps(dict(zip(("cl", "cd", "cm"), prediction.tolist())), indent=2))
