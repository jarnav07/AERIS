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
from .geometry import geometry_from_file
from .geometry_generation import DEFAULT_CAMBER, DEFAULT_CAMBER_POSITION, DEFAULT_THICKNESS, generate_camber_coordinates, sample_camber_grid
from .features import raw_input_matrix
from .large_dataset import CST_COLUMNS, FIXED_TARGET_COLUMNS, download_large_dataset, inverse_fixed_targets, load_fixed_re_dataset, train_fixed_re
from .large_evaluation import save_fixed_re_plots
from .multi_re_batch import generate_batch
from .models import ModelConfig
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
    acquire.add_argument("--xfoil-timeout", type=int, default=60, help="maximum seconds per airfoil/Re solver case")
    train = sub.add_parser("train", help="train grouped baseline and MLP models")
    train.add_argument("--polar-csv", default="data/raw/polars.csv")
    train.add_argument("--coordinates", default="data/raw/coordinates")
    train.add_argument("--output", default="models/initial")
    train.add_argument("--points", type=int, default=100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--mlp-hidden", default="128,64,32", help="comma-separated hidden layer sizes for the MLP")
    train.add_argument("--mlp-max-iter", type=int, default=600, help="maximum MLP epochs (early stopping usually stops sooner)")
    train.add_argument("--only", nargs="+", default=None, help="train only these models; existing metrics.json entries are preserved")
    train.add_argument("--log-cd", action="store_true", help="train Cd in log space so the fitted loss is relative drag error")
    train.add_argument("--test-airfoils-json", default=None, help="JSON file with a 'test_airfoils' list fixing the held-out airfoils (e.g. a prior split_manifest.json)")
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
    analyze_error.add_argument("--models", nargs="+", default=None, help="model names to include (default: all with saved predictions)")
    analyze_error.add_argument("--xfoil-benchmark", action="store_true", help="time real XFOIL polars on this machine (adds solver wall-clock)")
    analyze_error.add_argument("--benchmark-airfoils", nargs="+", default=None, help="airfoils to time with XFOIL (default: first two test airfoils)")
    analyze_error.add_argument("--benchmark-reynolds", nargs="+", type=float, default=[100000.0, 1000000.0])
    analyze = sub.add_parser("analyze-drag", help="drag-focused metrics and plots from saved test predictions")
    analyze.add_argument("--polar-csv", default="data/raw/polars.csv")
    analyze.add_argument("--coordinates", default="data/raw/coordinates")
    analyze.add_argument("--model-dir", default="models/initial")
    analyze.add_argument("--output", default="results/drag_analysis")
    analyze.add_argument("--points", type=int, default=100)
    analyze.add_argument("--models", nargs="+", default=None, help="model names to include (default: all with saved predictions)")
    large_download = sub.add_parser("download-large", help="download the large fixed-Re CST airfoil dataset")
    large_download.add_argument("--output", default="data/raw/external/kanakaero/compiled_airfoil_data.csv")
    train_large = sub.add_parser("train-large", help="train models on the large fixed-Re CST dataset")
    train_large.add_argument("--csv", default="data/raw/external/kanakaero/compiled_airfoil_data.csv")
    train_large.add_argument("--output", default="models/large_fixed_re")
    train_large.add_argument("--seed", type=int, default=42)
    train_large.add_argument("--no-log-cd", action="store_true", help="train on raw Cd instead of log(Cd)")
    evaluate_large = sub.add_parser("evaluate-large", help="plot the held-out airfoil test set for the fixed-Re dataset")
    evaluate_large.add_argument("--csv", default="data/raw/external/kanakaero/compiled_airfoil_data.csv")
    evaluate_large.add_argument("--model-dir", default="models/large_fixed_re")
    evaluate_large.add_argument("--model", default="mlp")
    evaluate_large.add_argument("--output", default="results/large_fixed_re")
    camber = sub.add_parser("generate-camber", help="generate parametric camber/thickness airfoils (optionally with XFOIL labels)")
    camber.add_argument("--output", default="data/raw/generated_coordinates", help="coordinate output directory")
    camber.add_argument("--n", type=int, default=None, help="number of designs to generate (default: the full camber grid)")
    camber.add_argument("--seed", type=int, default=42, help="seed for selecting a subset of the design grid")
    camber.add_argument("--points", type=int, default=100, help="geometry sampling points per surface")
    camber.add_argument("--camber-values", default=None, help="comma-separated max-camber fractions (default: 0,0.02,0.04,0.06)")
    camber.add_argument("--camber-positions", default=None, help="comma-separated max-camber positions (default: 0.2,0.4,0.6)")
    camber.add_argument("--thickness-values", default=None, help="comma-separated thickness fractions (default: 0.08,0.12,0.16)")
    camber.add_argument("--run-xfoil", action="store_true", help="run XFOIL polars over the generated coordinates")
    camber.add_argument("--polar-output", default="data/raw/generated_multi_re", help="XFOIL shard output directory")
    camber.add_argument("--reynolds", nargs="+", type=float, default=[30000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0])
    camber.add_argument("--alpha-start", type=float, default=-2.0, help="polar sweep start (degrees)")
    camber.add_argument("--alpha-end", type=float, default=10.0, help="polar sweep end (degrees)")
    camber.add_argument("--alpha-step", type=float, default=1.0, help="polar sweep step (degrees)")
    camber.add_argument("--workers", type=int, default=6, help="parallel XFOIL processes")
    batch = sub.add_parser("batch-generate", help="generate a sharded multi-Reynolds XFOIL dataset")
    batch.add_argument("--coordinates", default="data/raw/external/kanakaero/processed_coordinates")
    batch.add_argument("--output", default="data/raw/multi_re")
    batch.add_argument("--airfoils", nargs="+", default=None, help="coordinate stems to include")
    batch.add_argument("--limit", type=int, default=None, help="auto-select the first N coordinate files alphabetically")
    batch.add_argument("--reynolds", nargs="+", type=float, default=[30000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0])
    batch.add_argument("--alpha-start", type=float, default=-2.0)
    batch.add_argument("--alpha-end", type=float, default=10.0)
    batch.add_argument("--alpha-step", type=float, default=1.0)
    batch.add_argument("--mach", type=float, default=0.0)
    batch.add_argument("--xfoil-timeout", type=int, default=45)
    batch.add_argument("--workers", type=int, default=1, help="parallel XFOIL processes; each worker uses its own scratch directory")
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
            batch_result = generate_batch(
                args.output,
                args.polar_output,
                airfoil_ids,
                reynolds_values=args.reynolds,
                alpha_start=args.alpha_start,
                alpha_end=args.alpha_end,
                alpha_step=args.alpha_step,
                workers=args.workers,
            )
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
            holdout = json.loads(Path(args.test_airfoils_json).read_text())
            test_airfoils = holdout["test_airfoils"]
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
        frame = dataset.frame[dataset.frame.airfoil_id.isin(test_ids)].copy()
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for prediction_path in sorted(Path(args.model_dir).glob("test_predictions_*.npz")):
            name = prediction_path.stem.replace("test_predictions_", "")
            if args.models is not None and name not in args.models:
                continue
            saved = np.load(prediction_path)
            order = np.argsort(saved["indices"])
            predictions[name] = (saved["actual"][order], saved["predicted"][order])
        if not predictions:
            raise SystemExit("no saved test predictions found in the model directory")
        summary: dict[str, object] = {"n_test_rows": len(frame), "n_test_airfoils": len(test_ids)}
        summary["overall"] = {name: percentage_metrics(actual, predicted) for name, (actual, predicted) in predictions.items()}
        summary["by_regime"] = error_by_regime(frame, predictions)
        # ML wall-clock on the held-out test inputs (feature construction is
        # excluded so the measurement reflects model inference alone).
        model, processor = load_model_bundle(args.model_dir, next(iter(predictions)))
        x_raw = []
        for row in frame.itertuples():
            x_raw.append(raw_input_matrix([dataset.geometries[str(row.airfoil_id)]], np.array([row.alpha_deg]), np.array([row.reynolds]), np.array([row.mach]))[0])
        x_scaled = processor.input_scaler.transform(np.asarray(x_raw))
        model_names = sorted(predictions)
        ml_benchmark = benchmark_ml(args.model_dir, model_names, x_scaled)
        timing: dict[str, object] = {"ml_benchmark": ml_benchmark}
        if args.xfoil_benchmark:
            benchmark_airfoils = args.benchmark_airfoils or sorted(test_ids)[:2]
            xfoil_benchmark = benchmark_xfoil(args.coordinates, benchmark_airfoils, args.benchmark_reynolds)
            timing["xfoil_benchmark"] = xfoil_benchmark
            timing["projected_sweep"] = {name: time_saved_summary(xfoil_benchmark, ml_benchmark, name) for name in model_names}
        save_error_summary(frame, predictions, args.output, timing, summary)
        print(json.dumps({"overall": summary["overall"], "timing": timing}, indent=2))
    elif args.command == "analyze-drag":
        dataset = load_dataset(args.polar_csv, args.coordinates, args.points)
        manifest = json.loads((Path(args.model_dir) / "split_manifest.json").read_text())
        test_ids = set(manifest["test_airfoils"])
        frame = dataset.frame[dataset.frame.airfoil_id.isin(test_ids)].copy()
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for prediction_path in sorted(Path(args.model_dir).glob("test_predictions_*.npz")):
            name = prediction_path.stem.replace("test_predictions_", "")
            if args.models is not None and name not in args.models:
                continue
            saved = np.load(prediction_path)
            actual, predicted = saved["actual"], saved["predicted"]
            order = np.argsort(saved["indices"])
            predictions[name] = (actual[order], predicted[order])
        summary = compute_drag_summary(frame, predictions)
        save_drag_analysis_plots(frame, predictions, args.output, summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "predict":
        model, processor = load_model_bundle(args.model_dir, args.model)
        config_path = Path(args.model_dir) / "training_config.json"
        log_cd = json.loads(config_path.read_text()).get("log_cd", False) if config_path.exists() else False
        geometry = geometry_from_file(args.coordinates, args.points)
        raw = raw_input_matrix([geometry], np.array([args.alpha]), np.array([args.reynolds]), np.array([args.mach]))
        prediction = processor.inverse_targets(model.predict(processor.input_scaler.transform(raw)))[0]
        if log_cd:
            prediction[1] = float(np.exp(prediction[1]))
        print(json.dumps(dict(zip(("cl", "cd", "cm"), prediction.tolist())), indent=2))


if __name__ == "__main__":
    main()
