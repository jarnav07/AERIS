"""Command-line interface for the canonical airfoil ML workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import KULFAN_COLUMNS, load_dataset
from .evaluation import evaluate_cfd_reference, evaluate_xfoil_test
from .features import build_feature_matrix
from .geometry import kulfan_from_file
from .models import ModelConfig
from .training import load_model_bundle, train_all
from .training_data_generation import TrainingDataConfig, generate_training_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airfoil-ml",
        description=(
            "Generate one XFOIL dataset, train several surrogate models, "
            "and compare their predictions with XFOIL and CFD."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate-data", help="generate the canonical stochastic Kulfan/XFOIL dataset"
    )
    generate.add_argument("--cases", type=int, default=1000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--training-output", default="data/raw/xfoil_training.csv")
    generate.add_argument("--coordinates-output", default="data/raw/generated_airfoils")
    generate.add_argument("--database-coordinates", default=None)
    generate.add_argument("--xfoil", default="xfoil")
    generate.add_argument("--xfoil-timeout", type=int, default=30)
    generate.add_argument("--xfoil-iterations", type=int, default=200)

    train = sub.add_parser("train", help="train the common set of surrogate models")
    train.add_argument("--dataset", default="data/raw/xfoil_training.csv")
    train.add_argument("--output", default="models/main")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--only", nargs="+", default=None)
    train.add_argument("--no-log-cd", action="store_true")
    train.add_argument("--mlp-epochs", type=int, default=250)
    train.add_argument("--mlp-hidden", default="128,64,32")

    evaluate = sub.add_parser(
        "evaluate", help="evaluate saved models against XFOIL and optionally CFD"
    )
    evaluate.add_argument("--dataset", default="data/raw/xfoil_training.csv")
    evaluate.add_argument("--model-dir", default="models/main")
    evaluate.add_argument("--output", default="results/main")
    evaluate.add_argument("--cfd-csv", default=None)
    evaluate.add_argument("--models", nargs="+", default=None)

    predict = sub.add_parser(
        "predict", help="predict Cl, Cd, and Cm for a coordinate file and operating point"
    )
    predict.add_argument("--model-dir", default="models/main")
    predict.add_argument("--model", default="mlp_torch")
    predict.add_argument("--coordinates", required=True)
    predict.add_argument("--alpha", type=float, required=True)
    predict.add_argument("--reynolds", type=float, required=True)
    predict.add_argument("--mach", type=float, default=0.0)

    return parser


def _predict_frame_from_coordinate(path: str | Path, alpha: float, reynolds: float, mach: float) -> pd.DataFrame:
    airfoil = kulfan_from_file(path)
    vector = np.array(
        [
            *airfoil.upper_weights,
            *airfoil.lower_weights,
            float(airfoil.leading_edge_weight),
            float(airfoil.TE_thickness),
        ],
        dtype=float,
    )
    return pd.DataFrame(
        [[*vector, alpha, reynolds, mach]],
        columns=[*KULFAN_COLUMNS, "alpha_deg", "reynolds", "mach"],
    )


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "generate-data":
        config = TrainingDataConfig(
            xfoil_timeout=args.xfoil_timeout,
            xfoil_iterations=args.xfoil_iterations,
        )
        result = generate_training_dataset(
            args.training_output,
            args.coordinates_output,
            n_cases=args.cases,
            seed=args.seed,
            database_coordinates_dir=args.database_coordinates,
            xfoil_executable=args.xfoil,
            config=config,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "train":
        dataset = load_dataset(args.dataset)
        config = ModelConfig(
            seed=args.seed,
            mlp_hidden_layers=tuple(int(v) for v in args.mlp_hidden.split(",")),
            mlp_epochs=args.mlp_epochs,
        )
        result = train_all(
            dataset,
            args.output,
            seed=args.seed,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            model_config=config,
            only=args.only,
            log_cd=not args.no_log_cd,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "evaluate":
        dataset = load_dataset(args.dataset)
        output = Path(args.output)
        result = {
            "xfoil": evaluate_xfoil_test(
                dataset, args.model_dir, output / "xfoil", model_names=args.models
            )
        }
        if args.cfd_csv:
            result["cfd"] = evaluate_cfd_reference(
                args.cfd_csv, args.model_dir, output / "cfd", model_names=args.models
            )
        print(json.dumps(result, indent=2))
        return

    if args.command == "predict":
        model, processor = load_model_bundle(args.model_dir, args.model)
        frame = _predict_frame_from_coordinate(
            args.coordinates, args.alpha, args.reynolds, args.mach
        )
        prediction = processor.inverse_targets(
            model.predict(processor.transform_inputs(build_feature_matrix(frame)))
        )
        config = json.loads(
            (Path(args.model_dir) / "training_config.json").read_text(encoding="utf-8")
        )
        if config.get("log_cd", True):
            prediction[:, 1] = np.exp(prediction[:, 1])
        print(
            json.dumps(
                dict(zip(("cl", "cd", "cm"), prediction[0].tolist())), indent=2
            )
        )
