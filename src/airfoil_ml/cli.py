"""Command-line interface for the reproducible research workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import ModelConfig
from .training_data_generation import TrainingDataConfig, generate_training_dataset
from .training import train_from_kulfan_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airfoil-ml")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate-training-data",
        help="generate stochastic Kulfan airfoils and analyse them with XFOIL",
    )
    generate.add_argument("--cases", type=int, default=30000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output-dir", default="data/generated")
    generate.add_argument("--xfoil", default="xfoil")
    generate.add_argument("--xfoil-timeout", type=int, default=30)
    generate.add_argument("--xfoil-iterations", type=int, default=200)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--resume", action="store_true")

    train = sub.add_parser("train", help="train grouped baseline and MLP models")
    train.add_argument("--csv", default="data/generated/training_data.csv")
    train.add_argument("--output-dir", default="models")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--mlp-hidden", default="128,64,32")
    train.add_argument("--mlp-max-iter", type=int, default=600)
    train.add_argument("--only", nargs="+", default=None)
    train.add_argument("--log-cd", action="store_true")

    evaluate = sub.add_parser("evaluate", help="create evaluation plots")
    evaluate.add_argument("--csv", default="data/generated/training_data.csv")
    evaluate.add_argument("--model-dir", default="models")
    evaluate.add_argument("--model", default="mlp")
    evaluate.add_argument("--output", default="results/evaluation")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "generate-training-data":
        config = TrainingDataConfig(
            xfoil_timeout=args.xfoil_timeout,
            xfoil_iterations=args.xfoil_iterations,
        )
        result = generate_training_dataset(
            args.output_dir,
            n_cases=args.cases,
            seed=args.seed,
            workers=args.workers,
            xfoil_executable=args.xfoil,
            config=config,
            resume=args.resume,
        )
    elif args.command == "train":
        config = ModelConfig(
            seed=args.seed,
            hidden_layers=tuple(int(x) for x in args.mlp_hidden.split(",")),
            max_iter=args.mlp_max_iter,
        )
        result = train_from_kulfan_csv(
            csv_path=args.csv,
            output_dir=args.output_dir,
            seed=args.seed,
            model_config=config,
            only=args.only,
            log_cd=args.log_cd,
        )
    else:
        raise NotImplementedError(
            "Evaluation orchestration remains in the existing research modules; "
            "use scripts/evaluate_models.py for now."
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
