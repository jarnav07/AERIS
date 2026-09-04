#!/usr/bin/env python3
"""Backward-compatible wrapper for model training.

Prefer: ``airfoil-ml train``.
"""
from __future__ import annotations

import argparse
import json

from airfoil_ml.models import ModelConfig
from airfoil_ml.training import train_from_kulfan_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/generated/training_data.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    # Default is None, not a hardcoded architecture string: ModelConfig's own
    # hidden_layers default (tuned via scripts/tune_models.py) should apply unless
    # the caller explicitly overrides it here. A prior version defaulted this to
    # "128,64,32" unconditionally, which silently discarded every ModelConfig
    # architecture change made after that default was written -- models_final's
    # mlp/mlp_torch were trained on the untuned (128,64,32) architecture as a
    # result, despite (256,128,64,32) having won the hyperparameter search.
    parser.add_argument("--mlp-hidden", default=None, help="e.g. 256,128,64,32; defaults to ModelConfig's own default")
    parser.add_argument("--mlp-max-iter", type=int, default=600)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--log-cd", action="store_true")
    parser.add_argument("--target-cols", nargs="+", default=None, help="e.g. CL CD CM to train on only those targets instead of the full 197-column set")
    parser.add_argument("--test-airfoils", nargs="+", default=None, help="fixed set of airfoil_ids to hold out as the test partition")
    args = parser.parse_args()

    config_kwargs = {"seed": args.seed, "max_iter": args.mlp_max_iter}
    if args.mlp_hidden is not None:
        config_kwargs["hidden_layers"] = tuple(int(x) for x in args.mlp_hidden.split(","))
    config = ModelConfig(**config_kwargs)
    result = train_from_kulfan_csv(
        csv_path=args.csv,
        output_dir=args.output_dir,
        seed=args.seed,
        test_airfoils=args.test_airfoils,
        model_config=config,
        only=args.only,
        log_cd=args.log_cd,
        target_cols=args.target_cols,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
