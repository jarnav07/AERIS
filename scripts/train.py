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
    parser.add_argument("--mlp-hidden", default="128,64,32")
    parser.add_argument("--mlp-max-iter", type=int, default=600)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--log-cd", action="store_true")
    args = parser.parse_args()

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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
