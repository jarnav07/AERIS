#!/usr/bin/env python3
"""Evaluate a trained canonical model on its held-out test split and generate research plots.

Prefer: ``airfoil-ml evaluate``.
"""
from __future__ import annotations

import argparse
import json

from airfoil_ml.training import evaluate_kulfan_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/generated/training_data.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--output", default="results/evaluation")
    args = parser.parse_args()

    metrics = evaluate_kulfan_model(
        csv_path=args.csv,
        model_dir=args.model_dir,
        model_name=args.model,
        output_dir=args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
