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
    parser.add_argument(
        "--max-polar-plots",
        type=int,
        default=None,
        help="cap on per-airfoil polar PNGs (and the error-by-airfoil bar chart); "
        "0 skips them entirely, unset plots every test airfoil. At full generated-"
        "dataset scale (tens of thousands of test airfoils) the unbounded default "
        "can take hours per model, so pass 0 or a small sample size (e.g. 30).",
    )
    args = parser.parse_args()

    metrics = evaluate_kulfan_model(
        csv_path=args.csv,
        model_dir=args.model_dir,
        model_name=args.model,
        output_dir=args.output,
        max_polar_plots=args.max_polar_plots,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
