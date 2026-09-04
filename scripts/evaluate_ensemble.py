#!/usr/bin/env python3
"""Evaluate a weighted-average ensemble of already-trained models on the held-out test split."""
from __future__ import annotations

import argparse
import json

from airfoil_ml.training import evaluate_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/generated/training_data.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--models", nargs="+", required=True, help="model names to average, e.g. mlp mlp_torch hist_gb")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="one weight per --models entry; defaults to equal weights")
    parser.add_argument("--output", default="results/evaluation/ensemble")
    parser.add_argument("--max-polar-plots", type=int, default=None)
    args = parser.parse_args()

    metrics = evaluate_ensemble(
        csv_path=args.csv,
        model_dir=args.model_dir,
        model_names=args.models,
        output_dir=args.output,
        weights=args.weights,
        max_polar_plots=args.max_polar_plots,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
