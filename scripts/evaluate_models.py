#!/usr/bin/env python3
"""Evaluate saved model predictions and generate research plots."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/generated/training_data.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--output", default="results/evaluation")
    args = parser.parse_args()

    # Keep evaluation orchestration in the package; this script is only an entry point.
    from airfoil_ml.training import load_model_bundle
    from airfoil_ml.evaluation import save_evaluation_plots

    model_dir = Path(args.model_dir)
    bundle = load_model_bundle(model_dir / args.model)
    save_evaluation_plots(bundle, args.csv, args.output)


if __name__ == "__main__":
    main()
