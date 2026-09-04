#!/usr/bin/env python3
"""Fit and evaluate a per-target learned-weight stacking ensemble over CL/CD/CM.

Fits one non-negative-weighted linear combination per target on the validation
split (not the equal-weight average scripts/evaluate_ensemble.py uses), saves the
weights to <model-dir>/stacking_weights.json, and scores on the held-out test split.
"""
from __future__ import annotations

import argparse
import json

from airfoil_ml.training import fit_stacking_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/generated/training_data.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--models", nargs="+", required=True, help="model names to combine, e.g. mlp mlp_torch hist_gb")
    parser.add_argument("--output", default="results/evaluation/stacking_ensemble")
    parser.add_argument("--max-polar-plots", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    metrics = fit_stacking_ensemble(
        csv_path=args.csv,
        model_dir=args.model_dir,
        model_names=args.models,
        output_dir=args.output,
        max_polar_plots=args.max_polar_plots,
        alpha=args.ridge_alpha,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
