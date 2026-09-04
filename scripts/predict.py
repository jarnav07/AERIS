#!/usr/bin/env python3
"""Predict CL/CD/CM for a single aerofoil with the trained stacking ensemble.

Prefer: ``airfoil-ml predict``.

Examples::

    uv run python scripts/predict.py --model-dir models_dedicated \
        --airfoil naca2412 --alpha-range -5 15 1 --re 5e5 --plot results/predictions/naca2412.png
    uv run python scripts/predict.py --model-dir models_dedicated --random --seed 7 --alpha 0 5 10
"""
from __future__ import annotations

import argparse

from airfoil_ml.cli import add_predict_arguments, run_predict


def main() -> None:
    parser = add_predict_arguments(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    run_predict(parser.parse_args())


if __name__ == "__main__":
    main()
