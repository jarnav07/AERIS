#!/usr/bin/env python3
"""Generate the canonical Kulfan/XFOIL training dataset.

This is a thin command-line wrapper around the package implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from airfoil_ml.training_data_generation import TrainingDataConfig, generate_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--xfoil", default="xfoil")
    parser.add_argument("--xfoil-timeout", type=int, default=30)
    parser.add_argument("--xfoil-iterations", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = TrainingDataConfig(
        xfoil_timeout=args.xfoil_timeout,
        xfoil_iterations=args.xfoil_iterations,
    )
    manifest = generate_training_dataset(
        Path(args.output_dir),
        n_cases=args.cases,
        seed=args.seed,
        workers=args.workers,
        xfoil_executable=args.xfoil,
        config=config,
        resume=args.resume,
    )
    print(f"Generated {manifest['successful_vectors']} rows across "
          f"{manifest['successful_airfoils']} airfoils.")


if __name__ == "__main__":
    main()
