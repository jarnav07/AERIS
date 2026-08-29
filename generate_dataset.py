#!/usr/bin/env python3
"""Backward-compatible entry point for dataset generation.

Prefer: ``python scripts/generate_dataset.py`` or
``airfoil-ml generate-training-data``.
"""
from scripts.generate_dataset import main


if __name__ == "__main__":
    main()
