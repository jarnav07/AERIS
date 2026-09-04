#!/usr/bin/env python3
"""Sample custom aerofoils with the dataset generator's own sampler and view them.

Prefer: ``airfoil-ml generate-airfoils``.

Unlike ``scripts/generate_dataset.py`` this only draws *shapes* -- it never calls
XFOIL -- so it runs anywhere and is the quick way to produce aerofoils to feed to
``scripts/predict.py``.
"""
from __future__ import annotations

import argparse

from airfoil_ml.cli import add_generate_airfoils_arguments, run_generate_airfoils


def main() -> None:
    parser = add_generate_airfoils_arguments(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    run_generate_airfoils(parser.parse_args())


if __name__ == "__main__":
    main()
