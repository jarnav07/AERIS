"""Command-line interface for the reproducible research workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .airfoil_sources import (
    describe_airfoil,
    open_in_viewer,
    plot_airfoils,
    resolve_airfoil,
    sample_random_airfoils,
    save_airfoil,
)
from .models import ModelConfig
from .predict import (
    DEFAULT_MACH,
    DEFAULT_N_CRIT,
    DEFAULT_XTR,
    StackingEnsemblePredictor,
    kulfan_feature_frame,
    out_of_distribution_warnings,
    plot_prediction_polars,
)
from .training_data_generation import TrainingDataConfig, generate_training_dataset
from .training import evaluate_kulfan_model, train_from_kulfan_csv


def add_predict_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments for the stacking-ensemble inference command.

    Shared with ``scripts/predict.py`` so the standalone script and the
    ``airfoil-ml predict`` subcommand can never drift apart.
    """
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--airfoil",
        help="aerofoil name ('naca2412', 'e63', ...), coordinate .dat/.txt file, "
        "or a Kulfan .json written by `airfoil-ml generate-airfoils`",
    )
    source.add_argument(
        "--random",
        action="store_true",
        help="sample one aerofoil from the dataset generator's own sampler instead",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for --random")
    parser.add_argument("--model-dir", default="models", help="directory holding stacking_weights.json and its component models")
    # argparse accepts a leading-minus value here because no option string on this
    # parser looks like a negative number, so `--alpha -5 0 5` works as written.
    parser.add_argument("--alpha", nargs="+", type=float, default=None, help="angles of attack in degrees (default: 0)")
    parser.add_argument(
        "--alpha-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="alpha sweep, inclusive of STOP, e.g. --alpha-range -5 15 1",
    )
    parser.add_argument("--re", type=float, default=5e5, dest="reynolds", help="chord Reynolds number (default: 5e5)")
    parser.add_argument("--mach", type=float, default=DEFAULT_MACH)
    parser.add_argument("--n-crit", type=float, default=DEFAULT_N_CRIT, help="XFOIL e^N transition criterion (default: 9)")
    parser.add_argument("--xtr-upper", type=float, default=DEFAULT_XTR, help="forced upper-surface transition x/c; 1.0 = free")
    parser.add_argument("--xtr-lower", type=float, default=DEFAULT_XTR, help="forced lower-surface transition x/c; 1.0 = free")
    parser.add_argument("--per-model", action="store_true", help="also report each component model's own prediction")
    parser.add_argument("--csv-out", default=None, help="write the prediction table to this CSV")
    parser.add_argument("--plot", default=None, help="write predicted CL/CD/CM/drag-polar curves to this PNG")
    parser.add_argument("--show", action="store_true", help="open the plot (and the aerofoil section) in the platform viewer")
    parser.add_argument("--json", action="store_true", help="print the prediction table as JSON instead of a text table")
    return parser


def _alpha_values(args: argparse.Namespace) -> np.ndarray:
    if args.alpha is not None and args.alpha_range is not None:
        raise SystemExit("pass either --alpha or --alpha-range, not both")
    if args.alpha_range is not None:
        start, stop, step = args.alpha_range
        if step <= 0:
            raise SystemExit("--alpha-range STEP must be positive")
        # +step/2 so an exactly-representable STOP is included despite float drift.
        return np.arange(start, stop + step / 2, step)
    return np.asarray(args.alpha if args.alpha is not None else [0.0], dtype=float)


def run_predict(args: argparse.Namespace) -> None:
    airfoil = sample_random_airfoils(1, seed=args.seed)[0] if args.random else resolve_airfoil(args.airfoil)
    alpha = _alpha_values(args)

    query = kulfan_feature_frame(
        airfoil, alpha, args.reynolds, args.mach, args.n_crit, args.xtr_upper, args.xtr_lower
    )
    warnings = out_of_distribution_warnings(query)

    predictor = StackingEnsemblePredictor.load(args.model_dir)
    table = predictor.predict(
        airfoil,
        alpha=alpha,
        Re=args.reynolds,
        mach=args.mach,
        n_crit=args.n_crit,
        xtr_upper=args.xtr_upper,
        xtr_lower=args.xtr_lower,
        per_model=args.per_model,
    )

    # In --json mode stdout has to stay parseable, so progress/warning chatter goes
    # to stderr and the JSON document is the only thing printed on stdout, last.
    def note(message: str) -> None:
        print(message, file=sys.stderr if args.json else sys.stdout)

    if not args.json:
        print(f"aerofoil: {airfoil.name}")
        print("geometry: " + "  ".join(f"{key}={value:.4f}" for key, value in describe_airfoil(airfoil).items()))
        print(f"ensemble: {' + '.join(predictor.model_names)}  (from {args.model_dir})")
        print(
            f"flow:     Re={args.reynolds:.3g}  mach={args.mach:g}  n_crit={args.n_crit:g}  "
            f"xtr_upper={args.xtr_upper:g}  xtr_lower={args.xtr_lower:g}"
        )
        print()
        columns = ["alpha", "CL", "CD", "CM", "L_over_D"]
        if args.per_model:
            columns += [c for c in table.columns if any(c.endswith(f"_{t}") for t in ("CL", "CD", "CM"))]
        print(table[columns].to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

    for warning in warnings:
        note(f"warning: {warning}")

    written: dict[str, str] = {}
    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv_out, index=False)
        written["csv"] = str(args.csv_out)
        note(f"wrote {args.csv_out}")

    plot_path = Path(args.plot) if args.plot else None
    if plot_path is None and args.show:
        # --show with no explicit --plot still needs something on disk to open.
        plot_path = Path("results/predictions") / f"{airfoil.name}_polar.png"
    if plot_path is not None:
        if len(alpha) < 2:
            note("note: skipping the polar plot; it needs at least 2 alphas (see --alpha-range)")
            plot_path = None
        else:
            plot_prediction_polars(table, plot_path, title=f"{airfoil.name} @ Re={args.reynolds:.3g} (predicted)")
            written["polar_plot"] = str(plot_path)
            note(f"wrote {plot_path}")

    if args.show:
        base = plot_path.parent if plot_path else Path("results/predictions")
        section_path = base / f"{airfoil.name}_section.png"
        plot_airfoils([airfoil], section_path)
        written["section_plot"] = str(section_path)
        note(f"wrote {section_path}")
        for path in filter(None, (plot_path, section_path)):
            if not open_in_viewer(path):
                note(f"note: could not open {path} in a viewer; open it manually")

    if args.json:
        print(json.dumps({
            "airfoil": str(airfoil.name),
            "geometry": describe_airfoil(airfoil),
            "model_dir": str(args.model_dir),
            "models": predictor.model_names,
            "warnings": warnings,
            "written": written,
            "predictions": table.to_dict(orient="records"),
        }, indent=2))


def add_generate_airfoils_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments for sampling and viewing custom aerofoils (no XFOIL required)."""
    parser.add_argument("--count", type=int, default=6, help="how many aerofoils to sample")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="data/generated/custom_airfoils", help="where the .json/.dat pairs and the PNG are written")
    parser.add_argument("--no-save", action="store_true", help="only plot; don't write .json/.dat files")
    parser.add_argument("--plot", default=None, help="PNG path for the section grid (default: <output-dir>/airfoils.png)")
    parser.add_argument("--columns", type=int, default=3, help="columns in the section grid")
    parser.add_argument("--show", action="store_true", help="open the section grid in the platform viewer")
    return parser


def run_generate_airfoils(args: argparse.Namespace) -> None:
    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    airfoils = sample_random_airfoils(args.count, seed=args.seed)
    output_dir = Path(args.output_dir)

    if not args.no_save:
        for airfoil in airfoils:
            paths = save_airfoil(airfoil, output_dir)
            print(f"{airfoil.name}: {paths['json']}  {paths['dat']}")

    plot_path = Path(args.plot) if args.plot else output_dir / "airfoils.png"
    plot_airfoils(airfoils, plot_path, columns=args.columns)
    print(f"wrote {plot_path}")

    if args.show and not open_in_viewer(plot_path):
        print(f"note: could not open {plot_path} in a viewer; open it manually")

    if not args.no_save:
        print(f"\npredict on one with:\n  airfoil-ml predict --airfoil {output_dir / (airfoils[0].name + '.json')} --alpha-range -5 15 1 --re 5e5")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airfoil-ml")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate-training-data",
        help="generate stochastic Kulfan airfoils and analyse them with XFOIL",
    )
    generate.add_argument("--cases", type=int, default=30000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output-dir", default="data/generated")
    generate.add_argument("--xfoil", default="xfoil")
    generate.add_argument("--xfoil-timeout", type=int, default=30)
    generate.add_argument("--xfoil-iterations", type=int, default=200)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--resume", action="store_true")

    train = sub.add_parser("train", help="train grouped baseline and MLP models")
    train.add_argument("--csv", default="data/generated/training_data.csv")
    train.add_argument("--output-dir", default="models")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--mlp-hidden", default=None, help="e.g. 256,128,64,32; defaults to ModelConfig's own default")
    train.add_argument("--mlp-max-iter", type=int, default=600)
    train.add_argument("--only", nargs="+", default=None)
    train.add_argument("--log-cd", action="store_true")
    train.add_argument("--target-cols", nargs="+", default=None, help="e.g. CL CD CM to train on only those targets instead of the full 197-column set")
    train.add_argument("--test-airfoils", nargs="+", default=None, help="fixed set of airfoil_ids to hold out as the test partition")

    evaluate = sub.add_parser("evaluate", help="create evaluation plots")
    evaluate.add_argument("--csv", default="data/generated/training_data.csv")
    evaluate.add_argument("--model-dir", default="models")
    evaluate.add_argument("--model", default="mlp")
    evaluate.add_argument("--output", default="results/evaluation")
    evaluate.add_argument(
        "--max-polar-plots",
        type=int,
        default=None,
        help="cap on per-airfoil polar PNGs/error-by-airfoil bars; 0 skips them, unset plots every test airfoil",
    )

    add_predict_arguments(sub.add_parser(
        "predict",
        help="predict CL/CD/CM for one aerofoil with the trained stacking ensemble",
    ))
    add_generate_airfoils_arguments(sub.add_parser(
        "generate-airfoils",
        help="sample and view custom aerofoils from the dataset generator (no XFOIL needed)",
    ))

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "generate-training-data":
        config = TrainingDataConfig(
            xfoil_timeout=args.xfoil_timeout,
            xfoil_iterations=args.xfoil_iterations,
        )
        result = generate_training_dataset(
            args.output_dir,
            n_cases=args.cases,
            seed=args.seed,
            workers=args.workers,
            xfoil_executable=args.xfoil,
            config=config,
            resume=args.resume,
        )
    elif args.command == "train":
        config_kwargs = {"seed": args.seed, "max_iter": args.mlp_max_iter}
        if args.mlp_hidden is not None:
            config_kwargs["hidden_layers"] = tuple(int(x) for x in args.mlp_hidden.split(","))
        config = ModelConfig(**config_kwargs)
        result = train_from_kulfan_csv(
            csv_path=args.csv,
            output_dir=args.output_dir,
            seed=args.seed,
            test_airfoils=args.test_airfoils,
            model_config=config,
            only=args.only,
            log_cd=args.log_cd,
            target_cols=args.target_cols,
        )
    elif args.command == "predict":
        return run_predict(args)
    elif args.command == "generate-airfoils":
        return run_generate_airfoils(args)
    else:
        assert args.command == "evaluate"
        result = evaluate_kulfan_model(
            csv_path=args.csv,
            model_dir=args.model_dir,
            model_name=args.model,
            output_dir=args.output,
            max_polar_plots=args.max_polar_plots,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
