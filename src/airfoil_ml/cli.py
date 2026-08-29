"""Command-line interface for the reproducible research workflow."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

from .drag_analysis import compute_drag_summary, save_drag_analysis_plots
from .error_analysis import benchmark_ml, benchmark_xfoil, error_by_regime, percentage_metrics, save_error_summary, time_saved_summary
from .evaluation import save_evaluation_plots
from .models import ModelConfig
from .training_data_generation import TrainingDataConfig, generate_training_dataset
from .training import load_model_bundle, train_from_kulfan_csv


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="airfoil-ml")
    sub=p.add_subparsers(dest="command",required=True)
    
    g=sub.add_parser("generate-training-data",help="generate stochastic airfoil/XFOIL training vectors")
    g.add_argument("--cases",type=int,default=5000)
    g.add_argument("--seed",type=int,default=42)
    g.add_argument("--output-dir",default="output")
    g.add_argument("--xfoil",default="xfoil")
    g.add_argument("--xfoil-timeout",type=int,default=30)
    g.add_argument("--xfoil-iterations",type=int,default=200)
    g.add_argument("--workers",type=int,default=1)
    g.add_argument("--resume",action="store_true")

    t=sub.add_parser("train",help="train grouped baseline and MLP models")
    t.add_argument("--csv",default="output/training_data.csv")
    t.add_argument("--output-dir",default="models")
    t.add_argument("--seed",type=int,default=42)
    t.add_argument("--mlp-hidden",default="128,64,32")
    t.add_argument("--mlp-max-iter",type=int,default=600)
    t.add_argument("--only",nargs="+",default=None)
    t.add_argument("--log-cd",action="store_true")

    e=sub.add_parser("evaluate",help="create plots from a saved test prediction")
    e.add_argument("--csv",default="output/training_data.csv")
    e.add_argument("--model-dir",default="models")
    e.add_argument("--model",default="mlp")
    e.add_argument("--output",default="results/evaluation")

    return p


def main() -> None:
    a=build_parser().parse_args()
    
    if a.command=="generate-training-data":
        cfg=TrainingDataConfig(xfoil_timeout=a.xfoil_timeout,xfoil_iterations=a.xfoil_iterations)
        result=generate_training_dataset(
            a.output_dir,
            n_cases=a.cases,
            seed=a.seed,
            workers=a.workers,
            xfoil_executable=a.xfoil,
            config=cfg,
            resume=a.resume
        )
        print(json.dumps(result,indent=2))
        
    elif a.command=="train":
        cfg=ModelConfig(seed=a.seed,hidden_layers=tuple(int(x) for x in a.mlp_hidden.split(",")),max_iter=a.mlp_max_iter)
        result=train_from_kulfan_csv(
            csv_path=a.csv,
            output_dir=a.output_dir,
            seed=a.seed,
            model_config=cfg,
            only=a.only,
            log_cd=a.log_cd
        )
        print(json.dumps(result,indent=2))

    elif a.command=="evaluate":
        # Simplified evaluation stub
        pass

if __name__ == "__main__":
    main()
