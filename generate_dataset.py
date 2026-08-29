#!/usr/bin/env python3
"""
Self-contained script to generate the canonical Kulfan/XFOIL dataset
and train models on a Debian VM.
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path

def ensure_dependencies():
    missing_sys = []
    if not shutil.which("xfoil"):
        missing_sys.append("xfoil")
    if not shutil.which("xvfb-run"):
        missing_sys.append("xvfb-run")
        
    if missing_sys:
        print(f"Missing system dependencies: {', '.join(missing_sys)}")
        print(f"Please run: sudo apt-get install {' '.join(missing_sys)}")
        sys.exit(1)

    try:
        import aerosandbox
        import numpy
        import pandas
        import pyarrow
        import sklearn
        import tqdm
    except ImportError as e:
        print(f"Missing python dependency: {e}")
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow", "tqdm"], check=True)
        print("Dependencies installed. Please re-run the script.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Generate dataset and train models.")
    parser.add_argument("--cases", type=int, default=30000, help="Number of airfoils to process (default: 30000)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2), help="Parallel XFOIL workers")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", default="./output", help="Base directory for data and models")
    parser.add_argument("--xfoil", default="xfoil", help="Path to XFOIL binary")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted generation")
    
    args = parser.parse_args()
    
    ensure_dependencies()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting data generation with {args.cases} cases using {args.workers} workers...")
    from airfoil_ml.training_data_generation import generate_training_dataset, TrainingDataConfig
    
    config = TrainingDataConfig()
    try:
        manifest = generate_training_dataset(
            output_dir=output_dir,
            n_cases=args.cases,
            seed=args.seed,
            workers=args.workers,
            xfoil_executable=args.xfoil,
            config=config,
            resume=args.resume
        )
        print("Data generation complete.")
        print(f"Generated {manifest['successful_vectors']} rows across {manifest['successful_airfoils']} airfoils.")
    except Exception as e:
        print(f"Data generation failed: {e}")
        sys.exit(1)
        
    if not args.skip_train:
        print("Starting model training...")
        from airfoil_ml.training import train_from_kulfan_csv
        from airfoil_ml.models import ModelConfig
        
        try:
            results = train_from_kulfan_csv(
                csv_path=output_dir / "training_data.csv",
                output_dir=output_dir / "models",
                seed=args.seed,
                model_config=ModelConfig(seed=args.seed),
                log_cd=False
            )
            print("Model training complete.")
            import json
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Model training failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
