"""Command-line interface for the reproducible research workflow."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .acquisition import download_uiuc_coordinates, generate_dataset
from .data import load_dataset
from .drag_analysis import compute_drag_summary, save_drag_analysis_plots
from .error_analysis import benchmark_ml, benchmark_xfoil, error_by_regime, percentage_metrics, save_error_summary, time_saved_summary
from .evaluation import save_evaluation_plots
from .features import raw_input_matrix
from .geometry import geometry_from_file
from .geometry_generation import DEFAULT_CAMBER, DEFAULT_CAMBER_POSITION, DEFAULT_THICKNESS, generate_camber_coordinates, sample_camber_grid
from .large_dataset import CST_COLUMNS, FIXED_TARGET_COLUMNS, download_large_dataset, inverse_fixed_targets, load_fixed_re_dataset, train_fixed_re
from .large_evaluation import save_fixed_re_plots
from .models import ModelConfig
from .multi_re_batch import generate_batch
from .training_data_generation import TrainingDataConfig, generate_training_dataset
from .training import load_model_bundle, train_all


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="airfoil-ml"); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("acquire",help="download UIUC coordinates and generate XFOIL polars"); a.add_argument("--airfoils",nargs="+",required=True); a.add_argument("--coordinates",default="data/raw/coordinates"); a.add_argument("--polar-csv",default="data/raw/polars.csv"); a.add_argument("--xfoil",default="xfoil"); a.add_argument("--reynolds",nargs="+",type=float,default=[500000.,1000000.]); a.add_argument("--alpha-start",type=float,default=-10.); a.add_argument("--alpha-end",type=float,default=20.); a.add_argument("--alpha-step",type=float,default=1.); a.add_argument("--xfoil-timeout",type=int,default=60)
    t=sub.add_parser("train",help="train grouped baseline and MLP models"); t.add_argument("--polar-csv",default="data/raw/polars.csv"); t.add_argument("--coordinates",default="data/raw/coordinates"); t.add_argument("--output",default="models/initial"); t.add_argument("--points",type=int,default=100); t.add_argument("--seed",type=int,default=42); t.add_argument("--mlp-hidden",default="128,64,32"); t.add_argument("--mlp-max-iter",type=int,default=600); t.add_argument("--only",nargs="+",default=None); t.add_argument("--log-cd",action="store_true"); t.add_argument("--test-airfoils-json",default=None)
    e=sub.add_parser("evaluate",help="create plots from a saved test prediction"); e.add_argument("--polar-csv",default="data/raw/polars.csv"); e.add_argument("--coordinates",default="data/raw/coordinates"); e.add_argument("--model-dir",default="models/initial"); e.add_argument("--model",default="mlp"); e.add_argument("--output",default="results/initial"); e.add_argument("--points",type=int,default=100)
    ae=sub.add_parser("analyze-error",help="percentage error vs XFOIL reference and wall-clock time saved"); ae.add_argument("--polar-csv",default="data/raw/polars.csv"); ae.add_argument("--coordinates",default="data/raw/coordinates"); ae.add_argument("--model-dir",default="models/initial"); ae.add_argument("--output",default="results/error_analysis"); ae.add_argument("--points",type=int,default=100); ae.add_argument("--models",nargs="+",default=None); ae.add_argument("--xfoil-benchmark",action="store_true"); ae.add_argument("--benchmark-airfoils",nargs="+",default=None); ae.add_argument("--benchmark-reynolds",nargs="+",type=float,default=[100000.,1000000.])
    ad=sub.add_parser("analyze-drag",help="drag-focused metrics and plots from saved test predictions"); ad.add_argument("--polar-csv",default="data/raw/polars.csv"); ad.add_argument("--coordinates",default="data/raw/coordinates"); ad.add_argument("--model-dir",default="models/initial"); ad.add_argument("--output",default="results/drag_analysis"); ad.add_argument("--points",type=int,default=100); ad.add_argument("--models",nargs="+",default=None)
    ld=sub.add_parser("download-large",help="download the large fixed-Re CST airfoil dataset"); ld.add_argument("--output",default="data/raw/external/kanakaero/compiled_airfoil_data.csv")
    tl=sub.add_parser("train-large",help="train models on the large fixed-Re CST dataset"); tl.add_argument("--csv",default="data/raw/external/kanakaero/compiled_airfoil_data.csv"); tl.add_argument("--output",default="models/large_fixed_re"); tl.add_argument("--seed",type=int,default=42); tl.add_argument("--no-log-cd",action="store_true")
    el=sub.add_parser("evaluate-large",help="plot the held-out airfoil test set for the fixed-Re dataset"); el.add_argument("--csv",default="data/raw/external/kanakaero/compiled_airfoil_data.csv"); el.add_argument("--model-dir",default="models/large_fixed_re"); el.add_argument("--model",default="mlp"); el.add_argument("--output",default="results/large_fixed_re")
    c=sub.add_parser("generate-camber",help="generate parametric camber/thickness airfoils"); c.add_argument("--output",default="data/raw/generated_coordinates"); c.add_argument("--n",type=int,default=None); c.add_argument("--seed",type=int,default=42); c.add_argument("--points",type=int,default=100); c.add_argument("--camber-values",default=None); c.add_argument("--camber-positions",default=None); c.add_argument("--thickness-values",default=None); c.add_argument("--run-xfoil",action="store_true"); c.add_argument("--polar-output",default="data/raw/generated_multi_re"); c.add_argument("--reynolds",nargs="+",type=float,default=[30000.,50000.,100000.,250000.,500000.,1000000.]); c.add_argument("--alpha-start",type=float,default=-2.); c.add_argument("--alpha-end",type=float,default=10.); c.add_argument("--alpha-step",type=float,default=1.); c.add_argument("--workers",type=int,default=6)
    b=sub.add_parser("batch-generate",help="generate a sharded multi-Reynolds XFOIL dataset"); b.add_argument("--coordinates",default="data/raw/external/kanakaero/processed_coordinates"); b.add_argument("--output",default="data/raw/multi_re"); b.add_argument("--airfoils",nargs="+",default=None); b.add_argument("--limit",type=int,default=None); b.add_argument("--reynolds",nargs="+",type=float,default=[30000.,50000.,100000.,250000.,500000.,1000000.]); b.add_argument("--alpha-start",type=float,default=-2.); b.add_argument("--alpha-end",type=float,default=10.); b.add_argument("--alpha-step",type=float,default=1.); b.add_argument("--mach",type=float,default=0.); b.add_argument("--xfoil-timeout",type=int,default=45); b.add_argument("--workers",type=int,default=1)
    g=sub.add_parser("generate-training-data",help="generate stochastic airfoil/XFOIL training vectors"); g.add_argument("--cases",type=int,default=100); g.add_argument("--seed",type=int,default=42); g.add_argument("--coordinates-output",default="data/raw/training_coordinates"); g.add_argument("--training-output",default="data/raw/training_data.csv"); g.add_argument("--database-coordinates",default=None); g.add_argument("--xfoil",default="xfoil"); g.add_argument("--xfoil-timeout",type=int,default=30); g.add_argument("--xfoil-iterations",type=int,default=200)
    pr=sub.add_parser("predict",help="predict one operating point"); pr.add_argument("--model-dir",default="models/initial"); pr.add_argument("--model",default="mlp"); pr.add_argument("--coordinates",required=True); pr.add_argument("--alpha",type=float,required=True); pr.add_argument("--reynolds",type=float,required=True); pr.add_argument("--mach",type=float,default=0.); pr.add_argument("--points",type=int,default=100)
    return p


def main() -> None:
    a=build_parser().parse_args()
    if a.command=="acquire":
        download_uiuc_coordinates(a.airfoils,a.coordinates); generate_dataset(a.coordinates,a.polar_csv,a.reynolds,airfoil_ids=a.airfoils,xfoil_executable=a.xfoil,alpha_start=a.alpha_start,alpha_end=a.alpha_end,alpha_step=a.alpha_step,timeout_seconds=a.xfoil_timeout)
    elif a.command=="generate-training-data":
        cfg=TrainingDataConfig(xfoil_timeout=a.xfoil_timeout,xfoil_iterations=a.xfoil_iterations); print(json.dumps(generate_training_dataset(a.training_output,a.coordinates_output,n_cases=a.cases,seed=a.seed,database_coordinates_dir=a.database_coordinates,xfoil_executable=a.xfoil,config=cfg),indent=2))
    elif a.command=="download-large": print(json.dumps({"path":str(download_large_dataset(a.output))}))
    elif a.command=="generate-camber":
        cv=tuple(float(v) for v in a.camber_values.split(",")) if a.camber_values else None; cp=tuple(float(v) for v in a.camber_positions.split(",")) if a.camber_positions else None; tv=tuple(float(v) for v in a.thickness_values.split(",")) if a.thickness_values else None
        records=sample_camber_grid(camber_values=cv or DEFAULT_CAMBER,positions=cp or DEFAULT_CAMBER_POSITION,thickness_values=tv or DEFAULT_THICKNESS)
        if a.n is not None: records=[records[i] for i in np.random.default_rng(a.seed).permutation(len(records))[:a.n]]
        generated=generate_camber_coordinates(a.output,records,n_points=a.points); result={"coordinates_dir":a.output,"airfoils":len(generated["airfoils"])}
        if a.run_xfoil:
            ids=[str(r["airfoil_id"]) for r in generated["airfoils"]]; result["xfoil"]=generate_batch(a.output,a.polar_output,ids,reynolds_values=a.reynolds,alpha_start=a.alpha_start,alpha_end=a.alpha_end,alpha_step=a.alpha_step,workers=a.workers); result["combined_csv"]=str(Path(a.polar_output)/"combined.csv")
        print(json.dumps(result,indent=2))
    elif a.command=="batch-generate":
        coords=sorted(Path(a.coordinates).glob("*.dat")); selected=a.airfoils if a.airfoils is not None else [p.stem for p in (coords[:a.limit] if a.limit else coords)]; print(json.dumps(generate_batch(a.coordinates,a.output,selected,reynolds_values=a.reynolds,alpha_start=a.alpha_start,alpha_end=a.alpha_end,alpha_step=a.alpha_step,mach=a.mach,timeout_seconds=a.xfoil_timeout,workers=a.workers),indent=2))
    elif a.command=="train-large": print(json.dumps(train_fixed_re(load_fixed_re_dataset(a.csv),a.output,seed=a.seed,log_cd=not a.no_log_cd),indent=2))
    elif a.command=="evaluate-large":
        d=load_fixed_re_dataset(a.csv); m,proc=load_model_bundle(a.model_dir,a.model); ids=set(json.loads((Path(a.model_dir)/"split_manifest.json").read_text())["test_airfoils"]); f=d.frame[d.frame.airfoil_id.isin(ids)].copy(); x=f[[*CST_COLUMNS,"alpha_deg"]].to_numpy(float); y=f[list(FIXED_TARGET_COLUMNS)].to_numpy(float); meta=json.loads((Path(a.model_dir)/"dataset_metadata.json").read_text()); pred=inverse_fixed_targets(proc,m.predict(proc.input_scaler.transform(x)),log_cd=meta.get("log_cd",False)); save_fixed_re_plots(f,y,pred,a.output); print(json.dumps({"model":a.model,"rows":len(f),"airfoils":int(f.airfoil_id.nunique())}))
    elif a.command=="train":
        d=load_dataset(a.polar_csv,a.coordinates,a.points); cfg=ModelConfig(seed=a.seed,hidden_layers=tuple(int(x) for x in a.mlp_hidden.split(",")),max_iter=a.mlp_max_iter); ids=json.loads(Path(a.test_airfoils_json).read_text())["test_airfoils"] if a.test_airfoils_json else None; print(json.dumps(train_all(d,a.output,seed=a.seed,model_config=cfg,only=a.only,log_cd=a.log_cd,test_airfoils=ids),indent=2))
    elif a.command in {"evaluate","analyze-error","analyze-drag"}:
        d=load_dataset(a.polar_csv,a.coordinates,a.points); ids=set(json.loads((Path(a.model_dir)/"split_manifest.json").read_text())["test_airfoils"]); f=d.frame[d.frame.airfoil_id.isin(ids)].copy(); names=a.models or [n for n in ["ridge","random_forest","hist_gb","mlp","mlp_torch"] if (Path(a.model_dir)/(f"{n}.joblib" if n!="mlp_torch" else "mlp_torch.pt")).exists()]
        x=np.vstack([raw_input_matrix([d.geometries[str(r.airfoil_id)]],np.array([r.alpha_deg]),np.array([r.reynolds]),np.array([r.mach]))[0] for r in f.itertuples()]); actual=f[["cl","cd","cm"]].to_numpy(float); predictions={}
        for n in names:
            m,proc=load_model_bundle(a.model_dir,n); predictions[n]=(actual,proc.inverse_targets(m.predict(proc.input_scaler.transform(x))))
        if a.command=="evaluate": save_evaluation_plots(f,actual,predictions[names[0]][1],a.output); print(json.dumps({"model":names[0],"rows":len(f)}))
        elif a.command=="analyze-drag":
            summary={n:compute_drag_summary(f,*predictions[n]) for n in names}; save_drag_analysis_plots(f,predictions,a.output,summary); print(json.dumps(summary,indent=2))
        else:
            summary={"overall":{n:percentage_metrics(*predictions[n]) for n in names},"regimes":error_by_regime(f,predictions)}; timing={}
            if a.xfoil_benchmark: timing["xfoil_benchmark"]=benchmark_xfoil(a.coordinates,a.benchmark_airfoils or sorted(ids)[:2],a.benchmark_reynolds)
            timing["ml_benchmark"]=benchmark_ml(a.model_dir,names,x)
            if "xfoil_benchmark" in timing and names: timing["time_saved"]=time_saved_summary(timing["xfoil_benchmark"],timing["ml_benchmark"],names[0])
            save_error_summary(f,predictions,a.output,timing,summary); print(json.dumps(summary,indent=2))
    elif a.command=="predict":
        m,proc=load_model_bundle(a.model_dir,a.model); g=geometry_from_file(a.coordinates,n_points=a.points); x=raw_input_matrix([g],np.array([a.alpha]),np.array([a.reynolds]),np.array([a.mach])); pred=proc.inverse_targets(m.predict(proc.input_scaler.transform(x)))[0]; print(json.dumps(dict(zip(("cl","cd","cm"),pred.tolist())),indent=2))
