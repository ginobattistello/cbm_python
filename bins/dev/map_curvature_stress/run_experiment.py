from __future__ import annotations
import argparse
from .experiment import run_experiment

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["binary","categorical","ces","all"],default="all")
    ap.add_argument("--n-replicates",type=int,default=20)
    ap.add_argument("--n-trials",type=int,default=250)
    ap.add_argument("--sigma",type=float,default=0.1)
    ap.add_argument("--prior-scale",type=float,default=1.0)
    ap.add_argument("--n-starts",type=int,default=5)
    ap.add_argument("--seed",type=int,default=123)
    ap.add_argument("--output-dir",default="results/map_curvature_stress")
    args=ap.parse_args()

    models=["binary","categorical","ces"] if args.model=="all" else [args.model]
    run_experiment(
        models=models,
        n_replicates=args.n_replicates,
        n_trials=args.n_trials,
        sigma=args.sigma,
        prior_scale=args.prior_scale,
        n_starts=args.n_starts,
        seed=args.seed,
        output_dir=args.output_dir,
    )

if __name__=="__main__":
    main()
