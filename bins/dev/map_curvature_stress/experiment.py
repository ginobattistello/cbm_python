from __future__ import annotations
from dataclasses import dataclass, asdict
import csv
from pathlib import Path
import numpy as np

from .grid import iter_grid
from .models import generate_dataset, trial_loglik_np
from .derivatives import (
    negative_log_posterior_np,
    original_cbm_hessian,
    central_fd_hessian,
    autodiff_hessian,
    relative_frobenius,
    eig_summary,
    laplace_logevidence,
)
from .optimizer import fit_map

@dataclass
class RunRecord:
    model: str
    param1_name: str
    param2_name: str
    param1: float
    param2: float
    replicate: int
    seed: int
    n_trials: int
    sigma: float
    prior_scale: float
    theta_map_1: float
    theta_map_2: float
    objective_map: float
    gradient_norm: float
    original_pd: bool
    fd_pd: bool
    ad_pd: bool
    original_min_eig: float
    fd_min_eig: float
    ad_min_eig: float
    original_log10_condition: float
    fd_log10_condition: float
    ad_log10_condition: float
    original_ad_rel_error: float
    fd_ad_rel_error: float
    delta_loge_original_ad: float
    delta_loge_fd_ad: float
    original_numerical_failure: bool
    fd_numerical_failure: bool
    structural_laplace_failure: bool

def run_one(model,cell,replicate,seed,n_trials=250,sigma=0.1,prior_scale=1.0,n_starts=5):
    rng=np.random.default_rng(seed)
    ds=generate_dataset(model,cell["theta_true"],rng,n_trials,sigma,prior_scale)

    objective=lambda th: negative_log_posterior_np(
        th,model,ds.data,ds.prior_mean,ds.prior_precision
    )
    trial_func=lambda th: trial_loglik_np(model,th,ds.data)

    fit=fit_map(objective,ds.bounds,rng,trial_func,ds.prior_precision,n_starts)
    th=fit.theta

    H0=original_cbm_hessian(objective,th,1e-5)
    Hf=central_fd_hessian(objective,th,1e-4)
    Ha=autodiff_hessian(th,model,ds.data,ds.prior_mean,ds.prior_precision)

    i0=eig_summary(H0); ifd=eig_summary(Hf); ia=eig_summary(Ha)

    e0=laplace_logevidence(fit.objective,H0)
    ef=laplace_logevidence(fit.objective,Hf)
    ea=laplace_logevidence(fit.objective,Ha)

    return RunRecord(
        model=model,
        param1_name=cell["param1_name"],
        param2_name=cell["param2_name"],
        param1=cell["param1"],
        param2=cell["param2"],
        replicate=replicate,
        seed=seed,
        n_trials=n_trials,
        sigma=sigma,
        prior_scale=prior_scale,
        theta_map_1=float(th[0]),
        theta_map_2=float(th[1]),
        objective_map=float(fit.objective),
        gradient_norm=float(fit.gradient_norm),
        original_pd=i0["is_pd"],
        fd_pd=ifd["is_pd"],
        ad_pd=ia["is_pd"],
        original_min_eig=i0["min_eig"],
        fd_min_eig=ifd["min_eig"],
        ad_min_eig=ia["min_eig"],
        original_log10_condition=i0["log10_condition"],
        fd_log10_condition=ifd["log10_condition"],
        ad_log10_condition=ia["log10_condition"],
        original_ad_rel_error=relative_frobenius(H0,Ha),
        fd_ad_rel_error=relative_frobenius(Hf,Ha),
        delta_loge_original_ad=float(e0-ea) if np.isfinite(e0) and np.isfinite(ea) else np.nan,
        delta_loge_fd_ad=float(ef-ea) if np.isfinite(ef) and np.isfinite(ea) else np.nan,
        original_numerical_failure=(not i0["is_pd"]) and ia["is_pd"],
        fd_numerical_failure=(not ifd["is_pd"]) and ia["is_pd"],
        structural_laplace_failure=(not ia["is_pd"]),
    )

def run_experiment(models=("binary","categorical","ces"),n_replicates=20,n_trials=250,
                   sigma=0.1,prior_scale=1.0,n_starts=5,seed=123,
                   output_dir="results/map_curvature_stress"):
    import jax
    jax.config.update("jax_enable_x64",True)

    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]

    for mi,model in enumerate(models):
        cells=list(iter_grid(model))
        print(f"\n=== {model}: {len(cells)} cells ===")
        for ci,cell in enumerate(cells):
            print(f"  {ci+1:02d}/{len(cells)}  {cell['param1_name']}={cell['param1']}, "
                  f"{cell['param2_name']}={cell['param2']}", flush=True)
            for rep in range(n_replicates):
                run_seed=seed + mi*1_000_000 + ci*10_000 + rep
                rec=run_one(model,cell,rep,run_seed,n_trials,sigma,prior_scale,n_starts)
                rows.append(asdict(rec))

    path=out/"stress_raw.csv"
    if rows:
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\nSaved: {path}")
    return rows
