"""Minimal HBI example.

HBI starts from individual-fit files, then refits subjects hierarchically.
"""
from pathlib import Path
import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config
from cbm.hbi import hbi_main
from cbm.hbi_config import HBIConfig

from models import binary_rw, binary_rw_trials
from simulate import binary_subject

rng=np.random.default_rng(11)
data=[binary_subject(rng,theta=(0.35,3.0)) for _ in range(12)]
out=Path("examples/output")
out.mkdir(parents=True,exist_ok=True)

# Candidate 1: free alpha + beta.
def m1(theta,data): return binary_rw(theta,data)
def m1_trials(theta,data): return binary_rw_trials(theta,data)

# Candidate 2: fixed alpha, free beta.
def m2(theta,data): return binary_rw(np.array([0.5,theta[0]]),data)
def m2_trials(theta,data): return binary_rw_trials(np.array([0.5,theta[0]]),data)

cfg1=Config(
    d=2,
    range_bounds=np.array([[0.02,0.1],[0.98,8.0]]),
    hard_bounds=np.array([[0.001,0.01],[0.999,20.0]]),
    num_init=5,verbose=False,
)
cfg2=Config(
    d=1,
    range_bounds=np.array([[0.1],[8.0]]),
    hard_bounds=np.array([[0.01],[20.0]]),
    num_init=5,verbose=False,
)

f1=out/"hbi_map_model1.pkl"
f2=out/"hbi_map_model2.pkl"

individual_fit(
    data,m1,model_trials=m1_trials,
    prior_mean=np.array([0.5,2.0]),
    prior_variance=np.array([1.0,16.0]),
    fname=str(f1),config=cfg1,
)
individual_fit(
    data,m2,model_trials=m2_trials,
    prior_mean=np.array([2.0]),
    prior_variance=np.array([16.0]),
    fname=str(f2),config=cfg2,
)

hbi=hbi_main(
    data=data,
    models=[m1,m2],
    fcbm_maps=[str(f1),str(f2)],
    config=HBIConfig(verbose=True,maxiter=20),
    model_trials=[m1_trials,m2_trials],
)

print("\nmodel frequency:",hbi.output.model_frequency)
print("exceedance probability:",hbi.output.exceedance_prob)
print("protected exceedance probability:",hbi.output.protected_exceedance_prob)
print("group means:",hbi.output.group_mean)
