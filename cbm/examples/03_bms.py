"""Simple random-effects Bayesian model selection."""
import numpy as np
from cbm.individual_fit import individual_fit
from cbm.optimization import Config
from cbm.group_bms import group_bms

from models import binary_rw, binary_rw_trials
from simulate import binary_subject

rng=np.random.default_rng(10)
data=[binary_subject(rng, theta=(0.35,3.0)) for _ in range(20)]

# Two candidate models with different parameter dimensions:
# M1 fits alpha and beta.
def m1(theta,data): return binary_rw(theta,data)
def m1_trials(theta,data): return binary_rw_trials(theta,data)

# M2 fixes alpha=0.5 and fits only beta.
def m2(theta,data): return binary_rw(np.array([0.5,theta[0]]),data)
def m2_trials(theta,data): return binary_rw_trials(np.array([0.5,theta[0]]),data)

fit1=individual_fit(
    data,m1,model_trials=m1_trials,
    prior_mean=np.array([0.5,2.0]),
    prior_variance=np.array([1.0,16.0]),
    config=Config(
        d=2,
        range_bounds=np.array([[0.02,0.1],[0.98,8.0]]),
        hard_bounds=np.array([[0.001,0.01],[0.999,20.0]]),
        num_init=5,verbose=False,
    ),
)

fit2=individual_fit(
    data,m2,model_trials=m2_trials,
    prior_mean=np.array([2.0]),
    prior_variance=np.array([16.0]),
    config=Config(
        d=1,
        range_bounds=np.array([[0.1],[8.0]]),
        hard_bounds=np.array([[0.01],[20.0]]),
        num_init=5,verbose=False,
    ),
)

L=np.column_stack([fit1.output.log_evidence,fit2.output.log_evidence])
result=group_bms(L,n_samples=100_000)

print(result)
print("model frequency:",result.model_frequency)
print("exceedance probability:",result.exceedance_prob)
print("protected exceedance probability:",result.protected_exceedance_prob)
