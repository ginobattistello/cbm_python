"""Model specification, options, and outputs."""
import numpy as np
from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import binary_rw, binary_rw_trials, binary_rw_jax
from simulate import binary_subject

rng=np.random.default_rng(1)
data=[binary_subject(rng) for _ in range(5)]

config=Config(
    d=2,
    range_bounds=np.array([[0.02,0.1],[0.98,8.0]]),
    hard_bounds=np.array([[0.001,0.01],[0.999,20.0]]),
    num_init=5,
    verbose=True,       # print fitting progress
    display=True,       # retain diagnostic traces for fit.plot()
    hessian_method="central_fd",  # default after the refactor
)

fit=individual_fit(
    data,
    binary_rw,
    model_trials=binary_rw_trials,
    model_jax=binary_rw_jax,   # optional; only used if hessian_method="autodiff"
    prior_mean=np.array([0.5,2.0]),
    prior_variance=np.array([1.0,16.0]),
    config=config,
)

print("\nMAP parameters")
print(fit.output.parameters)

print("\nLog model evidence")
print(fit.output.log_evidence)

print("\nFirst-subject diagnostics")
print(fit.math.diagnostics[0])

# With display=True:
fit.plot(subject=0, show=True)
