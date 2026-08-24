"""Random-effects Bayesian model selection.

M1 estimates alpha and beta.
M2 fixes alpha=0.5 through a zero prior variance and estimates beta.
"""

import numpy as np

from cbm.bms_group import bms_group
from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import binary_model
from simulate import binary_subject


rng = np.random.default_rng(10)
data = [binary_subject(rng, theta=(0.35, 3.0)) for _ in range(20)]

config = Config(
    d=2,
    range_bounds=np.array([[0.02, 0.10], [0.98, 8.00]]),
    hard_bounds=np.array([[0.001, 0.01], [0.999, 20.0]]),
    num_init=5,
    verbose=False,
    display=False,
)

fit_free = individual_fit(
    data=data,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

fit_fixed = individual_fit(
    data=data,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([0.0, 16.0]),
    config=config,
)

log_evidence = np.column_stack([
    fit_free.output.log_evidence,
    fit_fixed.output.log_evidence,
])

result = bms_group(log_evidence, n_samples=100_000, verbose=True)