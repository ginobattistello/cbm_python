"""Unified individual-fit example.

The same public API handles:
1. binary choice data,
2. categorical choice data,
3. continuous outcomes,
4. legacy/scalar likelihood functions.

Key model rule
--------------
model(theta, data) may return either:
- a scalar summed log-likelihood -> L-BFGS-B only
- a vector of per-trial log-likelihoods -> L-BFGS-B + automatic GN polish
"""

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import (
    binary_model,
    binary_observation,
    categorical_model,
    categorical_observation,
    continuous_model,
    continuous_model_scalar,
    continuous_observation,
)
from simulate import (
    binary_subject,
    categorical_subject,
    continuous_subject,
)


rng = np.random.default_rng(1)

config = Config(
    d=2,
    num_init=5,
    verbose=True,
    display=True,
    hessian_method="central_fd",
)


# =====================================================================
# 1. Binary choice: trialwise likelihood -> GN available
# =====================================================================

binary_data = [
    binary_subject(rng)
    for _ in range(5)
]

binary_fit = individual_fit(
    data=binary_data,
    model=binary_model,
    observation=binary_observation,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

print("\nBinary MAP parameters")
print(binary_fit.output.parameters)

print("\nBinary log model evidence")
print(binary_fit.output.log_evidence)

binary_fit.plot(subject=0)

# =====================================================================
# 2. Categorical choice: trialwise likelihood -> GN available
# =====================================================================

categorical_data = [
    categorical_subject(rng)
    for _ in range(5)
]

categorical_fit = individual_fit(
    data=categorical_data,
    model=categorical_model,
    observation=categorical_observation,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

print("\nCategorical MAP parameters")
print(categorical_fit.output.parameters)

print("\nCategorical log model evidence")
print(categorical_fit.output.log_evidence)

categorical_fit.plot(subject=0)

# =====================================================================
# 3. Continuous output: trialwise likelihood -> GN available
# =====================================================================

continuous_data = [
    continuous_subject(rng)
    for _ in range(5)
]

continuous_fit = individual_fit(
    data=continuous_data,
    model=continuous_model,
    observation=continuous_observation,
    prior_mean=np.array([0.0, 0.0]),
    prior_variance=np.array([10.0, 10.0]),
    config=config,
)

print("\nContinuous MAP parameters")
print(continuous_fit.output.parameters)

print("\nContinuous log model evidence")
print(continuous_fit.output.log_evidence)

continuous_fit.plot(subject=0)

# =====================================================================
# 4. Scalar likelihood: fully supported, GN simply unavailable
# =====================================================================

scalar_config = Config(
    d=2,
    num_init=5,
    verbose=True,
    display=False,
    hessian_method="central_fd",
)

scalar_fit = individual_fit(
    data=continuous_data,
    model=continuous_model_scalar,
    observation=continuous_observation,
    prior_mean=np.array([0.0, 0.0]),
    prior_variance=np.array([10.0, 10.0]),
    config=scalar_config,
)

print("\nScalar model diagnostics")
print(scalar_fit.math.diagnostics[0])
