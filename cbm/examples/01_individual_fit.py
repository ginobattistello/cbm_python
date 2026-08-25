"""Unified individual-fit example.

The same API handles binary, categorical, continuous, and scalar likelihoods.
Dynamic models may additionally provide ``evolution=`` to expose deterministic
trialwise latent variables at the final MAP.
"""

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import (
    binary_model,
    binary_observation,
    binary_evolution,
    categorical_model,
    categorical_observation,
    categorical_evolution,
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
# 1. Binary RL model: prediction + latent trajectories
# =====================================================================

binary_data = [
    binary_subject(rng)
    for _ in range(5)
]

binary_fit = individual_fit(
    data=binary_data,
    model=binary_model,
    observation=binary_observation,
    evolution=binary_evolution,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

binary_fit.plot(subject=0)

# =====================================================================
# 2. Categorical RL model: prediction + latent trajectories
# =====================================================================

categorical_data = [
    categorical_subject(rng)
    for _ in range(5)
]

categorical_fit = individual_fit(
    data=categorical_data,
    model=categorical_model,
    observation=categorical_observation,
    evolution=categorical_evolution,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

categorical_fit.plot(subject=0)

# =====================================================================
# 3. Continuous model: no latent dynamics
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

continuous_fit.plot(subject=0)

# =====================================================================
# 4. Scalar likelihood: GN unavailable, latent tracking still optional
# =====================================================================

scalar_config = Config(
    d=2,
    num_init=5,
    verbose=True,
    display=True,
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

scalar_fit.plot(subject=0)
