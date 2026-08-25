"""Regression check: latent tracking must not change inference."""

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from cbm.examples.models import binary_model, binary_evolution
from cbm.examples.simulate import binary_subject


rng = np.random.default_rng(42)
data = [
    binary_subject(rng, n_trials=60)
    for _ in range(3)
]

config = Config(
    d=2,
    num_init=3,
    verbose=False,
    display=False,
    hessian_method="central_fd",
)

np.random.seed(123)
fit_plain = individual_fit(
    data=data,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

np.random.seed(123)
fit_latent = individual_fit(
    data=data,
    model=binary_model,
    evolution=binary_evolution,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

theta_error = np.nanmax(
    np.abs(
        fit_plain.output.parameters
        - fit_latent.output.parameters
    )
)

evidence_error = np.nanmax(
    np.abs(
        fit_plain.output.log_evidence
        - fit_latent.output.log_evidence
    )
)

print("max |Δtheta_MAP|:", theta_error)
print("max |Δlog evidence|:", evidence_error)

assert theta_error < 1e-12
assert evidence_error < 1e-12

latent = fit_latent.output.latent[0]

assert latent["Q"].shape == (60, 2)
assert latent["prediction_error"].shape == (60,)

print("latent tracking regression test: PASS")
