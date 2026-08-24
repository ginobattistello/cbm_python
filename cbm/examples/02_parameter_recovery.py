"""Parameter recovery for binary, categorical, and continuous models."""

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import binary_model, categorical_model, continuous_model
from simulate import binary_subject, categorical_subject, continuous_subject


def recover(
    name,
    model,
    simulator,
    true_theta,
    prior_mean,
    prior_variance,
    range_bounds,
    hard_bounds,
):
    """Simulate subjects and compare recovered parameters with the truth."""
    rng = np.random.default_rng(7)

    data = [
        simulator(rng, theta=true_theta)
        for _ in range(20)
    ]

    fit = individual_fit(
        data=data,
        model=model,
        prior_mean=np.asarray(prior_mean, dtype=float),
        prior_variance=np.asarray(prior_variance, dtype=float),
        config=Config(
            d=len(true_theta),
            range_bounds=range_bounds,
            hard_bounds=hard_bounds,
            num_init=5,
            verbose=False,
            display=False,
            hessian_method="central_fd",
        ),
    )

    estimate = np.nanmean(fit.output.parameters, axis=0)

    print(f"\n{name}")
    print("true           :", np.asarray(true_theta))
    print("mean recovered :", estimate)
    print("absolute error :", np.abs(estimate - np.asarray(true_theta)))


recover(
    "Binary RW",
    binary_model,
    binary_subject,
    true_theta=(0.35, 3.0),
    prior_mean=(0.5, 2.0),
    prior_variance=(4.0, 64.0),
    range_bounds=np.array([[0.02, 0.10], [0.98, 8.00]]),
    hard_bounds=np.array([[0.001, 0.01], [0.999, 20.0]]),
)

recover(
    "Categorical RW",
    categorical_model,
    categorical_subject,
    true_theta=(0.35, 3.0),
    prior_mean=(0.5, 2.0),
    prior_variance=(4.0, 64.0),
    range_bounds=np.array([[0.02, 0.10], [0.98, 8.00]]),
    hard_bounds=np.array([[0.001, 0.01], [0.999, 20.0]]),
)

recover(
    "Continuous linear",
    continuous_model,
    continuous_subject,
    true_theta=(1.0, 2.0),
    prior_mean=(0.0, 0.0),
    prior_variance=(25.0, 25.0),
    range_bounds=np.array([[-4.0, -4.0], [4.0, 4.0]]),
    hard_bounds=np.array([[-20.0, -20.0], [20.0, 20.0]]),
)
