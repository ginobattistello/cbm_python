"""Parameter recovery for binary, categorical, and continuous-output models."""
import numpy as np
from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import (
    binary_rw, binary_rw_trials,
    categorical_rw, categorical_rw_trials,
    ces, ces_trials,
)
from simulate import binary_subject, categorical_subject, ces_subject


def recover(name, model, trials, simulator, true_theta, bounds, hard_bounds):
    rng=np.random.default_rng(7)
    n_subjects=20
    # Noisy data + weak prior.
    data=[simulator(rng, theta=true_theta) for _ in range(n_subjects)]

    config=Config(
        d=2,
        range_bounds=bounds,
        hard_bounds=hard_bounds,
        num_init=5,
        verbose=False,
        display=False,
        hessian_method="central_fd",
    )

    fit=individual_fit(
        data,
        model,
        model_trials=trials,
        prior_mean=np.array([0.5, 2.0]) if "CES" not in name else np.array([0.5,0.0]),
        prior_variance=np.array([4.0,64.0]) if "CES" not in name else np.array([4.0,4.0]),
        config=config,
    )

    estimate=np.nanmean(fit.output.parameters,axis=0)
    print(f"\n{name}")
    print("true :", np.asarray(true_theta))
    print("mean recovered:", estimate)
    print("absolute error:", np.abs(estimate-np.asarray(true_theta)))


recover(
    "Binary RW", binary_rw, binary_rw_trials, binary_subject,
    (0.35,3.0),
    np.array([[0.02,0.1],[0.98,8.0]]),
    np.array([[0.001,0.01],[0.999,20.0]]),
)
recover(
    "Categorical RW", categorical_rw, categorical_rw_trials, categorical_subject,
    (0.35,3.0),
    np.array([[0.02,0.1],[0.98,8.0]]),
    np.array([[0.001,0.01],[0.999,20.0]]),
)
recover(
    "CES", ces, ces_trials, ces_subject,
    (0.60,0.30),
    np.array([[0.02,-0.8],[0.98,0.8]]),
    np.array([[0.001,-0.95],[0.999,0.95]]),
)
