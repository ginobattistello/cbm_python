import copy
import numpy as np
import pytest

from cbm.hbi import (
    _validate_hbi_map_files,
    hbi_init,
)
from cbm.individual_fit import individual_fit
from cbm.optimization import Config


def _model(theta, data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["X"]["x"], dtype=float)
    mu = theta[0] + theta[1] * x
    return -0.5 * (y - mu) ** 2


def _fit():
    data = [
        {
            "y": np.array([-1.0, 0.0, 1.0]),
            "X": {"x": np.array([-1.0, 0.0, 1.0])},
        },
        {
            "y": np.array([-0.8, 0.2, 1.2]),
            "X": {"x": np.array([-1.0, 0.0, 1.0])},
        },
    ]

    return individual_fit(
        data,
        _model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
        config=Config(
            d=2,
            num_init=3,
            random_state=1,
            verbose=False,
            display=False,
        ),
    )


def test_hbi_validation_accepts_valid_individual_fit():
    fit = _fit()
    _validate_hbi_map_files([fit])


def test_hbi_validation_rejects_invalid_laplace():
    fit = _fit()
    bad = copy.deepcopy(fit)

    bad.math.diagnostics[0].laplace_valid = False

    with pytest.raises(
        ValueError,
        match="valid Laplace",
    ):
        _validate_hbi_map_files([bad])


def test_hbi_init_produces_finite_initial_state():
    fit = _fit()

    hyper = {
        "b": 1.0,
        "v": 0.5,
        "s": 0.01,
    }

    inits, priors, configs = hbi_init(
        [fit],
        hyper,
    )

    assert inits is not None
    assert priors is not None
    assert len(configs) == 1


def test_hbi_preserves_fixed_parameter_information():
    data = [
        {
            "y": np.array([0.0, 1.0, 2.0]),
            "X": {"x": np.array([0.0, 1.0, 2.0])},
        },
    ]

    fit = individual_fit(
        data,
        _model,
        prior_mean=np.array([0.0, 1.0]),
        prior_variance=np.array([10.0, 0.0]),
        config=Config(
            d=2,
            num_init=2,
            random_state=1,
            verbose=False,
        ),
    )

    hyper = {
        "b": 1.0,
        "v": 0.5,
        "s": 0.01,
    }

    inits, priors, configs = hbi_init(
        [fit],
        hyper,
    )

    # HBI operates only on free dimensions.
    assert fit.input.fixed_mask[1]
    assert fit.output.parameters[0, 1] == 1.0
