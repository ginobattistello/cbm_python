import numpy as np
import pytest

from cbm.individual_fit import individual_fit
from cbm.optimization import Config


def linear_model(theta, data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["X"]["x"], dtype=float)
    sigma = float(data["X"].get("sigma", 1.0))
    mu = theta[0] + theta[1] * x
    return (
        -0.5 * ((y - mu) / sigma) ** 2
        - np.log(sigma * np.sqrt(2 * np.pi))
    )


def test_exact_deterministic_parameter_recovery():
    """Noise-free, exactly identifiable model -> theta_hat ~= theta_true."""
    theta_true = np.array([0.7, -1.3])

    x = np.linspace(-2, 2, 41)
    y = theta_true[0] + theta_true[1] * x

    data = [{
        "y": y,
        "X": {
            "x": x,
            "sigma": 1.0,
        },
    }]

    fit = individual_fit(
        data,
        linear_model,
        prior_mean=np.zeros(2),
        # Effectively flat relative to the likelihood.
        prior_variance=np.ones(2) * 1e12,
        config=Config(
            d=2,
            num_init=5,
            random_state=1,
            verbose=False,
            display=False,
        ),
    )

    np.testing.assert_allclose(
        fit.output.parameters[0],
        theta_true,
        atol=1e-5,
        rtol=1e-5,
    )

    diag = fit.math.diagnostics[0]

    assert diag.laplace_valid
    assert not diag.laplace_fragile
    assert diag.hess_n_clipped == 0


@pytest.mark.slow
def test_stochastic_parameter_recovery_smoke():
    """Realistic noisy generative process: assess correlation, bias, RMSE.

    This is deliberately a generic continuous smoke test. Replace or extend it
    with the binary RW, categorical RW, and CES simulators from cbm/examples
    once their final simulation API is frozen.

    Thresholds should be calibrated from repeated simulations before the PR.
    """
    rng = np.random.default_rng(123)
    n_subjects = 20
    n_trials = 80

    true = np.column_stack([
        rng.uniform(-1.0, 1.0, n_subjects),
        rng.uniform(-1.5, 1.5, n_subjects),
    ])

    data = []
    for intercept, slope in true:
        x = rng.normal(size=n_trials)
        y = (
            intercept
            + slope * x
            + rng.normal(scale=0.5, size=n_trials)
        )
        data.append({
            "y": y,
            "X": {
                "x": x,
                "sigma": 0.5,
            },
        })

    fit = individual_fit(
        data,
        linear_model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 100,
        config=Config(
            d=2,
            num_init=3,
            random_state=42,
            verbose=False,
            display=False,
        ),
    )

    estimated = fit.output.parameters

    for j in range(2):
        r = np.corrcoef(
            true[:, j],
            estimated[:, j],
        )[0, 1]
        bias = np.mean(
            estimated[:, j] - true[:, j]
        )
        rmse = np.sqrt(
            np.mean(
                (estimated[:, j] - true[:, j]) ** 2
            )
        )

        # Conservative generic smoke thresholds.
        # Cognitive-model-specific thresholds should be set empirically.
        assert r > 0.9
        assert abs(bias) < 0.15
        assert rmse < 0.25


@pytest.mark.slow
@pytest.mark.jax
def test_fd_vs_ad_scientific_regression():
    jnp = pytest.importorskip("jax.numpy")

    theta_true = np.array([0.5, -0.8])
    x = np.linspace(-2, 2, 51)
    y = theta_true[0] + theta_true[1] * x

    data = [{
        "y": y,
        "X": {
            "x": x,
            "sigma": 1.0,
        },
    }]

    def model_jax(theta, data):
        yj = jnp.asarray(data["y"])
        xj = jnp.asarray(data["X"]["x"])
        sigma = data["X"]["sigma"]
        mu = theta[0] + theta[1] * xj
        return (
            -0.5 * ((yj - mu) / sigma) ** 2
            - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
        )

    kwargs = dict(
        data=data,
        model=linear_model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 100,
    )

    fd = individual_fit(
        **kwargs,
        config=Config(
            d=2,
            num_init=3,
            random_state=1,
            hessian_method="central_fd",
            verbose=False,
        ),
    )

    ad = individual_fit(
        **kwargs,
        model_jax=model_jax,
        config=Config(
            d=2,
            num_init=3,
            random_state=1,
            hessian_method="autodiff",
            verbose=False,
        ),
    )

    H_fd = np.asarray(fd.math.hessian[0])
    H_ad = np.asarray(ad.math.hessian[0])

    rel = np.linalg.norm(
        H_fd - H_ad,
        ord="fro",
    ) / np.linalg.norm(
        H_ad,
        ord="fro",
    )

    assert rel < 1e-4
