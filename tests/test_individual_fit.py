import numpy as np
import pytest

from cbm.individual_fit import individual_fit
from cbm.optimization import Config


def _config(
    *,
    display=False,
    verbose=False,
    hessian_method="central_fd",
):
    return Config(
        d=2,
        num_init=4,
        random_state=123,
        display=display,
        verbose=verbose,
        hessian_method=hessian_method,
    )


def test_trialwise_and_scalar_likelihood_recover_same_map_and_evidence(
    simple_continuous_data,
    simple_continuous_model,
):
    def scalar_model(theta, data):
        return float(
            np.sum(
                simple_continuous_model(theta, data)
            )
        )

    prior_mean = np.zeros(2)
    prior_variance = np.array([1e6, 1e6])

    trial = individual_fit(
        simple_continuous_data,
        simple_continuous_model,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        config=_config(),
    )

    scalar = individual_fit(
        simple_continuous_data,
        scalar_model,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        config=_config(),
    )

    np.testing.assert_allclose(
        trial.output.parameters,
        scalar.output.parameters,
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        trial.output.log_evidence,
        scalar.output.log_evidence,
        atol=1e-5,
        rtol=1e-5,
    )


def test_zero_prior_variance_fixes_parameter_exactly(
    simple_continuous_data,
    simple_continuous_model,
):
    fit = individual_fit(
        simple_continuous_data,
        simple_continuous_model,
        prior_mean=np.array([0.75, 0.0]),
        prior_variance=np.array([0.0, 1e6]),
        config=_config(),
    )

    assert fit.output.parameters[0, 0] == 0.75
    assert fit.input.fixed_mask[0]
    assert not fit.input.free_mask[0]


def test_valid_fit_has_valid_laplace_state(
    simple_continuous_data,
    simple_continuous_model,
):
    fit = individual_fit(
        simple_continuous_data,
        simple_continuous_model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
        config=_config(),
    )

    diag = fit.math.diagnostics[0]

    assert np.all(np.isfinite(fit.output.parameters))
    assert np.isfinite(fit.output.log_evidence[0])

    # COMPLETE diagnostic state.
    assert diag.hess_n_clipped == 0
    assert diag.hess_raw_min_eig > 0
    assert diag.hess_raw_max_eig > 0
    assert np.isfinite(diag.hess_condition_number)
    assert diag.hess_ill_conditioned is False
    assert diag.laplace_valid is True
    assert diag.laplace_fragile is False


def test_latent_tracking_does_not_change_map_or_evidence(
    simple_continuous_data,
    simple_continuous_model,
):
    def evolution(theta, data):
        x = np.asarray(data["X"]["x"])
        return {
            "latent_mean": theta[0] + theta[1] * x,
        }

    kwargs = dict(
        data=simple_continuous_data,
        model=simple_continuous_model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
    )

    plain = individual_fit(
        **kwargs,
        config=_config(),
    )

    latent = individual_fit(
        **kwargs,
        evolution=evolution,
        config=_config(),
    )

    np.testing.assert_allclose(
        plain.output.parameters,
        latent.output.parameters,
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        plain.output.log_evidence,
        latent.output.log_evidence,
        atol=1e-12,
        rtol=0,
    )

    assert latent.output.latent[0]["latent_mean"].shape == (5,)


def test_verbose_does_not_change_numerical_result(
    simple_continuous_data,
    simple_continuous_model,
):
    kwargs = dict(
        data=simple_continuous_data,
        model=simple_continuous_model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
    )

    quiet = individual_fit(
        **kwargs,
        config=_config(verbose=False),
    )
    loud = individual_fit(
        **kwargs,
        config=_config(verbose=True),
    )

    np.testing.assert_allclose(
        quiet.output.parameters,
        loud.output.parameters,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        quiet.output.log_evidence,
        loud.output.log_evidence,
        atol=1e-12,
    )


def test_verbose_mentions_core_diagnostics(
    simple_continuous_data,
    simple_continuous_model,
    capsys,
):
    individual_fit(
        simple_continuous_data,
        simple_continuous_model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
        config=_config(verbose=True),
    )

    out = capsys.readouterr().out

    assert "L-BFGS-B" in out
    assert "GN" in out
    assert "Observed Hessian" in out
    assert "Laplace valid" in out


@pytest.mark.jax
def test_autodiff_and_central_fd_agree_on_individual_fit(
    simple_continuous_data,
    simple_continuous_model,
):
    jnp = pytest.importorskip("jax.numpy")

    def model_jax(theta, data):
        y = jnp.asarray(data["y"])
        x = jnp.asarray(data["X"]["x"])
        sigma = data["X"]["sigma"]
        mu = theta[0] + theta[1] * x
        return (
            -0.5 * ((y - mu) / sigma) ** 2
            - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))
        )

    kwargs = dict(
        data=simple_continuous_data,
        model=simple_continuous_model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
    )

    fd = individual_fit(
        **kwargs,
        config=_config(
            hessian_method="central_fd",
        ),
    )

    ad = individual_fit(
        **kwargs,
        model_jax=model_jax,
        config=_config(
            hessian_method="autodiff",
        ),
    )

    np.testing.assert_allclose(
        fd.output.parameters,
        ad.output.parameters,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        fd.output.log_evidence,
        ad.output.log_evidence,
        atol=1e-5,
        rtol=1e-5,
    )

    d_fd = fd.math.diagnostics[0]
    d_ad = ad.math.diagnostics[0]

    for d in (d_fd, d_ad):
        assert d.hess_n_clipped == 0
        assert d.laplace_valid
        assert not d.laplace_fragile
