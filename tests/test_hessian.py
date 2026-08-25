import numpy as np
import pytest

from cbm.optimization import BFGSOptimizer, Config


def _optimizer(
    *,
    method="central_fd",
    threshold=1e12,
    step=1e-4,
):
    return BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=1,
            hessian_method=method,
            hessian_step=step,
            condition_number_warn=threshold,
            random_state=1,
            verbose=False,
            display=False,
        ),
    )


def _assert_hessian_diagnostics(
    diag,
    *,
    pd,
    ill,
):
    # Test the COMPLETE Hessian diagnostic state.
    assert diag["is_positive_definite"] is pd
    assert diag["ill_conditioned"] is ill
    assert diag["n_clipped"] == 0
    assert np.isfinite(diag["raw_min_eig"])
    assert np.isfinite(diag["raw_max_eig"])
    assert diag["raw_max_eig"] >= diag["raw_min_eig"]

    if pd:
        assert diag["raw_min_eig"] > 0
        assert np.isfinite(diag["condition_number"])
        assert diag["condition_number"] >= 1
    else:
        assert diag["raw_min_eig"] <= 0
        assert np.isinf(diag["condition_number"])


def test_central_fd_matches_analytic_quadratic(quadratic_matrix):
    A = quadratic_matrix

    def objective(theta):
        theta = np.asarray(theta)
        return 0.5 * theta @ A @ theta

    opt = _optimizer()

    H, diag = opt.compute_hessian(
        objective,
        np.array([0.3, -0.7]),
        return_diagnostics=True,
    )

    np.testing.assert_allclose(
        H,
        A,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        H,
        H.T,
        atol=1e-12,
    )

    _assert_hessian_diagnostics(
        diag,
        pd=True,
        ill=False,
    )


def test_pd_well_conditioned_full_fit_state():
    def objective(theta):
        return 0.5 * np.sum(np.asarray(theta) ** 2)

    result = _optimizer().optimize(objective)

    # COMPLETE final state.
    assert result.is_hess_pos is True
    assert result.hess_ill_conditioned is False
    assert result.hess_n_clipped == 0
    assert result.laplace_valid is True
    assert result.laplace_fragile is False
    assert result.hess_raw_min_eig > 0
    assert result.hess_raw_max_eig > 0
    assert np.isfinite(result.hess_condition_number)
    assert result.flag == 1.0


def test_pd_ill_conditioned_full_fit_state():
    def objective(theta):
        theta = np.asarray(theta)
        return 0.5 * (
            theta[0] ** 2
            + 1e-8 * theta[1] ** 2
        )

    result = _optimizer(
        threshold=1e6,
    ).optimize(objective)

    assert result.is_hess_pos is True
    assert result.hess_ill_conditioned is True
    assert result.hess_n_clipped == 0
    assert result.laplace_valid is True
    assert result.laplace_fragile is True
    assert result.hess_raw_min_eig > 0
    assert result.hess_raw_max_eig > result.hess_raw_min_eig
    assert result.hess_condition_number > 1e6
    assert result.flag == 0.5


def test_non_pd_full_fit_state():
    # Concave in theta[1] at the stationary point.
    def objective(theta):
        theta = np.asarray(theta)
        return 0.5 * theta[0] ** 2 - 0.5 * theta[1] ** 2

    opt = _optimizer()
    H, diag = opt.compute_hessian(
        objective,
        np.zeros(2),
        return_diagnostics=True,
    )

    _assert_hessian_diagnostics(
        diag,
        pd=False,
        ill=False,
    )

    eig = np.linalg.eigvalsh(H)
    assert eig[0] < 0

    # Crucial regression: no clipping/repair.
    assert diag["n_clipped"] == 0


@pytest.mark.jax
def test_central_fd_matches_autodiff(quadratic_matrix):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    A = quadratic_matrix

    def objective(theta):
        theta = np.asarray(theta)
        return 0.5 * theta @ A @ theta

    A_jax = jnp.asarray(A)

    def objective_jax(theta):
        return 0.5 * theta @ A_jax @ theta

    x = np.array([0.2, -0.4])

    fd = _optimizer(method="central_fd")
    ad = _optimizer(method="autodiff")

    H_fd, d_fd = fd.compute_hessian(
        objective,
        x,
        return_diagnostics=True,
    )
    H_ad, d_ad = ad.compute_hessian(
        objective,
        x,
        neg_log_post_jax=objective_jax,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(
        H_fd,
        H_ad,
        rtol=1e-5,
        atol=1e-6,
    )

    for diag in (d_fd, d_ad):
        _assert_hessian_diagnostics(
            diag,
            pd=True,
            ill=False,
        )
