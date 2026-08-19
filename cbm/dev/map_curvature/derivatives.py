"""Derivative and curvature utilities for the MAP curvature experiment."""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np


@dataclass
class HessianResult:
    hessian: np.ndarray
    seconds: float


def negative_log_posterior_np(theta, model, data, prior_mean, prior_precision):
    from .models import trial_loglik_np

    theta = np.asarray(theta, dtype=float)
    ll = np.sum(trial_loglik_np(model, theta, data))

    d = theta - prior_mean
    prior = -0.5 * d @ prior_precision @ d

    return float(-(ll + prior))


def negative_log_posterior_jax(theta, model, data, prior_mean, prior_precision):
    import jax.numpy as jnp
    from .models import trial_loglik_jax

    ll = jnp.sum(trial_loglik_jax(model, theta, data))

    d = theta - jnp.asarray(prior_mean)
    P = jnp.asarray(prior_precision)
    prior = -0.5 * (d @ P @ d)

    return -(ll + prior)


def finite_difference_gradient(fun, x, eps=1e-6):
    """Central finite-difference gradient."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)

    for i in range(len(x)):
        h = eps * max(1.0, abs(x[i]))
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (fun(xp) - fun(xm)) / (2.0 * h)

    return g


def finite_difference_hessian(fun, x, eps=1e-4):
    """Central finite-difference Hessian of a scalar objective.

    The diagonal uses a standard second difference. Off-diagonal entries
    use the symmetric four-point formula.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)

    hs = np.array([eps * max(1.0, abs(v)) for v in x])

    f0 = float(fun(x))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = hs[i]

        H[i, i] = (
            fun(x + ei) - 2.0 * f0 + fun(x - ei)
        ) / hs[i] ** 2

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = hs[j]

            value = (
                fun(x + ei + ej)
                - fun(x + ei - ej)
                - fun(x - ei + ej)
                + fun(x - ei - ej)
            ) / (4.0 * hs[i] * hs[j])

            H[i, j] = value
            H[j, i] = value

    return 0.5 * (H + H.T)


def gn_curvature(
    theta,
    trial_func,
    prior_precision=None,
    relative_step=1e-4,
    absolute_floor=1e-4,
):
    """Reproduce the fork's current J^T J + prior-precision curvature.

    The per-trial Jacobian uses the same one-sided relative finite difference
    rule as cbm/optimization.py.
    """
    theta = np.asarray(theta, dtype=float)
    f0 = np.asarray(trial_func(theta), dtype=float)

    J = np.zeros((f0.size, theta.size), dtype=float)

    for i in range(theta.size):
        dx = relative_step * theta[i]
        if abs(dx) <= absolute_floor:
            dx = absolute_floor

        x_step = theta.copy()
        x_step[i] += dx

        J[:, i] = (
            np.asarray(trial_func(x_step), dtype=float) - f0
        ) / dx

    H = J.T @ J

    if prior_precision is not None:
        H = H + np.asarray(prior_precision, dtype=float)

    return 0.5 * (H + H.T)


def autodiff_gradient_hessian(
    theta,
    model,
    data,
    prior_mean,
    prior_precision,
):
    """Return exact AD gradient/Hessian of the full negative log posterior."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required. Install it with `pip install jax`."
        ) from exc

    fun = lambda z: negative_log_posterior_jax(
        z, model, data, prior_mean, prior_precision
    )

    theta_j = jnp.asarray(theta, dtype=jnp.float64)

    # JAX may default to float32 unless x64 is enabled. The experiment
    # enables x64 in run_experiment.py.
    g = np.asarray(jax.grad(fun)(theta_j), dtype=float)
    H = np.asarray(jax.hessian(fun)(theta_j), dtype=float)

    return g, 0.5 * (H + H.T)


def timed_autodiff_hessian(theta, model, data, prior_mean, prior_precision):
    t0 = time.perf_counter()
    g, H = autodiff_gradient_hessian(
        theta, model, data, prior_mean, prior_precision
    )
    return g, HessianResult(H, time.perf_counter() - t0)


def timed_fd_hessian(fun, theta, eps):
    t0 = time.perf_counter()
    H = finite_difference_hessian(fun, theta, eps)
    return HessianResult(H, time.perf_counter() - t0)


def compare_hessians(H1, H2):
    """Relative Frobenius error, plus spectra and log-determinant."""
    H1 = 0.5 * (np.asarray(H1) + np.asarray(H1).T)
    H2 = 0.5 * (np.asarray(H2) + np.asarray(H2).T)

    denom = max(np.linalg.norm(H2, ord="fro"), np.finfo(float).eps)
    rel_fro = np.linalg.norm(H1 - H2, ord="fro") / denom

    eig1 = np.linalg.eigvalsh(H1)
    eig2 = np.linalg.eigvalsh(H2)

    sign1, logdet1 = np.linalg.slogdet(H1)
    sign2, logdet2 = np.linalg.slogdet(H2)

    return {
        "relative_frobenius_error": float(rel_fro),
        "min_eig_1": float(eig1[0]),
        "min_eig_2": float(eig2[0]),
        "logdet_1": float(logdet1) if sign1 > 0 else np.nan,
        "logdet_2": float(logdet2) if sign2 > 0 else np.nan,
        "logdet_difference": (
            float(logdet1 - logdet2)
            if sign1 > 0 and sign2 > 0
            else np.nan
        ),
    }
