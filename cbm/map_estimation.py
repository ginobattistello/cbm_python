"""MAP estimation for CBM.

This module connects cognitive models to the MAP optimizer.

Final architecture
------------------
1. ``model`` defines the scalar NumPy log-likelihood.
2. ``model_trials`` is optional and returns per-trial log-likelihoods.
   When supplied, it enables the Gauss-Newton polish used ONLY for MAP
   optimization.
3. The final observed Hessian is computed independently at the MAP:
       - central finite differences by default;
       - JAX autodiff when ``Config.hessian_method == "autodiff"`` and
         the modeller supplies ``model_jax``.
4. ``result.hess`` is therefore always the observed posterior Hessian,
   never the Gauss-Newton optimization curvature.

The returned field historically named ``loglik`` is retained for backward
compatibility. It is the log joint at the MAP,

    log p(y, theta_MAP | model)
      = log likelihood + log prior,

not the bare log-likelihood.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

from .optimization import BFGSOptimizer, Config, OptimizationResult


def _validate_prior(
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate and normalize Gaussian-prior inputs."""
    prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
    prior_precision = np.asarray(prior_precision, dtype=float)

    d = prior_mean.size

    if prior_precision.shape != (d, d):
        raise ValueError(
            f"prior_precision must have shape ({d}, {d}), "
            f"got {prior_precision.shape}"
        )

    prior_precision = 0.5 * (
        prior_precision + prior_precision.T
    )

    sign, log_det_precision = np.linalg.slogdet(prior_precision)
    if sign <= 0:
        raise ValueError("prior_precision must be positive definite.")

    try:
        np.linalg.cholesky(prior_precision)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "prior_precision must be positive definite."
        ) from exc

    return prior_mean, prior_precision, float(log_det_precision)


def log_posterior(
    parameters: np.ndarray,
    model: Callable[[np.ndarray, Any], float],
    data: Any,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    log_det_precision: Optional[float] = None,
) -> float:
    """Compute the positive Gaussian-prior log joint.

    Returns
    -------
    float
        log p(y, theta | model)
        = log p(y | theta, model) + log p(theta | model).

    Notes
    -----
    The optimizer minimizes the negative of this quantity. The sign
    conversion occurs once, inside ``optimize_map``.
    """
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
    prior_precision = np.asarray(prior_precision, dtype=float)

    log_likelihood = float(model(parameters, data))
    diff = parameters - prior_mean

    if log_det_precision is None:
        sign, log_det_precision = np.linalg.slogdet(
            prior_precision
        )
        if sign <= 0:
            raise ValueError(
                "prior_precision must have positive determinant."
            )

    d = parameters.size
    log_prior = (
        -0.5 * d * np.log(2.0 * np.pi)
        + 0.5 * float(log_det_precision)
        - 0.5 * float(diff @ prior_precision @ diff)
    )

    return log_likelihood + log_prior


def _make_jax_neg_log_posterior(
    model_jax: Callable,
    data: Any,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    log_det_precision: float,
) -> Callable:
    """Create the JAX version of the full negative log posterior."""
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "A JAX model was requested but JAX is not installed."
        ) from exc

    prior_mean_jax = jnp.asarray(prior_mean)
    prior_precision_jax = jnp.asarray(prior_precision)

    d = prior_mean.size
    prior_constant = (
        -0.5 * d * np.log(2.0 * np.pi)
        + 0.5 * log_det_precision
    )

    def neg_log_post_jax(theta_vec):
        theta_vec = jnp.asarray(theta_vec)
        diff = theta_vec - prior_mean_jax

        log_prior = (
            prior_constant
            - 0.5 * diff @ prior_precision_jax @ diff
        )
        log_likelihood = model_jax(theta_vec, data)

        return -(log_likelihood + log_prior)

    return neg_log_post_jax


def optimize_map(
    data: Any,
    model: Callable[[np.ndarray, Any], float],
    config: Config,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    method: str = "LAP",
    model_trials: Optional[
        Callable[[np.ndarray, Any], np.ndarray]
    ] = None,
    model_jax: Optional[Callable] = None,
) -> Tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    OptimizationResult,
]:
    """Estimate one subject's MAP and observed posterior Hessian.

    Parameters
    ----------
    data
        Subject data.
    model
        NumPy model returning the summed log-likelihood.
    config
        Optimization configuration.
    prior_mean
        Gaussian-prior mean.
    prior_precision
        Gaussian-prior precision.
    method
        Only ``"LAP"`` is supported.
    model_trials
        Optional NumPy model returning per-trial log-likelihoods.
        When supplied, it enables the Gauss-Newton MAP polish:

            H_opt = J.T @ J + prior_precision.

        This curvature is used ONLY for optimization.
    model_jax
        Optional JAX implementation of the same summed log-likelihood.

        Required only when:

            config.hessian_method == "autodiff".

        The same Gaussian prior is added internally so AD differentiates
        the full negative log posterior.

    Returns
    -------
    loglik
        Backward-compatible name for the log joint at the MAP.
    parameters
        MAP parameter vector.
    hessian
        Observed Hessian of the full negative log posterior at the MAP.
    grad
        Gradient of the full negative log posterior at the MAP.
    flag
        MAP/curvature quality flag.
    result
        Full ``OptimizationResult``.

    Notes
    -----
    A non-PD observed Hessian does not erase a valid MAP. In that case
    ``result.is_hess_pos=False`` and ``result.laplace_valid=False``.
    Higher-level code must avoid computing Laplace evidence for that fit.
    """
    if method != "LAP":
        raise ValueError(
            f"Method '{method}' is not recognized. "
            "Only 'LAP' is supported."
        )

    prior_mean, prior_precision, log_det_precision = (
        _validate_prior(prior_mean, prior_precision)
    )

    d = prior_mean.size

    if config.d is not None and config.d != d:
        raise ValueError(
            f"config.d={config.d} but prior_mean has length {d}."
        )

    hessian_method = str(
        getattr(config, "hessian_method", "central_fd")
    ).lower()

    if hessian_method == "autodiff" and model_jax is None:
        raise ValueError(
            "Config.hessian_method='autodiff' requires model_jax."
        )

    optimizer = BFGSOptimizer(d, config=config)

    # NumPy full negative log posterior.
    def neg_log_post(theta_vec):
        return -log_posterior(
            theta_vec,
            model,
            data,
            prior_mean,
            prior_precision,
            log_det_precision=log_det_precision,
        )

    # Optional per-trial likelihood for GN optimization only.
    trial_func = None
    if model_trials is not None:

        def trial_func(theta_vec):
            values = np.asarray(
                model_trials(theta_vec, data),
                dtype=float,
            ).reshape(-1)

            if values.size == 0:
                raise ValueError(
                    "model_trials returned an empty array."
                )

            return values

    # Optional JAX full negative log posterior for AD Hessian only.
    neg_log_post_jax = None
    if model_jax is not None:
        neg_log_post_jax = _make_jax_neg_log_posterior(
            model_jax=model_jax,
            data=data,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            log_det_precision=log_det_precision,
        )

    result = optimizer.optimize(
        neg_log_post,
        x_init=prior_mean.copy(),
        trial_func=trial_func,
        prior_precision=prior_precision,
        neg_log_post_jax=neg_log_post_jax,
    )

    if result.flag == 0:
        parameters = np.full(d, np.nan)
        hessian = np.full((d, d), np.nan)
        grad = np.full(d, np.nan)
        loglik = np.nan

    else:
        # Keep the MAP even when the observed Hessian is non-PD.
        parameters = np.asarray(result.x, dtype=float)
        hessian = np.asarray(result.hess, dtype=float)
        grad = np.asarray(result.grad, dtype=float)

        loglik = log_posterior(
            result.x,
            model,
            data,
            prior_mean,
            prior_precision,
            log_det_precision=log_det_precision,
        )

    return (
        loglik,
        parameters,
        hessian,
        grad,
        result.flag,
        result,
    )
