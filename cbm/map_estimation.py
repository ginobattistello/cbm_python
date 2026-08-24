"""MAP estimation with a single model-likelihood interface.

Public model contract
---------------------
``model(theta, data)`` may return either:

1. a scalar
       summed log-likelihood
       -> L-BFGS-B optimization
       -> GN polish unavailable

2. a one-dimensional vector of length T
       per-trial/per-observation log-likelihoods
       -> internally summed for L-BFGS-B
       -> retained separately for GN polishing

This removes the redundant ``model_trials`` argument. The toolbox detects the
model output type once at the prior mean and uses the same convention for the
entire fit.

The optional ``model_jax`` follows the same scalar-or-vector contract. Its
output is internally summed before the full negative log posterior is
differentiated for the optional AD Hessian.

Fixed parameters
----------------
When a ``ParameterSpace`` is supplied, cognitive models still receive the full
parameter vector while optimization and Laplace inference operate only on the
free coordinates.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

from .optimization import (
    BFGSOptimizer,
    Config,
    ConvergenceStatus,
    OptimizationResult,
)
from .parameter_space import ParameterSpace


def _validate_prior(
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the Gaussian prior over FREE parameters."""
    prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
    prior_precision = np.asarray(prior_precision, dtype=float)
    d = prior_mean.size

    if prior_precision.shape != (d, d):
        raise ValueError(
            f"prior_precision must have shape ({d}, {d}), "
            f"got {prior_precision.shape}"
        )

    if d == 0:
        return prior_mean, np.empty((0, 0)), 0.0

    prior_precision = 0.5 * (prior_precision + prior_precision.T)

    try:
        np.linalg.cholesky(prior_precision)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "prior_precision over free parameters must be positive definite."
        ) from exc

    sign, logdet = np.linalg.slogdet(prior_precision)
    if sign <= 0:
        raise ValueError(
            "prior_precision over free parameters must have positive "
            "determinant."
        )

    return prior_mean, prior_precision, float(logdet)


def _model_output(
    model: Callable,
    theta_full: np.ndarray,
    data: Any,
) -> tuple[float, Optional[np.ndarray]]:
    """Evaluate a NumPy model and classify its output.

    Returns
    -------
    summed_loglik
        Scalar likelihood used by the optimizer.
    trial_loglik
        ``None`` for scalar models, otherwise the one-dimensional trialwise
        log-likelihood vector used by GN.
    """
    raw = model(theta_full, data)
    arr = np.asarray(raw, dtype=float)

    if arr.ndim == 0:
        value = float(arr)
        if not np.isfinite(value):
            raise ValueError("model returned a non-finite scalar likelihood")
        return value, None

    if arr.ndim != 1:
        raise ValueError(
            "model must return either a scalar summed log-likelihood or a "
            "one-dimensional vector of per-trial log-likelihoods; "
            f"got shape {arr.shape}."
        )

    if arr.size == 0:
        raise ValueError("model returned an empty likelihood vector")

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "model returned non-finite per-trial log-likelihood values"
        )

    return float(np.sum(arr)), arr


def _detect_trialwise_model(
    model: Callable,
    theta_full: np.ndarray,
    data: Any,
) -> bool:
    """Return True when ``model`` exposes per-trial likelihood values."""
    _, trial = _model_output(model, theta_full, data)
    return trial is not None


def log_posterior(
    parameters: np.ndarray,
    model: Callable[[np.ndarray, Any], Any],
    data: Any,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    log_det_precision: Optional[float] = None,
) -> float:
    """Compute the positive log joint.

    ``model`` may return either a scalar log-likelihood or a per-trial vector.
    In the latter case the vector is summed internally.
    """
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
    prior_precision = np.asarray(prior_precision, dtype=float)
    d = parameters.size

    log_likelihood, _ = _model_output(model, parameters, data)

    if d == 0:
        return log_likelihood

    diff = parameters - prior_mean

    if log_det_precision is None:
        sign, log_det_precision = np.linalg.slogdet(prior_precision)
        if sign <= 0:
            raise ValueError("prior_precision must have positive determinant")

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
    parameter_space: ParameterSpace,
) -> Callable:
    """Build the JAX full negative log posterior over FREE parameters."""
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "hessian_method='autodiff' requires JAX."
        ) from exc

    mean_jax = jnp.asarray(prior_mean)
    precision_jax = jnp.asarray(prior_precision)
    d = prior_mean.size

    prior_constant = (
        -0.5 * d * np.log(2.0 * np.pi)
        + 0.5 * log_det_precision
    )

    def neg_log_post_jax(theta_free):
        theta_full = parameter_space.expand_jax(theta_free)
        raw = jnp.asarray(model_jax(theta_full, data))

        # Scalar and trialwise JAX models share one objective.
        log_likelihood = jnp.sum(raw)

        if d == 0:
            return -log_likelihood

        diff = theta_free - mean_jax
        log_prior = (
            prior_constant
            - 0.5 * diff @ precision_jax @ diff
        )
        return -(log_likelihood + log_prior)

    return neg_log_post_jax


def _fixed_only_result(
    data,
    model,
    parameter_space: ParameterSpace,
) -> tuple:
    """Return a valid zero-dimensional MAP when every parameter is fixed."""
    theta_full = parameter_space.full_mean.copy()
    log_joint, _ = _model_output(model, theta_full, data)

    result = OptimizationResult(
        x=np.empty(0, dtype=float),
        f=-log_joint,
        hess=np.empty((0, 0), dtype=float),
        grad=np.empty(0, dtype=float),
        flag=1.0,
        success=True,
        nit=0,
        n_runs=0,
        is_hess_pos=True,
        abs_g=0.0,
        x_init=np.empty(0, dtype=float),
        hess_method="fixed",
        convergence_status=ConvergenceStatus.SKIPPED_NO_TRIAL_FUNC,
        hess_raw_min_eig=None,
        hess_n_clipped=0,
        hess_condition_number=1.0,
        laplace_valid=True,
        n_inits_agreeing=0,
        at_hard_bounds=np.empty(0, dtype=bool),
        weak_identifiability=None,
        search_path=None,
        search_f=None,
        polish_path=None,
        polish_f=None,
        n_polish_steps=0,
    )

    return (
        log_joint,
        theta_full,
        np.empty((0, 0), dtype=float),
        np.zeros(parameter_space.d_full, dtype=float),
        1.0,
        result,
    )


def optimize_map(
    data: Any,
    model: Callable[[np.ndarray, Any], Any],
    config: Config,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    method: str = "LAP",
    model_jax: Optional[Callable] = None,
    parameter_space: Optional[ParameterSpace] = None,
) -> Tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    OptimizationResult,
]:
    """Estimate one subject's MAP.

    The model output determines the optimization path automatically:

    scalar model
        L-BFGS-B -> observed Hessian

    trialwise model
        L-BFGS-B -> GN polish -> observed Hessian

    The final observed Hessian remains independent of GN in both cases.
    """
    if method != "LAP":
        raise ValueError(
            f"Method '{method}' is not recognized. Only 'LAP' is supported."
        )

    prior_mean, prior_precision, log_det_precision = _validate_prior(
        prior_mean,
        prior_precision,
    )

    if parameter_space is None:
        parameter_space = ParameterSpace.all_free(
            prior_mean,
            prior_precision,
        )

    if parameter_space.d_free != prior_mean.size:
        raise ValueError(
            "parameter_space.d_free must equal the free prior dimension"
        )

    if parameter_space.d_free == 0:
        return _fixed_only_result(data, model, parameter_space)

    reduced_config = parameter_space.reduce_config(config)

    hessian_method = str(
        getattr(reduced_config, "hessian_method", "central_fd")
    ).lower()

    if hessian_method == "autodiff" and model_jax is None:
        raise ValueError(
            "Config.hessian_method='autodiff' requires model_jax."
        )

    # Detect the likelihood interface once at the prior mean.
    theta_probe = parameter_space.expand(prior_mean)
    is_trialwise = _detect_trialwise_model(
        model,
        theta_probe,
        data,
    )

    # Cognitive model wrapper used by the scalar posterior objective.
    def model_free(theta_free, subject_data):
        theta_full = parameter_space.expand(theta_free)
        summed, _ = _model_output(model, theta_full, subject_data)
        return summed

    def neg_log_post(theta_free):
        return -log_posterior(
            theta_free,
            model_free,
            data,
            prior_mean,
            prior_precision,
            log_det_precision=log_det_precision,
        )

    # GN receives the vector model only when it actually exists.
    trial_func = None
    if is_trialwise:
        def trial_func(theta_free):
            theta_full = parameter_space.expand(theta_free)
            _, trial = _model_output(model, theta_full, data)

            if trial is None:
                raise RuntimeError(
                    "model output changed from trialwise to scalar during "
                    "optimization; model return shape must be stable."
                )
            return trial

    neg_log_post_jax = None
    if model_jax is not None:
        neg_log_post_jax = _make_jax_neg_log_posterior(
            model_jax=model_jax,
            data=data,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            log_det_precision=log_det_precision,
            parameter_space=parameter_space,
        )

    optimizer = BFGSOptimizer(
        parameter_space.d_free,
        config=reduced_config,
    )

    result = optimizer.optimize(
        neg_log_post,
        x_init=prior_mean.copy(),
        trial_func=trial_func,
        prior_precision=prior_precision,
        neg_log_post_jax=neg_log_post_jax,
    )

    if result.flag == 0:
        parameters_full = np.full(parameter_space.d_full, np.nan)
        gradient_full = np.full(parameter_space.d_full, np.nan)
        hessian = np.full(
            (parameter_space.d_free, parameter_space.d_free),
            np.nan,
        )
        log_joint = np.nan

    else:
        parameters_full = parameter_space.expand(result.x)
        gradient_full = parameter_space.expand_free_vector(
            result.grad,
            fixed_value=0.0,
        )
        hessian = np.asarray(result.hess, dtype=float)

        log_joint = log_posterior(
            result.x,
            model_free,
            data,
            prior_mean,
            prior_precision,
            log_det_precision=log_det_precision,
        )

    return (
        log_joint,
        parameters_full,
        hessian,
        gradient_full,
        result.flag,
        result,
    )
