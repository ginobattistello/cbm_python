"""Observed-posterior Hessian backends.

This module deliberately separates the Hessian used for Laplace inference
from the Gauss-Newton curvature used to polish the MAP.

Default
-------
central finite differences of the full negative log posterior.

Optional
--------
automatic differentiation when the modeller supplies a JAX-compatible
negative-log-posterior callable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass(frozen=True)
class HessianDiagnostics:
    method: str
    min_eigenvalue: float
    is_positive_definite: bool
    log_determinant: float
    condition_number: float


def central_fd_hessian(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    relative_step: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference Hessian of a scalar objective.

    Uses h_i = relative_step * max(1, |x_i|). This is the estimator
    validated against autodiff in cbm/dev/map_curvature_experiment.py.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    h = relative_step * np.maximum(1.0, np.abs(x))
    f0 = float(fun(x))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (
            fun(x + ei) - 2.0 * f0 + fun(x - ei)
        ) / (h[i] ** 2)

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            Hij = (
                fun(x + ei + ej)
                - fun(x + ei - ej)
                - fun(x - ei + ej)
                + fun(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
            H[i, j] = Hij
            H[j, i] = Hij

    return 0.5 * (H + H.T)


def autodiff_hessian(
    fun_jax: Callable,
    x: np.ndarray,
) -> np.ndarray:
    """Observed Hessian from JAX automatic differentiation."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "hessian_method='autodiff' requires JAX. Install with `pip install jax`."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    xj = jnp.asarray(x, dtype=jnp.float64)
    H = np.asarray(jax.hessian(fun_jax)(xj), dtype=float)
    return 0.5 * (H + H.T)


def diagnose_hessian(H: np.ndarray, method: str) -> HessianDiagnostics:
    """Check the observed Hessian without modifying it."""
    H = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    eig = np.linalg.eigvalsh(H)
    min_eig = float(eig[0])
    max_eig = float(eig[-1])
    is_pd = bool(min_eig > 0.0)

    if is_pd:
        sign, logdet = np.linalg.slogdet(H)
        logdet = float(logdet) if sign > 0 else np.nan
        condition = float(max_eig / min_eig)
    else:
        logdet = np.nan
        condition = np.inf

    return HessianDiagnostics(
        method=method,
        min_eigenvalue=min_eig,
        is_positive_definite=is_pd,
        log_determinant=logdet,
        condition_number=condition,
    )


def observed_hessian(
    neg_log_post: Callable[[np.ndarray], float],
    x: np.ndarray,
    method: str = "central_fd",
    neg_log_post_jax: Optional[Callable] = None,
    relative_step: float = 1e-4,
):
    """Compute and diagnose the Hessian used for Laplace inference."""
    method = str(method).lower()

    if method == "central_fd":
        H = central_fd_hessian(neg_log_post, x, relative_step=relative_step)
    elif method in {"autodiff", "ad", "jax"}:
        if neg_log_post_jax is None:
            raise ValueError(
                "Autodiff Hessian requested but no JAX objective was supplied."
            )
        H = autodiff_hessian(neg_log_post_jax, x)
        method = "autodiff"
    else:
        raise ValueError(
            "Unknown Hessian method. Use 'central_fd' or 'autodiff'."
        )

    return H, diagnose_hessian(H, method)
