"""Small MAP optimizers used only for the curvature experiment."""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize

from .derivatives import (
    finite_difference_gradient,
    gn_curvature,
    autodiff_gradient_hessian,
)


@dataclass
class FitResult:
    theta: np.ndarray
    objective: float
    gradient_norm: float
    method: str
    seconds: float
    n_lbfgs_runs: int
    n_polish_steps: int


def _clip_to_bounds(x, bounds):
    lower = np.asarray([b[0] for b in bounds])
    upper = np.asarray([b[1] for b in bounds])
    return np.clip(x, lower, upper)


def _polish(
    objective,
    theta,
    bounds,
    curvature_fn,
    gradient_fn,
    max_steps=30,
    tol_df=1e-8,
    min_step=2.0 ** -20,
):
    """Generic damped Newton polish.

    The line search requires strict objective decrease. If a Hessian gives
    a non-descent Newton direction, we fall back to steepest descent for that
    iteration. This is important for the observed AD Hessian away from the
    optimum, which need not be positive definite.
    """
    theta = np.asarray(theta, dtype=float).copy()
    f_current = float(objective(theta))
    n_steps = 0

    for _ in range(max_steps):
        H = np.asarray(curvature_fn(theta), dtype=float)
        H = 0.5 * (H + H.T)
        g = np.asarray(gradient_fn(theta), dtype=float)

        try:
            dx = np.linalg.solve(H, g)
            # For minimization, x - dx is a descent step only when g^T dx > 0.
            if not np.isfinite(dx).all() or float(g @ dx) <= 0.0:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            dx = g.copy()

        step = 1.0
        improved = False

        while step >= min_step:
            candidate = _clip_to_bounds(theta - step * dx, bounds)
            f_new = float(objective(candidate))

            if np.isfinite(f_new) and f_new < f_current:
                theta = candidate
                f_previous = f_current
                f_current = f_new
                improved = True
                n_steps += 1
                break

            step *= 0.5

        if not improved:
            break

        if abs(f_previous - f_current) / (1.0 + abs(f_current)) < tol_df:
            break

    return theta, f_current, n_steps


def lbfgsb_multistart(
    objective,
    bounds,
    rng,
    n_starts=5,
    maxiter=1000,
    gtol=1e-8,
    ftol=1e-12,
):
    """Simple multi-start L-BFGS-B."""
    starts = []

    for _ in range(n_starts):
        starts.append(
            np.array(
                [rng.uniform(lo, hi) for lo, hi in bounds],
                dtype=float,
            )
        )

    results = []

    for x0 in starts:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "gtol": gtol,
                "ftol": ftol,
            },
        )
        results.append(result)

    best = min(
        results,
        key=lambda r: r.fun if np.isfinite(r.fun) else np.inf,
    )

    return best.x.copy(), float(best.fun), len(results)


def fit_map(
    objective,
    bounds,
    rng,
    polish="none",
    trial_func=None,
    prior_precision=None,
    model=None,
    data=None,
    prior_mean=None,
    n_starts=5,
    maxiter=1000,
):
    """Fit a MAP using L-BFGS-B, optionally followed by GN or AD polish."""
    t0 = time.perf_counter()

    theta, f, n_runs = lbfgsb_multistart(
        objective,
        bounds,
        rng,
        n_starts=n_starts,
        maxiter=maxiter,
    )

    n_polish_steps = 0

    if polish == "gn":
        if trial_func is None:
            raise ValueError("GN polish requires trial_func.")

        curvature = lambda z: gn_curvature(
            z,
            trial_func,
            prior_precision=prior_precision,
        )
        gradient = lambda z: finite_difference_gradient(
            objective, z, eps=1e-6
        )

        theta, f, n_polish_steps = _polish(
            objective,
            theta,
            bounds,
            curvature,
            gradient,
        )

    elif polish == "ad":
        if model is None or data is None or prior_mean is None:
            raise ValueError("AD polish requires model/data/prior_mean.")

        def ad_derivatives(z):
            g, H = autodiff_gradient_hessian(
                z, model, data, prior_mean, prior_precision
            )
            return g, H

        curvature = lambda z: ad_derivatives(z)[1]
        gradient = lambda z: ad_derivatives(z)[0]

        theta, f, n_polish_steps = _polish(
            objective,
            theta,
            bounds,
            curvature,
            gradient,
        )

    elif polish != "none":
        raise ValueError("polish must be 'none', 'gn', or 'ad'.")

    # Independent finite-difference gradient for reporting.
    g = finite_difference_gradient(objective, theta, eps=1e-6)

    return FitResult(
        theta=theta,
        objective=float(f),
        gradient_norm=float(np.linalg.norm(g)),
        method=f"lbfgsb+{polish}",
        seconds=float(time.perf_counter() - t0),
        n_lbfgs_runs=n_runs,
        n_polish_steps=n_polish_steps,
    )
