"""Minimal validation for the final MAP architecture.

Answers only two questions:
1. Does GN polishing reach the same MAP as an exact AD-Hessian Newton polish,
   while being faster?
2. Is the proposed central-FD observed Hessian closer to AD than the original
   nested-forward-FD CBM Hessian?

Run:
    python -m cbm.dev.map_curvature_experiment
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize, approx_fprime

from cbm.hessian import central_fd_hessian, autodiff_hessian
from .models import (
    binary_trials_np, binary_trials_jax, simulate_binary,
    categorical_trials_np, categorical_trials_jax, simulate_categorical,
    ces_trials_np, ces_trials_jax, simulate_ces,
)


def original_cbm_hessian(fun, x, eps=1e-5):
    g0 = approx_fprime(x, fun, eps)
    H = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        gp = approx_fprime(xp, fun, eps)
        H[i, :] = (gp - g0) / eps
    return 0.5 * (H + H.T)


def score_outer_product(trials, theta, prior_precision):
    f0 = np.asarray(trials(theta), float)
    J = np.zeros((f0.size, len(theta)))
    for i in range(len(theta)):
        h = 1e-4 * theta[i]
        if abs(h) <= 1e-4:
            h = 1e-4
        xp = theta.copy()
        xp[i] += h
        J[:, i] = (np.asarray(trials(xp), float) - f0) / h
    return J.T @ J + prior_precision


def fd_gradient(fun, x, eps=1e-6):
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        h = eps * max(1, abs(x[i]))
        xp=x.copy(); xm=x.copy()
        xp[i]+=h; xm[i]-=h
        g[i]=(fun(xp)-fun(xm))/(2*h)
    return g


def polish(fun, x, hessian_fun, gradient_fun, bounds, max_steps=20):
    x = x.copy()
    f = float(fun(x))
    for _ in range(max_steps):
        H = hessian_fun(x)
        g = gradient_fun(x)
        try:
            step = np.linalg.solve(H, g)
            if float(g @ step) <= 0:
                break
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        improved = False
        for _ in range(20):
            xn = np.clip(x - scale*step, bounds[0], bounds[1])
            fn = float(fun(xn))
            if np.isfinite(fn) and fn < f:
                x, f = xn, fn
                improved = True
                break
            scale *= 0.5
        if not improved:
            break
    return x, f


def run_model(name, data, trials_np, trials_jax, bounds, prior_mean, prior_sd):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    P = np.diag(1.0 / np.asarray(prior_sd)**2)
    m = np.asarray(prior_mean, float)

    def obj_np(theta):
        d = theta - m
        return -np.sum(trials_np(theta, data)) + 0.5*d@P@d

    def obj_jax(theta):
        d = theta - jnp.asarray(m)
        return -jnp.sum(trials_jax(theta, data)) + 0.5*d@jnp.asarray(P)@d

    bnds = list(zip(bounds[0], bounds[1]))
    base = minimize(obj_np, m, method="L-BFGS-B", bounds=bnds)
    x0 = base.x

    trial_fun = lambda th: trials_np(th, data)

    t0 = time.perf_counter()
    x_gn, f_gn = polish(
        obj_np, x0,
        lambda th: score_outer_product(trial_fun, th, P),
        lambda th: fd_gradient(obj_np, th),
        bounds,
    )
    gn_time = time.perf_counter() - t0

    ad_grad = jax.grad(obj_jax)
    ad_hess = jax.hessian(obj_jax)
    t0 = time.perf_counter()
    x_ad, f_ad = polish(
        obj_np, x0,
        lambda th: np.asarray(ad_hess(jnp.asarray(th)), float),
        lambda th: np.asarray(ad_grad(jnp.asarray(th)), float),
        bounds,
    )
    ad_time = time.perf_counter() - t0

    H_ad = autodiff_hessian(obj_jax, x_gn)
    H_cfd = central_fd_hessian(obj_np, x_gn)
    H_old = original_cbm_hessian(obj_np, x_gn)

    denom = np.linalg.norm(H_ad, ord="fro")
    e_cfd = np.linalg.norm(H_cfd-H_ad, ord="fro") / denom
    e_old = np.linalg.norm(H_old-H_ad, ord="fro") / denom

    print(f"\n{name}")
    print("-" * len(name))
    print(f"|MAP_GN - MAP_AD|       : {np.linalg.norm(x_gn-x_ad):.3e}")
    print(f"|objective_GN - AD|     : {abs(f_gn-f_ad):.3e}")
    print(f"GN polish time           : {gn_time:.4f} s")
    print(f"AD polish time           : {ad_time:.4f} s")
    print(f"AD/GN time ratio         : {ad_time/max(gn_time,1e-12):.2f}x")
    print(f"central-FD vs AD Hessian : {e_cfd:.3e}")
    print(f"original FD vs AD Hessian: {e_old:.3e}")


def main():
    rng = np.random.default_rng(123)
    run_model(
        "binary RW + softmax",
        simulate_binary(rng), binary_trials_np, binary_trials_jax,
        np.array([[0.02,0.1],[0.98,8.0]]),
        [0.5,2.0], [0.25,2.0],
    )
    run_model(
        "categorical RW + softmax",
        simulate_categorical(rng), categorical_trials_np, categorical_trials_jax,
        np.array([[0.02,0.1],[0.98,8.0]]),
        [0.5,2.0], [0.25,2.0],
    )
    run_model(
        "CES + continuous output",
        simulate_ces(rng), ces_trials_np, ces_trials_jax,
        np.array([[0.02,-0.8],[0.98,0.8]]),
        [0.5,0.0], [0.25,0.5],
    )


if __name__ == "__main__":
    main()
