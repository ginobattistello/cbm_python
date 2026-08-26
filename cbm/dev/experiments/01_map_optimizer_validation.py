"""Validate multi-step Gauss-Newton polishing.

Question
--------
Can GN perform several genuine refinement steps while converging to the same
minimum as a high-accuracy reference optimization?

This is a development experiment, not a unit test. A Rosenbrock least-squares
problem is used because:
- its optimum is known exactly;
- it has a non-trivial curved valley;
- the residual representation gives a natural J.T @ J GN curvature;
- several GN steps are required from the chosen starting point.

Run
---
python cbm/dev/01_map_optimizer_validation.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from cbm.optimization import BFGSOptimizer, Config


VERBOSE = True
DISPLAY = True


def residuals(theta):
    """Rosenbrock residual vector."""
    x, y = np.asarray(theta, dtype=float)
    return np.array([
        10.0 * (y - x ** 2),
        1.0 - x,
    ])


def objective(theta):
    r = residuals(theta)
    return 0.5 * np.dot(r, r)


def header(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    x0 = np.array([-1.2, 1.0])
    x_true = np.array([1.0, 1.0])

    header("MAP OPTIMIZER VALIDATION — MULTI-STEP GN POLISH")
    print("Objective: Rosenbrock nonlinear least squares")
    print(f"Initial point: {x0}")
    print(f"Known optimum: {x_true}")
    print("-" * 78)

    # ---------------------------------------------------------------
    # High-accuracy reference solution.
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    reference = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={
            "maxiter": 5000,
            "gtol": 1e-12,
            "ftol": 1e-15,
        },
    )
    t_reference = time.perf_counter() - t0

    # ---------------------------------------------------------------
    # GN polish.
    #
    # We call the polish stage directly because the scientific question
    # here is specifically about GN dynamics. A full L-BFGS-B run often
    # reaches the gradient tolerance before GN has anything left to do.
    # ---------------------------------------------------------------
    config = Config(
        d=2,
        num_init=1,
        range_bounds=5,
        hard_bounds=np.array([
            [-np.inf, -np.inf],
            [ np.inf,  np.inf],
        ]),
        display=True,   # retain the GN path
        verbose=False,
        condition_number_warn=1e12,
        random_state=1,
    )

    optimizer = BFGSOptimizer(
        d=2,
        config=config,
        gtol=1e-10,
    )

    t0 = time.perf_counter()
    (
        x_gn,
        f_gn,
        status,
        n_steps,
        gn_diag,
    ) = optimizer._newton_polish(
        objective,
        x0,
        trial_func=residuals,
        prior_precision=np.zeros((2, 2)),
        n_steps=30,
        tol_df=1e-12,
    )
    t_gn = time.perf_counter() - t0

    trace = optimizer._temp_polish_trace
    path = np.vstack([row[0] for row in trace])
    f_path = np.array([row[1] for row in trace])

    delta_theta = np.linalg.norm(x_gn - reference.x)
    delta_f = abs(f_gn - float(reference.fun))

    if VERBOSE:
        print("Reference L-BFGS-B")
        print(f"  success:              {reference.success}")
        print(f"  theta:                {reference.x}")
        print(f"  objective:            {reference.fun:.6e}")
        print(f"  elapsed:              {t_reference * 1e3:.3f} ms")
        print()

        print("GN polish")
        print(f"  status:               {status.value}")
        print(f"  accepted steps:       {n_steps}")
        print(f"  theta:                {x_gn}")
        print(f"  objective:            {f_gn:.6e}")
        print(f"  elapsed:              {t_gn * 1e3:.3f} ms")
        print(f"  GN PD:                {gn_diag['is_positive_definite']}")
        print(f"  GN condition number:  {gn_diag['condition_number']:.6e}")
        print(f"  GN ill-conditioned:   {gn_diag['ill_conditioned']}")
        print()

        print("Agreement")
        print(f"  ||delta theta||:      {delta_theta:.6e}")
        print(f"  |delta objective|:    {delta_f:.6e}")
        print("-" * 78)

        if n_steps < 2:
            print("WARNING: fewer than two GN steps were accepted.")
        else:
            print("PASS: several GN polishing steps were accepted.")

        if delta_theta < 1e-5 and delta_f < 1e-10:
            print("PASS: GN and reference optimization reach the same minimum.")
        else:
            print("CHECK: GN/reference agreement is weaker than expected.")

    if DISPLAY:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.0, 3.8),
        )

        ax = axes[0]
        ax.plot(
            np.arange(len(f_path)),
            -f_path,
            marker="o",
            linewidth=1.2,
        )
        ax.set_title("A  GN log-joint trajectory")
        ax.set_xlabel("GN state")
        ax.set_ylabel("log-joint")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(
            np.arange(len(path)),
            path[:, 0],
            marker="o",
            label=r"$\theta_0$",
        )
        ax.plot(
            np.arange(len(path)),
            path[:, 1],
            marker="o",
            linestyle="--",
            label=r"$\theta_1$",
        )
        ax.axhline(
            1.0,
            linewidth=0.8,
            linestyle=":",
        )
        ax.set_title(
            f"B  parameter path · {n_steps} accepted steps"
        )
        ax.set_xlabel("GN state")
        ax.set_ylabel(r"$\theta$")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
