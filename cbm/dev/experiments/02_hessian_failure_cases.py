"""Original-vs-current Hessian and Laplace failure experiment.

Question
--------
Can a Hessian be positive definite while the Laplace evidence is nevertheless
unreliable, and does the original CBM identify that situation?

The experiment reproduces the ORIGINAL CBM nested-forward-FD Hessian exactly:

    grad_x = scipy.optimize.approx_fprime(x, f, epsilon)
    grad_step = approx_fprime(x + epsilon*e_i, f, epsilon)
    H[i] = (grad_step - grad_x) / epsilon

The original implementation effectively uses positive definiteness as the
curvature criterion for Laplace evidence. The current implementation also
diagnoses the Hessian condition number and marks extreme cases as fragile.

We compare both numerical Hessians with the analytic local Hessian and compare
the Laplace approximation with the EXACT posterior integral.

Run
---
python cbm/dev/02_hessian_failure_cases.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import approx_fprime

from cbm.optimization import BFGSOptimizer, Config


VERBOSE = True
DISPLAY = True

CONDITION_WARN = 1e12

# Healthy to extremely flat local curvature.
EPS_VALUES = np.logspace(-10, -18, 9)

# Locally invisible at the MAP, but globally relevant once the quadratic
# direction becomes sufficiently flat.
QUARTIC = 1e-26


def original_cbm_hessian(func, x, epsilon=1e-5):
    """Reproduce the original CBM nested-forward-FD Hessian."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))

    grad_x = approx_fprime(
        x,
        func,
        epsilon,
    )

    for i in range(n):
        x_step = x.copy()
        x_step[i] += epsilon

        grad_step = approx_fprime(
            x_step,
            func,
            epsilon,
        )

        H[i, :] = (
            grad_step - grad_x
        ) / epsilon

    return 0.5 * (H + H.T)


def objective(theta, eps):
    """Negative log posterior, normalized to zero at the MAP."""
    x, y = np.asarray(theta, dtype=float)

    return (
        0.5 * x ** 2
        + 0.5 * eps * y ** 2
        + 0.25 * QUARTIC * y ** 4
    )


def exact_log_evidence(eps):
    """Exact log integral of exp(-objective) over R^2.

    Rescale y = QUARTIC^(-1/4) z to keep numerical integration stable.
    """
    scale = QUARTIC ** (-0.25)
    a = eps / np.sqrt(QUARTIC)

    integral_z, _ = quad(
        lambda z: np.exp(
            -0.5 * a * z ** 2
            - 0.25 * z ** 4
        ),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=500,
    )

    integral_y = scale * integral_z

    return (
        0.5 * np.log(2.0 * np.pi)
        + np.log(integral_y)
    )


def laplace_log_evidence(H):
    """Laplace log integral; log joint at the MAP is zero."""
    sign, logdet = np.linalg.slogdet(H)

    if sign <= 0:
        return np.nan

    d = H.shape[0]

    return (
        0.5 * d * np.log(2.0 * np.pi)
        - 0.5 * logdet
    )


def header(title):
    print("=" * 90)
    print(title)
    print("=" * 90)


def main():
    x_map = np.zeros(2)

    current = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=1,
            hessian_method="central_fd",
            hessian_step=1e-4,
            condition_number_warn=CONDITION_WARN,
            verbose=False,
            display=False,
        ),
    )

    rows = []

    header(
        "HESSIAN FAILURE CASES — ORIGINAL CBM VS CURRENT IMPLEMENTATION"
    )
    print(
        f"Posterior: 0.5*x^2 + 0.5*eps*y^2 + "
        f"0.25*{QUARTIC:.0e}*y^4"
    )
    print(
        "As eps -> 0 the local Hessian remains PD, but the y posterior "
        "is increasingly non-Gaussian."
    )
    print("-" * 90)

    for eps in EPS_VALUES:
        fun = lambda theta, e=eps: objective(theta, e)

        H_true = np.diag([1.0, eps])
        true_cond = 1.0 / eps

        H_old = original_cbm_hessian(
            fun,
            x_map,
            epsilon=1e-5,
        )

        H_new, diag_new = current.compute_hessian(
            fun,
            x_map,
            return_diagnostics=True,
        )

        eig_old = np.linalg.eigvalsh(H_old)
        old_pd = bool(eig_old[0] > 0.0)
        old_cond = (
            float(eig_old[-1] / eig_old[0])
            if old_pd
            else np.inf
        )

        logE_exact = exact_log_evidence(eps)

        logE_old = (
            laplace_log_evidence(H_old)
            if old_pd
            else np.nan
        )

        logE_new = (
            laplace_log_evidence(H_new)
            if diag_new["is_positive_definite"]
            else np.nan
        )

        old_accepts_laplace = old_pd

        new_laplace_valid = bool(
            diag_new["is_positive_definite"]
        )
        new_laplace_fragile = bool(
            new_laplace_valid
            and diag_new["ill_conditioned"]
        )

        rows.append({
            "eps": eps,
            "true_cond": true_cond,
            "true_min_eig": eps,

            "old_pd": old_pd,
            "old_min_eig": float(eig_old[0]),
            "old_cond": old_cond,
            "old_accepts": old_accepts_laplace,

            "new_pd": diag_new["is_positive_definite"],
            "new_min_eig": float(diag_new["raw_min_eig"]),
            "new_cond": diag_new["condition_number"],
            "new_ill": diag_new["ill_conditioned"],
            "new_valid": new_laplace_valid,
            "new_fragile": new_laplace_fragile,

            "logE_exact": logE_exact,
            "logE_old": logE_old,
            "logE_new": logE_new,

            "old_error": (
                abs(logE_old - logE_exact)
                if np.isfinite(logE_old)
                else np.nan
            ),
            "new_error": (
                abs(logE_new - logE_exact)
                if np.isfinite(logE_new)
                else np.nan
            ),
        })

    if VERBOSE:
        print(
            f"{'eps':>10} {'cond':>12} "
            f"{'old PD':>7} {'old |dlogE|':>13} "
            f"{'new ill':>8} {'new fragile':>12} "
            f"{'new |dlogE|':>13}"
        )
        print("-" * 90)

        for row in rows:
            print(
                f"{row['eps']:10.1e} "
                f"{row['true_cond']:12.1e} "
                f"{str(row['old_pd']):>7} "
                f"{row['old_error']:13.4f} "
                f"{str(row['new_ill']):>8} "
                f"{str(row['new_fragile']):>12} "
                f"{row['new_error']:13.4f}"
            )

        print("-" * 90)

        fragile_rows = [
            row
            for row in rows
            if row["new_fragile"]
        ]

        if fragile_rows:
            first = fragile_rows[0]
            print(
                "Current toolbox first marks Laplace as fragile at "
                f"cond ~= {first['new_cond']:.3e}."
            )

        misleading = [
            row
            for row in rows
            if (
                row["old_accepts"]
                and row["old_error"] > 1.0
            )
        ]

        if misleading:
            first = misleading[0]
            print(
                "Original CBM still accepts a PD Hessian when the exact "
                "Laplace error already exceeds 1 log-evidence unit:"
            )
            print(
                f"  eps={first['eps']:.1e}, "
                f"condition={first['old_cond']:.3e}, "
                f"|delta logE|={first['old_error']:.3f}"
            )

        print()
        print("Interpretation")
        print(
            "  Both Hessian estimators recover the local flat direction."
        )
        print(
            "  Original CBM: PD Hessian -> finite Laplace evidence."
        )
        print(
            "  Current CBM: PD is retained, but extreme conditioning is "
            "explicitly marked laplace_fragile."
        )
        print(
            "  Exact integral: quantifies whether the local Gaussian "
            "approximation is actually accurate."
        )

    if DISPLAY:
        condition = np.array([
            row["true_cond"]
            for row in rows
        ])

        true_min_eig = np.array([
            row["true_min_eig"]
            for row in rows
        ])

        old_min_eig = np.array([
            row["old_min_eig"]
            for row in rows
        ])

        new_min_eig = np.array([
            row["new_min_eig"]
            for row in rows
        ])

        laplace_error = np.array([
            row["new_error"]
            for row in rows
        ])

        fragile = np.array([
            row["new_fragile"]
            for row in rows
        ], dtype=bool)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10.5, 4.2),
        )

        # -----------------------------------------------------------
        # A. Both methods recover the local flat direction.
        # -----------------------------------------------------------
        ax = axes[0]

        lo = true_min_eig.min()
        hi = true_min_eig.max()

        ax.loglog(
            [lo, hi],
            [lo, hi],
            linestyle=":",
            linewidth=1.0,
            label="analytic identity",
        )

        ax.loglog(
            true_min_eig,
            old_min_eig,
            marker="o",
            linestyle="none",
            label="original nested-forward FD",
        )

        ax.loglog(
            true_min_eig,
            new_min_eig,
            marker="s",
            linestyle="none",
            label="current central FD",
        )

        ax.set_xlabel(
            r"analytic smallest eigenvalue $\lambda_{\min}=\epsilon$"
        )
        ax.set_ylabel(
            r"estimated smallest eigenvalue $\hat{\lambda}_{\min}$"
        )
        ax.set_title(
            "A  both methods recover the local curvature"
        )
        ax.legend(
            frameon=False,
            fontsize=8,
        )
        ax.grid(alpha=0.3)

        # -----------------------------------------------------------
        # B. The methods agree locally, but differ in diagnostics.
        # -----------------------------------------------------------
        ax = axes[1]

        ax.semilogx(
            condition,
            laplace_error,
            linewidth=1.4,
            label="Laplace error vs exact integral",
        )

        # Original CBM accepts every point because all Hessians remain PD.
        ax.scatter(
            condition,
            laplace_error,
            marker="o",
            facecolors="none",
            linewidths=1.4,
            label="original: PD -> accepted",
        )

        # Current implementation marks the extreme-conditioning regime.
        if np.any(~fragile):
            ax.scatter(
                condition[~fragile],
                laplace_error[~fragile],
                marker=".",
                s=45,
                label="current: non-fragile",
            )

        if np.any(fragile):
            ax.scatter(
                condition[fragile],
                laplace_error[fragile],
                marker="x",
                s=60,
                linewidths=1.8,
                label="current: laplace_fragile",
            )

        ax.axvline(
            CONDITION_WARN,
            linestyle=":",
            linewidth=1.0,
            label="conditioning warning threshold",
        )

        ax.axhline(
            1.0,
            linestyle=":",
            linewidth=0.8,
            label=r"$|\Delta \log E|=1$",
        )

        ax.set_xlabel(
            "analytic Hessian condition number"
        )
        ax.set_ylabel(
            r"$|\Delta \log E|$ vs exact integral"
        )
        ax.set_title(
            "B  PD curvature can still yield fragile Laplace evidence"
        )
        ax.legend(
            frameon=False,
            fontsize=8,
        )
        ax.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
