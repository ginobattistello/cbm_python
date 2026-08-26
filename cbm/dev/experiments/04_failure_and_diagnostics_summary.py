"""Failure-mode and diagnostic summary for the current optimizer.

Question
--------
Do the new warning/flag fields distinguish the main numerical failure modes
that were previously collapsed into coarse success/failure decisions?

The script deliberately creates:
- a healthy fit;
- a PD but ill-conditioned observed Hessian;
- an unsuccessful L-BFGS-B run;
- invalid objective evaluations;
- a hard-bound MAP;
- an ill-conditioned GN curvature.

Run
---
python cbm/dev/04_failure_and_diagnostics_summary.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from cbm.optimization import (
    BFGSOptimizer,
    Config,
)


VERBOSE = True
DISPLAY = True


def run_case(
    name,
    objective,
    *,
    d=2,
    condition_number_warn=1e12,
    max_iter=1000,
    hard_bounds=None,
    inits=None,
):
    if hard_bounds is None:
        hard_bounds = np.array([
            -np.inf * np.ones(d),
            np.inf * np.ones(d),
        ])

    # Keep random-start bounds finite and strictly inside any finite hard
    # constraints. These experiments use explicit starts, but Config still
    # validates the initialization region.
    range_low = np.full(d, -5.0)
    range_high = np.full(d, 5.0)

    for j in range(d):
        lo = hard_bounds[0, j]
        hi = hard_bounds[1, j]

        if np.isfinite(lo) and np.isfinite(hi):
            width = hi - lo
            range_low[j] = lo + 0.1 * width
            range_high[j] = hi - 0.1 * width

        elif np.isfinite(lo):
            range_low[j] = max(
                lo + 0.1,
                -5.0,
            )
            range_high[j] = max(
                range_low[j] + 1.0,
                5.0,
            )

        elif np.isfinite(hi):
            range_high[j] = min(
                hi - 0.1,
                5.0,
            )
            range_low[j] = min(
                range_high[j] - 1.0,
                -5.0,
            )

    config = Config(
        d=d,
        num_init=1,
        inits=inits,
        range_bounds=np.vstack([
            range_low,
            range_high,
        ]),
        hard_bounds=hard_bounds,
        condition_number_warn=condition_number_warn,
        max_iter=max_iter,
        random_state=1,
        verbose=False,
        display=True,
    )

    optimizer = BFGSOptimizer(
        d,
        config,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimizer.optimize(
            objective,
        )

    return {
        "name": name,
        "result": result,
        "warnings": [
            str(w.message)
            for w in caught
        ],
    }


def gn_ill_conditioned_case():
    def objective(theta):
        theta = np.asarray(theta)
        return 0.5 * np.sum(
            (theta - 1.0) ** 2
        )

    def residuals(theta):
        theta = np.asarray(theta)
        return np.array([
            theta[0],
            1e-8 * theta[1],
        ])

    optimizer = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=1,
            condition_number_warn=1e10,
            hard_bounds=np.array([
                [-np.inf, -np.inf],
                [ np.inf,  np.inf],
            ]),
            display=True,
            verbose=False,
        ),
    )

    (
        _,
        _,
        status,
        n_steps,
        diag,
    ) = optimizer._newton_polish(
        objective,
        np.zeros(2),
        trial_func=residuals,
        prior_precision=np.zeros((2, 2)),
    )

    return {
        "name": "GN ill-conditioned",
        "status": status.value,
        "n_steps": n_steps,
        "diag": diag,
    }


def main():
    cases = []

    # Healthy.
    cases.append(
        run_case(
            "healthy",
            lambda x: 0.5 * np.sum(
                (np.asarray(x) - 1.0) ** 2
            ),
            inits=np.array([
                [0.0, 0.0],
            ]),
        )
    )

    # PD but ill-conditioned final observed Hessian.
    cases.append(
        run_case(
            "observed H ill-cond",
            lambda x: 0.5 * (
                np.asarray(x)[0] ** 2
                + 1e-8 * np.asarray(x)[1] ** 2
            ),
            condition_number_warn=1e6,
            inits=np.array([
                [0.1, 0.1],
            ]),
        )
    )

    # L-BFGS-B iteration limit.
    cases.append(
        run_case(
            "L-BFGS-B limit",
            lambda x: np.sum(
                (np.asarray(x) - 1.0) ** 2
            ),
            max_iter=0,
            inits=np.array([
                [5.0, 5.0],
            ]),
        )
    )

    # Invalid model domain for part of the search.
    def invalid_objective(x):
        x = np.asarray(x)

        if x[0] < 0:
            raise ValueError(
                "x[0] outside model domain"
            )

        return (
            (x[0] - 0.5) ** 2
            + x[1] ** 2
        )

    cases.append(
        run_case(
            "invalid evaluations",
            invalid_objective,
            inits=np.array([
                [-0.5, 0.0],
            ]),
        )
    )

    # True unconstrained optimum x=2, but hard bound x<=1.
    cases.append(
        run_case(
            "hard-bound MAP",
            lambda x: (
                np.asarray(x)[0] - 2.0
            ) ** 2,
            d=1,
            hard_bounds=np.array([
                [-1.0],
                [ 1.0],
            ]),
            inits=np.array([
                [0.0],
            ]),
        )
    )

    gn_case = gn_ill_conditioned_case()

    if VERBOSE:
        print("=" * 118)
        print("FAILURE + DIAGNOSTIC SUMMARY")
        print("=" * 118)

        print(
            f"{'case':<22} "
            f"{'flag':>5} "
            f"{'LBFGS':>7} "
            f"{'PD':>5} "
            f"{'ill-cond':>9} "
            f"{'Laplace':>8} "
            f"{'fragile':>8} "
            f"{'invalid':>8} "
            f"{'bound':>6}"
        )
        print("-" * 118)

        for case in cases:
            r = case["result"]

            print(
                f"{case['name']:<22} "
                f"{r.flag:5.1f} "
                f"{str(r.success):>7} "
                f"{str(r.is_hess_pos):>5} "
                f"{str(r.hess_ill_conditioned):>9} "
                f"{str(r.laplace_valid):>8} "
                f"{str(r.laplace_fragile):>8} "
                f"{r.n_invalid_evaluations:8d} "
                f"{str(bool(np.any(r.at_hard_bounds))):>6}"
            )

        print("-" * 118)
        print("GN-only curvature case")
        print(f"  status:               {gn_case['status']}")
        print(f"  accepted steps:       {gn_case['n_steps']}")
        print(
            "  positive definite:    "
            f"{gn_case['diag']['is_positive_definite']}"
        )
        print(
            "  condition number:      "
            f"{gn_case['diag']['condition_number']:.3e}"
        )
        print(
            "  ill-conditioned:       "
            f"{gn_case['diag']['ill_conditioned']}"
        )

        print()
        print("Warnings captured")
        for case in cases:
            if not case["warnings"]:
                continue

            print(f"  {case['name']}:")
            for message in case["warnings"]:
                print(f"    - {message}")

    if DISPLAY:
        names = [
            case["name"]
            for case in cases
        ]
        flags = [
            case["result"].flag
            for case in cases
        ]

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.0, 4.0),
        )

        ax = axes[0]
        ax.bar(
            np.arange(len(names)),
            flags,
        )
        ax.set_xticks(
            np.arange(len(names)),
            names,
            rotation=30,
            ha="right",
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fit flag")
        ax.set_title("A  diagnostic flag by failure case")
        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax = axes[1]

        conditions = []
        labels = []

        for case in cases:
            cond = case[
                "result"
            ].hess_condition_number

            if cond is not None and np.isfinite(cond):
                conditions.append(cond)
                labels.append(case["name"])

        x = np.arange(len(conditions))
        ax.bar(
            x,
            np.log10(
                np.maximum(
                    conditions,
                    1.0,
                )
            ),
        )
        ax.set_xticks(
            x,
            labels,
            rotation=30,
            ha="right",
        )
        ax.set_ylabel(
            r"$\log_{10}$ observed-Hessian condition number"
        )
        ax.set_title("B  observed-curvature conditioning")
        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
