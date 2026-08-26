# CBM development validation experiments

These scripts are intentionally separate from `tests/`.

`tests/` asks whether the software has regressed.

`cbm/dev/` asks why the revised numerical design is scientifically useful.

## 01 — MAP optimizer validation

`01_map_optimizer_validation.py`

Uses a nonlinear least-squares Rosenbrock problem to force several genuine
Gauss-Newton polishing steps. It compares the GN result with a high-accuracy
L-BFGS-B reference and displays the log-joint and parameter trajectories.

Main outputs:

- number of accepted GN steps;
- GN convergence status;
- GN condition number;
- final parameter difference;
- final objective difference.

## 02 — Hessian and Laplace failure cases

`02_hessian_failure_cases.py`

Reproduces the original CBM nested-forward finite-difference Hessian exactly
and compares it with the current central-FD estimator.

The posterior contains an increasingly flat quartic direction. It therefore
provides a controlled case where the Hessian remains positive definite while
the Gaussian/Laplace approximation becomes poor.

The exact posterior integral is computed numerically, allowing direct
measurement of log-evidence error.

The central question is whether PD alone is sufficient to accept Laplace
evidence. The current toolbox additionally diagnoses conditioning and marks
extreme cases as `laplace_fragile`.

## 03 — Parameter and latent recovery

`03_parameter_and_latent_recovery.py`

Simulates a binary Rescorla-Wagner model with known subject parameters and
known trialwise Q values/prediction errors.

It then fits the subjects with `individual_fit(..., evolution=...)` and reports:

- parameter correlation, bias, RMSE;
- pooled Q-value correlation, bias, RMSE;
- prediction-error correlation, bias, RMSE;
- number of valid Laplace fits.

The figure shows parameter recovery and true-versus-reconstructed latent
trajectories.

## 04 — Failure and diagnostic summary

`04_failure_and_diagnostics_summary.py`

Deliberately constructs several pathological fits:

- healthy reference;
- PD but ill-conditioned observed Hessian;
- L-BFGS-B iteration limit;
- invalid objective evaluations;
- hard-bound MAP;
- ill-conditioned GN curvature.

It prints a compact table of flags and diagnostic fields and lists the warnings
that the modeller would receive.
