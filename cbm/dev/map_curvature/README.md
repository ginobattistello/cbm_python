# MAP curvature experiment

This is a deliberately self-contained experiment for the CBM fork.

It tests:

1. the fork's current Gauss-Newton/score-outer-product curvature,
2. a direct finite-difference observed Hessian,
3. an automatic-differentiation observed Hessian,
4. whether AD curvature is useful for the Newton polish.

The experiment does **not** modify the production CBM optimizer and does not
run BMS/HBI/evidence calculations.

## Models

### 1. Binary RW + softmax

Parameters:

- `alpha`: learning rate
- `beta`: inverse temperature

### 2. Three-choice RW + categorical softmax

Same two parameters, but three alternatives.

### 3. CES + continuous Gaussian output

Parameters:

- `alpha`
- `rho`

The CES value is the Gaussian mean:

    V = [alpha*x1^rho + (1-alpha)*x2^rho]^(1/rho)

The observation SD is fixed at 0.10.

## Installation

From the repository root:

    pip install jax scipy numpy

If JAX is already installed, nothing else is required.

## Run

Quick test:

    python -m cbm.dev.map_curvature.run_experiment \
        --model binary \
        --n-datasets 2

Full initial experiment:

    python -m cbm.dev.map_curvature.run_experiment \
        --model all \
        --n-datasets 20 \
        --n-trials 250 \
        --n-starts 5

Results are written to:

    results/map_curvature/

including:

- `map_curvature_results.csv`
- `map_curvature_matrices.npz`

## Interpretation

The main curvature comparison is performed at the same GN-polished MAP:

    H_GN = J.T @ J + prior_precision
    H_AD = autodiff Hessian of the full negative log posterior
    H_FD = finite-difference Hessian of the full negative log posterior

The main metric is the relative Frobenius error:

    ||H_GN - H_AD||_F / ||H_AD||_F

FD is compared against AD to check whether finite differences are stable.

The experiment also compares the objective obtained by:

- L-BFGS-B + GN polish
- L-BFGS-B + AD Hessian polish

These two optimization comparisons should be interpreted separately from the
curvature comparison because they can end at slightly different MAP estimates.

## Important implementation note

The GN calculation reproduces the current fork's curvature construction:

    J = d(per-trial log-likelihood) / d(theta)
    H_GN = J.T @ J + prior_precision

using the fork's one-sided finite-difference step rule.

The finite-difference observed Hessian is deliberately implemented
independently as a central finite difference of the scalar negative
log-posterior.

The AD Hessian differentiates the same mathematical negative log-posterior
using a JAX implementation of each model.


## Second-stage decision analysis

This version adds two diagnostics requested after the first experiment.

### Gradient validation

Relative gradient error is no longer assessed only at the MAP, where the true
gradient is close to zero and relative errors can be misleading.

Instead, a deterministic interior probe point is constructed near the MAP and
finite-difference gradients are compared with AD for:

    h = 1e-3, 1e-4, 1e-5, 1e-6, 1e-7

### Curvature-only Laplace consequence

At the exact same GN-polished MAP, the script computes:

    logE(H) = -objective_MAP
              + d/2 * log(2*pi)
              - 1/2 * logdet(H)

for `H_GN`, `H_FD`, and `H_AD`.

Only differences between these values should be interpreted here. Because the
MAP and objective are held fixed, the GN-vs-AD difference is exactly:

    logE_GN - logE_AD
        = -0.5 * [logdet(H_GN) - logdet(H_AD)]

This isolates the consequence of using the GN curvature in the Laplace term
without involving BMS or HBI.

### Run the decision summary

After running the experiment:

    python -m cbm.dev.map_curvature.analyze_results

or, for a non-default results path:

    python -m cbm.dev.map_curvature.analyze_results \
        --path results/map_curvature/map_curvature_results.csv

The summary is designed to answer:

1. Are the NumPy and JAX models identical?
2. Are finite-difference gradients stable?
3. Does FD reproduce the AD observed Hessian?
4. How different is the current GN curvature from the observed Hessian?
5. Does AD improve the MAP optimization?
6. How much would the curvature choice change the local Laplace term?
