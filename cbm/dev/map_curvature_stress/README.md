# MAP curvature stress experiment

Location:

    cbm/dev/map_curvature_stress/

This experiment maps where the original CBM evidence-Hessian estimator becomes
inaccurate or non-positive-definite across generative parameter space.

## Models and grids

Binary RW + softmax:

    alpha = [0.05, 0.25, 0.50, 0.75, 0.95]
    beta  = [0.25, 0.50, 1.0, 3.0, 8.0]

Categorical RW + softmax:

    alpha = [0.05, 0.25, 0.50, 0.75, 0.95]
    beta  = [0.25, 0.50, 1.0, 3.0, 8.0]

CES + Gaussian continuous output:

    alpha = [0.05, 0.25, 0.50, 0.75, 0.95]
    rho   = [-0.70, -0.30, -0.10, 0.30, 0.70]

The CES observation model is:

    y ~ Normal(V_CES, sigma^2)

Default sigma is 0.1.

## Pipeline

For every grid cell and replicate:

    synthetic data
        ->
    multi-start L-BFGS-B
        ->
    GN polish
        ->
    freeze MAP
        ->
    original CBM nested forward-FD Hessian
    new central-FD Hessian
    JAX AD Hessian
        ->
    compare PD, conditioning, Hessian error, and curvature-only log evidence

The experiment intentionally does NOT repeat the earlier gradient-validation,
AD-optimization, or finite-difference step-size sweeps.

## Failure definitions

Original numerical failure:

    H_original is non-PD
    H_AD is PD

New central-FD numerical failure:

    H_FD is non-PD
    H_AD is PD

Structural Laplace failure:

    H_AD is non-PD

This distinction is important: an AD non-PD Hessian should not be attributed
to a finite-difference numerical method.

## Quick test

From the repository root:

    python -m cbm.dev.map_curvature_stress.run_experiment \
        --model binary \
        --n-replicates 2 \
        --n-trials 100 \
        --n-starts 2

Then:

    python -m cbm.dev.map_curvature_stress.analyze_results

## Main experiment

    python -m cbm.dev.map_curvature_stress.run_experiment \
        --model all \
        --n-replicates 20 \
        --n-trials 250 \
        --n-starts 5

Then:

    python -m cbm.dev.map_curvature_stress.analyze_results

Outputs:

    results/map_curvature_stress/stress_raw.csv
    results/map_curvature_stress/stress_grid_summary.csv

The analysis also writes heatmap-ready CSV tables.

## Optional heatmaps

    python -m cbm.dev.map_curvature_stress.plot_heatmaps

Default figures:

- original numerical-failure rate
- original-vs-AD Hessian error
- original-vs-AD curvature-only evidence difference

## If the baseline grid is too easy

Repeat exactly the same parameter grid under less information, rather than
changing several factors at once.

Fewer trials:

    python -m cbm.dev.map_curvature_stress.run_experiment \
        --model all \
        --n-replicates 20 \
        --n-trials 50 \
        --output-dir results/map_curvature_stress_T50

Weaker prior:

    python -m cbm.dev.map_curvature_stress.run_experiment \
        --model all \
        --n-replicates 20 \
        --prior-scale 0.1 \
        --output-dir results/map_curvature_stress_weak_prior

Noisier CES:

    python -m cbm.dev.map_curvature_stress.run_experiment \
        --model ces \
        --n-replicates 20 \
        --sigma 0.3 \
        --output-dir results/map_curvature_stress_ces_sigma03
