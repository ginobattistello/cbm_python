# CBM (Computational Brain/Behavior Modeling)

CBM provides individual-level computational-model fitting, Bayesian model
selection (BMS), and hierarchical Bayesian inference (HBI).

## Individual fitting

The current MAP pipeline separates optimization from Laplace curvature:

1. multi-start L-BFGS-B searches for the MAP;
2. trialwise likelihood models optionally receive a Gauss-Newton polish;
3. the final observed posterior Hessian is recomputed independently at the MAP
   using central finite differences by default, or JAX autodiff when supplied;
4. the observed Hessian is never clipped for evidence.

The result retains explicit diagnostics for L-BFGS-B termination, GN curvature,
observed-Hessian positive definiteness/conditioning, invalid objective
evaluations, hard-bound solutions, and Laplace validity.

A zero prior variance fixes a parameter exactly at its prior mean.

## Reproducibility

`Config(random_state=...)` controls MAP random initializations without touching
NumPy's global random state.

BMS functions similarly accept `random_state=...` for Monte-Carlo exceedance
probabilities.

## Data/model interface

Each subject uses:

```python
data[n] = {
    "y": observed_outcomes,
    "X": model_inputs,
}
```

`model(theta, data)` may return either a scalar summed log likelihood or a
one-dimensional vector of per-observation contributions. A vector enables GN
polishing automatically.

Optional `observation(theta, data)` supplies prediction diagnostics.
Optional `evolution(theta, data)` supplies deterministic latent trajectories
evaluated once at the final MAP.

## Installation

```bash
git clone https://github.com/payampiray/cbm_python.git
cd cbm_python
pip install -e .
```

For plotting:

```bash
pip install -e ".[display]"
```

For optional autodiff Hessians:

```bash
pip install -e ".[autodiff]"
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Reference

Piray P, Dezfouli A, Heskes T, Frank MJ, Daw ND. Hierarchical Bayesian
inference for concurrent model fitting and comparison for group studies.
PLoS Computational Biology (2019).
