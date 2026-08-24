# Examples

The examples deliberately reuse the same three model families throughout.

1. `models.py`
   - NumPy and JAX model specification.
   - Binary RW + softmax, categorical RW + softmax, CES continuous output.

2. `01_individual_fit.py`
   - `verbose`, `display`, optimizer configuration, Hessian backend.
   - Main outputs and diagnostics.

3. `02_parameter_recovery.py`
   - Parameter recovery from noisy synthetic data under weak priors.

4. `03_bms.py`
   - Random-effects Bayesian model selection from individual log evidences.

5. `04_hbi.py`
   - Hierarchical Bayesian inference using individual MAP files as initialization.

Run examples from the repository root, e.g.

```bash
python examples/01_individual_fit.py
```
