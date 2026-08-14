# Examples

This folder contains runnable examples for the CBM Python package.
- `example_individual_fit.py`: Demonstrates fitting three linear models seperately for each individual.
- `example_model_selection.py`: Demonstrates fitting three linear models seperately for each individual and performing Bayesian model selection.
- `example.py`: Demonstrates fitting three linear models, Bayesian model selection (BMS), and group-level inference using HBI.
- `example_RL.py`: Demonstrates fitting two RL models and performing group-level inference using HBI.
- `example_group_bms.py`: Group-level BMS between families, conditions and groups.
- `example_display.py`: **Start here for the diagnostic plots.** Minimal
  example of `config=dict(display=True)` and `fit.plot()` on a straight line —
  the model is trivial so the figures are the subject.
- `example_regression.py`: **Classical `y = f(X | theta)` models with continuous
  outcomes** — how to supply `model_trials` so the fit uses the Gauss-Newton
  curvature, and what changes when sigma is estimated rather than profiled out.
  Start here if your models are regression-style rather than choice-based.
  Also demonstrates `config=dict(display=True)` and `fit.plot()` (MOD 14).

Generated artifacts (pickle files, logs) are written to `examples/output/`.
