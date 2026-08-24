"""
Diagnostic plots in ~20 lines  (MODIFICATION 14).

The shortest useful example of `config=dict(display=True)` and
`fit.plot()`. A straight line with two parameters and Gaussian noise —
nothing interesting about the model, so the figures are the subject.

`example_regression.py` shows display inside a wider discussion (profiled
vs estimated sigma, Gauss-Newton curvature). This file is display alone.

Run:  python examples/example_display.py           # matplotlib windows
      python examples/example_display.py --html    # browser instead
      python examples/example_display.py --no-show # files only

Needs matplotlib, which is an optional extra:  pip install matplotlib
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbm.individual_fit import individual_fit          # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1. A model, and the two functions the toolbox wants
# ══════════════════════════════════════════════════════════════════
#   y = slope * x + intercept + noise
#
# theta is unconstrained here (a slope and an intercept are both free on
# the whole real line), so no sigmoid/exp mapping is needed. Most real
# models do need one — see example_regression.py.
def line(param, X):
    slope, intercept = param
    return slope * X + intercept


SIG2_FLOOR = 1e-6


def objective(param, data):
    """Log-likelihood, summed. This is what gets optimised."""
    X, y = data
    resid = y - line(param, X)
    sigma2 = np.mean(resid ** 2) + SIG2_FLOOR
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * sigma2)
                                + resid ** 2 / sigma2)))


def objective_trials(param, data):
    """The SAME thing, per observation instead of summed.

    Two jobs: it selects the Gauss-Newton curvature, and it gives the
    plot a per-trial panel. It must sum to `objective` exactly — asserted
    at the bottom of this file.
    """
    X, y = data
    resid = y - line(param, X)
    sigma2 = np.mean(resid ** 2) + SIG2_FLOOR
    return -0.5 * (np.log(2.0 * np.pi * sigma2) + resid ** 2 / sigma2)


# ══════════════════════════════════════════════════════════════════
# 2. Simulate 8 subjects
# ══════════════════════════════════════════════════════════════════
TRUE = np.array([2.0, -1.0])          # slope, intercept
rng = np.random.default_rng(7)
data = [(x, line(TRUE, x) + rng.normal(0, 0.5, 60))
        for x in (rng.uniform(-3, 3, 60) for _ in range(8))]

print("=" * 68)
print("Display example ·  y = 2x - 1 + noise  ·  8 subjects x 60 obs")
print("=" * 68)


# ══════════════════════════════════════════════════════════════════
# 3. Fit with display on
# ══════════════════════════════════════════════════════════════════
# display=True   makes the optimizer keep its search path, the Newton
#                polish trace, and any warnings raised per subject.
#                Off by default; costs nothing when off.
#
# predict/observed  optional, and only used by the plot. They let panel A
#                be a real observed-vs-predicted scatter. The toolbox
#                cannot infer them: `data` here is (x, y), but for an RL
#                model it would be (choices, rewards), so it has no way
#                to know which part is the outcome. Omit them and the
#                panel falls back to per-trial log-likelihood — try it,
#                the toolbox warns and tells you what you are missing.
np.random.seed(0)
fit = individual_fit(
    data, objective,
    prior_mean=np.zeros(2),
    prior_variance=25.0,                      # SD 5, weak
    config=dict(num_init=4, verbose=False, display=True),
    model_trials=objective_trials,
    predict=lambda param, dat: line(param, dat[0]),
    observed=lambda dat: dat[1],
)

print(f"\ntrue      slope {TRUE[0]:+.3f}   intercept {TRUE[1]:+.3f}")
print(f"recovered slope {fit.output.parameters[:, 0].mean():+.3f}   "
      f"intercept {fit.output.parameters[:, 1].mean():+.3f}"
      f"   (mean of 8 subjects)")


# ══════════════════════════════════════════════════════════════════
# 4. Plot
# ══════════════════════════════════════════════════════════════════
# plot() dispatches on how many subjects were fitted:
#   many  -> the population view (distributions, evidence, quality, cost)
#   one   -> the per-subject view, so a single fit never shows a
#            "distribution" of one point
# plot(subject=i) forces the per-subject view for any i.
#
# show=True opens a window; save=path writes a file. You can do both.
#
# ON WINDOWS NOT APPEARING. A figure only pops up if matplotlib is using
# an interactive backend AND something calls plt.show() — `show=True` is
# what calls it. Running with `python example_display.py` on macOS or
# Windows normally just works. If nothing appears:
#   * check the backend:  python -c "import matplotlib; print(matplotlib.get_backend())"
#     "Agg" means non-interactive (headless/SSH/CI) — no window is possible.
#     On Linux install a GUI toolkit, e.g. `pip install PyQt5`.
#   * in Jupyter, use `%matplotlib inline` (or `widget`) and just call
#     `fit.plot(subject=0)` — the notebook renders the returned figure and
#     `show=True` is unnecessary.
#   * this script forces nothing, so your own matplotlib config applies.
#
# BACKEND-AGNOSTIC ALTERNATIVE.  `backend="html"` sidesteps all of the
# above: the same figure is embedded in a self-contained HTML page and
# opened in your browser. No GUI toolkit, no backend question, works over
# SSH and on headless machines. Run this script with --html to try it.
#
#     fit.plot(subject=0, backend="html")
#
# --no-show skips the windows (used by the regression suite, which runs
# headless and must not block on a GUI).
SHOW = "--no-show" not in sys.argv and "--html" not in sys.argv
HTML = "--html" in sys.argv

print("\n" + "-" * 68)
try:
    if HTML:
        fit.plot(subject=0, backend="html",
                 html_path=str(OUT / "display_subject0.html"))
        fit.plot(backend="html",
                 html_path=str(OUT / "display_group.html"))
    fit.plot(subject=0, save=str(OUT / "display_subject0.png"), show=SHOW)
    fit.plot(save=str(OUT / "display_group.png"), show=SHOW)
except ImportError as e:
    print(f"skipped: {e}")
    sys.exit(0)

print(f"""wrote  {OUT / 'display_subject0.png'}
       {OUT / 'display_group.png'}

PER SUBJECT
  A  observed vs predicted, identity line, R2 and RMSE
  B  the parameter path, with the final +/-1 SE as a band
  C  objective evolution (see the note below)
  D  final estimates with 95% CI
  E  status, cost and any warnings

GROUP
  parameter distributions, log-evidence, fit-quality counts, cost.
  No trajectories: eight overlaid zigzags say nothing useful.

TWO THINGS WORTH KNOWING BEFORE YOU READ PANEL B AND C

  Panel B's x-axis is function EVALUATIONS, not iterations. It includes
  L-BFGS-B's line-search probes, so the path zigzags and may shoot far
  out before settling. That is the optimizer working normally.

  Panel C plots two DIFFERENT quantities, deliberately not joined into
  one line. The log-joint is what is being optimised, over the whole
  search. The log-evidence appears only for the Newton-polish steps,
  because the Laplace evidence needs |H| and the polish is the only loop
  that recomputes it — during the search there is no Hessian at all. The
  inset shows the CHANGE from the first polish step, usually a few
  millionths of a nat, meaning the fit had already converged.""")


# ══════════════════════════════════════════════════════════════════
# 5. Two checks worth copying into your own scripts
# ══════════════════════════════════════════════════════════════════
print("\n" + "-" * 68)

# (a) model_trials must sum to the scalar objective at EVERY parameter
#     value, not just at the optimum. If it does not, the Jacobian
#     describes a different function from the one being optimised and the
#     evidence is quietly wrong.
worst = 0.0
for p in ([2.0, -1.0], [0.0, 0.0], [-1.5, 4.0], [10.0, -10.0]):
    p = np.array(p, dtype=float)
    worst = max(worst, abs(objective(p, data[0])
                           - float(np.sum(objective_trials(p, data[0])))))
assert worst < 1e-9, f"per-trial mismatch: {worst:.2e}"
print(f"per-trial function matches the scalar objective   "
      f"(max diff {worst:.1e})")

# (b) confirm the curvature is the Gauss-Newton one, not the fallback.
print(f"curvature in use: {fit.math.diagnostics[0].hess_method}")

# The fit itself is unchanged by display — it only decides what is kept.
np.random.seed(0)
plain = individual_fit(data, objective, np.zeros(2), 25.0,
                       config=dict(num_init=4, verbose=False),
                       model_trials=objective_trials)
delta = float(np.max(np.abs(plain.output.parameters
                            - fit.output.parameters)))
print(f"display=True vs display=False, max parameter difference: {delta:.1e}")

print("\n" + "-" * 68)
print(fit.summary(max_subjects=4))


# ══════════════════════════════════════════════════════════════════
# 6. Keep the windows open
# ══════════════════════════════════════════════════════════════════
# plot(show=True) draws non-blocking, so both figures appear at once
# instead of the second waiting for you to close the first. The cost is
# that a plain script would exit immediately and take the windows with
# it — hence this final blocking call.
#
# Not needed in Jupyter, or in `python -i`, or in an IDE that keeps the
# interpreter alive.
if SHOW:
    import matplotlib.pyplot as plt
    print("\nclose the figure windows to exit…")
    plt.show()
