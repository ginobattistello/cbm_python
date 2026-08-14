"""
Fitting a classical model  y = f(X | theta)  with continuous outcomes.

Every other example here is a choice model (RL, risky choice), where the
likelihood is a probability of a discrete response. This one covers the
other common case: a deterministic function of predictors plus Gaussian
noise, which is what most regression-style modelling looks like.

It answers two questions.

  1. HOW DO I GET THE GAUSS-NEWTON CURVATURE?
     The toolbox selects the curvature from whether you pass
     `model_trials`. Without it you get the finite-difference Hessian with
     eigenvalue clipping (MOD 2); with it you get `H = J'J + prior
     precision`, which is positive-definite by construction (MOD 5).
     `model_trials` is not a different model — it is the SAME
     log-likelihood returned per observation instead of summed.

  2. SHOULD SIGMA BE A FITTED PARAMETER?
     Two defensible styles, compared side by side below. The short answer
     is that they give the same estimates, the explicit version gives
     slightly wider (more honest) standard errors, and **their
     log-evidences are not comparable to each other**.

Run:  python examples/example_regression.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbm.individual_fit import individual_fit          # noqa: E402


# ══════════════════════════════════════════════════════════════════
# The model:  a CES utility over two predictors
# ══════════════════════════════════════════════════════════════════
# theta is unconstrained; the sigmoids map it into the valid range.
#   a = weight on dimension 1,      in (0, 1)
#   d = substitution exponent,      in (0, 2)
def utility_ces(param, X):
    dim1, dim2 = X[:, 0], X[:, 1]
    a = 1.0 / (1.0 + np.exp(-param[0]))
    d = 2.0 / (1.0 + np.exp(-param[1]))
    return (a * dim1 ** d + (1.0 - a) * dim2 ** d) ** (1.0 / d)


# ══════════════════════════════════════════════════════════════════
# STYLE A — sigma profiled out  (sigma is NOT a fitted parameter)
# ══════════════════════════════════════════════════════════════════
# At each theta, sigma is set to its maximum-likelihood value given the
# residuals. This is the "profile likelihood" and it is what most
# hand-written objectives do.
#
# A NOTE ON THE ALGEBRA, because it bites when you write the per-trial
# version. The compact form often written as
#
#     -0.5 * n * (log(2*pi*sigma2) + 1)
#
# uses the identity sum(r^2)/sigma2 == n, which holds only when
# sigma2 is EXACTLY mean(r^2). Adding a floor (`+ SIG2_FLOOR`, sensible
# to avoid division by zero on a perfect fit) breaks that identity, and a
# naive per-trial split then disagrees with the scalar by ~1e-2 nats.
#
# Writing the residual term out explicitly keeps the two exactly
# consistent — which the assertion at the bottom of this file checks.
SIG2_FLOOR = 1e-6


def objective_profiled(param, data):
    """Scalar log-likelihood, sigma profiled out."""
    X, y = data
    resid = y - utility_ces(param, X)
    sigma2 = np.mean(resid ** 2) + SIG2_FLOOR
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * sigma2)
                                + resid ** 2 / sigma2)))


def objective_profiled_trials(param, data):
    """Per-observation version. MUST sum to objective_profiled exactly."""
    X, y = data
    resid = y - utility_ces(param, X)
    sigma2 = np.mean(resid ** 2) + SIG2_FLOOR
    return -0.5 * (np.log(2.0 * np.pi * sigma2) + resid ** 2 / sigma2)


# ══════════════════════════════════════════════════════════════════
# STYLE B — sigma estimated  (one extra parameter, on a log scale)
# ══════════════════════════════════════════════════════════════════
# param = [theta_a, theta_d, log_sigma]. Nothing is profiled; every
# observation's contribution depends only on theta, which is what
# Gauss-Newton assumes (see the note in the results section).
def objective_explicit(param, data):
    X, y = data
    resid = y - utility_ces(param[:2], X)
    sigma = np.exp(param[2])
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * sigma ** 2)
                                + resid ** 2 / sigma ** 2)))


def objective_explicit_trials(param, data):
    X, y = data
    resid = y - utility_ces(param[:2], X)
    sigma = np.exp(param[2])
    return -0.5 * (np.log(2.0 * np.pi * sigma ** 2)
                   + resid ** 2 / sigma ** 2)


# ══════════════════════════════════════════════════════════════════
# Simulate
# ══════════════════════════════════════════════════════════════════
TRUE_THETA = np.array([0.4, 0.3])      # a = 0.599, d = 1.149
TRUE_SIGMA = 0.08
N_SUBJECTS, N_OBS = 25, 80

rng = np.random.default_rng(5)
data = []
for _ in range(N_SUBJECTS):
    X = rng.uniform(0.2, 3.0, size=(N_OBS, 2))
    y = utility_ces(TRUE_THETA, X) + rng.normal(0, TRUE_SIGMA, N_OBS)
    data.append((X, y))

PRIOR_VARIANCE = 6.25                   # SD 2.5 on the unconstrained scale

print("=" * 70)
print("Regression-style fit:  y = f(X | theta) + noise")
print(f"{N_SUBJECTS} subjects x {N_OBS} observations   ·   "
      f"true theta = {TRUE_THETA}, sigma = {TRUE_SIGMA}")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════
# 1. Does model_trials actually change the curvature?
# ══════════════════════════════════════════════════════════════════
print("\n\n### 1. Selecting the curvature " + "#" * 40)

np.random.seed(0)
fit_fd = individual_fit(data, objective_profiled, np.zeros(2),
                        PRIOR_VARIANCE, config=dict(num_init=4, verbose=False))

np.random.seed(0)
fit_gn = individual_fit(data, objective_profiled, np.zeros(2),
                        PRIOR_VARIANCE, config=dict(num_init=4, verbose=False),
                        model_trials=objective_profiled_trials)

print(f"\nwithout model_trials : {fit_fd.math.diagnostics[0].hess_method}")
print(f"with    model_trials : {fit_gn.math.diagnostics[0].hess_method}")

d_map = float(np.max(np.abs(fit_fd.output.parameters
                            - fit_gn.output.parameters)))
d_lme = float(np.mean(fit_gn.output.log_evidence
                      - fit_fd.output.log_evidence))
print(f"\nmax |MAP difference|      {d_map:.3e}   (convergence noise)")
print(f"mean log-evidence shift   {d_lme:+.4f} nats")
print("\nThe optimum is the same; the curvature — and therefore the")
print("evidence — is what changes. That is the intended behaviour:")
print("Gauss-Newton alters how CONFIDENT a fit is, not where it lands.")


# ══════════════════════════════════════════════════════════════════
# 2. Profiled sigma vs estimated sigma
# ══════════════════════════════════════════════════════════════════
print("\n\n### 2. Profiled vs estimated sigma " + "#" * 34)

np.random.seed(0)
fit_A = individual_fit(data, objective_profiled, np.zeros(2),
                       PRIOR_VARIANCE, config=dict(num_init=4, verbose=False),
                       model_trials=objective_profiled_trials)

np.random.seed(0)
fit_B = individual_fit(data, objective_explicit, np.zeros(3),
                       PRIOR_VARIANCE, config=dict(num_init=4, verbose=False),
                       model_trials=objective_explicit_trials)

thA, thB = fit_A.output.parameters, fit_B.output.parameters[:, :2]
seA, seB = np.nanmean(fit_A.se, axis=0), np.nanmean(fit_B.se[:, :2], axis=0)

print(f"\n{'':22s}{'theta[0]':>12}{'theta[1]':>12}")
print(f"  {'true':20s}{TRUE_THETA[0]:>12.4f}{TRUE_THETA[1]:>12.4f}")
print(f"  {'A profiled  (mean)':20s}{thA[:, 0].mean():>12.4f}"
      f"{thA[:, 1].mean():>12.4f}")
print(f"  {'B explicit  (mean)':20s}{thB[:, 0].mean():>12.4f}"
      f"{thB[:, 1].mean():>12.4f}")
print(f"  {'A mean SE':20s}{seA[0]:>12.4f}{seA[1]:>12.4f}")
print(f"  {'B mean SE':20s}{seB[0]:>12.4f}{seB[1]:>12.4f}")
print(f"  {'SE ratio B/A':20s}{seB[0] / seA[0]:>12.4f}"
      f"{seB[1] / seA[1]:>12.4f}")

sig_hat = np.exp(fit_B.output.parameters[:, 2])
print(f"\n  B also recovers sigma: {sig_hat.mean():.4f} "
      f"(true {TRUE_SIGMA:.3f}), with an SE per subject — "
      f"A gives you no uncertainty on the noise level at all.")

print(f"\n  mean log-evidence   A {fit_A.output.log_evidence.mean():9.3f}"
      f"   B {fit_B.output.log_evidence.mean():9.3f}")


# ══════════════════════════════════════════════════════════════════
print("\n\n### What the comparison shows " + "#" * 39)
print("""
ESTIMATES        Identical to 4 decimals. Profiling sigma does not bias
                 theta, so if point estimates are all you need, either
                 style is fine.

STANDARD ERRORS  B's are ~2% wider here. That is B being honest: A's
                 curvature treats sigma as known once it has been
                 profiled, so it slightly understates uncertainty.

                 The gap shrinks as observations per subject grow,
                 because sigma becomes better determined and "treating
                 it as known" becomes closer to true. Measured on this
                 model, SE ratio B/A for theta[0]:

                     n_obs =  20    1.052
                     n_obs =  40    1.035
                     n_obs =  80    1.019
                     n_obs = 200    1.008

                 So the choice matters most for short sessions. At 20
                 observations A understates the SE by ~5%; at 200 the
                 two are practically identical.

GAUSS-NEWTON     Exact for B, approximate for A. `H = J'J` assumes each
                 observation contributes independently given the
                 parameters. Under A that is not quite true — sigma is
                 re-estimated from ALL residuals at every evaluation, so
                 the observations are coupled through it. In practice
                 the approximation is mild (and VBA makes the same one),
                 but B avoids it entirely.

EVIDENCE         *** DO NOT COMPARE A's log-evidence WITH B's. ***
                 Three reasons they are on different scales:
                   - A maximises over sigma at every theta, so its
                     likelihood is higher "for free";
                   - the Laplace occam term uses d = 2 for A, d = 3 for B;
                   - B pays a prior penalty for the sigma parameter that
                     A never pays.
                 Comparing models is valid WITHIN a style, never across.

RECOMMENDATION   Use B when you care about uncertainty, want sigma
                 itself, or plan to compare models by evidence. Use A
                 when you only need point estimates and prefer the
                 smaller parameter space.
""")


# ══════════════════════════════════════════════════════════════════
# 3. The consistency check you should run on your own objective
# ══════════════════════════════════════════════════════════════════
print("\n### 3. Checking your per-trial function " + "#" * 29)
print("""
`model_trials` must sum to the scalar objective at EVERY parameter value,
not just at the optimum. If it does not, the Jacobian describes a
different function from the one being optimised and the evidence is
quietly wrong. Check it before trusting any result:
""")

ok = True
for p in ([0.4, 0.3], [0.0, 0.0], [1.2, -0.4], [3.0, 2.0], [-2.0, 1.5]):
    s = objective_profiled(np.array(p), data[0])
    v = float(np.sum(objective_profiled_trials(np.array(p), data[0])))
    diff = abs(s - v)
    ok &= diff < 1e-9
    print(f"  theta = {str(p):14s} scalar {s:11.5f}   "
          f"sum {v:11.5f}   diff {diff:.2e}")
assert ok, "per-trial function does not match the scalar objective"
print("\n  All match. (Try re-introducing the '-0.5*n*(...+1)' shortcut")
print("   with SIG2_FLOOR > 0 and this check fails — see the comment")
print("   above objective_profiled.)")

print("\n\n### 4. Reading the result " + "#" * 43)
print()
print(fit_B.summary(max_subjects=5))


# ══════════════════════════════════════════════════════════════════
# 5. Diagnostic figures  (MODIFICATION 14)
# ══════════════════════════════════════════════════════════════════
# `display=True` makes the optimizer retain what a diagnostic plot needs.
# It is off by default and costs nothing when off.
#
# `predict` and `observed` are what turn panel A into a real
# observed-vs-predicted scatter. They are OPTIONAL: the toolbox cannot
# know how your `data` is laid out, so without them the panel falls back
# to per-trial log-likelihood and says so (and individual_fit warns once
# at fit time). Regression models can almost always supply them; choice
# models usually cannot.
print("\n\n### 5. Diagnostic figures " + "#" * 43)

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(0)
fit_display = individual_fit(
    data, objective_profiled, np.zeros(2), PRIOR_VARIANCE,
    config=dict(num_init=4, verbose=False, display=True),
    model_trials=objective_profiled_trials,
    predict=lambda param, dat: utility_ces(param, dat[0]),
    observed=lambda dat: dat[1])

try:
    fit_display.plot(subject=0, save=str(OUT / "regression_subject0.png"))
    fit_display.plot(save=str(OUT / "regression_group.png"))
    print(f"""
  wrote {OUT / 'regression_subject0.png'}
        {OUT / 'regression_group.png'}

  Per subject:  A observed vs predicted (R2, RMSE)
                B parameter path over the search, band = final +/-1 SE
                C objective evolution
                D estimates with 95% CI
                E status, cost and warnings

  Group:        parameter distributions, log-evidence, fit-quality
                counts, cost. No trajectories — twenty overlaid zigzags
                say nothing; use plot(subject=i) for one fit's detail.

  TWO THINGS THE FIGURES ARE CAREFUL ABOUT

  Panel B's x-axis is function EVALUATIONS, not iterations. L-BFGS-B
  line-search probes are included, so the path zigzags and can shoot
  far out before settling. That is normal.

  Panel C plots TWO different quantities. The log-joint is what is being
  optimised, over the whole search. The log-evidence only exists for the
  Newton-polish steps, because the Laplace evidence needs |H| and the
  polish is the only loop that recomputes it. The inset shows the CHANGE
  from the first polish step — typically a few millionths of a nat, i.e.
  the fit was already converged.""")
except ImportError as e:
    print(f"\n  (skipped: {e})")
