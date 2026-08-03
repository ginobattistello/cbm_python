"""
Verification harness: DEV.md §3 remaining items + §4 three-layer checks
(optimization.py Mods 1, 7, 8, 9 and individual_fit's pre-flight layer,
all 2026-08-03).

Layers covered
--------------
Pre-flight   Config bounds validation (Mod 1: expansion, shapes,
             range ⊂ hard) and individual_fit._preflight_checks
             (empty data, dims, PD prior, model crash/non-finite).
Per-iter     Monotonicity invariant (Mod 7) at the optimize() boundary.
             (The polish-internal guard is a tripwire for future edits:
             f_current ≤ f_entry is airtight by induction through the
             acceptance test, so it cannot be triggered from outside.)
Post-fit     Mod 9 diagnostics: raw min eigenvalue, clip count,
             cross-init agreement, hard-bound mask + warning, and the
             forwarding chain optimize_map → FitMath.diagnostics.
Sign/naming  Mod 8: neg_log_post/F properties, log_posterior sign.

Run:  python cbm/dev/checks_verify.py   (exit 0 = all pass)
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cbm.optimization import (  # noqa: E402
    BFGSOptimizer, Config, ConvergenceStatus, PostFitDiagnostics,
)
from cbm.individual_fit import individual_fit  # noqa: E402
from cbm.map_estimation import log_posterior, optimize_map  # noqa: E402

FAILURES = []


def check(label, ok):
    print(f"  {'PASS' if ok else 'FAIL':4s}  {label}")
    if not ok:
        FAILURES.append(label)


def expect_raise(label, fn, exc=ValueError):
    try:
        fn()
        check(label + " [did not raise]", False)
    except exc:
        check(label, True)
    except Exception as e:  # wrong exception type is also a failure
        check(label + f" [raised {type(e).__name__} instead]", False)


# ---------------------------------------------------------------------------
print("=" * 70)
print("1. MOD 1 — Config bounds validation (pre-flight, bounds part)")
print("=" * 70)
c = Config(d=3)
check("scalar range_bounds expanded to 2×d", c.range_bounds.shape == (2, 3))
check("scalar hard_bounds expanded to 2×d", c.hard_bounds.shape == (2, 3))
check("defaults ±5 / ±100 preserved",
      bool(np.all(c.range_bounds[1] == 5) and np.all(c.hard_bounds[1] == 100)))
expect_raise("wrong range_bounds shape raises",
             lambda: Config(d=3, range_bounds=np.zeros((3, 2))))
expect_raise("wrong hard_bounds shape raises",
             lambda: Config(d=3, hard_bounds=np.zeros((3, 2))))
expect_raise("range_bounds ⊄ hard_bounds raises",
             lambda: Config(d=2, range_bounds=200))
check("list bounds coerced to ndarray",
      isinstance(Config(d=2, range_bounds=[[-1, -1], [1, 1]]).range_bounds,
                 np.ndarray))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Pre-flight layer — individual_fit._preflight_checks")
print("=" * 70)


def linmodel(theta, data):
    X, y = data
    return float(-0.5 * np.sum((y - (X * theta[0] + theta[1])) ** 2))


rng = np.random.default_rng(1)
X = np.linspace(0, 1, 30)
DATA = [(X, 2 * X + 1 + 0.1 * rng.standard_normal(30)) for _ in range(3)]
CFG = {"num_init": 1, "verbose": False}

expect_raise("empty data raises",
             lambda: individual_fit([], linmodel, np.zeros(2), 10.0, config=CFG))
expect_raise("non-PD prior covariance raises",
             lambda: individual_fit(DATA, linmodel, np.zeros(2),
                                    np.array([[-1., 0.], [0., 1.]]), config=CFG))
expect_raise("model crashing on subject 2 raises (named)",
             lambda: individual_fit([DATA[0], "garbage", DATA[2]], linmodel,
                                    np.zeros(2), 10.0, config=CFG))


def nf_model(theta, data):
    return -np.inf if np.abs(theta).sum() < 1e-12 else linmodel(theta, data)


np.random.seed(0)
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    fit_nf = individual_fit(DATA, nf_model, np.zeros(2), 10.0,
                            config={"num_init": 2, "verbose": False})
check("non-finite at prior mean warns (does not stop)",
      any("non-finite" in str(w.message) for w in rec))
check("...and the fit still succeeds",
      bool(np.all(np.isfinite(fit_nf.output.log_evidence))))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. MOD 7 — monotonicity invariant (per-iteration layer)")
print("=" * 70)


def quad(x):
    return float(np.sum(x ** 2))


np.random.seed(0)
opt = BFGSOptimizer(2, config=Config(d=2, num_init=1, verbose=False))
r = opt.optimize(quad, x_init=np.array([2., 2.]))
check("normal path passes the invariant", r.f <= quad(np.array([2., 2.])))

np.random.seed(0)
opt = BFGSOptimizer(2, config=Config(d=2, num_init=1, verbose=False))
opt._newton_polish = lambda *a, **k: (np.array([2., 2.]), 1e9,
                                      ConvergenceStatus.CONVERGED_DF)
expect_raise("polish worsening the objective raises RuntimeError",
             lambda: opt.optimize(quad, x_init=np.array([2., 2.])),
             exc=RuntimeError)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. MOD 8 — sign/naming coherence")
print("=" * 70)
np.random.seed(0)
opt = BFGSOptimizer(2, config=Config(d=2, num_init=1, verbose=False))
r = opt.optimize(lambda x: float(np.sum((x - 1) ** 2)) + 3.0,
                 x_init=np.array([0., 0.]))
check("neg_log_post aliases f", r.neg_log_post == r.f)
check("F = −neg_log_post", r.F == -r.f)

# log_posterior returns the POSITIVE log joint (docstring fixed, code right)
lp = log_posterior(np.zeros(2), linmodel, DATA[0], np.zeros(2), np.eye(2) / 10)
by_hand = (linmodel(np.zeros(2), DATA[0])
           - 2 / 2 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(np.eye(2) / 10)))
check("log_posterior = loglik + logprior (positive convention)",
      abs(lp - by_hand) < 1e-12)

# optimize_map's loglik return equals OptimizationResult.F
np.random.seed(0)
ll, _, _, _, _, res = optimize_map(DATA[0], linmodel, Config(d=2, **CFG),
                                   np.zeros(2), np.eye(2) / 10)
check("optimize_map loglik == result.F (log joint at MAP)",
      abs(ll - res.F) < 1e-9)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. MOD 9 — post-fit diagnostics layer")
print("=" * 70)

# Flat direction -> Mod 2 clip fires and is reported
np.random.seed(0)
opt = BFGSOptimizer(2, config=Config(d=2, num_init=3, verbose=False))
r = opt.optimize(lambda x: float(x[0] ** 2), x_init=np.array([1., 1.]))
check("flat direction: n_clipped >= 1 reported", r.hess_n_clipped >= 1)
check("flat direction: raw_min_eig below the 1e-4 floor",
      r.hess_raw_min_eig < 1e-4)
check("cross-init agreement counted (all runs found same optimum)",
      r.n_inits_agreeing == r.n_runs == 3)

# GN path: PD by construction, nothing clipped
np.random.seed(0)
opt = BFGSOptimizer(2, config=Config(d=2, num_init=1, verbose=False))
r = opt.optimize(quad, x_init=np.array([1., 1.]),
                 trial_func=lambda x: x ** 2, prior_precision=np.eye(2))
check("GN path: n_clipped == 0 by construction", r.hess_n_clipped == 0)
check("GN path: raw_min_eig >= prior precision floor", r.hess_raw_min_eig >= 1.0 - 1e-6)

# Railed at hard bound -> mask + warning
np.random.seed(0)
opt = BFGSOptimizer(1, config=Config(d=1, num_init=1, verbose=False))
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    r = opt.optimize(lambda x: float((x[0] - 500.) ** 2), x_init=np.array([0.]))
check("railed parameter flagged in at_hard_bounds", bool(r.at_hard_bounds[0]))
check("railed parameter emits Laplace-validity warning",
      any("hard_bounds" in str(w.message) for w in rec))

# Forwarding chain: individual_fit -> FitMath.diagnostics
np.random.seed(0)
fit = individual_fit(DATA, linmodel, np.zeros(2), 10.0,
                     config={"num_init": 2, "verbose": False})
d0 = fit.math.diagnostics[0]
check("FitMath.diagnostics populated per subject",
      len(fit.math.diagnostics) == len(DATA))
check("diagnostics are PostFitDiagnostics dataclasses",
      isinstance(d0, PostFitDiagnostics))
check("convergence status forwarded (Mod 6 -> §4)",
      d0.convergence_status in {s.value for s in ConvergenceStatus})
check("hess_method forwarded", d0.hess_method == "finite_diff_clipped")
check("at_hard_bounds forwarded", d0.at_hard_bounds == [False, False])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("All checks passed.")
