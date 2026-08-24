"""
Verification harness: convergence status enum (optimization.py Mod 6).

Forces each of the four ConvergenceStatus exit paths of `_newton_polish`
and checks (a) the right status is reported, (b) FLAG_FROM_STATUS maps it
to the agreed flag, (c) `optimize()` surfaces the status on the result.

Exit paths and how they are forced
----------------------------------
CONVERGED_DF      quadratic, polish started near the minimum: the first
                  Newton step improves f by < tol_df relative -> DF exit.
NO_IMPROVEMENT    quadratic, polish started AT the minimum: no halved
                  step can reduce f -> backtracking exhausted.
MAX_STEPS         quadratic, polish started far away with n_steps=1: the
                  single step improves by >> tol_df, loop ends -> cap.
SINGULAR_HESSIAN  compute_hessian monkeypatched to return the zero
                  matrix -> np.linalg.solve raises LinAlgError.

Run:  python cbm/dev/convergence_status_verify.py   (exit 0 = all pass)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cbm.optimization import (  # noqa: E402
    BFGSOptimizer, Config, ConvergenceStatus, FLAG_FROM_STATUS,
)

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f"  {'PASS' if ok else 'FAIL':4s}  {label:55s} {got!r}")
    if not ok:
        FAILURES.append(f"{label}: expected {expected!r}, got {got!r}")


def make_opt():
    return BFGSOptimizer(2, config=Config(d=2, num_init=1, verbose=False))


def quadratic(x):
    return float(np.sum(x ** 2))


print("=" * 70)
print("1. Enum/table integrity")
print("=" * 70)
check("enum has exactly 4 members", len(ConvergenceStatus), 4)
check("FLAG_FROM_STATUS covers every member",
      set(FLAG_FROM_STATUS) == set(ConvergenceStatus), True)
check("str-enum equality (serialization contract)",
      ConvergenceStatus.CONVERGED_DF == "converged_df", True)
# The trap Mod 6 exists to avoid: every member is truthy, so a
# truthiness test can never distinguish them.
check("all members truthy (why `if status:` must never be used)",
      all(bool(s) for s in ConvergenceStatus), True)

print("\n" + "=" * 70)
print("2. _newton_polish exit paths")
print("=" * 70)

# CONVERGED_DF — near the minimum: step improves, but relative df < tol
opt = make_opt()
_, _, s = opt._newton_polish(quadratic, np.array([1e-3, 1e-3]))
check("near-minimum start -> CONVERGED_DF", s, ConvergenceStatus.CONVERGED_DF)

# NO_IMPROVEMENT — at the minimum: nothing to gain
opt = make_opt()
_, _, s = opt._newton_polish(quadratic, np.array([0.0, 0.0]))
check("at-minimum start -> NO_IMPROVEMENT", s, ConvergenceStatus.NO_IMPROVEMENT)

# MAX_STEPS — one big improving step allowed, then the cap
opt = make_opt()
x, f, s = opt._newton_polish(quadratic, np.array([3.0, 3.0]), n_steps=1)
check("far start, n_steps=1 -> MAX_STEPS", s, ConvergenceStatus.MAX_STEPS)
check("MAX_STEPS still improved f (monotonic descent)", f < quadratic(np.array([3.0, 3.0])), True)

# SINGULAR_HESSIAN — zero curvature makes the solve fail
opt = make_opt()
opt.compute_hessian = lambda *a, **k: np.zeros((2, 2))
_, _, s = opt._newton_polish(quadratic, np.array([1.0, 1.0]))
check("singular H -> SINGULAR_HESSIAN", s, ConvergenceStatus.SINGULAR_HESSIAN)

print("\n" + "=" * 70)
print("3. optimize() surfaces status and derives flag from the table")
print("=" * 70)

np.random.seed(0)
opt = make_opt()
res = opt.optimize(quadratic, x_init=np.array([2.0, 2.0]))
check("result.convergence_status is a ConvergenceStatus",
      isinstance(res.convergence_status, ConvergenceStatus), True)
check("result.flag == FLAG_FROM_STATUS[status]",
      res.flag, FLAG_FROM_STATUS[res.convergence_status])
check("reachable path keeps flag = 1.0 (baseline invariant)", res.flag, 1.0)

# Forced singular path end-to-end: flag must drop to 0.5 with a warning
import warnings as _w


def _singular_hessian(*a, **k):
    """Fake compute_hessian honoring the Mod 9 return_diagnostics kwarg."""
    H = np.zeros((2, 2))
    if k.get("return_diagnostics"):
        return H, {"raw_min_eig": 0.0, "n_clipped": 2}
    return H


np.random.seed(0)
opt = make_opt()
opt.compute_hessian = _singular_hessian
with _w.catch_warnings(record=True) as rec:
    _w.simplefilter("always")
    res = opt.optimize(quadratic, x_init=np.array([2.0, 2.0]))
check("forced singular -> status", res.convergence_status,
      ConvergenceStatus.SINGULAR_HESSIAN)
check("forced singular -> flag 0.5", res.flag, 0.5)
check("forced singular -> warning emitted",
      any("singular" in str(w.message).lower() for w in rec), True)
check("flag 0.5 != 0, so no prior substitution downstream", res.flag != 0, True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("All checks passed.")
