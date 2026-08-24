"""
Verification harness: cbm/group_bms.py (DEV.md §5, workstream 3,
promoted 2026-08-03) + the fe_null family-branch orientation fix in
model_selection.py.

What is pinned here
-------------------
1. fe_null C-orientation fix: family branch works for K != nf (it
   crashed before — dead code until group_bms passed C), and two
   analytic invariants: C = I  =>  F0f == F0m, and equal-sized
   partitions => F0f == F0m (both make f0 uniform = 1/K).
2. Partition validation: gap, overlap, empty family, bad indices,
   wrong names length all raise ValueError (not assert).
3. Family BOR correctness invariant: with each model its own family
   (C = I), the family BOR must equal the model-level BOR exactly.
   The draft's rescaling heuristic fails this badly.
4. Exactness/MC agreement: family_frequency == alpha-aggregation
   exactly; exceedance_prob agrees with the draft's ad-hoc gamma
   sampler within Monte-Carlo tolerance.
5. Directional sanity of both between modes (equal-favoring data =>
   high xp(equal); model-switching data => low), plus the dict shim.

Run:  python cbm/dev/group_bms_verify.py   (exit 0 = all pass)
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cbm.group_bms import (  # noqa: E402
    group_bms, group_bms_btw_conds, group_bms_btw_groups,
    GroupBMSResult, _validate_partition,
)
from cbm.model_selection import bms, fe_null  # noqa: E402

FAILURES = []
NSAMP = 200_000   # MC samples: enough for ~0.002 std on exceedance


def check(label, ok):
    print(f"  {'PASS' if ok else 'FAIL':4s}  {label}")
    if not ok:
        FAILURES.append(label)


def expect_valueerror(label, fn):
    try:
        fn()
        check(label + " [did not raise]", False)
    except ValueError:
        check(label, True)


def synth_L(n_sub, n_mod, favored, delta=3.0, seed=0):
    """Log-evidence with `favored[i]` the best model for subject i."""
    rng = np.random.default_rng(seed)
    L = rng.normal(scale=0.5, size=(n_sub, n_mod))
    for i, k in enumerate(favored):
        L[i, k] += delta
    return L


# ---------------------------------------------------------------------------
print("=" * 70)
print("1. fe_null family branch (orientation fix in model_selection.py)")
print("=" * 70)
rng = np.random.default_rng(0)
Lms = rng.normal(size=(4, 12))          # models × subjects (fe_null layout)

C42 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
try:
    _, F0f = fe_null(Lms, {"families": True, "C": C42})
    check("K != nf no longer crashes (was dead-code bug)", True)
except ValueError:
    check("K != nf no longer crashes (was dead-code bug)", False)
    F0f = np.nan

F0m, _ = fe_null(Lms, {"families": False})
_, F0f_eye = fe_null(Lms, {"families": True, "C": np.eye(4)})
check("C = I  =>  F0f == F0m (each model its own family)",
      abs(F0f_eye - F0m) < 1e-10)
check("equal-sized partition  =>  F0f == F0m (f0 uniform)",
      abs(F0f - F0m) < 1e-10)
# unequal partition breaks the coincidence — f0 is no longer uniform
C_uneq = np.array([[1, 0], [0, 1], [0, 1], [0, 1]], float)
_, F0f_uneq = fe_null(Lms, {"families": True, "C": C_uneq})
check("unequal partition  =>  F0f != F0m (f0 non-uniform)",
      abs(F0f_uneq - F0m) > 1e-6)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Partition validation (ValueError, not assert)")
print("=" * 70)
L = synth_L(20, 4, favored=[0] * 12 + [2] * 8)

expect_valueerror("gap (model in no family) raises",
                  lambda: group_bms(L, families=[[0, 1], [2]]))
expect_valueerror("overlap (model in two families) raises",
                  lambda: group_bms(L, families=[[0, 1, 2], [2, 3]]))
expect_valueerror("empty family raises",
                  lambda: group_bms(L, families=[[0, 1, 2, 3], []]))
expect_valueerror("out-of-range index raises",
                  lambda: group_bms(L, families=[[0, 1], [2, 4]]))
expect_valueerror("wrong family_names length raises",
                  lambda: group_bms(L, families=[[0, 1], [2, 3]],
                                    family_names=["only_one"]))
expect_valueerror("3D L to group_bms raises",
                  lambda: group_bms(np.zeros((5, 3, 2))))
expect_valueerror("1 condition to btw_conds raises",
                  lambda: group_bms_btw_conds(np.zeros((5, 3, 1))))
expect_valueerror("1 group to btw_groups raises",
                  lambda: group_bms_btw_groups([np.zeros((5, 3))]))
expect_valueerror("mismatched model counts across groups raises",
                  lambda: group_bms_btw_groups([np.zeros((5, 3)),
                                                np.zeros((5, 4))]))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. Family BOR: free-energy route vs the deleted heuristic")
print("=" * 70)

# INVARIANT: each model its own family => family inference IS model
# inference: same prior (1/K), same free energies => fam_bor == bor.
np.random.seed(1)
res_I = group_bms(L, families=[[0], [1], [2], [3]], n_samples=NSAMP)
check("C = I: family BOR == model BOR (exact)",
      abs(res_I.families.bor - res_I.bor) < 1e-12)
check("C = I: family frequencies == model frequencies",
      bool(np.allclose(res_I.families.family_frequency,
                       res_I.model_frequency)))

# fam_bor is a probability and behaves like one on a 2-family split
np.random.seed(1)
res_F = group_bms(L, families=[[0, 1], [2, 3]],
                  family_names=["A", "B"], n_samples=NSAMP)

# The draft's heuristic, on the case where it actually differs
# (nf != n_mod — for C = I its correction term log(K/nf) vanishes and
# it coincides with the correct value, which is why it looked
# plausible). Here nf=2, K=4, n=20: correction = 20·log(2) ≈ 13.9
# nats, blowing the BOR toward 1 regardless of the data.
n_sub, n_mod = L.shape
bor_c = np.clip(res_F.bor, 1e-16, 1 - 1e-16)
heur = 1 / (1 + np.exp(np.log((1 - bor_c) / bor_c) - n_sub * np.log(n_mod / 2)))
print(f"        [info] nf=2,K=4: free-energy fam_bor = "
      f"{res_F.families.bor:.3e}; draft heuristic = {heur:.3e}")
check("draft heuristic diverges wildly for nf != K (why it was deleted)",
      abs(heur - res_F.families.bor) > 0.1)
check("family BOR in [0, 1]", 0.0 <= res_F.families.bor <= 1.0)
check("strong family effect => low family BOR", res_F.families.bor < 0.5)
check("fam_pxp = bor/nf + (1-bor)*xp (Rigoux Eq. 7)",
      bool(np.allclose(
          res_F.families.protected_exceedance_prob,
          res_F.families.bor / 2
          + (1 - res_F.families.bor) * res_F.families.exceedance_prob)))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. Family frequency exact; exceedance agrees with draft sampler")
print("=" * 70)
a = res_F.posterior_parameters
a_fam = np.array([a[[0, 1]].sum(), a[[2, 3]].sum()])
check("family posterior_parameters == Cᵀa exactly",
      bool(np.allclose(res_F.families.posterior_parameters, a_fam)))
check("family_frequency == α_fam/Σα exactly (no MC noise)",
      bool(np.allclose(res_F.families.family_frequency, a_fam / a_fam.sum())))

# Draft's ad-hoc sampler (dirichlet.rvs over models, then sum) — the
# Dirichlet aggregation property says both estimate the same quantity.
from scipy.stats import dirichlet  # noqa: E402
rng = np.random.default_rng(7)
samp = dirichlet.rvs(a, size=NSAMP, random_state=rng)
ff = np.stack([samp[:, [0, 1]].sum(axis=1), samp[:, [2, 3]].sum(axis=1)], axis=1)
xp_draft = np.bincount(ff.argmax(axis=1), minlength=2) / NSAMP
diff = np.max(np.abs(xp_draft - res_F.families.exceedance_prob))
check(f"exceedance matches draft sampler within MC tol (diff={diff:.4f})",
      diff < 0.01)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. Between-conditions / between-groups directional sanity")
print("=" * 70)

# Same model in both conditions for every subject -> 'equal' should win
n_sub = 24
fav = [1] * n_sub
L3_same = np.stack([synth_L(n_sub, 3, fav, seed=2),
                    synth_L(n_sub, 3, fav, seed=3)], axis=2)
np.random.seed(2)
r_same = group_bms_btw_conds(L3_same, n_samples=NSAMP)
check(f"same model across conds: xp(equal) high ({r_same.xp:.3f})",
      r_same.xp > 0.9)
check("best tuple is on the diagonal (is_equal)", r_same.best.is_equal)

# Model switches between conditions -> 'equal' should lose
L3_diff = np.stack([synth_L(n_sub, 3, [0] * n_sub, seed=4),
                    synth_L(n_sub, 3, [2] * n_sub, seed=5)], axis=2)
np.random.seed(3)
r_diff = group_bms_btw_conds(L3_diff, n_samples=NSAMP)
check(f"model switch between conds: xp(equal) low ({r_diff.xp:.3f})",
      r_diff.xp < 0.1)
check("best tuple is off-diagonal", not r_diff.best.is_equal)
check("tuple bookkeeping: n_equal + n_not_equal == n_tuples",
      r_diff.n_equal + r_diff.n_not_equal == r_diff.n_tuples == 9)

# Groups (VBA free-energy test, 2026-08-03 — the draft's tuple
# construction gave xp(equal)=1.0 on the 'different' case below, the
# empirical disproof recorded in DEV.md §5):
# same favored model -> p_equal high; different -> reject equality.
np.random.seed(4)
g_same = group_bms_btw_groups(
    [synth_L(15, 3, [1] * 15, seed=6), synth_L(18, 3, [1] * 18, seed=7)],
    n_samples=NSAMP)
check(f"same model across groups: p_equal high ({g_same.p_equal:.3f})",
      g_same.p_equal > 0.5)
check("same model: equality not rejected", not g_same.h_reject_equality)
np.random.seed(5)
g_diff = group_bms_btw_groups(
    [synth_L(15, 3, [0] * 15, seed=8), synth_L(18, 3, [2] * 18, seed=9)],
    n_samples=NSAMP)
check(f"different models across groups: p_equal low ({g_diff.p_equal:.3e})",
      g_diff.p_equal < 0.05)
check("different models: equality rejected (h)", g_diff.h_reject_equality)
check("F bookkeeping: p_equal = 1/(1+exp(Fd-Fe))",
      abs(g_diff.p_equal
          - 1 / (1 + np.exp(g_diff.F_diff - g_diff.F_equal))) < 1e-12)
check("group sizes recorded", g_diff.group_sizes == [15, 18])
check("pooled + per-group fits carry free energies",
      np.isfinite(g_diff.F_equal) and np.isfinite(g_diff.F_diff)
      and all(np.isfinite(r.F) for r in g_diff.per_group))
check("VBA alias shim: g['p'] == p_equal", g_diff["p"] == g_diff.p_equal)

# per-condition results match standalone group_bms on the same slice
np.random.seed(6)
solo = group_bms(L3_same[:, :, 0], n_samples=NSAMP)
check("per_cond[0] model frequencies == standalone group_bms",
      bool(np.allclose(r_same.per_cond[0].model_frequency,
                       solo.model_frequency, atol=1e-12)))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. Dataclass results + draft dict-access shim")
print("=" * 70)
check("result is a GroupBMSResult dataclass", isinstance(res_F, GroupBMSResult))
check("draft access res['models']['ef'] still works",
      bool(np.allclose(res_F["models"]["ef"], res_F.model_frequency)))
check("draft access res['families']['pxp'] still works",
      bool(np.allclose(res_F["families"]["pxp"],
                       res_F.families.protected_exceedance_prob)))
check("within-family results carry name + member indices",
      res_F.families.within[0].name == "A"
      and res_F.families.within[0].models.tolist() == [0, 1])
check("to_dict() round-trips", isinstance(res_F.to_dict(), dict))
try:
    res_F["nope"]
    check("unknown key raises KeyError", False)
except KeyError:
    check("unknown key raises KeyError", True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("All checks passed.")
