"""
Worked example: group-level Bayesian model selection (cbm/group_bms.py).

Covers all three modes on synthetic log-evidence:
  1. group_bms            — standard BMS with model families
  2. group_bms_btw_conds  — same subjects, do they keep their model
                            across two conditions?
  3. group_bms_btw_groups — do two different groups of subjects share
                            one model-frequency profile?

The log-evidence matrices are synthesized directly so the example runs
in seconds. In a real analysis each column of L comes from a fit's
log_evidence (FitResult.output.log_evidence); check where that
evidence came from first — see check_evidence_provenance() and
DEV.md §2.1/§5:

    fits = [cbm_rl1, cbm_rl2]                       # one FitResult per model
    cbm.check_evidence_provenance(fits)             # warns on Mod 2 fallback
    L = np.column_stack([f.output.log_evidence for f in fits])
"""
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so this script runs without installing the
# package (`python examples/example_group_bms.py` from anywhere).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbm import group_bms, group_bms_btw_conds, group_bms_btw_groups

rng = np.random.default_rng(42)
N_SAMPLES = 200_000  # MC samples for exceedance probabilities


def synth_L(n_sub, n_mod, favored, delta=2.5, seed=0):
    """Synthetic per-subject log-evidence; favored[i] wins by ~delta nats."""
    r = np.random.default_rng(seed)
    L = r.normal(scale=0.8, size=(n_sub, n_mod))
    for i, k in enumerate(favored):
        L[i, k] += delta
    return L


# ============================================================================
# 1. Standard group BMS with families
# ============================================================================
print("=" * 70)
print("1. group_bms — 4 models, 2 families, 24 subjects")
print("=" * 70)

# 16 subjects best fit by model 1, 8 by model 3 -> family A (models 0,1)
# should dominate, model 1 within it.
favored = [1] * 16 + [3] * 8
L = synth_L(24, 4, favored, seed=1)

res = group_bms(L, families=[[0, 1], [2, 3]], family_names=["A", "B"],
                n_samples=N_SAMPLES)

print(f"\nModel level:")
print(f"  frequencies        : {np.round(res.model_frequency, 3)}")
print(f"  exceedance         : {np.round(res.exceedance_prob, 3)}")
print(f"  protected exceed.  : {np.round(res.protected_exceedance_prob, 3)}")
print(f"  BOR (model level)  : {res.bor:.4f}")

fam = res.families
print(f"\nFamily level ({fam.names}):")
print(f"  frequencies        : {np.round(fam.family_frequency, 3)}  (exact)")
print(f"  exceedance         : {np.round(fam.exceedance_prob, 3)}")
print(f"  BOR (family FE)    : {fam.bor:.4f}   <- free energy, not heuristic")
print(f"  protected exceed.  : {np.round(fam.protected_exceedance_prob, 3)}")
for w in fam.within:
    print(f"  within '{w.name}' (models {w.models.tolist()}): "
          f"freq {np.round(w.model_frequency, 3)}")

# ============================================================================
# 2. Between conditions (within-subject)
# ============================================================================
print("\n" + "=" * 70)
print("2. group_bms_btw_conds — 3 models, 2 conditions, 20 subjects")
print("=" * 70)

n_sub = 20
# Scenario A: everyone keeps model 1 in both conditions
L_stable = np.stack([synth_L(n_sub, 3, [1] * n_sub, seed=2),
                     synth_L(n_sub, 3, [1] * n_sub, seed=3)], axis=2)
r_stable = group_bms_btw_conds(L_stable, n_samples=N_SAMPLES)

# Scenario B: everyone switches from model 0 to model 2
L_switch = np.stack([synth_L(n_sub, 3, [0] * n_sub, seed=4),
                     synth_L(n_sub, 3, [2] * n_sub, seed=5)], axis=2)
r_switch = group_bms_btw_conds(L_switch, n_samples=N_SAMPLES)

for label, r in [("stable (model 1 -> model 1)", r_stable),
                 ("switch (model 0 -> model 2)", r_switch)]:
    b = r.best
    print(f"\n{label}:")
    print(f"  P(same model across conds)  xp = {r.xp:.3f}   pxp = {r.pxp:.3f}")
    print(f"  family BOR of the tuple fit : {r.bor:.4f}")
    print(f"  best tuple: models {b.models.tolist()} "
          f"(equal across conds: {b.is_equal})")
    print(f"  tuples: {r.n_tuples} = {r.n_equal} equal + {r.n_not_equal} not")

# ============================================================================
# 3. Between groups (free-energy test, = VBA_groupBMC_btwGroups)
# ============================================================================
print("\n" + "=" * 70)
print("3. group_bms_btw_groups — 3 models, groups of 15 and 18 subjects")
print("=" * 70)

# Scenario A: both groups favor model 1
g_same = group_bms_btw_groups(
    [synth_L(15, 3, [1] * 15, seed=6), synth_L(18, 3, [1] * 18, seed=7)],
    n_samples=N_SAMPLES)

# Scenario B: group 1 favors model 0, group 2 favors model 2
g_diff = group_bms_btw_groups(
    [synth_L(15, 3, [0] * 15, seed=8), synth_L(18, 3, [2] * 18, seed=9)],
    n_samples=N_SAMPLES)

for label, g in [("same model in both groups", g_same),
                 ("different models per group", g_diff)]:
    print(f"\n{label}:")
    print(f"  F(equal freqs)  = {g.F_equal:.2f}")
    print(f"  F(per-group)    = {g.F_diff:.2f}")
    print(f"  p(same freqs)   = {g.p_equal:.4g}")
    print(f"  reject equality : {g.h_reject_equality}  (VBA rule: p < 0.05)")
    for i, pg in enumerate(g.per_group):
        print(f"  group {i + 1} frequencies: {np.round(pg.model_frequency, 3)}")

print("\ndone :]")
