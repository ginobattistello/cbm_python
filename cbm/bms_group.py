"""
Group-level Bayesian Model Selection between families, conditions and groups
----------------------------------------------------------------------------
Reproduces the VBA toolbox's group-level BMC routines on top of the
existing `bms()` / `model_selection.py` machinery:

    bms_group            ≈ VBA_groupBMC          (2D L, optional families)
    bms_group_btw_conds  ≈ VBA_groupBMC_btwConds (3D L: subj × model × cond)
    bms_group_btw_groups ≈ VBA_groupBMC_btwGroups(list of 2D L, one per group)

Promoted out of cbm/dev/ on 2026-08-03 (DEV.md §5) with these changes
against the staged draft:
  * families must PARTITION the model set (disjoint + exhaustive) —
    validated once, shared by the Dirichlet prior and the membership
    matrix C (the draft left `a0 = 0` for unassigned models: an invalid
    Dirichlet prior).
  * the family-level BOR comes from the family free energy,
    `compute_bor(L.T, posterior, priors, C=C)` (Rigoux et al. 2014,
    Eq. 5) — the draft's `log((1-bor)/bor) - n·log(K/nf)` rescaling
    heuristic is gone. Cross-checked against VBA_groupBMC.m: with
    families, VBA's out.bor IS this family free-energy BOR, and its
    prior (`priors.a(f) = 1/|f|`, normalized to one total count) is
    the a0 used here.
  * family frequencies are exact (`α_fam/Σα`, Dirichlet aggregation —
    VBA_dirichlet_moments does the same) and family exceedance uses
    the toolbox's `dirichlet_exceedance` on `α_fam = Cᵀ·a`, VBA's own
    choice (`VBA_ExceedanceProb(out.families.a,[],'dirichlet')`).
  * BETWEEN-GROUPS IS A DIFFERENT TEST THAN THE DRAFT HAD: the draft's
    tuple construction is both empirically degenerate and absent from
    VBA — VBA_groupBMC_btwGroups is a free-energy comparison of pooled
    vs per-group fits. See BtwGroupsResult for the full record.
  * typed dataclass results (with a `['key']`/`to_dict()` shim for the
    draft's dict access); explicit entry points, no work in __init__;
    `raise ValueError` instead of `assert` for input validation.

PROVENANCE WARNING (DEV.md §5): every input `L` is a per-subject ×
per-model Laplace log-evidence. If a fit did not opt in to the
Gauss-Newton curvature (optimization.py Mod 5, via `model_trials`),
its evidence inherited the eigenvalue-clip artifact of Mod 2 (§2.1)
and every statistic here inherits it too. Use
`check_evidence_provenance(fit_results)` before trusting the output.

REFERENCES:
Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017

Rigoux L, Stephan KE, Friston KJ, Daunizeau J (2014)
Bayesian model selection for group studies - Revisited.
NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product as iterproduct
from typing import List, Optional, Sequence
import warnings

import numpy as np

from .model_selection import bms, compute_bor, compute_fe, dirichlet_exceedance


def _bms_rng(random_state=None):
    """Return/reuse a local Generator for reproducible Monte-Carlo BMS."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)



# ═══════════════════════════════════════════════════════════════════
# Result dataclasses (DEV.md §5: "Return dataclasses, not nested dicts")
# ═══════════════════════════════════════════════════════════════════
class _DictShim:
    """Backward-compat shim: the cbm/dev draft returned nested dicts
    with short keys ('a', 'ef', 'xp', 'pxp'); allow the same access on
    the dataclasses so existing exploration code keeps working."""
    _ALIASES: dict = {}

    def __getitem__(self, key):
        key = self._ALIASES.get(key, key)
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def keys(self):
        return list(self.__dataclass_fields__)  # type: ignore[attr-defined]

    def to_dict(self):
        return asdict(self)


@dataclass
class WithinFamilyResult(_DictShim):
    """BMS restricted to the members of one family (uniform prior
    within the family)."""
    name: str
    models: np.ndarray              # member indices into the full model set
    posterior_parameters: np.ndarray
    model_frequency: np.ndarray
    exceedance_prob: np.ndarray
    bor: float
    protected_exceedance_prob: np.ndarray

    _ALIASES = {"a": "posterior_parameters", "ef": "model_frequency",
                "xp": "exceedance_prob", "pxp": "protected_exceedance_prob"}


@dataclass
class FamilyResult(_DictShim):
    """Family-level inference (Rigoux et al. 2014, §'family inference').

    posterior_parameters: α_fam = Cᵀ·a — exact, by Dirichlet aggregation.
    family_frequency:     E[r_fam] = α_fam / Σα_fam — exact, no sampling.
    exceedance_prob:      dirichlet_exceedance(α_fam) (Monte-Carlo).
    bor:                  from the FAMILY free energy (compute_bor with
                          C — Rigoux Eq. 5), NOT the draft's rescaling
                          heuristic.
    protected_exceedance_prob: bor/nf + (1−bor)·xp (Rigoux Eq. 7).
    """
    names: List[str]
    members: List[np.ndarray]       # index arrays, one per family
    posterior_parameters: np.ndarray
    family_frequency: np.ndarray
    exceedance_prob: np.ndarray
    bor: float
    protected_exceedance_prob: np.ndarray
    within: List[WithinFamilyResult]

    _ALIASES = {"a": "posterior_parameters", "ef": "family_frequency",
                "xp": "exceedance_prob", "pxp": "protected_exceedance_prob"}


@dataclass
class GroupBMSResult(_DictShim):
    """Standard group BMS (≈ VBA_groupBMC). Field names match BMSResult.

    g: posterior responsibilities q(m_i = k | y_i), subjects × models.
    F: variational free energy of the fitted posterior (Rigoux A.20;
       VBA's out.F(end)) — what the between-groups test compares.
    bor: MODEL-level BOR (F vs equal-model-frequency null). With
       families, VBA reports the family-level BOR as its out.bor;
       that value lives in .families.bor here — both are kept.
    """
    posterior_parameters: np.ndarray
    model_frequency: np.ndarray
    exceedance_prob: np.ndarray
    bor: float
    protected_exceedance_prob: np.ndarray
    alpha0: np.ndarray
    g: Optional[np.ndarray] = None
    F: Optional[float] = None
    families: Optional[FamilyResult] = None

    _ALIASES = {"a": "posterior_parameters", "ef": "model_frequency",
                "xp": "exceedance_prob", "pxp": "protected_exceedance_prob"}

    def __getitem__(self, key):
        # draft compat: result["models"]["ef"] — the model level was a
        # nested dict; here the model level IS this object.
        if key == "models":
            return self
        return super().__getitem__(key)

    # ── MODIFICATION 13 — readable output (DEV.md §16). Presentation
    # only; reads existing fields, never called during inference.
    def summary(self, model_names=None) -> str:
        from .reporting import bms_summary
        return bms_summary(self, model_names=model_names)

    def table(self, model_names=None, pandas: bool = True):
        """One row per model: frequency, exceedance, protected xp, alpha."""
        from .reporting import bms_table
        return bms_table(self, model_names=model_names, pandas=pandas)

    def plot(self, model_names=None, display=True, **kwargs):
        """Plot posterior attribution and model-level BMS quantities."""
        return _plot_bms(
            self,
            model_names=model_names,
            display=display,
            **kwargs,
        )

    def __repr__(self) -> str:
        try:
            return self.summary()
        except Exception as e:
            return (f"<GroupBMSResult "
                    f"(summary failed: {type(e).__name__}: {e})>")


@dataclass
class BestTuple(_DictShim):
    """The most frequent model assignment across conditions/groups."""
    tuple_idx: int
    models: np.ndarray              # model index per condition/group slot
    families: np.ndarray            # family index per slot
    family_names: Optional[List[str]]
    is_equal: bool                  # same model/family in every slot?


@dataclass
class BtwCondsResult(_DictShim):
    """Between-conditions BMS (≈ VBA_groupBMC_btwConds).

    xp / pxp: P(same model-or-family across all conditions) — the
    'equal' family's exceedance / protected exceedance in the
    tuple-level BMS (`btw`). VBA cross-check (2026-08-03):
    VBA_groupBMC_btwConds reports ep = out.families.ep(1) and
    pep = ep·(1−bor) + bor/2 with bor the FAMILY free-energy BOR of
    the tuple fit — algebraically identical to our xp and pxp. `bor`
    here is that same family-level BOR (VBA's out.bor when families
    are given); the tuple/model-level BOR remains at `btw.bor`.
    """
    xp: float
    pxp: float
    bor: float
    n_tuples: int
    n_equal: int
    n_not_equal: int
    tuples: np.ndarray
    best: BestTuple
    btw: GroupBMSResult
    per_cond: List[GroupBMSResult]

    # ── MODIFICATION 13 — readable output (DEV.md §16).
    def summary(self) -> str:
        from .reporting import btw_conds_summary
        return btw_conds_summary(self)

    def plot(self, model_names=None, display=True, **kwargs):
        """Plot condition-specific attribution and BMS comparison."""
        return _plot_btw_conds(
            self,
            model_names=model_names,
            display=display,
            **kwargs,
        )

    def __repr__(self) -> str:
        try:
            return self.summary()
        except Exception as e:
            return (f"<BtwCondsResult "
                    f"(summary failed: {type(e).__name__}: {e})>")


@dataclass
class BtwGroupsResult(_DictShim):
    """Between-groups BMS (= VBA_groupBMC_btwGroups): a free-energy test
    of H0 "all groups share one model-frequency profile" against H1
    "each group has its own".

    F_equal:  free energy of the pooled fit (one Dirichlet for everyone).
    F_diff:   sum of the per-group fits' free energies.
    p_equal:  posterior probability of H0 = 1/(1 + exp(F_diff − F_equal)).
    h_reject_equality: VBA's decision rule, p_equal < 0.05.

    [2026-08-03 — REPLACES the cbm/dev draft's tuple construction for
    groups. Two independent reasons (DEV.md §5 correction):
      (a) empirically degenerate: a subject never informs the OTHER
          group's tuple slot, so within each subject's tie-set the
          family-fair prior (each 'equal' tuple carries twice the
          per-tuple mass of a 'not equal' one) is amplified by the
          rich-get-richer ψ(α) VB update into certainty — the harness
          measured xp(equal) = 1.0 for groups favoring OPPOSITE models;
      (b) it never was the VBA construction: VBA_groupBMC_btwGroups.m
          contains no tuples at all — it is exactly this free-energy
          comparison (`p = 1./(1+exp(Fd-Fe))`).
    VBA hardcodes 2 groups; this accepts G >= 2 (F_diff sums all
    per-group free energies).]
    """
    p_equal: float
    h_reject_equality: bool
    F_equal: float
    F_diff: float
    n_groups: int
    group_sizes: List[int]
    pooled: GroupBMSResult
    per_group: List[GroupBMSResult]

    _ALIASES = {"p": "p_equal", "h": "h_reject_equality"}

    # ── MODIFICATION 13 — readable output (DEV.md §16).
    def summary(self) -> str:
        from .reporting import btw_groups_summary
        return btw_groups_summary(self)

    def plot(self, model_names=None, display=True, **kwargs):
        """Plot group-specific attribution and BMS comparison."""
        return _plot_btw_groups(
            self,
            model_names=model_names,
            display=display,
            **kwargs,
        )

    def __repr__(self) -> str:
        try:
            return self.summary()
        except Exception as e:
            return (f"<BtwGroupsResult "
                    f"(summary failed: {type(e).__name__}: {e})>")


# ═══════════════════════════════════════════════════════════════════
# Input validation (DEV.md §5: partition check; ValueError, not assert)
# ═══════════════════════════════════════════════════════════════════
def _validate_partition(families: Sequence[Sequence[int]],
                        n_mod: int,
                        family_names: Optional[Sequence[str]]):
    """Families must PARTITION {0..n_mod-1}: disjoint and exhaustive.

    A model in no family would keep a Dirichlet prior mass of 0
    (invalid); a model in two families would double-count evidence in
    C. One validation, shared by the a0 prior and the membership
    matrix C (DEV.md §5). Returns (index arrays, names, C[n_mod, nf]).
    """
    if not isinstance(families, (list, tuple)) or len(families) == 0:
        raise ValueError("families must be a non-empty list of index lists")
    fams = []
    for f, idx in enumerate(families):
        arr = np.atleast_1d(np.asarray(idx, dtype=int))
        if arr.size == 0:
            raise ValueError(f"family {f} is empty")
        if np.any(arr < 0) or np.any(arr >= n_mod):
            raise ValueError(
                f"family {f} has model indices outside 0..{n_mod - 1}: "
                f"{arr.tolist()}")
        fams.append(arr)

    all_idx = np.concatenate(fams)
    unique = np.unique(all_idx)
    if len(all_idx) != len(unique):
        raise ValueError(
            "families must be DISJOINT: model(s) "
            f"{sorted(set(int(i) for i in all_idx if list(all_idx).count(i) > 1))} "
            "appear in more than one family")
    if len(unique) != n_mod:
        missing = sorted(set(range(n_mod)) - set(int(i) for i in unique))
        raise ValueError(
            "families must be EXHAUSTIVE: model(s) "
            f"{missing} belong to no family (their Dirichlet prior mass "
            "would be 0, which is invalid)")

    nf = len(fams)
    if family_names is None:
        names = [f"f_{i}" for i in range(nf)]
    else:
        names = list(family_names)
        if len(names) != nf:
            raise ValueError(
                f"family_names has {len(names)} entries for {nf} families")

    C = np.zeros((n_mod, nf))
    for f, arr in enumerate(fams):
        C[arr, f] = 1.0
    return fams, names, C



# ═══════════════════════════════════════════════════════════════════
# Console output
# ═══════════════════════════════════════════════════════════════════

def _print_bms_header(
    n_subjects: int,
    n_models: int,
    families,
    n_samples: int,
) -> None:
    """Print a compact CBM-style run header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"{'bms_group':<40}{now:>30}")
    print("=" * 70)
    print(f"Number of subjects: {n_subjects}")
    print(f"Number of models: {n_models}")
    print(
        "Families: "
        + ("none" if families is None else str(len(families)))
    )
    print(f"Exceedance samples: {int(n_samples):,}")
    print("-" * 70)


def _print_bms_result(result: "GroupBMSResult") -> None:
    """Print the main BMS quantities after inference."""
    print("Model results")

    for k in range(len(result.model_frequency)):
        print(
            f"  Model {k + 1:02d}: "
            f"frequency={result.model_frequency[k]:.3f}  "
            f"XP={result.exceedance_prob[k]:.3f}  "
            f"PXP={result.protected_exceedance_prob[k]:.3f}"
        )

    print(f"Bayes omnibus risk: {result.bor:.3f}")

    if result.families is not None:
        fam = result.families
        print("\nFamily results")

        for f, name in enumerate(fam.names):
            print(
                f"  {name}: "
                f"frequency={fam.family_frequency[f]:.3f}  "
                f"XP={fam.exceedance_prob[f]:.3f}  "
                f"PXP={fam.protected_exceedance_prob[f]:.3f}"
            )

        print(f"Family Bayes omnibus risk: {fam.bor:.3f}")

    print("-" * 70)
    print("done :]")


# ═══════════════════════════════════════════════════════════════════
# Entry points (DEV.md §5: "no work in __init__")
# ═══════════════════════════════════════════════════════════════════
def bms_group(L: np.ndarray,
              families: Optional[Sequence[Sequence[int]]] = None,
              family_names: Optional[Sequence[str]] = None,
              n_samples: int = 1_000_000,
              verbose: bool = True,
              display: bool = False,
              random_state=None) -> GroupBMSResult:
    """
    Standard group-level BMS (≈ VBA_groupBMC).

    Args:
        L: (n_subjects × n_models) log model evidences.
        families: optional partition of the models into families,
            e.g. [[0, 1], [2, 3]]. Must be disjoint and exhaustive.
        family_names: optional names, one per family.
        n_samples: Monte-Carlo samples for exceedance probabilities.
        verbose: If True, print a compact CBM-style run/result summary.
        display: If True, automatically show the BMS diagnostic figure.
        random_state: Integer seed, NumPy Generator, or None for exceedance
            Monte-Carlo sampling. The VB solution itself is deterministic.

    Returns:
        GroupBMSResult (with .families set when families were given).
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 2:
        raise ValueError(f"L must be 2D (subjects × models), got ndim={L.ndim}")
    n_sub, n_mod = L.shape
    rng = _bms_rng(random_state)

    if verbose:
        _print_bms_header(
            n_subjects=n_sub,
            n_models=n_mod,
            families=families,
            n_samples=n_samples,
        )

    if families is None:
        # VBA_groupBMC normalizes the default prior to ONE total prior
        # count (`priors.a = priors.a./sum(priors.a)`), i.e. 1/K per
        # model — not bms()'s default of one count PER model.
        a0 = np.ones(n_mod) / n_mod
        res = bms(L, Nsamp=int(n_samples), alpha0=a0, random_state=rng)
        F = compute_fe(L.T, {"a": res.posterior_parameters, "r": res.g.T},
                       {"a": a0})
        result = GroupBMSResult(
            posterior_parameters=res.posterior_parameters,
            model_frequency=res.model_frequency,
            exceedance_prob=res.exceedance_prob,
            bor=res.bor,
            protected_exceedance_prob=res.protected_exceedance_prob,
            alpha0=a0,
            g=res.g,
            F=F,
        )

        if verbose:
            _print_bms_result(result)

        if display:
            result.plot()

        return result

    fams, names, C = _validate_partition(families, n_mod, family_names)
    nf = len(fams)

    # Family-aware Dirichlet prior: equal mass per family, split equally
    # within (this is what makes the model-level BOR/PXP family-fair).
    # The partition check above guarantees every entry is filled.
    a0 = np.zeros(n_mod)
    for arr in fams:
        a0[arr] = 1.0 / (nf * len(arr))

    res = bms(L, Nsamp=int(n_samples), alpha0=a0, random_state=rng)
    a = res.posterior_parameters

    # Family posterior: aggregating Dirichlet components sums their
    # parameters exactly — no sampling needed for α_fam or E[r_fam].
    a_fam = C.T @ a
    fam_ef = a_fam / a_fam.sum()
    fam_xp = dirichlet_exceedance(a_fam, int(n_samples), random_state=rng)

    # Family BOR from the family free energy (Rigoux et al. 2014 Eq. 5)
    # via the toolbox's own compute_bor — same call pattern as bms()'s
    # model-level BOR, plus the membership matrix C.
    # ORIENTATION (DEV.md §5 caveat): compute_bor/compute_fe/fe_null
    # take L as models × subjects and r as [model, subject]; bms/
    # bms_group use subjects × models. Transpose deliberately.
    posterior = {"a": a, "r": res.g.T}
    priors = {"a": a0}
    fam_bor = compute_bor(L.T, posterior, priors, C=C)
    F = compute_fe(L.T, posterior, priors)

    # Protected exceedance (Rigoux Eq. 7, family level) — now on the
    # correct family BOR.
    fam_pxp = fam_bor / nf + (1.0 - fam_bor) * fam_xp

    # Within-family BMS: uniform prior over the family's own members.
    within = []
    for f, arr in enumerate(fams):
        a0w = np.ones(len(arr)) / len(arr)
        rw = bms(L[:, arr], Nsamp=int(n_samples), alpha0=a0w, random_state=rng)
        within.append(WithinFamilyResult(
            name=names[f],
            models=arr,
            posterior_parameters=rw.posterior_parameters,
            model_frequency=rw.model_frequency,
            exceedance_prob=rw.exceedance_prob,
            bor=rw.bor,
            protected_exceedance_prob=rw.protected_exceedance_prob,
        ))

    result = GroupBMSResult(
        posterior_parameters=a,
        model_frequency=res.model_frequency,
        exceedance_prob=res.exceedance_prob,
        bor=res.bor,
        protected_exceedance_prob=res.protected_exceedance_prob,
        alpha0=a0,
        g=res.g,
        F=F,
        families=FamilyResult(
            names=names,
            members=fams,
            posterior_parameters=a_fam,
            family_frequency=fam_ef,
            exceedance_prob=fam_xp,
            bor=fam_bor,
            protected_exceedance_prob=fam_pxp,
            within=within,
        ),
    )

    if verbose:
        _print_bms_result(result)

    if display:
        result.plot()

    return result


def _print_btw_conds_result(result: "BtwCondsResult") -> None:
    """Print one compact between-condition summary."""
    print("Between-condition results")
    print(f"  P(same model/family): {result.xp:.3f}")
    print(f"  Protected P(same):    {result.pxp:.3f}")
    print(f"  Bayes omnibus risk:   {result.bor:.3f}")
    print(
        "  Best tuple: "
        + " -> ".join(str(int(m) + 1) for m in result.best.models)
    )
    print(
        "  Best tuple status: "
        + ("same" if result.best.is_equal else "different")
    )
    print("\nCondition-specific model frequencies")
    for c, cond in enumerate(result.per_cond, start=1):
        freq = "  ".join(
            f"M{k + 1}={value:.3f}"
            for k, value in enumerate(cond.model_frequency)
        )
        print(f"  Condition {c}: {freq}")
    print("-" * 70)
    print("done :]")


def _print_btw_groups_result(result: "BtwGroupsResult") -> None:
    """Print one compact between-group summary."""
    print("Between-group results")
    print(f"  P(equal model-frequency profile): {result.p_equal:.3f}")
    print(f"  Equality rejected (p < 0.05): {result.h_reject_equality}")
    print(f"  Pooled free energy:   {result.F_equal:.3f}")
    print(f"  Separate free energy: {result.F_diff:.3f}")
    print("\nGroup-specific model frequencies")
    for g, grp in enumerate(result.per_group, start=1):
        freq = "  ".join(
            f"M{k + 1}={value:.3f}"
            for k, value in enumerate(grp.model_frequency)
        )
        print(f"  Group {g}: {freq}")
    print("-" * 70)
    print("done :]")


def _tuple_machinery(n_mod: int, n_slots: int, cfam: np.ndarray):
    """Enumerate all model tuples over slots and split them into
    'equal' (same family in every slot) vs 'not equal'. Shared by the
    between-conditions and between-groups constructions."""
    tuples = np.array(list(iterproduct(range(n_mod), repeat=n_slots)))
    nt = len(tuples)  # == n_mod ** n_slots by construction of iterproduct
    if nt > 100_000:
        warnings.warn(f"{n_mod}^{n_slots} = {nt} tuples — may be slow")
    is_eq = np.array([len(set(cfam[tuples[t]])) == 1 for t in range(nt)])
    eq_idx = np.where(is_eq)[0].tolist()
    neq_idx = np.where(~is_eq)[0].tolist()
    return tuples, nt, is_eq, eq_idx, neq_idx


def _family_assignment(n_mod, families, family_names):
    """Per-model family index (identity when no families given)."""
    if families is not None:
        fams, names, _ = _validate_partition(families, n_mod, family_names)
        cfam = np.zeros(n_mod, dtype=int)
        for f, arr in enumerate(fams):
            cfam[arr] = f
        return cfam, names
    return np.arange(n_mod), None


def _best_tuple(btw: GroupBMSResult, tuples, is_eq, cfam, names) -> BestTuple:
    best_t = int(np.argmax(btw.model_frequency))
    best_models = tuples[best_t]
    best_families = cfam[best_models]
    return BestTuple(
        tuple_idx=best_t,
        models=best_models,
        families=best_families,
        family_names=[names[f] for f in best_families] if names else None,
        is_equal=bool(is_eq[best_t]),
    )


def bms_group_btw_conds(L: np.ndarray,
                        families: Optional[Sequence[Sequence[int]]] = None,
                        family_names: Optional[Sequence[str]] = None,
                        n_samples: int = 1_000_000,
                        verbose: bool = True,
                        display: bool = False,
                        random_state=None) -> BtwCondsResult:
    """
    Between-conditions BMS (≈ VBA_groupBMC_btwConds): do the SAME
    subjects use the same model (or family) across conditions?

    Within-subject (repeated measures) construction: tuple log-evidence
    is the SUM across conditions of each condition's evidence for that
    tuple's model; the 'equal' family of tuples is then compared to
    'not equal' (DEV.md §5: construction verified, do not re-litigate).

    Args:
        L: (n_subjects × n_models × n_conditions) log evidences.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 3:
        raise ValueError(
            f"L must be 3D (subjects × models × conditions), got ndim={L.ndim}")
    n_sub, n_mod, n_cond = L.shape
    rng = _bms_rng(random_state)
    if n_cond < 2:
        raise ValueError(
            "between-conditions BMS needs >= 2 conditions; "
            "use bms_group(L[:, :, 0]) for a single condition")
    if n_mod < 2:
        raise ValueError("between-conditions BMS needs >= 2 models")

    cfam, names = _family_assignment(n_mod, families, family_names)
    tuples, nt, is_eq, eq_idx, neq_idx = _tuple_machinery(n_mod, n_cond, cfam)

    # Tuple log-evidence: sum across conditions (within-subject).
    Lt = sum(L[:, tuples[:, c], c] for c in range(n_cond))

    # Internal BMS fits are implementation details: keep them quiet.
    btw = bms_group(
        Lt,
        families=[eq_idx, neq_idx],
        family_names=["equal", "not_equal"],
        n_samples=n_samples,
        verbose=False,
        display=False,
        random_state=rng,
    )
    per_cond = [
        bms_group(
            L[:, :, c],
            families=families,
            family_names=family_names,
            n_samples=n_samples,
            verbose=False,
            display=False,
            random_state=rng,
        )
        for c in range(n_cond)
    ]

    result = BtwCondsResult(
        xp=float(btw.families.exceedance_prob[0]),
        pxp=float(btw.families.protected_exceedance_prob[0]),
        bor=btw.families.bor,
        n_tuples=nt,
        n_equal=len(eq_idx),
        n_not_equal=len(neq_idx),
        tuples=tuples,
        best=_best_tuple(btw, tuples, is_eq, cfam, names),
        btw=btw,
        per_cond=per_cond,
    )

    if verbose:
        _print_btw_conds_result(result)

    if display:
        result.plot()

    return result


def bms_group_btw_groups(Ls: Sequence[np.ndarray],
                         families: Optional[Sequence[Sequence[int]]] = None,
                         family_names: Optional[Sequence[str]] = None,
                         n_samples: int = 1_000_000,
                         verbose: bool = True,
                         display: bool = False,
                         random_state=None) -> BtwGroupsResult:
    """
    Between-groups BMS (= VBA_groupBMC_btwGroups): do DIFFERENT groups
    of subjects have the same model frequencies?

    Free-energy comparison (see BtwGroupsResult for why this replaced
    the draft's tuple construction): fit all subjects pooled (H0: one
    frequency profile) and each group separately (H1: per-group
    profiles); p_equal = 1/(1 + exp(F_diff − F_equal)). Families, when
    given, shape the Dirichlet prior of every fit exactly as VBA does
    (options.families is passed through to each VBA_groupBMC call).

    Args:
        Ls: list of G arrays, each (n_subjects_g × n_models).
    """
    if not isinstance(Ls, (list, tuple)):
        raise ValueError("Ls must be a list of per-group 2D arrays")
    Ls = [np.asarray(Lg, dtype=float) for Lg in Ls]
    n_grp = len(Ls)
    if n_grp < 2:
        raise ValueError("between-groups BMS needs >= 2 groups")
    for g, Lg in enumerate(Ls):
        if Lg.ndim != 2:
            raise ValueError(f"group {g}: L must be 2D, got ndim={Lg.ndim}")
    n_mod = Ls[0].shape[1]
    if any(Lg.shape[1] != n_mod for Lg in Ls):
        raise ValueError("all groups must have the same number of models")
    if n_mod < 2:
        raise ValueError("between-groups BMS needs >= 2 models")

    rng = _bms_rng(random_state)

    # H0: one frequency vector for all subjects (pooled fit)
    # Internal pooled/group fits are implementation details: keep them quiet.
    pooled = bms_group(
        np.vstack(Ls),
        families=families,
        family_names=family_names,
        n_samples=n_samples,
        verbose=False,
        display=False,
        random_state=rng,
    )

    # H1: each group gets its own frequency vector (separate fits)
    per_group = [
        bms_group(
            Lg,
            families=families,
            family_names=family_names,
            n_samples=n_samples,
            verbose=False,
            display=False,
            random_state=rng,
        )
        for Lg in Ls
    ]

    F_equal = float(pooled.F)
    F_diff = float(sum(r.F for r in per_group))
    p_equal = float(1.0 / (1.0 + np.exp(np.clip(F_diff - F_equal, -700, 700))))

    result = BtwGroupsResult(
        p_equal=p_equal,
        h_reject_equality=bool(p_equal < 0.05),
        F_equal=F_equal,
        F_diff=F_diff,
        n_groups=n_grp,
        group_sizes=[Lg.shape[0] for Lg in Ls],
        pooled=pooled,
        per_group=per_group,
    )

    if verbose:
        _print_btw_groups_result(result)

    if display:
        result.plot()

    return result



# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════
#
# Kept in this module intentionally so the toolbox has one compact BMS file.
# Plotting consumes the already-computed result objects and never changes the
# statistical inference.
#
def _mpl():
    import matplotlib.pyplot as plt
    return plt


def _model_names(n_models: int, model_names=None):
    if model_names is None:
        return [f"M{k + 1}" for k in range(n_models)]

    names = list(model_names)
    if len(names) != n_models:
        raise ValueError(
            f"model_names has {len(names)} entries for {n_models} models"
        )
    return names


# Same grayscale palette and typography as cbm/display.py.
_INK_LIGHT, _INK, _INK_DARK = "#C9C9CE", "#7A7A80", "#3A3A3E"
_GREY = "#888888"


def _plot_style():
    """Matplotlib rcParams shared with the individual-fit display."""
    return {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "DejaVu Sans",
        ],
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "grid.linewidth": 0.4,
        "grid.color": "#DDDDDD",
    }


def _style_axes(ax):
    ax.tick_params(labelsize=7, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _panel(
    ax,
    letter,
    title=None,
    subtitle=None,
    dx=-0.13,
    dy=1.03,
):
    """Panel label/title/subtitle matching cbm/display.py."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    if title:
        ax.set_title(
            title,
            pad=14 if subtitle else 5,
        )

    if subtitle:
        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=_GREY,
        )


def _responsibility_matrix(

    ax,
    g,
    model_names,
    *,
    title="subject model attribution",
    column_labels=None,
    separators=None,
):
    """Show q(model | subject) as models x subjects grayscale matrix."""
    g = np.asarray(g, dtype=float)

    if g.ndim != 2:
        raise ValueError("responsibility matrix g must be subjects x models")

    matrix = g.T

    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        cmap="Greys",
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_ylabel("model")
    ax.set_xlabel("subject")

    n_subjects = g.shape[0]
    if n_subjects <= 20:
        ax.set_xticks(np.arange(n_subjects))
        if column_labels is None:
            labels = [str(i + 1) for i in range(n_subjects)]
        else:
            labels = list(column_labels)
        ax.set_xticklabels(labels, rotation=0)
    else:
        # Keep the matrix readable for larger samples.
        step = max(1, int(np.ceil(n_subjects / 10)))
        ticks = np.arange(0, n_subjects, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(i + 1) for i in ticks])

    if separators is not None:
        for x in separators:
            ax.axvline(
                x - 0.5,
                color=_INK,
                lw=0.8,
                linestyle=":",
            )

    # Outline the most probable model for each subject.
    winner = np.argmax(g, axis=1)
    for n, k in enumerate(winner):
        ax.add_patch(
            __import__("matplotlib").patches.Rectangle(
                (n - 0.5, k - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=_INK_DARK,
                linewidth=0.6,
            )
        )

    _style_axes(ax)
    _panel(
        ax,
        "A",
        title,
        subtitle="grayscale = posterior responsibility q(model | subject)",
    )

    return image


def _paired_bars(
    ax,
    frequencies,
    exceedance,
    model_names,
    *,
    title="group model probabilities",
    protected=None,
):
    """Paired bars for expected model frequency and exceedance probability."""
    frequencies = np.asarray(frequencies, dtype=float)
    exceedance = np.asarray(exceedance, dtype=float)

    x = np.arange(len(model_names))
    width = 0.34

    ax.bar(
        x - width / 2,
        frequencies,
        width,
        label="frequency",
        color=_INK_DARK,
    )
    ax.bar(
        x + width / 2,
        exceedance,
        width,
        label="exceedance probability",
        color=_INK_LIGHT,
    )

    if protected is not None:
        protected = np.asarray(protected, dtype=float)
        ax.plot(
            x,
            protected,
            marker="o",
            linestyle="none",
            markersize=4,
            color=_INK_DARK,
            label="protected XP",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("probability")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _style_axes(ax)
    _panel(ax, "B", title)


def _multi_bars(
    ax,
    results,
    model_names,
    labels,
    *,
    title,
):
    """Condition/group-specific frequency and exceedance bars."""
    n_sets = len(results)
    n_models = len(model_names)

    x = np.arange(n_models)
    total_width = 0.82
    width = total_width / max(1, 2 * n_sets)

    offsets = (
        np.arange(2 * n_sets) - (2 * n_sets - 1) / 2
    ) * width

    for j, (label, result) in enumerate(zip(labels, results)):
        ax.bar(
            x + offsets[2 * j],
            result.model_frequency,
            width,
            label=f"{label} frequency",
            alpha=0.95,
        )
        ax.bar(
            x + offsets[2 * j + 1],
            result.exceedance_prob,
            width,
            label=f"{label} XP",
            alpha=0.45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("probability")
    ax.legend(
        fontsize=5.8,
        frameon=False,
        ncol=2,
    )
    ax.grid(axis="y", alpha=0.25)
    _style_axes(ax)
    _panel(ax, "C", title)


def _summary_standard(result, model_names):
    best_f = int(np.argmax(result.model_frequency))
    best_xp = int(np.argmax(result.exceedance_prob))

    lines = [
        "Standard BMS",
        "",
        f"subjects: {result.g.shape[0] if result.g is not None else '?'}",
        f"models: {len(model_names)}",
        f"highest frequency: {model_names[best_f]} "
        f"({result.model_frequency[best_f]:.3f})",
        f"highest XP: {model_names[best_xp]} "
        f"({result.exceedance_prob[best_xp]:.3f})",
        f"BOR: {result.bor:.3f}",
    ]

    if result.families is not None:
        fam = result.families
        best_family = int(np.argmax(fam.family_frequency))
        lines += [
            "",
            "Family inference",
            f"best family: {fam.names[best_family]} "
            f"({fam.family_frequency[best_family]:.3f})",
            f"family BOR: {fam.bor:.3f}",
        ]

    return lines


def _summary_conds(result):
    tuple_models = " -> ".join(
        f"M{int(m) + 1}"
        for m in result.best.models
    )

    return [
        "Between conditions",
        "",
        f"conditions: {len(result.per_cond)}",
        f"P(same model/family): {result.xp:.3f}",
        f"protected P(same): {result.pxp:.3f}",
        f"BOR: {result.bor:.3f}",
        f"best tuple: {tuple_models}",
        "best tuple status: "
        + ("same" if result.best.is_equal else "different"),
    ]


def _summary_groups(result):
    return [
        "Between groups",
        "",
        f"groups: {result.n_groups}",
        "group sizes: "
        + ", ".join(str(n) for n in result.group_sizes),
        f"P(equal frequency profile): {result.p_equal:.3f}",
        f"reject equality (p<.05): {result.h_reject_equality}",
        f"pooled free energy: {result.F_equal:.3f}",
        f"separate free energy: {result.F_diff:.3f}",
    ]


def _summary_panel(ax, lines, letter="D"):
    ax.axis("off")
    _panel(ax, letter, "result summary")

    y = 0.88
    for i, line in enumerate(lines):
        if line == "":
            y -= 0.055
            continue

        ax.text(
            0.04,
            y,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="bold" if i == 0 else "normal",
        )
        y -= 0.085


def _plot_bms(
    result,
    model_names=None,
    figsize=(9.0, 6.8),
    display=True,
):
    """Plot standard group BMS."""
    plt = _mpl()

    n_models = len(result.model_frequency)
    names = _model_names(n_models, model_names)

    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.55, 1.0],
            height_ratios=[1.0, 0.82],
            hspace=0.58,
            wspace=0.30,
            top=0.90,
            bottom=0.08,
            left=0.09,
            right=0.97,
        )

        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 0])
        axD = fig.add_subplot(gs[1, 1])

        if result.g is None:
            axA.axis("off")
            _panel(
                axA,
                "A",
                "subject model attribution",
                subtitle="posterior responsibilities unavailable",
            )
        else:
            _responsibility_matrix(
                axA,
                result.g,
                names,
            )

        _paired_bars(
            axB,
            result.model_frequency,
            result.exceedance_prob,
            names,
            title="model frequency and exceedance",
            protected=result.protected_exceedance_prob,
        )

        x = np.arange(n_models)
        axC.bar(
            x,
            result.posterior_parameters,
            color=_INK,
        )
        axC.set_xticks(x)
        axC.set_xticklabels(names)
        axC.set_ylabel("posterior Dirichlet α")
        axC.grid(axis="y", alpha=0.25)
        _style_axes(axC)
        _panel(
            axC,
            "C",
            "posterior model mass",
        )

        _summary_panel(
            axD,
            _summary_standard(result, names),
        )

        if display:
            plt.show()

        return fig

def _plot_btw_conds(
    result,
    model_names=None,
    figsize=(9.0, 6.2),
    display=True,
):
    """Plot between-condition BMS.

    Only the subject/model attribution matrix, the higher-level same-vs-
    different comparison, and a compact text summary are shown.
    """
    plt = _mpl()

    n_models = len(result.per_cond[0].model_frequency)
    names = _model_names(n_models, model_names)

    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.55, 1.0],
            height_ratios=[1.0, 0.72],
            hspace=0.62,
            wspace=0.30,
            top=0.89,
            bottom=0.08,
            left=0.09,
            right=0.97,
        )

        axA = fig.add_subplot(gs[0, :])
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[1, 1])

        gs_cond = [r.g for r in result.per_cond]

        if all(g is not None for g in gs_cond):
            matrix = np.vstack(gs_cond)
            n_subjects = gs_cond[0].shape[0]
            separators = [
                n_subjects * c
                for c in range(1, len(gs_cond))
            ]
            _responsibility_matrix(
                axA,
                matrix,
                names,
                title="model attribution across conditions",
                separators=separators,
            )
        else:
            axA.axis("off")
            _panel(
                axA,
                "A",
                "model attribution across conditions",
                subtitle="posterior responsibilities unavailable",
            )

        axB.bar(
            ["same", "different"],
            [result.xp, 1.0 - result.xp],
            color=[_INK_DARK, _INK_LIGHT],
        )
        axB.set_ylim(0.0, 1.03)
        axB.set_ylabel("exceedance probability")
        axB.grid(axis="y", alpha=0.25)
        _style_axes(axB)
        _panel(
            axB,
            "B",
            "same vs different across conditions",
            subtitle=(
                f"protected P(same) = {result.pxp:.3f} · "
                f"BOR = {result.bor:.3f}"
            ),
        )

        _summary_panel(
            axC,
            _summary_conds(result),
            letter="C",
        )

        if display:
            plt.show()

        return fig

def _plot_btw_groups(
    result,
    model_names=None,
    figsize=(9.0, 6.2),
    display=True,
):
    """Plot between-group BMS.

    Only the subject/model attribution matrix, the higher-level equality
    comparison, and a compact text summary are shown.
    """
    plt = _mpl()

    n_models = len(result.pooled.model_frequency)
    names = _model_names(n_models, model_names)

    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.55, 1.0],
            height_ratios=[1.0, 0.72],
            hspace=0.62,
            wspace=0.30,
            top=0.89,
            bottom=0.08,
            left=0.09,
            right=0.97,
        )

        axA = fig.add_subplot(gs[0, :])
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[1, 1])

        gs_group = [r.g for r in result.per_group]

        if all(g is not None for g in gs_group):
            matrix = np.vstack(gs_group)
            cumulative = np.cumsum(
                [g.shape[0] for g in gs_group]
            )[:-1]
            _responsibility_matrix(
                axA,
                matrix,
                names,
                title="model attribution across groups",
                separators=cumulative,
            )
        else:
            axA.axis("off")
            _panel(
                axA,
                "A",
                "model attribution across groups",
                subtitle="posterior responsibilities unavailable",
            )

        axB.bar(
            ["equal", "different"],
            [result.p_equal, 1.0 - result.p_equal],
            color=[_INK_DARK, _INK_LIGHT],
        )
        axB.set_ylim(0.0, 1.03)
        axB.set_ylabel("posterior probability")
        axB.grid(axis="y", alpha=0.25)
        _style_axes(axB)
        _panel(
            axB,
            "B",
            "equal vs different group profiles",
            subtitle=(
                "equality rejected"
                if result.h_reject_equality
                else "equality not rejected"
            ),
        )

        _summary_panel(
            axC,
            _summary_groups(result),
            letter="C",
        )

        if display:
            plt.show()

        return fig

# ═══════════════════════════════════════════════════════════════════
# Evidence provenance (DEV.md §5 promotion note)
# ═══════════════════════════════════════════════════════════════════
def check_evidence_provenance(fit_results) -> dict:
    """
    Check where the log-evidence in each FitResult came from before
    feeding it to bms_group* (DEV.md §5): evidence from the Mod 2
    eigenvalue-clip fallback (`hess_method == 'finite_diff_clipped'`,
    or any clipped eigenvalues) carries the §2.1 flat-direction
    artifact into every frequency/exceedance/PXP computed here.

    Args:
        fit_results: iterable of FitResult (one per model), as produced
            by individual_fit.

    Returns:
        Summary dict {model_index: {"methods": set, "n_clipped_total":
        int, "n_missing": int}}. Warns (never raises — the statistics
        are computable, just less trustworthy) when the GN path was not
        used everywhere or diagnostics are absent (pre-Mod-9 pickles).
    """
    summary = {}
    for m, fit in enumerate(fit_results):
        diags = getattr(fit.math, "diagnostics", None)
        if diags is None:
            warnings.warn(
                f"model {m}: FitResult has no diagnostics (fitted before "
                "Mod 9?) — evidence provenance cannot be verified")
            summary[m] = {"methods": set(), "n_clipped_total": 0,
                          "n_missing": None}
            continue
        methods = {d.hess_method for d in diags if d is not None}
        n_clip = sum(d.hess_n_clipped or 0 for d in diags if d is not None)
        n_missing = sum(1 for d in diags if d is None)
        if "finite_diff_clipped" in methods:
            warnings.warn(
                f"model {m}: evidence uses the finite-difference/clip "
                "fallback (Mod 2) — pass model_trials to individual_fit "
                "for Gauss-Newton evidence (optimization.py Mod 5, §2.1)")
        if n_clip > 0:
            warnings.warn(
                f"model {m}: {n_clip} eigenvalue(s) were clipped across "
                "subjects — the §2.1 evidence artifact is present")
        if n_missing:
            warnings.warn(
                f"model {m}: {n_missing} subject(s) fell back to the "
                "prior (no fit diagnostics)")
        summary[m] = {"methods": methods, "n_clipped_total": n_clip,
                      "n_missing": n_missing}
    return summary
