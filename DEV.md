# CBM-Python — Project Brief

**Repo:** `ginobattistello/cbm_python` (fork of `payampiray/cbm_python`)
**Working copy:** `~/Documents/` (local; may be ahead of GitHub)
**Staging folder:** `cbm/dev/` (was `bmc_TO_ADD/`, renamed for clarity) — as of 2026-08-03
holds **only** regression/verification harnesses, kept alongside the code they check
(`rl_jax_verify.py` for §2.3's JAX option;
`gn_numpy_verify.py` for §2.1/§2.3(C)'s NumPy Gauss-Newton curvature, done 2026-08-03;
`baseline_snapshot.py` + `baseline.json`, the §7 step 2/4 A/B regression check, 2026-08-03;
`convergence_status_verify.py` for §2.2's status enum (Mod 6), done 2026-08-03;
`checks_verify.py` for §3's Mods 1/7/8 + §4's three-layer checks (Mod 9 + pre-flight),
done 2026-08-03; `group_bms_verify.py` for §5's promoted `cbm/group_bms.py`, 2026-08-03;
`hbi_verify.py` for §14's Mods 11/12 in the hierarchical layer, 2026-08-13).
The last work-in-progress module (`group_bms.py`) was promoted to `cbm/`.
**Separately, `benchmark/`** holds workstream 5's three-arm robustness benchmark (§8) —
its own harnesses, the vendored pristine-CBM arm, and the MATLAB VBA driver; it is not
part of the `cbm` package. Local only for now; not yet pushed to GitHub.
**Reference:** MBB-team/VBA-toolbox (MATLAB)
**Guiding constraint:** simple, clear, transparent. Every change traceable and justified.

---

## 0. Scope

Workstreams, in dependency order:

1. **Optimization mechanics** — damping, Gauss-Newton, Hessian regularization (VBA-like)
2. **Convergence rules** — stopping criteria, monotonic-improvement checks
3. **BMC between groups and conditions** — already started, code to be supplied
4. **Manual** — prepare fitting / model fitting process / interpreting outputs
5. **Robustness benchmark** (added 2026-08-12, before the manual) — three-way comparison
   on simulated noisy data: this fork vs the pristine pre-fork CBM vs the MATLAB VBA
   toolbox. See §8.

Explicitly **out of scope**: turning CBM into VBA. CBM does Laplace-approximation MAP + HBI;
VBA does full Variational Bayesian inversion. We port *behaviors*, not the framework.
(§5's benchmark compares the two as they actually are — it does not merge them.)

---

## 0b. Remaining steps — consolidated TO-DO (added 2026-08-03)

Ordered by the dependency chain in §7. `[x]` done, `[ ]` open, `[~]` deferred by decision.
Section references point to the detail. This list is the single place to see what is left;
the prose sections below remain the source of truth for *how*.

### Done (for context — do not redo)
- [x] **§7 step 2 — Runnable baseline established (2026-08-03).** Two blockers fixed first:
  (a) all four `examples/*.py` imported `code.methods.utils.cbm_python.cbm.*`, a path from a
  different project tree — none of them could run, including `example_RL.py`, the script §7
  designates as *the* baseline; rewritten to `from cbm.*`. (b) the package is not pip-installed
  (`cbm_local.egg-info/` is a stale build artifact), so the examples now prepend the repo root
  to `sys.path` and run from any directory without an install step.
  `examples/example_RL.py` now runs end to end (~101 s, exit 0).
  **Baseline harness: `cbm/dev/baseline_snapshot.py` → `cbm/dev/baseline.json`.** Records
  per-subject MAP parameters, log-evidence, log-likelihood, `log|H|`, Hessian eigenvalues and
  flags, for all 4 blocks (RL / RL2 × Gauss-Newton / finite-diff-clipped), on the same 40
  subjects `example_RL.py` generates (seed 42, same call order). Verified bit-reproducible.
  Runs in ~9 s — far cheaper than the 101 s example, so it is the per-block regression check:
  - `python cbm/dev/baseline_snapshot.py --save cbm/dev/baseline.json` (before changes)
  - `python cbm/dev/baseline_snapshot.py --compare cbm/dev/baseline.json` (after each block;
    exit 1 + a per-field `max|delta|` report if anything moved beyond `--tol`, default 1e-8)
  HBI is deliberately excluded: slow, and its exceedance probabilities are Monte-Carlo, so it
  is a poor regression signal. Baseline totals, for reference — `sum(lme)`:
  RL/GN `-1963.067979`, RL/FD `-1966.443726`, RL2/GN `-1954.540346`, RL2/FD `-1966.836162`.
  (The GN-vs-FD gap of 3.4 / 12.3 nats across 40 subjects is §2.1's evidence contamination,
  measured at scale — it was 0.36 nat on the single subject in `gn_numpy_verify.py`.)
- [x] **§2.1 / §3 — Gauss-Newton curvature (Mod 5).** `H = JᵀJ + prior precision`, opt-in via
  `trial_func`; verified in `cbm/dev/gn_numpy_verify.py`. Resolves evidence contamination.
- [x] **§3 — Damping.** Investigated; VBA uses move-halving, already matched by `_newton_polish`
  backtracking. Nothing to implement (`H + λI` was a wrong instruction).
- [x] **§1 — Wiring.** `individual_fit.py` / `map_estimation.py` now import `optimization.py`;
  `optimization_legacy.py` frozen as the MOD 1-4 A/B baseline.
- [x] **§2.2 — Explicit convergence status (MODIFICATION 6), done 2026-08-03.**
  `ConvergenceStatus` (str-enum) in `cbm/optimization.py`, surfaced as
  `OptimizationResult.convergence_status`; `flag` now derives from the explicit
  `FLAG_FROM_STATUS` table (never a truthiness test — every member is truthy):
  `CONVERGED_DF` / `NO_IMPROVEMENT` / `MAX_STEPS` → 1.0, `SINGULAR_HESSIAN` → 0.5 + warn.
  `CONVERGED_GRAD` deliberately dropped (no code path; would be a permanently-dead member —
  if a gradient early-exit is ever wanted it is a new convergence path needing its own block
  and baseline diff). Mod 4d's flag branch superseded (its else-arm was dead code); full
  rationale lives in Mod 6's comment block, decision log in §2.2 below.
  **Verified:** `cbm/dev/convergence_status_verify.py` (16 checks — forces all four exit
  paths, incl. singular-Hessian end-to-end: flag 0.5, warning emitted, no prior substitution).
  **Baseline delta: zero** (`baseline_snapshot.py --compare` exit 0) — by design, every
  reachable path still maps to flag 1.0. Status distribution on the 40 baseline subjects
  (RL2/GN): 37 `converged_df`, 3 `no_improvement`, 0 `max_steps` — the boolean really was
  hiding structure (3 subjects where L-BFGS-B's point was already unimprovable).
  *Scope note:* the status lives on `OptimizationResult` only; `optimize_map`'s tuple and
  `FitMath` don't carry it yet — that surfacing belongs to §4's post-fit diagnostics layer,
  which lists "explicit convergence status (→ §2.2)" as one of its outputs.

### Done — workstream 2 complete (2026-08-03, second pass; verification in
### `cbm/dev/checks_verify.py`, 30 checks; **baseline delta: zero** on every block,
### end-to-end `example_RL.py` re-run exit 0 with all key numbers identical)
- [x] **§3 — MODIFICATION 1: activated** (decided activate, not delete: §4 pre-flight needs
  exactly these checks and Config is where `d` and both bounds first meet). Scalar→2×d
  expansion, shape rejection, and the `range_bounds ⊂ hard_bounds` §4 check now run in
  `Config.__post_init__` when `d` is given. `BFGSOptimizer.__init__` keeps its own expansion
  as second line of defence — deepcopy/unpickle do NOT re-run `__post_init__`, and HBI
  restores Configs from pickles (`profile.config`). Also fixed the `optimization.py`
  `__main__` demo, which predated the Config-based constructor and could not run.
- [x] **§3 — Monotonic objective as a checked invariant (MODIFICATION 7).** Two guards, both
  `raise RuntimeError` (violation ⇒ results untrustworthy ⇒ stop, per §4): at every
  `_newton_polish` exit (`f_return ≤ f_entry`) and at the `optimize()` boundary (polish never
  worsens the L-BFGS-B optimum). Note: the polish-internal guard is a tripwire for future
  edits — `f_current ≤ f_entry` is airtight by induction through the acceptance test
  (`f_new < f_current`), so only the boundary guard is externally testable (and tested).
- [x] **§3 — Sign/naming coherence (MODIFICATION 8).** The incoherence was worse than listed:
  `log_posterior`'s docstring claimed "negative log posterior (for minimization)" while the
  code returned the POSITIVE log joint (code right, docs wrong); `FitMath.loglik` actually
  stores the log JOINT, not the log-likelihood. Fixes: every objective callable renamed
  `neg_log_post` through `optimize`/`_newton_polish`/`compute_hessian`/`optimize_map` (the
  single negation lives in `optimize_map` and nowhere else); `OptimizationResult` gains
  `neg_log_post` (= `f`) and `F` (= `−f`) properties with a precision note (F is the log
  joint at the MAP, NOT the Laplace evidence — that adds `(d/2)log 2π − ½log|H|`). Field
  names `f`/`loglik`/`lme` kept — renaming would break pickles and user analysis code;
  docstrings now state exactly what each stores.
- [x] **§4 — Checks, three layers (MODIFICATION 9 + pre-flight in `individual_fit.py`).**
  - *Pre-flight* (`_preflight_checks`, runs once before fitting): empty data → raise; dim
    agreement (config.d vs prior, precision d×d) → raise; prior covariance PD (Cholesky —
    `inv()` succeeding does not imply PD) → raise; model evaluated at prior mean for EVERY
    subject (was: subject 0 only) — raises naming the subject if the model crashes (before,
    Mod 4a's defensive wrapper would silently turn every call into a 1e20 penalty and "fit"
    garbage), warns-and-continues if merely non-finite (random restarts may recover). Bounds
    checks live in Config (Mod 1 above). Data NaN inspection deliberately omitted: data is
    an opaque per-subject object, so evaluating the model IS the universal probe.
  - *Per-iteration*: monotonicity = Mod 7; solve-failure = Mod 6 (`SINGULAR_HESSIAN`);
    finiteness & hard-bounds containment are structural (acceptance test requires
    `isfinite`; every candidate is clipped) — documented rather than re-checked per step.
  - *Post-fit* (Mod 9, all informational, never alter the fit): `hess_raw_min_eig` +
    `hess_n_clipped` (raw spectrum before the Mod 2 floor — the §2.1 contamination
    diagnostic; 0 clipped on the GN path by construction), `n_inits_agreeing` (pre-polish
    optima within the same ΔF tolerance the polish uses — the practical multimodality test),
    `at_hard_bounds` (railed parameter ⇒ boundary MAP ⇒ Laplace invalid in that direction —
    the one §4 check that warns), plus `abs_g` (pre-existing) and the Mod 6 status.
  - *Forwarding* (closes §2.2's scope note): `optimize_map` now returns a 6th element (the
    full `OptimizationResult`); `FitMath.diagnostics` carries a per-subject
    `PostFitDiagnostics` dataclass (None for prior-substituted subjects); `hbi_updates.py`
    unpack updated (HBI receives but does not yet surface the diagnostics — noted for later).

### Done — workstream 3 complete (2026-08-03: `cbm/group_bms.py`, verified in
### `cbm/dev/group_bms_verify.py` (42 checks) against the ACTUAL VBA sources
### fetched from GitHub; baseline delta zero; `examples/example_group_bms.py` runs all
### three modes in ~2 s)
- [x] **All seven §5 items done.** Partition check (disjoint+exhaustive, shared by `a0` and
  `C`); family BOR from the family free energy via `compute_bor(L.T, …, C=C)` with the
  orientation transposes done deliberately; `dirichlet_exceedance` on `α_fam = Cᵀ·a`;
  `raise ValueError` everywhere (asserts gone); typed dataclasses (`GroupBMSResult` /
  `FamilyResult` / `WithinFamilyResult` / `BtwCondsResult` / `BtwGroupsResult` /
  `BestTuple`) with a `['key']`/`to_dict()` shim for the draft's dict access; explicit
  entry points `group_bms` / `group_bms_btw_conds` / `group_bms_btw_groups` (class with
  working `__init__` removed); module docstring with REFERENCES; promoted to `cbm/`,
  registered in `__init__.py`, worked example added. Provenance item shipped as
  `check_evidence_provenance(fit_results)` — inspects `FitMath.diagnostics` (Mod 9) and
  warns on Mod 2 fallback evidence, clipped eigenvalues, or prior-substituted subjects.
- [x] **VBA cross-check (sources fetched from MBB-team/VBA-toolbox@master).** Confirmed
  equal-per-family-then-normalize prior (`a0[k] = 1/(nf·|f(k)|)`, one TOTAL prior count —
  also why `group_bms` without families uses `1/K` per model, not `bms()`'s 1-per-model
  default); confirmed with families VBA's `out.bor` IS the family free-energy BOR
  (`VBA_groupBMC.m:228`), and btwConds' `pep = ep·(1−bor) + bor/2` is algebraically our
  `fam_pxp` — the deleted heuristic appears nowhere in VBA. Family frequencies computed
  exactly (`α_fam/Σα`, = `VBA_dirichlet_moments`) instead of the draft's MC estimate.
- [x] **Found & fixed: `fe_null` family branch had an axis bug** (`model_selection.py`).
  `np.sum(C, axis=1)` (per-model counts) cannot right-multiply the K×nf membership matrix —
  crashed for any K≠nf. Dead code until now (nothing ever passed `C`). VBA's original sums
  per FAMILY (`sum(options.C,1)'` = column sums) → `axis=0`. Verified against
  `VBA_groupBMC.m`'s `FE_null` line-by-line, plus invariants (C=I ⇒ F0f=F0m; equal-sized
  partition ⇒ F0f=F0m; unequal ⇒ differ).
- [x] **CORRECTION to this brief (§5 "already checks out" was wrong about between-groups).**
  The draft's between-groups tuple construction is NOT VBA's and does not work:
  (a) *empirical disproof* — a subject never informs the other group's tuple slot, so within
  each subject's tie-set the family-fair prior (each 'equal' tuple carries 2× the per-tuple
  mass of a 'not equal' one) is amplified by the rich-get-richer ψ(α) VB update into
  certainty: the harness measured **xp(equal) = 1.000 for groups favoring OPPOSITE models**;
  (b) *the reference disagrees* — `VBA_groupBMC_btwGroups.m` contains no tuples at all; it is
  a free-energy comparison: pooled fit (`Fe`) vs per-group fits (`Fd`),
  `p = 1/(1+exp(Fd−Fe))`, `h = p < .05`. Reimplemented exactly that (generalized to G ≥ 2);
  `BtwGroupsResult` carries `p_equal`/`h_reject_equality`/`F_equal`/`F_diff` + pooled and
  per-group fits. Sanity restored: same-model groups → p=0.916; opposite-model groups →
  p=6e-9, equality rejected. The between-CONDITIONS tuple construction is unaffected (every
  subject informs all slots — verified line-by-line against `VBA_groupBMC_btwConds.m` and
  directionally in the harness).

### Open — workstream 4 (§6, manual — write last, outline first)
- [ ] **a. Prepare fitting** — data format, model contract, priors, bounds, Config fields
  (consumes §4 pre-flight).
- [ ] **b. Model fitting process** — what the optimizer does step by step; where VBA behavior
  was adopted vs deliberately not (consumes §3, §4).
- [ ] **c. Interpret outputs** — every result field, meaning of the §2.2 status, when evidence
  is trustworthy, how to read BMS/HBI/BMC (consumes §2.1, §4 post-fit).

### Deferred by decision (not blocking)
- [~] **§2.3 A/B — JAX autodiff port + `jac=grad` in L-BFGS-B.** JAX side verified in
  `cbm/dev/rl_jax_verify.py` but not wired into the package. Ship NumPy first.
- [~] **§2.3 A→C — exact `jacfwd` Jacobian in Mod 5.** Swap the finite-difference Jacobian for
  an exact one later, without changing the `H = JᵀJ + prior` formula.

---

## 1. State of play (read from the pushed fork)

**[2026-08-03 update]** `cbm/optimization.py` (the annotated file, below) was **not actually
wired into the package** — `individual_fit.py` and `map_estimation.py` both imported
`optimization_legacy.py`, which silently carried the same MOD 1-4 logic without the
comments. Fixed: both now import from `optimization.py`. `optimization_legacy.py` is kept,
frozen, as the pre-Gauss-Newton (MOD 1-4 only) baseline for A/B comparison — it was never
actually the pristine pre-fork original (that only exists at commit `e72193f`), so treat it
as "baseline as of 2026-08-03", not "upstream original".

Five annotated modifications now exist in `cbm/optimization.py`:

| # | What | Status |
|---|------|--------|
| 1 | Bounds validation in `Config.__post_init__` (+ `range ⊂ hard` §4 check) | **Activated 2026-08-03** (was commented out) |
| 2 | Hessian eigenvalue regularization (floor 1e-4) | Active — fallback when no `trial_func` given (see Mod 5); clip now *reported* via Mod 9 |
| 3 | `_newton_polish` — Gauss-Newton refinement w/ backtracking | Active |
| 4 | Rewritten `optimize` method | Active |
| 5 | `_gauss_newton_curvature` — VBA-style `H = JᵀJ + prior_precision` | **Active, opt-in via `trial_func`** — resolves §2.1/§2.3, see below |
| 6 | `ConvergenceStatus` enum + `FLAG_FROM_STATUS` — explicit polish exit status, supersedes Mod 4d's flag branch | Active — resolves §2.2, see below |
| 7 | Monotonicity invariant — checked at polish exit and `optimize()` boundary, `RuntimeError` on violation | Active — resolves §3 monotonic item |
| 8 | Sign/naming coherence — `neg_log_post` naming; `neg_log_post`/`F` properties on `OptimizationResult` | Active — resolves §3 sign item |
| 9 | Post-fit diagnostics — raw min eig, clip count, cross-init agreement, hard-bound mask; `PostFitDiagnostics` forwarded to `FitMath.diagnostics` | Active — resolves §4 layer 3 (pre-flight lives in `individual_fit._preflight_checks`) |
| 10 | Weak-identifiability warning — smallest curvature eigenvalue vs prior precision, threshold 2× | Active — added 2026-08-12, see §11 |
| 11 | `model_trials` threaded `hbi_main` → `hbi_run` → `hbi_qhquad`, so HBI's internal refits can reach Mod 5's Gauss-Newton curvature | Active, opt-in — added 2026-08-13, see §14 |
| 12 | HBI keeps the per-refit `PostFitDiagnostics` it previously discarded, on `IndividualPosterior.diagnostics` | Active — added 2026-08-13, see §14 |

Mods 1–10 live in `cbm/optimization.py`; **11–12 live in `cbm/hbi.py` and
`cbm/hbi_updates.py`** — they concern the hierarchical layer, not the optimizer.

The `MODIFICATION n — WHAT / WHY / REFERENCE` comment convention is good. **Keep it.**
It is the single best transparency asset in the repo — the manual can be generated from it.

---

## 2. Three issues to settle BEFORE writing more code

These are design decisions, not bugs to patch blindly.

### 2.1 The regularized Hessian contaminates model evidence  ← RESOLVED 2026-08-03

`compute_hessian` clips all eigenvalues to `max(λ, 1e-4)`. This is correct and necessary
for *taking Newton steps*. But the same Hessian feeds the Laplace log-evidence:

```
log p(y|m) ≈ log p(y|θ*,m) + log p(θ*|m) + (d/2)·log(2π) − ½·log|H|
```

Clipping raises `log|H|` whenever the surface is flat. If model A has a flat direction and
model B does not, A is penalized by an amount that is an **artifact of the regularizer**,
not evidence. This propagates directly into BMS, HBI, and the group/condition BMC of
workstream 3 — i.e. into your actual scientific conclusions.

**Originally proposed resolution (superseded — do not implement):** split the two uses,
`compute_hessian(..., regularize=True/False)`. Checking the actual VBA source
(`C:\Users\Paul\Documents\VBA-toolbox`) before implementing this showed VBA doesn't have
this problem in the first place — it never needs two Hessians. See below.

**Actual resolution — Gauss-Newton curvature (MODIFICATION 5, `cbm/optimization.py`):**
`core/VBA_Iphi.m:100` / `core/VBA_Itheta.m:82` build the curvature as
`H = Σₜ Jₜᵀ Qₜ Jₜ + Σ₀⁻¹` (Jacobian outer-product sum + prior precision) — the classical
Gauss-Newton approximation for nonlinear least squares (drops the residual-weighted
second-derivative term of the exact Hessian, which is exactly the term responsible for
indefinite curvature). This sum is **positive-definite by construction** (sum of quadratic
forms + prior precision), so there is nothing to clip, and VBA reuses this *same* curvature
for both the Newton step *and* the evidence (`core/VBA_Hpost.m:56` → `core/VBA_FreeEnergy.m:132`
use `posterior.SigmaPhi`/`SigmaTheta`, the un-modified GN output) — no split needed. In a flat
direction, `JᵀQJ → 0` so `H → Σ₀⁻¹`: the posterior covariance falls back to the *prior*
covariance, not to an arbitrary constant like the old 1e-4 floor — this is what actually fixes
the evidence-contamination problem, not a reliability flag on top of the old clip.

Implemented as `BFGSOptimizer._gauss_newton_curvature` (opt-in via `trial_func`, a per-trial
log-likelihood callable — most models already compute this internally before summing it into
a scalar). `J` is obtained by a single finite difference per parameter (step rule copied from
`utils/VBA_numericDiff.m`: `1e-4·θ`, floored at `1e-4`) — not JAX yet, see §2.3. Falls back to
the old eigenvalue-clipped Hessian (Mod 2) when `trial_func` isn't supplied, so existing models
keep working unchanged.

**Verified** (`cbm/dev/gn_numpy_verify.py`, RL2 model, 100 trials, 1 subject): MAP parameters
match the old path to `1.8e-8` (`f` to `7e-15`) — as expected, GN and the old Hessian only
disagree on curvature, not on where the optimum is. But the curvature itself differs
meaningfully: eigenvalues `[0.29, 2.48, 26.0]` (old, clipped) vs `[0.75, 1.96, 26.0]` (GN),
and log-evidence differs by **0.36 nat** on this one subject — confirming this was a real,
not cosmetic, contamination of §5's BMS/HBI inputs. Backward compatibility (no `trial_func`)
re-verified via `individual_fit` on multiple subjects — unchanged behavior.

*This was the hinge between workstream 1 and workstream 3 — resolved before continuing either.*

### 2.2 `converged` currently carries no information  ← RESOLVED 2026-08-03

In `_newton_polish`, every exit path set `converged = True` — including the
"cannot reduce f" path and the exhausted-steps path. The flag was a constant.

**Originally proposed resolution:** replace the boolean with an explicit status, e.g.
`CONVERGED_DF` / `CONVERGED_GRAD` / `NO_IMPROVEMENT` / `MAX_STEPS` / `SINGULAR_HESSIAN`.
This *is* workstream 2 — a check that cannot report failure is not a check.

**Implemented — MODIFICATION 6 (`cbm/optimization.py`):** four-state `ConvergenceStatus`
str-enum (one member per real exit path of `_newton_polish`), surfaced as
`OptimizationResult.convergence_status`. Two deliberate deviations from the proposal above:

- **`CONVERGED_GRAD` dropped.** The polish never tests the gradient, so the member would be
  permanently unreachable — the same "inactive documented thing" problem as MODIFICATION 1.
  A gradient early-exit, if ever wanted, is a *new convergence path* (it can move where fits
  stop) and gets its own MODIFICATION block + baseline diff.
- **The coupled Mod 4d rewrite was mandatory, not optional.** Mod 4d's
  `if converged … elif converged … else flag = 0.5` had a dead else-arm; with an enum,
  `elif status:` stays truthy for every member (change would be inert) while an identity
  test on `CONVERGED_DF` would demote `MAX_STEPS`/`NO_IMPROVEMENT` to 0.5 and, transitively,
  risk prior substitution (`individual_fit.py:215-228`, and *unconditionally* in
  `hbi_updates.py:264-267` — HBI has no `prior_for_failed` opt-out). Hence the explicit
  `FLAG_FROM_STATUS` table: `CONVERGED_DF`/`NO_IMPROVEMENT`/`MAX_STEPS` → 1.0 (VBA accepts
  at max iter; monotonic descent means the point is never worse than L-BFGS-B's),
  `SINGULAR_HESSIAN` → 0.5 + warning (not 0.0 — never destroy a found MAP; near-unreachable
  anyway since both curvature paths are PD by construction).

**Verified** (`cbm/dev/convergence_status_verify.py`, 16 checks): all four exit paths forced
and correctly reported; forced-singular end-to-end gives flag 0.5 + warning and no prior
substitution; reachable paths keep flag 1.0. **Baseline delta: zero** (`--compare` exit 0).
On the 40 baseline subjects (RL2/GN): 37 `converged_df`, 3 `no_improvement` — information
the constant boolean was provably hiding. Status is not yet forwarded past
`OptimizationResult` (through `optimize_map`/`FitMath`); that surfacing is §4 post-fit work.

### 2.3 Replace finite-difference derivatives with autodiff / Gauss-Newton

**Problem (why this matters).** `compute_hessian` finite-differences an already
finite-differenced gradient: `(n+1)²` objective evaluations per Hessian (49 at d=6, 441 at
d=20), and its rounding noise is `~u·|f| / (ε_grad·ε) ≈ 2×10⁻⁶·|f|` — essentially the `10⁻⁴`
floor that Modification 2 clips to. Because that *same* Hessian sets the Laplace evidence
`−½·log|H|`, the noise doesn't just cost time, it biases §2.1. Cost and accuracy are one
problem, fixed by computing derivatives exactly instead of by differencing.

**Modifications to implement, in order:**

| # | What | Why | Status |
|---|------|-----|--------|
| A | Port each model log-likelihood to JAX | exact gradients at `O(1)` cost, zero differencing noise | **verified** (both RL models, `cbm/dev/rl_jax_verify.py`) — not yet wired into the package |
| B | Pass `jac=grad(loglik)` to the `L-BFGS-B` call in `_single_optimization` | removes the restart loop's dominant cost (repeated gradient differencing) | proposed — needs A wired in first |
| C | At the MAP, replace `compute_hessian` with Gauss-Newton `Σₜ ∇ℓₜ∇ℓₜᵀ + prior precision` | PSD by construction → the Mod 2 clip retires; matches VBA exactly (`core/VBA_Iphi.m:100`) | **DONE 2026-08-03**, NumPy version (single finite-difference Jacobian, not JAX) — see §2.1. `jax.hessian` (the other C option originally listed) was dropped: it is *not* PSD (verified indefinite even at the MAP in `rl_jax_verify.py`), so it doesn't actually solve §2.1 the way GN does — GN was always the right half of "C". |
| A→C via JAX | Swap the NumPy finite-difference Jacobian in Mod 5 for an exact one (`jax.jacfwd` on a JAX port of the per-trial log-lik) | removes the last `O(√ε)` finite-difference error from the now-PSD curvature | **future option**, deliberately deferred (2026-08-03 decision: ship NumPy first, ship correct, add exact derivatives later without changing the `H = JᵀJ + prior` formula). `cbm/dev/rl_jax_verify.py` already has the JAX side ready when this is picked back up — prefer `jacfwd` over the `jacrev` used there (d≪T for these models, forward-mode is O(d) passes vs reverse-mode's O(T)). |
| D | *Fallback only if a model can't be made differentiable:* central-difference the evidence Hessian, and reuse curvature across polish steps (quasi-Newton / LM damping) | `O(ε²)` accuracy at ~2× cost; avoids rebuilding `(n+1)²` Hessian up to 30× | superseded by Mod 2 acting as the fallback (already existed, now correctly framed as "fallback for non-trial-decomposable models" rather than the default path) |

**Porting recipe for (A) — three mechanical edits** (verified in `cbm/dev/rl_jax_verify.py`):
trial loop → `jax.lax.scan` (Q as carry); `Q[a] = …` → `Q.at[a].set(…)`; and the only edit
that bites, RL2's `if delta >= 0: … else: …` → `jnp.where(delta >= 0, alpha_pos, alpha_neg)`
(you cannot branch on a traced value). Smooth transforms and the stable softmax carry over
unchanged. Evidence it is the *same* model, not an approximation: log-lik matches NumPy to
`2×10⁻¹⁴`, gradient to `~5×10⁻¹⁰`; raw Hessian indefinite (`[-15.7, -0.35]`) vs Gauss-Newton
strictly positive (`[1.26, 17.1]`) — GN approximates `−∇²log-lik`, hence the positive spectrum.

**Two caveats for the port:**
- HBI's per-subject objective is *not* a plain trial-sum (it adds prior/responsibility terms):
  Gauss-Newton applies to the *likelihood* Hessian; add the prior's known precision on top.
- A genuinely non-differentiable op (a hard `argmax` instead of a softmax) needs a smoothed
  form or a custom derivative. Neither RL model has one — that is why they are the first
  test case. `rl_jax_verify.py` doubles as the regression check for A–C.

---

## 3. Workstream 1 — VBA-like optimization mechanics

Port these, each as its own `MODIFICATION` block:

- **Damping — CORRECTED 2026-08-03, do not implement `H + λI`.** Checked the actual VBA
  source: `grep -rniE "lambda|damp" core/` returns nothing. VBA does **not** inflate/relax a
  precision term. What it actually does on a rejected step (`core/VBA_GN.m:159-160`,
  `utils/VBA_GaussNewton.m:85-86`): `deltaMu = 0.5 * deltaMu` — halve the *proposed move* from
  the quadratic approximation and retry from the same point, i.e. backtracking on the Newton
  direction, not a Levenberg-Marquardt precision inflation. `_newton_polish`'s existing
  backtracking (halving `step` until `f` improves) is already closer to what VBA does than the
  `H + λI` description below ever was — **nothing to change here**, the brief's original text
  was wrong, not the code. (There *is* a real LM regularizer in VBA — `utils/VBA_checkGN.m` —
  but it acts on the covariance only when its smallest eigenvalue is ≤ 0, a rare-case safety
  valve with its own `flag`/count diagnostic, not a per-iteration damping loop. Irrelevant here
  since Mod 5's GN curvature is PD by construction and never triggers it.)
- **Gauss-Newton curvature — DONE 2026-08-03 (MODIFICATION 5, §2.1).** Build the stepping
  curvature as `Σₜ ∇ℓₜ∇ℓₜᵀ + prior precision` rather than a finite-difference Hessian. It is
  positive-definite by construction, confirmed on RL2 (`cbm/dev/gn_numpy_verify.py`) and on
  both RL models via JAX (`cbm/dev/rl_jax_verify.py`) — so the Mod 2 eigenvalue clip is now the
  fallback, not the default path. This is the same curvature VBA uses for both stepping and
  evidence (§2.1), which is why it was implemented together with §2.1 rather than separately.
- **Monotonic objective guarantee — DONE 2026-08-03 (MODIFICATION 7).** Never accept a step
  that worsens the objective. Was structural (backtracking); now *checked* at both the polish
  exit and the `optimize()` boundary, `RuntimeError` on violation. See §0b Done entry.
- **Sign/naming coherence — DONE 2026-08-03 (MODIFICATION 8).** Objective callables named
  `neg_log_post` throughout; `OptimizationResult.F = −neg_log_post` property at the boundary.
  Found and fixed en route: `log_posterior`'s docstring claimed the negative of what the code
  (correctly) returns, and `FitMath.loglik` stores the log joint, not the likelihood — both
  now documented; field names kept for pickle/user-code compatibility. See §0b Done entry.
- **MODIFICATION 1 — ACTIVATED 2026-08-03** (chose activate over delete: §4's pre-flight
  bounds checks needed a home and Config is it). Includes the `range ⊂ hard` containment
  check. See §0b Done entry.

## 4. Workstream 2 — checks  ← RESOLVED 2026-08-03 (see §0b Done entry for detail)

Three layers, each with a defined failure behavior (warn / flag / raise):

**Pre-flight (before any fitting)** — `individual_fit._preflight_checks` + Config (Mod 1)
- parameter dimension vs. prior dimension agreement → raise ✓
- `range_bounds` ⊂ `hard_bounds`, shapes are 2×d → raise (Config, Mod 1) ✓
- objective is finite at initialization (every subject, at prior mean; model crash → raise
  naming the subject, non-finite → warn); data non-empty → raise. Data NaN/Inf inspection
  deliberately delegated to the model probe — `data` is an opaque per-subject object. ✓
- prior covariance is positive-definite (Cholesky of precision) → raise ✓

**Per-iteration (invariants)**
- objective is finite at every accepted point — structural (acceptance requires `isfinite`) ✓
- objective is non-increasing (monotonicity) → checked, raise (Mod 7) ✓
- Hessian solve succeeded → `SINGULAR_HESSIAN` status, flag 0.5 + warn (Mod 6); "damping
  did not diverge" is subsumed — backtracking cannot diverge, it only shrinks the step ✓
- iterate stayed inside `hard_bounds` — structural (every candidate clipped); the *residual*
  risk, a MAP railed *on* the bound, is caught post-fit (`at_hard_bounds`, below) ✓

**Post-fit (diagnostics, surfaced in results — Mod 9, `FitMath.diagnostics`)**
- gradient norm at the optimum (`abs_grad`) ✓
- raw min eigenvalue + number clipped (→ §2.1): `hess_raw_min_eig`, `hess_n_clipped` ✓
- agreement across random initializations (`n_inits_agreeing`, pre-polish optima within the
  polish's own ΔF tolerance) — the practical multimodality test for the manual ✓
- explicit convergence status (→ §2.2) — forwarded through `optimize_map`'s new 6th return ✓
- (added) `at_hard_bounds`: railed parameter ⇒ boundary MAP ⇒ Laplace invalid there → warn ✓

**Principle:** a check never silently changes a result. It either flags it or stops.
Verified end to end in `cbm/dev/checks_verify.py` (30 checks); baseline delta zero.

## 5. Workstream 3 — BMC between groups and conditions  ← RESOLVED 2026-08-03

**Shipped as `cbm/group_bms.py`** (promoted from the `cbm/dev/group_bms.py` draft, which is
deleted). Three module-level entry points, no dispatch-in-constructor:
`group_bms` (2D `L`, optional families) = `VBA_groupBMC`,
`group_bms_btw_conds` (3D `L`) = `VBA_groupBMC_btwConds`,
`group_bms_btw_groups` (list of 2D `L`) = `VBA_groupBMC_btwGroups`.
Built on the existing `bms()` / `compute_bor` / `compute_fe` / `dirichlet_exceedance` in
`model_selection.py`. Verified in `cbm/dev/group_bms_verify.py` (42 checks); worked example
`examples/example_group_bms.py`. The subsections below are kept as the decision record —
they document *why* each change was made, and where this brief itself was wrong.

**Depended on §2.1 — resolved 2026-08-03.** Every input is a per-subject × per-model
log-evidence `L`. That `L` comes from the Laplace approximation, whose Hessian was
eigenvalue-clipped, so a flat-direction artifact would have been inherited by every
frequency, exceedance and PXP number here. §2.1 is now fixed (Gauss-Newton curvature,
Mod 5) *but only for models that pass `model_trials` into `individual_fit`/`optimize_map`* —
if group_bms.py is fed evidence from a fit that didn't opt in, it's still on the Mod 2
fallback. Check which path produced `L` before trusting these statistics (`hess_method` field
on the fit's `OptimizationResult`, or `cbm.math.hessian` provenance in `FitResult`).
**Shipped as `check_evidence_provenance(fit_results)`** — reads `FitMath.diagnostics` (Mod 9)
and warns on Mod 2 fallback evidence, clipped eigenvalues, or prior-substituted subjects.

### What checked out — and the one item that did NOT (corrected 2026-08-03)
- **Between-conditions** sums tuple log-evidence across conditions (`Lt`), then tests the
  "same model/family across all conditions" family against "not equal" — the correct
  within-subject (repeated-measures) construction. ✓ Confirmed line-by-line against
  `VBA_groupBMC_btwConds.m:112-116` (same `Lt(i,:) += L(Ci(j),:,j)` accumulation) and
  directionally in the harness. Kept as-is.
- ~~**Between-groups** stacks each group's per-model evidence into the tuple's group-slot
  (`Lt = vstack([Ls[g][:, tuples[:, g]] ...])`) so a subject only informs its own group's
  slot — the correct construction.~~ **← THIS BRIEF WAS WRONG. Do not restore it.**
  Two independent disproofs, found while writing the verification harness:
  - *Empirical.* Because a subject never informs the other group's slot, every tuple sharing
    that subject's own-slot model is tied in its likelihood. Within that tie-set the
    family-fair prior gives each 'equal' tuple 2× the per-tuple mass of a 'not equal' one
    (`1/(nf·|f|)` with |equal| = K but |not equal| = K²−K), and the rich-get-richer ψ(α) VB
    update amplifies that head start to certainty. Measured: **xp(equal) = 1.000 for two
    groups favouring OPPOSITE models** — the statistic cannot detect the difference it exists
    to detect.
  - *The reference disagrees.* `VBA_groupBMC_btwGroups.m` (fetched from
    MBB-team/VBA-toolbox@master) contains **no tuples at all**. It is a free-energy
    comparison of one pooled fit against per-group fits:
    `L = [Ls{1} Ls{2}]; Fe = out.F(end);` … `Fd = out1.F(end) + out2.F(end);`
    `p = 1./(1+exp(Fd-Fe)); h = p<.05;`
  Reimplemented exactly that, generalized to G ≥ 2 (VBA hardcodes 2). `BtwGroupsResult` now
  carries `p_equal` / `h_reject_equality` / `F_equal` / `F_diff` + the pooled and per-group
  fits — a different result shape from the other two modes, because it answers a different
  question (a hypothesis test, not an exceedance probability). Sanity restored: same-model
  groups → p = 0.916 (equality kept); opposite-model groups → p = 6e-9 (equality rejected).
  *Lesson for the remaining workstreams: "verified construction" in this brief means
  "read and believed", not "executed". Prefer a harness that can fail.*
- Result field names (`posterior_parameters`, `model_frequency`, `exceedance_prob`,
  `bor`, `protected_exceedance_prob`) already match `BMSResult`. Good — kept.

### Correctness to verify / flag — ALL DONE 2026-08-03
- [x] **Family prior leaves gaps.** In `_standard`, `a0` is filled only for indices listed in
  `families`; any model not in a family keeps `a0 = 0`, an invalid Dirichlet prior. Add a
  check that families **partition** all models (disjoint + exhaustive), or handle leftovers.
  → `_validate_partition()`, one function shared by the `a0` prior and the `C` matrix; raises
  naming the offending models for gap / overlap / empty family / out-of-range / bad names
  length. Matches VBA's own checks (`VBA_groupBMC_btwConds.m:64-81`).
- [x] **Family-level BOR/PXP: use the family-partition free energy, not the rescaling.**
  *Decision made — the heuristic is out.* Delete the block
  `lbf = log((1-bor)/bor) − n_sub·log(n_mod/nf)` → `fam_bor` entirely and compute the family
  BOR from the family free energy instead. **The toolbox already implements this**, so no new
  math is needed — reuse `model_selection.py`:
  - Build the family-membership matrix `C` (shape `n_mod × nf`, `C[k,f] = 1` iff model `k`
    is in family `f`). This requires families to partition all models — same validation as
    the `a0`-gap bullet below, so do it once and share it.
  - Call `compute_bor(L, posterior, priors, C=C)`. With `C` given, it takes the family branch:
    `fe_null(..., {'families': True, 'C': C})` returns the family null free energy `F0f`,
    and `bor = 1/(1 + exp(F1 − F0f))` is the correct family BOR (Rigoux et al. 2014, Eq. 5).
  - **Orientation caveat:** `compute_bor` / `compute_fe` / `fe_null` expect `L` as
    **models × subjects** and a `posterior` dict with `'a'` (Dirichlet counts) and `'r'`
    (responsibilities, `[model, subject]`), whereas `bms` / `group_bms` use
    **subjects × models**. Transpose deliberately; don't pass the wrong orientation silently.
  - `bms()` already calls `compute_bor` internally for the model level — copy that call
    pattern for the family level rather than inventing one.
  - Then `fam_pxp = fam_bor/nf + (1 − fam_bor)·fam_xp` stands, now on the *correct* `fam_bor`.
    For `fam_xp`, prefer the toolbox's `dirichlet_exceedance` on the family counts
    `α_fam = Cᵀ·a` over the ad-hoc sampler, for consistency with the rest of `model_selection.py`.
  - **Done exactly as specified.** Two additions found while implementing:
    (i) **`fe_null`'s family branch had an axis bug and had never run.** `f0 = C @ (np.sum(C,
    axis=1) ** -1) / C.shape[1]` sums per MODEL, shape (K,), which cannot right-multiply the
    K×nf matrix `C` — it raised `ValueError` for any K ≠ nf. Nothing had ever passed `C`, so
    the branch was dead code and the bug invisible. VBA sums per FAMILY
    (`f0 = options.C*sum(options.C,1)'.^-1/size(options.C,2)`; MATLAB `sum(·,1)` = column
    sums) → `axis=0`. Fixed and pinned with three invariants (C=I ⇒ F0f=F0m; equal-sized
    partition ⇒ F0f=F0m; unequal partition ⇒ they differ).
    (ii) **Family frequencies are computed exactly**, `α_fam/Σα` by Dirichlet aggregation
    (what `VBA_dirichlet_moments` does), replacing the draft's 1e6-sample MC estimate — same
    quantity, no sampling noise, ~free. Only the *exceedance* probability still needs MC.
    Invariant pinned: with C = I the family BOR equals the model BOR exactly. (Note the
    deleted heuristic also passes that particular case — its `n·log(K/nf)` correction
    vanishes when nf = K, which is presumably why it looked plausible. At nf=2, K=4, n=20 it
    is off by ~13.9 nats and pins the BOR near 1 regardless of data; that is the case the
    harness pins.)
- [x] **`assert` used for input validation** (`assert len(names) == nf`, tuple-count assert) is
  stripped under `python -O`. Use explicit `raise ValueError`. → done; the tuple-count assert
  is gone entirely (it restated `itertools.product`'s postcondition).

### Style/logic adaptation to CBM (the "fit the toolbox" part) — ALL DONE 2026-08-03
- [x] **Return dataclasses, not nested dicts.** → `GroupBMSResult` / `FamilyResult` /
  `WithinFamilyResult` / `BtwCondsResult` / `BtwGroupsResult` / `BestTuple`, all with a
  `_DictShim` base giving `result["ef"]`, `result["models"]["ef"]`, `.keys()` and
  `.to_dict()` so the draft's dict access keeps working (unknown key → `KeyError`).
- [x] **No work in `__init__`.** → three module-level functions; the `GroupBMS` class is gone.
- [x] **Add a module docstring with REFERENCES** (Rigoux et al. 2014; Stephan et al. 2009). →
  done, plus a promotion changelog and the provenance warning.

**Promotion — DONE 2026-08-03.** `cbm/group_bms.py` (the `cbm/dev/` draft is deleted),
exported from `__init__.py` (`group_bms`, `group_bms_btw_conds`, `group_bms_btw_groups`,
`check_evidence_provenance` + the six result dataclasses), worked example at
`examples/example_group_bms.py` covering all three modes (~2 s).

**Public API note for the manual (§6c):** the three modes deliberately return *different*
shapes, because they answer different questions. `group_bms` and `group_bms_btw_conds`
report exceedance/protected-exceedance probabilities; `group_bms_btw_groups` reports a
hypothesis test (`p_equal`, `h_reject_equality`, `F_equal`, `F_diff`) because that is what
VBA's between-groups routine actually is. Do not "harmonize" them.

## 6. Workstream 4 — manual

Sections map onto the above, so it should be written *last* but *outlined first*:

- **a. Prepare fitting** — data format, model function contract, priors, bounds, Config
  fields and their real effect. Consumes §4 pre-flight.
- **b. Model fitting process** — what the optimizer actually does, step by step;
  where VBA behavior was adopted and where it deliberately was not. Consumes §3, §4.
- **c. How to interpret outputs** — every field of the result object, what "converged"
  means, when evidence is trustworthy, how to read BMS/HBI/BMC. Consumes §2.1, §4 post-fit.

---

## 7. How to run this in Claude Code

1. Start in the repo root; confirm the local copy vs. GitHub (`git status`, `git diff`).
2. Establish a baseline **before touching anything**. *Done 2026-08-03 — see §0b.* The
   baseline is `cbm/dev/baseline.json`, produced by `cbm/dev/baseline_snapshot.py --save`;
   it captures per-subject parameters, log-evidence, `log|H|`, Hessian eigenvalues and flags
   for both models on both curvature paths. `examples/example_RL.py` still runs the full
   pipeline (incl. HBI) if an end-to-end sanity check is wanted, but the snapshot is what
   changes get diffed against — 9 s vs 101 s, and it compares numbers instead of eyeballed logs.
   Keep `cbm/dev/rl_jax_verify.py` alongside it — it already pins the autodiff port to the
   NumPy models numerically, so it doubles as the regression check when wiring §2.3 (A)–(C) in.
   (Note: JAX is **not installed** in the current environment, so `rl_jax_verify.py` cannot run
   as-is. Not blocking — every open item is NumPy-side — but §2.3 A/B needs `pip install jax`.)
3. Work **one MODIFICATION block at a time**, in the order: ~~§2.2~~ → §2.1 → §3 → §4 → §5.
   The §2.3 autodiff/Gauss-Newton work slots into §2.1+§3 (it supplies the exact/PSD Hessian
   both depend on); do the JAX port of the model likelihood before those two blocks.
   **[2026-08-03: order changed — did §2.1+§3's Gauss-Newton curvature (Mod 5) before §2.2,
   since it was already fully scoped after the VBA cross-check and §2.2 doesn't depend on it.
   §2.2 done later the same day (Mod 6, baseline delta zero). §3 remainder + §4 done the same
   day (Mods 1/7/8/9 + pre-flight; baseline delta zero on every block; end-to-end example_RL
   re-run identical). Workstreams 1 and 2 are COMPLETE.
   2026-08-03, later: workstream 3 (§5) COMPLETE too — `cbm/group_bms.py` promoted, with a
   correction to this brief's between-groups claim (see §5) and a latent `fe_null` axis-bug
   fix in `model_selection.py`. **Next: the §6 manual — the only workstream left.**]**
4. After each block: run `python cbm/dev/baseline_snapshot.py --compare cbm/dev/baseline.json`,
   and record the delta in the block's comment. Exit 0 means nothing moved; exit 1 prints the
   per-field `max|delta|`. If a change moves the numbers, that must be explained, not just
   observed — and only then is the baseline re-saved (`--save`) with a note saying why.
   Done for Mod 5: see §2.1's verification note and `cbm/dev/gn_numpy_verify.py`.
5. Only then write the manual, drawing text from the MODIFICATION blocks.

**Non-negotiables:** no silent corrections; no undocumented change; keep a frozen baseline
for A/B comparison — **as of 2026-08-03 this is `optimization_legacy.py`** (MOD 1-4 only, pre-
Gauss-Newton), not the pristine upstream original (that's commit `e72193f` only). Do not edit
`optimization_legacy.py` going forward; `cbm/optimization.py` is now the live file (imported by
`individual_fit.py`/`map_estimation.py` — this wasn't true before 2026-08-03, see §1).
---

## 8. Workstream 5 — Robustness benchmark (started 2026-08-12)

**Question.** How robust is this toolbox on noisy simulated data, and how does it compare
with (a) the MATLAB VBA toolbox and (b) the original pre-fork CBM?

**Everything lives under `benchmark/`.** Not part of the `cbm` package; nothing here is
imported by the toolbox.

| file | role |
|---|---|
| `simulate.py` | ground-truth generator; writes each cell as **both** `.npz` (Python arms) and `.mat` (VBA arm) |
| `models.py` | the RL/RL2 likelihoods, defined once so both Python arms fit the identical objective |
| `run_python_arms.py` | arms `fork_gn` (this fork, Gauss-Newton) and `cbm_orig` (pristine) |
| `run_vba_arm.m` | arm `vba`; MATLAB, headless via `matlab -batch` |
| `matlab/f_QlearnAsym2.m` | CBM's two-sigmoid RL2 written for VBA (see "comparability" below) |
| `analyze.py` | metrics + report → `results/report_<grid>.md`, `results/summary_<grid>.csv` *(removed 2026-08-13 — superseded by `make_report_figures.py`, §10)* |
| `external/cbm_original/` | pristine CBM, extracted verbatim from commit `e72193f` (**tracked**; do not edit) |
| `external/VBA-toolbox/` | shallow clone, **gitignored**; re-create per `external/README.md` |

**Environment (verified 2026-08-12).** MATLAB R2025b at
`/Users/gino.diez/Applications/Matlab_R2025b.app`, licensed, runs headless. VBA inverts an
RL model in ~2.5 s/subject. The pristine CBM arm needs nothing external — commit `e72193f`
is in this repo, and DEV.md §7 already identified it as the only pristine copy.

**Reproduce:**
```
python benchmark/simulate.py                     # or --quick
python benchmark/run_python_arms.py --grid grid
matlab -batch "grid='grid'; run('benchmark/run_vba_arm.m')"
python benchmark/analyze.py --grid grid
```

### Design
Grid: generator {RL, RL2} × n_trials {30, 100, 300} × beta {0.5, 1, 3, 8}, 60 subjects per
cell, plus four degenerate cells (perseverative, alpha≈0, alpha≈1, 15 trials) = 28 cells /
1680 subjects. Every cell is fitted with **both** candidate models by **every** arm, which
is what makes the model-recovery confusion matrix possible. Seeds come from `blake2b` of the
cell identity — deliberately **not** builtin `hash()`, which is per-process randomized and
would silently give each arm different data.

Metrics: parameter recovery (bias / RMSE / Pearson r vs truth, on correctly-specified fits;
beta on the log scale since it is fitted as log-beta), model recovery (confusion matrix from
each arm's own evidence), and robustness (non-finite estimates, railed parameters, clipped
eigenvalues, convergence status, runtime).

### Comparability — what is and is not a fair comparison
Three differences are real and are handled explicitly rather than papered over:

1. **Inference procedure.** CBM does Laplace-MAP; VBA does variational Bayes on a
   state-space model with the Q-values as hidden *states*. The VBA arm therefore fixes
   evolution to be deterministic (`priors.a_alpha = Inf`, `b_alpha = 0`) so VBA's recursion
   is CBM's recursion. Without this VBA would fit a strictly richer model.
2. **Evidence scale.** VBA's `out.F` is a variational free energy; CBM's `log_evidence` is a
   Laplace approximation. Comparable in *kind*, not in value. Therefore **model selection**
   (which model wins) is compared, never absolute nats — each arm is scored against its own
   evidence, which is how each toolbox is used in practice.
3. **RL2 parameterization.** VBA's own `f_QlearningAsym` uses `sigmoid(P1 + sign(δ)·P2)` —
   a *different model family* from CBM's two independent sigmoids; it cannot represent every
   (α_pos, α_neg) pair. Comparing it to CBM's RL2 would compare different models and blame
   the toolbox. `benchmark/matlab/f_QlearnAsym2.m` implements CBM's exact form for VBA;
   analytic gradients checked against finite differences (max err 2.5e-9) and verified to
   reduce exactly to VBA's own `f_Qlearn` when both rates are equal. VBA's file is untouched.

### Calibration finding (2026-08-12) — the first grid measured the simulator, not the toolboxes
The initial RL2 truth (α_pos ~ U(0.5,0.85), α_neg ~ U(0.1,0.4); mean gap 0.42) produced a
mean RL2-vs-RL evidence gap of **−0.05 nats** at T=100: the Laplace complexity penalty for
the third parameter almost exactly cancelled the likelihood gain, so model recovery sat near
chance **for all three arms**. That all three agreed is the tell — it was the simulation, not
the fitters. Direct check with a strong asymmetry (0.85 / 0.10) confirmed the evidence
machinery is sound: +2.6 nats and 92% correct at T=100, +20.3 nats and 100% at T=1000.
`draw_true_params` now enforces `MIN_ASYMMETRY = 0.35`, and `N_SUBJECTS` went 40 → 60 so a
~10-point accuracy difference between arms is not noise. **Do not weaken either constant
without re-checking the evidence gap** — a benchmark that cannot separate the models cannot
rank the toolboxes.

### Results — full grid, 2026-08-12 (`benchmark/results/report_grid.md`)
28 cells × 60 subjects × 2 fitted models × 3 arms = **10 080 fits, zero failures**, all
estimates finite in every arm. Runtime per subject: fork 0.33 s, original CBM 0.38 s,
VBA 0.54 s. Wall clock: Python arms 49 min, VBA arm 30 min.

**1. §2.1's central claim is confirmed at scale.** Paired per subject (same data, same
seeds, same objective — the only difference is the curvature):

| quantity | mean Δ (fork − CBM) | fits that differ |
|---|---:|---|
| MAP `alpha_pos` | 0.0014 | 14 / 3360 (0.4 %) |
| `log\|H\|` | −1.016 | — |
| log-evidence | +0.507 | **3360 / 3360 (100 %)** |

Exactly as predicted in §2.1: Gauss-Newton moves the **evidence**, not the optimum. The 14
MAP differences are multimodality, not a curvature bug — in each case where the fork found
the *better* optimum (higher log-joint) Mod 9's `n_inits_agreeing` was 2-3/5, i.e. the
diagnostic correctly flagged those surfaces as unreliable; where all 5 inits agreed the
differences are numerical ties (|Δ log-joint| < 1e-3).

**2. Model recovery — the fork tracks VBA exactly; original CBM discriminates better.**
Degenerate cells excluded; AUC is of the evidence gap `lme(RL2) − lme(RL)`, threshold-free.

| arm | RL-recall | RL2-recall | balanced acc | **AUC** |
|---|---:|---:|---:|---:|
| this fork (Gauss-Newton) | 0.707 | 0.543 | 0.625 | **0.6631** |
| original CBM (pre-fork) | 0.779 | 0.519 | 0.649 | **0.6929** |
| MATLAB VBA | 0.706 | 0.551 | 0.628 | **0.6631** |

Two things to read carefully here:

- **Raw accuracy is a trap and is labelled as such in the report.** Original CBM's higher
  headline number (0.677 vs 0.639) is substantially a *complexity bias*: it prefers the
  simpler model more often (RL-recall 0.779 vs 0.707) and pays for it on RL2-generated data
  (0.519 vs 0.543). Because the grid holds more RL cells than RL2, that bias inflates its
  average. Balanced accuracy narrows the gap (0.649 vs 0.625) but does not close it.
- **The fork reproduces MATLAB VBA to four decimals (0.6631 vs 0.6631)** on the
  threshold-free metric, at every trial count. That is strong independent evidence the
  Gauss-Newton port is faithful — the fork now behaves like the reference implementation.
  **But CBM's eigenvalue-clipped Hessian genuinely discriminates better on this benchmark
  (AUC 0.693).** Mechanism: the fork raises the 3-parameter model's evidence by **+0.13 nats
  relative** to the 2-parameter model versus CBM, i.e. it is *less* conservative about
  complexity. That is a real, measured trade-off, not a defect: §2.1's argument was that the
  clip's penalty is an artifact of a tuning constant rather than evidence, and that argument
  still holds — but on this grid the artifact happens to help. **Do not "fix" this by
  reverting Mod 5**; the honest statement is that GN is more principled and CBM's clip is
  better calibrated *for this model pair at these sample sizes*. Worth revisiting with a
  model pair whose flat directions are more pronounced.

**3. Robustness.** No arm produced a single non-finite estimate, railed parameter, clipped
eigenvalue, or `flag < 1` anywhere in the grid, including all four degenerate cells. On the
degenerate cells all three arms degrade *gracefully and near-identically* (median |bias| in
alpha within 0.03 of each other), except `degen_alpha0` where the fork is modestly better
(0.135 vs CBM 0.162 vs VBA 0.194). Convergence status (fork, Mod 6): 87.4 % `converged_df`,
12.6 % `no_improvement`, 0 % `max_steps` / `singular_hessian`.

**Open follow-ups (not blocking the manual):**
- The AUC gap deserves a second model pair before drawing a general conclusion — RL/RL2
  differ by one parameter with a strong asymmetry, which is a mild test of flat directions.
- HBI is not in the benchmark yet (individual fits only). The evidence shift measured here
  propagates into HBI's responsibilities, so an HBI arm is the natural extension.
- The fork's `n_inits_agreeing` proved its worth (it flagged exactly the multimodal fits);
  worth surfacing in the manual's interpretation section as the recommended triage field.

### Figures (2026-08-12) — *superseded, see §10*
`benchmark/make_figures.py` → `results/figdata_<grid>.json` → `results/figures_<grid>.html`
(self-contained page, light/dark, with a table view of every plotted number).
Published: https://claude.ai/code/artifact/c21fcc57-6757-4544-83b2-a8b8acdc86e4

*Removed 2026-08-13.* The HTML/JS renderers (`make_figures.py`, `build_report.py`,
`make_clean_figures.py`, `report_template.html`, `analyze.py`) were replaced by a
single matplotlib script producing publication-layout figures — see §10.

Six panels. Three were requested; three were added because the requested set could
not answer questions a reader immediately asks:

| panel | what it shows |
|---|---|
| a | parameter estimates — **error** (est − truth) beside raw, per arm |
| b | parameter-recovery matrices, estimated × true, per arm |
| c | model-recovery confusion + bias-corrected metrics |
| d | evidence shift: fork vs CBM, paired |
| e | recovery vs the stress axes (trials, β) |
| f | degenerate cells + the fork's per-fit diagnostics |

Three deliberate deviations from the figures as first specified, each for a reason:
- **(a) boxes the error, not the raw estimate.** True values vary per subject by
  design, so a raw boxplot mostly renders the prior spread and all three arms look
  identical (they nearly are — the MAP differs on 0.4 % of fits). Boxing est − truth
  makes box centre = bias and box width = variance. The raw panel is kept beside it.
- **(b) is a recovery matrix, not a correlation matrix of estimates.** Recovery is a
  relation between two vectors (est, true), not a square matrix. Correlating every
  estimated parameter against every true one gives a real matrix whose diagonal is
  recovery and whose off-diagonal is parameter trade-off — the conventional figure.
- **(c) ships the bias-corrected metrics alongside the confusion matrices.** The
  confusion matrices alone invite the "original CBM is better" misread that §8's
  results section warns about.

**New finding surfaced by (b) — worth carrying into the manual.** The learning rates
barely recover at all: on RL2, r(est α_pos, true α_pos) ≈ −0.01 and
r(est α_neg, true α_neg) ≈ 0.21, while β recovers strongly (r ≈ 0.89). The α
estimates correlate *more* with true β (+0.15 / −0.31) than with their own true
value. **All three arms show this to two decimals**, so it is the model and the task,
not the software — a genuine identifiability limit of 2-armed-bandit RL at these
trial counts. §6c should say plainly that a recovered learning rate from this design
is not trustworthy at the individual level, and that β is the parameter this task
actually identifies.

---

## 9. Benchmark round 2 — analysis answers, design corrections, value models (2026-08-12)

Round 1 (§8) raised six questions. Answering them exposed **two design flaws in my own
benchmark** and one **real bug in the original CBM**. Recorded here because several of
round 1's numbers were measuring the simulation, not the toolboxes.

### 9.1 Why every estimate distribution is shifted vs the truth box (fig a)
**Prior shrinkage, working as designed.** The N(0,10) prior pulls estimates toward its
mean (α = 0.5, β = 1). The proof is RL2, where the two learning rates shift in *opposite*
directions — α_pos −0.154 (true median 0.76), α_neg +0.089 (true median 0.18) — both
toward 0.5. A systematic error would shift them the same way.

β is a different mechanism: it shifts **up regardless of true value** (+0.16), but the
bias vanishes with data (T=30 +0.157, T=100 +0.166, **T=300 −0.004**). That is
finite-sample bias in a log-scaled parameter, not a defect. Both are now stated as the
section conclusion in the figure page.

### 9.2 Why RL α recovery is only ~0.50, and RL2's is near zero
- **RL's 0.50 is a pooling artifact.** Per cell it runs from **+0.16** (T=100, β=0.5) to
  **+0.91** (T=300, β≥3). Pooling averages cells where recovery is physically impossible
  with cells where it is excellent.
- **RL2's near-zero recovery was caused by `MIN_ASYMMETRY`** — the constant added in
  round 1 to make model recovery detectable. It narrows the α ranges, and correlation is
  bounded by true variance. Widening the ranges (0.05–0.95):

  | | round-1 ranges | wide ranges |
  |---|---:|---:|
  | α_pos | +0.57 | **+0.87** |
  | α_neg | +0.60 | **+0.91** |

  **The two benchmark goals are in direct conflict** — separation-for-selection narrows
  the ranges that recovery needs. Round 1 optimized one and silently sacrificed the
  other. Fixed by splitting into two sub-grids (§9.4).
- **Model recovery is trial-count limited, not toolbox limited.** At β=3 with strong
  asymmetry: T=100 → 0.70/0.62, T=300 → 0.93/0.90, **T=600 → 0.97/1.00**.

### 9.3 What the log-evidence shift means (fig d)
The fork's +0.5 nat shift is **almost entirely a level shift, not a discrimination gain**:
+0.480 nats on correctly-specified fits vs +0.504 on misspecified ones — a **−0.024
differential**. The shift tracks *parameter count* (+0.43 for 2-param RL, +0.56 for
3-param RL2), i.e. Gauss-Newton systematically reduces the complexity penalty. That is
precisely why its AUC sits slightly below CBM's: a uniformly smaller Occam factor moves
every model's evidence without separating them better.

**Fork vs MATLAB VBA, paired across 2880 fits** (the comparison §8 lacked): median
|Δα| = **0.0001**, median Δ evidence = **0.005 nats**, corr(α) = **0.978**. Two different
inference procedures — Laplace-MAP and variational Bayes — landing on the same answer is
the strongest evidence yet that the port is faithful.

### 9.4 Design correction — two sub-grids per question
Because separation and recovery pull the ranges in opposite directions, each question now
gets the design that can answer it (`--grid` on `benchmark/simulate.py`):

| grid | intent | ranges |
|---|---|---|
| `grid` | round-1 RL, selection-tuned (kept; §8's published figures rest on it) | MIN_ASYMMETRY 0.35 |
| `rl_wide` | RL/RL2 parameter recovery | α ∈ 0.05–0.95 |
| `value_recovery` | LIN/POW parameter recovery | ρ ∈ 0.30–1.70 (spans, and includes, ρ≈1) |
| `value_selection` | LIN/POW model recovery | ρ ∈ 0.35–0.60 (far from the ρ=1 nesting point) |

### 9.5 Value-function models (neuroeconomics)
Risky choice: sure amount *s* vs gamble (*g* with probability *p*).
`U(sure) = v(s)`, `U(gamble) = p·v(g)`, softmax on the difference.

- **LIN** — `v(x) = x`, risk-neutral. 1 parameter (β).
- **POW** — `v(x) = x^ρ`, CRRA. 2 parameters (ρ, β). ρ<1 concave = risk averse.

**Nested at ρ = 1**, verified exact to machine precision (0.00e+00) in both Python and
MATLAB. That nesting is what makes the pair a clean selection test: LIN-generated data
lies inside POW's parameter space, so preferring POW is purely a complexity question.

**Task-design calibration — the amount scale matters more than anything else.** `x^ρ` is
nearly flat for x<1, so amounts on a 0–1 scale leave ρ weakly identified and estimates run
away (observed ρ up to 31.5). Measured ρ recovery (T=100 / T=300):

| amounts | r |
|---|---|
| 0.05–1 | 0.60 / 0.82 ← the original, bad choice |
| **0.5–10** | **0.92 / 0.97** ← adopted (`AMOUNT_SCALE = 10`) |
| 5–100 | 0.67 / 0.79 |

VBA arm: `benchmark/run_vba_value.m` + `benchmark/matlab/g_valuePower.m`. These are
**static** models (`dim.n = 0`, no evolution function), which makes this the cleaner of
the two comparisons — with no state-space machinery involved, VBA and CBM differ only in
how they approximate the posterior. Analytic gradients verified against finite differences
(POW 2.8e-10, LIN 8.9e-11) and the ρ=1 nesting reproduced exactly.

### 9.6 A real bug in the original CBM, found by the value models
`cbm_original/optimization.py:394` crashes with
`AttributeError: 'NoneType' object has no attribute 'x'` when every initialization returns
a non-finite objective: `best_result` is never assigned and the retry loop has no guard.
On the value grids this cost the pristine arm **360 of 1080 POW fits** on
`value_recovery` (6 whole cells) and 60 on `value_selection`. **The fork completed all
4320 fits** — MODIFICATION 4a's defensive wrapper (non-finite → 1e20 penalty) is exactly
what prevents it. This is the first case in the benchmark where a fork modification
changes whether a fit happens at all, rather than by how much.

### 9.7 Why the "selection" grids show poor recovery — the design trade-off, quantified
Asked why RL-selection and Value-selection look worse than Value-recovery. Two things came
out of checking, one of which corrects the premise.

**Premise correction: Value-selection has the BEST model recovery of any grid** (AUC
0.946 vs 0.870 recovery / 0.663 RL-selection). Only its *parameter* recovery collapses
(ρ r = 0.159). The pattern is not "selection grids are worse" — it is a clean trade-off:

| grid | primary-param recovery | model recovery (AUC) |
|---|---:|---:|
| RL · selection | α 0.55 / RL2 α_pos −0.01 | 0.663 |
| RL · recovery (`rl_wide`) | α 0.62 / RL2 α_pos 0.44, α_neg 0.60 | 0.623 |
| Value · recovery | ρ **0.86** | 0.870 |
| Value · selection | ρ 0.16 | **0.946** |

**Mechanism — recovery is capped by true spread, not by fitting accuracy.**
`r ≈ SD_true / sqrt(SD_true² + SD_error²)`. The estimation error is nearly identical
across the two value grids; only the spread differs (6×):

| grid | true SD | error SD | predicted r | observed r |
|---|---:|---:|---:|---:|
| Value · recovery | 0.408 | 0.224 | 0.877 | 0.864 |
| Value · selection | 0.070 | 0.277 | 0.245 | 0.159 |

Value-selection samples ρ ∈ [0.35, 0.60] *on purpose*, to hold it away from the ρ=1
nesting point where LIN and POW become the same model. **Separating the models requires
clustering the parameters, and clustered parameters cannot be told apart from each other.**
Same fits, same accuracy — different question. Value-selection's ρ = 0.16 is near its
ceiling (~0.25 for a perfect estimator at that spread), not a failure.

**Second mechanism, RL2 only: shrinkage correlated with truth.** RL2's α_pos comes in
*below* even the spread prediction (−0.01 vs +0.41 predicted) because the formula assumes
independent noise. Measured `corr(error, true α_pos) = −0.455`: the higher the true value,
the harder the prior pulls it down, which cancels exactly the variation the correlation
would detect. In Value-recovery the same quantity is only −0.115 — identical absolute
shrinkage removes a far smaller *fraction* of a 6× wider spread. RL2 suffers both effects
at once. Parameter trade-offs were checked and are NOT the cause (all estimate-estimate
correlations |r| < 0.30).

**Correction to a prediction I made.** I said `rl_wide` would lift RL2 α recovery to
"roughly 0.87". Pooled, it reached 0.44 (α_pos) / 0.60 (α_neg). The 0.87 came from a
single favourable cell; `rl_wide` pools all 12. In the *matching* cell (T=300, β=3) it
does reach **0.80 / 0.94**, so the prediction was right for that condition and wrong as a
pooled claim. Per-cell α_pos on `rl_wide` ranges −0.09 (T=30, β=1) to +0.80 (T=300, β=3).
**Never quote a pooled recovery number without the cell breakdown** — this is the same
pooling trap that made RL's α look like 0.50 in §9.2.

**For the manual (§6c).** A low recovery correlation in a narrow-range study does not mean
the fitter failed. Report the *estimation error* (SD, design-independent) next to the
correlation (design-dependent), and never pool across cells that differ in identifiability.

---

## 10. The clean cross-arm benchmark (2026-08-12) — the one to re-run

§8-§9 were exploratory: many grids, many stress axes, several design corrections.
This section is the **consolidated regression check** that replaces them for routine
use. Procedure: `benchmark/RERUN.md`.
Output: `benchmark/results/figures/` — five figures (PDF + PNG) and a generated
`figures.md` caption sheet, all from `benchmark/make_report_figures.py`.

**Design.** Two generating models, one cell each, 120 subjects × 200 trials:
- **RL** — 2-armed bandit, single learning rate. θ = (α, β). α ∈ [0.05, 0.95],
  β ∈ [0.8, 8] log-uniform.
- **POW** — non-linear value function v(x)=x^ρ on risky choice. θ = (ρ, β).
  ρ ∈ [0.30, 1.70], β ∈ [0.3, 3] log-uniform.

Both with **10% lapse trials** (`LAPSE_RATE`): on one trial in ten the agent ignores
the model and responds at random. This is the right noise for *choice* models — there
is no residual to perturb as there would be in regression — and it is a genuine
misspecification stress, since neither fitted model has a lapse parameter. (Jittering
the true parameters was considered and rejected: it keeps the model perfectly
specified and only widens the spread.)

Each dataset is still fitted with both candidates of its family (RL vs RL2, POW vs
LIN) because §6's AUC needs an evidence gap; the confusion-matrix figure was dropped.

**Reference values — compare any future run against these.**

| section | quantity | fork | CBM | VBA |
|---|---|---:|---:|---:|
| 2 | RL recovery α / β | 0.818 / 0.900 | 0.818 / 0.900 | 0.818 / 0.902 |
| 2 | POW recovery ρ / β | 0.943 / 0.773 | 0.944 / 0.773 | 0.943 / 0.773 |
| 3 | cross-arm, worst off-diagonal | **0.9999** (any pair, any parameter) | | |
| 6 | AUC | 0.6215 | 0.6338 | 0.6223 |
| — | failed fits (of 960) | 0 | 1 | 0 |

**Main message: the three implementations agree essentially perfectly.** The weakest
cross-arm correlation anywhere is **r = 0.9999**. Two arms share a codebase, but
MATLAB VBA does not — and it uses variational Bayes rather than Laplace-MAP — so this
is a real cross-implementation check, not a shared-code artifact. Section 3 is the
number to watch: a drop there means the fork has diverged from *both* references.

Secondary findings, all consistent with §8-§9:
- **Section 1**: the three arms are biased *identically* (medians within 0.003). The
  residual offsets are prior shrinkage plus the lapse contamination — properties of
  the estimator and the data, not of any implementation.
- **Section 5**: across 239 paired fits the MAP moved on **15**, the evidence on
  **239**. Gauss-Newton is an evidence-scale change, which is exactly why sections 2
  and 3 are untouched by it.
- **Section 6**: AUC spread across arms is 0.012 — the arms discriminate equally well.
  The fork's evidence shift is a level change, and a level change cannot move a
  threshold-free metric.
- **The 1 CBM failure is the upstream crash from §9.6**, now isolated: the runner fits
  the CBM arm subject-by-subject, so one pathological subject costs one subject rather
  than a whole cell of 120. Without that fallback the entire CBM arm vanished from the
  POW figures.

---

## 11. MODIFICATION 10 — weak-identifiability warning (2026-08-12)

**Why it exists.** Probing extreme scenarios (see §12) found a gap in Mod 9: on RL data
with α at the boundary, recovery collapsed to *worse than chance* (r = −0.18) while
`at_hard_bounds` and `hess_n_clipped` both stayed at **0** for every fit. The boolean
flags are tuned for catastrophic failure (railed, singular); they say nothing about a
*gradual* loss of identifiability. But the continuous diagnostic did see it —
`hess_raw_min_eig` fell from 4.56 (α=0.5) to 0.138 (α=0.001), a 33× collapse.

**The quantity.** `weak_identifiability = hess_raw_min_eig / min(eig(prior_precision))`,
warned on below `WEAK_IDENTIFIABILITY_RATIO = 2.0`.

**Why a ratio, not an absolute floor.** `min_eig` is log-posterior curvature, so it grows
with data. Measured on RL at α=0.5: T=50 → 1.44, T=150 → 5.46, T=450 → 14.6. An absolute
threshold would flag every small study and miss every large one. Dividing by the prior
precision removes the scale *and* gives the number a meaning: how much more the data know
than the prior did. Empirically the ratio is also the better predictor —
corr(log ratio, |error|) = **−0.38** vs **−0.27** for the raw eigenvalue.

**Threshold calibration** (n = 360 fits, RL α ∈ 0.001–0.999 × T ∈ 60/150/300):

| ratio band | median abs error | fits with abs error > 0.15 |
|---|---:|---:|
| 0–2 | **0.206** | **53 %** |
| 2–5 | 0.045 | 12 % |
| 5–20 | 0.066 | 25 % |
| 20–100 | 0.051 | 7 % |

At ratio < 2: precision 0.53, recall 0.48.

**Honest limits, also stated in the code.** Precision ≈ 0.5 means about half the flagged
fits are fine — this is a **triage signal, not a rejection rule**, which is why it warns
rather than raises (§4: a check flags or stops; stopping would discard usable fits).
Recall ≈ 0.5 means it also misses half the bad fits: a fit can be wrong while curvature
stays healthy, typically through multimodality, which `n_inits_agreeing` covers instead.
The two diagnostics are complementary, not redundant.

**Scope.** Computed only on the Gauss-Newton path, where `prior_precision` is known
exactly and enters H by construction. On the Mod 2 fallback the eigenvalues have already
been clipped, so the raw spectrum is not comparable to a prior scale — the field is None
there rather than misleading.

**Verification.** Behaves as designed: median ratio 37.9 with 0 warnings when
well-identified (α=0.5), 1.10 with 8/10 warnings at the boundary (α=0.001). All four
harnesses pass and **`baseline_snapshot --compare` is unchanged** — Mod 10 is purely
additive, it computes and reports without touching any fit.

---

## 12. Boundary stress investigation (2026-08-12)

**Question.** How do the arms behave when parameters approach or exceed the edges of
their range — and does any diagnostic *predict* the failure?

**Scope, chosen after a screening probe.** Candidate stressors were probed before
committing to a grid, and the probe overturned the intuition: **skewed predictors are
nearly harmless** (POW ρ recovery 0.933 → 0.927 with lognormal amounts), while
**parameters near a boundary are destructive** (ρ ∈ [0.02, 0.15] → r = −0.18, worse than
chance). So the grid walks each model's own parameter across its range with everything
else fixed: `--grid boundary`, 18 cells × 40 subjects × 150 trials, no lapse (so the
boundary effect is not confounded with contamination).

RL α ∈ {0.001 … 0.999}, POW ρ ∈ {0.02 … 4.0}. Each cell **fixes** the parameter, so
per-cell recovery correlation is undefined — read |bias| and the diagnostics instead.

### Recovery degrades, but asymmetrically
Median |bias| in the native parameter (this fork):

| RL α | 0.001 | 0.01 | 0.05 | 0.5 | 0.95 | 0.999 |
|---|---:|---:|---:|---:|---:|---:|
| |bias| | **0.414** | 0.105 | 0.019 | 0.090 | 0.055 | 0.048 |

| POW ρ | 0.02 | 0.08 | 0.5 | 1.3 | 2.5 | 4.0 |
|---|---:|---:|---:|---:|---:|---:|
| |bias| | 0.173 | 0.189 | 0.098 | 0.039 | 0.060 | **0.480** |

The failures are at α → 0 and ρ → 4, not symmetric about the range. α → 1 is *fine*
(0.048) because the sigmoid saturates: a large error in the fitted θ maps to a tiny
error in α. ρ → 4 fails because choices saturate (gamble fraction 0.97), leaving no
information about curvature.

### The diagnostic gap that prompted MODIFICATION 10
`at_hard_bounds` and `hess_n_clipped` were **0 in all 18 cells**, including the ones
with 8× the baseline error. The Mod 9 booleans detect catastrophic failure, not gradual
loss of identifiability. Mod 10 (§11) was added to close that gap.

### Validating Mod 10 — and a correction to my own first analysis
Scored as a predictor of "this fit is bad", AUC, this fork only:

| diagnostic | RL (in-sample) | POW (**out of sample**) |
|---|---:|---:|
| **weak_identifiability (Mod 10)** | **0.824** | **0.766** |
| n_inits_agreeing | 0.503 | 0.206 |
| abs_grad | 0.543 | 0.271 |

Flagged fits (ratio < 2) carry **4.7×** the median θ-error on RL (4.08 vs 0.86) and
**5.2×** on POW (0.47 vs 0.09). The threshold was calibrated on RL alone, so POW at
0.766 is genuine out-of-sample transfer.

**The correction:** my first pass scored these against error in the *native* parameter
(α, ρ) and got AUC 0.510 — chance — which I initially read as "Mod 10 doesn't work".
That target was wrong. The curvature refers to the **unconstrained (θ) space**, and
native error is not a monotone function of θ error: at α → 1 the sigmoid saturates, so a
badly-identified θ still yields an α close to truth. Against θ-space error the same data
give 0.824. Recorded because the wrong target produced a plausible-looking null result:
**a diagnostic must be validated in the space it describes.**

Practical reading, now stated in the code: Mod 10 means *"this parameter is poorly
constrained"*, not *"this number is far from the truth"*.

### Harness bug found and fixed
`run_python_arms.py` was not copying `weak_identifiability` into its result rows, so the
first boundary analysis showed NaN everywhere. Fixed with `getattr` so older pickles
predating Mod 10 still load. Worth noting the failure mode: the toolbox was correct and
the *measurement* harness was silently dropping the field.

---

## 13. Is HBI still valid? (2026-08-13)

**Question.** The examples and the whole §10 benchmark validate `individual_fit`.
HBI is a separate code path that the benchmark never touches. Does it still work,
and do the ten modifications reach it?

### 13.1 How HBI couples to the modified code

Two entry points, and the distinction between them turns out to be the whole story:

| Path | Where | Gauss-Newton reachable? |
|---|---|---|
| the `cbm_map` files the user supplies | consumed at `hbi.py:379` | **yes** — the user chose when they called `individual_fit` |
| HBI's own per-iteration refit | `hbi_updates.py:264` | **no** — calls `optimize_map` with 6 positional args; `model_trials` is the 7th parameter and is never passed |

So HBI's internal refits are **always finite-difference**, regardless of how the
input MAPs were produced. The supplied fits act only as (a) the initialization
`cfg.inits` for the refit, and (b) the source of `logdetA` for the first bound.

### 13.2 Does the curvature choice change HBI's answer? No.

Held the supplied MAP fixed and swapped ONLY the curvature (`log_det_hessian`,
`hessian_inv_diag`). POW-vs-LIN, 60 subjects:

| supplied | model_frequency [POW, LIN] |
|---|---|
| GN map + GN curvature | 0.996416, 0.003584 |
| GN map + **FD** curvature | 0.996928, 0.003072 |

A 137-nat difference in summed `log_det_hessian` moves the group frequency by
5e-4. **This is correct behaviour, not a bug**: HBI refits every subject on its
first iteration under the group prior, so the supplied curvature is overwritten
before it can influence anything but the initial bound.

### 13.3 But the MAP choice changes it completely — and that IS the finding

Same comparison, letting the supplied MAP differ as it naturally would:

| supplied | model_frequency [POW, LIN] | verdict |
|---|---|---|
| GN map + GN curvature | **0.996**, 0.004 | POW wins |
| FD map + FD curvature | 0.294, **0.706** | LIN wins |

Opposite conclusions from identical data. The cause is not the curvature (13.2
rules that out) but the MAP: on 11 of 60 subjects the GN and FD optima differ,
one of them by 2.88 in θ.

**The GN fits are better on every affected subject**, by log-evidence:

| subject | θ (GN) | θ (FD) | LME GN | LME FD |
|---|---|---|---|---|
| 50 | [0.51, −1.12] | [0.74, −4.00] | **−103.86** | −114.75 |
| 58 | [0.53, −1.01] | [−0.00, −0.16] | **−92.31** | −135.00 |
| 41 | [0.27, 0.51] | [0.00, 0.00] | **−76.66** | −103.86 |
| 31 | [0.26, 0.45] | [0.00, 0.00] | **−85.63** | −107.87 |

The FD path is landing at or near the prior mean `[0, 0]` — it is failing to
find the optimum, not finding a different one. GN recovers up to 43 nats.

### 13.4 What this means

1. **HBI runs and converges.** `examples/example.py` completes; 4 iterations,
   protected exceedance probabilities produced.
2. **HBI is valid, but it does not inherit MOD 5.** Its internal refits use the
   old finite-difference curvature. Nothing is broken — this is the pre-fork
   behaviour, unchanged.
3. **The quality of the supplied MAPs matters a great deal**, because they seed
   every refit and HBI is evidently sensitive to that seed on hard subjects.
   Fitting with `model_trials` before calling HBI is therefore worth doing even
   though HBI discards the curvature.
4. **`n_inits_agreeing` flagged the trouble**: 58 of 60 subjects had fewer than 3
   of 3 initializations agreeing, and the worst subject had 0. Mod 9's
   multimodality diagnostic is doing its job on exactly the fits that go on to
   destabilise the group result.

### 13.5 Gaps — HBI is the least-verified part of the toolbox

- `cbm/dev/baseline_snapshot.py` **excludes HBI by design** ("it is slow and
  ...") — so the regression baseline that protects `individual_fit` does not
  protect HBI at all.
- No cross-arm benchmark arm for HBI; §10 covers `individual_fit` only.
- `optimize_map`'s 6th return value (the `OptimizationResult` with all the Mod 9
  diagnostics) is discarded at `hbi_updates.py:264` — the comment there says
  "not yet surfaced in HBI's own result structures".
- `protected_exceedance_prob` came back NaN in the two-model probe (it is
  produced by `hbi_null`, run separately); worth checking whether that is
  expected or a gap.

### 13.6 Suggested next steps, in priority order

1. **Pass `model_trials` through HBI** so its refits can use Gauss-Newton. It is
   a plumbing change (thread the argument from `hbi_main` to `hbi_qhquad`), and
   §13.3 shows the MAP quality genuinely changes group conclusions.
2. **Add HBI to the baseline harness**, even at reduced size — a small fixed
   dataset, a recorded `model_frequency`, an exact-match assertion.
3. **Surface the diagnostics** currently thrown away at `hbi_updates.py:264`,
   at minimum `n_inits_agreeing` aggregated per model.
4. Add an HBI arm to the cross-arm benchmark against VBA's `VBA_MFX`.

Probes for all of the above: scratchpad `hbi_probe{,2,3,4}.py` (not committed —
they are one-shot investigations, reproducible from this section).

---

## 14. MODIFICATIONS 11 & 12 — HBI reaches Gauss-Newton, and reports on itself (2026-08-13)

Both answer §13. Mod 11 closes the functional gap, Mod 12 closes the reporting gap.
Neither changes any default behaviour: `model_trials=None` — what every existing
caller passes implicitly — is verified bit-identical to the pre-modification code.

### 14.1 MODIFICATION 11 — `model_trials` threaded into HBI

**Why it exists.** `hbi_qhquad` called `optimize_map` with six positional arguments.
`model_trials` is the seventh, so HBI's internal refits used the Mod 2
finite-difference Hessian unconditionally — no matter how the supplied `cbm_map`
files had been produced. Since HBI refits every subject on its first iteration under
the group prior, a user who deliberately fitted with `model_trials` had that work
silently discarded.

**The change.** A `model_trials` parameter on `hbi_main`, carried through
`user_input` → `hbi_run` → `hbi_qhquad` → `optimize_map`. One entry per model,
aligned with `models`; any entry may be None for a model that cannot expose a
per-trial decomposition, and that model alone keeps the fallback.

| File | Change |
|---|---|
| `cbm/hbi.py` | `hbi_main` parameter + length validation; `hbi_run` reads it with `.get()`; forwarded at the `hbi_qhquad` call |
| `cbm/hbi_updates.py` | `hbi_qhquad` parameter; `mt_k` selected per model; passed as `optimize_map`'s 7th argument |

**Why `.get()` and not `["model_trials"]`** in `hbi_run`: `user_input` dicts are
deep-copied into every iteration's `math` record and pickled. Indexing would break
on any HBI result written before this modification.

### 14.2 MODIFICATION 12 — HBI surfaces its per-refit diagnostics

**Why it exists.** The sixth return value of `optimize_map` — the
`OptimizationResult` carrying all the Mod 9/10 diagnostics — was discarded, with a
comment admitting it ("not yet surfaced in HBI's own result structures"). HBI was
the only path in the toolbox with no fit-quality reporting at all.

**The change.** A `(K, N)` object array on `IndividualPosterior.diagnostics`, one
`PostFitDiagnostics` per (model, subject) refit, None where the optimizer reported
none. The field is `Optional` with a None default so pre-Mod-12 pickles still load —
HBI results are routinely saved and re-read by `hbi_null`.

**Implementation note worth keeping.** `diagnostics` is a **method** on
`OptimizationResult`, not an attribute — it builds the record on demand and copies
the arrays. The first implementation used `getattr(res, "diagnostics", None)`, which
silently stored the *bound method*; nothing failed until something tried to read
`.hess_method` off it, several layers away. Check 4c in the harness pins this
specific mistake.

### 14.3 Verification

**Backward compatibility — the one that matters.** Reference recorded by checking out
the pre-Mod-11 files (`git stash`) and calling the old six-argument `hbi_main`:

```
REF_DEFAULT_FREQ = [0.5564341971439771, 0.4435658028560228]
```

Post-modification, the default path reproduces it with **max delta 0.000e+00**. This
is a genuine pre-modification value, not a post-hoc recording of current output.

**Mod 11 does what it claims** (30 subjects × 120 trials, POW vs LIN, 10% lapse):

| supplied maps | HBI refits | model_frequency [POW, LIN] | refit `hess_method` |
|---|---|---|---|
| FD | FD (old default) | 0.556, 0.444 | `finite_diff_clipped` ×60 |
| GN | FD | 0.939, 0.061 | `finite_diff_clipped` ×60 |
| GN | **GN (Mod 11)** | **0.940**, 0.060 | `gauss_newton` ×60 |
| FD | **GN (Mod 11)** | **0.941**, 0.059 | `gauss_newton` ×60 |

**The last two rows are the result.** Under FD refits the verdict depends on how the
*supplied* maps were fitted — spread **0.382** between rows 1 and 2. Under GN refits
it does not — spread **0.0004**, a 950× reduction. Mod 11 does not merely change
which answer you get; it removes the dependence on an arbitrary upstream choice.

This reproduces §13.3 on independently generated data (different seed, different
subject count, self-contained generator in the harness), so it is not an artefact of
the one dataset that first exposed it.

**Regression suite.** All five harnesses pass and the baseline is unchanged:

| Harness | Result |
|---|---|
| `hbi_verify.py` (new, 12 checks) | 12 passed, 0 failed |
| `convergence_status_verify.py` | PASS |
| `checks_verify.py` | PASS |
| `group_bms_verify.py` | PASS |
| `gn_numpy_verify.py` | PASS |
| `baseline_snapshot.py --compare` | **UNCHANGED** (tol 1e-8, all fields) |
| `examples/example.py`, `examples/example_RL.py` | both run clean |

### 14.4 `cbm/dev/hbi_verify.py` — the new harness

Closes the §13.5 gap that `baseline_snapshot.py` excludes HBI. That exclusion was
half right: exceedance probabilities are Monte-Carlo and make a poor regression
signal, but `model_frequency` is deterministic given the data and the initial fits,
and pins perfectly well.

Twelve checks: backward compatibility against the hard-coded reference (1); the
Gauss-Newton path is actually reached and not merely requested (2a-b); the
seed-dependence defect exists under FD and is gone under GN (3a-c); diagnostics are
present, populated, real records rather than bound methods, and carry the Mod 9
fields (4a-d); input validation (5); pre-Mod-12 construction still works (6).

Self-contained — it generates its own data rather than reading `benchmark/data/`, so
it runs anywhere in well under a minute. Regenerate the reference only with
`--record`, and only when deliberately changing default behaviour: if check 1 fails,
the default path has moved, which is precisely what Mod 11 promised would not happen.

### 14.5 What is still open

- **Recommendation for users unchanged in substance, but now easier to follow.**
  Pass `model_trials` to `hbi_main` for any model that can expose per-trial
  log-likelihoods. Without it HBI silently uses finite differences.
- Mod 12 stores the diagnostics but nothing yet *aggregates* them — no per-model
  summary of `n_inits_agreeing` or weak-identifiability counts in the printed HBI
  output. That is a reporting layer, best designed alongside the manual (§6).
- Still no HBI arm in the cross-arm benchmark (against VBA's `VBA_MFX`). §10 remains
  `individual_fit`-only.
- `protected_exceedance_prob` returned NaN in the §13 two-model probe. It is produced
  by `hbi_null` run separately, so this is most likely correct usage rather than a
  defect, but it has not been confirmed.

---

## 15. HBI benchmark — this fork vs pristine CBM (2026-08-13)

Answers "what did Mods 11-12 actually buy?" with a measurement rather than the
single-dataset probe of §13.3.

### 15.1 Why two arms and not three

§10's design has three arms, the third (MATLAB VBA) being the independent check
that stops the two CBM-lineage arms from drifting together unnoticed. **That does
not transfer to the hierarchical layer**, and the reason is worth recording rather
than quietly working around:

`VBA_MFX` fits **one model at a time** and returns a group-level free energy. CBM's
HBI fits all K models jointly with a Dirichlet over model identity and reports how
the population divides between them. There is no `model_frequency` in `VBA_MFX` to
correlate against. A third arm would mean comparing incommensurable quantities, so
this benchmark has two and says so in the figure captions.

The two arms are:

| arm | module | refit curvature |
|---|---|---|
| `cbm_orig` | `cbm/hbi_legacy.py` | finite difference (no `model_trials` parameter exists) |
| `fork_gn` | `cbm/hbi.py` | Gauss-Newton (MOD 11) |

The HBI variational update equations are **byte-identical** between them — `diff`
shows only the Mod 11/12 plumbing. So any difference measured is attributable to
the curvature and nothing else. This is a stronger control than §10 has.

### 15.2 The frozen legacy snapshot

`cbm/hbi_legacy.py`, `hbi_updates_legacy.py`, `hbi_types_legacy.py` — copies of
commit `93a0be8`, the HBI analogue of `optimization_legacy.py`. Nothing in the
package imports them.

**Their internal imports are rewired to each other.** Without that,
`hbi_legacy.py` would `from .hbi_updates import hbi_qhquad` and silently call the
MODIFIED refit — it would not be legacy behaviour at all. Verified: the frozen
path reproduces `hbi_verify.py`'s pinned pre-Mod-11 reference at delta 0.000e+00.

### 15.3 Grid design — mixed populations

`benchmark/simulate.py --grid hbi`, 10 cells, 40 subjects x 150 trials, 10% lapse.

Every other grid here draws all subjects from one generator. That is useless for
HBI: the group frequency is pinned at ~1 and cannot move. Each cell instead mixes
the two candidates of a family at a known ratio — 0 / 30 / 50 / 70 / 100 % from the
complex model — and records per-subject `true_model` as ground truth.

The **50 and 70 % cells carry the experiment**: that is where a small change in the
individual fits can flip the group verdict. The 0 and 100 % cells are controls that
should be stable under either arm.

### 15.4 The two analyses

**1. Stability under seed perturbation.** HBI refits every subject *starting from
the supplied individual fits*, so those fits are a seed. Rerunning your own fitting
pipeline with different random restarts gives an equally defensible seed, and a
trustworthy group inference should not depend much on which one it got. Each cell
is run from 4 such seeds; the measured quantity is the **spread** (max − min) in
the recovered group frequency.

**2. Convergence behaviour.** Iterations to converge, free-energy bound at exit,
wall-clock cost, and the fraction of refits flagged weakly identified. The last is
only available for the fork — `cbm_orig` discards those records — and that absence
is itself one of the differences under test.

### 15.5 Design correction — the first perturbation axis was wrong

**Recording this because the first version of the benchmark measured nothing,
and the failure looked exactly like a clean result.**

The plan was to perturb HBI's seed by varying the random restarts inside
`individual_fit` — same `num_init`, different `np.random.seed` — on the
reasoning that this is the variation a user gets by rerunning their pipeline.

The first 20 runs came back with all four seeds agreeing to four decimals
(0.4367, 0.4367, 0.4367, 0.4367). That is not stability; it is a dead
instrument. Measured directly: changing the restart seed moves the supplied
MAPs by **~1e-5**. The multi-start optimizer converges to the same optimum
regardless of where it starts, so there was nothing for HBI to be sensitive
to. Four seeds were costing four HBI runs each to re-measure the same number.

What actually moves the MAPs is the **curvature used to fit them**, which is
also a choice a real user makes — whether they pass `model_trials` to
`individual_fit`. Per-cell, GN vs FD map difference:

| cells | max abs difference in theta | subjects moved > 0.01 |
|---|---:|---:|
| all RL cells (RL and RL2) | ~1e-5 | 0 / 12 |
| VALmix000 / 030 (POW) | 1e-3 – 5e-3 | 0 / 12 |
| VALmix050 (POW) | 0.52 | 3 / 12 |
| **VALmix070 (POW)** | **2.72** | **6 / 12** |
| VALmix100 (POW) | 0.82 | 3 / 12 |

So the divergence is **entirely in the POW model** — the RL cells are
genuinely insensitive and would have shown nothing under any perturbation.
The grid did contain the phenomenon; the axis was pointed the wrong way.

The corrected design runs each cell from BOTH map sources and crosses that
with the two arms — 4 combinations per cell instead of 8, so it is also
cheaper. Confirmed on VALmix070 before committing to the sweep:

| arm | supplied maps | freq(complex) | iterations |
|---|---|---:|---:|
| legacy | GN | 0.9123 | 10 |
| legacy | **FD** | **0.3583** | **2** |
| fork (Mod 11) | GN | 0.9140 | 10 |
| fork (Mod 11) | FD | 0.9121 | 11 |

Spread **0.554** for the legacy arm versus **0.002** for the fork. Note the
legacy FD run also terminated in 2 iterations against 10 — it did not merely
land elsewhere, it quit early at a worse optimum.

**The lesson generalises**: before trusting a null result from a perturbation
experiment, verify the perturbation actually perturbs. Related to the §9.7
lesson about pooled numbers — both are cases where a plausible-looking
measurement was measuring the harness rather than the thing.

### 15.6 Implementation notes worth keeping

- **Iteration count is not on the result object.** The per-iteration `prog` list is
  local to `hbi_run` and never returned; `HBIMath` has no `prog` field. It *is* in
  the `.log` file `hbi_run` opens alongside `fname`, and that is written regardless
  of `verbose` because `hbi_log` writes to the file handle independently of the
  console flag. The runner counts "Iteration" lines there — reading HBI's own
  record rather than inferring.
- **Final bound** is `res.math.bound.bound.L` (last iteration only, not a
  trajectory).
- Figure 2B plots the bound as a **within-cell difference** because the absolute
  bound varies by hundreds of nats between cells, which would hide the effect.
- Figure 1C exists because a large recovery offset would visually drown a small
  stability difference if only A/B were shown.

### 15.7 Files

| File | Role |
|---|---|
| `benchmark/simulate.py --grid hbi` | 10 mixed-population cells |
| `benchmark/run_hbi_arms.py` | both arms x 4 seeds -> `results/hbi_hbi.pkl` |
| `benchmark/make_hbi_figures.py` | 2 figures + generated captions |
| `cbm/hbi_legacy.py` + 2 siblings | frozen pre-fork snapshot |

Procedure in `benchmark/RERUN.md`. Results reported in §15.8 below once the
sweep completes.

### 15.8 Results (2026-08-13, 40 runs)

**Sensitivity to how the supplied maps were fitted** — |freq(GN maps) −
freq(FD maps)|, per cell. Reported per cell, never pooled: the RL cells are
controls where GN and FD maps are nearly identical, and averaging them in
would halve the headline for no reason.

| cell | CBM gn | CBM fd | **spread** | fork gn | fork fd | **spread** |
|---|---:|---:|---:|---:|---:|---:|
| RLmix000 | 0.0005 | 0.0005 | 0.0000 | 0.0005 | 0.0005 | 0.0000 |
| RLmix030 | 0.4161 | 0.4367 | 0.0205 | 0.4488 | 0.4700 | 0.0212 |
| RLmix050 | 0.7033 | 0.7088 | 0.0055 | 0.7005 | 0.7068 | 0.0063 |
| RLmix070 | 0.7983 | 0.8306 | 0.0323 | 0.7984 | 0.8034 | 0.0050 |
| RLmix100 | 0.9976 | 0.9976 | 0.0000 | 0.9973 | 0.9973 | 0.0000 |
| VALmix000 | 0.2544 | 0.2556 | 0.0013 | 0.2616 | 0.2628 | 0.0013 |
| VALmix030 | 0.4375 | 0.4412 | 0.0038 | 0.4280 | 0.4318 | 0.0038 |
| **VALmix050** | 0.5469 | 0.2981 | **0.2487** | 0.5464 | 0.6706 | **0.1242** |
| **VALmix070** | 0.9123 | 0.3583 | **0.5540** | 0.9140 | 0.9121 | **0.0018** |
| **VALmix100** | 0.9999 | 0.5819 | **0.4180** | 0.9999 | 0.9999 | **0.0000** |

**The three value cells where GN and FD maps genuinely differ are the whole
result.** On VALmix070 the legacy arm's verdict swings from 0.91 to 0.36 —
from "the complex model dominates" to "it does not" — purely on an upstream
choice the user may not know they made. The fork's swing on the same cell is
0.0018. VALmix100 is the same story: 0.42 versus exactly 0.

Everywhere else both arms are stable, and the fork is never meaningfully
worse (RLmix030: 0.0212 vs 0.0205, a difference of 7e-4 that is not a finding).

**One cell does not fit the clean story, and it should be reported.**
VALmix050 halves rather than closes: 0.2487 → 0.1242. Looking at the numbers,
the fork's two runs are 0.5464 and 0.6706 — the FD-seeded run overshoots
rather than matching. Mod 11 improved this cell substantially but did not make
it seed-independent. **The honest claim is therefore "closes the gap on 2 of
3 affected cells, halves it on the third", not "removes seed-dependence".**

**Convergence** (Figure 2):

- *Iterations.* Similar on RL cells. On the affected value cells the legacy FD
  runs **terminate early at a worse optimum** rather than converging elsewhere
  — VALmix050 legacy takes 2 iterations against the fork's 5-9, VALmix070
  takes 2 against 10-11. Early termination is the mechanism, not a
  coincidence.
- *Free-energy bound.* The fork reaches a bound higher by **71 / 110 / 161
  nats** on VALmix050/070/100, and is within ~1 nat everywhere else. Never
  worse on any cell.
- *Wall clock.* The fork is slightly faster on RL cells (89 vs 99 s on
  RLmix000; 42 vs 54 s on RLmix070) — Gauss-Newton reuses the Jacobian instead
  of running a fresh finite-difference Hessian. Value cells are ~1-2 s for
  both.
- *Fit-quality reporting.* The fork flags 43-99 % of refits weakly identified
  on RL cells and 0-55 % on value cells. The legacy arm reports nothing at
  all — it has no `diagnostics` field.

**Interpretation.** Mod 11's benefit is **conditional, not universal**. It
does nothing on data where GN and FD individual fits agree, which is all five
RL cells and two of five value cells. Where they disagree it is the difference
between a stable group verdict and one that depends on an arbitrary upstream
choice. Since a user cannot know in advance which case they are in, passing
`model_trials` to `hbi_main` is the right default — but the honest summary is
"insurance that pays out on some datasets", not "uniformly better".
