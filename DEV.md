# CBM-Python — Project Brief

**Repo:** `ginobattistello/cbm_python` (fork of `payampiray/cbm_python`)
**Working copy:** `~/Documents/` (local; may be ahead of GitHub)
**Staging folder:** `cbm/dev/` (was `bmc_TO_ADD/`, renamed for clarity) — holds work in
progress not yet wired into the package (`group_bms.py`, §5) plus regression/verification
harnesses kept alongside the code they check (`rl_jax_verify.py` for §2.3's JAX option;
`gn_numpy_verify.py` for §2.1/§2.3(C)'s NumPy Gauss-Newton curvature, done 2026-08-03;
`baseline_snapshot.py` + `baseline.json`, the §7 step 2/4 A/B regression check, 2026-08-03;
`convergence_status_verify.py` for §2.2's status enum (Mod 6), done 2026-08-03;
`checks_verify.py` for §3's Mods 1/7/8 + §4's three-layer checks (Mod 9 + pre-flight),
done 2026-08-03). Local only for now; not yet pushed to GitHub.
**Reference:** MBB-team/VBA-toolbox (MATLAB)
**Guiding constraint:** simple, clear, transparent. Every change traceable and justified.

---

## 0. Scope

Three workstreams, in dependency order:

1. **Optimization mechanics** — damping, Gauss-Newton, Hessian regularization (VBA-like)
2. **Convergence rules** — stopping criteria, monotonic-improvement checks
3. **BMC between groups and conditions** — already started, code to be supplied
4. **Manual** — prepare fitting / model fitting process / interpreting outputs

Explicitly **out of scope**: turning CBM into VBA. CBM does Laplace-approximation MAP + HBI;
VBA does full Variational Bayesian inversion. We port *behaviors*, not the framework.

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

### Open — workstream 3 (§5, `cbm/dev/group_bms.py`)
- [ ] **Family prior must partition models.** In `_standard`, `a0` is left at 0 for models not
  in any family (invalid Dirichlet). Add a disjoint+exhaustive partition check (shared with the
  BOR item below).
- [ ] **Family BOR/PXP from family free energy, not the rescaling heuristic.** Delete the
  `lbf = log((1-bor)/bor) − …` block; build membership matrix `C` and call
  `compute_bor(L, posterior, priors, C=C)` from `model_selection.py`. Mind the orientation
  caveat (models×subjects vs subjects×models — transpose deliberately). Use
  `dirichlet_exceedance` on `α_fam = Cᵀ·a` for `fam_xp`.
- [ ] **Replace `assert` input validation with `raise ValueError`** (survives `python -O`).
- [ ] **Return dataclasses, not nested dicts** (`GroupBMSResult` / `FamilyResult` /
  `BtwCondsResult` / `BtwGroupsResult`), with optional `to_dict()`/`__getitem__` shim.
- [ ] **No work in `__init__`.** Give explicit entry points (`group_bms`,
  `group_bms_btw_conds`, `group_bms_btw_groups`) or a `.fit()` method.
- [ ] **Add module docstring with REFERENCES** (Rigoux et al. 2014; Stephan et al. 2009).
- [ ] **Promote out of `cbm/dev/`.** Move to `cbm/`, register in `__init__.py`, add a worked
  `examples/` script covering all three modes. Verify each input `L`'s `hess_method` provenance
  (Mod 5 vs Mod 2 fallback) before trusting the statistics.

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

## 5. Workstream 3 — BMC between groups and conditions

**Code supplied:** `cbm/dev/group_bms.py` (264 lines). A `GroupBMS` class that
dispatches on input shape to three routines reproducing VBA:
`_standard` (2D `L`, with optional families) ≈ `VBA_groupBMC`,
`_btw_conds` (3D `L`) ≈ `VBA_groupBMC_btwConds`,
`_btw_groups` (list of 2D `L`) ≈ `VBA_groupBMC_btwGroups`.
It builds on the existing `bms()` / `BMSResult` in `model_selection.py`.

**Depended on §2.1 — resolved 2026-08-03.** Every input is a per-subject × per-model
log-evidence `L`. That `L` comes from the Laplace approximation, whose Hessian was
eigenvalue-clipped, so a flat-direction artifact would have been inherited by every
frequency, exceedance and PXP number here. §2.1 is now fixed (Gauss-Newton curvature,
Mod 5) *but only for models that pass `model_trials` into `individual_fit`/`optimize_map`* —
if group_bms.py is fed evidence from a fit that didn't opt in, it's still on the Mod 2
fallback. Check which path produced `L` before trusting these statistics (`hess_method` field
on the fit's `OptimizationResult`, or `cbm.math.hessian` provenance in `FitResult`).

### What already checks out (do not re-litigate)
- **Between-conditions** sums tuple log-evidence across conditions (`Lt`), then tests the
  "same model/family across all conditions" family against "not equal" — the correct
  within-subject (repeated-measures) construction.
- **Between-groups** stacks each group's per-model evidence into the tuple's group-slot
  (`Lt = vstack([Ls[g][:, tuples[:, g]] ...])`) so a subject only informs its own group's
  slot — the correct construction.
- Result field names (`posterior_parameters`, `model_frequency`, `exceedance_prob`,
  `bor`, `protected_exceedance_prob`) already match `BMSResult`. Good.

### Correctness to verify / flag
- **Family prior leaves gaps.** In `_standard`, `a0` is filled only for indices listed in
  `families`; any model not in a family keeps `a0 = 0`, an invalid Dirichlet prior. Add a
  check that families **partition** all models (disjoint + exhaustive), or handle leftovers.
- **Family-level BOR/PXP: use the family-partition free energy, not the rescaling.**
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
- **`assert` used for input validation** (`assert len(names) == nf`, tuple-count assert) is
  stripped under `python -O`. Use explicit `raise ValueError`.

### Style/logic adaptation to CBM (the "fit the toolbox" part)
- **Return dataclasses, not nested dicts.** The toolbox uses `@dataclass` results
  (`BMSResult`, `OptimizationResult`). Convert to e.g. `GroupBMSResult` / `FamilyResult` /
  `BtwCondsResult` / `BtwGroupsResult` so fields are discoverable and typed. Keep a
  `to_dict()` / `__getitem__` shim if backward-compat with the current dict access matters.
- **No work in `__init__`.** Doing the whole computation in the constructor and dispatching
  on shape is VBA-idiomatic but not Pythonic here. Prefer explicit entry points
  (`group_bms`, `group_bms_btw_conds`, `group_bms_btw_groups`) or a `.fit()` method.
- **Add a module docstring with REFERENCES**, matching `model_selection.py` (Rigoux et al.
  2014; Stephan et al. 2009) so provenance is visible.

**When you move it in Claude Code:** promote from `cbm/dev/` into `cbm/`, add it to
`__init__.py`, and add a worked example under `examples/` (mirroring
`example_model_selection.py`) covering all three modes.

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
   re-run identical). Workstreams 1 and 2 are COMPLETE. Next: workstream 3, §5
   (`cbm/dev/group_bms.py`); then the §6 manual.]**
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