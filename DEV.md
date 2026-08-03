# CBM-Python — Project Brief

**Repo:** `ginobattistello/cbm_python` (fork of `payampiray/cbm_python`)
**Working copy:** `~/Documents/` (local; may be ahead of GitHub)
**Staging folder:** `cbm/dev/` (was `bmc_TO_ADD/`, renamed for clarity) — holds work in
progress not yet wired into the package: `group_bms.py` (§5) and `rl_jax_verify.py` (§2.3).
Local only for now; not yet pushed to GitHub.
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

## 1. State of play (read from the pushed fork)

Four annotated modifications exist in `cbm/optimization.py`:

| # | What | Status |
|---|------|--------|
| 1 | Bounds validation in `Config.__post_init__` | **Commented out — inactive** |
| 2 | Hessian eigenvalue regularization (floor 1e-4) | Active |
| 3 | `_newton_polish` — Gauss-Newton refinement w/ backtracking | Active |
| 4 | Rewritten `optimize` method | Active |

The `MODIFICATION n — WHAT / WHY / REFERENCE` comment convention is good. **Keep it.**
It is the single best transparency asset in the repo — the manual can be generated from it.

---

## 2. Three issues to settle BEFORE writing more code

These are design decisions, not bugs to patch blindly.

### 2.1 The regularized Hessian contaminates model evidence  ← highest priority

`compute_hessian` clips all eigenvalues to `max(λ, 1e-4)`. This is correct and necessary
for *taking Newton steps*. But the same Hessian feeds the Laplace log-evidence:

```
log p(y|m) ≈ log p(y|θ*,m) + log p(θ*|m) + (d/2)·log(2π) − ½·log|H|
```

Clipping raises `log|H|` whenever the surface is flat. If model A has a flat direction and
model B does not, A is penalized by an amount that is an **artifact of the regularizer**,
not evidence. This propagates directly into BMS, HBI, and the group/condition BMC of
workstream 3 — i.e. into your actual scientific conclusions.

**Proposed resolution:** split the two uses.
- `compute_hessian(..., regularize=True)` → optimization steps
- `compute_hessian(..., regularize=False)` → inference / log-evidence
- Record `n_eigenvalues_clipped` and `min_eigenvalue_raw` in the result object.
  If clipping was needed at the MAP, that fit's evidence is **flagged as unreliable**,
  not silently corrected.

*This decision is the hinge between workstream 1 and workstream 3 — resolve it first.*

### 2.2 `converged` currently carries no information

In `_newton_polish`, every exit path sets `converged = True` — including the
"cannot reduce f" path and the exhausted-steps path. The flag is a constant.

**Proposed resolution:** replace the boolean with an explicit status, e.g.
`CONVERGED_DF` / `CONVERGED_GRAD` / `NO_IMPROVEMENT` / `MAX_STEPS` / `SINGULAR_HESSIAN`.
This *is* workstream 2 — a check that cannot report failure is not a check.

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
| A | Port each model log-likelihood to JAX | exact gradients at `O(1)` cost, zero differencing noise | **verified** (both RL models) |
| B | Pass `jac=grad(loglik)` to the `L-BFGS-B` call in `_single_optimization` | removes the restart loop's dominant cost (repeated gradient differencing) | proposed |
| C | At the MAP, replace `compute_hessian` with `jax.hessian` (for the evidence) **or** Gauss-Newton `Σₜ ∇ℓₜ∇ℓₜᵀ + prior precision` (for stepping) | exact evidence Hessian; GN is PSD by construction → the Mod 2 clip retires | **verified** PSD |
| D | *Fallback only if a model can't be made differentiable:* central-difference the evidence Hessian, and reuse curvature across polish steps (quasi-Newton / LM damping) | `O(ε²)` accuracy at ~2× cost; avoids rebuilding `(n+1)²` Hessian up to 30× | proposed |

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

- **Damping (Levenberg-Marquardt style).** VBA does not take raw Newton steps; it inflates
  the precision when a step fails and relaxes it when steps succeed. This is more robust
  than pure step-halving and adapts to local curvature.
  → `H + λI`, with λ↑ on rejection, λ↓ on acceptance.
- **Gauss-Newton curvature (verified — see §2.3 / `rl_jax_verify.py`).** Build the stepping
  curvature as `Σₜ ∇ℓₜ∇ℓₜᵀ + prior precision` rather than a finite-difference Hessian. It is
  positive-definite by construction (confirmed on both RL models), so damping only ever needs
  to *accelerate*, never to rescue an indefinite Hessian — and the Mod 2 eigenvalue clip can
  retire. This is the same curvature VBA damps, which is why it lives here and in §2.3.
- **Monotonic objective guarantee.** Never accept a step that worsens the objective.
  Partly present (backtracking in `_newton_polish`); make it a global invariant.
- **Sign/naming coherence.** The objective here is the *negative* log posterior and is
  minimized; VBA maximizes free energy F. Name it consistently (`neg_log_post`, and
  report `F = −neg_log_post` at the boundary) so the manual isn't self-contradictory.
- **Activate MODIFICATION 1** or delete it — an inactive documented modification is the
  opposite of transparent.

## 4. Workstream 2 — checks

Three layers, each with a defined failure behavior (warn / flag / raise):

**Pre-flight (before any fitting)**
- parameter dimension vs. prior dimension agreement
- `range_bounds` ⊂ `hard_bounds`, shapes are 2×d
- objective is finite at initialization; data non-empty, no NaN/Inf
- prior covariance is positive-definite

**Per-iteration (invariants)**
- objective is finite at every accepted point
- objective is non-increasing (monotonicity)
- Hessian solve succeeded; damping did not diverge
- iterate stayed inside `hard_bounds`

**Post-fit (diagnostics, surfaced in results)**
- gradient norm at the optimum
- raw min eigenvalue + number clipped (→ §2.1)
- agreement across random initializations (how many reached the best optimum) — this is
  the practical multimodality test and belongs in the manual's interpretation section
- explicit convergence status (→ §2.2)

**Principle:** a check never silently changes a result. It either flags it or stops.

## 5. Workstream 3 — BMC between groups and conditions

**Code supplied:** `cbm/dev/group_bms.py` (264 lines). A `GroupBMS` class that
dispatches on input shape to three routines reproducing VBA:
`_standard` (2D `L`, with optional families) ≈ `VBA_groupBMC`,
`_btw_conds` (3D `L`) ≈ `VBA_groupBMC_btwConds`,
`_btw_groups` (list of 2D `L`) ≈ `VBA_groupBMC_btwGroups`.
It builds on the existing `bms()` / `BMSResult` in `model_selection.py`.

**Depends on §2.1.** Every input is a per-subject × per-model log-evidence `L`. That `L`
comes from the Laplace approximation, whose Hessian is currently eigenvalue-clipped. So a
flat-direction artifact in §2.1 is inherited by every frequency, exceedance and PXP number
here. Fix §2.1 first, or this workstream computes clean statistics on contaminated evidence.

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
2. Establish a baseline **before touching anything**: run `examples/example_RL.py`,
   save parameter estimates, log-evidence, timings. Every later change is judged against it.
   Keep `cbm/dev/rl_jax_verify.py` alongside it — it already pins the autodiff port to the
   NumPy models numerically, so it doubles as the regression check when wiring §2.3 (A)–(C) in.
3. Work **one MODIFICATION block at a time**, in the order: §2.2 → §2.1 → §3 → §4 → §5.
   The §2.3 autodiff/Gauss-Newton work slots into §2.1+§3 (it supplies the exact/PSD Hessian
   both depend on); do the JAX port of the model likelihood before those two blocks.
4. After each block: re-run the baseline, diff the numbers, record the delta in the block's
   comment. If a change moves the numbers, that must be explained, not just observed.
5. Only then write the manual, drawing text from the MODIFICATION blocks.

**Non-negotiables:** no silent corrections; no undocumented change; keep
`optimization_legacy.py` as the untouched reference for A/B comparison.
