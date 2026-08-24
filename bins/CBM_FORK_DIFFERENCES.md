# Differences between the original CBM-Python toolbox and the present fork

**Original** — `payampiray/cbm_python`, HEAD at commit `ccb8aa8` ("Update examples").
**Fork** — `ginobattistello/cbm_python`, HEAD at commit `65abe7d` ("New version").
**Reference toolbox for the ported behaviours** — `MBB-team/VBA-toolbox` (MATLAB).

The fork branches from `ccb8aa8` and is **11 commits ahead with no divergence**: every
difference described below is an addition or an in-place edit, and **no file present in the
original has been deleted**. The upstream code as it existed before any modification is
preserved verbatim inside the fork at `benchmark/external/cbm_original/`, extracted from
commit `e72193f`.

Summary of the file-level comparison:

| | count |
|---|---:|
| Files byte-identical to the original | 6 |
| Files modified in place | 14 |
| Files added by the fork | 77 |
| Files removed | 0 |

This document is self-contained: it does not require reading `DEV.md`, the fork's internal
development log.

---

## 1. MOTIVATION

### 1.1 What the original toolbox does, and where it stops

The original CBM-Python implements the method of Piray et al. (2019): per-subject
maximum-a-posteriori (MAP) estimation under a Laplace approximation, random-effects Bayesian
model selection (BMS), and hierarchical Bayesian inference (HBI). Its estimation core
(`cbm/optimization.py`) is a multi-start L-BFGS-B optimiser followed by a finite-difference
Hessian, whose determinant supplies the Laplace log-evidence used by every downstream
inference.

Three properties of that core motivated the fork. They are not implementation bugs so much as
design decisions whose consequences propagate into the scientific conclusions the toolbox is
used to draw.

**(a) The regularised Hessian contaminates the model evidence.**
In the original, `compute_hessian` produces a Hessian that is then checked for positive
definiteness by Cholesky; the fork's early modifications clipped its eigenvalues to a floor of
`1e-4` so that Newton steps and the Laplace formula never fail. Clipping is correct for
*taking a step*, but the *same* matrix enters

```
log p(y | m) ≈ log p(y | θ*, m) + log p(θ* | m) + (d/2)·log(2π) − ½·log|H|
```

so any clipping raises `log|H|` and penalises the model. The size of the penalty depends on how
flat the likelihood surface is, and on the arbitrary constant `1e-4`. If model A has a weakly
identified direction and model B does not, A is penalised by an artifact of the regulariser
rather than by evidence — and that artifact propagates into BMS, into HBI, and into any
group-level comparison built on them.

**(b) The convergence flag carried no information.**
The original derives a three-valued `flag` (1.0 / 0.5 / 0.0) from a gradient-norm test and a
Cholesky check, and escalates the number of restarts when the flag is not 1.0. In the fork's
first Newton-polish implementation, every exit path of the refinement loop set `converged =
True`, including the "cannot improve" and "ran out of steps" paths. A check that cannot report
failure is not a check.

**(c) Derivatives are obtained by differencing an already-differenced gradient.**
The original has no analytic derivatives anywhere. The gradient is a finite difference of the
objective, and the Hessian is then a finite difference *of that gradient* — a difference of a
difference. Two consequences follow.

*Cost.* One gradient costs `n+1` evaluations of the model, and the Hessian needs a fresh
gradient for each of the `n` parameters, so a single Hessian costs `(n+1)²` model evaluations:
49 at d = 6, 441 at d = 20. Any Newton-style refinement makes this worse still, because it needs
the Hessian afresh at every step.

*Accuracy.* Each differencing step divides a small difference by a small step size, which
amplifies floating-point rounding error; doing it twice compounds that amplification, leaving
noise of order `2×10⁻⁶·|f|` in the Hessian entries. Crucially, that is the same order of
magnitude as the `1e-4` eigenvalue floor from problem (a) — so the clip can be triggered by
numerical noise rather than by genuine flatness in the likelihood.

These are not two independent problems but one. Because the same noisy Hessian supplies
`−½·log|H|`, its error passes straight into the Laplace evidence: differencing twice does not
merely make fitting slow, it biases the quantity that BMS, HBI and every group-level comparison
are built on.

Beyond the estimation core, three capabilities were absent: group-level Bayesian model
comparison between families, conditions and groups; any reporting of fit quality (HBI in
particular discarded the optimiser's diagnostics entirely); and any human-readable rendering of
results (`print(fit)` emitted the raw dataclass, leading with configuration and prior arrays).

### 1.2 The new optimisation algorithm — concept

The change is best stated as a change of *which curvature the toolbox uses*, and where.

**Original pipeline** (`BFGSOptimizer.optimize` in `cbm/optimization.py`):

```
multi-start L-BFGS-B  →  keep best  →  finite-difference Hessian
                      →  Cholesky test  →  flag ∈ {1.0, 0.5, 0.0}
                      →  if flag < 1.0, restart with more initialisations (loop)
```

**Fork pipeline** (same method, rewritten):

```
single pass of L-BFGS-B over num_init starts  →  keep best
   →  Gauss-Newton / Newton polish with backtracking line search
   →  curvature at the optimum:  H = JᵀJ + Σ₀⁻¹   (Gauss-Newton, opt-in)
                             or  eigenvalue-clipped finite differences (fallback)
   →  explicit ConvergenceStatus  →  flag via an explicit lookup table
   →  post-fit diagnostics + weak-identifiability triage
```

**The Gauss-Newton curvature.** The exact Hessian of the negative log posterior decomposes as

```
∇²J(θ)  =  Σₜ Jₜᵀ Qₜ Jₜ   −   Σₜ rₜᵀ Qₜ ∂²gₜ/∂θ²   +   Σ₀⁻¹
           └──── kept ────┘    └──── dropped ─────┘     └ prior ┘
```

Dropping the residual-weighted second-derivative term is the classical Gauss-Newton
approximation for non-linear least squares (Nocedal & Wright, 2006, ch. 10). It is exact in the
limit of vanishing residuals, and — critically — it is precisely the term responsible for
indefinite curvature. What survives, `Σ Jᵀ Q J`, is a sum of quadratic forms with `Q` positive
definite, hence positive *semi*-definite by construction; adding the prior precision `Σ₀⁻¹`
(itself positive definite) makes the total strictly positive definite.

Three consequences follow, and together they are the point of the modification:

- **Nothing needs clipping.** The eigenvalue floor of problem (a) becomes unnecessary rather
  than better-tuned. There is no arbitrary constant left in the evidence.
- **Flat directions fall back to the prior, not to a constant.** Where the data are
  uninformative, `JᵀQJ → 0`, so `H → Σ₀⁻¹` in that direction and the posterior covariance
  reverts to the *prior* covariance. That is the correct Bayesian answer; a `1e-4` floor is not.
- **One curvature serves both uses.** The same `H` is used for the Newton step and for the
  Laplace evidence, so the "regularise for stepping, don't regularise for evidence" split never
  arises. This mirrors the VBA toolbox, which builds `iSigma = iQ + Σ Jᵀ J` in `VBA_Iphi.m` /
  `VBA_Itheta.m` and reuses it for the free energy via `VBA_Hpost.m` → `VBA_FreeEnergy.m`.

**How `J` is obtained.** The Jacobian of the *per-trial* log-likelihood is built by a single
forward difference per parameter — `n+1` model calls, each returning all `T` trials at once —
using VBA's step rule (relative step `1e-4·θ`, floored at `1e-4` in magnitude;
`VBA_numericDiff.m`). This replaces the `(n+1)²` double-differencing of the original and
removes the compounding of differencing noise identified in problem (c).

**How it is switched on.** The Gauss-Newton path requires the model to expose its per-trial
log-likelihood vector rather than only the summed scalar. Most models already compute this
internally before summing, so the change is usually to stop summing early. The toolbox takes it
as an optional argument — `model_trials=` on `individual_fit` and on `hbi_main`, `trial_func=`
at the optimiser level. **When it is not supplied, the fork reproduces the previous
finite-difference behaviour exactly**, so existing model code keeps working unchanged.

**The polish loop.** `_newton_polish` iterates the Newton direction `H⁻¹g` with a backtracking
line search (up to 20 halvings, i.e. down to a step of ~`1e-6`), accepting only steps that
strictly reduce the objective, and stops on VBA's relative free-energy criterion
`|ΔF| / (1 + |F|) < 1e-4`, or after 30 steps. Because only improving steps are accepted,
monotonic descent is structural — and the fork promotes it to a *checked* invariant, raising a
`RuntimeError` rather than warning if the objective ever rises, on the grounds that a violation
means the evidence values cannot be trusted.

**What the new algorithm does and does not change.** Verified on matched fits: the MAP itself
moves very little (parameters agree with the old path to ~`1.8e-8`), because Gauss-Newton and
the clipped Hessian disagree about *curvature*, not about *where the optimum is*. What changes
is the evidence: in a paired benchmark of 239 fits, the MAP moved on 15 and the log-evidence
moved on all 239. This is exactly the intended behaviour — the modification is an evidence-scale
correction, and evidence is what BMS and HBI consume.

### 1.3 Empirical justification

The fork ships the benchmark that tests these claims, in two parts.

**Cross-implementation agreement (individual fits).** The same simulated data — two generating
models (2-armed bandit RL, and a power value function on risky choice), 120 subjects × 200
trials, with 10 % lapse trials as a genuine misspecification stress — fitted by three
independent arms: this fork, the pristine pre-fork CBM, and the MATLAB VBA toolbox. Because the
ground truth is known by construction, each arm can be scored against it and the scores then
compared across arms.

Three quantities are reported, and they answer different questions:

- **Parameter recovery** — the Pearson correlation, across subjects, between each parameter's
  estimated value and the value actually used to generate that subject's data. 1.0 would be
  perfect recovery; the ceiling in practice is set by how strongly the simulated data constrain
  that parameter, not by the fitting code. *Does the arm find the right parameters?*
- **Model-selection AUC** — every subject's data are fitted with both candidate models of its
  family, giving an evidence gap (complex minus simple). AUC is the Mann-Whitney U statistic on
  those gaps: the probability that a subject generated by the complex model has a larger gap
  than a randomly chosen subject generated by the simple one. 0.5 is chance. *Does the arm's
  evidence discriminate between models?*
- **Failed fits** — fits that did not complete at all, out of the full set attempted per arm.
  *Is the arm robust?*

| quantity | fork | pristine CBM | MATLAB VBA |
|---|---:|---:|---:|
| RL parameter recovery (α / β) | 0.818 / 0.900 | 0.818 / 0.900 | 0.818 / 0.902 |
| POW parameter recovery (ρ / β) | 0.943 / 0.773 | 0.944 / 0.773 | 0.943 / 0.773 |
| Model-selection AUC | 0.6215 | 0.6338 | 0.6223 |
| Failed fits (of 960) | 0 | 1 | 0 |

**What the table shows is that the three columns are the same**, which is exactly what a change
that must not silently move estimates should produce. Recovery agrees to the third decimal
throughout. The AUC spread across arms is 0.012, so all three discriminate equally well; the
fork's marginally lower value is not a loss of discriminability, because the Gauss-Newton
curvature shifts the *level* of the evidence and a level shift cannot move a threshold-free
metric.

A fourth quantity is not a per-arm score against the truth but a direct arm-to-arm comparison:
correlating the same parameter, for the same subject, as estimated by two different arms. The
weakest such correlation anywhere in the grid — any pair of arms, any parameter — is
**r = 0.9999**. Since VBA shares no code with either Python arm and uses variational Bayes
rather than Laplace-MAP, this is a genuine cross-implementation check rather than a shared-code
artifact.

The single pristine-CBM failure is an upstream crash: in the original `optimize`, when every
initialisation returns a non-finite objective, `best_result` is never assigned and the retry
loop has no guard, producing `AttributeError: 'NoneType' object has no attribute 'x'`. On the
value-function grids this cost the pristine arm 360 of 1080 fits in one configuration. The
fork's defensive objective wrapper (non-finite → `1e20` penalty) is what prevents it.

**Hierarchical stability (HBI).** A two-arm comparison — the frozen pre-modification HBI versus
the modified HBI — asks whether the group verdict depends on how the supplied individual maps
were fitted, which is an upstream choice a user may not realise they are making. On the three
mixed-population cells where the two curvature paths genuinely differ, the legacy arm's model
frequency swings by 0.25 / 0.55 / 0.42, in one case moving from "the complex model dominates"
(0.91) to "it does not" (0.36). The fork's swing on the same cells is 0.12 / 0.0018 / 0.0000,
and it reaches a free-energy bound higher by 71 / 110 / 161 nats. Elsewhere the two arms are
indistinguishable.

The honest summary recorded in the fork is that this benefit is **conditional**: it does nothing
on data where the two curvatures agree, and on one of the three affected cells it halves the
instability rather than removing it. Since a user cannot know in advance which case they are in,
supplying `model_trials` is the recommended default, but it is insurance rather than a uniform
improvement.

---

## 2. EDITS SUMMARY

### 2.1 The fifteen numbered modifications

The fork uses a consistent in-code convention: every substantive change is wrapped in a comment
block of the form `MODIFICATION n — WHAT / WHY / REFERENCE`. The numbering is stable and is used
throughout the codebase, so it is reproduced here as the index of substantive edits.

| # | Change | Where it lives | Default |
|---|---|---|---|
| 1 | Bounds validation in `Config.__post_init__` (`range ⊂ hard`) | `cbm/optimization.py` | active |
| 2 | Hessian eigenvalue regularisation (floor `1e-4`) | `cbm/optimization.py` | active (fallback) |
| 3 | `_newton_polish` — Gauss-Newton refinement with backtracking | `cbm/optimization.py` | active |
| 4 | Rewritten `optimize`: defensive wrapping, single pass, polish | `cbm/optimization.py` | active |
| 5 | `_gauss_newton_curvature` — `H = JᵀJ + Σ₀⁻¹` | `cbm/optimization.py` | **opt-in** via `trial_func` |
| 6 | `ConvergenceStatus` enum + explicit `FLAG_FROM_STATUS` table | `cbm/optimization.py` | active |
| 7 | Monotonicity invariant, checked at two boundaries | `cbm/optimization.py` | active |
| 8 | Sign/naming coherence: `neg_log_post`, `F = −neg_log_post` | `cbm/optimization.py`, `cbm/map_estimation.py` | active |
| 9 | Post-fit diagnostics (`PostFitDiagnostics`) | `cbm/optimization.py` | active |
| 10 | Weak-identifiability warning (curvature vs prior precision) | `cbm/optimization.py` | active |
| 11 | `model_trials` threaded into HBI's internal refits | `cbm/hbi.py`, `cbm/hbi_updates.py` | **opt-in** |
| 12 | HBI retains per-refit diagnostics instead of discarding them | `cbm/hbi_updates.py`, `cbm/hbi_types.py` | active |
| 13 | Readable output: `summary()` / `table()` / `__repr__` / `se` | `cbm/reporting.py` (new) | active |
| 14 | Display options: trace capture and diagnostic figures | `cbm/optimization.py`, `cbm/individual_fit.py`, `cbm/display.py` (new) | **opt-in** via `Config.display` |
| 15 | Default weakly-informative prior N(0, 6.25) in θ-space | `cbm/individual_fit.py` | active when prior omitted |

### 2.2 The substantive differences, grouped

**A. Estimation core rewritten** (Mods 1–10, `cbm/optimization.py`, 539 → 1684 lines).
The dominant change. Gauss-Newton curvature as an opt-in replacement for the clipped
finite-difference Hessian; a Newton polish with backtracking; the retry-escalation loop replaced
by a single pass; a defensive objective wrapper; an explicit convergence status in place of a
constant boolean; a monotonicity invariant enforced by raising; and a diagnostics record
attached to every result. Roughly half of the added lines are the `WHAT / WHY / REFERENCE`
annotation blocks rather than executable code.

**B. The hierarchical layer can now reach the new curvature, and reports on itself**
(Mods 11–12). Previously `optimize_map` was called from HBI with six positional arguments, so
the seventh — the per-trial function — could never be passed; HBI refits every subject on its
first iteration, which meant a user's carefully Gauss-Newton-fitted maps were silently
discarded. The returned `OptimizationResult` was likewise thrown away with a comment saying so,
leaving HBI as the only path in the toolbox with no fit-quality reporting.

**C. Group-level Bayesian model comparison added** (`cbm/group_bms.py`, new, 642 lines).
Family-level, between-conditions and between-groups BMC, mapping onto VBA's `VBA_groupBMC`,
`VBA_groupBMC_btwConds` and `VBA_groupBMC_btwGroups`, built on the existing `bms()` machinery.
Includes `check_evidence_provenance()`, which flags input evidence produced without the
Gauss-Newton path.

**D. Usability layer added** (Mods 13–15). Readable summaries and tables on every result type;
optional diagnostic figures with a matplotlib and a self-contained HTML backend; a documented
default prior so that `prior_mean` / `prior_variance` become optional. Mods 13 and 15 are
presentation and defaults; Mod 14 is off by default and retains nothing when off.

**E. A verification and benchmarking infrastructure added.**
Nine regression harnesses under `cbm/dev/` pinning each modification against a committed
baseline, and a `benchmark/` tree implementing the three-arm cross-implementation comparison
(including a vendored copy of the pristine upstream code and MATLAB drivers for the VBA arm).

**F. Two upstream correctness fixes.**
A family-prior orientation error in `compute_bor` (`cbm/model_selection.py`) that crashed
whenever the number of models differed from the number of families, and a docstring in
`cbm/map_estimation.py` that documented the opposite sign to the one the code returned.

### 2.3 What is deliberately unchanged

- **The statistical framework.** The fork remains Laplace-approximation MAP plus HBI. It ports
  *behaviours* from VBA; it does not adopt VBA's full variational Bayesian inversion.
- **The HBI update equations**, which are byte-identical to the original apart from the
  threading of the two new arguments — this is what makes the HBI benchmark attributable to the
  curvature alone.
- **Backward compatibility.** Every new capability is either opt-in or additive. Callers written
  against the original API continue to run and, where verified, produce bit-identical numbers.
- **Four core modules** (`hbi_bound.py`, `hbi_config.py`, `hbi_exceedance.py`,
  `hbi_logging.py`), the `LICENSE`, and the top-level `README.md`.

**Note on the README.** `README.md` is byte-identical to the original's. The fork's public
front page therefore still describes the upstream toolbox and mentions none of the changes
above; the entirety of the fork's documentation currently lives in `DEV.md` and in the in-code
annotation blocks.

---

## 3. EDITIONS

Every file that differs from the original, by location. `M` = modified in place,
`N` = new in the fork. Line counts are given as `original → fork` for modified files.

### 3.1 Repository root

| File | | Lines |
|---|---|---|
| `pyproject.toml` | M | 20 → 33 |

- Declares dependencies that were always required but never stated: `numpy>=1.20`,
  `scipy>=1.7`, plus `pandas>=1.3`, newly required because `result.table()` returns DataFrames
  (Mod 13).
- Adds an optional extra `display = ["matplotlib>=3.5"]`. Deliberately optional: `cbm/display.py`
  imports matplotlib lazily, so `import cbm` and every fit work without it, and only calling
  `plot()` needs it.

| File | | Lines |
|---|---|---|
| `DEV.md` | N | 2053 |

- The fork's development log and design record: the three motivating problems and their
  resolutions, one section per modification, the benchmark designs and results, and the
  outstanding to-do list. Written as a working document, not as user documentation.

| File | | |
|---|---|---|
| `.gitignore` | N | |

- Excludes build and cache artefacts, benchmark datasets and per-arm result binaries
  (`benchmark/data/`, `*.npz`, `*.mat`, `*.pkl`), and the gitignored VBA-toolbox clone.
- Two deliberate exceptions are documented in the file itself: `cbm/dev/baseline.json` is
  tracked because it is the committed A/B reference, and `benchmark/external/cbm_original/` is
  tracked because it is the pristine comparison arm.

| File | | |
|---|---|---|
| `.Rhistory` | N | empty |

- Empty file, committed at the repository root. No R code exists anywhere in the project;
  this appears to be an accidental commit and is a candidate for removal.

| File | | |
|---|---|---|
| `README.md` | — | unchanged |
| `LICENSE` | — | unchanged |

### 3.2 `cbm/` — modified core modules

#### `cbm/optimization.py` — M, 539 → 1684 lines

The largest single change; hosts Modifications 1–10.

- **Header block** rewritten as an index of all modifications, stating explicitly that Mods 2–4
  are interdependent, that Mod 5 supersedes Mod 2 whenever `trial_func` is supplied, and that
  Mod 6 supersedes the flag branch of Mod 4.
- **`ConvergenceStatus`** (new `str`-`Enum`, Mod 6) with exactly four members, one per real exit
  path of the polish: `CONVERGED_DF`, `NO_IMPROVEMENT`, `MAX_STEPS`, `SINGULAR_HESSIAN`. A
  `CONVERGED_GRAD` member was considered and dropped because the polish never tests the
  gradient and it would be permanently unreachable.
- **`FLAG_FROM_STATUS`** — an explicit dict mapping status → legacy flag (`CONVERGED_DF`,
  `NO_IMPROVEMENT`, `MAX_STEPS` → 1.0; `SINGULAR_HESSIAN` → 0.5 with a warning). Explicit rather
  than a truthiness test, because `if status:` is true for every enum member — the original bug
  in enum form — and an identity test would demote two legitimate accept-paths to 0.5, which
  transitively risks prior substitution downstream.
- **`WEAK_IDENTIFIABILITY_RATIO = 2.0`** (Mod 10) — module-level threshold, calibrated on 360
  fits, with the full justification recorded at the warning site.
- **`Config`** — bounds validation activated in `__post_init__` (Mod 1, previously present but
  commented out); new `display` field (Mod 14), off by default.
- **`OptimizationResult`** — carries `convergence_status`, raw minimum eigenvalue, clip count,
  cross-initialisation agreement and hard-bound mask; gains `neg_log_post` and `F` properties
  making the sign convention explicit at the boundary (Mod 8), and a `diagnostics()` method
  building a `PostFitDiagnostics` record on demand.
- **`PostFitDiagnostics`** (new dataclass, Mod 9) — the transport object for fit-quality
  information into `FitResult` and, via Mod 12, into HBI.
- **`compute_hessian`** — after symmetrising, eigendecomposes, records the raw minimum
  eigenvalue and clip count *before* clipping (so the contamination is measurable), clips to
  `1e-4`, reconstructs (Mod 2). Now optionally returns the diagnostics alongside the matrix.
- **`_gauss_newton_curvature`** (new method, Mod 5) — builds `H = JᵀJ + prior_precision` from a
  single forward difference per parameter of the per-trial log-likelihood, using VBA's step rule
  (`1e-4·θ`, floored at `1e-4`).
- **`_newton_polish`** (new method, Mod 3) — Newton direction from the chosen curvature,
  backtracking line search of up to 20 halvings clipped to `hard_bounds`, VBA's relative
  free-energy stopping rule with `tol_df = 1e-4`, at most 30 steps; returns an explicit status;
  raises `RuntimeError` if the objective rose between entry and exit (Mod 7); optionally records
  a per-step trace of objective and Laplace log-evidence (Mod 14).
- **`optimize`** (rewritten, Mod 4) — four sub-changes: (a) a defensive wrapper turning
  exceptions and non-finite values into a `1e20` penalty; (b) a single pass over the
  initialisation points, replacing the escalating retry loop; (c) the Newton polish, followed by
  a second monotonicity check at the `optimize()` boundary; (d) the flag now derived from the
  status table.
- **Post-fit block** (Mods 9–10) — cross-initialisation agreement measured on pre-polish optima
  with the same relative tolerance the polish uses; a warning when the MAP sits exactly on a hard
  bound, because the Laplace approximation assumes an interior optimum; and the
  weak-identifiability ratio, computed only on the Gauss-Newton path where the prior precision
  is known exactly and nothing has been clipped. The accompanying comment states the measured
  precision (~0.5) and recall (~0.5) and frames the warning as triage, not rejection.
- The original method bodies are retained above each block as commented reference, so the
  before/after is readable in place.

#### `cbm/individual_fit.py` — M, 370 → 733 lines

- **`_preflight_checks`** (new, the pre-flight validation layer) — raises on empty data, on
  disagreement between `config.d` and the prior dimension, on a wrongly shaped precision matrix,
  on a non-positive-definite prior covariance (Cholesky, because `np.linalg.inv` succeeding does
  not establish positive definiteness), and on a model that raises at the prior mean; warns when
  the model returns a non-finite log-likelihood there.
- **`individual_fit` signature extended** — `prior_mean` and `prior_variance` become optional
  (Mod 15); `model_trials=` added (routes the fit onto the Gauss-Newton curvature);
  `predict=` and `observed=` added (display only, Mod 14).
- **`_resolve_prior`** (new, Mod 15) — supplies N(0, 6.25) in unconstrained θ-space for whichever
  of mean/variance is missing, infers `d` without introducing a new argument, and announces
  itself in four places (a `UserWarning`, the verbose header, `FitInput.prior_defaults` on the
  result, and both `summary()` and the figure). The value is taken from Piray et al. (2019)
  rather than from VBA's N(0, I): unit variance is a *strong* prior in θ-space, excluding
  learning rates below ≈ 0.12.
- **`FitResult` extended** — `summary()`, `table()`, `__repr__` (Mod 13), an `se` property
  returning standard errors from the posterior covariance, and `plot()` (Mod 14). The custom
  `__repr__` exists because the dataclass default led with configuration and prior arrays,
  burying the estimates.
- **Display payload** attached as a plain attribute rather than a dataclass field, so it never
  enters `asdict()` and never reaches the pickle written by `fname=`.
- Threads the sixth return value of `optimize_map` (the full `OptimizationResult`) through to
  the result object.

#### `cbm/map_estimation.py` — M, 99 → 129 lines

- **Sign/naming correction (Mod 8).** `log_posterior`'s docstring claimed it returned the
  *negative* log posterior while the code returned the positive log joint. The code was right;
  only the documentation was wrong. The single negation for minimisation is now located in
  `optimize_map` and named `neg_log_post`.
- **`optimize_map` accepts `model_trials`** and constructs the `trial_func` closure passed to the
  optimiser, together with the exact `prior_precision`.
- **Returns six values instead of five** — the trailing `OptimizationResult` carries the
  convergence status and the post-fit diagnostics. This is the signature change that Mods 11 and
  12 depend on.
- Docstring corrected on a long-standing naming trap: the first return value named `loglik` is
  the log *joint* at the MAP, not the bare log-likelihood. The name is kept for backward
  compatibility and the discrepancy documented.

#### `cbm/hbi.py` — M, 656 → 683 lines

- **`hbi_main` accepts `model_trials`** (Mod 11): one per-trial log-likelihood function per
  model, aligned with `models`, each entry independently allowed to be `None` for a model that
  cannot expose a per-trial decomposition.
- Validates the length of that list against `models` with an explanatory error.
- Stores it in `user_input` and reads it back with `.get()` rather than `[]`, so a `user_input`
  dict built by older code — or unpickled from before the change — still loads and simply keeps
  the finite-difference path.
- Forwards it to `hbi_qhquad`.
- Import corrected from absolute (`from cbm.hbi_exceedance import …`) to relative
  (`from .hbi_exceedance import …`), consistent with the rest of the package.

#### `cbm/hbi_updates.py` — M, 346 → 424 lines

- **Mod 11 annotation block** recording why the change was necessary: `optimize_map` was called
  with six positional arguments, so the seventh could never be passed, and every HBI refit used
  finite differences regardless of how the supplied maps had been produced.
- **Mod 12 annotation block** recording that the returned `OptimizationResult` was previously
  discarded with a comment acknowledging it.
- **`hbi_qhquad` accepts `model_trials`** and selects per model (`None` → finite-difference
  fallback, i.e. exactly the previous behaviour).
- Unpacks six return values from `optimize_map` and collects the diagnostics into a `(K, N)`
  object array attached to `IndividualPosterior.diagnostics`.
- Explicit `callable()` check when harvesting diagnostics, because `diagnostics` is a *method*
  on `OptimizationResult`; a bare `getattr` would store the bound method and fail only later, at
  the point of use.

#### `cbm/hbi_types.py` — M, 148 → 184 lines

- **`IndividualPosterior.diagnostics`** (Mod 12) — optional `(K, N)` object array, defaulting to
  `None` so that HBI results pickled before the change still load. This matters in practice
  because HBI results are routinely written to disk and re-read by `hbi_null`.
- **`HBIResult` gains `summary()`, `table()`, `subject_table()` and `__repr__`** (Mod 13), all
  delegating to `cbm/reporting.py` and importing it lazily inside the methods. The `__repr__`
  degrades gracefully to a short diagnostic string if summarising fails.

#### `cbm/model_selection.py` — M, 287 → 295 lines

- **Correctness fix in `compute_bor`'s family branch.** The family-null prior was computed as
  `C @ (np.sum(C, axis=1) ** -1)` — per-*model* membership counts, shape `(K,)`, which cannot
  right-multiply the `K × nf` matrix `C` and raised whenever the number of models differed from
  the number of families. VBA sums per *family* (`sum(options.C,1)'` in MATLAB is the column
  sum), so the axis is corrected to `axis=0`.
- The branch was unreachable dead code in the original — nothing passed `C` — and only became
  live when `cbm/group_bms.py` started using it, which is how it was found.

#### `cbm/__init__.py` — M, 4 → 16 lines

- Re-exports the new group-BMS entry points and result types (`group_bms`,
  `group_bms_btw_conds`, `group_bms_btw_groups`, `check_evidence_provenance`, and the five
  associated dataclasses) alongside the original exports.

#### Unchanged core modules

`cbm/hbi_bound.py`, `cbm/hbi_config.py`, `cbm/hbi_exceedance.py`, `cbm/hbi_logging.py` are
byte-identical to the original.

### 3.3 `cbm/` — new modules

#### `cbm/group_bms.py` — N, 642 lines

- Group-level Bayesian model comparison built on the existing `bms()` / `model_selection.py`
  machinery, mapping onto VBA: `group_bms ≈ VBA_groupBMC` (2-D log-evidence, optional families);
  `group_bms_btw_conds ≈ VBA_groupBMC_btwConds` (3-D, subject × model × condition);
  `group_bms_btw_groups ≈ VBA_groupBMC_btwGroups` (one 2-D matrix per group).
- Families are required to **partition** the model set — disjoint and exhaustive, validated once
  and shared by the Dirichlet prior and the membership matrix `C`. An earlier draft left `a0 = 0`
  for unassigned models, which is an invalid Dirichlet prior.
- The family-level Bayesian omnibus risk is computed from the family free energy rather than by
  a rescaling heuristic, cross-checked against `VBA_groupBMC.m`.
- Family frequencies are exact Dirichlet aggregations (`α_fam/Σα`), and family exceedance uses
  the toolbox's own `dirichlet_exceedance` on `α_fam = Cᵀ·a`, matching VBA's choice.
- The between-groups test is a free-energy comparison of pooled versus per-group fits — VBA's
  actual test — replacing a tuple construction that was both empirically degenerate and absent
  from VBA.
- **`check_evidence_provenance(fit_results)`** — warns when input log-evidence came from fits
  that did not opt into the Gauss-Newton curvature, since every statistic computed here inherits
  the eigenvalue-clip artifact if so.
- Typed dataclass results with a `['key']` / `to_dict()` shim for dict-style access;
  `raise ValueError` rather than `assert` for input validation.

#### `cbm/reporting.py` — N, 701 lines

- Mod 13's implementation: `summary()`, `table()` and row-builders for every result type —
  `FitResult`, `HBIResult`, `GroupBMSResult`, `BtwConds`, `BtwGroups` — plus `standard_errors()`
  and a `prior_spec()` renderer.
- **Presentation only.** It reads fields that already existed and is never called during
  fitting; the accompanying harness pins that adding it changes no numerical result.
- Includes a per-fit `quality` column derived from the convergence status, the diagnostics and
  the flag, which is the single field intended for triage.

#### `cbm/display.py` — N, 913 lines

- Mod 14's figures: `plot_subject`, `plot_group`, and a dispatching `plot()`.
- **matplotlib is imported lazily** inside the plotting functions, which is what allows it to be
  an optional extra.
- Two backends: matplotlib windows/files, and `to_html()`, which writes a self-contained HTML
  file — chosen over pyqtgraph or Plotly to avoid adding a heavyweight or JavaScript-bundled
  dependency.
- The figures are explicit about three limitations rather than hiding them: search trajectories
  are function *evaluations* including line-search probes, not iterations; a per-step Laplace
  log-evidence exists only during the polish (L-BFGS-B has no Hessian, so no evidence exists for
  those evaluations) and the two segments are plotted separately rather than joined into one
  misleading curve; and warnings are *copied* into the figure, never intercepted, so they still
  reach the user's log.
- Not built, deliberately: live plotting during the fit.

#### Frozen legacy snapshots

| File | | Lines |
|---|---|---|
| `cbm/optimization_legacy.py` | N | 709 |
| `cbm/hbi_legacy.py` | N | 682 |
| `cbm/hbi_updates_legacy.py` | N | 375 |
| `cbm/hbi_types_legacy.py` | N | 174 |

- Frozen snapshots used as A/B comparison arms, not as live code.
- The three `hbi_*_legacy` files are the hierarchical layer as of commit `93a0be8`, immediately
  before Mods 11 and 12; they form the legacy arm of the HBI benchmark, which is what makes that
  benchmark's differences attributable to the curvature alone.
- **`optimization_legacy.py` carries an important caveat**: it is *not* the pristine upstream
  file. It is the state as of 2026-08-03, already carrying Mods 1–4 without their annotation
  blocks, and it was for a period the module actually imported by `individual_fit.py` and
  `map_estimation.py` while the annotated `optimization.py` sat unused. The genuinely pristine
  code is the vendored copy under `benchmark/external/cbm_original/`.

### 3.4 `cbm/dev/` — verification harnesses (all new)

Regression harnesses kept alongside the code they check. Each pins the behaviour of one
modification and is run against a committed baseline; none is imported by the package.

| File | Lines | Pins |
|---|---:|---|
| `cbm/dev/baseline_snapshot.py` | 267 | Per-subject MAP parameters, log-evidence, Hessian eigenvalues, flags and timings for both RL models. Provides the `--compare` A/B check every later modification is judged against. |
| `cbm/dev/baseline.json` | 2313 | The committed reference numbers themselves. Deliberately not gitignored. |
| `cbm/dev/gn_numpy_verify.py` | 136 | Mod 5 versus the Mod 2 fallback on an identical subject: parameters, eigenvalues, log-evidence. |
| `cbm/dev/convergence_status_verify.py` | 134 | Mod 6 — forces all four exit paths, checks the status reported, the flag mapped, and that a forced singular Hessian does not trigger prior substitution. |
| `cbm/dev/checks_verify.py` | 216 | Mods 1, 7, 8, 9 and the pre-flight layer (30 checks). |
| `cbm/dev/group_bms_verify.py` | 270 | `cbm/group_bms.py` and the `compute_bor` orientation fix, against the actual VBA sources (42 checks). |
| `cbm/dev/hbi_verify.py` | 296 | Mods 11 and 12, including bit-identity of the `model_trials=None` path against a pre-change reference. |
| `cbm/dev/reporting_verify.py` | 539 | Mod 13 — chiefly that the addition is *purely additive* and changes no numerical result. |
| `cbm/dev/rl_jax_verify.py` | 190 | The JAX autodiff option for both RL models: log-likelihood to ~2e-14, gradient to ~5e-10, Gauss-Newton curvature positive definite where the exact Hessian is indefinite. **Verified but deliberately not wired into the package** — the shipped Jacobian is the NumPy finite difference. |

### 3.5 `examples/`

#### Modified

| File | | Lines |
|---|---|---|
| `examples/example_RL.py` | M | 245 → 332 |

- Adds `RL_model_trials` (and its counterpart for the second model), returning the per-trial
  log-likelihood vector instead of the summed scalar, as the worked demonstration of how to opt
  into the Gauss-Newton curvature.
- Passes the per-trial functions to `hbi_main` as well as to `individual_fit` — the point of
  Mod 11, and the case a user is most likely to miss.

| File | | Lines |
|---|---|---|
| `examples/example.py` | M | 184 → 189 |
| `examples/example_model_selection.py` | M | 180 → 186 |
| `examples/exampla_individual_fit.py` | M | 69 → 75 |

- All four examples previously imported through a path rooted outside the repository, so none of
  them ran from a clean checkout. Each now inserts the repository root on `sys.path` before
  importing, so `python examples/<name>.py` works without installing the package. (The
  misspelling of `exampla_individual_fit.py` is inherited from the original and left as is.)

| File | | Lines |
|---|---|---|
| `examples/README.md` | M | 10 → 19 |

- Documents the three new examples and points users at `example_regression.py` when their models
  are regression-style rather than choice-based.

#### New

| File | Lines | Purpose |
|---|---:|---|
| `examples/example_regression.py` | 335 | Classical `y = f(X | θ)` models with continuous outcomes: how to supply `model_trials`, and what changes when σ is estimated rather than profiled out. |
| `examples/example_group_bms.py` | 132 | The three group-BMS modes — families, between conditions, between groups. |
| `examples/example_display.py` | 238 | Minimal Mod 14 demonstration on a straight-line model, chosen so the figures rather than the model are the subject. |
| `examples/output/*.png`, `*.html` (6 files) | — | Committed example figures produced by the two display examples. Generated artefacts, regenerable by re-running the scripts. |

### 3.6 `benchmark/` — cross-implementation benchmark (all new)

Not part of the `cbm` package; it is the evidence base for §1.3.

**Drivers and shared code**

| File | Lines | Purpose |
|---|---:|---|
| `benchmark/simulate.py` | 699 | Deterministic ground-truth data generator, written once in two formats reading the same bytes so the Python and MATLAB arms fit literally identical data. |
| `benchmark/models.py` | 202 | The model likelihoods, defined once and shared by both Python arms, so that any difference in results is attributable to the optimiser rather than the model. |
| `benchmark/run_python_arms.py` | 286 | Runs the fork arm and the pristine-CBM arm over a grid. Fits the CBM arm subject-by-subject so that one pathological subject costs one subject rather than a whole cell. |
| `benchmark/run_hbi_arms.py` | 347 | The two-arm hierarchical benchmark (modified HBI vs frozen legacy HBI), including the seeding design that isolates sensitivity to how the supplied maps were fitted. |
| `benchmark/make_report_figures.py` | 1099 | Produces the five cross-arm figures and their generated caption sheet. |
| `benchmark/make_hbi_figures.py` | 488 | Produces the two HBI figures and their caption sheet. |
| `benchmark/RERUN.md` | 142 | The four commands that reproduce the pipeline end to end, with expected runtimes. Contains absolute paths from the author's machine, which need editing to run elsewhere. |

**MATLAB VBA arm**

| File | Lines |
|---|---:|
| `benchmark/run_vba_arm.m` | 151 |
| `benchmark/run_vba_grid.m` | 134 |
| `benchmark/run_vba_clean.m` | 78 |
| `benchmark/run_vba_value.m` | 103 |
| `benchmark/matlab/f_QlearnAsym2.m` | 65 |
| `benchmark/matlab/g_valuePower.m` | 63 |

- Read the same `.mat` datasets the Python arms read and invert each subject with
  `VBA_NLStateSpaceModel` under both candidate models of its family.
- The two `matlab/` files are the VBA-format evolution and observation functions for the
  asymmetric-learning-rate and power-value models.
- The `run_vba_*.m` drivers contain hard-coded absolute paths to the author's machine.

**Vendored pristine baseline**

| File | |
|---|---|
| `benchmark/external/cbm_original/*.py` (12 files) | N |
| `benchmark/external/README.md` | N, 13 lines |

- The unmodified upstream package, extracted verbatim from commit `e72193f` and verified
  pristine (539-line `optimization.py`, zero `MODIFICATION` markers). Tracked deliberately, and
  marked do-not-edit: it is the comparison baseline.
- The README also documents how to re-create the gitignored VBA-toolbox clone (shallow clone of
  `MBB-team/VBA-toolbox`; tested against MATLAB R2025b, headless via `matlab -batch`).

**Committed results**

| File | |
|---|---|
| `benchmark/results/figures/fig1…fig5` (PDF + PNG) | N |
| `benchmark/results/figures/hbi_fig1, hbi_fig2` (PDF + PNG) | N |
| `benchmark/results/figures/figures.md` (58 l) | N |
| `benchmark/results/figures/hbi_figures.md` (63 l) | N |
| `benchmark/results/*.log` (3 files) | N |

- Generated artefacts, committed so the reported numbers are inspectable without re-running the
  pipeline. The two `.md` files are generated caption sheets stating what each figure shows and
  how to read it; `hbi_figures.md` also records why there is no MATLAB arm in the hierarchical
  benchmark (`VBA_MFX` fits one model at a time and has no Dirichlet over model identity, so it
  produces no comparable model frequency — a property of VBA's design, not a gap in the
  benchmark).
- The corresponding datasets and per-arm result binaries are gitignored and regenerated by
  `RERUN.md`.

### 3.7 Incidental / generated

| File | |
|---|---|
| `cbm_local.egg-info/PKG-INFO`, `SOURCES.txt`, `dependency_links.txt`, `top_level.txt` | N |

- Setuptools build metadata from a local editable install (`cbm-local` 0.1.0), committed
  despite `*.egg-info/` appearing in `.gitignore` — it was presumably added to the index before
  the ignore rule existed. Not part of the source; a candidate for removal from tracking.
