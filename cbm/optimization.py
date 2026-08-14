"""
optimization_annotated.py

Base code  : optimization_original.py — CBM toolbox
             https://github.com/payampiray/cbm_python

Modifications documented inline, inspired by the VBA toolbox
             https://github.com/MBB-team/VBA-toolbox

─────────────────────────────────────────────────────────────────
OVERVIEW OF MODIFICATIONS
─────────────────────────────────────────────────────────────────
  1  Config.__post_init__  — activate bounds validation
  2  compute_hessian       — eigenvalue regularisation (always PD)
  3  _newton_polish        — new VBA-style Gauss-Newton refinement
  4  optimize              — defensive wrapping, single-pass,
                             Newton polish, VBA convergence flag
  5  compute_hessian       — VBA-style Gauss-Newton curvature,
                             opt-in via `trial_func` (replaces Mod 2
                             for models that expose per-trial log-lik)
  6  ConvergenceStatus     — explicit convergence status enum replacing
                             the always-True `converged` boolean of
                             Mod 3/4; explicit status→flag mapping
  7  monotonicity invariant — polish exit + optimize() boundary both
                             verify descent; RuntimeError on violation
  8  sign/naming coherence — objective is `neg_log_post` (minimized);
                             `F = −neg_log_post` property at boundary
  9  post-fit diagnostics  — raw min eigenvalue, clip count, cross-init
                             agreement, hard-bound mask; surfaced via
                             PostFitDiagnostics → FitMath.diagnostics
 10  weak-identifiability  — min curvature eigenvalue vs prior
                             precision; warns below ratio 2.0
─────────────────────────────────────────────────────────────────
Modifications 11-12 live in cbm/hbi.py + cbm/hbi_updates.py, not in
this file, because they concern the hierarchical layer:
 11  model_trials threaded through hbi_main → hbi_run → hbi_qhquad,
     so HBI's internal refits can use Mod 5's Gauss-Newton curvature
 12  HBI keeps the per-refit diagnostics it used to discard, on
     IndividualPosterior.diagnostics
─────────────────────────────────────────────────────────────────
Modifications 2-4 are interdependent and should be applied together:
  • Mod 2 guarantees a positive-definite Hessian for Mod 3's Newton step
  • Mod 3 provides the refinement used by Mod 4's optimize loop
  • Mod 4's flag logic relies on Mods 2-3 (Hessian always PD, ΔF
    convergence replaces gradient-norm check)
Modification 5 supersedes Modification 2 whenever the caller supplies
`trial_func`; Mod 2 remains the fallback for models that cannot expose
a per-trial decomposition (see Mod 5's docstring below).
Modification 6 supersedes Mod 4d's flag branch (the boolean it
branched on carried no information; see Mod 6's block below).
Modifications 7-9 implement DEV.md §3's remaining items and §4's
check layers (2026-08-03); Mod 1 was activated the same day (§4's
bounds checks live there). The pre-flight layer itself lives in
individual_fit._preflight_checks — checks that need `data` and the
prior cannot live in this file.
─────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.optimize import minimize, approx_fprime
from typing import Callable, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings


# ══════════════════════════════════════════════════════════════════
# MODIFICATION 6 — Explicit convergence status (replaces the
#                  always-True `converged` boolean of Mods 3/4)
# ──────────────────────────────────────────────────────────────────
#
class ConvergenceStatus(str, Enum):
    """
    How `_newton_polish` (Mod 3) exited. Exactly one status per fit;
    each member corresponds to one real exit path — no member is
    unreachable by design (a `CONVERGED_GRAD` member was considered
    and deliberately dropped: the polish never tests the gradient,
    so it would be permanently dead — DEV.md §2.2, 2026-08-03).

    Members
    -------
    CONVERGED_DF      The VBA criterion |ΔF|/(1+|F|) < tol_df was met.
    NO_IMPROVEMENT    Backtracking exhausted: 20 step-halvings (down to
                      ~1e-6) could not reduce f. Signature of already
                      sitting at the minimum, not of failure.
    MAX_STEPS         The 30-step Newton loop ended without meeting
                      tol_df. Monotonic descent guarantees f never
                      worsened, so the point is no worse than the
                      L-BFGS-B optimum it started from.
    SINGULAR_HESSIAN  np.linalg.solve raised LinAlgError: the Newton
                      direction could not be computed. Near-unreachable
                      (GN curvature is PD by construction; the Mod 2
                      fallback clips eigenvalues to 1e-4) — safety valve.

    Inherits from `str` so it serializes cleanly (JSON, pickle, repr)
    and compares equal to its literal value, e.g.
    `status == "converged_df"`.
    """
    CONVERGED_DF = "converged_df"
    NO_IMPROVEMENT = "no_improvement"
    MAX_STEPS = "max_steps"
    SINGULAR_HESSIAN = "singular_hessian"


# Explicit status → flag mapping. NEVER replace this with a truthiness
# test: `if status:` is true for EVERY member (the old bug, in enum
# form), and `if status is CONVERGED_DF:` silently demotes the two
# accept-paths below. Every member MUST appear here — a KeyError on a
# new member is the desired failure mode (forces a deliberate decision
# rather than a silent default).
# MODIFICATION 10 — threshold for the weak-identifiability warning,
# expressed as a multiple of the prior precision. Calibrated on 360 fits;
# the full justification lives at the warning site in `optimize`.
# Raising it flags more fits (higher recall, lower precision).
WEAK_IDENTIFIABILITY_RATIO = 2.0

FLAG_FROM_STATUS = {
    ConvergenceStatus.CONVERGED_DF: 1.0,
    ConvergenceStatus.NO_IMPROVEMENT: 1.0,
    ConvergenceStatus.MAX_STEPS: 1.0,
    ConvergenceStatus.SINGULAR_HESSIAN: 0.5,
}
#
# WHAT
#   Replace the boolean returned by `_newton_polish` — which was set
#   to True on every one of its four exit paths, i.e. a constant —
#   with a four-state enum naming the actual exit path, surfaced on
#   `OptimizationResult.convergence_status`. Map status → flag through
#   the explicit table above instead of Mod 4d's `if converged … elif
#   converged … else` branch (whose else-arm was dead code).
#
# WHY — a check that cannot report failure is not a check
#   Workstream 2 (DEV.md §2.2/§4) builds post-fit diagnostics on the
#   convergence status. With a constant `converged=True`, downstream
#   code (Mod 4d, individual_fit, HBI) could not distinguish "ΔF
#   criterion met" from "hit the iteration cap" from "Hessian solve
#   failed". The enum records the distinction; the flag mapping keeps
#   the *operational* behaviour identical for every currently-reachable
#   path (all → 1.0), so this Mod adds information without moving any
#   baseline number (verified: cbm/dev/baseline_snapshot.py --compare).
#
# WHY — flag values (decided 2026-08-03, DEV.md §2.2)
#   CONVERGED_DF    → 1.0  unambiguous success (the VBA criterion).
#   NO_IMPROVEMENT  → 1.0  already at the minimum (see enum docstring).
#   MAX_STEPS       → 1.0  VBA also accepts at max iter; Mod 4d's
#                          argument — a stable F with a noisy gradient
#                          is a valid fit — applies. The status field,
#                          not the flag, carries the nuance.
#   SINGULAR_HESSIAN→ 0.5 + warn. Not 0.0: flag == 0 triggers prior
#                          substitution in map_estimation.py:98,
#                          individual_fit.py:215-228 and (opt-out-free)
#                          hbi_updates.py:264-267, discarding the MAP
#                          that L-BFGS-B already found — exactly the
#                          silent data loss Mod 4d was written to
#                          prevent. 0.5 records the concern without
#                          destroying data (§4: a check never silently
#                          changes a result — it flags or stops).
#
# REFERENCE
#   • DEV.md §2.2 (design + decision log, 2026-08-03).
#   • VBA_NLStateSpaceModel.m: |ΔF|/(1+|F|) < tol is the sole
#     convergence criterion; VBA accepts at max iterations.
#   • Consumers of flag (all test `== 0` only; nothing reads 0.5):
#     map_estimation.py:98, individual_fit.py:215-228,
#     hbi_updates.py:264-267.
# ══════════════════════════════════════════════════════════════════


@dataclass
class Config:
    """
    Configuration for individual fitting.

    These parameters match BFGSOptimizer configuration.

    Attributes:
        d: Dimension of parameters
        range_bounds: 2×d array for parameter ranges
        tol_grad: Tolerance for gradient
        tol_grad_liberal: Liberal tolerance for bad subjects
        num_init: Number of random initializations
        num_init_med: Increased number for bad subjects
        num_init_up: Maximum number for bad subjects
        inits: Optional custom initialization points (n_inits × d array)
        max_iter: Maximum iterations per optimization run
        prior_for_failed: Whether to use prior for subjects with no good fit
        verbose: Whether to print progress
        save_data: Whether to save data in output
    """
    d: Optional[int] = None
    range_bounds: Optional[int | np.ndarray] = 5
    hard_bounds: Optional[int | np.ndarray] = 100
    tol_grad: float = 0.001001
    tol_grad_liberal: float = 0.1
    num_init: Optional[int] = None
    num_init_med: Optional[int] = None
    num_init_up: Optional[int] = None
    inits: Optional[np.ndarray] = None
    max_iter: int = 1000
    prior_for_failed: bool = True
    verbose: bool = True
    save_data: bool = False
    # MODIFICATION 14 — display. Off by default: when False the optimizer
    # retains nothing extra, so the cost is exactly zero for every existing
    # caller. When True it keeps the L-BFGS-B evaluation path, the Newton
    # polish trace and any warnings raised during the fit, which is what
    # FitResult.plot() draws. See DEV.md §17.
    display: bool = False

    def __post_init__(self):
        """Set defaults based on dimension."""
        if self.num_init is None:
            self.num_init = min(7 * self.d, 100)

        if self.num_init_med is None:
            self.num_init_med = self.num_init + 10
        elif self.num_init_med < self.num_init:
            raise ValueError("num_init_med must be >= num_init")

        if self.num_init_up is None:
            self.num_init_up = self.num_init_med + 10
        elif self.num_init_up < self.num_init_med:
            raise ValueError("num_init_up must be >= num_init_med")

        # ══════════════════════════════════════════════════════════════
        # MODIFICATION 1 — Bounds validation in Config  [ACTIVATED
        #                  2026-08-03 — was commented out; DEV.md §3
        #                  required "activate or delete", decided
        #                  activate: §4's pre-flight layer needs exactly
        #                  these checks (shapes 2×d, range ⊂ hard), and
        #                  Config is where d and both bounds first meet]
        # ──────────────────────────────────────────────────────────────
        #
        # WHAT
        #   Expand scalar/None bounds to proper 2×d arrays as soon as
        #   Config is created; reject wrong shapes; reject range_bounds
        #   not contained in hard_bounds (random initializations are
        #   drawn from range_bounds, and L-BFGS-B constrains iterates to
        #   hard_bounds — an init outside hard_bounds is a contradiction).
        #
        # RATIONALE — defence in depth
        #   In the original, scalar→array expansion only happens inside
        #   BFGSOptimizer.__init__.  If a Config object is inspected,
        #   logged, or passed to another component *before* reaching the
        #   optimizer, bounds may still be scalar and cause downstream
        #   shape-mismatch errors.  Validating early in Config guarantees
        #   a consistent 2×d shape regardless of usage path.
        #   BFGSOptimizer.__init__ keeps its own expansion as a second
        #   line of defence: deepcopy/unpickle do NOT re-run
        #   __post_init__, so a Config restored from an old pickle (as
        #   HBI does via profile.config) may still carry scalars.
        #
        # FAILURE BEHAVIOUR (§4: pre-flight → stop)
        #   Wrong shape or range ⊄ hard → raise ValueError.
        #   d is None → skip (nothing to validate against; the
        #   optimizer's own expansion covers that path).
        # ══════════════════════════════════════════════════════════════
        if self.d is not None:
            if self.range_bounds is None:
                self.range_bounds = np.array([
                    -5.0 * np.ones(self.d),
                    5.0 * np.ones(self.d)
                ])
            elif np.isscalar(self.range_bounds):
                self.range_bounds = np.array([
                    -self.range_bounds * np.ones(self.d),
                    self.range_bounds * np.ones(self.d)
                ])
            else:
                self.range_bounds = np.asarray(self.range_bounds, dtype=float)
                if self.range_bounds.shape != (2, self.d):
                    raise ValueError(
                        f"range_bounds must be 2×{self.d} array, "
                        f"got shape {self.range_bounds.shape}")
            if self.hard_bounds is None:
                self.hard_bounds = np.array([
                    -100.0 * np.ones(self.d),
                    100.0 * np.ones(self.d)
                ])
            elif np.isscalar(self.hard_bounds):
                self.hard_bounds = np.array([
                    -self.hard_bounds * np.ones(self.d),
                    self.hard_bounds * np.ones(self.d)
                ])
            else:
                self.hard_bounds = np.asarray(self.hard_bounds, dtype=float)
                if self.hard_bounds.shape != (2, self.d):
                    raise ValueError(
                        f"hard_bounds must be 2×{self.d} array, "
                        f"got shape {self.hard_bounds.shape}")
            # range_bounds ⊂ hard_bounds (§4 pre-flight)
            if (np.any(self.range_bounds[0] < self.hard_bounds[0])
                    or np.any(self.range_bounds[1] > self.hard_bounds[1])):
                raise ValueError(
                    "range_bounds must lie within hard_bounds "
                    f"(range {self.range_bounds.tolist()} vs "
                    f"hard {self.hard_bounds.tolist()})")

@dataclass
class OptimizationResult:
    """
    Result from BFGS optimization.

    Attributes:
        x: Optimized parameters (d-dimensional array)
        f: Optimal function value (scalar). SIGN CONVENTION (Mod 8):
           this is the minimized NEGATIVE log joint, f = −log p(y,θ*|m)
           = neg_log_post at the optimum. Use the `F` property for the
           VBA-convention (maximized) value.
        hess: Hessian matrix at optimum (d × d array), computed via finite differences
              Can be None for intermediate results
        grad: Gradient at optimum (d-dimensional array)
        flag: Success flag (1.0=full success, 0.5=partial success, 0.0=failed)
        success: Boolean indicating if scipy optimization succeeded
        nit: Number of iterations in best run
        n_runs: Total number of optimization runs attempted
        is_hess_pos: Whether Hessian is positive definite
        abs_g: Mean absolute gradient at optimum
        x_init: Initial point used for the best run
        hess_method: Which curvature `hess` came from — "gauss_newton"
              (Mod 5, VBA-style, PD by construction) or
              "finite_diff_clipped" (Mod 2 fallback, eigenvalue-floored)
        convergence_status: How the Newton polish exited (Mod 6) —
              see ConvergenceStatus. None on intermediate results
              (from _single_optimization, before the polish runs).
              `flag` is derived from this via FLAG_FROM_STATUS.
        hess_raw_min_eig: Smallest eigenvalue of the curvature BEFORE
              any regularisation (Mod 9). On the Mod 2 path a value
              below 1e-4 means the clip fired and log|H| — hence the
              evidence — is partly artifact (§2.1); on the GN path
              nothing is clipped and this is simply the smallest
              eigenvalue of JᵀJ + prior precision.
        hess_n_clipped: How many eigenvalues the Mod 2 floor raised
              (0 on the GN path by construction). Non-zero = the
              evidence for this fit inherited the §2.1 contamination.
        n_inits_agreeing: How many of the n_runs initializations ended
              within the ΔF tolerance of the best pre-polish optimum —
              the practical multimodality diagnostic (Mod 9):
              n_inits_agreeing == n_runs suggests a well-behaved
              surface; a low fraction means multiple local optima.
              Trivially 1 when num_init=1.
        at_hard_bounds: Boolean mask (d,) — parameters sitting exactly
              on a hard bound at the optimum (Mod 9). Any True means
              the MAP is a boundary point: the Laplace approximation
              (interior-optimum assumption) is invalid in that
              direction and a warning is emitted.
    """
    x: np.ndarray
    f: float
    hess: Optional[np.ndarray]
    grad: np.ndarray
    flag: float
    success: bool
    nit: int
    n_runs: int
    is_hess_pos: bool
    abs_g: float
    x_init: np.ndarray
    hess_method: str = "finite_diff_clipped"
    convergence_status: Optional[ConvergenceStatus] = None
    # Post-fit diagnostics (Mod 9) — informational only, never alter
    # the fit. None on intermediate results.
    hess_raw_min_eig: Optional[float] = None
    hess_n_clipped: Optional[int] = None
    n_inits_agreeing: Optional[int] = None
    at_hard_bounds: Optional[np.ndarray] = None
    # Mod 10: smallest curvature eigenvalue as a multiple of the prior
    # precision. Below WEAK_IDENTIFIABILITY_RATIO the fit is flagged as
    # weakly identified. None on the Mod 2 fallback (see Mod 10 block).
    weak_identifiability: Optional[float] = None
    # ── Mod 14 (display=True only; None otherwise) ────────────────
    # search_path: every point L-BFGS-B EVALUATED on the winning run,
    #   (n_evals, d). These are function evaluations including
    #   line-search probes, NOT clean iterations — the path zigzags, and
    #   the display labels the axis accordingly.
    # search_f: the objective at each of those points, sign-flipped to
    #   log-joint (Mod 8: the optimizer minimises the negative).
    # polish_path / polish_f / polish_lme: the Newton-polish steps.
    #   polish_lme is the ONLY place a genuine per-step log-evidence
    #   exists, because it is the only loop that recomputes H each step.
    search_path: Optional[np.ndarray] = None
    search_f: Optional[np.ndarray] = None
    polish_path: Optional[np.ndarray] = None
    polish_f: Optional[np.ndarray] = None
    polish_lme: Optional[np.ndarray] = None

    # ══════════════════════════════════════════════════════════════
    # MODIFICATION 8 — Sign/naming coherence (DEV.md §3)
    # ──────────────────────────────────────────────────────────────
    # WHAT
    #   Fix the sign convention at the result boundary. The optimizer
    #   MINIMIZES the negative log joint (neg_log_post); VBA MAXIMIZES
    #   free energy F. `f` is the minimized value; the two properties
    #   below give both conventions their unambiguous name so the
    #   manual (workstream 4) never has to hedge about signs.
    #
    # WHY — the incoherence was real, not cosmetic
    #   Before this Mod: map_estimation.log_posterior returned the
    #   POSITIVE log joint while its own docstring claimed "negative
    #   log posterior (for minimization)"; optimize_map's local
    #   `objective` silently negated it; FitMath.loglik actually
    #   stores the log JOINT (lik + prior), not the log-likelihood.
    #   Docstrings corrected in map_estimation.py/individual_fit.py;
    #   field names (`f`, `loglik`, `lme`) are kept — renaming them
    #   would break existing pickles and user analysis code — and the
    #   properties below provide the coherent vocabulary instead.
    #
    # PRECISION — what F is and is NOT
    #   F = −f = log p(y,θ*|m), the log JOINT at the optimum.
    #   It is NOT the Laplace log-evidence: that adds the curvature
    #   correction, lme = F + (d/2)·log(2π) − ½·log|H|
    #   (computed in individual_fit.py). Calling this quantity "F"
    #   follows DEV.md §3's instruction; do not read it as VBA's full
    #   variational free energy (which additionally bounds the
    #   evidence from below).
    # ══════════════════════════════════════════════════════════════
    @property
    def neg_log_post(self) -> float:
        """The minimized objective: −log p(y,θ*|m). Alias for `f`."""
        return self.f

    @property
    def F(self) -> float:
        """VBA-convention (maximized) value: F = −neg_log_post
        = log p(y,θ*|m), the log joint at the optimum. See Mod 8
        note above — this is not yet the Laplace evidence."""
        return -self.f

    def diagnostics(self) -> "PostFitDiagnostics":
        """Compact per-fit diagnostics record (Mod 9) for surfacing in
        higher-level results (FitMath.diagnostics). Arrays copied to
        plain lists so the record pickles small and prints readably."""
        return PostFitDiagnostics(
            convergence_status=(self.convergence_status.value
                                if self.convergence_status is not None else None),
            flag=self.flag,
            hess_method=self.hess_method,
            abs_grad=float(self.abs_g),
            hess_raw_min_eig=self.hess_raw_min_eig,
            hess_n_clipped=self.hess_n_clipped,
            n_inits_agreeing=self.n_inits_agreeing,
            n_runs=self.n_runs,
            at_hard_bounds=(self.at_hard_bounds.tolist()
                            if self.at_hard_bounds is not None else None),
            weak_identifiability=self.weak_identifiability,
            # Mod 14 — kept as arrays (not lists): they can run to a few
            # hundred rows and only exist when display=True anyway.
            search_path=self.search_path, search_f=self.search_f,
            polish_path=self.polish_path, polish_f=self.polish_f,
            polish_lme=self.polish_lme,
        )


@dataclass
class PostFitDiagnostics:
    """
    Per-subject post-fit diagnostics (DEV.md §4, layer 3; Mod 9).

    Everything here is informational — none of it alters the fit.
    Fields mirror OptimizationResult's diagnostic fields; see there
    for semantics. `convergence_status` is the ConvergenceStatus
    value string (Mod 6); `at_hard_bounds` a per-parameter bool list.
    """
    convergence_status: Optional[str]
    flag: float
    hess_method: str
    abs_grad: float
    hess_raw_min_eig: Optional[float]
    hess_n_clipped: Optional[int]
    n_inits_agreeing: Optional[int]
    n_runs: int
    at_hard_bounds: Optional[list]
    weak_identifiability: Optional[float] = None
    # Mod 14 — optimizer traces, populated only when Config.display is
    # True. See OptimizationResult for what each one holds and the
    # important caveat that search_path is function EVALUATIONS, not
    # iterations.
    search_path: Optional[np.ndarray] = None
    search_f: Optional[np.ndarray] = None
    polish_path: Optional[np.ndarray] = None
    polish_f: Optional[np.ndarray] = None
    polish_lme: Optional[np.ndarray] = None
    # Mod 14 — warnings raised while fitting THIS subject, recorded by
    # individual_fit when display=True (and re-emitted, never swallowed).
    warnings: Optional[list] = None


class BFGSOptimizer:
    """
    BFGS optimizer with multiple initializations and convergence criteria.
    The optimizer is configured at initialization and can be run multiple times
    with different functions.
    """

    def __init__(self,
                 d: int,
                 config: Config,
                 gtol: float = 1e-5,
                 ftol: float = 1e-9):
        """
        Initialize BFGS optimizer with configuration parameters.

        Args:
            config: Configuration object with optimization parameters
            gtol: Gradient tolerance for scipy optimizer
            ftol: Function tolerance for scipy optimizer
        """
        self.d = d
        self.tol_grad = config.tol_grad
        self.tol_grad_liberal = config.tol_grad_liberal
        self.num_init = config.num_init
        self.num_init_med = config.num_init_med
        self.num_init_up = config.num_init_up
        self.max_iter = config.max_iter
        self.range_bounds = config.range_bounds
        self.hard_bounds = config.hard_bounds
        self.inits = config.inits
        # Mod 14 — getattr so a Config built by older code (or unpickled
        # from before this modification) still constructs.
        self.display = bool(getattr(config, "display", False))
        self._display = False
        self._temp_polish_trace = None
        self.gtol = gtol
        self.ftol = ftol

        # History tracking
        self.history_x = []
        self.history_f = []
        self.all_results = []
        """Set defaults based on dimension."""
        if self.range_bounds is None:
            self.range_bounds = np.array([
                -5 * np.ones(self.d),
                5 * np.ones(self.d)
            ])
        elif np.isscalar(self.range_bounds):
            self.range_bounds = np.array([
                -self.range_bounds * np.ones(self.d),
                self.range_bounds * np.ones(self.d)
            ])
        if self.hard_bounds is None:
            self.hard_bounds = np.array([
                -100 * np.ones(self.d),
                100 * np.ones(self.d)
            ])
        elif np.isscalar(self.hard_bounds):
            self.hard_bounds = np.array([
                -self.hard_bounds * np.ones(self.d),
                self.hard_bounds * np.ones(self.d)
            ])
        else:
            if self.hard_bounds.shape != (2, self.d):
                raise ValueError(f"hard_bounds must be 2×{self.d} array, got shape {self.hard_bounds.shape}")
            self.hard_bounds = self.hard_bounds


    def compute_hessian(self,
                        neg_log_post: Callable[[np.ndarray], float],
                        x: np.ndarray,
                        epsilon: float = 1e-5,
                        trial_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                        prior_precision: Optional[np.ndarray] = None,
                        return_diagnostics: bool = False):
        """
        Compute the curvature used for the Newton step and (at the MAP)
        for the Laplace evidence.

        Args:
            neg_log_post: Objective (negative log joint, MINIMIZED —
                sign convention per Mod 8)
            x: Point at which to compute the curvature
            epsilon: Step size for the Mod 2 fallback (finite-difference Hessian)
            trial_func: Optional. Per-trial log-likelihood, shape (T,).
                If given, uses the Mod 5 Gauss-Newton curvature instead
                of Mod 2's finite-difference Hessian.
            prior_precision: Optional d×d prior precision, added exactly
                to the Gauss-Newton curvature (only used with trial_func).
            return_diagnostics: If True, return (H, diag) where diag is
                {"raw_min_eig": smallest eigenvalue BEFORE any clipping,
                 "n_clipped": how many eigenvalues the Mod 2 floor
                 raised (always 0 on the Gauss-Newton path — nothing is
                 clipped there)}. Post-fit diagnostics layer (Mod 9);
                 purely informational, never alters H.

        Returns:
            Hessian/curvature matrix (d × d), or (H, diag) if
            return_diagnostics.
        """
        if trial_func is not None:
            H = self._gauss_newton_curvature(trial_func, x, prior_precision)
            if return_diagnostics:
                diag = {"raw_min_eig": float(np.linalg.eigvalsh(H).min()),
                        "n_clipped": 0}
                return H, diag
            return H

        n = len(x)
        H = np.zeros((n, n))

        # Compute gradient at x
        grad_x = approx_fprime(x, neg_log_post, epsilon)

        # Compute gradient at x + epsilon*e_i for each dimension
        for i in range(n):
            x_step = x.copy()
            x_step[i] += epsilon
            grad_step = approx_fprime(x_step, neg_log_post, epsilon)
            H[i, :] = (grad_step - grad_x) / epsilon

        # Symmetrize
        # return (H + H.T) / 2

        # ══════════════════════════════════════════════════════════════
        # MODIFICATION 2 — Hessian eigenvalue regularisation
        # ──────────────────────────────────────────────────────────────
        #
        # Replace the return above with the following block:
        #
        H = (H + H.T) / 2
        eigvals, eigvecs = np.linalg.eigh(H)
        # Mod 9: record what the clip is about to do, BEFORE doing it —
        # the raw spectrum is the §2.1 contamination diagnostic.
        raw_min_eig = float(eigvals.min())
        n_clipped = int(np.sum(eigvals < 1e-4))
        eigvals = np.maximum(eigvals, 1e-4)
        H = (eigvecs * eigvals) @ eigvecs.T
        if return_diagnostics:
            return H, {"raw_min_eig": raw_min_eig, "n_clipped": n_clipped}
        return H
        #
        # WHAT
        #   After symmetrising the finite-difference Hessian, decompose
        #   it, clip every eigenvalue to a floor of 1e-4, and
        #   reconstruct.  The returned Hessian is positive-definite by
        #   construction.
        #
        # WHY — theory
        #   In the Laplace approximation the posterior covariance is
        #
        #       Σ_post = H⁻¹
        #
        #   where H is the Hessian of the negative log-posterior
        #   evaluated at the MAP estimate (Bishop, 2006).
        #   If H has zero or negative eigenvalues:
        #     • H is singular → Σ_post is undefined.
        #     • H⁻¹ can have negative diagonal → "negative variance",
        #       which is nonsensical.
        #     • The Laplace log-evidence
        #         log p(y|m) ≈ log p(y|θ*,m) + log p(θ*|m)
        #                      + (d/2) log(2π) − ½ log|H|
        #       requires  |H| > 0.
        #
        #   Clipping  λ_i → max(λ_i, ε)  is equivalent to adding a
        #   small ridge  H ← H + ε I  (restricted to the offending
        #   directions).  In Bayesian terms this encodes a vague prior
        #   that prevents infinite posterior variance in any direction,
        #   i.e. no parameter may have infinite posterior variance.
        #
        # WHY — numerical stability
        #   Finite-difference Hessians are noisy, especially for:
        #     • flat or nearly-flat likelihood surfaces
        #     • points near a saddle (some negative curvature)
        #     • objectives with numerical noise (e.g. from simulation)
        #   The regularised Hessian is always invertible, so Newton
        #   steps (Mod 3) and the Laplace approximation never fail.
        #
        # REFERENCE — VBA toolbox
        #   spm_nlsi_GN.m (SPM12) and VBA_GaussNewton.m both
        #   regularise the posterior precision (= Hessian) at every
        #   Gauss-Newton iteration to guarantee positive-definiteness.
        #   See Friston et al. (2007) "Variational free energy and the
        #   Laplace approximation", NeuroImage 34(1):220-234.
        # ══════════════════════════════════════════════════════════════


    # ══════════════════════════════════════════════════════════════════
    # MODIFICATION 5 — Gauss-Newton curvature (VBA-style, opt-in)
    # ──────────────────────────────────────────────────────────────────
    #
    # Add the following NEW method (does not exist in the original):
    #
    def _gauss_newton_curvature(self,
                                trial_func: Callable[[np.ndarray], np.ndarray],
                                x: np.ndarray,
                                prior_precision: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gauss-Newton curvature:  H = J^T J  (+ prior_precision).

        J = d(per-trial log-lik)/d(theta), shape (T, d), obtained by a
        SINGLE finite difference per parameter (T evaluations per column,
        not (n+1)^2 like Mod 2) — same step rule as VBA_numericDiff.m:
        relative step 1e-4*x, floored at 1e-4 in magnitude.
        """
        n = len(x)
        f0 = np.asarray(trial_func(x))          # (T,) per-trial log-lik
        J = np.zeros((f0.shape[0], n))

        for i in range(n):
            dx = 1e-4 * x[i]
            if abs(dx) <= 1e-4:
                dx = 1e-4
            x_step = x.copy()
            x_step[i] += dx
            J[:, i] = (np.asarray(trial_func(x_step)) - f0) / dx

        H = J.T @ J
        if prior_precision is not None:
            H = H + prior_precision
        return H
    #
    # WHAT
    #   Build the curvature as the outer product of the per-trial
    #   log-likelihood Jacobian, JᵀJ, plus the (exact) prior precision.
    #   Requires the caller to expose a `trial_func` returning the
    #   per-trial log-likelihood vector — the model's normal inner loop
    #   already computes this before summing it into a scalar, so most
    #   models only need to stop summing early to supply it.
    #
    # WHY — this is not an approximation of Mod 2, it replaces it
    #   The exact Hessian of the negative log-posterior is
    #
    #       ∇²J(θ) = Σₜ Jₜᵀ Qₜ Jₜ  −  Σₜ rₜᵀ Qₜ ∂²gₜ/∂θ²  +  Σ₀⁻¹
    #                └──────┬──────┘   └───────────┬───────────┘
    #                 kept (this Mod)      dropped (Gauss-Newton approx.)
    #
    #   Dropping the residual-weighted second-derivative term is the
    #   classical Gauss-Newton approximation for nonlinear least squares
    #   (Nocedal & Wright, 2006, ch.10) — exact as residuals → 0, and it
    #   is precisely the term responsible for indefinite curvature. The
    #   surviving term ΣJᵀQJ is a sum of quadratic forms with Q positive
    #   definite → positive-SEMI-definite by construction; adding the
    #   prior precision Σ₀⁻¹ (also PD) makes the sum strictly PD. No
    #   eigenvalue clipping is needed — there is nothing to clip.
    #
    #   In a flat (weakly-identified) direction, JᵀQJ → 0, so H → Σ₀⁻¹ in
    #   that direction: the posterior covariance falls back to the PRIOR
    #   covariance, not to an arbitrary constant like Mod 2's 1e-4 floor.
    #   This is the correct Bayesian answer and is exactly what resolves
    #   the evidence-contamination issue of Mod 2 (flat directions no
    #   longer get an evidence penalty that depends on a tuning constant).
    #
    # WHY — single differencing, not double
    #   Mod 2 finite-differences an already finite-differenced gradient
    #   ((n+1)^2 objective calls, noise ~1e-6, sitting next to its own
    #   1e-4 clip floor). This Mod differences the per-trial vector ONCE
    #   (n+1 model calls, each returning all T trials at once) — no
    #   compounding of differencing noise.
    #
    # REFERENCE — VBA toolbox (verified against this exact recipe)
    #   • core/VBA_Iphi.m:100 / core/VBA_Itheta.m:82 —
    #     `iSigma = iQ + precision * (Jacobian outer-product sum)`,
    #     the same H = JᵀJ + prior-precision construction, used for BOTH
    #     the Gauss-Newton step and (via core/VBA_Hpost.m:56 →
    #     core/VBA_FreeEnergy.m:132) the Laplace evidence — VBA never
    #     maintains two separate Hessians.
    #   • utils/VBA_numericDiff.m:46,72-79 — single forward-difference
    #     Jacobian, step `epsilon=1e-4 * x`, floored at `1e-4` — the
    #     exact step rule reproduced above.
    #   • utils/VBA_checkGN.m — VBA's rare-case safety valve (only fires
    #     if this curvature's smallest eigenvalue is non-positive, which
    #     cannot happen here by construction) also returns a flag
    #     counting how often it fired — worth mirroring later if this
    #     path ever needs its own diagnostic (see DEV.md §2.1).
    #
    # FUTURE OPTION — exact autodiff (not implemented here)
    #   Replace the finite-difference Jacobian above with an exact one
    #   (e.g. JAX `jacfwd`/`jacrev` on a JAX port of `trial_func`) to
    #   remove the remaining O(sqrt(eps)) finite-difference error. The
    #   H = JᵀJ + prior_precision formula is unchanged — only how J is
    #   obtained changes. See cbm/dev/rl_jax_verify.py, which already
    #   verifies this for both RL models (log-lik matches NumPy to
    #   2e-14, gradient to 5e-10, GN curvature PSD).
    # ══════════════════════════════════════════════════════════════════


    # ══════════════════════════════════════════════════════════════════
    # MODIFICATION 3 — VBA-style Gauss-Newton refinement
    # ──────────────────────────────────────────────────────────────────
    #
    # Add the following NEW method (does not exist in the original):
    #
    def _newton_polish(self, neg_log_post, x, n_steps=30, trial_func=None, prior_precision=None):
        """
        VBA-style Gauss-Newton refinement.
        Iterates until the free energy stops improving.
        Returns (x, f, status) with status a ConvergenceStatus (Mod 6)
        naming the exit path taken.

        neg_log_post: the MINIMIZED objective (sign convention, Mod 8).
        trial_func/prior_precision: see compute_hessian (Mod 5). If
        trial_func is None, falls back to the Mod 2 finite-difference
        Hessian, as before.
        """
        f_current = neg_log_post(x)
        f_entry = f_current    # for the Mod 7 monotonicity invariant
        status = None          # set by exactly one exit path (Mod 6)
        tol_df = 1e-4          # relative free-energy tolerance

        # MODIFICATION 14 — polish trace. Only here is a per-step LOG-EVIDENCE
        # actually available: the Laplace evidence needs |H|, and this loop is
        # the only place H is recomputed at every step. During L-BFGS-B there
        # is no Hessian, so no evidence exists for those evaluations — the
        # display panel plots the two segments separately rather than
        # pretending one curve spans the whole fit.
        trace = [] if getattr(self, "_display", False) else None

        for _ in range(n_steps):
            H = self.compute_hessian(neg_log_post, x,
                                      trial_func=trial_func,
                                      prior_precision=prior_precision)
            g = approx_fprime(x, neg_log_post, 1e-5)

            if trace is not None:
                # lme = −f + (d/2)·log(2π) − ½·log|H|  (Mod 8 sign
                # convention: f is the NEGATIVE log joint).
                try:
                    sign, logdet = np.linalg.slogdet(H)
                    lme_step = (-f_current + 0.5 * self.d * np.log(2 * np.pi)
                                - 0.5 * logdet) if sign > 0 else np.nan
                except np.linalg.LinAlgError:
                    lme_step = np.nan
                trace.append((x.copy(), float(f_current), float(lme_step)))

            try:
                dx = np.linalg.solve(H, g)       # Newton direction
            except np.linalg.LinAlgError:
                status = ConvergenceStatus.SINGULAR_HESSIAN
                break

            # Backtracking line search (halving up to 2^-20 ≈ 1e-6)
            step = 1.0
            improved = False
            for _ in range(20):
                x_new = x - step * dx
                x_new = np.clip(x_new,
                                self.hard_bounds[0],
                                self.hard_bounds[1])
                f_new = neg_log_post(x_new)
                if np.isfinite(f_new) and f_new < f_current:
                    improved = True
                    break
                step *= 0.5

            if not improved:
                # Cannot reduce f → already at minimum
                status = ConvergenceStatus.NO_IMPROVEMENT
                break

            delta_f = abs(f_current - f_new)
            x, f_current = x_new, f_new

            # VBA convergence criterion
            if delta_f / (1.0 + abs(f_current)) < tol_df:
                status = ConvergenceStatus.CONVERGED_DF
                break

        # Exhausted steps → accept (VBA also accepts at max iter)
        if status is None:
            status = ConvergenceStatus.MAX_STEPS

        # ══════════════════════════════════════════════════════════════
        # MODIFICATION 7 — Monotonic objective as a checked invariant
        # ──────────────────────────────────────────────────────────────
        # WHAT
        #   Verify, at every exit of the polish, that the returned
        #   objective is no worse than the objective at entry. A second
        #   check at the optimize() boundary (see 4c') verifies the
        #   polish never worsened the L-BFGS-B optimum.
        #
        # WHY
        #   Backtracking (Mod 3) only accepts steps with f_new <
        #   f_current, so monotonic descent is structural — but a
        #   structural property that is never *checked* can be silently
        #   broken by a future edit (e.g. someone "simplifying" the
        #   acceptance test) or by a non-deterministic objective, and
        #   would then corrupt every downstream evidence value. DEV.md
        #   §3/§4: promote the local guard to a checked global
        #   invariant; a violation means the results cannot be trusted,
        #   so the defined failure behaviour is STOP (raise), not warn.
        #
        # REFERENCE
        #   VBA's move-halving (core/VBA_GN.m:159-160) provides the same
        #   structural guarantee; DEV.md §3 (damping investigation,
        #   2026-08-03) established that backtracking is the correct
        #   VBA-equivalent mechanism, this makes it verified.
        # ══════════════════════════════════════════════════════════════
        if f_current > f_entry:
            raise RuntimeError(
                "MONOTONICITY INVARIANT VIOLATED in _newton_polish: "
                f"objective rose from {f_entry!r} to {f_current!r}. "
                "This indicates a bug or a non-deterministic objective; "
                "results cannot be trusted (DEV.md §3/§4).")

        # MOD 14 — record the final state too, so the trace ends at the
        # accepted optimum rather than at the last point BEFORE the last
        # accepted step. Stashed on self; optimize() moves it onto the
        # winning run's result.
        if trace is not None:
            try:
                H = self.compute_hessian(neg_log_post, x,
                                         trial_func=trial_func,
                                         prior_precision=prior_precision)
                sign, logdet = np.linalg.slogdet(H)
                lme_end = (-f_current + 0.5 * self.d * np.log(2 * np.pi)
                           - 0.5 * logdet) if sign > 0 else np.nan
            except np.linalg.LinAlgError:
                lme_end = np.nan
            trace.append((x.copy(), float(f_current), float(lme_end)))
            self._temp_polish_trace = trace

        return x, f_current, status
    #
    # WHAT
    #   After L-BFGS-B finds an approximate minimum, refine it with
    #   full Newton steps  x ← x − α H⁻¹ g  using the exact
    #   (regularised) Hessian.  Convergence is determined by the
    #   relative change in the objective ("free energy"), NOT by the
    #   gradient norm.
    #
    # WHY — theory (Newton refinement)
    #   L-BFGS-B uses a limited-memory Hessian approximation that can
    #   be inaccurate near the optimum, especially for ill-conditioned
    #   problems.  A full Newton step with the true Hessian has
    #   QUADRATIC convergence near the minimum (Nocedal & Wright, 2006), 
    #   meaning the distance to the optimum squares at each
    #   iteration — far faster than the super-linear convergence of
    #   quasi-Newton.  With the regularised Hessian (Mod 2) the Newton
    #   direction is always well-defined.
    #
    # WHY — theory (ΔF convergence criterion)
    #   In variational Bayes the quantity being maximised is the free
    #   energy  F = log p(y|m) − KL(q||p)  (evidence lower bound).
    #   The natural stopping rule is therefore
    #
    #       |ΔF| / (1 + |F|)  <  tol
    #
    #   rather than  |∇F| < tol, because:
    #
    #   (a) F is the actual optimisation target — if it has stopped
    #       changing, the MAP/variational estimate is stable.
    #
    #   (b) In flat directions the gradient is numerically noisy even
    #       when the function value is perfectly stable.  A gradient-
    #       norm criterion would reject these valid fits or demand
    #       wasteful extra random restarts.
    #
    #   (c) Gradient norms are scale-dependent: a gradient of 0.01 may
    #       be "large" for one parameter and "small" for another,
    #       depending on the parameter's curvature.  The ΔF criterion
    #       is scale-free.
    #
    # WHY — line search
    #   The raw Newton step can overshoot (especially early on or when
    #   the quadratic approximation is poor far from the minimum).
    #   Halving the step length until f decreases guarantees MONOTONIC
    #   descent, preventing divergence.
    #
    # REFERENCE
    #   • VBA_GaussNewton.m implements this refinement loop.
    #   • The criterion |ΔF|/(1+|F|) < tol appears in
    #     VBA_NLStateSpaceModel.m and originates from SPM's
    #     spm_nlsi_GN.m.
    #   • Friston et al. (2007) "Variational free energy and the
    #     Laplace approximation", NeuroImage 34(1):220-234.
    #   • Daunizeau et al. (2014) "VBA: A Probabilistic Treatment of
    #     Nonlinear Models for Neurobiological and Behavioural Data",
    #     PLoS Computational Biology 10(1):e1003441.
    # ══════════════════════════════════════════════════════════════════


    def _single_optimization(self,
                             neg_log_post: Callable[[np.ndarray], float],
                             x_init: np.ndarray) -> OptimizationResult:
        """
        Run a single optimization from given initial point.

        Args:
            neg_log_post: Objective (minimized; sign convention per Mod 8)
            x_init: Initial point

        Returns:
            OptimizationResult with hess=None (computed later for best run only)
        """
        # Track function evaluations for this run
        run_history_x = []
        run_history_f = []

        def func_wrapper(x):
            f = neg_log_post(x)
            run_history_x.append(x.copy())
            run_history_f.append(f)
            return f

        # Convert range_bounds to scipy bounds format
        bounds = [(self.hard_bounds[0, i], self.hard_bounds[1, i])
                  for i in range(self.d)]

        # Run L-BFGS-B optimizer
        result = minimize(
            func_wrapper,
            x_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'gtol': self.gtol,
                'ftol': self.ftol,
                'disp': False
            }
        )

        # Extract results
        x_opt = result.x
        f_opt = result.fun

        # Compute gradient at optimum using finite differences
        epsilon = 1e-8
        grad = approx_fprime(x_opt, neg_log_post, epsilon)

        # Check if inverse Hessian from L-BFGS is positive definite
        # This is cheap and good enough for selecting best run
        try:
            # Convert to dense if needed
            if hasattr(result.hess_inv, 'todense'):
                hess_inv_dense = result.hess_inv.todense()
            elif hasattr(result.hess_inv, 'matvec'):
                n = self.d
                hess_inv_dense = np.zeros((n, n))
                for i in range(n):
                    e = np.zeros(n)
                    e[i] = 1.0
                    hess_inv_dense[:, i] = result.hess_inv.matvec(e)
            else:
                hess_inv_dense = result.hess_inv

            # Check if positive definite
            np.linalg.cholesky(hess_inv_dense)
            is_hess_pos = True
        except (np.linalg.LinAlgError, AttributeError):
            is_hess_pos = False

        # Compute mean absolute gradient
        abs_g = np.mean(np.abs(grad))

        # Store history temporarily (not in OptimizationResult)
        self._temp_history_x = run_history_x
        self._temp_history_f = run_history_f

        return OptimizationResult(
            x=x_opt,
            f=f_opt,
            hess=None,  # Computed later for best run only
            grad=grad,
            flag=0.0,  # Computed later
            success=result.success,
            nit=result.nit,
            n_runs=1,  # Single run
            is_hess_pos=is_hess_pos,
            abs_g=abs_g,
            x_init=x_init.copy()
        )

    # def optimize(self,
    #              func: Callable[[np.ndarray], float],
    #              x_init: Optional[np.ndarray] = None) -> OptimizationResult:
    #     """
    #     Optimize the given function.

    #     If x_init is provided, uses it PLUS num_init random initializations.
    #     Otherwise, uses only num_init random initializations.

    #     The number of initializations adapts based on convergence quality:
    #     - If flag=1.0: Uses num_init initializations
    #     - If flag=0.5: Tries up to num_init_med initializations
    #     - If flag=0.0: Tries up to num_init_up initializations

    #     Args:
    #         func: Objective function that takes x (numpy array of length d) and returns scalar
    #         x_init: Optional initial point (length d array). If provided, will be used in addition to random starts

    #     Returns:
    #         OptimizationResult dataclass with all optimization results
    #     """
    #     self.all_results = []

    #     # Determine initial number of attempts
    #     n_attempts = self.num_init

    #     while True:
    #         # Generate list of initial points for this round
    #         init_points = []

    #         # Add user-provided initial point if given (only on first iteration)
    #         if x_init is not None and len(self.all_results) == 0:
    #             if len(x_init) != self.d:
    #                 raise ValueError(f"x_init must have length {self.d}, got {len(x_init)}")
    #             init_points.append(x_init)

    #         # Add user-provided initial point given through config
    #         if self.inits is not None and len(self.all_results) == 0:
    #             if len(self.inits) != self.d:
    #                 raise ValueError(f"inits must have length {self.d}, got {len(self.inits)}")
    #             init_points.append(self.inits)

    #         # Determine how many random inits to add this round
    #         n_random = n_attempts - len(self.all_results)

    #         # Fill remaining with random initializations
    #         n_needed = n_random - (len(init_points) - (1 if x_init is not None and len(self.all_results) == 0 else 0))
    #         if n_needed > 0:
    #             random_inits = np.random.uniform(
    #                 low=self.range_bounds[0, :],
    #                 high=self.range_bounds[1, :],
    #                 size=(n_needed, self.d)
    #             )
    #             for init_pt in random_inits:
    #                 init_points.append(init_pt)

    #         # Run optimization from each initial point
    #         best_f = np.inf
    #         best_result = None
    #         best_history_x = []
    #         best_history_f = []

    #         for i, x0 in enumerate(init_points):
    #             result = self._single_optimization(func, x0)
    #             self.all_results.append(result)

    #             # Keep track of best result (lowest function value)
    #             if result.f < best_f:
    #                 best_f = result.f
    #                 best_result = result
    #                 best_history_x = self._temp_history_x
    #                 best_history_f = self._temp_history_f

    #         # Store history from best run
    #         self.history_x = best_history_x
    #         self.history_f = best_history_f

    #         # Compute Hessian only for the best result using finite differences
    #         hess = self.compute_hessian(func, best_result.x, epsilon=1e-5)

    #         # Re-check if Hessian is positive definite (using actual Hessian this time)
    #         try:
    #             np.linalg.cholesky(hess)
    #             is_hess_pos = True
    #         except np.linalg.LinAlgError:
    #             is_hess_pos = False

    #         # Determine flag based on convergence criteria
    #         flag = 0.0

    #         if best_result.success and is_hess_pos and (best_result.abs_g < self.tol_grad):
    #             flag = 1.0  # Full success
    #         elif is_hess_pos and (best_result.abs_g < self.tol_grad_liberal):
    #             flag = 0.5  # Partial success
    #         else:
    #             flag = 0.0  # Failed

    #         # Check if we need more attempts
    #         if flag == 1.0:
    #             # Success! We're done
    #             break
    #         elif flag == 0.5 and len(self.all_results) < self.num_init_med:
    #             # Partial success, try more initializations
    #             n_attempts = self.num_init_med
    #             continue
    #         elif flag == 0.0 and len(self.all_results) < self.num_init_up:
    #             # Failed, try even more initializations
    #             n_attempts = self.num_init_up
    #             continue
    #         else:
    #             # We've tried enough, stop here
    #             break

    #     # Throw warnings based on final flag
    #     if flag == 0.0:
    #         warnings.warn(f"--- No positive hessian found in spite of {len(self.all_results)} initialization.")
    #     elif flag == 0.5:
    #         warnings.warn(
    #             f"Positive hessian found, but not a good gradient in spite of {len(self.all_results)} initialization.")

    #     return OptimizationResult(
    #         x=best_result.x,
    #         f=best_result.f,
    #         hess=hess,
    #         grad=best_result.grad,
    #         flag=flag,
    #         success=best_result.success,
    #         nit=best_result.nit,
    #         n_runs=len(self.all_results),
    #         is_hess_pos=is_hess_pos,
    #         abs_g=best_result.abs_g,
    #         x_init=best_result.x_init
    #     )

    # ══════════════════════════════════════════════════════════════════
    # MODIFICATION 4 — Rewritten optimize method
    # ──────────────────────────────────────────────────────────────────
    #
    # Replace the ENTIRE optimize method above with the version below.
    # It combines four sub-changes (4a–4d) explained after the code.
    #
    def optimize(self, neg_log_post, x_init=None, trial_func=None, prior_precision=None):
        """
        neg_log_post: the objective to MINIMIZE — the negative log
        joint −log p(y,θ|m) (sign convention per Mod 8; VBA's F is the
        negative of this, see OptimizationResult.F).

        trial_func/prior_precision: optional, see compute_hessian (Mod 5).
        When supplied, the Newton polish and the returned Hessian use the
        VBA-style Gauss-Newton curvature instead of the Mod 2 fallback.
        """
        self.all_results = []
        n_attempts = self.num_init

        # ── 4a. Defensive function wrapping ──────────────────────
        _raw_neg_log_post = neg_log_post
        def neg_log_post(x):
            with np.errstate(over='ignore', invalid='ignore',
                            divide='ignore'):
                try:
                    f = float(_raw_neg_log_post(x))
                    if np.isfinite(f):
                        return f
                except Exception:
                    pass
            return 1e20

        # ── 4b. Single-pass initialisation (no retry loop) ───────
        init_points = []
        if x_init is not None:
            if len(x_init) != self.d:
                raise ValueError(
                    f"x_init must have length {self.d}")
            init_points.append(x_init)
        if self.inits is not None:
            if len(self.inits) != self.d:
                raise ValueError(
                    f"inits must have length {self.d}")
            init_points.append(self.inits)

        n_needed = n_attempts - len(init_points)
        if n_needed > 0:
            random_inits = np.random.uniform(
                low=self.range_bounds[0, :],
                high=self.range_bounds[1, :],
                size=(n_needed, self.d)
            )
            for init_pt in random_inits:
                init_points.append(init_pt)

        best_f = np.inf
        best_result = None
        best_history_x, best_history_f = [], []

        for x0 in init_points:
            result = self._single_optimization(neg_log_post, x0)
            self.all_results.append(result)
            if result.f < best_f:
                best_f = result.f
                best_result = result
                best_history_x = self._temp_history_x
                best_history_f = self._temp_history_f

        self.history_x = best_history_x
        self.history_f = best_history_f

        # MOD 14 — arm the polish trace for this fit only. Read by
        # _newton_polish; left unset (falsy) when display is off, so the
        # tracing branch there never runs.
        self._display = bool(getattr(self, "display", False))
        self._temp_polish_trace = None

        # ── 4c. Newton polish (requires Mod 3) ───────────────────
        f_before_polish = best_result.f
        best_result.x, best_result.f, status = (
            self._newton_polish(neg_log_post, best_result.x,
                                 trial_func=trial_func,
                                 prior_precision=prior_precision)
        )
        # ── 4c'. Monotonicity at the optimize() boundary (Mod 7) ─
        # The polish must never return a worse point than L-BFGS-B
        # found. Structural (backtracking) — but checked, per §4.
        if best_result.f > f_before_polish:
            raise RuntimeError(
                "MONOTONICITY INVARIANT VIOLATED in optimize: Newton "
                f"polish worsened the objective ({f_before_polish!r} "
                f"→ {best_result.f!r}); results cannot be trusted "
                "(DEV.md §3/§4, MODIFICATION 7).")
        best_result.grad = approx_fprime(
            best_result.x, neg_log_post, 1e-8)
        best_result.abs_g = np.mean(np.abs(best_result.grad))

        # Curvature at the optimum — Gauss-Newton (Mod 5) if trial_func
        # was supplied, else the regularised finite-difference Hessian
        # (Mod 2). Both are positive-definite by construction.
        hess, hess_diag = self.compute_hessian(
            neg_log_post, best_result.x, epsilon=1e-5,
            trial_func=trial_func, prior_precision=prior_precision,
            return_diagnostics=True)
        hess_method = "gauss_newton" if trial_func is not None else "finite_diff_clipped"
        is_hess_pos = True

        # ══════════════════════════════════════════════════════════
        # MODIFICATION 9 — Post-fit diagnostics (DEV.md §4, layer 3)
        # ──────────────────────────────────────────────────────────
        # WHAT
        #   Surface, on the result object, the four §4 post-fit
        #   diagnostics: gradient norm (abs_g, pre-existing), raw
        #   minimum eigenvalue + clip count (hess_diag above),
        #   cross-initialization agreement, and the Mod 6 convergence
        #   status. Plus a boundary check: parameters railed at
        #   hard_bounds invalidate the Laplace approximation in that
        #   direction (interior-optimum assumption).
        #
        # WHY
        #   §4 principle: a check never silently changes a result —
        #   every value below is informational; the only side effect
        #   is a warning when the MAP sits on a hard bound.
        #   n_inits_agreeing is the practical multimodality test the
        #   manual's interpretation section is built on: agreement is
        #   measured on PRE-polish optima with the same ΔF-style
        #   relative tolerance the polish uses (tol_df = 1e-4).
        #
        # REFERENCE
        #   DEV.md §4 (post-fit layer); §2.1 (why the raw spectrum
        #   matters for evidence); utils/VBA_checkGN.m (VBA's own
        #   fired-regularizer counter, mirrored here by n_clipped).
        # ══════════════════════════════════════════════════════════
        # Cross-initialization agreement (pre-polish optima)
        tol_df = 1e-4
        n_inits_agreeing = int(sum(
            1 for res in self.all_results
            if abs(res.f - f_before_polish) / (1.0 + abs(f_before_polish)) < tol_df))

        # Boundary check — exact equality is correct here: the polish
        # clips to hard_bounds, and L-BFGS-B respects them as box
        # constraints, so a railed parameter sits exactly on the bound.
        at_hard_bounds = ((best_result.x <= self.hard_bounds[0])
                          | (best_result.x >= self.hard_bounds[1]))
        if np.any(at_hard_bounds):
            warnings.warn(
                "MAP estimate sits on hard_bounds for parameter(s) "
                f"{np.where(at_hard_bounds)[0].tolist()}; the Laplace "
                "approximation assumes an interior optimum, so the "
                "evidence for this fit is unreliable (Mod 9, DEV.md §4).")

        # ══════════════════════════════════════════════════════════
        # MODIFICATION 10 — Weak-identifiability warning
        # ──────────────────────────────────────────────────────────
        # WHAT
        #   Compare the smallest curvature eigenvalue against the PRIOR
        #   precision. If the data added little curvature relative to
        #   the prior in the weakest direction, the posterior there is
        #   essentially the prior: the parameter is weakly identified
        #   and its estimate should not be interpreted.
        #
        # WHY a RATIO, not an absolute floor
        #   min_eig is the curvature of the log-posterior, so it grows
        #   with the amount of data. Measured on RL at alpha=0.5:
        #   T=50 -> 1.44, T=150 -> 5.46, T=450 -> 14.6. An absolute
        #   threshold would therefore flag every small study and miss
        #   every large one. Dividing by the prior precision removes the
        #   scale and gives the quantity a meaning: "how much more does
        #   the data know than the prior did?"
        #
        # WHY the threshold is 2 (calibrated 2026-08-12, n=360 fits over
        # RL alpha 0.001-0.999 x T 60/150/300)
        #   ratio band   median |error|   fits with |error| > 0.15
        #     0-2            0.206              53%
        #     2-5            0.045              12%
        #     5-20           0.066              25%
        #     20-100         0.051               7%
        #   corr(log ratio, |error|) = -0.38, vs -0.27 for the raw
        #   eigenvalue — the ratio is the better predictor.
        #   At ratio < 2: precision 0.53, recall 0.48.
        #
        # HONEST LIMITS — read before trusting it
        #   WHAT IT PREDICTS is error in the UNCONSTRAINED (theta) space,
        #   which is the space this curvature refers to. Validated on an
        #   18-cell boundary sweep (2026-08-12): AUC 0.824 on RL and 0.766
        #   on POW (out of sample — the threshold was calibrated on RL
        #   only), versus ~0.5 for n_inits_agreeing and abs_grad. Flagged
        #   fits carry 4.7x (RL) / 5.2x (POW) the median theta-error.
        #   Against error in the NATIVE space (alpha, rho) it is near
        #   chance on RL, and that is expected rather than a defect: as
        #   alpha -> 1 the sigmoid saturates, so a large theta error maps
        #   to a tiny alpha error. A parameter can be badly identified in
        #   the fitting space while its transformed estimate still looks
        #   close to truth. Read the warning as "this parameter is poorly
        #   constrained", not as "this number is far from the truth".
        #
        #   Precision ~0.5 means about half of flagged fits are fine.
        #   This is a TRIAGE signal ("look at this one"), not a
        #   rejection rule, and it is deliberately a warning rather than
        #   an error: §4's principle is that a check flags or stops, and
        #   stopping here would discard usable fits. Recall ~0.5 also
        #   means it misses half of the bad fits — a fit can be wrong
        #   for reasons that leave curvature healthy (multimodality,
        #   which n_inits_agreeing covers instead).
        #
        # Only computed on the Gauss-Newton path: there `prior_precision`
        # is known exactly and enters H by construction. On the Mod 2
        # fallback the eigenvalues were already clipped, so the raw
        # spectrum is not comparable to a prior scale.
        weak_identifiability = None
        if prior_precision is not None and hess_diag["raw_min_eig"] is not None:
            pp = np.asarray(prior_precision, dtype=float)
            pp_min = float(np.linalg.eigvalsh(pp).min()) if pp.ndim == 2 else float(pp)
            if pp_min > 0:
                weak_identifiability = float(hess_diag["raw_min_eig"] / pp_min)
                if weak_identifiability < WEAK_IDENTIFIABILITY_RATIO:
                    warnings.warn(
                        "Weakly identified fit: the smallest curvature "
                        f"eigenvalue is only {weak_identifiability:.2f}x the "
                        "prior precision, so the data barely constrain the "
                        "weakest parameter direction and the posterior there "
                        "is close to the prior. Treat the affected estimate "
                        "as unreliable (Mod 10; triage signal, precision "
                        "~0.5 — see the block in optimization.py).")

        # ── 4d. Convergence flag from explicit status (Mod 6) ────
        # Explicit table lookup — see FLAG_FROM_STATUS for why this
        # must never be a truthiness or identity test on the status.
        flag = FLAG_FROM_STATUS[status]
        if status is ConvergenceStatus.SINGULAR_HESSIAN:
            warnings.warn(
                "Newton polish aborted: singular Hessian in "
                "np.linalg.solve (convergence_status="
                f"{status.value}); accepting the L-BFGS-B optimum "
                "with flag=0.5.")

        # MOD 14 — attach the traces, but only when display is on. The
        # arrays are built here rather than during the fit so the hot loop
        # stays untouched; when display is off every field below is None
        # and nothing was retained in the first place.
        s_path = s_f = p_path = p_f = p_lme = None
        if self._display:
            if best_history_x:
                s_path = np.asarray(best_history_x, dtype=float)
                # Mod 8: the optimizer minimises −log joint; flip the sign
                # so the display plots something that goes UP as the fit
                # improves, which is what a reader expects.
                s_f = -np.asarray(best_history_f, dtype=float)
            tr = getattr(self, "_temp_polish_trace", None)
            if tr:
                p_path = np.asarray([t[0] for t in tr], dtype=float)
                p_f = -np.asarray([t[1] for t in tr], dtype=float)
                p_lme = np.asarray([t[2] for t in tr], dtype=float)

        return OptimizationResult(
            x=best_result.x, f=best_result.f, hess=hess,
            grad=best_result.grad, flag=flag,
            success=True,
            nit=best_result.nit,
            n_runs=len(self.all_results),
            is_hess_pos=is_hess_pos,
            abs_g=best_result.abs_g,
            x_init=best_result.x_init,
            hess_method=hess_method,
            convergence_status=status,
            hess_raw_min_eig=hess_diag["raw_min_eig"],
            hess_n_clipped=hess_diag["n_clipped"],
            n_inits_agreeing=n_inits_agreeing,
            at_hard_bounds=at_hard_bounds,
            weak_identifiability=weak_identifiability,
            search_path=s_path, search_f=s_f,
            polish_path=p_path, polish_f=p_f, polish_lme=p_lme
        )
    #
    # ──────────────────────────────────────────────────────────────────
    # SUB-CHANGE 4a — Defensive function wrapping
    #
    #   WHAT:  Wrap the user-supplied objective so that any call that
    #          returns NaN / Inf / raises an exception silently returns
    #          a large finite penalty (1e20).
    #
    #   WHY:   Complex computational models (reinforcement learning,
    #          neural/DCM, simulation-based) routinely produce non-
    #          finite values for extreme parameter combinations (e.g.
    #          exp(100), log(0), division by near-zero probabilities).
    #          Without protection the L-BFGS-B call crashes and the
    #          entire fit is lost.  Returning 1e20 naturally steers the
    #          search away from pathological regions while keeping the
    #          optimiser alive.
    #          np.errstate suppresses the cascade of NumPy warnings
    #          that would otherwise flood the console.
    #
    #   REFERENCE:  Standard practice in any production-grade
    #          optimisation wrapper.  VBA_GaussNewton.m uses try/catch
    #          around the observation/evolution functions.
    #
    # ──────────────────────────────────────────────────────────────────
    # SUB-CHANGE 4b — Single-pass (no retry loop)
    #
    #   WHAT:  Remove the while-True loop that increases the number of
    #          random initialisations when flag < 1.0.  Run all
    #          L-BFGS-B starts in a single pass, then polish.
    #
    #   WHY:   In the original, the retry loop is driven by two
    #          failure modes:
    #            (i)  non-positive-definite Hessian  → flag = 0.0
    #            (ii) gradient norm too large         → flag ≤ 0.5
    #          With Mods 2 and 3:
    #            • The regularised Hessian is ALWAYS positive-definite
    #              → failure mode (i) is eliminated.
    #            • Newton polish converges via ΔF criterion → failure
    #              mode (ii) is replaced by a criterion that does not
    #              depend on gradient norm.
    #          Therefore the retry mechanism has nothing left to retry
    #          for.  Removing it avoids running 10-20 extra L-BFGS-B
    #          starts (each expensive) that would not change the
    #          outcome.
    #
    #   REFERENCE:  VBA_NLStateSpaceModel.m does not retry with new
    #          random starts; it relies on the Gauss-Newton loop to
    #          converge from the given initial point.
    #
    # ──────────────────────────────────────────────────────────────────
    # SUB-CHANGE 4c — Newton polish after L-BFGS-B
    #
    #   (See Modification 3 for full rationale.)
    #
    # ──────────────────────────────────────────────────────────────────
    # SUB-CHANGE 4d — VBA-style convergence flag
    #          [flag *branch* superseded by MODIFICATION 6 — the
    #          boolean it tested was constant, so the else-arm was
    #          dead code. The flag is now FLAG_FROM_STATUS[status];
    #          see Mod 6's block at the top of this file. The WHY
    #          below still holds: it is the rationale for mapping
    #          MAX_STEPS / NO_IMPROVEMENT to flag=1.0.]
    #
    #   WHAT:  Accept the fit (flag = 1.0) when the Newton polish
    #          exits through any descent path, REGARDLESS of the
    #          gradient norm.
    #
    #   Original flag logic:
    #     flag=1.0 : scipy success  AND  Hessian PD  AND  |∇| < 0.001
    #     flag=0.5 : Hessian PD  AND  |∇| < 0.1
    #     flag=0.0 : otherwise  →  may trigger prior_for_failed
    #
    #   Flag logic as of Mod 6 (explicit, see FLAG_FROM_STATUS):
    #     flag=1.0 : CONVERGED_DF / NO_IMPROVEMENT / MAX_STEPS
    #     flag=0.5 : SINGULAR_HESSIAN (+ warning; near-unreachable,
    #                curvature is PD by construction on both paths)
    #
    #   WHY — theory:
    #   The gradient norm is not the right convergence diagnostic for
    #   the Laplace approximation.  Consider a parameter θ_k that the
    #   data barely constrain (flat likelihood direction).  At the MAP:
    #
    #     • ∂F/∂θ_k  may be noisy / non-zero  (flat → gradient is
    #       dominated by numerical noise in finite differences)
    #     • yet the free energy F has genuinely converged  (the
    #       function value is stable to many decimal places)
    #
    #   Rejecting this fit (flag=0.0) and substituting the prior loses
    #   whatever information the likelihood *did* provide, and biases
    #   the group-level estimate (the affected subject contributes the
    #   prior rather than their actual posterior).
    #
    #   The correct treatment is to ACCEPT the fit and let the Laplace
    #   machinery handle the uncertainty: the large posterior variance
    #   in the flat direction (H⁻¹ is large there) correctly downweights
    #   that subject in hierarchical / random-effects analyses and
    #   correctly penalises the log-model-evidence (BIC / Laplace
    #   approximation).
    #
    #   In other words: a converged free energy with a large gradient
    #   is not a failure — it is a well-characterised uncertain
    #   parameter, which is exactly the information the Bayesian
    #   framework is designed to propagate.
    #
    #   WHY — practical consequence:
    #   In the original, subjects whose models are hard to fit (noisy
    #   data, many parameters, weak effects) frequently end up with
    #   flag=0.0.  If prior_for_failed=True, their fit is replaced by
    #   the prior — discarding valid likelihood information.  The
    #   modified flag avoids this silent data loss.
    #
    #   REFERENCE:
    #     • VBA_NLStateSpaceModel.m uses  |ΔF|/(1+|F|) < tol  as the
    #       sole convergence criterion and does not check gradient
    #       norms.
    #     • Friston et al. (2007) NeuroImage 34(1):220-234.
    #     • Daunizeau et al. (2014) PLoS Comp Biol 10(1):e1003441.
    # ══════════════════════════════════════════════════════════════════

    def get_all_results(self) -> List[OptimizationResult]:
        """
        Get detailed results from all optimization runs.

        Returns:
            List of OptimizationResult objects, one for each run (with hess=None)
        """
        return self.all_results

    def get_history(self) -> tuple:
        """
        Get optimization history from the best run.

        Returns:
            Tuple of (history_x, history_f) where:
                - history_x: List of x values tried
                - history_f: List of function values
        """
        return self.history_x, self.history_f


# Example usage
if __name__ == "__main__":
    # Define test function (Rosenbrock)
    def rosenbrock(x):
        """Rosenbrock function"""
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


    # Create optimizer for 4-dimensional problem
    # [MOD 1 cleanup 2026-08-03: this demo predated the Config-based
    #  constructor and no longer matched BFGSOptimizer's signature]
    optimizer = BFGSOptimizer(
        d=4,
        config=Config(
            d=4,
            range_bounds=np.array([[-2, -2, -2, -2], [2, 2, 2, 2]]),
            num_init=10
        )
    )

    print("=" * 70)
    print("Test 1: Multiple Random Initializations (no x_init provided)")
    print("=" * 70)

    # Optimize without providing initial point (uses num_init random starts)
    result = optimizer.optimize(rosenbrock)

    print(f"Optimal x: {result.x}")
    print(f"Optimal f: {result.f:.6e}")
    print(f"Flag: {result.flag} (1.0=full success, 0.5=partial, 0.0=failed)")
    print(f"Success: {result.success}")
    print(f"Mean |grad|: {result.abs_g:.6e}")
    print(f"Number of runs: {result.n_runs}")
    print(f"Iterations (best run): {result.nit}")
    print(f"Hessian positive definite: {result.is_hess_pos}")
    print(f"Initial point of best run: {result.x_init}")

    print("\nHessian matrix:")
    print(result.hess)
    print(f"Condition number: {np.linalg.cond(result.hess):.2e}")
    print(f"Determinant: {np.linalg.det(result.hess):.2e}")

    print("\n" + "=" * 70)
    print("Test 2: With Provided Initial Point (x_init + num_init random)")
    print("=" * 70)

    # Optimize with specific initial point PLUS random initializations
    x_init = np.array([0.5, 0.5, 0.5, 0.5])
    result2 = optimizer.optimize(rosenbrock, x_init=x_init)

    print(f"Provided x_init: {x_init}")
    print(f"Total runs: {result2.n_runs} (1 from x_init + {result2.n_runs - 1} random)")
    print(f"Optimal x: {result2.x}")
    print(f"Optimal f: {result2.f:.6e}")
    print(f"Flag: {result2.flag}")
    print(f"Mean |grad|: {result2.abs_g:.6e}")
    print(f"Initial point of best run: {result2.x_init}")

    print("\nHessian matrix:")
    print(result2.hess)
    print(f"Condition number: {np.linalg.cond(result2.hess):.2e}")

    # Get optimization history
    history_x, history_f = optimizer.get_history()
    print(f"\nOptimization trajectory: {len(history_f)} function evaluations")
    print(f"Function value progress: {history_f[0]:.3e} -> {history_f[-1]:.3e}")

    print("\n" + "=" * 70)
    print("Accessing result fields")
    print("=" * 70)
    print(f"result.x = {result.x}")
    print(f"result.f = {result.f:.6e}")
    print(f"result.flag = {result.flag}")
    print(f"result.hess = {result.hess}")