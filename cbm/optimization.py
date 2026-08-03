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
─────────────────────────────────────────────────────────────────
Modifications 2-4 are interdependent and should be applied together:
  • Mod 2 guarantees a positive-definite Hessian for Mod 3's Newton step
  • Mod 3 provides the refinement used by Mod 4's optimize loop
  • Mod 4's flag logic relies on Mods 2-3 (Hessian always PD, ΔF
    convergence replaces gradient-norm check)
Modification 5 supersedes Modification 2 whenever the caller supplies
`trial_func`; Mod 2 remains the fallback for models that cannot expose
a per-trial decomposition (see Mod 5's docstring below).
─────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.optimize import minimize, approx_fprime
from typing import Callable, Optional, List
from dataclasses import dataclass
import warnings


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
        # MODIFICATION 1 — Activate bounds validation in Config
        # ──────────────────────────────────────────────────────────────
        #
        # In the modified version the following commented-out block is
        # UNCOMMENTED so that range_bounds and hard_bounds are expanded
        # to proper 2×d arrays as soon as Config is created.
        #
        # RATIONALE — defence in depth
        #   In the original, scalar→array expansion only happens inside
        #   BFGSOptimizer.__init__.  If a Config object is inspected,
        #   logged, or passed to another component *before* reaching the
        #   optimizer, bounds may still be scalar and cause downstream
        #   shape-mismatch errors.  Validating early in Config guarantees
        #   a consistent 2×d shape regardless of usage path.
        # ══════════════════════════════════════════════════════════════
        # if self.range_bounds is None:
        #     self.range_bounds = np.array([
        #         -5 * np.ones(self.d),
        #         5 * np.ones(self.d)
        #     ])
        # elif np.isscalar(self.range_bounds):
        #     self.range_bounds = np.array([
        #         -self.range_bounds * np.ones(self.d),
        #         self.range_bounds * np.ones(self.d)
        #     ])
        # else:
        #     if self.range_bounds.shape != (2, self.d):
        #         raise ValueError(f"range_bounds must be 2×{self.d} array, got shape {self.range_bounds.shape}")
        #     self.range_bounds = self.range_bounds
        # if self.hard_bounds is None:
        #     self.hard_bounds = np.array([
        #         -100 * np.ones(self.d),
        #         100 * np.ones(self.d)
        #     ])
        # elif np.isscalar(self.hard_bounds):
        #     self.hard_bounds = np.array([
        #         -self.hard_bounds * np.ones(self.d),
        #         self.hard_bounds * np.ones(self.d)
        #     ])
        # else:
        #     if self.hard_bounds.shape != (2, self.d):
        #         raise ValueError(f"hard_bounds must be 2×{self.d} array, got shape {self.hard_bounds.shape}")
        #     self.hard_bounds = self.hard_bounds

@dataclass
class OptimizationResult:
    """
    Result from BFGS optimization.

    Attributes:
        x: Optimized parameters (d-dimensional array)
        f: Optimal function value (scalar)
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
                        func: Callable[[np.ndarray], float],
                        x: np.ndarray,
                        epsilon: float = 1e-5,
                        trial_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                        prior_precision: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the curvature used for the Newton step and (at the MAP)
        for the Laplace evidence.

        Args:
            func: Objective function (negative log posterior)
            x: Point at which to compute the curvature
            epsilon: Step size for the Mod 2 fallback (finite-difference Hessian)
            trial_func: Optional. Per-trial log-likelihood, shape (T,).
                If given, uses the Mod 5 Gauss-Newton curvature instead
                of Mod 2's finite-difference Hessian.
            prior_precision: Optional d×d prior precision, added exactly
                to the Gauss-Newton curvature (only used with trial_func).

        Returns:
            Hessian/curvature matrix (d × d)
        """
        if trial_func is not None:
            return self._gauss_newton_curvature(trial_func, x, prior_precision)

        n = len(x)
        H = np.zeros((n, n))

        # Compute gradient at x
        grad_x = approx_fprime(x, func, epsilon)

        # Compute gradient at x + epsilon*e_i for each dimension
        for i in range(n):
            x_step = x.copy()
            x_step[i] += epsilon
            grad_step = approx_fprime(x_step, func, epsilon)
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
        eigvals = np.maximum(eigvals, 1e-4)
        H = (eigvecs * eigvals) @ eigvecs.T
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
    def _newton_polish(self, func, x, n_steps=30, trial_func=None, prior_precision=None):
        """
        VBA-style Gauss-Newton refinement.
        Iterates until the free energy stops improving.
        Returns (x, f, converged).

        trial_func/prior_precision: see compute_hessian (Mod 5). If
        trial_func is None, falls back to the Mod 2 finite-difference
        Hessian, as before.
        """
        f_current = func(x)
        converged = False
        tol_df = 1e-4          # relative free-energy tolerance

        for _ in range(n_steps):
            H = self.compute_hessian(func, x,
                                      trial_func=trial_func,
                                      prior_precision=prior_precision)
            g = approx_fprime(x, func, 1e-5)

            try:
                dx = np.linalg.solve(H, g)       # Newton direction
            except np.linalg.LinAlgError:
                converged = True
                break

            # Backtracking line search (halving up to 2^-20 ≈ 1e-6)
            step = 1.0
            improved = False
            for _ in range(20):
                x_new = x - step * dx
                x_new = np.clip(x_new,
                                self.hard_bounds[0],
                                self.hard_bounds[1])
                f_new = func(x_new)
                if np.isfinite(f_new) and f_new < f_current:
                    improved = True
                    break
                step *= 0.5

            if not improved:
                # Cannot reduce f → already at minimum
                converged = True
                break

            delta_f = abs(f_current - f_new)
            x, f_current = x_new, f_new

            # VBA convergence criterion
            if delta_f / (1.0 + abs(f_current)) < tol_df:
                converged = True
                break

        # Exhausted steps → accept (VBA also accepts at max iter)
        if not converged:
            converged = True

        return x, f_current, converged
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
                             func: Callable[[np.ndarray], float],
                             x_init: np.ndarray) -> OptimizationResult:
        """
        Run a single optimization from given initial point.

        Args:
            func: Objective function
            x_init: Initial point

        Returns:
            OptimizationResult with hess=None (computed later for best run only)
        """
        # Track function evaluations for this run
        run_history_x = []
        run_history_f = []

        def func_wrapper(x):
            f = func(x)
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
        grad = approx_fprime(x_opt, func, epsilon)

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
    def optimize(self, func, x_init=None, trial_func=None, prior_precision=None):
        """
        trial_func/prior_precision: optional, see compute_hessian (Mod 5).
        When supplied, the Newton polish and the returned Hessian use the
        VBA-style Gauss-Newton curvature instead of the Mod 2 fallback.
        """
        self.all_results = []
        n_attempts = self.num_init

        # ── 4a. Defensive function wrapping ──────────────────────
        _raw_func = func
        def func(x):
            with np.errstate(over='ignore', invalid='ignore',
                            divide='ignore'):
                try:
                    f = float(_raw_func(x))
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
            result = self._single_optimization(func, x0)
            self.all_results.append(result)
            if result.f < best_f:
                best_f = result.f
                best_result = result
                best_history_x = self._temp_history_x
                best_history_f = self._temp_history_f

        self.history_x = best_history_x
        self.history_f = best_history_f

        # ── 4c. Newton polish (requires Mod 3) ───────────────────
        best_result.x, best_result.f, converged = (
            self._newton_polish(func, best_result.x,
                                 trial_func=trial_func,
                                 prior_precision=prior_precision)
        )
        best_result.grad = approx_fprime(
            best_result.x, func, 1e-8)
        best_result.abs_g = np.mean(np.abs(best_result.grad))

        # Curvature at the optimum — Gauss-Newton (Mod 5) if trial_func
        # was supplied, else the regularised finite-difference Hessian
        # (Mod 2). Both are positive-definite by construction.
        hess = self.compute_hessian(
            func, best_result.x, epsilon=1e-5,
            trial_func=trial_func, prior_precision=prior_precision)
        hess_method = "gauss_newton" if trial_func is not None else "finite_diff_clipped"
        is_hess_pos = True

        # ── 4d. VBA-style convergence flag ───────────────────────
        if converged and best_result.abs_g < self.tol_grad:
            flag = 1.0       # perfect convergence
        elif converged:
            flag = 1.0       # ΔF converged → accept
        else:
            flag = 0.5       # should not occur
            warnings.warn(
                "Newton polish did not converge after "
                f"{len(self.all_results)} initialisations.")

        return OptimizationResult(
            x=best_result.x, f=best_result.f, hess=hess,
            grad=best_result.grad, flag=flag,
            success=True,
            nit=best_result.nit,
            n_runs=len(self.all_results),
            is_hess_pos=is_hess_pos,
            abs_g=best_result.abs_g,
            x_init=best_result.x_init,
            hess_method=hess_method
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
    #
    #   WHAT:  Always set flag = 1.0 when Newton polish converges (ΔF
    #          criterion met), REGARDLESS of the gradient norm.
    #
    #   Original flag logic:
    #     flag=1.0 : scipy success  AND  Hessian PD  AND  |∇| < 0.001
    #     flag=0.5 : Hessian PD  AND  |∇| < 0.1
    #     flag=0.0 : otherwise  →  may trigger prior_for_failed
    #
    #   Modified flag logic:
    #     flag=1.0 : Newton polish converged (ΔF)
    #     flag=0.5 : should never occur (_newton_polish always returns
    #                converged=True, even at max iterations)
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
    optimizer = BFGSOptimizer(
        d=4,
        range_bounds=np.array([[-2, -2, -2, -2], [2, 2, 2, 2]]),
        num_init=10
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