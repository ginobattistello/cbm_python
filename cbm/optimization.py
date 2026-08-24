"""MAP optimization for CBM.

Final architecture
------------------
1. Multi-start L-BFGS-B finds a robust MAP candidate.
2. If a per-trial log-likelihood is supplied, a Gauss-Newton (GN)
   polish refines the MAP:
       H_opt = J.T @ J + prior_precision
   This curvature is used ONLY for optimization.
3. At the final MAP, compute a separate observed Hessian of the full
   negative log posterior:
       - "central_fd" (default): direct central finite differences
       - "autodiff": JAX Hessian, when the modeller supplies a JAX
         negative-log-posterior callable.
4. Never clip the final observed Hessian. If it is non-positive-
   definite, keep the MAP but flag the Laplace approximation as invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional
import warnings

import numpy as np
from scipy.optimize import minimize


class ConvergenceStatus(str, Enum):
    """How the post-L-BFGS optimization stage exited."""

    CONVERGED_DF = "converged_df"
    NO_IMPROVEMENT = "no_improvement"
    MAX_STEPS = "max_steps"
    SINGULAR_CURVATURE = "singular_curvature"
    SKIPPED_NO_TRIAL_FUNC = "skipped_no_trial_func"


@dataclass
class Config:
    """Configuration for individual MAP fitting.

    ``tol_grad``, ``tol_grad_liberal``, ``num_init_med`` and
    ``num_init_up`` are retained for compatibility with existing CBM
    configs/pickles. The final optimizer does not use adaptive restart
    logic or gradient-threshold acceptance.
    """

    d: Optional[int] = None
    range_bounds: Optional[int | np.ndarray] = 5
    hard_bounds: Optional[int | np.ndarray] = 100

    # Backward-compatible fields.
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
    display: bool = False

    # Final observed-Hessian backend.
    hessian_method: str = "central_fd"
    hessian_step: float = 1e-4

    def __post_init__(self):
        if self.d is None:
            return

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

        self.range_bounds = _expand_bounds(
            self.range_bounds, self.d, default=5.0, name="range_bounds"
        )
        self.hard_bounds = _expand_bounds(
            self.hard_bounds, self.d, default=100.0, name="hard_bounds"
        )

        if (
            np.any(self.range_bounds[0] < self.hard_bounds[0])
            or np.any(self.range_bounds[1] > self.hard_bounds[1])
        ):
            raise ValueError("range_bounds must lie within hard_bounds")

        method = str(self.hessian_method).lower()
        if method not in {"central_fd", "autodiff"}:
            raise ValueError(
                "hessian_method must be 'central_fd' or 'autodiff'"
            )
        self.hessian_method = method

        if self.hessian_step <= 0:
            raise ValueError("hessian_step must be > 0")


def _expand_bounds(bounds, d: int, default: float, name: str) -> np.ndarray:
    """Return bounds as a validated 2 x d float array."""
    if bounds is None:
        return np.array([-default * np.ones(d), default * np.ones(d)])

    if np.isscalar(bounds):
        value = float(bounds)
        return np.array([-value * np.ones(d), value * np.ones(d)])

    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, d):
        raise ValueError(
            f"{name} must be a 2 x {d} array, got {bounds.shape}"
        )
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError(f"Each lower {name} must be < upper bound")
    return bounds


@dataclass
class PostFitDiagnostics:
    """Compact diagnostics for one MAP fit."""

    convergence_status: Optional[str]
    flag: float
    hess_method: str
    abs_grad: float
    hess_raw_min_eig: Optional[float]
    hess_n_clipped: int
    hess_condition_number: Optional[float]
    laplace_valid: bool
    n_inits_agreeing: Optional[int]
    n_runs: int
    at_hard_bounds: Optional[list]
    weak_identifiability: Optional[float] = None

    # Only populated when Config.display=True.
    search_path: Optional[np.ndarray] = None
    search_f: Optional[np.ndarray] = None
    polish_path: Optional[np.ndarray] = None
    polish_f: Optional[np.ndarray] = None
    polish_lme: Optional[np.ndarray] = None
    warnings: Optional[list] = None


@dataclass
class OptimizationResult:
    """Result of MAP optimization.

    ``hess`` is always the independent post-MAP observed Hessian, never
    the GN optimization curvature.

    ``flag`` describes MAP/diagnostic quality:
      1.0 = accepted MAP with valid Laplace curvature
      0.5 = MAP retained with an optimization/Laplace warning
      0.0 = reserved for higher-level catastrophic failure handling
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

    hess_method: str = "central_fd"
    convergence_status: Optional[ConvergenceStatus] = None

    # Minimum eigenvalue of the unmodified observed Hessian.
    hess_raw_min_eig: Optional[float] = None
    # Kept at zero for compatibility; the final Hessian is never clipped.
    hess_n_clipped: int = 0
    hess_condition_number: Optional[float] = None
    laplace_valid: bool = False

    n_inits_agreeing: Optional[int] = None
    at_hard_bounds: Optional[np.ndarray] = None
    weak_identifiability: Optional[float] = None

    search_path: Optional[np.ndarray] = None
    search_f: Optional[np.ndarray] = None
    polish_path: Optional[np.ndarray] = None
    polish_f: Optional[np.ndarray] = None

    # Retained for compatibility with existing display/result code.
    # Evidence is intentionally not computed during GN polishing.
    polish_lme: Optional[np.ndarray] = None

    @property
    def neg_log_post(self) -> float:
        """Minimized objective: -log p(y, theta | model)."""
        return self.f

    @property
    def F(self) -> float:
        """Log joint at the MAP, before the Laplace curvature term."""
        return -self.f

    def diagnostics(self) -> PostFitDiagnostics:
        return PostFitDiagnostics(
            convergence_status=(
                self.convergence_status.value
                if self.convergence_status is not None
                else None
            ),
            flag=self.flag,
            hess_method=self.hess_method,
            abs_grad=float(self.abs_g),
            hess_raw_min_eig=self.hess_raw_min_eig,
            hess_n_clipped=0,
            hess_condition_number=self.hess_condition_number,
            laplace_valid=self.laplace_valid,
            n_inits_agreeing=self.n_inits_agreeing,
            n_runs=self.n_runs,
            at_hard_bounds=(
                self.at_hard_bounds.tolist()
                if self.at_hard_bounds is not None
                else None
            ),
            weak_identifiability=self.weak_identifiability,
            search_path=self.search_path,
            search_f=self.search_f,
            polish_path=self.polish_path,
            polish_f=self.polish_f,
            polish_lme=None,
        )


class BFGSOptimizer:
    """Multi-start L-BFGS-B + optional GN polish + observed Hessian."""

    def __init__(
        self,
        d: int,
        config: Config,
        gtol: float = 1e-5,
        ftol: float = 1e-9,
    ):
        self.d = d
        self.max_iter = config.max_iter
        self.num_init = config.num_init

        # Defensive re-expansion supports old/unpickled Config objects.
        self.range_bounds = _expand_bounds(
            config.range_bounds, d, default=5.0, name="range_bounds"
        )
        self.hard_bounds = _expand_bounds(
            config.hard_bounds, d, default=100.0, name="hard_bounds"
        )

        self.inits = config.inits
        self.display = bool(getattr(config, "display", False))
        self.hessian_method = str(
            getattr(config, "hessian_method", "central_fd")
        ).lower()
        self.hessian_step = float(
            getattr(config, "hessian_step", 1e-4)
        )

        self.gtol = gtol
        self.ftol = ftol

        self.history_x: list[np.ndarray] = []
        self.history_f: list[float] = []
        self.all_results: list[OptimizationResult] = []

        self._temp_history_x = []
        self._temp_history_f = []
        self._temp_polish_trace = None

    # -----------------------------------------------------------------
    # Numerical derivatives
    # -----------------------------------------------------------------

    @staticmethod
    def _central_gradient(
        fun: Callable[[np.ndarray], float],
        x: np.ndarray,
        relative_step: float = 1e-6,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)

        for i in range(x.size):
            h = relative_step * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            g[i] = (fun(xp) - fun(xm)) / (2.0 * h)

        return g

    def _gauss_newton_curvature(
        self,
        trial_func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        prior_precision: Optional[np.ndarray],
    ) -> np.ndarray:
        """Optimization-only score-outer-product/GN curvature.

        H_opt = J.T @ J + prior_precision

        This matrix is used only to generate the polish direction. It is
        never reused as the final evidence Hessian.
        """
        x = np.asarray(x, dtype=float)
        f0 = np.asarray(trial_func(x), dtype=float).reshape(-1)
        J = np.zeros((f0.size, x.size), dtype=float)

        for i in range(x.size):
            dx = 1e-4 * x[i]
            if abs(dx) <= 1e-4:
                dx = 1e-4

            xp = x.copy()
            xp[i] += dx
            fp = np.asarray(trial_func(xp), dtype=float).reshape(-1)
            J[:, i] = (fp - f0) / dx

        H = J.T @ J
        if prior_precision is not None:
            H = H + np.asarray(prior_precision, dtype=float)

        return 0.5 * (H + H.T)

    def _central_fd_hessian(
        self,
        neg_log_post: Callable[[np.ndarray], float],
        x: np.ndarray,
    ) -> np.ndarray:
        """Observed Hessian of the full negative log posterior.

        Uses h_i = hessian_step * max(1, |x_i|). This replaces the
        original CBM nested-forward finite-difference Hessian as the
        default post-MAP estimator.
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        H = np.zeros((n, n), dtype=float)

        h = self.hessian_step * np.maximum(1.0, np.abs(x))
        f0 = float(neg_log_post(x))

        for i in range(n):
            ei = np.zeros(n)
            ei[i] = h[i]

            H[i, i] = (
                neg_log_post(x + ei)
                - 2.0 * f0
                + neg_log_post(x - ei)
            ) / (h[i] ** 2)

            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = h[j]

                Hij = (
                    neg_log_post(x + ei + ej)
                    - neg_log_post(x + ei - ej)
                    - neg_log_post(x - ei + ej)
                    + neg_log_post(x - ei - ej)
                ) / (4.0 * h[i] * h[j])

                H[i, j] = Hij
                H[j, i] = Hij

        return 0.5 * (H + H.T)

    @staticmethod
    def _autodiff_hessian(
        neg_log_post_jax: Callable,
        x: np.ndarray,
    ) -> np.ndarray:
        """Observed Hessian from a modeller-supplied JAX objective."""
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                "hessian_method='autodiff' requires JAX. "
                "Install JAX and supply neg_log_post_jax."
            ) from exc

        jax.config.update("jax_enable_x64", True)
        xj = jnp.asarray(x, dtype=jnp.float64)
        H = np.asarray(jax.hessian(neg_log_post_jax)(xj), dtype=float)
        return 0.5 * (H + H.T)

    def compute_hessian(
        self,
        neg_log_post: Callable[[np.ndarray], float],
        x: np.ndarray,
        neg_log_post_jax: Optional[Callable] = None,
        return_diagnostics: bool = False,
    ):
        """Compute the final observed Hessian at the MAP.

        This method never returns GN curvature and never clips
        eigenvalues.
        """
        if self.hessian_method == "central_fd":
            H = self._central_fd_hessian(neg_log_post, x)
            method = "central_fd"

        elif self.hessian_method == "autodiff":
            if neg_log_post_jax is None:
                raise ValueError(
                    "Config.hessian_method='autodiff' but no "
                    "neg_log_post_jax was supplied."
                )
            H = self._autodiff_hessian(neg_log_post_jax, x)
            method = "autodiff"

        else:
            raise ValueError(
                f"Unknown hessian_method: {self.hessian_method}"
            )

        eigvals = np.linalg.eigvalsh(H)
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])
        is_pd = bool(min_eig > 0.0)

        condition_number = (
            float(max_eig / min_eig) if is_pd else np.inf
        )

        diag = {
            "method": method,
            "raw_min_eig": min_eig,
            "n_clipped": 0,
            "is_positive_definite": is_pd,
            "condition_number": condition_number,
        }

        if return_diagnostics:
            return H, diag
        return H

    # -----------------------------------------------------------------
    # L-BFGS-B
    # -----------------------------------------------------------------

    def _single_optimization(
        self,
        neg_log_post: Callable[[np.ndarray], float],
        x_init: np.ndarray,
    ) -> OptimizationResult:
        run_history_x = []
        run_history_f = []

        def wrapper(x):
            f = neg_log_post(x)
            run_history_x.append(x.copy())
            run_history_f.append(float(f))
            return f

        bounds = [
            (self.hard_bounds[0, i], self.hard_bounds[1, i])
            for i in range(self.d)
        ]

        scipy_result = minimize(
            wrapper,
            x_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self.max_iter,
                "gtol": self.gtol,
                "ftol": self.ftol,
                "disp": False,
            },
        )

        x_opt = np.asarray(scipy_result.x, dtype=float)
        f_opt = float(scipy_result.fun)
        grad = self._central_gradient(neg_log_post, x_opt)
        abs_g = float(np.mean(np.abs(grad)))

        self._temp_history_x = run_history_x
        self._temp_history_f = run_history_f

        return OptimizationResult(
            x=x_opt,
            f=f_opt,
            hess=None,
            grad=grad,
            flag=0.0,
            success=bool(scipy_result.success),
            nit=int(scipy_result.nit),
            n_runs=1,
            is_hess_pos=False,
            abs_g=abs_g,
            x_init=np.asarray(x_init, dtype=float).copy(),
        )

    # -----------------------------------------------------------------
    # GN polish
    # -----------------------------------------------------------------

    def _newton_polish(
        self,
        neg_log_post: Callable[[np.ndarray], float],
        x: np.ndarray,
        trial_func: Callable[[np.ndarray], np.ndarray],
        prior_precision: Optional[np.ndarray],
        n_steps: int = 30,
        tol_df: float = 1e-4,
    ):
        """GN refinement of the MAP.

        The final observed-Hessian backend is intentionally not used
        here.
        """
        x = np.asarray(x, dtype=float).copy()
        f_current = float(neg_log_post(x))
        f_entry = f_current

        trace = [] if self.display else None
        status = None

        for _ in range(n_steps):
            H_opt = self._gauss_newton_curvature(
                trial_func, x, prior_precision
            )
            g = self._central_gradient(neg_log_post, x)

            if trace is not None:
                trace.append((x.copy(), f_current))

            try:
                dx = np.linalg.solve(H_opt, g)
            except np.linalg.LinAlgError:
                status = ConvergenceStatus.SINGULAR_CURVATURE
                break

            if not np.all(np.isfinite(dx)) or float(g @ dx) <= 0.0:
                status = ConvergenceStatus.SINGULAR_CURVATURE
                break

            step = 1.0
            improved = False

            for _ in range(20):
                x_new = np.clip(
                    x - step * dx,
                    self.hard_bounds[0],
                    self.hard_bounds[1],
                )
                f_new = float(neg_log_post(x_new))

                if np.isfinite(f_new) and f_new < f_current:
                    improved = True
                    break

                step *= 0.5

            if not improved:
                status = ConvergenceStatus.NO_IMPROVEMENT
                break

            delta_f = abs(f_current - f_new)
            x = x_new
            f_current = f_new

            if delta_f / (1.0 + abs(f_current)) < tol_df:
                status = ConvergenceStatus.CONVERGED_DF
                break

        if status is None:
            status = ConvergenceStatus.MAX_STEPS

        if f_current > f_entry:
            raise RuntimeError(
                "GN polish violated the monotonicity invariant: "
                f"{f_entry} -> {f_current}."
            )

        if trace is not None:
            trace.append((x.copy(), f_current))
            self._temp_polish_trace = trace

        return x, f_current, status

    # -----------------------------------------------------------------
    # Full optimization
    # -----------------------------------------------------------------

    def optimize(
        self,
        neg_log_post: Callable[[np.ndarray], float],
        x_init: Optional[np.ndarray] = None,
        trial_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        prior_precision: Optional[np.ndarray] = None,
        neg_log_post_jax: Optional[Callable] = None,
    ) -> OptimizationResult:
        """Fit one MAP.

        Parameters
        ----------
        neg_log_post
            Full negative log posterior to minimize.
        x_init
            Optional initialization added to the multi-start set.
        trial_func
            Optional per-trial log-likelihood. When supplied, enables
            the GN polish. GN is never reused as the final Hessian.
        prior_precision
            Prior precision added to the GN optimization curvature.
        neg_log_post_jax
            JAX version of the full negative log posterior. Required
            only when Config.hessian_method='autodiff'.

        Returns
        -------
        OptimizationResult
            ``result.hess`` is the independent observed Hessian at the
            final MAP.
        """
        self.all_results = []

        # Defensive NumPy/scipy wrapper.
        raw_fun = neg_log_post

        def safe_fun(x):
            with np.errstate(
                over="ignore",
                invalid="ignore",
                divide="ignore",
                under="ignore",
            ):
                try:
                    f = float(raw_fun(x))
                except Exception:
                    return 1e20
            return f if np.isfinite(f) else 1e20

        init_points = []

        if x_init is not None:
            x_init = np.asarray(x_init, dtype=float)
            if x_init.shape != (self.d,):
                raise ValueError(
                    f"x_init must have shape ({self.d},), "
                    f"got {x_init.shape}"
                )
            init_points.append(x_init)

        if self.inits is not None:
            custom = np.asarray(self.inits, dtype=float)
            if custom.ndim == 1:
                if custom.shape != (self.d,):
                    raise ValueError(
                        f"inits must have shape ({self.d},) or (n, {self.d})"
                    )
                init_points.append(custom)
            elif custom.ndim == 2:
                if custom.shape[1] != self.d:
                    raise ValueError(
                        f"inits must have {self.d} columns"
                    )
                init_points.extend(custom)
            else:
                raise ValueError(
                    f"inits must have shape ({self.d},) or (n, {self.d})"
                )

        n_needed = max(0, self.num_init - len(init_points))
        if n_needed:
            random_inits = np.random.uniform(
                low=self.range_bounds[0],
                high=self.range_bounds[1],
                size=(n_needed, self.d),
            )
            init_points.extend(random_inits)

        if not init_points:
            raise RuntimeError("No optimization initializations available")

        best_result = None
        best_history_x = []
        best_history_f = []

        for start in init_points:
            result = self._single_optimization(safe_fun, start)
            self.all_results.append(result)

            if (
                np.isfinite(result.f)
                and (
                    best_result is None
                    or result.f < best_result.f
                )
            ):
                best_result = result
                best_history_x = list(self._temp_history_x)
                best_history_f = list(self._temp_history_f)

        if best_result is None:
            raise RuntimeError(
                "All L-BFGS-B initializations returned non-finite objectives"
            )

        self.history_x = best_history_x
        self.history_f = best_history_f

        # -------------------------------------------------------------
        # Optimization refinement: GN only.
        # -------------------------------------------------------------
        f_before_polish = best_result.f

        if trial_func is not None:
            best_result.x, best_result.f, status = self._newton_polish(
                safe_fun,
                best_result.x,
                trial_func=trial_func,
                prior_precision=prior_precision,
            )
        else:
            status = ConvergenceStatus.SKIPPED_NO_TRIAL_FUNC
            self._temp_polish_trace = None

        if best_result.f > f_before_polish:
            raise RuntimeError(
                "Optimization polish worsened the L-BFGS-B objective."
            )

        best_result.grad = self._central_gradient(
            safe_fun, best_result.x
        )
        best_result.abs_g = float(
            np.mean(np.abs(best_result.grad))
        )

        # -------------------------------------------------------------
        # Independent post-MAP observed Hessian.
        # -------------------------------------------------------------
        hess, hess_diag = self.compute_hessian(
            safe_fun,
            best_result.x,
            neg_log_post_jax=neg_log_post_jax,
            return_diagnostics=True,
        )

        is_hess_pos = hess_diag["is_positive_definite"]

        # -------------------------------------------------------------
        # Diagnostics.
        # -------------------------------------------------------------
        tol_df = 1e-4
        n_inits_agreeing = int(
            sum(
                abs(res.f - f_before_polish)
                / (1.0 + abs(f_before_polish))
                < tol_df
                for res in self.all_results
            )
        )

        at_hard_bounds = (
            best_result.x <= self.hard_bounds[0]
        ) | (
            best_result.x >= self.hard_bounds[1]
        )

        if np.any(at_hard_bounds):
            warnings.warn(
                "MAP lies on hard_bounds for parameter(s) "
                f"{np.where(at_hard_bounds)[0].tolist()}. "
                "The Laplace approximation assumes an interior MAP."
            )

        laplace_valid = bool(
            is_hess_pos and not np.any(at_hard_bounds)
        )

        if not is_hess_pos:
            warnings.warn(
                "Observed Hessian at the MAP is not positive definite "
                f"(minimum eigenvalue={hess_diag['raw_min_eig']:.3g}). "
                "The MAP is retained, but Laplace evidence should not "
                "be computed from this fit."
            )

        condition = hess_diag["condition_number"]
        if np.isfinite(condition) and condition > 1e12:
            warnings.warn(
                "Observed Hessian is extremely ill-conditioned "
                f"(condition number={condition:.3e}). "
                "Treat Laplace uncertainty/evidence cautiously."
            )

        # -------------------------------------------------------------
        # Flagging: do not replace a valid MAP with the prior merely
        # because Laplace curvature is problematic.
        # -------------------------------------------------------------
        if status is ConvergenceStatus.SINGULAR_CURVATURE:
            flag = 0.5
            warnings.warn(
                "GN polish stopped because the optimization curvature "
                "could not provide a valid Newton direction. "
                "The best L-BFGS-B/GN point is retained."
            )
        else:
            flag = 1.0

        if not laplace_valid:
            flag = min(flag, 0.5)

        # -------------------------------------------------------------
        # Optional traces. Evidence is not defined during GN polishing,
        # so polish_lme remains None by design.
        # -------------------------------------------------------------
        search_path = search_f = None
        polish_path = polish_f = None

        if self.display:
            if best_history_x:
                search_path = np.asarray(best_history_x, dtype=float)
                search_f = -np.asarray(best_history_f, dtype=float)

            trace = self._temp_polish_trace
            if trace:
                polish_path = np.asarray(
                    [item[0] for item in trace], dtype=float
                )
                polish_f = -np.asarray(
                    [item[1] for item in trace], dtype=float
                )

        return OptimizationResult(
            x=best_result.x,
            f=best_result.f,
            hess=hess,
            grad=best_result.grad,
            flag=flag,
            success=best_result.success,
            nit=best_result.nit,
            n_runs=len(self.all_results),
            is_hess_pos=is_hess_pos,
            abs_g=best_result.abs_g,
            x_init=best_result.x_init,
            hess_method=hess_diag["method"],
            convergence_status=status,
            hess_raw_min_eig=hess_diag["raw_min_eig"],
            hess_n_clipped=0,
            hess_condition_number=hess_diag["condition_number"],
            laplace_valid=laplace_valid,
            n_inits_agreeing=n_inits_agreeing,
            at_hard_bounds=at_hard_bounds,
            weak_identifiability=None,
            search_path=search_path,
            search_f=search_f,
            polish_path=polish_path,
            polish_f=polish_f,
            polish_lme=None,
        )

    def get_all_results(self) -> List[OptimizationResult]:
        """Return the pre-polish result from each L-BFGS-B start."""
        return self.all_results

    def get_history(self) -> tuple:
        """Return function-evaluation history of the winning L-BFGS-B run."""
        return self.history_x, self.history_f
