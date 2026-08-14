"""
Individual subject fitting using Laplace approximation.

This module provides the individual_fit function for fitting computational models
to multiple subjects using Laplace approximation (MAP estimation).

"""

import numpy as np
import pickle
from typing import Callable, Optional, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import time

from .map_estimation import optimize_map, log_posterior
from .optimization import BFGSOptimizer, Config, PostFitDiagnostics


# ══════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS (DEV.md §4, layer 1 — added 2026-08-03)
# ──────────────────────────────────────────────────────────────────
# Run once, before any fitting. Defined failure behaviour per check:
#   raise  — the fit could not mean anything (dimension mismatch,
#            empty data, non-PD prior, model that crashes): STOP.
#   warn   — the fit may still succeed (objective non-finite at the
#            prior mean: random restarts can land in a finite region,
#            and the optimizer's defensive wrapper (Mod 4a) penalises
#            non-finite points): FLAG and continue.
# §4 principle: a check never silently changes a result.
#
# The bounds checks of this layer (shapes 2×d, range ⊂ hard) live in
# Config.__post_init__ (MODIFICATION 1, activated 2026-08-03) because
# that is where d and both bounds first meet; they run before this
# function is ever reached.
#
# Data content (NaN/Inf) is deliberately NOT inspected here: `data`
# is an opaque per-subject object (any type the model accepts), so
# the only universal probe is evaluating the model itself — which is
# exactly what the finiteness check below does.
# ══════════════════════════════════════════════════════════════════
def _preflight_checks(data, model, prior, config):
    """Validate inputs before fitting (DEV.md §4 pre-flight layer)."""
    d = len(prior.mean)

    # -- data non-empty -------------------------------------- raise
    if len(data) == 0:
        raise ValueError("data is empty: nothing to fit")

    # -- dimension agreement --------------------------------- raise
    if config.d != d:
        raise ValueError(
            f"config.d ({config.d}) != prior dimension ({d})")
    if prior.precision.shape != (d, d):
        raise ValueError(
            f"prior precision must be {d}×{d}, "
            f"got {prior.precision.shape}")

    # -- prior covariance positive-definite ------------------ raise
    # (covariance PD ⇔ precision PD; Prior stores the precision.
    #  np.linalg.inv succeeding in Prior.__post_init__ does NOT
    #  guarantee PD — a negative-definite matrix inverts fine.)
    try:
        np.linalg.cholesky(prior.precision)
    except np.linalg.LinAlgError:
        raise ValueError(
            "prior covariance is not positive-definite "
            "(Cholesky of the precision failed)")

    # -- objective finite at initialization ------------- raise/warn
    # Every subject is probed at the prior mean (the deterministic
    # x_init used by optimize_map). A model that RAISES is broken —
    # stop now rather than let Mod 4a silently turn every evaluation
    # into a 1e20 penalty and "fit" garbage. A model that returns
    # non-finite may recover from other start points — warn.
    theta0 = prior.mean.flatten()
    nonfinite = []
    for n, dat in enumerate(data):
        try:
            ll = model(theta0, dat)
        except Exception as e:
            raise ValueError(
                f"model raised at prior mean for subject {n + 1:02d}: "
                f"{e!r}") from e
        if not np.isfinite(ll):
            nonfinite.append(n + 1)
    if nonfinite:
        warnings.warn(
            f"Model returns non-finite log-likelihood at the prior "
            f"mean for subject(s) {nonfinite}; fitting continues "
            "(random initializations may still succeed) but these "
            "fits deserve scrutiny (DEV.md §4 pre-flight).")


@dataclass
class Prior:
    """
    Gaussian prior specification.

    Attributes:
        mean: Prior mean (d-dimensional array)
        variance: Prior variance (scalar, vector, or d×d matrix)
        precision: Prior precision matrix (inverse of covariance), computed automatically
    """
    mean: np.ndarray
    variance: np.ndarray
    precision: Optional[np.ndarray] = None

    def __post_init__(self):
        """Compute precision matrix from variance."""
        d = len(self.mean)

        # Convert variance to covariance matrix
        if np.isscalar(self.variance):
            cov = self.variance * np.eye(d)
        elif self.variance.ndim == 1:
            cov = np.diag(self.variance)
        else:
            cov = self.variance

        # Compute precision (inverse of covariance)
        self.precision = np.linalg.inv(cov)

        # Ensure mean is column vector
        if self.mean.ndim == 1:
            self.mean = self.mean.reshape(-1, 1)

@dataclass
class FitInput:
    """Profile and input information."""
    model_name: str
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    fname: Optional[str]
    # MODIFICATION 15 — the prior VARIANCE as supplied (the precision
    # above is its inverse, and a matrix, so the original scalar/vector
    # is not recoverable from it for display).
    prior_variance: Optional[np.ndarray] = None
    # Which of ("prior_mean", "prior_variance") the toolbox filled in.
    # Empty when the caller supplied both. Recorded on the RESULT, not
    # just warned about, so a saved fit still says how it was priored.
    prior_defaults: tuple = ()

@dataclass
class FitProfile:
    """Profile and input information."""
    datetime: str
    filename: str  # function name
    telapsed: float
    config: Config
    prior_mean: np.ndarray  
    prior_precision: np.ndarray    

@dataclass
class FitMath:
    """Mathematical details from fitting.

    Sign/naming note (optimization.py Mod 8): `loglik` stores the log
    JOINT log p(y,θ*|m) at each subject's MAP — likelihood PLUS prior —
    not the bare log-likelihood. The name is kept for backward
    compatibility with existing pickles/analysis code. `lme` is the
    Laplace evidence: lme = loglik + (d/2)·log(2π) − ½·log|H|.
    """
    loglik: np.ndarray
    parameters: List[np.ndarray]
    hessian: List[np.ndarray]
    lme: np.ndarray
    hessian_inv_diag: List[np.ndarray]
    log_det_hessian: np.ndarray
    flag: np.ndarray
    gradient: np.ndarray
    # Per-subject post-fit diagnostics (DEV.md §4 layer 3; Mod 9).
    # None entries mark subjects that fell back to the prior
    # (prior_for_failed) — no optimizer diagnostics exist for them.
    diagnostics: Optional[List[Optional[PostFitDiagnostics]]] = None

@dataclass
class FitOutput:
    """Main output from fitting."""
    parameters: np.ndarray  # N×d matrix
    log_evidence: np.ndarray  # N×1 vector


@dataclass
class FitResult:
    """
    Result from individual fitting.

    Attributes:
        method: Method name ('individual_fit')
        profile: Profile and input information
        math: Mathematical details
        output: Main output (parameters and log_evidence)

    Readable output (MODIFICATION 13, DEV.md §16):
        print(fit)      compact summary table
        fit.summary()   the same table as a string
        fit.table()     per-subject DataFrame (or list of dicts)
        fit.se          posterior standard errors, (n_subjects, d)
    """
    method: str
    input: FitInput
    profile: FitProfile
    math: FitMath
    output: FitOutput
    # MODIFICATION 14 — set by individual_fit only when display=True;
    # holds {data, predict, observed, model_trials} for plot(). Excluded
    # from repr/compare so it cannot clutter output or affect equality.
    _display_data: Optional[dict] = field(default=None, repr=False,
                                          compare=False)

    # ── MODIFICATION 13 — readable output ────────────────────────────
    # Everything below is presentation only. It reads fields that already
    # exist, is never called during fitting, and adds no dependency: the
    # table falls back to a list of dicts when pandas is absent. A bug in
    # this code cannot change a fit.

    def summary(self, max_subjects: int = 12) -> str:
        """Compact, copy-pasteable text summary. See cbm.reporting."""
        from .reporting import summary as _summary
        return _summary(self, max_subjects=max_subjects)

    def table(self, pandas: bool = True):
        """Per-subject table as a DataFrame (pandas=False for dicts).

        Works for a single-subject fit too — `parameters` stays
        two-dimensional, so you simply get a one-row table.
        """
        from .reporting import table as _table
        return _table(self, pandas=pandas)

    def plot(self, subject: Optional[int] = None, backend: str = "auto",
             **kwargs):
        """Diagnostic figure (MODIFICATION 14, DEV.md §17).

        Requires the fit to have been run with `config=dict(display=True)`;
        raises a ValueError explaining why if not, since the optimizer
        retains nothing otherwise.

        subject=None  group figure for a multi-subject fit, per-subject
                      figure when only one subject was fitted
        subject=i     force the per-subject figure

        backend="html" renders the same figure into a self-contained HTML
        page and opens it in a browser — independent of which matplotlib
        backend is active, so it works headless and over SSH. Default
        "auto" keeps the matplotlib behaviour.

        Extra keyword arguments go to the underlying plotter: `save=path`
        writes the figure, `show=True` opens a matplotlib window,
        `html_path=path` chooses where the HTML page is written.
        """
        from .display import plot as _plot
        return _plot(self, subject=subject, backend=backend, **kwargs)

    @property
    def se(self) -> np.ndarray:
        """Posterior standard errors in theta space, (n_subjects, d).

        sqrt(diag(H^-1)) under the Laplace approximation. Exposed as a
        property because it is the single most-requested quantity that
        the raw result does not name anywhere.
        """
        from .reporting import standard_errors
        return standard_errors(self)

    def __repr__(self) -> str:
        # The default dataclass repr dumps config and prior arrays before
        # reaching the estimates, which is why nobody printed these objects.
        try:
            return self.summary()
        except Exception as e:                       # never mask a result
            return (f"<FitResult {self.method!r} "
                    f"(summary failed: {type(e).__name__}: {e})>")


# ══════════════════════════════════════════════════════════════════
# MODIFICATION 15 — default prior (DEV.md §18)
# ──────────────────────────────────────────────────────────────────
# VBA supplies a default prior when none is given (VBA_defaultPriors.m:
# N(0, I) on parameters). This toolbox required both `prior_mean` and
# `prior_variance` on every call, so every example and harness hard-coded
# the same numbers with no stated justification.
#
# WHY NOT VBA'S VARIANCE OF 1. The two toolboxes put priors on different
# things. VBA's unit variance sits on parameters at roughly natural
# scale; this toolbox fits in UNCONSTRAINED theta-space, where models
# typically map alpha = sigmoid(theta) and beta = exp(theta). What N(0,v)
# implies in native space:
#
#     v      SD    alpha = sigmoid(theta)   beta = exp(theta)
#     1.00  1.00   [0.123, 0.877]           [0.14,   7.1]
#     6.25  2.50   [0.007, 0.993]           [0.007, 134]
#    10.00  3.16   [0.002, 0.998]           [0.002, 492]
#
# Variance 1 is a STRONG prior here — it excludes learning rates below
# 0.12, which are perfectly plausible. Copying VBA's number would import
# an assumption that does not transfer.
#
# THE VALUE CHOSEN: 6.25 (SD 2.5), from Piray et al. 2019 — the CBM paper
# this toolbox implements. Weakly informative rather than neutral: in
# unconstrained space no Gaussian is truly uninformative, so this is
# documented as an assumption, never claimed as an absence of one.
#
# NOT SILENT. The prior measurably moves results (on the benchmark RL
# cell, summed log-evidence spans 66 nats between v=1 and v=100), so a
# default is announced — in a warning, in the verbose header, on
# `FitInput`, and in `summary()`.
DEFAULT_PRIOR_VARIANCE = 6.25          # SD 2.5; Piray et al. 2019
DEFAULT_PRIOR_MEAN = 0.0               # per parameter, in theta-space


def _resolve_prior(prior_mean, prior_variance, config, model_name):
    """Fill in whichever of (mean, variance) was not supplied.

    Returns (prior_mean, prior_variance, defaults_used) where
    `defaults_used` is a tuple of the field names that were defaulted.

    HOW `d` IS OBTAINED WITHOUT A NEW ARGUMENT. Normally `d` comes from
    `len(prior_mean)`. When the mean is omitted it is read from
    `config.d`, which `Config` requires anyway — so a caller who omits
    the prior has already stated the dimension by passing a config.

    Probing the model (calling it with growing d until it stops raising)
    was tried and REJECTED: it silently returns the wrong answer for two
    ordinary patterns — a model summing over all parameters, or one that
    happens to index only the first. A wrong `d` would fit the wrong
    model without erroring, which is far worse than asking for a config.
    """
    used = []

    if prior_variance is None:
        prior_variance = DEFAULT_PRIOR_VARIANCE
        used.append("prior_variance")

    if prior_mean is None:
        d_cfg = None
        if config is not None:
            d_cfg = (config.get("d") if isinstance(config, dict)
                     else getattr(config, "d", None))
        if d_cfg is None:
            raise ValueError(
                "prior_mean was not given and the number of parameters "
                "could not be determined. Either pass prior_mean (e.g. "
                "np.zeros(3)), or state the dimension in the config: "
                "config=dict(d=3). The toolbox cannot infer it — `model` "
                "is an opaque callable, and probing it for a parameter "
                "count is unreliable (see MODIFICATION 15).")
        prior_mean = np.full(int(d_cfg), DEFAULT_PRIOR_MEAN, dtype=float)
        used.append("prior_mean")

    if used:
        d = len(np.ravel(prior_mean))
        v = prior_variance
        v_txt = (f"{float(v):g}" if np.isscalar(v)
                 else np.array2string(np.ravel(np.asarray(v)),
                                      precision=3))
        warnings.warn(
            f"{model_name}: using the DEFAULT prior for "
            f"{' and '.join(used)} — N(mean 0, variance {v_txt}) on each "
            f"of {d} parameter(s), in unconstrained theta-space "
            f"(SD {np.sqrt(float(np.mean(v))):.2f}; Piray et al. 2019). "
            f"This is a weakly informative assumption, not an absence of "
            f"one, and it does affect the estimates and the evidence. "
            f"Pass prior_mean/prior_variance explicitly to choose your "
            f"own (MODIFICATION 15, DEV.md §18).",
            UserWarning, stacklevel=3)
    return prior_mean, prior_variance, tuple(used)


def individual_fit(data: List[Any],
                   model: Callable[[np.ndarray, Any], float],
                   prior_mean: Optional[np.ndarray] = None,
                   prior_variance: Optional[np.ndarray | float] = None,
                   fname: Optional[str] = None,
                   config: Optional[Union[Config, dict]] = None,
                   model_trials: Optional[Callable[[np.ndarray, Any], np.ndarray]] = None,
                   predict: Optional[Callable[[np.ndarray, Any], np.ndarray]] = None,
                   observed: Optional[Callable[[Any], np.ndarray]] = None
                   ) -> FitResult:
    """
    Individual subject fitting using Laplace approximation.

    Args:
        data: List of data for N subjects (each element can be any type)
        model: Function that computes log-likelihood given parameters and data
               Signature: model(theta, data) -> log_likelihood
        prior: Prior object with mean and variance
        fname: Filename for saving output using pickle (None for no saving)
        config: Configuration object (optional)
        model_trials: Optional. Same signature as `model` but returns the
            per-trial log-likelihood array instead of the summed scalar
            (sum(model_trials(theta, data)) == model(theta, data)). When
            given, the Hessian/evidence at each subject's MAP uses the
            VBA-style Gauss-Newton curvature (see optimization.py Mod 5)
            instead of the finite-difference/eigenvalue-clip fallback.
        predict: MODIFICATION 14, display only. `predict(theta, data) ->
            y_hat`, the model's predicted outcome for one subject. Used
            solely by `FitResult.plot()` to draw observed-vs-predicted;
            it never enters the fit.
        observed: MODIFICATION 14, display only. `observed(data) -> y`,
            pulling the measured outcome out of one subject's data (e.g.
            `lambda d: d[1]`). Needed alongside `predict`, because `data`
            is opaque to the toolbox — it may be (X, y), (choices,
            rewards), or anything else.

            With `config=dict(display=True)` but WITHOUT these two, the
            plot falls back to per-trial log-likelihood and warns once,
            naming both arguments. Choice models generally cannot supply
            them meaningfully, so the fallback is a first-class path, not
            a degraded one.

    Returns:
        Tuple of (cbm, success) where:
            - cbm: CBM dataclass with all results
    """
    # Setup
    N = len(data)  # Number of subjects

    # MODIFICATION 15 — fill in a default prior for whichever of
    # (mean, variance) is missing. Runs before `d` is read, since the
    # mean is what defines it.
    prior_mean, prior_variance, prior_defaults = _resolve_prior(
        prior_mean, prior_variance, config,
        getattr(model, "__name__", "model"))

    d = len(np.ravel(prior_mean))  # Number of parameters

    prior = Prior(
        mean=np.asarray(prior_mean, dtype=float).reshape(-1),
        variance=prior_variance  # prior variances
    )

    # Configuration handling: allow Config or dict
    if config is None:
        config = Config(d=d)
    else:
        if isinstance(config, dict):
            # Ensure dimension d is present/consistent
            cfg_kwargs = dict(config)
            cfg_kwargs["d"] = d
            config = Config(**cfg_kwargs)
        elif isinstance(config, Config):
            # If provided Config has different d, overwrite to current d
            # to keep consistency with provided prior dimensions
            if getattr(config, "d", None) != d:
                # Reconstruct config preserving fields while updating d
                cfg_kwargs = config.__dict__.copy()
                cfg_kwargs["d"] = d
                config = Config(**cfg_kwargs)
        else:
            raise TypeError("config must be a Config or dict or None")

    # Initial report
    start_time = datetime.now()
    if config.verbose:
        print("=" * 70)
        print(f"{'individual_fit':<40}{start_time.strftime('%Y-%m-%d %H:%M:%S'):>30}")
        print("=" * 70)
        print(f"Number of samples: {N}")
        print(f"Number of parameters: {d}")
        # Mod 15 — the prior is a modelling choice that moves the result,
        # so state it in the header whether or not it was defaulted.
        _pv = prior_variance
        _pv_txt = (f"{float(_pv):g}" if np.isscalar(_pv)
                   else np.array2string(np.ravel(np.asarray(_pv)),
                                        precision=3))
        _tag = ("  [DEFAULT: " + ", ".join(prior_defaults) + "]"
                if prior_defaults else "")
        print(f"Prior: N(mean {np.array2string(np.ravel(prior.mean), precision=3)}, "
              f"variance {_pv_txt}){_tag}\n")
        print(f"Number of initializations: {config.num_init}")
        print("-" * 70)

    # Pre-flight checks (DEV.md §4 layer 1) — replaces the old
    # subject-0-only probe: dims, PD prior, and per-subject model
    # evaluation at the prior mean. Raises on unfittable inputs.
    _preflight_checks(data, model, prior, config)

    # ── MODIFICATION 14 — display fallback notice ────────────────────
    # Warn at FIT time, not at plot time: the user should learn what the
    # figure will be missing before waiting for the fit, not after. Once
    # per call, never per subject.
    if getattr(config, "display", False) and (predict is None
                                              or observed is None):
        missing = [n for n, v in (("predict", predict),
                                  ("observed", observed)) if v is None]
        warnings.warn(
            f"display=True without {' and '.join(missing)}: the "
            f"observed-vs-predicted panel will fall back to per-trial "
            f"log-likelihood. Pass predict(theta, data) -> y_hat and "
            f"observed(data) -> y to get the residual plot "
            f"(Mod 14, DEV.md §17).",
            UserWarning, stacklevel=2)

    # Initialize storage
    flags = np.full(N, np.nan)
    loglik = np.full(N, np.nan)
    parameters_list = []
    hessian_list = []
    G = np.full((d, N), np.nan)
    lme = np.full(N, np.nan)  # log-model-evidence
    hessian_inv_diag = []
    log_det_hessian = np.full(N, np.nan)
    diagnostics_list = []


    # Main loop over subjects
    t_start = time.time()

    for n in range(N):
        if config.verbose:
            print(f"Subject: {n + 1:02d}")

        dat = data[n]

        # Create optimizer for this subject

        # Call optimize_map for this subject.
        #
        # MOD 14 — when display is on, record this subject's warnings so
        # the status panel can show them. They are RE-EMITTED immediately
        # afterwards, so capturing never hides a warning from the user or
        # from their own warning filters; the recording is a copy, not an
        # interception. When display is off there is no wrapper at all.
        if getattr(config, "display", False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                (loglik_n, parameters_n, hessian_n, grad_n, flag_n,
                 result_n) = optimize_map(
                    dat, model, config, prior.mean.flatten(),
                    prior.precision, method='LAP', model_trials=model_trials)
            subject_warnings = [str(w.message) for w in caught]
            for w in caught:
                warnings.warn_explicit(w.message, w.category, w.filename,
                                       w.lineno)
        else:
            subject_warnings = None
            (loglik_n, parameters_n, hessian_n, grad_n, flag_n,
             result_n) = optimize_map(
                dat, model, config, prior.mean.flatten(), prior.precision,
                method='LAP', model_trials=model_trials)
        diag_n = result_n.diagnostics()
        if subject_warnings is not None:
            diag_n.warnings = subject_warnings

        # Handle failed optimization
        if flag_n == 0:
            if config.verbose:
                print(f"No minimum found for subject {n + 1:02d}")

            if config.prior_for_failed:
                if config.verbose:
                    print("No minimum found, use prior values as individual parameters")
                parameters_n = prior.mean.flatten()
                loglik_n = log_posterior(parameters_n, model, dat, prior.mean.flatten(), prior.precision)
                hessian_n = prior.precision.copy()
                grad_n = np.full(d, np.nan)
                diag_n = None  # prior substitution: no fit diagnostics
            else:
                print(f"No minimum found for subject {n + 1:02d}")
                raise RuntimeError(f"Optimization failed: No minimum found for subject {n+1:02d}")

        # Store results
        flags[n] = flag_n
        parameters_list.append(parameters_n)
        loglik[n] = loglik_n
        hessian_list.append(hessian_n)
        diagnostics_list.append(diag_n)
        hessian_inv_diag.append(np.diag(np.linalg.inv(hessian_n)))
        G[:, n] = grad_n

        log_det_hess = np.linalg.slogdet(hessian_n)[1]
        log_det_hessian[n] = log_det_hess

        # Compute log model evidence (Laplace approximation)
        lme[n] = loglik_n + 0.5 * d * np.log(2 * np.pi) - 0.5 * log_det_hess

    t_elapsed = time.time() - t_start

    # Prepare output using dataclasses (no data stored)
    fit_input = FitInput(
        model_name=model.__name__ if hasattr(model, '__name__') else str(model),
        prior_mean=prior.mean,
        prior_precision=prior.precision,        
        fname=fname,
        prior_variance=prior_variance,          # Mod 15
        prior_defaults=prior_defaults
    )

    profile_info = FitProfile(
        datetime=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        filename='individual_fit',
        telapsed=t_elapsed,
        config=config,
        prior_mean=prior.mean,
        prior_precision=prior.precision    
    )

    math_details = FitMath(
        loglik=loglik,
        parameters=parameters_list,
        hessian=hessian_list,
        lme=lme,
        hessian_inv_diag=hessian_inv_diag,
        log_det_hessian=log_det_hessian,
        flag=flags,
        gradient=G,
        diagnostics=diagnostics_list
    )

    # Stack parameters into N×d matrix
    parameters_array = np.vstack(parameters_list)

    output = FitOutput(
        parameters=parameters_array,
        log_evidence=lme
    )

    cbm = FitResult(
        method='LAP individual',
        input=fit_input,
        profile=profile_info,
        math=math_details,
        output=output
    )

    # ── MODIFICATION 14 — display payload ────────────────────────────
    # What plot() needs that is not already on the result: the data
    # itself, and the two optional accessors. Held on a private attribute
    # rather than a dataclass field so it never enters `asdict()`, never
    # gets pickled by accident, and cannot be mistaken for a fit output.
    #
    # NOTE this keeps a reference to `data`. That is deliberate — the
    # alternative is copying every subject's dataset — but it means a
    # display=True result holds the data alive. Off by default, so the
    # usual path is unaffected.
    if getattr(config, "display", False):
        cbm._display_data = dict(data=data, predict=predict,
                                 observed=observed,
                                 model_trials=model_trials)

    # Save if filename provided
    if fname is not None:
        with open(fname, 'wb') as f:
            pickle.dump(cbm, f)

    if config.verbose:
        print("done :]")

    return cbm


# Example usage
if __name__ == "__main__":
    # Example: Simple linear model
    def linear_model(theta, data):
        """
        Simple linear model: y ~ N(X*theta, sigma^2)
        theta = [slope, intercept, log(sigma)]
        """
        X, y = data
        slope, intercept, log_sigma = theta

        y_pred = X * slope + intercept
        sigma = np.exp(log_sigma)

        # Log-likelihood of Gaussian
        log_lik = -0.5 * np.sum((y - y_pred) ** 2 / sigma ** 2) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))

        return log_lik


    # Generate synthetic data for 5 subjects
    np.random.seed(42)
    N_subjects = 5
    data = []
    true_parameters = []

    X = np.linspace(0, 10, 50)
    noise = np.random.randn(len(X))
    print(X)

    for i in range(N_subjects):

        true_slope = 2.0 + np.random.randn() * 0.5
        true_intercept = 1.0 + np.random.randn() * 0.5
        true_log_sigma = np.log(0.5)

        y = true_slope * X + true_intercept + noise * np.exp(true_log_sigma)
        data.append((X, y))
        true_parameters.append([true_slope, true_intercept, true_log_sigma])

    true_parameters = np.array(true_parameters)

    # Define prior
    prior = Prior(
        mean=np.array([0, 0, 0]),  # slope, intercept, log(sigma)
        variance=np.array([10, 10, 10])  # prior variances
    )

    # Configure
    config = Config(
        d=3,
        num_init=20,
        tol_grad=1e-4,
        verbose=True
    )

    # Run individual fitting
    print("\n" + "=" * 70)
    print("Running Individual Fit Example")
    print("=" * 70 + "\n")

    cbm = individual_fit(data, linear_model, prior, fname=None, config=config)

    llp = log_posterior(cbm.output.parameters[0, :], linear_model, data[0], prior.mean, prior.precision)

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"\nTrue parameters (N×d):")
    print(true_parameters)
    print(f"\nFitted parameters (N×d):")
    print(cbm.output.parameters)
    print(f"\nLog model evidence:\n{cbm.output.log_evidence}")
    print(f"\nLog model loglik:\n{cbm.math.loglik}")
    print(f"\nLog det hessian:\n{cbm.math.log_det_hessian}")

    print(f"\ndiff:\n{cbm.output.log_evidence - cbm.math.loglik}")

