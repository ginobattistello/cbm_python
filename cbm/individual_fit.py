"""Individual-subject fitting with Laplace approximation.

Final architecture
------------------
For each subject:

    multi-start L-BFGS-B
        -> optional Gauss-Newton polish
        -> MAP
        -> independent observed Hessian
             central finite difference (default)
             or JAX autodiff (optional)
        -> Laplace evidence only when the observed Hessian is valid

Important
---------
A valid MAP is NOT discarded merely because the observed Hessian is
non-positive-definite. In that case:

    parameters         = MAP estimate
    hessian            = raw observed Hessian
    log_evidence       = NaN
    log_det_hessian    = NaN
    hessian_inv_diag   = NaN
    result diagnostics = laplace_valid=False

This preserves the optimization result while explicitly flagging that the
local Gaussian/Laplace approximation is not valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional, Union
import pickle
import time
import warnings

import numpy as np

from .map_estimation import optimize_map, log_posterior
from .optimization import Config, PostFitDiagnostics


# ---------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------

DEFAULT_PRIOR_VARIANCE = 6.25
DEFAULT_PRIOR_MEAN = 0.0


@dataclass
class Prior:
    """Gaussian prior specification."""

    mean: np.ndarray
    variance: np.ndarray | float
    precision: Optional[np.ndarray] = None

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        d = mean.size

        variance = np.asarray(self.variance, dtype=float)

        if variance.ndim == 0:
            covariance = float(variance) * np.eye(d)
        elif variance.ndim == 1:
            if variance.size != d:
                raise ValueError(
                    f"prior_variance has length {variance.size}; "
                    f"expected {d}."
                )
            covariance = np.diag(variance)
        elif variance.shape == (d, d):
            covariance = variance
        else:
            raise ValueError(
                "prior_variance must be a scalar, length-d vector, "
                f"or ({d}, {d}) covariance matrix."
            )

        covariance = 0.5 * (covariance + covariance.T)

        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Prior covariance must be positive definite."
            ) from exc

        self.mean = mean.reshape(-1, 1)
        self.precision = np.linalg.inv(covariance)


def _resolve_prior(
    prior_mean,
    prior_variance,
    config,
    model_name: str,
):
    """Fill missing prior fields while recording that defaults were used."""
    defaults_used = []

    if prior_variance is None:
        prior_variance = DEFAULT_PRIOR_VARIANCE
        defaults_used.append("prior_variance")

    if prior_mean is None:
        d = None

        if config is not None:
            if isinstance(config, dict):
                d = config.get("d")
            else:
                d = getattr(config, "d", None)

        if d is None:
            raise ValueError(
                "prior_mean was not supplied and parameter dimension "
                "could not be inferred. Pass prior_mean or config.d."
            )

        prior_mean = np.full(
            int(d), DEFAULT_PRIOR_MEAN, dtype=float
        )
        defaults_used.append("prior_mean")

    if defaults_used:
        warnings.warn(
            f"{model_name}: using default "
            f"{' and '.join(defaults_used)}. "
            "The prior is a modelling assumption and affects both "
            "MAP estimates and model evidence.",
            UserWarning,
            stacklevel=3,
        )

    return (
        np.asarray(prior_mean, dtype=float).reshape(-1),
        prior_variance,
        tuple(defaults_used),
    )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _preflight_checks(data, model, prior: Prior, config: Config):
    """Validate inputs once before subject-level fitting."""
    if len(data) == 0:
        raise ValueError("data is empty: nothing to fit")

    d = prior.mean.size

    if config.d != d:
        raise ValueError(
            f"config.d ({config.d}) != prior dimension ({d})"
        )

    if prior.precision.shape != (d, d):
        raise ValueError(
            f"prior precision must be ({d}, {d}), "
            f"got {prior.precision.shape}"
        )

    try:
        np.linalg.cholesky(prior.precision)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "prior precision must be positive definite"
        ) from exc

    theta0 = prior.mean.flatten()
    nonfinite = []

    for n, subject_data in enumerate(data):
        try:
            value = model(theta0, subject_data)
        except Exception as exc:
            raise ValueError(
                f"model raised at prior mean for subject {n + 1}: "
                f"{exc!r}"
            ) from exc

        if not np.isfinite(value):
            nonfinite.append(n + 1)

    if nonfinite:
        warnings.warn(
            "Model returns a non-finite log-likelihood at the prior "
            f"mean for subject(s) {nonfinite}. Random starts may still "
            "find a finite region, but these fits deserve scrutiny."
        )


# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------

@dataclass
class FitInput:
    model_name: str
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    fname: Optional[str]
    prior_variance: Optional[np.ndarray | float] = None
    prior_defaults: tuple = ()


@dataclass
class FitProfile:
    datetime: str
    filename: str
    telapsed: float
    config: Config
    prior_mean: np.ndarray
    prior_precision: np.ndarray


@dataclass
class FitMath:
    """Mathematical details from individual fitting.

    ``loglik`` is retained for backward compatibility but stores the
    log JOINT at the MAP:

        log p(y, theta_MAP | model)

    not the bare log-likelihood.

    ``lme`` is the Laplace log model evidence and is NaN when the
    observed Hessian is not suitable for the Laplace approximation.
    """

    loglik: np.ndarray
    parameters: List[np.ndarray]
    hessian: List[np.ndarray]
    lme: np.ndarray
    hessian_inv_diag: List[np.ndarray]
    log_det_hessian: np.ndarray
    flag: np.ndarray
    gradient: np.ndarray
    diagnostics: Optional[
        List[Optional[PostFitDiagnostics]]
    ] = None


@dataclass
class FitOutput:
    parameters: np.ndarray
    log_evidence: np.ndarray


@dataclass
class FitResult:
    """Result from individual fitting."""

    method: str
    input: FitInput
    profile: FitProfile
    math: FitMath
    output: FitOutput

    _display_data: Optional[dict] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def summary(self, max_subjects: int = 12) -> str:
        from .reporting import summary as _summary
        return _summary(self, max_subjects=max_subjects)

    def table(self, pandas: bool = True):
        from .reporting import table as _table
        return _table(self, pandas=pandas)

    def plot(
        self,
        subject: Optional[int] = None,
        backend: str = "auto",
        **kwargs,
    ):
        from .display import plot as _plot
        return _plot(
            self,
            subject=subject,
            backend=backend,
            **kwargs,
        )

    @property
    def se(self) -> np.ndarray:
        """Posterior standard errors in parameter space.

        Rows corresponding to invalid Laplace fits are NaN.
        """
        from .reporting import standard_errors
        return standard_errors(self)

    def __repr__(self) -> str:
        try:
            return self.summary()
        except Exception as exc:
            return (
                f"<FitResult {self.method!r} "
                f"(summary failed: "
                f"{type(exc).__name__}: {exc})>"
            )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def _resolve_config(
    config: Optional[Union[Config, dict]],
    d: int,
) -> Config:
    if config is None:
        return Config(d=d)

    if isinstance(config, dict):
        values = dict(config)
        values["d"] = d
        return Config(**values)

    if isinstance(config, Config):
        if getattr(config, "d", None) == d:
            return config

        values = config.__dict__.copy()
        values["d"] = d
        return Config(**values)

    raise TypeError("config must be Config, dict, or None")


# ---------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------

def individual_fit(
    data: List[Any],
    model: Callable[[np.ndarray, Any], float],
    prior_mean: Optional[np.ndarray] = None,
    prior_variance: Optional[np.ndarray | float] = None,
    fname: Optional[str] = None,
    config: Optional[Union[Config, dict]] = None,
    model_trials: Optional[
        Callable[[np.ndarray, Any], np.ndarray]
    ] = None,
    model_jax: Optional[Callable] = None,
    predict: Optional[
        Callable[[np.ndarray, Any], np.ndarray]
    ] = None,
    observed: Optional[
        Callable[[Any], np.ndarray]
    ] = None,
) -> FitResult:
    """Fit a computational model independently to multiple subjects.

    Parameters
    ----------
    data
        List containing one data object per subject.
    model
        NumPy model returning the SUMMED log-likelihood:

            model(theta, subject_data) -> scalar

    prior_mean, prior_variance
        Gaussian prior in parameter space. Missing values are filled
        with the documented toolbox defaults.
    fname
        Optional pickle output path.
    config
        ``Config`` or dictionary.

        Relevant optimizer options include:

            display
            verbose
            num_init
            hessian_method = "central_fd" | "autodiff"
            hessian_step

    model_trials
        Optional NumPy function returning per-trial log-likelihoods:

            model_trials(theta, subject_data) -> (T,)

        This is used ONLY for the Gauss-Newton MAP polish.
        It does not define the final Hessian.

    model_jax
        Optional JAX implementation of the same SUMMED
        log-likelihood as ``model``.

        Required only for:

            config.hessian_method == "autodiff"

        and used only for the final AD observed Hessian.

    predict, observed
        Optional display helpers. They do not enter the fit.

    Returns
    -------
    FitResult
        Individual MAP parameters, evidence, Hessians and diagnostics.

    Notes
    -----
    MAP validity and Laplace validity are separate.

    If optimization succeeds but the final observed Hessian is
    non-positive-definite, the MAP is kept but evidence-related
    quantities for that subject are NaN.
    """
    model_name = getattr(model, "__name__", "model")

    prior_mean, prior_variance, prior_defaults = (
        _resolve_prior(
            prior_mean,
            prior_variance,
            config,
            model_name,
        )
    )

    d = prior_mean.size
    prior = Prior(
        mean=prior_mean,
        variance=prior_variance,
    )
    config = _resolve_config(config, d)

    if (
        getattr(config, "hessian_method", "central_fd")
        == "autodiff"
        and model_jax is None
    ):
        raise ValueError(
            "config.hessian_method='autodiff' requires model_jax."
        )

    _preflight_checks(data, model, prior, config)

    if (
        getattr(config, "display", False)
        and (predict is None or observed is None)
    ):
        missing = [
            name
            for name, value in (
                ("predict", predict),
                ("observed", observed),
            )
            if value is None
        ]

        warnings.warn(
            f"display=True without {' and '.join(missing)}. "
            "Observed-vs-predicted display will use its fallback "
            "representation."
        )

    n_subjects = len(data)

    start_time = datetime.now()

    if config.verbose:
        print("=" * 70)
        print(
            f"{'individual_fit':<40}"
            f"{start_time.strftime('%Y-%m-%d %H:%M:%S'):>30}"
        )
        print("=" * 70)
        print(f"Number of subjects: {n_subjects}")
        print(f"Number of parameters: {d}")
        print(f"Number of initializations: {config.num_init}")
        print(
            "Observed Hessian: "
            f"{getattr(config, 'hessian_method', 'central_fd')}"
        )
        print("-" * 70)

    flags = np.full(n_subjects, np.nan)
    log_joint = np.full(n_subjects, np.nan)
    lme = np.full(n_subjects, np.nan)
    log_det_hessian = np.full(n_subjects, np.nan)

    parameters_list: List[np.ndarray] = []
    hessian_list: List[np.ndarray] = []
    hessian_inv_diag: List[np.ndarray] = []
    diagnostics_list: List[
        Optional[PostFitDiagnostics]
    ] = []

    gradients = np.full((d, n_subjects), np.nan)

    t_start = time.time()

    for n, subject_data in enumerate(data):
        if config.verbose:
            print(f"Subject: {n + 1:02d}")

        # ---------------------------------------------------------
        # Fit subject. When display=True, capture warnings for the
        # diagnostic panel but re-emit them immediately.
        # ---------------------------------------------------------
        if getattr(config, "display", False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                (
                    log_joint_n,
                    parameters_n,
                    hessian_n,
                    grad_n,
                    flag_n,
                    result_n,
                ) = optimize_map(
                    subject_data,
                    model,
                    config,
                    prior.mean.flatten(),
                    prior.precision,
                    method="LAP",
                    model_trials=model_trials,
                    model_jax=model_jax,
                )

            subject_warnings = [
                str(w.message) for w in caught
            ]

            for w in caught:
                warnings.warn_explicit(
                    w.message,
                    w.category,
                    w.filename,
                    w.lineno,
                )

        else:
            subject_warnings = None

            (
                log_joint_n,
                parameters_n,
                hessian_n,
                grad_n,
                flag_n,
                result_n,
            ) = optimize_map(
                subject_data,
                model,
                config,
                prior.mean.flatten(),
                prior.precision,
                method="LAP",
                model_trials=model_trials,
                model_jax=model_jax,
            )

        diag_n = result_n.diagnostics()

        if subject_warnings is not None:
            diag_n.warnings = subject_warnings

        # ---------------------------------------------------------
        # True optimization failure: preserve legacy prior fallback.
        #
        # This path is distinct from a valid MAP with invalid Laplace
        # curvature. A non-PD observed Hessian does NOT enter here.
        # ---------------------------------------------------------
        if flag_n == 0:
            if config.verbose:
                print(
                    f"No finite MAP found for subject {n + 1:02d}"
                )

            if not config.prior_for_failed:
                raise RuntimeError(
                    "Optimization failed: no finite MAP found for "
                    f"subject {n + 1:02d}"
                )

            if config.verbose:
                print(
                    "Using prior mean because prior_for_failed=True"
                )

            parameters_n = prior.mean.flatten()
            log_joint_n = log_posterior(
                parameters_n,
                model,
                subject_data,
                prior.mean.flatten(),
                prior.precision,
            )
            hessian_n = prior.precision.copy()
            grad_n = np.full(d, np.nan)

            # Legacy fallback has no optimizer diagnostics.
            diag_n = None

            # Preserve legacy behavior for a genuine optimization
            # failure: the substituted Gaussian prior is PD, so its
            # curvature quantities remain defined.
            laplace_valid_n = True

        else:
            # This is the new distinction.
            laplace_valid_n = bool(
                result_n.laplace_valid
            )

        # ---------------------------------------------------------
        # Store MAP-level outputs.
        # ---------------------------------------------------------
        flags[n] = flag_n
        parameters_list.append(
            np.asarray(parameters_n, dtype=float)
        )
        log_joint[n] = log_joint_n
        hessian_list.append(
            np.asarray(hessian_n, dtype=float)
        )
        gradients[:, n] = grad_n
        diagnostics_list.append(diag_n)

        # ---------------------------------------------------------
        # Laplace quantities.
        #
        # Never invert or take logdet of an observed Hessian that the
        # optimizer diagnosed as invalid.
        # ---------------------------------------------------------
        if laplace_valid_n:
            try:
                hessian_inv = np.linalg.inv(hessian_n)
                sign, log_det_hess = np.linalg.slogdet(
                    hessian_n
                )

                if sign <= 0:
                    raise np.linalg.LinAlgError(
                        "Hessian determinant is non-positive"
                    )

            except np.linalg.LinAlgError:
                # Defensive consistency check. If diagnostics said the
                # Hessian was valid but linear algebra disagrees, do
                # not manufacture evidence.
                warnings.warn(
                    f"Subject {n + 1:02d}: Hessian failed inversion/"
                    "log-determinant despite being marked valid. "
                    "Evidence is set to NaN."
                )

                hessian_inv_diag.append(
                    np.full(d, np.nan)
                )
                log_det_hessian[n] = np.nan
                lme[n] = np.nan

                if diag_n is not None:
                    diag_n.laplace_valid = False

            else:
                hessian_inv_diag.append(
                    np.diag(hessian_inv)
                )
                log_det_hessian[n] = log_det_hess

                lme[n] = (
                    log_joint_n
                    + 0.5 * d * np.log(2.0 * np.pi)
                    - 0.5 * log_det_hess
                )

        else:
            hessian_inv_diag.append(
                np.full(d, np.nan)
            )
            log_det_hessian[n] = np.nan
            lme[n] = np.nan

            if config.verbose:
                print(
                    "  MAP retained; Laplace evidence unavailable "
                    "(invalid observed Hessian/boundary MAP)."
                )

    t_elapsed = time.time() - t_start

    # -----------------------------------------------------------------
    # Assemble result
    # -----------------------------------------------------------------

    fit_input = FitInput(
        model_name=model_name,
        prior_mean=prior.mean,
        prior_precision=prior.precision,
        fname=fname,
        prior_variance=prior_variance,
        prior_defaults=prior_defaults,
    )

    profile = FitProfile(
        datetime=datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        filename="individual_fit",
        telapsed=t_elapsed,
        config=config,
        prior_mean=prior.mean,
        prior_precision=prior.precision,
    )

    math = FitMath(
        loglik=log_joint,
        parameters=parameters_list,
        hessian=hessian_list,
        lme=lme,
        hessian_inv_diag=hessian_inv_diag,
        log_det_hessian=log_det_hessian,
        flag=flags,
        gradient=gradients,
        diagnostics=diagnostics_list,
    )

    output = FitOutput(
        parameters=np.vstack(parameters_list),
        log_evidence=lme,
    )

    fit = FitResult(
        method="LAP individual",
        input=fit_input,
        profile=profile,
        math=math,
        output=output,
    )

    if getattr(config, "display", False):
        fit._display_data = {
            "data": data,
            "predict": predict,
            "observed": observed,
            "model_trials": model_trials,
        }

    if fname is not None:
        with open(fname, "wb") as file:
            pickle.dump(fit, file)

    if config.verbose:
        n_invalid = int(np.sum(~np.isfinite(lme)))
        print("-" * 70)
        print(f"Finished in {t_elapsed:.2f} s")
        if n_invalid:
            print(
                f"Laplace evidence unavailable for "
                f"{n_invalid}/{n_subjects} subject(s)."
            )
        print("done :]")

    return fit
