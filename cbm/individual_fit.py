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

    parameters         = full MAP vector (fixed values included)
    hessian            = free-parameter observed Hessian
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
from .parameter_space import ParameterSpace


# ---------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------

DEFAULT_PRIOR_VARIANCE = 6.25
DEFAULT_PRIOR_MEAN = 0.0


@dataclass
class Prior:
    """Gaussian prior plus free/fixed parameter mapping.

    A zero prior variance fixes that model parameter exactly at its prior mean.
    Only parameters with positive variance enter optimization and Laplace
    integration.
    """

    mean: np.ndarray
    variance: np.ndarray | float
    precision: Optional[np.ndarray] = None
    space: Optional[ParameterSpace] = None

    def __post_init__(self):
        self.space = ParameterSpace.from_prior(
            self.mean,
            self.variance,
        )

        self.mean = self.space.full_mean.reshape(-1, 1)

        # ``precision`` is kept in full model coordinates for result metadata.
        # Fixed rows/columns are zero; the true Gaussian precision used for
        # inference is ``space.free_precision``.
        self.precision = self.space.full_precision

    @property
    def free_mean(self) -> np.ndarray:
        return self.space.free_mean

    @property
    def free_precision(self) -> np.ndarray:
        return self.space.free_precision

    @property
    def free_mask(self) -> np.ndarray:
        return self.space.free_mask

    @property
    def fixed_mask(self) -> np.ndarray:
        return self.space.fixed_mask

    @property
    def d_free(self) -> int:
        return self.space.d_free

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
    """Validate standardized data and model output before fitting."""
    if len(data) == 0:
        raise ValueError("data is empty: nothing to fit")

    d_full = prior.mean.size

    if config.d != d_full:
        raise ValueError(
            f"config.d ({config.d}) != model/prior dimension ({d_full})"
        )

    theta0 = prior.mean.flatten()
    nonfinite = []

    for n, subject_data in enumerate(data):
        if not isinstance(subject_data, dict):
            raise TypeError(
                "Each subject must be a dictionary with keys 'y' and 'X'. "
                f"Subject {n + 1} has type {type(subject_data).__name__}."
            )

        missing = [key for key in ("y", "X") if key not in subject_data]
        if missing:
            raise ValueError(
                f"Subject {n + 1} is missing required data key(s): {missing}. "
                "Use data[n] = {'y': observed_outcomes, 'X': model_inputs}."
            )

        try:
            raw = model(theta0, subject_data)
            arr = np.asarray(raw, dtype=float)
        except Exception as exc:
            raise ValueError(
                f"model raised at prior mean for subject {n + 1}: {exc!r}"
            ) from exc

        if arr.ndim > 1:
            raise ValueError(
                f"model for subject {n + 1} returned shape {arr.shape}. "
                "It must return either a scalar or a one-dimensional "
                "per-trial log-likelihood vector."
            )

        if arr.size == 0 or not np.all(np.isfinite(arr)):
            nonfinite.append(n + 1)

        if arr.ndim == 1:
            y = np.asarray(subject_data["y"])
            if y.ndim == 0:
                raise ValueError(
                    f"Subject {n + 1}: trialwise model output requires "
                    "data['y'] to contain one outcome per observation."
                )
            if arr.size != y.shape[0]:
                raise ValueError(
                    f"Subject {n + 1}: model returned {arr.size} likelihood "
                    f"terms but data['y'] contains {y.shape[0]} outcomes."
                )

    if nonfinite:
        warnings.warn(
            "Model returns a non-finite likelihood at the prior mean for "
            f"subject(s) {nonfinite}. Random starts may still find a finite "
            "region, but these fits deserve scrutiny."
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
    free_mask: Optional[np.ndarray] = None
    fixed_mask: Optional[np.ndarray] = None
    fixed_values: Optional[np.ndarray] = None
    n_free_parameters: Optional[int] = None


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

    # One optional dictionary per subject. Values are deterministic model
    # trajectories evaluated ONCE at the final MAP. They are not optimization
    # traces and do not enter MAP estimation or Laplace evidence.
    latent: Optional[List[Optional[dict]]] = None


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
        display: bool = True,
        **kwargs,
    ):
        """Plot retained diagnostics.

        ``display`` is the only visibility switch. The same option is used by
        ``Config(display=True)`` during fitting. No separate ``show`` or
        ``block`` option is exposed.
        """
        from .display import plot as _plot
        return _plot(
            self,
            subject=subject,
            backend=backend,
            display=display,
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
# Optional latent-state tracking
# ---------------------------------------------------------------------

def _validate_evolution_output(
    latent,
    subject_data,
    *,
    subject_index: int,
) -> dict:
    """Validate an evolution() result without imposing a model-specific schema.

    ``evolution(theta, data)`` must return a dictionary. Numeric scalars and
    arrays are allowed. Trialwise arrays conventionally have first dimension T,
    but non-trialwise quantities are also retained and simply omitted from the
    automatic trajectory plot.
    """
    if not isinstance(latent, dict):
        raise TypeError(
            "evolution(theta, data) must return a dictionary; "
            f"subject {subject_index + 1} returned "
            f"{type(latent).__name__}."
        )

    clean = {}
    for name, value in latent.items():
        if not isinstance(name, str):
            raise TypeError(
                "evolution() dictionary keys must be strings; "
                f"got key {name!r}."
            )

        try:
            arr = np.asarray(value, dtype=float)
        except Exception as exc:
            raise TypeError(
                f"latent variable {name!r} for subject "
                f"{subject_index + 1} is not numeric."
            ) from exc

        if arr.size == 0:
            raise ValueError(
                f"latent variable {name!r} for subject "
                f"{subject_index + 1} is empty."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"latent variable {name!r} for subject "
                f"{subject_index + 1} contains non-finite values."
            )

        clean[name] = arr.copy()

    return clean


def _probe_evolution(evolution, theta0, data):
    """Validate the optional evolution function once before fitting."""
    if evolution is None:
        return ()

    latent = _validate_evolution_output(
        evolution(theta0, data[0]),
        data[0],
        subject_index=0,
    )
    return tuple(latent.keys())


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
    model: Callable[[np.ndarray, Any], Any],
    prior_mean: Optional[np.ndarray] = None,
    prior_variance: Optional[np.ndarray | float] = None,
    fname: Optional[str] = None,
    config: Optional[Union[Config, dict]] = None,
    model_jax: Optional[Callable] = None,
    observation: Optional[
        Callable[[np.ndarray, Any], np.ndarray]
    ] = None,
    evolution: Optional[
        Callable[[np.ndarray, Any], dict]
    ] = None,
) -> FitResult:
    """Fit a computational model independently to multiple subjects.

    Parameters
    ----------
    data
        List containing one data object per subject.
    model
        NumPy likelihood function. It may return either:

            model(theta, subject_data) -> scalar

        for a summed log-likelihood, or:

            model(theta, subject_data) -> (T,)

        for per-trial/per-observation log-likelihoods. A trialwise return
        enables the GN polish automatically; the toolbox sums it internally
        for L-BFGS-B.

    prior_mean, prior_variance
        Gaussian prior in model-parameter space. A zero variance fixes
        that parameter exactly at its prior mean; positive-variance
        parameters are estimated. Missing values use toolbox defaults.
    fname
        Optional pickle output path.
    config
        ``Config`` or dictionary.

        Relevant optimizer options include:

            display  # retain diagnostics and show the plot after fitting
            verbose
            num_init
            hessian_method = "central_fd" | "autodiff"
            hessian_step

    model_jax
        Optional JAX implementation following the same scalar-or-vector
        likelihood contract as ``model``. Required only when
        ``config.hessian_method == "autodiff"``.

    observation
        Optional observation function used only for display:

            observation(theta, subject_data) -> predictions

        Subject data must use the standardized structure:

            {"y": observed_outcomes, "X": model_inputs}

        ``display.py`` reads ``y`` directly and calls ``observation`` with
        the fitted full parameter vector and the complete subject data.

    evolution
        Optional deterministic latent-state function:

            evolution(theta, subject_data) -> dict[str, array-like]

        It is evaluated once per subject at the FINAL MAP after optimization.
        Returned trajectories are stored in ``fit.output.latent`` and may be
        displayed trial-by-trial. ``evolution`` never enters the optimization,
        observed-Hessian, or evidence calculations.

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
    d_free = prior.d_free

    if (
        getattr(config, "hessian_method", "central_fd")
        == "autodiff"
        and model_jax is None
    ):
        raise ValueError(
            "config.hessian_method='autodiff' requires model_jax."
        )

    _preflight_checks(data, model, prior, config)

    probe = np.asarray(
        model(prior.mean.flatten(), data[0]),
        dtype=float,
    )
    model_is_trialwise = probe.ndim == 1

    latent_names = _probe_evolution(
        evolution,
        prior.mean.flatten(),
        data,
    )

    if getattr(config, "display", False) and observation is None:
        warnings.warn(
            "display=True without observation=. Prediction diagnostics will "
            "be omitted; optimization, parameter, evidence, and status "
            "diagnostics will still be shown."
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
        print(f"Number of model parameters: {d}")
        print(f"Free parameters: {d_free}")
        print(f"Fixed parameters: {d - d_free}")
        print(f"Number of initializations: {config.num_init}")
        print(
            "Model likelihood: "
            + ("per-trial (GN available)" if model_is_trialwise
               else "scalar (L-BFGS-B only)")
        )
        print(
            "Observed Hessian: "
            f"{getattr(config, 'hessian_method', 'central_fd')}"
        )
        print(
            "Latent tracking: "
            + (", ".join(latent_names) if latent_names else "none")
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

    latent_list: List[Optional[dict]] = []

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
                    prior.free_mean,
                    prior.free_precision,
                    method="LAP",
                    model_jax=model_jax,
                    parameter_space=prior.space,
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
                prior.free_mean,
                prior.free_precision,
                method="LAP",
                model_jax=model_jax,
                parameter_space=prior.space,
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

            # Evaluate the fallback in free coordinates while the cognitive
            # model still receives the complete parameter vector.
            def _model_free(theta_free, subject_data_):
                return model(
                    prior.space.expand(theta_free),
                    subject_data_,
                )

            log_joint_n = log_posterior(
                prior.free_mean,
                _model_free,
                subject_data,
                prior.free_mean,
                prior.free_precision,
            )
            hessian_n = prior.free_precision.copy()
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
        # Deterministic latent trajectories at the FINAL MAP.
        #
        # This is deliberately post-fit. The evolution function is never
        # called by optimization.py or map_estimation.py, so adding latent
        # tracking cannot change theta_MAP or evidence.
        # ---------------------------------------------------------
        latent_n = None
        if evolution is not None:
            try:
                latent_n = _validate_evolution_output(
                    evolution(parameters_n, subject_data),
                    subject_data,
                    subject_index=n,
                )
            except Exception as exc:
                warnings.warn(
                    f"Subject {n + 1:02d}: latent tracking failed at the "
                    f"final MAP ({type(exc).__name__}: {exc}). "
                    "The fit is retained and latent output is set to None."
                )
                latent_n = None

        latent_list.append(latent_n)

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

                invalid_diag = np.zeros(d, dtype=float)
                invalid_diag[prior.free_mask] = np.nan
                hessian_inv_diag.append(invalid_diag)
                log_det_hessian[n] = np.nan
                lme[n] = np.nan

                if diag_n is not None:
                    diag_n.laplace_valid = False

            else:
                hessian_inv_diag.append(
                    prior.space.expand_free_vector(
                        np.diag(hessian_inv),
                        fixed_value=0.0,
                    )
                )
                log_det_hessian[n] = log_det_hess

                lme[n] = (
                    log_joint_n
                    + 0.5 * d_free * np.log(2.0 * np.pi)
                    - 0.5 * log_det_hess
                )

        else:
            invalid_diag = np.zeros(d, dtype=float)
            invalid_diag[prior.free_mask] = np.nan
            hessian_inv_diag.append(invalid_diag)
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
        free_mask=prior.free_mask.copy(),
        fixed_mask=prior.fixed_mask.copy(),
        fixed_values=prior.space.fixed_values.copy(),
        n_free_parameters=d_free,
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
        latent=latent_list,
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
            "observation": observation,
            "evolution": evolution,
            "latent_names": latent_names,
            "free_mask": prior.free_mask.copy(),
            "fixed_mask": prior.fixed_mask.copy(),
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

    # ``display`` is the single plotting switch: when enabled it both
    # retained the diagnostics above and now shows the appropriate figure.
    if getattr(config, "display", False):
        fit.plot(display=True)

    return fit
