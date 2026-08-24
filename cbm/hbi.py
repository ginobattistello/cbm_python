"""Hierarchical Bayesian inference for CBM.

This module is the public HBI orchestrator. The variational update equations
live in ``hbi_updates.py``; dataclasses live in ``hbi_types.py``.

Relationship to MAP fitting
---------------------------
HBI repeatedly refits each subject under the current hierarchical prior.

The final MAP architecture is respected throughout:
- multi-start L-BFGS-B
- automatic GN polish when a model returns per-trial likelihoods
- independent observed Hessian at the MAP
- central finite differences by default
- optional JAX autodiff Hessian through ``models_jax``

HBI requires valid local Gaussian curvature. A finite individual MAP with an
invalid observed Hessian can still be reported by ``individual_fit``, but it
cannot initialize or enter HBI because HBI explicitly uses Hessian covariance
and log-determinant terms.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import pickle

import numpy as np
from scipy.special import gammaln, psi

from .hbi_config import HBIConfig
from .parameter_space import ParameterSpace
from .hbi_exceedance import cbm_hbi_exceedance
from .hbi_logging import (
    hbi_log,
    log_header,
    log_iteration,
)
from .hbi_types import (
    BoundQM,
    BoundQHZ,
    BoundQMutau,
    BoundState,
    BoundTerms,
    DirichletDistribution,
    GaussianGammaDistribution,
    HBIInput,
    HBIMath,
    HBIOutput,
    HBIProfile,
    HBIResult,
    IndividualPosterior,
    ProgressChange,
    ProgressState,
)
from .hbi_updates import (
    hbi_bound,
    hbi_qHZ,
    hbi_qhquad,
    hbi_qm,
    hbi_qmutau,
    hbi_sumstats,
)


__all__ = [
    "hbi_main",
    "hbi_run",
    "hbi_init",
    "hbi_null",
    "HBIResult",
]


def _validate_optional_model_list(
    values,
    models,
    name: str,
) -> None:
    if values is None:
        return

    if len(values) != len(models):
        raise ValueError(
            f"{name} must contain one entry per model: "
            f"got {len(values)} for {len(models)} models. "
            "Use None for a model without that optional backend."
        )


def _hbi_prog(
    prog: List[ProgressState],
    L: float,
    alpha: np.ndarray,
    thetabar: List[np.ndarray],
    Sdiag: List[np.ndarray],
) -> Tuple[ProgressChange, List[ProgressState]]:
    """Track changes used by the HBI convergence criterion."""
    last = prog[-1]

    L_pre = float(last.bound)
    alpha_pre = np.asarray(last.model_freq)

    thetabar_vec = np.concatenate(
        [tb.ravel() for tb in thetabar]
    )
    Sdiag_vec = np.concatenate(
        [sd.ravel() for sd in Sdiag]
    )

    # Normalize parameter changes by their hierarchical variance.
    scale = np.sqrt(
        np.maximum(
            Sdiag_vec,
            np.finfo(float).eps,
        )
    )
    x = thetabar_vec / scale

    if np.isscalar(last.normalized_params) and np.isnan(
        last.normalized_params
    ):
        dx = np.inf
    else:
        x_pre = np.asarray(
            last.normalized_params,
            dtype=float,
        )
        dx = float(
            np.sqrt(
                np.mean((x - x_pre) ** 2)
            )
        )

    dL = float(L - L_pre)

    ibest = int(np.argmax(alpha))
    if np.isinf(alpha[ibest]):
        dalpha = np.nan
    else:
        dalpha = float(
            abs(alpha[ibest] - alpha_pre[ibest])
        )

    change = ProgressChange(
        change_bound=dL,
        change_model_freq=dalpha,
        change_parameters=dx,
    )

    prog.append(
        ProgressState(
            bound=float(L),
            model_freq=np.asarray(
                alpha,
                dtype=float,
            ).copy(),
            normalized_params=x,
        )
    )

    return change, prog


def _load_map_result(source):
    """Load one individual-fit result used to initialize HBI."""
    if isinstance(source, str):
        with open(source, "rb") as file:
            loaded = pickle.load(file)

        # Support older wrappers that stored {'cbm': result}.
        if isinstance(loaded, dict) and "cbm" in loaded:
            return loaded["cbm"]

        return loaded

    # Retain support for already-loaded result objects.
    return source


def _validate_hbi_map_files(cbm_maps) -> None:
    """Require valid Laplace quantities before HBI starts.

    HBI's q(h) approximation needs:
    - a finite individual log joint,
    - finite Hessian inverse diagonal,
    - finite Hessian log determinant.

    New individual-fit results additionally expose ``laplace_valid`` in their
    diagnostics. Older valid files remain usable because the finite numerical
    quantities provide the fallback check.
    """
    failures = []

    for k, cbm_map in enumerate(cbm_maps):
        try:
            logf = np.asarray(
                cbm_map.math.loglik,
                dtype=float,
            )
            lme = np.asarray(
                cbm_map.math.lme,
                dtype=float,
            )
            logdet = np.asarray(
                cbm_map.math.log_det_hessian,
                dtype=float,
            )
            invdiag = cbm_map.math.hessian_inv_diag
        except AttributeError as exc:
            raise ValueError(
                f"HBI initialization for model {k + 1} is not a "
                "compatible individual-fit result."
            ) from exc

        diagnostics = getattr(
            cbm_map.math,
            "diagnostics",
            None,
        )

        for n in range(len(logf)):
            diag = (
                diagnostics[n]
                if diagnostics is not None
                and n < len(diagnostics)
                else None
            )

            laplace_valid = (
                getattr(
                    diag,
                    "laplace_valid",
                    None,
                )
                if diag is not None
                else None
            )

            invdiag_n = np.asarray(
                invdiag[n],
                dtype=float,
            )

            invalid = (
                not np.isfinite(logf[n])
                or not np.isfinite(lme[n])
                or not np.isfinite(logdet[n])
                or not np.all(
                    np.isfinite(invdiag_n)
                )
                or laplace_valid is False
            )

            if invalid:
                failures.append((k, n))

    if failures:
        preview = ", ".join(
            f"model {k + 1}/subject {n + 1}"
            for k, n in failures[:10]
        )

        if len(failures) > 10:
            preview += (
                f", ... (+{len(failures) - 10} more)"
            )

        raise ValueError(
            "HBI requires a valid Laplace approximation for every "
            "subject-model initialization. Invalid entries were found "
            f"for {preview}. A MAP may still exist for those fits, "
            "but HBI cannot use a MAP without valid Gaussian curvature."
        )


def hbi_main(
    data: List[Any],
    models: List[Any],
    fcbm_maps: List[Any],
    fname: str = "",
    config: Optional[
        Union[HBIConfig, Dict[str, Any]]
    ] = None,
    optimconfigs: Optional[List[Any]] = None,
    models_jax: Optional[
        List[Optional[Any]]
    ] = None,
) -> HBIResult:
    """Run hierarchical Bayesian inference.

    Parameters
    ----------
    data
        List of subject-level data objects.
    models
        One NumPy summed log-likelihood function per model.
    fcbm_maps
        Individual-fit files/results, one per model. These provide the initial
        subject posterior approximations.
    fname
        Optional output pickle.
    config
        HBI-level configuration.
    optimconfigs
        Optional MAP optimizer configs, one per model. If omitted, HBI uses
        the config stored in each individual-fit result.
    models
        Each model may return either a scalar log-likelihood or a one-dimensional
        vector of per-trial log-likelihoods. Trialwise models automatically
        receive GN polishing during HBI refits.
    models_jax
        Optional per-model JAX summed likelihoods. A model uses this only when
        its optimizer config requests ``hessian_method='autodiff'``.
    """
    if len(models) != len(fcbm_maps):
        raise ValueError(
            "models and fcbm_maps must have the same length"
        )

    _validate_optional_model_list(
        models_jax,
        models,
        "models_jax",
    )

    if optimconfigs is not None and len(
        optimconfigs
    ) != len(models):
        raise ValueError(
            "optimconfigs must contain one config per model"
        )

    config = (
        HBIConfig()
        if config is None
        else config
    )

    user_input = {
        "models": models,
        "fcbm_maps": fcbm_maps,
        "fname": fname,
        "config": config,
        "optimconfigs": optimconfigs,
        "models_jax": models_jax,
    }

    # Hyperpriors retained from the original CBM HBI implementation.
    hyper = {
        "b": 1.0,
        "v": 0.5,
        "s": 0.01,
    }

    inits, priors, map_configs = hbi_init(
        fcbm_maps,
        hyper,
        limInf=0,
        initialize_r="all_r_1",
    )

    if optimconfigs is not None:
        map_configs = [
            deepcopy(cfg)
            for cfg in optimconfigs
        ]

    return hbi_run(
        data,
        user_input,
        inits,
        priors,
        map_configs,
    )


def hbi_run(
    data: List[Any],
    user_input: Dict[str, Any],
    inits: Dict[str, Any],
    priors: Dict[str, Any],
    opt_configs: List[Any],
) -> HBIResult:
    """Run the HBI variational coordinate-ascent loop."""
    models = user_input["models"]
    fcbm_maps = user_input["fcbm_maps"]
    fname = user_input.get("fname", "")
    models_jax = user_input.get(
        "models_jax"
    )
    parameter_spaces = inits["parameter_spaces"]

    K = len(models)
    N = len(data)

    config_in = user_input.get("config")
    config = (
        config_in
        if isinstance(config_in, HBIConfig)
        else HBIConfig(
            **(config_in or {})
        )
    )

    qhquad = inits["qh"]
    r = np.asarray(
        inits["r"],
        dtype=float,
    )
    bound = deepcopy(inits["bound"])

    hyper = priors["hyper"]
    pmutau = deepcopy(priors["pmutau"])
    pm = deepcopy(priors["pm"])

    isnull = bool(pm.limInf)

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------
    flog = config.flog

    if (
        (flog is None or flog == "")
        and fname
    ):
        directory, filename = os.path.split(
            fname
        )
        directory = directory or "."
        root, _ = os.path.splitext(filename)
        flog = os.path.join(
            directory,
            f"{root}.log",
        )

    log_file = None
    if (
        flog != -1
        and isinstance(flog, str)
        and flog
    ):
        log_file = open(
            flog,
            "w",
            encoding="utf-8",
        )

    fid = log_file

    verbose = bool(config.verbose)
    verbose_multiK = bool(
        verbose and K > 1 and not isnull
    )
    fid_multiK = (
        fid
        if K > 1 and not isnull
        else None
    )

    # --------------------------------------------------------------
    # Subject-refit optimizer configs
    # --------------------------------------------------------------
    optconfigs = []

    for k in range(K):
        cfg = deepcopy(opt_configs[k])

        # HBI uses the previous subject posterior as an explicit start;
        # random multi-starts are therefore unnecessary inside every
        # hierarchical iteration.
        cfg.num_init = 0

        # Kept for compatibility with older Config objects.
        if hasattr(cfg, "num_init_med"):
            cfg.num_init_med = 0
        if hasattr(cfg, "num_init_up"):
            cfg.num_init_up = 0

        cfg.verbose = False
        optconfigs.append(cfg)

    log_header(
        verbose,
        fid,
        K,
        N,
        fcbm_maps,
        isnull,
    )

    prog = [
        ProgressState(
            bound=bound.bound.L,
            model_freq=np.asarray(
                pm.alpha,
                dtype=float,
            ),
            normalized_params=np.nan,
        )
    ]

    terminate = False
    iteration = 0
    math_history = []

    try:
        while (
            not terminate
            and iteration < config.maxiter
        ):
            iteration += 1

            hbi_log(
                verbose,
                fid,
                f"Iteration {iteration:02d}\n",
            )

            # q(mu, tau)
            Nbar, thetabar, Sdiag = (
                hbi_sumstats(r, qhquad)
            )

            qmutau, bound_qmutau = (
                hbi_qmutau(
                    pmutau,
                    Nbar,
                    thetabar,
                    Sdiag,
                )
            )
            bound.qmutau = bound_qmutau
            bound, _ = hbi_bound(
                bound,
                "qmutau",
            )

            # q(m)
            qm, bound_qm = hbi_qm(
                pm,
                Nbar,
            )
            bound.qm = bound_qm
            bound, _ = hbi_bound(
                bound,
                "qm",
            )

            # q(h): subject refits under the current hierarchical prior.
            qhquad = hbi_qhquad(
                models,
                data,
                optconfigs,
                qmutau,
                qhquad,
                fid,
                models_jax=models_jax,
                parameter_spaces=parameter_spaces,
            )

            # q(H, Z): responsibilities.
            r, bound_qHZ = hbi_qHZ(
                qmutau,
                qm,
                qhquad,
                thetabar,
                Sdiag,
            )
            bound.qHZ = bound_qHZ
            bound, _ = hbi_bound(
                bound,
                "qHZ",
            )

            progress_change, prog = _hbi_prog(
                prog,
                bound.bound.L,
                qm.alpha,
                thetabar,
                Sdiag,
            )

            terminate = bool(
                np.isfinite(
                    progress_change.change_parameters
                )
                and progress_change.change_parameters
                < config.tolx
            )

            log_iteration(
                verbose,
                fid,
                verbose_multiK,
                fid_multiK,
                iteration,
                Nbar,
                N,
                progress_change,
                terminate,
                K,
            )

            math_history.append(
                {
                    "qhquad": deepcopy(qhquad),
                    "r": r.copy(),
                    "Nbar": Nbar.copy(),
                    "thetabar": [
                        x.copy()
                        for x in thetabar
                    ],
                    "Sdiag": [
                        x.copy()
                        for x in Sdiag
                    ],
                    "qm": deepcopy(qm),
                    "qmutau": deepcopy(qmutau),
                    "bound": deepcopy(bound),
                    "prog": deepcopy(prog),
                    "prog_change": deepcopy(
                        progress_change
                    ),
                }
            )

            if (
                config.save_prog
                and config.fname_prog
            ):
                with open(
                    config.fname_prog,
                    "wb",
                ) as file:
                    pickle.dump(
                        math_history,
                        file,
                    )

        # ----------------------------------------------------------
        # Final output
        # ----------------------------------------------------------
        he_list: List[np.ndarray] = [None] * K
        nk_vec = np.zeros(K, dtype=float)
        group_mean = [None] * K

        for k in range(K):
            nu = qmutau[k].nu
            beta = qmutau[k].beta
            sigma = np.asarray(qmutau[k].sigma)
            space = parameter_spaces[k]

            if space.d_free:
                s2 = 2.0 * sigma / beta
                nk = 2.0 * nu
                he_free = np.sqrt(s2 / nk)
            else:
                nk = 2.0 * nu
                he_free = np.empty(0, dtype=float)

            # User-facing HBI summaries stay in complete model coordinates.
            he_list[k] = space.expand_free_vector(
                he_free,
                fixed_value=0.0,
            )
            group_mean[k] = space.expand(
                np.asarray(qmutau[k].a, dtype=float).reshape(-1)
            )
            nk_vec[k] = nk

        exceedance = cbm_hbi_exceedance(
            qm.alpha,
            is_null=isnull,
        )

        theta_out = []
        for k in range(K):
            space = parameter_spaces[k]
            theta_free = qhquad.parameters[k]
            theta_full = np.vstack(
                [
                    space.expand(theta_free[:, n])
                    for n in range(N)
                ]
            )
            theta_out.append(theta_full)

        output = HBIOutput(
            parameters=theta_out,
            responsibility=r.T,
            group_mean=group_mean,
            group_hierarchical_errorbar=he_list,
            model_frequency=Nbar / N,
            exceedance_prob=exceedance.xp,
            protected_exceedance_prob=exceedance.pxp,
        )

        profile = HBIProfile(
            datetime=datetime.now().isoformat(),
            filename="cbm_hbi_hbi",
            config=config,
            optimconfigs=optconfigs,
            hyperparameters=hyper,
        )

        cbm_input = HBIInput(
            models=models,
            fcbm_maps=fcbm_maps,
            fname=fname,
            config=config,
            optimconfigs=optconfigs,
            models_jax=models_jax,
            parameter_spaces=parameter_spaces,
        )

        cbm_math = HBIMath(
            qhquad=qhquad,
            r=r,
            qmutau=qmutau,
            qm=qm,
            bound=bound,
            Nbar=Nbar,
            hyper=hyper,
            he_list=he_list,
            nk_vec=nk_vec,
            exceedance=exceedance,
        )

        result = HBIResult(
            method="hbi",
            input=cbm_input,
            profile=profile,
            math=cbm_math,
            output=output,
        )

        if fname:
            with open(fname, "wb") as file:
                pickle.dump(
                    result,
                    file,
                )

        return result

    finally:
        if log_file is not None:
            log_file.close()


def hbi_init(
    flap,
    hyper,
    limInf=0,
    initialize_r="all_r_1",
    families=None,
):
    """Initialize HBI from individual-fit results."""
    families = [] if families is None else families

    b = hyper["b"]
    v = hyper["v"]
    s = hyper["s"]

    K = len(flap)

    cbm_maps = [
        _load_map_result(source)
        for source in flap
    ]

    _validate_hbi_map_files(cbm_maps)

    # All candidate models must describe the same subjects.
    N = cbm_maps[0].output.parameters.shape[0]

    for k, cbm_map in enumerate(cbm_maps[1:], start=1):
        Nk = cbm_map.output.parameters.shape[0]
        if Nk != N:
            raise ValueError(
                f"Model {k + 1} has {Nk} subjects but model 1 has {N}."
            )

    # Optimization configs stored with the individual fits.
    opt_configs = [
        deepcopy(cbm_map.profile.config)
        for cbm_map in cbm_maps
    ]

    # Initial individual posterior approximation. HBI group distributions
    # contain only free parameters; fixed parameters are reconstructed only
    # for user-facing outputs and model evaluation.
    theta = []
    Ainvdiag = []
    logdetA = []
    logf = []
    a0 = []
    parameter_spaces = []

    for k, cbm_map in enumerate(cbm_maps):
        prior_mean_full = np.asarray(
            cbm_map.profile.prior_mean,
            dtype=float,
        ).reshape(-1)

        prior_variance = getattr(
            cbm_map.input,
            "prior_variance",
            None,
        )

        if prior_variance is not None:
            space = ParameterSpace.from_prior(
                prior_mean_full,
                prior_variance,
            )
        else:
            # Backward compatibility for older map files: all parameters
            # were free in the original CBM convention.
            prior_precision = np.asarray(
                cbm_map.profile.prior_precision,
                dtype=float,
            )
            space = ParameterSpace.all_free(
                prior_mean_full,
                prior_precision,
            )

        parameter_spaces.append(space)

        logf.append(
            np.asarray(cbm_map.math.loglik, dtype=float)
        )
        a0.append(space.free_mean.reshape(-1, 1))

        theta_full = np.column_stack(
            cbm_map.math.parameters
        )
        theta.append(theta_full[space.free_mask, :])

        invdiag_full = np.column_stack(
            cbm_map.math.hessian_inv_diag
        )
        Ainvdiag.append(invdiag_full[space.free_mask, :])

        logdetA.append(
            np.asarray(
                cbm_map.math.log_det_hessian,
                dtype=float,
            )
        )

    qh = IndividualPosterior(
        loglik=np.vstack(logf),
        parameters=theta,
        hessian_inv_diag=Ainvdiag,
        log_det_hessian=np.vstack(logdetA),
    )

    # --------------------------------------------------------------
    # Hyperpriors for group parameter means/precisions.
    # --------------------------------------------------------------
    pmutau: List[
        GaussianGammaDistribution
    ] = []

    for k in range(K):
        a_k = np.asarray(a0[k])
        sigma_k = (
            s * np.ones_like(a_k)
            if not isinstance(s, (list, tuple))
            else np.asarray(s[k])
        )

        if sigma_k.shape != a_k.shape:
            raise ValueError(
                f"Hyperparameter s for model {k + 1} has shape "
                f"{sigma_k.shape}, expected {a_k.shape}."
            )

        Elogtau = psi(v) - np.log(
            sigma_k
        )
        Etau = v / sigma_k
        logG = np.sum(
            -gammaln(v)
            + v * np.log(sigma_k)
        )

        pmutau.append(
            GaussianGammaDistribution(
                a=a_k,
                beta=b,
                sigma=sigma_k,
                nu=v,
                Etau=Etau,
                Elogtau=Elogtau,
                logG=float(logG),
            )
        )

    # --------------------------------------------------------------
    # Prior over model frequencies.
    # --------------------------------------------------------------
    alpha0 = np.ones(K)

    if len(families) > 0:
        families_arr = np.asarray(families)
        if len(families_arr) != K:
            raise ValueError(
                "families must have one entry per model"
            )

        alpha0[:] = np.nan
        for family in np.unique(
            families_arr
        ):
            mask = families_arr == family
            alpha0[mask] = 1.0 / mask.sum()

    pm = DirichletDistribution(
        limInf=bool(limInf),
        alpha=alpha0,
        Elogm=np.zeros_like(
            alpha0,
            dtype=float,
        ),
        logC=0.0,
    )

    alpha_star = np.sum(pm.alpha)
    pm.Elogm = (
        psi(pm.alpha)
        - psi(alpha_star)
    )
    pm.logC = float(
        gammaln(alpha_star)
        - np.sum(gammaln(pm.alpha))
    )

    if pm.limInf:
        pm.alpha = np.full_like(
            pm.alpha,
            np.inf,
            dtype=float,
        )
        pm.Elogm = np.full_like(
            pm.alpha,
            np.inf,
            dtype=float,
        )
        pm.logC = 0.0

    # --------------------------------------------------------------
    # Initial responsibilities.
    # --------------------------------------------------------------
    if initialize_r == "all_r_1":
        r = np.ones(
            (K, N),
            dtype=float,
        )
    else:
        raise NotImplementedError(
            f"initialize option '{initialize_r}' is not implemented"
        )

    # --------------------------------------------------------------
    # Empty bound state; modules fill these values during iteration.
    # --------------------------------------------------------------
    bound_terms = BoundTerms(
        ElogpX=np.nan,
        ElogpH=np.nan,
        ElogpZ=np.nan,
        Elogpmu=np.nan,
        Elogptau=np.nan,
        Elogpm=0.0,
        ElogqH=np.nan,
        ElogqZ=np.nan,
        Elogqmu=np.nan,
        Elogqtau=np.nan,
        Elogqm=0.0,
        pmlimInf=bool(limInf),
        lastmodule="",
        L=np.nan,
        dL=np.nan,
    )

    bound = BoundState(
        bound=bound_terms,
        qHZ=BoundQHZ(
            ElogpX=np.full(K, np.nan),
            ElogpH=np.full(K, np.nan),
            ElogpZ=np.full(K, np.nan),
            ElogqH=np.full(K, np.nan),
            ElogqZ=np.full(K, np.nan),
        ),
        qmutau=BoundQMutau(
            ElogpH=np.full(K, np.nan),
            Elogpmu=np.full(K, np.nan),
            Elogptau=np.full(K, np.nan),
            Elogqmu=np.full(K, np.nan),
            Elogqtau=np.full(K, np.nan),
        ),
        qm=BoundQM(
            ElogpZ=np.full(K, np.nan),
            Elogpm=np.nan,
            Elogqm=np.nan,
        ),
    )

    inits = {
        "qh": qh,
        "r": r,
        "bound": bound,
        "parameter_spaces": parameter_spaces,
    }

    priors = {
        "hyper": hyper,
        "pmutau": pmutau,
        "pm": pm,
    }

    return inits, priors, opt_configs


def hbi_null(
    data: List[Any],
    fname_cbm: Union[str, HBIResult],
) -> HBIResult:
    """Compute protected exceedance probabilities using the HBI null model.

    The null rerun preserves the same optional ``models_jax`` backend as
    the original HBI run. GN availability continues to be inferred directly
    from each model output.
    """
    input_is_file = isinstance(
        fname_cbm,
        str,
    )

    if input_is_file:
        with open(
            fname_cbm,
            "rb",
        ) as file:
            loaded = pickle.load(file)

        cbm = (
            loaded["cbm"]
            if isinstance(loaded, dict)
            and "cbm" in loaded
            else loaded
        )
        source_filename = fname_cbm

    elif isinstance(
        fname_cbm,
        HBIResult,
    ):
        cbm = fname_cbm
        source_filename = None

    else:
        raise TypeError(
            "fname_cbm must be a filename or HBIResult"
        )

    fname0 = ""

    if source_filename:
        directory, basename = os.path.split(
            source_filename
        )
        root, extension = os.path.splitext(
            basename
        )
        extension = extension or ".pkl"
        fname0 = os.path.join(
            directory,
            f"{root}_null{extension}",
        )

    config = deepcopy(cbm.input.config)

    if isinstance(config, HBIConfig):
        config_null = config
    else:
        config_null = HBIConfig(
            **config
        )

    # Avoid overwriting progress/log files from the non-null run.
    if isinstance(config_null.flog, str):
        directory, basename = os.path.split(
            config_null.flog
        )
        root, extension = os.path.splitext(
            basename
        )
        config_null.flog = os.path.join(
            directory or ".",
            f"{root}_null{extension}",
        )

    if isinstance(
        config_null.fname_prog,
        str,
    ):
        directory, basename = os.path.split(
            config_null.fname_prog
        )
        root, extension = os.path.splitext(
            basename
        )
        extension = extension or ".pkl"
        config_null.fname_prog = os.path.join(
            directory or ".",
            f"{root}_null{extension}",
        )

    hyper = cbm.profile.hyperparameters

    inits, priors, opt_configs = hbi_init(
        cbm.input.fcbm_maps,
        hyper,
        limInf=1,
        initialize_r=config_null.initialize,
    )

    user_input = {
        "models": cbm.input.models,
        "fcbm_maps": cbm.input.fcbm_maps,
        "fname": fname0,
        "config": config_null,
        "optimconfigs": cbm.input.optimconfigs,
        "models_jax": getattr(
            cbm.input,
            "models_jax",
            None,
        ),
    }

    cbm0 = hbi_run(
        data,
        user_input,
        inits,
        priors,
        opt_configs,
    )

    alpha = np.asarray(
        cbm.math.qm.alpha,
        dtype=float,
    )
    L = float(cbm.math.bound.bound.L)
    L0 = float(cbm0.math.bound.bound.L)

    exceedance = cbm_hbi_exceedance(
        alpha,
        L=L,
        L0=L0,
    )

    cbm.math.exceedance = exceedance
    cbm.output.protected_exceedance_prob = (
        exceedance.pxp
    )

    if source_filename:
        with open(
            source_filename,
            "wb",
        ) as file:
            pickle.dump(cbm, file)

    return cbm
