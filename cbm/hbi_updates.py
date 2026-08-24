"""Variational update equations for hierarchical Bayesian inference.

Only ``hbi_qhquad`` depends on the MAP optimizer. The other functions are the
existing HBI mathematics and are intentionally independent of the Hessian
backend.

Final MAP/Hessian policy inside HBI
-----------------------------------
- ``model_trials[k]`` enables GN polishing of the subject MAP.
- ``models_jax[k]`` optionally enables the AD observed Hessian.
- The final Hessian consumed by HBI is always the independent observed
  posterior Hessian returned by ``optimize_map``.
- HBI requires valid positive-definite Laplace curvature. A MAP-only fit is
  useful for individual analysis but cannot define the Gaussian q(h) required
  by HBI, so HBI raises an informative error rather than clipping or replacing
  the Hessian.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.special import gammaln, psi

from .hbi_types import (
    BoundQHZ,
    BoundQM,
    BoundQMutau,
    BoundState,
    DirichletDistribution,
    GaussianDistribution,
    GaussianGammaDistribution,
    IndividualPosterior,
)


def hbi_sumstats(
    r: np.ndarray,
    qh: IndividualPosterior,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Responsibility-weighted subject parameter moments."""
    theta = qh.parameters
    Ainvdiag = qh.hessian_inv_diag

    K, _ = r.shape
    Nbar = np.zeros(K, dtype=float)
    thetabar: List[np.ndarray] = [None] * K
    Sdiag: List[np.ndarray] = [None] * K

    for k in range(K):
        r_k = r[k, :]
        Nk = float(r_k.sum())

        if Nk <= 0:
            raise RuntimeError(
                f"Model {k + 1} has zero effective subjects in HBI."
            )

        Nbar[k] = Nk

        theta_k = theta[k]
        Ainvdiag_k = Ainvdiag[k]

        thetabar_k = (
            np.sum(
                theta_k * r_k[np.newaxis, :],
                axis=1,
                keepdims=True,
            )
            / Nk
        )

        Sdiag_k = (
            np.sum(
                (theta_k**2 + Ainvdiag_k)
                * r_k[np.newaxis, :],
                axis=1,
                keepdims=True,
            )
            / Nk
            - thetabar_k**2
        )

        thetabar[k] = thetabar_k
        Sdiag[k] = Sdiag_k

    return Nbar, thetabar, Sdiag


def hbi_qmutau(
    pmutau: List[GaussianGammaDistribution],
    Nbar: np.ndarray,
    thetabar: List[np.ndarray],
    Sdiag: List[np.ndarray],
) -> Tuple[List[GaussianGammaDistribution], BoundQMutau]:
    """Update group parameter means and precisions."""
    K = len(Nbar)

    ElogpH = np.full(K, np.nan)
    Elogpmu = np.full(K, np.nan)
    Elogqmu = np.full(K, np.nan)
    Elogptau = np.full(K, np.nan)
    Elogqtau = np.full(K, np.nan)

    qmutau_out: List[GaussianGammaDistribution] = []

    for k in range(K):
        a0k = np.asarray(pmutau[k].a, dtype=float)
        beta0k = float(pmutau[k].beta)
        nu0k = float(pmutau[k].nu)
        sigma0k = np.asarray(pmutau[k].sigma, dtype=float)

        Nk = float(Nbar[k])
        tb_k = thetabar[k]
        Sd_k = Sdiag[k]

        beta_k = beta0k + Nk
        a_k = (beta0k * a0k + Nk * tb_k) / beta_k
        nu_k = nu0k + 0.5 * Nk

        sigma_k = sigma0k + 0.5 * (
            Nk * Sd_k
            + Nk
            * beta0k
            / (Nk + beta0k)
            * (tb_k - a0k) ** 2
        )

        Elogtau_k = psi(nu_k) - np.log(sigma_k)
        Etau_k = nu_k / sigma_k

        logG_k = np.sum(
            -gammaln(nu_k)
            + nu_k * np.log(sigma_k)
        )
        logG0 = np.sum(
            -gammaln(nu0k)
            + nu0k * np.log(sigma0k)
        )

        Dk = len(a0k)
        ElogdetT = np.sum(Elogtau_k)

        diff = a_k - a0k
        quad_term = beta0k * np.sum(
            Etau_k * diff**2
        )

        Elogpmu[k] = (
            -Dk / 2 * np.log(2 * np.pi)
            + 0.5 * Dk * np.log(beta0k)
            + 0.5 * ElogdetT
            - 0.5 * quad_term
            - Dk / 2 * beta0k / beta_k
        )

        Elogptau[k] = (
            (nu0k - 1) * ElogdetT
            - np.sum(sigma0k * Etau_k)
            + logG0
        )

        Elogqmu[k] = (
            -Dk / 2 * np.log(2 * np.pi)
            + 0.5 * Dk * np.log(beta_k)
            + 0.5 * ElogdetT
            - Dk / 2
        )

        Elogqtau[k] = (
            (nu_k - 1) * ElogdetT
            - Dk * nu_k
            + logG_k
        )

        ElogpH[k] = (
            0.5 * Nk * ElogdetT
            - 0.5 * Nk * Dk * np.log(2 * np.pi)
            - 0.5 * Nk * Dk / beta_k
            - 0.5
            * np.sum(
                Etau_k
                * (
                    Nk * Sd_k
                    + Nk * (tb_k - a_k) ** 2
                )
            )
        )

        qmutau_out.append(
            GaussianGammaDistribution(
                a=a_k,
                beta=beta_k,
                sigma=sigma_k,
                nu=nu_k,
                Etau=Etau_k,
                Elogtau=Elogtau_k,
                logG=logG_k,
            )
        )

    bound = BoundQMutau(
        ElogpH=ElogpH,
        Elogpmu=Elogpmu,
        Elogptau=Elogptau,
        Elogqmu=Elogqmu,
        Elogqtau=Elogqtau,
    )

    return qmutau_out, bound


def hbi_qm(
    pm: DirichletDistribution,
    Nbar: np.ndarray,
) -> Tuple[DirichletDistribution, BoundQM]:
    """Update the group-level model-frequency posterior."""
    limInf = bool(pm.limInf)
    logC0 = float(pm.logC)
    alpha0 = np.asarray(pm.alpha, dtype=float)

    alpha = alpha0 + Nbar
    alpha_star = np.sum(alpha)

    if not np.isfinite(alpha_star):
        Elogm = np.full_like(alpha, np.nan)
        logC = np.nan
    else:
        Elogm = psi(alpha) - psi(alpha_star)
        logC = (
            gammaln(alpha_star)
            - np.sum(gammaln(alpha))
        )

    Elogpm = logC0 + np.sum(
        (alpha0 - 1) * Elogm
    )
    Elogqm = logC + np.sum(
        (alpha - 1) * Elogm
    )
    ElogpZ = Nbar * Elogm

    if limInf:
        K = len(alpha)
        alpha = np.full(K, np.inf)
        Elogm = np.log(np.ones(K) / K)
        logC = np.inf
        Elogpm = np.nan
        Elogqm = np.nan
        ElogpZ = Nbar * Elogm

    qm = DirichletDistribution(
        limInf=limInf,
        alpha=alpha,
        Elogm=Elogm,
        logC=logC,
    )

    bound = BoundQM(
        ElogpZ=ElogpZ,
        Elogpm=Elogpm,
        Elogqm=Elogqm,
    )

    return qm, bound


def hbi_qHZ(
    qmutau: List[GaussianGammaDistribution],
    qm: DirichletDistribution,
    qh: IndividualPosterior,
    thetabar: List[np.ndarray],
    Sdiag: List[np.ndarray],
) -> Tuple[np.ndarray, BoundQHZ]:
    """Update subject-by-model responsibilities."""
    qmlimInf = bool(qm.limInf)

    logf = np.asarray(qh.loglik, dtype=float)
    logdetA = np.asarray(
        qh.log_det_hessian,
        dtype=float,
    )

    if not np.all(np.isfinite(logf)):
        raise RuntimeError(
            "HBI received non-finite subject/model log-joint values."
        )

    if not np.all(np.isfinite(logdetA)):
        raise RuntimeError(
            "HBI received non-finite subject/model Hessian "
            "log-determinants. HBI requires valid Laplace curvature."
        )

    K, N = logf.shape
    r = np.zeros((K, N), dtype=float)

    ElogpH = np.full(K, np.nan)
    ElogpZ = np.full(K, np.nan)
    ElogpX = np.full(K, np.nan)
    ElogqH = np.full(K, np.nan)
    ElogqZ = np.full(K, np.nan)

    D = np.array(
        [len(qmutau[k].a) for k in range(K)],
        dtype=float,
    )

    ElogdetT = np.array(
        [
            np.sum(qmutau[k].Elogtau)
            for k in range(K)
        ],
        dtype=float,
    )

    logdetET = np.array(
        [
            np.sum(np.log(qmutau[k].Etau))
            for k in range(K)
        ],
        dtype=float,
    )

    beta = np.array(
        [qmutau[k].beta for k in range(K)],
        dtype=float,
    )

    lambda_vec = (
        0.5 * ElogdetT
        - 0.5 * logdetET
        - 0.5 * D / beta
    )

    shift = (
        0.5 * D * np.log(2 * np.pi)
        + lambda_vec
        + qm.Elogm
    )

    logrho = logf - 0.5 * logdetA
    logrho = logrho + shift[:, np.newaxis]

    if qmlimInf:
        r[:, :] = 1.0 / K
    else:
        # Stable softmax over models for each subject.
        centered = logrho - np.max(
            logrho,
            axis=0,
            keepdims=True,
        )
        exp_centered = np.exp(centered)
        r = exp_centered / np.sum(
            exp_centered,
            axis=0,
            keepdims=True,
        )

    tiny = np.finfo(float).tiny

    for k in range(K):
        Nk = float(r[k, :].sum())
        Dk = D[k]
        ElogdetT_k = ElogdetT[k]
        beta_k = beta[k]
        Etau_k = qmutau[k].Etau
        a_k = qmutau[k].a

        Sd_k = Sdiag[k].ravel()
        tb_k = thetabar[k].ravel()

        ElogpH[k] = (
            0.5 * Nk * ElogdetT_k
            - 0.5 * Nk * Dk * np.log(2 * np.pi)
            - 0.5 * Nk * Dk / beta_k
            - 0.5
            * Nk
            * np.sum(
                Etau_k
                * (
                    Sd_k
                    + (tb_k - a_k) ** 2
                )
            )
        )

        ElogpZ[k] = Nk * qm.Elogm[k]

        ElogpXH = np.sum(
            r[k, :]
            * (
                logf[k, :]
                - 0.5 * Dk
                + lambda_vec[k]
            )
        )
        ElogpX[k] = ElogpXH - ElogpH[k]

        r_k = r[k, :]
        ElogqH[k] = np.sum(
            r_k
            * (
                -Dk / 2
                - Dk / 2 * np.log(2 * np.pi)
                + 0.5 * logdetA[k, :]
            )
        )

        mask = r_k > tiny
        ElogqZ[k] = np.sum(
            r_k[mask] * np.log(r_k[mask])
        )

    bound = BoundQHZ(
        ElogpX=ElogpX,
        ElogpH=ElogpH,
        ElogpZ=ElogpZ,
        ElogqH=ElogqH,
        ElogqZ=ElogqZ,
    )

    return r, bound


def _validate_optional_model_list(
    values,
    models,
    name: str,
):
    if values is None:
        return

    if len(values) != len(models):
        raise ValueError(
            f"{name} must contain one entry per model: "
            f"got {len(values)} for {len(models)} models."
        )


def hbi_qhquad(
    models: List[Any],
    data: List[Any],
    pconfig: List[Any],
    qmutau: List[GaussianGammaDistribution],
    qh: IndividualPosterior,
    fid,
    model_trials: Optional[List[Optional[Any]]] = None,
    models_jax: Optional[List[Optional[Any]]] = None,
) -> IndividualPosterior:
    """Refit each subject under the current hierarchical prior.

    The previous subject posterior mean is used as the initialization. The
    optimizer may use GN curvature for the MAP polish, but HBI always consumes
    the independent observed Hessian returned at the final MAP.

    HBI cannot continue from a MAP-only fit because q(h) explicitly requires
    an invertible Gaussian covariance and Hessian log-determinant.
    """
    from .map_estimation import optimize_map

    N = len(data)
    K = len(models)

    _validate_optional_model_list(
        model_trials,
        models,
        "model_trials",
    )
    _validate_optional_model_list(
        models_jax,
        models,
        "models_jax",
    )

    # Subject refits are intentionally quiet; HBI owns the high-level log.
    if fid is not None:
        fid = None

    theta_list = []
    Ainvdiag_list = []

    logf = np.full((K, N), np.nan)
    logdetA = np.full((K, N), np.nan)
    diagnostics = np.empty((K, N), dtype=object)

    for k in range(K):
        a_k = np.asarray(qmutau[k].a)
        Etau_k = np.asarray(qmutau[k].Etau)
        Dk = len(a_k)

        prior = GaussianDistribution(
            mean=a_k,
            precision=np.diagflat(Etau_k),
        )

        cfg = deepcopy(pconfig[k])

        theta_k = np.full((Dk, N), np.nan)
        Ainvdiag_k = np.full((Dk, N), np.nan)

        mt_k = (
            None
            if model_trials is None
            else model_trials[k]
        )
        mj_k = (
            None
            if models_jax is None
            else models_jax[k]
        )

        if (
            getattr(
                cfg,
                "hessian_method",
                "central_fd",
            )
            == "autodiff"
            and mj_k is None
        ):
            raise ValueError(
                f"HBI model {k + 1} requests an autodiff Hessian "
                "but no JAX model was supplied in models_jax."
            )

        for n in range(N):
            # Previous HBI posterior mean provides a deterministic start.
            cfg.inits = qh.parameters[k][:, n]

            (
                logf_kn,
                theta_kn,
                A_kn,
                _,
                flag_kn,
                result_kn,
            ) = optimize_map(
                data=data[n],
                model=models[k],
                config=cfg,
                prior_mean=prior.mean.flatten(),
                prior_precision=prior.precision,
                method="LAP",
                model_trials=mt_k,
                model_jax=mj_k,
            )

            diag_fn = getattr(
                result_kn,
                "diagnostics",
                None,
            )
            diagnostics[k, n] = (
                diag_fn()
                if callable(diag_fn)
                else None
            )

            if flag_kn == 0 or not np.all(
                np.isfinite(theta_kn)
            ):
                raise RuntimeError(
                    f"HBI could not obtain a finite MAP for "
                    f"model {k + 1}, subject {n + 1}."
                )

            laplace_valid = bool(
                getattr(
                    result_kn,
                    "laplace_valid",
                    result_kn.is_hess_pos,
                )
            )

            if not laplace_valid:
                min_eig = getattr(
                    result_kn,
                    "hess_raw_min_eig",
                    np.nan,
                )
                raise RuntimeError(
                    "HBI found a finite MAP but no valid local "
                    f"Gaussian approximation for model {k + 1}, "
                    f"subject {n + 1} "
                    f"(minimum observed-Hessian eigenvalue="
                    f"{min_eig:.3g}). "
                    "The MAP may be retained for individual analysis, "
                    "but HBI requires valid Laplace curvature."
                )

            A_kn = np.asarray(A_kn, dtype=float)

            try:
                cholA = np.linalg.cholesky(A_kn)
                Ainv = np.linalg.inv(A_kn)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"Invalid Hessian reached HBI for model {k + 1}, "
                    f"subject {n + 1} despite laplace_valid=True."
                ) from exc

            logdetA_kn = float(
                2.0
                * np.sum(
                    np.log(np.diag(cholA))
                )
            )

            if not (
                np.isfinite(logf_kn)
                and np.isfinite(logdetA_kn)
                and np.all(np.isfinite(Ainv))
            ):
                raise RuntimeError(
                    f"Non-finite HBI Gaussian quantities for "
                    f"model {k + 1}, subject {n + 1}."
                )

            logf[k, n] = logf_kn
            theta_k[:, n] = theta_kn
            Ainvdiag_k[:, n] = np.diag(Ainv)
            logdetA[k, n] = logdetA_kn

        theta_list.append(theta_k)
        Ainvdiag_list.append(Ainvdiag_k)

    return IndividualPosterior(
        loglik=logf,
        parameters=theta_list,
        hessian_inv_diag=Ainvdiag_list,
        log_det_hessian=logdetA,
        diagnostics=diagnostics,
    )


def hbi_bound(
    bound: BoundState,
    lastmodule: str,
) -> Tuple[BoundState, float]:
    """Update the variational bound after one HBI factor update."""
    bb = bound.bound
    pmlimInf = bool(bb.pmlimInf)

    Elogpm_minus_Elogqm = bb.Elogpm - bb.Elogqm
    if pmlimInf:
        Elogpm_minus_Elogqm = 0.0

    L_pre = (
        bb.ElogpX
        + bb.ElogpH
        + bb.ElogpZ
        + bb.Elogpmu
        + bb.Elogptau
        - bb.ElogqH
        - bb.ElogqZ
        - bb.Elogqmu
        - bb.Elogqtau
        + Elogpm_minus_Elogqm
    )

    if lastmodule == "qHZ":
        q = bound.qHZ
        bb.ElogpX = float(np.sum(q.ElogpX))
        bb.ElogpH = float(np.sum(q.ElogpH))
        bb.ElogpZ = float(np.sum(q.ElogpZ))
        bb.ElogqH = float(np.sum(q.ElogqH))
        bb.ElogqZ = float(np.sum(q.ElogqZ))

    elif lastmodule == "qmutau":
        q = bound.qmutau
        bb.ElogpH = float(np.sum(q.ElogpH))
        bb.Elogpmu = float(np.sum(q.Elogpmu))
        bb.Elogptau = float(np.sum(q.Elogptau))
        bb.Elogqmu = float(np.sum(q.Elogqmu))
        bb.Elogqtau = float(np.sum(q.Elogqtau))

    elif lastmodule == "qm":
        q = bound.qm
        bb.ElogpZ = float(np.sum(q.ElogpZ))
        bb.Elogpm = float(q.Elogpm)
        bb.Elogqm = float(q.Elogqm)

    else:
        raise ValueError(
            f"Unknown HBI bound module: {lastmodule}"
        )

    Elogpm_minus_Elogqm = bb.Elogpm - bb.Elogqm
    if pmlimInf:
        Elogpm_minus_Elogqm = 0.0

    L = (
        bb.ElogpX
        + bb.ElogpH
        + bb.ElogpZ
        + bb.Elogpmu
        + bb.Elogptau
        - bb.ElogqH
        - bb.ElogqZ
        - bb.Elogqmu
        - bb.Elogqtau
        + Elogpm_minus_Elogqm
    )

    dL = float(L - L_pre)

    bb.lastmodule = lastmodule
    bb.L = float(L)
    bb.dL = dL

    return bound, dL
