from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.special import psi, gammaln
from copy import deepcopy

from .hbi_types import (
    IndividualPosterior,
    GaussianGammaDistribution,
    DirichletDistribution,
    BoundQMutau,
    BoundQM,
    BoundQHZ,
    GaussianDistribution,
    BoundState,
    BoundTerms,
)

# hbi_sumstats

def hbi_sumstats(r: np.ndarray, qh: IndividualPosterior) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    theta = qh.parameters
    Ainvdiag = qh.hessian_inv_diag

    K, _ = r.shape
    thetabar: List[np.ndarray] = [None] * K
    Sdiag: List[np.ndarray] = [None] * K
    Nbar = np.zeros(K, dtype=float)

    for k in range(K):
        r_k = r[k, :]
        Nk = float(r_k.sum())
        Nbar[k] = Nk
        theta_k = theta[k]
        Ainvdiag_k = Ainvdiag[k]
        thetabar_k = np.sum(theta_k * r_k[np.newaxis, :], axis=1, keepdims=True) / Nk
        Sdiag_k = (
            np.sum((theta_k ** 2 + Ainvdiag_k) * r_k[np.newaxis, :], axis=1, keepdims=True) / Nk
            - thetabar_k ** 2
        )
        thetabar[k] = thetabar_k
        Sdiag[k] = Sdiag_k
    return Nbar, thetabar, Sdiag

# hbi_qmutau

def hbi_qmutau(
    pmutau: List[GaussianGammaDistribution],
    Nbar: np.ndarray,
    thetabar: List[np.ndarray],
    Sdiag: List[np.ndarray],
) -> Tuple[List[GaussianGammaDistribution], BoundQMutau]:
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
            Nk * Sd_k + Nk * beta0k / (Nk + beta0k) * (tb_k - a0k) ** 2
        )
        Elogtau_k = psi(nu_k) - np.log(sigma_k)
        Etau_k = nu_k / sigma_k
        logG_k = np.sum(-gammaln(nu_k) + nu_k * np.log(sigma_k))
        logG0 = np.sum(-gammaln(nu0k) + nu0k * np.log(sigma0k))
        Dk = len(a0k)
        ElogdetT = np.sum(Elogtau_k)
        diff = a_k - a0k
        quad_term = beta0k * np.sum(Etau_k * diff ** 2)
        Elogpmu[k] = (
            -Dk / 2 * np.log(2 * np.pi)
            + 0.5 * Dk * np.log(beta0k)
            + 0.5 * ElogdetT
            - 0.5 * quad_term
            - Dk / 2 * beta0k / beta_k
        )
        Elogptau[k] = (nu0k - 1) * ElogdetT - np.sum(sigma0k * Etau_k) + logG0
        Elogqmu[k] = (
            -Dk / 2 * np.log(2 * np.pi)
            + 0.5 * Dk * np.log(beta_k)
            + 0.5 * ElogdetT
            - Dk / 2
        )
        Elogqtau[k] = (nu_k - 1) * ElogdetT - Dk * nu_k + logG_k
        ElogpH[k] = (
            0.5 * Nk * ElogdetT
            - 0.5 * Nk * Dk * np.log(2 * np.pi)
            - 0.5 * Nk * Dk / beta_k
            - 0.5 * np.sum(Etau_k * (Nk * Sd_k + Nk * (tb_k - a_k) ** 2))
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

# hbi_qm

def hbi_qm(pm: DirichletDistribution, Nbar: np.ndarray) -> Tuple[DirichletDistribution, BoundQM]:
    limInf = bool(pm.limInf)
    logC0 = float(pm.logC)
    alpha0 = np.asarray(pm.alpha, dtype=float)
    alpha = alpha0 + Nbar
    alpha_star = np.sum(alpha)
    if ~np.isfinite(alpha_star):
        Elogm = np.nan * np.ones_like(alpha)
        logC = np.nan
    else:
        Elogm = psi(alpha) - psi(alpha_star)
        loggamma = gammaln(alpha)
        logC = gammaln(alpha_star) - np.sum(loggamma)
    Elogpm = logC0 + np.sum((alpha0 - 1) * Elogm)
    Elogqm = logC + np.sum((alpha - 1) * Elogm)
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

# hbi_qHZ

def hbi_qHZ(
    qmutau: List[GaussianGammaDistribution],
    qm: DirichletDistribution,
    qh: IndividualPosterior,
    thetabar: List[np.ndarray],
    Sdiag: List[np.ndarray],
) -> Tuple[np.ndarray, BoundQHZ]:
    qmlimInf = bool(qm.limInf)
    logf = np.asarray(qh.loglik, dtype=float)
    logdetA = np.asarray(qh.log_det_hessian, dtype=float)
    K, N = logf.shape
    r = np.zeros((K, N), dtype=float)
    ElogpH = np.full(K, np.nan)
    ElogpZ = np.full(K, np.nan)
    ElogpX = np.full(K, np.nan)
    ElogqH = np.full(K, np.nan)
    ElogqZ = np.full(K, np.nan)
    D = np.array([len(qmutau[k].a) for k in range(K)], dtype=float)
    ElogdetT = np.array([np.sum(qmutau[k].Elogtau) for k in range(K)], dtype=float)
    logdetET = np.array([np.sum(np.log(qmutau[k].Etau)) for k in range(K)], dtype=float)
    beta = np.array([qmutau[k].beta for k in range(K)], dtype=float)
    lambda_vec = 0.5 * ElogdetT - 0.5 * logdetET - 0.5 * D / beta
    shift = 0.5 * D * np.log(2 * np.pi) + lambda_vec + qm.Elogm
    logrho = logf - 0.5 * logdetA
    logrho = logrho + shift[:, np.newaxis]
    if qmlimInf:
        r[:, :] = 1.0 / K
    else:
        for k in range(K):
            rarg = logrho - logrho[k, :][np.newaxis, :]
            r[k, :] = 1.0 / np.sum(np.exp(rarg), axis=0)
    logeps = np.exp(np.log1p(-1 + np.finfo(float).eps))
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
            - 0.5 * Nk * np.sum(Etau_k * (Sd_k + (tb_k - a_k) ** 2))
        )
        ElogpZ[k] = Nk * qm.Elogm[k]
        ElogpXH = np.sum(r[k, :] * (logf[k, :] - 0.5 * Dk + lambda_vec[k]))
        ElogpX[k] = ElogpXH - ElogpH[k]
        r_k = r[k, :]
        rlogr = r_k * np.log1p(r_k - 1.0)
        rlogr[r_k < logeps] = 0.0
        ElogqH[k] = np.sum(
            r_k * (-Dk / 2 - Dk / 2 * np.log(2 * np.pi) + 0.5 * logdetA[k, :])
        )
        ElogqZ[k] = np.sum(rlogr)
    bound = BoundQHZ(
        ElogpX=ElogpX,
        ElogpH=ElogpH,
        ElogpZ=ElogpZ,
        ElogqH=ElogqH,
        ElogqZ=ElogqZ,
    )
    return r, bound


# hbi_qhquad (moved from hbi_all)

def hbi_qhquad(
    models: List[Any],
    data: List[Any],
    pconfig: List[Dict[str, Any]],
    qmutau: List[GaussianGammaDistribution],
    qh: IndividualPosterior,
    fid,
) -> IndividualPosterior:
    N = len(data)
    K = len(models)
    verbose_vec = np.zeros(K, dtype=int)
    if not np.all(verbose_vec > 0):
        fid = None
    theta_list = []
    Ainvdiag_list = []
    logf = np.zeros((K, N), dtype=float)
    flag = np.zeros((K, N), dtype=int)
    logdetA = np.zeros((K, N), dtype=float)
    for k in range(K):
        a_k = np.asarray(qmutau[k].a)
        Etau_k = np.asarray(qmutau[k].Etau)
        Dk = len(a_k)
        prior = GaussianDistribution(mean=a_k, precision=np.diagflat(Etau_k))
        cfg = deepcopy(pconfig[k])
        theta_k = np.zeros((Dk, N), dtype=float)
        Ainvdiag_k = np.zeros((Dk, N), dtype=float)
        for n in range(N):
            cfg.inits = qh.parameters[k][:, n]
            from .map_estimation import optimize_map, log_posterior
            logf_kn, theta_kn, A_kn, _, flag_kn = optimize_map(
                data[n], models[k], cfg, prior.mean.flatten(), prior.precision, 'LAP'
            )
            if flag_kn == 0:
                theta_kn = prior.mean.flatten()
                logf_kn = log_posterior(theta_kn, models[k], data[n], prior.mean.flatten(), prior.precision)
                A_kn = prior.precision
            logf[k, n] = logf_kn
            theta_k[:, n] = theta_kn
            flag[k, n] = flag_kn
            cholA = np.linalg.cholesky(A_kn)
            logdetA_kn = 2.0 * np.sum(np.log(np.diag(cholA)))
            Ainv = np.linalg.inv(A_kn)
            Ainvdiag_k[:, n] = np.diag(Ainv)
            logdetA[k, n] = logdetA_kn
        Ainvdiag_list.append(Ainvdiag_k)
        theta_list.append(theta_k)
    qh_new = IndividualPosterior(
        loglik=logf,
        parameters=theta_list,
        hessian_inv_diag=Ainvdiag_list,
        log_det_hessian=logdetA,
    )
    return qh_new


# hbi_bound (moved from hbi_all)

def hbi_bound(bound: BoundState, lastmodule: str) -> Tuple[BoundState, float]:
    bb = bound.bound
    pmlimInf = bool(bb.pmlimInf)
    Elogpm_Elogqm = bb.Elogpm - bb.Elogqm
    if pmlimInf:
        Elogpm_Elogqm = 0.0
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
        + Elogpm_Elogqm
    )
    if lastmodule == "qHZ":
        bh = bound.qHZ
        bb.ElogpX = float(np.sum(bh.ElogpX))
        bb.ElogpH = float(np.sum(bh.ElogpH))
        bb.ElogpZ = float(np.sum(bh.ElogpZ))
        bb.ElogqH = float(np.sum(bh.ElogqH))
        bb.ElogqZ = float(np.sum(bh.ElogqZ))
    elif lastmodule == "qmutau":
        bq = bound.qmutau
        bb.ElogpH = float(np.sum(bq.ElogpH))
        bb.Elogpmu = float(np.sum(bq.Elogpmu))
        bb.Elogptau = float(np.sum(bq.Elogptau))
        bb.Elogqmu = float(np.sum(bq.Elogqmu))
        bb.Elogqtau = float(np.sum(bq.Elogqtau))
    elif lastmodule == "qm":
        bm = bound.qm
        bb.ElogpZ = float(np.sum(bm.ElogpZ))
        bb.Elogpm = float(bm.Elogpm)
        bb.Elogqm = float(bm.Elogqm)
    Elogpm_Elogqm = bb.Elogpm - bb.Elogqm
    if pmlimInf:
        Elogpm_Elogqm = 0.0
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
        + Elogpm_Elogqm
    )
    dL = float(L - L_pre)
    bb.lastmodule = lastmodule
    bb.L = float(L)
    bb.dL = dL
    return bound, dL
