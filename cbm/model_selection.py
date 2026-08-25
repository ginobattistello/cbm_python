"""Bayesian Model Selection for Group Studies.

References
----------
Stephan KE et al. (2009), NeuroImage 46:1004-1017.
Rigoux L et al. (2014), NeuroImage 84:971-985.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import psi, gammaln


RandomStateLike = Optional[Union[int, np.random.Generator]]


def _resolve_rng(random_state: RandomStateLike) -> np.random.Generator:
    """Return a local Generator without touching NumPy's global RNG."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


@dataclass
class BMSResult:
    """Result from Bayesian model selection."""

    model_frequency: np.ndarray
    exceedance_prob: Optional[np.ndarray]
    protected_exceedance_prob: Optional[np.ndarray]
    posterior_parameters: np.ndarray
    bor: float
    g: np.ndarray


def bms(
    lme: np.ndarray,
    Nsamp: int = int(1e6),
    alpha0: Optional[np.ndarray] = None,
    random_state: RandomStateLike = None,
) -> BMSResult:
    """Bayesian model selection for group studies.

    Parameters
    ----------
    lme
        Subjects x models log-model-evidence matrix.
    Nsamp
        Monte-Carlo samples used for exceedance probabilities.
    alpha0
        Prior Dirichlet model counts.
    random_state
        Integer seed, ``numpy.random.Generator``, or ``None``. This controls
        only Monte-Carlo exceedance sampling; the VB solution is deterministic.
    """
    lme = np.asarray(lme, dtype=float)
    if lme.ndim != 2:
        raise ValueError("lme must be a 2D subjects x models array")
    if not np.all(np.isfinite(lme)):
        raise ValueError("lme contains non-finite values")

    Ni, Nk = lme.shape
    cc = 10e-4

    if alpha0 is None:
        alpha0 = np.ones(Nk)
    alpha0 = np.asarray(alpha0, dtype=float).reshape(-1)
    if alpha0.shape != (Nk,) or np.any(alpha0 <= 0):
        raise ValueError("alpha0 must contain one positive value per model")

    alpha = alpha0.copy()

    converged = False
    while not converged:
        log_u = np.zeros((Ni, Nk))
        for i in range(Ni):
            for k in range(Nk):
                log_u[i, k] = (
                    lme[i, k]
                    + psi(alpha[k])
                    - psi(np.sum(alpha))
                )

        u = np.exp(log_u - np.max(log_u, axis=1, keepdims=True))
        g = u / np.sum(u, axis=1, keepdims=True)
        beta = np.sum(g, axis=0)

        prev = alpha.copy()
        alpha = alpha0 + beta
        converged = np.linalg.norm(alpha - prev) <= cc

    exp_r = alpha / np.sum(alpha)

    rng = _resolve_rng(random_state)
    xp = dirichlet_exceedance(alpha, Nsamp, random_state=rng)

    posterior = {"a": alpha, "r": g.T}
    priors = {"a": alpha0}
    bor = compute_bor(lme.T, posterior, priors)

    pxp = (1 - bor) * xp + bor / Nk if xp is not None else None

    return BMSResult(
        posterior_parameters=alpha,
        model_frequency=exp_r,
        exceedance_prob=xp,
        protected_exceedance_prob=pxp,
        bor=bor,
        g=g,
    )


def compute_bor(
    L: np.ndarray,
    posterior: dict,
    priors: dict,
    C: Optional[np.ndarray] = None,
) -> float:
    """Compute Bayes Omnibus Risk."""
    if C is None:
        options = {"families": False}
        F0, _ = fe_null(L, options)
    else:
        options = {"families": True, "C": C}
        _, F0 = fe_null(L, options)

    F1 = compute_fe(L, posterior, priors)
    return float(1.0 / (1.0 + np.exp(F1 - F0)))


def compute_fe(L: np.ndarray, posterior: dict, priors: dict) -> float:
    """Variational free energy for the current approximate posterior."""
    K, n = L.shape
    a0 = np.sum(posterior["a"])
    Elogr = psi(posterior["a"]) - psi(np.sum(posterior["a"]))

    Sqf = (
        np.sum(gammaln(posterior["a"]))
        - gammaln(a0)
        - np.sum((posterior["a"] - 1) * Elogr)
    )

    Sqm = 0.0
    for i in range(n):
        Sqm -= np.sum(
            posterior["r"][:, i]
            * np.log(posterior["r"][:, i] + np.finfo(float).eps)
        )

    ELJ = (
        gammaln(np.sum(priors["a"]))
        - np.sum(gammaln(priors["a"]))
        + np.sum((priors["a"] - 1) * Elogr)
    )

    for i in range(n):
        for k in range(K):
            ELJ += posterior["r"][k, i] * (
                Elogr[k] + L[k, i]
            )

    return float(ELJ + Sqf + Sqm)


def fe_null(
    L: np.ndarray,
    options: dict,
) -> Tuple[float, Optional[float]]:
    """Free energy of the null hypothesis of equal frequencies."""
    K, n = L.shape

    if options["families"]:
        C = options["C"]
        # Equal total mass per family, divided equally between its models.
        f0 = C @ (np.sum(C, axis=0) ** -1.0) / C.shape[1]
        F0f = 0.0
    else:
        F0f = None

    F0m = 0.0

    for i in range(n):
        tmp = L[:, i] - np.max(L[:, i])
        g = np.exp(tmp) / np.sum(np.exp(tmp))

        for k in range(K):
            F0m += g[k] * (
                L[k, i]
                - np.log(K)
                - np.log(g[k] + np.finfo(float).eps)
            )

            if options["families"]:
                F0f += g[k] * (
                    L[k, i]
                    - np.log(g[k] + np.finfo(float).eps)
                    + np.log(f0[k])
                )

    return float(F0m), None if F0f is None else float(F0f)


def dirichlet_exceedance(
    alpha: np.ndarray,
    Nsamp: int,
    random_state: RandomStateLike = None,
) -> np.ndarray:
    """Monte-Carlo exceedance probabilities for a Dirichlet distribution.

    Uses a local ``numpy.random.Generator`` so callers can reproduce BMS
    figures/results without modifying global NumPy random state.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if np.any(alpha <= 0):
        raise ValueError("Dirichlet parameters must be positive")
    if int(Nsamp) <= 0:
        raise ValueError("Nsamp must be > 0")

    rng = _resolve_rng(random_state)
    Nk = len(alpha)
    Nsamp = int(Nsamp)

    blk_n = int(np.ceil(Nsamp * Nk * 8 / 2 ** 28))
    blk = np.floor(
        Nsamp / blk_n * np.ones(blk_n)
    ).astype(int)
    blk[-1] = Nsamp - np.sum(blk[:-1])

    xp = np.zeros(Nk)

    for n_block in blk:
        r = np.zeros((n_block, Nk))
        for k in range(Nk):
            r[:, k] = rng.gamma(alpha[k], 1.0, n_block)

        r /= np.sum(r, axis=1, keepdims=True)
        winner = np.argmax(r, axis=1)
        xp += np.bincount(winner, minlength=Nk)

    return xp / Nsamp
