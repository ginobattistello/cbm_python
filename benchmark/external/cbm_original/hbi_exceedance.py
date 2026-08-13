import numpy as np
from typing import Optional, Tuple

from scipy.stats import beta as beta_dist
from .hbi_types import ExceedanceResult

def cbm_hbi_exceedance(
    alpha,
    L: Optional[float] = np.nan,
    L0: Optional[float] = np.nan,
    Nsamp: int = int(1e6),
    is_null: bool = False,
) -> Tuple[ExceedanceResult, np.ndarray, np.ndarray, float]:
    """
    Parameters
    ----------
    alpha : array-like
        Dirichlet parameters of q(m).
    L : float or NaN, optional
        Log-evidence under full model.
    L0 : float or NaN, optional
        Log-evidence under null model.
    Nsamp : int, optional
        Number of samples for Dirichlet exceedance when K > 2.

    Returns
    -------
    exceedance : ExceedanceResult
        Struct-like container (xp, pxp, bor, alpha, L, L0).
    xp : np.ndarray
        Exceedance probabilities.
    pxp : np.ndarray
        Protected exceedance probabilities (may be NaN if L/L0 are NaN).
    bor : float
        Probability that model differences are due to chance.
    """
    alpha = np.asarray(alpha, dtype=float)

    xp, pxp, bor = _compute_exceedance(alpha, L, L0, Nsamp, is_null)

    exceedance = ExceedanceResult(
        xp=xp,
        pxp=pxp,
        bor=bor,
        alpha=alpha,
        L=L,
        L0=L0,
    )
    return exceedance


def _compute_exceedance(
    alpha: np.ndarray, L: float, L0: float, Nsamp: int, is_null: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Helper corresponding to compute_exceedance.m
    """
    K = len(alpha)

    if is_null:
        # In null mode, all models are equally likely
        xp = np.ones(K, dtype=float) / K
        bor = 1.0
        pxp = np.ones(K, dtype=float) / K
        return xp, pxp, bor
    else:
        # 1) Exceedance probs xp
        if K == 2:
            # for two models: analytic via Beta CDF
            xp = np.zeros(2, dtype=float)
            xp[0] = beta_dist.cdf(0.5, a=alpha[1], b=alpha[0])
            xp[1] = beta_dist.cdf(0.5, a=alpha[0], b=alpha[1])
        else:
            # for K > 2: sampling approach
            xp = _dirichlet_exceedance(alpha, Nsamp)

        # 2) Probability that differences are due to chance (BOR)
        # bor = 1 / (1 + exp(L - L0))
        bor = 1.0 / (1.0 + np.exp(L - L0))

        # 3) Protected exceedance probabilities (Rigoux et al. Eq. 7)
        pxp = (1.0 - bor) * xp + bor / K

        return xp, pxp, bor


def _dirichlet_exceedance(alpha: np.ndarray, Nsamp: int = int(1e6)) -> np.ndarray:
    """
    Sampling approximation to Dirichlet exceedance probabilities.

    Parameters
    ----------
    alpha : array-like, shape (K,)
        Dirichlet parameters.
    Nsamp : int, optional
        Number of samples.

    Returns
    -------
    xp : np.ndarray, shape (K,)
        Exceedance probability for each model.
    """
    alpha = np.asarray(alpha, dtype=float)
    K = len(alpha)

    # Rough bytes ~ Nsamp * K * 8; split so this stays under ~2^28 bytes.
    nblocks = int(np.ceil(Nsamp * K * 8 / 2**28))
    nblocks = max(1, nblocks)

    # Sizes for each block
    base_block = Nsamp // nblocks
    blk_sizes = np.full(nblocks, base_block, dtype=int)
    blk_sizes[-1] = Nsamp - base_block * (nblocks - 1)

    xp = np.zeros(K, dtype=float)

    for blk in blk_sizes:
        if blk <= 0:
            continue

        # Sample gamma for each component and normalize → Dirichlet samples
        # r: (blk, K)
        r = np.zeros((blk, K), dtype=float)
        for k in range(K):
            # shape = alpha[k], scale = 1
            r[:, k] = np.random.gamma(shape=alpha[k], scale=1.0, size=blk)

        sr = r.sum(axis=1, keepdims=True)
        r = r / sr

        # For each sample, find argmax model
        j = np.argmax(r, axis=1)  # 0..K-1

        # Count wins per model
        counts = np.bincount(j, minlength=K)
        xp += counts.astype(float)

    # Normalize by number of samples
    xp /= float(Nsamp)
    return xp