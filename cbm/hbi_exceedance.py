"""Exceedance and protected exceedance probabilities for HBI."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from .hbi_types import ExceedanceResult


def cbm_hbi_exceedance(
    alpha,
    L: Optional[float] = np.nan,
    L0: Optional[float] = np.nan,
    Nsamp: int = int(1e6),
    is_null: bool = False,
) -> ExceedanceResult:
    """Compute exceedance probabilities from Dirichlet model frequencies.

    For two models the exceedance probability is analytic. For more than two
    models it is estimated by Dirichlet sampling.

    Protected exceedance probability requires full and null-model variational
    bounds ``L`` and ``L0``. When these are unavailable the protected values
    are NaN; ordinary exceedance probabilities remain defined.
    """
    alpha = np.asarray(alpha, dtype=float)

    xp, pxp, bor = _compute_exceedance(
        alpha,
        L,
        L0,
        Nsamp,
        is_null,
    )

    return ExceedanceResult(
        xp=xp,
        pxp=pxp,
        bor=bor,
        alpha=alpha,
        L=L,
        L0=L0,
    )


def _compute_exceedance(
    alpha: np.ndarray,
    L: float,
    L0: float,
    Nsamp: int,
    is_null: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    K = len(alpha)

    if is_null:
        xp = np.ones(K, dtype=float) / K
        return xp, xp.copy(), 1.0

    if K == 2:
        xp = np.zeros(2, dtype=float)
        xp[0] = beta_dist.cdf(
            0.5,
            a=alpha[1],
            b=alpha[0],
        )
        xp[1] = beta_dist.cdf(
            0.5,
            a=alpha[0],
            b=alpha[1],
        )
    else:
        xp = _dirichlet_exceedance(alpha, Nsamp)

    if np.isfinite(L) and np.isfinite(L0):
        # Bayesian omnibus risk (BOR).
        delta = np.clip(L - L0, -700.0, 700.0)
        bor = 1.0 / (1.0 + np.exp(delta))
        pxp = (1.0 - bor) * xp + bor / K
    else:
        bor = np.nan
        pxp = np.full(K, np.nan)

    return xp, pxp, float(bor)


def _dirichlet_exceedance(
    alpha: np.ndarray,
    Nsamp: int = int(1e6),
) -> np.ndarray:
    """Monte-Carlo exceedance probability for K > 2 models."""
    alpha = np.asarray(alpha, dtype=float)
    K = len(alpha)

    if Nsamp <= 0:
        raise ValueError("Nsamp must be positive")

    # Keep each temporary sample block below roughly 2^28 bytes.
    nblocks = max(
        1,
        int(np.ceil(Nsamp * K * 8 / 2**28)),
    )

    base = Nsamp // nblocks
    sizes = np.full(nblocks, base, dtype=int)
    sizes[-1] = Nsamp - base * (nblocks - 1)

    wins = np.zeros(K, dtype=float)

    for block_size in sizes:
        if block_size <= 0:
            continue

        gamma_samples = np.zeros(
            (block_size, K),
            dtype=float,
        )

        for k in range(K):
            gamma_samples[:, k] = np.random.gamma(
                shape=alpha[k],
                scale=1.0,
                size=block_size,
            )

        samples = gamma_samples / gamma_samples.sum(
            axis=1,
            keepdims=True,
        )
        winner = np.argmax(samples, axis=1)
        wins += np.bincount(
            winner,
            minlength=K,
        ).astype(float)

    return wins / float(Nsamp)
