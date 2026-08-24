"""Dataclasses used by hierarchical Bayesian inference.

This module contains containers only. No HBI update equations or optimizer
logic should live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class IndividualPosterior:
    """Subject-level Gaussian approximations used inside HBI.

    Shapes follow the existing CBM convention:
    - loglik: (K, N), log joint at each subject/model MAP
    - parameters[k]: (D_k, N)
    - hessian_inv_diag[k]: (D_k, N)
    - log_det_hessian: (K, N)
    - diagnostics: optional (K, N) object array
    """

    loglik: np.ndarray
    parameters: List[np.ndarray]
    hessian_inv_diag: List[np.ndarray]
    log_det_hessian: np.ndarray
    diagnostics: Optional[np.ndarray] = None


@dataclass
class ProgressChange:
    change_bound: float
    change_model_freq: float
    change_parameters: float


@dataclass
class ProgressState:
    bound: float
    model_freq: np.ndarray
    normalized_params: Any


@dataclass
class GaussianDistribution:
    mean: np.ndarray
    precision: np.ndarray


@dataclass
class GaussianGammaDistribution:
    """Factorized Gaussian-Gamma posterior for group parameters."""

    a: np.ndarray
    beta: float
    sigma: np.ndarray
    nu: float
    Etau: np.ndarray
    Elogtau: np.ndarray
    logG: float


@dataclass
class DirichletDistribution:
    limInf: bool
    alpha: np.ndarray
    Elogm: np.ndarray
    logC: float


@dataclass
class BoundTerms:
    ElogpX: float
    ElogpH: float
    ElogpZ: float
    Elogpmu: float
    Elogptau: float
    Elogpm: float
    ElogqH: float
    ElogqZ: float
    Elogqmu: float
    Elogqtau: float
    Elogqm: float
    pmlimInf: bool
    lastmodule: str
    L: float
    dL: float


@dataclass
class BoundQHZ:
    ElogpX: np.ndarray
    ElogpH: np.ndarray
    ElogpZ: np.ndarray
    ElogqH: np.ndarray
    ElogqZ: np.ndarray


@dataclass
class BoundQMutau:
    ElogpH: np.ndarray
    Elogpmu: np.ndarray
    Elogptau: np.ndarray
    Elogqmu: np.ndarray
    Elogqtau: np.ndarray


@dataclass
class BoundQM:
    ElogpZ: np.ndarray
    Elogpm: float
    Elogqm: float


@dataclass
class BoundState:
    bound: BoundTerms
    qHZ: BoundQHZ
    qmutau: BoundQMutau
    qm: BoundQM


@dataclass
class HBIInput:
    """Inputs required to reproduce an HBI run.

    ``model_trials`` and ``models_jax`` are optional derivative backends:
    - model_trials[k]: per-trial NumPy likelihood for GN MAP polishing
    - models_jax[k]: summed JAX likelihood for optional AD Hessians
    """

    models: List[Any]
    fcbm_maps: List[Any]
    fname: str
    config: Any
    optimconfigs: Any
    model_trials: Optional[List[Any]] = None
    models_jax: Optional[List[Any]] = None


@dataclass
class HBIProfile:
    datetime: str
    filename: str
    config: Any
    optimconfigs: List[Any]
    hyperparameters: Dict[str, Any]


@dataclass
class HBIMath:
    qhquad: IndividualPosterior
    r: np.ndarray
    qmutau: List[GaussianGammaDistribution]
    qm: DirichletDistribution
    bound: BoundState
    Nbar: np.ndarray
    hyper: Dict[str, Any]
    he_list: List[np.ndarray]
    nk_vec: np.ndarray
    exceedance: Any


@dataclass
class HBIOutput:
    parameters: List[np.ndarray]
    responsibility: np.ndarray
    group_mean: List[np.ndarray]
    group_hierarchical_errorbar: List[np.ndarray]
    model_frequency: np.ndarray
    exceedance_prob: np.ndarray
    protected_exceedance_prob: np.ndarray


@dataclass
class HBIResult:
    """Result of hierarchical Bayesian inference."""

    method: str
    input: HBIInput
    profile: HBIProfile
    math: HBIMath
    output: HBIOutput

    def summary(self, max_models: int = 12) -> str:
        from .reporting import hbi_summary
        return hbi_summary(self, max_models=max_models)

    def table(self, pandas: bool = True):
        from .reporting import hbi_table
        return hbi_table(self, pandas=pandas)

    def subject_table(self, pandas: bool = True):
        from .reporting import hbi_subject_table
        return hbi_subject_table(self, pandas=pandas)

    def __repr__(self) -> str:
        try:
            return self.summary()
        except Exception as exc:
            return (
                f"<HBIResult {self.method!r} "
                f"(summary failed: {type(exc).__name__}: {exc})>"
            )


@dataclass
class ExceedanceResult:
    xp: np.ndarray
    pxp: np.ndarray
    bor: float
    alpha: np.ndarray
    L: float
    L0: float
