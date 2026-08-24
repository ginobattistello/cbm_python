"""
FROZEN pre-fork snapshot — hbi_types.py as of commit 93a0be8 (2026-08-13),
before MODIFICATIONS 11 and 12.

Role
----
This is the HBI analogue of `optimization_legacy.py`: an unmodified reference
copy kept for A/B comparison. **Nothing in the package imports it.** The live,
modified code is at `cbm/hbi_types.py` and that is what you want unless you are
deliberately measuring what Mods 11-12 changed.

Difference from the live version
--------------------------------
MOD 11  no `model_trials` parameter, so `hbi_qhquad` calls `optimize_map` with
        six positional arguments and every internal refit uses the Mod 2
        finite-difference Hessian, whatever curvature the supplied cbm_maps
        were fitted with.
MOD 12  the `OptimizationResult` returned sixth by `optimize_map` is discarded;
        `IndividualPosterior` has no `diagnostics` field.

Its imports are rewired to the other `*_legacy` modules, so this snapshot is
self-consistent. Without that it would import the MODIFIED `hbi_qhquad` and
silently not be legacy behaviour at all.

Used by `benchmark/run_hbi_arms.py` as the `cbm_orig` arm. See DEV.md §14-§15.
"""
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

@dataclass
class IndividualPosterior:
    loglik: np.ndarray
    parameters: List[np.ndarray]
    hessian_inv_diag: List[np.ndarray]
    log_det_hessian: np.ndarray

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
    models: List[Any]
    fcbm_maps: List[str]
    fname: str
    config: Any
    optimconfigs: Any

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
    method: str
    input: HBIInput
    profile: HBIProfile
    math: HBIMath
    output: HBIOutput

@dataclass
class ExceedanceResult:
    xp: np.ndarray
    pxp: np.ndarray
    bor: float
    alpha: np.ndarray
    L: float
    L0: float
