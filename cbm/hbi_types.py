from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class IndividualPosterior:
    loglik: np.ndarray
    parameters: List[np.ndarray]
    hessian_inv_diag: List[np.ndarray]
    log_det_hessian: np.ndarray
    # MODIFICATION 12 — (K, N) object array of PostFitDiagnostics from
    # each refit, or None where the optimizer reported none. Optional
    # with a None default so pickles written before Mod 12 still load
    # (HBI results are routinely saved to disk and re-read by hbi_null).
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
    """Result of a hierarchical Bayesian inference run.

    Readable output (MODIFICATION 13, DEV.md §16):
        print(result)             compact summary table
        result.summary()          the same table as a string
        result.table()            one row per model
        result.subject_table()    one row per subject, with responsibilities
    """
    method: str
    input: HBIInput
    profile: HBIProfile
    math: HBIMath
    output: HBIOutput

    # ── MODIFICATION 13 — presentation only; reads existing fields,
    # never called during inference. See cbm/reporting.py.
    def summary(self, max_models: int = 12) -> str:
        from .reporting import hbi_summary
        return hbi_summary(self, max_models=max_models)

    def table(self, pandas: bool = True):
        """Model-level table: frequency, exceedance, attributed subjects."""
        from .reporting import hbi_table
        return hbi_table(self, pandas=pandas)

    def subject_table(self, pandas: bool = True):
        """Subject-level table: p(model) per candidate plus the best fit."""
        from .reporting import hbi_subject_table
        return hbi_subject_table(self, pandas=pandas)

    def __repr__(self) -> str:
        try:
            return self.summary()
        except Exception as e:
            return (f"<HBIResult {self.method!r} "
                    f"(summary failed: {type(e).__name__}: {e})>")

@dataclass
class ExceedanceResult:
    xp: np.ndarray
    pxp: np.ndarray
    bor: float
    alpha: np.ndarray
    L: float
    L0: float
