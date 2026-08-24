from typing import Tuple
import numpy as np
from .hbi_types import BoundState


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
