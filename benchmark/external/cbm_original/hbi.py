import os
import pickle
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from scipy.special import psi, gammaln

from cbm.hbi_exceedance import cbm_hbi_exceedance
from .hbi_types import (
    IndividualPosterior,
    ProgressChange,
    ProgressState,
    GaussianGammaDistribution,
    DirichletDistribution,
    BoundTerms,
    BoundQHZ,
    BoundQMutau,
    BoundQM,
    BoundState,
    HBIInput,
    HBIProfile,
    HBIMath,
    HBIOutput,
    HBIResult,
)

from .hbi_config import HBIConfig
from .hbi_updates import hbi_sumstats, hbi_qmutau, hbi_qm, hbi_qHZ, hbi_qhquad, hbi_bound
from .hbi_logging import hbi_log, log_header, log_iteration
__all__ = ["hbi_run", "hbi_init", "hbi_null", "HBIResult"]

# Private convergence helper (not exported)
def _hbi_prog(
    prog: List[ProgressState],
    L: float,
    alpha: np.ndarray,
    thetabar: List[np.ndarray],
    Sdiag: List[np.ndarray],
) -> Tuple[ProgressChange, List[ProgressState]]:
    last = prog[-1]
    L_pre = float(last.bound)
    alpha_pre = np.asarray(last.model_freq)
    x_pre = last.normalized_params

    thetabar_vec = np.concatenate([tb.ravel() for tb in thetabar])
    Sdiag_vec = np.concatenate([sd.ravel() for sd in Sdiag])
    x = thetabar_vec / np.sqrt(Sdiag_vec)

    dx = np.sqrt(np.mean((x - x_pre) ** 2))
    dL = float(L - L_pre)

    ibest = int(np.argmax(alpha))
    if np.isinf(alpha[ibest]):
        dalpha = np.nan
    else:
        dalpha = float(abs(alpha[ibest] - alpha_pre[ibest]))

    prog_change = ProgressChange(change_bound=dL, change_model_freq=dalpha, change_parameters=float(dx))
    prog.append(ProgressState(bound=float(L), model_freq=alpha.copy(), normalized_params=x))

    return prog_change, prog

def hbi_main(data: List[Any], models: List[Any], fcbm_maps: List[str], fname: str = "", config: Union[HBIConfig, Dict[str, Any]] = None, optimconfigs: List[Any] = None) -> HBIResult:
    """
    Main function to run HBI.

    Parameters
    ----------
    data : list
        List of subject-level data objects.
    models : list
        List of model functions.
    fcbm_maps : list
        List of file paths or dicts for CBM maps.
    config : HBIConfig or dict
        Configuration for HBI.
    optimconfigs : list
        List of optimization configurations for each model.
    fname : str, optional
        Filename to save the resulting CBM object, by default "".

    Returns
    -------
    HBIResult
        The result of the HBI run.
    """
    user_input = {
        "models": models,
        "fcbm_maps": fcbm_maps,
        "fname": fname,
        "config": config,
        "optimconfigs": optimconfigs,
    }

    # Hyper (prior) parameters
    b = 1.0
    v = 0.5
    s = 0.01
    hyper = {"b": b, "v": v, "s": s}

    # Initialize HBI
    inits, priors, opt_configs = hbi_init(
        fcbm_maps,
        hyper,
        limInf=0,
        initialize_r='all_r_1',
    )

    # Run HBI
    cbm = hbi_run(data, user_input, inits, priors, opt_configs)
    return cbm

def hbi_run(data: List[Any], user_input: Dict[str, Any], inits: Dict[str, Any], priors: Dict[str, Any], opt_configs: List[Any]) -> HBIResult:
    models = user_input["models"]
    fcbm_maps = user_input["fcbm_maps"]
    fname = user_input.get("fname", None)
    config_in = user_input["config"]
    optconfigs_in = opt_configs

    K = len(models)
    N = len(data)

    qhquad = inits["qh"]
    r = np.asarray(inits["r"], dtype=float)
    bound = deepcopy(inits["bound"])

    hyper = priors["hyper"]
    pmutau = deepcopy(priors["pmutau"]) if isinstance(priors["pmutau"], list) else [deepcopy(priors["pmutau"])]
    pm = deepcopy(priors["pm"])

    isnull = bool(pm.limInf) if isinstance(pm, DirichletDistribution) else bool(pm["limInf"]) 

    if isinstance(config_in, HBIConfig):
        config = config_in
    else:
        config = HBIConfig(**config_in)

    flog = config.flog
    fname_prog = config.fname_prog
    save_prog = bool(config.save_prog)
    verbose = bool(config.verbose)
    maxiter = config.maxiter
    tolx = config.tolx

    if (flog is None or flog == "") and fname:
        fdir, fn = os.path.split(fname)
        if fdir == "":
            fdir = "."
        flog = os.path.join(fdir, f"{os.path.splitext(fn)[0]}.log")

    fid_file = None
    if flog != -1 and isinstance(flog, str) and flog != "":
        fid_file = open(flog, "w")
    fid = fid_file

    verbose_multiK = bool(verbose and (K > 1) and (not isnull))
    fid_multiK = fid if (K > 1 and not isnull) else None

    optconfigs = []
    for k in range(K):
        d = len(pmutau[k].a) if isinstance(pmutau[k], GaussianGammaDistribution) else len(pmutau[k]["a"])
        optfigk = {}
        if len(optconfigs_in) > 0:
            optfigk = deepcopy(optconfigs_in[k])
        optfigk.num_init = 0
        optfigk.num_init_med = 0
        optfigk.num_init_up = 3
        optfigk.verbose = False
        optconfigs.append(optfigk)

    log_header(verbose, fid, K, N, fcbm_maps, isnull)

    prog = [
        ProgressState(
            bound=bound.bound.L,
            model_freq=np.asarray(pm.alpha, dtype=float) if isinstance(pm, DirichletDistribution) else np.asarray(pm["alpha"], dtype=float),
            normalized_params=np.nan,
        )
    ]

    terminate = False
    it = 0
    math_list: List[Dict[str, Any]] = []

    while not terminate and it <= maxiter:
        it += 1
        hbi_log(verbose, fid, f"Iteration {it:02d}\n")
        Nbar, thetabar, Sdiag = hbi_sumstats(r, qhquad)
        qmutau, bound_qmutau = hbi_qmutau(pmutau, Nbar, thetabar, Sdiag)
        bound.qmutau = bound_qmutau
        bound, _ = hbi_bound(bound, "qmutau")
        qm, bound_qm = hbi_qm(pm, Nbar)
        bound.qm = bound_qm
        bound, _ = hbi_bound(bound, "qm")
        qhquad = hbi_qhquad(models, data, optconfigs, qmutau, qhquad, fid)
        r, bound_qHZ = hbi_qHZ(qmutau, qm, qhquad, thetabar, Sdiag)
        bound.qHZ = bound_qHZ
        bound, _ = hbi_bound(bound, "qHZ")
        prog_change, prog = _hbi_prog(prog, bound.bound.L, qm.alpha, thetabar, Sdiag)
        if prog_change.change_parameters < tolx:
            terminate = True
        if it > 1:
            log_iteration(verbose, fid, verbose_multiK, fid_multiK, it, Nbar, N, prog_change, terminate, K)
        math_iter = {
            "qhquad": deepcopy(qhquad),
            "r": r.copy(),
            "Nbar": Nbar.copy(),
            "thetabar": [tb.copy() for tb in thetabar],
            "Sdiag": [sd.copy() for sd in Sdiag],
            "pm": deepcopy(pm),
            "pmutau": deepcopy(pmutau),
            "qm": deepcopy(qm),
            "qmutau": deepcopy(qmutau),
            "bound": deepcopy(bound),
            "prog": deepcopy(prog),
            "prog_change": deepcopy(prog_change),
            "input": deepcopy(user_input),
            "hyper": deepcopy(hyper),
        }
        math_list.append(math_iter)
        if save_prog and fname_prog:
            with open(fname_prog, "wb") as f:
                pickle.dump(math_list, f)

    qmutau_list: List[GaussianGammaDistribution] = qmutau
    he_list: List[np.ndarray] = [None] * K
    nk_vec: np.ndarray = np.zeros(K, dtype=float)
    for k in range(K):
        nu = qmutau_list[k].nu
        beta = qmutau_list[k].beta
        sigma = np.asarray(qmutau_list[k].sigma)
        s2 = 2.0 * sigma / beta
        nk = 2.0 * nu
        he_list[k] = np.sqrt(s2 / nk)
        nk_vec[k] = nk

    exceedance = cbm_hbi_exceedance(qm.alpha, is_null = isnull)

    theta_list = qhquad.parameters
    r_mat = r
    r_out = r_mat.T

    a_list: List[np.ndarray] = [None] * K
    # he_list already calculated above, don't reinitialize
    # nk_vec already calculated above, don't reinitialize

    theta_out: List[np.ndarray] = [None] * K
    for k in range(K):
        theta_k = theta_list[k].T
        theta_out[k] = theta_k
        a_list[k] = qmutau[k].a.copy()

    xp = exceedance.xp
    pxp = exceedance.pxp

    output = HBIOutput(
        parameters=theta_out,
        responsibility=r_out,
        group_mean=a_list,
        group_hierarchical_errorbar=he_list,
        model_frequency=Nbar / N,
        exceedance_prob=xp,
        protected_exceedance_prob=pxp,
    )

    hyper_out = hyper
    profile = HBIProfile(
        datetime=datetime.now().isoformat(),
        filename="cbm_hbi_hbi",
        config=config,
        optimconfigs=optconfigs,
        hyperparameters=hyper_out,
    )

    cbm_input = HBIInput(
        models=user_input["models"],
        fcbm_maps=user_input["fcbm_maps"],
        fname=user_input.get("fname", ""),
        config=user_input["config"],
        optimconfigs=user_input.get("optimconfigs", None),
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

    cbm = HBIResult(
        method="hbi",
        input=cbm_input,
        profile=profile,
        math=cbm_math,
        output=output,
    )

    # log_final(verbose, fid, output)

    if fname:
        with open(fname, "wb") as f:
            pickle.dump(cbm, f)

    if fid_file is not None:
        fid_file.close()

    return cbm


def hbi_init(flap, hyper, limInf=0, initialize_r='all_r_1', families=None):
    if families is None:
        families = []
    b = hyper['b']
    v = hyper['v']
    s = hyper['s']
    K = len(flap)
    allfiles_map = True
    cbm_maps = []
    for k in range(K):
        fcbm_map = flap[k]
        if isinstance(fcbm_map, str):
            with open(fcbm_map, 'rb') as f:
                cbm = pickle.load(f)
                cbm_maps.append(cbm)
        elif isinstance(fcbm_map, dict):
            allfiles_map = allfiles_map and False
            cbm_maps.append(fcbm_map)
        else:
            raise ValueError(
                f"fcbm_map input has not properly been specified for model {k + 1}!"
            )
    bb = BoundTerms(
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
        lastmodule='',
        L=np.nan,
        dL=np.nan,
    )
    opt_configs = []
    for k in range(K):
        cbm_map = cbm_maps[k]
        opt_config = cbm_map.profile.config
        opt_configs.append(opt_config)
    logrho = []
    theta = []
    Ainvdiag = []
    logdetA = []
    logf = []
    D = []
    a0 = []
    N = cbm_maps[0].output.parameters.shape[0]
    for k in range(K):
        cbm_map = cbm_maps[k]
        logrho.append(np.asarray(cbm_map.math.lme))
        logf.append(np.asarray(cbm_map.math.loglik))
        a0.append(np.asarray(cbm_map.profile.prior_mean))
        theta_k = cbm_map.math.parameters
        Ainvdiag_k = cbm_map.math.hessian_inv_diag
        theta.append(np.column_stack(theta_k))
        Ainvdiag.append(np.column_stack(Ainvdiag_k))
        logdetA.append(np.asarray(cbm_map.math.log_det_hessian))
        D.append(theta[k].shape[0])
    logf_mat = np.vstack(logf)
    logdetA_mat = np.vstack(logdetA)
    D = np.array(D)
    qh = IndividualPosterior(
        loglik=logf_mat,
        parameters=theta,
        hessian_inv_diag=Ainvdiag,
        log_det_hessian=logdetA_mat,
    )
    a = []
    beta = []
    sigma = []
    nu = []
    alpha0 = np.ones(K)
    for k in range(K):
        a.append(np.asarray(a0[k]))
        beta.append(b)
        if not isinstance(s, (list, tuple)):
            sigma.append(s * np.ones_like(a[k]))
        else:
            sigma_k = np.asarray(s[k])
            if sigma_k.shape != a[k].shape:
                raise ValueError(
                    f"length of s is not match with that for a for model {k + 1}"
                )
            sigma.append(sigma_k)
        nu.append(v)
    if len(families) > 0:
        families_arr = np.asarray(families)
        alpha0[:] = np.nan
        uf = np.unique(families_arr)
        for f in uf:
            mask = (families_arr == f)
            nf = mask.sum()
            alpha0[mask] = 1.0 / nf
    pmutau: List[GaussianGammaDistribution] = []
    for k in range(K):
        pmutau.append(
            GaussianGammaDistribution(
                a=a[k],
                beta=beta[k],
                sigma=sigma[k],
                nu=nu[k],
                Etau=np.zeros_like(a[k]),
                Elogtau=np.zeros_like(a[k]),
                logG=0.0,
            )
        )
    pm_alpha = alpha0.copy()
    pm = DirichletDistribution(
        limInf=bool(limInf),
        alpha=pm_alpha,
        Elogm=np.zeros_like(pm_alpha, dtype=float),
        logC=0.0,
    )
    for k in range(K):
        a_k = pmutau[k].a
        beta_k = pmutau[k].beta
        nu_k = np.asarray(pmutau[k].nu)
        sigma_k = np.asarray(pmutau[k].sigma)
        Elogtau = psi(nu_k) - np.log(sigma_k)
        Etau = nu_k / sigma_k
        logG = np.sum(-gammaln(nu_k) + nu_k * np.log(sigma_k))
        pmutau[k] = GaussianGammaDistribution(
            a=a_k,
            beta=beta_k,
            sigma=np.asarray(sigma_k),
            nu=float(nu_k),
            Etau=np.asarray(Etau),
            Elogtau=np.asarray(Elogtau),
            logG=float(logG),
        )
    alpha = pm.alpha
    alpha_star = np.sum(alpha)
    Elogm = psi(alpha) - psi(alpha_star)
    loggamma1 = gammaln(alpha)
    logC = gammaln(alpha_star) - np.sum(loggamma1)
    if pm.limInf:
        pm.alpha = np.full_like(alpha, np.inf, dtype=float)
        Elogm = np.full_like(alpha, np.inf, dtype=float)
        logC = 0.0
    pm.Elogm = Elogm
    pm.logC = float(logC)
    lme = np.vstack(logrho).T
    if initialize_r == 'all_r_1':
        r = np.ones((K, N))
    else:
        raise NotImplementedError(
            f"initialize_r option '{initialize_r}' not implemented."
        )
    bound = BoundState(
        bound=bb,
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
        'qh': qh,
        'r': r,
        'bound': bound,
    }
    priors = {
        'hyper': hyper,
        'pmutau': pmutau,
        'pm': pm,
    }
    return inits, priors, opt_configs


def hbi_null(
    data: List[Any],
    fname_cbm: Union[str, HBIResult],
) -> HBIResult:
    """
    Parameters
    ----------
    data : list
        List of subject-level data objects.
    fname_cbm : str or CBMResult
        If str: path to a saved cbm object (pickled).
        If CBMResult: already-loaded cbm structure.

    Returns
    -------
    cbm : CBMResult
        Original HBI result, updated with protected exceedance probs.
    cbm0 : CBMResult
        HBI result under the null hypothesis.
    """
    # ------------------------------------------------------------------
    # Load cbm if a filename is given
    # ------------------------------------------------------------------
    inputisfile = False
    fname = None

    if isinstance(fname_cbm, str):
        inputisfile = True
        fname = fname_cbm
        with open(fname, "rb") as f:
            loaded = pickle.load(f)
        # handle either cbm or {'cbm': cbm}
        if isinstance(loaded, dict) and "cbm" in loaded:
            cbm = loaded["cbm"]
        else:
            cbm = loaded
    elif isinstance(fname_cbm, HBIResult):
        cbm = fname_cbm
    else:
        raise TypeError("fname_cbm must be either a filename (str) or a CBMResult")

    # ------------------------------------------------------------------
    # Derive output filename for null model if we loaded from file
    # ------------------------------------------------------------------
    fname0 = None
    if inputisfile:
        fdir, fbase = os.path.split(fname)
        root, ext = os.path.splitext(fbase)
        if not ext:
            ext = ".pkl"
        fname0 = os.path.join(fdir, f"{root}_null{ext}")

    # ------------------------------------------------------------------
    # Extract input components
    # ------------------------------------------------------------------
    models = cbm.input.models
    fcbm_maps = cbm.input.fcbm_maps
    config = cbm.input.config
    optimconfigs = cbm.input.optimconfigs if cbm.input.optimconfigs is not None else []
    hyper = cbm.profile.hyperparameters
    isnull = 1  # used for limInf in cbm_hbi_init

    # ------------------------------------------------------------------
    # Adjust config for null run: flog and fname_prog get "_null"
    # ------------------------------------------------------------------
    # config may be an HBIConfig dataclass or a dict
    if isinstance(config, HBIConfig):
        config_null = deepcopy(config)
        # flog
        if isinstance(config_null.flog, str):
            fdir, fbase = os.path.split(config_null.flog)
            root, ext = os.path.splitext(fbase)
            config_null.flog = os.path.join(fdir or ".", f"{root}_null{ext}")
        # fname_prog
        if isinstance(config_null.fname_prog, str):
            fdir, fbase = os.path.split(config_null.fname_prog)
            root, ext = os.path.splitext(fbase)
            if not ext:
                ext = ".pkl"
            config_null.fname_prog = os.path.join(fdir or ".", f"{root}_null{ext}")
    else:
        # assume dict-like
        config_null = deepcopy(config)
        # flog
        flog = config_null.get("flog", None)
        if isinstance(flog, str):
            fdir, fbase = os.path.split(flog)
            root, ext = os.path.splitext(fbase)
            config_null["flog"] = os.path.join(fdir or ".", f"{root}_null{ext}")
        # fname_prog
        fname_prog = config_null.get("fname_prog", None)
        if isinstance(fname_prog, str):
            fdir, fbase = os.path.split(fname_prog)
            root, ext = os.path.splitext(fbase)
            if not ext:
                ext = ".pkl"
            config_null["fname_prog"] = os.path.join(fdir or ".", f"{root}_null{ext}")

    # ------------------------------------------------------------------
    # Build user_input for the null HBI run
    # ------------------------------------------------------------------
    user_input = {
        "models": models,
        "fcbm_maps": fcbm_maps,
        "fname": fname0,
        "config": config_null,
        "optimconfigs": optimconfigs,
    }

    # ensure we have an HBIConfig instance for initialization
    if isinstance(config_null, HBIConfig):
        config_for_init = config_null
    else:
        config_for_init = HBIConfig(**config_null)

    # ------------------------------------------------------------------
    # Initialize HBI under the null (limInf = isnull)
    # ------------------------------------------------------------------
    inits, priors, opt_configs = hbi_init(
        fcbm_maps,
        hyper,
        isnull,
        config_for_init.initialize,
    )

    # ------------------------------------------------------------------
    # Run HBI under null hypothesis
    # ------------------------------------------------------------------
    cbm0 = hbi_run(data, user_input, inits, priors, opt_configs)

    # ------------------------------------------------------------------
    # Use cbm0 to compute protected exceedance probability
    # ------------------------------------------------------------------
    alpha = np.asarray(cbm.math.qm.alpha, dtype=float)
    L = float(cbm.math.bound.bound.L)
    L0 = float(cbm0.math.bound.bound.L)

    exceedance = cbm_hbi_exceedance(alpha, L=L, L0=L0)

    # Update cbm with exceedance results
    cbm.math.exceedance = exceedance
    cbm.output.protected_exceedance_prob = exceedance.pxp

    # ------------------------------------------------------------------
    # Save updated cbm back to file if needed
    # ------------------------------------------------------------------
    if inputisfile and fname is not None:
        with open(fname, "wb") as f:
            pickle.dump(cbm, f)

    return cbm