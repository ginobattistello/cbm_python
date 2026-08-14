"""
Readable output for every result the toolbox returns (MODIFICATION 13).

WHY THIS EXISTS
---------------
Everything a user needs was already on the result objects, spread across
four branches, and printing one dumped the raw dataclass — the first several
hundred characters being `config` and prior arrays, with the estimates
somewhere after. Three specific costs:

  1. `print(fit)` is unusable, so people index into `.output.parameters`
     and never discover the rest.
  2. There is no standard error anywhere in the output. The information is
     in `math.hessian_inv_diag`, but you have to know that its square root
     is the posterior SD.
  3. MOD 9/10 compute per-subject convergence and identifiability
     diagnostics that nothing surfaces. A user who does not know they exist
     gets no warning when a fit is untrustworthy.

COVERAGE
--------
    FitResult       individual_fit    summary() · table() · se
    HBIResult       hbi_main          summary() · table() · subject_table()
    GroupBMSResult  group_bms         summary() · table()
    BtwCondsResult / BtwGroupsResult  summary()

All of it is **purely additive** — no existing field changes, and nothing
here is called during fitting or inference, so a broken formatter can never
affect a result. Every `__repr__` falls back to a short tag if formatting
raises, so a cosmetic bug can never make a result look lost.

ON PARAMETER SPACE
------------------
`output.parameters` holds theta, the UNCONSTRAINED parameters the optimizer
works in. Models typically map these to native values (alpha = sigmoid(theta),
beta = exp(theta)). This module deliberately does NOT transform them: the
toolbox cannot know your parameterisation, and silently guessing would be
worse than printing theta and saying so. Every table labels the column
`theta[i]` and the summary states the space explicitly.

REFERENCE  DEV.md §16.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Column width for the parameter block; wide enough for "theta[10]" and a
# sign, narrow enough that a 4-parameter model still fits in 80 columns.
_W = 11


def _as_2d(params) -> np.ndarray:
    """Parameters as (n_subjects, d), whatever the caller stored."""
    a = np.asarray(params, dtype=float)
    return a.reshape(1, -1) if a.ndim == 1 else a


def standard_errors(result) -> np.ndarray:
    """Posterior standard errors, (n_subjects, d).

    The Laplace approximation treats the posterior as Gaussian with
    covariance H^-1 at the MAP, so the marginal SD of each parameter is
    sqrt(diag(H^-1)) — which is exactly what `math.hessian_inv_diag`
    stores. NaN for any subject whose fit failed.

    Note these are SDs in THETA space, matching `output.parameters`. They
    do not transform to native space by simple rescaling; a delta-method
    step would be needed, and that requires knowing the transform.
    """
    theta = _as_2d(result.output.parameters)
    n, d = theta.shape
    se = np.full((n, d), np.nan)
    hid = getattr(result.math, "hessian_inv_diag", None)
    if hid is None:
        return se
    for i in range(n):
        try:
            v = np.asarray(hid[i], dtype=float).ravel()
        except Exception:
            continue
        if v.size == d:
            # A negative diagonal would mean a non-PD curvature; report NaN
            # rather than a nan-from-sqrt warning.
            se[i] = np.sqrt(np.where(v >= 0, v, np.nan))
    return se


def _diag_list(result, n: int) -> List[Optional[Any]]:
    dg = getattr(result.math, "diagnostics", None)
    if dg is None:
        return [None] * n
    return list(dg) + [None] * (n - len(dg))


def _quality(dg, flag: float) -> str:
    """One short word per subject, worst-first.

    Deliberately coarse. The point is to make a problem *visible* in a
    table someone skims; the full record stays on `math.diagnostics` for
    anyone who wants to look properly.
    """
    if dg is None:
        # No optimizer record: the fit fell back to the prior (see
        # `prior_for_failed` in Config).
        return "prior" if flag == 0.0 else "-"
    bad = []
    if getattr(dg, "at_hard_bounds", None) is not None \
            and np.any(dg.at_hard_bounds):
        bad.append("bounds")
    wi = getattr(dg, "weak_identifiability", None)
    if wi is not None and wi < 2.0:
        bad.append("weak")
    nia = getattr(dg, "n_inits_agreeing", None)
    if nia is not None and nia == 0:
        bad.append("multimodal")
    if float(flag) == 0.5:
        bad.append("singular")
    return ",".join(bad) if bad else "ok"


def rows(result) -> List[Dict[str, Any]]:
    """Per-subject records: estimates, standard errors, evidence, quality.

    Works for one subject or many — `individual_fit` keeps `parameters`
    two-dimensional either way, so a single-subject fit simply returns a
    one-row list.
    """
    theta = _as_2d(result.output.parameters)
    n, d = theta.shape
    se = standard_errors(result)
    lme = np.asarray(result.output.log_evidence, dtype=float).ravel()
    flag = np.asarray(getattr(result.math, "flag", np.full(n, np.nan)),
                      dtype=float).ravel()
    loglik = np.asarray(getattr(result.math, "loglik", np.full(n, np.nan)),
                        dtype=float).ravel()
    diags = _diag_list(result, n)

    out = []
    for i in range(n):
        r: Dict[str, Any] = {"subject": i}
        for j in range(d):
            r[f"theta[{j}]"] = float(theta[i, j])
        for j in range(d):
            r[f"se[{j}]"] = float(se[i, j])
        r["log_evidence"] = float(lme[i]) if i < lme.size else np.nan
        # Named to match the docstring on FitMath: this is the log JOINT
        # (likelihood + prior) at the MAP, not the bare log-likelihood.
        r["log_joint"] = float(loglik[i]) if i < loglik.size else np.nan
        r["flag"] = float(flag[i]) if i < flag.size else np.nan
        dg = diags[i]
        r["quality"] = _quality(dg, r["flag"])
        r["convergence"] = getattr(dg, "convergence_status", None)
        r["hess_method"] = getattr(dg, "hess_method", None)
        r["n_inits_agreeing"] = getattr(dg, "n_inits_agreeing", None)
        r["weak_identifiability"] = getattr(dg, "weak_identifiability", None)
        out.append(r)
    return out


def table(result, pandas: bool = True):
    """Per-subject table: one row per subject, ready to inspect or export.

    Returns a `pandas.DataFrame` indexed by subject. Pass `pandas=False`
    for the underlying list of dicts (useful for JSON, or for code that
    would rather not carry a DataFrame around).

    A single-subject fit gives a one-row table; no special case is needed,
    because `individual_fit` keeps `parameters` two-dimensional for n=1.
    """
    recs = rows(result)
    if not pandas:
        return recs
    return pd.DataFrame(recs).set_index("subject")


def prior_spec(result):
    """Per-parameter prior as (means, variances, defaulted_fields).

    MODIFICATION 15. `prior_variance` may be a scalar, a vector or a full
    covariance matrix; this normalises all three to one variance per
    parameter so callers do not each re-implement the unpacking.
    Falls back to inverting the precision for results saved before the
    variance was recorded.
    """
    inp = getattr(result, "input", None)
    mean = np.ravel(np.asarray(getattr(inp, "prior_mean", []), dtype=float))
    d = mean.size

    var = getattr(inp, "prior_variance", None)
    if var is None:
        prec = getattr(inp, "prior_precision", None)
        if prec is not None:
            try:
                var = np.diag(np.linalg.inv(np.asarray(prec, dtype=float)))
            except Exception:
                var = None
    if var is None:
        var = np.full(d, np.nan)
    else:
        var = np.asarray(var, dtype=float)
        if var.ndim == 0:
            var = np.full(d, float(var))
        elif var.ndim == 2:
            var = np.diag(var)
        else:
            var = np.ravel(var)
        if var.size != d:
            var = np.full(d, np.nan)
    return mean, var, tuple(getattr(inp, "prior_defaults", ()) or ())


def _fmt(v, width=_W, prec=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—".rjust(width)
    return f"{v:>{width}.{prec}f}"


def summary(result, max_subjects: int = 12) -> str:
    """A compact, copy-pasteable text summary of a fit.

    Returns the string rather than printing it, so it composes in scripts
    and notebooks; `FitResult.__repr__` calls this, so `print(fit)` and
    plain evaluation in a REPL both give the table.

    Layout is plain monospace with no box-drawing characters, so it
    survives being pasted into a plaintext email, a GitHub comment, or a
    lab notebook.
    """
    theta = _as_2d(result.output.parameters)
    n, d = theta.shape
    se = standard_errors(result)
    lme = np.asarray(result.output.log_evidence, dtype=float).ravel()
    recs = rows(result)

    model = getattr(getattr(result, "input", None), "model_name", "?")
    prof = getattr(result, "profile", None)
    elapsed = getattr(prof, "telapsed", None)
    method = getattr(result, "method", "?")
    hm = recs[0].get("hess_method") if recs else None
    curv = {"gauss_newton": "Gauss-Newton",
            "finite_diff_clipped": "finite-difference"}.get(hm, hm or "?")

    L = []
    head = f"{method} · {model} · {n} subject{'s' if n != 1 else ''}"
    L.append(head)
    L.append("=" * max(len(head), 64))
    meta = f"curvature: {curv}"
    if elapsed is not None:
        meta += f"   ·   elapsed: {elapsed:.2f}s"
    L.append(meta)
    L.append("")

    # ---- parameter block ------------------------------------------------
    # Across subjects: the distribution of estimates. For one subject the
    # mean IS the estimate, so the header says so rather than implying a
    # population summary that does not exist.
    if n == 1:
        L.append("Parameters (theta, unconstrained space)")
        L.append(f"  {'':10s}{'estimate':>{_W}}{'SE':>{_W}}"
                 f"{'95% CI':>{2 * _W + 3}}")
        for j in range(d):
            lo, hi = theta[0, j] - 1.96 * se[0, j], theta[0, j] + 1.96 * se[0, j]
            ci = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "—"
            L.append(f"  theta[{j}]{'':4s}{_fmt(theta[0, j])}"
                     f"{_fmt(se[0, j])}{ci:>{2 * _W + 3}}")
    else:
        L.append("Parameters (theta, unconstrained space) — across subjects")
        L.append(f"  {'':10s}{'mean':>{_W}}{'SD':>{_W}}{'min':>{_W}}"
                 f"{'max':>{_W}}{'mean SE':>{_W}}")
        for j in range(d):
            col = theta[:, j]
            ok = np.isfinite(col)
            L.append(
                f"  theta[{j}]{'':4s}"
                f"{_fmt(np.mean(col[ok]) if ok.any() else np.nan)}"
                f"{_fmt(np.std(col[ok]) if ok.any() else np.nan)}"
                f"{_fmt(np.min(col[ok]) if ok.any() else np.nan)}"
                f"{_fmt(np.max(col[ok]) if ok.any() else np.nan)}"
                f"{_fmt(np.nanmean(se[:, j]) if np.isfinite(se[:, j]).any() else np.nan)}")
    L.append("")
    L.append("  theta is the space the optimizer works in. Convert to your")
    L.append("  model's native parameters yourself (e.g. sigmoid / exp).")
    L.append("")

    # ---- prior (Mod 15) -------------------------------------------------
    # Stated always, not only when defaulted: the prior moves both the
    # estimates and the evidence, so a summary that omits it is
    # incomplete as a record of how the fit was produced.
    pm, pv, pdef = prior_spec(result)
    if pm.size:
        same = (np.allclose(pm, pm[0]) and
                (np.allclose(pv, pv[0]) if np.isfinite(pv).all() else False))
        tag = ("   [DEFAULT: " + ", ".join(pdef) + "]") if pdef else ""
        if same:
            L.append(f"Prior          N({pm[0]:g}, {pv[0]:g}) on all "
                     f"{pm.size} parameters{tag}")
        else:
            L.append(f"Prior{tag}")
            for j in range(pm.size):
                L.append(f"  theta[{j}]{'':4s}N({pm[j]:g}, {pv[j]:g})")
        if pdef:
            L.append("  a weakly informative default, not an absence of")
            L.append("  assumption — pass prior_mean/prior_variance to choose")
        L.append("")

    # ---- evidence -------------------------------------------------------
    fin = lme[np.isfinite(lme)]
    if fin.size:
        if n == 1:
            L.append(f"Log-evidence   {fin[0]:.3f}")
        else:
            L.append(f"Log-evidence   mean {fin.mean():.3f}   "
                     f"sum {fin.sum():.3f}   "
                     f"range [{fin.min():.3f}, {fin.max():.3f}]")
        L.append("  Laplace approximation; comparable across models fitted")
        L.append("  to the SAME data by the same method.")
    else:
        L.append("Log-evidence   —  (no finite values)")
    L.append("")

    # ---- diagnostics ----------------------------------------------------
    q = [r["quality"] for r in recs]
    n_ok = sum(1 for x in q if x == "ok")
    L.append("Fit quality")
    if all(x == "-" for x in q):
        L.append("  no per-fit diagnostics available for this result")
    else:
        L.append(f"  {n_ok}/{n} clean")
        counts: Dict[str, int] = {}
        for x in q:
            if x in ("ok", "-"):
                continue
            for part in x.split(","):
                counts[part] = counts.get(part, 0) + 1
        label = {"weak": "weakly identified (Mod 10)",
                 "multimodal": "multimodal — inits disagreed",
                 "bounds": "parameter at a hard bound",
                 "singular": "singular Hessian at the optimum",
                 "prior": "fit failed, prior substituted"}
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            L.append(f"  {v:>3d}  {label.get(k, k)}")
        if counts:
            L.append("  these are triage signals, not rejections — see")
            L.append("  fit.math.diagnostics[i] for the full record")
    L.append("")

    # ---- per-subject rows ----------------------------------------------
    # Skipped for a single subject: the parameter block above already shows
    # everything, and a one-row table would just repeat it.
    if n > 1:
        show = min(n, max_subjects)
        L.append(f"Per subject (first {show} of {n})" if show < n
                 else "Per subject")
        hdr = f"  {'subj':>4s}"
        for j in range(d):
            hdr += f"{f'theta[{j}]':>{_W}}"
        hdr += f"{'log_eV':>{_W}}  quality"
        L.append(hdr)
        for r in recs[:show]:
            line = f"  {r['subject']:>4d}"
            for j in range(d):
                line += _fmt(r[f"theta[{j}]"])
            line += _fmt(r["log_evidence"], prec=2) + "  " + r["quality"]
            L.append(line)
        if show < n:
            L.append(f"  … {n - show} more — use fit.table() for all rows")
        L.append("")

    L.append("fit.table() for a per-subject table · "
             "fit.math for hessians and gradients")
    return "\n".join(L)


# ===========================================================================
# HBI results
# ===========================================================================
#
# The quantities a reader wants from a hierarchical fit are different from
# an individual one. There is no single "the estimate" — there is a group
# mean per model, a population frequency per model, and a per-subject
# assignment. So this is its own layout rather than a reuse of the fit one.
#
# On the three model-level probabilities, which are easy to confuse:
#   model_frequency  E[r_k]  — expected share of the population using k
#   exceedance_prob  P(r_k > r_j for all j) — confidence that k is the
#                    single most common model
#   protected xp     the same, discounted by the Bayes Omnibus Risk, i.e.
#                    allowing for the possibility that no model is more
#                    common than any other. Produced by `hbi_null`, so it
#                    is NaN until you run it — the summary says so rather
#                    than printing a bare NaN.
# ===========================================================================

def _model_names(result, K: int) -> List[str]:
    """Names for the K models, from whatever the result carries."""
    inp = getattr(result, "input", None)
    models = getattr(inp, "models", None) if inp is not None else None
    out = []
    for k in range(K):
        nm = None
        if models is not None and k < len(models):
            nm = getattr(models[k], "__name__", None)
        out.append(nm or f"model {k + 1}")
    return out


def hbi_rows(result) -> List[Dict[str, Any]]:
    """One record per model: frequency, exceedance, attributed subjects."""
    o = result.output
    freq = np.ravel(np.asarray(o.model_frequency, dtype=float))
    K = freq.size
    xp = np.ravel(np.asarray(o.exceedance_prob, dtype=float))
    pxp = np.ravel(np.asarray(
        getattr(o, "protected_exceedance_prob", np.full(K, np.nan)),
        dtype=float))
    resp = np.asarray(o.responsibility, dtype=float)
    # responsibility is (K, N) in some paths and (N, K) in others; normalise
    # to (N, K) by matching the model count rather than guessing.
    if resp.ndim == 2 and resp.shape[0] == K and resp.shape[1] != K:
        resp = resp.T
    names = _model_names(result, K)

    recs = []
    for k in range(K):
        r: Dict[str, Any] = {"model": names[k]}
        r["frequency"] = float(freq[k])
        r["exceedance_prob"] = float(xp[k]) if k < xp.size else np.nan
        r["protected_xp"] = float(pxp[k]) if k < pxp.size else np.nan
        if resp.ndim == 2 and resp.shape[1] == K:
            r["attributed_subjects"] = float(resp[:, k].sum())
            r["n_best"] = int((np.argmax(resp, axis=1) == k).sum())
        else:
            r["attributed_subjects"] = np.nan
            r["n_best"] = -1
        gm = getattr(o, "group_mean", None)
        if gm is not None and k < len(gm):
            r["group_mean"] = np.ravel(np.asarray(gm[k], float)).tolist()
        recs.append(r)
    return recs


def hbi_table(result, pandas: bool = True):
    """Model-level table: one row per candidate model."""
    recs = hbi_rows(result)
    if not pandas:
        return recs
    return pd.DataFrame(recs).set_index("model")


def hbi_subject_table(result, pandas: bool = True):
    """Subject-level table: responsibilities and the best-fitting model.

    One row per subject, one `p(model)` column per candidate, plus the
    argmax and its probability — which is what most people actually want
    when they ask "who used which model?".
    """
    o = result.output
    K = np.ravel(np.asarray(o.model_frequency, dtype=float)).size
    resp = np.asarray(o.responsibility, dtype=float)
    if resp.ndim == 2 and resp.shape[0] == K and resp.shape[1] != K:
        resp = resp.T
    names = _model_names(result, K)
    params = getattr(o, "parameters", None)

    recs = []
    for i in range(resp.shape[0]):
        r: Dict[str, Any] = {"subject": i}
        for k in range(K):
            r[f"p({names[k]})"] = float(resp[i, k])
        best = int(np.argmax(resp[i]))
        r["best_model"] = names[best]
        r["p_best"] = float(resp[i, best])
        # Parameters under the subject's own best-fitting model — the
        # sensible default when reporting one number per subject.
        if params is not None and best < len(params):
            pk = np.asarray(params[best], dtype=float)
            if pk.ndim == 2 and i < pk.shape[0]:
                for j, v in enumerate(pk[i]):
                    r[f"theta[{j}]"] = float(v)
        recs.append(r)
    if not pandas:
        return recs
    return pd.DataFrame(recs).set_index("subject")


def hbi_summary(result, max_models: int = 12) -> str:
    """Compact, copy-pasteable summary of a hierarchical fit."""
    recs = hbi_rows(result)
    K = len(recs)
    resp = np.asarray(result.output.responsibility, dtype=float)
    if resp.ndim == 2 and resp.shape[0] == K and resp.shape[1] != K:
        resp = resp.T
    N = resp.shape[0] if resp.ndim == 2 else 0

    prof = getattr(result, "profile", None)
    method = getattr(result, "method", "HBI")
    L = []
    head = f"{method} · {K} models · {N} subjects"
    L.append(head)
    L.append("=" * max(len(head), 64))
    bits = []
    dt = getattr(prof, "datetime", None)
    if dt:
        bits.append(str(dt))
    if bits:
        L.append("   ·   ".join(bits))
    L.append("")

    name_w = max(12, min(28, max(len(r["model"]) for r in recs) + 2))
    L.append("Model comparison")
    L.append(f"  {'model':<{name_w}}{'freq':>9}{'xp':>9}{'pxp':>9}"
             f"{'best fit':>10}")
    for r in recs[:max_models]:
        pxp = r["protected_xp"]
        pxp_s = f"{pxp:>9.4f}" if np.isfinite(pxp) else f"{'—':>9}"
        nb = f"{r['n_best']:>10d}" if r["n_best"] >= 0 else f"{'—':>10}"
        L.append(f"  {r['model']:<{name_w}}{r['frequency']:>9.4f}"
                 f"{r['exceedance_prob']:>9.4f}{pxp_s}{nb}")
    if K > max_models:
        L.append(f"  … {K - max_models} more — use result.table()")
    L.append("")
    L.append("  freq = expected share of the population using that model")
    L.append("  xp   = P(this model is the most common one)")
    if not np.isfinite(recs[0]["protected_xp"]):
        L.append("  pxp  = not computed — run hbi_null() for protected xp")
    else:
        L.append("  pxp  = xp discounted by the Bayes Omnibus Risk")
    L.append("")

    # Winner, stated plainly, with the caveat attached rather than implied.
    best = max(recs, key=lambda r: r["frequency"])
    L.append(f"Most frequent   {best['model']}  "
             f"(freq {best['frequency']:.3f}, xp {best['exceedance_prob']:.3f})")
    if best["exceedance_prob"] < 0.95:
        L.append("  xp < 0.95 — the evidence does not clearly single out one")
        L.append("  model; treat this as a ranking, not a decision.")
    L.append("")

    # ---- MOD 12 diagnostics, when present ------------------------------
    dg = getattr(getattr(result, "math", None), "qhquad", None)
    dg = getattr(dg, "diagnostics", None) if dg is not None else None
    L.append("Fit quality")
    if dg is None:
        L.append("  no per-refit diagnostics on this result")
        L.append("  (pass model_trials= to hbi_main for Gauss-Newton refits")
        L.append("   and the Mod 9/10 records — see DEV.md §14)")
    else:
        flat = [x for x in np.asarray(dg, dtype=object).ravel()
                if x is not None]
        n_slots = np.asarray(dg, dtype=object).size
        weak = [getattr(x, "weak_identifiability", None) for x in flat]
        weak = [v for v in weak if v is not None]
        n_weak = sum(1 for v in weak if v < 2.0)
        methods = {getattr(x, "hess_method", None) for x in flat}
        L.append(f"  {len(flat)}/{n_slots} refits recorded   ·   "
                 f"curvature: {'/'.join(sorted(m for m in methods if m))}")
        if n_weak:
            L.append(f"  {n_weak} weakly identified (Mod 10) — see "
                     f"result.math.qhquad.diagnostics")
        if n_slots - len(flat):
            L.append(f"  {n_slots - len(flat)} refits fell back to the prior")
    L.append("")
    L.append("result.table() per model · result.subject_table() per subject")
    return "\n".join(L)


# ===========================================================================
# group_bms results
# ===========================================================================

def bms_rows(result, model_names: Optional[List[str]] = None
             ) -> List[Dict[str, Any]]:
    """One record per model for a GroupBMSResult."""
    freq = np.ravel(np.asarray(result.model_frequency, dtype=float))
    K = freq.size
    xp = np.ravel(np.asarray(result.exceedance_prob, dtype=float))
    pxp = np.ravel(np.asarray(result.protected_exceedance_prob, dtype=float))
    alpha = np.ravel(np.asarray(result.posterior_parameters, dtype=float))
    g = getattr(result, "g", None)
    names = model_names or [f"model {k + 1}" for k in range(K)]

    recs = []
    for k in range(K):
        r: Dict[str, Any] = {"model": names[k] if k < len(names)
                             else f"model {k + 1}"}
        r["frequency"] = float(freq[k])
        r["exceedance_prob"] = float(xp[k]) if k < xp.size else np.nan
        r["protected_xp"] = float(pxp[k]) if k < pxp.size else np.nan
        r["alpha"] = float(alpha[k]) if k < alpha.size else np.nan
        if g is not None:
            ga = np.asarray(g, dtype=float)
            if ga.ndim == 2 and ga.shape[1] == K:
                r["attributed_subjects"] = float(ga[:, k].sum())
                r["n_best"] = int((np.argmax(ga, axis=1) == k).sum())
        recs.append(r)
    return recs


def bms_table(result, model_names: Optional[List[str]] = None,
              pandas: bool = True):
    recs = bms_rows(result, model_names)
    if not pandas:
        return recs
    return pd.DataFrame(recs).set_index("model")


def bms_summary(result, model_names: Optional[List[str]] = None) -> str:
    """Summary for a GroupBMSResult (random-effects model comparison)."""
    recs = bms_rows(result, model_names)
    K = len(recs)
    bor = float(getattr(result, "bor", np.nan))

    L = []
    head = f"Group BMS · {K} models"
    L.append(head)
    L.append("=" * max(len(head), 64))
    L.append("")
    name_w = max(12, min(28, max(len(r["model"]) for r in recs) + 2))
    has_g = "n_best" in recs[0]
    L.append(f"  {'model':<{name_w}}{'freq':>9}{'xp':>9}{'pxp':>9}"
             + (f"{'best fit':>10}" if has_g else ""))
    for r in recs:
        line = (f"  {r['model']:<{name_w}}{r['frequency']:>9.4f}"
                f"{r['exceedance_prob']:>9.4f}{r['protected_xp']:>9.4f}")
        if has_g:
            line += f"{r['n_best']:>10d}"
        L.append(line)
    L.append("")

    best = max(recs, key=lambda r: r["frequency"])
    L.append(f"Most frequent   {best['model']}  "
             f"(freq {best['frequency']:.3f}, xp {best['exceedance_prob']:.3f})")
    L.append("")

    # BOR is the honest caveat on any BMS result, so it is not optional
    # reading — it gets its own line with an interpretation.
    L.append(f"Bayes Omnibus Risk   {bor:.4f}")
    if np.isfinite(bor):
        if bor > 0.25:
            L.append("  HIGH — the data may be equally well explained by all")
            L.append("  models having the same frequency. Prefer pxp over xp.")
        else:
            L.append("  low — the models do differ in population frequency")
    L.append("")
    L.append("  freq = expected population share · xp = P(most common)")
    L.append("  pxp  = xp corrected for the BOR (Rigoux et al. Eq. 7)")
    L.append("")
    L.append("result.table() for a per-model table")
    return "\n".join(L)


def btw_conds_summary(result) -> str:
    """Summary for a between-conditions BMS."""
    L = []
    L.append("Between-conditions BMS")
    L.append("=" * 64)
    L.append("")
    L.append(f"  P(same model in every condition)             "
             f"{float(result.xp):.4f}")
    L.append(f"  … protected for the omnibus risk             "
             f"{float(result.pxp):.4f}")
    L.append(f"  Bayes Omnibus Risk (family level)            "
             f"{float(result.bor):.4f}")
    L.append("")
    L.append(f"  {int(result.n_equal)} of {int(result.n_tuples)} tuples "
             f"have the same model in every slot")
    L.append("")
    pxp = float(result.pxp)
    if pxp > 0.95:
        L.append("  Strong evidence that ONE model holds across conditions.")
    elif pxp < 0.05:
        L.append("  Strong evidence that the best model DIFFERS by condition.")
    else:
        L.append("  Inconclusive — neither hypothesis is clearly supported.")
    return "\n".join(L)


def btw_groups_summary(result) -> str:
    """Summary for a between-groups BMS (free-energy test of H0)."""
    L = []
    L.append("Between-groups BMS")
    L.append("=" * 64)
    L.append("")
    L.append(f"  F(one shared profile)      {float(result.F_equal):>12.3f}")
    L.append(f"  F(per-group profiles)      {float(result.F_diff):>12.3f}")
    L.append(f"  difference                 "
             f"{float(result.F_diff) - float(result.F_equal):>12.3f}")
    L.append("")
    L.append(f"  P(groups share one profile)   {float(result.p_equal):.4f}")
    L.append("")
    if bool(getattr(result, "h_reject_equality", False)):
        L.append("  REJECT equality (p < 0.05): the groups differ in which")
        L.append("  models their subjects use.")
    else:
        L.append("  Cannot reject equality: no evidence the groups differ in")
        L.append("  model usage. (Absence of evidence, not evidence of")
        L.append("  absence — a small sample rejects nothing.)")
    return "\n".join(L)
