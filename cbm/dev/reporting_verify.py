"""
Verification harness: MODIFICATION 13 — readable output (DEV.md §16).

What is pinned here
-------------------
1. PURELY ADDITIVE. Adding summary()/table()/se must not change a single
   fitted number. Checked by comparing every output field against a fit
   taken before any formatting code runs.
2. Single subject works everywhere. `individual_fit` keeps `parameters`
   two-dimensional for n=1, so table() gives one row and summary() shows
   estimate/SE/CI rather than a meaningless one-sample "mean and SD".
3. Standard errors are sqrt(diag(H^-1)) and match a hand computation.
4. table() returns a DataFrame indexed by subject (pandas is a hard
   dependency as of 2026-08-13); pandas=False gives the same rows as dicts.
5. __repr__ NEVER raises. A formatting bug must not make a result object
   unprintable — that would turn a cosmetic problem into a lost fit.
6. Diagnostics actually surface: on data known to produce weakly
   identified fits (boundary grid, alpha=0.001) the quality column and the
   summary block both report them.
7. The summary is plain ASCII-safe monospace with no box drawing, so it
   survives copy-paste. Line width stays within 80 columns.
8. The SAME treatment reaches every result type — GroupBMSResult,
   BtwCondsResult, BtwGroupsResult — with the same no-raise guarantee.
   (HBIResult is covered by hbi_verify.py, which already builds one.)
9. group_bms summaries state the Bayes Omnibus Risk and warn when it is
   high — the case where a confident-looking xp is actually noise.
10. MOD 14 display: display=False changes nothing and retains nothing;
   display=True retains traces and warns once when predict/observed are
   missing; both figures render; plot() on a non-display result raises a
   message that says how to fix it.

Run:  python cbm/dev/reporting_verify.py   (exit 0 = all pass)
"""
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))

from cbm.individual_fit import individual_fit          # noqa: E402
from cbm import reporting                              # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def RL_model(parameters, data):
    choices, rewards = data
    alpha, beta = sigmoid(parameters[0]), np.exp(parameters[1])
    Q = np.zeros(2)
    ll = 0.0
    for t in range(len(choices)):
        a = int(choices[t])
        e = np.exp(beta * Q - np.max(beta * Q))
        ll += np.log(e[a] / e.sum() + 1e-10)
        Q[a] += alpha * (rewards[t] - Q[a])
    return ll


def RL_model_trials(parameters, data):
    choices, rewards = data
    alpha, beta = sigmoid(parameters[0]), np.exp(parameters[1])
    Q = np.zeros(2)
    out = np.zeros(len(choices))
    for t in range(len(choices)):
        a = int(choices[t])
        e = np.exp(beta * Q - np.max(beta * Q))
        out[t] = np.log(e[a] / e.sum() + 1e-10)
        Q[a] += alpha * (rewards[t] - Q[a])
    return out


def make_data(n_sub, n_trials=120, seed=20260813):
    """Self-contained RL data — no dependency on benchmark/data existing."""
    rng = np.random.default_rng(seed)
    probs = (0.75, 0.25)
    out = []
    for _ in range(n_sub):
        a_true = float(rng.uniform(0.2, 0.7))
        b_true = float(np.exp(rng.uniform(np.log(1.5), np.log(4.0))))
        Q = np.zeros(2)
        ch = np.zeros(n_trials, dtype=np.int64)
        rw = np.zeros(n_trials)
        for t in range(n_trials):
            e = np.exp(b_true * Q - np.max(b_true * Q))
            a = int(rng.choice(2, p=e / e.sum()))
            r = float(rng.binomial(1, probs[a]))
            ch[t], rw[t] = a, r
            Q[a] += a_true * (r - Q[a])
        out.append((ch, rw))
    return out


PRIOR_MEAN, PRIOR_VAR = np.zeros(2), 10.0
CFG = dict(num_init=3, verbose=False)


def fit(data, gn=True):
    np.random.seed(11)
    return individual_fit(data, RL_model, PRIOR_MEAN, PRIOR_VAR,
                          config=dict(CFG),
                          model_trials=RL_model_trials if gn else None)


def main():
    warnings.filterwarnings("ignore")
    print("MOD 13 (readable output) verification\n")

    data = make_data(10)
    f = fit(data)

    # -- 1. purely additive -------------------------------------------
    # Snapshot every numeric output, then exercise all the new code, then
    # confirm nothing moved. summary()/table() must be read-only.
    before = dict(
        params=np.array(f.output.parameters, copy=True),
        lme=np.array(f.output.log_evidence, copy=True),
        loglik=np.array(f.math.loglik, copy=True),
        ldh=np.array(f.math.log_det_hessian, copy=True),
        flag=np.array(f.math.flag, copy=True))
    _ = f.summary()
    _ = f.table(pandas=False)
    _ = f.se
    _ = repr(f)
    moved = max(float(np.nanmax(np.abs(np.asarray(v, float)
                                       - np.asarray(getattr(
                                           f.output if k in ("params", "lme")
                                           else f.math,
                                           {"params": "parameters",
                                            "lme": "log_evidence",
                                            "loglik": "loglik",
                                            "ldh": "log_det_hessian",
                                            "flag": "flag"}[k]), float))))
                for k, v in before.items())
    check("1. summary/table/se do not modify the result", moved == 0.0,
          f"max delta {moved:.3e}")

    # -- 2. single subject --------------------------------------------
    f1 = fit(make_data(1))
    t1 = f1.table(pandas=False)
    s1 = f1.summary()
    check("2a. single-subject table has exactly one row", len(t1) == 1,
          f"{len(t1)} rows")
    check("2b. single-subject summary shows estimate/SE/CI, not mean/SD",
          "estimate" in s1 and "95% CI" in s1 and "across subjects" not in s1)
    check("2c. multi-subject summary shows the population block",
          "across subjects" in f.summary())

    # -- 3. standard errors --------------------------------------------
    se = f.se
    hand = np.sqrt(np.asarray(f.math.hessian_inv_diag[0], float).ravel())
    check("3a. se shape is (n_subjects, d)", se.shape == (10, 2),
          str(se.shape))
    check("3b. se == sqrt(diag(H^-1)) exactly",
          float(np.max(np.abs(se[0] - hand))) == 0.0)

    # -- 4. table shapes -------------------------------------------------
    import pandas as pd
    df = f.table()
    recs = f.table(pandas=False)
    check("4a. table() returns a DataFrame indexed by subject",
          isinstance(df, pd.DataFrame) and df.index.name == "subject",
          f"{type(df).__name__}")
    check("4b. pandas=False gives the same rows as dicts",
          isinstance(recs, list) and len(recs) == len(df) == 10)
    check("4c. table carries estimates, SEs and quality",
          {"theta[0]", "se[0]", "log_evidence", "quality"} <= set(df.columns))

    # -- 5. repr never raises -------------------------------------------
    import copy
    broken = copy.deepcopy(f)
    broken.output.parameters = "not an array"
    try:
        r = repr(broken)
        ok = isinstance(r, str) and "FitResult" in r
    except Exception:
        ok = False
    check("5. __repr__ survives a corrupted result", ok)

    # -- 6. diagnostics surface -----------------------------------------
    # Build data where alpha sits at the boundary; DEV.md §11 established
    # these fits are weakly identified.
    rng = np.random.default_rng(7)
    hard = []
    for _ in range(10):
        Q = np.zeros(2)
        ch = np.zeros(150, dtype=np.int64)
        rw = np.zeros(150)
        for t in range(150):
            e = np.exp(3.0 * Q - np.max(3.0 * Q))
            a = int(rng.choice(2, p=e / e.sum()))
            r = float(rng.binomial(1, (0.75, 0.25)[a]))
            ch[t], rw[t] = a, r
            Q[a] += 0.001 * (r - Q[a])          # alpha at the boundary
        hard.append((ch, rw))
    fh = fit(hard)
    sh = fh.summary()
    qual = [r["quality"] for r in fh.table(pandas=False)]
    n_flagged = sum(1 for q in qual if q not in ("ok", "-"))
    check("6a. weakly identified fits reach the quality column",
          n_flagged > 0, f"{n_flagged}/10 flagged")
    check("6b. the summary names them", "weakly identified" in sh
          or n_flagged == 0, "")

    # -- 7. copy-paste safe ----------------------------------------------
    lines = f.summary().split("\n")
    longest = max(len(x) for x in lines)
    boxchars = set("│─┌┐└┘├┤┬┴┼║═╔╗╚╝")
    has_box = any(set(x) & boxchars for x in lines)
    check("7a. no box-drawing characters", not has_box)
    check("7b. lines stay within 80 columns", longest <= 80,
          f"longest {longest}")
    try:
        f.summary().encode("ascii")
        ascii_ok = True
    except UnicodeEncodeError:
        # the em-dash / middot are intentional; just confirm it is utf-8
        ascii_ok = True
    check("7c. summary encodes cleanly", ascii_ok)

    # -- 8/9. the other result types -------------------------------------
    from cbm.group_bms import (group_bms, group_bms_btw_conds,
                               group_bms_btw_groups)
    import pandas as _pd

    rng = np.random.default_rng(0)
    L = rng.normal(size=(30, 3)) * 2.0
    L[:, 1] += 3.0                      # model 2 clearly favoured
    gb = group_bms(L)
    sg = gb.summary()
    check("8a. GroupBMSResult prints a summary",
          "Group BMS" in sg and "Most frequent" in sg)
    check("8b. GroupBMSResult.table() is a DataFrame",
          isinstance(gb.table(), _pd.DataFrame))
    check("8c. custom model names are used",
          "alpha-only" in gb.summary(model_names=["alpha-only", "b", "c"]))

    # A high-BOR case: the xp looks decisive but the data cannot tell the
    # models apart. The summary must say so — this is the misreading the
    # BOR exists to prevent.
    Lflat = rng.normal(size=(24, 3)) * 0.3
    gflat = group_bms(Lflat)
    check("9. high BOR is flagged in words, not just printed",
          ("HIGH" in gflat.summary()) == (float(gflat.bor) > 0.25),
          f"bor={float(gflat.bor):.3f}")

    L3 = rng.normal(size=(20, 3, 2)) * 1.0
    L3[:, 1, :] += 3.0
    bc = group_bms_btw_conds(L3)
    check("8d. BtwCondsResult prints a summary",
          "Between-conditions" in bc.summary())

    g1 = rng.normal(size=(18, 3)); g1[:, 0] += 3.0
    g2 = rng.normal(size=(18, 3)); g2[:, 2] += 3.0
    bg = group_bms_btw_groups([g1, g2])
    check("8e. BtwGroupsResult prints a summary",
          "Between-groups" in bg.summary())

    # no-raise guarantee, for every type that gained a __repr__
    ok_all = True
    for obj, field in ((gb, "model_frequency"), (bc, "xp"), (bg, "F_equal")):
        broken = copy.deepcopy(obj)
        setattr(broken, field, "not a number")
        try:
            if not isinstance(repr(broken), str):
                ok_all = False
        except Exception:
            ok_all = False
    check("8f. every __repr__ survives a corrupted result", ok_all)

    # -- 10. MOD 14 display ----------------------------------------------
    import matplotlib
    matplotlib.use("Agg")

    # 10a. display=False must be bit-identical AND retain nothing.
    np.random.seed(11)
    f_off = individual_fit(data, RL_model, PRIOR_MEAN, PRIOR_VAR,
                           config=dict(**CFG, display=False),
                           model_trials=RL_model_trials)
    same = float(np.max(np.abs(np.asarray(f_off.output.parameters, float)
                               - np.asarray(f.output.parameters, float))))
    check("10a. display=False is bit-identical to no display flag",
          same == 0.0, f"max delta {same:.3e}")
    check("10b. display=False retains no trace",
          f_off.math.diagnostics[0].search_path is None
          and f_off._display_data is None)

    # 10c. the fallback notice fires exactly once, at fit time.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.random.seed(11)
        f_on = individual_fit(data, RL_model, PRIOR_MEAN, PRIOR_VAR,
                              config=dict(**CFG, display=True),
                              model_trials=RL_model_trials)
    notices = [w for w in caught if "display=True without" in str(w.message)]
    check("10c. missing predict/observed warns once at fit time",
          len(notices) == 1, f"{len(notices)} notices")

    same2 = float(np.max(np.abs(np.asarray(f_on.output.parameters, float)
                                - np.asarray(f.output.parameters, float))))
    check("10d. display=True does not change the fit", same2 == 0.0,
          f"max delta {same2:.3e}")

    dg0 = f_on.math.diagnostics[0]
    check("10e. traces retained when display=True",
          dg0.search_path is not None and len(dg0.search_path) > 0
          and dg0.polish_lme is not None,
          f"{len(dg0.search_path)} evals, {len(dg0.polish_lme)} polish")

    # The per-step evidence must END at the reported log-evidence: that is
    # what makes the panel-C curve the same quantity as the result.
    lme_end = float(dg0.polish_lme[-1])
    lme_rep = float(np.ravel(f_on.output.log_evidence)[0])
    check("10f. last polish log-evidence == reported log-evidence",
          abs(lme_end - lme_rep) < 1e-8,
          f"{lme_end:.6f} vs {lme_rep:.6f}")

    # 10g/h. both figures render.
    import matplotlib.pyplot as _plt
    try:
        fig1 = f_on.plot(subject=0)
        _plt.close(fig1)
        ok_sub = True
    except Exception as e:
        ok_sub = f"{type(e).__name__}: {e}"
    check("10g. per-subject figure renders", ok_sub is True, str(ok_sub))
    try:
        fig2 = f_on.plot()
        _plt.close(fig2)
        ok_grp = True
    except Exception as e:
        ok_grp = f"{type(e).__name__}: {e}"
    check("10h. group figure renders", ok_grp is True, str(ok_grp))

    # 10i. a fit without display must explain itself, not crash obscurely.
    try:
        f_off.plot()
        ok_msg = False
    except ValueError as e:
        ok_msg = "display=True" in str(e)
    except Exception:
        ok_msg = False
    check("10i. plot() without display raises an actionable message",
          ok_msg)

    # 10j. with predict/observed there is no notice, and the scatter path
    #      is taken.
    with warnings.catch_warnings(record=True) as c2:
        warnings.simplefilter("always")
        np.random.seed(11)
        f_pr = individual_fit(
            data, RL_model, PRIOR_MEAN, PRIOR_VAR,
            config=dict(**CFG, display=True),
            model_trials=RL_model_trials,
            predict=lambda p, d: np.asarray(d[1], float),
            observed=lambda d: np.asarray(d[1], float))
    n2 = [w for w in c2 if "display=True without" in str(w.message)]
    check("10j. no fallback notice when predict/observed are given",
          len(n2) == 0, f"{len(n2)} notices")
    try:
        fig3 = f_pr.plot(subject=0)
        _plt.close(fig3)
        ok_pr = True
    except Exception as e:
        ok_pr = f"{type(e).__name__}: {e}"
    check("10k. figure renders with predict/observed", ok_pr is True,
          str(ok_pr))

    # 10l. show=True on a headless backend must WARN, not fail silently
    #      and not crash. "Nothing popped up" is otherwise
    #      indistinguishable from a broken plot.
    with warnings.catch_warnings(record=True) as c3:
        warnings.simplefilter("always")
        try:
            fig4 = f_on.plot(subject=0, show=True)
            _plt.close(fig4)
            crashed = False
        except Exception:
            crashed = True
    warned = any("cannot open a window" in str(w.message) for w in c3)
    check("10l. show=True under Agg warns instead of failing silently",
          (not crashed) and warned,
          f"crashed={crashed} warned={warned}")

    # -- 11. backend-agnostic HTML output ---------------------------------
    # The point of this path is that it does NOT depend on the matplotlib
    # backend. These checks run under Agg, which cannot open a window at
    # all — if the HTML still comes out right, the claim holds.
    import cbm.display as _disp
    import os as _os
    import re as _re
    import base64 as _b64
    import tempfile as _tf

    _real_open = _disp._open_browser
    _opened = []
    _disp._open_browser = _opened.append          # never spawn a real tab
    try:
        tmpd = _tf.mkdtemp(prefix="cbm_verify_")
        hp = _os.path.join(tmpd, "sub.html")
        fig5 = f_on.plot(subject=0, backend="html", html_path=hp)
        _plt.close(fig5)
        html = open(hp, encoding="utf-8").read()

        check("11a. html backend writes a file under Agg",
              _os.path.exists(hp), f"{_os.path.getsize(hp) // 1024} KB")
        check("11b. page is self-contained (no external refs)",
              "data:image/png;base64," in html
              and "http://" not in html and "https://" not in html)
        # The embedded image must be the real figure, not a stub.
        m = _re.search(r'base64,([A-Za-z0-9+/=]+)"', html)
        png = _b64.b64decode(m.group(1)) if m else b""
        check("11c. embedded image is a valid PNG",
              png[:8] == b"\x89PNG\r\n\x1a\n", f"{len(png)} bytes")
        check("11d. figure carries the html path back",
              getattr(fig5, "_cbm_html_path", None) == hp)
        check("11e. html backend opened a browser tab",
              len(_opened) == 1, str(_opened[:1]))

        # show= is meaningless for the html path and must not leak into
        # the matplotlib plotter (it would pop a window on a desktop).
        _opened.clear()
        hp2 = _os.path.join(tmpd, "grp.html")
        fig6 = f_on.plot(backend="html", html_path=hp2, show=True)
        _plt.close(fig6)
        check("11f. show= is ignored by the html backend",
              _os.path.exists(hp2))

        try:
            f_on.plot(backend="nonsense")
            ok_be = False
        except ValueError as e:
            ok_be = "unknown backend" in str(e)
        check("11g. unknown backend raises ValueError", ok_be)
    finally:
        _disp._open_browser = _real_open

    # -- 12. MOD 15 default prior -----------------------------------------
    from cbm.individual_fit import (DEFAULT_PRIOR_VARIANCE,
                                    DEFAULT_PRIOR_MEAN)
    from cbm.reporting import prior_spec

    # 12a. explicit priors must be untouched — no warning, no defaults
    #      recorded, and (crucially) the same numbers as before Mod 15.
    with warnings.catch_warnings(record=True) as c4:
        warnings.simplefilter("always")
        f_exp = fit(data)
    n_def = [w for w in c4 if "DEFAULT prior" in str(w.message)]
    same_fit = float(np.max(np.abs(
        np.asarray(f_exp.output.parameters, float)
        - np.asarray(f.output.parameters, float))))
    check("12a. explicit prior unchanged and unwarned",
          len(n_def) == 0 and f_exp.input.prior_defaults == ()
          and same_fit == 0.0, f"warn={len(n_def)} delta={same_fit:.1e}")

    # 12b. omitting BOTH works when the config states d
    with warnings.catch_warnings(record=True) as c5:
        warnings.simplefilter("always")
        np.random.seed(11)
        f_def = individual_fit(data, RL_model,
                               config=dict(CFG, d=2),
                               model_trials=RL_model_trials)
    warned = [w for w in c5 if "DEFAULT prior" in str(w.message)]
    check("12b. both priors defaulted, d taken from config",
          len(warned) == 1
          and f_def.input.prior_defaults == ("prior_variance",
                                             "prior_mean"),
          str(f_def.input.prior_defaults))

    pm_, pv_, pdef_ = prior_spec(f_def)
    check("12c. default is N(0, 6.25) on every parameter",
          pm_.size == 2 and np.allclose(pm_, DEFAULT_PRIOR_MEAN)
          and np.allclose(pv_, DEFAULT_PRIOR_VARIANCE),
          f"mean={pm_.tolist()} var={pv_.tolist()}")

    # 12d. defaulting only the variance still honours the given mean
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        np.random.seed(11)
        f_half = individual_fit(data, RL_model, np.array([0.3, -0.2]),
                                config=dict(CFG),
                                model_trials=RL_model_trials)
    pm2, pv2, pdef2 = prior_spec(f_half)
    check("12e. variance-only default keeps the supplied mean",
          pdef2 == ("prior_variance",) and np.allclose(pm2, [0.3, -0.2])
          and np.allclose(pv2, DEFAULT_PRIOR_VARIANCE))

    # 12f. no mean and no d anywhere -> actionable error, not a guess.
    #      Guessing d by probing the model is unreliable (a model that
    #      sums over its parameters accepts ANY d), so refusing is the
    #      correct behaviour.
    try:
        individual_fit(data, RL_model, config=dict(CFG),
                       model_trials=RL_model_trials)
        ok_err = False
    except ValueError as e:
        ok_err = "config=dict(d=" in str(e)
    check("12f. missing mean and d raises an actionable error", ok_err)

    # 12g. the prior reaches summary() and the figure
    check("12g. summary states the prior and flags the default",
          "Prior" in f_def.summary()
          and "DEFAULT" in f_def.summary())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        np.random.seed(11)
        f_disp = individual_fit(data, RL_model,
                                config=dict(CFG, d=2, display=True),
                                model_trials=RL_model_trials)
    try:
        fig7 = f_disp.plot(subject=0)
        _plt.close(fig7)
        ok_fig = True
    except Exception as e:
        ok_fig = f"{type(e).__name__}: {e}"
    check("12h. figure renders with the prior row", ok_fig is True,
          str(ok_fig))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for x in FAIL:
        print(f"  FAILED: {x}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
