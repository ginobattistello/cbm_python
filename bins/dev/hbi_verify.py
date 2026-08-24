"""
Verification harness: MODIFICATION 11 and 12 (DEV.md §13, §14).

Why this file exists
--------------------
Until 2026-08-13 HBI was the least-verified subsystem in the toolbox.
`baseline_snapshot.py` excludes it by design — its stated reason is that
HBI is slow and its exceedance probabilities are Monte-Carlo, which is a
fair objection to pinning *those*. But `model_frequency` and the free-energy
bound are deterministic given the data and the initial fits, so they pin
perfectly well, and that is what this harness does.

What is pinned here
-------------------
1. BACKWARD COMPATIBILITY (the important one). `model_trials=None` — the
   default and what every pre-Mod-11 caller passes implicitly — must
   reproduce the old six-argument `optimize_map` call EXACTLY. Pinned to
   hard-coded reference values, bit-for-bit.
2. Mod 11 reaches the Gauss-Newton path: with `model_trials` supplied every
   refit reports hess_method == "gauss_newton"; without it, every refit
   reports "finite_diff_clipped".
3. Mod 11 removes a real instability. Before it, HBI's verdict depended on
   how the SUPPLIED maps were fitted, because those maps seed the refits and
   the finite-difference refit could not escape a bad seed (DEV.md §13.3).
   With GN refits the verdict is the same from either seed.
4. Mod 12 populates `IndividualPosterior.diagnostics` with a (K, N) array of
   real PostFitDiagnostics — the regression guarded here is the one that
   actually occurred during implementation: `diagnostics` is a METHOD on
   OptimizationResult, so a bare getattr silently stores a bound method
   that only fails much later, at the point of use.
5. Input validation: a `model_trials` list whose length does not match
   `models` raises ValueError rather than failing obscurely mid-run.
6. Mod 12 is backward-compatible on unpickling: an IndividualPosterior
   built without the field still constructs, since HBI results are routinely
   written to disk and re-read by hbi_null.

Run:  python cbm/dev/hbi_verify.py   (exit 0 = all pass)
"""
import sys
import pickle
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from cbm.individual_fit import individual_fit          # noqa: E402
from cbm.hbi import hbi_main                           # noqa: E402
from cbm.hbi_types import IndividualPosterior          # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# Models — a non-linear value function v(x) = x^rho versus its linear special
# case, on risky-choice data. Chosen deliberately: this is the comparison that
# exposed the Mod 11 problem (DEV.md §13.3). POW nests LIN at rho = 1, so the
# two genuinely compete and the group frequency is not pinned at ~1.
# ---------------------------------------------------------------------------
def _choice_loglik(p_gamble, chose):
    p = np.clip(p_gamble, 1e-10, 1 - 1e-10)
    return float(np.sum(np.where(chose == 1, np.log(p), np.log1p(-p))))


def POW_model(parameters, data):
    sure, gamble, prob, chose = data
    rho, beta = np.exp(parameters[0]), np.exp(parameters[1])
    dv = prob * gamble ** rho - sure ** rho
    return _choice_loglik(1.0 / (1.0 + np.exp(-beta * dv)), chose)


def POW_model_trials(parameters, data):
    sure, gamble, prob, chose = data
    rho, beta = np.exp(parameters[0]), np.exp(parameters[1])
    dv = prob * gamble ** rho - sure ** rho
    p = np.clip(1.0 / (1.0 + np.exp(-beta * dv)), 1e-10, 1 - 1e-10)
    return np.where(chose == 1, np.log(p), np.log1p(-p))


def LIN_model(parameters, data):
    sure, gamble, prob, chose = data
    beta = np.exp(parameters[0])
    dv = prob * gamble - sure
    return _choice_loglik(1.0 / (1.0 + np.exp(-beta * dv)), chose)


def LIN_model_trials(parameters, data):
    sure, gamble, prob, chose = data
    beta = np.exp(parameters[0])
    dv = prob * gamble - sure
    p = np.clip(1.0 / (1.0 + np.exp(-beta * dv)), 1e-10, 1 - 1e-10)
    return np.where(chose == 1, np.log(p), np.log1p(-p))


MODELS = [POW_model, LIN_model]
TRIALS = [POW_model_trials, LIN_model_trials]
PRIOR_MEANS = [np.zeros(2), np.zeros(1)]
PRIOR_VARIANCE = 10.0

# Small enough to run in well under a minute, large enough that the group
# frequency is a meaningful quantity rather than noise.
N_SUBJECTS, N_TRIALS, LAPSE = 30, 120, 0.10


def build_data(seed=20260813):
    """Risky-choice data from the POW generator, with lapse trials.

    Self-contained and seeded, so this harness does not depend on the
    benchmark grid existing on disk.
    """
    rng = np.random.default_rng(seed)
    true_rho = rng.uniform(0.4, 1.6, N_SUBJECTS)
    true_beta = np.exp(rng.uniform(np.log(0.5), np.log(4.0), N_SUBJECTS))
    data = []
    for i in range(N_SUBJECTS):
        sure = rng.uniform(1.0, 9.0, N_TRIALS)
        gamble = sure + rng.uniform(0.5, 8.0, N_TRIALS)
        prob = rng.uniform(0.2, 0.8, N_TRIALS)
        dv = prob * gamble ** true_rho[i] - sure ** true_rho[i]
        p = 1.0 / (1.0 + np.exp(-true_beta[i] * dv))
        chose = (rng.random(N_TRIALS) < p).astype(int)
        lapse = rng.random(N_TRIALS) < LAPSE
        chose[lapse] = (rng.random(int(lapse.sum())) < 0.5).astype(int)
        data.append((sure, gamble, prob, chose))
    return data


def fit_maps(data, tmpdir, use_gn, tag):
    """Individual fits, written to pickles (hbi_main wants paths)."""
    paths = []
    for k, model in enumerate(MODELS):
        np.random.seed(7)
        fit = individual_fit(data, model, PRIOR_MEANS[k], PRIOR_VARIANCE,
                             config=dict(num_init=3, verbose=False),
                             model_trials=TRIALS[k] if use_gn else None)
        p = str(Path(tmpdir) / f"map_{tag}_{k}.pkl")
        with open(p, "wb") as f:
            pickle.dump(fit, f)
        paths.append(p)
    return paths


def run_hbi(data, paths, tmpdir, tag, model_trials):
    np.random.seed(7)
    return hbi_main(data, MODELS, paths, str(Path(tmpdir) / f"hbi_{tag}.pkl"),
                    config={"save_prog": False, "verbose": False},
                    model_trials=model_trials)


def hess_methods(result):
    dg = result.math.qhquad.diagnostics
    if dg is None:
        return Counter()
    return Counter(x.hess_method for x in dg.ravel() if x is not None)


# ---------------------------------------------------------------------------
# REFERENCE VALUES — recorded 2026-08-13 from the pre-Mod-11 code path.
#
# These pin backward compatibility. They must NOT be regenerated to make a
# failing test pass: if check 1 fails, the default path has changed
# behaviour, which is exactly what Mod 11 promised would not happen.
# ---------------------------------------------------------------------------
# Recorded by checking out the pre-Mod-11 hbi.py/hbi_updates.py/hbi_types.py
# (git stash) and calling the OLD six-argument hbi_main. So this is a genuine
# pre-modification value, not a post-hoc rationalisation of current output.
REF_DEFAULT_FREQ = [0.5564341971439771, 0.4435658028560228]


def main():
    record = "--record" in sys.argv
    warnings.filterwarnings("ignore")
    print(__doc__.split("Run:")[0].strip()[:0] or "", end="")
    print(f"HBI verification — {N_SUBJECTS} subjects x {N_TRIALS} trials, "
          f"{LAPSE:.0%} lapse\n")

    data = build_data()
    with tempfile.TemporaryDirectory() as tmp:
        gn_maps = fit_maps(data, tmp, True, "gn")
        fd_maps = fit_maps(data, tmp, False, "fd")

        # Four combinations of (how the maps were fitted) x (how HBI refits).
        r_fd_fd = run_hbi(data, fd_maps, tmp, "fdfd", None)
        r_gn_fd = run_hbi(data, gn_maps, tmp, "gnfd", None)
        r_gn_gn = run_hbi(data, gn_maps, tmp, "gngn", TRIALS)
        r_fd_gn = run_hbi(data, fd_maps, tmp, "fdgn", TRIALS)

        f_fd_fd = np.ravel(r_fd_fd.output.model_frequency)
        f_gn_fd = np.ravel(r_gn_fd.output.model_frequency)
        f_gn_gn = np.ravel(r_gn_gn.output.model_frequency)
        f_fd_gn = np.ravel(r_fd_gn.output.model_frequency)

        if record:
            print("Reference values (paste into REF_* constants):")
            print(f"  REF_DEFAULT_FREQ = {f_fd_fd.tolist()!r}")
            return 0

        print("model_frequency [POW, LIN] by (maps, refits):")
        for lbl, f in (("FD maps, FD refits", f_fd_fd),
                       ("GN maps, FD refits", f_gn_fd),
                       ("GN maps, GN refits", f_gn_gn),
                       ("FD maps, GN refits", f_fd_gn)):
            print(f"    {lbl:20s} {np.round(f, 6)}")
        print()

        # -- 1. backward compatibility -----------------------------------
        if REF_DEFAULT_FREQ is None:
            check("1. default path pinned to reference", False,
                  "REF_DEFAULT_FREQ unset — run with --record once")
        else:
            d = float(np.max(np.abs(f_fd_fd - np.asarray(REF_DEFAULT_FREQ))))
            check("1. default path bit-identical to pre-Mod-11 reference",
                  d == 0.0, f"max delta {d:.3e}")

        # -- 2. Mod 11 reaches the Gauss-Newton path ----------------------
        m_off, m_on = hess_methods(r_gn_fd), hess_methods(r_gn_gn)
        check("2a. model_trials=None  -> every refit finite_diff_clipped",
              set(m_off) == {"finite_diff_clipped"}, str(dict(m_off)))
        check("2b. model_trials given -> every refit gauss_newton",
              set(m_on) == {"gauss_newton"}, str(dict(m_on)))

        # -- 3. Mod 11 removes the seed-dependence ------------------------
        # The defect: under FD refits the verdict depends on how the
        # supplied maps were produced, because a bad seed is inescapable.
        spread_fd = float(np.max(np.abs(f_fd_fd - f_gn_fd)))
        spread_gn = float(np.max(np.abs(f_fd_gn - f_gn_gn)))
        check("3a. FD refits are seed-dependent (the defect Mod 11 fixes)",
              spread_fd > 1e-3, f"spread {spread_fd:.4f}")
        check("3b. GN refits agree from either seed",
              spread_gn < 1e-2, f"spread {spread_gn:.2e}")
        check("3c. GN refits are the more stable path",
              spread_gn < spread_fd,
              f"{spread_gn:.2e} < {spread_fd:.4f}")

        # -- 4. Mod 12 diagnostics ---------------------------------------
        dg = r_gn_gn.math.qhquad.diagnostics
        check("4a. diagnostics field present and correctly shaped",
              dg is not None and dg.shape == (len(MODELS), N_SUBJECTS),
              None if dg is None else str(dg.shape))
        if dg is not None:
            n_ok = sum(1 for x in dg.ravel() if x is not None)
            check("4b. every (model, subject) slot populated",
                  n_ok == dg.size, f"{n_ok}/{dg.size}")
            # The regression that actually happened during implementation.
            first = next((x for x in dg.ravel() if x is not None), None)
            check("4c. entries are records, not bound methods",
                  first is not None and not callable(first)
                  and hasattr(first, "hess_method"),
                  type(first).__name__)
            nia = [getattr(x, "n_inits_agreeing", None) for x in dg.ravel()
                   if x is not None]
            check("4d. Mod 9 fields carried through",
                  all(v is not None for v in nia),
                  f"n_inits_agreeing min={min(v for v in nia if v is not None)}")

        # -- 5. input validation ------------------------------------------
        try:
            run_hbi(data, gn_maps, tmp, "bad", TRIALS[:1])
            check("5. mismatched model_trials length raises ValueError", False,
                  "no exception")
        except ValueError:
            check("5. mismatched model_trials length raises ValueError", True)
        except Exception as e:
            check("5. mismatched model_trials length raises ValueError", False,
                  f"raised {type(e).__name__}")

        # -- 6. Mod 12 is backward-compatible on construction -------------
        try:
            IndividualPosterior(loglik=np.zeros((1, 1)), parameters=[],
                                hessian_inv_diag=[],
                                log_det_hessian=np.zeros((1, 1)))
            check("6. IndividualPosterior constructs without diagnostics",
                  True)
        except Exception as e:
            check("6. IndividualPosterior constructs without diagnostics",
                  False, f"{type(e).__name__}: {e}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
